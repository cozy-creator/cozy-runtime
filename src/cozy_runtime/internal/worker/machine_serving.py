"""Prepare captured inference bindings asynchronously, then use the existing attempt lanes."""

from __future__ import annotations

import base64
import contextlib
import hashlib
import sys
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import msgspec

from cozy_runtime import canonical_json
from cozy_runtime.author._artifacts import ModelArtifact
from cozy_runtime.author._calls import MAX_ACTIVE_CALLS
from cozy_runtime.internal import package_environment
from cozy_runtime.protocol import documents
from cozy_runtime.protocol import worker_pb2 as pb

from . import (
    grants,
    machine_byte_inputs,
    machine_byte_results,
    machine_capture,
    machine_checkpoint_inputs,
    machine_model_defaults,
    machine_models,
    package_prepare,
)
from .machine_child_target import Target
from .workspace import WorkspaceRefusal
from .workspace_calls import Call
from .workspace_calls import Calls as JournalCalls

if TYPE_CHECKING:
    from .session import Worker


def arguments(target: Target, raw: bytes) -> tuple[bytes, dict[str, Any]]:
    """Generated library callers separate Model overrides from the handler payload.

    A registered self-call uses the original invocable's flat request. The frozen
    caller/callee binding identifies that convention; user field names never do.
    """
    value = canonical_json.decode(raw)
    if not isinstance(value, dict):
        raise WorkspaceRefusal("captured serving arguments must be an object")
    if not target.serving_envelope:
        return raw, value
    names = {model["path"].rpartition(".")[2] for model in target.declaration.get("models", [])}
    if (
        set(value) != {"payload", "models"}
        or not isinstance(value["payload"], dict)
        or not isinstance(value["models"], dict)
        or not set(value["models"]) <= names
    ):
        raise WorkspaceRefusal("generated serving call has invalid payload or Model slots")
    return canonical_json.encode(value["payload"]), {
        name: model for name, model in value["models"].items() if model is not None
    }


@dataclass
class Prepared:
    placement: dict[str, Any]
    defaults: list[dict[str, Any]]
    access: contextlib.ExitStack


class Serving:
    def __init__(self, worker: Worker):
        self.worker = worker
        self.pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="machine-prepare")
        self.pending: dict[tuple[str, str], Future[Prepared]] = {}

    def close(self) -> None:
        self.pool.shutdown(wait=True, cancel_futures=True)
        for future in self.pending.values():
            if not future.cancelled():
                with contextlib.suppress(Exception):
                    future.result().access.close()

    def sweep(self, owner: str) -> None:
        workspace, executions = self.worker.workspace, self.worker.executions
        assert workspace is not None and executions is not None
        for key, future in list(self.pending.items()):
            if key[0] != owner or not future.done():
                continue
            with workspace.locked() as db:
                row = db.execute(
                    "SELECT e.desired FROM execution_calls c JOIN executions e "
                    "ON e.owner=c.owner AND e.request=c.parent_request "
                    "WHERE c.owner=? AND c.child_request=?",
                    key,
                ).fetchone()
            if row is not None and row["desired"] == "cancel":
                with contextlib.suppress(Exception):
                    future.result().access.close()
                machine_models.release(workspace, owner, key[1])
                self.pending.pop(key, None)

    def drive(self, owner: str, call: Call, target: Target, request: pb.ChildCallRequest) -> None:
        workspace, executions = self.worker.workspace, self.worker.executions
        assert workspace is not None and executions is not None
        if target.declaration.get("invocable", {}).get("memoize"):
            raise WorkspaceRefusal("inference calls cannot enter the memo index")
        journal = JournalCalls(workspace)
        if not call.prepared:
            journal.freeze(
                owner,
                call,
                canonical_json.encode(
                    {
                        "kind": "serving",
                        "revision": target.revision,
                        "entrypoint": target.entrypoint,
                        "parent_attempt": request.parent_attempt_ordinal,
                        "result_schema": target.declaration["result"],
                    }
                ),
            )
        key = owner, call.child_request
        future = self.pending.get(key)
        if future is None:
            if len(self.pending) >= MAX_ACTIVE_CALLS:
                raise WorkspaceRefusal("captured inference preparation capacity is exhausted")
            self.pending[key] = self.pool.submit(self._prepare, owner, call, target, request)
            return
        if not future.done():
            return
        prepared = None
        try:
            prepared = future.result()
            placement = prepared.placement
            journal.check_active(owner, call)
            selected = next(
                (entry for entry in placement["entrypoints"] if entry["name"] == target.entrypoint),
                None,
            )
            if selected is None:
                raise WorkspaceRefusal("captured inference has no complete prepared Model binding")
            payload, _ = arguments(target, request.request_canonical_bytes)
            digest = documents.spell(hashlib.sha256(payload).digest())
            with workspace.locked() as db:
                row = db.execute(
                    "SELECT invocation FROM attempts WHERE owner=? AND request=? AND ordinal=?",
                    (owner, call.parent_request, request.parent_attempt_ordinal),
                ).fetchone()
            parent = documents.read(row["invocation"], pb.InvocationSpec)
            spec = pb.InvocationSpec(
                environment_digest=placement["environment_digest"],
                payload_digest=digest,
                inputs=[
                    pb.InputBinding(
                        input_id="payload",
                        digest=digest,
                        length=len(payload),
                        kind_mime="application/json",
                    )
                ],
                outputs=[
                    pb.OutputBinding(output_id=path, max_bytes=bound)
                    for path, bound in machine_byte_results.paths(target.declaration["result"])
                ],
                deadline_unix_ms=parent.get("deadline_unix_ms", 0),
                serving=pb.ServingInvocationSpec(
                    entrypoint_binding_digest=selected["entrypoint_binding_digest"],
                    attempt_binding_id=selected["entrypoint_binding_digest"],
                    bindings_digest=placement["bindings_digest"],
                ),
            )
            if request.HasField("capture"):
                spec.capture.CopyFrom(request.capture)
                spec.outputs.append(
                    pb.OutputBinding(output_id="runtime.capture", max_bytes=64 << 20)
                )
            live_parent = self.worker.engine.live.get(call.parent_request)
            if live_parent is None or live_parent.attempt != request.parent_attempt_ordinal:
                raise WorkspaceRefusal("native child input parent is no longer active")
            byte_bindings, byte_access = machine_byte_inputs.inputs(
                workspace,
                owner,
                live_parent,
                call.child_request,
                target.declaration["request"],
                canonical_json.decode(payload),
                self.worker.calls,
            )
            spec.inputs.extend(byte_bindings)
            default_bindings, default_access = machine_model_defaults.inputs(prepared.defaults)
            spec.inputs.extend(default_bindings)
            spec.inputs.sort(key=lambda entry: entry.input_id)
            raw, identity = documents.identity(spec)
            offer = pb.AttemptOffer(
                request_id=call.child_request,
                attempt_ordinal=1,
                placement_id=placement["placement_id"],
                invocation_spec_canonical_bytes=raw,
                invocation_spec_digest=identity,
                grant=pb.DeliveryGrant(
                    invocation_spec_digest=identity,
                    inputs=[
                        pb.InputAccess(
                            input_id="payload",
                            url="data:application/json;base64,"
                            + base64.b64encode(payload).decode(),
                        )
                    ],
                    outputs=[
                        pb.OutputAccess(output_id=output.output_id) for output in spec.outputs
                    ],
                ),
            )
            offer.grant.inputs.extend(byte_access)
            offer.grant.inputs.extend(default_access)
            offer.grant.inputs.sort(key=lambda entry: entry.input_id)
            placement_bytes = canonical_json.encode(
                {"format": "cozy.worker.v1.PlacementSet/1", "placements": [placement]}
            )
            desired = pb.DesiredWorkerState(
                wire_minor=self.worker.fence.wire_minor,
                posture=pb.POSTURE_ACCEPTING,
                placement_set=pb.DesiredPlacementSet(
                    placement_set_digest=hashlib.sha256(placement_bytes).digest(),
                    placement_set_canonical_bytes=placement_bytes,
                ),
            )
            preparation = canonical_json.encode(
                {
                    "revisions": {
                        target.revision: {
                            "placement": placement,
                            "receipt_digest": target.prepared_revision["receipt_digest"],
                        }
                    },
                    "state": base64.b64encode(
                        desired.SerializeToString(deterministic=True)
                    ).decode(),
                    "parent_attempt": request.parent_attempt_ordinal,
                }
            )
            capture = canonical_json.encode(executions.capture(owner, call.parent_request))
            attempts, delay = executions.retry_policy(owner, call.parent_request)
            with machine_checkpoint_inputs.admission(workspace, owner, offer.request_id):
                catalog = grants.model_inputs(
                    grants.bind(documents.read(raw, pb.InvocationSpec), offer.grant, identity)
                )
                machine_checkpoint_inputs.preflight(workspace, owner, offer, catalog)
                for entry in catalog.values():
                    machine_checkpoint_inputs.retain(workspace, owner, offer, entry)
                executions.submit(
                    owner,
                    call.child_request,
                    hashlib.sha256(capture).digest(),
                    offer,
                    result_schema=canonical_json.encode(target.declaration["result"]),
                    worker_boot=self.worker.fence.worker_boot_id,
                    memoize=False,
                    preparation=preparation,
                    worker_id=self.worker.options.worker_id,
                    max_attempts=attempts,
                    retry_delay_ms=delay,
                )
        finally:
            if prepared is not None:
                prepared.access.close()
            self.pending.pop(key, None)

    def _prepare(
        self, owner: str, call: Call, target: Target, request: pb.ChildCallRequest
    ) -> Prepared:
        worker, workspace = self.worker, self.worker.workspace
        assert workspace is not None
        prepared = target.prepared_revision["placement"]
        if not target.declaration.get("models"):
            return Prepared(dict(prepared), [], contextlib.ExitStack())
        _, model_arguments = arguments(target, request.request_canonical_bytes)
        native: list[pb.NativeModelBinding] = []
        defaults = machine_model_defaults.select(worker, owner, call, target, model_arguments)
        access = contextlib.ExitStack()
        try:
            machine_model_defaults.materialize(worker, defaults, access)
            for model in target.declaration["models"]:
                prefix = target.entrypoint + ".models."
                path = model["path"]
                if not path.startswith(prefix) or not path.removeprefix(prefix).isidentifier():
                    raise WorkspaceRefusal("captured serving Model has no exact parameter")
                if model_arguments.get(path.removeprefix(prefix)) is None:
                    continue
                supplied = model_arguments.get(path.removeprefix(prefix))
                if supplied is None:
                    raise WorkspaceRefusal("captured serving Model default has not been retained")
                artifact = msgspec.convert(supplied, type=ModelArtifact, strict=True)
                retention, held = machine_models.retain(
                    workspace,
                    owner,
                    call.child_request,
                    "input/" + path.removeprefix(prefix),
                    artifact,
                )
                access.enter_context(workspace.held_model(owner, retention, held.manifest))
                native.append(
                    pb.NativeModelBinding(
                        slot=path,
                        model="native/" + artifact.manifest.digest[7:39],
                        manifest=held.manifest,
                        retention=retention,
                    )
                )
            # The existing immutable preparation registry already owns this work.
            # Reusing its exact checkpoint bindings does not reuse an inference result.
            expected = {
                model.slot.removeprefix(target.entrypoint + ".models."): documents.body(
                    model.manifest
                )
                for model in native
            }
            expected.update({row["parameter"]: row["manifest"] for row in defaults})
            with worker.control_lock:
                candidates = [dict(value.document) for value in worker.prepared_revisions.values()]
            for candidate in candidates:
                if (
                    machine_capture.code_identity(candidate) != target.revision
                    or candidate.get("environment_digest") != prepared["environment_digest"]
                    or candidate.get("package_interface") != prepared["package_interface"]
                ):
                    continue
                entry = next(
                    (
                        row
                        for row in candidate.get("entrypoints", [])
                        if row["name"] == target.entrypoint
                    ),
                    None,
                )
                models = {row["id"]: row["manifest"] for row in candidate.get("models", [])}
                if entry is None:
                    continue
                used = {slot["reference_model_id"] for slot in entry["slots"]}
                if used != models.keys() or any(
                    component["model_id"] != slot["reference_model_id"]
                    for slot in entry["slots"]
                    for component in slot.get("components", [])
                ):
                    continue
                if {
                    slot["slot"]: models[slot["reference_model_id"]] for slot in entry["slots"]
                } == expected:
                    return Prepared(candidate, defaults, access)
            installed = package_environment.open_generation(
                Path(worker.options.install_root or ""),
                receipt_digest=target.prepared_revision["receipt_digest"],
                environment_digest=prepared["environment_digest"],
                python=Path(worker.options.python or sys.executable),
            )
            with worker.preparation_lock:

                def verified(digest: str, length: int, path: Path) -> None:
                    worker.verified_artifacts.record(digest, length, path)

                native.sort(key=lambda model: model.slot)
                result = package_prepare.prepare_model_placement(
                    native_models=native,
                    models=[
                        {
                            "package": (prepared.get("package") or prepared["development"])[
                                "package"
                            ],
                            "slot": target.entrypoint + ".models." + row["parameter"],
                            "model": row["repository"],
                            "manifest": row["manifest"]["digest"],
                            "manifest_length": row["manifest"]["length"],
                        }
                        for row in defaults
                    ],
                    prepared=prepared,
                    installed=installed,
                    artifact_cache=Path(worker.options.artifact_cache or ""),
                    tensorfs_root=workspace.store_root,
                    verified=verified,
                )
            return Prepared(
                dict(
                    documents.read(
                        result.placement_set.placement_set_canonical_bytes, pb.PlacementSet
                    )["placements"][0]
                ),
                defaults,
                access,
            )
        except BaseException:
            access.close()
            raise
