"""Prepare captured inference bindings asynchronously, then use the existing attempt lanes."""

from __future__ import annotations

import base64
import contextlib
import hashlib
import sys
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any

import msgspec

from cozy_runtime import canonical_json
from cozy_runtime.author._calls import MAX_ACTIVE_CALLS
from cozy_runtime.author._model_bindings import read_binding
from cozy_runtime.internal import package_environment
from cozy_runtime.protocol import documents
from cozy_runtime.protocol import worker_pb2 as pb

from . import (
    machine_byte_inputs,
    machine_byte_results,
    machine_capture,
    machine_models,
    package_prepare,
)
from .machine_child_target import Target
from .workspace import Workspace, WorkspaceRefusal
from .workspace_calls import Call
from .workspace_calls import Calls as JournalCalls

if TYPE_CHECKING:
    from .session import Worker


class Serving:
    def __init__(self, worker: Worker):
        self.worker = worker
        self.pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="machine-prepare")
        self.pending: dict[tuple[str, str], Future[dict[str, Any]]] = {}

    def close(self) -> None:
        self.pool.shutdown(wait=True, cancel_futures=True)

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
        try:
            placement = future.result()
            journal.check_active(owner, call)
            selected = next(
                (entry for entry in placement["entrypoints"] if entry["name"] == target.entrypoint),
                None,
            )
            if selected is None:
                raise WorkspaceRefusal("captured inference has no complete prepared Model binding")
            payload = canonical_json.encode(_call_arguments(target, request)[0])
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
            self.pending.pop(key, None)

    def _prepare(
        self, owner: str, call: Call, target: Target, request: pb.ChildCallRequest
    ) -> dict[str, Any]:
        worker, workspace = self.worker, self.worker.workspace
        assert workspace is not None
        prepared = target.prepared_revision["placement"]
        if not target.declaration.get("models"):
            return dict(prepared)
        _, arguments = _call_arguments(target, request)
        with contextlib.ExitStack() as access:
            native, native_adapters = _retain_models(
                workspace,
                owner,
                call.child_request,
                target.entrypoint,
                target.declaration["models"],
                arguments,
                access,
            )
            # Reuse preparation only when every base, component and ordered adapter
            # selector matches. This never reuses a generation result.
            with worker.control_lock:
                candidates = [dict(value.document) for value in worker.prepared_revisions.values()]
            for candidate in candidates:
                if (
                    machine_capture.code_identity(candidate) == target.revision
                    and candidate.get("environment_digest") == prepared["environment_digest"]
                    and candidate.get("package_interface") == prepared["package_interface"]
                    and _matches_bindings(candidate, target.entrypoint, native, native_adapters)
                ):
                    return candidate
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
                    native_adapters=native_adapters,
                    prepared=prepared,
                    installed=installed,
                    artifact_cache=Path(worker.options.artifact_cache or ""),
                    tensorfs_root=workspace.store_root,
                    verified=verified,
                )
            return dict(
                documents.read(result.placement_set.placement_set_canonical_bytes, pb.PlacementSet)[
                    "placements"
                ][0]
            )


def _call_arguments(
    target: Target,
    request: pb.ChildCallRequest,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Separate injected Model arguments using the frozen callable declaration.

    Generated serving callers carry payload/models; authored @invocable calls carry
    flat signature arguments. Only declared injected slots can contain wrappers.
    The original request bytes remain the journaled child intent in both cases.
    """
    arguments = canonical_json.decode(request.request_canonical_bytes)
    if not isinstance(arguments, dict):
        raise WorkspaceRefusal("serving call arguments must be an object")
    names = {
        row["path"].removeprefix(target.entrypoint + ".models.")
        for row in target.declaration.get("models", [])
    }
    fields = {row["name"] for row in target.declaration["request"].get("fields", [])}
    if names <= fields and (names or set(arguments) != {"payload", "models"}):
        payload, models = dict(arguments), {name: arguments.get(name) for name in names}
        for name, value in models.items():
            if value is not None:
                base, _ = read_binding(value)
                payload[name] = msgspec.to_builtins(base)
    else:
        if set(arguments) != {"payload", "models"} or not isinstance(arguments["payload"], dict):
            raise WorkspaceRefusal("serving caller requires its declared payload/models envelope")
        payload, models = dict(arguments["payload"]), arguments["models"]
        if not isinstance(models, dict) or set(models) != names:
            raise WorkspaceRefusal("serving caller model slots differ from its declaration")
    if any(value is None for value in models.values()):
        raise WorkspaceRefusal("captured serving requires exact retained Model arguments")
    return payload, models


def _matches_bindings(
    candidate: dict[str, Any],
    entrypoint: str,
    native: list[pb.NativeModelBinding],
    adapters: dict[str, tuple[package_prepare.NativeAdapterSelection, ...]],
) -> bool:
    entry = next(
        (row for row in candidate.get("entrypoints", []) if row["name"] == entrypoint), None
    )
    if entry is None:
        return False
    models = {row["id"]: row["manifest"] for row in candidate.get("models", [])}
    slots = {row["slot"]: row for row in entry["slots"]}
    prefix = entrypoint + ".models."
    if set(slots) != {row.slot.removeprefix(prefix) for row in native}:
        return False
    used: set[str] = set()
    for row in native:
        slot = slots[row.slot.removeprefix(prefix)]
        reference = slot["reference_model_id"]
        used.add(reference)
        if models.get(reference) != documents.body(row.manifest) or any(
            component["model_id"] != reference for component in slot.get("components", [])
        ):
            return False
        selected = slot.get("adapters", [])
        expected = adapters.get(row.slot, ())
        if len(selected) != len(expected):
            return False
        for actual, adapter in zip(selected, expected, strict=True):
            used.add(actual["model_id"])
            if (actual["component"], actual["source_component"], actual["scale"]) != (
                adapter.component,
                adapter.source_component,
                adapter.scale,
            ) or models.get(actual["model_id"]) != {
                "digest": adapter.manifest,
                "length": adapter.manifest_length,
            }:
                return False
    return used == models.keys()


def _retain_models(
    workspace: Workspace,
    owner: str,
    recipient: str,
    entrypoint: str,
    declared: list[dict[str, Any]],
    arguments: dict[str, Any],
    access: contextlib.ExitStack,
) -> tuple[
    list[pb.NativeModelBinding], dict[str, tuple[package_prepare.NativeAdapterSelection, ...]]
]:
    native: list[pb.NativeModelBinding] = []
    native_adapters: dict[str, tuple[package_prepare.NativeAdapterSelection, ...]] = {}
    for model in declared:
        prefix = entrypoint + ".models."
        path = model["path"]
        if not path.startswith(prefix) or not path.removeprefix(prefix).isidentifier():
            raise WorkspaceRefusal("captured serving Model has no exact parameter")
        artifact, selected_adapters = read_binding(arguments[path.removeprefix(prefix)])
        retention, held = machine_models.retain(
            workspace,
            owner,
            recipient,
            "input/" + path.removeprefix(prefix),
            artifact,
        )
        access.enter_context(workspace.held_model(owner, retention, held.manifest))
        adapter_rows = []
        for index, adapter in enumerate(selected_adapters):
            adapter_retention, adapter_held = machine_models.retain(
                workspace,
                owner,
                recipient,
                "input/" + path.removeprefix(prefix) + "/adapters/" + str(index),
                adapter.source,
            )
            access.enter_context(
                workspace.held_model(
                    owner,
                    adapter_retention,
                    adapter_held.manifest,
                )
            )
            adapter_rows.append(
                package_prepare.NativeAdapterSelection(
                    adapter.component,
                    adapter.source.manifest.digest,
                    adapter.source.manifest.length,
                    adapter.source_component,
                    adapter.scale,
                )
            )
        if adapter_rows:
            native_adapters[path] = tuple(adapter_rows)
        native.append(
            pb.NativeModelBinding(
                slot=path,
                model="native/" + artifact.manifest.digest[7:39],
                manifest=held.manifest,
                retention=retention,
            )
        )
    return native, native_adapters
