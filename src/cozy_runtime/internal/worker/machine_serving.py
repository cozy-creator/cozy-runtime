"""Prepare captured inference bindings asynchronously, then use the existing attempt lanes."""

from __future__ import annotations

import base64
import contextlib
import hashlib
import time
from collections.abc import Callable, Hashable, Iterator, Mapping
from concurrent.futures import Future
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import msgspec

from cozy_runtime import canonical_json
from cozy_runtime.author._artifacts import ModelArtifact
from cozy_runtime.author._calls import MAX_ACTIVE_CALLS
from cozy_runtime.internal import fill, package_installation
from cozy_runtime.protocol import documents
from cozy_runtime.protocol import worker_pb2 as pb

from . import (
    grants,
    machine_byte_inputs,
    machine_byte_results,
    machine_capture,
    machine_checkpoint_inputs,
    machine_child_target,
    machine_model_defaults,
    machine_models,
    package_prepare,
)
from .machine_child_target import Target
from .preparations import Preparations
from .workspace import WorkspaceRefusal
from .workspace_calls import Call
from .workspace_calls import Calls as JournalCalls

if TYPE_CHECKING:
    from .attempts import AttemptRecord
    from .session import Worker


def arguments(target: Target, raw: bytes) -> tuple[bytes, dict[str, Any]]:
    """Generated library callers separate Model overrides from the handler payload.

    A registered self-call uses the original invocable's flat request. The frozen
    caller/callee binding identifies that convention; user field names never do.
    """
    value = canonical_json.decode(raw)
    if not isinstance(value, dict):
        raise WorkspaceRefusal("captured serving arguments must be an object")
    names = {model["path"].rpartition(".")[2] for model in target.declaration.get("models", [])}
    if not target.serving_envelope:
        payload = {key: item for key, item in value.items() if key not in names}
        return canonical_json.encode(payload), value
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
        self.preparations = Preparations[Prepared](MAX_ACTIVE_CALLS)
        self.pending: dict[tuple[str, str], Future[Prepared]] = {}

    @contextlib.contextmanager
    def phase(self, call: Call, request: pb.ChildCallRequest, name: str) -> Iterator[None]:
        """Separate immutable preparation and per-child custody in the parent journal."""
        started = time.perf_counter()
        label = self.worker.calls.progress_label(
            call.parent_request, request.parent_attempt_ordinal, call.call_index
        )
        stage = f"{label} / {name}"[:120] if label else name
        self.worker.emit_progress(
            call.parent_request,
            request.parent_attempt_ordinal,
            {"kind": "progress", "stage": stage},
        )
        succeeded = False
        try:
            yield
            succeeded = True
        finally:
            self.worker.emit_progress(
                call.parent_request,
                request.parent_attempt_ordinal,
                {
                    "kind": "log",
                    "name": name,
                    "value": "info",
                    "at_unix_ms": int(time.time() * 1000),
                    "fields": {
                        "child_request": call.child_request,
                        "phase": name,
                        "completed": succeeded,
                        "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
                    },
                },
            )

    def close(self) -> None:
        self.preparations.close()

    def prefetch(self, parent: AttemptRecord, frame: Mapping[str, Any]) -> dict[str, Any]:
        """A model-only hint; never accepts a child execution or chooses a GPU lane."""
        owner = self.worker.fence.record_owner_id
        target = machine_child_target.resolve_export(
            self.worker, parent, str(frame.get("module", "")), str(frame.get("export", ""))
        )
        if target.binding is not None:
            raise WorkspaceRefusal("model prefetch requires a serving callable")
        raw = frame.get("payload")
        if not isinstance(raw, str) or len(raw.encode()) > 48 << 10:
            raise WorkspaceRefusal("model prefetch exceeds its inline bound")
        models = canonical_json.decode(raw.encode())
        names = {row["path"].rpartition(".")[2] for row in target.declaration.get("models", [])}
        if not isinstance(models, dict) or not models.keys() <= names:
            raise WorkspaceRefusal("model prefetch names an undeclared Model slot")
        for artifact in models.values():
            if artifact is not None:
                msgspec.convert(artifact, type=ModelArtifact, strict=True)
        # Only the preparation context is needed; this is never journal.accept'ed
        # and does not reserve a managed-call index or a child execution.
        call = Call(
            parent.request_id,
            (1 << 32) - 1,
            parent.attempt,
            0,
            b"",
            b"",
            "",
            b"",
            False,
            b"",
            "",
            "",
        )
        request = pb.ChildCallRequest(parent_attempt_ordinal=parent.attempt)
        try:
            future = self._shared(owner, call, target, request, models, speculative=True)
        except WorkspaceRefusal as error:
            self.worker.note("model-prefetch", str(error))
            return {"ok": True, "scheduled": False}
        return {"ok": True, "scheduled": future is not None}

    def sweep(self, owner: str) -> None:
        def active(interest: Hashable) -> bool:
            request_id, ordinal = cast(tuple[str, int], interest)
            parent = self.worker.engine.live.get(request_id)
            return bool(
                parent is not None
                and parent.attempt == ordinal
                and parent.state == "running"
                and not parent.canceling
            )

        self.preparations.retain(active)
        workspace = self.worker.workspace
        assert workspace is not None
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
            if row is not None and row["desired"] != "run":
                machine_models.release(workspace, owner, key[1])
                self.pending.pop(key, None)

    def _shared(
        self,
        owner: str,
        call: Call,
        target: Target,
        request: pb.ChildCallRequest,
        models: dict[str, Any],
        *,
        speculative: bool,
    ) -> Future[Prepared] | None:
        names = {row["path"].rpartition(".")[2] for row in target.declaration.get("models", [])}
        models = {
            name: value for name, value in models.items() if name in names and value is not None
        }
        defaults = machine_model_defaults.select(self.worker, owner, call, target, models)
        identity = hashlib.sha256(
            canonical_json.encode(
                {
                    "owner": owner,
                    "installation": target.installation_id,
                    "interface": target.prepared_installation["placement"]["package_interface"],
                    "entrypoint": target.entrypoint,
                    "models": models,
                    "defaults": defaults,
                }
            )
        ).hexdigest()
        source = replace(call, child_request="model-preparation." + identity)
        tensorfs = fill.tensorfs_module()
        token = tensorfs.PullCancellation()

        def load(cancelled: Callable[[], bool]) -> Prepared:
            try:
                if cancelled():
                    raise WorkspaceRefusal("model preparation canceled")
                prepared = self._prepare(
                    owner,
                    source,
                    target,
                    request,
                    model_arguments=models,
                    defaults=defaults,
                    cancellation=token,
                )
                # Consumers acquire their own holds at actual admission. Shared
                # preparation retains only immutable binding metadata on completion.
                prepared.access.close()
                return prepared
            finally:
                assert self.worker.workspace is not None
                machine_models.release(self.worker.workspace, owner, source.child_request)

        return self.preparations.request(
            identity,
            (call.parent_request, request.parent_attempt_ordinal),
            load,
            speculative=speculative,
            cancel=token.cancel,
        )

    def drive(self, owner: str, call: Call, target: Target, request: pb.ChildCallRequest) -> None:
        workspace, executions = self.worker.workspace, self.worker.executions
        assert workspace is not None and executions is not None
        workspace_id = executions.workspace_id
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
                        "installation_id": target.installation_id,
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
            _, models = arguments(target, request.request_canonical_bytes)
            future = self._shared(owner, call, target, request, models, speculative=False)
            if future is not None:
                self.pending[key] = future
            return
        if not future.done():
            return
        prepared = None
        try:
            prepared = future.result()
            # A shared CPU preparation owns no recipient custody. Actual B retains
            # its explicit artifacts exactly as ordinary demand preparation did.
            _, models = arguments(target, request.request_canonical_bytes)
            names = {row["path"].rpartition(".")[2] for row in target.declaration.get("models", [])}
            for name, supplied in models.items():
                if name in names and supplied is not None:
                    machine_models.retain(
                        workspace,
                        owner,
                        call.child_request,
                        "input/" + name,
                        msgspec.convert(supplied, type=ModelArtifact, strict=True),
                    )
            from . import machine_lanes

            placement, pins = machine_lanes.replica(
                self.worker,
                owner,
                prepared.placement,
                target.declaration.get("models", []),
                prepared.defaults,
            )
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
                installation_id=placement["installation_id"],
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
                    device_pins=pins,
                ),
            )
            preparation = canonical_json.encode(
                {
                    "installations": {
                        target.installation_id: {
                            "placement": placement,
                            "installation_id": target.installation_id,
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
            with (
                self.phase(call, request, "Checking model inputs"),
                machine_checkpoint_inputs.admission(workspace, owner, offer.request_id),
            ):
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
                    expected_execution_workspace_id=workspace_id,
                    result_schema=canonical_json.encode(target.declaration["result"]),
                    worker_boot=self.worker.fence.worker_boot_id,
                    memoize=False,
                    preparation=preparation,
                    worker_id=self.worker.options.worker_id,
                    max_attempts=attempts,
                    retry_delay_ms=delay,
                )
            if executions.status(owner, call.child_request).state == "queued":
                label = self.worker.calls.progress_label(
                    call.parent_request, request.parent_attempt_ordinal, call.call_index
                )
                self.worker.emit_progress(
                    call.parent_request,
                    request.parent_attempt_ordinal,
                    {
                        "kind": "progress",
                        "stage": (label + " / " if label else "") + "Waiting for GPUs",
                    },
                )
        finally:
            if prepared is not None:
                prepared.access.close()
            self.pending.pop(key, None)

    def _prepare(
        self,
        owner: str,
        call: Call,
        target: Target,
        request: pb.ChildCallRequest,
        *,
        model_arguments: dict[str, Any] | None = None,
        defaults: list[dict[str, Any]] | None = None,
        cancellation: Any = None,
    ) -> Prepared:
        worker, workspace = self.worker, self.worker.workspace
        assert workspace is not None
        prepared = target.prepared_installation["placement"]
        if not target.declaration.get("models"):
            return Prepared(dict(prepared), [], contextlib.ExitStack())
        if model_arguments is None:
            _, model_arguments = arguments(target, request.request_canonical_bytes)
        native: list[pb.NativeModelBinding] = []
        if defaults is None:
            with self.phase(call, request, "Selecting model"):
                defaults = machine_model_defaults.select(
                    worker, owner, call, target, model_arguments
                )
        access = contextlib.ExitStack()
        try:
            for model in target.declaration["models"]:
                prefix = target.entrypoint + ".models."
                path = model["path"]
                if not path.startswith(prefix) or not path.removeprefix(prefix).isidentifier():
                    raise WorkspaceRefusal("captured serving Model has no exact parameter")
                supplied = model_arguments.get(path.removeprefix(prefix))
                if supplied is None:
                    continue
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
                candidates = [
                    dict(value.document) for value in worker.prepared_installations.values()
                ]
            matched = None
            for candidate in candidates:
                if (
                    machine_capture.installation_identity(candidate) != target.installation_id
                    or candidate.get("installation_id") != prepared["installation_id"]
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
                    matched = candidate
                    break
            machine_model_defaults.materialize(
                worker,
                defaults,
                access if matched is None else None,
                downloading=lambda: self.phase(call, request, "Downloading model weights"),
                cancellation=cancellation,
                progress=lambda available, total: worker.emit_progress(
                    call.parent_request,
                    request.parent_attempt_ordinal,
                    {
                        "kind": "progress",
                        "stage": target.entrypoint + " / Downloading model weights",
                        "position": available,
                        "total": total,
                    },
                ),
            )
            if cancellation is not None and cancellation.cancelled:
                raise WorkspaceRefusal("model preparation canceled")
            if matched is not None:
                return Prepared(matched, defaults, access)
            installed = package_installation.open_installation(
                Path(worker.options.install_root or ""),
                target.installation_id,
            )
            with worker.preparation_lock, self.phase(call, request, "Checking model compatibility"):

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
