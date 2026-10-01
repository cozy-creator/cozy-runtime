"""Prepare captured inference bindings asynchronously, then use the existing attempt lanes."""

from __future__ import annotations

import base64
import contextlib
import hashlib
import threading
import time
import uuid
from collections.abc import Callable, Hashable, Iterator, Mapping
from concurrent.futures import Future
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import msgspec

from cozy_runtime import canonical_json
from cozy_runtime.author._artifacts import ModelArtifact, ObjectRef
from cozy_runtime.author._calls import MAX_ACTIVE_CALLS, MAX_CALL_BYTES
from cozy_runtime.author._executor_requests import Answer, ModelPrefetch
from cozy_runtime.internal import fill, package_installation
from cozy_runtime.internal.canonical import Json
from cozy_runtime.protocol import documents
from cozy_runtime.protocol import worker_pb2 as pb

from . import (
    grants,
    machine_builtin,
    machine_byte_inputs,
    machine_byte_results,
    machine_capture,
    machine_checkpoint_inputs,
    machine_child_target,
    machine_lanes,
    machine_model_defaults,
    machine_model_overrides,
    machine_models,
    package_prepare,
    store_gc,
)
from .machine_child_target import Target
from .preparations import Preparations
from .stage_progress import Emit, StageProgress
from .supervisor import Unit
from .workspace import WorkspaceRefusal
from .workspace_calls import Call
from .workspace_calls import Calls as JournalCalls

if TYPE_CHECKING:
    from .attempts import AttemptRecord
    from .session import Worker


def arguments(target: Target, raw: bytes) -> tuple[bytes, dict[str, ModelArtifact]]:
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
        return canonical_json.encode(payload), _models(value, names)
    # Envelope fields another generator adds are ignored; absent models select defaults.
    models = value.get("models", {})
    if (
        not isinstance(value.get("payload"), dict)
        or not isinstance(models, dict)
        or not set(models) <= names
    ):
        raise WorkspaceRefusal("generated serving call has invalid payload or Model slots")
    return canonical_json.encode(value["payload"]), _models(models, names)


def _models(values: dict[str, Json], names: set[str]) -> dict[str, ModelArtifact]:
    return {
        name: msgspec.convert(value, type=ModelArtifact, strict=True)
        for name, value in values.items()
        if name in names and value is not None
    }


class _Default(msgspec.Struct, frozen=True):
    """A selected Model default (`machine_model_defaults.select`), as preparation reads it."""

    parameter: str
    repository: str
    manifest: ObjectRef
    adapters: tuple[package_prepare.AdapterSelection, ...] = ()
    composed: package_prepare.CheckpointSelection | None = None


class _Cancellation(Protocol):
    @property
    def cancelled(self) -> bool: ...

    def cancel(self) -> None: ...


@dataclass
class Prepared:
    #: the prepared `Placement` document
    placement: dict[str, object]
    #: `machine_model_defaults.select`'s rows, which only its `inputs` reads
    defaults: list[dict[str, object]]
    access: contextlib.ExitStack
    generation: int = 0  # `store_gc.generation` when its bytes were verified


class _Compatibility:
    """The model compatibility check of a fresh preparation (run 1510). It needs each
    checkpoint's head (manifest, header and asset segments) and the installation, never the
    weight bytes, so it runs as soon as the heads are held, beside the rest of the download."""

    def __init__(
        self,
        check: Callable[[], pb.PreparePackageSetResult],
        heads: list[str],
        store_root: Path,
        cancel: Callable[[], None],
    ) -> None:
        self.check, self.heads, self.store_root, self.cancel = check, heads, store_root, cancel
        self.changed = threading.Condition()
        self.landed = self.stopped = False
        self.result: pb.PreparePackageSetResult | None = None
        self.error: BaseException | None = None
        self.thread = threading.Thread(target=self._run, name="model-compatibility", daemon=True)
        self.thread.start()

    def poke(self, landed: bool = False) -> None:
        """New bytes landed (or all of them): look at the heads again."""
        with self.changed:
            self.landed = self.landed or landed
            self.changed.notify_all()

    def _held(self) -> bool:
        tensorfs, store = fill.tensorfs_module(), fill.store(self.store_root)
        for digest in self.heads:
            try:
                header = store.manifest(digest)["header"]
                if header is None:
                    return False
                segments = sorted(
                    {
                        ("sha256:" + str(row["sha256"]), int(row["length"]))
                        for asset in tensorfs.parse_header(header).get("assets", {}).values()
                        for row in asset.get("segments", [])
                    }
                )
                if segments:
                    store.acquire(digest, segments).release()
            except Exception:  # not landed yet
                return False
        return True

    def _run(self) -> None:
        try:
            with self.changed:
                while not (self.stopped or self.landed or self._held()):
                    self.changed.wait()
                if self.stopped:
                    return
            self.result = self.check()
        except BaseException as exc:
            self.error = exc
            self.cancel()  # the download cannot be used: stop it

    def stop(self) -> BaseException | None:
        """End a check the download did not complete for: a waiting one never runs, a running
        one finishes. Its own refusal, if it failed, is why the download stopped."""
        with self.changed:
            self.stopped = True
            self.changed.notify_all()
        self.thread.join()
        return self.error

    def join(self) -> pb.PreparePackageSetResult:
        self.poke(landed=True)
        self.thread.join()
        if self.error is not None:
            raise self.error
        assert self.result is not None
        return self.result


class Serving:
    def __init__(self, worker: Worker):
        self.worker = worker
        self.preparations = Preparations[Prepared](MAX_ACTIVE_CALLS, changed=self._changed)
        #: call units waiting for a free preparation entry; any completion pokes them
        self.waiting: set[str] = set()
        # A preparation with no parent execution (a release root) reports here instead.
        self.observers: dict[str, Emit] = {}
        #: the tenth of its download each prefetch last narrated, until the download lands
        self.narrated: dict[str, int] = {}
        #: the parent each paused preparation's transfer yielded to, by its source request
        self.paused: dict[str, str] = {}
        # Read leases die with the old process, but durable temporary holds need
        # release through the existing journal after a worker restart.
        workspace = worker.workspace
        if workspace is not None:
            with workspace.locked() as db:
                abandoned = db.execute(
                    "SELECT DISTINCT owner,recipient FROM execution_model_holds "
                    "WHERE path LIKE 'preparation/%'"
                ).fetchall()
            for row in abandoned:
                self._release_preparation(row["owner"], row["recipient"])

    def _release_preparation(self, owner: str, recipient: str) -> None:
        workspace = self.worker.workspace
        assert workspace is not None
        machine_models.release(workspace, owner, recipient)
        with workspace.locked() as db:
            db.execute(
                "DELETE FROM execution_model_holds WHERE owner=? AND recipient=? "
                "AND path LIKE 'preparation/%'",
                (owner, recipient),
            )

    def _progress(self, call: Call, request: pb.ChildCallRequest, frame: dict[str, object]) -> None:
        if (observer := self.observers.get(call.child_request)) is not None:
            observer(frame)
            return
        if call.call_index == (1 << 32) - 1 and frame.get("kind") == "progress":
            # A prefetch narrates on its parent's durable log, so once per hundredth of its
            # bytes: every position filled 4,065 of run 1510's 4,096 events and hid its
            # segments, and a tenth left a watcher a minute and a half between readings.
            total, position = frame.get("total"), frame.get("position", 0)
            if isinstance(total, int) and isinstance(position, int) and total > 0:
                hundredth = position * 100 // total
                if hundredth <= self.narrated.get(call.child_request, -1):
                    return
                self.narrated[call.child_request] = hundredth
                if hundredth >= 100:
                    self.narrated.pop(call.child_request, None)
            frame = {
                "kind": "log",
                "name": "model-prefetch",
                "value": "info",
                "fields": {key: value for key, value in frame.items() if key != "kind"},
            }
        # Optional shared observations can outlive their initiating parent while
        # another parent still needs the same work. They cannot abort that work.
        with contextlib.suppress(WorkspaceRefusal, OSError):
            self.worker.emit_progress(call.parent_request, request.parent_attempt_ordinal, frame)

    def _stage(self, call: Call, request: pb.ChildCallRequest, name: str, scope: str = "") -> str:
        """`name` under its call's step label (else `scope`): one line per step for clients."""
        label = self.worker.calls.progress_label(
            call.parent_request, request.parent_attempt_ordinal, call.call_index
        )
        return f"{label or scope} / {name}"[:120] if label or scope else name

    @contextlib.contextmanager
    def phase(
        self, call: Call, request: pb.ChildCallRequest, name: str, since: float = 0.0
    ) -> Iterator[None]:
        """Separate immutable preparation and per-child custody in the parent journal. `since`
        (a `perf_counter` reading) starts the timing at work done before the phase opened."""
        started = since or time.perf_counter()
        self._progress(
            call, request, {"kind": "progress", "stage": self._stage(call, request, name)}
        )
        succeeded = False
        try:
            yield
            succeeded = True
        finally:
            self._progress(
                call,
                request,
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

    @contextlib.contextmanager
    def fetching(
        self,
        call: Call,
        request: pb.ChildCallRequest,
        target: Target,
        row: machine_model_defaults.Selected,
    ) -> Iterator[Callable[[int, int], None]]:
        """One model's pull, on its parent's durable log: a record when it starts, naming the
        model and the step it blocks (none for a prefetch), and one when it ends, with its
        bytes, time and rate. A call that waits on it says so itself (`holding`)."""
        prefetch = call.call_index == (1 << 32) - 1
        step = (
            ""
            if prefetch
            else self.worker.calls.progress_label(
                call.parent_request, request.parent_attempt_ordinal, call.call_index
            )
        )
        started, started_ms = time.perf_counter(), int(time.time() * 1000)
        landed: list[int] = []  # first sample, latest sample, total

        def sample(available: int, total: int) -> None:
            landed[:] = [landed[0] if landed else available, available, total]

        def record(fields: dict[str, object]) -> None:
            self._progress(
                call,
                request,
                {
                    "kind": "log",
                    "name": "model fetch",
                    "value": "info",
                    "at_unix_ms": int(time.time() * 1000),
                    "fields": {
                        "model": row.repository,
                        "manifest": row.manifest.digest,
                        "entrypoint": target.entrypoint,
                        "prefetch": prefetch,
                        "step": step,
                        "started_unix_ms": started_ms,
                        **fields,
                    },
                },
            )

        record({"event": "start"})
        failure: BaseException | None = None
        try:
            with self.phase(call, request, "Downloading model weights"):
                yield sample
        except BaseException as error:
            failure = error
            raise
        finally:
            elapsed = time.perf_counter() - started
            fields: dict[str, object] = {
                "event": "end",
                "completed": failure is None,
                "elapsed_ms": round(elapsed * 1000, 3),
            }
            if failure is not None:
                fields["reason"] = str(failure) or type(failure).__name__
                if call.child_request in self.paused:
                    fields["paused_for"] = self.paused[call.child_request]
            if landed:
                first, latest, total = landed
                fields.update(bytes=latest, total_bytes=total, moved_bytes=latest - first)
                if elapsed > 0:
                    fields["rate_bytes_per_second"] = round((latest - first) / elapsed, 1)
            record(fields)

    def holding(
        self, call: Call, request: pb.ChildCallRequest, target: Target
    ) -> Callable[[], None]:
        """A call held for its callee's model preparation (a download, then its check): the
        step it runs in, when it started waiting and, from the returned callable, for how
        long, so the step is timed without the wait."""
        step = self.worker.calls.progress_label(
            call.parent_request, request.parent_attempt_ordinal, call.call_index
        )
        started = time.perf_counter()

        def record(fields: dict[str, object]) -> None:
            self._progress(
                call,
                request,
                {
                    "kind": "log",
                    "name": "model wait",
                    "value": "info",
                    "at_unix_ms": int(time.time() * 1000),
                    "fields": {
                        "step": step,
                        "call": call.child_request,
                        "entrypoint": target.entrypoint,
                        **fields,
                    },
                },
            )

        record({"event": "start"})
        return lambda: record(
            {"event": "end", "waited_ms": round((time.perf_counter() - started) * 1000, 3)}
        )

    def close(self) -> None:
        self.preparations.close()

    def prefetch(self, parent: AttemptRecord, hint: ModelPrefetch) -> Answer:
        """A model-only hint; never accepts a child execution or chooses a GPU lane."""
        owner = self.worker.fence.record_owner_id
        try:
            target = machine_child_target.resolve_export(
                self.worker, parent, hint.module, hint.export, wait=False
            )
        except machine_builtin.Preparing:
            return Answer(ok=True)  # the callee installs on its call
        if target.binding is not None:
            raise WorkspaceRefusal("model prefetch requires a serving callable")
        raw = hint.payload.encode()
        if len(raw) > MAX_CALL_BYTES:
            raise WorkspaceRefusal("model prefetch exceeds its inline bound")
        selected = canonical_json.decode_as(raw, dict[str, ModelArtifact | None])
        names = {row["path"].rpartition(".")[2] for row in target.declaration.get("models", [])}
        if not selected.keys() <= names:
            raise WorkspaceRefusal("model prefetch names an undeclared Model slot")
        models = {name: artifact for name, artifact in selected.items() if artifact is not None}
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
            self._shared(owner, call, target, request, models, speculative=True)
        except WorkspaceRefusal as error:
            self.worker.note("model-prefetch", str(error))
        return Answer(ok=True)

    def _changed(self) -> None:
        for key in tuple(self.waiting):
            self.worker.supervisor.poke(key)

    def forget(self, parent: str, ordinal: int) -> None:
        """A parent attempt ended: the preparations only it wanted stop."""
        self.preparations.retain(lambda interest: interest != (parent, ordinal))

    def _shared(
        self,
        owner: str,
        call: Call,
        target: Target,
        request: pb.ChildCallRequest,
        models: dict[str, ModelArtifact],
        *,
        speculative: bool,
    ) -> Future[Prepared] | None:
        choices = machine_model_overrides.captured(self.worker, owner, call, target)
        defaults = machine_model_defaults.select(
            self.worker, owner, call, target, models, model_choices=choices
        )
        identity = hashlib.sha256(
            canonical_json.encode(
                {
                    "owner": owner,
                    "installation": target.installation_id,
                    "interface": target.prepared_installation["placement"]["package_interface"],
                    "entrypoint": target.entrypoint,
                    "models": msgspec.to_builtins(models),
                    "defaults": defaults,
                    "overrides": {name: documents.body(choice) for name, choice in choices.items()},
                }
            )
        ).hexdigest()
        source = replace(
            call,
            child_request="model-preparation." + uuid.uuid4().hex,
            call_index=(1 << 32) - 1 if speculative else call.call_index,
        )
        tensorfs = fill.tensorfs_module()
        cancellation_lock = threading.Lock()
        active_token: _Cancellation | None = None

        def warm() -> None:
            self.worker.prespawns.request(
                owner,
                call.parent_request,
                target,
                lambda frame: self._progress(call, request, frame),
            )

        if not speculative:
            # The immutable binding may be cached after its old executor was trimmed.
            # A real call still warms its process while awaiting admission.
            warm()

        def stop(demand: Hashable | None) -> None:
            with cancellation_lock:
                if active_token is not None:
                    if isinstance(demand, tuple):  # (parent request, attempt)
                        self.paused[source.child_request] = str(demand[0])
                    active_token.cancel()

        def load(cancelled: Callable[[], bool]) -> Prepared:
            nonlocal active_token
            # A paused hint resumes with a fresh native cancellation token. The store
            # retains completed bytes; only this attempt's transfer and holds ended.
            token = tensorfs.PullCancellation()
            with cancellation_lock:
                active_token = token
                if cancelled():
                    token.cancel()
            generation = store_gc.generation
            try:
                if cancelled():
                    raise WorkspaceRefusal("model preparation canceled")
                # Warming belongs to scheduled work too: a queued hint must not consume
                # CPU preparing an executor yet.
                if speculative:
                    warm()
                prepared = self._prepare(
                    owner,
                    source,
                    target,
                    request,
                    model_arguments=models,
                    defaults=defaults,
                    cancellation=token,
                    retention_prefix="preparation/",
                    model_choices=choices,
                )
                # Consumers acquire their own holds at actual admission. Shared
                # preparation retains only immutable binding metadata on completion.
                prepared.access.close()
                prepared.generation = generation
                return prepared
            finally:
                with cancellation_lock:
                    active_token = None
                self.paused.pop(source.child_request, None)
                assert self.worker.workspace is not None
                self._release_preparation(owner, source.child_request)

        return self.preparations.request(
            identity,
            (call.parent_request, request.parent_attempt_ordinal),
            load,
            speculative=speculative,
            cancel=stop,
            # A hint's exact selection serves its demand unless the store collected since.
            stale=lambda prepared: prepared.generation != store_gc.generation,
        )

    def submit(
        self, unit: Unit, owner: str, call: Call, target: Target, request: pb.ChildCallRequest
    ) -> bool:
        """Wait for this call's shared preparation, then submit its serving child. True once
        submitted; False when its parent stopped waiting first."""
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
                        "installation_id": target.installation_id,
                        "entrypoint": target.entrypoint,
                        "parent_attempt": request.parent_attempt_ordinal,
                        "result_schema": target.declaration["result"],
                    }
                ),
            )
        payload, models = arguments(target, request.request_canonical_bytes)

        def waiting() -> bool:
            parent = self.worker.engine.live.get(call.parent_request)
            return (
                parent is None
                or parent.attempt != request.parent_attempt_ordinal
                or parent.state != "running"
                or bool(parent.canceling)
            )

        self.waiting.add(unit.key)
        try:
            # Every preparation entry is busy: the next one to finish pokes this unit.
            while (
                future := self._shared(owner, call, target, request, models, speculative=False)
            ) is None:
                if waiting():
                    return False
                unit.wait()
        finally:
            self.waiting.discard(unit.key)
        future.add_done_callback(lambda _: unit.poke())
        waited = None if future.done() else self.holding(call, request, target)
        try:
            if not unit.wait_for(lambda: future.done() or waiting()) or not future.done():
                return False
        finally:
            if waited is not None:
                waited()
        prepared = None
        try:
            prepared = future.result()
            placement = documents.from_body(prepared.placement, pb.Placement)
            # The durable row holds the placement template; its devices are chosen at grant.
            for name, supplied in models.items():
                machine_models.retain(
                    workspace, owner, call.child_request, "input/" + name, supplied
                )
            journal.check_active(owner, call)
            selected = next(
                (entry for entry in placement.entrypoints if entry.name == target.entrypoint), None
            )
            if selected is None:
                raise WorkspaceRefusal("captured inference has no complete prepared Model binding")
            binding = documents.spell(selected.entrypoint_binding_digest)
            digest = documents.spell(hashlib.sha256(payload).digest())
            with workspace.locked() as db:
                row = db.execute(
                    "SELECT invocation FROM attempts WHERE owner=? AND request=? AND ordinal=?",
                    (owner, call.parent_request, request.parent_attempt_ordinal),
                ).fetchone()
            parent = documents.parse(row["invocation"], pb.InvocationSpec)
            spec = pb.InvocationSpec(
                installation_id=placement.installation_id,
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
                deadline_unix_ms=parent.deadline_unix_ms,
                attention_kernel=parent.attention_kernel,
                serving=pb.ServingInvocationSpec(
                    entrypoint_binding_digest=binding,
                    attempt_binding_id=binding,
                    bindings_digest=documents.spell(placement.bindings_digest),
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
            # Custody of the call's byte inputs is part of checking them: one durable hold per
            # input, six per H3 segment. Run 1516 showed it only as a gap before the phase.
            checking = time.perf_counter()
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
                placement_id=placement.placement_id,
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
                {"format": "cozy.worker.v1.PlacementSet/1", "placements": [prepared.placement]}
            )
            desired = pb.DesiredWorkerState(
                wire_minor=executions.wire_minor(owner, call.parent_request),
                posture=pb.POSTURE_ACCEPTING,
                placement_set=pb.DesiredPlacementSet(
                    placement_set_digest=hashlib.sha256(placement_bytes).digest(),
                    placement_set_canonical_bytes=placement_bytes,
                ),
            )
            preparation = canonical_json.encode(
                {
                    "installations": {
                        target.installation_id: {
                            "placement": prepared.placement,
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
            machine_lanes.admit(self.worker, call.parent_request, preparation, offer)
            with (
                self.phase(call, request, "Checking model inputs", checking),
                machine_checkpoint_inputs.admission(workspace, owner, offer.request_id),
            ):
                catalog = grants.model_inputs(
                    grants.bind(documents.read(raw, pb.InvocationSpec), offer.grant, identity)
                )
                machine_checkpoint_inputs.preflight(
                    workspace, owner, offer, catalog, held=self.worker.held_manifests
                )
                for entry in catalog.values():
                    machine_checkpoint_inputs.retain(workspace, owner, offer, entry)
                executions.submit(
                    owner,
                    call.child_request,
                    hashlib.sha256(capture).digest(),
                    offer,
                    expected_execution_workspace_id=executions.workspace_id,
                    result_schema=canonical_json.encode(target.declaration["result"]),
                    worker_boot=self.worker.fence.worker_boot_id,
                    memoize=False,
                    preparation=preparation,
                    worker_id=self.worker.options.worker_id,
                )
        finally:
            if prepared is not None:
                prepared.access.close()
        return True

    def _prepare(
        self,
        owner: str,
        call: Call,
        target: Target,
        request: pb.ChildCallRequest,
        *,
        model_arguments: dict[str, ModelArtifact] | None = None,
        defaults: list[dict[str, object]] | None = None,
        cancellation: _Cancellation | None = None,
        retention_prefix: str = "input/",
        model_choices: Mapping[str, pb.ModelChoice] | None = None,
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
            if model_choices is None:
                model_choices = machine_model_overrides.captured(worker, owner, call, target)
            with self.phase(call, request, "Selecting model"):
                defaults = machine_model_defaults.select(
                    worker, owner, call, target, model_arguments, model_choices=model_choices
                )
        access = contextlib.ExitStack()
        try:
            for model in target.declaration["models"]:
                prefix = target.entrypoint + ".models."
                path = model["path"]
                if not path.startswith(prefix) or not path.removeprefix(prefix).isidentifier():
                    raise WorkspaceRefusal("captured serving Model has no exact parameter")
                artifact = model_arguments.get(path.removeprefix(prefix))
                if artifact is None:
                    continue
                retention, held = machine_models.retain(
                    workspace,
                    owner,
                    call.child_request,
                    retention_prefix + path.removeprefix(prefix),
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
            if model_choices:
                hub, account, credentials = machine_model_overrides.context(worker, owner, call)
                package = str((prepared.get("package") or prepared["development"])["package"])

                def check_adapter_preparation() -> None:
                    if worker.stop.is_set() or (
                        cancellation is not None and cancellation.cancelled
                    ):
                        raise WorkspaceRefusal("model preparation canceled")

                selected_defaults = {str(row["parameter"]): dict(row) for row in defaults}
                for parameter, choice in model_choices.items():
                    if not choice.adapters:
                        continue
                    original = next(
                        (
                            row
                            for row in native
                            if row.slot == target.entrypoint + ".models." + parameter
                        ),
                        None,
                    )
                    if original is not None:
                        base: dict[str, Json] = {
                            "parameter": parameter,
                            "repository": original.model,
                            "manifest": documents.body(original.manifest),
                            "native": True,
                        }
                    else:
                        base = selected_defaults[parameter]
                    selected_defaults[parameter] = machine_model_overrides.apply(
                        worker,
                        base,
                        choice,
                        package=package,
                        hub=hub,
                        owner=account,
                        credentials=credentials,
                        note=lambda *_: None,
                        check=check_adapter_preparation,
                        cancellation=cancellation,
                        native_base=original is not None,
                    )
                    if original is not None:
                        native.remove(original)
                defaults = list(selected_defaults.values())
            selected = [msgspec.convert(row, _Default, strict=True) for row in defaults]

            # The existing immutable preparation registry already owns this work.
            # Reusing its exact checkpoint bindings does not reuse an inference result.
            expected = {
                model.slot.removeprefix(target.entrypoint + ".models."): documents.body(
                    model.manifest
                )
                for model in native
            }
            expected.update({row.parameter: msgspec.to_builtins(row.manifest) for row in selected})
            components = {
                row.parameter: row.composed.manifest
                if row.composed is not None
                else row.manifest.digest
                for row in selected
            }
            components.update(
                {
                    parameter: str(manifest["digest"])
                    for parameter, manifest in expected.items()
                    if parameter not in components
                }
            )
            adapters = {
                row.parameter: [msgspec.to_builtins(adapter) for adapter in row.adapters]
                for row in selected
            }
            with worker.control_lock:
                candidates = [
                    documents.body(value.document)
                    for value in worker.prepared_installations.values()
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
                models = {row["id"]: row for row in candidate.get("models", [])}
                if entry is None:
                    continue
                used = {
                    identifier
                    for slot in entry["slots"]
                    for identifier in (
                        slot["reference_model_id"],
                        *(row["model_id"] for row in slot.get("components", [])),
                        *(row["model_id"] for row in slot.get("adapters", [])),
                    )
                }
                if used != models.keys() or any(
                    models[component["model_id"]]["manifest"]["digest"]
                    != components.get(slot["slot"])
                    for slot in entry["slots"]
                    for component in slot.get("components", [])
                ):
                    continue
                observed_adapters = {
                    slot["slot"]: [
                        {
                            "component": row["component"],
                            "model": models[row["model_id"]]["repo"],
                            "manifest": models[row["model_id"]]["manifest"]["digest"],
                            "release": models[row["model_id"]].get("version", ""),
                            "lane": models[row["model_id"]].get("lane", ""),
                            "source_component": row.get("source_component", "adapter"),
                            "scale": row.get("scale", "1"),
                        }
                        for row in slot.get("adapters", [])
                    ]
                    for slot in entry["slots"]
                }
                if any(
                    observed_adapters[slot] != adapters.get(slot, []) for slot in observed_adapters
                ):
                    continue
                if {
                    slot["slot"]: models[slot["reference_model_id"]]["manifest"]
                    for slot in entry["slots"]
                } == expected:
                    matched = candidate
                    break
            check: _Compatibility | None = None
            if matched is None:
                installed = package_installation.open_installation(
                    Path(worker.options.install_root or ""),
                    target.installation_id,
                )
                if cancellation is None:
                    cancellation = fill.tensorfs_module().PullCancellation()

                def verified(digest: str, length: int, path: Path) -> None:
                    worker.verified_artifacts.record(digest, length, path)

                def compatible() -> pb.PreparePackageSetResult:
                    with (
                        worker.preparation_lock,
                        self.phase(call, request, "Checking model compatibility"),
                    ):
                        native.sort(key=lambda model: model.slot)
                        return package_prepare.prepare_model_placement(
                            native_models=native,
                            models=[
                                {
                                    "package": (prepared.get("package") or prepared["development"])[
                                        "package"
                                    ],
                                    "slot": target.entrypoint + ".models." + row.parameter,
                                    "model": row.repository,
                                    "manifest": row.manifest.digest,
                                    "manifest_length": row.manifest.length,
                                    "adapters": msgspec.to_builtins(row.adapters),
                                    "composed": msgspec.to_builtins(row.composed),
                                }
                                for row in selected
                            ],
                            prepared=documents.from_body(prepared, pb.Placement),
                            installed=installed,
                            artifact_cache=Path(worker.options.artifact_cache or ""),
                            tensorfs_root=workspace.store_root,
                            verified=verified,
                        )

                check = _Compatibility(
                    compatible,
                    [row.manifest.digest for row in selected],
                    workspace.store_root,
                    cancellation.cancel,
                )

            # The download's bytes and rate, once a second, on the ordinary progress lane.
            stage: list[StageProgress] = []

            def progress(available: int, total: int) -> None:
                if not stage:
                    stage.append(
                        StageProgress(
                            self._stage(
                                call, request, "Downloading model weights", target.entrypoint
                            ),
                            lambda frame: self._progress(call, request, frame),
                        )
                    )
                stage[0].update(available, total)
                if check is not None:
                    check.poke()

            try:
                machine_model_defaults.materialize(
                    worker,
                    defaults,
                    downloading=lambda row: self.fetching(call, request, target, row),
                    cancellation=cancellation,
                    progress=progress,
                    waiting=lambda text: self._progress(
                        call,
                        request,
                        {
                            "kind": "progress",
                            "stage": self._stage(call, request, text, target.entrypoint),
                        },
                    ),
                )
                if cancellation is not None and cancellation.cancelled:
                    raise WorkspaceRefusal("model preparation canceled")
            except BaseException:
                if check is not None and (failed := check.stop()) is not None:
                    raise failed from None
                raise
            if check is None:
                assert matched is not None
                return Prepared(matched, defaults, access)
            result = check.join()
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
