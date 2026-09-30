"""Stateless transport of exact admitted parent calls to their RecordOwner."""

from __future__ import annotations

import contextlib
import hashlib
import socket
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import msgspec

from cozy_runtime.author._calls import MAX_ACTIVE_CALLS, MAX_CALL_BYTES
from cozy_runtime.author._errors import InvalidRequest
from cozy_runtime.author._executor_requests import (
    Answer,
    ByteGrant,
    ByteMetadata,
    CallProgress,
    CallRequest,
    CallState,
    ChildCall,
    ChildCancel,
    ChildEvents,
    ChildForget,
    DescriptorReply,
    GpuRelease,
    ModelPrefetch,
    Reply,
    refuse,
)
from cozy_runtime.internal import (
    builtin_operations,
    canonical,
    native_interfaces,
    output_budget,
    package_interface,
)
from cozy_runtime.internal.call_intent import canonical_intent
from cozy_runtime.internal.capture_observation import document as observation_document
from cozy_runtime.internal.executor_commands import CallInterface
from cozy_runtime.internal.executor_replies import OutputRow
from cozy_runtime.internal.seam import MAX_FRAME
from cozy_runtime.internal.worker import byte_inputs, child_byte_results
from cozy_runtime.internal.worker.workspace import Workspace
from cozy_runtime.protocol import documents
from cozy_runtime.protocol import worker_pb2 as pb

if TYPE_CHECKING:
    from cozy_runtime.internal.worker.attempts import AttemptRecord


class CapabilityUnavailable(ValueError):
    """The executor expects a native operation this worker cannot serve."""

    code = "child_capability_unavailable"


@dataclass(slots=True)
class _Pending:
    request: pb.ChildCallRequest
    result: pb.ChildCallResult
    byte_grants: list[ByteGrant] | None = None
    spool: Path | None = None
    progress_label: str = ""


class _ProgressEvent(msgspec.Struct, frozen=True):
    """A journaled progress event; its payload is the child's own narration."""

    payload: dict[str, canonical.Json]


def _framed(answer: CallState) -> int:
    """The answer's control-frame size, with room for its envelope."""
    return len(msgspec.json.encode(answer)) + 128


@dataclass(frozen=True, slots=True)
class _Received:
    metadata: ByteMetadata
    source: pb.NativeByteRetentionRequest


class Calls:
    def __init__(
        self,
        call: Callable[[AttemptRecord, pb.ChildCallRequest, str], pb.ChildCallResult],
        *,
        workspace: Workspace | None = None,
        owner: Callable[[], str] = lambda: "",
        installed_bindings: Callable[[AttemptRecord], list[dict[str, Any]]] | None = None,
        prefetch: Callable[[AttemptRecord, ModelPrefetch], Answer] | None = None,
        release_gpus: Callable[[AttemptRecord], None] | None = None,
    ):
        self.prefetch = prefetch
        self.release_gpus = release_gpus
        self.installed_bindings = installed_bindings
        self.call = call
        self.workspace, self.owner = workspace, owner
        self.pending: dict[tuple[str, int, int], _Pending] = {}
        self.materialized: dict[tuple[str, int], tuple[Path, int]] = {}
        self.received: dict[tuple[str, int], dict[str, _Received]] = {}
        #: per parent attempt: the socket a nudge is written to when its calls may have moved
        self.watchers: dict[tuple[str, int], socket.socket] = {}
        self.lock = threading.RLock()
        #: one lock per parent attempt: parents never wait on each other's calls
        self.parents: dict[tuple[str, int], threading.RLock] = {}

    def bindings(self, attempt: AttemptRecord) -> tuple[CallInterface, ...]:
        if attempt.job is None:
            return ()
        result = [dict(row) for row in attempt.job.call_interfaces]
        if self.installed_bindings is not None:
            captured = self.installed_bindings(attempt)
            selected = {(row["module"], row["export"]) for row in captured}
            # Installed metadata inventories imports; captured bindings determine
            # their execution role. A full implementation wheel also embeds its
            # own exports, which must retain their captured self-call binding.
            result = [row for row in result if (row["module"], row["export"]) not in selected]
            result.extend(captured)
        executor = attempt.executor
        result.extend(native_interfaces.rows(executor.hello if executor is not None else {}))
        if builtin_operations.supports_environment(attempt.job.python):
            result.extend(builtin_operations.rows())
        raw = Path(attempt.job.package_interface).read_bytes()
        interface = package_interface.read_bytes(raw)
        for entry in [*interface["jobs"], *interface["entrypoints"]]:
            declaration = entry.get("invocable")
            if declaration is not None and not any(
                (row["module"], row["export"]) == (declaration["module"], declaration["export"])
                for row in result
            ):
                # Published SDK wheels already carry their exact generated request
                # convention. Do not replace that binding with a raw self-call shape.
                result.append(
                    {
                        "module": declaration["module"],
                        "export": declaration["export"],
                        "interface_path": attempt.job.package_interface,
                        "self": True,
                    }
                )
        return msgspec.convert(result, tuple[CallInterface, ...])

    def handle(
        self,
        attempt: AttemptRecord,
        request: CallRequest,
    ) -> Reply:
        if isinstance(request, GpuRelease):
            # Worker-local and always safe: no owner capability, and a canceling attempt
            # may release too.
            try:
                if self.release_gpus is not None:
                    self.release_gpus(attempt)
            except (OSError, ValueError) as exc:
                return refuse("gpu_release_refused", str(exc))
            return Answer(ok=True)
        with self.lock:
            parent_lock = self.parents.setdefault(
                (attempt.request_id, attempt.attempt), threading.RLock()
            )
        with parent_lock:
            try:
                if attempt.state != "running" or attempt.canceling:
                    raise ValueError("parent call is not active")
                if isinstance(request, ModelPrefetch):
                    if self.prefetch is None:
                        return Answer(ok=True)
                    return self.prefetch(attempt, request)
                if isinstance(request, ChildEvents):
                    return self._watch(attempt)
                key = (attempt.request_id, attempt.attempt, request.call_index)
                pending = self.pending.get(key)
                if isinstance(request, ChildForget):
                    if pending is not None and pending.result.state != pb.CHILD_CALL_STATE_PENDING:
                        del self.pending[key]
                    return Answer(ok=True)
                if isinstance(request, ChildCall):
                    intent = self._request(attempt, request)
                    if (
                        pending is not None
                        and pending.request.intent_digest != intent.intent_digest
                    ):
                        raise ValueError("an existing parent call index changed its exact intent")
                    if pending is None:
                        if sum(k[:2] == key[:2] for k in list(self.pending)) >= MAX_ACTIVE_CALLS:
                            raise ValueError("active parent call bound exceeded")
                        label = request.progress_label
                        if len(label) > 120 or not label.isprintable():
                            label = ""
                        pending = self.pending[key] = _Pending(
                            intent, pb.ChildCallResult(), progress_label=label
                        )
                        try:
                            pending.result = self.call(attempt, intent, "call")
                        except BaseException:
                            self.pending.pop(key, None)
                            raise
                if pending is None:
                    raise ValueError("parent call has not been activated")
                canceling = isinstance(request, ChildCancel)
                pending.result = self.call(
                    attempt, pending.request, "cancel" if canceling else "poll"
                )
                if canceling:
                    return CallState(ok=True, state="pending")
                result = pending.result
                if result.state == pb.CHILD_CALL_STATE_PENDING:
                    return CallState(
                        ok=True,
                        state="pending",
                        child_request_id=result.child_request_id,
                        progress=self._progress(pending),
                    )
                observation = (
                    observation_document(result.observation)
                    if result.HasField("observation")
                    else None
                )
                if result.state != pb.CHILD_CALL_STATE_SUCCEEDED:
                    return CallState(
                        ok=False,
                        code=result.safe_code or "child_failed",
                        detail=result.safe_detail,
                        child_request_id=result.child_request_id,
                        progress=self._progress(pending),
                        observation=observation,
                    )
                if pending.byte_grants is None:
                    pending.byte_grants = self._materialize(attempt, pending)
                answer = CallState(
                    ok=True,
                    state="succeeded",
                    child_request_id=result.child_request_id,
                    result=result.result_canonical_bytes.decode(),
                    byte_grants=tuple(pending.byte_grants),
                    progress=self._progress(pending),
                    observation=observation,
                )
                if _framed(answer) > MAX_FRAME:
                    # Optional narration cannot turn a valid child result into a refusal.
                    answer = msgspec.structs.replace(answer, progress=None)
                    if _framed(answer) > MAX_FRAME:
                        raise ValueError("child result and native grants exceed the control frame")
                return answer
            except CapabilityUnavailable as exc:
                return refuse(exc.code, str(exc))
            except (
                OSError,
                ValueError,
                InvalidRequest,
                canonical.CanonicalError,
                documents.DocumentError,
            ) as exc:
                return refuse("child_call_refused", str(exc))

    def _watch(self, attempt: AttemptRecord) -> DescriptorReply:
        """The parent's wakeup: it awaits a nudge instead of polling on a clock."""
        ours, theirs = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
        ours.setblocking(False)
        previous = self.watchers.pop((attempt.request_id, attempt.attempt), None)
        if previous is not None:
            previous.close()
        self.watchers[(attempt.request_id, attempt.attempt)] = ours
        return DescriptorReply(Answer(ok=True), theirs)

    def wake(self, parent: str) -> None:
        """A call result, child progress or a cancel of `parent`'s may have moved: that
        parent re-polls. Lock-free, so any journal writer may call it; a nudge already
        queued is enough."""
        for (request, _), watcher in tuple(self.watchers.items()):
            if request == parent:
                with contextlib.suppress(OSError):
                    watcher.send(b"\0")

    def _progress(self, pending: _Pending) -> CallProgress | None:
        """Reuse the child's existing journal; polling progress does not change call identity."""
        result, workspace = pending.result, self.workspace
        if not result.child_request_id or workspace is None:
            return None
        with workspace.locked() as db:
            row = db.execute(
                "SELECT p.sequence,p.ordinal,p.body FROM execution_events p "
                "JOIN executions e ON e.owner=p.owner AND e.request=p.request "
                "AND e.ordinal=p.ordinal "
                "JOIN execution_calls c ON c.owner=e.owner AND c.child_request=e.request "
                "WHERE e.owner=? AND e.request=? AND c.parent_request=? AND c.parent_ordinal=? "
                "AND c.call_index=? AND p.kind='progress' ORDER BY p.sequence DESC LIMIT 1",
                (
                    self.owner(),
                    result.child_request_id,
                    pending.request.parent_request_id,
                    pending.request.parent_attempt_ordinal,
                    pending.request.call_index,
                ),
            ).fetchone()
        if row is None:
            return None
        try:
            event = msgspec.json.decode(row["body"], type=_ProgressEvent)
        except msgspec.ValidationError:
            return None  # progress is narration; the child's answer stands without it
        return CallProgress(
            sequence=row["sequence"],
            payload={
                "call_request": result.child_request_id,
                "call_attempt": row["ordinal"],
                **event.payload,
            },
        )

    def progress_label(self, parent: str, ordinal: int, index: int) -> str:
        with self.lock:
            pending = self.pending.get((parent, ordinal, index))
            return pending.progress_label if pending is not None else ""

    def _request(self, attempt: AttemptRecord, call: ChildCall) -> pb.ChildCallRequest:
        module, export = call.module, call.export
        allowed = next(
            (row for row in self.bindings(attempt) if (row.module, row.export) == (module, export)),
            None,
        )
        if allowed is None:
            raise ValueError(
                "call does not name a locked interface dependency or current App export"
            )
        if allowed.unavailable:
            raise CapabilityUnavailable(allowed.unavailable)
        payload = call.payload.encode()
        if len(payload) > MAX_CALL_BYTES:
            raise ValueError("call request exceeds its inline bound")
        if not isinstance(canonical.parse_canonical(payload), dict):
            raise ValueError("call request must be a canonical object")
        capture = call.capture
        if capture is not None and (allowed.native_source or allowed.native_effect):
            raise ValueError("activation capture requires an ordinary serving invocation")
        request = pb.ChildCallRequest(
            parent_request_id=attempt.request_id,
            parent_attempt_ordinal=attempt.attempt,
            parent_invocation_spec_digest=attempt.digest,
            call_index=call.call_index,
            module=module,
            export=export,
            request_canonical_bytes=payload,
            capture=pb.ActivationCapture(components=capture.components, steps=capture.steps)
            if capture is not None
            else None,
        )
        request.intent_digest = hashlib.sha256(canonical_intent(request)).digest()
        return request

    def _materialize(self, attempt: AttemptRecord, pending: _Pending) -> list[ByteGrant]:
        result = pending.result
        if not result.byte_result_grants:
            return []
        if self.workspace is None or attempt.spool is None:
            raise ValueError("parent has no native result workspace")
        parent = (attempt.request_id, attempt.attempt)
        spool, held_bytes = self.materialized.get(parent, (attempt.spool, 0))
        if spool != attempt.spool:
            raise ValueError("parent artifact spool changed")
        pending.spool = attempt.spool / "child-results" / str(result.call_index)
        try:
            granted = child_byte_results.materialize(
                self.workspace,
                self.owner(),
                result,
                attempt.spool,
                max_bytes=output_budget.intermediate(attempt.spec) - held_bytes,
            )
            received = dict(self.received.get(parent, {}))
            for metadata, grant in zip(granted, result.byte_result_grants, strict=True):
                identity = canonical.write(
                    [metadata["kind"], metadata["digest"], metadata["media_type"]]
                ).decode()
                received.setdefault(
                    identity,
                    _Received(
                        metadata.copy(),
                        pb.NativeByteRetentionRequest(
                            source=grant.source, retention_id=grant.retention_id
                        ),
                    ),
                )
            # Re-export retains metadata after call-forget, under the existing
            # bounded call inventory's aggregate control-byte allowance.
            if (
                sum(
                    len(canonical.write(value.metadata)) + value.source.ByteSize()
                    for value in received.values()
                )
                > MAX_ACTIVE_CALLS * MAX_CALL_BYTES
            ):
                raise ValueError("received native output metadata exceeds its bound")
            self.received[parent] = received
            self.materialized[parent] = (
                spool,
                held_bytes + sum(int(row["content_bytes"]) for row in granted),
            )
            return granted
        except Exception as exc:
            byte_inputs.remove_checkout(pending.spool)
            if isinstance(exc, (ValueError, OSError)):
                raise
            raise ValueError("native child artifact bytes are unavailable") from exc

    def received_inputs(
        self, attempt: AttemptRecord
    ) -> list[tuple[ByteMetadata, pb.NativeByteRetentionRequest]]:
        """Snapshot only native byte capabilities actually delivered to this attempt."""
        with self.lock:
            return [
                (
                    value.metadata.copy(),
                    pb.NativeByteRetentionRequest.FromString(
                        value.source.SerializeToString(deterministic=True)
                    ),
                )
                for value in self.received.get((attempt.request_id, attempt.attempt), {}).values()
            ]

    def received_output(
        self, attempt: AttemptRecord, row: OutputRow
    ) -> tuple[Path, pb.NativeByteRetentionRequest] | None:
        """Resolve an explicit returned handle only against this parent's actual grants."""
        ref = row.asset_ref
        if not ref.startswith("sha256:"):
            return None
        media = byte_inputs.TREE_MIME if row.kind == "tree" else row.media_type
        identity = canonical.write([row.kind, ref, media]).decode()
        with self.lock:
            grant = self.received.get((attempt.request_id, attempt.attempt), {}).get(identity)
            if (
                grant is None
                or row.digest != ref
                or row.size_bytes != grant.metadata["content_bytes"]
            ):
                raise ValueError("returned native handle has no exact received grant")
            return Path(grant.metadata["local"]), grant.source

    def close(self, request_id: str, attempt: int) -> None:
        with self.lock:
            self.parents.pop((request_id, attempt), None)
            if watcher := self.watchers.pop((request_id, attempt), None):
                watcher.close()
            self.received.pop((request_id, attempt), None)
            if held := self.materialized.pop((request_id, attempt), None):
                byte_inputs.remove_checkout(held[0] / "child-results")
            for key in list(self.pending):
                if key[:2] == (request_id, attempt):
                    if self.pending[key].spool is not None:
                        byte_inputs.remove_checkout(self.pending[key].spool)
                    del self.pending[key]
