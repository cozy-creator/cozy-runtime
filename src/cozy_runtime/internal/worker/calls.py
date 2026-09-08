"""Stateless transport of exact admitted parent calls to their RecordOwner."""

from __future__ import annotations

import hashlib
import json
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import msgspec

from cozy_runtime.author._calls import MAX_CALL_BYTES, MAX_DIRECT_CALLS
from cozy_runtime.author._capture import ActivationCapture
from cozy_runtime.author._errors import InvalidRequest
from cozy_runtime.internal import canonical, package_interface
from cozy_runtime.internal.capture_observation import document as observation_document
from cozy_runtime.internal.seam import MAX_FRAME
from cozy_runtime.internal.worker import byte_inputs, child_byte_results
from cozy_runtime.internal.worker.workspace import Workspace
from cozy_runtime.protocol import documents
from cozy_runtime.protocol import worker_pb2 as pb

if TYPE_CHECKING:
    from cozy_runtime.internal.worker.attempts import AttemptRecord


@dataclass(slots=True)
class _Pending:
    request: pb.ChildCallRequest
    stream_epoch: int
    result: pb.ChildCallResult | None = None
    byte_grants: list[dict[str, Any]] | None = None
    spool: Path | None = None


class Calls:
    def __init__(
        self,
        send: Callable[[pb.WorkerFrame], None],
        stream: Callable[[], int],
        minor: Callable[[], int],
        *,
        workspace: Workspace | None = None,
        owner: Callable[[], str] = lambda: "",
    ):
        self.send, self.stream, self.minor = send, stream, minor
        self.workspace, self.owner = workspace, owner
        self.pending: dict[tuple[str, int, int], _Pending] = {}
        self.lock = threading.RLock()

    @staticmethod
    def bindings(attempt: AttemptRecord) -> list[dict[str, Any]]:
        if attempt.job is None:
            return []
        from cozy_runtime.internal import effect_interfaces, source_interfaces

        result = [dict(row) for row in attempt.job.call_interfaces]
        result.extend(source_interfaces.rows())
        result.extend(effect_interfaces.rows())
        raw = Path(attempt.job.package_interface).read_bytes()
        interface = package_interface.read_bytes(raw)
        digest = "sha256:" + hashlib.sha256(raw).hexdigest()
        for entry in [*interface["entrypoints"], *interface["jobs"]]:
            declaration = entry.get("invocable")
            if declaration is not None:
                result.append(
                    {
                        "interface_digest": digest,
                        "module": declaration["module"],
                        "export": declaration["export"],
                        "implementation_digest": attempt.job.build_id,
                        "interface_path": attempt.job.package_interface,
                        "self": True,
                    }
                )
        return result

    def handle(self, attempt: AttemptRecord, frame: Mapping[str, Any]) -> dict[str, Any]:
        with self.lock:
            try:
                if self.minor() < 40 or attempt.state != "running" or attempt.canceling:
                    raise ValueError("parent call is not active under a capable owner")
                index = frame.get("call_index")
                if type(index) is not int or not 0 <= index < MAX_DIRECT_CALLS:
                    raise ValueError("parent call index exceeds its bound")
                key = (attempt.request_id, attempt.attempt, index)
                kind = frame.get("kind")
                pending = self.pending.get(key)
                if kind == "child_call":
                    request = self._request(attempt, frame, index)
                    if (
                        pending is not None
                        and pending.request.intent_digest != request.intent_digest
                    ):
                        raise ValueError("an existing parent call index changed its exact intent")
                    if pending is None:
                        pending = self.pending[key] = _Pending(request, self.stream())
                        self.send(pb.WorkerFrame(child_call_request=request))
                if pending is None:
                    raise ValueError("parent call has not been activated")
                if kind == "child_cancel":
                    request = pending.request
                    self.send(
                        pb.WorkerFrame(
                            child_call_cancel=pb.ChildCallCancel(
                                parent_request_id=request.parent_request_id,
                                parent_attempt_ordinal=request.parent_attempt_ordinal,
                                parent_invocation_spec_digest=request.parent_invocation_spec_digest,
                                call_index=request.call_index,
                                intent_digest=request.intent_digest,
                            )
                        )
                    )
                    return {"ok": True, "state": "pending"}
                if kind not in ("child_call", "child_poll"):
                    raise ValueError("unknown managed call operation")
                # A new owner stream re-establishes accepted calls by the same
                # canonical index/intent. No durable cursor lives in Runtime.
                if pending.stream_epoch != self.stream():
                    pending.stream_epoch = self.stream()
                    self.send(pb.WorkerFrame(child_call_request=pending.request))
                result = pending.result
                if result is None or result.state == pb.CHILD_CALL_STATE_PENDING:
                    return {"ok": True, "state": "pending"}
                if result.state == pb.CHILD_CALL_STATE_SUCCEEDED:
                    if pending.byte_grants is None:
                        pending.byte_grants = self._materialize(attempt, pending)
                    answer = {
                        "ok": True,
                        "state": "succeeded",
                        "child_request_id": result.child_request_id,
                        "result": result.result_canonical_bytes.decode(),
                        "byte_grants": pending.byte_grants,
                        **(
                            {"observation": observation_document(result.observation)}
                            if result.HasField("observation")
                            else {}
                        ),
                    }
                    if (
                        len(json.dumps(answer, separators=(",", ":"), sort_keys=True).encode())
                        + 128
                        > MAX_FRAME
                    ):
                        raise ValueError("child result and native grants exceed the control frame")
                    return answer
                return {
                    "ok": False,
                    **(
                        {"observation": observation_document(result.observation)}
                        if result.HasField("observation")
                        else {}
                    ),
                    "code": result.safe_code or "child_failed",
                    "detail": result.safe_detail,
                    "child_request_id": result.child_request_id,
                }
            except (
                OSError,
                ValueError,
                InvalidRequest,
                canonical.CanonicalError,
                documents.DocumentError,
            ) as exc:
                return {"ok": False, "code": "child_call_refused", "detail": str(exc)[:1024]}

    def _request(
        self, attempt: AttemptRecord, frame: Mapping[str, Any], index: int
    ) -> pb.ChildCallRequest:
        interface = str(frame.get("interface_digest", ""))
        module, export = str(frame.get("module", "")), str(frame.get("export", ""))
        allowed = next(
            (
                row
                for row in self.bindings(attempt)
                if row["interface_digest"] == interface
                and row["module"] == module
                and row["export"] == export
            ),
            None,
        )
        if allowed is None:
            raise ValueError(
                "call does not name a locked interface dependency or current App export"
            )
        payload = frame.get("payload")
        if not isinstance(payload, str) or len(payload.encode()) > MAX_CALL_BYTES:
            raise ValueError("call request exceeds its inline bound")
        value = canonical.parse_canonical(payload.encode())
        if not isinstance(value, dict):
            raise ValueError("call request must be a canonical object")
        capture = None
        if frame.get("capture") is not None:
            if self.minor() < 43:
                raise ValueError("owner does not support capture")
            capture = msgspec.convert(frame["capture"], type=ActivationCapture, strict=True)
        digest = hashlib.sha256(
            canonical.write(
                {
                    "interface_digest": interface,
                    "module": module,
                    "export": export,
                    "request": value,
                    **({"capture": capture.document()} if capture is not None else {}),
                }
            )
        ).digest()
        return pb.ChildCallRequest(
            parent_request_id=attempt.request_id,
            parent_attempt_ordinal=attempt.attempt,
            parent_invocation_spec_digest=attempt.digest,
            call_index=index,
            interface_digest=documents.raw(interface),
            module=module,
            export=export,
            request_canonical_bytes=payload.encode(),
            intent_digest=digest,
            capture=pb.ActivationCapture(components=capture.components, steps=capture.steps)
            if capture is not None
            else None,
        )

    def absorb(self, result: pb.ChildCallResult) -> bool:
        with self.lock:
            pending = self.pending.get(
                (result.parent_request_id, result.parent_attempt_ordinal, result.call_index)
            )
            if (
                pending is None
                or result.parent_invocation_spec_digest
                != pending.request.parent_invocation_spec_digest
                or result.intent_digest != pending.request.intent_digest
            ):
                return False
            if (
                result.state
                not in (
                    pb.CHILD_CALL_STATE_PENDING,
                    pb.CHILD_CALL_STATE_SUCCEEDED,
                    pb.CHILD_CALL_STATE_REFUSED,
                    pb.CHILD_CALL_STATE_FAILED,
                    pb.CHILD_CALL_STATE_CANCELED,
                )
                or len(result.result_canonical_bytes) > MAX_CALL_BYTES
                or len(result.safe_detail.encode()) > 1024
            ):
                return False
            if (
                result.state in (pb.CHILD_CALL_STATE_PENDING, pb.CHILD_CALL_STATE_SUCCEEDED)
                and not result.child_request_id
            ):
                return False
            try:
                child_byte_results.validate(result)
            except ValueError:
                return False
            if result.HasField("observation"):
                try:
                    observation_document(result.observation)
                except (ValueError, documents.DocumentError):
                    return False
            if result.state == pb.CHILD_CALL_STATE_SUCCEEDED:
                try:
                    canonical.parse_canonical(result.result_canonical_bytes)
                except (ValueError, canonical.CanonicalError):
                    return False
            elif result.result_canonical_bytes:
                return False
            if pending.result is not None and pending.result.state != pb.CHILD_CALL_STATE_PENDING:
                return pending.result == result
            held = pb.ChildCallResult()
            held.CopyFrom(result)
            pending.result = held
            return True

    def _materialize(self, attempt: AttemptRecord, pending: _Pending) -> list[dict[str, Any]]:
        result = pending.result
        assert result is not None
        if not result.byte_result_grants:
            return []
        if self.workspace is None or attempt.spool is None:
            raise ValueError("parent has no native result workspace")
        held_bytes = sum(
            int(row["content_bytes"])
            for key, earlier in self.pending.items()
            if key[:2] == (attempt.request_id, attempt.attempt)
            for row in (earlier.byte_grants or [])
        )
        pending.spool = attempt.spool / "child-results" / str(result.call_index)
        try:
            return child_byte_results.materialize(
                self.workspace,
                self.owner(),
                result,
                attempt.spool,
                max_bytes=byte_inputs.MAX_BYTES - held_bytes,
            )
        except Exception as exc:
            byte_inputs.remove_checkout(pending.spool)
            if isinstance(exc, (ValueError, OSError)):
                raise
            raise ValueError("native child artifact bytes are unavailable") from exc

    def close(self, request_id: str, attempt: int) -> None:
        with self.lock:
            for key in list(self.pending):
                if key[:2] == (request_id, attempt):
                    if self.pending[key].spool is not None:
                        byte_inputs.remove_checkout(self.pending[key].spool)
                    del self.pending[key]
