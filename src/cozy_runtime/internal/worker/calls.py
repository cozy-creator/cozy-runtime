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

from cozy_runtime.author._calls import MAX_ACTIVE_CALLS, MAX_CALL_BYTES, MAX_CALL_INDEX
from cozy_runtime.author._capture import ActivationCapture
from cozy_runtime.author._errors import InvalidRequest
from cozy_runtime.internal import canonical, output_budget, package_interface
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
    forgotten: bool = False
    local: bool = False
    progress_label: str = ""


@dataclass(frozen=True, slots=True)
class _Received:
    metadata: dict[str, Any]
    source: pb.NativeByteRetentionRequest


class Calls:
    def __init__(
        self,
        send: Callable[[pb.WorkerFrame], None],
        stream: Callable[[], int],
        minor: Callable[[], int],
        *,
        workspace: Workspace | None = None,
        owner: Callable[[], str] = lambda: "",
        local: Callable[[AttemptRecord, pb.ChildCallRequest, str], pb.ChildCallResult | None]
        | None = None,
    ):
        self.send, self.stream, self.minor = send, stream, minor
        self.workspace, self.owner = workspace, owner
        self.local = local
        self.pending: dict[tuple[str, int, int], _Pending] = {}
        self.materialized: dict[tuple[str, int], tuple[Path, int]] = {}
        self.received: dict[tuple[str, int], dict[str, _Received]] = {}
        self.lock = threading.RLock()

    @staticmethod
    def bindings(attempt: AttemptRecord) -> list[dict[str, Any]]:
        if attempt.job is None:
            return []
        from cozy_runtime.internal import builtin_operations, effect_interfaces, source_interfaces

        result = [dict(row) for row in attempt.job.call_interfaces]
        result.extend(source_interfaces.rows())
        result.extend(effect_interfaces.rows())
        if builtin_operations.supports_environment(attempt.job.python):
            result.extend(builtin_operations.rows())
        raw = Path(attempt.job.package_interface).read_bytes()
        interface = package_interface.read_bytes(raw)
        digest = "sha256:" + hashlib.sha256(raw).hexdigest()
        for entry in [*interface["jobs"], *interface["entrypoints"]]:
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
                if self.minor() < 44 or attempt.state != "running" or attempt.canceling:
                    raise ValueError("parent call is not active under a capable owner")
                index = frame.get("call_index")
                if type(index) is not int or not 0 <= index <= MAX_CALL_INDEX:
                    raise ValueError("parent call index exceeds its bound")
                key = (attempt.request_id, attempt.attempt, index)
                kind = frame.get("kind")
                pending = self.pending.get(key)
                if kind == "child_forget":
                    if pending is not None:
                        if (
                            pending.result is None
                            or pending.result.state == pb.CHILD_CALL_STATE_PENDING
                        ):
                            pending.forgotten = True
                        else:
                            del self.pending[key]
                    return {"ok": True}
                if kind == "child_call":
                    request = self._request(attempt, frame, index)
                    if (
                        pending is not None
                        and pending.request.intent_digest != request.intent_digest
                    ):
                        raise ValueError("an existing parent call index changed its exact intent")
                    if pending is None:
                        if sum(k[:2] == key[:2] for k in self.pending) >= MAX_ACTIVE_CALLS:
                            raise ValueError("active parent call bound exceeded")
                        label = frame.get("progress_label", "")
                        if (
                            not isinstance(label, str)
                            or len(label) > 120
                            or not label.isprintable()
                        ):
                            label = ""
                        pending = self.pending[key] = _Pending(
                            request, self.stream(), progress_label=label
                        )
                        local = (
                            self.local(attempt, request, "call") if self.local is not None else None
                        )
                        if local is not None:
                            pending.local, pending.result = True, local
                        else:
                            self.send(pb.WorkerFrame(child_call_request=request))
                if pending is None:
                    raise ValueError("parent call has not been activated")
                if pending.local:
                    assert self.local is not None
                    pending.result = self.local(
                        attempt, pending.request, "cancel" if kind == "child_cancel" else "poll"
                    )
                if kind == "child_cancel":
                    if pending.local:
                        return {"ok": True, "state": "pending"}
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
                if not pending.local and pending.stream_epoch != self.stream():
                    pending.stream_epoch = self.stream()
                    self.send(pb.WorkerFrame(child_call_request=pending.request))
                result = pending.result
                if result is None or result.state == pb.CHILD_CALL_STATE_PENDING:
                    return {"ok": True, "state": "pending", **self._progress(pending)}
                if result.state == pb.CHILD_CALL_STATE_SUCCEEDED:
                    if pending.byte_grants is None:
                        pending.byte_grants = self._materialize(attempt, pending)
                    answer = {
                        "ok": True,
                        "state": "succeeded",
                        "child_request_id": result.child_request_id,
                        "result": result.result_canonical_bytes.decode(),
                        "byte_grants": pending.byte_grants,
                        **self._progress(pending),
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
                        # Optional narration cannot turn a valid child result into a refusal.
                        answer.pop("progress", None)
                        if (
                            len(json.dumps(answer, separators=(",", ":"), sort_keys=True).encode())
                            + 128
                            > MAX_FRAME
                        ):
                            raise ValueError(
                                "child result and native grants exceed the control frame"
                            )
                    return answer
                return {
                    "ok": False,
                    **self._progress(pending),
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

    def _progress(self, pending: _Pending) -> dict[str, Any]:
        """Reuse the child's existing journal; polling progress does not change call identity."""
        result, workspace = pending.result, self.workspace
        if result is None or not result.child_request_id:
            return {}
        answer: dict[str, Any] = {"child_request_id": result.child_request_id}
        if not pending.local or workspace is None:
            return answer
        with workspace.locked() as db:
            row = db.execute(
                "SELECT p.sequence,p.body FROM execution_events p "
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
        if row is not None:
            document = json.loads(row["body"])
            answer["progress"] = {"sequence": row["sequence"], "payload": document["payload"]}
        return answer

    def progress_label(self, parent: str, ordinal: int, index: int) -> str:
        with self.lock:
            pending = self.pending.get((parent, ordinal, index))
            return pending.progress_label if pending is not None else ""

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
            if allowed.get("native_source") or allowed.get("native_effect"):
                raise ValueError("activation capture requires an ordinary serving invocation")
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
            if pending.forgotten and result.state != pb.CHILD_CALL_STATE_PENDING:
                del self.pending[
                    (result.parent_request_id, result.parent_attempt_ordinal, result.call_index)
                ]
            return True

    def _materialize(self, attempt: AttemptRecord, pending: _Pending) -> list[dict[str, Any]]:
        result = pending.result
        assert result is not None
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
                        dict(metadata),
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
    ) -> list[tuple[dict[str, Any], pb.NativeByteRetentionRequest]]:
        """Snapshot only native byte capabilities actually delivered to this attempt."""
        with self.lock:
            return [
                (
                    dict(value.metadata),
                    pb.NativeByteRetentionRequest.FromString(
                        value.source.SerializeToString(deterministic=True)
                    ),
                )
                for value in self.received.get((attempt.request_id, attempt.attempt), {}).values()
            ]

    def received_output(
        self, attempt: AttemptRecord, row: Mapping[str, Any]
    ) -> tuple[Path, pb.NativeByteRetentionRequest] | None:
        """Resolve an explicit returned handle only against this parent's actual grants."""
        ref = str(row.get("asset_ref", ""))
        if not ref.startswith("sha256:"):
            return None
        kind = row.get("kind")
        media = byte_inputs.TREE_MIME if kind == "tree" else row.get("media_type")
        identity = canonical.write([kind, ref, media]).decode()
        with self.lock:
            grant = self.received.get((attempt.request_id, attempt.attempt), {}).get(identity)
            if (
                grant is None
                or row.get("digest") != ref
                or row.get("size_bytes") != grant.metadata["content_bytes"]
            ):
                raise ValueError("returned native handle has no exact received grant")
            return Path(grant.metadata["local"]), grant.source

    def close(self, request_id: str, attempt: int) -> None:
        with self.lock:
            self.received.pop((request_id, attempt), None)
            if held := self.materialized.pop((request_id, attempt), None):
                byte_inputs.remove_checkout(held[0] / "child-results")
            for key in list(self.pending):
                if key[:2] == (request_id, attempt):
                    if self.pending[key].spool is not None:
                        byte_inputs.remove_checkout(self.pending[key].spool)
                    del self.pending[key]
