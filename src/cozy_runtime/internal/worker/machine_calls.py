"""Machine-local invocable dispatch through the existing attempt engine and memo store."""

from __future__ import annotations

import base64
import hashlib
import threading
from typing import TYPE_CHECKING, Any, Literal

from cozy_runtime import canonical_json
from cozy_runtime.author._calls import MAX_ACTIVE_CALLS
from cozy_runtime.internal import effect_interfaces
from cozy_runtime.internal.worker import machine_child_target, workspace_memo
from cozy_runtime.internal.worker.workspace import WorkspaceRefusal
from cozy_runtime.internal.worker.workspace_calls import Call
from cozy_runtime.internal.worker.workspace_calls import Calls as JournalCalls
from cozy_runtime.protocol import documents
from cozy_runtime.protocol import worker_pb2 as pb

if TYPE_CHECKING:
    from .attempts import AttemptRecord
    from .session import Worker


class Calls:
    def __init__(self, worker: Worker):
        assert worker.workspace is not None
        self.worker = worker
        self.journal = JournalCalls(worker.workspace)
        self.lock = threading.Lock()
        self.pending: dict[tuple[str, int], pb.ChildCallRequest] = {}

    def handle(
        self, parent: AttemptRecord, request: pb.ChildCallRequest, action: str
    ) -> pb.ChildCallResult:
        owner = self.worker.fence.record_owner_id
        if action == "call":
            call = self.journal.accept(owner, request)
            with self.lock:
                self.pending[(parent.request_id, request.call_index)] = request
        else:
            call = self.journal.get(owner, parent.request_id, request.call_index)
        if call.intent_digest != request.intent_digest:
            raise WorkspaceRefusal("local call changed its accepted intent")
        executions = self.worker.executions
        assert executions is not None
        if executions.owns(owner, call.child_request):
            state = executions.status(owner, call.child_request)
            previous_attempt = canonical_json.decode(call.prepared)["parent_attempt"]
            if (
                action == "call"
                and parent.attempt > previous_attempt
                and state.state
                in (
                    "failed",
                    "paused",
                )
            ):
                executions.control(
                    owner,
                    call.child_request,
                    f"parent-resume.{parent.attempt}",
                    state.generation,
                    "resume",
                    worker_boot=self.worker.fence.worker_boot_id,
                )
            elif action == "cancel" and state.state not in (
                "succeeded",
                "canceled",
                "failed",
                "paused",
                "pausing",
                "canceling",
            ):
                executions.control(
                    owner,
                    call.child_request,
                    f"parent-pause.{parent.attempt}",
                    state.generation,
                    "pause",
                )
        return self.result(owner, call, request)

    def result(self, owner: str, call: Call, request: pb.ChildCallRequest) -> pb.ChildCallResult:
        result = pb.ChildCallResult(
            parent_request_id=request.parent_request_id,
            parent_attempt_ordinal=request.parent_attempt_ordinal,
            parent_invocation_spec_digest=request.parent_invocation_spec_digest,
            call_index=request.call_index,
            intent_digest=request.intent_digest,
            child_request_id=call.child_request,
            state=pb.CHILD_CALL_STATE_PENDING,
        )
        if call.result:
            result.state, result.result_canonical_bytes = pb.CHILD_CALL_STATE_SUCCEEDED, call.result
            return result
        if call.safe_code:
            result.state, result.safe_code, result.safe_detail = (
                pb.CHILD_CALL_STATE_REFUSED,
                call.safe_code,
                call.safe_detail,
            )
            return result
        executions = self.worker.executions
        assert executions is not None
        if not executions.owns(owner, call.child_request):
            return result
        state = executions.status(owner, call.child_request)
        if state.state not in ("succeeded", "failed", "canceled", "paused"):
            return result
        terminal = executions.collect(owner, call.child_request)
        body = documents.read(terminal.outcome_canonical_bytes, pb.AttemptOutcomeBody)
        if state.state != "succeeded":
            result.state = pb.CHILD_CALL_STATE_FAILED
            result.safe_code = "child_failed"
            result.safe_detail = str(body.get("safe_message", "child operation stopped"))[:1024]
            return result
        held = self.worker.engine.history.get((call.child_request, state.attempt_ordinal))
        if held is not None and held.state != "closed":
            return result
        self._finish(owner, call)
        result.result_canonical_bytes = base64.b64decode(
            body["result"]["inline_result"], validate=True
        )
        result.state = pb.CHILD_CALL_STATE_SUCCEEDED
        return result

    def drive(self) -> None:
        owner, executions = self.worker.fence.record_owner_id, self.worker.executions
        assert executions is not None
        self._stop_descendants(owner)
        with self.lock:
            requests = list(self.pending.values())
        for request in requests:
            parent = self.worker.engine.live.get(request.parent_request_id)
            if (
                parent is None
                or parent.attempt != request.parent_attempt_ordinal
                or parent.state != "running"
                or parent.canceling
            ):
                with self.lock:
                    self.pending.pop((request.parent_request_id, request.call_index), None)
                continue
            call = self.journal.get(owner, parent.request_id, request.call_index)
            if call.result or call.safe_code:
                with self.lock:
                    self.pending.pop((parent.request_id, request.call_index), None)
                continue
            if executions.owns(owner, call.child_request):
                self._finish(owner, call)
                continue
            try:
                if request.module == effect_interfaces.MODULE:
                    assert self.worker.machine_effects is not None
                    self.worker.machine_effects.schedule(owner, call)
                    continue
                if request.HasField("capture"):
                    raise WorkspaceRefusal("activation capture requires a serving child")
                target = machine_child_target.resolve(self.worker, parent, call)
                if target.declaration.get("models") or target.declaration.get("weights_outputs"):
                    raise WorkspaceRefusal(
                        "model child requires native input and output delegation"
                    )
                payload = request.request_canonical_bytes
                arguments = canonical_json.decode(payload)
                memoized = bool(target.declaration.get("invocable", {}).get("memoize"))
                key = (
                    machine_child_target.computation(
                        target, arguments, self.worker.numerical_environment()
                    )
                    if memoized
                    else b""
                )
                plan: dict[str, Any] = {
                    "kind": "job",
                    "revision": target.revision,
                    "entrypoint": target.entrypoint,
                    "parent_attempt": parent.attempt,
                    "computation": key.hex(),
                }
                self.journal.freeze(owner, call, canonical_json.encode(plan))
                call = self.journal.get(owner, parent.request_id, request.call_index)
                if key:
                    found = workspace_memo.lookup(
                        executions.workspace,
                        owner,
                        pb.LookupOperationCall(
                            computation_digest=key, consumer_request_id=call.child_request
                        ),
                    )
                    if found.found:
                        body = documents.read(
                            found.source.outcome_canonical_bytes, pb.AttemptOutcomeBody
                        )
                        self.journal.observe_result(
                            owner,
                            call,
                            base64.b64decode(body["result"]["inline_result"], validate=True),
                        )
                        workspace_memo.delivered(executions.workspace, owner, call.child_request)
                        continue
                self._check_ancestors(owner, parent, target, payload)
                spec = pb.InvocationSpec(
                    environment_digest=target.prepared_revision["placement"]["environment_digest"],
                    payload_digest=documents.spell(hashlib.sha256(payload).digest()),
                    inputs=[
                        pb.InputBinding(
                            input_id="payload",
                            digest=documents.spell(hashlib.sha256(payload).digest()),
                            length=len(payload),
                            kind_mime="application/json",
                        )
                    ],
                    deadline_unix_ms=int(parent.spec.get("deadline_unix_ms", 0)),
                    job=pb.JobInvocationSpec(
                        build_id=target.revision, job_descriptor_id=target.binding.job_descriptor_id
                    ),
                )
                raw, digest = documents.identity(spec)
                offer = pb.AttemptOffer(
                    request_id=call.child_request,
                    attempt_ordinal=1,
                    invocation_spec_digest=digest,
                    invocation_spec_canonical_bytes=raw,
                    grant=pb.DeliveryGrant(
                        invocation_spec_digest=digest,
                        inputs=[
                            pb.InputAccess(
                                input_id="payload",
                                url="data:application/json;base64,"
                                + base64.b64encode(payload).decode(),
                            )
                        ],
                    ),
                )
                desired = pb.DesiredWorkerState(
                    wire_minor=self.worker.fence.wire_minor,
                    posture=pb.POSTURE_ACCEPTING,
                    job=pb.JobDirective(
                        build_id=target.revision,
                        job_descriptor_id=target.binding.job_descriptor_id,
                        orchestration=True,
                        reclaim_on_terminal=True,
                    ),
                )
                preparation = canonical_json.encode(
                    {
                        "revisions": {target.revision: target.prepared_revision},
                        "state": base64.b64encode(
                            desired.SerializeToString(deterministic=True)
                        ).decode(),
                        "parent_attempt": parent.attempt,
                    }
                )
                capture = canonical_json.encode(executions.capture(owner, parent.request_id))
                executions.submit(
                    owner,
                    call.child_request,
                    hashlib.sha256(capture).digest(),
                    offer,
                    result_schema=canonical_json.encode(target.declaration["result"]),
                    worker_boot=self.worker.fence.worker_boot_id,
                    memoize=memoized,
                    preparation=preparation,
                    worker_id=self.worker.options.worker_id,
                )
            except Exception as exc:
                # Admission refusal has no accepted child attempt to duplicate. Keep its
                # exact failure beside the frozen call so a polling parent receives it.
                with executions.workspace.locked() as db:
                    db.execute(
                        "UPDATE execution_calls SET safe_code=?,safe_detail=? "
                        "WHERE owner=? AND parent_request=? AND call_index=?",
                        (
                            "child_admission_refused",
                            str(exc)[:1024],
                            owner,
                            parent.request_id,
                            request.call_index,
                        ),
                    )

    def _stop_descendants(self, owner: str) -> None:
        """A stopped ancestor fences and stops its live descendants after a restart too."""
        executions = self.worker.executions
        assert executions is not None
        with executions.workspace.locked() as db:
            rows = db.execute(
                "SELECT c.child_request,p.request AS parent,p.generation,p.desired,p.state "
                "FROM execution_calls c JOIN executions p "
                "ON p.owner=c.owner AND p.request=c.parent_request "
                "JOIN executions e ON e.owner=c.owner AND e.request=c.child_request "
                "WHERE c.owner=? AND (p.desired<>'run' OR p.state IN ('failed','canceled')) "
                "AND e.state IN ('queued','running','retrying','pausing')",
                (owner,),
            ).fetchall()
        for row in rows:
            state = executions.status(owner, row["child_request"])
            action: Literal["cancel", "pause"] = "cancel" if row["desired"] == "cancel" else "pause"
            if action == "pause" and state.state == "pausing":
                continue
            executions.control(
                owner,
                row["child_request"],
                f"ancestor-{action}.{row['parent']}.{row['generation']}",
                state.generation,
                action,
            )

    def _finish(self, owner: str, call: Call) -> None:
        executions = self.worker.executions
        assert executions is not None
        state = executions.status(owner, call.child_request)
        if state.state != "succeeded" or state.collected:
            return
        held = self.worker.engine.history.get((call.child_request, state.attempt_ordinal))
        if held is not None and held.state != "closed":
            return
        terminal = executions.collect(owner, call.child_request)
        plan = canonical_json.decode(call.prepared)
        if key := plan.get("computation"):
            workspace_memo.record(
                executions.workspace,
                owner,
                pb.RecordOperationResultCall(
                    computation_digest=bytes.fromhex(key),
                    request_id=call.child_request,
                    attempt_ordinal=terminal.attempt_ordinal,
                    invocation_spec_digest=terminal.invocation_spec_digest,
                    outcome_id=terminal.outcome_id,
                    outcome_digest=terminal.outcome_digest,
                ),
            )
        ack = pb.AttemptOutcomeAck(
            request_id=call.child_request,
            attempt_ordinal=terminal.attempt_ordinal,
            invocation_spec_digest=terminal.invocation_spec_digest,
            outcome_id=terminal.outcome_id,
            outcome_digest=terminal.outcome_digest,
        )
        executions.acknowledge_collection(owner, ack)
        executions.workspace.acknowledge(owner, ack)

    def _check_ancestors(
        self, owner: str, parent: AttemptRecord, target: machine_child_target.Target, payload: bytes
    ) -> None:
        assert self.worker.workspace is not None
        request = parent.request_id
        seen: set[str] = set()
        with self.worker.workspace.locked() as db:
            while request not in seen and len(seen) < MAX_ACTIVE_CALLS - 1:
                seen.add(request)
                row = db.execute(
                    "SELECT a.invocation FROM attempts a JOIN executions e "
                    "ON a.owner=e.owner AND a.request=e.request AND a.ordinal=e.ordinal "
                    "WHERE a.owner=? AND a.request=?",
                    (owner, request),
                ).fetchone()
                if row is not None:
                    spec = documents.read(row["invocation"], pb.InvocationSpec)
                    if (
                        spec.get("job", {}).get("build_id") == target.revision
                        and spec.get("job", {}).get("job_descriptor_id")
                        == target.binding.job_descriptor_id
                        and spec.get("payload_digest")
                        == documents.spell(hashlib.sha256(payload).digest())
                    ):
                        raise WorkspaceRefusal(
                            "managed call repeats an active ancestor without progress"
                        )
                above = db.execute(
                    "SELECT parent_request FROM execution_calls WHERE owner=? AND child_request=?",
                    (owner, request),
                ).fetchone()
                if above is None:
                    return
                request = above[0]
        raise WorkspaceRefusal("managed call depth exceeds the composition bound")
