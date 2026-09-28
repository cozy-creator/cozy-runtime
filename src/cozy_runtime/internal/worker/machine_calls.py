"""Machine-local child calls: each call runs on its child's own unit (`execution_unit`).

That unit answers the call (a memo hit, a native source, an effect) or submits the child and
goes on as its execution, then settles the result into the call row. It is that row's one
writer: a parent's poll only reads.
"""

from __future__ import annotations

import base64
import hashlib
import threading
from typing import TYPE_CHECKING, Any

from cozy_runtime import canonical_json
from cozy_runtime.author._calls import MAX_ACTIVE_CALLS
from cozy_runtime.internal import effect_interfaces, source_interfaces
from cozy_runtime.internal.worker import (
    machine_byte_inputs,
    machine_byte_results,
    machine_child_target,
    machine_lanes,
    machine_models,
    workspace_memo,
)
from cozy_runtime.internal.worker.machine_builtin import Builtins
from cozy_runtime.internal.worker.machine_deferred import Deferred
from cozy_runtime.internal.worker.machine_release_roots import ReleaseRoots
from cozy_runtime.internal.worker.machine_serving import Serving
from cozy_runtime.internal.worker.supervisor import Unit, fault_text
from cozy_runtime.internal.worker.workspace import WorkspaceRefusal
from cozy_runtime.internal.worker.workspace_calls import Call, CallFenced
from cozy_runtime.internal.worker.workspace_calls import Calls as JournalCalls
from cozy_runtime.internal.worker.workspace_executions import ExecutionRow
from cozy_runtime.protocol import documents
from cozy_runtime.protocol import worker_pb2 as pb

if TYPE_CHECKING:
    from .attempts import AttemptRecord
    from .session import Worker


class Calls:
    def __init__(self, worker: Worker):
        assert worker.workspace is not None and worker.executions is not None
        self.worker = worker
        self.workspace = worker.workspace
        self.executions = worker.executions
        self.journal = JournalCalls(worker.workspace)
        self.lock = threading.Lock()
        #: child request -> the parent call that named it last (a resumed parent calls again)
        self.requests: dict[str, pb.ChildCallRequest] = {}
        self.serving = Serving(worker)
        self.builtins = Builtins(worker)
        self.deferred = Deferred(worker)
        self.release_roots = ReleaseRoots(worker)

    def close(self) -> None:
        self.builtins.close()
        self.serving.close()

    # ------------------------------------------------------------------ the parent's side

    def handle(
        self, parent: AttemptRecord, request: pb.ChildCallRequest, action: str
    ) -> pb.ChildCallResult:
        owner = self.worker.fence.record_owner_id
        if action == "call":
            call = self.journal.accept(owner, request)
            with self.lock:
                self.requests[call.child_request] = request
        else:
            call = self.journal.get(owner, parent.request_id, request.call_index)
        if call.intent_digest != request.intent_digest:
            raise WorkspaceRefusal("local call changed its accepted intent")
        if request.module == source_interfaces.MODULE and action == "cancel":
            self.worker.machine_sources.cancel(owner, request)
        state = (
            self.executions.status(owner, call.child_request)
            if self.executions.owns(owner, call.child_request)
            else None
        )
        if state is not None and action == "call" and state.state in ("failed", "paused"):
            if parent.attempt > call.plan().parent_attempt:
                # The parent's next attempt calls again: its paused or failed child resumes.
                self.executions.control(
                    owner,
                    call.child_request,
                    f"parent-resume.{parent.attempt}",
                    state.generation,
                    "resume",
                    worker_boot=self.worker.fence.worker_boot_id,
                )
        elif state is not None and action == "cancel" and state.state in ("queued", "running"):
            self.worker.control_tree(owner, call.child_request, "pause")
        if action in ("call", "cancel"):
            self.worker.execution(owner, call.child_request)
        return self.result(owner, call, request)

    def result(self, owner: str, call: Call, request: pb.ChildCallRequest) -> pb.ChildCallResult:
        """What the parent's poll reads. It writes nothing: the child's unit settles the row."""
        result = pb.ChildCallResult(
            parent_request_id=request.parent_request_id,
            parent_attempt_ordinal=request.parent_attempt_ordinal,
            parent_invocation_spec_digest=request.parent_invocation_spec_digest,
            call_index=request.call_index,
            intent_digest=request.intent_digest,
            child_request_id=call.child_request,
            state=pb.CHILD_CALL_STATE_PENDING,
        )
        if call.safe_code:
            result.state, result.safe_code, result.safe_detail = (
                pb.CHILD_CALL_STATE_REFUSED,
                call.safe_code,
                call.safe_detail,
            )
            return result
        if request.module == source_interfaces.MODULE:
            return self.worker.machine_sources.result(owner, call, request, result)
        owned = self.executions.owns(owner, call.child_request)
        if call.result:
            result.state, result.result_canonical_bytes = pb.CHILD_CALL_STATE_SUCCEEDED, call.result
            capture = False
            if owned:
                body = self._terminal(owner, call.child_request)
                if body.HasField("observation"):
                    result.observation.CopyFrom(body.observation)
                capture = body.observation.HasField("capture")
            # An effect froze its own plan in `prepared`; its result declares no byte grants.
            schema = (
                {}
                if request.module == effect_interfaces.MODULE
                else call.plan().result_schema or {}
            )
            result.byte_result_grants.extend(
                machine_byte_results.grants(
                    self.workspace,
                    owner,
                    call.parent_request,
                    call.call_index,
                    schema,
                    capture=capture,
                )
            )
            return result
        if owned and self.executions.status(owner, call.child_request).state in (
            "failed",
            "canceled",
            "paused",
        ):
            result.state = pb.CHILD_CALL_STATE_FAILED
            result.safe_code = "child_failed"
            message = self._terminal(owner, call.child_request).safe_message
            result.safe_detail = (message or "child operation stopped")[:1024]
        return result

    def _terminal(self, owner: str, request: str) -> pb.AttemptOutcomeBody:
        outcome = self.executions.outcome(owner, request)
        return documents.parse(outcome.outcome_canonical_bytes, pb.AttemptOutcomeBody)

    # ------------------------------------------------------------------ the child's unit

    def call(self, unit: Unit, owner: str, child: str) -> bool:
        """The call phase, before the child's execution exists. True once the child is
        submitted (the unit goes on as its execution); False when the call is settled some
        other way, or no parent is waiting for it."""
        call = self.journal.child(owner, child)
        if call is None or call.result or (call.safe_code and not call.sent):
            return False
        module = call.target().module
        if module == effect_interfaces.MODULE:
            effects = self.worker.machine_effects
            assert effects is not None
            effects.run(unit, owner, call)
            return False
        with self.lock:
            request = self.requests.get(child)
        parent = self.worker.engine.live.get(call.parent_request)
        if (
            request is None
            or parent is None
            or parent.attempt != request.parent_attempt_ordinal
            or parent.state != "running"
            or parent.canceling
        ):
            return False  # nobody awaits it: a restarted or finished parent
        try:
            if module == source_interfaces.MODULE:
                self.worker.machine_sources.run(unit, owner, call, request)
                return False
            target = machine_child_target.resolve(self.worker, parent, call)
            self._check_ancestors(owner, parent, target, request.request_canonical_bytes)
            if target.binding is None:
                return self.serving.submit(unit, owner, call, target, request)
            return self._job(owner, parent, call, target, request)
        except CallFenced:
            return False  # its parent moved on: no one reads this call

    def _job(
        self,
        owner: str,
        parent: AttemptRecord,
        call: Call,
        target: machine_child_target.Target,
        request: pb.ChildCallRequest,
    ) -> bool:
        assert target.binding is not None
        if request.HasField("capture"):
            raise WorkspaceRefusal("activation capture requires a serving child")
        executions, workspace = self.executions, self.workspace
        payload = request.request_canonical_bytes
        arguments = canonical_json.decode(payload)
        memoized = bool(target.declaration.get("invocable", {}).get("memoize"))
        key = (
            machine_child_target.computation(target, arguments, self.worker.numerical_environment())
            if memoized
            else b""
        )
        plan: dict[str, Any] = {
            "kind": "job",
            "installation_id": target.installation_id,
            "entrypoint": target.entrypoint,
            "parent_attempt": parent.attempt,
            "computation": key.hex(),
            "result_schema": target.declaration["result"],
        }
        if "runtime_builtin" in target.prepared_installation:
            plan["builtin_preparation"] = target.prepared_installation
        self.journal.freeze(owner, call, canonical_json.encode(plan))
        call = self.journal.get(owner, call.parent_request, call.call_index)
        if key:
            found = workspace_memo.lookup(
                workspace,
                owner,
                pb.LookupOperationCall(
                    computation_digest=key, consumer_request_id=call.child_request
                ),
            )
            if found.found:
                body = documents.parse(found.source.outcome_canonical_bytes, pb.AttemptOutcomeBody)
                self._retain(owner, call, body, target.declaration["result"])
                workspace_memo.delivered(workspace, owner, call.child_request)
                workspace_memo.release_lookup(workspace, owner, call.child_request)
                self.worker.calls.wake(call.parent_request)
                return False
        model_inputs, model_access = machine_models.inputs(
            workspace, owner, call.child_request, target.entrypoint, target.declaration, arguments
        )
        byte_bindings, byte_access = machine_byte_inputs.inputs(
            workspace,
            owner,
            parent,
            call.child_request,
            target.declaration["request"],
            arguments,
            self.worker.calls,
        )
        digest = documents.spell(hashlib.sha256(payload).digest())
        spec = pb.InvocationSpec(
            installation_id=target.prepared_installation["placement"]["installation_id"],
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
                pb.OutputBinding(**output)
                for output in target.declaration.get("weights_outputs", [])
            ],
            deadline_unix_ms=parent.deadline_ms,
            attention_kernel=str(parent.spec.get("attention_kernel") or ""),
            job=pb.JobInvocationSpec(
                installation_id=target.installation_id,
                job_descriptor_id=target.binding.job_descriptor_id,
            ),
        )
        spec.inputs.extend(model_inputs)
        spec.inputs.extend(byte_bindings)
        spec.outputs.extend(
            pb.OutputBinding(output_id=path, max_bytes=bound)
            for path, bound in machine_byte_results.paths(target.declaration["result"])
        )
        raw, identity = documents.identity(spec)
        offer = pb.AttemptOffer(
            request_id=call.child_request,
            attempt_ordinal=1,
            invocation_spec_digest=identity,
            invocation_spec_canonical_bytes=raw,
            grant=pb.DeliveryGrant(
                invocation_spec_digest=identity,
                inputs=[
                    pb.InputAccess(
                        input_id="payload",
                        url="data:application/json;base64," + base64.b64encode(payload).decode(),
                    )
                ],
                outputs=[pb.OutputAccess(output_id=output.output_id) for output in spec.outputs],
            ),
        )
        offer.grant.inputs.extend(model_access)
        offer.grant.inputs.extend(byte_access)
        desired = pb.DesiredWorkerState(
            wire_minor=executions.wire_minor(owner, parent.request_id),
            posture=pb.POSTURE_ACCEPTING,
            job=pb.JobDirective(
                installation_id=target.installation_id,
                job_descriptor_id=target.binding.job_descriptor_id,
                orchestration=not bool(
                    target.declaration.get("models") or target.declaration.get("weights_outputs")
                ),
            ),
        )
        preparation = canonical_json.encode(
            {
                "installations": {target.installation_id: target.prepared_installation},
                "state": base64.b64encode(desired.SerializeToString(deterministic=True)).decode(),
                "parent_attempt": parent.attempt,
            }
        )
        capture = canonical_json.encode(executions.capture(owner, parent.request_id))
        machine_lanes.admit(self.worker, parent.request_id, preparation, offer)
        executions.submit(
            owner,
            call.child_request,
            hashlib.sha256(capture).digest(),
            offer,
            expected_execution_workspace_id=executions.workspace_id,
            result_schema=canonical_json.encode(target.declaration["result"]),
            worker_boot=self.worker.fence.worker_boot_id,
            memoize=memoized,
            preparation=preparation,
            worker_id=self.worker.options.worker_id,
        )
        return True

    def fail_call(self, owner: str, child: str, exc: BaseException) -> None:
        """A call that could not be admitted: its parent's await reads this refusal."""
        call = self.journal.child(owner, child)
        if call is None or call.result or call.safe_code:
            return
        if not self.executions.owns(owner, child):
            machine_models.release(self.workspace, owner, child)
        self._record(owner, call, "child_admission_refused", fault_text(exc))

    def deliver(self, unit: Unit, owner: str, request: str, row: ExecutionRow) -> None:
        """A terminal child settles into its parent's call row: its result's custody moves to
        the parent, once. A failed custody step fails this call, never another."""
        call = self.journal.child(owner, request)
        if call is None:
            return
        if row.state == "succeeded" and not call.result and not call.safe_code:
            engine = self.worker.engine
            # Custody moves after the child's attempt closed (its outcome acknowledged).
            unit.wait_for(
                lambda: (
                    getattr(engine.history.get((request, row.ordinal)), "state", "closed")
                    == "closed"
                )
            )
            try:
                self._custody(owner, call)
            except CallFenced:
                raise  # the parent was resumed under us: settle again against its generation
            except Exception as exc:
                self._record(owner, call, "child_result_unavailable", fault_text(exc))
        self.worker.calls.wake(call.parent_request)

    def _custody(self, owner: str, call: Call) -> None:
        executions, workspace = self.executions, self.workspace
        terminal = executions.collect(owner, call.child_request)
        plan = call.plan()
        if plan.computation:
            workspace_memo.record(
                workspace,
                owner,
                pb.RecordOperationResultCall(
                    computation_digest=bytes.fromhex(plan.computation),
                    request_id=call.child_request,
                    attempt_ordinal=terminal.attempt_ordinal,
                    invocation_spec_digest=terminal.invocation_spec_digest,
                    outcome_id=terminal.outcome_id,
                    outcome_digest=terminal.outcome_digest,
                ),
            )
        body = documents.parse(terminal.outcome_canonical_bytes, pb.AttemptOutcomeBody)
        self._retain(owner, call, body, plan.result_schema or {})
        ack = pb.AttemptOutcomeAck(
            request_id=call.child_request,
            attempt_ordinal=terminal.attempt_ordinal,
            invocation_spec_digest=terminal.invocation_spec_digest,
            outcome_id=terminal.outcome_id,
            outcome_digest=terminal.outcome_digest,
        )
        executions.acknowledge_collection(owner, ack)
        for reference in body.weights_receipts:
            receipt = documents.parse(reference.weights_receipt_canonical_bytes, pb.WeightsReceipt)
            workspace.release_result(
                owner,
                pb.DerivedResultReleaseRequest(
                    weights_transaction_id=receipt.weights_transaction_id,
                    tensorfs_receipt_digest=documents.raw(receipt.tensorfs_receipt_digest),
                ),
            )
        workspace.acknowledge(owner, ack)
        workspace_memo.release_lookup(workspace, owner, call.child_request)

    def _retain(self, owner: str, call: Call, body: pb.AttemptOutcomeBody, schema: Any) -> None:
        """The child's typed result and its native bytes, held for the parent, then recorded."""
        raw = body.result.inline_result
        machine_models.retain_result(
            self.workspace,
            owner,
            call.parent_request,
            f"call.{call.call_index}.result",
            canonical_json.decode(raw),
            schema,
        )
        machine_byte_results.retain(
            self.workspace,
            owner,
            call.parent_request,
            call.call_index,
            canonical_json.decode(raw),
            schema,
            body.output_manifest.outputs,
            capture=body.observation.HasField("capture"),
        )
        self.journal.observe_result(owner, call, raw)

    def _record(self, owner: str, call: Call, code: str, detail: str) -> None:
        self.worker.note("call", f"{call.parent_request}#{call.call_index} {code}: {detail}"[:400])
        with self.workspace.locked() as db:
            db.execute(
                "UPDATE execution_calls SET safe_code=?,safe_detail=? "
                "WHERE owner=? AND parent_request=? AND call_index=? AND result=x''",
                (code, detail[:1024], owner, call.parent_request, call.call_index),
            )
        self.worker.calls.wake(call.parent_request)

    def _check_ancestors(
        self, owner: str, parent: AttemptRecord, target: machine_child_target.Target, payload: bytes
    ) -> None:
        request = parent.request_id
        seen: set[str] = set()
        with self.workspace.locked() as db:
            while request not in seen and len(seen) < MAX_ACTIVE_CALLS - 1:
                seen.add(request)
                if (
                    target.binding is None
                    or target.declaration.get("models")
                    or target.declaration.get("weights_outputs")
                ):
                    held = self.worker.engine.live.get(request)
                    if held is not None and not held.lane_id.startswith("cpu-"):
                        raise WorkspaceRefusal(
                            "managed child cannot coexist with an ancestor on the device lane"
                        )
                row = db.execute(
                    "SELECT a.invocation FROM attempts a JOIN executions e "
                    "ON a.owner=e.owner AND a.request=e.request AND a.ordinal=e.ordinal "
                    "WHERE a.owner=? AND a.request=?",
                    (owner, request),
                ).fetchone()
                if row is not None:
                    spec = documents.parse(row["invocation"], pb.InvocationSpec)
                    if (
                        target.binding is not None
                        and spec.job.installation_id == target.installation_id
                        and spec.job.job_descriptor_id == target.binding.job_descriptor_id
                        and spec.payload_digest == documents.spell(hashlib.sha256(payload).digest())
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

    def release_stranded(self, owner: str) -> None:
        """Boot: model holds of calls no parent will read (no child execution was submitted,
        or an effect settled) go with the process that took them."""
        with self.workspace.locked() as db:
            rows = db.execute(
                "SELECT c.child_request FROM execution_calls c JOIN executions p "
                "ON p.owner=c.owner AND p.request=c.parent_request "
                "WHERE c.owner=? AND p.state<>'running' AND NOT EXISTS("
                "SELECT 1 FROM executions e WHERE e.owner=c.owner AND e.request=c.child_request) "
                "AND (c.sent=0 OR c.result<>x'') "
                "AND EXISTS(SELECT 1 FROM execution_model_holds h "
                "WHERE h.owner=c.owner AND h.recipient=c.child_request)",
                (owner,),
            ).fetchall()
        for row in rows:
            machine_models.release(self.workspace, owner, row["child_request"])
