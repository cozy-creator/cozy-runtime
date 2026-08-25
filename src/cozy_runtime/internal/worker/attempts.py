"""The attempt engine: offer -> prepare -> accept -> execute -> terminal, and the warm pass.

Everything the worker may CLAIM about an attempt passes through here, in the one order
worker-protocol/02 fixes:

    VALIDATING  fences, digest recomputation over the resident bytes, local safety
    PREPARING   the PlanChooser fixes ONE plan; the ExecutorPreparer applies it
    ACCEPTED    acceptance journaled (spec + plan + model-construction digests) FIRST
    TERMINAL    the canonical TerminalBody journaled BEFORE it is sent, replayed byte-for-
                byte until a TerminalAck echoes its id AND its digest
    CLOSED      closure recorded

Two things are deliberately NOT here, because they are not the worker's: retryability (a
coordinator projection over status/cause/origin) and attempt-selection order beyond this
worker (dispatch is the coordinator's; below that boundary service class does not exist and
there is no field to carry it). Two attempts assigned to this worker SERIALIZE on the single
device lease, in arrival order, and that is the whole of local arbitration.

Metric attestation lives here too: what the executor reports is a CLAIM. Wall-clock is
clamped to the supervisor-observed dispatch->result window, and the fields the supervisor
structurally cannot verify are NAMED in `unverified_fields` rather than passed off as
measurements.
"""

from __future__ import annotations

import base64
import json
import re
import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from dataclasses import field as dataclasses_field
from pathlib import Path
from typing import Any

from cozy_runtime.internal import canonical
from cozy_runtime.internal.journal import Journal, JournalError
from cozy_runtime.internal.worker import grants
from cozy_runtime.internal.worker.child import ExecutorGone, ExecutorSupervision
from cozy_runtime.internal.worker.ledger import Ledger, LedgerRefusal
from cozy_runtime.internal.worker.plan import AttemptPlan, BindingPlan, PlanRefusal
from cozy_runtime.protocol import documents
from cozy_runtime.protocol import worker_pb2 as pb

#: `safe_message` and `cause.detail` are bounded and SANITIZED (01 §3): printable ASCII
#: only, no secrets, no local paths, no signed URLs. The document profile is integer-only
#: and printable-ASCII, so an unsanitized message is not merely impolite — it cannot be
#: canonicalized at all, and the terminal that carries it could not be journaled.
_PATHY = re.compile(r"(file|https?)://\S+|(?<![\w.])/[\w./-]{4,}")


def safe(text: str, cap: int = 1024) -> str:
    """Bound and sanitize a message for the wire."""
    stripped = _PATHY.sub("<redacted>", text)
    ascii_only = "".join(c if 0x20 <= ord(c) <= 0x7E else "." for c in stripped)
    return ascii_only[:cap]


def _features(payload: Mapping[str, Any], entrypoint: str) -> dict[str, Any]:
    """The request shape the plan is chosen AGAINST, read off the decoded payload.

    Bounded and structural: which fields the caller actually sent, the scalar values that
    can move demand, and the payload's own size. cr-008b replaces this with the declared
    normalized feature basis; what must not come back is a "shape" invented from the plan
    id, which is a fact about the deployment and says nothing about the request.
    """
    scalars = {
        name: value
        for name, value in sorted(payload.items())
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }
    return {
        "entrypoint": entrypoint,
        "fields": sorted(payload),
        "scalars": scalars,
        "payload_bytes": len(canonical.write(dict(payload))),
    }


#: Fields the supervisor structurally CANNOT verify. Named on every terminal that carries
#: them, sorted, because a claim printed as a measurement is worse than no number.
UNVERIFIABLE = ("input_tokens", "output_tokens", "peak_vram_bytes")

CAUSE = pb.CauseCode
ORIGIN = pb.CauseOrigin
STATUS = pb.TerminalStatus

#: The author-surface outcome vocabulary, mapped onto the wire's NEUTRAL one. The mapping is
#: total and closed: a new author terminal without a wire meaning is a compile-time hole.
_TERMINAL = {
    "succeeded": (STATUS.TERMINAL_STATUS_SUCCEEDED, CAUSE.CAUSE_CODE_UNSPECIFIED),
    "refused": (STATUS.TERMINAL_STATUS_REFUSED, CAUSE.CAUSE_CODE_INVALID_REQUEST),
    "failed": (STATUS.TERMINAL_STATUS_FAILED, CAUSE.CAUSE_CODE_AUTHOR_EXCEPTION),
    "canceled": (STATUS.TERMINAL_STATUS_CANCELED, CAUSE.CAUSE_CODE_CLIENT_CANCEL),
}
_ORIGIN = {
    "author": ORIGIN.CAUSE_ORIGIN_AUTHOR,
    "request": ORIGIN.CAUSE_ORIGIN_CLIENT,
    "runtime": ORIGIN.CAUSE_ORIGIN_RUNTIME,
}


class AttemptRefusal(Exception):
    """A typed refusal that becomes a REFUSED terminal (or a dropped message, when the
    fence says the message was never ours to read)."""

    def __init__(self, code: str, detail: str, *, cause: pb.CauseCode, drop: bool = False) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail
        self.cause = cause
        self.drop = drop


@dataclass(slots=True)
class Attempt:
    request_id: str
    attempt: int
    digest: bytes
    spec: dict[str, Any]
    grant: pb.DeliveryGrant
    plan: AttemptPlan | None = None
    binding: BindingPlan | None = None
    state: str = "validating"
    spool: Path | None = None
    terminal_id: str = ""
    terminal_digest: bytes = b""
    terminal_bytes: bytes = b""
    accepted_at: float = 0.0
    dispatched_at: float = 0.0
    resulted_at: float = 0.0
    replays: int = 0
    payload: dict[str, Any] = dataclasses_field(default_factory=dict)
    canceling: str = ""
    cancel_started: float = 0.0
    executions: int = 0
    #: every ledger class as it stood BEFORE this attempt's envelope was granted. The
    #: reconciliation at close compares against this line, never against the mid-attempt
    #: state, because "the attempt gave back what it took" is a claim about the whole
    #: envelope — the grant included.
    opened: dict[str, int] = dataclasses_field(default_factory=dict)

    def key(self) -> tuple[str, int]:
        return (self.request_id, self.attempt)


# --------------------------------------------------------------------------- the engine


class AttemptEngine:
    """One worker's attempts. Single device lease, so one execution at a time."""

    def __init__(
        self,
        *,
        journal: Journal,
        ledger: Ledger,
        chooser: Any,
        supervision: ExecutorSupervision,
        authorizer: grants.Authorizer,
        bindings: Mapping[str, BindingPlan],
        spool_root: Path,
        instance_id: str,
        send: Callable[[pb.WorkerMessage], None],
        progress: Callable[[str, int, dict[str, Any]], None],
    ) -> None:
        self.journal = journal
        self.ledger = ledger
        self.chooser = chooser
        self.supervision = supervision
        self.authorizer = authorizer
        self.bindings = dict(bindings)
        self.spool_root = spool_root
        self.instance_id = instance_id
        self.send = send
        self.progress = progress
        self.live: dict[str, Attempt] = {}  # request_id -> the attempt held right now
        self.history: dict[tuple[str, int], Attempt] = {}
        self.session_id = ""
        self.recovered: list[pb.ActiveAttempt] = []
        self.executed = 0
        self.faults: list[pb.Fault] = []
        self.warm_outcomes: list[dict[str, Any]] = []

    # ------------------------------------------------------------------ offer

    def offer(self, start: pb.StartAttempt) -> Attempt:
        """VALIDATING: fences and local safety, before any body field is trusted."""
        if self.journal.stopped:
            raise AttemptRefusal(
                "journal_stopped",
                f"the journal cannot durably commit ({self.journal.stopped}); this worker "
                "acknowledges nothing and executes nothing",
                cause=CAUSE.CAUSE_CODE_LOCAL_SAFETY,
            )
        recomputed = documents.digest_of(start.exec_spec_canonical)
        if recomputed != start.exec_spec_digest:
            raise AttemptRefusal(
                "exec_spec_digest_mismatch",
                f"carried {start.exec_spec_digest.hex()[:16]}… but the resident bytes hash "
                f"to {recomputed.hex()[:16]}… — a digest never bypasses the lower check",
                cause=CAUSE.CAUSE_CODE_PROTOCOL,
            )
        try:
            spec = documents.read(start.exec_spec_canonical, pb.ExecutionSpec)
        except documents.DocumentError as exc:
            raise AttemptRefusal(
                f"exec_spec_{exc.code}",
                f"the ExecutionSpec document refuses: {exc.detail}",
                cause=CAUSE.CAUSE_CODE_PROTOCOL,
            ) from exc

        # The (attempt, digest) fence binds the ORDINAL, not the current state: an attempt
        # is one execution of one spec whether it is running, journaled or already closed.
        known = self.history.get((start.request_id, start.attempt))
        if known is not None and known.digest != start.exec_spec_digest:
            raise AttemptRefusal(
                "attempt_digest_conflict",
                f"attempt {start.attempt} of {start.request_id} is bound to "
                f"{known.digest.hex()[:16]} (state {known.state}); a different spec under "
                "the same ordinal is unrepresentable, so neither spec runs",
                cause=CAUSE.CAUSE_CODE_PROTOCOL,
            )

        held = self.live.get(start.request_id)
        if held is not None:
            if held.attempt == start.attempt:
                held.grant = start.grant  # a refresh swaps authority; identity never moves
                return held  # idempotent: re-send AttemptAccepted, run once
            if held.state in ("accepted", "running"):
                raise AttemptRefusal(
                    "live_attempt_supersession",
                    f"attempt {start.attempt} arrived while attempt {held.attempt} is live "
                    "with no journaled terminal: supersession is explicit "
                    "(CancelAttempt(SUPERSEDED) -> journaled terminal -> attempt+1), never "
                    "the arrival of a successor",
                    cause=CAUSE.CAUSE_CODE_PROTOCOL,
                )
        previous = self.history.get((start.request_id, start.attempt))
        if previous is not None and previous.state in ("terminal", "closed"):
            previous.replays += 1
            return previous  # duplicate StartAttempt for a finished attempt: replay, no run

        return Attempt(
            request_id=start.request_id,
            attempt=start.attempt,
            digest=start.exec_spec_digest,
            spec=spec,
            grant=start.grant,
        )

    # ------------------------------------------------------------------ prepare/accept

    def prepare_and_accept(self, attempt: Attempt) -> pb.AttemptAccepted:
        """The ATOMIC ACCEPTANCE BOUNDARY, in exactly this order:

            resolve binding -> fetch/verify/decode the payload -> derive REAL features from
            it -> choose the plan -> prepare the executor -> reserve the envelope -> JOURNAL
            -> publish AttemptAccepted.

        Everything before the journal either completes or raises a typed PRE-ACCEPT refusal,
        and the reservation rolls back if the journal cannot commit. The payload is fetched
        HERE and not at execution time on purpose: a missing, expired or malformed payload
        must be a refusal the coordinator can read, not an accepted attempt that later dies
        on a thread with no terminal to show for it.

        Features are derived from the DECODED PAYLOAD. Deriving them from the plan id would
        be inventing the very input the plan is supposed to be chosen against.
        """
        serving = attempt.spec.get("serving")
        if serving is None:
            raise AttemptRefusal(
                "job_mode_unsupported",
                "this worker serves; the job lane (bounded preparation under a fresh "
                "per-attempt executor, checkpoints, reclaim) is cr-009",
                cause=CAUSE.CAUSE_CODE_CONSTRAINT_INFEASIBLE,
            )
        plan_id = serving["entrypoint_binding_plan_id"]
        binding = self.bindings.get(plan_id)
        if binding is None:
            raise AttemptRefusal(
                "unknown_binding_plan",
                f"no EntrypointBindingPlan {plan_id[:24]}… is bound on this worker",
                cause=CAUSE.CAUSE_CODE_LOCAL_SAFETY,
            )
        attempt.binding = binding
        attempt.state = "preparing"

        try:
            attempt.payload = self._payload(attempt)
        except grants.GrantRefusal as exc:
            raise AttemptRefusal(
                exc.code,
                exc.detail,
                cause=CAUSE.CAUSE_CODE_GRANT_EXPIRED
                if exc.code in ("grant_expired", "input_absent")
                else CAUSE.CAUSE_CODE_LOCAL_SAFETY,
            ) from exc
        except (ValueError, OSError) as exc:
            raise AttemptRefusal(
                "payload_undecodable",
                f"the granted payload input is not a JSON object: {exc}",
                cause=CAUSE.CAUSE_CODE_INVALID_REQUEST,
            ) from exc
        features = _features(attempt.payload, binding.entrypoint)
        try:
            plan = self.chooser.choose(binding, features)
        except PlanRefusal as exc:
            raise AttemptRefusal(
                exc.code, exc.detail, cause=CAUSE.CAUSE_CODE_CONSTRAINT_INFEASIBLE
            ) from exc
        attempt.plan = plan

        # The ExecutorPreparer: when the selected construction is not already ready, prepare
        # it BEFORE durable acceptance. Serving never accepts-then-cold-loads.
        executor = self.supervision.current
        if executor is None or not executor.alive():
            raise AttemptRefusal(
                "executor_absent",
                "no live executor: an online attempt is accepted only against a Ready one",
                cause=CAUSE.CAUSE_CODE_EXECUTOR_FAULT,
            )
        if plan_id not in executor.ready_plans:
            raise AttemptRefusal(
                "executor_not_ready",
                f"the executor is ready for {sorted(executor.ready_plans) or 'nothing'}, "
                f"not {plan_id[:24]}… — the directive prepares an executor before an "
                "attempt may be accepted against it, and serving never accepts-then-"
                "cold-loads",
                cause=CAUSE.CAUSE_CODE_EXECUTOR_FAULT,
            )
        if executor.poisoned:
            raise AttemptRefusal(
                "poisoned_generation",
                f"the executor generation is poisoned ({executor.poisoned}) and has no "
                "recovery edge to Ready",
                cause=CAUSE.CAUSE_CODE_EXECUTOR_FAULT,
            )

        attempt.opened = self.ledger.snapshot()
        try:
            self.ledger.grant(
                f"{attempt.request_id}#{attempt.attempt}",
                plan.reserved_host_bytes,
                plan.reserved_pinned_bytes,
                f"attempt {attempt.request_id}#{attempt.attempt}",
            )
        except LedgerRefusal as exc:
            raise AttemptRefusal(
                exc.code, exc.detail, cause=CAUSE.CAUSE_CODE_CONSTRAINT_INFEASIBLE
            ) from exc

        plan_digest = plan.digest()
        try:
            self.journal.append(
                "accepted",
                {
                    "request_id": attempt.request_id,
                    "attempt": attempt.attempt,
                    "session_id": self.session_id,
                    "instance_id": self.instance_id,
                    "exec_spec_digest": documents.spell(attempt.digest),
                    "plan_digest": plan_digest,
                    "model_construction_digest": plan.model_construction_digest,
                    "grant_epoch": int(attempt.grant.credential.epoch),
                    "entrypoint_binding_plan_id": plan_id,
                },
            )
        except Exception:
            # The envelope was granted a moment ago and nothing will ever run under it:
            # an unjournaled acceptance must not leave the worker's host ledger charged.
            self.ledger.release(f"{attempt.request_id}#{attempt.attempt}")
            raise
        attempt.state = "accepted"
        attempt.accepted_at = time.perf_counter()
        self.live[attempt.request_id] = attempt
        self.history[attempt.key()] = attempt
        return pb.AttemptAccepted(
            session_id=self.session_id,
            executor_incarnation=self.supervision.incarnation,
            request_id=attempt.request_id,
            attempt=attempt.attempt,
            exec_spec_digest=attempt.digest,
            plan_digest=documents.raw(plan_digest),
            model_construction_digest=documents.raw(plan.model_construction_digest),
            plan=pb.AttemptPlanSummary(
                delivery=plan.delivery,
                materialization=plan.materialization,
                compute_dtype=plan.compute_dtype,
                placement=plan.placement,
                reserved_vram_bytes=plan.reserved_vram_bytes,
                reserved_host_bytes=plan.reserved_host_bytes,
                quantified_choice=plan.quantified_choice,
            ),
        )

    # ------------------------------------------------------------------ execute

    def execute(self, attempt: Attempt) -> pb.AttemptTerminal:
        """Run ONE accepted attempt, then CLOSE it against the ledger.

        The two halves are separate on purpose. `_run` ends at a journaled terminal and
        nothing may come between the journal and the send — that is the protocol's
        durability rule. Reconciliation happens after it, before the attempt leaves this
        engine, and its own failure can never invalidate a terminal that is already durable:
        a ledger that cannot be closed is a FAULT the operator sees, not a lost result.

        The opening line it reconciles against was taken at ACCEPTANCE, before the envelope
        was granted (`attempt.opened`).
        """
        terminal = self._run(attempt)
        self.reconcile(attempt)
        return terminal

    def reconcile(self, attempt: Attempt) -> dict[str, Any]:
        """Every byte class, before and after, with the executor's after-state re-read.

        The device numbers come from the process that owns the CUDA context, so a dead
        executor makes them UNREADABLE rather than zero (§3.6). That distinction is the
        whole point on the kill path: "0 B allocated" would be a claim about a device this
        worker can no longer see, while "unreadable, the context died with the process" is
        what actually happened — and the bytes really are gone, because the kernel reclaims
        a dead process's device context, which `nvidia-smi` confirms from outside.
        """
        key = f"{attempt.request_id}#{attempt.attempt}"
        executor = self.supervision.current
        if executor is not None and executor.alive() and not executor.poisoned:
            try:
                self.ledger.observe_probe(executor.call({"cmd": "probe"}, timeout=30.0))
            except ExecutorGone:
                self.ledger.device_unreadable("the executor died before the ledger was read")
        else:
            self.ledger.device_unreadable(
                "no live usable executor: its CUDA context died with the process"
            )
        record = self.ledger.close_attempt(key, attempt.opened)
        try:
            self.journal.append("ledger", record)
        except JournalError as exc:
            record["journaled"] = False
            record["journal_error"] = exc.code
        if record["leaks"]:
            self.faults.append(
                pb.Fault(
                    kind=pb.FaultKind.FAULT_KIND_LOCAL_SAFETY_REFUSAL,
                    subject=key,
                    reason="ledger_unreconciled",
                    detail=safe(
                        "the attempt reached a terminal without returning every class to "
                        "its opening value: " + "; ".join(record["leaks"])
                    ),
                )
            )
        return record

    def _run(self, attempt: Attempt) -> pb.AttemptTerminal:
        """The attempt itself: dispatch, freeze point, tail, journaled terminal."""
        assert attempt.binding is not None and attempt.plan is not None
        executor = self.supervision.current
        if executor is None:
            return self.abandon(attempt, "the executor vanished before dispatch")
        spool = self.spool_root / f"{attempt.request_id}-{attempt.attempt}"
        spool.mkdir(parents=True, exist_ok=True)
        attempt.spool = spool
        attempt.state = "running"
        attempt.executions += 1
        self.executed += 1

        deadline_ms = int(attempt.spec.get("deadline_unix_ms", 0))
        seconds = max(0.5, deadline_ms / 1000 - time.time()) if deadline_ms else 600.0
        attempt.dispatched_at = time.perf_counter()
        try:
            reply = executor.call(
                {
                    "cmd": "invoke",
                    "request_id": f"{attempt.request_id}#{attempt.attempt}",
                    "entrypoint": attempt.binding.entrypoint,
                    "payload": attempt.payload,
                    "deadline_s": seconds,
                    "boot_warmup": False,
                    "spool": str(spool),
                },
                timeout=None,
                on_progress=lambda frame: self.progress(attempt.request_id, attempt.attempt, frame),
            )
        except ExecutorGone as exc:
            attempt.resulted_at = time.perf_counter()
            # WHICHEVER THREAD SEES THE DEATH MARKS IT. The forceful-cancel path kills on
            # the watchdog thread while this one is blocked in recv, so a rebuild decision
            # that waits for the killer to finish its bookkeeping races a dying process and
            # loses (`waitpid` answers 0 for a few ms). Invalidating at the OBSERVATION
            # removes the race instead of narrowing it.
            self.supervision.invalidate("the executor died during an attempt")
            # A cancel that ESCALATED killed this executor on purpose: the attempt's
            # terminal is CANCELED, and the executor is already provably dead. Any other
            # death is an invalidation the attempt did not ask for.
            if attempt.canceling:
                return self.finish_cancel(attempt, attempt.canceling)
            return self.abandon(attempt, str(exc))
        attempt.resulted_at = time.perf_counter()
        # What this attempt's kernel allocated ON TOP of the resident weights. Recorded from
        # whatever came back, success or fault: a failed attempt's activation peak is
        # exactly the number an operator needs, and dropping it on the failure path would
        # leave the ledger blind in the one case that matters.
        self.ledger.observe_attempt(reply.get("metrics", {}))

        if attempt.canceling:
            return self.finish_cancel(attempt, attempt.canceling)
        if not reply.get("ok"):
            return self.terminal(
                attempt,
                STATUS.TERMINAL_STATUS_FAILED,
                CAUSE.CAUSE_CODE_EXECUTOR_FAULT,
                ORIGIN.CAUSE_ORIGIN_EXECUTOR,
                f"{reply.get('code')}: {reply.get('detail', '')}",
            )
        if not reply.get("quiescent"):
            # No terminal is journaled while author computation might still mutate GPU or
            # output state. Unproven quiescence is an executor fault, not a success.
            self.supervision.invalidate("unproven quiescence")
            return self.terminal(
                attempt,
                STATUS.TERMINAL_STATUS_FAILED,
                CAUSE.CAUSE_CODE_EXECUTOR_FAULT,
                ORIGIN.CAUSE_ORIGIN_SUPERVISOR,
                "the executor could not prove quiescence after the handler returned; a "
                "terminal journaled under unproven quiescence would be a claim about state "
                "that may still be moving",
            )
        if not reply.get("env_intact", True):
            self.supervision.invalidate("env seal broken")
            return self.terminal(
                attempt,
                STATUS.TERMINAL_STATUS_FAILED,
                CAUSE.CAUSE_CODE_EXECUTOR_FAULT,
                ORIGIN.CAUSE_ORIGIN_SUPERVISOR,
                "env_seal_broken: the executor's environment changed after CUDA init — the "
                "allocator read it once, so the imposition was a silent no-op",
            )
        if "state_seq" in reply:
            # Monotone wire counters belong to the SUPERVISOR. A disposable process that
            # restarts a counter at 1 must never become the worker's truth.
            self.faults.append(
                pb.Fault(
                    kind=pb.FaultKind.FAULT_KIND_LOCAL_SAFETY_REFUSAL,
                    subject=f"{attempt.request_id}#{attempt.attempt}",
                    reason="executor_owned_counter",
                    detail=f"the executor claimed state_seq={reply['state_seq']}; monotone "
                    "wire counters are the supervisor's and this claim is dropped",
                )
            )
        if reply.get("poisoned"):
            self.supervision.invalidate(str(reply["poisoned"]))

        outcome = reply["outcome"]
        status, cause = _TERMINAL[outcome["terminal"]]
        if outcome["terminal"] == "refused" and outcome["code"] == "unsupported_input":
            cause = CAUSE.CAUSE_CODE_UNSUPPORTED_INPUT
        if outcome["terminal"] == "canceled":
            cause = (
                CAUSE.CAUSE_CODE_DEADLINE_EXPIRED
                if outcome["code"] == "deadline"
                else CAUSE.CAUSE_CODE_CLIENT_CANCEL
            )
        oversize = int(reply.get("oversize_result_bytes", 0))
        if oversize:
            grants.abort_outputs(attempt.spool) if attempt.spool else None
            return self.terminal(
                attempt,
                STATUS.TERMINAL_STATUS_FAILED,
                CAUSE.CAUSE_CODE_CAPABILITY_UNAVAILABLE,
                ORIGIN.CAUSE_ORIGIN_RUNTIME,
                f"result_too_large: the typed result canonicalizes to {oversize} B, over "
                "the inline door; the blob-receipt branch needs a real write-and-receipt "
                "transaction and this runtime will not mint a receipt for bytes nobody "
                "wrote (cr-017 owns that transaction)",
            )
        # The output transaction belongs to a SUCCEEDING attempt. A refused or failed one
        # produced no outputs by definition, so running the tail over it would find the
        # grant unsatisfied and report `destination_absent` — replacing the author's real
        # cause with a story about a transaction that was never owed. Found live: a handler
        # that refused came back to the coordinator as a missing destination, and the actual
        # reason was nowhere on the terminal.
        manifest: pb.OutputManifest | None = None
        if status == STATUS.TERMINAL_STATUS_SUCCEEDED:
            manifest, tail_error = self._tail(attempt, reply)
            if tail_error is not None:
                return self.terminal(
                    attempt,
                    STATUS.TERMINAL_STATUS_FAILED,
                    CAUSE.CAUSE_CODE_CAPABILITY_UNAVAILABLE,
                    ORIGIN.CAUSE_ORIGIN_SUPERVISOR,
                    tail_error,
                )
        elif attempt.spool is not None:
            grants.abort_outputs(attempt.spool)
        return self.terminal(
            attempt,
            status,
            cause,
            _ORIGIN[outcome["origin"]],
            outcome["message"] or outcome["code"],
            manifest=manifest,
            metrics=reply.get("metrics", {}),
            result=reply if status == STATUS.TERMINAL_STATUS_SUCCEEDED else None,
        )

    def _payload(self, attempt: Attempt) -> dict[str, Any]:
        """The payload travels as a grant INPUT, fetched supervisor-side under the grant."""
        data = grants.read_input(attempt.grant, "payload", self.authorizer)
        value = json.loads(data)
        if not isinstance(value, dict):
            raise ValueError(f"a payload document is an object, not {type(value).__name__}")
        return value

    def _tail(
        self, attempt: Attempt, reply: dict[str, Any]
    ) -> tuple[pb.OutputManifest | None, str | None]:
        """§3.4 steps 4-5, supervisor-side: the device lease is already released, the bytes
        are already immutable in the attempt spool, and this is where they are written under
        the exact output grant and become a durable manifest.

        The tail runs SYNCHRONOUSLY with the attempt (launch). Overlapping it with the next
        attempt's GPU work is cr-018's door and is not claimed here.

        Outputs bind to granted destinations by STABLE OUTPUT ID — the result's own field
        path (`image`, `pair.thumb`, `frames.0`) — with EXACT SET EQUALITY against the
        grant. Never by list index: two outputs that arrive in a different order must still
        land in their own destinations, and a position cannot express that. A produced
        output the coordinator did not grant, or a granted destination the attempt did not
        fill, refuses the tail WHOLE — an output transaction is not partially successful.
        """
        rows = list(reply.get("outputs", ()))
        spool = attempt.spool
        if not rows and not attempt.grant.outputs:
            return None, None
        if spool is None:
            return None, "the attempt has no spool to read outputs from"
        produced = {str(row["output_id"]): row for row in rows}
        granted = {destination.output_id for destination in attempt.grant.outputs}
        if set(produced) != granted:
            grants.abort_outputs(spool)
            return None, (
                f"destination_absent: the attempt produced {sorted(produced) or 'nothing'} "
                f"and the grant names {sorted(granted) or 'nothing'}; an output the "
                "coordinator did not grant is unwritable however it was produced, and a "
                "granted destination the attempt never filled is not a success"
            )
        entries = []
        try:
            for output_id in sorted(produced):
                row = produced[output_id]
                blob = self._blob(spool, str(row["asset_ref"]))
                if blob is None:
                    return None, (f"output {output_id} has no bytes in the attempt spool")
                entries.append(
                    grants.write_output(
                        attempt.grant, output_id, blob.read_bytes(), self.authorizer
                    )
                )
        except grants.GrantRefusal as exc:
            grants.abort_outputs(spool)
            return None, f"{exc.code}: {exc.detail}"
        return (
            pb.OutputManifest(
                manifest_id=f"man-{attempt.request_id}-{attempt.attempt}", outputs=entries
            ),
            None,
        )

    @staticmethod
    def _blob(spool: Path, ref: str) -> Path | None:
        tail = ref.rsplit("/", 2)
        if len(tail) < 3:
            return None
        candidate = spool / f"{tail[-2]}-{tail[-1]}"
        return candidate if candidate.is_file() else None

    # ------------------------------------------------------------------ terminals

    def terminal(
        self,
        attempt: Attempt,
        status: pb.TerminalStatus,
        cause: pb.CauseCode,
        origin: pb.CauseOrigin,
        detail: str,
        *,
        manifest: pb.OutputManifest | None = None,
        metrics: Mapping[str, Any] | None = None,
        result: Mapping[str, Any] | None = None,
        shortfall: Mapping[str, Any] | None = None,
    ) -> pb.AttemptTerminal:
        """Build, JOURNAL, and only then return the terminal for sending."""
        body = pb.TerminalBody(
            request_id=attempt.request_id,
            attempt=attempt.attempt,
            exec_spec_digest=attempt.digest,
            status=status,
            safe_message=safe(detail, 4096),
            cause=pb.TerminalCause(code=cause, origin=origin, detail=safe(detail)),
        )
        if manifest is not None:
            body.output_manifest.CopyFrom(manifest)
        if shortfall is not None:
            body.cause.shortfall.CopyFrom(
                pb.ResourceShortfall(
                    resource=str(shortfall["resource"]),
                    scope=str(shortfall["scope"]),
                    needed_bytes=int(shortfall["needed_bytes"]),
                    available_bytes=int(shortfall["available_bytes"]),
                    evidence_class=str(shortfall.get("evidence_class", "measured")),
                )
            )
        if metrics:
            body.metrics.CopyFrom(self.attest(attempt, metrics, manifest))
        if result is not None:
            body.result.CopyFrom(self._envelope(result))

        canonical_bytes, digest = documents.identity(body)
        attempt.terminal_id = f"trm-{uuid.uuid4().hex[:24]}"
        attempt.terminal_digest = digest
        attempt.terminal_bytes = canonical_bytes
        self.journal.append(
            "terminal",
            {
                "request_id": attempt.request_id,
                "attempt": attempt.attempt,
                "session_id": self.session_id,
                "instance_id": self.instance_id,
                "exec_spec_digest": documents.spell(attempt.digest),
                "terminal_id": attempt.terminal_id,
                "terminal_digest": documents.spell(digest),
                "terminal_canonical_b64": base64.b64encode(canonical_bytes).decode(),
            },
        )
        attempt.state = "terminal"
        self.ledger.release(f"{attempt.request_id}#{attempt.attempt}")
        return self.wrap(attempt)

    def accepted_message(self, attempt: Attempt) -> pb.AttemptAccepted:
        """Re-send the SAME acceptance for a held attempt. AttemptAccepted always carries
        the digests that were JOURNALED, so a re-send can never move them."""
        assert attempt.plan is not None
        plan = attempt.plan
        return pb.AttemptAccepted(
            session_id=self.session_id,
            executor_incarnation=self.supervision.incarnation,
            request_id=attempt.request_id,
            attempt=attempt.attempt,
            exec_spec_digest=attempt.digest,
            plan_digest=documents.raw(plan.digest()),
            model_construction_digest=documents.raw(plan.model_construction_digest),
            plan=pb.AttemptPlanSummary(
                delivery=plan.delivery,
                materialization=plan.materialization,
                compute_dtype=plan.compute_dtype,
                placement=plan.placement,
                reserved_vram_bytes=plan.reserved_vram_bytes,
                reserved_host_bytes=plan.reserved_host_bytes,
                quantified_choice=plan.quantified_choice,
            ),
        )

    def wrap(self, attempt: Attempt) -> pb.AttemptTerminal:
        """Wrap the SAME journaled body in the CURRENT session envelope. The body is a byte
        fact; only the envelope moves across a reconnect."""
        return pb.AttemptTerminal(
            session_id=self.session_id,
            executor_incarnation=self.supervision.incarnation,
            request_id=attempt.request_id,
            attempt=attempt.attempt,
            exec_spec_digest=attempt.digest,
            terminal_id=attempt.terminal_id,
            terminal_digest=attempt.terminal_digest,
            terminal_canonical=attempt.terminal_bytes,
        )

    def _envelope(self, reply: Mapping[str, Any]) -> pb.ResultEnvelope:
        """The TYPED function result — never the asset manifest under another name.

        Small results ride the terminal INLINE (<= 4 MiB), skipping the file-store round
        trip; a larger one takes the blob-receipt door. The choice is kept here explicitly
        rather than lost: it is the whole reason the field is a XOR pair.
        """
        document = reply.get("result") or {}
        data = canonical.write(document)
        # The DECLARED annotated result surface's identity, computed by the executor from
        # the endpoint's own schema — never a digest of this particular value, which would
        # move every request and fence nothing.
        schema_digest = str(reply.get("result_schema_digest") or canonical.digest(document))
        envelope = pb.ResultEnvelope(
            result_schema_digest=documents.raw(schema_digest),
            adjustments=[
                pb.AdjustmentRow(
                    field=row["field"],
                    requested=row["requested"],
                    applied=row["applied"],
                    reason=row["reason"],
                )
                for row in reply.get("adjustments", ())
            ],
        )
        envelope.inline_result = data  # the oversize case never reaches here (§3.4)
        return envelope

    def attest(
        self,
        attempt: Attempt,
        claim: Mapping[str, Any],
        manifest: pb.OutputManifest | None,
    ) -> pb.AttemptMetrics:
        """The executor's numbers are a CLAIM; this is the supervisor's attestation.

        Wall-clock is clamped to the window the supervisor itself observed (dispatch ->
        result). Anything it cannot observe is NAMED, never silently promoted.
        """
        window_ms = (attempt.resulted_at - attempt.dispatched_at) * 1000
        queue_ms = (attempt.dispatched_at - attempt.accepted_at) * 1000
        return pb.AttemptMetrics(
            runtime_ms=int(min(float(claim.get("handler_ms", 0.0)), window_ms)),
            queue_ms=int(max(queue_ms, 0.0)),
            peak_vram_bytes=int(claim.get("peak_vram_bytes", 0)),
            rss_at_end_bytes=int(claim.get("rss_at_end_bytes", 0)),
            output_count=len(manifest.outputs) if manifest is not None else 0,
            device_lease_ms=int(min(float(claim.get("device_lease_ms", 0.0)), window_ms)),
            gpu_count=int(claim.get("gpu_count", 0)),
            handler_ms=int(min(float(claim.get("handler_ms", 0.0)), window_ms)),
            finalization_ms=int(max((time.perf_counter() - attempt.resulted_at) * 1000, 0)),
            unverified_fields=list(UNVERIFIABLE),
        )

    # ------------------------------------------------------------------ cancel / abandon

    def cancel(self, message: pb.CancelAttempt) -> str:
        """Cooperative, then forceful. Returns what was done, for the activity lane."""
        held = self.live.get(message.request_id)
        if held is None or held.attempt != message.attempt:
            return "dropped: no such held attempt"
        if held.digest != message.exec_spec_digest:
            return "dropped: cancel digest does not match the journaled acceptance"
        if held.state == "terminal":
            return "ignored: the attempt already reached a natural terminal"
        held.canceling = {
            pb.CancelReason.CANCEL_REASON_CLIENT: "client",
            pb.CancelReason.CANCEL_REASON_DRAIN: "drain",
            pb.CancelReason.CANCEL_REASON_SUPERSEDED: "superseded",
            pb.CancelReason.CANCEL_REASON_POLICY: "policy",
            pb.CancelReason.CANCEL_REASON_DEADLINE: "deadline",
        }.get(message.reason, "client")
        executor = self.supervision.current
        if executor is None:
            return "executor already gone"
        executor.cooperative_cancel()
        return f"cooperative cancel signalled ({held.canceling})"

    def escalate(self, attempt: Attempt, grace: float) -> dict[str, Any]:
        """The FORCEFUL half, and the exact ordering §5 requires: forced kill -> output
        abort -> epoch invalidation -> journal CANCELED. Rebuilding readiness happens
        AFTER, off the cancellation critical path."""
        executor = self.supervision.current
        started = time.perf_counter()
        killed = executor.kill_group() if executor is not None else 0
        status = executor.reap() if executor is not None else None
        gone_at = time.perf_counter()
        aborted = grants.abort_outputs(attempt.spool) if attempt.spool else 0
        epoch = self.supervision.invalidate("forceful cancel")
        return {
            "grace_s": grace,
            "killed": killed,
            "exit_status": status,
            "kill_to_gone_ms": round((gone_at - started) * 1000, 2),
            "aborted_outputs": aborted,
            "readiness_epoch": epoch,
        }

    def finish_cancel(self, attempt: Attempt, reason: str) -> pb.AttemptTerminal:
        cause = {
            "client": CAUSE.CAUSE_CODE_CLIENT_CANCEL,
            "drain": CAUSE.CAUSE_CODE_DRAIN_CANCEL,
            "superseded": CAUSE.CAUSE_CODE_SUPERSEDED_CANCEL,
            "policy": CAUSE.CAUSE_CODE_POLICY_CANCEL,
            "deadline": CAUSE.CAUSE_CODE_DEADLINE_EXPIRED,
        }[reason]
        return self.terminal(
            attempt,
            STATUS.TERMINAL_STATUS_CANCELED,
            cause,
            ORIGIN.CAUSE_ORIGIN_COORDINATOR
            if reason in ("drain", "policy", "superseded")
            else ORIGIN.CAUSE_ORIGIN_CLIENT,
            f"canceled ({reason}); the executor was dead and uncommitted outputs aborted "
            "before this terminal was journaled",
        )

    def abandon(self, attempt: Attempt, why: str) -> pb.AttemptTerminal:
        """An accepted-but-incomplete attempt ALWAYS produces a terminal, never silence."""
        if attempt.spool is not None:
            grants.abort_outputs(attempt.spool)
        self.supervision.invalidate("executor invalidated")
        return self.terminal(
            attempt,
            STATUS.TERMINAL_STATUS_ABANDONED,
            CAUSE.CAUSE_CODE_EXECUTOR_INVALIDATED,
            ORIGIN.CAUSE_ORIGIN_EXECUTOR,
            f"executor invalidated: {why}",
        )

    def close(self, ack: pb.TerminalAck) -> str:
        """Closure records ONLY on an ack bound by terminal_id AND terminal_digest."""
        attempt = self.history.get((ack.request_id, ack.attempt))
        if attempt is None or attempt.state != "terminal":
            return "dropped: no journaled terminal for that attempt"
        if ack.terminal_id != attempt.terminal_id or ack.terminal_digest != attempt.terminal_digest:
            return (
                f"NOT AN ACK: id {ack.terminal_id!r}/digest {ack.terminal_digest.hex()[:12]} "
                f"does not bind the journaled terminal — replay continues"
            )
        self.journal.append(
            "closed",
            {
                "request_id": attempt.request_id,
                "attempt": attempt.attempt,
                "terminal_id": attempt.terminal_id,
                "terminal_digest": documents.spell(attempt.terminal_digest),
            },
        )
        attempt.state = "closed"
        # Only the attempt that IS held clears the hold. A refusal terminal for ordinal N+1
        # closing while ordinal N runs must not evict the running attempt from `live`.
        if self.live.get(attempt.request_id) is attempt:
            self.live.pop(attempt.request_id, None)
        return "closed"

    # ------------------------------------------------------------------ recovery

    def recover(self) -> list[pb.ActiveAttempt]:
        """Replay the journal into `recovered_attempts` for the new session's Register.

        Two states survive a restart and each becomes a DURABLE, REPLAYABLE TERMINAL here —
        never a half-built live attempt:

        * a journaled TERMINAL replays verbatim, if and only if the three-authorization rule
          admits it (persisted assignment, stable `instance_id`, matching execution digest);
        * an ACCEPTED row with no terminal is accepted-but-incomplete. Its execution context
          died with the process, so recovery MINTS `ABANDONED/EXECUTOR_INVALIDATED`,
          journals it, and reports it as terminal-pending-ack. It is never re-executed and
          it never re-enters `live` as an attempt with no plan and no grant — an attempt
          object that cannot answer "what would you run" has no business being dispatchable.

        The identity a recovered ABANDONED terminal needs — request id, ordinal, execution
        digest, instance — is journaled at ACCEPTANCE, before the attempt can start, which
        is what makes this reconstruction possible without guessing.
        """
        recovered: list[pb.ActiveAttempt] = []
        for (request_id, number), row in self.journal.attempts().items():
            if row["state"] == "closed":
                continue
            digest = documents.raw(row["exec_spec_digest"])
            authorized = row.get("instance_id") == self.instance_id and row.get(
                "exec_spec_digest"
            ) == documents.spell(digest)
            if not authorized:
                continue  # the three-authorization rule: unauthorized never replays
            attempt = Attempt(
                request_id=request_id,
                attempt=number,
                digest=digest,
                spec={},
                grant=pb.DeliveryGrant(),
                state="terminal",
            )
            if row["state"] == "terminal":
                attempt.terminal_id = row["terminal_id"]
                attempt.terminal_digest = documents.raw(row["terminal_digest"])
                attempt.terminal_bytes = base64.b64decode(row["terminal_canonical_b64"])
            elif row["state"] == "accepted":
                self.terminal(
                    attempt,
                    STATUS.TERMINAL_STATUS_ABANDONED,
                    CAUSE.CAUSE_CODE_EXECUTOR_INVALIDATED,
                    ORIGIN.CAUSE_ORIGIN_SUPERVISOR,
                    "accepted-but-incomplete across a control-parent restart: the execution "
                    "context is gone, so this attempt reports ABANDONED and is never "
                    "re-executed",
                )
            else:
                continue
            self.history[attempt.key()] = attempt
            recovered.append(
                pb.ActiveAttempt(
                    request_id=request_id,
                    attempt=number,
                    kind=pb.AttemptKind.ATTEMPT_KIND_SERVING,
                    state=pb.AttemptState.ATTEMPT_STATE_TERMINAL_PENDING_ACK,
                    exec_spec_digest=digest,
                )
            )
        self.recovered = recovered
        return recovered

    def pending_terminals(self) -> list[Attempt]:
        return [a for a in self.history.values() if a.state == "terminal"]

    def active(self) -> list[pb.ActiveAttempt]:
        out = []
        for attempt in self.live.values():
            out.append(
                pb.ActiveAttempt(
                    request_id=attempt.request_id,
                    attempt=attempt.attempt,
                    kind=pb.AttemptKind.ATTEMPT_KIND_SERVING,
                    state=pb.AttemptState.ATTEMPT_STATE_TERMINAL_PENDING_ACK
                    if attempt.state == "terminal"
                    else pb.AttemptState.ATTEMPT_STATE_RUNNING,
                    exec_spec_digest=attempt.digest,
                )
            )
        return out
