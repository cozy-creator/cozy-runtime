"""The machine execution methods on the existing authenticated WorkerControl server."""

from __future__ import annotations

import base64
import dataclasses
import threading
from collections.abc import Callable
from contextlib import nullcontext
from typing import TYPE_CHECKING, Literal, cast

import grpc

from cozy_runtime.author import ConformanceError
from cozy_runtime.internal import canonical, prepare_diagnostics
from cozy_runtime.internal.package_environment import EnvironmentRefusal
from cozy_runtime.internal.worker import (
    machine_capture,
    machine_model_resolve,
    machine_reads,
    machine_release_roots,
    triage,
)
from cozy_runtime.internal.worker.plan import PlanRefusal
from cozy_runtime.internal.worker.servicer_context import ServicerContext
from cozy_runtime.internal.worker.workspace import (
    WorkspaceBusy,
    WorkspaceRefusal,
)
from cozy_runtime.internal.worker.workspace_executions import (
    MAX_PAGE,
    TERMINAL,
    EventPage,
    ExecutionWorkspaceRefusal,
    Run,
    StaleExecutionGeneration,
    State,
)
from cozy_runtime.protocol import documents
from cozy_runtime.protocol import worker_pb2 as pb

if TYPE_CHECKING:
    from .session import Worker


class _ExecutionNotFound(WorkspaceRefusal):
    """Authenticated lookup proved absence in the selected execution workspace."""


class MachineExecutionRPC:
    def __init__(self, worker: Worker):
        self.worker = worker

    def _call[T](self, claim: pb.Claim, context: ServicerContext, operation: Callable[[], T]) -> T:
        try:
            if self.worker.executions is None:
                raise WorkspaceRefusal("machine execution workspace is unavailable")
            self.worker.authorize_workspace(claim)
            return operation()
        except _ExecutionNotFound as exc:
            context.abort(grpc.StatusCode.NOT_FOUND, str(exc))
        except machine_release_roots.Preparing as exc:
            trailers = [("cozy-error-code", "release_root_preparing")]
            if exc.total:  # bytes landed of the Models' total, while they download
                trailers.append(("cozy-progress-bytes", f"{exc.moved} {exc.total}"))
            context.set_trailing_metadata(tuple(trailers))
            context.abort(grpc.StatusCode.UNAVAILABLE, str(exc)[:512])
        except ExecutionWorkspaceRefusal as exc:
            context.set_trailing_metadata((("cozy-error-code", exc.code),))
            context.abort(grpc.StatusCode.FAILED_PRECONDITION, str(exc))
        except StaleExecutionGeneration as exc:
            context.set_trailing_metadata((("cozy-error-code", exc.code),))
            context.abort(grpc.StatusCode.ABORTED, str(exc))
        except WorkspaceBusy as exc:
            # A foreground transition is "not now", never a verdict: the caller asks again.
            context.set_trailing_metadata((("cozy-error-code", "workspace_not_now"),))
            context.abort(grpc.StatusCode.UNAVAILABLE, str(exc)[:512])
        except (
            ValueError,
            canonical.CanonicalError,
            OSError,
            documents.DocumentError,
            EnvironmentRefusal,
            ConformanceError,
            PlanRefusal,
        ) as exc:
            context.abort(grpc.StatusCode.FAILED_PRECONDITION, str(exc)[:512])
        except Exception as exc:
            code = getattr(exc, "code", None)
            if code in {"ROOT_ABSENT", "OBJECT_ABSENT", "OBJECT_CORRUPT", "LENGTH_MISMATCH"}:
                context.abort(
                    grpc.StatusCode.FAILED_PRECONDITION,
                    f"retained execution output is unavailable: {code}",
                )
            # Unexpected: its type and frame locations (never its message) go to the log,
            # and its type to the caller, so the failure is diagnosable from either side.
            name = type(exc).__name__
            self.worker.note("machine_execution", f"operation failed unexpectedly: {name}")
            trace = prepare_diagnostics.exception_trace(exc)
            for line in prepare_diagnostics.activity_lines(trace):
                self.worker.note("machine_execution", f"traceback: {line}")
            context.abort(
                grpc.StatusCode.UNAVAILABLE, f"machine execution operation failed ({name})"
            )

    def _query(self, query: pb.MachineExecutionQuery) -> None:
        executions = self.worker.executions
        if executions is None:
            raise WorkspaceRefusal("machine execution workspace is unavailable")
        executions.require_workspace(query.expected_execution_workspace_id)

    def GetMachineExecutionWorkspace(
        self, request: pb.MachineExecutionWorkspaceQuery, context: ServicerContext
    ) -> pb.MachineExecutionWorkspace:
        def get() -> pb.MachineExecutionWorkspace:
            assert self.worker.executions is not None
            options, calls = self.worker.options, self.worker.machine_calls
            described = (
                calls.release_roots.describe(request.describe)
                if calls is not None and request.HasField("describe")
                else None
            )
            return pb.MachineExecutionWorkspace(
                worker_id=options.worker_id,
                worker_boot_id=self.worker.fence.worker_boot_id,
                execution_workspace_id=self.worker.executions.workspace_id,
                devices=machine_reads.devices(self.worker),
                accelerator_backend=machine_reads.accelerator_backend(self.worker),
                executor_uid_isolation=options.executor_uid >= 0,
                described_release=described,
                run_output_log=True,
                release_root_owner=True,
                submission_close=True,
                model_overrides=True,
            )

        return self._call(request.claim, context, get)

    def _state(self, state: State, run: Run | None = None) -> pb.MachineExecutionState:
        assert self.worker.executions is not None
        owner = self.worker.fence.record_owner_id
        if run is None:
            run = self.worker.executions.run(owner, state.request_id)
        message = pb.MachineExecutionState(
            request_id=state.request_id,
            attempt_ordinal=state.attempt_ordinal,
            generation=state.generation,
            state=state.state,
            collected=state.collected,
            sequence=state.sequence,
            worker_id=self.worker.options.worker_id,
            worker_boot_id=self.worker.fence.worker_boot_id,
            execution_workspace_id=self.worker.executions.workspace_id,
            gpu=self.worker.gpu_status(state.request_id),
            awaiting_source_credentials=[]
            if state.state in TERMINAL
            else self.worker.machine_sources.awaiting(owner, state.request_id),
        )
        if run is not None:
            message.number = run.number
            message.accepted_at_ms = run.accepted_ms
            # A replayed command answers its journaled state, which may no longer be current.
            message.finished_at_ms = run.finished_ms if run.status() == state else 0
            message.target.CopyFrom(machine_reads.target(self.worker, run))
        return message

    def SubmitMachineExecution(
        self, request: pb.MachineExecutionSubmit, context: ServicerContext
    ) -> pb.MachineExecutionReceipt:
        if request.HasField("release_root"):
            return self._submit_release_root(request, context)

        def submit() -> pb.MachineExecutionReceipt:
            assert self.worker.executions is not None
            self.worker.executions.require_workspace(request.expected_execution_workspace_id)
            payload = request.payload_canonical_bytes
            if not 0 < len(payload) <= canonical.DOC_MAX_BYTES:
                raise WorkspaceRefusal("execution payload exceeds its inline argument bound")
            value = canonical.parse_canonical(payload)
            if not isinstance(value, dict):
                raise WorkspaceRefusal("execution arguments must be a canonical object")
            assert self.worker.executions is not None
            self.worker.executions.workspace.validate_offer(request.offer)
            spec = documents.read(request.offer.invocation_spec_canonical_bytes, pb.InvocationSpec)
            if documents.spell(documents.digest_of(payload)) != spec.get("payload_digest"):
                raise WorkspaceRefusal("execution payload differs from its invocation digest")
            offered = pb.AttemptOffer()
            offered.CopyFrom(request.offer)
            inputs = [entry for entry in offered.grant.inputs if entry.input_id == "payload"]
            if len(inputs) != 1:
                raise WorkspaceRefusal("execution payload has no unique input grant")
            inputs[0].url = "data:application/json;base64," + base64.b64encode(payload).decode()
            executions = self.worker.executions
            owner = self.worker.authorize_workspace(request.claim)
            if executions.owns(owner, offered.request_id):
                # Reconcile lost acceptance replies from the durable record itself,
                # even after the worker's in-memory prepared registry disappeared.
                prepared = executions.preparation(owner, offered.request_id)
                # Reconciliation uses the scope accepted in the journal, even after
                # that Hub's bearer expires. It never selects a new current grant.
                hub = request.hub
                own = self.worker.options.publication_authority
                if own is not None and machine_model_resolve.same_origin(hub, own.origin):
                    hub = ""
                held_hub = str(prepared.get("hub", ""))
                if hub != held_hub and not machine_model_resolve.same_origin(hub, held_hub):
                    raise WorkspaceRefusal("execution submission changed its Hub scope")
                if prepared.get("state") != machine_capture.state_identity(request.prepared_state):
                    raise WorkspaceRefusal("execution submission changed its prepared state")
                receipt = executions.submit(
                    owner,
                    request.submission_id,
                    request.capture_digest,
                    offered,
                    expected_execution_workspace_id=request.expected_execution_workspace_id,
                    capture_document=request.capture_canonical_bytes,
                    preparation=canonical.write(prepared),
                    publication_authorization_id=request.publication_authorization_id,
                    owner_memo=request.owner_memo,
                )
                self.worker.machine_sources.hold(
                    owner, offered.request_id, request.source_credentials
                )
            else:
                preparation = machine_capture.verify(self.worker, request)
                receipt = self.worker.submit_execution(
                    request.claim,
                    request.submission_id,
                    request.capture_digest,
                    offered,
                    expected_execution_workspace_id=request.expected_execution_workspace_id,
                    capture_document=request.capture_canonical_bytes,
                    preparation=preparation,
                    publication_authorization_id=request.publication_authorization_id,
                    arguments=value,
                    owner_memo=request.owner_memo,
                    source_credentials=request.source_credentials,
                )
            self.worker.machine_sources.declare(owner, offered.request_id)
            return pb.MachineExecutionReceipt(**dataclasses.asdict(receipt))

        def checked_submit() -> pb.MachineExecutionReceipt:
            try:
                return submit()
            except (ExecutionWorkspaceRefusal, WorkspaceBusy):
                raise
            except Exception:
                # A lost reply after commit stays ambiguous. Only an actual
                # journal read proving absence permits the client to stop replay.
                try:
                    executions = self.worker.executions
                    if executions is not None and executions.submission_absent(
                        request.claim.record_owner_id,
                        request.offer.request_id,
                        request.expected_execution_workspace_id,
                    ):
                        context.set_trailing_metadata(
                            (("cozy-error-code", "execution_submission_refused"),)
                        )
                except ExecutionWorkspaceRefusal:
                    raise
                except Exception:
                    pass  # Failed readback is ambiguous, never definitive nonacceptance.
                raise

        return self._call(request.claim, context, checked_submit)

    def _submit_release_root(
        self, request: pb.MachineExecutionSubmit, context: ServicerContext
    ) -> pb.MachineExecutionReceipt:
        """Answer a receipt or the preparation's progress."""

        def submit() -> pb.MachineExecutionReceipt:
            calls, executions = self.worker.machine_calls, self.worker.executions
            if calls is None or executions is None:
                raise WorkspaceRefusal("release roots need the machine execution workspace")
            if (
                request.capture_digest
                or request.capture_canonical_bytes
                or request.HasField("prepared_state")
                or request.offer.invocation_spec_canonical_bytes
                or not request.offer.request_id
            ):
                raise WorkspaceRefusal("a release root names only its request, never a capture")
            owner = self.worker.authorize_workspace(request.claim)
            try:
                return calls.release_roots.submit(owner, request, context)
            except WorkspaceBusy:
                raise
            except WorkspaceRefusal as exc:
                # Read back acceptance; clients close this key before rerouting it.
                if executions.submission_absent(
                    request.claim.record_owner_id,
                    request.offer.request_id,
                    request.expected_execution_workspace_id,
                ):
                    if isinstance(exc, ExecutionWorkspaceRefusal) and exc.code in {
                        "hub_access_absent",
                        "hub_access_expired",
                        "hub_access_principal_conflict",
                    }:
                        # An access refusal is actionable at the controller. Preserve
                        # disposition plus cause instead of losing either to _call.
                        raise ExecutionWorkspaceRefusal(
                            "execution_submission_refused", f"{exc.code}: {exc}"
                        ) from exc
                    # Other typed refusals retain their operation-specific recovery
                    # code, e.g. release_root_installation_absent asks for reprepare.
                    context.set_trailing_metadata(
                        (("cozy-error-code", "execution_submission_refused"),)
                    )
                raise

        return self._call(request.claim, context, submit)

    def CloseMachineSubmission(
        self, request: pb.MachineSubmissionClose, context: ServicerContext
    ) -> pb.MachineSubmissionClosure:
        def close() -> pb.MachineSubmissionClosure:
            assert self.worker.executions is not None
            owner = self.worker.authorize_workspace(request.claim)
            receipt = self.worker.executions.close_submission(
                owner,
                request.submission_id,
                request.request_id,
                request.expected_execution_workspace_id,
            )
            return pb.MachineSubmissionClosure(
                submission_id=request.submission_id,
                request_id=request.request_id,
                execution_workspace_id=request.expected_execution_workspace_id,
                receipt=pb.MachineExecutionReceipt(**dataclasses.asdict(receipt))
                if receipt
                else None,
            )

        return self._call(request.claim, context, close)

    def GetMachineExecution(
        self, request: pb.MachineExecutionQuery, context: ServicerContext
    ) -> pb.MachineExecutionState:
        def get() -> pb.MachineExecutionState:
            self._query(request)
            assert self.worker.executions is not None
            if not self.worker.executions.owns(request.claim.record_owner_id, request.request_id):
                raise _ExecutionNotFound("execution is not held by this owner")
            return self._state(self.worker.execution_status(request.claim, request.request_id))

        return self._call(request.claim, context, get)

    def ListMachineExecutionEvents(
        self, request: pb.MachineExecutionEventsQuery, context: ServicerContext
    ) -> pb.MachineExecutionEventPage:
        def events() -> pb.MachineExecutionEventPage:
            self._query(request.execution)
            if self.worker.owner_memo is not None:
                self.worker.owner_memo.read(
                    request.execution.claim.record_owner_id,
                    request.execution.request_id,
                    request.after,
                )
            if request.wait:
                page = self._wait_events(request, context)
            else:
                page = self.worker.execution_events(
                    request.execution.claim,
                    request.execution.request_id,
                    request.after,
                    request.limit or 256,
                )
            reply = pb.MachineExecutionEventPage(
                next_after=page.next_after,
                head_sequence=page.head_sequence,
                compacted_through=page.compacted_through,
            )
            for e in page.events:
                event = reply.events.add(
                    sequence=e.sequence,
                    attempt_ordinal=e.attempt_ordinal,
                    at_ms=e.at_ms,
                    kind=e.kind,
                    body_canonical_bytes=e.body,
                )
                if e.kind == "product":
                    event.product.CopyFrom(documents.parse(e.body, pb.RunProduct))
                elif e.kind == "outcome":
                    # The run's terminal entry carries its exact outcome, and ends the page:
                    # the outcome may be as large as a page.
                    assert self.worker.executions is not None
                    event.outcome.CopyFrom(
                        self.worker.executions.outcome(
                            self.worker.authorize_workspace(request.execution.claim),
                            request.execution.request_id,
                            e.attempt_ordinal,
                        )
                    )
                    reply.next_after = e.sequence
                    break
            return reply

        return self._call(request.execution.claim, context, events)

    def _wait_events(
        self, request: pb.MachineExecutionEventsQuery, context: ServicerContext
    ) -> EventPage:
        executions = self.worker.executions
        assert executions is not None
        owner = self.worker.authorize_workspace(request.execution.claim)
        changes = executions.workspace.changes()
        left = threading.Event()

        def leave() -> None:
            left.set()
            changes.bump()

        if not context.add_callback(leave):
            leave()
        memo = self.worker.owner_memo
        with (
            memo.wait_read(owner, request.execution.request_id, left.is_set)
            if memo is not None
            else nullcontext()
        ):
            return executions.wait_events(
                owner,
                request.execution.request_id,
                request.after,
                request.limit or 256,
                lambda: left.is_set() or self.worker.stop.is_set(),
            )

    def ControlMachineExecution(
        self, request: pb.MachineExecutionControl, context: ServicerContext
    ) -> pb.MachineExecutionState:
        def control() -> pb.MachineExecutionState:
            self._query(request.execution)
            if request.action == pb.MACHINE_EXECUTION_ACTION_RECONCILE_PUBLICATION:
                return self._state(
                    self.worker.reconcile_publication(
                        request.execution.claim, request.execution.request_id, request.publication
                    )
                )
            if request.action == pb.MACHINE_EXECUTION_ACTION_ANSWER_MEMO:
                return self._state(self._answer_memo(request))
            action = {
                pb.MACHINE_EXECUTION_ACTION_PAUSE: "pause",
                pb.MACHINE_EXECUTION_ACTION_RESUME: "resume",
                pb.MACHINE_EXECUTION_ACTION_CANCEL: "cancel",
            }.get(request.action)
            if action is None:
                raise WorkspaceRefusal("execution control action is unspecified")
            state = self.worker.control_execution(
                request.execution.claim,
                request.execution.request_id,
                request.command_id,
                request.expected_generation,
                cast(Literal["pause", "resume", "cancel"], action),
            )
            return self._state(state)

        return self._call(request.execution.claim, context, control)

    def _answer_memo(self, request: pb.MachineExecutionControl) -> State:
        memo, executions = self.worker.owner_memo, self.worker.executions
        claim, request_id = request.execution.claim, request.execution.request_id
        if memo is None or executions is None:
            raise ExecutionWorkspaceRefusal("memo_lookup_unsupported", "workspace has no memo")
        if not executions.owns(claim.record_owner_id, request_id):
            raise _ExecutionNotFound("execution is not held by this owner")
        memo.answer(claim.record_owner_id, request_id, request.memo)
        return self.worker.execution_status(claim, request_id)

    def CollectMachineExecution(
        self, request: pb.MachineExecutionCollect, context: ServicerContext
    ) -> pb.AttemptOutcome:
        def collect() -> pb.AttemptOutcome:
            self._query(request.execution)
            return self.worker.collect_execution(
                request.execution.claim,
                request.execution.request_id,
                request.attempt_ordinal or None,
            )

        return self._call(request.execution.claim, context, collect)

    def ReadMachineExecutionTriage(
        self, request: pb.MachineExecutionTriageQuery, context: ServicerContext
    ) -> pb.MachineExecutionTriage:
        def read() -> pb.MachineExecutionTriage:
            self._query(request.execution)
            executions = self.worker.executions
            assert executions is not None
            owner, request_id = (
                request.execution.claim.record_owner_id,
                request.execution.request_id,
            )
            if not executions.owns(owner, request_id):
                raise _ExecutionNotFound("execution is not held by this owner")
            outcome = executions.outcome(owner, request_id, request.attempt_ordinal or None)
            body = documents.read(outcome.outcome_canonical_bytes, pb.AttemptOutcomeBody)
            ref = body.get("triage_bundle")
            if not ref:
                raise _ExecutionNotFound("the attempt's outcome names no triage bundle")
            try:
                data = triage.retained_bytes(
                    self.worker.bundles.root,
                    str(ref["subject_id"]),
                    str(ref["write_receipt_digest"]),
                    int(ref["length"]),
                )
            except triage.TriageError as exc:
                if exc.code == "bundle_absent":
                    raise _ExecutionNotFound(exc.detail) from exc
                raise WorkspaceRefusal(f"{exc.code}: {exc.detail}") from exc
            return pb.MachineExecutionTriage(
                bundle=pb.TriageBundleRef(
                    subject_id=str(ref["subject_id"]),
                    write_receipt_digest=documents.raw(str(ref["write_receipt_digest"])),
                    length=int(ref["length"]),
                ),
                bundle_canonical_bytes=data,
            )

        return self._call(request.execution.claim, context, read)

    def ListMachineExecutions(
        self, request: pb.MachineExecutionListQuery, context: ServicerContext
    ) -> pb.MachineExecutionList:
        def listing() -> pb.MachineExecutionList:
            executions = self.worker.executions
            assert executions is not None
            owner = self.worker.authorize_workspace(request.claim)
            limit, states = min(request.limit or 64, MAX_PAGE), tuple(request.states)
            if request.wait and not request.newest_first:
                runs, head = self._wait_runs(owner, request.after_number, limit, states, context)
            else:
                runs, head = executions.runs(
                    owner,
                    after=request.after_number,
                    before=request.before_number,
                    newest_first=request.newest_first,
                    limit=limit,
                    states=states,
                )
            return pb.MachineExecutionList(
                executions=[self._state(run.status(), run) for run in runs],
                head_number=head,
                execution_workspace_id=executions.workspace_id,
            )

        return self._call(request.claim, context, listing)

    def _wait_runs(
        self, owner: str, after: int, limit: int, states: tuple[str, ...], context: ServicerContext
    ) -> tuple[list[Run], int]:
        executions = self.worker.executions
        assert executions is not None
        changes = executions.workspace.changes()
        left = threading.Event()

        def leave() -> None:
            left.set()
            changes.bump()

        if not context.add_callback(leave):
            leave()
        return executions.wait_runs(
            owner, after, limit, states, lambda: left.is_set() or self.worker.stop.is_set()
        )

    def ListPackages(
        self, request: pb.PackageListQuery, context: ServicerContext
    ) -> pb.PackageList:
        return self._call(request.claim, context, lambda: machine_reads.packages(self.worker))

    def ListModels(self, request: pb.ModelListQuery, context: ServicerContext) -> pb.ModelList:
        return self._call(request.claim, context, lambda: machine_reads.models(self.worker))

    def DescribeMachine(
        self, request: pb.DescribeMachineQuery, context: ServicerContext
    ) -> pb.MachineDescription:
        def describe() -> pb.MachineDescription:
            return pb.MachineDescription(
                worker_id=self.worker.options.worker_id,
                worker_boot_id=self.worker.fence.worker_boot_id,
                runtime=machine_reads.runtime(self.worker),
            )

        return self._call(request.claim, context, describe)

    def AcknowledgeMachineExecutionCollection(
        self, request: pb.MachineExecutionCollectionAck, context: ServicerContext
    ) -> pb.MachineExecutionState:
        def acknowledge() -> pb.MachineExecutionState:
            self._query(request.execution)
            if request.outcome.request_id != request.execution.request_id:
                raise WorkspaceRefusal("collection acknowledgment names another request")
            state = self.worker.acknowledge_execution_collection(
                request.execution.claim, request.outcome
            )
            self.worker.machine_sources.forget(
                request.execution.claim.record_owner_id, request.execution.request_id
            )
            return self._state(state)

        return self._call(request.execution.claim, context, acknowledge)
