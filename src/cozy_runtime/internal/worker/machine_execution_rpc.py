"""The machine execution methods on the existing authenticated WorkerControl server."""

from __future__ import annotations

import base64
import dataclasses
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal, cast

import grpc

from cozy_runtime.author import ConformanceError
from cozy_runtime.internal import canonical
from cozy_runtime.internal.package_environment import EnvironmentRefusal
from cozy_runtime.internal.worker import machine_capture, machine_slots, triage
from cozy_runtime.internal.worker.plan import PlanRefusal
from cozy_runtime.internal.worker.workspace import WorkspaceRefusal
from cozy_runtime.internal.worker.workspace_executions import (
    ExecutionWorkspaceRefusal,
    StaleExecutionGeneration,
)
from cozy_runtime.protocol import (
    MACHINE_PUBLICATION_AUTHORITY_WIRE_MINOR,
    WORKSPACE_FENCED_EXECUTION_WIRE_MINOR,
    documents,
)
from cozy_runtime.protocol import worker_pb2 as pb

if TYPE_CHECKING:
    from .session import Worker


class _ExecutionNotFound(WorkspaceRefusal):
    """Authenticated lookup proved absence in the selected execution workspace."""


class MachineExecutionRPC:
    def __init__(self, worker: Worker):
        self.worker = worker

    def _call(self, claim: pb.Claim, context: Any, operation: Callable[[], Any]) -> Any:
        try:
            if claim.wire_minor < WORKSPACE_FENCED_EXECUTION_WIRE_MINOR:
                raise WorkspaceRefusal("workspace-fenced machine execution requires wire59")
            if self.worker.executions is None:
                raise WorkspaceRefusal("machine execution workspace is unavailable")
            self.worker.authorize_workspace(claim)
            return operation()
        except _ExecutionNotFound as exc:
            return context.abort(grpc.StatusCode.NOT_FOUND, str(exc))
        except ExecutionWorkspaceRefusal as exc:
            context.set_trailing_metadata((("cozy-error-code", exc.code),))
            return context.abort(grpc.StatusCode.FAILED_PRECONDITION, str(exc))
        except StaleExecutionGeneration as exc:
            context.set_trailing_metadata((("cozy-error-code", exc.code),))
            return context.abort(grpc.StatusCode.ABORTED, str(exc))
        except (
            ValueError,
            canonical.CanonicalError,
            OSError,
            documents.DocumentError,
            EnvironmentRefusal,
            ConformanceError,
            PlanRefusal,
        ) as exc:
            return context.abort(grpc.StatusCode.FAILED_PRECONDITION, str(exc)[:512])
        except Exception as exc:
            code = getattr(exc, "code", None)
            if code in {"ROOT_ABSENT", "OBJECT_ABSENT", "OBJECT_CORRUPT", "LENGTH_MISMATCH"}:
                return context.abort(
                    grpc.StatusCode.FAILED_PRECONDITION,
                    f"retained execution output is unavailable: {code}",
                )
            return context.abort(grpc.StatusCode.UNAVAILABLE, "machine execution operation failed")

    def _query(self, query: pb.MachineExecutionQuery) -> None:
        executions = self.worker.executions
        if executions is None:
            raise WorkspaceRefusal("machine execution workspace is unavailable")
        executions.require_workspace(query.expected_execution_workspace_id)

    def GetMachineExecutionWorkspace(
        self, request: pb.MachineExecutionWorkspaceQuery, context: Any
    ) -> Any:
        def get() -> pb.MachineExecutionWorkspace:
            assert self.worker.executions is not None
            options = self.worker.options
            return pb.MachineExecutionWorkspace(
                worker_id=options.worker_id,
                worker_boot_id=self.worker.fence.worker_boot_id,
                execution_workspace_id=self.worker.executions.workspace_id,
                devices=[
                    pb.MachineDevice(
                        ordinal=ordinal,
                        name=gpu["device_name"],
                        uuid=gpu["device_uuid"],
                        memory_bytes=gpu["memory_bytes"],
                        pci_bus_id=gpu["pci_bus_id"],
                        driver_version=gpu["driver_version"],
                    )
                    for ordinal, gpu in enumerate(options.gpus)
                ],
                # A backend without a measured inventory (a flag-launched `serve`) is unknown.
                accelerator_backend=options.accelerator_backend
                if options.gpus or options.accelerator_backend == "none"
                else "",
                executor_uid_isolation=options.executor_uid >= 0,
                cpu_slot_model_inputs=machine_slots.MODEL_INPUTS,
                exact_execution_gpus=True,
            )

        return self._call(request.claim, context, get)

    def _state(self, state: Any) -> pb.MachineExecutionState:
        assert self.worker.executions is not None
        return pb.MachineExecutionState(
            **state.document(),
            worker_id=self.worker.options.worker_id,
            worker_boot_id=self.worker.fence.worker_boot_id,
            execution_workspace_id=self.worker.executions.workspace_id,
            gpu=self.worker.gpu_status(state.request_id),
        )

    def SubmitMachineExecution(self, request: pb.MachineExecutionSubmit, context: Any) -> Any:
        def submit() -> pb.MachineExecutionReceipt:
            assert self.worker.executions is not None
            self.worker.executions.require_workspace(request.expected_execution_workspace_id)
            if (
                request.publication_authorization_id
                and request.claim.wire_minor < MACHINE_PUBLICATION_AUTHORITY_WIRE_MINOR
            ):
                raise WorkspaceRefusal("publication authority requires Runtime wire52")
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
                if prepared.get("state") != machine_capture.state_identity(request.prepared_state):
                    raise WorkspaceRefusal("execution submission changed its prepared state")
                receipt = executions.submit(
                    owner,
                    request.submission_id,
                    request.capture_digest,
                    offered,
                    expected_execution_workspace_id=request.expected_execution_workspace_id,
                    max_attempts=request.max_attempts or 1,
                    retry_delay_ms=request.retry_delay_ms,
                    capture_document=request.capture_canonical_bytes,
                    preparation=canonical.write(prepared),
                    publication_authorization_id=request.publication_authorization_id,
                )
            else:
                preparation = machine_capture.verify(self.worker, request)
                receipt = self.worker.submit_execution(
                    request.claim,
                    request.submission_id,
                    request.capture_digest,
                    offered,
                    expected_execution_workspace_id=request.expected_execution_workspace_id,
                    max_attempts=request.max_attempts or 1,
                    retry_delay_ms=request.retry_delay_ms,
                    capture_document=request.capture_canonical_bytes,
                    preparation=preparation,
                    publication_authorization_id=request.publication_authorization_id,
                    arguments=value,
                )
            assert self.worker.executions is not None
            return pb.MachineExecutionReceipt(**dataclasses.asdict(receipt))

        def checked_submit() -> pb.MachineExecutionReceipt:
            try:
                return submit()
            except ExecutionWorkspaceRefusal:
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

    def GetMachineExecution(self, request: pb.MachineExecutionQuery, context: Any) -> Any:
        def get() -> pb.MachineExecutionState:
            self._query(request)
            assert self.worker.executions is not None
            if not self.worker.executions.owns(request.claim.record_owner_id, request.request_id):
                raise _ExecutionNotFound("execution is not held by this owner")
            return self._state(self.worker.execution_status(request.claim, request.request_id))

        return self._call(request.claim, context, get)

    def ListMachineExecutionEvents(
        self, request: pb.MachineExecutionEventsQuery, context: Any
    ) -> Any:
        def events() -> pb.MachineExecutionEventPage:
            self._query(request.execution)
            page = self.worker.execution_events(
                request.execution.claim,
                request.execution.request_id,
                request.after,
                request.limit or 256,
            )
            return pb.MachineExecutionEventPage(
                events=[
                    pb.MachineExecutionEvent(
                        sequence=e.sequence,
                        attempt_ordinal=e.attempt_ordinal,
                        at_ms=e.at_ms,
                        kind=e.kind,
                        body_canonical_bytes=e.body,
                    )
                    for e in page.events
                ],
                next_after=page.next_after,
                head_sequence=page.head_sequence,
                compacted_through=page.compacted_through,
            )

        return self._call(request.execution.claim, context, events)

    def ControlMachineExecution(self, request: pb.MachineExecutionControl, context: Any) -> Any:
        def control() -> pb.MachineExecutionState:
            self._query(request.execution)
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

    def CollectMachineExecution(self, request: pb.MachineExecutionCollect, context: Any) -> Any:
        def collect() -> pb.AttemptOutcome:
            self._query(request.execution)
            return self.worker.collect_execution(
                request.execution.claim,
                request.execution.request_id,
                request.attempt_ordinal or None,
            )

        return self._call(request.execution.claim, context, collect)

    def ReadMachineExecutionTriage(
        self, request: pb.MachineExecutionTriageQuery, context: Any
    ) -> Any:
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

    def AcknowledgeMachineExecutionCollection(
        self, request: pb.MachineExecutionCollectionAck, context: Any
    ) -> Any:
        def acknowledge() -> pb.MachineExecutionState:
            self._query(request.execution)
            if request.outcome.request_id != request.execution.request_id:
                raise WorkspaceRefusal("collection acknowledgment names another request")
            return self._state(
                self.worker.acknowledge_execution_collection(
                    request.execution.claim, request.outcome
                )
            )

        return self._call(request.execution.claim, context, acknowledge)
