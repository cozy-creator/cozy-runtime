from google.protobuf import empty_pb2 as _empty_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class Posture(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    POSTURE_UNSPECIFIED: _ClassVar[Posture]
    POSTURE_ACCEPTING: _ClassVar[Posture]
    POSTURE_DRAINING: _ClassVar[Posture]

class IntakeState(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    INTAKE_STATE_UNSPECIFIED: _ClassVar[IntakeState]
    INTAKE_STATE_BOOTING: _ClassVar[IntakeState]
    INTAKE_STATE_DOWNLOADING: _ClassVar[IntakeState]
    INTAKE_STATE_LOADING: _ClassVar[IntakeState]
    INTAKE_STATE_READY: _ClassVar[IntakeState]
    INTAKE_STATE_DRAINING: _ClassVar[IntakeState]
    INTAKE_STATE_ERROR: _ClassVar[IntakeState]

class AttemptKind(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    ATTEMPT_KIND_UNSPECIFIED: _ClassVar[AttemptKind]
    ATTEMPT_KIND_SERVING: _ClassVar[AttemptKind]
    ATTEMPT_KIND_JOB: _ClassVar[AttemptKind]

class AttemptState(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    ATTEMPT_STATE_UNSPECIFIED: _ClassVar[AttemptState]
    ATTEMPT_STATE_RUNNING: _ClassVar[AttemptState]
    ATTEMPT_STATE_TERMINAL_PENDING_ACK: _ClassVar[AttemptState]

class TerminalStatus(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    TERMINAL_STATUS_UNSPECIFIED: _ClassVar[TerminalStatus]
    TERMINAL_STATUS_SUCCEEDED: _ClassVar[TerminalStatus]
    TERMINAL_STATUS_REFUSED: _ClassVar[TerminalStatus]
    TERMINAL_STATUS_FAILED: _ClassVar[TerminalStatus]
    TERMINAL_STATUS_CANCELED: _ClassVar[TerminalStatus]
    TERMINAL_STATUS_ABANDONED: _ClassVar[TerminalStatus]

class CauseCode(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    CAUSE_CODE_UNSPECIFIED: _ClassVar[CauseCode]
    CAUSE_CODE_INVALID_REQUEST: _ClassVar[CauseCode]
    CAUSE_CODE_UNSUPPORTED_INPUT: _ClassVar[CauseCode]
    CAUSE_CODE_LOCAL_SAFETY: _ClassVar[CauseCode]
    CAUSE_CODE_PROTOCOL: _ClassVar[CauseCode]
    CAUSE_CODE_CONSTRAINT_INFEASIBLE: _ClassVar[CauseCode]
    CAUSE_CODE_AUTHOR_EXCEPTION: _ClassVar[CauseCode]
    CAUSE_CODE_EXECUTOR_FAULT: _ClassVar[CauseCode]
    CAUSE_CODE_GRANT_EXPIRED: _ClassVar[CauseCode]
    CAUSE_CODE_ARTIFACT_UNFETCHABLE: _ClassVar[CauseCode]
    CAUSE_CODE_CAPABILITY_UNAVAILABLE: _ClassVar[CauseCode]
    CAUSE_CODE_CLIENT_CANCEL: _ClassVar[CauseCode]
    CAUSE_CODE_DEADLINE_EXPIRED: _ClassVar[CauseCode]
    CAUSE_CODE_DRAIN_CANCEL: _ClassVar[CauseCode]
    CAUSE_CODE_POLICY_CANCEL: _ClassVar[CauseCode]
    CAUSE_CODE_SUPERSEDED_CANCEL: _ClassVar[CauseCode]
    CAUSE_CODE_EXECUTOR_INVALIDATED: _ClassVar[CauseCode]

class CauseOrigin(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    CAUSE_ORIGIN_UNSPECIFIED: _ClassVar[CauseOrigin]
    CAUSE_ORIGIN_AUTHOR: _ClassVar[CauseOrigin]
    CAUSE_ORIGIN_RUNTIME: _ClassVar[CauseOrigin]
    CAUSE_ORIGIN_EXECUTOR: _ClassVar[CauseOrigin]
    CAUSE_ORIGIN_SUPERVISOR: _ClassVar[CauseOrigin]
    CAUSE_ORIGIN_INFRA: _ClassVar[CauseOrigin]
    CAUSE_ORIGIN_CLIENT: _ClassVar[CauseOrigin]
    CAUSE_ORIGIN_COORDINATOR: _ClassVar[CauseOrigin]

class CancelReason(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    CANCEL_REASON_UNSPECIFIED: _ClassVar[CancelReason]
    CANCEL_REASON_CLIENT: _ClassVar[CancelReason]
    CANCEL_REASON_DRAIN: _ClassVar[CancelReason]
    CANCEL_REASON_SUPERSEDED: _ClassVar[CancelReason]
    CANCEL_REASON_POLICY: _ClassVar[CancelReason]
    CANCEL_REASON_DEADLINE: _ClassVar[CancelReason]

class RegisterRejection(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    REGISTER_REJECTION_UNSPECIFIED: _ClassVar[RegisterRejection]
    REGISTER_REJECTION_UNAUTHENTICATED: _ClassVar[RegisterRejection]
    REGISTER_REJECTION_SESSION_COLLISION: _ClassVar[RegisterRejection]
    REGISTER_REJECTION_STALE_SESSION: _ClassVar[RegisterRejection]
    REGISTER_REJECTION_WORKER_ID_MISMATCH: _ClassVar[RegisterRejection]
    REGISTER_REJECTION_RELEASE_ID_MISMATCH: _ClassVar[RegisterRejection]
    REGISTER_REJECTION_UNSUPPORTED_MINOR: _ClassVar[RegisterRejection]

class FaultKind(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    FAULT_KIND_UNSPECIFIED: _ClassVar[FaultKind]
    FAULT_KIND_BINDING_UNAVAILABLE: _ClassVar[FaultKind]
    FAULT_KIND_BINDING_DEGRADED: _ClassVar[FaultKind]
    FAULT_KIND_HARDWARE_UNSUITABLE: _ClassVar[FaultKind]
    FAULT_KIND_ARTIFACT_FETCH_FAILED: _ClassVar[FaultKind]
    FAULT_KIND_GRANT_EXPIRED: _ClassVar[FaultKind]
    FAULT_KIND_CREDENTIAL_UNAPPLIED: _ClassVar[FaultKind]
    FAULT_KIND_LOCAL_SAFETY_REFUSAL: _ClassVar[FaultKind]
    FAULT_KIND_EXECUTOR_POISONED: _ClassVar[FaultKind]
    FAULT_KIND_CONFIG_REFUSED: _ClassVar[FaultKind]
    FAULT_KIND_UNKNOWN_DEPLOYMENT: _ClassVar[FaultKind]

class SubjectKind(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    SUBJECT_KIND_UNSPECIFIED: _ClassVar[SubjectKind]
    SUBJECT_KIND_OBJECT: _ClassVar[SubjectKind]
    SUBJECT_KIND_SNAPSHOT: _ClassVar[SubjectKind]
    SUBJECT_KIND_TENSOR_HEADER: _ClassVar[SubjectKind]
    SUBJECT_KIND_CHECKPOINT: _ClassVar[SubjectKind]
    SUBJECT_KIND_ADAPTER_ARTIFACT: _ClassVar[SubjectKind]
    SUBJECT_KIND_LAYOUT: _ClassVar[SubjectKind]
    SUBJECT_KIND_RELEASE: _ClassVar[SubjectKind]

class PurgeFaultCode(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    PURGE_FAULT_CODE_UNSPECIFIED: _ClassVar[PurgeFaultCode]
    PURGE_FAULT_CODE_STALE_TOMBSTONE_GENERATION: _ClassVar[PurgeFaultCode]
    PURGE_FAULT_CODE_STALE_WORKER_GENERATION: _ClassVar[PurgeFaultCode]
    PURGE_FAULT_CODE_CHANGED_BODY_REPLAY: _ClassVar[PurgeFaultCode]
    PURGE_FAULT_CODE_CROSS_TENANT_SUBJECT: _ClassVar[PurgeFaultCode]
    PURGE_FAULT_CODE_LIVE_UNRELATED_BINDING: _ClassVar[PurgeFaultCode]
    PURGE_FAULT_CODE_ERASURE_FAILED: _ClassVar[PurgeFaultCode]
    PURGE_FAULT_CODE_UNKNOWN_SUBJECT: _ClassVar[PurgeFaultCode]
    PURGE_FAULT_CODE_DEADLINE_EXCEEDED: _ClassVar[PurgeFaultCode]

class BootFailureReason(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    BOOT_FAILURE_REASON_UNSPECIFIED: _ClassVar[BootFailureReason]
    BOOT_FAILURE_REASON_HARDWARE_VERDICT: _ClassVar[BootFailureReason]
    BOOT_FAILURE_REASON_IMAGE_FAULT: _ClassVar[BootFailureReason]
    BOOT_FAILURE_REASON_DRIVER_FAULT: _ClassVar[BootFailureReason]
    BOOT_FAILURE_REASON_DISK_SHAPE: _ClassVar[BootFailureReason]
    BOOT_FAILURE_REASON_CONFIG_INVALID: _ClassVar[BootFailureReason]
    BOOT_FAILURE_REASON_OTHER_FATAL: _ClassVar[BootFailureReason]

class CheckpointOutcome(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    CHECKPOINT_OUTCOME_UNSPECIFIED: _ClassVar[CheckpointOutcome]
    CHECKPOINT_OUTCOME_RECORDED: _ClassVar[CheckpointOutcome]
    CHECKPOINT_OUTCOME_REPLAYED: _ClassVar[CheckpointOutcome]
    CHECKPOINT_OUTCOME_CONFLICT: _ClassVar[CheckpointOutcome]
    CHECKPOINT_OUTCOME_REFUSED: _ClassVar[CheckpointOutcome]

class CheckpointFaultCode(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    CHECKPOINT_FAULT_CODE_UNSPECIFIED: _ClassVar[CheckpointFaultCode]
    CHECKPOINT_FAULT_CODE_IDENTITY_CONFLICT: _ClassVar[CheckpointFaultCode]
    CHECKPOINT_FAULT_CODE_UNKNOWN_ATTEMPT: _ClassVar[CheckpointFaultCode]
    CHECKPOINT_FAULT_CODE_NOT_JOB_MODE: _ClassVar[CheckpointFaultCode]
    CHECKPOINT_FAULT_CODE_STALE_SEQUENCE: _ClassVar[CheckpointFaultCode]
    CHECKPOINT_FAULT_CODE_QUOTA_EXCEEDED: _ClassVar[CheckpointFaultCode]
POSTURE_UNSPECIFIED: Posture
POSTURE_ACCEPTING: Posture
POSTURE_DRAINING: Posture
INTAKE_STATE_UNSPECIFIED: IntakeState
INTAKE_STATE_BOOTING: IntakeState
INTAKE_STATE_DOWNLOADING: IntakeState
INTAKE_STATE_LOADING: IntakeState
INTAKE_STATE_READY: IntakeState
INTAKE_STATE_DRAINING: IntakeState
INTAKE_STATE_ERROR: IntakeState
ATTEMPT_KIND_UNSPECIFIED: AttemptKind
ATTEMPT_KIND_SERVING: AttemptKind
ATTEMPT_KIND_JOB: AttemptKind
ATTEMPT_STATE_UNSPECIFIED: AttemptState
ATTEMPT_STATE_RUNNING: AttemptState
ATTEMPT_STATE_TERMINAL_PENDING_ACK: AttemptState
TERMINAL_STATUS_UNSPECIFIED: TerminalStatus
TERMINAL_STATUS_SUCCEEDED: TerminalStatus
TERMINAL_STATUS_REFUSED: TerminalStatus
TERMINAL_STATUS_FAILED: TerminalStatus
TERMINAL_STATUS_CANCELED: TerminalStatus
TERMINAL_STATUS_ABANDONED: TerminalStatus
CAUSE_CODE_UNSPECIFIED: CauseCode
CAUSE_CODE_INVALID_REQUEST: CauseCode
CAUSE_CODE_UNSUPPORTED_INPUT: CauseCode
CAUSE_CODE_LOCAL_SAFETY: CauseCode
CAUSE_CODE_PROTOCOL: CauseCode
CAUSE_CODE_CONSTRAINT_INFEASIBLE: CauseCode
CAUSE_CODE_AUTHOR_EXCEPTION: CauseCode
CAUSE_CODE_EXECUTOR_FAULT: CauseCode
CAUSE_CODE_GRANT_EXPIRED: CauseCode
CAUSE_CODE_ARTIFACT_UNFETCHABLE: CauseCode
CAUSE_CODE_CAPABILITY_UNAVAILABLE: CauseCode
CAUSE_CODE_CLIENT_CANCEL: CauseCode
CAUSE_CODE_DEADLINE_EXPIRED: CauseCode
CAUSE_CODE_DRAIN_CANCEL: CauseCode
CAUSE_CODE_POLICY_CANCEL: CauseCode
CAUSE_CODE_SUPERSEDED_CANCEL: CauseCode
CAUSE_CODE_EXECUTOR_INVALIDATED: CauseCode
CAUSE_ORIGIN_UNSPECIFIED: CauseOrigin
CAUSE_ORIGIN_AUTHOR: CauseOrigin
CAUSE_ORIGIN_RUNTIME: CauseOrigin
CAUSE_ORIGIN_EXECUTOR: CauseOrigin
CAUSE_ORIGIN_SUPERVISOR: CauseOrigin
CAUSE_ORIGIN_INFRA: CauseOrigin
CAUSE_ORIGIN_CLIENT: CauseOrigin
CAUSE_ORIGIN_COORDINATOR: CauseOrigin
CANCEL_REASON_UNSPECIFIED: CancelReason
CANCEL_REASON_CLIENT: CancelReason
CANCEL_REASON_DRAIN: CancelReason
CANCEL_REASON_SUPERSEDED: CancelReason
CANCEL_REASON_POLICY: CancelReason
CANCEL_REASON_DEADLINE: CancelReason
REGISTER_REJECTION_UNSPECIFIED: RegisterRejection
REGISTER_REJECTION_UNAUTHENTICATED: RegisterRejection
REGISTER_REJECTION_SESSION_COLLISION: RegisterRejection
REGISTER_REJECTION_STALE_SESSION: RegisterRejection
REGISTER_REJECTION_WORKER_ID_MISMATCH: RegisterRejection
REGISTER_REJECTION_RELEASE_ID_MISMATCH: RegisterRejection
REGISTER_REJECTION_UNSUPPORTED_MINOR: RegisterRejection
FAULT_KIND_UNSPECIFIED: FaultKind
FAULT_KIND_BINDING_UNAVAILABLE: FaultKind
FAULT_KIND_BINDING_DEGRADED: FaultKind
FAULT_KIND_HARDWARE_UNSUITABLE: FaultKind
FAULT_KIND_ARTIFACT_FETCH_FAILED: FaultKind
FAULT_KIND_GRANT_EXPIRED: FaultKind
FAULT_KIND_CREDENTIAL_UNAPPLIED: FaultKind
FAULT_KIND_LOCAL_SAFETY_REFUSAL: FaultKind
FAULT_KIND_EXECUTOR_POISONED: FaultKind
FAULT_KIND_CONFIG_REFUSED: FaultKind
FAULT_KIND_UNKNOWN_DEPLOYMENT: FaultKind
SUBJECT_KIND_UNSPECIFIED: SubjectKind
SUBJECT_KIND_OBJECT: SubjectKind
SUBJECT_KIND_SNAPSHOT: SubjectKind
SUBJECT_KIND_TENSOR_HEADER: SubjectKind
SUBJECT_KIND_CHECKPOINT: SubjectKind
SUBJECT_KIND_ADAPTER_ARTIFACT: SubjectKind
SUBJECT_KIND_LAYOUT: SubjectKind
SUBJECT_KIND_RELEASE: SubjectKind
PURGE_FAULT_CODE_UNSPECIFIED: PurgeFaultCode
PURGE_FAULT_CODE_STALE_TOMBSTONE_GENERATION: PurgeFaultCode
PURGE_FAULT_CODE_STALE_WORKER_GENERATION: PurgeFaultCode
PURGE_FAULT_CODE_CHANGED_BODY_REPLAY: PurgeFaultCode
PURGE_FAULT_CODE_CROSS_TENANT_SUBJECT: PurgeFaultCode
PURGE_FAULT_CODE_LIVE_UNRELATED_BINDING: PurgeFaultCode
PURGE_FAULT_CODE_ERASURE_FAILED: PurgeFaultCode
PURGE_FAULT_CODE_UNKNOWN_SUBJECT: PurgeFaultCode
PURGE_FAULT_CODE_DEADLINE_EXCEEDED: PurgeFaultCode
BOOT_FAILURE_REASON_UNSPECIFIED: BootFailureReason
BOOT_FAILURE_REASON_HARDWARE_VERDICT: BootFailureReason
BOOT_FAILURE_REASON_IMAGE_FAULT: BootFailureReason
BOOT_FAILURE_REASON_DRIVER_FAULT: BootFailureReason
BOOT_FAILURE_REASON_DISK_SHAPE: BootFailureReason
BOOT_FAILURE_REASON_CONFIG_INVALID: BootFailureReason
BOOT_FAILURE_REASON_OTHER_FATAL: BootFailureReason
CHECKPOINT_OUTCOME_UNSPECIFIED: CheckpointOutcome
CHECKPOINT_OUTCOME_RECORDED: CheckpointOutcome
CHECKPOINT_OUTCOME_REPLAYED: CheckpointOutcome
CHECKPOINT_OUTCOME_CONFLICT: CheckpointOutcome
CHECKPOINT_OUTCOME_REFUSED: CheckpointOutcome
CHECKPOINT_FAULT_CODE_UNSPECIFIED: CheckpointFaultCode
CHECKPOINT_FAULT_CODE_IDENTITY_CONFLICT: CheckpointFaultCode
CHECKPOINT_FAULT_CODE_UNKNOWN_ATTEMPT: CheckpointFaultCode
CHECKPOINT_FAULT_CODE_NOT_JOB_MODE: CheckpointFaultCode
CHECKPOINT_FAULT_CODE_STALE_SEQUENCE: CheckpointFaultCode
CHECKPOINT_FAULT_CODE_QUOTA_EXCEEDED: CheckpointFaultCode

class WorkerMessage(_message.Message):
    __slots__ = ("register", "report", "attempt_accepted", "attempt_terminal", "boot_failure", "checkpoint_request", "checkpoint_ack")
    REGISTER_FIELD_NUMBER: _ClassVar[int]
    REPORT_FIELD_NUMBER: _ClassVar[int]
    ATTEMPT_ACCEPTED_FIELD_NUMBER: _ClassVar[int]
    ATTEMPT_TERMINAL_FIELD_NUMBER: _ClassVar[int]
    BOOT_FAILURE_FIELD_NUMBER: _ClassVar[int]
    CHECKPOINT_REQUEST_FIELD_NUMBER: _ClassVar[int]
    CHECKPOINT_ACK_FIELD_NUMBER: _ClassVar[int]
    register: Register
    report: Report
    attempt_accepted: AttemptAccepted
    attempt_terminal: AttemptTerminal
    boot_failure: BootFailure
    checkpoint_request: JobCheckpointRequest
    checkpoint_ack: JobCheckpointAck
    def __init__(self, register: _Optional[_Union[Register, _Mapping]] = ..., report: _Optional[_Union[Report, _Mapping]] = ..., attempt_accepted: _Optional[_Union[AttemptAccepted, _Mapping]] = ..., attempt_terminal: _Optional[_Union[AttemptTerminal, _Mapping]] = ..., boot_failure: _Optional[_Union[BootFailure, _Mapping]] = ..., checkpoint_request: _Optional[_Union[JobCheckpointRequest, _Mapping]] = ..., checkpoint_ack: _Optional[_Union[JobCheckpointAck, _Mapping]] = ...) -> None: ...

class CoordinatorMessage(_message.Message):
    __slots__ = ("register_ack", "directive", "start_attempt", "cancel_attempt", "terminal_ack", "checkpoint_receipt")
    REGISTER_ACK_FIELD_NUMBER: _ClassVar[int]
    DIRECTIVE_FIELD_NUMBER: _ClassVar[int]
    START_ATTEMPT_FIELD_NUMBER: _ClassVar[int]
    CANCEL_ATTEMPT_FIELD_NUMBER: _ClassVar[int]
    TERMINAL_ACK_FIELD_NUMBER: _ClassVar[int]
    CHECKPOINT_RECEIPT_FIELD_NUMBER: _ClassVar[int]
    register_ack: RegisterAck
    directive: Directive
    start_attempt: StartAttempt
    cancel_attempt: CancelAttempt
    terminal_ack: TerminalAck
    checkpoint_receipt: JobCheckpointReceipt
    def __init__(self, register_ack: _Optional[_Union[RegisterAck, _Mapping]] = ..., directive: _Optional[_Union[Directive, _Mapping]] = ..., start_attempt: _Optional[_Union[StartAttempt, _Mapping]] = ..., cancel_attempt: _Optional[_Union[CancelAttempt, _Mapping]] = ..., terminal_ack: _Optional[_Union[TerminalAck, _Mapping]] = ..., checkpoint_receipt: _Optional[_Union[JobCheckpointReceipt, _Mapping]] = ...) -> None: ...

class PurgeMessage(_message.Message):
    __slots__ = ("purge_artifact", "purge_artifact_report")
    PURGE_ARTIFACT_FIELD_NUMBER: _ClassVar[int]
    PURGE_ARTIFACT_REPORT_FIELD_NUMBER: _ClassVar[int]
    purge_artifact: PurgeArtifact
    purge_artifact_report: PurgeArtifactReport
    def __init__(self, purge_artifact: _Optional[_Union[PurgeArtifact, _Mapping]] = ..., purge_artifact_report: _Optional[_Union[PurgeArtifactReport, _Mapping]] = ...) -> None: ...

class Register(_message.Message):
    __slots__ = ("session_id", "executor_incarnation", "wire_minor", "worker_id", "instance_id", "release_id", "image_digest", "git_commit", "resources", "recovered_attempts")
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    EXECUTOR_INCARNATION_FIELD_NUMBER: _ClassVar[int]
    WIRE_MINOR_FIELD_NUMBER: _ClassVar[int]
    WORKER_ID_FIELD_NUMBER: _ClassVar[int]
    INSTANCE_ID_FIELD_NUMBER: _ClassVar[int]
    RELEASE_ID_FIELD_NUMBER: _ClassVar[int]
    IMAGE_DIGEST_FIELD_NUMBER: _ClassVar[int]
    GIT_COMMIT_FIELD_NUMBER: _ClassVar[int]
    RESOURCES_FIELD_NUMBER: _ClassVar[int]
    RECOVERED_ATTEMPTS_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    executor_incarnation: int
    wire_minor: int
    worker_id: str
    instance_id: str
    release_id: str
    image_digest: str
    git_commit: str
    resources: WorkerResources
    recovered_attempts: _containers.RepeatedCompositeFieldContainer[ActiveAttempt]
    def __init__(self, session_id: _Optional[str] = ..., executor_incarnation: _Optional[int] = ..., wire_minor: _Optional[int] = ..., worker_id: _Optional[str] = ..., instance_id: _Optional[str] = ..., release_id: _Optional[str] = ..., image_digest: _Optional[str] = ..., git_commit: _Optional[str] = ..., resources: _Optional[_Union[WorkerResources, _Mapping]] = ..., recovered_attempts: _Optional[_Iterable[_Union[ActiveAttempt, _Mapping]]] = ...) -> None: ...

class BootFailure(_message.Message):
    __slots__ = ("worker_id", "instance_id", "reason", "detail", "resources", "image_digest", "release_id")
    WORKER_ID_FIELD_NUMBER: _ClassVar[int]
    INSTANCE_ID_FIELD_NUMBER: _ClassVar[int]
    REASON_FIELD_NUMBER: _ClassVar[int]
    DETAIL_FIELD_NUMBER: _ClassVar[int]
    RESOURCES_FIELD_NUMBER: _ClassVar[int]
    IMAGE_DIGEST_FIELD_NUMBER: _ClassVar[int]
    RELEASE_ID_FIELD_NUMBER: _ClassVar[int]
    worker_id: str
    instance_id: str
    reason: BootFailureReason
    detail: str
    resources: WorkerResources
    image_digest: str
    release_id: str
    def __init__(self, worker_id: _Optional[str] = ..., instance_id: _Optional[str] = ..., reason: _Optional[_Union[BootFailureReason, str]] = ..., detail: _Optional[str] = ..., resources: _Optional[_Union[WorkerResources, _Mapping]] = ..., image_digest: _Optional[str] = ..., release_id: _Optional[str] = ...) -> None: ...

class RegisterAck(_message.Message):
    __slots__ = ("session_id", "executor_incarnation", "wire_minor", "accepted", "rejection")
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    EXECUTOR_INCARNATION_FIELD_NUMBER: _ClassVar[int]
    WIRE_MINOR_FIELD_NUMBER: _ClassVar[int]
    ACCEPTED_FIELD_NUMBER: _ClassVar[int]
    REJECTION_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    executor_incarnation: int
    wire_minor: int
    accepted: bool
    rejection: RegisterRejection
    def __init__(self, session_id: _Optional[str] = ..., executor_incarnation: _Optional[int] = ..., wire_minor: _Optional[int] = ..., accepted: bool = ..., rejection: _Optional[_Union[RegisterRejection, str]] = ...) -> None: ...

class Directive(_message.Message):
    __slots__ = ("session_id", "executor_incarnation", "revision", "posture", "drain_policy", "artifact_grant", "credential", "wire_minor", "serving", "job")
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    EXECUTOR_INCARNATION_FIELD_NUMBER: _ClassVar[int]
    REVISION_FIELD_NUMBER: _ClassVar[int]
    POSTURE_FIELD_NUMBER: _ClassVar[int]
    DRAIN_POLICY_FIELD_NUMBER: _ClassVar[int]
    ARTIFACT_GRANT_FIELD_NUMBER: _ClassVar[int]
    CREDENTIAL_FIELD_NUMBER: _ClassVar[int]
    WIRE_MINOR_FIELD_NUMBER: _ClassVar[int]
    SERVING_FIELD_NUMBER: _ClassVar[int]
    JOB_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    executor_incarnation: int
    revision: int
    posture: Posture
    drain_policy: DrainPolicy
    artifact_grant: ArtifactGrant
    credential: Credential
    wire_minor: int
    serving: ServingDirective
    job: JobDirective
    def __init__(self, session_id: _Optional[str] = ..., executor_incarnation: _Optional[int] = ..., revision: _Optional[int] = ..., posture: _Optional[_Union[Posture, str]] = ..., drain_policy: _Optional[_Union[DrainPolicy, _Mapping]] = ..., artifact_grant: _Optional[_Union[ArtifactGrant, _Mapping]] = ..., credential: _Optional[_Union[Credential, _Mapping]] = ..., wire_minor: _Optional[int] = ..., serving: _Optional[_Union[ServingDirective, _Mapping]] = ..., job: _Optional[_Union[JobDirective, _Mapping]] = ...) -> None: ...

class ServingDirective(_message.Message):
    __slots__ = ("endpoint_release_id", "entrypoint_binding_plan_ids")
    ENDPOINT_RELEASE_ID_FIELD_NUMBER: _ClassVar[int]
    ENTRYPOINT_BINDING_PLAN_IDS_FIELD_NUMBER: _ClassVar[int]
    endpoint_release_id: str
    entrypoint_binding_plan_ids: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, endpoint_release_id: _Optional[str] = ..., entrypoint_binding_plan_ids: _Optional[_Iterable[str]] = ...) -> None: ...

class JobDirective(_message.Message):
    __slots__ = ("build_id", "job_descriptor_id", "resource_caps", "publication_contract", "reclaim_on_terminal", "gpu_count")
    BUILD_ID_FIELD_NUMBER: _ClassVar[int]
    JOB_DESCRIPTOR_ID_FIELD_NUMBER: _ClassVar[int]
    RESOURCE_CAPS_FIELD_NUMBER: _ClassVar[int]
    PUBLICATION_CONTRACT_FIELD_NUMBER: _ClassVar[int]
    RECLAIM_ON_TERMINAL_FIELD_NUMBER: _ClassVar[int]
    GPU_COUNT_FIELD_NUMBER: _ClassVar[int]
    build_id: str
    job_descriptor_id: str
    resource_caps: ResourceCaps
    publication_contract: PublicationContract
    reclaim_on_terminal: bool
    gpu_count: int
    def __init__(self, build_id: _Optional[str] = ..., job_descriptor_id: _Optional[str] = ..., resource_caps: _Optional[_Union[ResourceCaps, _Mapping]] = ..., publication_contract: _Optional[_Union[PublicationContract, _Mapping]] = ..., reclaim_on_terminal: bool = ..., gpu_count: _Optional[int] = ...) -> None: ...

class Report(_message.Message):
    __slots__ = ("session_id", "executor_incarnation", "applied_revision", "intake_state", "credential_epoch", "readiness_epoch", "active_attempts", "faults", "activity", "applied_wire_minor", "serving_capacity", "job_capacity")
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    EXECUTOR_INCARNATION_FIELD_NUMBER: _ClassVar[int]
    APPLIED_REVISION_FIELD_NUMBER: _ClassVar[int]
    INTAKE_STATE_FIELD_NUMBER: _ClassVar[int]
    CREDENTIAL_EPOCH_FIELD_NUMBER: _ClassVar[int]
    READINESS_EPOCH_FIELD_NUMBER: _ClassVar[int]
    ACTIVE_ATTEMPTS_FIELD_NUMBER: _ClassVar[int]
    FAULTS_FIELD_NUMBER: _ClassVar[int]
    ACTIVITY_FIELD_NUMBER: _ClassVar[int]
    APPLIED_WIRE_MINOR_FIELD_NUMBER: _ClassVar[int]
    SERVING_CAPACITY_FIELD_NUMBER: _ClassVar[int]
    JOB_CAPACITY_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    executor_incarnation: int
    applied_revision: int
    intake_state: IntakeState
    credential_epoch: int
    readiness_epoch: int
    active_attempts: _containers.RepeatedCompositeFieldContainer[ActiveAttempt]
    faults: _containers.RepeatedCompositeFieldContainer[Fault]
    activity: _containers.RepeatedCompositeFieldContainer[ActivityEvent]
    applied_wire_minor: int
    serving_capacity: ServingCapacity
    job_capacity: JobCapacity
    def __init__(self, session_id: _Optional[str] = ..., executor_incarnation: _Optional[int] = ..., applied_revision: _Optional[int] = ..., intake_state: _Optional[_Union[IntakeState, str]] = ..., credential_epoch: _Optional[int] = ..., readiness_epoch: _Optional[int] = ..., active_attempts: _Optional[_Iterable[_Union[ActiveAttempt, _Mapping]]] = ..., faults: _Optional[_Iterable[_Union[Fault, _Mapping]]] = ..., activity: _Optional[_Iterable[_Union[ActivityEvent, _Mapping]]] = ..., applied_wire_minor: _Optional[int] = ..., serving_capacity: _Optional[_Union[ServingCapacity, _Mapping]] = ..., job_capacity: _Optional[_Union[JobCapacity, _Mapping]] = ...) -> None: ...

class ActivityEvent(_message.Message):
    __slots__ = ("seq", "kind", "step", "at_unix_ms")
    SEQ_FIELD_NUMBER: _ClassVar[int]
    KIND_FIELD_NUMBER: _ClassVar[int]
    STEP_FIELD_NUMBER: _ClassVar[int]
    AT_UNIX_MS_FIELD_NUMBER: _ClassVar[int]
    seq: int
    kind: str
    step: str
    at_unix_ms: int
    def __init__(self, seq: _Optional[int] = ..., kind: _Optional[str] = ..., step: _Optional[str] = ..., at_unix_ms: _Optional[int] = ...) -> None: ...

class StartAttempt(_message.Message):
    __slots__ = ("session_id", "executor_incarnation", "request_id", "attempt", "exec_spec_digest", "grant", "exec_spec_canonical")
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    EXECUTOR_INCARNATION_FIELD_NUMBER: _ClassVar[int]
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    ATTEMPT_FIELD_NUMBER: _ClassVar[int]
    EXEC_SPEC_DIGEST_FIELD_NUMBER: _ClassVar[int]
    GRANT_FIELD_NUMBER: _ClassVar[int]
    EXEC_SPEC_CANONICAL_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    executor_incarnation: int
    request_id: str
    attempt: int
    exec_spec_digest: bytes
    grant: DeliveryGrant
    exec_spec_canonical: bytes
    def __init__(self, session_id: _Optional[str] = ..., executor_incarnation: _Optional[int] = ..., request_id: _Optional[str] = ..., attempt: _Optional[int] = ..., exec_spec_digest: _Optional[bytes] = ..., grant: _Optional[_Union[DeliveryGrant, _Mapping]] = ..., exec_spec_canonical: _Optional[bytes] = ...) -> None: ...

class AttemptAccepted(_message.Message):
    __slots__ = ("session_id", "executor_incarnation", "request_id", "attempt", "exec_spec_digest", "plan_digest", "model_construction_digest", "plan")
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    EXECUTOR_INCARNATION_FIELD_NUMBER: _ClassVar[int]
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    ATTEMPT_FIELD_NUMBER: _ClassVar[int]
    EXEC_SPEC_DIGEST_FIELD_NUMBER: _ClassVar[int]
    PLAN_DIGEST_FIELD_NUMBER: _ClassVar[int]
    MODEL_CONSTRUCTION_DIGEST_FIELD_NUMBER: _ClassVar[int]
    PLAN_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    executor_incarnation: int
    request_id: str
    attempt: int
    exec_spec_digest: bytes
    plan_digest: bytes
    model_construction_digest: bytes
    plan: AttemptPlanSummary
    def __init__(self, session_id: _Optional[str] = ..., executor_incarnation: _Optional[int] = ..., request_id: _Optional[str] = ..., attempt: _Optional[int] = ..., exec_spec_digest: _Optional[bytes] = ..., plan_digest: _Optional[bytes] = ..., model_construction_digest: _Optional[bytes] = ..., plan: _Optional[_Union[AttemptPlanSummary, _Mapping]] = ...) -> None: ...

class AttemptPlanSummary(_message.Message):
    __slots__ = ("delivery", "materialization", "compute_dtype", "placement", "reserved_vram_bytes", "reserved_host_bytes", "quantified_choice")
    DELIVERY_FIELD_NUMBER: _ClassVar[int]
    MATERIALIZATION_FIELD_NUMBER: _ClassVar[int]
    COMPUTE_DTYPE_FIELD_NUMBER: _ClassVar[int]
    PLACEMENT_FIELD_NUMBER: _ClassVar[int]
    RESERVED_VRAM_BYTES_FIELD_NUMBER: _ClassVar[int]
    RESERVED_HOST_BYTES_FIELD_NUMBER: _ClassVar[int]
    QUANTIFIED_CHOICE_FIELD_NUMBER: _ClassVar[int]
    delivery: str
    materialization: str
    compute_dtype: str
    placement: str
    reserved_vram_bytes: int
    reserved_host_bytes: int
    quantified_choice: str
    def __init__(self, delivery: _Optional[str] = ..., materialization: _Optional[str] = ..., compute_dtype: _Optional[str] = ..., placement: _Optional[str] = ..., reserved_vram_bytes: _Optional[int] = ..., reserved_host_bytes: _Optional[int] = ..., quantified_choice: _Optional[str] = ...) -> None: ...

class CancelAttempt(_message.Message):
    __slots__ = ("session_id", "executor_incarnation", "request_id", "attempt", "reason", "grace_ms", "exec_spec_digest")
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    EXECUTOR_INCARNATION_FIELD_NUMBER: _ClassVar[int]
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    ATTEMPT_FIELD_NUMBER: _ClassVar[int]
    REASON_FIELD_NUMBER: _ClassVar[int]
    GRACE_MS_FIELD_NUMBER: _ClassVar[int]
    EXEC_SPEC_DIGEST_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    executor_incarnation: int
    request_id: str
    attempt: int
    reason: CancelReason
    grace_ms: int
    exec_spec_digest: bytes
    def __init__(self, session_id: _Optional[str] = ..., executor_incarnation: _Optional[int] = ..., request_id: _Optional[str] = ..., attempt: _Optional[int] = ..., reason: _Optional[_Union[CancelReason, str]] = ..., grace_ms: _Optional[int] = ..., exec_spec_digest: _Optional[bytes] = ...) -> None: ...

class AttemptTerminal(_message.Message):
    __slots__ = ("session_id", "executor_incarnation", "request_id", "attempt", "exec_spec_digest", "terminal_id", "terminal_digest", "terminal_canonical")
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    EXECUTOR_INCARNATION_FIELD_NUMBER: _ClassVar[int]
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    ATTEMPT_FIELD_NUMBER: _ClassVar[int]
    EXEC_SPEC_DIGEST_FIELD_NUMBER: _ClassVar[int]
    TERMINAL_ID_FIELD_NUMBER: _ClassVar[int]
    TERMINAL_DIGEST_FIELD_NUMBER: _ClassVar[int]
    TERMINAL_CANONICAL_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    executor_incarnation: int
    request_id: str
    attempt: int
    exec_spec_digest: bytes
    terminal_id: str
    terminal_digest: bytes
    terminal_canonical: bytes
    def __init__(self, session_id: _Optional[str] = ..., executor_incarnation: _Optional[int] = ..., request_id: _Optional[str] = ..., attempt: _Optional[int] = ..., exec_spec_digest: _Optional[bytes] = ..., terminal_id: _Optional[str] = ..., terminal_digest: _Optional[bytes] = ..., terminal_canonical: _Optional[bytes] = ...) -> None: ...

class TerminalBody(_message.Message):
    __slots__ = ("request_id", "attempt", "exec_spec_digest", "status", "output_manifest", "metrics", "triage_bundle", "safe_message", "cause", "result")
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    ATTEMPT_FIELD_NUMBER: _ClassVar[int]
    EXEC_SPEC_DIGEST_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_MANIFEST_FIELD_NUMBER: _ClassVar[int]
    METRICS_FIELD_NUMBER: _ClassVar[int]
    TRIAGE_BUNDLE_FIELD_NUMBER: _ClassVar[int]
    SAFE_MESSAGE_FIELD_NUMBER: _ClassVar[int]
    CAUSE_FIELD_NUMBER: _ClassVar[int]
    RESULT_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    attempt: int
    exec_spec_digest: bytes
    status: TerminalStatus
    output_manifest: OutputManifest
    metrics: AttemptMetrics
    triage_bundle: TriageBundleRef
    safe_message: str
    cause: TerminalCause
    result: ResultEnvelope
    def __init__(self, request_id: _Optional[str] = ..., attempt: _Optional[int] = ..., exec_spec_digest: _Optional[bytes] = ..., status: _Optional[_Union[TerminalStatus, str]] = ..., output_manifest: _Optional[_Union[OutputManifest, _Mapping]] = ..., metrics: _Optional[_Union[AttemptMetrics, _Mapping]] = ..., triage_bundle: _Optional[_Union[TriageBundleRef, _Mapping]] = ..., safe_message: _Optional[str] = ..., cause: _Optional[_Union[TerminalCause, _Mapping]] = ..., result: _Optional[_Union[ResultEnvelope, _Mapping]] = ...) -> None: ...

class ResultEnvelope(_message.Message):
    __slots__ = ("result_schema_digest", "inline_result", "result_blob", "adjustments", "checkpoint_ref", "adapters")
    RESULT_SCHEMA_DIGEST_FIELD_NUMBER: _ClassVar[int]
    INLINE_RESULT_FIELD_NUMBER: _ClassVar[int]
    RESULT_BLOB_FIELD_NUMBER: _ClassVar[int]
    ADJUSTMENTS_FIELD_NUMBER: _ClassVar[int]
    CHECKPOINT_REF_FIELD_NUMBER: _ClassVar[int]
    ADAPTERS_FIELD_NUMBER: _ClassVar[int]
    result_schema_digest: bytes
    inline_result: bytes
    result_blob: OutputEntry
    adjustments: _containers.RepeatedCompositeFieldContainer[AdjustmentRow]
    checkpoint_ref: str
    adapters: _containers.RepeatedCompositeFieldContainer[AppliedAdapter]
    def __init__(self, result_schema_digest: _Optional[bytes] = ..., inline_result: _Optional[bytes] = ..., result_blob: _Optional[_Union[OutputEntry, _Mapping]] = ..., adjustments: _Optional[_Iterable[_Union[AdjustmentRow, _Mapping]]] = ..., checkpoint_ref: _Optional[str] = ..., adapters: _Optional[_Iterable[_Union[AppliedAdapter, _Mapping]]] = ...) -> None: ...

class AdjustmentRow(_message.Message):
    __slots__ = ("field", "requested", "applied", "reason")
    FIELD_FIELD_NUMBER: _ClassVar[int]
    REQUESTED_FIELD_NUMBER: _ClassVar[int]
    APPLIED_FIELD_NUMBER: _ClassVar[int]
    REASON_FIELD_NUMBER: _ClassVar[int]
    field: str
    requested: str
    applied: str
    reason: str
    def __init__(self, field: _Optional[str] = ..., requested: _Optional[str] = ..., applied: _Optional[str] = ..., reason: _Optional[str] = ...) -> None: ...

class AppliedAdapter(_message.Message):
    __slots__ = ("ref", "kind")
    REF_FIELD_NUMBER: _ClassVar[int]
    KIND_FIELD_NUMBER: _ClassVar[int]
    ref: str
    kind: str
    def __init__(self, ref: _Optional[str] = ..., kind: _Optional[str] = ...) -> None: ...

class TerminalCause(_message.Message):
    __slots__ = ("code", "origin", "detail", "shortfall")
    CODE_FIELD_NUMBER: _ClassVar[int]
    ORIGIN_FIELD_NUMBER: _ClassVar[int]
    DETAIL_FIELD_NUMBER: _ClassVar[int]
    SHORTFALL_FIELD_NUMBER: _ClassVar[int]
    code: CauseCode
    origin: CauseOrigin
    detail: str
    shortfall: ResourceShortfall
    def __init__(self, code: _Optional[_Union[CauseCode, str]] = ..., origin: _Optional[_Union[CauseOrigin, str]] = ..., detail: _Optional[str] = ..., shortfall: _Optional[_Union[ResourceShortfall, _Mapping]] = ...) -> None: ...

class ResourceShortfall(_message.Message):
    __slots__ = ("resource", "scope", "needed_bytes", "available_bytes", "request_shape", "evidence_class")
    RESOURCE_FIELD_NUMBER: _ClassVar[int]
    SCOPE_FIELD_NUMBER: _ClassVar[int]
    NEEDED_BYTES_FIELD_NUMBER: _ClassVar[int]
    AVAILABLE_BYTES_FIELD_NUMBER: _ClassVar[int]
    REQUEST_SHAPE_FIELD_NUMBER: _ClassVar[int]
    EVIDENCE_CLASS_FIELD_NUMBER: _ClassVar[int]
    resource: str
    scope: str
    needed_bytes: int
    available_bytes: int
    request_shape: str
    evidence_class: str
    def __init__(self, resource: _Optional[str] = ..., scope: _Optional[str] = ..., needed_bytes: _Optional[int] = ..., available_bytes: _Optional[int] = ..., request_shape: _Optional[str] = ..., evidence_class: _Optional[str] = ...) -> None: ...

class TerminalAck(_message.Message):
    __slots__ = ("session_id", "executor_incarnation", "request_id", "attempt", "exec_spec_digest", "terminal_id", "terminal_digest")
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    EXECUTOR_INCARNATION_FIELD_NUMBER: _ClassVar[int]
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    ATTEMPT_FIELD_NUMBER: _ClassVar[int]
    EXEC_SPEC_DIGEST_FIELD_NUMBER: _ClassVar[int]
    TERMINAL_ID_FIELD_NUMBER: _ClassVar[int]
    TERMINAL_DIGEST_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    executor_incarnation: int
    request_id: str
    attempt: int
    exec_spec_digest: bytes
    terminal_id: str
    terminal_digest: bytes
    def __init__(self, session_id: _Optional[str] = ..., executor_incarnation: _Optional[int] = ..., request_id: _Optional[str] = ..., attempt: _Optional[int] = ..., exec_spec_digest: _Optional[bytes] = ..., terminal_id: _Optional[str] = ..., terminal_digest: _Optional[bytes] = ...) -> None: ...

class JobCheckpointRequest(_message.Message):
    __slots__ = ("session_id", "executor_incarnation", "request_id", "attempt", "operation_key", "logical_key", "content_digest", "seq", "artifact")
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    EXECUTOR_INCARNATION_FIELD_NUMBER: _ClassVar[int]
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    ATTEMPT_FIELD_NUMBER: _ClassVar[int]
    OPERATION_KEY_FIELD_NUMBER: _ClassVar[int]
    LOGICAL_KEY_FIELD_NUMBER: _ClassVar[int]
    CONTENT_DIGEST_FIELD_NUMBER: _ClassVar[int]
    SEQ_FIELD_NUMBER: _ClassVar[int]
    ARTIFACT_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    executor_incarnation: int
    request_id: str
    attempt: int
    operation_key: str
    logical_key: str
    content_digest: bytes
    seq: int
    artifact: OutputEntry
    def __init__(self, session_id: _Optional[str] = ..., executor_incarnation: _Optional[int] = ..., request_id: _Optional[str] = ..., attempt: _Optional[int] = ..., operation_key: _Optional[str] = ..., logical_key: _Optional[str] = ..., content_digest: _Optional[bytes] = ..., seq: _Optional[int] = ..., artifact: _Optional[_Union[OutputEntry, _Mapping]] = ...) -> None: ...

class JobCheckpointReceipt(_message.Message):
    __slots__ = ("session_id", "executor_incarnation", "request_id", "attempt", "operation_key", "logical_key", "content_digest", "receipt_id", "outcome", "fault")
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    EXECUTOR_INCARNATION_FIELD_NUMBER: _ClassVar[int]
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    ATTEMPT_FIELD_NUMBER: _ClassVar[int]
    OPERATION_KEY_FIELD_NUMBER: _ClassVar[int]
    LOGICAL_KEY_FIELD_NUMBER: _ClassVar[int]
    CONTENT_DIGEST_FIELD_NUMBER: _ClassVar[int]
    RECEIPT_ID_FIELD_NUMBER: _ClassVar[int]
    OUTCOME_FIELD_NUMBER: _ClassVar[int]
    FAULT_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    executor_incarnation: int
    request_id: str
    attempt: int
    operation_key: str
    logical_key: str
    content_digest: bytes
    receipt_id: str
    outcome: CheckpointOutcome
    fault: CheckpointFault
    def __init__(self, session_id: _Optional[str] = ..., executor_incarnation: _Optional[int] = ..., request_id: _Optional[str] = ..., attempt: _Optional[int] = ..., operation_key: _Optional[str] = ..., logical_key: _Optional[str] = ..., content_digest: _Optional[bytes] = ..., receipt_id: _Optional[str] = ..., outcome: _Optional[_Union[CheckpointOutcome, str]] = ..., fault: _Optional[_Union[CheckpointFault, _Mapping]] = ...) -> None: ...

class CheckpointFault(_message.Message):
    __slots__ = ("code", "detail", "recorded_content_digest")
    CODE_FIELD_NUMBER: _ClassVar[int]
    DETAIL_FIELD_NUMBER: _ClassVar[int]
    RECORDED_CONTENT_DIGEST_FIELD_NUMBER: _ClassVar[int]
    code: CheckpointFaultCode
    detail: str
    recorded_content_digest: bytes
    def __init__(self, code: _Optional[_Union[CheckpointFaultCode, str]] = ..., detail: _Optional[str] = ..., recorded_content_digest: _Optional[bytes] = ...) -> None: ...

class JobCheckpointAck(_message.Message):
    __slots__ = ("session_id", "executor_incarnation", "request_id", "attempt", "operation_key", "receipt_id", "content_digest")
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    EXECUTOR_INCARNATION_FIELD_NUMBER: _ClassVar[int]
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    ATTEMPT_FIELD_NUMBER: _ClassVar[int]
    OPERATION_KEY_FIELD_NUMBER: _ClassVar[int]
    RECEIPT_ID_FIELD_NUMBER: _ClassVar[int]
    CONTENT_DIGEST_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    executor_incarnation: int
    request_id: str
    attempt: int
    operation_key: str
    receipt_id: str
    content_digest: bytes
    def __init__(self, session_id: _Optional[str] = ..., executor_incarnation: _Optional[int] = ..., request_id: _Optional[str] = ..., attempt: _Optional[int] = ..., operation_key: _Optional[str] = ..., receipt_id: _Optional[str] = ..., content_digest: _Optional[bytes] = ...) -> None: ...

class AttemptProgress(_message.Message):
    __slots__ = ("session_id", "executor_incarnation", "request_id", "attempt", "seq", "content_type", "data")
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    EXECUTOR_INCARNATION_FIELD_NUMBER: _ClassVar[int]
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    ATTEMPT_FIELD_NUMBER: _ClassVar[int]
    SEQ_FIELD_NUMBER: _ClassVar[int]
    CONTENT_TYPE_FIELD_NUMBER: _ClassVar[int]
    DATA_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    executor_incarnation: int
    request_id: str
    attempt: int
    seq: int
    content_type: str
    data: bytes
    def __init__(self, session_id: _Optional[str] = ..., executor_incarnation: _Optional[int] = ..., request_id: _Optional[str] = ..., attempt: _Optional[int] = ..., seq: _Optional[int] = ..., content_type: _Optional[str] = ..., data: _Optional[bytes] = ...) -> None: ...

class PurgeArtifact(_message.Message):
    __slots__ = ("session_id", "executor_incarnation", "purge_id", "purge_body_digest", "purge_body_canonical")
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    EXECUTOR_INCARNATION_FIELD_NUMBER: _ClassVar[int]
    PURGE_ID_FIELD_NUMBER: _ClassVar[int]
    PURGE_BODY_DIGEST_FIELD_NUMBER: _ClassVar[int]
    PURGE_BODY_CANONICAL_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    executor_incarnation: int
    purge_id: str
    purge_body_digest: bytes
    purge_body_canonical: bytes
    def __init__(self, session_id: _Optional[str] = ..., executor_incarnation: _Optional[int] = ..., purge_id: _Optional[str] = ..., purge_body_digest: _Optional[bytes] = ..., purge_body_canonical: _Optional[bytes] = ...) -> None: ...

class PurgeBody(_message.Message):
    __slots__ = ("tenant_id", "subjects", "tombstone_generation", "target_instance_id", "target_generation", "deadline_unix")
    TENANT_ID_FIELD_NUMBER: _ClassVar[int]
    SUBJECTS_FIELD_NUMBER: _ClassVar[int]
    TOMBSTONE_GENERATION_FIELD_NUMBER: _ClassVar[int]
    TARGET_INSTANCE_ID_FIELD_NUMBER: _ClassVar[int]
    TARGET_GENERATION_FIELD_NUMBER: _ClassVar[int]
    DEADLINE_UNIX_FIELD_NUMBER: _ClassVar[int]
    tenant_id: str
    subjects: _containers.RepeatedCompositeFieldContainer[ArtifactSubject]
    tombstone_generation: int
    target_instance_id: str
    target_generation: int
    deadline_unix: int
    def __init__(self, tenant_id: _Optional[str] = ..., subjects: _Optional[_Iterable[_Union[ArtifactSubject, _Mapping]]] = ..., tombstone_generation: _Optional[int] = ..., target_instance_id: _Optional[str] = ..., target_generation: _Optional[int] = ..., deadline_unix: _Optional[int] = ...) -> None: ...

class PurgeArtifactReport(_message.Message):
    __slots__ = ("session_id", "executor_incarnation", "purge_id", "purge_body_digest", "applied_tombstone_generation", "pending", "absent", "fault")
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    EXECUTOR_INCARNATION_FIELD_NUMBER: _ClassVar[int]
    PURGE_ID_FIELD_NUMBER: _ClassVar[int]
    PURGE_BODY_DIGEST_FIELD_NUMBER: _ClassVar[int]
    APPLIED_TOMBSTONE_GENERATION_FIELD_NUMBER: _ClassVar[int]
    PENDING_FIELD_NUMBER: _ClassVar[int]
    ABSENT_FIELD_NUMBER: _ClassVar[int]
    FAULT_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    executor_incarnation: int
    purge_id: str
    purge_body_digest: bytes
    applied_tombstone_generation: int
    pending: PendingLeases
    absent: AbsenceEvidence
    fault: PurgeFault
    def __init__(self, session_id: _Optional[str] = ..., executor_incarnation: _Optional[int] = ..., purge_id: _Optional[str] = ..., purge_body_digest: _Optional[bytes] = ..., applied_tombstone_generation: _Optional[int] = ..., pending: _Optional[_Union[PendingLeases, _Mapping]] = ..., absent: _Optional[_Union[AbsenceEvidence, _Mapping]] = ..., fault: _Optional[_Union[PurgeFault, _Mapping]] = ...) -> None: ...

class ExecutionSpec(_message.Message):
    __slots__ = ("endpoint_release_id", "image_digest", "config_digest", "deadline_unix_ms", "serving", "job")
    ENDPOINT_RELEASE_ID_FIELD_NUMBER: _ClassVar[int]
    IMAGE_DIGEST_FIELD_NUMBER: _ClassVar[int]
    CONFIG_DIGEST_FIELD_NUMBER: _ClassVar[int]
    DEADLINE_UNIX_MS_FIELD_NUMBER: _ClassVar[int]
    SERVING_FIELD_NUMBER: _ClassVar[int]
    JOB_FIELD_NUMBER: _ClassVar[int]
    endpoint_release_id: str
    image_digest: str
    config_digest: str
    deadline_unix_ms: int
    serving: ServingExecutionSpec
    job: JobExecutionSpec
    def __init__(self, endpoint_release_id: _Optional[str] = ..., image_digest: _Optional[str] = ..., config_digest: _Optional[str] = ..., deadline_unix_ms: _Optional[int] = ..., serving: _Optional[_Union[ServingExecutionSpec, _Mapping]] = ..., job: _Optional[_Union[JobExecutionSpec, _Mapping]] = ...) -> None: ...

class ServingExecutionSpec(_message.Message):
    __slots__ = ("entrypoint_binding_plan_id", "attempt_binding_id")
    ENTRYPOINT_BINDING_PLAN_ID_FIELD_NUMBER: _ClassVar[int]
    ATTEMPT_BINDING_ID_FIELD_NUMBER: _ClassVar[int]
    entrypoint_binding_plan_id: str
    attempt_binding_id: str
    def __init__(self, entrypoint_binding_plan_id: _Optional[str] = ..., attempt_binding_id: _Optional[str] = ...) -> None: ...

class JobExecutionSpec(_message.Message):
    __slots__ = ("build_id", "job_descriptor_id", "publication_contract")
    BUILD_ID_FIELD_NUMBER: _ClassVar[int]
    JOB_DESCRIPTOR_ID_FIELD_NUMBER: _ClassVar[int]
    PUBLICATION_CONTRACT_FIELD_NUMBER: _ClassVar[int]
    build_id: str
    job_descriptor_id: str
    publication_contract: PublicationContract
    def __init__(self, build_id: _Optional[str] = ..., job_descriptor_id: _Optional[str] = ..., publication_contract: _Optional[_Union[PublicationContract, _Mapping]] = ...) -> None: ...

class DeliveryGrant(_message.Message):
    __slots__ = ("credential", "file_base_url", "expires_at_unix", "inputs", "outputs")
    CREDENTIAL_FIELD_NUMBER: _ClassVar[int]
    FILE_BASE_URL_FIELD_NUMBER: _ClassVar[int]
    EXPIRES_AT_UNIX_FIELD_NUMBER: _ClassVar[int]
    INPUTS_FIELD_NUMBER: _ClassVar[int]
    OUTPUTS_FIELD_NUMBER: _ClassVar[int]
    credential: Credential
    file_base_url: str
    expires_at_unix: int
    inputs: _containers.RepeatedCompositeFieldContainer[InputLocation]
    outputs: _containers.RepeatedCompositeFieldContainer[OutputDestination]
    def __init__(self, credential: _Optional[_Union[Credential, _Mapping]] = ..., file_base_url: _Optional[str] = ..., expires_at_unix: _Optional[int] = ..., inputs: _Optional[_Iterable[_Union[InputLocation, _Mapping]]] = ..., outputs: _Optional[_Iterable[_Union[OutputDestination, _Mapping]]] = ...) -> None: ...

class InputLocation(_message.Message):
    __slots__ = ("digest", "url", "length", "input_id", "order", "kind_mime")
    DIGEST_FIELD_NUMBER: _ClassVar[int]
    URL_FIELD_NUMBER: _ClassVar[int]
    LENGTH_FIELD_NUMBER: _ClassVar[int]
    INPUT_ID_FIELD_NUMBER: _ClassVar[int]
    ORDER_FIELD_NUMBER: _ClassVar[int]
    KIND_MIME_FIELD_NUMBER: _ClassVar[int]
    digest: bytes
    url: str
    length: int
    input_id: str
    order: int
    kind_mime: str
    def __init__(self, digest: _Optional[bytes] = ..., url: _Optional[str] = ..., length: _Optional[int] = ..., input_id: _Optional[str] = ..., order: _Optional[int] = ..., kind_mime: _Optional[str] = ...) -> None: ...

class OutputDestination(_message.Message):
    __slots__ = ("output_id", "url", "max_bytes")
    OUTPUT_ID_FIELD_NUMBER: _ClassVar[int]
    URL_FIELD_NUMBER: _ClassVar[int]
    MAX_BYTES_FIELD_NUMBER: _ClassVar[int]
    output_id: str
    url: str
    max_bytes: int
    def __init__(self, output_id: _Optional[str] = ..., url: _Optional[str] = ..., max_bytes: _Optional[int] = ...) -> None: ...

class Credential(_message.Message):
    __slots__ = ("issuer", "key_id", "epoch", "expires_at_unix", "token")
    ISSUER_FIELD_NUMBER: _ClassVar[int]
    KEY_ID_FIELD_NUMBER: _ClassVar[int]
    EPOCH_FIELD_NUMBER: _ClassVar[int]
    EXPIRES_AT_UNIX_FIELD_NUMBER: _ClassVar[int]
    TOKEN_FIELD_NUMBER: _ClassVar[int]
    issuer: str
    key_id: str
    epoch: int
    expires_at_unix: int
    token: bytes
    def __init__(self, issuer: _Optional[str] = ..., key_id: _Optional[str] = ..., epoch: _Optional[int] = ..., expires_at_unix: _Optional[int] = ..., token: _Optional[bytes] = ...) -> None: ...

class ArtifactGrant(_message.Message):
    __slots__ = ("file_base_url", "expires_at_unix", "subjects")
    FILE_BASE_URL_FIELD_NUMBER: _ClassVar[int]
    EXPIRES_AT_UNIX_FIELD_NUMBER: _ClassVar[int]
    SUBJECTS_FIELD_NUMBER: _ClassVar[int]
    file_base_url: str
    expires_at_unix: int
    subjects: _containers.RepeatedCompositeFieldContainer[ArtifactSubject]
    def __init__(self, file_base_url: _Optional[str] = ..., expires_at_unix: _Optional[int] = ..., subjects: _Optional[_Iterable[_Union[ArtifactSubject, _Mapping]]] = ...) -> None: ...

class ArtifactSubject(_message.Message):
    __slots__ = ("kind", "digest")
    KIND_FIELD_NUMBER: _ClassVar[int]
    DIGEST_FIELD_NUMBER: _ClassVar[int]
    kind: SubjectKind
    digest: bytes
    def __init__(self, kind: _Optional[_Union[SubjectKind, str]] = ..., digest: _Optional[bytes] = ...) -> None: ...

class DrainPolicy(_message.Message):
    __slots__ = ("deadline_ms", "force_cancel_at_deadline")
    DEADLINE_MS_FIELD_NUMBER: _ClassVar[int]
    FORCE_CANCEL_AT_DEADLINE_FIELD_NUMBER: _ClassVar[int]
    deadline_ms: int
    force_cancel_at_deadline: bool
    def __init__(self, deadline_ms: _Optional[int] = ..., force_cancel_at_deadline: bool = ...) -> None: ...

class ResourceCaps(_message.Message):
    __slots__ = ("gpu_required", "max_vram_bytes", "max_rss_bytes", "max_disk_bytes")
    GPU_REQUIRED_FIELD_NUMBER: _ClassVar[int]
    MAX_VRAM_BYTES_FIELD_NUMBER: _ClassVar[int]
    MAX_RSS_BYTES_FIELD_NUMBER: _ClassVar[int]
    MAX_DISK_BYTES_FIELD_NUMBER: _ClassVar[int]
    gpu_required: bool
    max_vram_bytes: int
    max_rss_bytes: int
    max_disk_bytes: int
    def __init__(self, gpu_required: bool = ..., max_vram_bytes: _Optional[int] = ..., max_rss_bytes: _Optional[int] = ..., max_disk_bytes: _Optional[int] = ...) -> None: ...

class PublicationContract(_message.Message):
    __slots__ = ("outputs", "grant_id")
    OUTPUTS_FIELD_NUMBER: _ClassVar[int]
    GRANT_ID_FIELD_NUMBER: _ClassVar[int]
    outputs: _containers.RepeatedCompositeFieldContainer[OutputSpec]
    grant_id: str
    def __init__(self, outputs: _Optional[_Iterable[_Union[OutputSpec, _Mapping]]] = ..., grant_id: _Optional[str] = ...) -> None: ...

class OutputSpec(_message.Message):
    __slots__ = ("output_id", "mime_type", "max_bytes")
    OUTPUT_ID_FIELD_NUMBER: _ClassVar[int]
    MIME_TYPE_FIELD_NUMBER: _ClassVar[int]
    MAX_BYTES_FIELD_NUMBER: _ClassVar[int]
    output_id: str
    mime_type: str
    max_bytes: int
    def __init__(self, output_id: _Optional[str] = ..., mime_type: _Optional[str] = ..., max_bytes: _Optional[int] = ...) -> None: ...

class WorkerResources(_message.Message):
    __slots__ = ("gpu_count", "gpu_name", "gpu_sm", "vram_total_bytes", "driver_version", "cuda_version", "torch_version", "platform", "host_ram_total_bytes", "vcpu_count", "disk_total_bytes", "pinned_capacity_bytes", "power_cap_watts", "mig_profile", "interconnect", "peer_access", "h2d_gbps_measured", "d2h_gbps_measured", "unreadable")
    GPU_COUNT_FIELD_NUMBER: _ClassVar[int]
    GPU_NAME_FIELD_NUMBER: _ClassVar[int]
    GPU_SM_FIELD_NUMBER: _ClassVar[int]
    VRAM_TOTAL_BYTES_FIELD_NUMBER: _ClassVar[int]
    DRIVER_VERSION_FIELD_NUMBER: _ClassVar[int]
    CUDA_VERSION_FIELD_NUMBER: _ClassVar[int]
    TORCH_VERSION_FIELD_NUMBER: _ClassVar[int]
    PLATFORM_FIELD_NUMBER: _ClassVar[int]
    HOST_RAM_TOTAL_BYTES_FIELD_NUMBER: _ClassVar[int]
    VCPU_COUNT_FIELD_NUMBER: _ClassVar[int]
    DISK_TOTAL_BYTES_FIELD_NUMBER: _ClassVar[int]
    PINNED_CAPACITY_BYTES_FIELD_NUMBER: _ClassVar[int]
    POWER_CAP_WATTS_FIELD_NUMBER: _ClassVar[int]
    MIG_PROFILE_FIELD_NUMBER: _ClassVar[int]
    INTERCONNECT_FIELD_NUMBER: _ClassVar[int]
    PEER_ACCESS_FIELD_NUMBER: _ClassVar[int]
    H2D_GBPS_MEASURED_FIELD_NUMBER: _ClassVar[int]
    D2H_GBPS_MEASURED_FIELD_NUMBER: _ClassVar[int]
    UNREADABLE_FIELD_NUMBER: _ClassVar[int]
    gpu_count: int
    gpu_name: str
    gpu_sm: int
    vram_total_bytes: int
    driver_version: str
    cuda_version: str
    torch_version: str
    platform: str
    host_ram_total_bytes: int
    vcpu_count: int
    disk_total_bytes: int
    pinned_capacity_bytes: int
    power_cap_watts: int
    mig_profile: str
    interconnect: str
    peer_access: bool
    h2d_gbps_measured: int
    d2h_gbps_measured: int
    unreadable: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, gpu_count: _Optional[int] = ..., gpu_name: _Optional[str] = ..., gpu_sm: _Optional[int] = ..., vram_total_bytes: _Optional[int] = ..., driver_version: _Optional[str] = ..., cuda_version: _Optional[str] = ..., torch_version: _Optional[str] = ..., platform: _Optional[str] = ..., host_ram_total_bytes: _Optional[int] = ..., vcpu_count: _Optional[int] = ..., disk_total_bytes: _Optional[int] = ..., pinned_capacity_bytes: _Optional[int] = ..., power_cap_watts: _Optional[int] = ..., mig_profile: _Optional[str] = ..., interconnect: _Optional[str] = ..., peer_access: bool = ..., h2d_gbps_measured: _Optional[int] = ..., d2h_gbps_measured: _Optional[int] = ..., unreadable: _Optional[_Iterable[str]] = ...) -> None: ...

class ServingCapacity(_message.Message):
    __slots__ = ("ready_entrypoint_binding_plan_ids", "loadable_entrypoint_binding_plan_ids", "free_vram_bytes")
    READY_ENTRYPOINT_BINDING_PLAN_IDS_FIELD_NUMBER: _ClassVar[int]
    LOADABLE_ENTRYPOINT_BINDING_PLAN_IDS_FIELD_NUMBER: _ClassVar[int]
    FREE_VRAM_BYTES_FIELD_NUMBER: _ClassVar[int]
    ready_entrypoint_binding_plan_ids: _containers.RepeatedScalarFieldContainer[str]
    loadable_entrypoint_binding_plan_ids: _containers.RepeatedScalarFieldContainer[str]
    free_vram_bytes: int
    def __init__(self, ready_entrypoint_binding_plan_ids: _Optional[_Iterable[str]] = ..., loadable_entrypoint_binding_plan_ids: _Optional[_Iterable[str]] = ..., free_vram_bytes: _Optional[int] = ...) -> None: ...

class JobCapacity(_message.Message):
    __slots__ = ("jobs_in_flight", "jobs_available", "free_vram_bytes", "free_disk_bytes")
    JOBS_IN_FLIGHT_FIELD_NUMBER: _ClassVar[int]
    JOBS_AVAILABLE_FIELD_NUMBER: _ClassVar[int]
    FREE_VRAM_BYTES_FIELD_NUMBER: _ClassVar[int]
    FREE_DISK_BYTES_FIELD_NUMBER: _ClassVar[int]
    jobs_in_flight: int
    jobs_available: int
    free_vram_bytes: int
    free_disk_bytes: int
    def __init__(self, jobs_in_flight: _Optional[int] = ..., jobs_available: _Optional[int] = ..., free_vram_bytes: _Optional[int] = ..., free_disk_bytes: _Optional[int] = ...) -> None: ...

class ActiveAttempt(_message.Message):
    __slots__ = ("request_id", "attempt", "kind", "state", "exec_spec_digest")
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    ATTEMPT_FIELD_NUMBER: _ClassVar[int]
    KIND_FIELD_NUMBER: _ClassVar[int]
    STATE_FIELD_NUMBER: _ClassVar[int]
    EXEC_SPEC_DIGEST_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    attempt: int
    kind: AttemptKind
    state: AttemptState
    exec_spec_digest: bytes
    def __init__(self, request_id: _Optional[str] = ..., attempt: _Optional[int] = ..., kind: _Optional[_Union[AttemptKind, str]] = ..., state: _Optional[_Union[AttemptState, str]] = ..., exec_spec_digest: _Optional[bytes] = ...) -> None: ...

class Fault(_message.Message):
    __slots__ = ("kind", "subject", "reason", "detail")
    KIND_FIELD_NUMBER: _ClassVar[int]
    SUBJECT_FIELD_NUMBER: _ClassVar[int]
    REASON_FIELD_NUMBER: _ClassVar[int]
    DETAIL_FIELD_NUMBER: _ClassVar[int]
    kind: FaultKind
    subject: str
    reason: str
    detail: str
    def __init__(self, kind: _Optional[_Union[FaultKind, str]] = ..., subject: _Optional[str] = ..., reason: _Optional[str] = ..., detail: _Optional[str] = ...) -> None: ...

class OutputManifest(_message.Message):
    __slots__ = ("manifest_id", "outputs")
    MANIFEST_ID_FIELD_NUMBER: _ClassVar[int]
    OUTPUTS_FIELD_NUMBER: _ClassVar[int]
    manifest_id: str
    outputs: _containers.RepeatedCompositeFieldContainer[OutputEntry]
    def __init__(self, manifest_id: _Optional[str] = ..., outputs: _Optional[_Iterable[_Union[OutputEntry, _Mapping]]] = ...) -> None: ...

class OutputEntry(_message.Message):
    __slots__ = ("output_id", "digest", "length", "mime_type")
    OUTPUT_ID_FIELD_NUMBER: _ClassVar[int]
    DIGEST_FIELD_NUMBER: _ClassVar[int]
    LENGTH_FIELD_NUMBER: _ClassVar[int]
    MIME_TYPE_FIELD_NUMBER: _ClassVar[int]
    output_id: str
    digest: bytes
    length: int
    mime_type: str
    def __init__(self, output_id: _Optional[str] = ..., digest: _Optional[bytes] = ..., length: _Optional[int] = ..., mime_type: _Optional[str] = ...) -> None: ...

class AttemptMetrics(_message.Message):
    __slots__ = ("runtime_ms", "queue_ms", "peak_vram_bytes", "rss_at_end_bytes", "output_count", "input_tokens", "output_tokens", "device_lease_ms", "gpu_count", "handler_ms", "finalization_ms", "unverified_fields")
    RUNTIME_MS_FIELD_NUMBER: _ClassVar[int]
    QUEUE_MS_FIELD_NUMBER: _ClassVar[int]
    PEAK_VRAM_BYTES_FIELD_NUMBER: _ClassVar[int]
    RSS_AT_END_BYTES_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_COUNT_FIELD_NUMBER: _ClassVar[int]
    INPUT_TOKENS_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_TOKENS_FIELD_NUMBER: _ClassVar[int]
    DEVICE_LEASE_MS_FIELD_NUMBER: _ClassVar[int]
    GPU_COUNT_FIELD_NUMBER: _ClassVar[int]
    HANDLER_MS_FIELD_NUMBER: _ClassVar[int]
    FINALIZATION_MS_FIELD_NUMBER: _ClassVar[int]
    UNVERIFIED_FIELDS_FIELD_NUMBER: _ClassVar[int]
    runtime_ms: int
    queue_ms: int
    peak_vram_bytes: int
    rss_at_end_bytes: int
    output_count: int
    input_tokens: int
    output_tokens: int
    device_lease_ms: int
    gpu_count: int
    handler_ms: int
    finalization_ms: int
    unverified_fields: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, runtime_ms: _Optional[int] = ..., queue_ms: _Optional[int] = ..., peak_vram_bytes: _Optional[int] = ..., rss_at_end_bytes: _Optional[int] = ..., output_count: _Optional[int] = ..., input_tokens: _Optional[int] = ..., output_tokens: _Optional[int] = ..., device_lease_ms: _Optional[int] = ..., gpu_count: _Optional[int] = ..., handler_ms: _Optional[int] = ..., finalization_ms: _Optional[int] = ..., unverified_fields: _Optional[_Iterable[str]] = ...) -> None: ...

class TriageBundleRef(_message.Message):
    __slots__ = ("subject_id", "write_receipt_digest", "length")
    SUBJECT_ID_FIELD_NUMBER: _ClassVar[int]
    WRITE_RECEIPT_DIGEST_FIELD_NUMBER: _ClassVar[int]
    LENGTH_FIELD_NUMBER: _ClassVar[int]
    subject_id: str
    write_receipt_digest: bytes
    length: int
    def __init__(self, subject_id: _Optional[str] = ..., write_receipt_digest: _Optional[bytes] = ..., length: _Optional[int] = ...) -> None: ...

class PendingLeases(_message.Message):
    __slots__ = ("lease_ids",)
    LEASE_IDS_FIELD_NUMBER: _ClassVar[int]
    lease_ids: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, lease_ids: _Optional[_Iterable[str]] = ...) -> None: ...

class AbsenceEvidence(_message.Message):
    __slots__ = ("evidence_id", "evidence_digest")
    EVIDENCE_ID_FIELD_NUMBER: _ClassVar[int]
    EVIDENCE_DIGEST_FIELD_NUMBER: _ClassVar[int]
    evidence_id: str
    evidence_digest: bytes
    def __init__(self, evidence_id: _Optional[str] = ..., evidence_digest: _Optional[bytes] = ...) -> None: ...

class PurgeFault(_message.Message):
    __slots__ = ("code", "detail")
    CODE_FIELD_NUMBER: _ClassVar[int]
    DETAIL_FIELD_NUMBER: _ClassVar[int]
    code: PurgeFaultCode
    detail: str
    def __init__(self, code: _Optional[_Union[PurgeFaultCode, str]] = ..., detail: _Optional[str] = ...) -> None: ...
