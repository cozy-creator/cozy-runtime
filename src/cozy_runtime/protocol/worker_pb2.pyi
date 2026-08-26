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

class ClaimRejection(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    CLAIM_REJECTION_UNSPECIFIED: _ClassVar[ClaimRejection]
    CLAIM_REJECTION_UNAUTHENTICATED: _ClassVar[ClaimRejection]
    CLAIM_REJECTION_STALE_OWNER_EPOCH: _ClassVar[ClaimRejection]
    CLAIM_REJECTION_EPOCH_HELD: _ClassVar[ClaimRejection]
    CLAIM_REJECTION_WORKER_ID_MISMATCH: _ClassVar[ClaimRejection]
    CLAIM_REJECTION_RELEASE_ID_MISMATCH: _ClassVar[ClaimRejection]
    CLAIM_REJECTION_UNSUPPORTED_MINOR: _ClassVar[ClaimRejection]

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
    FAULT_KIND_DEPLOYMENT_SET_UNSUPPORTED: _ClassVar[FaultKind]

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
CLAIM_REJECTION_UNSPECIFIED: ClaimRejection
CLAIM_REJECTION_UNAUTHENTICATED: ClaimRejection
CLAIM_REJECTION_STALE_OWNER_EPOCH: ClaimRejection
CLAIM_REJECTION_EPOCH_HELD: ClaimRejection
CLAIM_REJECTION_WORKER_ID_MISMATCH: ClaimRejection
CLAIM_REJECTION_RELEASE_ID_MISMATCH: ClaimRejection
CLAIM_REJECTION_UNSUPPORTED_MINOR: ClaimRejection
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
FAULT_KIND_DEPLOYMENT_SET_UNSUPPORTED: FaultKind
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

class OwnerFrame(_message.Message):
    __slots__ = ("claim", "directive", "start_attempt", "cancel_attempt", "terminal_ack", "checkpoint_receipt", "snapshot_ack")
    CLAIM_FIELD_NUMBER: _ClassVar[int]
    DIRECTIVE_FIELD_NUMBER: _ClassVar[int]
    START_ATTEMPT_FIELD_NUMBER: _ClassVar[int]
    CANCEL_ATTEMPT_FIELD_NUMBER: _ClassVar[int]
    TERMINAL_ACK_FIELD_NUMBER: _ClassVar[int]
    CHECKPOINT_RECEIPT_FIELD_NUMBER: _ClassVar[int]
    SNAPSHOT_ACK_FIELD_NUMBER: _ClassVar[int]
    claim: Claim
    directive: Directive
    start_attempt: StartAttempt
    cancel_attempt: CancelAttempt
    terminal_ack: TerminalAck
    checkpoint_receipt: JobCheckpointReceipt
    snapshot_ack: SnapshotAck
    def __init__(self, claim: _Optional[_Union[Claim, _Mapping]] = ..., directive: _Optional[_Union[Directive, _Mapping]] = ..., start_attempt: _Optional[_Union[StartAttempt, _Mapping]] = ..., cancel_attempt: _Optional[_Union[CancelAttempt, _Mapping]] = ..., terminal_ack: _Optional[_Union[TerminalAck, _Mapping]] = ..., checkpoint_receipt: _Optional[_Union[JobCheckpointReceipt, _Mapping]] = ..., snapshot_ack: _Optional[_Union[SnapshotAck, _Mapping]] = ...) -> None: ...

class WorkerFrame(_message.Message):
    __slots__ = ("claim_ack", "report", "attempt_accepted", "attempt_terminal", "boot_failure", "checkpoint_request", "checkpoint_ack", "snapshot_begin", "snapshot_end", "snapshot_entry")
    CLAIM_ACK_FIELD_NUMBER: _ClassVar[int]
    REPORT_FIELD_NUMBER: _ClassVar[int]
    ATTEMPT_ACCEPTED_FIELD_NUMBER: _ClassVar[int]
    ATTEMPT_TERMINAL_FIELD_NUMBER: _ClassVar[int]
    BOOT_FAILURE_FIELD_NUMBER: _ClassVar[int]
    CHECKPOINT_REQUEST_FIELD_NUMBER: _ClassVar[int]
    CHECKPOINT_ACK_FIELD_NUMBER: _ClassVar[int]
    SNAPSHOT_BEGIN_FIELD_NUMBER: _ClassVar[int]
    SNAPSHOT_END_FIELD_NUMBER: _ClassVar[int]
    SNAPSHOT_ENTRY_FIELD_NUMBER: _ClassVar[int]
    claim_ack: ClaimAck
    report: Report
    attempt_accepted: AttemptAccepted
    attempt_terminal: AttemptTerminal
    boot_failure: BootFailure
    checkpoint_request: JobCheckpointRequest
    checkpoint_ack: JobCheckpointAck
    snapshot_begin: SnapshotBegin
    snapshot_end: SnapshotEnd
    snapshot_entry: SnapshotEntry
    def __init__(self, claim_ack: _Optional[_Union[ClaimAck, _Mapping]] = ..., report: _Optional[_Union[Report, _Mapping]] = ..., attempt_accepted: _Optional[_Union[AttemptAccepted, _Mapping]] = ..., attempt_terminal: _Optional[_Union[AttemptTerminal, _Mapping]] = ..., boot_failure: _Optional[_Union[BootFailure, _Mapping]] = ..., checkpoint_request: _Optional[_Union[JobCheckpointRequest, _Mapping]] = ..., checkpoint_ack: _Optional[_Union[JobCheckpointAck, _Mapping]] = ..., snapshot_begin: _Optional[_Union[SnapshotBegin, _Mapping]] = ..., snapshot_end: _Optional[_Union[SnapshotEnd, _Mapping]] = ..., snapshot_entry: _Optional[_Union[SnapshotEntry, _Mapping]] = ...) -> None: ...

class Claim(_message.Message):
    __slots__ = ("owner_epoch", "control_generation", "worker_boot_id", "controller_id", "worker_id", "wire_minor", "proof")
    OWNER_EPOCH_FIELD_NUMBER: _ClassVar[int]
    CONTROL_GENERATION_FIELD_NUMBER: _ClassVar[int]
    WORKER_BOOT_ID_FIELD_NUMBER: _ClassVar[int]
    CONTROLLER_ID_FIELD_NUMBER: _ClassVar[int]
    WORKER_ID_FIELD_NUMBER: _ClassVar[int]
    WIRE_MINOR_FIELD_NUMBER: _ClassVar[int]
    PROOF_FIELD_NUMBER: _ClassVar[int]
    owner_epoch: int
    control_generation: int
    worker_boot_id: str
    controller_id: str
    worker_id: str
    wire_minor: int
    proof: bytes
    def __init__(self, owner_epoch: _Optional[int] = ..., control_generation: _Optional[int] = ..., worker_boot_id: _Optional[str] = ..., controller_id: _Optional[str] = ..., worker_id: _Optional[str] = ..., wire_minor: _Optional[int] = ..., proof: _Optional[bytes] = ...) -> None: ...

class ClaimAck(_message.Message):
    __slots__ = ("owner_epoch", "control_generation", "worker_boot_id", "accepted", "rejection", "wire_minor", "worker_id", "instance_id", "release_id", "image_digest", "git_commit", "resources")
    OWNER_EPOCH_FIELD_NUMBER: _ClassVar[int]
    CONTROL_GENERATION_FIELD_NUMBER: _ClassVar[int]
    WORKER_BOOT_ID_FIELD_NUMBER: _ClassVar[int]
    ACCEPTED_FIELD_NUMBER: _ClassVar[int]
    REJECTION_FIELD_NUMBER: _ClassVar[int]
    WIRE_MINOR_FIELD_NUMBER: _ClassVar[int]
    WORKER_ID_FIELD_NUMBER: _ClassVar[int]
    INSTANCE_ID_FIELD_NUMBER: _ClassVar[int]
    RELEASE_ID_FIELD_NUMBER: _ClassVar[int]
    IMAGE_DIGEST_FIELD_NUMBER: _ClassVar[int]
    GIT_COMMIT_FIELD_NUMBER: _ClassVar[int]
    RESOURCES_FIELD_NUMBER: _ClassVar[int]
    owner_epoch: int
    control_generation: int
    worker_boot_id: str
    accepted: bool
    rejection: ClaimRejection
    wire_minor: int
    worker_id: str
    instance_id: str
    release_id: str
    image_digest: str
    git_commit: str
    resources: WorkerResources
    def __init__(self, owner_epoch: _Optional[int] = ..., control_generation: _Optional[int] = ..., worker_boot_id: _Optional[str] = ..., accepted: bool = ..., rejection: _Optional[_Union[ClaimRejection, str]] = ..., wire_minor: _Optional[int] = ..., worker_id: _Optional[str] = ..., instance_id: _Optional[str] = ..., release_id: _Optional[str] = ..., image_digest: _Optional[str] = ..., git_commit: _Optional[str] = ..., resources: _Optional[_Union[WorkerResources, _Mapping]] = ...) -> None: ...

class BootFailure(_message.Message):
    __slots__ = ("owner_epoch", "control_generation", "worker_boot_id", "worker_id", "instance_id", "reason", "detail", "resources", "image_digest", "release_id")
    OWNER_EPOCH_FIELD_NUMBER: _ClassVar[int]
    CONTROL_GENERATION_FIELD_NUMBER: _ClassVar[int]
    WORKER_BOOT_ID_FIELD_NUMBER: _ClassVar[int]
    WORKER_ID_FIELD_NUMBER: _ClassVar[int]
    INSTANCE_ID_FIELD_NUMBER: _ClassVar[int]
    REASON_FIELD_NUMBER: _ClassVar[int]
    DETAIL_FIELD_NUMBER: _ClassVar[int]
    RESOURCES_FIELD_NUMBER: _ClassVar[int]
    IMAGE_DIGEST_FIELD_NUMBER: _ClassVar[int]
    RELEASE_ID_FIELD_NUMBER: _ClassVar[int]
    owner_epoch: int
    control_generation: int
    worker_boot_id: str
    worker_id: str
    instance_id: str
    reason: BootFailureReason
    detail: str
    resources: WorkerResources
    image_digest: str
    release_id: str
    def __init__(self, owner_epoch: _Optional[int] = ..., control_generation: _Optional[int] = ..., worker_boot_id: _Optional[str] = ..., worker_id: _Optional[str] = ..., instance_id: _Optional[str] = ..., reason: _Optional[_Union[BootFailureReason, str]] = ..., detail: _Optional[str] = ..., resources: _Optional[_Union[WorkerResources, _Mapping]] = ..., image_digest: _Optional[str] = ..., release_id: _Optional[str] = ...) -> None: ...

class SnapshotBegin(_message.Message):
    __slots__ = ("owner_epoch", "control_generation", "worker_boot_id", "snapshot_id", "journal_highwater")
    OWNER_EPOCH_FIELD_NUMBER: _ClassVar[int]
    CONTROL_GENERATION_FIELD_NUMBER: _ClassVar[int]
    WORKER_BOOT_ID_FIELD_NUMBER: _ClassVar[int]
    SNAPSHOT_ID_FIELD_NUMBER: _ClassVar[int]
    JOURNAL_HIGHWATER_FIELD_NUMBER: _ClassVar[int]
    owner_epoch: int
    control_generation: int
    worker_boot_id: str
    snapshot_id: str
    journal_highwater: int
    def __init__(self, owner_epoch: _Optional[int] = ..., control_generation: _Optional[int] = ..., worker_boot_id: _Optional[str] = ..., snapshot_id: _Optional[str] = ..., journal_highwater: _Optional[int] = ...) -> None: ...

class SnapshotEntry(_message.Message):
    __slots__ = ("owner_epoch", "control_generation", "worker_boot_id", "snapshot_id", "attempt")
    OWNER_EPOCH_FIELD_NUMBER: _ClassVar[int]
    CONTROL_GENERATION_FIELD_NUMBER: _ClassVar[int]
    WORKER_BOOT_ID_FIELD_NUMBER: _ClassVar[int]
    SNAPSHOT_ID_FIELD_NUMBER: _ClassVar[int]
    ATTEMPT_FIELD_NUMBER: _ClassVar[int]
    owner_epoch: int
    control_generation: int
    worker_boot_id: str
    snapshot_id: str
    attempt: ActiveAttempt
    def __init__(self, owner_epoch: _Optional[int] = ..., control_generation: _Optional[int] = ..., worker_boot_id: _Optional[str] = ..., snapshot_id: _Optional[str] = ..., attempt: _Optional[_Union[ActiveAttempt, _Mapping]] = ...) -> None: ...

class SnapshotEnd(_message.Message):
    __slots__ = ("owner_epoch", "control_generation", "worker_boot_id", "snapshot_id", "entry_count")
    OWNER_EPOCH_FIELD_NUMBER: _ClassVar[int]
    CONTROL_GENERATION_FIELD_NUMBER: _ClassVar[int]
    WORKER_BOOT_ID_FIELD_NUMBER: _ClassVar[int]
    SNAPSHOT_ID_FIELD_NUMBER: _ClassVar[int]
    ENTRY_COUNT_FIELD_NUMBER: _ClassVar[int]
    owner_epoch: int
    control_generation: int
    worker_boot_id: str
    snapshot_id: str
    entry_count: int
    def __init__(self, owner_epoch: _Optional[int] = ..., control_generation: _Optional[int] = ..., worker_boot_id: _Optional[str] = ..., snapshot_id: _Optional[str] = ..., entry_count: _Optional[int] = ...) -> None: ...

class SnapshotAck(_message.Message):
    __slots__ = ("owner_epoch", "control_generation", "worker_boot_id", "snapshot_id")
    OWNER_EPOCH_FIELD_NUMBER: _ClassVar[int]
    CONTROL_GENERATION_FIELD_NUMBER: _ClassVar[int]
    WORKER_BOOT_ID_FIELD_NUMBER: _ClassVar[int]
    SNAPSHOT_ID_FIELD_NUMBER: _ClassVar[int]
    owner_epoch: int
    control_generation: int
    worker_boot_id: str
    snapshot_id: str
    def __init__(self, owner_epoch: _Optional[int] = ..., control_generation: _Optional[int] = ..., worker_boot_id: _Optional[str] = ..., snapshot_id: _Optional[str] = ...) -> None: ...

class Directive(_message.Message):
    __slots__ = ("owner_epoch", "control_generation", "worker_boot_id", "revision", "posture", "wire_minor", "drain_grace_ms", "job", "deployment_set")
    OWNER_EPOCH_FIELD_NUMBER: _ClassVar[int]
    CONTROL_GENERATION_FIELD_NUMBER: _ClassVar[int]
    WORKER_BOOT_ID_FIELD_NUMBER: _ClassVar[int]
    REVISION_FIELD_NUMBER: _ClassVar[int]
    POSTURE_FIELD_NUMBER: _ClassVar[int]
    WIRE_MINOR_FIELD_NUMBER: _ClassVar[int]
    DRAIN_GRACE_MS_FIELD_NUMBER: _ClassVar[int]
    JOB_FIELD_NUMBER: _ClassVar[int]
    DEPLOYMENT_SET_FIELD_NUMBER: _ClassVar[int]
    owner_epoch: int
    control_generation: int
    worker_boot_id: str
    revision: int
    posture: Posture
    wire_minor: int
    drain_grace_ms: int
    job: JobDirective
    deployment_set: DeploymentSetDirective
    def __init__(self, owner_epoch: _Optional[int] = ..., control_generation: _Optional[int] = ..., worker_boot_id: _Optional[str] = ..., revision: _Optional[int] = ..., posture: _Optional[_Union[Posture, str]] = ..., wire_minor: _Optional[int] = ..., drain_grace_ms: _Optional[int] = ..., job: _Optional[_Union[JobDirective, _Mapping]] = ..., deployment_set: _Optional[_Union[DeploymentSetDirective, _Mapping]] = ...) -> None: ...

class DeploymentSetDirective(_message.Message):
    __slots__ = ("set_digest", "set")
    SET_DIGEST_FIELD_NUMBER: _ClassVar[int]
    SET_FIELD_NUMBER: _ClassVar[int]
    set_digest: bytes
    set: DeploymentSet
    def __init__(self, set_digest: _Optional[bytes] = ..., set: _Optional[_Union[DeploymentSet, _Mapping]] = ...) -> None: ...

class DeploymentSet(_message.Message):
    __slots__ = ("deployments",)
    DEPLOYMENTS_FIELD_NUMBER: _ClassVar[int]
    deployments: _containers.RepeatedCompositeFieldContainer[Deployment]
    def __init__(self, deployments: _Optional[_Iterable[_Union[Deployment, _Mapping]]] = ...) -> None: ...

class Deployment(_message.Message):
    __slots__ = ("deployment_id", "endpoint_release_id", "entrypoint_binding_plan_ids")
    DEPLOYMENT_ID_FIELD_NUMBER: _ClassVar[int]
    ENDPOINT_RELEASE_ID_FIELD_NUMBER: _ClassVar[int]
    ENTRYPOINT_BINDING_PLAN_IDS_FIELD_NUMBER: _ClassVar[int]
    deployment_id: str
    endpoint_release_id: str
    entrypoint_binding_plan_ids: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, deployment_id: _Optional[str] = ..., endpoint_release_id: _Optional[str] = ..., entrypoint_binding_plan_ids: _Optional[_Iterable[str]] = ...) -> None: ...

class JobDirective(_message.Message):
    __slots__ = ("build_id", "job_descriptor_id", "resource_caps", "publication_contract", "reclaim_on_terminal", "device_count")
    BUILD_ID_FIELD_NUMBER: _ClassVar[int]
    JOB_DESCRIPTOR_ID_FIELD_NUMBER: _ClassVar[int]
    RESOURCE_CAPS_FIELD_NUMBER: _ClassVar[int]
    PUBLICATION_CONTRACT_FIELD_NUMBER: _ClassVar[int]
    RECLAIM_ON_TERMINAL_FIELD_NUMBER: _ClassVar[int]
    DEVICE_COUNT_FIELD_NUMBER: _ClassVar[int]
    build_id: str
    job_descriptor_id: str
    resource_caps: ResourceCaps
    publication_contract: PublicationContract
    reclaim_on_terminal: bool
    device_count: int
    def __init__(self, build_id: _Optional[str] = ..., job_descriptor_id: _Optional[str] = ..., resource_caps: _Optional[_Union[ResourceCaps, _Mapping]] = ..., publication_contract: _Optional[_Union[PublicationContract, _Mapping]] = ..., reclaim_on_terminal: bool = ..., device_count: _Optional[int] = ...) -> None: ...

class Report(_message.Message):
    __slots__ = ("owner_epoch", "control_generation", "worker_boot_id", "applied_revision", "intake_state", "applied_wire_minor", "active_attempts", "faults", "activity", "job_capacity", "applied_set_digest", "deployments")
    OWNER_EPOCH_FIELD_NUMBER: _ClassVar[int]
    CONTROL_GENERATION_FIELD_NUMBER: _ClassVar[int]
    WORKER_BOOT_ID_FIELD_NUMBER: _ClassVar[int]
    APPLIED_REVISION_FIELD_NUMBER: _ClassVar[int]
    INTAKE_STATE_FIELD_NUMBER: _ClassVar[int]
    APPLIED_WIRE_MINOR_FIELD_NUMBER: _ClassVar[int]
    ACTIVE_ATTEMPTS_FIELD_NUMBER: _ClassVar[int]
    FAULTS_FIELD_NUMBER: _ClassVar[int]
    ACTIVITY_FIELD_NUMBER: _ClassVar[int]
    JOB_CAPACITY_FIELD_NUMBER: _ClassVar[int]
    APPLIED_SET_DIGEST_FIELD_NUMBER: _ClassVar[int]
    DEPLOYMENTS_FIELD_NUMBER: _ClassVar[int]
    owner_epoch: int
    control_generation: int
    worker_boot_id: str
    applied_revision: int
    intake_state: IntakeState
    applied_wire_minor: int
    active_attempts: _containers.RepeatedCompositeFieldContainer[ActiveAttempt]
    faults: _containers.RepeatedCompositeFieldContainer[Fault]
    activity: _containers.RepeatedCompositeFieldContainer[ActivityEvent]
    job_capacity: JobCapacity
    applied_set_digest: bytes
    deployments: _containers.RepeatedCompositeFieldContainer[DeploymentStatus]
    def __init__(self, owner_epoch: _Optional[int] = ..., control_generation: _Optional[int] = ..., worker_boot_id: _Optional[str] = ..., applied_revision: _Optional[int] = ..., intake_state: _Optional[_Union[IntakeState, str]] = ..., applied_wire_minor: _Optional[int] = ..., active_attempts: _Optional[_Iterable[_Union[ActiveAttempt, _Mapping]]] = ..., faults: _Optional[_Iterable[_Union[Fault, _Mapping]]] = ..., activity: _Optional[_Iterable[_Union[ActivityEvent, _Mapping]]] = ..., job_capacity: _Optional[_Union[JobCapacity, _Mapping]] = ..., applied_set_digest: _Optional[bytes] = ..., deployments: _Optional[_Iterable[_Union[DeploymentStatus, _Mapping]]] = ...) -> None: ...

class DeploymentStatus(_message.Message):
    __slots__ = ("deployment_id", "intake_state", "readiness_epoch", "executor_generation", "attempt_credits", "ready_entrypoint_binding_plan_ids", "loadable_entrypoint_binding_plan_ids", "faults", "accelerator")
    DEPLOYMENT_ID_FIELD_NUMBER: _ClassVar[int]
    INTAKE_STATE_FIELD_NUMBER: _ClassVar[int]
    READINESS_EPOCH_FIELD_NUMBER: _ClassVar[int]
    EXECUTOR_GENERATION_FIELD_NUMBER: _ClassVar[int]
    ATTEMPT_CREDITS_FIELD_NUMBER: _ClassVar[int]
    READY_ENTRYPOINT_BINDING_PLAN_IDS_FIELD_NUMBER: _ClassVar[int]
    LOADABLE_ENTRYPOINT_BINDING_PLAN_IDS_FIELD_NUMBER: _ClassVar[int]
    FAULTS_FIELD_NUMBER: _ClassVar[int]
    ACCELERATOR_FIELD_NUMBER: _ClassVar[int]
    deployment_id: str
    intake_state: IntakeState
    readiness_epoch: int
    executor_generation: int
    attempt_credits: int
    ready_entrypoint_binding_plan_ids: _containers.RepeatedScalarFieldContainer[str]
    loadable_entrypoint_binding_plan_ids: _containers.RepeatedScalarFieldContainer[str]
    faults: _containers.RepeatedCompositeFieldContainer[Fault]
    accelerator: AcceleratorQualification
    def __init__(self, deployment_id: _Optional[str] = ..., intake_state: _Optional[_Union[IntakeState, str]] = ..., readiness_epoch: _Optional[int] = ..., executor_generation: _Optional[int] = ..., attempt_credits: _Optional[int] = ..., ready_entrypoint_binding_plan_ids: _Optional[_Iterable[str]] = ..., loadable_entrypoint_binding_plan_ids: _Optional[_Iterable[str]] = ..., faults: _Optional[_Iterable[_Union[Fault, _Mapping]]] = ..., accelerator: _Optional[_Union[AcceleratorQualification, _Mapping]] = ...) -> None: ...

class AcceleratorQualification(_message.Message):
    __slots__ = ("qualified", "capabilities", "recommended_max_bytes", "pinned_capacity_bytes", "h2d_gbps_measured", "d2h_gbps_measured", "torch_version", "unreadable")
    QUALIFIED_FIELD_NUMBER: _ClassVar[int]
    CAPABILITIES_FIELD_NUMBER: _ClassVar[int]
    RECOMMENDED_MAX_BYTES_FIELD_NUMBER: _ClassVar[int]
    PINNED_CAPACITY_BYTES_FIELD_NUMBER: _ClassVar[int]
    H2D_GBPS_MEASURED_FIELD_NUMBER: _ClassVar[int]
    D2H_GBPS_MEASURED_FIELD_NUMBER: _ClassVar[int]
    TORCH_VERSION_FIELD_NUMBER: _ClassVar[int]
    UNREADABLE_FIELD_NUMBER: _ClassVar[int]
    qualified: bool
    capabilities: _containers.RepeatedScalarFieldContainer[str]
    recommended_max_bytes: int
    pinned_capacity_bytes: int
    h2d_gbps_measured: int
    d2h_gbps_measured: int
    torch_version: str
    unreadable: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, qualified: bool = ..., capabilities: _Optional[_Iterable[str]] = ..., recommended_max_bytes: _Optional[int] = ..., pinned_capacity_bytes: _Optional[int] = ..., h2d_gbps_measured: _Optional[int] = ..., d2h_gbps_measured: _Optional[int] = ..., torch_version: _Optional[str] = ..., unreadable: _Optional[_Iterable[str]] = ...) -> None: ...

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
    __slots__ = ("owner_epoch", "control_generation", "worker_boot_id", "request_id", "attempt", "invocation_digest", "grant", "invocation_canonical", "deployment_id")
    OWNER_EPOCH_FIELD_NUMBER: _ClassVar[int]
    CONTROL_GENERATION_FIELD_NUMBER: _ClassVar[int]
    WORKER_BOOT_ID_FIELD_NUMBER: _ClassVar[int]
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    ATTEMPT_FIELD_NUMBER: _ClassVar[int]
    INVOCATION_DIGEST_FIELD_NUMBER: _ClassVar[int]
    GRANT_FIELD_NUMBER: _ClassVar[int]
    INVOCATION_CANONICAL_FIELD_NUMBER: _ClassVar[int]
    DEPLOYMENT_ID_FIELD_NUMBER: _ClassVar[int]
    owner_epoch: int
    control_generation: int
    worker_boot_id: str
    request_id: str
    attempt: int
    invocation_digest: bytes
    grant: DeliveryGrant
    invocation_canonical: bytes
    deployment_id: str
    def __init__(self, owner_epoch: _Optional[int] = ..., control_generation: _Optional[int] = ..., worker_boot_id: _Optional[str] = ..., request_id: _Optional[str] = ..., attempt: _Optional[int] = ..., invocation_digest: _Optional[bytes] = ..., grant: _Optional[_Union[DeliveryGrant, _Mapping]] = ..., invocation_canonical: _Optional[bytes] = ..., deployment_id: _Optional[str] = ...) -> None: ...

class AttemptAccepted(_message.Message):
    __slots__ = ("owner_epoch", "control_generation", "worker_boot_id", "request_id", "attempt", "invocation_digest", "plan_digest", "model_construction_digest", "plan", "deployment_id", "executor_generation")
    OWNER_EPOCH_FIELD_NUMBER: _ClassVar[int]
    CONTROL_GENERATION_FIELD_NUMBER: _ClassVar[int]
    WORKER_BOOT_ID_FIELD_NUMBER: _ClassVar[int]
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    ATTEMPT_FIELD_NUMBER: _ClassVar[int]
    INVOCATION_DIGEST_FIELD_NUMBER: _ClassVar[int]
    PLAN_DIGEST_FIELD_NUMBER: _ClassVar[int]
    MODEL_CONSTRUCTION_DIGEST_FIELD_NUMBER: _ClassVar[int]
    PLAN_FIELD_NUMBER: _ClassVar[int]
    DEPLOYMENT_ID_FIELD_NUMBER: _ClassVar[int]
    EXECUTOR_GENERATION_FIELD_NUMBER: _ClassVar[int]
    owner_epoch: int
    control_generation: int
    worker_boot_id: str
    request_id: str
    attempt: int
    invocation_digest: bytes
    plan_digest: bytes
    model_construction_digest: bytes
    plan: AttemptPlanSummary
    deployment_id: str
    executor_generation: int
    def __init__(self, owner_epoch: _Optional[int] = ..., control_generation: _Optional[int] = ..., worker_boot_id: _Optional[str] = ..., request_id: _Optional[str] = ..., attempt: _Optional[int] = ..., invocation_digest: _Optional[bytes] = ..., plan_digest: _Optional[bytes] = ..., model_construction_digest: _Optional[bytes] = ..., plan: _Optional[_Union[AttemptPlanSummary, _Mapping]] = ..., deployment_id: _Optional[str] = ..., executor_generation: _Optional[int] = ...) -> None: ...

class AttemptPlanSummary(_message.Message):
    __slots__ = ("delivery", "materialization", "compute_dtype", "placement", "reserved_device_memory_bytes", "reserved_host_bytes", "decision_explanation")
    DELIVERY_FIELD_NUMBER: _ClassVar[int]
    MATERIALIZATION_FIELD_NUMBER: _ClassVar[int]
    COMPUTE_DTYPE_FIELD_NUMBER: _ClassVar[int]
    PLACEMENT_FIELD_NUMBER: _ClassVar[int]
    RESERVED_DEVICE_MEMORY_BYTES_FIELD_NUMBER: _ClassVar[int]
    RESERVED_HOST_BYTES_FIELD_NUMBER: _ClassVar[int]
    DECISION_EXPLANATION_FIELD_NUMBER: _ClassVar[int]
    delivery: str
    materialization: str
    compute_dtype: str
    placement: str
    reserved_device_memory_bytes: int
    reserved_host_bytes: int
    decision_explanation: str
    def __init__(self, delivery: _Optional[str] = ..., materialization: _Optional[str] = ..., compute_dtype: _Optional[str] = ..., placement: _Optional[str] = ..., reserved_device_memory_bytes: _Optional[int] = ..., reserved_host_bytes: _Optional[int] = ..., decision_explanation: _Optional[str] = ...) -> None: ...

class CancelAttempt(_message.Message):
    __slots__ = ("owner_epoch", "control_generation", "worker_boot_id", "request_id", "attempt", "reason", "grace_ms", "invocation_digest")
    OWNER_EPOCH_FIELD_NUMBER: _ClassVar[int]
    CONTROL_GENERATION_FIELD_NUMBER: _ClassVar[int]
    WORKER_BOOT_ID_FIELD_NUMBER: _ClassVar[int]
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    ATTEMPT_FIELD_NUMBER: _ClassVar[int]
    REASON_FIELD_NUMBER: _ClassVar[int]
    GRACE_MS_FIELD_NUMBER: _ClassVar[int]
    INVOCATION_DIGEST_FIELD_NUMBER: _ClassVar[int]
    owner_epoch: int
    control_generation: int
    worker_boot_id: str
    request_id: str
    attempt: int
    reason: CancelReason
    grace_ms: int
    invocation_digest: bytes
    def __init__(self, owner_epoch: _Optional[int] = ..., control_generation: _Optional[int] = ..., worker_boot_id: _Optional[str] = ..., request_id: _Optional[str] = ..., attempt: _Optional[int] = ..., reason: _Optional[_Union[CancelReason, str]] = ..., grace_ms: _Optional[int] = ..., invocation_digest: _Optional[bytes] = ...) -> None: ...

class AttemptTerminal(_message.Message):
    __slots__ = ("owner_epoch", "control_generation", "worker_boot_id", "request_id", "attempt", "invocation_digest", "terminal_id", "terminal_digest", "terminal_canonical", "deployment_id")
    OWNER_EPOCH_FIELD_NUMBER: _ClassVar[int]
    CONTROL_GENERATION_FIELD_NUMBER: _ClassVar[int]
    WORKER_BOOT_ID_FIELD_NUMBER: _ClassVar[int]
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    ATTEMPT_FIELD_NUMBER: _ClassVar[int]
    INVOCATION_DIGEST_FIELD_NUMBER: _ClassVar[int]
    TERMINAL_ID_FIELD_NUMBER: _ClassVar[int]
    TERMINAL_DIGEST_FIELD_NUMBER: _ClassVar[int]
    TERMINAL_CANONICAL_FIELD_NUMBER: _ClassVar[int]
    DEPLOYMENT_ID_FIELD_NUMBER: _ClassVar[int]
    owner_epoch: int
    control_generation: int
    worker_boot_id: str
    request_id: str
    attempt: int
    invocation_digest: bytes
    terminal_id: str
    terminal_digest: bytes
    terminal_canonical: bytes
    deployment_id: str
    def __init__(self, owner_epoch: _Optional[int] = ..., control_generation: _Optional[int] = ..., worker_boot_id: _Optional[str] = ..., request_id: _Optional[str] = ..., attempt: _Optional[int] = ..., invocation_digest: _Optional[bytes] = ..., terminal_id: _Optional[str] = ..., terminal_digest: _Optional[bytes] = ..., terminal_canonical: _Optional[bytes] = ..., deployment_id: _Optional[str] = ...) -> None: ...

class TerminalBody(_message.Message):
    __slots__ = ("request_id", "attempt", "invocation_digest", "status", "output_manifest", "metrics", "triage_bundle", "safe_message", "cause", "result")
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    ATTEMPT_FIELD_NUMBER: _ClassVar[int]
    INVOCATION_DIGEST_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_MANIFEST_FIELD_NUMBER: _ClassVar[int]
    METRICS_FIELD_NUMBER: _ClassVar[int]
    TRIAGE_BUNDLE_FIELD_NUMBER: _ClassVar[int]
    SAFE_MESSAGE_FIELD_NUMBER: _ClassVar[int]
    CAUSE_FIELD_NUMBER: _ClassVar[int]
    RESULT_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    attempt: int
    invocation_digest: str
    status: TerminalStatus
    output_manifest: OutputManifest
    metrics: AttemptMetrics
    triage_bundle: TriageBundleRef
    safe_message: str
    cause: TerminalCause
    result: ResultEnvelope
    def __init__(self, request_id: _Optional[str] = ..., attempt: _Optional[int] = ..., invocation_digest: _Optional[str] = ..., status: _Optional[_Union[TerminalStatus, str]] = ..., output_manifest: _Optional[_Union[OutputManifest, _Mapping]] = ..., metrics: _Optional[_Union[AttemptMetrics, _Mapping]] = ..., triage_bundle: _Optional[_Union[TriageBundleRef, _Mapping]] = ..., safe_message: _Optional[str] = ..., cause: _Optional[_Union[TerminalCause, _Mapping]] = ..., result: _Optional[_Union[ResultEnvelope, _Mapping]] = ...) -> None: ...

class ResultEnvelope(_message.Message):
    __slots__ = ("result_schema_digest", "inline_result", "result_blob", "adjustments", "checkpoint_ref")
    RESULT_SCHEMA_DIGEST_FIELD_NUMBER: _ClassVar[int]
    INLINE_RESULT_FIELD_NUMBER: _ClassVar[int]
    RESULT_BLOB_FIELD_NUMBER: _ClassVar[int]
    ADJUSTMENTS_FIELD_NUMBER: _ClassVar[int]
    CHECKPOINT_REF_FIELD_NUMBER: _ClassVar[int]
    result_schema_digest: bytes
    inline_result: bytes
    result_blob: OutputEntry
    adjustments: _containers.RepeatedCompositeFieldContainer[AdjustmentRow]
    checkpoint_ref: str
    def __init__(self, result_schema_digest: _Optional[bytes] = ..., inline_result: _Optional[bytes] = ..., result_blob: _Optional[_Union[OutputEntry, _Mapping]] = ..., adjustments: _Optional[_Iterable[_Union[AdjustmentRow, _Mapping]]] = ..., checkpoint_ref: _Optional[str] = ...) -> None: ...

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
    __slots__ = ("owner_epoch", "control_generation", "worker_boot_id", "request_id", "attempt", "invocation_digest", "terminal_id", "terminal_digest")
    OWNER_EPOCH_FIELD_NUMBER: _ClassVar[int]
    CONTROL_GENERATION_FIELD_NUMBER: _ClassVar[int]
    WORKER_BOOT_ID_FIELD_NUMBER: _ClassVar[int]
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    ATTEMPT_FIELD_NUMBER: _ClassVar[int]
    INVOCATION_DIGEST_FIELD_NUMBER: _ClassVar[int]
    TERMINAL_ID_FIELD_NUMBER: _ClassVar[int]
    TERMINAL_DIGEST_FIELD_NUMBER: _ClassVar[int]
    owner_epoch: int
    control_generation: int
    worker_boot_id: str
    request_id: str
    attempt: int
    invocation_digest: bytes
    terminal_id: str
    terminal_digest: bytes
    def __init__(self, owner_epoch: _Optional[int] = ..., control_generation: _Optional[int] = ..., worker_boot_id: _Optional[str] = ..., request_id: _Optional[str] = ..., attempt: _Optional[int] = ..., invocation_digest: _Optional[bytes] = ..., terminal_id: _Optional[str] = ..., terminal_digest: _Optional[bytes] = ...) -> None: ...

class JobCheckpointRequest(_message.Message):
    __slots__ = ("owner_epoch", "control_generation", "worker_boot_id", "request_id", "attempt", "operation_key", "logical_key", "content_digest", "seq", "artifact")
    OWNER_EPOCH_FIELD_NUMBER: _ClassVar[int]
    CONTROL_GENERATION_FIELD_NUMBER: _ClassVar[int]
    WORKER_BOOT_ID_FIELD_NUMBER: _ClassVar[int]
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    ATTEMPT_FIELD_NUMBER: _ClassVar[int]
    OPERATION_KEY_FIELD_NUMBER: _ClassVar[int]
    LOGICAL_KEY_FIELD_NUMBER: _ClassVar[int]
    CONTENT_DIGEST_FIELD_NUMBER: _ClassVar[int]
    SEQ_FIELD_NUMBER: _ClassVar[int]
    ARTIFACT_FIELD_NUMBER: _ClassVar[int]
    owner_epoch: int
    control_generation: int
    worker_boot_id: str
    request_id: str
    attempt: int
    operation_key: str
    logical_key: str
    content_digest: bytes
    seq: int
    artifact: OutputEntry
    def __init__(self, owner_epoch: _Optional[int] = ..., control_generation: _Optional[int] = ..., worker_boot_id: _Optional[str] = ..., request_id: _Optional[str] = ..., attempt: _Optional[int] = ..., operation_key: _Optional[str] = ..., logical_key: _Optional[str] = ..., content_digest: _Optional[bytes] = ..., seq: _Optional[int] = ..., artifact: _Optional[_Union[OutputEntry, _Mapping]] = ...) -> None: ...

class JobCheckpointReceipt(_message.Message):
    __slots__ = ("owner_epoch", "control_generation", "worker_boot_id", "request_id", "attempt", "operation_key", "logical_key", "content_digest", "receipt_id", "outcome", "fault")
    OWNER_EPOCH_FIELD_NUMBER: _ClassVar[int]
    CONTROL_GENERATION_FIELD_NUMBER: _ClassVar[int]
    WORKER_BOOT_ID_FIELD_NUMBER: _ClassVar[int]
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    ATTEMPT_FIELD_NUMBER: _ClassVar[int]
    OPERATION_KEY_FIELD_NUMBER: _ClassVar[int]
    LOGICAL_KEY_FIELD_NUMBER: _ClassVar[int]
    CONTENT_DIGEST_FIELD_NUMBER: _ClassVar[int]
    RECEIPT_ID_FIELD_NUMBER: _ClassVar[int]
    OUTCOME_FIELD_NUMBER: _ClassVar[int]
    FAULT_FIELD_NUMBER: _ClassVar[int]
    owner_epoch: int
    control_generation: int
    worker_boot_id: str
    request_id: str
    attempt: int
    operation_key: str
    logical_key: str
    content_digest: bytes
    receipt_id: str
    outcome: CheckpointOutcome
    fault: CheckpointFault
    def __init__(self, owner_epoch: _Optional[int] = ..., control_generation: _Optional[int] = ..., worker_boot_id: _Optional[str] = ..., request_id: _Optional[str] = ..., attempt: _Optional[int] = ..., operation_key: _Optional[str] = ..., logical_key: _Optional[str] = ..., content_digest: _Optional[bytes] = ..., receipt_id: _Optional[str] = ..., outcome: _Optional[_Union[CheckpointOutcome, str]] = ..., fault: _Optional[_Union[CheckpointFault, _Mapping]] = ...) -> None: ...

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
    __slots__ = ("owner_epoch", "control_generation", "worker_boot_id", "request_id", "attempt", "operation_key", "receipt_id", "content_digest")
    OWNER_EPOCH_FIELD_NUMBER: _ClassVar[int]
    CONTROL_GENERATION_FIELD_NUMBER: _ClassVar[int]
    WORKER_BOOT_ID_FIELD_NUMBER: _ClassVar[int]
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    ATTEMPT_FIELD_NUMBER: _ClassVar[int]
    OPERATION_KEY_FIELD_NUMBER: _ClassVar[int]
    RECEIPT_ID_FIELD_NUMBER: _ClassVar[int]
    CONTENT_DIGEST_FIELD_NUMBER: _ClassVar[int]
    owner_epoch: int
    control_generation: int
    worker_boot_id: str
    request_id: str
    attempt: int
    operation_key: str
    receipt_id: str
    content_digest: bytes
    def __init__(self, owner_epoch: _Optional[int] = ..., control_generation: _Optional[int] = ..., worker_boot_id: _Optional[str] = ..., request_id: _Optional[str] = ..., attempt: _Optional[int] = ..., operation_key: _Optional[str] = ..., receipt_id: _Optional[str] = ..., content_digest: _Optional[bytes] = ...) -> None: ...

class ProgressOpen(_message.Message):
    __slots__ = ("owner_epoch", "control_generation", "worker_boot_id", "request_id")
    OWNER_EPOCH_FIELD_NUMBER: _ClassVar[int]
    CONTROL_GENERATION_FIELD_NUMBER: _ClassVar[int]
    WORKER_BOOT_ID_FIELD_NUMBER: _ClassVar[int]
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    owner_epoch: int
    control_generation: int
    worker_boot_id: str
    request_id: str
    def __init__(self, owner_epoch: _Optional[int] = ..., control_generation: _Optional[int] = ..., worker_boot_id: _Optional[str] = ..., request_id: _Optional[str] = ...) -> None: ...

class AttemptProgress(_message.Message):
    __slots__ = ("owner_epoch", "control_generation", "worker_boot_id", "request_id", "attempt", "seq", "content_type", "data", "deployment_id")
    OWNER_EPOCH_FIELD_NUMBER: _ClassVar[int]
    CONTROL_GENERATION_FIELD_NUMBER: _ClassVar[int]
    WORKER_BOOT_ID_FIELD_NUMBER: _ClassVar[int]
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    ATTEMPT_FIELD_NUMBER: _ClassVar[int]
    SEQ_FIELD_NUMBER: _ClassVar[int]
    CONTENT_TYPE_FIELD_NUMBER: _ClassVar[int]
    DATA_FIELD_NUMBER: _ClassVar[int]
    DEPLOYMENT_ID_FIELD_NUMBER: _ClassVar[int]
    owner_epoch: int
    control_generation: int
    worker_boot_id: str
    request_id: str
    attempt: int
    seq: int
    content_type: str
    data: bytes
    deployment_id: str
    def __init__(self, owner_epoch: _Optional[int] = ..., control_generation: _Optional[int] = ..., worker_boot_id: _Optional[str] = ..., request_id: _Optional[str] = ..., attempt: _Optional[int] = ..., seq: _Optional[int] = ..., content_type: _Optional[str] = ..., data: _Optional[bytes] = ..., deployment_id: _Optional[str] = ...) -> None: ...

class InvocationSpec(_message.Message):
    __slots__ = ("endpoint_release_id", "image_digest", "config_digest", "payload_digest", "inputs", "outputs", "deadline_unix_ms", "serving", "job")
    ENDPOINT_RELEASE_ID_FIELD_NUMBER: _ClassVar[int]
    IMAGE_DIGEST_FIELD_NUMBER: _ClassVar[int]
    CONFIG_DIGEST_FIELD_NUMBER: _ClassVar[int]
    PAYLOAD_DIGEST_FIELD_NUMBER: _ClassVar[int]
    INPUTS_FIELD_NUMBER: _ClassVar[int]
    OUTPUTS_FIELD_NUMBER: _ClassVar[int]
    DEADLINE_UNIX_MS_FIELD_NUMBER: _ClassVar[int]
    SERVING_FIELD_NUMBER: _ClassVar[int]
    JOB_FIELD_NUMBER: _ClassVar[int]
    endpoint_release_id: str
    image_digest: str
    config_digest: str
    payload_digest: str
    inputs: _containers.RepeatedCompositeFieldContainer[InputBinding]
    outputs: _containers.RepeatedCompositeFieldContainer[OutputBinding]
    deadline_unix_ms: int
    serving: ServingInvocationSpec
    job: JobInvocationSpec
    def __init__(self, endpoint_release_id: _Optional[str] = ..., image_digest: _Optional[str] = ..., config_digest: _Optional[str] = ..., payload_digest: _Optional[str] = ..., inputs: _Optional[_Iterable[_Union[InputBinding, _Mapping]]] = ..., outputs: _Optional[_Iterable[_Union[OutputBinding, _Mapping]]] = ..., deadline_unix_ms: _Optional[int] = ..., serving: _Optional[_Union[ServingInvocationSpec, _Mapping]] = ..., job: _Optional[_Union[JobInvocationSpec, _Mapping]] = ...) -> None: ...

class InputBinding(_message.Message):
    __slots__ = ("input_id", "digest", "length", "kind_mime", "order")
    INPUT_ID_FIELD_NUMBER: _ClassVar[int]
    DIGEST_FIELD_NUMBER: _ClassVar[int]
    LENGTH_FIELD_NUMBER: _ClassVar[int]
    KIND_MIME_FIELD_NUMBER: _ClassVar[int]
    ORDER_FIELD_NUMBER: _ClassVar[int]
    input_id: str
    digest: str
    length: int
    kind_mime: str
    order: int
    def __init__(self, input_id: _Optional[str] = ..., digest: _Optional[str] = ..., length: _Optional[int] = ..., kind_mime: _Optional[str] = ..., order: _Optional[int] = ...) -> None: ...

class OutputBinding(_message.Message):
    __slots__ = ("output_id", "mime_type", "max_bytes")
    OUTPUT_ID_FIELD_NUMBER: _ClassVar[int]
    MIME_TYPE_FIELD_NUMBER: _ClassVar[int]
    MAX_BYTES_FIELD_NUMBER: _ClassVar[int]
    output_id: str
    mime_type: str
    max_bytes: int
    def __init__(self, output_id: _Optional[str] = ..., mime_type: _Optional[str] = ..., max_bytes: _Optional[int] = ...) -> None: ...

class ServingInvocationSpec(_message.Message):
    __slots__ = ("entrypoint_binding_plan_id", "attempt_binding_id")
    ENTRYPOINT_BINDING_PLAN_ID_FIELD_NUMBER: _ClassVar[int]
    ATTEMPT_BINDING_ID_FIELD_NUMBER: _ClassVar[int]
    entrypoint_binding_plan_id: str
    attempt_binding_id: str
    def __init__(self, entrypoint_binding_plan_id: _Optional[str] = ..., attempt_binding_id: _Optional[str] = ...) -> None: ...

class JobInvocationSpec(_message.Message):
    __slots__ = ("build_id", "job_descriptor_id", "publication_contract")
    BUILD_ID_FIELD_NUMBER: _ClassVar[int]
    JOB_DESCRIPTOR_ID_FIELD_NUMBER: _ClassVar[int]
    PUBLICATION_CONTRACT_FIELD_NUMBER: _ClassVar[int]
    build_id: str
    job_descriptor_id: str
    publication_contract: PublicationContract
    def __init__(self, build_id: _Optional[str] = ..., job_descriptor_id: _Optional[str] = ..., publication_contract: _Optional[_Union[PublicationContract, _Mapping]] = ...) -> None: ...

class DeliveryGrant(_message.Message):
    __slots__ = ("invocation_digest", "credential", "file_base_url", "expires_at_unix", "inputs", "outputs")
    INVOCATION_DIGEST_FIELD_NUMBER: _ClassVar[int]
    CREDENTIAL_FIELD_NUMBER: _ClassVar[int]
    FILE_BASE_URL_FIELD_NUMBER: _ClassVar[int]
    EXPIRES_AT_UNIX_FIELD_NUMBER: _ClassVar[int]
    INPUTS_FIELD_NUMBER: _ClassVar[int]
    OUTPUTS_FIELD_NUMBER: _ClassVar[int]
    invocation_digest: bytes
    credential: Credential
    file_base_url: str
    expires_at_unix: int
    inputs: _containers.RepeatedCompositeFieldContainer[InputAccess]
    outputs: _containers.RepeatedCompositeFieldContainer[OutputAccess]
    def __init__(self, invocation_digest: _Optional[bytes] = ..., credential: _Optional[_Union[Credential, _Mapping]] = ..., file_base_url: _Optional[str] = ..., expires_at_unix: _Optional[int] = ..., inputs: _Optional[_Iterable[_Union[InputAccess, _Mapping]]] = ..., outputs: _Optional[_Iterable[_Union[OutputAccess, _Mapping]]] = ...) -> None: ...

class InputAccess(_message.Message):
    __slots__ = ("input_id", "url")
    INPUT_ID_FIELD_NUMBER: _ClassVar[int]
    URL_FIELD_NUMBER: _ClassVar[int]
    input_id: str
    url: str
    def __init__(self, input_id: _Optional[str] = ..., url: _Optional[str] = ...) -> None: ...

class OutputAccess(_message.Message):
    __slots__ = ("output_id", "url")
    OUTPUT_ID_FIELD_NUMBER: _ClassVar[int]
    URL_FIELD_NUMBER: _ClassVar[int]
    output_id: str
    url: str
    def __init__(self, output_id: _Optional[str] = ..., url: _Optional[str] = ...) -> None: ...

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

class ResourceCaps(_message.Message):
    __slots__ = ("device_required", "max_device_memory_bytes", "max_rss_bytes", "max_disk_bytes")
    DEVICE_REQUIRED_FIELD_NUMBER: _ClassVar[int]
    MAX_DEVICE_MEMORY_BYTES_FIELD_NUMBER: _ClassVar[int]
    MAX_RSS_BYTES_FIELD_NUMBER: _ClassVar[int]
    MAX_DISK_BYTES_FIELD_NUMBER: _ClassVar[int]
    device_required: bool
    max_device_memory_bytes: int
    max_rss_bytes: int
    max_disk_bytes: int
    def __init__(self, device_required: bool = ..., max_device_memory_bytes: _Optional[int] = ..., max_rss_bytes: _Optional[int] = ..., max_disk_bytes: _Optional[int] = ...) -> None: ...

class PublicationContract(_message.Message):
    __slots__ = ("outputs", "grant_id")
    OUTPUTS_FIELD_NUMBER: _ClassVar[int]
    GRANT_ID_FIELD_NUMBER: _ClassVar[int]
    outputs: _containers.RepeatedCompositeFieldContainer[OutputBinding]
    grant_id: str
    def __init__(self, outputs: _Optional[_Iterable[_Union[OutputBinding, _Mapping]]] = ..., grant_id: _Optional[str] = ...) -> None: ...

class WorkerResources(_message.Message):
    __slots__ = ("platform", "backend", "memory_model", "device_count", "device_name", "device_memory_total_bytes", "driver_version", "backend_version", "host_ram_total_bytes", "vcpu_count", "disk_total_bytes", "power_cap_watts", "partition_profile", "interconnect", "peer_access", "unreadable")
    PLATFORM_FIELD_NUMBER: _ClassVar[int]
    BACKEND_FIELD_NUMBER: _ClassVar[int]
    MEMORY_MODEL_FIELD_NUMBER: _ClassVar[int]
    DEVICE_COUNT_FIELD_NUMBER: _ClassVar[int]
    DEVICE_NAME_FIELD_NUMBER: _ClassVar[int]
    DEVICE_MEMORY_TOTAL_BYTES_FIELD_NUMBER: _ClassVar[int]
    DRIVER_VERSION_FIELD_NUMBER: _ClassVar[int]
    BACKEND_VERSION_FIELD_NUMBER: _ClassVar[int]
    HOST_RAM_TOTAL_BYTES_FIELD_NUMBER: _ClassVar[int]
    VCPU_COUNT_FIELD_NUMBER: _ClassVar[int]
    DISK_TOTAL_BYTES_FIELD_NUMBER: _ClassVar[int]
    POWER_CAP_WATTS_FIELD_NUMBER: _ClassVar[int]
    PARTITION_PROFILE_FIELD_NUMBER: _ClassVar[int]
    INTERCONNECT_FIELD_NUMBER: _ClassVar[int]
    PEER_ACCESS_FIELD_NUMBER: _ClassVar[int]
    UNREADABLE_FIELD_NUMBER: _ClassVar[int]
    platform: str
    backend: str
    memory_model: str
    device_count: int
    device_name: str
    device_memory_total_bytes: int
    driver_version: str
    backend_version: str
    host_ram_total_bytes: int
    vcpu_count: int
    disk_total_bytes: int
    power_cap_watts: int
    partition_profile: str
    interconnect: str
    peer_access: bool
    unreadable: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, platform: _Optional[str] = ..., backend: _Optional[str] = ..., memory_model: _Optional[str] = ..., device_count: _Optional[int] = ..., device_name: _Optional[str] = ..., device_memory_total_bytes: _Optional[int] = ..., driver_version: _Optional[str] = ..., backend_version: _Optional[str] = ..., host_ram_total_bytes: _Optional[int] = ..., vcpu_count: _Optional[int] = ..., disk_total_bytes: _Optional[int] = ..., power_cap_watts: _Optional[int] = ..., partition_profile: _Optional[str] = ..., interconnect: _Optional[str] = ..., peer_access: bool = ..., unreadable: _Optional[_Iterable[str]] = ...) -> None: ...

class JobCapacity(_message.Message):
    __slots__ = ("jobs_in_flight", "jobs_available", "free_disk_bytes")
    JOBS_IN_FLIGHT_FIELD_NUMBER: _ClassVar[int]
    JOBS_AVAILABLE_FIELD_NUMBER: _ClassVar[int]
    FREE_DISK_BYTES_FIELD_NUMBER: _ClassVar[int]
    jobs_in_flight: int
    jobs_available: int
    free_disk_bytes: int
    def __init__(self, jobs_in_flight: _Optional[int] = ..., jobs_available: _Optional[int] = ..., free_disk_bytes: _Optional[int] = ...) -> None: ...

class ActiveAttempt(_message.Message):
    __slots__ = ("request_id", "attempt", "kind", "state", "invocation_digest", "deployment_id", "executor_generation")
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    ATTEMPT_FIELD_NUMBER: _ClassVar[int]
    KIND_FIELD_NUMBER: _ClassVar[int]
    STATE_FIELD_NUMBER: _ClassVar[int]
    INVOCATION_DIGEST_FIELD_NUMBER: _ClassVar[int]
    DEPLOYMENT_ID_FIELD_NUMBER: _ClassVar[int]
    EXECUTOR_GENERATION_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    attempt: int
    kind: AttemptKind
    state: AttemptState
    invocation_digest: bytes
    deployment_id: str
    executor_generation: int
    def __init__(self, request_id: _Optional[str] = ..., attempt: _Optional[int] = ..., kind: _Optional[_Union[AttemptKind, str]] = ..., state: _Optional[_Union[AttemptState, str]] = ..., invocation_digest: _Optional[bytes] = ..., deployment_id: _Optional[str] = ..., executor_generation: _Optional[int] = ...) -> None: ...

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
    __slots__ = ("runtime_ms", "queue_ms", "peak_device_memory_bytes", "rss_at_end_bytes", "output_count", "input_tokens", "output_tokens", "device_lease_ms", "device_count", "handler_ms", "finalization_ms", "unverified_fields")
    RUNTIME_MS_FIELD_NUMBER: _ClassVar[int]
    QUEUE_MS_FIELD_NUMBER: _ClassVar[int]
    PEAK_DEVICE_MEMORY_BYTES_FIELD_NUMBER: _ClassVar[int]
    RSS_AT_END_BYTES_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_COUNT_FIELD_NUMBER: _ClassVar[int]
    INPUT_TOKENS_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_TOKENS_FIELD_NUMBER: _ClassVar[int]
    DEVICE_LEASE_MS_FIELD_NUMBER: _ClassVar[int]
    DEVICE_COUNT_FIELD_NUMBER: _ClassVar[int]
    HANDLER_MS_FIELD_NUMBER: _ClassVar[int]
    FINALIZATION_MS_FIELD_NUMBER: _ClassVar[int]
    UNVERIFIED_FIELDS_FIELD_NUMBER: _ClassVar[int]
    runtime_ms: int
    queue_ms: int
    peak_device_memory_bytes: int
    rss_at_end_bytes: int
    output_count: int
    input_tokens: int
    output_tokens: int
    device_lease_ms: int
    device_count: int
    handler_ms: int
    finalization_ms: int
    unverified_fields: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, runtime_ms: _Optional[int] = ..., queue_ms: _Optional[int] = ..., peak_device_memory_bytes: _Optional[int] = ..., rss_at_end_bytes: _Optional[int] = ..., output_count: _Optional[int] = ..., input_tokens: _Optional[int] = ..., output_tokens: _Optional[int] = ..., device_lease_ms: _Optional[int] = ..., device_count: _Optional[int] = ..., handler_ms: _Optional[int] = ..., finalization_ms: _Optional[int] = ..., unverified_fields: _Optional[_Iterable[str]] = ...) -> None: ...

class TriageBundleRef(_message.Message):
    __slots__ = ("subject_id", "write_receipt_digest", "length")
    SUBJECT_ID_FIELD_NUMBER: _ClassVar[int]
    WRITE_RECEIPT_DIGEST_FIELD_NUMBER: _ClassVar[int]
    LENGTH_FIELD_NUMBER: _ClassVar[int]
    subject_id: str
    write_receipt_digest: bytes
    length: int
    def __init__(self, subject_id: _Optional[str] = ..., write_receipt_digest: _Optional[bytes] = ..., length: _Optional[int] = ...) -> None: ...
