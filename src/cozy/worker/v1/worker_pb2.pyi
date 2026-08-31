from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class LocalDownloadKind(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    LOCAL_DOWNLOAD_KIND_UNSPECIFIED: _ClassVar[LocalDownloadKind]
    LOCAL_DOWNLOAD_KIND_PROJECT_WHEEL: _ClassVar[LocalDownloadKind]
    LOCAL_DOWNLOAD_KIND_DEPENDENCY_WHEEL: _ClassVar[LocalDownloadKind]
    LOCAL_DOWNLOAD_KIND_MODEL_MANIFEST: _ClassVar[LocalDownloadKind]
    LOCAL_DOWNLOAD_KIND_MODEL_OBJECT: _ClassVar[LocalDownloadKind]

class Posture(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    POSTURE_UNSPECIFIED: _ClassVar[Posture]
    POSTURE_ACCEPTING: _ClassVar[Posture]
    POSTURE_DRAINING: _ClassVar[Posture]

class WorkerPhase(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    WORKER_PHASE_UNSPECIFIED: _ClassVar[WorkerPhase]
    WORKER_PHASE_BOOTING: _ClassVar[WorkerPhase]
    WORKER_PHASE_ONLINE: _ClassVar[WorkerPhase]
    WORKER_PHASE_DRAINING: _ClassVar[WorkerPhase]
    WORKER_PHASE_FAILED: _ClassVar[WorkerPhase]

class MaterializationState(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    MATERIALIZATION_STATE_UNSPECIFIED: _ClassVar[MaterializationState]
    MATERIALIZATION_STATE_ABSENT: _ClassVar[MaterializationState]
    MATERIALIZATION_STATE_MATERIALIZING: _ClassVar[MaterializationState]
    MATERIALIZATION_STATE_STAGED: _ClassVar[MaterializationState]
    MATERIALIZATION_STATE_FAILED: _ClassVar[MaterializationState]

class ServingState(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    SERVING_STATE_UNSPECIFIED: _ClassVar[ServingState]
    SERVING_STATE_OFFLINE: _ClassVar[ServingState]
    SERVING_STATE_ACTIVATING: _ClassVar[ServingState]
    SERVING_STATE_DISPATCHABLE: _ClassVar[ServingState]
    SERVING_STATE_DRAINING: _ClassVar[ServingState]

class AdmissionState(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    ADMISSION_STATE_UNSPECIFIED: _ClassVar[AdmissionState]
    ADMISSION_STATE_OPEN: _ClassVar[AdmissionState]
    ADMISSION_STATE_CLOSED: _ClassVar[AdmissionState]

class AttemptKind(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    ATTEMPT_KIND_UNSPECIFIED: _ClassVar[AttemptKind]
    ATTEMPT_KIND_SERVING: _ClassVar[AttemptKind]
    ATTEMPT_KIND_JOB: _ClassVar[AttemptKind]

class AttemptState(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    ATTEMPT_STATE_UNSPECIFIED: _ClassVar[AttemptState]
    ATTEMPT_STATE_RUNNING: _ClassVar[AttemptState]
    ATTEMPT_STATE_OUTCOME_PENDING_ACK: _ClassVar[AttemptState]
    ATTEMPT_STATE_HELD_UNDURABLE: _ClassVar[AttemptState]

class OutcomeStatus(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    OUTCOME_STATUS_UNSPECIFIED: _ClassVar[OutcomeStatus]
    OUTCOME_STATUS_SUCCEEDED: _ClassVar[OutcomeStatus]
    OUTCOME_STATUS_REFUSED: _ClassVar[OutcomeStatus]
    OUTCOME_STATUS_FAILED: _ClassVar[OutcomeStatus]
    OUTCOME_STATUS_CANCELED: _ClassVar[OutcomeStatus]
    OUTCOME_STATUS_ABANDONED: _ClassVar[OutcomeStatus]

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
    CAUSE_CODE_NO_CAPACITY: _ClassVar[CauseCode]
    CAUSE_CODE_ADMISSION_GENERATION_STALE: _ClassVar[CauseCode]
    CAUSE_CODE_UNKNOWN_PLACEMENT: _ClassVar[CauseCode]
    CAUSE_CODE_PLACEMENT_NOT_DISPATCHABLE: _ClassVar[CauseCode]

class CauseOrigin(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    CAUSE_ORIGIN_UNSPECIFIED: _ClassVar[CauseOrigin]
    CAUSE_ORIGIN_AUTHOR: _ClassVar[CauseOrigin]
    CAUSE_ORIGIN_RUNTIME: _ClassVar[CauseOrigin]
    CAUSE_ORIGIN_EXECUTOR: _ClassVar[CauseOrigin]
    CAUSE_ORIGIN_WORKER: _ClassVar[CauseOrigin]
    CAUSE_ORIGIN_INFRA: _ClassVar[CauseOrigin]
    CAUSE_ORIGIN_CLIENT: _ClassVar[CauseOrigin]
    CAUSE_ORIGIN_RECORD_OWNER: _ClassVar[CauseOrigin]

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
    CLAIM_REJECTION_STALE_RECORD_OWNER_EPOCH: _ClassVar[ClaimRejection]
    CLAIM_REJECTION_EPOCH_HELD: _ClassVar[ClaimRejection]
    CLAIM_REJECTION_WORKER_ID_MISMATCH: _ClassVar[ClaimRejection]
    CLAIM_REJECTION_RELEASE_ID_MISMATCH: _ClassVar[ClaimRejection]
    CLAIM_REJECTION_UNDURABLE: _ClassVar[ClaimRejection]

class FaultKind(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    FAULT_KIND_UNSPECIFIED: _ClassVar[FaultKind]
    FAULT_KIND_BINDING_UNAVAILABLE: _ClassVar[FaultKind]
    FAULT_KIND_BINDING_DEGRADED: _ClassVar[FaultKind]
    FAULT_KIND_HARDWARE_UNSUITABLE: _ClassVar[FaultKind]
    FAULT_KIND_ARTIFACT_FETCH_FAILED: _ClassVar[FaultKind]
    FAULT_KIND_CREDENTIAL_UNAPPLIED: _ClassVar[FaultKind]
    FAULT_KIND_LOCAL_SAFETY_REFUSAL: _ClassVar[FaultKind]
    FAULT_KIND_EXECUTOR_POISONED: _ClassVar[FaultKind]
    FAULT_KIND_CONFIG_REFUSED: _ClassVar[FaultKind]
    FAULT_KIND_UNKNOWN_PLACEMENT: _ClassVar[FaultKind]
    FAULT_KIND_PLACEMENT_SET_UNSUPPORTED: _ClassVar[FaultKind]
    FAULT_KIND_PLACEMENT_SET_DIGEST_MISMATCH: _ClassVar[FaultKind]
    FAULT_KIND_ENVIRONMENT_RECEIPT_MISMATCH: _ClassVar[FaultKind]
    FAULT_KIND_ARTIFACT_DIGEST_MISMATCH: _ClassVar[FaultKind]
    FAULT_KIND_FALLBACK_PIN_MISSING: _ClassVar[FaultKind]

class BootFailureReason(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    BOOT_FAILURE_REASON_UNSPECIFIED: _ClassVar[BootFailureReason]
    BOOT_FAILURE_REASON_HARDWARE_VERDICT: _ClassVar[BootFailureReason]
    BOOT_FAILURE_REASON_ARTIFACT_FAULT: _ClassVar[BootFailureReason]
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

class ArtifactHostStage(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    ARTIFACT_HOST_STAGE_UNSPECIFIED: _ClassVar[ArtifactHostStage]
    ARTIFACT_HOST_STAGE_INTENT: _ClassVar[ArtifactHostStage]
    ARTIFACT_HOST_STAGE_RECEIPT: _ClassVar[ArtifactHostStage]

class ArtifactHostOutcome(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    ARTIFACT_HOST_OUTCOME_UNSPECIFIED: _ClassVar[ArtifactHostOutcome]
    ARTIFACT_HOST_OUTCOME_RECORDED: _ClassVar[ArtifactHostOutcome]
    ARTIFACT_HOST_OUTCOME_REPLAYED: _ClassVar[ArtifactHostOutcome]
    ARTIFACT_HOST_OUTCOME_REFUSED: _ClassVar[ArtifactHostOutcome]

class ArtifactHostRefusal(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    ARTIFACT_HOST_REFUSAL_UNSPECIFIED: _ClassVar[ArtifactHostRefusal]
    ARTIFACT_HOST_REFUSAL_UNKNOWN_ATTEMPT: _ClassVar[ArtifactHostRefusal]
    ARTIFACT_HOST_REFUSAL_INTENT_CONFLICT: _ClassVar[ArtifactHostRefusal]
    ARTIFACT_HOST_REFUSAL_STALE_WRITER: _ClassVar[ArtifactHostRefusal]
    ARTIFACT_HOST_REFUSAL_RECEIPT_CONFLICT: _ClassVar[ArtifactHostRefusal]
    ARTIFACT_HOST_REFUSAL_INVENTORY_INVALID: _ClassVar[ArtifactHostRefusal]

class ArtifactTransactionState(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    ARTIFACT_TRANSACTION_STATE_UNSPECIFIED: _ClassVar[ArtifactTransactionState]
    ARTIFACT_TRANSACTION_STATE_INTENT: _ClassVar[ArtifactTransactionState]
    ARTIFACT_TRANSACTION_STATE_RECEIPT: _ClassVar[ArtifactTransactionState]

class ArtifactReadOutcome(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    ARTIFACT_READ_OUTCOME_UNSPECIFIED: _ClassVar[ArtifactReadOutcome]
    ARTIFACT_READ_OUTCOME_DATA: _ClassVar[ArtifactReadOutcome]
    ARTIFACT_READ_OUTCOME_REFUSED: _ClassVar[ArtifactReadOutcome]

class ArtifactReadRefusal(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    ARTIFACT_READ_REFUSAL_UNSPECIFIED: _ClassVar[ArtifactReadRefusal]
    ARTIFACT_READ_REFUSAL_UNKNOWN_TRANSACTION: _ClassVar[ArtifactReadRefusal]
    ARTIFACT_READ_REFUSAL_STALE_WRITER: _ClassVar[ArtifactReadRefusal]
    ARTIFACT_READ_REFUSAL_UNKNOWN_OBJECT: _ClassVar[ArtifactReadRefusal]
    ARTIFACT_READ_REFUSAL_SOURCE_REF_MISMATCH: _ClassVar[ArtifactReadRefusal]
    ARTIFACT_READ_REFUSAL_RANGE_INVALID: _ClassVar[ArtifactReadRefusal]
    ARTIFACT_READ_REFUSAL_SOURCE_UNAVAILABLE: _ClassVar[ArtifactReadRefusal]

class ArtifactTransferState(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    ARTIFACT_TRANSFER_STATE_UNSPECIFIED: _ClassVar[ArtifactTransferState]
    ARTIFACT_TRANSFER_STATE_ACCEPTED: _ClassVar[ArtifactTransferState]
    ARTIFACT_TRANSFER_STATE_UPLOADING: _ClassVar[ArtifactTransferState]
    ARTIFACT_TRANSFER_STATE_UPLOADED: _ClassVar[ArtifactTransferState]
    ARTIFACT_TRANSFER_STATE_ALREADY_PRESENT: _ClassVar[ArtifactTransferState]
    ARTIFACT_TRANSFER_STATE_HELD: _ClassVar[ArtifactTransferState]
    ARTIFACT_TRANSFER_STATE_FAILED: _ClassVar[ArtifactTransferState]

class ArtifactFinalizeDisposition(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    ARTIFACT_FINALIZE_DISPOSITION_UNSPECIFIED: _ClassVar[ArtifactFinalizeDisposition]
    ARTIFACT_FINALIZE_DISPOSITION_ADOPT: _ClassVar[ArtifactFinalizeDisposition]
    ARTIFACT_FINALIZE_DISPOSITION_ABANDON: _ClassVar[ArtifactFinalizeDisposition]
    ARTIFACT_FINALIZE_DISPOSITION_ABANDON_UNCOMMITTED: _ClassVar[ArtifactFinalizeDisposition]

class ArtifactFinalizeOutcome(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    ARTIFACT_FINALIZE_OUTCOME_UNSPECIFIED: _ClassVar[ArtifactFinalizeOutcome]
    ARTIFACT_FINALIZE_OUTCOME_ADOPTED: _ClassVar[ArtifactFinalizeOutcome]
    ARTIFACT_FINALIZE_OUTCOME_ABANDONED: _ClassVar[ArtifactFinalizeOutcome]
LOCAL_DOWNLOAD_KIND_UNSPECIFIED: LocalDownloadKind
LOCAL_DOWNLOAD_KIND_PROJECT_WHEEL: LocalDownloadKind
LOCAL_DOWNLOAD_KIND_DEPENDENCY_WHEEL: LocalDownloadKind
LOCAL_DOWNLOAD_KIND_MODEL_MANIFEST: LocalDownloadKind
LOCAL_DOWNLOAD_KIND_MODEL_OBJECT: LocalDownloadKind
POSTURE_UNSPECIFIED: Posture
POSTURE_ACCEPTING: Posture
POSTURE_DRAINING: Posture
WORKER_PHASE_UNSPECIFIED: WorkerPhase
WORKER_PHASE_BOOTING: WorkerPhase
WORKER_PHASE_ONLINE: WorkerPhase
WORKER_PHASE_DRAINING: WorkerPhase
WORKER_PHASE_FAILED: WorkerPhase
MATERIALIZATION_STATE_UNSPECIFIED: MaterializationState
MATERIALIZATION_STATE_ABSENT: MaterializationState
MATERIALIZATION_STATE_MATERIALIZING: MaterializationState
MATERIALIZATION_STATE_STAGED: MaterializationState
MATERIALIZATION_STATE_FAILED: MaterializationState
SERVING_STATE_UNSPECIFIED: ServingState
SERVING_STATE_OFFLINE: ServingState
SERVING_STATE_ACTIVATING: ServingState
SERVING_STATE_DISPATCHABLE: ServingState
SERVING_STATE_DRAINING: ServingState
ADMISSION_STATE_UNSPECIFIED: AdmissionState
ADMISSION_STATE_OPEN: AdmissionState
ADMISSION_STATE_CLOSED: AdmissionState
ATTEMPT_KIND_UNSPECIFIED: AttemptKind
ATTEMPT_KIND_SERVING: AttemptKind
ATTEMPT_KIND_JOB: AttemptKind
ATTEMPT_STATE_UNSPECIFIED: AttemptState
ATTEMPT_STATE_RUNNING: AttemptState
ATTEMPT_STATE_OUTCOME_PENDING_ACK: AttemptState
ATTEMPT_STATE_HELD_UNDURABLE: AttemptState
OUTCOME_STATUS_UNSPECIFIED: OutcomeStatus
OUTCOME_STATUS_SUCCEEDED: OutcomeStatus
OUTCOME_STATUS_REFUSED: OutcomeStatus
OUTCOME_STATUS_FAILED: OutcomeStatus
OUTCOME_STATUS_CANCELED: OutcomeStatus
OUTCOME_STATUS_ABANDONED: OutcomeStatus
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
CAUSE_CODE_NO_CAPACITY: CauseCode
CAUSE_CODE_ADMISSION_GENERATION_STALE: CauseCode
CAUSE_CODE_UNKNOWN_PLACEMENT: CauseCode
CAUSE_CODE_PLACEMENT_NOT_DISPATCHABLE: CauseCode
CAUSE_ORIGIN_UNSPECIFIED: CauseOrigin
CAUSE_ORIGIN_AUTHOR: CauseOrigin
CAUSE_ORIGIN_RUNTIME: CauseOrigin
CAUSE_ORIGIN_EXECUTOR: CauseOrigin
CAUSE_ORIGIN_WORKER: CauseOrigin
CAUSE_ORIGIN_INFRA: CauseOrigin
CAUSE_ORIGIN_CLIENT: CauseOrigin
CAUSE_ORIGIN_RECORD_OWNER: CauseOrigin
CANCEL_REASON_UNSPECIFIED: CancelReason
CANCEL_REASON_CLIENT: CancelReason
CANCEL_REASON_DRAIN: CancelReason
CANCEL_REASON_SUPERSEDED: CancelReason
CANCEL_REASON_POLICY: CancelReason
CANCEL_REASON_DEADLINE: CancelReason
CLAIM_REJECTION_UNSPECIFIED: ClaimRejection
CLAIM_REJECTION_UNAUTHENTICATED: ClaimRejection
CLAIM_REJECTION_STALE_RECORD_OWNER_EPOCH: ClaimRejection
CLAIM_REJECTION_EPOCH_HELD: ClaimRejection
CLAIM_REJECTION_WORKER_ID_MISMATCH: ClaimRejection
CLAIM_REJECTION_RELEASE_ID_MISMATCH: ClaimRejection
CLAIM_REJECTION_UNDURABLE: ClaimRejection
FAULT_KIND_UNSPECIFIED: FaultKind
FAULT_KIND_BINDING_UNAVAILABLE: FaultKind
FAULT_KIND_BINDING_DEGRADED: FaultKind
FAULT_KIND_HARDWARE_UNSUITABLE: FaultKind
FAULT_KIND_ARTIFACT_FETCH_FAILED: FaultKind
FAULT_KIND_CREDENTIAL_UNAPPLIED: FaultKind
FAULT_KIND_LOCAL_SAFETY_REFUSAL: FaultKind
FAULT_KIND_EXECUTOR_POISONED: FaultKind
FAULT_KIND_CONFIG_REFUSED: FaultKind
FAULT_KIND_UNKNOWN_PLACEMENT: FaultKind
FAULT_KIND_PLACEMENT_SET_UNSUPPORTED: FaultKind
FAULT_KIND_PLACEMENT_SET_DIGEST_MISMATCH: FaultKind
FAULT_KIND_ENVIRONMENT_RECEIPT_MISMATCH: FaultKind
FAULT_KIND_ARTIFACT_DIGEST_MISMATCH: FaultKind
FAULT_KIND_FALLBACK_PIN_MISSING: FaultKind
BOOT_FAILURE_REASON_UNSPECIFIED: BootFailureReason
BOOT_FAILURE_REASON_HARDWARE_VERDICT: BootFailureReason
BOOT_FAILURE_REASON_ARTIFACT_FAULT: BootFailureReason
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
ARTIFACT_HOST_STAGE_UNSPECIFIED: ArtifactHostStage
ARTIFACT_HOST_STAGE_INTENT: ArtifactHostStage
ARTIFACT_HOST_STAGE_RECEIPT: ArtifactHostStage
ARTIFACT_HOST_OUTCOME_UNSPECIFIED: ArtifactHostOutcome
ARTIFACT_HOST_OUTCOME_RECORDED: ArtifactHostOutcome
ARTIFACT_HOST_OUTCOME_REPLAYED: ArtifactHostOutcome
ARTIFACT_HOST_OUTCOME_REFUSED: ArtifactHostOutcome
ARTIFACT_HOST_REFUSAL_UNSPECIFIED: ArtifactHostRefusal
ARTIFACT_HOST_REFUSAL_UNKNOWN_ATTEMPT: ArtifactHostRefusal
ARTIFACT_HOST_REFUSAL_INTENT_CONFLICT: ArtifactHostRefusal
ARTIFACT_HOST_REFUSAL_STALE_WRITER: ArtifactHostRefusal
ARTIFACT_HOST_REFUSAL_RECEIPT_CONFLICT: ArtifactHostRefusal
ARTIFACT_HOST_REFUSAL_INVENTORY_INVALID: ArtifactHostRefusal
ARTIFACT_TRANSACTION_STATE_UNSPECIFIED: ArtifactTransactionState
ARTIFACT_TRANSACTION_STATE_INTENT: ArtifactTransactionState
ARTIFACT_TRANSACTION_STATE_RECEIPT: ArtifactTransactionState
ARTIFACT_READ_OUTCOME_UNSPECIFIED: ArtifactReadOutcome
ARTIFACT_READ_OUTCOME_DATA: ArtifactReadOutcome
ARTIFACT_READ_OUTCOME_REFUSED: ArtifactReadOutcome
ARTIFACT_READ_REFUSAL_UNSPECIFIED: ArtifactReadRefusal
ARTIFACT_READ_REFUSAL_UNKNOWN_TRANSACTION: ArtifactReadRefusal
ARTIFACT_READ_REFUSAL_STALE_WRITER: ArtifactReadRefusal
ARTIFACT_READ_REFUSAL_UNKNOWN_OBJECT: ArtifactReadRefusal
ARTIFACT_READ_REFUSAL_SOURCE_REF_MISMATCH: ArtifactReadRefusal
ARTIFACT_READ_REFUSAL_RANGE_INVALID: ArtifactReadRefusal
ARTIFACT_READ_REFUSAL_SOURCE_UNAVAILABLE: ArtifactReadRefusal
ARTIFACT_TRANSFER_STATE_UNSPECIFIED: ArtifactTransferState
ARTIFACT_TRANSFER_STATE_ACCEPTED: ArtifactTransferState
ARTIFACT_TRANSFER_STATE_UPLOADING: ArtifactTransferState
ARTIFACT_TRANSFER_STATE_UPLOADED: ArtifactTransferState
ARTIFACT_TRANSFER_STATE_ALREADY_PRESENT: ArtifactTransferState
ARTIFACT_TRANSFER_STATE_HELD: ArtifactTransferState
ARTIFACT_TRANSFER_STATE_FAILED: ArtifactTransferState
ARTIFACT_FINALIZE_DISPOSITION_UNSPECIFIED: ArtifactFinalizeDisposition
ARTIFACT_FINALIZE_DISPOSITION_ADOPT: ArtifactFinalizeDisposition
ARTIFACT_FINALIZE_DISPOSITION_ABANDON: ArtifactFinalizeDisposition
ARTIFACT_FINALIZE_DISPOSITION_ABANDON_UNCOMMITTED: ArtifactFinalizeDisposition
ARTIFACT_FINALIZE_OUTCOME_UNSPECIFIED: ArtifactFinalizeOutcome
ARTIFACT_FINALIZE_OUTCOME_ADOPTED: ArtifactFinalizeOutcome
ARTIFACT_FINALIZE_OUTCOME_ABANDONED: ArtifactFinalizeOutcome

class PreparePackageSetRequest(_message.Message):
    __slots__ = ("download_delegation", "files", "environment_root")
    DOWNLOAD_DELEGATION_FIELD_NUMBER: _ClassVar[int]
    FILES_FIELD_NUMBER: _ClassVar[int]
    ENVIRONMENT_ROOT_FIELD_NUMBER: _ClassVar[int]
    download_delegation: bytes
    files: _containers.RepeatedCompositeFieldContainer[LocalDownloadFile]
    environment_root: str
    def __init__(self, download_delegation: _Optional[bytes] = ..., files: _Optional[_Iterable[_Union[LocalDownloadFile, _Mapping]]] = ..., environment_root: _Optional[str] = ...) -> None: ...

class LocalDownloadFile(_message.Message):
    __slots__ = ("digest", "filename", "kind", "length", "path")
    DIGEST_FIELD_NUMBER: _ClassVar[int]
    FILENAME_FIELD_NUMBER: _ClassVar[int]
    KIND_FIELD_NUMBER: _ClassVar[int]
    LENGTH_FIELD_NUMBER: _ClassVar[int]
    PATH_FIELD_NUMBER: _ClassVar[int]
    digest: bytes
    filename: str
    kind: LocalDownloadKind
    length: int
    path: str
    def __init__(self, digest: _Optional[bytes] = ..., filename: _Optional[str] = ..., kind: _Optional[_Union[LocalDownloadKind, str]] = ..., length: _Optional[int] = ..., path: _Optional[str] = ...) -> None: ...

class PreparePackageSetResult(_message.Message):
    __slots__ = ("placement_set",)
    PLACEMENT_SET_FIELD_NUMBER: _ClassVar[int]
    placement_set: DesiredPlacementSet
    def __init__(self, placement_set: _Optional[_Union[DesiredPlacementSet, _Mapping]] = ...) -> None: ...

class RecordOwnerFrame(_message.Message):
    __slots__ = ("claim", "desired_state", "attempt_offer", "cancel_attempt", "outcome_ack", "checkpoint_receipt", "snapshot_ack", "artifact_finalize_request", "artifact_host_ack", "artifact_transfer_request", "artifact_read_request")
    CLAIM_FIELD_NUMBER: _ClassVar[int]
    DESIRED_STATE_FIELD_NUMBER: _ClassVar[int]
    ATTEMPT_OFFER_FIELD_NUMBER: _ClassVar[int]
    CANCEL_ATTEMPT_FIELD_NUMBER: _ClassVar[int]
    OUTCOME_ACK_FIELD_NUMBER: _ClassVar[int]
    CHECKPOINT_RECEIPT_FIELD_NUMBER: _ClassVar[int]
    SNAPSHOT_ACK_FIELD_NUMBER: _ClassVar[int]
    ARTIFACT_FINALIZE_REQUEST_FIELD_NUMBER: _ClassVar[int]
    ARTIFACT_HOST_ACK_FIELD_NUMBER: _ClassVar[int]
    ARTIFACT_TRANSFER_REQUEST_FIELD_NUMBER: _ClassVar[int]
    ARTIFACT_READ_REQUEST_FIELD_NUMBER: _ClassVar[int]
    claim: Claim
    desired_state: DesiredWorkerState
    attempt_offer: AttemptOffer
    cancel_attempt: CancelAttempt
    outcome_ack: AttemptOutcomeAck
    checkpoint_receipt: JobCheckpointReceipt
    snapshot_ack: SnapshotAck
    artifact_finalize_request: ArtifactFinalizeRequest
    artifact_host_ack: ArtifactHostAck
    artifact_transfer_request: ArtifactTransferRequest
    artifact_read_request: ArtifactReadRequest
    def __init__(self, claim: _Optional[_Union[Claim, _Mapping]] = ..., desired_state: _Optional[_Union[DesiredWorkerState, _Mapping]] = ..., attempt_offer: _Optional[_Union[AttemptOffer, _Mapping]] = ..., cancel_attempt: _Optional[_Union[CancelAttempt, _Mapping]] = ..., outcome_ack: _Optional[_Union[AttemptOutcomeAck, _Mapping]] = ..., checkpoint_receipt: _Optional[_Union[JobCheckpointReceipt, _Mapping]] = ..., snapshot_ack: _Optional[_Union[SnapshotAck, _Mapping]] = ..., artifact_finalize_request: _Optional[_Union[ArtifactFinalizeRequest, _Mapping]] = ..., artifact_host_ack: _Optional[_Union[ArtifactHostAck, _Mapping]] = ..., artifact_transfer_request: _Optional[_Union[ArtifactTransferRequest, _Mapping]] = ..., artifact_read_request: _Optional[_Union[ArtifactReadRequest, _Mapping]] = ...) -> None: ...

class WorkerFrame(_message.Message):
    __slots__ = ("claim_ack", "observed_state", "attempt_accepted", "attempt_outcome", "boot_failure", "checkpoint_request", "checkpoint_ack", "snapshot", "artifact_finalize_result", "artifact_intent", "artifact_receipt", "artifact_read_result", "artifact_transfer_status")
    CLAIM_ACK_FIELD_NUMBER: _ClassVar[int]
    OBSERVED_STATE_FIELD_NUMBER: _ClassVar[int]
    ATTEMPT_ACCEPTED_FIELD_NUMBER: _ClassVar[int]
    ATTEMPT_OUTCOME_FIELD_NUMBER: _ClassVar[int]
    BOOT_FAILURE_FIELD_NUMBER: _ClassVar[int]
    CHECKPOINT_REQUEST_FIELD_NUMBER: _ClassVar[int]
    CHECKPOINT_ACK_FIELD_NUMBER: _ClassVar[int]
    SNAPSHOT_FIELD_NUMBER: _ClassVar[int]
    ARTIFACT_FINALIZE_RESULT_FIELD_NUMBER: _ClassVar[int]
    ARTIFACT_INTENT_FIELD_NUMBER: _ClassVar[int]
    ARTIFACT_RECEIPT_FIELD_NUMBER: _ClassVar[int]
    ARTIFACT_READ_RESULT_FIELD_NUMBER: _ClassVar[int]
    ARTIFACT_TRANSFER_STATUS_FIELD_NUMBER: _ClassVar[int]
    claim_ack: ClaimAck
    observed_state: ObservedWorkerState
    attempt_accepted: AttemptAccepted
    attempt_outcome: AttemptOutcome
    boot_failure: BootFailure
    checkpoint_request: JobCheckpointRequest
    checkpoint_ack: JobCheckpointAck
    snapshot: WorkerSnapshot
    artifact_finalize_result: ArtifactFinalizeResult
    artifact_intent: ArtifactIntentFrame
    artifact_receipt: ArtifactReceiptFrame
    artifact_read_result: ArtifactReadResult
    artifact_transfer_status: ArtifactTransferStatus
    def __init__(self, claim_ack: _Optional[_Union[ClaimAck, _Mapping]] = ..., observed_state: _Optional[_Union[ObservedWorkerState, _Mapping]] = ..., attempt_accepted: _Optional[_Union[AttemptAccepted, _Mapping]] = ..., attempt_outcome: _Optional[_Union[AttemptOutcome, _Mapping]] = ..., boot_failure: _Optional[_Union[BootFailure, _Mapping]] = ..., checkpoint_request: _Optional[_Union[JobCheckpointRequest, _Mapping]] = ..., checkpoint_ack: _Optional[_Union[JobCheckpointAck, _Mapping]] = ..., snapshot: _Optional[_Union[WorkerSnapshot, _Mapping]] = ..., artifact_finalize_result: _Optional[_Union[ArtifactFinalizeResult, _Mapping]] = ..., artifact_intent: _Optional[_Union[ArtifactIntentFrame, _Mapping]] = ..., artifact_receipt: _Optional[_Union[ArtifactReceiptFrame, _Mapping]] = ..., artifact_read_result: _Optional[_Union[ArtifactReadResult, _Mapping]] = ..., artifact_transfer_status: _Optional[_Union[ArtifactTransferStatus, _Mapping]] = ...) -> None: ...

class Claim(_message.Message):
    __slots__ = ("record_owner_epoch", "control_stream_generation", "worker_boot_id", "record_owner_id", "worker_id", "wire_minor", "proof")
    RECORD_OWNER_EPOCH_FIELD_NUMBER: _ClassVar[int]
    CONTROL_STREAM_GENERATION_FIELD_NUMBER: _ClassVar[int]
    WORKER_BOOT_ID_FIELD_NUMBER: _ClassVar[int]
    RECORD_OWNER_ID_FIELD_NUMBER: _ClassVar[int]
    WORKER_ID_FIELD_NUMBER: _ClassVar[int]
    WIRE_MINOR_FIELD_NUMBER: _ClassVar[int]
    PROOF_FIELD_NUMBER: _ClassVar[int]
    record_owner_epoch: int
    control_stream_generation: int
    worker_boot_id: str
    record_owner_id: str
    worker_id: str
    wire_minor: int
    proof: bytes
    def __init__(self, record_owner_epoch: _Optional[int] = ..., control_stream_generation: _Optional[int] = ..., worker_boot_id: _Optional[str] = ..., record_owner_id: _Optional[str] = ..., worker_id: _Optional[str] = ..., wire_minor: _Optional[int] = ..., proof: _Optional[bytes] = ...) -> None: ...

class ClaimProof(_message.Message):
    __slots__ = ("record_owner_epoch", "worker_boot_id", "worker_id", "worker_tls_certificate_digest")
    RECORD_OWNER_EPOCH_FIELD_NUMBER: _ClassVar[int]
    WORKER_BOOT_ID_FIELD_NUMBER: _ClassVar[int]
    WORKER_ID_FIELD_NUMBER: _ClassVar[int]
    WORKER_TLS_CERTIFICATE_DIGEST_FIELD_NUMBER: _ClassVar[int]
    record_owner_epoch: int
    worker_boot_id: str
    worker_id: str
    worker_tls_certificate_digest: bytes
    def __init__(self, record_owner_epoch: _Optional[int] = ..., worker_boot_id: _Optional[str] = ..., worker_id: _Optional[str] = ..., worker_tls_certificate_digest: _Optional[bytes] = ...) -> None: ...

class ClaimAck(_message.Message):
    __slots__ = ("record_owner_epoch", "control_stream_generation", "worker_boot_id", "accepted", "rejection", "wire_minor", "worker_id", "worker_instance_id", "worker_release_id", "control_runtime_digest", "git_commit", "resources")
    RECORD_OWNER_EPOCH_FIELD_NUMBER: _ClassVar[int]
    CONTROL_STREAM_GENERATION_FIELD_NUMBER: _ClassVar[int]
    WORKER_BOOT_ID_FIELD_NUMBER: _ClassVar[int]
    ACCEPTED_FIELD_NUMBER: _ClassVar[int]
    REJECTION_FIELD_NUMBER: _ClassVar[int]
    WIRE_MINOR_FIELD_NUMBER: _ClassVar[int]
    WORKER_ID_FIELD_NUMBER: _ClassVar[int]
    WORKER_INSTANCE_ID_FIELD_NUMBER: _ClassVar[int]
    WORKER_RELEASE_ID_FIELD_NUMBER: _ClassVar[int]
    CONTROL_RUNTIME_DIGEST_FIELD_NUMBER: _ClassVar[int]
    GIT_COMMIT_FIELD_NUMBER: _ClassVar[int]
    RESOURCES_FIELD_NUMBER: _ClassVar[int]
    record_owner_epoch: int
    control_stream_generation: int
    worker_boot_id: str
    accepted: bool
    rejection: ClaimRejection
    wire_minor: int
    worker_id: str
    worker_instance_id: str
    worker_release_id: str
    control_runtime_digest: str
    git_commit: str
    resources: WorkerResources
    def __init__(self, record_owner_epoch: _Optional[int] = ..., control_stream_generation: _Optional[int] = ..., worker_boot_id: _Optional[str] = ..., accepted: bool = ..., rejection: _Optional[_Union[ClaimRejection, str]] = ..., wire_minor: _Optional[int] = ..., worker_id: _Optional[str] = ..., worker_instance_id: _Optional[str] = ..., worker_release_id: _Optional[str] = ..., control_runtime_digest: _Optional[str] = ..., git_commit: _Optional[str] = ..., resources: _Optional[_Union[WorkerResources, _Mapping]] = ...) -> None: ...

class BootFailure(_message.Message):
    __slots__ = ("record_owner_epoch", "control_stream_generation", "worker_boot_id", "worker_id", "worker_instance_id", "reason", "detail", "resources", "control_runtime_digest", "worker_release_id")
    RECORD_OWNER_EPOCH_FIELD_NUMBER: _ClassVar[int]
    CONTROL_STREAM_GENERATION_FIELD_NUMBER: _ClassVar[int]
    WORKER_BOOT_ID_FIELD_NUMBER: _ClassVar[int]
    WORKER_ID_FIELD_NUMBER: _ClassVar[int]
    WORKER_INSTANCE_ID_FIELD_NUMBER: _ClassVar[int]
    REASON_FIELD_NUMBER: _ClassVar[int]
    DETAIL_FIELD_NUMBER: _ClassVar[int]
    RESOURCES_FIELD_NUMBER: _ClassVar[int]
    CONTROL_RUNTIME_DIGEST_FIELD_NUMBER: _ClassVar[int]
    WORKER_RELEASE_ID_FIELD_NUMBER: _ClassVar[int]
    record_owner_epoch: int
    control_stream_generation: int
    worker_boot_id: str
    worker_id: str
    worker_instance_id: str
    reason: BootFailureReason
    detail: str
    resources: WorkerResources
    control_runtime_digest: str
    worker_release_id: str
    def __init__(self, record_owner_epoch: _Optional[int] = ..., control_stream_generation: _Optional[int] = ..., worker_boot_id: _Optional[str] = ..., worker_id: _Optional[str] = ..., worker_instance_id: _Optional[str] = ..., reason: _Optional[_Union[BootFailureReason, str]] = ..., detail: _Optional[str] = ..., resources: _Optional[_Union[WorkerResources, _Mapping]] = ..., control_runtime_digest: _Optional[str] = ..., worker_release_id: _Optional[str] = ...) -> None: ...

class WorkerSnapshot(_message.Message):
    __slots__ = ("record_owner_epoch", "control_stream_generation", "worker_boot_id", "snapshot_id", "snapshot_digest", "snapshot_canonical_bytes", "accepted_placement_set_canonical_bytes")
    RECORD_OWNER_EPOCH_FIELD_NUMBER: _ClassVar[int]
    CONTROL_STREAM_GENERATION_FIELD_NUMBER: _ClassVar[int]
    WORKER_BOOT_ID_FIELD_NUMBER: _ClassVar[int]
    SNAPSHOT_ID_FIELD_NUMBER: _ClassVar[int]
    SNAPSHOT_DIGEST_FIELD_NUMBER: _ClassVar[int]
    SNAPSHOT_CANONICAL_BYTES_FIELD_NUMBER: _ClassVar[int]
    ACCEPTED_PLACEMENT_SET_CANONICAL_BYTES_FIELD_NUMBER: _ClassVar[int]
    record_owner_epoch: int
    control_stream_generation: int
    worker_boot_id: str
    snapshot_id: str
    snapshot_digest: bytes
    snapshot_canonical_bytes: bytes
    accepted_placement_set_canonical_bytes: bytes
    def __init__(self, record_owner_epoch: _Optional[int] = ..., control_stream_generation: _Optional[int] = ..., worker_boot_id: _Optional[str] = ..., snapshot_id: _Optional[str] = ..., snapshot_digest: _Optional[bytes] = ..., snapshot_canonical_bytes: _Optional[bytes] = ..., accepted_placement_set_canonical_bytes: _Optional[bytes] = ...) -> None: ...

class WorkerSnapshotBody(_message.Message):
    __slots__ = ("accepted_desired_state_revision", "accepted_placement_set_digest", "journal_highwater", "worker_phase", "placements", "converged_revision", "admission_generation", "admission_state", "available_attempt_slots", "held_attempts", "artifact_transactions")
    ACCEPTED_DESIRED_STATE_REVISION_FIELD_NUMBER: _ClassVar[int]
    ACCEPTED_PLACEMENT_SET_DIGEST_FIELD_NUMBER: _ClassVar[int]
    JOURNAL_HIGHWATER_FIELD_NUMBER: _ClassVar[int]
    WORKER_PHASE_FIELD_NUMBER: _ClassVar[int]
    PLACEMENTS_FIELD_NUMBER: _ClassVar[int]
    CONVERGED_REVISION_FIELD_NUMBER: _ClassVar[int]
    ADMISSION_GENERATION_FIELD_NUMBER: _ClassVar[int]
    ADMISSION_STATE_FIELD_NUMBER: _ClassVar[int]
    AVAILABLE_ATTEMPT_SLOTS_FIELD_NUMBER: _ClassVar[int]
    HELD_ATTEMPTS_FIELD_NUMBER: _ClassVar[int]
    ARTIFACT_TRANSACTIONS_FIELD_NUMBER: _ClassVar[int]
    accepted_desired_state_revision: int
    accepted_placement_set_digest: bytes
    journal_highwater: int
    worker_phase: WorkerPhase
    placements: _containers.RepeatedCompositeFieldContainer[PlacementStatus]
    converged_revision: int
    admission_generation: int
    admission_state: AdmissionState
    available_attempt_slots: int
    held_attempts: _containers.RepeatedCompositeFieldContainer[HeldAttempt]
    artifact_transactions: _containers.RepeatedCompositeFieldContainer[ArtifactTransactionStatus]
    def __init__(self, accepted_desired_state_revision: _Optional[int] = ..., accepted_placement_set_digest: _Optional[bytes] = ..., journal_highwater: _Optional[int] = ..., worker_phase: _Optional[_Union[WorkerPhase, str]] = ..., placements: _Optional[_Iterable[_Union[PlacementStatus, _Mapping]]] = ..., converged_revision: _Optional[int] = ..., admission_generation: _Optional[int] = ..., admission_state: _Optional[_Union[AdmissionState, str]] = ..., available_attempt_slots: _Optional[int] = ..., held_attempts: _Optional[_Iterable[_Union[HeldAttempt, _Mapping]]] = ..., artifact_transactions: _Optional[_Iterable[_Union[ArtifactTransactionStatus, _Mapping]]] = ...) -> None: ...

class SnapshotAck(_message.Message):
    __slots__ = ("record_owner_epoch", "control_stream_generation", "worker_boot_id", "snapshot_id", "snapshot_digest")
    RECORD_OWNER_EPOCH_FIELD_NUMBER: _ClassVar[int]
    CONTROL_STREAM_GENERATION_FIELD_NUMBER: _ClassVar[int]
    WORKER_BOOT_ID_FIELD_NUMBER: _ClassVar[int]
    SNAPSHOT_ID_FIELD_NUMBER: _ClassVar[int]
    SNAPSHOT_DIGEST_FIELD_NUMBER: _ClassVar[int]
    record_owner_epoch: int
    control_stream_generation: int
    worker_boot_id: str
    snapshot_id: str
    snapshot_digest: bytes
    def __init__(self, record_owner_epoch: _Optional[int] = ..., control_stream_generation: _Optional[int] = ..., worker_boot_id: _Optional[str] = ..., snapshot_id: _Optional[str] = ..., snapshot_digest: _Optional[bytes] = ...) -> None: ...

class DesiredWorkerState(_message.Message):
    __slots__ = ("record_owner_epoch", "control_stream_generation", "worker_boot_id", "revision", "posture", "wire_minor", "drain_grace_ms", "job", "placement_set", "package_set")
    RECORD_OWNER_EPOCH_FIELD_NUMBER: _ClassVar[int]
    CONTROL_STREAM_GENERATION_FIELD_NUMBER: _ClassVar[int]
    WORKER_BOOT_ID_FIELD_NUMBER: _ClassVar[int]
    REVISION_FIELD_NUMBER: _ClassVar[int]
    POSTURE_FIELD_NUMBER: _ClassVar[int]
    WIRE_MINOR_FIELD_NUMBER: _ClassVar[int]
    DRAIN_GRACE_MS_FIELD_NUMBER: _ClassVar[int]
    JOB_FIELD_NUMBER: _ClassVar[int]
    PLACEMENT_SET_FIELD_NUMBER: _ClassVar[int]
    PACKAGE_SET_FIELD_NUMBER: _ClassVar[int]
    record_owner_epoch: int
    control_stream_generation: int
    worker_boot_id: str
    revision: int
    posture: Posture
    wire_minor: int
    drain_grace_ms: int
    job: JobDirective
    placement_set: DesiredPlacementSet
    package_set: DesiredPackageSet
    def __init__(self, record_owner_epoch: _Optional[int] = ..., control_stream_generation: _Optional[int] = ..., worker_boot_id: _Optional[str] = ..., revision: _Optional[int] = ..., posture: _Optional[_Union[Posture, str]] = ..., wire_minor: _Optional[int] = ..., drain_grace_ms: _Optional[int] = ..., job: _Optional[_Union[JobDirective, _Mapping]] = ..., placement_set: _Optional[_Union[DesiredPlacementSet, _Mapping]] = ..., package_set: _Optional[_Union[DesiredPackageSet, _Mapping]] = ...) -> None: ...

class DesiredPackageSet(_message.Message):
    __slots__ = ("download_delegation", "download_delegation_signature")
    DOWNLOAD_DELEGATION_FIELD_NUMBER: _ClassVar[int]
    DOWNLOAD_DELEGATION_SIGNATURE_FIELD_NUMBER: _ClassVar[int]
    download_delegation: bytes
    download_delegation_signature: bytes
    def __init__(self, download_delegation: _Optional[bytes] = ..., download_delegation_signature: _Optional[bytes] = ...) -> None: ...

class DesiredPlacementSet(_message.Message):
    __slots__ = ("placement_set_digest", "placement_set_canonical_bytes")
    PLACEMENT_SET_DIGEST_FIELD_NUMBER: _ClassVar[int]
    PLACEMENT_SET_CANONICAL_BYTES_FIELD_NUMBER: _ClassVar[int]
    placement_set_digest: bytes
    placement_set_canonical_bytes: bytes
    def __init__(self, placement_set_digest: _Optional[bytes] = ..., placement_set_canonical_bytes: _Optional[bytes] = ...) -> None: ...

class PlacementSet(_message.Message):
    __slots__ = ("placements",)
    PLACEMENTS_FIELD_NUMBER: _ClassVar[int]
    placements: _containers.RepeatedCompositeFieldContainer[Placement]
    def __init__(self, placements: _Optional[_Iterable[_Union[Placement, _Mapping]]] = ...) -> None: ...

class DownloadDelegation(_message.Message):
    __slots__ = ("expires_at_unix", "models", "packages", "rental_id", "worker_boot_id", "worker_id", "worker_tls_certificate_digest")
    EXPIRES_AT_UNIX_FIELD_NUMBER: _ClassVar[int]
    MODELS_FIELD_NUMBER: _ClassVar[int]
    PACKAGES_FIELD_NUMBER: _ClassVar[int]
    RENTAL_ID_FIELD_NUMBER: _ClassVar[int]
    WORKER_BOOT_ID_FIELD_NUMBER: _ClassVar[int]
    WORKER_ID_FIELD_NUMBER: _ClassVar[int]
    WORKER_TLS_CERTIFICATE_DIGEST_FIELD_NUMBER: _ClassVar[int]
    expires_at_unix: int
    models: _containers.RepeatedCompositeFieldContainer[DownloadModelRef]
    packages: _containers.RepeatedCompositeFieldContainer[DownloadPackageRef]
    rental_id: str
    worker_boot_id: str
    worker_id: str
    worker_tls_certificate_digest: bytes
    def __init__(self, expires_at_unix: _Optional[int] = ..., models: _Optional[_Iterable[_Union[DownloadModelRef, _Mapping]]] = ..., packages: _Optional[_Iterable[_Union[DownloadPackageRef, _Mapping]]] = ..., rental_id: _Optional[str] = ..., worker_boot_id: _Optional[str] = ..., worker_id: _Optional[str] = ..., worker_tls_certificate_digest: _Optional[bytes] = ...) -> None: ...

class DownloadModelRef(_message.Message):
    __slots__ = ("manifest", "model", "release", "package", "slot")
    MANIFEST_FIELD_NUMBER: _ClassVar[int]
    MODEL_FIELD_NUMBER: _ClassVar[int]
    RELEASE_FIELD_NUMBER: _ClassVar[int]
    PACKAGE_FIELD_NUMBER: _ClassVar[int]
    SLOT_FIELD_NUMBER: _ClassVar[int]
    manifest: str
    model: str
    release: str
    package: str
    slot: str
    def __init__(self, manifest: _Optional[str] = ..., model: _Optional[str] = ..., release: _Optional[str] = ..., package: _Optional[str] = ..., slot: _Optional[str] = ...) -> None: ...

class DownloadPackageRef(_message.Message):
    __slots__ = ("package", "release", "release_digest")
    PACKAGE_FIELD_NUMBER: _ClassVar[int]
    RELEASE_FIELD_NUMBER: _ClassVar[int]
    RELEASE_DIGEST_FIELD_NUMBER: _ClassVar[int]
    package: str
    release: str
    release_digest: str
    def __init__(self, package: _Optional[str] = ..., release: _Optional[str] = ..., release_digest: _Optional[str] = ...) -> None: ...

class Placement(_message.Message):
    __slots__ = ("placement_id", "package", "development", "environment_digest", "environment_receipt_digest", "package_descriptor", "bindings_digest", "models", "entrypoints", "qualification", "environment")
    PLACEMENT_ID_FIELD_NUMBER: _ClassVar[int]
    PACKAGE_FIELD_NUMBER: _ClassVar[int]
    DEVELOPMENT_FIELD_NUMBER: _ClassVar[int]
    ENVIRONMENT_DIGEST_FIELD_NUMBER: _ClassVar[int]
    ENVIRONMENT_RECEIPT_DIGEST_FIELD_NUMBER: _ClassVar[int]
    PACKAGE_DESCRIPTOR_FIELD_NUMBER: _ClassVar[int]
    BINDINGS_DIGEST_FIELD_NUMBER: _ClassVar[int]
    MODELS_FIELD_NUMBER: _ClassVar[int]
    ENTRYPOINTS_FIELD_NUMBER: _ClassVar[int]
    QUALIFICATION_FIELD_NUMBER: _ClassVar[int]
    ENVIRONMENT_FIELD_NUMBER: _ClassVar[int]
    placement_id: str
    package: PackageSelection
    development: DevelopmentPackage
    environment_digest: bytes
    environment_receipt_digest: bytes
    package_descriptor: Ref
    bindings_digest: bytes
    models: _containers.RepeatedCompositeFieldContainer[Model]
    entrypoints: _containers.RepeatedCompositeFieldContainer[Entrypoint]
    qualification: Ref
    environment: Environment
    def __init__(self, placement_id: _Optional[str] = ..., package: _Optional[_Union[PackageSelection, _Mapping]] = ..., development: _Optional[_Union[DevelopmentPackage, _Mapping]] = ..., environment_digest: _Optional[bytes] = ..., environment_receipt_digest: _Optional[bytes] = ..., package_descriptor: _Optional[_Union[Ref, _Mapping]] = ..., bindings_digest: _Optional[bytes] = ..., models: _Optional[_Iterable[_Union[Model, _Mapping]]] = ..., entrypoints: _Optional[_Iterable[_Union[Entrypoint, _Mapping]]] = ..., qualification: _Optional[_Union[Ref, _Mapping]] = ..., environment: _Optional[_Union[Environment, _Mapping]] = ...) -> None: ...

class Ref(_message.Message):
    __slots__ = ("digest", "length")
    DIGEST_FIELD_NUMBER: _ClassVar[int]
    LENGTH_FIELD_NUMBER: _ClassVar[int]
    digest: bytes
    length: int
    def __init__(self, digest: _Optional[bytes] = ..., length: _Optional[int] = ...) -> None: ...

class WheelFact(_message.Message):
    __slots__ = ("ref", "distribution", "version", "filename", "import_roots", "tags")
    REF_FIELD_NUMBER: _ClassVar[int]
    DISTRIBUTION_FIELD_NUMBER: _ClassVar[int]
    VERSION_FIELD_NUMBER: _ClassVar[int]
    FILENAME_FIELD_NUMBER: _ClassVar[int]
    IMPORT_ROOTS_FIELD_NUMBER: _ClassVar[int]
    TAGS_FIELD_NUMBER: _ClassVar[int]
    ref: Ref
    distribution: str
    version: str
    filename: str
    import_roots: _containers.RepeatedScalarFieldContainer[str]
    tags: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, ref: _Optional[_Union[Ref, _Mapping]] = ..., distribution: _Optional[str] = ..., version: _Optional[str] = ..., filename: _Optional[str] = ..., import_roots: _Optional[_Iterable[str]] = ..., tags: _Optional[_Iterable[str]] = ...) -> None: ...

class PackageSelection(_message.Message):
    __slots__ = ("package", "release", "release_digest", "project_wheel")
    PACKAGE_FIELD_NUMBER: _ClassVar[int]
    RELEASE_FIELD_NUMBER: _ClassVar[int]
    RELEASE_DIGEST_FIELD_NUMBER: _ClassVar[int]
    PROJECT_WHEEL_FIELD_NUMBER: _ClassVar[int]
    package: str
    release: str
    release_digest: bytes
    project_wheel: WheelFact
    def __init__(self, package: _Optional[str] = ..., release: _Optional[str] = ..., release_digest: _Optional[bytes] = ..., project_wheel: _Optional[_Union[WheelFact, _Mapping]] = ...) -> None: ...

class DevelopmentPackage(_message.Message):
    __slots__ = ("package", "release", "source_digest")
    PACKAGE_FIELD_NUMBER: _ClassVar[int]
    RELEASE_FIELD_NUMBER: _ClassVar[int]
    SOURCE_DIGEST_FIELD_NUMBER: _ClassVar[int]
    package: str
    release: str
    source_digest: bytes
    def __init__(self, package: _Optional[str] = ..., release: _Optional[str] = ..., source_digest: _Optional[bytes] = ...) -> None: ...

class Environment(_message.Message):
    __slots__ = ("wheelhouse_manifest", "wheels")
    WHEELHOUSE_MANIFEST_FIELD_NUMBER: _ClassVar[int]
    WHEELS_FIELD_NUMBER: _ClassVar[int]
    wheelhouse_manifest: Ref
    wheels: _containers.RepeatedCompositeFieldContainer[WheelFact]
    def __init__(self, wheelhouse_manifest: _Optional[_Union[Ref, _Mapping]] = ..., wheels: _Optional[_Iterable[_Union[WheelFact, _Mapping]]] = ...) -> None: ...

class Model(_message.Message):
    __slots__ = ("id", "repo", "version", "lane", "manifest")
    ID_FIELD_NUMBER: _ClassVar[int]
    REPO_FIELD_NUMBER: _ClassVar[int]
    VERSION_FIELD_NUMBER: _ClassVar[int]
    LANE_FIELD_NUMBER: _ClassVar[int]
    MANIFEST_FIELD_NUMBER: _ClassVar[int]
    id: str
    repo: str
    version: str
    lane: str
    manifest: Ref
    def __init__(self, id: _Optional[str] = ..., repo: _Optional[str] = ..., version: _Optional[str] = ..., lane: _Optional[str] = ..., manifest: _Optional[_Union[Ref, _Mapping]] = ...) -> None: ...

class Entrypoint(_message.Message):
    __slots__ = ("name", "entrypoint_binding_digest", "slots")
    NAME_FIELD_NUMBER: _ClassVar[int]
    ENTRYPOINT_BINDING_DIGEST_FIELD_NUMBER: _ClassVar[int]
    SLOTS_FIELD_NUMBER: _ClassVar[int]
    name: str
    entrypoint_binding_digest: bytes
    slots: _containers.RepeatedCompositeFieldContainer[Slot]
    def __init__(self, name: _Optional[str] = ..., entrypoint_binding_digest: _Optional[bytes] = ..., slots: _Optional[_Iterable[_Union[Slot, _Mapping]]] = ...) -> None: ...

class Slot(_message.Message):
    __slots__ = ("slot", "reference_model_id", "config", "components", "model_construction_contract", "stamps")
    SLOT_FIELD_NUMBER: _ClassVar[int]
    REFERENCE_MODEL_ID_FIELD_NUMBER: _ClassVar[int]
    CONFIG_FIELD_NUMBER: _ClassVar[int]
    COMPONENTS_FIELD_NUMBER: _ClassVar[int]
    MODEL_CONSTRUCTION_CONTRACT_FIELD_NUMBER: _ClassVar[int]
    STAMPS_FIELD_NUMBER: _ClassVar[int]
    slot: str
    reference_model_id: str
    config: Config
    components: _containers.RepeatedCompositeFieldContainer[Component]
    model_construction_contract: Ref
    stamps: _containers.RepeatedCompositeFieldContainer[Stamp]
    def __init__(self, slot: _Optional[str] = ..., reference_model_id: _Optional[str] = ..., config: _Optional[_Union[Config, _Mapping]] = ..., components: _Optional[_Iterable[_Union[Component, _Mapping]]] = ..., model_construction_contract: _Optional[_Union[Ref, _Mapping]] = ..., stamps: _Optional[_Iterable[_Union[Stamp, _Mapping]]] = ...) -> None: ...

class Config(_message.Message):
    __slots__ = ("document", "assets")
    DOCUMENT_FIELD_NUMBER: _ClassVar[int]
    ASSETS_FIELD_NUMBER: _ClassVar[int]
    document: Ref
    assets: _containers.RepeatedCompositeFieldContainer[Asset]
    def __init__(self, document: _Optional[_Union[Ref, _Mapping]] = ..., assets: _Optional[_Iterable[_Union[Asset, _Mapping]]] = ...) -> None: ...

class Asset(_message.Message):
    __slots__ = ("name", "model_id", "ref")
    NAME_FIELD_NUMBER: _ClassVar[int]
    MODEL_ID_FIELD_NUMBER: _ClassVar[int]
    REF_FIELD_NUMBER: _ClassVar[int]
    name: str
    model_id: str
    ref: Ref
    def __init__(self, name: _Optional[str] = ..., model_id: _Optional[str] = ..., ref: _Optional[_Union[Ref, _Mapping]] = ...) -> None: ...

class Component(_message.Message):
    __slots__ = ("component", "model_id")
    COMPONENT_FIELD_NUMBER: _ClassVar[int]
    MODEL_ID_FIELD_NUMBER: _ClassVar[int]
    component: str
    model_id: str
    def __init__(self, component: _Optional[str] = ..., model_id: _Optional[str] = ...) -> None: ...

class Stamp(_message.Message):
    __slots__ = ("component", "key", "values")
    COMPONENT_FIELD_NUMBER: _ClassVar[int]
    KEY_FIELD_NUMBER: _ClassVar[int]
    VALUES_FIELD_NUMBER: _ClassVar[int]
    component: str
    key: str
    values: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, component: _Optional[str] = ..., key: _Optional[str] = ..., values: _Optional[_Iterable[str]] = ...) -> None: ...

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

class ObservedWorkerState(_message.Message):
    __slots__ = ("record_owner_epoch", "control_stream_generation", "worker_boot_id", "applied_wire_minor", "held_attempts", "faults", "activity", "job_capacity", "placements", "admission_generation", "admission_state", "available_attempt_slots", "accepted_desired_state_revision", "accepted_placement_set_digest", "converged_revision", "worker_phase")
    RECORD_OWNER_EPOCH_FIELD_NUMBER: _ClassVar[int]
    CONTROL_STREAM_GENERATION_FIELD_NUMBER: _ClassVar[int]
    WORKER_BOOT_ID_FIELD_NUMBER: _ClassVar[int]
    APPLIED_WIRE_MINOR_FIELD_NUMBER: _ClassVar[int]
    HELD_ATTEMPTS_FIELD_NUMBER: _ClassVar[int]
    FAULTS_FIELD_NUMBER: _ClassVar[int]
    ACTIVITY_FIELD_NUMBER: _ClassVar[int]
    JOB_CAPACITY_FIELD_NUMBER: _ClassVar[int]
    PLACEMENTS_FIELD_NUMBER: _ClassVar[int]
    ADMISSION_GENERATION_FIELD_NUMBER: _ClassVar[int]
    ADMISSION_STATE_FIELD_NUMBER: _ClassVar[int]
    AVAILABLE_ATTEMPT_SLOTS_FIELD_NUMBER: _ClassVar[int]
    ACCEPTED_DESIRED_STATE_REVISION_FIELD_NUMBER: _ClassVar[int]
    ACCEPTED_PLACEMENT_SET_DIGEST_FIELD_NUMBER: _ClassVar[int]
    CONVERGED_REVISION_FIELD_NUMBER: _ClassVar[int]
    WORKER_PHASE_FIELD_NUMBER: _ClassVar[int]
    record_owner_epoch: int
    control_stream_generation: int
    worker_boot_id: str
    applied_wire_minor: int
    held_attempts: _containers.RepeatedCompositeFieldContainer[HeldAttempt]
    faults: _containers.RepeatedCompositeFieldContainer[Fault]
    activity: _containers.RepeatedCompositeFieldContainer[ActivityEvent]
    job_capacity: JobCapacity
    placements: _containers.RepeatedCompositeFieldContainer[PlacementStatus]
    admission_generation: int
    admission_state: AdmissionState
    available_attempt_slots: int
    accepted_desired_state_revision: int
    accepted_placement_set_digest: bytes
    converged_revision: int
    worker_phase: WorkerPhase
    def __init__(self, record_owner_epoch: _Optional[int] = ..., control_stream_generation: _Optional[int] = ..., worker_boot_id: _Optional[str] = ..., applied_wire_minor: _Optional[int] = ..., held_attempts: _Optional[_Iterable[_Union[HeldAttempt, _Mapping]]] = ..., faults: _Optional[_Iterable[_Union[Fault, _Mapping]]] = ..., activity: _Optional[_Iterable[_Union[ActivityEvent, _Mapping]]] = ..., job_capacity: _Optional[_Union[JobCapacity, _Mapping]] = ..., placements: _Optional[_Iterable[_Union[PlacementStatus, _Mapping]]] = ..., admission_generation: _Optional[int] = ..., admission_state: _Optional[_Union[AdmissionState, str]] = ..., available_attempt_slots: _Optional[int] = ..., accepted_desired_state_revision: _Optional[int] = ..., accepted_placement_set_digest: _Optional[bytes] = ..., converged_revision: _Optional[int] = ..., worker_phase: _Optional[_Union[WorkerPhase, str]] = ...) -> None: ...

class PlacementStatus(_message.Message):
    __slots__ = ("placement_id", "executor_generation", "dispatchable_binding_digests", "materializable_binding_digests", "faults", "accelerator", "placement_set_digest", "materialization", "serving", "retained_fallback_placement_set_digest", "acquisition", "package_revision_digest", "environment_digest", "config_digest")
    PLACEMENT_ID_FIELD_NUMBER: _ClassVar[int]
    EXECUTOR_GENERATION_FIELD_NUMBER: _ClassVar[int]
    DISPATCHABLE_BINDING_DIGESTS_FIELD_NUMBER: _ClassVar[int]
    MATERIALIZABLE_BINDING_DIGESTS_FIELD_NUMBER: _ClassVar[int]
    FAULTS_FIELD_NUMBER: _ClassVar[int]
    ACCELERATOR_FIELD_NUMBER: _ClassVar[int]
    PLACEMENT_SET_DIGEST_FIELD_NUMBER: _ClassVar[int]
    MATERIALIZATION_FIELD_NUMBER: _ClassVar[int]
    SERVING_FIELD_NUMBER: _ClassVar[int]
    RETAINED_FALLBACK_PLACEMENT_SET_DIGEST_FIELD_NUMBER: _ClassVar[int]
    ACQUISITION_FIELD_NUMBER: _ClassVar[int]
    PACKAGE_REVISION_DIGEST_FIELD_NUMBER: _ClassVar[int]
    ENVIRONMENT_DIGEST_FIELD_NUMBER: _ClassVar[int]
    CONFIG_DIGEST_FIELD_NUMBER: _ClassVar[int]
    placement_id: str
    executor_generation: int
    dispatchable_binding_digests: _containers.RepeatedScalarFieldContainer[bytes]
    materializable_binding_digests: _containers.RepeatedScalarFieldContainer[bytes]
    faults: _containers.RepeatedCompositeFieldContainer[Fault]
    accelerator: AcceleratorQualification
    placement_set_digest: bytes
    materialization: MaterializationState
    serving: ServingState
    retained_fallback_placement_set_digest: bytes
    acquisition: PlacementAcquisitionObservation
    package_revision_digest: str
    environment_digest: str
    config_digest: str
    def __init__(self, placement_id: _Optional[str] = ..., executor_generation: _Optional[int] = ..., dispatchable_binding_digests: _Optional[_Iterable[bytes]] = ..., materializable_binding_digests: _Optional[_Iterable[bytes]] = ..., faults: _Optional[_Iterable[_Union[Fault, _Mapping]]] = ..., accelerator: _Optional[_Union[AcceleratorQualification, _Mapping]] = ..., placement_set_digest: _Optional[bytes] = ..., materialization: _Optional[_Union[MaterializationState, str]] = ..., serving: _Optional[_Union[ServingState, str]] = ..., retained_fallback_placement_set_digest: _Optional[bytes] = ..., acquisition: _Optional[_Union[PlacementAcquisitionObservation, _Mapping]] = ..., package_revision_digest: _Optional[str] = ..., environment_digest: _Optional[str] = ..., config_digest: _Optional[str] = ...) -> None: ...

class PlacementAcquisitionObservation(_message.Message):
    __slots__ = ("package", "model")
    PACKAGE_FIELD_NUMBER: _ClassVar[int]
    MODEL_FIELD_NUMBER: _ClassVar[int]
    package: AcquisitionLegObservation
    model: AcquisitionLegObservation
    def __init__(self, package: _Optional[_Union[AcquisitionLegObservation, _Mapping]] = ..., model: _Optional[_Union[AcquisitionLegObservation, _Mapping]] = ...) -> None: ...

class AcquisitionLegObservation(_message.Message):
    __slots__ = ("started_monotonic_ns", "ended_monotonic_ns", "downloaded_bytes", "reused_bytes")
    STARTED_MONOTONIC_NS_FIELD_NUMBER: _ClassVar[int]
    ENDED_MONOTONIC_NS_FIELD_NUMBER: _ClassVar[int]
    DOWNLOADED_BYTES_FIELD_NUMBER: _ClassVar[int]
    REUSED_BYTES_FIELD_NUMBER: _ClassVar[int]
    started_monotonic_ns: int
    ended_monotonic_ns: int
    downloaded_bytes: int
    reused_bytes: int
    def __init__(self, started_monotonic_ns: _Optional[int] = ..., ended_monotonic_ns: _Optional[int] = ..., downloaded_bytes: _Optional[int] = ..., reused_bytes: _Optional[int] = ...) -> None: ...

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

class AttemptOffer(_message.Message):
    __slots__ = ("record_owner_epoch", "control_stream_generation", "worker_boot_id", "request_id", "attempt_ordinal", "invocation_spec_digest", "grant", "invocation_spec_canonical_bytes", "placement_id", "admission_generation")
    RECORD_OWNER_EPOCH_FIELD_NUMBER: _ClassVar[int]
    CONTROL_STREAM_GENERATION_FIELD_NUMBER: _ClassVar[int]
    WORKER_BOOT_ID_FIELD_NUMBER: _ClassVar[int]
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    ATTEMPT_ORDINAL_FIELD_NUMBER: _ClassVar[int]
    INVOCATION_SPEC_DIGEST_FIELD_NUMBER: _ClassVar[int]
    GRANT_FIELD_NUMBER: _ClassVar[int]
    INVOCATION_SPEC_CANONICAL_BYTES_FIELD_NUMBER: _ClassVar[int]
    PLACEMENT_ID_FIELD_NUMBER: _ClassVar[int]
    ADMISSION_GENERATION_FIELD_NUMBER: _ClassVar[int]
    record_owner_epoch: int
    control_stream_generation: int
    worker_boot_id: str
    request_id: str
    attempt_ordinal: int
    invocation_spec_digest: bytes
    grant: DeliveryGrant
    invocation_spec_canonical_bytes: bytes
    placement_id: str
    admission_generation: int
    def __init__(self, record_owner_epoch: _Optional[int] = ..., control_stream_generation: _Optional[int] = ..., worker_boot_id: _Optional[str] = ..., request_id: _Optional[str] = ..., attempt_ordinal: _Optional[int] = ..., invocation_spec_digest: _Optional[bytes] = ..., grant: _Optional[_Union[DeliveryGrant, _Mapping]] = ..., invocation_spec_canonical_bytes: _Optional[bytes] = ..., placement_id: _Optional[str] = ..., admission_generation: _Optional[int] = ...) -> None: ...

class AttemptAccepted(_message.Message):
    __slots__ = ("record_owner_epoch", "control_stream_generation", "worker_boot_id", "request_id", "attempt_ordinal", "invocation_spec_digest", "plan_digest", "model_construction_digest", "plan", "placement_id", "executor_generation")
    RECORD_OWNER_EPOCH_FIELD_NUMBER: _ClassVar[int]
    CONTROL_STREAM_GENERATION_FIELD_NUMBER: _ClassVar[int]
    WORKER_BOOT_ID_FIELD_NUMBER: _ClassVar[int]
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    ATTEMPT_ORDINAL_FIELD_NUMBER: _ClassVar[int]
    INVOCATION_SPEC_DIGEST_FIELD_NUMBER: _ClassVar[int]
    PLAN_DIGEST_FIELD_NUMBER: _ClassVar[int]
    MODEL_CONSTRUCTION_DIGEST_FIELD_NUMBER: _ClassVar[int]
    PLAN_FIELD_NUMBER: _ClassVar[int]
    PLACEMENT_ID_FIELD_NUMBER: _ClassVar[int]
    EXECUTOR_GENERATION_FIELD_NUMBER: _ClassVar[int]
    record_owner_epoch: int
    control_stream_generation: int
    worker_boot_id: str
    request_id: str
    attempt_ordinal: int
    invocation_spec_digest: bytes
    plan_digest: bytes
    model_construction_digest: bytes
    plan: AttemptPlanSummary
    placement_id: str
    executor_generation: int
    def __init__(self, record_owner_epoch: _Optional[int] = ..., control_stream_generation: _Optional[int] = ..., worker_boot_id: _Optional[str] = ..., request_id: _Optional[str] = ..., attempt_ordinal: _Optional[int] = ..., invocation_spec_digest: _Optional[bytes] = ..., plan_digest: _Optional[bytes] = ..., model_construction_digest: _Optional[bytes] = ..., plan: _Optional[_Union[AttemptPlanSummary, _Mapping]] = ..., placement_id: _Optional[str] = ..., executor_generation: _Optional[int] = ...) -> None: ...

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
    __slots__ = ("record_owner_epoch", "control_stream_generation", "worker_boot_id", "request_id", "attempt_ordinal", "reason", "grace_ms", "invocation_spec_digest")
    RECORD_OWNER_EPOCH_FIELD_NUMBER: _ClassVar[int]
    CONTROL_STREAM_GENERATION_FIELD_NUMBER: _ClassVar[int]
    WORKER_BOOT_ID_FIELD_NUMBER: _ClassVar[int]
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    ATTEMPT_ORDINAL_FIELD_NUMBER: _ClassVar[int]
    REASON_FIELD_NUMBER: _ClassVar[int]
    GRACE_MS_FIELD_NUMBER: _ClassVar[int]
    INVOCATION_SPEC_DIGEST_FIELD_NUMBER: _ClassVar[int]
    record_owner_epoch: int
    control_stream_generation: int
    worker_boot_id: str
    request_id: str
    attempt_ordinal: int
    reason: CancelReason
    grace_ms: int
    invocation_spec_digest: bytes
    def __init__(self, record_owner_epoch: _Optional[int] = ..., control_stream_generation: _Optional[int] = ..., worker_boot_id: _Optional[str] = ..., request_id: _Optional[str] = ..., attempt_ordinal: _Optional[int] = ..., reason: _Optional[_Union[CancelReason, str]] = ..., grace_ms: _Optional[int] = ..., invocation_spec_digest: _Optional[bytes] = ...) -> None: ...

class AttemptOutcome(_message.Message):
    __slots__ = ("record_owner_epoch", "control_stream_generation", "worker_boot_id", "request_id", "attempt_ordinal", "invocation_spec_digest", "outcome_id", "outcome_digest", "outcome_canonical_bytes", "placement_id")
    RECORD_OWNER_EPOCH_FIELD_NUMBER: _ClassVar[int]
    CONTROL_STREAM_GENERATION_FIELD_NUMBER: _ClassVar[int]
    WORKER_BOOT_ID_FIELD_NUMBER: _ClassVar[int]
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    ATTEMPT_ORDINAL_FIELD_NUMBER: _ClassVar[int]
    INVOCATION_SPEC_DIGEST_FIELD_NUMBER: _ClassVar[int]
    OUTCOME_ID_FIELD_NUMBER: _ClassVar[int]
    OUTCOME_DIGEST_FIELD_NUMBER: _ClassVar[int]
    OUTCOME_CANONICAL_BYTES_FIELD_NUMBER: _ClassVar[int]
    PLACEMENT_ID_FIELD_NUMBER: _ClassVar[int]
    record_owner_epoch: int
    control_stream_generation: int
    worker_boot_id: str
    request_id: str
    attempt_ordinal: int
    invocation_spec_digest: bytes
    outcome_id: str
    outcome_digest: bytes
    outcome_canonical_bytes: bytes
    placement_id: str
    def __init__(self, record_owner_epoch: _Optional[int] = ..., control_stream_generation: _Optional[int] = ..., worker_boot_id: _Optional[str] = ..., request_id: _Optional[str] = ..., attempt_ordinal: _Optional[int] = ..., invocation_spec_digest: _Optional[bytes] = ..., outcome_id: _Optional[str] = ..., outcome_digest: _Optional[bytes] = ..., outcome_canonical_bytes: _Optional[bytes] = ..., placement_id: _Optional[str] = ...) -> None: ...

class AttemptOutcomeBody(_message.Message):
    __slots__ = ("request_id", "attempt_ordinal", "invocation_spec_digest", "status", "output_manifest", "metrics", "triage_bundle", "safe_message", "cause", "result", "execution_started", "artifact_receipts")
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    ATTEMPT_ORDINAL_FIELD_NUMBER: _ClassVar[int]
    INVOCATION_SPEC_DIGEST_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_MANIFEST_FIELD_NUMBER: _ClassVar[int]
    METRICS_FIELD_NUMBER: _ClassVar[int]
    TRIAGE_BUNDLE_FIELD_NUMBER: _ClassVar[int]
    SAFE_MESSAGE_FIELD_NUMBER: _ClassVar[int]
    CAUSE_FIELD_NUMBER: _ClassVar[int]
    RESULT_FIELD_NUMBER: _ClassVar[int]
    EXECUTION_STARTED_FIELD_NUMBER: _ClassVar[int]
    ARTIFACT_RECEIPTS_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    attempt_ordinal: int
    invocation_spec_digest: str
    status: OutcomeStatus
    output_manifest: OutputManifest
    metrics: AttemptMetrics
    triage_bundle: TriageBundleRef
    safe_message: str
    cause: OutcomeCause
    result: ResultEnvelope
    execution_started: bool
    artifact_receipts: _containers.RepeatedCompositeFieldContainer[ArtifactReceiptRef]
    def __init__(self, request_id: _Optional[str] = ..., attempt_ordinal: _Optional[int] = ..., invocation_spec_digest: _Optional[str] = ..., status: _Optional[_Union[OutcomeStatus, str]] = ..., output_manifest: _Optional[_Union[OutputManifest, _Mapping]] = ..., metrics: _Optional[_Union[AttemptMetrics, _Mapping]] = ..., triage_bundle: _Optional[_Union[TriageBundleRef, _Mapping]] = ..., safe_message: _Optional[str] = ..., cause: _Optional[_Union[OutcomeCause, _Mapping]] = ..., result: _Optional[_Union[ResultEnvelope, _Mapping]] = ..., execution_started: bool = ..., artifact_receipts: _Optional[_Iterable[_Union[ArtifactReceiptRef, _Mapping]]] = ...) -> None: ...

class ArtifactReceiptRef(_message.Message):
    __slots__ = ("artifact_receipt_digest", "artifact_receipt_canonical_bytes")
    ARTIFACT_RECEIPT_DIGEST_FIELD_NUMBER: _ClassVar[int]
    ARTIFACT_RECEIPT_CANONICAL_BYTES_FIELD_NUMBER: _ClassVar[int]
    artifact_receipt_digest: bytes
    artifact_receipt_canonical_bytes: bytes
    def __init__(self, artifact_receipt_digest: _Optional[bytes] = ..., artifact_receipt_canonical_bytes: _Optional[bytes] = ...) -> None: ...

class ArtifactReceipt(_message.Message):
    __slots__ = ("owner_authority_scope", "request_id", "invocation_spec_digest", "output_slot", "artifact_transaction_id", "tensorfs_receipt_digest", "tensorfs_receipt_canonical_bytes")
    OWNER_AUTHORITY_SCOPE_FIELD_NUMBER: _ClassVar[int]
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    INVOCATION_SPEC_DIGEST_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_SLOT_FIELD_NUMBER: _ClassVar[int]
    ARTIFACT_TRANSACTION_ID_FIELD_NUMBER: _ClassVar[int]
    TENSORFS_RECEIPT_DIGEST_FIELD_NUMBER: _ClassVar[int]
    TENSORFS_RECEIPT_CANONICAL_BYTES_FIELD_NUMBER: _ClassVar[int]
    owner_authority_scope: str
    request_id: str
    invocation_spec_digest: str
    output_slot: str
    artifact_transaction_id: str
    tensorfs_receipt_digest: str
    tensorfs_receipt_canonical_bytes: bytes
    def __init__(self, owner_authority_scope: _Optional[str] = ..., request_id: _Optional[str] = ..., invocation_spec_digest: _Optional[str] = ..., output_slot: _Optional[str] = ..., artifact_transaction_id: _Optional[str] = ..., tensorfs_receipt_digest: _Optional[str] = ..., tensorfs_receipt_canonical_bytes: _Optional[bytes] = ...) -> None: ...

class ArtifactObjectSource(_message.Message):
    __slots__ = ("object_id", "length", "source_ref")
    OBJECT_ID_FIELD_NUMBER: _ClassVar[int]
    LENGTH_FIELD_NUMBER: _ClassVar[int]
    SOURCE_REF_FIELD_NUMBER: _ClassVar[int]
    object_id: str
    length: int
    source_ref: str
    def __init__(self, object_id: _Optional[str] = ..., length: _Optional[int] = ..., source_ref: _Optional[str] = ...) -> None: ...

class ArtifactIntentFrame(_message.Message):
    __slots__ = ("record_owner_epoch", "control_stream_generation", "worker_boot_id", "request_id", "attempt_ordinal", "invocation_spec_digest", "output_slot", "tensorfs_declaration_digest", "requested_writer_generation", "tensorfs_declaration_canonical_bytes")
    RECORD_OWNER_EPOCH_FIELD_NUMBER: _ClassVar[int]
    CONTROL_STREAM_GENERATION_FIELD_NUMBER: _ClassVar[int]
    WORKER_BOOT_ID_FIELD_NUMBER: _ClassVar[int]
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    ATTEMPT_ORDINAL_FIELD_NUMBER: _ClassVar[int]
    INVOCATION_SPEC_DIGEST_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_SLOT_FIELD_NUMBER: _ClassVar[int]
    TENSORFS_DECLARATION_DIGEST_FIELD_NUMBER: _ClassVar[int]
    REQUESTED_WRITER_GENERATION_FIELD_NUMBER: _ClassVar[int]
    TENSORFS_DECLARATION_CANONICAL_BYTES_FIELD_NUMBER: _ClassVar[int]
    record_owner_epoch: int
    control_stream_generation: int
    worker_boot_id: str
    request_id: str
    attempt_ordinal: int
    invocation_spec_digest: bytes
    output_slot: str
    tensorfs_declaration_digest: bytes
    requested_writer_generation: int
    tensorfs_declaration_canonical_bytes: bytes
    def __init__(self, record_owner_epoch: _Optional[int] = ..., control_stream_generation: _Optional[int] = ..., worker_boot_id: _Optional[str] = ..., request_id: _Optional[str] = ..., attempt_ordinal: _Optional[int] = ..., invocation_spec_digest: _Optional[bytes] = ..., output_slot: _Optional[str] = ..., tensorfs_declaration_digest: _Optional[bytes] = ..., requested_writer_generation: _Optional[int] = ..., tensorfs_declaration_canonical_bytes: _Optional[bytes] = ...) -> None: ...

class ArtifactHostAck(_message.Message):
    __slots__ = ("record_owner_epoch", "control_stream_generation", "worker_boot_id", "request_id", "attempt_ordinal", "invocation_spec_digest", "output_slot", "artifact_transaction_id", "writer_generation", "tensorfs_declaration_digest", "stage", "outcome", "refusal", "artifact_receipt", "manifest")
    RECORD_OWNER_EPOCH_FIELD_NUMBER: _ClassVar[int]
    CONTROL_STREAM_GENERATION_FIELD_NUMBER: _ClassVar[int]
    WORKER_BOOT_ID_FIELD_NUMBER: _ClassVar[int]
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    ATTEMPT_ORDINAL_FIELD_NUMBER: _ClassVar[int]
    INVOCATION_SPEC_DIGEST_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_SLOT_FIELD_NUMBER: _ClassVar[int]
    ARTIFACT_TRANSACTION_ID_FIELD_NUMBER: _ClassVar[int]
    WRITER_GENERATION_FIELD_NUMBER: _ClassVar[int]
    TENSORFS_DECLARATION_DIGEST_FIELD_NUMBER: _ClassVar[int]
    STAGE_FIELD_NUMBER: _ClassVar[int]
    OUTCOME_FIELD_NUMBER: _ClassVar[int]
    REFUSAL_FIELD_NUMBER: _ClassVar[int]
    ARTIFACT_RECEIPT_FIELD_NUMBER: _ClassVar[int]
    MANIFEST_FIELD_NUMBER: _ClassVar[int]
    record_owner_epoch: int
    control_stream_generation: int
    worker_boot_id: str
    request_id: str
    attempt_ordinal: int
    invocation_spec_digest: bytes
    output_slot: str
    artifact_transaction_id: str
    writer_generation: int
    tensorfs_declaration_digest: bytes
    stage: ArtifactHostStage
    outcome: ArtifactHostOutcome
    refusal: ArtifactHostRefusal
    artifact_receipt: ArtifactReceiptRef
    manifest: Ref
    def __init__(self, record_owner_epoch: _Optional[int] = ..., control_stream_generation: _Optional[int] = ..., worker_boot_id: _Optional[str] = ..., request_id: _Optional[str] = ..., attempt_ordinal: _Optional[int] = ..., invocation_spec_digest: _Optional[bytes] = ..., output_slot: _Optional[str] = ..., artifact_transaction_id: _Optional[str] = ..., writer_generation: _Optional[int] = ..., tensorfs_declaration_digest: _Optional[bytes] = ..., stage: _Optional[_Union[ArtifactHostStage, str]] = ..., outcome: _Optional[_Union[ArtifactHostOutcome, str]] = ..., refusal: _Optional[_Union[ArtifactHostRefusal, str]] = ..., artifact_receipt: _Optional[_Union[ArtifactReceiptRef, _Mapping]] = ..., manifest: _Optional[_Union[Ref, _Mapping]] = ...) -> None: ...

class ArtifactReceiptFrame(_message.Message):
    __slots__ = ("record_owner_epoch", "control_stream_generation", "worker_boot_id", "request_id", "attempt_ordinal", "invocation_spec_digest", "output_slot", "artifact_transaction_id", "writer_generation", "tensorfs_declaration_digest", "artifact_receipt", "objects", "manifest")
    RECORD_OWNER_EPOCH_FIELD_NUMBER: _ClassVar[int]
    CONTROL_STREAM_GENERATION_FIELD_NUMBER: _ClassVar[int]
    WORKER_BOOT_ID_FIELD_NUMBER: _ClassVar[int]
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    ATTEMPT_ORDINAL_FIELD_NUMBER: _ClassVar[int]
    INVOCATION_SPEC_DIGEST_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_SLOT_FIELD_NUMBER: _ClassVar[int]
    ARTIFACT_TRANSACTION_ID_FIELD_NUMBER: _ClassVar[int]
    WRITER_GENERATION_FIELD_NUMBER: _ClassVar[int]
    TENSORFS_DECLARATION_DIGEST_FIELD_NUMBER: _ClassVar[int]
    ARTIFACT_RECEIPT_FIELD_NUMBER: _ClassVar[int]
    OBJECTS_FIELD_NUMBER: _ClassVar[int]
    MANIFEST_FIELD_NUMBER: _ClassVar[int]
    record_owner_epoch: int
    control_stream_generation: int
    worker_boot_id: str
    request_id: str
    attempt_ordinal: int
    invocation_spec_digest: bytes
    output_slot: str
    artifact_transaction_id: str
    writer_generation: int
    tensorfs_declaration_digest: bytes
    artifact_receipt: ArtifactReceiptRef
    objects: _containers.RepeatedCompositeFieldContainer[ArtifactObjectSource]
    manifest: Ref
    def __init__(self, record_owner_epoch: _Optional[int] = ..., control_stream_generation: _Optional[int] = ..., worker_boot_id: _Optional[str] = ..., request_id: _Optional[str] = ..., attempt_ordinal: _Optional[int] = ..., invocation_spec_digest: _Optional[bytes] = ..., output_slot: _Optional[str] = ..., artifact_transaction_id: _Optional[str] = ..., writer_generation: _Optional[int] = ..., tensorfs_declaration_digest: _Optional[bytes] = ..., artifact_receipt: _Optional[_Union[ArtifactReceiptRef, _Mapping]] = ..., objects: _Optional[_Iterable[_Union[ArtifactObjectSource, _Mapping]]] = ..., manifest: _Optional[_Union[Ref, _Mapping]] = ...) -> None: ...

class ArtifactTransactionStatus(_message.Message):
    __slots__ = ("artifact_transaction_id", "request_id", "attempt_ordinal", "invocation_spec_digest", "output_slot", "writer_generation", "state", "tensorfs_declaration_digest", "artifact_receipt_digest")
    ARTIFACT_TRANSACTION_ID_FIELD_NUMBER: _ClassVar[int]
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    ATTEMPT_ORDINAL_FIELD_NUMBER: _ClassVar[int]
    INVOCATION_SPEC_DIGEST_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_SLOT_FIELD_NUMBER: _ClassVar[int]
    WRITER_GENERATION_FIELD_NUMBER: _ClassVar[int]
    STATE_FIELD_NUMBER: _ClassVar[int]
    TENSORFS_DECLARATION_DIGEST_FIELD_NUMBER: _ClassVar[int]
    ARTIFACT_RECEIPT_DIGEST_FIELD_NUMBER: _ClassVar[int]
    artifact_transaction_id: str
    request_id: str
    attempt_ordinal: int
    invocation_spec_digest: str
    output_slot: str
    writer_generation: int
    state: ArtifactTransactionState
    tensorfs_declaration_digest: bytes
    artifact_receipt_digest: bytes
    def __init__(self, artifact_transaction_id: _Optional[str] = ..., request_id: _Optional[str] = ..., attempt_ordinal: _Optional[int] = ..., invocation_spec_digest: _Optional[str] = ..., output_slot: _Optional[str] = ..., writer_generation: _Optional[int] = ..., state: _Optional[_Union[ArtifactTransactionState, str]] = ..., tensorfs_declaration_digest: _Optional[bytes] = ..., artifact_receipt_digest: _Optional[bytes] = ...) -> None: ...

class ArtifactReadRequest(_message.Message):
    __slots__ = ("record_owner_epoch", "control_stream_generation", "worker_boot_id", "artifact_transaction_id", "writer_generation", "object_id", "source_ref", "read_id", "offset", "max_bytes")
    RECORD_OWNER_EPOCH_FIELD_NUMBER: _ClassVar[int]
    CONTROL_STREAM_GENERATION_FIELD_NUMBER: _ClassVar[int]
    WORKER_BOOT_ID_FIELD_NUMBER: _ClassVar[int]
    ARTIFACT_TRANSACTION_ID_FIELD_NUMBER: _ClassVar[int]
    WRITER_GENERATION_FIELD_NUMBER: _ClassVar[int]
    OBJECT_ID_FIELD_NUMBER: _ClassVar[int]
    SOURCE_REF_FIELD_NUMBER: _ClassVar[int]
    READ_ID_FIELD_NUMBER: _ClassVar[int]
    OFFSET_FIELD_NUMBER: _ClassVar[int]
    MAX_BYTES_FIELD_NUMBER: _ClassVar[int]
    record_owner_epoch: int
    control_stream_generation: int
    worker_boot_id: str
    artifact_transaction_id: str
    writer_generation: int
    object_id: str
    source_ref: str
    read_id: str
    offset: int
    max_bytes: int
    def __init__(self, record_owner_epoch: _Optional[int] = ..., control_stream_generation: _Optional[int] = ..., worker_boot_id: _Optional[str] = ..., artifact_transaction_id: _Optional[str] = ..., writer_generation: _Optional[int] = ..., object_id: _Optional[str] = ..., source_ref: _Optional[str] = ..., read_id: _Optional[str] = ..., offset: _Optional[int] = ..., max_bytes: _Optional[int] = ...) -> None: ...

class ArtifactReadResult(_message.Message):
    __slots__ = ("record_owner_epoch", "control_stream_generation", "worker_boot_id", "artifact_transaction_id", "writer_generation", "object_id", "source_ref", "read_id", "offset", "outcome", "data", "data_digest", "refusal", "safe_detail")
    RECORD_OWNER_EPOCH_FIELD_NUMBER: _ClassVar[int]
    CONTROL_STREAM_GENERATION_FIELD_NUMBER: _ClassVar[int]
    WORKER_BOOT_ID_FIELD_NUMBER: _ClassVar[int]
    ARTIFACT_TRANSACTION_ID_FIELD_NUMBER: _ClassVar[int]
    WRITER_GENERATION_FIELD_NUMBER: _ClassVar[int]
    OBJECT_ID_FIELD_NUMBER: _ClassVar[int]
    SOURCE_REF_FIELD_NUMBER: _ClassVar[int]
    READ_ID_FIELD_NUMBER: _ClassVar[int]
    OFFSET_FIELD_NUMBER: _ClassVar[int]
    OUTCOME_FIELD_NUMBER: _ClassVar[int]
    DATA_FIELD_NUMBER: _ClassVar[int]
    DATA_DIGEST_FIELD_NUMBER: _ClassVar[int]
    REFUSAL_FIELD_NUMBER: _ClassVar[int]
    SAFE_DETAIL_FIELD_NUMBER: _ClassVar[int]
    record_owner_epoch: int
    control_stream_generation: int
    worker_boot_id: str
    artifact_transaction_id: str
    writer_generation: int
    object_id: str
    source_ref: str
    read_id: str
    offset: int
    outcome: ArtifactReadOutcome
    data: bytes
    data_digest: bytes
    refusal: ArtifactReadRefusal
    safe_detail: str
    def __init__(self, record_owner_epoch: _Optional[int] = ..., control_stream_generation: _Optional[int] = ..., worker_boot_id: _Optional[str] = ..., artifact_transaction_id: _Optional[str] = ..., writer_generation: _Optional[int] = ..., object_id: _Optional[str] = ..., source_ref: _Optional[str] = ..., read_id: _Optional[str] = ..., offset: _Optional[int] = ..., outcome: _Optional[_Union[ArtifactReadOutcome, str]] = ..., data: _Optional[bytes] = ..., data_digest: _Optional[bytes] = ..., refusal: _Optional[_Union[ArtifactReadRefusal, str]] = ..., safe_detail: _Optional[str] = ...) -> None: ...

class ArtifactObjectRef(_message.Message):
    __slots__ = ("object_id", "length")
    OBJECT_ID_FIELD_NUMBER: _ClassVar[int]
    LENGTH_FIELD_NUMBER: _ClassVar[int]
    object_id: str
    length: int
    def __init__(self, object_id: _Optional[str] = ..., length: _Optional[int] = ...) -> None: ...

class ArtifactUploadHeader(_message.Message):
    __slots__ = ("name", "value")
    NAME_FIELD_NUMBER: _ClassVar[int]
    VALUE_FIELD_NUMBER: _ClassVar[int]
    name: str
    value: str
    def __init__(self, name: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...

class ArtifactUploadGrant(_message.Message):
    __slots__ = ("object_id", "length", "url", "required_headers", "expires_at_unix")
    OBJECT_ID_FIELD_NUMBER: _ClassVar[int]
    LENGTH_FIELD_NUMBER: _ClassVar[int]
    URL_FIELD_NUMBER: _ClassVar[int]
    REQUIRED_HEADERS_FIELD_NUMBER: _ClassVar[int]
    EXPIRES_AT_UNIX_FIELD_NUMBER: _ClassVar[int]
    object_id: str
    length: int
    url: str
    required_headers: _containers.RepeatedCompositeFieldContainer[ArtifactUploadHeader]
    expires_at_unix: int
    def __init__(self, object_id: _Optional[str] = ..., length: _Optional[int] = ..., url: _Optional[str] = ..., required_headers: _Optional[_Iterable[_Union[ArtifactUploadHeader, _Mapping]]] = ..., expires_at_unix: _Optional[int] = ...) -> None: ...

class ArtifactTransferRequest(_message.Message):
    __slots__ = ("record_owner_epoch", "control_stream_generation", "worker_boot_id", "request_id", "attempt_ordinal", "invocation_spec_digest", "output_slot", "artifact_transaction_id", "artifact_receipt_digest", "operation_id", "grant_revision", "upload_grant", "held")
    RECORD_OWNER_EPOCH_FIELD_NUMBER: _ClassVar[int]
    CONTROL_STREAM_GENERATION_FIELD_NUMBER: _ClassVar[int]
    WORKER_BOOT_ID_FIELD_NUMBER: _ClassVar[int]
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    ATTEMPT_ORDINAL_FIELD_NUMBER: _ClassVar[int]
    INVOCATION_SPEC_DIGEST_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_SLOT_FIELD_NUMBER: _ClassVar[int]
    ARTIFACT_TRANSACTION_ID_FIELD_NUMBER: _ClassVar[int]
    ARTIFACT_RECEIPT_DIGEST_FIELD_NUMBER: _ClassVar[int]
    OPERATION_ID_FIELD_NUMBER: _ClassVar[int]
    GRANT_REVISION_FIELD_NUMBER: _ClassVar[int]
    UPLOAD_GRANT_FIELD_NUMBER: _ClassVar[int]
    HELD_FIELD_NUMBER: _ClassVar[int]
    record_owner_epoch: int
    control_stream_generation: int
    worker_boot_id: str
    request_id: str
    attempt_ordinal: int
    invocation_spec_digest: bytes
    output_slot: str
    artifact_transaction_id: str
    artifact_receipt_digest: bytes
    operation_id: str
    grant_revision: int
    upload_grant: ArtifactUploadGrant
    held: ArtifactObjectRef
    def __init__(self, record_owner_epoch: _Optional[int] = ..., control_stream_generation: _Optional[int] = ..., worker_boot_id: _Optional[str] = ..., request_id: _Optional[str] = ..., attempt_ordinal: _Optional[int] = ..., invocation_spec_digest: _Optional[bytes] = ..., output_slot: _Optional[str] = ..., artifact_transaction_id: _Optional[str] = ..., artifact_receipt_digest: _Optional[bytes] = ..., operation_id: _Optional[str] = ..., grant_revision: _Optional[int] = ..., upload_grant: _Optional[_Union[ArtifactUploadGrant, _Mapping]] = ..., held: _Optional[_Union[ArtifactObjectRef, _Mapping]] = ...) -> None: ...

class ArtifactTransferStatus(_message.Message):
    __slots__ = ("record_owner_epoch", "control_stream_generation", "worker_boot_id", "request_id", "attempt_ordinal", "invocation_spec_digest", "output_slot", "artifact_transaction_id", "operation_id", "grant_revision", "object_id", "length", "update_sequence", "state", "transferred_bytes", "http_status", "etag", "checksum_sha256", "attempts", "safe_code", "safe_detail")
    RECORD_OWNER_EPOCH_FIELD_NUMBER: _ClassVar[int]
    CONTROL_STREAM_GENERATION_FIELD_NUMBER: _ClassVar[int]
    WORKER_BOOT_ID_FIELD_NUMBER: _ClassVar[int]
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    ATTEMPT_ORDINAL_FIELD_NUMBER: _ClassVar[int]
    INVOCATION_SPEC_DIGEST_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_SLOT_FIELD_NUMBER: _ClassVar[int]
    ARTIFACT_TRANSACTION_ID_FIELD_NUMBER: _ClassVar[int]
    OPERATION_ID_FIELD_NUMBER: _ClassVar[int]
    GRANT_REVISION_FIELD_NUMBER: _ClassVar[int]
    OBJECT_ID_FIELD_NUMBER: _ClassVar[int]
    LENGTH_FIELD_NUMBER: _ClassVar[int]
    UPDATE_SEQUENCE_FIELD_NUMBER: _ClassVar[int]
    STATE_FIELD_NUMBER: _ClassVar[int]
    TRANSFERRED_BYTES_FIELD_NUMBER: _ClassVar[int]
    HTTP_STATUS_FIELD_NUMBER: _ClassVar[int]
    ETAG_FIELD_NUMBER: _ClassVar[int]
    CHECKSUM_SHA256_FIELD_NUMBER: _ClassVar[int]
    ATTEMPTS_FIELD_NUMBER: _ClassVar[int]
    SAFE_CODE_FIELD_NUMBER: _ClassVar[int]
    SAFE_DETAIL_FIELD_NUMBER: _ClassVar[int]
    record_owner_epoch: int
    control_stream_generation: int
    worker_boot_id: str
    request_id: str
    attempt_ordinal: int
    invocation_spec_digest: bytes
    output_slot: str
    artifact_transaction_id: str
    operation_id: str
    grant_revision: int
    object_id: str
    length: int
    update_sequence: int
    state: ArtifactTransferState
    transferred_bytes: int
    http_status: int
    etag: str
    checksum_sha256: str
    attempts: int
    safe_code: str
    safe_detail: str
    def __init__(self, record_owner_epoch: _Optional[int] = ..., control_stream_generation: _Optional[int] = ..., worker_boot_id: _Optional[str] = ..., request_id: _Optional[str] = ..., attempt_ordinal: _Optional[int] = ..., invocation_spec_digest: _Optional[bytes] = ..., output_slot: _Optional[str] = ..., artifact_transaction_id: _Optional[str] = ..., operation_id: _Optional[str] = ..., grant_revision: _Optional[int] = ..., object_id: _Optional[str] = ..., length: _Optional[int] = ..., update_sequence: _Optional[int] = ..., state: _Optional[_Union[ArtifactTransferState, str]] = ..., transferred_bytes: _Optional[int] = ..., http_status: _Optional[int] = ..., etag: _Optional[str] = ..., checksum_sha256: _Optional[str] = ..., attempts: _Optional[int] = ..., safe_code: _Optional[str] = ..., safe_detail: _Optional[str] = ...) -> None: ...

class ArtifactFinalizeRequest(_message.Message):
    __slots__ = ("record_owner_epoch", "control_stream_generation", "worker_boot_id", "request_id", "invocation_spec_digest", "output_slot", "disposition", "artifact_receipt_digest", "scratch_root_id", "owner_authority_scope")
    RECORD_OWNER_EPOCH_FIELD_NUMBER: _ClassVar[int]
    CONTROL_STREAM_GENERATION_FIELD_NUMBER: _ClassVar[int]
    WORKER_BOOT_ID_FIELD_NUMBER: _ClassVar[int]
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    INVOCATION_SPEC_DIGEST_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_SLOT_FIELD_NUMBER: _ClassVar[int]
    DISPOSITION_FIELD_NUMBER: _ClassVar[int]
    ARTIFACT_RECEIPT_DIGEST_FIELD_NUMBER: _ClassVar[int]
    SCRATCH_ROOT_ID_FIELD_NUMBER: _ClassVar[int]
    OWNER_AUTHORITY_SCOPE_FIELD_NUMBER: _ClassVar[int]
    record_owner_epoch: int
    control_stream_generation: int
    worker_boot_id: str
    request_id: str
    invocation_spec_digest: bytes
    output_slot: str
    disposition: ArtifactFinalizeDisposition
    artifact_receipt_digest: bytes
    scratch_root_id: str
    owner_authority_scope: str
    def __init__(self, record_owner_epoch: _Optional[int] = ..., control_stream_generation: _Optional[int] = ..., worker_boot_id: _Optional[str] = ..., request_id: _Optional[str] = ..., invocation_spec_digest: _Optional[bytes] = ..., output_slot: _Optional[str] = ..., disposition: _Optional[_Union[ArtifactFinalizeDisposition, str]] = ..., artifact_receipt_digest: _Optional[bytes] = ..., scratch_root_id: _Optional[str] = ..., owner_authority_scope: _Optional[str] = ...) -> None: ...

class ArtifactFinalizeResult(_message.Message):
    __slots__ = ("record_owner_epoch", "control_stream_generation", "worker_boot_id", "request_id", "invocation_spec_digest", "output_slot", "outcome", "artifact_receipt", "owner_authority_scope")
    RECORD_OWNER_EPOCH_FIELD_NUMBER: _ClassVar[int]
    CONTROL_STREAM_GENERATION_FIELD_NUMBER: _ClassVar[int]
    WORKER_BOOT_ID_FIELD_NUMBER: _ClassVar[int]
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    INVOCATION_SPEC_DIGEST_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_SLOT_FIELD_NUMBER: _ClassVar[int]
    OUTCOME_FIELD_NUMBER: _ClassVar[int]
    ARTIFACT_RECEIPT_FIELD_NUMBER: _ClassVar[int]
    OWNER_AUTHORITY_SCOPE_FIELD_NUMBER: _ClassVar[int]
    record_owner_epoch: int
    control_stream_generation: int
    worker_boot_id: str
    request_id: str
    invocation_spec_digest: bytes
    output_slot: str
    outcome: ArtifactFinalizeOutcome
    artifact_receipt: ArtifactReceiptRef
    owner_authority_scope: str
    def __init__(self, record_owner_epoch: _Optional[int] = ..., control_stream_generation: _Optional[int] = ..., worker_boot_id: _Optional[str] = ..., request_id: _Optional[str] = ..., invocation_spec_digest: _Optional[bytes] = ..., output_slot: _Optional[str] = ..., outcome: _Optional[_Union[ArtifactFinalizeOutcome, str]] = ..., artifact_receipt: _Optional[_Union[ArtifactReceiptRef, _Mapping]] = ..., owner_authority_scope: _Optional[str] = ...) -> None: ...

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

class OutcomeCause(_message.Message):
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

class AttemptOutcomeAck(_message.Message):
    __slots__ = ("record_owner_epoch", "control_stream_generation", "worker_boot_id", "request_id", "attempt_ordinal", "invocation_spec_digest", "outcome_id", "outcome_digest")
    RECORD_OWNER_EPOCH_FIELD_NUMBER: _ClassVar[int]
    CONTROL_STREAM_GENERATION_FIELD_NUMBER: _ClassVar[int]
    WORKER_BOOT_ID_FIELD_NUMBER: _ClassVar[int]
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    ATTEMPT_ORDINAL_FIELD_NUMBER: _ClassVar[int]
    INVOCATION_SPEC_DIGEST_FIELD_NUMBER: _ClassVar[int]
    OUTCOME_ID_FIELD_NUMBER: _ClassVar[int]
    OUTCOME_DIGEST_FIELD_NUMBER: _ClassVar[int]
    record_owner_epoch: int
    control_stream_generation: int
    worker_boot_id: str
    request_id: str
    attempt_ordinal: int
    invocation_spec_digest: bytes
    outcome_id: str
    outcome_digest: bytes
    def __init__(self, record_owner_epoch: _Optional[int] = ..., control_stream_generation: _Optional[int] = ..., worker_boot_id: _Optional[str] = ..., request_id: _Optional[str] = ..., attempt_ordinal: _Optional[int] = ..., invocation_spec_digest: _Optional[bytes] = ..., outcome_id: _Optional[str] = ..., outcome_digest: _Optional[bytes] = ...) -> None: ...

class JobCheckpointRequest(_message.Message):
    __slots__ = ("record_owner_epoch", "control_stream_generation", "worker_boot_id", "request_id", "attempt_ordinal", "operation_key", "logical_key", "content_digest", "seq", "artifact")
    RECORD_OWNER_EPOCH_FIELD_NUMBER: _ClassVar[int]
    CONTROL_STREAM_GENERATION_FIELD_NUMBER: _ClassVar[int]
    WORKER_BOOT_ID_FIELD_NUMBER: _ClassVar[int]
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    ATTEMPT_ORDINAL_FIELD_NUMBER: _ClassVar[int]
    OPERATION_KEY_FIELD_NUMBER: _ClassVar[int]
    LOGICAL_KEY_FIELD_NUMBER: _ClassVar[int]
    CONTENT_DIGEST_FIELD_NUMBER: _ClassVar[int]
    SEQ_FIELD_NUMBER: _ClassVar[int]
    ARTIFACT_FIELD_NUMBER: _ClassVar[int]
    record_owner_epoch: int
    control_stream_generation: int
    worker_boot_id: str
    request_id: str
    attempt_ordinal: int
    operation_key: str
    logical_key: str
    content_digest: bytes
    seq: int
    artifact: OutputEntry
    def __init__(self, record_owner_epoch: _Optional[int] = ..., control_stream_generation: _Optional[int] = ..., worker_boot_id: _Optional[str] = ..., request_id: _Optional[str] = ..., attempt_ordinal: _Optional[int] = ..., operation_key: _Optional[str] = ..., logical_key: _Optional[str] = ..., content_digest: _Optional[bytes] = ..., seq: _Optional[int] = ..., artifact: _Optional[_Union[OutputEntry, _Mapping]] = ...) -> None: ...

class JobCheckpointReceipt(_message.Message):
    __slots__ = ("record_owner_epoch", "control_stream_generation", "worker_boot_id", "request_id", "attempt_ordinal", "operation_key", "logical_key", "content_digest", "receipt_id", "outcome", "fault")
    RECORD_OWNER_EPOCH_FIELD_NUMBER: _ClassVar[int]
    CONTROL_STREAM_GENERATION_FIELD_NUMBER: _ClassVar[int]
    WORKER_BOOT_ID_FIELD_NUMBER: _ClassVar[int]
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    ATTEMPT_ORDINAL_FIELD_NUMBER: _ClassVar[int]
    OPERATION_KEY_FIELD_NUMBER: _ClassVar[int]
    LOGICAL_KEY_FIELD_NUMBER: _ClassVar[int]
    CONTENT_DIGEST_FIELD_NUMBER: _ClassVar[int]
    RECEIPT_ID_FIELD_NUMBER: _ClassVar[int]
    OUTCOME_FIELD_NUMBER: _ClassVar[int]
    FAULT_FIELD_NUMBER: _ClassVar[int]
    record_owner_epoch: int
    control_stream_generation: int
    worker_boot_id: str
    request_id: str
    attempt_ordinal: int
    operation_key: str
    logical_key: str
    content_digest: bytes
    receipt_id: str
    outcome: CheckpointOutcome
    fault: CheckpointFault
    def __init__(self, record_owner_epoch: _Optional[int] = ..., control_stream_generation: _Optional[int] = ..., worker_boot_id: _Optional[str] = ..., request_id: _Optional[str] = ..., attempt_ordinal: _Optional[int] = ..., operation_key: _Optional[str] = ..., logical_key: _Optional[str] = ..., content_digest: _Optional[bytes] = ..., receipt_id: _Optional[str] = ..., outcome: _Optional[_Union[CheckpointOutcome, str]] = ..., fault: _Optional[_Union[CheckpointFault, _Mapping]] = ...) -> None: ...

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
    __slots__ = ("record_owner_epoch", "control_stream_generation", "worker_boot_id", "request_id", "attempt_ordinal", "operation_key", "receipt_id", "content_digest")
    RECORD_OWNER_EPOCH_FIELD_NUMBER: _ClassVar[int]
    CONTROL_STREAM_GENERATION_FIELD_NUMBER: _ClassVar[int]
    WORKER_BOOT_ID_FIELD_NUMBER: _ClassVar[int]
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    ATTEMPT_ORDINAL_FIELD_NUMBER: _ClassVar[int]
    OPERATION_KEY_FIELD_NUMBER: _ClassVar[int]
    RECEIPT_ID_FIELD_NUMBER: _ClassVar[int]
    CONTENT_DIGEST_FIELD_NUMBER: _ClassVar[int]
    record_owner_epoch: int
    control_stream_generation: int
    worker_boot_id: str
    request_id: str
    attempt_ordinal: int
    operation_key: str
    receipt_id: str
    content_digest: bytes
    def __init__(self, record_owner_epoch: _Optional[int] = ..., control_stream_generation: _Optional[int] = ..., worker_boot_id: _Optional[str] = ..., request_id: _Optional[str] = ..., attempt_ordinal: _Optional[int] = ..., operation_key: _Optional[str] = ..., receipt_id: _Optional[str] = ..., content_digest: _Optional[bytes] = ...) -> None: ...

class ProgressOpen(_message.Message):
    __slots__ = ("record_owner_epoch", "control_stream_generation", "worker_boot_id", "request_id")
    RECORD_OWNER_EPOCH_FIELD_NUMBER: _ClassVar[int]
    CONTROL_STREAM_GENERATION_FIELD_NUMBER: _ClassVar[int]
    WORKER_BOOT_ID_FIELD_NUMBER: _ClassVar[int]
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    record_owner_epoch: int
    control_stream_generation: int
    worker_boot_id: str
    request_id: str
    def __init__(self, record_owner_epoch: _Optional[int] = ..., control_stream_generation: _Optional[int] = ..., worker_boot_id: _Optional[str] = ..., request_id: _Optional[str] = ...) -> None: ...

class AttemptProgress(_message.Message):
    __slots__ = ("record_owner_epoch", "control_stream_generation", "worker_boot_id", "request_id", "attempt_ordinal", "seq", "content_type", "data", "placement_id")
    RECORD_OWNER_EPOCH_FIELD_NUMBER: _ClassVar[int]
    CONTROL_STREAM_GENERATION_FIELD_NUMBER: _ClassVar[int]
    WORKER_BOOT_ID_FIELD_NUMBER: _ClassVar[int]
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    ATTEMPT_ORDINAL_FIELD_NUMBER: _ClassVar[int]
    SEQ_FIELD_NUMBER: _ClassVar[int]
    CONTENT_TYPE_FIELD_NUMBER: _ClassVar[int]
    DATA_FIELD_NUMBER: _ClassVar[int]
    PLACEMENT_ID_FIELD_NUMBER: _ClassVar[int]
    record_owner_epoch: int
    control_stream_generation: int
    worker_boot_id: str
    request_id: str
    attempt_ordinal: int
    seq: int
    content_type: str
    data: bytes
    placement_id: str
    def __init__(self, record_owner_epoch: _Optional[int] = ..., control_stream_generation: _Optional[int] = ..., worker_boot_id: _Optional[str] = ..., request_id: _Optional[str] = ..., attempt_ordinal: _Optional[int] = ..., seq: _Optional[int] = ..., content_type: _Optional[str] = ..., data: _Optional[bytes] = ..., placement_id: _Optional[str] = ...) -> None: ...

class InvocationSpec(_message.Message):
    __slots__ = ("package_revision_digest", "environment_digest", "config_digest", "payload_digest", "inputs", "outputs", "deadline_unix_ms", "serving", "job")
    PACKAGE_REVISION_DIGEST_FIELD_NUMBER: _ClassVar[int]
    ENVIRONMENT_DIGEST_FIELD_NUMBER: _ClassVar[int]
    CONFIG_DIGEST_FIELD_NUMBER: _ClassVar[int]
    PAYLOAD_DIGEST_FIELD_NUMBER: _ClassVar[int]
    INPUTS_FIELD_NUMBER: _ClassVar[int]
    OUTPUTS_FIELD_NUMBER: _ClassVar[int]
    DEADLINE_UNIX_MS_FIELD_NUMBER: _ClassVar[int]
    SERVING_FIELD_NUMBER: _ClassVar[int]
    JOB_FIELD_NUMBER: _ClassVar[int]
    package_revision_digest: str
    environment_digest: str
    config_digest: str
    payload_digest: str
    inputs: _containers.RepeatedCompositeFieldContainer[InputBinding]
    outputs: _containers.RepeatedCompositeFieldContainer[OutputBinding]
    deadline_unix_ms: int
    serving: ServingInvocationSpec
    job: JobInvocationSpec
    def __init__(self, package_revision_digest: _Optional[str] = ..., environment_digest: _Optional[str] = ..., config_digest: _Optional[str] = ..., payload_digest: _Optional[str] = ..., inputs: _Optional[_Iterable[_Union[InputBinding, _Mapping]]] = ..., outputs: _Optional[_Iterable[_Union[OutputBinding, _Mapping]]] = ..., deadline_unix_ms: _Optional[int] = ..., serving: _Optional[_Union[ServingInvocationSpec, _Mapping]] = ..., job: _Optional[_Union[JobInvocationSpec, _Mapping]] = ...) -> None: ...

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
    __slots__ = ("entrypoint_binding_digest", "attempt_binding_id")
    ENTRYPOINT_BINDING_DIGEST_FIELD_NUMBER: _ClassVar[int]
    ATTEMPT_BINDING_ID_FIELD_NUMBER: _ClassVar[int]
    entrypoint_binding_digest: str
    attempt_binding_id: str
    def __init__(self, entrypoint_binding_digest: _Optional[str] = ..., attempt_binding_id: _Optional[str] = ...) -> None: ...

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
    __slots__ = ("invocation_spec_digest", "credential", "file_base_url", "expires_at_unix", "inputs", "outputs")
    INVOCATION_SPEC_DIGEST_FIELD_NUMBER: _ClassVar[int]
    CREDENTIAL_FIELD_NUMBER: _ClassVar[int]
    FILE_BASE_URL_FIELD_NUMBER: _ClassVar[int]
    EXPIRES_AT_UNIX_FIELD_NUMBER: _ClassVar[int]
    INPUTS_FIELD_NUMBER: _ClassVar[int]
    OUTPUTS_FIELD_NUMBER: _ClassVar[int]
    invocation_spec_digest: bytes
    credential: DeliveryAccessCredential
    file_base_url: str
    expires_at_unix: int
    inputs: _containers.RepeatedCompositeFieldContainer[InputAccess]
    outputs: _containers.RepeatedCompositeFieldContainer[OutputAccess]
    def __init__(self, invocation_spec_digest: _Optional[bytes] = ..., credential: _Optional[_Union[DeliveryAccessCredential, _Mapping]] = ..., file_base_url: _Optional[str] = ..., expires_at_unix: _Optional[int] = ..., inputs: _Optional[_Iterable[_Union[InputAccess, _Mapping]]] = ..., outputs: _Optional[_Iterable[_Union[OutputAccess, _Mapping]]] = ...) -> None: ...

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

class DeliveryAccessCredential(_message.Message):
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

class HeldAttempt(_message.Message):
    __slots__ = ("request_id", "attempt_ordinal", "kind", "state", "invocation_spec_digest", "placement_id", "executor_generation", "outcome_id", "outcome_digest")
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    ATTEMPT_ORDINAL_FIELD_NUMBER: _ClassVar[int]
    KIND_FIELD_NUMBER: _ClassVar[int]
    STATE_FIELD_NUMBER: _ClassVar[int]
    INVOCATION_SPEC_DIGEST_FIELD_NUMBER: _ClassVar[int]
    PLACEMENT_ID_FIELD_NUMBER: _ClassVar[int]
    EXECUTOR_GENERATION_FIELD_NUMBER: _ClassVar[int]
    OUTCOME_ID_FIELD_NUMBER: _ClassVar[int]
    OUTCOME_DIGEST_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    attempt_ordinal: int
    kind: AttemptKind
    state: AttemptState
    invocation_spec_digest: bytes
    placement_id: str
    executor_generation: int
    outcome_id: str
    outcome_digest: bytes
    def __init__(self, request_id: _Optional[str] = ..., attempt_ordinal: _Optional[int] = ..., kind: _Optional[_Union[AttemptKind, str]] = ..., state: _Optional[_Union[AttemptState, str]] = ..., invocation_spec_digest: _Optional[bytes] = ..., placement_id: _Optional[str] = ..., executor_generation: _Optional[int] = ..., outcome_id: _Optional[str] = ..., outcome_digest: _Optional[bytes] = ...) -> None: ...

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
    __slots__ = ("publication_receipt_digest", "outputs")
    PUBLICATION_RECEIPT_DIGEST_FIELD_NUMBER: _ClassVar[int]
    OUTPUTS_FIELD_NUMBER: _ClassVar[int]
    publication_receipt_digest: str
    outputs: _containers.RepeatedCompositeFieldContainer[OutputEntry]
    def __init__(self, publication_receipt_digest: _Optional[str] = ..., outputs: _Optional[_Iterable[_Union[OutputEntry, _Mapping]]] = ...) -> None: ...

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
