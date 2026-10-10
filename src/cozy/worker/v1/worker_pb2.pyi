from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class MachineExecutionAction(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    MACHINE_EXECUTION_ACTION_UNSPECIFIED: _ClassVar[MachineExecutionAction]
    MACHINE_EXECUTION_ACTION_PAUSE: _ClassVar[MachineExecutionAction]
    MACHINE_EXECUTION_ACTION_RESUME: _ClassVar[MachineExecutionAction]
    MACHINE_EXECUTION_ACTION_CANCEL: _ClassVar[MachineExecutionAction]

class ChildCallState(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    CHILD_CALL_STATE_UNSPECIFIED: _ClassVar[ChildCallState]
    CHILD_CALL_STATE_PENDING: _ClassVar[ChildCallState]
    CHILD_CALL_STATE_SUCCEEDED: _ClassVar[ChildCallState]
    CHILD_CALL_STATE_REFUSED: _ClassVar[ChildCallState]
    CHILD_CALL_STATE_FAILED: _ClassVar[ChildCallState]
    CHILD_CALL_STATE_CANCELED: _ClassVar[ChildCallState]

class NativeSourcePhase(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    NATIVE_SOURCE_PHASE_UNSPECIFIED: _ClassVar[NativeSourcePhase]
    NATIVE_SOURCE_PHASE_RESOLVE: _ClassVar[NativeSourcePhase]
    NATIVE_SOURCE_PHASE_EXECUTE: _ClassVar[NativeSourcePhase]
    NATIVE_SOURCE_PHASE_CANCEL: _ClassVar[NativeSourcePhase]

class NativeSourceOperation(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    NATIVE_SOURCE_OPERATION_UNSPECIFIED: _ClassVar[NativeSourceOperation]
    NATIVE_SOURCE_OPERATION_HUGGINGFACE: _ClassVar[NativeSourceOperation]
    NATIVE_SOURCE_OPERATION_CIVITAI: _ClassVar[NativeSourceOperation]
    NATIVE_SOURCE_OPERATION_CONVERT: _ClassVar[NativeSourceOperation]
    NATIVE_SOURCE_OPERATION_SOURCE_FILES: _ClassVar[NativeSourceOperation]
    NATIVE_SOURCE_OPERATION_COMMIT_FILE: _ClassVar[NativeSourceOperation]

class NativeSourceState(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    NATIVE_SOURCE_STATE_UNSPECIFIED: _ClassVar[NativeSourceState]
    NATIVE_SOURCE_STATE_RESOLVED: _ClassVar[NativeSourceState]
    NATIVE_SOURCE_STATE_SUCCEEDED: _ClassVar[NativeSourceState]
    NATIVE_SOURCE_STATE_FAILED: _ClassVar[NativeSourceState]
    NATIVE_SOURCE_STATE_CANCELED: _ClassVar[NativeSourceState]

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
    ATTEMPT_STATE_QUEUED: _ClassVar[AttemptState]
    ATTEMPT_STATE_DEVICE_RELEASED: _ClassVar[AttemptState]

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
    CAUSE_CODE_ADMISSION_EPOCH_STALE: _ClassVar[CauseCode]
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
    CLAIM_REJECTION_PROTOCOL_INCOMPATIBLE: _ClassVar[ClaimRejection]

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

class PrepareStage(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    PREPARE_STAGE_UNSPECIFIED: _ClassVar[PrepareStage]
    PREPARE_STAGE_RESOLVED: _ClassVar[PrepareStage]
    PREPARE_STAGE_DOWNLOADING: _ClassVar[PrepareStage]
    PREPARE_STAGE_PREPARING: _ClassVar[PrepareStage]
    PREPARE_STAGE_PREPARED: _ClassVar[PrepareStage]
    PREPARE_STAGE_REFUSED: _ClassVar[PrepareStage]

class WeightsHostStage(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    WEIGHTS_HOST_STAGE_UNSPECIFIED: _ClassVar[WeightsHostStage]
    WEIGHTS_HOST_STAGE_INTENT: _ClassVar[WeightsHostStage]
    WEIGHTS_HOST_STAGE_RECEIPT: _ClassVar[WeightsHostStage]
    WEIGHTS_HOST_STAGE_CHECKPOINT: _ClassVar[WeightsHostStage]

class WeightsHostOutcome(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    WEIGHTS_HOST_OUTCOME_UNSPECIFIED: _ClassVar[WeightsHostOutcome]
    WEIGHTS_HOST_OUTCOME_RECORDED: _ClassVar[WeightsHostOutcome]
    WEIGHTS_HOST_OUTCOME_REPLAYED: _ClassVar[WeightsHostOutcome]
    WEIGHTS_HOST_OUTCOME_REFUSED: _ClassVar[WeightsHostOutcome]

class WeightsHostRefusal(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    WEIGHTS_HOST_REFUSAL_UNSPECIFIED: _ClassVar[WeightsHostRefusal]
    WEIGHTS_HOST_REFUSAL_UNKNOWN_ATTEMPT: _ClassVar[WeightsHostRefusal]
    WEIGHTS_HOST_REFUSAL_INTENT_CONFLICT: _ClassVar[WeightsHostRefusal]
    WEIGHTS_HOST_REFUSAL_STALE_WRITER: _ClassVar[WeightsHostRefusal]
    WEIGHTS_HOST_REFUSAL_RECEIPT_CONFLICT: _ClassVar[WeightsHostRefusal]
    WEIGHTS_HOST_REFUSAL_INVENTORY_INVALID: _ClassVar[WeightsHostRefusal]

class WeightsTransactionState(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    WEIGHTS_TRANSACTION_STATE_UNSPECIFIED: _ClassVar[WeightsTransactionState]
    WEIGHTS_TRANSACTION_STATE_INTENT: _ClassVar[WeightsTransactionState]
    WEIGHTS_TRANSACTION_STATE_RECEIPT: _ClassVar[WeightsTransactionState]

class WeightsReadOutcome(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    WEIGHTS_READ_OUTCOME_UNSPECIFIED: _ClassVar[WeightsReadOutcome]
    WEIGHTS_READ_OUTCOME_DATA: _ClassVar[WeightsReadOutcome]
    WEIGHTS_READ_OUTCOME_REFUSED: _ClassVar[WeightsReadOutcome]

class WeightsReadRefusal(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    WEIGHTS_READ_REFUSAL_UNSPECIFIED: _ClassVar[WeightsReadRefusal]
    WEIGHTS_READ_REFUSAL_UNKNOWN_TRANSACTION: _ClassVar[WeightsReadRefusal]
    WEIGHTS_READ_REFUSAL_STALE_WRITER: _ClassVar[WeightsReadRefusal]
    WEIGHTS_READ_REFUSAL_UNKNOWN_OBJECT: _ClassVar[WeightsReadRefusal]
    WEIGHTS_READ_REFUSAL_SOURCE_REF_MISMATCH: _ClassVar[WeightsReadRefusal]
    WEIGHTS_READ_REFUSAL_RANGE_INVALID: _ClassVar[WeightsReadRefusal]
    WEIGHTS_READ_REFUSAL_SOURCE_UNAVAILABLE: _ClassVar[WeightsReadRefusal]

class WeightsUploadOutcome(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    WEIGHTS_UPLOAD_OUTCOME_UNSPECIFIED: _ClassVar[WeightsUploadOutcome]
    WEIGHTS_UPLOAD_OUTCOME_UPLOADED: _ClassVar[WeightsUploadOutcome]
    WEIGHTS_UPLOAD_OUTCOME_ALREADY_PRESENT: _ClassVar[WeightsUploadOutcome]
    WEIGHTS_UPLOAD_OUTCOME_REFUSED: _ClassVar[WeightsUploadOutcome]

class WeightsUploadRefusal(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    WEIGHTS_UPLOAD_REFUSAL_UNSPECIFIED: _ClassVar[WeightsUploadRefusal]
    WEIGHTS_UPLOAD_REFUSAL_UNKNOWN_TRANSACTION: _ClassVar[WeightsUploadRefusal]
    WEIGHTS_UPLOAD_REFUSAL_STALE_WRITER: _ClassVar[WeightsUploadRefusal]
    WEIGHTS_UPLOAD_REFUSAL_UNKNOWN_OBJECT: _ClassVar[WeightsUploadRefusal]
    WEIGHTS_UPLOAD_REFUSAL_SOURCE_REF_MISMATCH: _ClassVar[WeightsUploadRefusal]
    WEIGHTS_UPLOAD_REFUSAL_SOURCE_UNAVAILABLE: _ClassVar[WeightsUploadRefusal]
    WEIGHTS_UPLOAD_REFUSAL_GRANT_REFUSED: _ClassVar[WeightsUploadRefusal]
    WEIGHTS_UPLOAD_REFUSAL_TRANSFER_FAILED: _ClassVar[WeightsUploadRefusal]

class WeightsTransferState(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    WEIGHTS_TRANSFER_STATE_UNSPECIFIED: _ClassVar[WeightsTransferState]
    WEIGHTS_TRANSFER_STATE_ACCEPTED: _ClassVar[WeightsTransferState]
    WEIGHTS_TRANSFER_STATE_UPLOADING: _ClassVar[WeightsTransferState]
    WEIGHTS_TRANSFER_STATE_UPLOADED: _ClassVar[WeightsTransferState]
    WEIGHTS_TRANSFER_STATE_ALREADY_PRESENT: _ClassVar[WeightsTransferState]
    WEIGHTS_TRANSFER_STATE_HELD: _ClassVar[WeightsTransferState]
    WEIGHTS_TRANSFER_STATE_FAILED: _ClassVar[WeightsTransferState]

class ModelSourceProvider(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    MODEL_SOURCE_PROVIDER_UNSPECIFIED: _ClassVar[ModelSourceProvider]
    MODEL_SOURCE_PROVIDER_HUGGING_FACE: _ClassVar[ModelSourceProvider]
    MODEL_SOURCE_PROVIDER_CIVITAI: _ClassVar[ModelSourceProvider]

class ModelSourceFileState(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    MODEL_SOURCE_FILE_STATE_UNSPECIFIED: _ClassVar[ModelSourceFileState]
    MODEL_SOURCE_FILE_STATE_ACCEPTED: _ClassVar[ModelSourceFileState]
    MODEL_SOURCE_FILE_STATE_DOWNLOADING: _ClassVar[ModelSourceFileState]
    MODEL_SOURCE_FILE_STATE_VERIFIED: _ClassVar[ModelSourceFileState]
    MODEL_SOURCE_FILE_STATE_FAILED: _ClassVar[ModelSourceFileState]
    MODEL_SOURCE_FILE_STATE_CONVERTED: _ClassVar[ModelSourceFileState]

class ModelSourcePrepareOutcome(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    MODEL_SOURCE_PREPARE_OUTCOME_UNSPECIFIED: _ClassVar[ModelSourcePrepareOutcome]
    MODEL_SOURCE_PREPARE_OUTCOME_PREPARED: _ClassVar[ModelSourcePrepareOutcome]
    MODEL_SOURCE_PREPARE_OUTCOME_REPLAYED: _ClassVar[ModelSourcePrepareOutcome]
    MODEL_SOURCE_PREPARE_OUTCOME_REFUSED: _ClassVar[ModelSourcePrepareOutcome]
    MODEL_SOURCE_PREPARE_OUTCOME_INCOMPLETE: _ClassVar[ModelSourcePrepareOutcome]

class LocalPackageFileState(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    LOCAL_PACKAGE_FILE_STATE_UNSPECIFIED: _ClassVar[LocalPackageFileState]
    LOCAL_PACKAGE_FILE_STATE_RECEIVING: _ClassVar[LocalPackageFileState]
    LOCAL_PACKAGE_FILE_STATE_VERIFIED: _ClassVar[LocalPackageFileState]
    LOCAL_PACKAGE_FILE_STATE_REFUSED: _ClassVar[LocalPackageFileState]

class LocalPackageAbortOutcome(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    LOCAL_PACKAGE_ABORT_OUTCOME_UNSPECIFIED: _ClassVar[LocalPackageAbortOutcome]
    LOCAL_PACKAGE_ABORT_OUTCOME_ABANDONED: _ClassVar[LocalPackageAbortOutcome]
    LOCAL_PACKAGE_ABORT_OUTCOME_REPLAYED: _ClassVar[LocalPackageAbortOutcome]
    LOCAL_PACKAGE_ABORT_OUTCOME_REFUSED: _ClassVar[LocalPackageAbortOutcome]

class WeightsFinalizeDisposition(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    WEIGHTS_FINALIZE_DISPOSITION_UNSPECIFIED: _ClassVar[WeightsFinalizeDisposition]
    WEIGHTS_FINALIZE_DISPOSITION_ADOPT: _ClassVar[WeightsFinalizeDisposition]
    WEIGHTS_FINALIZE_DISPOSITION_ABANDON: _ClassVar[WeightsFinalizeDisposition]
    WEIGHTS_FINALIZE_DISPOSITION_ABANDON_UNCOMMITTED: _ClassVar[WeightsFinalizeDisposition]

class WeightsFinalizeOutcome(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    WEIGHTS_FINALIZE_OUTCOME_UNSPECIFIED: _ClassVar[WeightsFinalizeOutcome]
    WEIGHTS_FINALIZE_OUTCOME_ADOPTED: _ClassVar[WeightsFinalizeOutcome]
    WEIGHTS_FINALIZE_OUTCOME_ABANDONED: _ClassVar[WeightsFinalizeOutcome]
MACHINE_EXECUTION_ACTION_UNSPECIFIED: MachineExecutionAction
MACHINE_EXECUTION_ACTION_PAUSE: MachineExecutionAction
MACHINE_EXECUTION_ACTION_RESUME: MachineExecutionAction
MACHINE_EXECUTION_ACTION_CANCEL: MachineExecutionAction
CHILD_CALL_STATE_UNSPECIFIED: ChildCallState
CHILD_CALL_STATE_PENDING: ChildCallState
CHILD_CALL_STATE_SUCCEEDED: ChildCallState
CHILD_CALL_STATE_REFUSED: ChildCallState
CHILD_CALL_STATE_FAILED: ChildCallState
CHILD_CALL_STATE_CANCELED: ChildCallState
NATIVE_SOURCE_PHASE_UNSPECIFIED: NativeSourcePhase
NATIVE_SOURCE_PHASE_RESOLVE: NativeSourcePhase
NATIVE_SOURCE_PHASE_EXECUTE: NativeSourcePhase
NATIVE_SOURCE_PHASE_CANCEL: NativeSourcePhase
NATIVE_SOURCE_OPERATION_UNSPECIFIED: NativeSourceOperation
NATIVE_SOURCE_OPERATION_HUGGINGFACE: NativeSourceOperation
NATIVE_SOURCE_OPERATION_CIVITAI: NativeSourceOperation
NATIVE_SOURCE_OPERATION_CONVERT: NativeSourceOperation
NATIVE_SOURCE_OPERATION_SOURCE_FILES: NativeSourceOperation
NATIVE_SOURCE_OPERATION_COMMIT_FILE: NativeSourceOperation
NATIVE_SOURCE_STATE_UNSPECIFIED: NativeSourceState
NATIVE_SOURCE_STATE_RESOLVED: NativeSourceState
NATIVE_SOURCE_STATE_SUCCEEDED: NativeSourceState
NATIVE_SOURCE_STATE_FAILED: NativeSourceState
NATIVE_SOURCE_STATE_CANCELED: NativeSourceState
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
ATTEMPT_STATE_QUEUED: AttemptState
ATTEMPT_STATE_DEVICE_RELEASED: AttemptState
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
CAUSE_CODE_ADMISSION_EPOCH_STALE: CauseCode
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
CLAIM_REJECTION_PROTOCOL_INCOMPATIBLE: ClaimRejection
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
PREPARE_STAGE_UNSPECIFIED: PrepareStage
PREPARE_STAGE_RESOLVED: PrepareStage
PREPARE_STAGE_DOWNLOADING: PrepareStage
PREPARE_STAGE_PREPARING: PrepareStage
PREPARE_STAGE_PREPARED: PrepareStage
PREPARE_STAGE_REFUSED: PrepareStage
WEIGHTS_HOST_STAGE_UNSPECIFIED: WeightsHostStage
WEIGHTS_HOST_STAGE_INTENT: WeightsHostStage
WEIGHTS_HOST_STAGE_RECEIPT: WeightsHostStage
WEIGHTS_HOST_STAGE_CHECKPOINT: WeightsHostStage
WEIGHTS_HOST_OUTCOME_UNSPECIFIED: WeightsHostOutcome
WEIGHTS_HOST_OUTCOME_RECORDED: WeightsHostOutcome
WEIGHTS_HOST_OUTCOME_REPLAYED: WeightsHostOutcome
WEIGHTS_HOST_OUTCOME_REFUSED: WeightsHostOutcome
WEIGHTS_HOST_REFUSAL_UNSPECIFIED: WeightsHostRefusal
WEIGHTS_HOST_REFUSAL_UNKNOWN_ATTEMPT: WeightsHostRefusal
WEIGHTS_HOST_REFUSAL_INTENT_CONFLICT: WeightsHostRefusal
WEIGHTS_HOST_REFUSAL_STALE_WRITER: WeightsHostRefusal
WEIGHTS_HOST_REFUSAL_RECEIPT_CONFLICT: WeightsHostRefusal
WEIGHTS_HOST_REFUSAL_INVENTORY_INVALID: WeightsHostRefusal
WEIGHTS_TRANSACTION_STATE_UNSPECIFIED: WeightsTransactionState
WEIGHTS_TRANSACTION_STATE_INTENT: WeightsTransactionState
WEIGHTS_TRANSACTION_STATE_RECEIPT: WeightsTransactionState
WEIGHTS_READ_OUTCOME_UNSPECIFIED: WeightsReadOutcome
WEIGHTS_READ_OUTCOME_DATA: WeightsReadOutcome
WEIGHTS_READ_OUTCOME_REFUSED: WeightsReadOutcome
WEIGHTS_READ_REFUSAL_UNSPECIFIED: WeightsReadRefusal
WEIGHTS_READ_REFUSAL_UNKNOWN_TRANSACTION: WeightsReadRefusal
WEIGHTS_READ_REFUSAL_STALE_WRITER: WeightsReadRefusal
WEIGHTS_READ_REFUSAL_UNKNOWN_OBJECT: WeightsReadRefusal
WEIGHTS_READ_REFUSAL_SOURCE_REF_MISMATCH: WeightsReadRefusal
WEIGHTS_READ_REFUSAL_RANGE_INVALID: WeightsReadRefusal
WEIGHTS_READ_REFUSAL_SOURCE_UNAVAILABLE: WeightsReadRefusal
WEIGHTS_UPLOAD_OUTCOME_UNSPECIFIED: WeightsUploadOutcome
WEIGHTS_UPLOAD_OUTCOME_UPLOADED: WeightsUploadOutcome
WEIGHTS_UPLOAD_OUTCOME_ALREADY_PRESENT: WeightsUploadOutcome
WEIGHTS_UPLOAD_OUTCOME_REFUSED: WeightsUploadOutcome
WEIGHTS_UPLOAD_REFUSAL_UNSPECIFIED: WeightsUploadRefusal
WEIGHTS_UPLOAD_REFUSAL_UNKNOWN_TRANSACTION: WeightsUploadRefusal
WEIGHTS_UPLOAD_REFUSAL_STALE_WRITER: WeightsUploadRefusal
WEIGHTS_UPLOAD_REFUSAL_UNKNOWN_OBJECT: WeightsUploadRefusal
WEIGHTS_UPLOAD_REFUSAL_SOURCE_REF_MISMATCH: WeightsUploadRefusal
WEIGHTS_UPLOAD_REFUSAL_SOURCE_UNAVAILABLE: WeightsUploadRefusal
WEIGHTS_UPLOAD_REFUSAL_GRANT_REFUSED: WeightsUploadRefusal
WEIGHTS_UPLOAD_REFUSAL_TRANSFER_FAILED: WeightsUploadRefusal
WEIGHTS_TRANSFER_STATE_UNSPECIFIED: WeightsTransferState
WEIGHTS_TRANSFER_STATE_ACCEPTED: WeightsTransferState
WEIGHTS_TRANSFER_STATE_UPLOADING: WeightsTransferState
WEIGHTS_TRANSFER_STATE_UPLOADED: WeightsTransferState
WEIGHTS_TRANSFER_STATE_ALREADY_PRESENT: WeightsTransferState
WEIGHTS_TRANSFER_STATE_HELD: WeightsTransferState
WEIGHTS_TRANSFER_STATE_FAILED: WeightsTransferState
MODEL_SOURCE_PROVIDER_UNSPECIFIED: ModelSourceProvider
MODEL_SOURCE_PROVIDER_HUGGING_FACE: ModelSourceProvider
MODEL_SOURCE_PROVIDER_CIVITAI: ModelSourceProvider
MODEL_SOURCE_FILE_STATE_UNSPECIFIED: ModelSourceFileState
MODEL_SOURCE_FILE_STATE_ACCEPTED: ModelSourceFileState
MODEL_SOURCE_FILE_STATE_DOWNLOADING: ModelSourceFileState
MODEL_SOURCE_FILE_STATE_VERIFIED: ModelSourceFileState
MODEL_SOURCE_FILE_STATE_FAILED: ModelSourceFileState
MODEL_SOURCE_FILE_STATE_CONVERTED: ModelSourceFileState
MODEL_SOURCE_PREPARE_OUTCOME_UNSPECIFIED: ModelSourcePrepareOutcome
MODEL_SOURCE_PREPARE_OUTCOME_PREPARED: ModelSourcePrepareOutcome
MODEL_SOURCE_PREPARE_OUTCOME_REPLAYED: ModelSourcePrepareOutcome
MODEL_SOURCE_PREPARE_OUTCOME_REFUSED: ModelSourcePrepareOutcome
MODEL_SOURCE_PREPARE_OUTCOME_INCOMPLETE: ModelSourcePrepareOutcome
LOCAL_PACKAGE_FILE_STATE_UNSPECIFIED: LocalPackageFileState
LOCAL_PACKAGE_FILE_STATE_RECEIVING: LocalPackageFileState
LOCAL_PACKAGE_FILE_STATE_VERIFIED: LocalPackageFileState
LOCAL_PACKAGE_FILE_STATE_REFUSED: LocalPackageFileState
LOCAL_PACKAGE_ABORT_OUTCOME_UNSPECIFIED: LocalPackageAbortOutcome
LOCAL_PACKAGE_ABORT_OUTCOME_ABANDONED: LocalPackageAbortOutcome
LOCAL_PACKAGE_ABORT_OUTCOME_REPLAYED: LocalPackageAbortOutcome
LOCAL_PACKAGE_ABORT_OUTCOME_REFUSED: LocalPackageAbortOutcome
WEIGHTS_FINALIZE_DISPOSITION_UNSPECIFIED: WeightsFinalizeDisposition
WEIGHTS_FINALIZE_DISPOSITION_ADOPT: WeightsFinalizeDisposition
WEIGHTS_FINALIZE_DISPOSITION_ABANDON: WeightsFinalizeDisposition
WEIGHTS_FINALIZE_DISPOSITION_ABANDON_UNCOMMITTED: WeightsFinalizeDisposition
WEIGHTS_FINALIZE_OUTCOME_UNSPECIFIED: WeightsFinalizeOutcome
WEIGHTS_FINALIZE_OUTCOME_ADOPTED: WeightsFinalizeOutcome
WEIGHTS_FINALIZE_OUTCOME_ABANDONED: WeightsFinalizeOutcome

class WeightsHostEvent(_message.Message):
    __slots__ = ("intent", "receipt", "checkpoint")
    INTENT_FIELD_NUMBER: _ClassVar[int]
    RECEIPT_FIELD_NUMBER: _ClassVar[int]
    CHECKPOINT_FIELD_NUMBER: _ClassVar[int]
    intent: WeightsIntentFrame
    receipt: WeightsReceiptFrame
    checkpoint: WeightsCheckpointFrame
    def __init__(self, intent: _Optional[_Union[WeightsIntentFrame, _Mapping]] = ..., receipt: _Optional[_Union[WeightsReceiptFrame, _Mapping]] = ..., checkpoint: _Optional[_Union[WeightsCheckpointFrame, _Mapping]] = ...) -> None: ...

class ProtocolInfoRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class KeepRentalAliveRequest(_message.Message):
    __slots__ = ("claim", "request_id")
    CLAIM_FIELD_NUMBER: _ClassVar[int]
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    claim: Claim
    request_id: str
    def __init__(self, claim: _Optional[_Union[Claim, _Mapping]] = ..., request_id: _Optional[str] = ...) -> None: ...

class KeepRentalAliveResult(_message.Message):
    __slots__ = ("request_id", "worker_id", "worker_boot_id", "acknowledged_at_unix_ms", "idle_deadline_unix_ms")
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    WORKER_ID_FIELD_NUMBER: _ClassVar[int]
    WORKER_BOOT_ID_FIELD_NUMBER: _ClassVar[int]
    ACKNOWLEDGED_AT_UNIX_MS_FIELD_NUMBER: _ClassVar[int]
    IDLE_DEADLINE_UNIX_MS_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    worker_id: str
    worker_boot_id: str
    acknowledged_at_unix_ms: int
    idle_deadline_unix_ms: int
    def __init__(self, request_id: _Optional[str] = ..., worker_id: _Optional[str] = ..., worker_boot_id: _Optional[str] = ..., acknowledged_at_unix_ms: _Optional[int] = ..., idle_deadline_unix_ms: _Optional[int] = ...) -> None: ...

class MachineExecutionCapture(_message.Message):
    __slots__ = ("root_revision_digest", "revisions", "bindings", "published_revisions", "model_defaults")
    ROOT_REVISION_DIGEST_FIELD_NUMBER: _ClassVar[int]
    REVISIONS_FIELD_NUMBER: _ClassVar[int]
    BINDINGS_FIELD_NUMBER: _ClassVar[int]
    PUBLISHED_REVISIONS_FIELD_NUMBER: _ClassVar[int]
    MODEL_DEFAULTS_FIELD_NUMBER: _ClassVar[int]
    root_revision_digest: bytes
    revisions: _containers.RepeatedCompositeFieldContainer[LocalPackageRevision]
    bindings: _containers.RepeatedCompositeFieldContainer[MachineCallableBinding]
    published_revisions: _containers.RepeatedCompositeFieldContainer[PublishedPackageRevision]
    model_defaults: _containers.RepeatedCompositeFieldContainer[MachineModelDefault]
    def __init__(self, root_revision_digest: _Optional[bytes] = ..., revisions: _Optional[_Iterable[_Union[LocalPackageRevision, _Mapping]]] = ..., bindings: _Optional[_Iterable[_Union[MachineCallableBinding, _Mapping]]] = ..., published_revisions: _Optional[_Iterable[_Union[PublishedPackageRevision, _Mapping]]] = ..., model_defaults: _Optional[_Iterable[_Union[MachineModelDefault, _Mapping]]] = ...) -> None: ...

class MachineModelDefault(_message.Message):
    __slots__ = ("callee_revision_digest", "entrypoint", "parameter", "public_origin", "rungs", "unavailable_code")
    CALLEE_REVISION_DIGEST_FIELD_NUMBER: _ClassVar[int]
    ENTRYPOINT_FIELD_NUMBER: _ClassVar[int]
    PARAMETER_FIELD_NUMBER: _ClassVar[int]
    PUBLIC_ORIGIN_FIELD_NUMBER: _ClassVar[int]
    RUNGS_FIELD_NUMBER: _ClassVar[int]
    UNAVAILABLE_CODE_FIELD_NUMBER: _ClassVar[int]
    callee_revision_digest: bytes
    entrypoint: str
    parameter: str
    public_origin: str
    rungs: _containers.RepeatedCompositeFieldContainer[MachineModelDefaultRung]
    unavailable_code: str
    def __init__(self, callee_revision_digest: _Optional[bytes] = ..., entrypoint: _Optional[str] = ..., parameter: _Optional[str] = ..., public_origin: _Optional[str] = ..., rungs: _Optional[_Iterable[_Union[MachineModelDefaultRung, _Mapping]]] = ..., unavailable_code: _Optional[str] = ...) -> None: ...

class MachineModelDefaultRung(_message.Message):
    __slots__ = ("gpu", "repository", "manifest")
    GPU_FIELD_NUMBER: _ClassVar[int]
    REPOSITORY_FIELD_NUMBER: _ClassVar[int]
    MANIFEST_FIELD_NUMBER: _ClassVar[int]
    gpu: str
    repository: str
    manifest: Ref
    def __init__(self, gpu: _Optional[str] = ..., repository: _Optional[str] = ..., manifest: _Optional[_Union[Ref, _Mapping]] = ...) -> None: ...

class PublishedPackageRevision(_message.Message):
    __slots__ = ("package", "environment", "package_interface")
    PACKAGE_FIELD_NUMBER: _ClassVar[int]
    ENVIRONMENT_FIELD_NUMBER: _ClassVar[int]
    PACKAGE_INTERFACE_FIELD_NUMBER: _ClassVar[int]
    package: PackageSelection
    environment: Environment
    package_interface: Ref
    def __init__(self, package: _Optional[_Union[PackageSelection, _Mapping]] = ..., environment: _Optional[_Union[Environment, _Mapping]] = ..., package_interface: _Optional[_Union[Ref, _Mapping]] = ...) -> None: ...

class MachineCallableBinding(_message.Message):
    __slots__ = ("caller_revision_digest", "interface_digest", "module", "export", "callee_revision_digest", "entrypoint")
    CALLER_REVISION_DIGEST_FIELD_NUMBER: _ClassVar[int]
    INTERFACE_DIGEST_FIELD_NUMBER: _ClassVar[int]
    MODULE_FIELD_NUMBER: _ClassVar[int]
    EXPORT_FIELD_NUMBER: _ClassVar[int]
    CALLEE_REVISION_DIGEST_FIELD_NUMBER: _ClassVar[int]
    ENTRYPOINT_FIELD_NUMBER: _ClassVar[int]
    caller_revision_digest: bytes
    interface_digest: bytes
    module: str
    export: str
    callee_revision_digest: bytes
    entrypoint: str
    def __init__(self, caller_revision_digest: _Optional[bytes] = ..., interface_digest: _Optional[bytes] = ..., module: _Optional[str] = ..., export: _Optional[str] = ..., callee_revision_digest: _Optional[bytes] = ..., entrypoint: _Optional[str] = ...) -> None: ...

class MachineExecutionWorkspaceQuery(_message.Message):
    __slots__ = ("claim",)
    CLAIM_FIELD_NUMBER: _ClassVar[int]
    claim: Claim
    def __init__(self, claim: _Optional[_Union[Claim, _Mapping]] = ...) -> None: ...

class MachineExecutionWorkspace(_message.Message):
    __slots__ = ("worker_id", "worker_boot_id", "execution_workspace_id")
    WORKER_ID_FIELD_NUMBER: _ClassVar[int]
    WORKER_BOOT_ID_FIELD_NUMBER: _ClassVar[int]
    EXECUTION_WORKSPACE_ID_FIELD_NUMBER: _ClassVar[int]
    worker_id: str
    worker_boot_id: str
    execution_workspace_id: str
    def __init__(self, worker_id: _Optional[str] = ..., worker_boot_id: _Optional[str] = ..., execution_workspace_id: _Optional[str] = ...) -> None: ...

class MachineExecutionSubmit(_message.Message):
    __slots__ = ("claim", "submission_id", "capture_digest", "capture_canonical_bytes", "offer", "max_attempts", "retry_delay_ms", "prepared_state", "payload_canonical_bytes", "publication_authorization_id", "expected_execution_workspace_id")
    CLAIM_FIELD_NUMBER: _ClassVar[int]
    SUBMISSION_ID_FIELD_NUMBER: _ClassVar[int]
    CAPTURE_DIGEST_FIELD_NUMBER: _ClassVar[int]
    CAPTURE_CANONICAL_BYTES_FIELD_NUMBER: _ClassVar[int]
    OFFER_FIELD_NUMBER: _ClassVar[int]
    MAX_ATTEMPTS_FIELD_NUMBER: _ClassVar[int]
    RETRY_DELAY_MS_FIELD_NUMBER: _ClassVar[int]
    PREPARED_STATE_FIELD_NUMBER: _ClassVar[int]
    PAYLOAD_CANONICAL_BYTES_FIELD_NUMBER: _ClassVar[int]
    PUBLICATION_AUTHORIZATION_ID_FIELD_NUMBER: _ClassVar[int]
    EXPECTED_EXECUTION_WORKSPACE_ID_FIELD_NUMBER: _ClassVar[int]
    claim: Claim
    submission_id: str
    capture_digest: bytes
    capture_canonical_bytes: bytes
    offer: AttemptOffer
    max_attempts: int
    retry_delay_ms: int
    prepared_state: DesiredWorkerState
    payload_canonical_bytes: bytes
    publication_authorization_id: str
    expected_execution_workspace_id: str
    def __init__(self, claim: _Optional[_Union[Claim, _Mapping]] = ..., submission_id: _Optional[str] = ..., capture_digest: _Optional[bytes] = ..., capture_canonical_bytes: _Optional[bytes] = ..., offer: _Optional[_Union[AttemptOffer, _Mapping]] = ..., max_attempts: _Optional[int] = ..., retry_delay_ms: _Optional[int] = ..., prepared_state: _Optional[_Union[DesiredWorkerState, _Mapping]] = ..., payload_canonical_bytes: _Optional[bytes] = ..., publication_authorization_id: _Optional[str] = ..., expected_execution_workspace_id: _Optional[str] = ...) -> None: ...

class MachineExecutionReceipt(_message.Message):
    __slots__ = ("request_id", "submission_id", "capture_digest", "invocation_spec_digest", "accepted_at_ms", "worker_id", "worker_boot_id", "execution_workspace_id", "publication_authorization_id")
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    SUBMISSION_ID_FIELD_NUMBER: _ClassVar[int]
    CAPTURE_DIGEST_FIELD_NUMBER: _ClassVar[int]
    INVOCATION_SPEC_DIGEST_FIELD_NUMBER: _ClassVar[int]
    ACCEPTED_AT_MS_FIELD_NUMBER: _ClassVar[int]
    WORKER_ID_FIELD_NUMBER: _ClassVar[int]
    WORKER_BOOT_ID_FIELD_NUMBER: _ClassVar[int]
    EXECUTION_WORKSPACE_ID_FIELD_NUMBER: _ClassVar[int]
    PUBLICATION_AUTHORIZATION_ID_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    submission_id: str
    capture_digest: bytes
    invocation_spec_digest: bytes
    accepted_at_ms: int
    worker_id: str
    worker_boot_id: str
    execution_workspace_id: str
    publication_authorization_id: str
    def __init__(self, request_id: _Optional[str] = ..., submission_id: _Optional[str] = ..., capture_digest: _Optional[bytes] = ..., invocation_spec_digest: _Optional[bytes] = ..., accepted_at_ms: _Optional[int] = ..., worker_id: _Optional[str] = ..., worker_boot_id: _Optional[str] = ..., execution_workspace_id: _Optional[str] = ..., publication_authorization_id: _Optional[str] = ...) -> None: ...

class MachineExecutionQuery(_message.Message):
    __slots__ = ("claim", "request_id", "expected_execution_workspace_id")
    CLAIM_FIELD_NUMBER: _ClassVar[int]
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    EXPECTED_EXECUTION_WORKSPACE_ID_FIELD_NUMBER: _ClassVar[int]
    claim: Claim
    request_id: str
    expected_execution_workspace_id: str
    def __init__(self, claim: _Optional[_Union[Claim, _Mapping]] = ..., request_id: _Optional[str] = ..., expected_execution_workspace_id: _Optional[str] = ...) -> None: ...

class MachineExecutionState(_message.Message):
    __slots__ = ("request_id", "attempt_ordinal", "generation", "state", "collected", "sequence", "worker_id", "worker_boot_id", "execution_workspace_id")
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    ATTEMPT_ORDINAL_FIELD_NUMBER: _ClassVar[int]
    GENERATION_FIELD_NUMBER: _ClassVar[int]
    STATE_FIELD_NUMBER: _ClassVar[int]
    COLLECTED_FIELD_NUMBER: _ClassVar[int]
    SEQUENCE_FIELD_NUMBER: _ClassVar[int]
    WORKER_ID_FIELD_NUMBER: _ClassVar[int]
    WORKER_BOOT_ID_FIELD_NUMBER: _ClassVar[int]
    EXECUTION_WORKSPACE_ID_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    attempt_ordinal: int
    generation: int
    state: str
    collected: bool
    sequence: int
    worker_id: str
    worker_boot_id: str
    execution_workspace_id: str
    def __init__(self, request_id: _Optional[str] = ..., attempt_ordinal: _Optional[int] = ..., generation: _Optional[int] = ..., state: _Optional[str] = ..., collected: bool = ..., sequence: _Optional[int] = ..., worker_id: _Optional[str] = ..., worker_boot_id: _Optional[str] = ..., execution_workspace_id: _Optional[str] = ...) -> None: ...

class MachineExecutionEventsQuery(_message.Message):
    __slots__ = ("execution", "after", "limit")
    EXECUTION_FIELD_NUMBER: _ClassVar[int]
    AFTER_FIELD_NUMBER: _ClassVar[int]
    LIMIT_FIELD_NUMBER: _ClassVar[int]
    execution: MachineExecutionQuery
    after: int
    limit: int
    def __init__(self, execution: _Optional[_Union[MachineExecutionQuery, _Mapping]] = ..., after: _Optional[int] = ..., limit: _Optional[int] = ...) -> None: ...

class MachineExecutionEvent(_message.Message):
    __slots__ = ("sequence", "attempt_ordinal", "at_ms", "kind", "body_canonical_bytes")
    SEQUENCE_FIELD_NUMBER: _ClassVar[int]
    ATTEMPT_ORDINAL_FIELD_NUMBER: _ClassVar[int]
    AT_MS_FIELD_NUMBER: _ClassVar[int]
    KIND_FIELD_NUMBER: _ClassVar[int]
    BODY_CANONICAL_BYTES_FIELD_NUMBER: _ClassVar[int]
    sequence: int
    attempt_ordinal: int
    at_ms: int
    kind: str
    body_canonical_bytes: bytes
    def __init__(self, sequence: _Optional[int] = ..., attempt_ordinal: _Optional[int] = ..., at_ms: _Optional[int] = ..., kind: _Optional[str] = ..., body_canonical_bytes: _Optional[bytes] = ...) -> None: ...

class MachineExecutionEventPage(_message.Message):
    __slots__ = ("events", "next_after", "head_sequence", "compacted_through")
    EVENTS_FIELD_NUMBER: _ClassVar[int]
    NEXT_AFTER_FIELD_NUMBER: _ClassVar[int]
    HEAD_SEQUENCE_FIELD_NUMBER: _ClassVar[int]
    COMPACTED_THROUGH_FIELD_NUMBER: _ClassVar[int]
    events: _containers.RepeatedCompositeFieldContainer[MachineExecutionEvent]
    next_after: int
    head_sequence: int
    compacted_through: int
    def __init__(self, events: _Optional[_Iterable[_Union[MachineExecutionEvent, _Mapping]]] = ..., next_after: _Optional[int] = ..., head_sequence: _Optional[int] = ..., compacted_through: _Optional[int] = ...) -> None: ...

class MachineExecutionControl(_message.Message):
    __slots__ = ("execution", "command_id", "expected_generation", "action")
    EXECUTION_FIELD_NUMBER: _ClassVar[int]
    COMMAND_ID_FIELD_NUMBER: _ClassVar[int]
    EXPECTED_GENERATION_FIELD_NUMBER: _ClassVar[int]
    ACTION_FIELD_NUMBER: _ClassVar[int]
    execution: MachineExecutionQuery
    command_id: str
    expected_generation: int
    action: MachineExecutionAction
    def __init__(self, execution: _Optional[_Union[MachineExecutionQuery, _Mapping]] = ..., command_id: _Optional[str] = ..., expected_generation: _Optional[int] = ..., action: _Optional[_Union[MachineExecutionAction, str]] = ...) -> None: ...

class MachineExecutionCollect(_message.Message):
    __slots__ = ("execution", "attempt_ordinal")
    EXECUTION_FIELD_NUMBER: _ClassVar[int]
    ATTEMPT_ORDINAL_FIELD_NUMBER: _ClassVar[int]
    execution: MachineExecutionQuery
    attempt_ordinal: int
    def __init__(self, execution: _Optional[_Union[MachineExecutionQuery, _Mapping]] = ..., attempt_ordinal: _Optional[int] = ...) -> None: ...

class MachineExecutionCollectionAck(_message.Message):
    __slots__ = ("execution", "outcome")
    EXECUTION_FIELD_NUMBER: _ClassVar[int]
    OUTCOME_FIELD_NUMBER: _ClassVar[int]
    execution: MachineExecutionQuery
    outcome: AttemptOutcomeAck
    def __init__(self, execution: _Optional[_Union[MachineExecutionQuery, _Mapping]] = ..., outcome: _Optional[_Union[AttemptOutcomeAck, _Mapping]] = ...) -> None: ...

class WorkspaceImportBegin(_message.Message):
    __slots__ = ("claim", "migration_id", "record_count", "source_schema")
    CLAIM_FIELD_NUMBER: _ClassVar[int]
    MIGRATION_ID_FIELD_NUMBER: _ClassVar[int]
    RECORD_COUNT_FIELD_NUMBER: _ClassVar[int]
    SOURCE_SCHEMA_FIELD_NUMBER: _ClassVar[int]
    claim: Claim
    migration_id: str
    record_count: int
    source_schema: int
    def __init__(self, claim: _Optional[_Union[Claim, _Mapping]] = ..., migration_id: _Optional[str] = ..., record_count: _Optional[int] = ..., source_schema: _Optional[int] = ...) -> None: ...

class WorkspaceImportRecord(_message.Message):
    __slots__ = ("index", "kind", "length", "digest")
    INDEX_FIELD_NUMBER: _ClassVar[int]
    KIND_FIELD_NUMBER: _ClassVar[int]
    LENGTH_FIELD_NUMBER: _ClassVar[int]
    DIGEST_FIELD_NUMBER: _ClassVar[int]
    index: int
    kind: str
    length: int
    digest: bytes
    def __init__(self, index: _Optional[int] = ..., kind: _Optional[str] = ..., length: _Optional[int] = ..., digest: _Optional[bytes] = ...) -> None: ...

class WorkspaceImportChunk(_message.Message):
    __slots__ = ("index", "offset", "data")
    INDEX_FIELD_NUMBER: _ClassVar[int]
    OFFSET_FIELD_NUMBER: _ClassVar[int]
    DATA_FIELD_NUMBER: _ClassVar[int]
    index: int
    offset: int
    data: bytes
    def __init__(self, index: _Optional[int] = ..., offset: _Optional[int] = ..., data: _Optional[bytes] = ...) -> None: ...

class WorkspaceImportCommit(_message.Message):
    __slots__ = ("digest", "record_count")
    DIGEST_FIELD_NUMBER: _ClassVar[int]
    RECORD_COUNT_FIELD_NUMBER: _ClassVar[int]
    digest: bytes
    record_count: int
    def __init__(self, digest: _Optional[bytes] = ..., record_count: _Optional[int] = ...) -> None: ...

class WorkspaceImportFrame(_message.Message):
    __slots__ = ("begin", "record", "chunk", "commit")
    BEGIN_FIELD_NUMBER: _ClassVar[int]
    RECORD_FIELD_NUMBER: _ClassVar[int]
    CHUNK_FIELD_NUMBER: _ClassVar[int]
    COMMIT_FIELD_NUMBER: _ClassVar[int]
    begin: WorkspaceImportBegin
    record: WorkspaceImportRecord
    chunk: WorkspaceImportChunk
    commit: WorkspaceImportCommit
    def __init__(self, begin: _Optional[_Union[WorkspaceImportBegin, _Mapping]] = ..., record: _Optional[_Union[WorkspaceImportRecord, _Mapping]] = ..., chunk: _Optional[_Union[WorkspaceImportChunk, _Mapping]] = ..., commit: _Optional[_Union[WorkspaceImportCommit, _Mapping]] = ...) -> None: ...

class WorkspaceImportResult(_message.Message):
    __slots__ = ("migration_id", "record_count", "imported")
    MIGRATION_ID_FIELD_NUMBER: _ClassVar[int]
    RECORD_COUNT_FIELD_NUMBER: _ClassVar[int]
    IMPORTED_FIELD_NUMBER: _ClassVar[int]
    migration_id: str
    record_count: int
    imported: bool
    def __init__(self, migration_id: _Optional[str] = ..., record_count: _Optional[int] = ..., imported: bool = ...) -> None: ...

class WorkspaceActivationCall(_message.Message):
    __slots__ = ("claim", "migration_id")
    CLAIM_FIELD_NUMBER: _ClassVar[int]
    MIGRATION_ID_FIELD_NUMBER: _ClassVar[int]
    claim: Claim
    migration_id: str
    def __init__(self, claim: _Optional[_Union[Claim, _Mapping]] = ..., migration_id: _Optional[str] = ...) -> None: ...

class ProtocolInfoResult(_message.Message):
    __slots__ = ("wire_minor", "minimum_wire_minor", "supports_mixed_model_inputs", "supports_model_materialization_recovery", "supports_local_installation_reuse", "supports_rental_keepalive")
    WIRE_MINOR_FIELD_NUMBER: _ClassVar[int]
    MINIMUM_WIRE_MINOR_FIELD_NUMBER: _ClassVar[int]
    SUPPORTS_MIXED_MODEL_INPUTS_FIELD_NUMBER: _ClassVar[int]
    SUPPORTS_MODEL_MATERIALIZATION_RECOVERY_FIELD_NUMBER: _ClassVar[int]
    SUPPORTS_LOCAL_INSTALLATION_REUSE_FIELD_NUMBER: _ClassVar[int]
    SUPPORTS_RENTAL_KEEPALIVE_FIELD_NUMBER: _ClassVar[int]
    wire_minor: int
    minimum_wire_minor: int
    supports_mixed_model_inputs: bool
    supports_model_materialization_recovery: bool
    supports_local_installation_reuse: bool
    supports_rental_keepalive: bool
    def __init__(self, wire_minor: _Optional[int] = ..., minimum_wire_minor: _Optional[int] = ..., supports_mixed_model_inputs: bool = ..., supports_model_materialization_recovery: bool = ..., supports_local_installation_reuse: bool = ..., supports_rental_keepalive: bool = ...) -> None: ...

class PreparePackageSetCall(_message.Message):
    __slots__ = ("claim", "package_set", "application", "model_slot_paths", "image_inventory", "locked_requirements", "supports_model_materialization_recovery", "python_requires", "python_version")
    CLAIM_FIELD_NUMBER: _ClassVar[int]
    PACKAGE_SET_FIELD_NUMBER: _ClassVar[int]
    APPLICATION_FIELD_NUMBER: _ClassVar[int]
    MODEL_SLOT_PATHS_FIELD_NUMBER: _ClassVar[int]
    IMAGE_INVENTORY_FIELD_NUMBER: _ClassVar[int]
    LOCKED_REQUIREMENTS_FIELD_NUMBER: _ClassVar[int]
    SUPPORTS_MODEL_MATERIALIZATION_RECOVERY_FIELD_NUMBER: _ClassVar[int]
    PYTHON_REQUIRES_FIELD_NUMBER: _ClassVar[int]
    PYTHON_VERSION_FIELD_NUMBER: _ClassVar[int]
    claim: Claim
    package_set: DesiredPackageSet
    application: str
    model_slot_paths: _containers.RepeatedScalarFieldContainer[str]
    image_inventory: ImageInventory
    locked_requirements: bytes
    supports_model_materialization_recovery: bool
    python_requires: str
    python_version: str
    def __init__(self, claim: _Optional[_Union[Claim, _Mapping]] = ..., package_set: _Optional[_Union[DesiredPackageSet, _Mapping]] = ..., application: _Optional[str] = ..., model_slot_paths: _Optional[_Iterable[str]] = ..., image_inventory: _Optional[_Union[ImageInventory, _Mapping]] = ..., locked_requirements: _Optional[bytes] = ..., supports_model_materialization_recovery: bool = ..., python_requires: _Optional[str] = ..., python_version: _Optional[str] = ...) -> None: ...

class PrepareLocalPackageCall(_message.Message):
    __slots__ = ("claim", "local_package_set")
    CLAIM_FIELD_NUMBER: _ClassVar[int]
    LOCAL_PACKAGE_SET_FIELD_NUMBER: _ClassVar[int]
    claim: Claim
    local_package_set: DesiredLocalPackageSet
    def __init__(self, claim: _Optional[_Union[Claim, _Mapping]] = ..., local_package_set: _Optional[_Union[DesiredLocalPackageSet, _Mapping]] = ...) -> None: ...

class PreparePrivatePlacementCall(_message.Message):
    __slots__ = ("claim", "private_placement_set", "supports_model_materialization_recovery")
    CLAIM_FIELD_NUMBER: _ClassVar[int]
    PRIVATE_PLACEMENT_SET_FIELD_NUMBER: _ClassVar[int]
    SUPPORTS_MODEL_MATERIALIZATION_RECOVERY_FIELD_NUMBER: _ClassVar[int]
    claim: Claim
    private_placement_set: DesiredPrivatePlacementSet
    supports_model_materialization_recovery: bool
    def __init__(self, claim: _Optional[_Union[Claim, _Mapping]] = ..., private_placement_set: _Optional[_Union[DesiredPrivatePlacementSet, _Mapping]] = ..., supports_model_materialization_recovery: bool = ...) -> None: ...

class PrepareEvent(_message.Message):
    __slots__ = ("stage", "total_bytes", "transferred_bytes", "placement_set", "safe_code", "safe_detail", "model_progress")
    STAGE_FIELD_NUMBER: _ClassVar[int]
    TOTAL_BYTES_FIELD_NUMBER: _ClassVar[int]
    TRANSFERRED_BYTES_FIELD_NUMBER: _ClassVar[int]
    PLACEMENT_SET_FIELD_NUMBER: _ClassVar[int]
    SAFE_CODE_FIELD_NUMBER: _ClassVar[int]
    SAFE_DETAIL_FIELD_NUMBER: _ClassVar[int]
    MODEL_PROGRESS_FIELD_NUMBER: _ClassVar[int]
    stage: PrepareStage
    total_bytes: int
    transferred_bytes: int
    placement_set: DesiredPlacementSet
    safe_code: str
    safe_detail: str
    model_progress: _containers.RepeatedCompositeFieldContainer[PrepareModelProgress]
    def __init__(self, stage: _Optional[_Union[PrepareStage, str]] = ..., total_bytes: _Optional[int] = ..., transferred_bytes: _Optional[int] = ..., placement_set: _Optional[_Union[DesiredPlacementSet, _Mapping]] = ..., safe_code: _Optional[str] = ..., safe_detail: _Optional[str] = ..., model_progress: _Optional[_Iterable[_Union[PrepareModelProgress, _Mapping]]] = ...) -> None: ...

class PrepareModelProgress(_message.Message):
    __slots__ = ("model", "total_bytes", "transferred_bytes", "origin_bytes", "cached_bytes")
    MODEL_FIELD_NUMBER: _ClassVar[int]
    TOTAL_BYTES_FIELD_NUMBER: _ClassVar[int]
    TRANSFERRED_BYTES_FIELD_NUMBER: _ClassVar[int]
    ORIGIN_BYTES_FIELD_NUMBER: _ClassVar[int]
    CACHED_BYTES_FIELD_NUMBER: _ClassVar[int]
    model: DownloadModelRef
    total_bytes: int
    transferred_bytes: int
    origin_bytes: int
    cached_bytes: int
    def __init__(self, model: _Optional[_Union[DownloadModelRef, _Mapping]] = ..., total_bytes: _Optional[int] = ..., transferred_bytes: _Optional[int] = ..., origin_bytes: _Optional[int] = ..., cached_bytes: _Optional[int] = ...) -> None: ...

class ModelSourceFileCall(_message.Message):
    __slots__ = ("claim", "request")
    CLAIM_FIELD_NUMBER: _ClassVar[int]
    REQUEST_FIELD_NUMBER: _ClassVar[int]
    claim: Claim
    request: ModelSourceFileRequest
    def __init__(self, claim: _Optional[_Union[Claim, _Mapping]] = ..., request: _Optional[_Union[ModelSourceFileRequest, _Mapping]] = ...) -> None: ...

class ModelSourcePrepareCall(_message.Message):
    __slots__ = ("claim", "request")
    CLAIM_FIELD_NUMBER: _ClassVar[int]
    REQUEST_FIELD_NUMBER: _ClassVar[int]
    claim: Claim
    request: ModelSourcePrepareRequest
    def __init__(self, claim: _Optional[_Union[Claim, _Mapping]] = ..., request: _Optional[_Union[ModelSourcePrepareRequest, _Mapping]] = ...) -> None: ...

class ModelSourceReleaseCall(_message.Message):
    __slots__ = ("claim", "operation_id", "source_selection_digest")
    CLAIM_FIELD_NUMBER: _ClassVar[int]
    OPERATION_ID_FIELD_NUMBER: _ClassVar[int]
    SOURCE_SELECTION_DIGEST_FIELD_NUMBER: _ClassVar[int]
    claim: Claim
    operation_id: str
    source_selection_digest: bytes
    def __init__(self, claim: _Optional[_Union[Claim, _Mapping]] = ..., operation_id: _Optional[str] = ..., source_selection_digest: _Optional[bytes] = ...) -> None: ...

class ReleaseModelSourceRequest(_message.Message):
    __slots__ = ("operation_id",)
    OPERATION_ID_FIELD_NUMBER: _ClassVar[int]
    operation_id: str
    def __init__(self, operation_id: _Optional[str] = ...) -> None: ...

class NumericalEnvironmentRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class ModelSourceControlCall(_message.Message):
    __slots__ = ("claim", "operation_id", "source_selection_digest", "control_revision", "paused")
    CLAIM_FIELD_NUMBER: _ClassVar[int]
    OPERATION_ID_FIELD_NUMBER: _ClassVar[int]
    SOURCE_SELECTION_DIGEST_FIELD_NUMBER: _ClassVar[int]
    CONTROL_REVISION_FIELD_NUMBER: _ClassVar[int]
    PAUSED_FIELD_NUMBER: _ClassVar[int]
    claim: Claim
    operation_id: str
    source_selection_digest: bytes
    control_revision: int
    paused: bool
    def __init__(self, claim: _Optional[_Union[Claim, _Mapping]] = ..., operation_id: _Optional[str] = ..., source_selection_digest: _Optional[bytes] = ..., control_revision: _Optional[int] = ..., paused: bool = ...) -> None: ...

class ModelSourceControlResult(_message.Message):
    __slots__ = ("operation_id", "source_selection_digest", "control_revision", "paused")
    OPERATION_ID_FIELD_NUMBER: _ClassVar[int]
    SOURCE_SELECTION_DIGEST_FIELD_NUMBER: _ClassVar[int]
    CONTROL_REVISION_FIELD_NUMBER: _ClassVar[int]
    PAUSED_FIELD_NUMBER: _ClassVar[int]
    operation_id: str
    source_selection_digest: bytes
    control_revision: int
    paused: bool
    def __init__(self, operation_id: _Optional[str] = ..., source_selection_digest: _Optional[bytes] = ..., control_revision: _Optional[int] = ..., paused: bool = ...) -> None: ...

class NumericalEnvironmentCall(_message.Message):
    __slots__ = ("claim",)
    CLAIM_FIELD_NUMBER: _ClassVar[int]
    claim: Claim
    def __init__(self, claim: _Optional[_Union[Claim, _Mapping]] = ...) -> None: ...

class NumericalEnvironmentResult(_message.Message):
    __slots__ = ("digest",)
    DIGEST_FIELD_NUMBER: _ClassVar[int]
    digest: bytes
    def __init__(self, digest: _Optional[bytes] = ...) -> None: ...

class DerivedRetentionCall(_message.Message):
    __slots__ = ("claim", "request")
    CLAIM_FIELD_NUMBER: _ClassVar[int]
    REQUEST_FIELD_NUMBER: _ClassVar[int]
    claim: Claim
    request: DerivedRetentionRequest
    def __init__(self, claim: _Optional[_Union[Claim, _Mapping]] = ..., request: _Optional[_Union[DerivedRetentionRequest, _Mapping]] = ...) -> None: ...

class DerivedRetentionRequest(_message.Message):
    __slots__ = ("weights_transaction_id", "tensorfs_receipt_digest", "retention_id")
    WEIGHTS_TRANSACTION_ID_FIELD_NUMBER: _ClassVar[int]
    TENSORFS_RECEIPT_DIGEST_FIELD_NUMBER: _ClassVar[int]
    RETENTION_ID_FIELD_NUMBER: _ClassVar[int]
    weights_transaction_id: str
    tensorfs_receipt_digest: bytes
    retention_id: str
    def __init__(self, weights_transaction_id: _Optional[str] = ..., tensorfs_receipt_digest: _Optional[bytes] = ..., retention_id: _Optional[str] = ...) -> None: ...

class DerivedRetentionResult(_message.Message):
    __slots__ = ("weights_transaction_id", "tensorfs_receipt_digest", "retention_id", "manifest", "released")
    WEIGHTS_TRANSACTION_ID_FIELD_NUMBER: _ClassVar[int]
    TENSORFS_RECEIPT_DIGEST_FIELD_NUMBER: _ClassVar[int]
    RETENTION_ID_FIELD_NUMBER: _ClassVar[int]
    MANIFEST_FIELD_NUMBER: _ClassVar[int]
    RELEASED_FIELD_NUMBER: _ClassVar[int]
    weights_transaction_id: str
    tensorfs_receipt_digest: bytes
    retention_id: str
    manifest: Ref
    released: bool
    def __init__(self, weights_transaction_id: _Optional[str] = ..., tensorfs_receipt_digest: _Optional[bytes] = ..., retention_id: _Optional[str] = ..., manifest: _Optional[_Union[Ref, _Mapping]] = ..., released: bool = ...) -> None: ...

class DerivedResultReleaseCall(_message.Message):
    __slots__ = ("claim", "request")
    CLAIM_FIELD_NUMBER: _ClassVar[int]
    REQUEST_FIELD_NUMBER: _ClassVar[int]
    claim: Claim
    request: DerivedResultReleaseRequest
    def __init__(self, claim: _Optional[_Union[Claim, _Mapping]] = ..., request: _Optional[_Union[DerivedResultReleaseRequest, _Mapping]] = ...) -> None: ...

class DerivedResultReleaseRequest(_message.Message):
    __slots__ = ("weights_transaction_id", "tensorfs_receipt_digest")
    WEIGHTS_TRANSACTION_ID_FIELD_NUMBER: _ClassVar[int]
    TENSORFS_RECEIPT_DIGEST_FIELD_NUMBER: _ClassVar[int]
    weights_transaction_id: str
    tensorfs_receipt_digest: bytes
    def __init__(self, weights_transaction_id: _Optional[str] = ..., tensorfs_receipt_digest: _Optional[bytes] = ...) -> None: ...

class DerivedResultReleaseResult(_message.Message):
    __slots__ = ("weights_transaction_id", "tensorfs_receipt_digest", "released")
    WEIGHTS_TRANSACTION_ID_FIELD_NUMBER: _ClassVar[int]
    TENSORFS_RECEIPT_DIGEST_FIELD_NUMBER: _ClassVar[int]
    RELEASED_FIELD_NUMBER: _ClassVar[int]
    weights_transaction_id: str
    tensorfs_receipt_digest: bytes
    released: bool
    def __init__(self, weights_transaction_id: _Optional[str] = ..., tensorfs_receipt_digest: _Optional[bytes] = ..., released: bool = ...) -> None: ...

class RecordOperationResultCall(_message.Message):
    __slots__ = ("claim", "computation_digest", "request_id", "attempt_ordinal", "invocation_spec_digest", "outcome_id", "outcome_digest")
    CLAIM_FIELD_NUMBER: _ClassVar[int]
    COMPUTATION_DIGEST_FIELD_NUMBER: _ClassVar[int]
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    ATTEMPT_ORDINAL_FIELD_NUMBER: _ClassVar[int]
    INVOCATION_SPEC_DIGEST_FIELD_NUMBER: _ClassVar[int]
    OUTCOME_ID_FIELD_NUMBER: _ClassVar[int]
    OUTCOME_DIGEST_FIELD_NUMBER: _ClassVar[int]
    claim: Claim
    computation_digest: bytes
    request_id: str
    attempt_ordinal: int
    invocation_spec_digest: bytes
    outcome_id: str
    outcome_digest: bytes
    def __init__(self, claim: _Optional[_Union[Claim, _Mapping]] = ..., computation_digest: _Optional[bytes] = ..., request_id: _Optional[str] = ..., attempt_ordinal: _Optional[int] = ..., invocation_spec_digest: _Optional[bytes] = ..., outcome_id: _Optional[str] = ..., outcome_digest: _Optional[bytes] = ...) -> None: ...

class RecordOperationResultResult(_message.Message):
    __slots__ = ("computation_digest", "recorded")
    COMPUTATION_DIGEST_FIELD_NUMBER: _ClassVar[int]
    RECORDED_FIELD_NUMBER: _ClassVar[int]
    computation_digest: bytes
    recorded: bool
    def __init__(self, computation_digest: _Optional[bytes] = ..., recorded: bool = ...) -> None: ...

class LookupOperationCall(_message.Message):
    __slots__ = ("claim", "computation_digest", "consumer_request_id")
    CLAIM_FIELD_NUMBER: _ClassVar[int]
    COMPUTATION_DIGEST_FIELD_NUMBER: _ClassVar[int]
    CONSUMER_REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    claim: Claim
    computation_digest: bytes
    consumer_request_id: str
    def __init__(self, claim: _Optional[_Union[Claim, _Mapping]] = ..., computation_digest: _Optional[bytes] = ..., consumer_request_id: _Optional[str] = ...) -> None: ...

class LookupOperationResult(_message.Message):
    __slots__ = ("computation_digest", "found", "source", "retentions", "consumer_request_id", "byte_retentions")
    COMPUTATION_DIGEST_FIELD_NUMBER: _ClassVar[int]
    FOUND_FIELD_NUMBER: _ClassVar[int]
    SOURCE_FIELD_NUMBER: _ClassVar[int]
    RETENTIONS_FIELD_NUMBER: _ClassVar[int]
    CONSUMER_REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    BYTE_RETENTIONS_FIELD_NUMBER: _ClassVar[int]
    computation_digest: bytes
    found: bool
    source: AttemptOutcome
    retentions: _containers.RepeatedCompositeFieldContainer[DerivedRetentionResult]
    consumer_request_id: str
    byte_retentions: _containers.RepeatedCompositeFieldContainer[NativeByteRetentionResult]
    def __init__(self, computation_digest: _Optional[bytes] = ..., found: bool = ..., source: _Optional[_Union[AttemptOutcome, _Mapping]] = ..., retentions: _Optional[_Iterable[_Union[DerivedRetentionResult, _Mapping]]] = ..., consumer_request_id: _Optional[str] = ..., byte_retentions: _Optional[_Iterable[_Union[NativeByteRetentionResult, _Mapping]]] = ...) -> None: ...

class PruneOperationCacheCall(_message.Message):
    __slots__ = ("claim",)
    CLAIM_FIELD_NUMBER: _ClassVar[int]
    claim: Claim
    def __init__(self, claim: _Optional[_Union[Claim, _Mapping]] = ...) -> None: ...

class PruneOperationCacheResult(_message.Message):
    __slots__ = ("removed_entries", "reclaimed_bytes", "store_busy")
    REMOVED_ENTRIES_FIELD_NUMBER: _ClassVar[int]
    RECLAIMED_BYTES_FIELD_NUMBER: _ClassVar[int]
    STORE_BUSY_FIELD_NUMBER: _ClassVar[int]
    removed_entries: int
    reclaimed_bytes: int
    store_busy: bool
    def __init__(self, removed_entries: _Optional[int] = ..., reclaimed_bytes: _Optional[int] = ..., store_busy: bool = ...) -> None: ...

class CollectStoreGarbageRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class CollectStoreGarbageResult(_message.Message):
    __slots__ = ("reclaimed_bytes", "store_busy")
    RECLAIMED_BYTES_FIELD_NUMBER: _ClassVar[int]
    STORE_BUSY_FIELD_NUMBER: _ClassVar[int]
    reclaimed_bytes: int
    store_busy: bool
    def __init__(self, reclaimed_bytes: _Optional[int] = ..., store_busy: bool = ...) -> None: ...

class ReleaseModelSourceResult(_message.Message):
    __slots__ = ("operation_id", "released")
    OPERATION_ID_FIELD_NUMBER: _ClassVar[int]
    RELEASED_FIELD_NUMBER: _ClassVar[int]
    operation_id: str
    released: bool
    def __init__(self, operation_id: _Optional[str] = ..., released: bool = ...) -> None: ...

class ModelSourceAdoptCall(_message.Message):
    __slots__ = ("claim", "from_operation_id", "request")
    CLAIM_FIELD_NUMBER: _ClassVar[int]
    FROM_OPERATION_ID_FIELD_NUMBER: _ClassVar[int]
    REQUEST_FIELD_NUMBER: _ClassVar[int]
    claim: Claim
    from_operation_id: str
    request: ModelSourcePrepareRequest
    def __init__(self, claim: _Optional[_Union[Claim, _Mapping]] = ..., from_operation_id: _Optional[str] = ..., request: _Optional[_Union[ModelSourcePrepareRequest, _Mapping]] = ...) -> None: ...

class CheckpointPageCall(_message.Message):
    __slots__ = ("claim", "request")
    CLAIM_FIELD_NUMBER: _ClassVar[int]
    REQUEST_FIELD_NUMBER: _ClassVar[int]
    claim: Claim
    request: CheckpointPageRequest
    def __init__(self, claim: _Optional[_Union[Claim, _Mapping]] = ..., request: _Optional[_Union[CheckpointPageRequest, _Mapping]] = ...) -> None: ...

class CheckpointTransferCall(_message.Message):
    __slots__ = ("claim", "request")
    CLAIM_FIELD_NUMBER: _ClassVar[int]
    REQUEST_FIELD_NUMBER: _ClassVar[int]
    claim: Claim
    request: CheckpointTransferRequest
    def __init__(self, claim: _Optional[_Union[Claim, _Mapping]] = ..., request: _Optional[_Union[CheckpointTransferRequest, _Mapping]] = ...) -> None: ...

class LocalPackageFetchCall(_message.Message):
    __slots__ = ("claim", "request")
    CLAIM_FIELD_NUMBER: _ClassVar[int]
    REQUEST_FIELD_NUMBER: _ClassVar[int]
    claim: Claim
    request: LocalPackageFetchRequest
    def __init__(self, claim: _Optional[_Union[Claim, _Mapping]] = ..., request: _Optional[_Union[LocalPackageFetchRequest, _Mapping]] = ...) -> None: ...

class LocalPackageUploadFrame(_message.Message):
    __slots__ = ("header", "chunk")
    HEADER_FIELD_NUMBER: _ClassVar[int]
    CHUNK_FIELD_NUMBER: _ClassVar[int]
    header: LocalPackageUploadHeader
    chunk: LocalPackageUploadChunk
    def __init__(self, header: _Optional[_Union[LocalPackageUploadHeader, _Mapping]] = ..., chunk: _Optional[_Union[LocalPackageUploadChunk, _Mapping]] = ...) -> None: ...

class LocalPackageUploadHeader(_message.Message):
    __slots__ = ("claim", "operation_id", "source_digest", "file")
    CLAIM_FIELD_NUMBER: _ClassVar[int]
    OPERATION_ID_FIELD_NUMBER: _ClassVar[int]
    SOURCE_DIGEST_FIELD_NUMBER: _ClassVar[int]
    FILE_FIELD_NUMBER: _ClassVar[int]
    claim: Claim
    operation_id: str
    source_digest: bytes
    file: LocalPackageFileRef
    def __init__(self, claim: _Optional[_Union[Claim, _Mapping]] = ..., operation_id: _Optional[str] = ..., source_digest: _Optional[bytes] = ..., file: _Optional[_Union[LocalPackageFileRef, _Mapping]] = ...) -> None: ...

class LocalPackageUploadChunk(_message.Message):
    __slots__ = ("offset", "data")
    OFFSET_FIELD_NUMBER: _ClassVar[int]
    DATA_FIELD_NUMBER: _ClassVar[int]
    offset: int
    data: bytes
    def __init__(self, offset: _Optional[int] = ..., data: _Optional[bytes] = ...) -> None: ...

class LocalPackageAbortCall(_message.Message):
    __slots__ = ("claim", "request")
    CLAIM_FIELD_NUMBER: _ClassVar[int]
    REQUEST_FIELD_NUMBER: _ClassVar[int]
    claim: Claim
    request: LocalPackageAbort
    def __init__(self, claim: _Optional[_Union[Claim, _Mapping]] = ..., request: _Optional[_Union[LocalPackageAbort, _Mapping]] = ...) -> None: ...

class WeightsTransferCall(_message.Message):
    __slots__ = ("claim", "request")
    CLAIM_FIELD_NUMBER: _ClassVar[int]
    REQUEST_FIELD_NUMBER: _ClassVar[int]
    claim: Claim
    request: WeightsTransferRequest
    def __init__(self, claim: _Optional[_Union[Claim, _Mapping]] = ..., request: _Optional[_Union[WeightsTransferRequest, _Mapping]] = ...) -> None: ...

class PreparePackageSetRequest(_message.Message):
    __slots__ = ("download_delegation", "application", "model_slot_paths", "image_inventory", "locked_requirements", "install_root", "download_delegation_signature", "python_requires", "python_version")
    DOWNLOAD_DELEGATION_FIELD_NUMBER: _ClassVar[int]
    APPLICATION_FIELD_NUMBER: _ClassVar[int]
    MODEL_SLOT_PATHS_FIELD_NUMBER: _ClassVar[int]
    IMAGE_INVENTORY_FIELD_NUMBER: _ClassVar[int]
    LOCKED_REQUIREMENTS_FIELD_NUMBER: _ClassVar[int]
    INSTALL_ROOT_FIELD_NUMBER: _ClassVar[int]
    DOWNLOAD_DELEGATION_SIGNATURE_FIELD_NUMBER: _ClassVar[int]
    PYTHON_REQUIRES_FIELD_NUMBER: _ClassVar[int]
    PYTHON_VERSION_FIELD_NUMBER: _ClassVar[int]
    download_delegation: bytes
    application: str
    model_slot_paths: _containers.RepeatedScalarFieldContainer[str]
    image_inventory: ImageInventory
    locked_requirements: bytes
    install_root: str
    download_delegation_signature: bytes
    python_requires: str
    python_version: str
    def __init__(self, download_delegation: _Optional[bytes] = ..., application: _Optional[str] = ..., model_slot_paths: _Optional[_Iterable[str]] = ..., image_inventory: _Optional[_Union[ImageInventory, _Mapping]] = ..., locked_requirements: _Optional[bytes] = ..., install_root: _Optional[str] = ..., download_delegation_signature: _Optional[bytes] = ..., python_requires: _Optional[str] = ..., python_version: _Optional[str] = ...) -> None: ...

class ImageInventory(_message.Message):
    __slots__ = ("profile", "python", "distributions", "interpreters")
    PROFILE_FIELD_NUMBER: _ClassVar[int]
    PYTHON_FIELD_NUMBER: _ClassVar[int]
    DISTRIBUTIONS_FIELD_NUMBER: _ClassVar[int]
    INTERPRETERS_FIELD_NUMBER: _ClassVar[int]
    profile: str
    python: str
    distributions: _containers.RepeatedCompositeFieldContainer[ImageDistribution]
    interpreters: _containers.RepeatedCompositeFieldContainer[PythonInterpreter]
    def __init__(self, profile: _Optional[str] = ..., python: _Optional[str] = ..., distributions: _Optional[_Iterable[_Union[ImageDistribution, _Mapping]]] = ..., interpreters: _Optional[_Iterable[_Union[PythonInterpreter, _Mapping]]] = ...) -> None: ...

class PythonInterpreter(_message.Message):
    __slots__ = ("version", "abi")
    VERSION_FIELD_NUMBER: _ClassVar[int]
    ABI_FIELD_NUMBER: _ClassVar[int]
    version: str
    abi: str
    def __init__(self, version: _Optional[str] = ..., abi: _Optional[str] = ...) -> None: ...

class ImageDistribution(_message.Message):
    __slots__ = ("distribution", "version")
    DISTRIBUTION_FIELD_NUMBER: _ClassVar[int]
    VERSION_FIELD_NUMBER: _ClassVar[int]
    distribution: str
    version: str
    def __init__(self, distribution: _Optional[str] = ..., version: _Optional[str] = ...) -> None: ...

class PreparePackageSetResult(_message.Message):
    __slots__ = ("placement_set",)
    PLACEMENT_SET_FIELD_NUMBER: _ClassVar[int]
    placement_set: DesiredPlacementSet
    def __init__(self, placement_set: _Optional[_Union[DesiredPlacementSet, _Mapping]] = ...) -> None: ...

class CheckPackageSetCompatibilityResult(_message.Message):
    __slots__ = ("refusal_code", "refusal_detail")
    REFUSAL_CODE_FIELD_NUMBER: _ClassVar[int]
    REFUSAL_DETAIL_FIELD_NUMBER: _ClassVar[int]
    refusal_code: str
    refusal_detail: str
    def __init__(self, refusal_code: _Optional[str] = ..., refusal_detail: _Optional[str] = ...) -> None: ...

class PrepareLocalPackageRequest(_message.Message):
    __slots__ = ("operation_id", "package", "wheels", "install_root", "dependency_requirements", "python_requires", "python_version")
    OPERATION_ID_FIELD_NUMBER: _ClassVar[int]
    PACKAGE_FIELD_NUMBER: _ClassVar[int]
    WHEELS_FIELD_NUMBER: _ClassVar[int]
    INSTALL_ROOT_FIELD_NUMBER: _ClassVar[int]
    DEPENDENCY_REQUIREMENTS_FIELD_NUMBER: _ClassVar[int]
    PYTHON_REQUIRES_FIELD_NUMBER: _ClassVar[int]
    PYTHON_VERSION_FIELD_NUMBER: _ClassVar[int]
    operation_id: str
    package: DevelopmentPackage
    wheels: _containers.RepeatedCompositeFieldContainer[LocalPackageWheel]
    install_root: str
    dependency_requirements: bytes
    python_requires: str
    python_version: str
    def __init__(self, operation_id: _Optional[str] = ..., package: _Optional[_Union[DevelopmentPackage, _Mapping]] = ..., wheels: _Optional[_Iterable[_Union[LocalPackageWheel, _Mapping]]] = ..., install_root: _Optional[str] = ..., dependency_requirements: _Optional[bytes] = ..., python_requires: _Optional[str] = ..., python_version: _Optional[str] = ...) -> None: ...

class LocalPackageWheel(_message.Message):
    __slots__ = ("digest", "filename", "length", "path")
    DIGEST_FIELD_NUMBER: _ClassVar[int]
    FILENAME_FIELD_NUMBER: _ClassVar[int]
    LENGTH_FIELD_NUMBER: _ClassVar[int]
    PATH_FIELD_NUMBER: _ClassVar[int]
    digest: bytes
    filename: str
    length: int
    path: str
    def __init__(self, digest: _Optional[bytes] = ..., filename: _Optional[str] = ..., length: _Optional[int] = ..., path: _Optional[str] = ...) -> None: ...

class PreparePrivatePlacementRequest(_message.Message):
    __slots__ = ("operation_id", "local_revision_digest", "download_delegation", "download_delegation_signature", "native_models", "claim")
    OPERATION_ID_FIELD_NUMBER: _ClassVar[int]
    LOCAL_REVISION_DIGEST_FIELD_NUMBER: _ClassVar[int]
    DOWNLOAD_DELEGATION_FIELD_NUMBER: _ClassVar[int]
    DOWNLOAD_DELEGATION_SIGNATURE_FIELD_NUMBER: _ClassVar[int]
    NATIVE_MODELS_FIELD_NUMBER: _ClassVar[int]
    CLAIM_FIELD_NUMBER: _ClassVar[int]
    operation_id: str
    local_revision_digest: bytes
    download_delegation: bytes
    download_delegation_signature: bytes
    native_models: _containers.RepeatedCompositeFieldContainer[NativeModelBinding]
    claim: Claim
    def __init__(self, operation_id: _Optional[str] = ..., local_revision_digest: _Optional[bytes] = ..., download_delegation: _Optional[bytes] = ..., download_delegation_signature: _Optional[bytes] = ..., native_models: _Optional[_Iterable[_Union[NativeModelBinding, _Mapping]]] = ..., claim: _Optional[_Union[Claim, _Mapping]] = ...) -> None: ...

class LocalModelSourceFile(_message.Message):
    __slots__ = ("member", "object_id", "length", "path", "header_path", "verified")
    MEMBER_FIELD_NUMBER: _ClassVar[int]
    OBJECT_ID_FIELD_NUMBER: _ClassVar[int]
    LENGTH_FIELD_NUMBER: _ClassVar[int]
    PATH_FIELD_NUMBER: _ClassVar[int]
    HEADER_PATH_FIELD_NUMBER: _ClassVar[int]
    VERIFIED_FIELD_NUMBER: _ClassVar[int]
    member: str
    object_id: str
    length: int
    path: str
    header_path: str
    verified: bool
    def __init__(self, member: _Optional[str] = ..., object_id: _Optional[str] = ..., length: _Optional[int] = ..., path: _Optional[str] = ..., header_path: _Optional[str] = ..., verified: bool = ...) -> None: ...

class ModelSourceProfile(_message.Message):
    __slots__ = ("slot", "profile")
    SLOT_FIELD_NUMBER: _ClassVar[int]
    PROFILE_FIELD_NUMBER: _ClassVar[int]
    slot: str
    profile: str
    def __init__(self, slot: _Optional[str] = ..., profile: _Optional[str] = ...) -> None: ...

class PreparedModelSource(_message.Message):
    __slots__ = ("slot", "profile", "manifest")
    SLOT_FIELD_NUMBER: _ClassVar[int]
    PROFILE_FIELD_NUMBER: _ClassVar[int]
    MANIFEST_FIELD_NUMBER: _ClassVar[int]
    slot: str
    profile: str
    manifest: Ref
    def __init__(self, slot: _Optional[str] = ..., profile: _Optional[str] = ..., manifest: _Optional[_Union[Ref, _Mapping]] = ...) -> None: ...

class ModelSourceCheckpoint(_message.Message):
    __slots__ = ("slot", "head", "plan_digest", "index", "bytes")
    SLOT_FIELD_NUMBER: _ClassVar[int]
    HEAD_FIELD_NUMBER: _ClassVar[int]
    PLAN_DIGEST_FIELD_NUMBER: _ClassVar[int]
    INDEX_FIELD_NUMBER: _ClassVar[int]
    BYTES_FIELD_NUMBER: _ClassVar[int]
    slot: str
    head: Ref
    plan_digest: bytes
    index: int
    bytes: int
    def __init__(self, slot: _Optional[str] = ..., head: _Optional[_Union[Ref, _Mapping]] = ..., plan_digest: _Optional[bytes] = ..., index: _Optional[int] = ..., bytes: _Optional[int] = ...) -> None: ...

class SourceCheckpointSubject(_message.Message):
    __slots__ = ("operation_id", "source_selection_digest", "slot")
    OPERATION_ID_FIELD_NUMBER: _ClassVar[int]
    SOURCE_SELECTION_DIGEST_FIELD_NUMBER: _ClassVar[int]
    SLOT_FIELD_NUMBER: _ClassVar[int]
    operation_id: str
    source_selection_digest: bytes
    slot: str
    def __init__(self, operation_id: _Optional[str] = ..., source_selection_digest: _Optional[bytes] = ..., slot: _Optional[str] = ...) -> None: ...

class WeightsCheckpointSubject(_message.Message):
    __slots__ = ("request_id", "invocation_spec_digest", "output_slot", "weights_transaction_id", "writer_epoch", "tensorfs_declaration_digest")
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    INVOCATION_SPEC_DIGEST_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_SLOT_FIELD_NUMBER: _ClassVar[int]
    WEIGHTS_TRANSACTION_ID_FIELD_NUMBER: _ClassVar[int]
    WRITER_EPOCH_FIELD_NUMBER: _ClassVar[int]
    TENSORFS_DECLARATION_DIGEST_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    invocation_spec_digest: bytes
    output_slot: str
    weights_transaction_id: str
    writer_epoch: int
    tensorfs_declaration_digest: bytes
    def __init__(self, request_id: _Optional[str] = ..., invocation_spec_digest: _Optional[bytes] = ..., output_slot: _Optional[str] = ..., weights_transaction_id: _Optional[str] = ..., writer_epoch: _Optional[int] = ..., tensorfs_declaration_digest: _Optional[bytes] = ...) -> None: ...

class CheckpointSubject(_message.Message):
    __slots__ = ("source", "weights")
    SOURCE_FIELD_NUMBER: _ClassVar[int]
    WEIGHTS_FIELD_NUMBER: _ClassVar[int]
    source: SourceCheckpointSubject
    weights: WeightsCheckpointSubject
    def __init__(self, source: _Optional[_Union[SourceCheckpointSubject, _Mapping]] = ..., weights: _Optional[_Union[WeightsCheckpointSubject, _Mapping]] = ...) -> None: ...

class CheckpointObject(_message.Message):
    __slots__ = ("ref", "manifest")
    REF_FIELD_NUMBER: _ClassVar[int]
    MANIFEST_FIELD_NUMBER: _ClassVar[int]
    ref: Ref
    manifest: bool
    def __init__(self, ref: _Optional[_Union[Ref, _Mapping]] = ..., manifest: bool = ...) -> None: ...

class CheckpointPageRequest(_message.Message):
    __slots__ = ("record_owner_epoch", "control_stream_epoch", "worker_boot_id", "subject", "plan_digest", "head", "offset", "limit")
    RECORD_OWNER_EPOCH_FIELD_NUMBER: _ClassVar[int]
    CONTROL_STREAM_EPOCH_FIELD_NUMBER: _ClassVar[int]
    WORKER_BOOT_ID_FIELD_NUMBER: _ClassVar[int]
    SUBJECT_FIELD_NUMBER: _ClassVar[int]
    PLAN_DIGEST_FIELD_NUMBER: _ClassVar[int]
    HEAD_FIELD_NUMBER: _ClassVar[int]
    OFFSET_FIELD_NUMBER: _ClassVar[int]
    LIMIT_FIELD_NUMBER: _ClassVar[int]
    record_owner_epoch: int
    control_stream_epoch: int
    worker_boot_id: str
    subject: CheckpointSubject
    plan_digest: bytes
    head: Ref
    offset: int
    limit: int
    def __init__(self, record_owner_epoch: _Optional[int] = ..., control_stream_epoch: _Optional[int] = ..., worker_boot_id: _Optional[str] = ..., subject: _Optional[_Union[CheckpointSubject, _Mapping]] = ..., plan_digest: _Optional[bytes] = ..., head: _Optional[_Union[Ref, _Mapping]] = ..., offset: _Optional[int] = ..., limit: _Optional[int] = ...) -> None: ...

class CheckpointPageResult(_message.Message):
    __slots__ = ("record_owner_epoch", "control_stream_epoch", "worker_boot_id", "subject", "plan_digest", "head", "index", "bytes", "previous", "progress", "objects", "next_offset", "has_more", "safe_code", "safe_detail")
    RECORD_OWNER_EPOCH_FIELD_NUMBER: _ClassVar[int]
    CONTROL_STREAM_EPOCH_FIELD_NUMBER: _ClassVar[int]
    WORKER_BOOT_ID_FIELD_NUMBER: _ClassVar[int]
    SUBJECT_FIELD_NUMBER: _ClassVar[int]
    PLAN_DIGEST_FIELD_NUMBER: _ClassVar[int]
    HEAD_FIELD_NUMBER: _ClassVar[int]
    INDEX_FIELD_NUMBER: _ClassVar[int]
    BYTES_FIELD_NUMBER: _ClassVar[int]
    PREVIOUS_FIELD_NUMBER: _ClassVar[int]
    PROGRESS_FIELD_NUMBER: _ClassVar[int]
    OBJECTS_FIELD_NUMBER: _ClassVar[int]
    NEXT_OFFSET_FIELD_NUMBER: _ClassVar[int]
    HAS_MORE_FIELD_NUMBER: _ClassVar[int]
    SAFE_CODE_FIELD_NUMBER: _ClassVar[int]
    SAFE_DETAIL_FIELD_NUMBER: _ClassVar[int]
    record_owner_epoch: int
    control_stream_epoch: int
    worker_boot_id: str
    subject: CheckpointSubject
    plan_digest: bytes
    head: Ref
    index: int
    bytes: int
    previous: Ref
    progress: Ref
    objects: _containers.RepeatedCompositeFieldContainer[CheckpointObject]
    next_offset: int
    has_more: bool
    safe_code: str
    safe_detail: str
    def __init__(self, record_owner_epoch: _Optional[int] = ..., control_stream_epoch: _Optional[int] = ..., worker_boot_id: _Optional[str] = ..., subject: _Optional[_Union[CheckpointSubject, _Mapping]] = ..., plan_digest: _Optional[bytes] = ..., head: _Optional[_Union[Ref, _Mapping]] = ..., index: _Optional[int] = ..., bytes: _Optional[int] = ..., previous: _Optional[_Union[Ref, _Mapping]] = ..., progress: _Optional[_Union[Ref, _Mapping]] = ..., objects: _Optional[_Iterable[_Union[CheckpointObject, _Mapping]]] = ..., next_offset: _Optional[int] = ..., has_more: bool = ..., safe_code: _Optional[str] = ..., safe_detail: _Optional[str] = ...) -> None: ...

class CheckpointTransferRequest(_message.Message):
    __slots__ = ("record_owner_epoch", "control_stream_epoch", "worker_boot_id", "subject", "plan_digest", "head", "object", "transfer_id", "grant_revision", "upload_grant", "download_url")
    RECORD_OWNER_EPOCH_FIELD_NUMBER: _ClassVar[int]
    CONTROL_STREAM_EPOCH_FIELD_NUMBER: _ClassVar[int]
    WORKER_BOOT_ID_FIELD_NUMBER: _ClassVar[int]
    SUBJECT_FIELD_NUMBER: _ClassVar[int]
    PLAN_DIGEST_FIELD_NUMBER: _ClassVar[int]
    HEAD_FIELD_NUMBER: _ClassVar[int]
    OBJECT_FIELD_NUMBER: _ClassVar[int]
    TRANSFER_ID_FIELD_NUMBER: _ClassVar[int]
    GRANT_REVISION_FIELD_NUMBER: _ClassVar[int]
    UPLOAD_GRANT_FIELD_NUMBER: _ClassVar[int]
    DOWNLOAD_URL_FIELD_NUMBER: _ClassVar[int]
    record_owner_epoch: int
    control_stream_epoch: int
    worker_boot_id: str
    subject: CheckpointSubject
    plan_digest: bytes
    head: Ref
    object: CheckpointObject
    transfer_id: str
    grant_revision: int
    upload_grant: WeightsUploadGrant
    download_url: str
    def __init__(self, record_owner_epoch: _Optional[int] = ..., control_stream_epoch: _Optional[int] = ..., worker_boot_id: _Optional[str] = ..., subject: _Optional[_Union[CheckpointSubject, _Mapping]] = ..., plan_digest: _Optional[bytes] = ..., head: _Optional[_Union[Ref, _Mapping]] = ..., object: _Optional[_Union[CheckpointObject, _Mapping]] = ..., transfer_id: _Optional[str] = ..., grant_revision: _Optional[int] = ..., upload_grant: _Optional[_Union[WeightsUploadGrant, _Mapping]] = ..., download_url: _Optional[str] = ...) -> None: ...

class CheckpointTransferStatus(_message.Message):
    __slots__ = ("record_owner_epoch", "control_stream_epoch", "worker_boot_id", "subject", "plan_digest", "head", "object", "transfer_id", "grant_revision", "state", "transferred_bytes", "http_status", "checksum_sha256", "safe_code", "safe_detail")
    RECORD_OWNER_EPOCH_FIELD_NUMBER: _ClassVar[int]
    CONTROL_STREAM_EPOCH_FIELD_NUMBER: _ClassVar[int]
    WORKER_BOOT_ID_FIELD_NUMBER: _ClassVar[int]
    SUBJECT_FIELD_NUMBER: _ClassVar[int]
    PLAN_DIGEST_FIELD_NUMBER: _ClassVar[int]
    HEAD_FIELD_NUMBER: _ClassVar[int]
    OBJECT_FIELD_NUMBER: _ClassVar[int]
    TRANSFER_ID_FIELD_NUMBER: _ClassVar[int]
    GRANT_REVISION_FIELD_NUMBER: _ClassVar[int]
    STATE_FIELD_NUMBER: _ClassVar[int]
    TRANSFERRED_BYTES_FIELD_NUMBER: _ClassVar[int]
    HTTP_STATUS_FIELD_NUMBER: _ClassVar[int]
    CHECKSUM_SHA256_FIELD_NUMBER: _ClassVar[int]
    SAFE_CODE_FIELD_NUMBER: _ClassVar[int]
    SAFE_DETAIL_FIELD_NUMBER: _ClassVar[int]
    record_owner_epoch: int
    control_stream_epoch: int
    worker_boot_id: str
    subject: CheckpointSubject
    plan_digest: bytes
    head: Ref
    object: CheckpointObject
    transfer_id: str
    grant_revision: int
    state: WeightsTransferState
    transferred_bytes: int
    http_status: int
    checksum_sha256: str
    safe_code: str
    safe_detail: str
    def __init__(self, record_owner_epoch: _Optional[int] = ..., control_stream_epoch: _Optional[int] = ..., worker_boot_id: _Optional[str] = ..., subject: _Optional[_Union[CheckpointSubject, _Mapping]] = ..., plan_digest: _Optional[bytes] = ..., head: _Optional[_Union[Ref, _Mapping]] = ..., object: _Optional[_Union[CheckpointObject, _Mapping]] = ..., transfer_id: _Optional[str] = ..., grant_revision: _Optional[int] = ..., state: _Optional[_Union[WeightsTransferState, str]] = ..., transferred_bytes: _Optional[int] = ..., http_status: _Optional[int] = ..., checksum_sha256: _Optional[str] = ..., safe_code: _Optional[str] = ..., safe_detail: _Optional[str] = ...) -> None: ...

class PrepareModelSourceRequest(_message.Message):
    __slots__ = ("operation_id", "source_selection_digest", "profiles", "files", "source_uri", "declared_license", "checkpoints", "adopt_from_operation_id")
    OPERATION_ID_FIELD_NUMBER: _ClassVar[int]
    SOURCE_SELECTION_DIGEST_FIELD_NUMBER: _ClassVar[int]
    PROFILES_FIELD_NUMBER: _ClassVar[int]
    FILES_FIELD_NUMBER: _ClassVar[int]
    SOURCE_URI_FIELD_NUMBER: _ClassVar[int]
    DECLARED_LICENSE_FIELD_NUMBER: _ClassVar[int]
    CHECKPOINTS_FIELD_NUMBER: _ClassVar[int]
    ADOPT_FROM_OPERATION_ID_FIELD_NUMBER: _ClassVar[int]
    operation_id: str
    source_selection_digest: bytes
    profiles: _containers.RepeatedCompositeFieldContainer[ModelSourceProfile]
    files: _containers.RepeatedCompositeFieldContainer[LocalModelSourceFile]
    source_uri: str
    declared_license: str
    checkpoints: _containers.RepeatedCompositeFieldContainer[ModelSourceCheckpoint]
    adopt_from_operation_id: str
    def __init__(self, operation_id: _Optional[str] = ..., source_selection_digest: _Optional[bytes] = ..., profiles: _Optional[_Iterable[_Union[ModelSourceProfile, _Mapping]]] = ..., files: _Optional[_Iterable[_Union[LocalModelSourceFile, _Mapping]]] = ..., source_uri: _Optional[str] = ..., declared_license: _Optional[str] = ..., checkpoints: _Optional[_Iterable[_Union[ModelSourceCheckpoint, _Mapping]]] = ..., adopt_from_operation_id: _Optional[str] = ...) -> None: ...

class PrepareModelSourceResult(_message.Message):
    __slots__ = ("outcome", "sources", "safe_code", "safe_detail", "checkpoints", "spent_members")
    OUTCOME_FIELD_NUMBER: _ClassVar[int]
    SOURCES_FIELD_NUMBER: _ClassVar[int]
    SAFE_CODE_FIELD_NUMBER: _ClassVar[int]
    SAFE_DETAIL_FIELD_NUMBER: _ClassVar[int]
    CHECKPOINTS_FIELD_NUMBER: _ClassVar[int]
    SPENT_MEMBERS_FIELD_NUMBER: _ClassVar[int]
    outcome: ModelSourcePrepareOutcome
    sources: _containers.RepeatedCompositeFieldContainer[PreparedModelSource]
    safe_code: str
    safe_detail: str
    checkpoints: _containers.RepeatedCompositeFieldContainer[ModelSourceCheckpoint]
    spent_members: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, outcome: _Optional[_Union[ModelSourcePrepareOutcome, str]] = ..., sources: _Optional[_Iterable[_Union[PreparedModelSource, _Mapping]]] = ..., safe_code: _Optional[str] = ..., safe_detail: _Optional[str] = ..., checkpoints: _Optional[_Iterable[_Union[ModelSourceCheckpoint, _Mapping]]] = ..., spent_members: _Optional[_Iterable[str]] = ...) -> None: ...

class RecordOwnerFrame(_message.Message):
    __slots__ = ("claim", "desired_state", "attempt_offer", "cancel_attempt", "outcome_ack", "checkpoint_receipt", "snapshot_ack", "weights_finalize_request", "weights_host_ack", "weights_transfer_request", "weights_read_request", "model_source_file_request", "model_source_prepare_request", "local_package_abort", "weights_upload_request", "local_package_fetch_request", "child_call_result", "native_source_command", "native_artifact_transfer")
    CLAIM_FIELD_NUMBER: _ClassVar[int]
    DESIRED_STATE_FIELD_NUMBER: _ClassVar[int]
    ATTEMPT_OFFER_FIELD_NUMBER: _ClassVar[int]
    CANCEL_ATTEMPT_FIELD_NUMBER: _ClassVar[int]
    OUTCOME_ACK_FIELD_NUMBER: _ClassVar[int]
    CHECKPOINT_RECEIPT_FIELD_NUMBER: _ClassVar[int]
    SNAPSHOT_ACK_FIELD_NUMBER: _ClassVar[int]
    WEIGHTS_FINALIZE_REQUEST_FIELD_NUMBER: _ClassVar[int]
    WEIGHTS_HOST_ACK_FIELD_NUMBER: _ClassVar[int]
    WEIGHTS_TRANSFER_REQUEST_FIELD_NUMBER: _ClassVar[int]
    WEIGHTS_READ_REQUEST_FIELD_NUMBER: _ClassVar[int]
    MODEL_SOURCE_FILE_REQUEST_FIELD_NUMBER: _ClassVar[int]
    MODEL_SOURCE_PREPARE_REQUEST_FIELD_NUMBER: _ClassVar[int]
    LOCAL_PACKAGE_ABORT_FIELD_NUMBER: _ClassVar[int]
    WEIGHTS_UPLOAD_REQUEST_FIELD_NUMBER: _ClassVar[int]
    LOCAL_PACKAGE_FETCH_REQUEST_FIELD_NUMBER: _ClassVar[int]
    CHILD_CALL_RESULT_FIELD_NUMBER: _ClassVar[int]
    NATIVE_SOURCE_COMMAND_FIELD_NUMBER: _ClassVar[int]
    NATIVE_ARTIFACT_TRANSFER_FIELD_NUMBER: _ClassVar[int]
    claim: Claim
    desired_state: DesiredWorkerState
    attempt_offer: AttemptOffer
    cancel_attempt: CancelAttempt
    outcome_ack: AttemptOutcomeAck
    checkpoint_receipt: JobCheckpointReceipt
    snapshot_ack: SnapshotAck
    weights_finalize_request: WeightsFinalizeRequest
    weights_host_ack: WeightsHostAck
    weights_transfer_request: WeightsTransferRequest
    weights_read_request: WeightsReadRequest
    model_source_file_request: ModelSourceFileRequest
    model_source_prepare_request: ModelSourcePrepareRequest
    local_package_abort: LocalPackageAbort
    weights_upload_request: WeightsUploadRequest
    local_package_fetch_request: LocalPackageFetchRequest
    child_call_result: ChildCallResult
    native_source_command: NativeSourceCommand
    native_artifact_transfer: NativeArtifactTransfer
    def __init__(self, claim: _Optional[_Union[Claim, _Mapping]] = ..., desired_state: _Optional[_Union[DesiredWorkerState, _Mapping]] = ..., attempt_offer: _Optional[_Union[AttemptOffer, _Mapping]] = ..., cancel_attempt: _Optional[_Union[CancelAttempt, _Mapping]] = ..., outcome_ack: _Optional[_Union[AttemptOutcomeAck, _Mapping]] = ..., checkpoint_receipt: _Optional[_Union[JobCheckpointReceipt, _Mapping]] = ..., snapshot_ack: _Optional[_Union[SnapshotAck, _Mapping]] = ..., weights_finalize_request: _Optional[_Union[WeightsFinalizeRequest, _Mapping]] = ..., weights_host_ack: _Optional[_Union[WeightsHostAck, _Mapping]] = ..., weights_transfer_request: _Optional[_Union[WeightsTransferRequest, _Mapping]] = ..., weights_read_request: _Optional[_Union[WeightsReadRequest, _Mapping]] = ..., model_source_file_request: _Optional[_Union[ModelSourceFileRequest, _Mapping]] = ..., model_source_prepare_request: _Optional[_Union[ModelSourcePrepareRequest, _Mapping]] = ..., local_package_abort: _Optional[_Union[LocalPackageAbort, _Mapping]] = ..., weights_upload_request: _Optional[_Union[WeightsUploadRequest, _Mapping]] = ..., local_package_fetch_request: _Optional[_Union[LocalPackageFetchRequest, _Mapping]] = ..., child_call_result: _Optional[_Union[ChildCallResult, _Mapping]] = ..., native_source_command: _Optional[_Union[NativeSourceCommand, _Mapping]] = ..., native_artifact_transfer: _Optional[_Union[NativeArtifactTransfer, _Mapping]] = ...) -> None: ...

class WorkerFrame(_message.Message):
    __slots__ = ("claim_ack", "observed_state", "attempt_accepted", "attempt_outcome", "boot_failure", "checkpoint_request", "checkpoint_ack", "snapshot", "weights_finalize_result", "weights_intent", "weights_receipt", "weights_read_result", "weights_transfer_status", "model_source_file_status", "model_source_prepared", "local_package_file_status", "local_package_abort_status", "weights_upload_result", "weights_checkpoint", "weights_transaction", "child_call_request", "child_call_cancel", "native_source_status", "native_artifact_transfer_status")
    CLAIM_ACK_FIELD_NUMBER: _ClassVar[int]
    OBSERVED_STATE_FIELD_NUMBER: _ClassVar[int]
    ATTEMPT_ACCEPTED_FIELD_NUMBER: _ClassVar[int]
    ATTEMPT_OUTCOME_FIELD_NUMBER: _ClassVar[int]
    BOOT_FAILURE_FIELD_NUMBER: _ClassVar[int]
    CHECKPOINT_REQUEST_FIELD_NUMBER: _ClassVar[int]
    CHECKPOINT_ACK_FIELD_NUMBER: _ClassVar[int]
    SNAPSHOT_FIELD_NUMBER: _ClassVar[int]
    WEIGHTS_FINALIZE_RESULT_FIELD_NUMBER: _ClassVar[int]
    WEIGHTS_INTENT_FIELD_NUMBER: _ClassVar[int]
    WEIGHTS_RECEIPT_FIELD_NUMBER: _ClassVar[int]
    WEIGHTS_READ_RESULT_FIELD_NUMBER: _ClassVar[int]
    WEIGHTS_TRANSFER_STATUS_FIELD_NUMBER: _ClassVar[int]
    MODEL_SOURCE_FILE_STATUS_FIELD_NUMBER: _ClassVar[int]
    MODEL_SOURCE_PREPARED_FIELD_NUMBER: _ClassVar[int]
    LOCAL_PACKAGE_FILE_STATUS_FIELD_NUMBER: _ClassVar[int]
    LOCAL_PACKAGE_ABORT_STATUS_FIELD_NUMBER: _ClassVar[int]
    WEIGHTS_UPLOAD_RESULT_FIELD_NUMBER: _ClassVar[int]
    WEIGHTS_CHECKPOINT_FIELD_NUMBER: _ClassVar[int]
    WEIGHTS_TRANSACTION_FIELD_NUMBER: _ClassVar[int]
    CHILD_CALL_REQUEST_FIELD_NUMBER: _ClassVar[int]
    CHILD_CALL_CANCEL_FIELD_NUMBER: _ClassVar[int]
    NATIVE_SOURCE_STATUS_FIELD_NUMBER: _ClassVar[int]
    NATIVE_ARTIFACT_TRANSFER_STATUS_FIELD_NUMBER: _ClassVar[int]
    claim_ack: ClaimAck
    observed_state: ObservedWorkerState
    attempt_accepted: AttemptAccepted
    attempt_outcome: AttemptOutcome
    boot_failure: BootFailure
    checkpoint_request: JobCheckpointRequest
    checkpoint_ack: JobCheckpointAck
    snapshot: WorkerSnapshot
    weights_finalize_result: WeightsFinalizeResult
    weights_intent: WeightsIntentFrame
    weights_receipt: WeightsReceiptFrame
    weights_read_result: WeightsReadResult
    weights_transfer_status: WeightsTransferStatus
    model_source_file_status: ModelSourceFileStatus
    model_source_prepared: ModelSourcePrepared
    local_package_file_status: LocalPackageFileStatus
    local_package_abort_status: LocalPackageAbortStatus
    weights_upload_result: WeightsUploadResult
    weights_checkpoint: WeightsCheckpointFrame
    weights_transaction: WeightsTransactionStatus
    child_call_request: ChildCallRequest
    child_call_cancel: ChildCallCancel
    native_source_status: NativeSourceStatus
    native_artifact_transfer_status: NativeArtifactTransferStatus
    def __init__(self, claim_ack: _Optional[_Union[ClaimAck, _Mapping]] = ..., observed_state: _Optional[_Union[ObservedWorkerState, _Mapping]] = ..., attempt_accepted: _Optional[_Union[AttemptAccepted, _Mapping]] = ..., attempt_outcome: _Optional[_Union[AttemptOutcome, _Mapping]] = ..., boot_failure: _Optional[_Union[BootFailure, _Mapping]] = ..., checkpoint_request: _Optional[_Union[JobCheckpointRequest, _Mapping]] = ..., checkpoint_ack: _Optional[_Union[JobCheckpointAck, _Mapping]] = ..., snapshot: _Optional[_Union[WorkerSnapshot, _Mapping]] = ..., weights_finalize_result: _Optional[_Union[WeightsFinalizeResult, _Mapping]] = ..., weights_intent: _Optional[_Union[WeightsIntentFrame, _Mapping]] = ..., weights_receipt: _Optional[_Union[WeightsReceiptFrame, _Mapping]] = ..., weights_read_result: _Optional[_Union[WeightsReadResult, _Mapping]] = ..., weights_transfer_status: _Optional[_Union[WeightsTransferStatus, _Mapping]] = ..., model_source_file_status: _Optional[_Union[ModelSourceFileStatus, _Mapping]] = ..., model_source_prepared: _Optional[_Union[ModelSourcePrepared, _Mapping]] = ..., local_package_file_status: _Optional[_Union[LocalPackageFileStatus, _Mapping]] = ..., local_package_abort_status: _Optional[_Union[LocalPackageAbortStatus, _Mapping]] = ..., weights_upload_result: _Optional[_Union[WeightsUploadResult, _Mapping]] = ..., weights_checkpoint: _Optional[_Union[WeightsCheckpointFrame, _Mapping]] = ..., weights_transaction: _Optional[_Union[WeightsTransactionStatus, _Mapping]] = ..., child_call_request: _Optional[_Union[ChildCallRequest, _Mapping]] = ..., child_call_cancel: _Optional[_Union[ChildCallCancel, _Mapping]] = ..., native_source_status: _Optional[_Union[NativeSourceStatus, _Mapping]] = ..., native_artifact_transfer_status: _Optional[_Union[NativeArtifactTransferStatus, _Mapping]] = ...) -> None: ...

class ChildCallRequest(_message.Message):
    __slots__ = ("record_owner_epoch", "control_stream_epoch", "worker_boot_id", "parent_request_id", "parent_attempt_ordinal", "parent_invocation_spec_digest", "call_index", "interface_digest", "module", "export", "request_canonical_bytes", "intent_digest", "capture")
    RECORD_OWNER_EPOCH_FIELD_NUMBER: _ClassVar[int]
    CONTROL_STREAM_EPOCH_FIELD_NUMBER: _ClassVar[int]
    WORKER_BOOT_ID_FIELD_NUMBER: _ClassVar[int]
    PARENT_REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    PARENT_ATTEMPT_ORDINAL_FIELD_NUMBER: _ClassVar[int]
    PARENT_INVOCATION_SPEC_DIGEST_FIELD_NUMBER: _ClassVar[int]
    CALL_INDEX_FIELD_NUMBER: _ClassVar[int]
    INTERFACE_DIGEST_FIELD_NUMBER: _ClassVar[int]
    MODULE_FIELD_NUMBER: _ClassVar[int]
    EXPORT_FIELD_NUMBER: _ClassVar[int]
    REQUEST_CANONICAL_BYTES_FIELD_NUMBER: _ClassVar[int]
    INTENT_DIGEST_FIELD_NUMBER: _ClassVar[int]
    CAPTURE_FIELD_NUMBER: _ClassVar[int]
    record_owner_epoch: int
    control_stream_epoch: int
    worker_boot_id: str
    parent_request_id: str
    parent_attempt_ordinal: int
    parent_invocation_spec_digest: bytes
    call_index: int
    interface_digest: bytes
    module: str
    export: str
    request_canonical_bytes: bytes
    intent_digest: bytes
    capture: ActivationCapture
    def __init__(self, record_owner_epoch: _Optional[int] = ..., control_stream_epoch: _Optional[int] = ..., worker_boot_id: _Optional[str] = ..., parent_request_id: _Optional[str] = ..., parent_attempt_ordinal: _Optional[int] = ..., parent_invocation_spec_digest: _Optional[bytes] = ..., call_index: _Optional[int] = ..., interface_digest: _Optional[bytes] = ..., module: _Optional[str] = ..., export: _Optional[str] = ..., request_canonical_bytes: _Optional[bytes] = ..., intent_digest: _Optional[bytes] = ..., capture: _Optional[_Union[ActivationCapture, _Mapping]] = ...) -> None: ...

class ChildCallCancel(_message.Message):
    __slots__ = ("record_owner_epoch", "control_stream_epoch", "worker_boot_id", "parent_request_id", "parent_attempt_ordinal", "parent_invocation_spec_digest", "call_index", "intent_digest")
    RECORD_OWNER_EPOCH_FIELD_NUMBER: _ClassVar[int]
    CONTROL_STREAM_EPOCH_FIELD_NUMBER: _ClassVar[int]
    WORKER_BOOT_ID_FIELD_NUMBER: _ClassVar[int]
    PARENT_REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    PARENT_ATTEMPT_ORDINAL_FIELD_NUMBER: _ClassVar[int]
    PARENT_INVOCATION_SPEC_DIGEST_FIELD_NUMBER: _ClassVar[int]
    CALL_INDEX_FIELD_NUMBER: _ClassVar[int]
    INTENT_DIGEST_FIELD_NUMBER: _ClassVar[int]
    record_owner_epoch: int
    control_stream_epoch: int
    worker_boot_id: str
    parent_request_id: str
    parent_attempt_ordinal: int
    parent_invocation_spec_digest: bytes
    call_index: int
    intent_digest: bytes
    def __init__(self, record_owner_epoch: _Optional[int] = ..., control_stream_epoch: _Optional[int] = ..., worker_boot_id: _Optional[str] = ..., parent_request_id: _Optional[str] = ..., parent_attempt_ordinal: _Optional[int] = ..., parent_invocation_spec_digest: _Optional[bytes] = ..., call_index: _Optional[int] = ..., intent_digest: _Optional[bytes] = ...) -> None: ...

class ChildCallResult(_message.Message):
    __slots__ = ("record_owner_epoch", "control_stream_epoch", "worker_boot_id", "parent_request_id", "parent_attempt_ordinal", "parent_invocation_spec_digest", "call_index", "intent_digest", "child_request_id", "state", "result_canonical_bytes", "safe_code", "safe_detail", "byte_result_grants", "observation")
    RECORD_OWNER_EPOCH_FIELD_NUMBER: _ClassVar[int]
    CONTROL_STREAM_EPOCH_FIELD_NUMBER: _ClassVar[int]
    WORKER_BOOT_ID_FIELD_NUMBER: _ClassVar[int]
    PARENT_REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    PARENT_ATTEMPT_ORDINAL_FIELD_NUMBER: _ClassVar[int]
    PARENT_INVOCATION_SPEC_DIGEST_FIELD_NUMBER: _ClassVar[int]
    CALL_INDEX_FIELD_NUMBER: _ClassVar[int]
    INTENT_DIGEST_FIELD_NUMBER: _ClassVar[int]
    CHILD_REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    STATE_FIELD_NUMBER: _ClassVar[int]
    RESULT_CANONICAL_BYTES_FIELD_NUMBER: _ClassVar[int]
    SAFE_CODE_FIELD_NUMBER: _ClassVar[int]
    SAFE_DETAIL_FIELD_NUMBER: _ClassVar[int]
    BYTE_RESULT_GRANTS_FIELD_NUMBER: _ClassVar[int]
    OBSERVATION_FIELD_NUMBER: _ClassVar[int]
    record_owner_epoch: int
    control_stream_epoch: int
    worker_boot_id: str
    parent_request_id: str
    parent_attempt_ordinal: int
    parent_invocation_spec_digest: bytes
    call_index: int
    intent_digest: bytes
    child_request_id: str
    state: ChildCallState
    result_canonical_bytes: bytes
    safe_code: str
    safe_detail: str
    byte_result_grants: _containers.RepeatedCompositeFieldContainer[ChildByteResultGrant]
    observation: ExecutionObservation
    def __init__(self, record_owner_epoch: _Optional[int] = ..., control_stream_epoch: _Optional[int] = ..., worker_boot_id: _Optional[str] = ..., parent_request_id: _Optional[str] = ..., parent_attempt_ordinal: _Optional[int] = ..., parent_invocation_spec_digest: _Optional[bytes] = ..., call_index: _Optional[int] = ..., intent_digest: _Optional[bytes] = ..., child_request_id: _Optional[str] = ..., state: _Optional[_Union[ChildCallState, str]] = ..., result_canonical_bytes: _Optional[bytes] = ..., safe_code: _Optional[str] = ..., safe_detail: _Optional[str] = ..., byte_result_grants: _Optional[_Iterable[_Union[ChildByteResultGrant, _Mapping]]] = ..., observation: _Optional[_Union[ExecutionObservation, _Mapping]] = ...) -> None: ...

class NativeSourceMember(_message.Message):
    __slots__ = ("member", "object", "url")
    MEMBER_FIELD_NUMBER: _ClassVar[int]
    OBJECT_FIELD_NUMBER: _ClassVar[int]
    URL_FIELD_NUMBER: _ClassVar[int]
    member: str
    object: Ref
    url: str
    def __init__(self, member: _Optional[str] = ..., object: _Optional[_Union[Ref, _Mapping]] = ..., url: _Optional[str] = ...) -> None: ...

class NativeSourceSelection(_message.Message):
    __slots__ = ("canonical", "selection_digest", "content_manifest", "members", "allowed_hosts", "credential_hosts")
    CANONICAL_FIELD_NUMBER: _ClassVar[int]
    SELECTION_DIGEST_FIELD_NUMBER: _ClassVar[int]
    CONTENT_MANIFEST_FIELD_NUMBER: _ClassVar[int]
    MEMBERS_FIELD_NUMBER: _ClassVar[int]
    ALLOWED_HOSTS_FIELD_NUMBER: _ClassVar[int]
    CREDENTIAL_HOSTS_FIELD_NUMBER: _ClassVar[int]
    canonical: str
    selection_digest: bytes
    content_manifest: Ref
    members: _containers.RepeatedCompositeFieldContainer[NativeSourceMember]
    allowed_hosts: _containers.RepeatedScalarFieldContainer[str]
    credential_hosts: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, canonical: _Optional[str] = ..., selection_digest: _Optional[bytes] = ..., content_manifest: _Optional[_Union[Ref, _Mapping]] = ..., members: _Optional[_Iterable[_Union[NativeSourceMember, _Mapping]]] = ..., allowed_hosts: _Optional[_Iterable[str]] = ..., credential_hosts: _Optional[_Iterable[str]] = ...) -> None: ...

class NativeSourceCommand(_message.Message):
    __slots__ = ("record_owner_epoch", "control_stream_epoch", "worker_boot_id", "parent_call", "service_id", "operation", "phase", "selection", "credential")
    RECORD_OWNER_EPOCH_FIELD_NUMBER: _ClassVar[int]
    CONTROL_STREAM_EPOCH_FIELD_NUMBER: _ClassVar[int]
    WORKER_BOOT_ID_FIELD_NUMBER: _ClassVar[int]
    PARENT_CALL_FIELD_NUMBER: _ClassVar[int]
    SERVICE_ID_FIELD_NUMBER: _ClassVar[int]
    OPERATION_FIELD_NUMBER: _ClassVar[int]
    PHASE_FIELD_NUMBER: _ClassVar[int]
    SELECTION_FIELD_NUMBER: _ClassVar[int]
    CREDENTIAL_FIELD_NUMBER: _ClassVar[int]
    record_owner_epoch: int
    control_stream_epoch: int
    worker_boot_id: str
    parent_call: ChildCallRequest
    service_id: str
    operation: NativeSourceOperation
    phase: NativeSourcePhase
    selection: NativeSourceSelection
    credential: str
    def __init__(self, record_owner_epoch: _Optional[int] = ..., control_stream_epoch: _Optional[int] = ..., worker_boot_id: _Optional[str] = ..., parent_call: _Optional[_Union[ChildCallRequest, _Mapping]] = ..., service_id: _Optional[str] = ..., operation: _Optional[_Union[NativeSourceOperation, str]] = ..., phase: _Optional[_Union[NativeSourcePhase, str]] = ..., selection: _Optional[_Union[NativeSourceSelection, _Mapping]] = ..., credential: _Optional[str] = ...) -> None: ...

class NativeSourceStatus(_message.Message):
    __slots__ = ("record_owner_epoch", "control_stream_epoch", "worker_boot_id", "parent_request_id", "parent_attempt_ordinal", "parent_invocation_spec_digest", "call_index", "intent_digest", "service_id", "state", "selection", "result_canonical_bytes", "safe_code", "safe_detail", "native_receipt_canonical_bytes", "computation_digest", "memo_hit", "byte_output", "byte_output_attempt_ordinal", "byte_output_invocation_spec_digest")
    RECORD_OWNER_EPOCH_FIELD_NUMBER: _ClassVar[int]
    CONTROL_STREAM_EPOCH_FIELD_NUMBER: _ClassVar[int]
    WORKER_BOOT_ID_FIELD_NUMBER: _ClassVar[int]
    PARENT_REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    PARENT_ATTEMPT_ORDINAL_FIELD_NUMBER: _ClassVar[int]
    PARENT_INVOCATION_SPEC_DIGEST_FIELD_NUMBER: _ClassVar[int]
    CALL_INDEX_FIELD_NUMBER: _ClassVar[int]
    INTENT_DIGEST_FIELD_NUMBER: _ClassVar[int]
    SERVICE_ID_FIELD_NUMBER: _ClassVar[int]
    STATE_FIELD_NUMBER: _ClassVar[int]
    SELECTION_FIELD_NUMBER: _ClassVar[int]
    RESULT_CANONICAL_BYTES_FIELD_NUMBER: _ClassVar[int]
    SAFE_CODE_FIELD_NUMBER: _ClassVar[int]
    SAFE_DETAIL_FIELD_NUMBER: _ClassVar[int]
    NATIVE_RECEIPT_CANONICAL_BYTES_FIELD_NUMBER: _ClassVar[int]
    COMPUTATION_DIGEST_FIELD_NUMBER: _ClassVar[int]
    MEMO_HIT_FIELD_NUMBER: _ClassVar[int]
    BYTE_OUTPUT_FIELD_NUMBER: _ClassVar[int]
    BYTE_OUTPUT_ATTEMPT_ORDINAL_FIELD_NUMBER: _ClassVar[int]
    BYTE_OUTPUT_INVOCATION_SPEC_DIGEST_FIELD_NUMBER: _ClassVar[int]
    record_owner_epoch: int
    control_stream_epoch: int
    worker_boot_id: str
    parent_request_id: str
    parent_attempt_ordinal: int
    parent_invocation_spec_digest: bytes
    call_index: int
    intent_digest: bytes
    service_id: str
    state: NativeSourceState
    selection: NativeSourceSelection
    result_canonical_bytes: bytes
    safe_code: str
    safe_detail: str
    native_receipt_canonical_bytes: bytes
    computation_digest: bytes
    memo_hit: bool
    byte_output: NativeByteTreeRef
    byte_output_attempt_ordinal: int
    byte_output_invocation_spec_digest: bytes
    def __init__(self, record_owner_epoch: _Optional[int] = ..., control_stream_epoch: _Optional[int] = ..., worker_boot_id: _Optional[str] = ..., parent_request_id: _Optional[str] = ..., parent_attempt_ordinal: _Optional[int] = ..., parent_invocation_spec_digest: _Optional[bytes] = ..., call_index: _Optional[int] = ..., intent_digest: _Optional[bytes] = ..., service_id: _Optional[str] = ..., state: _Optional[_Union[NativeSourceState, str]] = ..., selection: _Optional[_Union[NativeSourceSelection, _Mapping]] = ..., result_canonical_bytes: _Optional[bytes] = ..., safe_code: _Optional[str] = ..., safe_detail: _Optional[str] = ..., native_receipt_canonical_bytes: _Optional[bytes] = ..., computation_digest: _Optional[bytes] = ..., memo_hit: bool = ..., byte_output: _Optional[_Union[NativeByteTreeRef, _Mapping]] = ..., byte_output_attempt_ordinal: _Optional[int] = ..., byte_output_invocation_spec_digest: _Optional[bytes] = ...) -> None: ...

class Claim(_message.Message):
    __slots__ = ("record_owner_epoch", "control_stream_epoch", "worker_boot_id", "record_owner_id", "worker_id", "wire_minor", "proof")
    RECORD_OWNER_EPOCH_FIELD_NUMBER: _ClassVar[int]
    CONTROL_STREAM_EPOCH_FIELD_NUMBER: _ClassVar[int]
    WORKER_BOOT_ID_FIELD_NUMBER: _ClassVar[int]
    RECORD_OWNER_ID_FIELD_NUMBER: _ClassVar[int]
    WORKER_ID_FIELD_NUMBER: _ClassVar[int]
    WIRE_MINOR_FIELD_NUMBER: _ClassVar[int]
    PROOF_FIELD_NUMBER: _ClassVar[int]
    record_owner_epoch: int
    control_stream_epoch: int
    worker_boot_id: str
    record_owner_id: str
    worker_id: str
    wire_minor: int
    proof: bytes
    def __init__(self, record_owner_epoch: _Optional[int] = ..., control_stream_epoch: _Optional[int] = ..., worker_boot_id: _Optional[str] = ..., record_owner_id: _Optional[str] = ..., worker_id: _Optional[str] = ..., wire_minor: _Optional[int] = ..., proof: _Optional[bytes] = ...) -> None: ...

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
    __slots__ = ("record_owner_epoch", "control_stream_epoch", "worker_boot_id", "accepted", "rejection", "wire_minor", "worker_id", "worker_instance_id", "worker_release_id", "control_runtime_digest", "git_commit", "resources")
    RECORD_OWNER_EPOCH_FIELD_NUMBER: _ClassVar[int]
    CONTROL_STREAM_EPOCH_FIELD_NUMBER: _ClassVar[int]
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
    control_stream_epoch: int
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
    def __init__(self, record_owner_epoch: _Optional[int] = ..., control_stream_epoch: _Optional[int] = ..., worker_boot_id: _Optional[str] = ..., accepted: bool = ..., rejection: _Optional[_Union[ClaimRejection, str]] = ..., wire_minor: _Optional[int] = ..., worker_id: _Optional[str] = ..., worker_instance_id: _Optional[str] = ..., worker_release_id: _Optional[str] = ..., control_runtime_digest: _Optional[str] = ..., git_commit: _Optional[str] = ..., resources: _Optional[_Union[WorkerResources, _Mapping]] = ...) -> None: ...

class BootFailure(_message.Message):
    __slots__ = ("record_owner_epoch", "control_stream_epoch", "worker_boot_id", "worker_id", "worker_instance_id", "reason", "detail", "resources", "control_runtime_digest", "worker_release_id")
    RECORD_OWNER_EPOCH_FIELD_NUMBER: _ClassVar[int]
    CONTROL_STREAM_EPOCH_FIELD_NUMBER: _ClassVar[int]
    WORKER_BOOT_ID_FIELD_NUMBER: _ClassVar[int]
    WORKER_ID_FIELD_NUMBER: _ClassVar[int]
    WORKER_INSTANCE_ID_FIELD_NUMBER: _ClassVar[int]
    REASON_FIELD_NUMBER: _ClassVar[int]
    DETAIL_FIELD_NUMBER: _ClassVar[int]
    RESOURCES_FIELD_NUMBER: _ClassVar[int]
    CONTROL_RUNTIME_DIGEST_FIELD_NUMBER: _ClassVar[int]
    WORKER_RELEASE_ID_FIELD_NUMBER: _ClassVar[int]
    record_owner_epoch: int
    control_stream_epoch: int
    worker_boot_id: str
    worker_id: str
    worker_instance_id: str
    reason: BootFailureReason
    detail: str
    resources: WorkerResources
    control_runtime_digest: str
    worker_release_id: str
    def __init__(self, record_owner_epoch: _Optional[int] = ..., control_stream_epoch: _Optional[int] = ..., worker_boot_id: _Optional[str] = ..., worker_id: _Optional[str] = ..., worker_instance_id: _Optional[str] = ..., reason: _Optional[_Union[BootFailureReason, str]] = ..., detail: _Optional[str] = ..., resources: _Optional[_Union[WorkerResources, _Mapping]] = ..., control_runtime_digest: _Optional[str] = ..., worker_release_id: _Optional[str] = ...) -> None: ...

class WorkerSnapshot(_message.Message):
    __slots__ = ("record_owner_epoch", "control_stream_epoch", "worker_boot_id", "snapshot_id", "snapshot_digest", "snapshot_canonical_bytes", "accepted_placement_set_canonical_bytes", "host_snapshot_digest", "host_snapshot_canonical_bytes")
    RECORD_OWNER_EPOCH_FIELD_NUMBER: _ClassVar[int]
    CONTROL_STREAM_EPOCH_FIELD_NUMBER: _ClassVar[int]
    WORKER_BOOT_ID_FIELD_NUMBER: _ClassVar[int]
    SNAPSHOT_ID_FIELD_NUMBER: _ClassVar[int]
    SNAPSHOT_DIGEST_FIELD_NUMBER: _ClassVar[int]
    SNAPSHOT_CANONICAL_BYTES_FIELD_NUMBER: _ClassVar[int]
    ACCEPTED_PLACEMENT_SET_CANONICAL_BYTES_FIELD_NUMBER: _ClassVar[int]
    HOST_SNAPSHOT_DIGEST_FIELD_NUMBER: _ClassVar[int]
    HOST_SNAPSHOT_CANONICAL_BYTES_FIELD_NUMBER: _ClassVar[int]
    record_owner_epoch: int
    control_stream_epoch: int
    worker_boot_id: str
    snapshot_id: str
    snapshot_digest: bytes
    snapshot_canonical_bytes: bytes
    accepted_placement_set_canonical_bytes: bytes
    host_snapshot_digest: bytes
    host_snapshot_canonical_bytes: bytes
    def __init__(self, record_owner_epoch: _Optional[int] = ..., control_stream_epoch: _Optional[int] = ..., worker_boot_id: _Optional[str] = ..., snapshot_id: _Optional[str] = ..., snapshot_digest: _Optional[bytes] = ..., snapshot_canonical_bytes: _Optional[bytes] = ..., accepted_placement_set_canonical_bytes: _Optional[bytes] = ..., host_snapshot_digest: _Optional[bytes] = ..., host_snapshot_canonical_bytes: _Optional[bytes] = ...) -> None: ...

class WorkerSnapshotBody(_message.Message):
    __slots__ = ("accepted_desired_state_revision", "accepted_placement_set_digest", "journal_highwater", "worker_phase", "placements", "converged_revision", "admission_epoch", "admission_state", "available_attempt_slots", "held_attempts", "weights_transactions", "lanes", "held_manifests")
    ACCEPTED_DESIRED_STATE_REVISION_FIELD_NUMBER: _ClassVar[int]
    ACCEPTED_PLACEMENT_SET_DIGEST_FIELD_NUMBER: _ClassVar[int]
    JOURNAL_HIGHWATER_FIELD_NUMBER: _ClassVar[int]
    WORKER_PHASE_FIELD_NUMBER: _ClassVar[int]
    PLACEMENTS_FIELD_NUMBER: _ClassVar[int]
    CONVERGED_REVISION_FIELD_NUMBER: _ClassVar[int]
    ADMISSION_EPOCH_FIELD_NUMBER: _ClassVar[int]
    ADMISSION_STATE_FIELD_NUMBER: _ClassVar[int]
    AVAILABLE_ATTEMPT_SLOTS_FIELD_NUMBER: _ClassVar[int]
    HELD_ATTEMPTS_FIELD_NUMBER: _ClassVar[int]
    WEIGHTS_TRANSACTIONS_FIELD_NUMBER: _ClassVar[int]
    LANES_FIELD_NUMBER: _ClassVar[int]
    HELD_MANIFESTS_FIELD_NUMBER: _ClassVar[int]
    accepted_desired_state_revision: int
    accepted_placement_set_digest: bytes
    journal_highwater: int
    worker_phase: WorkerPhase
    placements: _containers.RepeatedCompositeFieldContainer[PlacementStatus]
    converged_revision: int
    admission_epoch: int
    admission_state: AdmissionState
    available_attempt_slots: int
    held_attempts: _containers.RepeatedCompositeFieldContainer[HeldAttempt]
    weights_transactions: _containers.RepeatedCompositeFieldContainer[WeightsTransactionStatus]
    lanes: _containers.RepeatedCompositeFieldContainer[DeviceLane]
    held_manifests: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, accepted_desired_state_revision: _Optional[int] = ..., accepted_placement_set_digest: _Optional[bytes] = ..., journal_highwater: _Optional[int] = ..., worker_phase: _Optional[_Union[WorkerPhase, str]] = ..., placements: _Optional[_Iterable[_Union[PlacementStatus, _Mapping]]] = ..., converged_revision: _Optional[int] = ..., admission_epoch: _Optional[int] = ..., admission_state: _Optional[_Union[AdmissionState, str]] = ..., available_attempt_slots: _Optional[int] = ..., held_attempts: _Optional[_Iterable[_Union[HeldAttempt, _Mapping]]] = ..., weights_transactions: _Optional[_Iterable[_Union[WeightsTransactionStatus, _Mapping]]] = ..., lanes: _Optional[_Iterable[_Union[DeviceLane, _Mapping]]] = ..., held_manifests: _Optional[_Iterable[str]] = ...) -> None: ...

class HostSnapshotBody(_message.Message):
    __slots__ = ("held_outcomes", "weights_transactions", "retained_desired_revision")
    HELD_OUTCOMES_FIELD_NUMBER: _ClassVar[int]
    WEIGHTS_TRANSACTIONS_FIELD_NUMBER: _ClassVar[int]
    RETAINED_DESIRED_REVISION_FIELD_NUMBER: _ClassVar[int]
    held_outcomes: _containers.RepeatedCompositeFieldContainer[HeldAttempt]
    weights_transactions: _containers.RepeatedCompositeFieldContainer[WeightsTransactionStatus]
    retained_desired_revision: int
    def __init__(self, held_outcomes: _Optional[_Iterable[_Union[HeldAttempt, _Mapping]]] = ..., weights_transactions: _Optional[_Iterable[_Union[WeightsTransactionStatus, _Mapping]]] = ..., retained_desired_revision: _Optional[int] = ...) -> None: ...

class SnapshotAck(_message.Message):
    __slots__ = ("record_owner_epoch", "control_stream_epoch", "worker_boot_id", "snapshot_id", "snapshot_digest", "host_snapshot_digest")
    RECORD_OWNER_EPOCH_FIELD_NUMBER: _ClassVar[int]
    CONTROL_STREAM_EPOCH_FIELD_NUMBER: _ClassVar[int]
    WORKER_BOOT_ID_FIELD_NUMBER: _ClassVar[int]
    SNAPSHOT_ID_FIELD_NUMBER: _ClassVar[int]
    SNAPSHOT_DIGEST_FIELD_NUMBER: _ClassVar[int]
    HOST_SNAPSHOT_DIGEST_FIELD_NUMBER: _ClassVar[int]
    record_owner_epoch: int
    control_stream_epoch: int
    worker_boot_id: str
    snapshot_id: str
    snapshot_digest: bytes
    host_snapshot_digest: bytes
    def __init__(self, record_owner_epoch: _Optional[int] = ..., control_stream_epoch: _Optional[int] = ..., worker_boot_id: _Optional[str] = ..., snapshot_id: _Optional[str] = ..., snapshot_digest: _Optional[bytes] = ..., host_snapshot_digest: _Optional[bytes] = ...) -> None: ...

class DesiredWorkerState(_message.Message):
    __slots__ = ("record_owner_epoch", "control_stream_epoch", "worker_boot_id", "revision", "posture", "wire_minor", "drain_grace_ms", "job", "placement_set", "package_set", "local_package_set", "private_placement_set", "runtime_revision")
    RECORD_OWNER_EPOCH_FIELD_NUMBER: _ClassVar[int]
    CONTROL_STREAM_EPOCH_FIELD_NUMBER: _ClassVar[int]
    WORKER_BOOT_ID_FIELD_NUMBER: _ClassVar[int]
    REVISION_FIELD_NUMBER: _ClassVar[int]
    POSTURE_FIELD_NUMBER: _ClassVar[int]
    WIRE_MINOR_FIELD_NUMBER: _ClassVar[int]
    DRAIN_GRACE_MS_FIELD_NUMBER: _ClassVar[int]
    JOB_FIELD_NUMBER: _ClassVar[int]
    PLACEMENT_SET_FIELD_NUMBER: _ClassVar[int]
    PACKAGE_SET_FIELD_NUMBER: _ClassVar[int]
    LOCAL_PACKAGE_SET_FIELD_NUMBER: _ClassVar[int]
    PRIVATE_PLACEMENT_SET_FIELD_NUMBER: _ClassVar[int]
    RUNTIME_REVISION_FIELD_NUMBER: _ClassVar[int]
    record_owner_epoch: int
    control_stream_epoch: int
    worker_boot_id: str
    revision: int
    posture: Posture
    wire_minor: int
    drain_grace_ms: int
    job: JobDirective
    placement_set: DesiredPlacementSet
    package_set: DesiredPackageSet
    local_package_set: DesiredLocalPackageSet
    private_placement_set: DesiredPrivatePlacementSet
    runtime_revision: RuntimeRevision
    def __init__(self, record_owner_epoch: _Optional[int] = ..., control_stream_epoch: _Optional[int] = ..., worker_boot_id: _Optional[str] = ..., revision: _Optional[int] = ..., posture: _Optional[_Union[Posture, str]] = ..., wire_minor: _Optional[int] = ..., drain_grace_ms: _Optional[int] = ..., job: _Optional[_Union[JobDirective, _Mapping]] = ..., placement_set: _Optional[_Union[DesiredPlacementSet, _Mapping]] = ..., package_set: _Optional[_Union[DesiredPackageSet, _Mapping]] = ..., local_package_set: _Optional[_Union[DesiredLocalPackageSet, _Mapping]] = ..., private_placement_set: _Optional[_Union[DesiredPrivatePlacementSet, _Mapping]] = ..., runtime_revision: _Optional[_Union[RuntimeRevision, _Mapping]] = ...) -> None: ...

class DesiredPackageSet(_message.Message):
    __slots__ = ("download_delegation", "download_delegation_signature")
    DOWNLOAD_DELEGATION_FIELD_NUMBER: _ClassVar[int]
    DOWNLOAD_DELEGATION_SIGNATURE_FIELD_NUMBER: _ClassVar[int]
    download_delegation: bytes
    download_delegation_signature: bytes
    def __init__(self, download_delegation: _Optional[bytes] = ..., download_delegation_signature: _Optional[bytes] = ...) -> None: ...

class DesiredLocalPackageSet(_message.Message):
    __slots__ = ("operation_id", "package", "files", "dependency_requirements", "python_requires", "python_version")
    OPERATION_ID_FIELD_NUMBER: _ClassVar[int]
    PACKAGE_FIELD_NUMBER: _ClassVar[int]
    FILES_FIELD_NUMBER: _ClassVar[int]
    DEPENDENCY_REQUIREMENTS_FIELD_NUMBER: _ClassVar[int]
    PYTHON_REQUIRES_FIELD_NUMBER: _ClassVar[int]
    PYTHON_VERSION_FIELD_NUMBER: _ClassVar[int]
    operation_id: str
    package: DevelopmentPackage
    files: _containers.RepeatedCompositeFieldContainer[LocalPackageFileRef]
    dependency_requirements: bytes
    python_requires: str
    python_version: str
    def __init__(self, operation_id: _Optional[str] = ..., package: _Optional[_Union[DevelopmentPackage, _Mapping]] = ..., files: _Optional[_Iterable[_Union[LocalPackageFileRef, _Mapping]]] = ..., dependency_requirements: _Optional[bytes] = ..., python_requires: _Optional[str] = ..., python_version: _Optional[str] = ...) -> None: ...

class DesiredPrivatePlacementSet(_message.Message):
    __slots__ = ("operation_id", "local_revision_digest", "download_delegation", "download_delegation_signature", "native_models")
    OPERATION_ID_FIELD_NUMBER: _ClassVar[int]
    LOCAL_REVISION_DIGEST_FIELD_NUMBER: _ClassVar[int]
    DOWNLOAD_DELEGATION_FIELD_NUMBER: _ClassVar[int]
    DOWNLOAD_DELEGATION_SIGNATURE_FIELD_NUMBER: _ClassVar[int]
    NATIVE_MODELS_FIELD_NUMBER: _ClassVar[int]
    operation_id: str
    local_revision_digest: bytes
    download_delegation: bytes
    download_delegation_signature: bytes
    native_models: _containers.RepeatedCompositeFieldContainer[NativeModelBinding]
    def __init__(self, operation_id: _Optional[str] = ..., local_revision_digest: _Optional[bytes] = ..., download_delegation: _Optional[bytes] = ..., download_delegation_signature: _Optional[bytes] = ..., native_models: _Optional[_Iterable[_Union[NativeModelBinding, _Mapping]]] = ...) -> None: ...

class NativeModelBinding(_message.Message):
    __slots__ = ("slot", "model", "manifest", "retention", "adapters")
    SLOT_FIELD_NUMBER: _ClassVar[int]
    MODEL_FIELD_NUMBER: _ClassVar[int]
    MANIFEST_FIELD_NUMBER: _ClassVar[int]
    RETENTION_FIELD_NUMBER: _ClassVar[int]
    ADAPTERS_FIELD_NUMBER: _ClassVar[int]
    slot: str
    model: str
    manifest: Ref
    retention: DerivedRetentionRequest
    adapters: _containers.RepeatedCompositeFieldContainer[DownloadAdapterRef]
    def __init__(self, slot: _Optional[str] = ..., model: _Optional[str] = ..., manifest: _Optional[_Union[Ref, _Mapping]] = ..., retention: _Optional[_Union[DerivedRetentionRequest, _Mapping]] = ..., adapters: _Optional[_Iterable[_Union[DownloadAdapterRef, _Mapping]]] = ...) -> None: ...

class LocalPackageFileRef(_message.Message):
    __slots__ = ("digest", "filename", "length")
    DIGEST_FIELD_NUMBER: _ClassVar[int]
    FILENAME_FIELD_NUMBER: _ClassVar[int]
    LENGTH_FIELD_NUMBER: _ClassVar[int]
    digest: bytes
    filename: str
    length: int
    def __init__(self, digest: _Optional[bytes] = ..., filename: _Optional[str] = ..., length: _Optional[int] = ...) -> None: ...

class LocalPackageRevision(_message.Message):
    __slots__ = ("package", "release", "source_digest", "package_interface", "files", "dependency_requirements", "python_requires", "python_version")
    PACKAGE_FIELD_NUMBER: _ClassVar[int]
    RELEASE_FIELD_NUMBER: _ClassVar[int]
    SOURCE_DIGEST_FIELD_NUMBER: _ClassVar[int]
    PACKAGE_INTERFACE_FIELD_NUMBER: _ClassVar[int]
    FILES_FIELD_NUMBER: _ClassVar[int]
    DEPENDENCY_REQUIREMENTS_FIELD_NUMBER: _ClassVar[int]
    PYTHON_REQUIRES_FIELD_NUMBER: _ClassVar[int]
    PYTHON_VERSION_FIELD_NUMBER: _ClassVar[int]
    package: str
    release: str
    source_digest: bytes
    package_interface: Ref
    files: _containers.RepeatedCompositeFieldContainer[LocalPackageFileRef]
    dependency_requirements: Ref
    python_requires: str
    python_version: str
    def __init__(self, package: _Optional[str] = ..., release: _Optional[str] = ..., source_digest: _Optional[bytes] = ..., package_interface: _Optional[_Union[Ref, _Mapping]] = ..., files: _Optional[_Iterable[_Union[LocalPackageFileRef, _Mapping]]] = ..., dependency_requirements: _Optional[_Union[Ref, _Mapping]] = ..., python_requires: _Optional[str] = ..., python_version: _Optional[str] = ...) -> None: ...

class DesiredPlacementSet(_message.Message):
    __slots__ = ("placement_set_digest", "placement_set_canonical_bytes", "device_pins", "orchestration_parent")
    PLACEMENT_SET_DIGEST_FIELD_NUMBER: _ClassVar[int]
    PLACEMENT_SET_CANONICAL_BYTES_FIELD_NUMBER: _ClassVar[int]
    DEVICE_PINS_FIELD_NUMBER: _ClassVar[int]
    ORCHESTRATION_PARENT_FIELD_NUMBER: _ClassVar[int]
    placement_set_digest: bytes
    placement_set_canonical_bytes: bytes
    device_pins: _containers.RepeatedCompositeFieldContainer[PlacementDevicePin]
    orchestration_parent: JobDirective
    def __init__(self, placement_set_digest: _Optional[bytes] = ..., placement_set_canonical_bytes: _Optional[bytes] = ..., device_pins: _Optional[_Iterable[_Union[PlacementDevicePin, _Mapping]]] = ..., orchestration_parent: _Optional[_Union[JobDirective, _Mapping]] = ...) -> None: ...

class PlacementDevicePin(_message.Message):
    __slots__ = ("placement_id", "device_ordinals")
    PLACEMENT_ID_FIELD_NUMBER: _ClassVar[int]
    DEVICE_ORDINALS_FIELD_NUMBER: _ClassVar[int]
    placement_id: str
    device_ordinals: _containers.RepeatedScalarFieldContainer[int]
    def __init__(self, placement_id: _Optional[str] = ..., device_ordinals: _Optional[_Iterable[int]] = ...) -> None: ...

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
    __slots__ = ("manifest", "model", "release", "package", "slot", "lane", "adapters")
    MANIFEST_FIELD_NUMBER: _ClassVar[int]
    MODEL_FIELD_NUMBER: _ClassVar[int]
    RELEASE_FIELD_NUMBER: _ClassVar[int]
    PACKAGE_FIELD_NUMBER: _ClassVar[int]
    SLOT_FIELD_NUMBER: _ClassVar[int]
    LANE_FIELD_NUMBER: _ClassVar[int]
    ADAPTERS_FIELD_NUMBER: _ClassVar[int]
    manifest: str
    model: str
    release: str
    package: str
    slot: str
    lane: str
    adapters: _containers.RepeatedCompositeFieldContainer[DownloadAdapterRef]
    def __init__(self, manifest: _Optional[str] = ..., model: _Optional[str] = ..., release: _Optional[str] = ..., package: _Optional[str] = ..., slot: _Optional[str] = ..., lane: _Optional[str] = ..., adapters: _Optional[_Iterable[_Union[DownloadAdapterRef, _Mapping]]] = ...) -> None: ...

class DownloadAdapterRef(_message.Message):
    __slots__ = ("component", "model", "release", "lane", "manifest", "source_component", "scale")
    COMPONENT_FIELD_NUMBER: _ClassVar[int]
    MODEL_FIELD_NUMBER: _ClassVar[int]
    RELEASE_FIELD_NUMBER: _ClassVar[int]
    LANE_FIELD_NUMBER: _ClassVar[int]
    MANIFEST_FIELD_NUMBER: _ClassVar[int]
    SOURCE_COMPONENT_FIELD_NUMBER: _ClassVar[int]
    SCALE_FIELD_NUMBER: _ClassVar[int]
    component: str
    model: str
    release: str
    lane: str
    manifest: str
    source_component: str
    scale: str
    def __init__(self, component: _Optional[str] = ..., model: _Optional[str] = ..., release: _Optional[str] = ..., lane: _Optional[str] = ..., manifest: _Optional[str] = ..., source_component: _Optional[str] = ..., scale: _Optional[str] = ...) -> None: ...

class DownloadPackageRef(_message.Message):
    __slots__ = ("package", "release")
    PACKAGE_FIELD_NUMBER: _ClassVar[int]
    RELEASE_FIELD_NUMBER: _ClassVar[int]
    package: str
    release: str
    def __init__(self, package: _Optional[str] = ..., release: _Optional[str] = ...) -> None: ...

class Placement(_message.Message):
    __slots__ = ("placement_id", "package", "development", "environment_digest", "package_interface", "bindings_digest", "models", "entrypoints", "environment")
    PLACEMENT_ID_FIELD_NUMBER: _ClassVar[int]
    PACKAGE_FIELD_NUMBER: _ClassVar[int]
    DEVELOPMENT_FIELD_NUMBER: _ClassVar[int]
    ENVIRONMENT_DIGEST_FIELD_NUMBER: _ClassVar[int]
    PACKAGE_INTERFACE_FIELD_NUMBER: _ClassVar[int]
    BINDINGS_DIGEST_FIELD_NUMBER: _ClassVar[int]
    MODELS_FIELD_NUMBER: _ClassVar[int]
    ENTRYPOINTS_FIELD_NUMBER: _ClassVar[int]
    ENVIRONMENT_FIELD_NUMBER: _ClassVar[int]
    placement_id: str
    package: PackageSelection
    development: DevelopmentPackage
    environment_digest: bytes
    package_interface: Ref
    bindings_digest: bytes
    models: _containers.RepeatedCompositeFieldContainer[Model]
    entrypoints: _containers.RepeatedCompositeFieldContainer[Entrypoint]
    environment: Environment
    def __init__(self, placement_id: _Optional[str] = ..., package: _Optional[_Union[PackageSelection, _Mapping]] = ..., development: _Optional[_Union[DevelopmentPackage, _Mapping]] = ..., environment_digest: _Optional[bytes] = ..., package_interface: _Optional[_Union[Ref, _Mapping]] = ..., bindings_digest: _Optional[bytes] = ..., models: _Optional[_Iterable[_Union[Model, _Mapping]]] = ..., entrypoints: _Optional[_Iterable[_Union[Entrypoint, _Mapping]]] = ..., environment: _Optional[_Union[Environment, _Mapping]] = ...) -> None: ...

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
    __slots__ = ("package", "release")
    PACKAGE_FIELD_NUMBER: _ClassVar[int]
    RELEASE_FIELD_NUMBER: _ClassVar[int]
    package: str
    release: str
    def __init__(self, package: _Optional[str] = ..., release: _Optional[str] = ...) -> None: ...

class DevelopmentPackage(_message.Message):
    __slots__ = ("package", "release", "source_digest", "project_wheel", "local_revision_digest")
    PACKAGE_FIELD_NUMBER: _ClassVar[int]
    RELEASE_FIELD_NUMBER: _ClassVar[int]
    SOURCE_DIGEST_FIELD_NUMBER: _ClassVar[int]
    PROJECT_WHEEL_FIELD_NUMBER: _ClassVar[int]
    LOCAL_REVISION_DIGEST_FIELD_NUMBER: _ClassVar[int]
    package: str
    release: str
    source_digest: bytes
    project_wheel: WheelFact
    local_revision_digest: bytes
    def __init__(self, package: _Optional[str] = ..., release: _Optional[str] = ..., source_digest: _Optional[bytes] = ..., project_wheel: _Optional[_Union[WheelFact, _Mapping]] = ..., local_revision_digest: _Optional[bytes] = ...) -> None: ...

class Environment(_message.Message):
    __slots__ = ("locked_requirements", "local_wheels")
    LOCKED_REQUIREMENTS_FIELD_NUMBER: _ClassVar[int]
    LOCAL_WHEELS_FIELD_NUMBER: _ClassVar[int]
    locked_requirements: Ref
    local_wheels: _containers.RepeatedCompositeFieldContainer[WheelFact]
    def __init__(self, locked_requirements: _Optional[_Union[Ref, _Mapping]] = ..., local_wheels: _Optional[_Iterable[_Union[WheelFact, _Mapping]]] = ...) -> None: ...

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
    __slots__ = ("slot", "reference_model_id", "components", "model_construction_contract", "stamps", "adapters")
    SLOT_FIELD_NUMBER: _ClassVar[int]
    REFERENCE_MODEL_ID_FIELD_NUMBER: _ClassVar[int]
    COMPONENTS_FIELD_NUMBER: _ClassVar[int]
    MODEL_CONSTRUCTION_CONTRACT_FIELD_NUMBER: _ClassVar[int]
    STAMPS_FIELD_NUMBER: _ClassVar[int]
    ADAPTERS_FIELD_NUMBER: _ClassVar[int]
    slot: str
    reference_model_id: str
    components: _containers.RepeatedCompositeFieldContainer[Component]
    model_construction_contract: Ref
    stamps: _containers.RepeatedCompositeFieldContainer[Stamp]
    adapters: _containers.RepeatedCompositeFieldContainer[ModelAdapter]
    def __init__(self, slot: _Optional[str] = ..., reference_model_id: _Optional[str] = ..., components: _Optional[_Iterable[_Union[Component, _Mapping]]] = ..., model_construction_contract: _Optional[_Union[Ref, _Mapping]] = ..., stamps: _Optional[_Iterable[_Union[Stamp, _Mapping]]] = ..., adapters: _Optional[_Iterable[_Union[ModelAdapter, _Mapping]]] = ...) -> None: ...

class ModelAdapter(_message.Message):
    __slots__ = ("component", "model_id", "source_component", "scale")
    COMPONENT_FIELD_NUMBER: _ClassVar[int]
    MODEL_ID_FIELD_NUMBER: _ClassVar[int]
    SOURCE_COMPONENT_FIELD_NUMBER: _ClassVar[int]
    SCALE_FIELD_NUMBER: _ClassVar[int]
    component: str
    model_id: str
    source_component: str
    scale: str
    def __init__(self, component: _Optional[str] = ..., model_id: _Optional[str] = ..., source_component: _Optional[str] = ..., scale: _Optional[str] = ...) -> None: ...

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
    __slots__ = ("build_id", "job_descriptor_id", "resource_caps", "publication_contract", "reclaim_on_terminal", "device_count", "orchestration", "orchestration_parent")
    BUILD_ID_FIELD_NUMBER: _ClassVar[int]
    JOB_DESCRIPTOR_ID_FIELD_NUMBER: _ClassVar[int]
    RESOURCE_CAPS_FIELD_NUMBER: _ClassVar[int]
    PUBLICATION_CONTRACT_FIELD_NUMBER: _ClassVar[int]
    RECLAIM_ON_TERMINAL_FIELD_NUMBER: _ClassVar[int]
    DEVICE_COUNT_FIELD_NUMBER: _ClassVar[int]
    ORCHESTRATION_FIELD_NUMBER: _ClassVar[int]
    ORCHESTRATION_PARENT_FIELD_NUMBER: _ClassVar[int]
    build_id: str
    job_descriptor_id: str
    resource_caps: ResourceCaps
    publication_contract: PublicationContract
    reclaim_on_terminal: bool
    device_count: int
    orchestration: bool
    orchestration_parent: JobDirective
    def __init__(self, build_id: _Optional[str] = ..., job_descriptor_id: _Optional[str] = ..., resource_caps: _Optional[_Union[ResourceCaps, _Mapping]] = ..., publication_contract: _Optional[_Union[PublicationContract, _Mapping]] = ..., reclaim_on_terminal: bool = ..., device_count: _Optional[int] = ..., orchestration: bool = ..., orchestration_parent: _Optional[_Union[JobDirective, _Mapping]] = ...) -> None: ...

class ObservedWorkerState(_message.Message):
    __slots__ = ("record_owner_epoch", "control_stream_epoch", "worker_boot_id", "applied_wire_minor", "held_attempts", "faults", "activity", "job_capacity", "placements", "admission_epoch", "admission_state", "available_attempt_slots", "accepted_desired_state_revision", "accepted_placement_set_digest", "converged_revision", "worker_phase", "lanes", "held_manifests")
    RECORD_OWNER_EPOCH_FIELD_NUMBER: _ClassVar[int]
    CONTROL_STREAM_EPOCH_FIELD_NUMBER: _ClassVar[int]
    WORKER_BOOT_ID_FIELD_NUMBER: _ClassVar[int]
    APPLIED_WIRE_MINOR_FIELD_NUMBER: _ClassVar[int]
    HELD_ATTEMPTS_FIELD_NUMBER: _ClassVar[int]
    FAULTS_FIELD_NUMBER: _ClassVar[int]
    ACTIVITY_FIELD_NUMBER: _ClassVar[int]
    JOB_CAPACITY_FIELD_NUMBER: _ClassVar[int]
    PLACEMENTS_FIELD_NUMBER: _ClassVar[int]
    ADMISSION_EPOCH_FIELD_NUMBER: _ClassVar[int]
    ADMISSION_STATE_FIELD_NUMBER: _ClassVar[int]
    AVAILABLE_ATTEMPT_SLOTS_FIELD_NUMBER: _ClassVar[int]
    ACCEPTED_DESIRED_STATE_REVISION_FIELD_NUMBER: _ClassVar[int]
    ACCEPTED_PLACEMENT_SET_DIGEST_FIELD_NUMBER: _ClassVar[int]
    CONVERGED_REVISION_FIELD_NUMBER: _ClassVar[int]
    WORKER_PHASE_FIELD_NUMBER: _ClassVar[int]
    LANES_FIELD_NUMBER: _ClassVar[int]
    HELD_MANIFESTS_FIELD_NUMBER: _ClassVar[int]
    record_owner_epoch: int
    control_stream_epoch: int
    worker_boot_id: str
    applied_wire_minor: int
    held_attempts: _containers.RepeatedCompositeFieldContainer[HeldAttempt]
    faults: _containers.RepeatedCompositeFieldContainer[Fault]
    activity: _containers.RepeatedCompositeFieldContainer[ActivityEvent]
    job_capacity: JobCapacity
    placements: _containers.RepeatedCompositeFieldContainer[PlacementStatus]
    admission_epoch: int
    admission_state: AdmissionState
    available_attempt_slots: int
    accepted_desired_state_revision: int
    accepted_placement_set_digest: bytes
    converged_revision: int
    worker_phase: WorkerPhase
    lanes: _containers.RepeatedCompositeFieldContainer[DeviceLane]
    held_manifests: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, record_owner_epoch: _Optional[int] = ..., control_stream_epoch: _Optional[int] = ..., worker_boot_id: _Optional[str] = ..., applied_wire_minor: _Optional[int] = ..., held_attempts: _Optional[_Iterable[_Union[HeldAttempt, _Mapping]]] = ..., faults: _Optional[_Iterable[_Union[Fault, _Mapping]]] = ..., activity: _Optional[_Iterable[_Union[ActivityEvent, _Mapping]]] = ..., job_capacity: _Optional[_Union[JobCapacity, _Mapping]] = ..., placements: _Optional[_Iterable[_Union[PlacementStatus, _Mapping]]] = ..., admission_epoch: _Optional[int] = ..., admission_state: _Optional[_Union[AdmissionState, str]] = ..., available_attempt_slots: _Optional[int] = ..., accepted_desired_state_revision: _Optional[int] = ..., accepted_placement_set_digest: _Optional[bytes] = ..., converged_revision: _Optional[int] = ..., worker_phase: _Optional[_Union[WorkerPhase, str]] = ..., lanes: _Optional[_Iterable[_Union[DeviceLane, _Mapping]]] = ..., held_manifests: _Optional[_Iterable[str]] = ...) -> None: ...

class DeviceLane(_message.Message):
    __slots__ = ("lane_id", "device_ordinals", "available_attempt_slots", "placement_ids", "resident_placement_ids")
    LANE_ID_FIELD_NUMBER: _ClassVar[int]
    DEVICE_ORDINALS_FIELD_NUMBER: _ClassVar[int]
    AVAILABLE_ATTEMPT_SLOTS_FIELD_NUMBER: _ClassVar[int]
    PLACEMENT_IDS_FIELD_NUMBER: _ClassVar[int]
    RESIDENT_PLACEMENT_IDS_FIELD_NUMBER: _ClassVar[int]
    lane_id: str
    device_ordinals: _containers.RepeatedScalarFieldContainer[int]
    available_attempt_slots: int
    placement_ids: _containers.RepeatedScalarFieldContainer[str]
    resident_placement_ids: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, lane_id: _Optional[str] = ..., device_ordinals: _Optional[_Iterable[int]] = ..., available_attempt_slots: _Optional[int] = ..., placement_ids: _Optional[_Iterable[str]] = ..., resident_placement_ids: _Optional[_Iterable[str]] = ...) -> None: ...

class PlacementStatus(_message.Message):
    __slots__ = ("placement_id", "executor_epoch", "dispatchable_binding_digests", "materializable_binding_digests", "faults", "accelerator", "placement_set_digest", "materialization", "serving", "retained_fallback_placement_set_digest", "acquisition", "environment_digest", "device_lane_id")
    PLACEMENT_ID_FIELD_NUMBER: _ClassVar[int]
    EXECUTOR_EPOCH_FIELD_NUMBER: _ClassVar[int]
    DISPATCHABLE_BINDING_DIGESTS_FIELD_NUMBER: _ClassVar[int]
    MATERIALIZABLE_BINDING_DIGESTS_FIELD_NUMBER: _ClassVar[int]
    FAULTS_FIELD_NUMBER: _ClassVar[int]
    ACCELERATOR_FIELD_NUMBER: _ClassVar[int]
    PLACEMENT_SET_DIGEST_FIELD_NUMBER: _ClassVar[int]
    MATERIALIZATION_FIELD_NUMBER: _ClassVar[int]
    SERVING_FIELD_NUMBER: _ClassVar[int]
    RETAINED_FALLBACK_PLACEMENT_SET_DIGEST_FIELD_NUMBER: _ClassVar[int]
    ACQUISITION_FIELD_NUMBER: _ClassVar[int]
    ENVIRONMENT_DIGEST_FIELD_NUMBER: _ClassVar[int]
    DEVICE_LANE_ID_FIELD_NUMBER: _ClassVar[int]
    placement_id: str
    executor_epoch: int
    dispatchable_binding_digests: _containers.RepeatedScalarFieldContainer[bytes]
    materializable_binding_digests: _containers.RepeatedScalarFieldContainer[bytes]
    faults: _containers.RepeatedCompositeFieldContainer[Fault]
    accelerator: AcceleratorQualification
    placement_set_digest: bytes
    materialization: MaterializationState
    serving: ServingState
    retained_fallback_placement_set_digest: bytes
    acquisition: PlacementAcquisitionObservation
    environment_digest: str
    device_lane_id: str
    def __init__(self, placement_id: _Optional[str] = ..., executor_epoch: _Optional[int] = ..., dispatchable_binding_digests: _Optional[_Iterable[bytes]] = ..., materializable_binding_digests: _Optional[_Iterable[bytes]] = ..., faults: _Optional[_Iterable[_Union[Fault, _Mapping]]] = ..., accelerator: _Optional[_Union[AcceleratorQualification, _Mapping]] = ..., placement_set_digest: _Optional[bytes] = ..., materialization: _Optional[_Union[MaterializationState, str]] = ..., serving: _Optional[_Union[ServingState, str]] = ..., retained_fallback_placement_set_digest: _Optional[bytes] = ..., acquisition: _Optional[_Union[PlacementAcquisitionObservation, _Mapping]] = ..., environment_digest: _Optional[str] = ..., device_lane_id: _Optional[str] = ...) -> None: ...

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
    __slots__ = ("record_owner_epoch", "control_stream_epoch", "worker_boot_id", "request_id", "attempt_ordinal", "invocation_spec_digest", "grant", "invocation_spec_canonical_bytes", "placement_id", "admission_epoch")
    RECORD_OWNER_EPOCH_FIELD_NUMBER: _ClassVar[int]
    CONTROL_STREAM_EPOCH_FIELD_NUMBER: _ClassVar[int]
    WORKER_BOOT_ID_FIELD_NUMBER: _ClassVar[int]
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    ATTEMPT_ORDINAL_FIELD_NUMBER: _ClassVar[int]
    INVOCATION_SPEC_DIGEST_FIELD_NUMBER: _ClassVar[int]
    GRANT_FIELD_NUMBER: _ClassVar[int]
    INVOCATION_SPEC_CANONICAL_BYTES_FIELD_NUMBER: _ClassVar[int]
    PLACEMENT_ID_FIELD_NUMBER: _ClassVar[int]
    ADMISSION_EPOCH_FIELD_NUMBER: _ClassVar[int]
    record_owner_epoch: int
    control_stream_epoch: int
    worker_boot_id: str
    request_id: str
    attempt_ordinal: int
    invocation_spec_digest: bytes
    grant: DeliveryGrant
    invocation_spec_canonical_bytes: bytes
    placement_id: str
    admission_epoch: int
    def __init__(self, record_owner_epoch: _Optional[int] = ..., control_stream_epoch: _Optional[int] = ..., worker_boot_id: _Optional[str] = ..., request_id: _Optional[str] = ..., attempt_ordinal: _Optional[int] = ..., invocation_spec_digest: _Optional[bytes] = ..., grant: _Optional[_Union[DeliveryGrant, _Mapping]] = ..., invocation_spec_canonical_bytes: _Optional[bytes] = ..., placement_id: _Optional[str] = ..., admission_epoch: _Optional[int] = ...) -> None: ...

class AttemptAccepted(_message.Message):
    __slots__ = ("record_owner_epoch", "control_stream_epoch", "worker_boot_id", "request_id", "attempt_ordinal", "invocation_spec_digest", "placement_id", "lane_id")
    RECORD_OWNER_EPOCH_FIELD_NUMBER: _ClassVar[int]
    CONTROL_STREAM_EPOCH_FIELD_NUMBER: _ClassVar[int]
    WORKER_BOOT_ID_FIELD_NUMBER: _ClassVar[int]
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    ATTEMPT_ORDINAL_FIELD_NUMBER: _ClassVar[int]
    INVOCATION_SPEC_DIGEST_FIELD_NUMBER: _ClassVar[int]
    PLACEMENT_ID_FIELD_NUMBER: _ClassVar[int]
    LANE_ID_FIELD_NUMBER: _ClassVar[int]
    record_owner_epoch: int
    control_stream_epoch: int
    worker_boot_id: str
    request_id: str
    attempt_ordinal: int
    invocation_spec_digest: bytes
    placement_id: str
    lane_id: str
    def __init__(self, record_owner_epoch: _Optional[int] = ..., control_stream_epoch: _Optional[int] = ..., worker_boot_id: _Optional[str] = ..., request_id: _Optional[str] = ..., attempt_ordinal: _Optional[int] = ..., invocation_spec_digest: _Optional[bytes] = ..., placement_id: _Optional[str] = ..., lane_id: _Optional[str] = ...) -> None: ...

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
    __slots__ = ("record_owner_epoch", "control_stream_epoch", "worker_boot_id", "request_id", "attempt_ordinal", "reason", "grace_ms", "invocation_spec_digest")
    RECORD_OWNER_EPOCH_FIELD_NUMBER: _ClassVar[int]
    CONTROL_STREAM_EPOCH_FIELD_NUMBER: _ClassVar[int]
    WORKER_BOOT_ID_FIELD_NUMBER: _ClassVar[int]
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    ATTEMPT_ORDINAL_FIELD_NUMBER: _ClassVar[int]
    REASON_FIELD_NUMBER: _ClassVar[int]
    GRACE_MS_FIELD_NUMBER: _ClassVar[int]
    INVOCATION_SPEC_DIGEST_FIELD_NUMBER: _ClassVar[int]
    record_owner_epoch: int
    control_stream_epoch: int
    worker_boot_id: str
    request_id: str
    attempt_ordinal: int
    reason: CancelReason
    grace_ms: int
    invocation_spec_digest: bytes
    def __init__(self, record_owner_epoch: _Optional[int] = ..., control_stream_epoch: _Optional[int] = ..., worker_boot_id: _Optional[str] = ..., request_id: _Optional[str] = ..., attempt_ordinal: _Optional[int] = ..., reason: _Optional[_Union[CancelReason, str]] = ..., grace_ms: _Optional[int] = ..., invocation_spec_digest: _Optional[bytes] = ...) -> None: ...

class AttemptOutcome(_message.Message):
    __slots__ = ("record_owner_epoch", "control_stream_epoch", "worker_boot_id", "request_id", "attempt_ordinal", "invocation_spec_digest", "outcome_id", "outcome_digest", "outcome_canonical_bytes", "placement_id")
    RECORD_OWNER_EPOCH_FIELD_NUMBER: _ClassVar[int]
    CONTROL_STREAM_EPOCH_FIELD_NUMBER: _ClassVar[int]
    WORKER_BOOT_ID_FIELD_NUMBER: _ClassVar[int]
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    ATTEMPT_ORDINAL_FIELD_NUMBER: _ClassVar[int]
    INVOCATION_SPEC_DIGEST_FIELD_NUMBER: _ClassVar[int]
    OUTCOME_ID_FIELD_NUMBER: _ClassVar[int]
    OUTCOME_DIGEST_FIELD_NUMBER: _ClassVar[int]
    OUTCOME_CANONICAL_BYTES_FIELD_NUMBER: _ClassVar[int]
    PLACEMENT_ID_FIELD_NUMBER: _ClassVar[int]
    record_owner_epoch: int
    control_stream_epoch: int
    worker_boot_id: str
    request_id: str
    attempt_ordinal: int
    invocation_spec_digest: bytes
    outcome_id: str
    outcome_digest: bytes
    outcome_canonical_bytes: bytes
    placement_id: str
    def __init__(self, record_owner_epoch: _Optional[int] = ..., control_stream_epoch: _Optional[int] = ..., worker_boot_id: _Optional[str] = ..., request_id: _Optional[str] = ..., attempt_ordinal: _Optional[int] = ..., invocation_spec_digest: _Optional[bytes] = ..., outcome_id: _Optional[str] = ..., outcome_digest: _Optional[bytes] = ..., outcome_canonical_bytes: _Optional[bytes] = ..., placement_id: _Optional[str] = ...) -> None: ...

class AttemptOutcomeBody(_message.Message):
    __slots__ = ("request_id", "attempt_ordinal", "invocation_spec_digest", "status", "output_manifest", "metrics", "triage_bundle", "safe_message", "cause", "result", "execution_started", "weights_receipts", "observation")
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
    WEIGHTS_RECEIPTS_FIELD_NUMBER: _ClassVar[int]
    OBSERVATION_FIELD_NUMBER: _ClassVar[int]
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
    weights_receipts: _containers.RepeatedCompositeFieldContainer[WeightsReceiptRef]
    observation: ExecutionObservation
    def __init__(self, request_id: _Optional[str] = ..., attempt_ordinal: _Optional[int] = ..., invocation_spec_digest: _Optional[str] = ..., status: _Optional[_Union[OutcomeStatus, str]] = ..., output_manifest: _Optional[_Union[OutputManifest, _Mapping]] = ..., metrics: _Optional[_Union[AttemptMetrics, _Mapping]] = ..., triage_bundle: _Optional[_Union[TriageBundleRef, _Mapping]] = ..., safe_message: _Optional[str] = ..., cause: _Optional[_Union[OutcomeCause, _Mapping]] = ..., result: _Optional[_Union[ResultEnvelope, _Mapping]] = ..., execution_started: bool = ..., weights_receipts: _Optional[_Iterable[_Union[WeightsReceiptRef, _Mapping]]] = ..., observation: _Optional[_Union[ExecutionObservation, _Mapping]] = ...) -> None: ...

class WeightsReceiptRef(_message.Message):
    __slots__ = ("weights_receipt_digest", "weights_receipt_canonical_bytes")
    WEIGHTS_RECEIPT_DIGEST_FIELD_NUMBER: _ClassVar[int]
    WEIGHTS_RECEIPT_CANONICAL_BYTES_FIELD_NUMBER: _ClassVar[int]
    weights_receipt_digest: bytes
    weights_receipt_canonical_bytes: bytes
    def __init__(self, weights_receipt_digest: _Optional[bytes] = ..., weights_receipt_canonical_bytes: _Optional[bytes] = ...) -> None: ...

class WeightsReceipt(_message.Message):
    __slots__ = ("owner_authority_scope", "request_id", "invocation_spec_digest", "output_slot", "weights_transaction_id", "tensorfs_receipt_digest", "tensorfs_receipt_canonical_bytes")
    OWNER_AUTHORITY_SCOPE_FIELD_NUMBER: _ClassVar[int]
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    INVOCATION_SPEC_DIGEST_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_SLOT_FIELD_NUMBER: _ClassVar[int]
    WEIGHTS_TRANSACTION_ID_FIELD_NUMBER: _ClassVar[int]
    TENSORFS_RECEIPT_DIGEST_FIELD_NUMBER: _ClassVar[int]
    TENSORFS_RECEIPT_CANONICAL_BYTES_FIELD_NUMBER: _ClassVar[int]
    owner_authority_scope: str
    request_id: str
    invocation_spec_digest: str
    output_slot: str
    weights_transaction_id: str
    tensorfs_receipt_digest: str
    tensorfs_receipt_canonical_bytes: bytes
    def __init__(self, owner_authority_scope: _Optional[str] = ..., request_id: _Optional[str] = ..., invocation_spec_digest: _Optional[str] = ..., output_slot: _Optional[str] = ..., weights_transaction_id: _Optional[str] = ..., tensorfs_receipt_digest: _Optional[str] = ..., tensorfs_receipt_canonical_bytes: _Optional[bytes] = ...) -> None: ...

class WeightsObjectSource(_message.Message):
    __slots__ = ("object_id", "length", "source_ref")
    OBJECT_ID_FIELD_NUMBER: _ClassVar[int]
    LENGTH_FIELD_NUMBER: _ClassVar[int]
    SOURCE_REF_FIELD_NUMBER: _ClassVar[int]
    object_id: str
    length: int
    source_ref: str
    def __init__(self, object_id: _Optional[str] = ..., length: _Optional[int] = ..., source_ref: _Optional[str] = ...) -> None: ...

class WeightsIntentFrame(_message.Message):
    __slots__ = ("record_owner_epoch", "control_stream_epoch", "worker_boot_id", "request_id", "attempt_ordinal", "invocation_spec_digest", "output_slot", "tensorfs_declaration_digest", "requested_writer_epoch", "tensorfs_declaration_canonical_bytes", "weights_transaction_id")
    RECORD_OWNER_EPOCH_FIELD_NUMBER: _ClassVar[int]
    CONTROL_STREAM_EPOCH_FIELD_NUMBER: _ClassVar[int]
    WORKER_BOOT_ID_FIELD_NUMBER: _ClassVar[int]
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    ATTEMPT_ORDINAL_FIELD_NUMBER: _ClassVar[int]
    INVOCATION_SPEC_DIGEST_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_SLOT_FIELD_NUMBER: _ClassVar[int]
    TENSORFS_DECLARATION_DIGEST_FIELD_NUMBER: _ClassVar[int]
    REQUESTED_WRITER_EPOCH_FIELD_NUMBER: _ClassVar[int]
    TENSORFS_DECLARATION_CANONICAL_BYTES_FIELD_NUMBER: _ClassVar[int]
    WEIGHTS_TRANSACTION_ID_FIELD_NUMBER: _ClassVar[int]
    record_owner_epoch: int
    control_stream_epoch: int
    worker_boot_id: str
    request_id: str
    attempt_ordinal: int
    invocation_spec_digest: bytes
    output_slot: str
    tensorfs_declaration_digest: bytes
    requested_writer_epoch: int
    tensorfs_declaration_canonical_bytes: bytes
    weights_transaction_id: str
    def __init__(self, record_owner_epoch: _Optional[int] = ..., control_stream_epoch: _Optional[int] = ..., worker_boot_id: _Optional[str] = ..., request_id: _Optional[str] = ..., attempt_ordinal: _Optional[int] = ..., invocation_spec_digest: _Optional[bytes] = ..., output_slot: _Optional[str] = ..., tensorfs_declaration_digest: _Optional[bytes] = ..., requested_writer_epoch: _Optional[int] = ..., tensorfs_declaration_canonical_bytes: _Optional[bytes] = ..., weights_transaction_id: _Optional[str] = ...) -> None: ...

class WeightsHostAck(_message.Message):
    __slots__ = ("record_owner_epoch", "control_stream_epoch", "worker_boot_id", "request_id", "attempt_ordinal", "invocation_spec_digest", "output_slot", "weights_transaction_id", "writer_epoch", "tensorfs_declaration_digest", "stage", "outcome", "refusal", "weights_receipt", "manifest", "checkpoint")
    RECORD_OWNER_EPOCH_FIELD_NUMBER: _ClassVar[int]
    CONTROL_STREAM_EPOCH_FIELD_NUMBER: _ClassVar[int]
    WORKER_BOOT_ID_FIELD_NUMBER: _ClassVar[int]
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    ATTEMPT_ORDINAL_FIELD_NUMBER: _ClassVar[int]
    INVOCATION_SPEC_DIGEST_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_SLOT_FIELD_NUMBER: _ClassVar[int]
    WEIGHTS_TRANSACTION_ID_FIELD_NUMBER: _ClassVar[int]
    WRITER_EPOCH_FIELD_NUMBER: _ClassVar[int]
    TENSORFS_DECLARATION_DIGEST_FIELD_NUMBER: _ClassVar[int]
    STAGE_FIELD_NUMBER: _ClassVar[int]
    OUTCOME_FIELD_NUMBER: _ClassVar[int]
    REFUSAL_FIELD_NUMBER: _ClassVar[int]
    WEIGHTS_RECEIPT_FIELD_NUMBER: _ClassVar[int]
    MANIFEST_FIELD_NUMBER: _ClassVar[int]
    CHECKPOINT_FIELD_NUMBER: _ClassVar[int]
    record_owner_epoch: int
    control_stream_epoch: int
    worker_boot_id: str
    request_id: str
    attempt_ordinal: int
    invocation_spec_digest: bytes
    output_slot: str
    weights_transaction_id: str
    writer_epoch: int
    tensorfs_declaration_digest: bytes
    stage: WeightsHostStage
    outcome: WeightsHostOutcome
    refusal: WeightsHostRefusal
    weights_receipt: WeightsReceiptRef
    manifest: Ref
    checkpoint: CheckpointRef
    def __init__(self, record_owner_epoch: _Optional[int] = ..., control_stream_epoch: _Optional[int] = ..., worker_boot_id: _Optional[str] = ..., request_id: _Optional[str] = ..., attempt_ordinal: _Optional[int] = ..., invocation_spec_digest: _Optional[bytes] = ..., output_slot: _Optional[str] = ..., weights_transaction_id: _Optional[str] = ..., writer_epoch: _Optional[int] = ..., tensorfs_declaration_digest: _Optional[bytes] = ..., stage: _Optional[_Union[WeightsHostStage, str]] = ..., outcome: _Optional[_Union[WeightsHostOutcome, str]] = ..., refusal: _Optional[_Union[WeightsHostRefusal, str]] = ..., weights_receipt: _Optional[_Union[WeightsReceiptRef, _Mapping]] = ..., manifest: _Optional[_Union[Ref, _Mapping]] = ..., checkpoint: _Optional[_Union[CheckpointRef, _Mapping]] = ...) -> None: ...

class CheckpointRef(_message.Message):
    __slots__ = ("head", "plan_digest", "index", "bytes")
    HEAD_FIELD_NUMBER: _ClassVar[int]
    PLAN_DIGEST_FIELD_NUMBER: _ClassVar[int]
    INDEX_FIELD_NUMBER: _ClassVar[int]
    BYTES_FIELD_NUMBER: _ClassVar[int]
    head: Ref
    plan_digest: bytes
    index: int
    bytes: int
    def __init__(self, head: _Optional[_Union[Ref, _Mapping]] = ..., plan_digest: _Optional[bytes] = ..., index: _Optional[int] = ..., bytes: _Optional[int] = ...) -> None: ...

class WeightsCheckpointFrame(_message.Message):
    __slots__ = ("record_owner_epoch", "control_stream_epoch", "worker_boot_id", "request_id", "attempt_ordinal", "invocation_spec_digest", "output_slot", "weights_transaction_id", "writer_epoch", "tensorfs_declaration_digest", "checkpoint")
    RECORD_OWNER_EPOCH_FIELD_NUMBER: _ClassVar[int]
    CONTROL_STREAM_EPOCH_FIELD_NUMBER: _ClassVar[int]
    WORKER_BOOT_ID_FIELD_NUMBER: _ClassVar[int]
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    ATTEMPT_ORDINAL_FIELD_NUMBER: _ClassVar[int]
    INVOCATION_SPEC_DIGEST_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_SLOT_FIELD_NUMBER: _ClassVar[int]
    WEIGHTS_TRANSACTION_ID_FIELD_NUMBER: _ClassVar[int]
    WRITER_EPOCH_FIELD_NUMBER: _ClassVar[int]
    TENSORFS_DECLARATION_DIGEST_FIELD_NUMBER: _ClassVar[int]
    CHECKPOINT_FIELD_NUMBER: _ClassVar[int]
    record_owner_epoch: int
    control_stream_epoch: int
    worker_boot_id: str
    request_id: str
    attempt_ordinal: int
    invocation_spec_digest: bytes
    output_slot: str
    weights_transaction_id: str
    writer_epoch: int
    tensorfs_declaration_digest: bytes
    checkpoint: CheckpointRef
    def __init__(self, record_owner_epoch: _Optional[int] = ..., control_stream_epoch: _Optional[int] = ..., worker_boot_id: _Optional[str] = ..., request_id: _Optional[str] = ..., attempt_ordinal: _Optional[int] = ..., invocation_spec_digest: _Optional[bytes] = ..., output_slot: _Optional[str] = ..., weights_transaction_id: _Optional[str] = ..., writer_epoch: _Optional[int] = ..., tensorfs_declaration_digest: _Optional[bytes] = ..., checkpoint: _Optional[_Union[CheckpointRef, _Mapping]] = ...) -> None: ...

class WeightsIntentReadyRequest(_message.Message):
    __slots__ = ("record_owner_epoch", "control_stream_epoch", "worker_boot_id", "weights", "attempt_ordinal", "checkpoint")
    RECORD_OWNER_EPOCH_FIELD_NUMBER: _ClassVar[int]
    CONTROL_STREAM_EPOCH_FIELD_NUMBER: _ClassVar[int]
    WORKER_BOOT_ID_FIELD_NUMBER: _ClassVar[int]
    WEIGHTS_FIELD_NUMBER: _ClassVar[int]
    ATTEMPT_ORDINAL_FIELD_NUMBER: _ClassVar[int]
    CHECKPOINT_FIELD_NUMBER: _ClassVar[int]
    record_owner_epoch: int
    control_stream_epoch: int
    worker_boot_id: str
    weights: WeightsCheckpointSubject
    attempt_ordinal: int
    checkpoint: CheckpointRef
    def __init__(self, record_owner_epoch: _Optional[int] = ..., control_stream_epoch: _Optional[int] = ..., worker_boot_id: _Optional[str] = ..., weights: _Optional[_Union[WeightsCheckpointSubject, _Mapping]] = ..., attempt_ordinal: _Optional[int] = ..., checkpoint: _Optional[_Union[CheckpointRef, _Mapping]] = ...) -> None: ...

class WeightsIntentReadyCall(_message.Message):
    __slots__ = ("claim", "request")
    CLAIM_FIELD_NUMBER: _ClassVar[int]
    REQUEST_FIELD_NUMBER: _ClassVar[int]
    claim: Claim
    request: WeightsIntentReadyRequest
    def __init__(self, claim: _Optional[_Union[Claim, _Mapping]] = ..., request: _Optional[_Union[WeightsIntentReadyRequest, _Mapping]] = ...) -> None: ...

class ValidateWeightsCheckpointRequest(_message.Message):
    __slots__ = ("intent", "tensorfs_declaration_canonical_bytes")
    INTENT_FIELD_NUMBER: _ClassVar[int]
    TENSORFS_DECLARATION_CANONICAL_BYTES_FIELD_NUMBER: _ClassVar[int]
    intent: WeightsIntentReadyRequest
    tensorfs_declaration_canonical_bytes: bytes
    def __init__(self, intent: _Optional[_Union[WeightsIntentReadyRequest, _Mapping]] = ..., tensorfs_declaration_canonical_bytes: _Optional[bytes] = ...) -> None: ...

class ValidateWeightsCheckpointResult(_message.Message):
    __slots__ = ("weights", "checkpoint", "valid", "safe_code", "safe_detail")
    WEIGHTS_FIELD_NUMBER: _ClassVar[int]
    CHECKPOINT_FIELD_NUMBER: _ClassVar[int]
    VALID_FIELD_NUMBER: _ClassVar[int]
    SAFE_CODE_FIELD_NUMBER: _ClassVar[int]
    SAFE_DETAIL_FIELD_NUMBER: _ClassVar[int]
    weights: WeightsCheckpointSubject
    checkpoint: CheckpointRef
    valid: bool
    safe_code: str
    safe_detail: str
    def __init__(self, weights: _Optional[_Union[WeightsCheckpointSubject, _Mapping]] = ..., checkpoint: _Optional[_Union[CheckpointRef, _Mapping]] = ..., valid: bool = ..., safe_code: _Optional[str] = ..., safe_detail: _Optional[str] = ...) -> None: ...

class WeightsReceiptFrame(_message.Message):
    __slots__ = ("record_owner_epoch", "control_stream_epoch", "worker_boot_id", "request_id", "attempt_ordinal", "invocation_spec_digest", "output_slot", "weights_transaction_id", "writer_epoch", "tensorfs_declaration_digest", "weights_receipt", "objects", "manifest")
    RECORD_OWNER_EPOCH_FIELD_NUMBER: _ClassVar[int]
    CONTROL_STREAM_EPOCH_FIELD_NUMBER: _ClassVar[int]
    WORKER_BOOT_ID_FIELD_NUMBER: _ClassVar[int]
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    ATTEMPT_ORDINAL_FIELD_NUMBER: _ClassVar[int]
    INVOCATION_SPEC_DIGEST_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_SLOT_FIELD_NUMBER: _ClassVar[int]
    WEIGHTS_TRANSACTION_ID_FIELD_NUMBER: _ClassVar[int]
    WRITER_EPOCH_FIELD_NUMBER: _ClassVar[int]
    TENSORFS_DECLARATION_DIGEST_FIELD_NUMBER: _ClassVar[int]
    WEIGHTS_RECEIPT_FIELD_NUMBER: _ClassVar[int]
    OBJECTS_FIELD_NUMBER: _ClassVar[int]
    MANIFEST_FIELD_NUMBER: _ClassVar[int]
    record_owner_epoch: int
    control_stream_epoch: int
    worker_boot_id: str
    request_id: str
    attempt_ordinal: int
    invocation_spec_digest: bytes
    output_slot: str
    weights_transaction_id: str
    writer_epoch: int
    tensorfs_declaration_digest: bytes
    weights_receipt: WeightsReceiptRef
    objects: _containers.RepeatedCompositeFieldContainer[WeightsObjectSource]
    manifest: Ref
    def __init__(self, record_owner_epoch: _Optional[int] = ..., control_stream_epoch: _Optional[int] = ..., worker_boot_id: _Optional[str] = ..., request_id: _Optional[str] = ..., attempt_ordinal: _Optional[int] = ..., invocation_spec_digest: _Optional[bytes] = ..., output_slot: _Optional[str] = ..., weights_transaction_id: _Optional[str] = ..., writer_epoch: _Optional[int] = ..., tensorfs_declaration_digest: _Optional[bytes] = ..., weights_receipt: _Optional[_Union[WeightsReceiptRef, _Mapping]] = ..., objects: _Optional[_Iterable[_Union[WeightsObjectSource, _Mapping]]] = ..., manifest: _Optional[_Union[Ref, _Mapping]] = ...) -> None: ...

class WeightsTransactionStatus(_message.Message):
    __slots__ = ("weights_transaction_id", "request_id", "attempt_ordinal", "invocation_spec_digest", "output_slot", "writer_epoch", "state", "tensorfs_declaration_digest", "weights_receipt_digest", "checkpoint", "intent_ready")
    WEIGHTS_TRANSACTION_ID_FIELD_NUMBER: _ClassVar[int]
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    ATTEMPT_ORDINAL_FIELD_NUMBER: _ClassVar[int]
    INVOCATION_SPEC_DIGEST_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_SLOT_FIELD_NUMBER: _ClassVar[int]
    WRITER_EPOCH_FIELD_NUMBER: _ClassVar[int]
    STATE_FIELD_NUMBER: _ClassVar[int]
    TENSORFS_DECLARATION_DIGEST_FIELD_NUMBER: _ClassVar[int]
    WEIGHTS_RECEIPT_DIGEST_FIELD_NUMBER: _ClassVar[int]
    CHECKPOINT_FIELD_NUMBER: _ClassVar[int]
    INTENT_READY_FIELD_NUMBER: _ClassVar[int]
    weights_transaction_id: str
    request_id: str
    attempt_ordinal: int
    invocation_spec_digest: str
    output_slot: str
    writer_epoch: int
    state: WeightsTransactionState
    tensorfs_declaration_digest: bytes
    weights_receipt_digest: bytes
    checkpoint: CheckpointRef
    intent_ready: bool
    def __init__(self, weights_transaction_id: _Optional[str] = ..., request_id: _Optional[str] = ..., attempt_ordinal: _Optional[int] = ..., invocation_spec_digest: _Optional[str] = ..., output_slot: _Optional[str] = ..., writer_epoch: _Optional[int] = ..., state: _Optional[_Union[WeightsTransactionState, str]] = ..., tensorfs_declaration_digest: _Optional[bytes] = ..., weights_receipt_digest: _Optional[bytes] = ..., checkpoint: _Optional[_Union[CheckpointRef, _Mapping]] = ..., intent_ready: bool = ...) -> None: ...

class WeightsReadRequest(_message.Message):
    __slots__ = ("record_owner_epoch", "control_stream_epoch", "worker_boot_id", "weights_transaction_id", "writer_epoch", "object_id", "source_ref", "read_id", "offset", "max_bytes")
    RECORD_OWNER_EPOCH_FIELD_NUMBER: _ClassVar[int]
    CONTROL_STREAM_EPOCH_FIELD_NUMBER: _ClassVar[int]
    WORKER_BOOT_ID_FIELD_NUMBER: _ClassVar[int]
    WEIGHTS_TRANSACTION_ID_FIELD_NUMBER: _ClassVar[int]
    WRITER_EPOCH_FIELD_NUMBER: _ClassVar[int]
    OBJECT_ID_FIELD_NUMBER: _ClassVar[int]
    SOURCE_REF_FIELD_NUMBER: _ClassVar[int]
    READ_ID_FIELD_NUMBER: _ClassVar[int]
    OFFSET_FIELD_NUMBER: _ClassVar[int]
    MAX_BYTES_FIELD_NUMBER: _ClassVar[int]
    record_owner_epoch: int
    control_stream_epoch: int
    worker_boot_id: str
    weights_transaction_id: str
    writer_epoch: int
    object_id: str
    source_ref: str
    read_id: str
    offset: int
    max_bytes: int
    def __init__(self, record_owner_epoch: _Optional[int] = ..., control_stream_epoch: _Optional[int] = ..., worker_boot_id: _Optional[str] = ..., weights_transaction_id: _Optional[str] = ..., writer_epoch: _Optional[int] = ..., object_id: _Optional[str] = ..., source_ref: _Optional[str] = ..., read_id: _Optional[str] = ..., offset: _Optional[int] = ..., max_bytes: _Optional[int] = ...) -> None: ...

class WeightsReadResult(_message.Message):
    __slots__ = ("record_owner_epoch", "control_stream_epoch", "worker_boot_id", "weights_transaction_id", "writer_epoch", "object_id", "source_ref", "read_id", "offset", "outcome", "data", "data_digest", "refusal", "safe_detail")
    RECORD_OWNER_EPOCH_FIELD_NUMBER: _ClassVar[int]
    CONTROL_STREAM_EPOCH_FIELD_NUMBER: _ClassVar[int]
    WORKER_BOOT_ID_FIELD_NUMBER: _ClassVar[int]
    WEIGHTS_TRANSACTION_ID_FIELD_NUMBER: _ClassVar[int]
    WRITER_EPOCH_FIELD_NUMBER: _ClassVar[int]
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
    control_stream_epoch: int
    worker_boot_id: str
    weights_transaction_id: str
    writer_epoch: int
    object_id: str
    source_ref: str
    read_id: str
    offset: int
    outcome: WeightsReadOutcome
    data: bytes
    data_digest: bytes
    refusal: WeightsReadRefusal
    safe_detail: str
    def __init__(self, record_owner_epoch: _Optional[int] = ..., control_stream_epoch: _Optional[int] = ..., worker_boot_id: _Optional[str] = ..., weights_transaction_id: _Optional[str] = ..., writer_epoch: _Optional[int] = ..., object_id: _Optional[str] = ..., source_ref: _Optional[str] = ..., read_id: _Optional[str] = ..., offset: _Optional[int] = ..., outcome: _Optional[_Union[WeightsReadOutcome, str]] = ..., data: _Optional[bytes] = ..., data_digest: _Optional[bytes] = ..., refusal: _Optional[_Union[WeightsReadRefusal, str]] = ..., safe_detail: _Optional[str] = ...) -> None: ...

class WeightsObjectRef(_message.Message):
    __slots__ = ("object_id", "length")
    OBJECT_ID_FIELD_NUMBER: _ClassVar[int]
    LENGTH_FIELD_NUMBER: _ClassVar[int]
    object_id: str
    length: int
    def __init__(self, object_id: _Optional[str] = ..., length: _Optional[int] = ...) -> None: ...

class WeightsUploadHeader(_message.Message):
    __slots__ = ("name", "value")
    NAME_FIELD_NUMBER: _ClassVar[int]
    VALUE_FIELD_NUMBER: _ClassVar[int]
    name: str
    value: str
    def __init__(self, name: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...

class WeightsUploadGrant(_message.Message):
    __slots__ = ("object_id", "length", "url", "required_headers", "expires_at_unix")
    OBJECT_ID_FIELD_NUMBER: _ClassVar[int]
    LENGTH_FIELD_NUMBER: _ClassVar[int]
    URL_FIELD_NUMBER: _ClassVar[int]
    REQUIRED_HEADERS_FIELD_NUMBER: _ClassVar[int]
    EXPIRES_AT_UNIX_FIELD_NUMBER: _ClassVar[int]
    object_id: str
    length: int
    url: str
    required_headers: _containers.RepeatedCompositeFieldContainer[WeightsUploadHeader]
    expires_at_unix: int
    def __init__(self, object_id: _Optional[str] = ..., length: _Optional[int] = ..., url: _Optional[str] = ..., required_headers: _Optional[_Iterable[_Union[WeightsUploadHeader, _Mapping]]] = ..., expires_at_unix: _Optional[int] = ...) -> None: ...

class WeightsTransferRequest(_message.Message):
    __slots__ = ("record_owner_epoch", "control_stream_epoch", "worker_boot_id", "request_id", "attempt_ordinal", "invocation_spec_digest", "output_slot", "weights_transaction_id", "weights_receipt_digest", "operation_id", "grant_revision", "upload_grant", "held")
    RECORD_OWNER_EPOCH_FIELD_NUMBER: _ClassVar[int]
    CONTROL_STREAM_EPOCH_FIELD_NUMBER: _ClassVar[int]
    WORKER_BOOT_ID_FIELD_NUMBER: _ClassVar[int]
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    ATTEMPT_ORDINAL_FIELD_NUMBER: _ClassVar[int]
    INVOCATION_SPEC_DIGEST_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_SLOT_FIELD_NUMBER: _ClassVar[int]
    WEIGHTS_TRANSACTION_ID_FIELD_NUMBER: _ClassVar[int]
    WEIGHTS_RECEIPT_DIGEST_FIELD_NUMBER: _ClassVar[int]
    OPERATION_ID_FIELD_NUMBER: _ClassVar[int]
    GRANT_REVISION_FIELD_NUMBER: _ClassVar[int]
    UPLOAD_GRANT_FIELD_NUMBER: _ClassVar[int]
    HELD_FIELD_NUMBER: _ClassVar[int]
    record_owner_epoch: int
    control_stream_epoch: int
    worker_boot_id: str
    request_id: str
    attempt_ordinal: int
    invocation_spec_digest: bytes
    output_slot: str
    weights_transaction_id: str
    weights_receipt_digest: bytes
    operation_id: str
    grant_revision: int
    upload_grant: WeightsUploadGrant
    held: WeightsObjectRef
    def __init__(self, record_owner_epoch: _Optional[int] = ..., control_stream_epoch: _Optional[int] = ..., worker_boot_id: _Optional[str] = ..., request_id: _Optional[str] = ..., attempt_ordinal: _Optional[int] = ..., invocation_spec_digest: _Optional[bytes] = ..., output_slot: _Optional[str] = ..., weights_transaction_id: _Optional[str] = ..., weights_receipt_digest: _Optional[bytes] = ..., operation_id: _Optional[str] = ..., grant_revision: _Optional[int] = ..., upload_grant: _Optional[_Union[WeightsUploadGrant, _Mapping]] = ..., held: _Optional[_Union[WeightsObjectRef, _Mapping]] = ...) -> None: ...

class WeightsUploadRequest(_message.Message):
    __slots__ = ("record_owner_epoch", "control_stream_epoch", "worker_boot_id", "weights_transaction_id", "writer_epoch", "object_id", "source_ref", "length", "operation_id", "grant_revision", "grant")
    RECORD_OWNER_EPOCH_FIELD_NUMBER: _ClassVar[int]
    CONTROL_STREAM_EPOCH_FIELD_NUMBER: _ClassVar[int]
    WORKER_BOOT_ID_FIELD_NUMBER: _ClassVar[int]
    WEIGHTS_TRANSACTION_ID_FIELD_NUMBER: _ClassVar[int]
    WRITER_EPOCH_FIELD_NUMBER: _ClassVar[int]
    OBJECT_ID_FIELD_NUMBER: _ClassVar[int]
    SOURCE_REF_FIELD_NUMBER: _ClassVar[int]
    LENGTH_FIELD_NUMBER: _ClassVar[int]
    OPERATION_ID_FIELD_NUMBER: _ClassVar[int]
    GRANT_REVISION_FIELD_NUMBER: _ClassVar[int]
    GRANT_FIELD_NUMBER: _ClassVar[int]
    record_owner_epoch: int
    control_stream_epoch: int
    worker_boot_id: str
    weights_transaction_id: str
    writer_epoch: int
    object_id: str
    source_ref: str
    length: int
    operation_id: str
    grant_revision: int
    grant: WeightsUploadGrant
    def __init__(self, record_owner_epoch: _Optional[int] = ..., control_stream_epoch: _Optional[int] = ..., worker_boot_id: _Optional[str] = ..., weights_transaction_id: _Optional[str] = ..., writer_epoch: _Optional[int] = ..., object_id: _Optional[str] = ..., source_ref: _Optional[str] = ..., length: _Optional[int] = ..., operation_id: _Optional[str] = ..., grant_revision: _Optional[int] = ..., grant: _Optional[_Union[WeightsUploadGrant, _Mapping]] = ...) -> None: ...

class WeightsUploadResult(_message.Message):
    __slots__ = ("record_owner_epoch", "control_stream_epoch", "worker_boot_id", "weights_transaction_id", "writer_epoch", "object_id", "operation_id", "grant_revision", "outcome", "transferred_bytes", "http_status", "etag", "checksum_sha256", "refusal", "safe_detail")
    RECORD_OWNER_EPOCH_FIELD_NUMBER: _ClassVar[int]
    CONTROL_STREAM_EPOCH_FIELD_NUMBER: _ClassVar[int]
    WORKER_BOOT_ID_FIELD_NUMBER: _ClassVar[int]
    WEIGHTS_TRANSACTION_ID_FIELD_NUMBER: _ClassVar[int]
    WRITER_EPOCH_FIELD_NUMBER: _ClassVar[int]
    OBJECT_ID_FIELD_NUMBER: _ClassVar[int]
    OPERATION_ID_FIELD_NUMBER: _ClassVar[int]
    GRANT_REVISION_FIELD_NUMBER: _ClassVar[int]
    OUTCOME_FIELD_NUMBER: _ClassVar[int]
    TRANSFERRED_BYTES_FIELD_NUMBER: _ClassVar[int]
    HTTP_STATUS_FIELD_NUMBER: _ClassVar[int]
    ETAG_FIELD_NUMBER: _ClassVar[int]
    CHECKSUM_SHA256_FIELD_NUMBER: _ClassVar[int]
    REFUSAL_FIELD_NUMBER: _ClassVar[int]
    SAFE_DETAIL_FIELD_NUMBER: _ClassVar[int]
    record_owner_epoch: int
    control_stream_epoch: int
    worker_boot_id: str
    weights_transaction_id: str
    writer_epoch: int
    object_id: str
    operation_id: str
    grant_revision: int
    outcome: WeightsUploadOutcome
    transferred_bytes: int
    http_status: int
    etag: str
    checksum_sha256: str
    refusal: WeightsUploadRefusal
    safe_detail: str
    def __init__(self, record_owner_epoch: _Optional[int] = ..., control_stream_epoch: _Optional[int] = ..., worker_boot_id: _Optional[str] = ..., weights_transaction_id: _Optional[str] = ..., writer_epoch: _Optional[int] = ..., object_id: _Optional[str] = ..., operation_id: _Optional[str] = ..., grant_revision: _Optional[int] = ..., outcome: _Optional[_Union[WeightsUploadOutcome, str]] = ..., transferred_bytes: _Optional[int] = ..., http_status: _Optional[int] = ..., etag: _Optional[str] = ..., checksum_sha256: _Optional[str] = ..., refusal: _Optional[_Union[WeightsUploadRefusal, str]] = ..., safe_detail: _Optional[str] = ...) -> None: ...

class WeightsTransferStatus(_message.Message):
    __slots__ = ("record_owner_epoch", "control_stream_epoch", "worker_boot_id", "request_id", "attempt_ordinal", "invocation_spec_digest", "output_slot", "weights_transaction_id", "operation_id", "grant_revision", "object_id", "length", "update_sequence", "state", "transferred_bytes", "http_status", "etag", "checksum_sha256", "attempts", "safe_code", "safe_detail")
    RECORD_OWNER_EPOCH_FIELD_NUMBER: _ClassVar[int]
    CONTROL_STREAM_EPOCH_FIELD_NUMBER: _ClassVar[int]
    WORKER_BOOT_ID_FIELD_NUMBER: _ClassVar[int]
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    ATTEMPT_ORDINAL_FIELD_NUMBER: _ClassVar[int]
    INVOCATION_SPEC_DIGEST_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_SLOT_FIELD_NUMBER: _ClassVar[int]
    WEIGHTS_TRANSACTION_ID_FIELD_NUMBER: _ClassVar[int]
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
    control_stream_epoch: int
    worker_boot_id: str
    request_id: str
    attempt_ordinal: int
    invocation_spec_digest: bytes
    output_slot: str
    weights_transaction_id: str
    operation_id: str
    grant_revision: int
    object_id: str
    length: int
    update_sequence: int
    state: WeightsTransferState
    transferred_bytes: int
    http_status: int
    etag: str
    checksum_sha256: str
    attempts: int
    safe_code: str
    safe_detail: str
    def __init__(self, record_owner_epoch: _Optional[int] = ..., control_stream_epoch: _Optional[int] = ..., worker_boot_id: _Optional[str] = ..., request_id: _Optional[str] = ..., attempt_ordinal: _Optional[int] = ..., invocation_spec_digest: _Optional[bytes] = ..., output_slot: _Optional[str] = ..., weights_transaction_id: _Optional[str] = ..., operation_id: _Optional[str] = ..., grant_revision: _Optional[int] = ..., object_id: _Optional[str] = ..., length: _Optional[int] = ..., update_sequence: _Optional[int] = ..., state: _Optional[_Union[WeightsTransferState, str]] = ..., transferred_bytes: _Optional[int] = ..., http_status: _Optional[int] = ..., etag: _Optional[str] = ..., checksum_sha256: _Optional[str] = ..., attempts: _Optional[int] = ..., safe_code: _Optional[str] = ..., safe_detail: _Optional[str] = ...) -> None: ...

class ModelSourceFileRequest(_message.Message):
    __slots__ = ("record_owner_epoch", "control_stream_epoch", "worker_boot_id", "operation_id", "source_selection_digest", "member", "object_id", "length", "provider", "url", "expires_at_unix", "capability_revision", "header")
    RECORD_OWNER_EPOCH_FIELD_NUMBER: _ClassVar[int]
    CONTROL_STREAM_EPOCH_FIELD_NUMBER: _ClassVar[int]
    WORKER_BOOT_ID_FIELD_NUMBER: _ClassVar[int]
    OPERATION_ID_FIELD_NUMBER: _ClassVar[int]
    SOURCE_SELECTION_DIGEST_FIELD_NUMBER: _ClassVar[int]
    MEMBER_FIELD_NUMBER: _ClassVar[int]
    OBJECT_ID_FIELD_NUMBER: _ClassVar[int]
    LENGTH_FIELD_NUMBER: _ClassVar[int]
    PROVIDER_FIELD_NUMBER: _ClassVar[int]
    URL_FIELD_NUMBER: _ClassVar[int]
    EXPIRES_AT_UNIX_FIELD_NUMBER: _ClassVar[int]
    CAPABILITY_REVISION_FIELD_NUMBER: _ClassVar[int]
    HEADER_FIELD_NUMBER: _ClassVar[int]
    record_owner_epoch: int
    control_stream_epoch: int
    worker_boot_id: str
    operation_id: str
    source_selection_digest: bytes
    member: str
    object_id: str
    length: int
    provider: ModelSourceProvider
    url: str
    expires_at_unix: int
    capability_revision: int
    header: bytes
    def __init__(self, record_owner_epoch: _Optional[int] = ..., control_stream_epoch: _Optional[int] = ..., worker_boot_id: _Optional[str] = ..., operation_id: _Optional[str] = ..., source_selection_digest: _Optional[bytes] = ..., member: _Optional[str] = ..., object_id: _Optional[str] = ..., length: _Optional[int] = ..., provider: _Optional[_Union[ModelSourceProvider, str]] = ..., url: _Optional[str] = ..., expires_at_unix: _Optional[int] = ..., capability_revision: _Optional[int] = ..., header: _Optional[bytes] = ...) -> None: ...

class ModelSourceFileStatus(_message.Message):
    __slots__ = ("record_owner_epoch", "control_stream_epoch", "worker_boot_id", "operation_id", "source_selection_digest", "member", "object_id", "length", "capability_revision", "state", "transferred_bytes", "attempts", "safe_code", "safe_detail")
    RECORD_OWNER_EPOCH_FIELD_NUMBER: _ClassVar[int]
    CONTROL_STREAM_EPOCH_FIELD_NUMBER: _ClassVar[int]
    WORKER_BOOT_ID_FIELD_NUMBER: _ClassVar[int]
    OPERATION_ID_FIELD_NUMBER: _ClassVar[int]
    SOURCE_SELECTION_DIGEST_FIELD_NUMBER: _ClassVar[int]
    MEMBER_FIELD_NUMBER: _ClassVar[int]
    OBJECT_ID_FIELD_NUMBER: _ClassVar[int]
    LENGTH_FIELD_NUMBER: _ClassVar[int]
    CAPABILITY_REVISION_FIELD_NUMBER: _ClassVar[int]
    STATE_FIELD_NUMBER: _ClassVar[int]
    TRANSFERRED_BYTES_FIELD_NUMBER: _ClassVar[int]
    ATTEMPTS_FIELD_NUMBER: _ClassVar[int]
    SAFE_CODE_FIELD_NUMBER: _ClassVar[int]
    SAFE_DETAIL_FIELD_NUMBER: _ClassVar[int]
    record_owner_epoch: int
    control_stream_epoch: int
    worker_boot_id: str
    operation_id: str
    source_selection_digest: bytes
    member: str
    object_id: str
    length: int
    capability_revision: int
    state: ModelSourceFileState
    transferred_bytes: int
    attempts: int
    safe_code: str
    safe_detail: str
    def __init__(self, record_owner_epoch: _Optional[int] = ..., control_stream_epoch: _Optional[int] = ..., worker_boot_id: _Optional[str] = ..., operation_id: _Optional[str] = ..., source_selection_digest: _Optional[bytes] = ..., member: _Optional[str] = ..., object_id: _Optional[str] = ..., length: _Optional[int] = ..., capability_revision: _Optional[int] = ..., state: _Optional[_Union[ModelSourceFileState, str]] = ..., transferred_bytes: _Optional[int] = ..., attempts: _Optional[int] = ..., safe_code: _Optional[str] = ..., safe_detail: _Optional[str] = ...) -> None: ...

class ModelSourcePrepareRequest(_message.Message):
    __slots__ = ("record_owner_epoch", "control_stream_epoch", "worker_boot_id", "operation_id", "source_selection_digest", "profiles", "source_uri", "declared_license", "checkpoints")
    RECORD_OWNER_EPOCH_FIELD_NUMBER: _ClassVar[int]
    CONTROL_STREAM_EPOCH_FIELD_NUMBER: _ClassVar[int]
    WORKER_BOOT_ID_FIELD_NUMBER: _ClassVar[int]
    OPERATION_ID_FIELD_NUMBER: _ClassVar[int]
    SOURCE_SELECTION_DIGEST_FIELD_NUMBER: _ClassVar[int]
    PROFILES_FIELD_NUMBER: _ClassVar[int]
    SOURCE_URI_FIELD_NUMBER: _ClassVar[int]
    DECLARED_LICENSE_FIELD_NUMBER: _ClassVar[int]
    CHECKPOINTS_FIELD_NUMBER: _ClassVar[int]
    record_owner_epoch: int
    control_stream_epoch: int
    worker_boot_id: str
    operation_id: str
    source_selection_digest: bytes
    profiles: _containers.RepeatedCompositeFieldContainer[ModelSourceProfile]
    source_uri: str
    declared_license: str
    checkpoints: _containers.RepeatedCompositeFieldContainer[ModelSourceCheckpoint]
    def __init__(self, record_owner_epoch: _Optional[int] = ..., control_stream_epoch: _Optional[int] = ..., worker_boot_id: _Optional[str] = ..., operation_id: _Optional[str] = ..., source_selection_digest: _Optional[bytes] = ..., profiles: _Optional[_Iterable[_Union[ModelSourceProfile, _Mapping]]] = ..., source_uri: _Optional[str] = ..., declared_license: _Optional[str] = ..., checkpoints: _Optional[_Iterable[_Union[ModelSourceCheckpoint, _Mapping]]] = ...) -> None: ...

class ModelSourcePrepared(_message.Message):
    __slots__ = ("record_owner_epoch", "control_stream_epoch", "worker_boot_id", "operation_id", "source_selection_digest", "outcome", "sources", "safe_code", "safe_detail", "checkpoints")
    RECORD_OWNER_EPOCH_FIELD_NUMBER: _ClassVar[int]
    CONTROL_STREAM_EPOCH_FIELD_NUMBER: _ClassVar[int]
    WORKER_BOOT_ID_FIELD_NUMBER: _ClassVar[int]
    OPERATION_ID_FIELD_NUMBER: _ClassVar[int]
    SOURCE_SELECTION_DIGEST_FIELD_NUMBER: _ClassVar[int]
    OUTCOME_FIELD_NUMBER: _ClassVar[int]
    SOURCES_FIELD_NUMBER: _ClassVar[int]
    SAFE_CODE_FIELD_NUMBER: _ClassVar[int]
    SAFE_DETAIL_FIELD_NUMBER: _ClassVar[int]
    CHECKPOINTS_FIELD_NUMBER: _ClassVar[int]
    record_owner_epoch: int
    control_stream_epoch: int
    worker_boot_id: str
    operation_id: str
    source_selection_digest: bytes
    outcome: ModelSourcePrepareOutcome
    sources: _containers.RepeatedCompositeFieldContainer[PreparedModelSource]
    safe_code: str
    safe_detail: str
    checkpoints: _containers.RepeatedCompositeFieldContainer[ModelSourceCheckpoint]
    def __init__(self, record_owner_epoch: _Optional[int] = ..., control_stream_epoch: _Optional[int] = ..., worker_boot_id: _Optional[str] = ..., operation_id: _Optional[str] = ..., source_selection_digest: _Optional[bytes] = ..., outcome: _Optional[_Union[ModelSourcePrepareOutcome, str]] = ..., sources: _Optional[_Iterable[_Union[PreparedModelSource, _Mapping]]] = ..., safe_code: _Optional[str] = ..., safe_detail: _Optional[str] = ..., checkpoints: _Optional[_Iterable[_Union[ModelSourceCheckpoint, _Mapping]]] = ...) -> None: ...

class LocalPackageFileGrant(_message.Message):
    __slots__ = ("digest", "filename", "length", "url")
    DIGEST_FIELD_NUMBER: _ClassVar[int]
    FILENAME_FIELD_NUMBER: _ClassVar[int]
    LENGTH_FIELD_NUMBER: _ClassVar[int]
    URL_FIELD_NUMBER: _ClassVar[int]
    digest: bytes
    filename: str
    length: int
    url: str
    def __init__(self, digest: _Optional[bytes] = ..., filename: _Optional[str] = ..., length: _Optional[int] = ..., url: _Optional[str] = ...) -> None: ...

class LocalPackageFetchRequest(_message.Message):
    __slots__ = ("record_owner_epoch", "control_stream_epoch", "worker_boot_id", "operation_id", "source_digest", "files")
    RECORD_OWNER_EPOCH_FIELD_NUMBER: _ClassVar[int]
    CONTROL_STREAM_EPOCH_FIELD_NUMBER: _ClassVar[int]
    WORKER_BOOT_ID_FIELD_NUMBER: _ClassVar[int]
    OPERATION_ID_FIELD_NUMBER: _ClassVar[int]
    SOURCE_DIGEST_FIELD_NUMBER: _ClassVar[int]
    FILES_FIELD_NUMBER: _ClassVar[int]
    record_owner_epoch: int
    control_stream_epoch: int
    worker_boot_id: str
    operation_id: str
    source_digest: bytes
    files: _containers.RepeatedCompositeFieldContainer[LocalPackageFileGrant]
    def __init__(self, record_owner_epoch: _Optional[int] = ..., control_stream_epoch: _Optional[int] = ..., worker_boot_id: _Optional[str] = ..., operation_id: _Optional[str] = ..., source_digest: _Optional[bytes] = ..., files: _Optional[_Iterable[_Union[LocalPackageFileGrant, _Mapping]]] = ...) -> None: ...

class LocalPackageFileStatus(_message.Message):
    __slots__ = ("record_owner_epoch", "control_stream_epoch", "worker_boot_id", "operation_id", "source_digest", "digest", "filename", "length", "state", "received_bytes", "safe_code", "safe_detail")
    RECORD_OWNER_EPOCH_FIELD_NUMBER: _ClassVar[int]
    CONTROL_STREAM_EPOCH_FIELD_NUMBER: _ClassVar[int]
    WORKER_BOOT_ID_FIELD_NUMBER: _ClassVar[int]
    OPERATION_ID_FIELD_NUMBER: _ClassVar[int]
    SOURCE_DIGEST_FIELD_NUMBER: _ClassVar[int]
    DIGEST_FIELD_NUMBER: _ClassVar[int]
    FILENAME_FIELD_NUMBER: _ClassVar[int]
    LENGTH_FIELD_NUMBER: _ClassVar[int]
    STATE_FIELD_NUMBER: _ClassVar[int]
    RECEIVED_BYTES_FIELD_NUMBER: _ClassVar[int]
    SAFE_CODE_FIELD_NUMBER: _ClassVar[int]
    SAFE_DETAIL_FIELD_NUMBER: _ClassVar[int]
    record_owner_epoch: int
    control_stream_epoch: int
    worker_boot_id: str
    operation_id: str
    source_digest: bytes
    digest: bytes
    filename: str
    length: int
    state: LocalPackageFileState
    received_bytes: int
    safe_code: str
    safe_detail: str
    def __init__(self, record_owner_epoch: _Optional[int] = ..., control_stream_epoch: _Optional[int] = ..., worker_boot_id: _Optional[str] = ..., operation_id: _Optional[str] = ..., source_digest: _Optional[bytes] = ..., digest: _Optional[bytes] = ..., filename: _Optional[str] = ..., length: _Optional[int] = ..., state: _Optional[_Union[LocalPackageFileState, str]] = ..., received_bytes: _Optional[int] = ..., safe_code: _Optional[str] = ..., safe_detail: _Optional[str] = ...) -> None: ...

class LocalPackageAbort(_message.Message):
    __slots__ = ("record_owner_epoch", "control_stream_epoch", "worker_boot_id", "operation_id", "source_digest", "local_revision_digest")
    RECORD_OWNER_EPOCH_FIELD_NUMBER: _ClassVar[int]
    CONTROL_STREAM_EPOCH_FIELD_NUMBER: _ClassVar[int]
    WORKER_BOOT_ID_FIELD_NUMBER: _ClassVar[int]
    OPERATION_ID_FIELD_NUMBER: _ClassVar[int]
    SOURCE_DIGEST_FIELD_NUMBER: _ClassVar[int]
    LOCAL_REVISION_DIGEST_FIELD_NUMBER: _ClassVar[int]
    record_owner_epoch: int
    control_stream_epoch: int
    worker_boot_id: str
    operation_id: str
    source_digest: bytes
    local_revision_digest: bytes
    def __init__(self, record_owner_epoch: _Optional[int] = ..., control_stream_epoch: _Optional[int] = ..., worker_boot_id: _Optional[str] = ..., operation_id: _Optional[str] = ..., source_digest: _Optional[bytes] = ..., local_revision_digest: _Optional[bytes] = ...) -> None: ...

class LocalPackageAbortStatus(_message.Message):
    __slots__ = ("record_owner_epoch", "control_stream_epoch", "worker_boot_id", "operation_id", "source_digest", "local_revision_digest", "outcome", "safe_code", "safe_detail")
    RECORD_OWNER_EPOCH_FIELD_NUMBER: _ClassVar[int]
    CONTROL_STREAM_EPOCH_FIELD_NUMBER: _ClassVar[int]
    WORKER_BOOT_ID_FIELD_NUMBER: _ClassVar[int]
    OPERATION_ID_FIELD_NUMBER: _ClassVar[int]
    SOURCE_DIGEST_FIELD_NUMBER: _ClassVar[int]
    LOCAL_REVISION_DIGEST_FIELD_NUMBER: _ClassVar[int]
    OUTCOME_FIELD_NUMBER: _ClassVar[int]
    SAFE_CODE_FIELD_NUMBER: _ClassVar[int]
    SAFE_DETAIL_FIELD_NUMBER: _ClassVar[int]
    record_owner_epoch: int
    control_stream_epoch: int
    worker_boot_id: str
    operation_id: str
    source_digest: bytes
    local_revision_digest: bytes
    outcome: LocalPackageAbortOutcome
    safe_code: str
    safe_detail: str
    def __init__(self, record_owner_epoch: _Optional[int] = ..., control_stream_epoch: _Optional[int] = ..., worker_boot_id: _Optional[str] = ..., operation_id: _Optional[str] = ..., source_digest: _Optional[bytes] = ..., local_revision_digest: _Optional[bytes] = ..., outcome: _Optional[_Union[LocalPackageAbortOutcome, str]] = ..., safe_code: _Optional[str] = ..., safe_detail: _Optional[str] = ...) -> None: ...

class WeightsFinalizeRequest(_message.Message):
    __slots__ = ("record_owner_epoch", "control_stream_epoch", "worker_boot_id", "request_id", "invocation_spec_digest", "output_slot", "disposition", "weights_receipt_digest", "scratch_root_id", "owner_authority_scope", "invocation_spec_canonical_bytes")
    RECORD_OWNER_EPOCH_FIELD_NUMBER: _ClassVar[int]
    CONTROL_STREAM_EPOCH_FIELD_NUMBER: _ClassVar[int]
    WORKER_BOOT_ID_FIELD_NUMBER: _ClassVar[int]
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    INVOCATION_SPEC_DIGEST_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_SLOT_FIELD_NUMBER: _ClassVar[int]
    DISPOSITION_FIELD_NUMBER: _ClassVar[int]
    WEIGHTS_RECEIPT_DIGEST_FIELD_NUMBER: _ClassVar[int]
    SCRATCH_ROOT_ID_FIELD_NUMBER: _ClassVar[int]
    OWNER_AUTHORITY_SCOPE_FIELD_NUMBER: _ClassVar[int]
    INVOCATION_SPEC_CANONICAL_BYTES_FIELD_NUMBER: _ClassVar[int]
    record_owner_epoch: int
    control_stream_epoch: int
    worker_boot_id: str
    request_id: str
    invocation_spec_digest: bytes
    output_slot: str
    disposition: WeightsFinalizeDisposition
    weights_receipt_digest: bytes
    scratch_root_id: str
    owner_authority_scope: str
    invocation_spec_canonical_bytes: bytes
    def __init__(self, record_owner_epoch: _Optional[int] = ..., control_stream_epoch: _Optional[int] = ..., worker_boot_id: _Optional[str] = ..., request_id: _Optional[str] = ..., invocation_spec_digest: _Optional[bytes] = ..., output_slot: _Optional[str] = ..., disposition: _Optional[_Union[WeightsFinalizeDisposition, str]] = ..., weights_receipt_digest: _Optional[bytes] = ..., scratch_root_id: _Optional[str] = ..., owner_authority_scope: _Optional[str] = ..., invocation_spec_canonical_bytes: _Optional[bytes] = ...) -> None: ...

class WeightsFinalizeResult(_message.Message):
    __slots__ = ("record_owner_epoch", "control_stream_epoch", "worker_boot_id", "request_id", "invocation_spec_digest", "output_slot", "outcome", "weights_receipt", "owner_authority_scope")
    RECORD_OWNER_EPOCH_FIELD_NUMBER: _ClassVar[int]
    CONTROL_STREAM_EPOCH_FIELD_NUMBER: _ClassVar[int]
    WORKER_BOOT_ID_FIELD_NUMBER: _ClassVar[int]
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    INVOCATION_SPEC_DIGEST_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_SLOT_FIELD_NUMBER: _ClassVar[int]
    OUTCOME_FIELD_NUMBER: _ClassVar[int]
    WEIGHTS_RECEIPT_FIELD_NUMBER: _ClassVar[int]
    OWNER_AUTHORITY_SCOPE_FIELD_NUMBER: _ClassVar[int]
    record_owner_epoch: int
    control_stream_epoch: int
    worker_boot_id: str
    request_id: str
    invocation_spec_digest: bytes
    output_slot: str
    outcome: WeightsFinalizeOutcome
    weights_receipt: WeightsReceiptRef
    owner_authority_scope: str
    def __init__(self, record_owner_epoch: _Optional[int] = ..., control_stream_epoch: _Optional[int] = ..., worker_boot_id: _Optional[str] = ..., request_id: _Optional[str] = ..., invocation_spec_digest: _Optional[bytes] = ..., output_slot: _Optional[str] = ..., outcome: _Optional[_Union[WeightsFinalizeOutcome, str]] = ..., weights_receipt: _Optional[_Union[WeightsReceiptRef, _Mapping]] = ..., owner_authority_scope: _Optional[str] = ...) -> None: ...

class ResultEnvelope(_message.Message):
    __slots__ = ("result_schema_digest", "inline_result", "result_blob", "adjustments", "checkpoint_ref", "retained_models")
    RESULT_SCHEMA_DIGEST_FIELD_NUMBER: _ClassVar[int]
    INLINE_RESULT_FIELD_NUMBER: _ClassVar[int]
    RESULT_BLOB_FIELD_NUMBER: _ClassVar[int]
    ADJUSTMENTS_FIELD_NUMBER: _ClassVar[int]
    CHECKPOINT_REF_FIELD_NUMBER: _ClassVar[int]
    RETAINED_MODELS_FIELD_NUMBER: _ClassVar[int]
    result_schema_digest: bytes
    inline_result: bytes
    result_blob: OutputEntry
    adjustments: _containers.RepeatedCompositeFieldContainer[AdjustmentRow]
    checkpoint_ref: str
    retained_models: _containers.RepeatedCompositeFieldContainer[RetainedModelResult]
    def __init__(self, result_schema_digest: _Optional[bytes] = ..., inline_result: _Optional[bytes] = ..., result_blob: _Optional[_Union[OutputEntry, _Mapping]] = ..., adjustments: _Optional[_Iterable[_Union[AdjustmentRow, _Mapping]]] = ..., checkpoint_ref: _Optional[str] = ..., retained_models: _Optional[_Iterable[_Union[RetainedModelResult, _Mapping]]] = ...) -> None: ...

class RetainedModelResult(_message.Message):
    __slots__ = ("result_pointer", "model_artifact_canonical_bytes", "retention")
    RESULT_POINTER_FIELD_NUMBER: _ClassVar[int]
    MODEL_ARTIFACT_CANONICAL_BYTES_FIELD_NUMBER: _ClassVar[int]
    RETENTION_FIELD_NUMBER: _ClassVar[int]
    result_pointer: str
    model_artifact_canonical_bytes: bytes
    retention: DerivedRetentionRequest
    def __init__(self, result_pointer: _Optional[str] = ..., model_artifact_canonical_bytes: _Optional[bytes] = ..., retention: _Optional[_Union[DerivedRetentionRequest, _Mapping]] = ...) -> None: ...

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
    __slots__ = ("record_owner_epoch", "control_stream_epoch", "worker_boot_id", "request_id", "attempt_ordinal", "invocation_spec_digest", "outcome_id", "outcome_digest", "retain_work")
    RECORD_OWNER_EPOCH_FIELD_NUMBER: _ClassVar[int]
    CONTROL_STREAM_EPOCH_FIELD_NUMBER: _ClassVar[int]
    WORKER_BOOT_ID_FIELD_NUMBER: _ClassVar[int]
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    ATTEMPT_ORDINAL_FIELD_NUMBER: _ClassVar[int]
    INVOCATION_SPEC_DIGEST_FIELD_NUMBER: _ClassVar[int]
    OUTCOME_ID_FIELD_NUMBER: _ClassVar[int]
    OUTCOME_DIGEST_FIELD_NUMBER: _ClassVar[int]
    RETAIN_WORK_FIELD_NUMBER: _ClassVar[int]
    record_owner_epoch: int
    control_stream_epoch: int
    worker_boot_id: str
    request_id: str
    attempt_ordinal: int
    invocation_spec_digest: bytes
    outcome_id: str
    outcome_digest: bytes
    retain_work: bool
    def __init__(self, record_owner_epoch: _Optional[int] = ..., control_stream_epoch: _Optional[int] = ..., worker_boot_id: _Optional[str] = ..., request_id: _Optional[str] = ..., attempt_ordinal: _Optional[int] = ..., invocation_spec_digest: _Optional[bytes] = ..., outcome_id: _Optional[str] = ..., outcome_digest: _Optional[bytes] = ..., retain_work: bool = ...) -> None: ...

class JobCheckpointRequest(_message.Message):
    __slots__ = ("record_owner_epoch", "control_stream_epoch", "worker_boot_id", "request_id", "attempt_ordinal", "operation_key", "logical_key", "content_digest", "seq", "artifact")
    RECORD_OWNER_EPOCH_FIELD_NUMBER: _ClassVar[int]
    CONTROL_STREAM_EPOCH_FIELD_NUMBER: _ClassVar[int]
    WORKER_BOOT_ID_FIELD_NUMBER: _ClassVar[int]
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    ATTEMPT_ORDINAL_FIELD_NUMBER: _ClassVar[int]
    OPERATION_KEY_FIELD_NUMBER: _ClassVar[int]
    LOGICAL_KEY_FIELD_NUMBER: _ClassVar[int]
    CONTENT_DIGEST_FIELD_NUMBER: _ClassVar[int]
    SEQ_FIELD_NUMBER: _ClassVar[int]
    ARTIFACT_FIELD_NUMBER: _ClassVar[int]
    record_owner_epoch: int
    control_stream_epoch: int
    worker_boot_id: str
    request_id: str
    attempt_ordinal: int
    operation_key: str
    logical_key: str
    content_digest: bytes
    seq: int
    artifact: OutputEntry
    def __init__(self, record_owner_epoch: _Optional[int] = ..., control_stream_epoch: _Optional[int] = ..., worker_boot_id: _Optional[str] = ..., request_id: _Optional[str] = ..., attempt_ordinal: _Optional[int] = ..., operation_key: _Optional[str] = ..., logical_key: _Optional[str] = ..., content_digest: _Optional[bytes] = ..., seq: _Optional[int] = ..., artifact: _Optional[_Union[OutputEntry, _Mapping]] = ...) -> None: ...

class JobCheckpointReceipt(_message.Message):
    __slots__ = ("record_owner_epoch", "control_stream_epoch", "worker_boot_id", "request_id", "attempt_ordinal", "operation_key", "logical_key", "content_digest", "receipt_id", "outcome", "fault")
    RECORD_OWNER_EPOCH_FIELD_NUMBER: _ClassVar[int]
    CONTROL_STREAM_EPOCH_FIELD_NUMBER: _ClassVar[int]
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
    control_stream_epoch: int
    worker_boot_id: str
    request_id: str
    attempt_ordinal: int
    operation_key: str
    logical_key: str
    content_digest: bytes
    receipt_id: str
    outcome: CheckpointOutcome
    fault: CheckpointFault
    def __init__(self, record_owner_epoch: _Optional[int] = ..., control_stream_epoch: _Optional[int] = ..., worker_boot_id: _Optional[str] = ..., request_id: _Optional[str] = ..., attempt_ordinal: _Optional[int] = ..., operation_key: _Optional[str] = ..., logical_key: _Optional[str] = ..., content_digest: _Optional[bytes] = ..., receipt_id: _Optional[str] = ..., outcome: _Optional[_Union[CheckpointOutcome, str]] = ..., fault: _Optional[_Union[CheckpointFault, _Mapping]] = ...) -> None: ...

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
    __slots__ = ("record_owner_epoch", "control_stream_epoch", "worker_boot_id", "request_id", "attempt_ordinal", "operation_key", "receipt_id", "content_digest")
    RECORD_OWNER_EPOCH_FIELD_NUMBER: _ClassVar[int]
    CONTROL_STREAM_EPOCH_FIELD_NUMBER: _ClassVar[int]
    WORKER_BOOT_ID_FIELD_NUMBER: _ClassVar[int]
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    ATTEMPT_ORDINAL_FIELD_NUMBER: _ClassVar[int]
    OPERATION_KEY_FIELD_NUMBER: _ClassVar[int]
    RECEIPT_ID_FIELD_NUMBER: _ClassVar[int]
    CONTENT_DIGEST_FIELD_NUMBER: _ClassVar[int]
    record_owner_epoch: int
    control_stream_epoch: int
    worker_boot_id: str
    request_id: str
    attempt_ordinal: int
    operation_key: str
    receipt_id: str
    content_digest: bytes
    def __init__(self, record_owner_epoch: _Optional[int] = ..., control_stream_epoch: _Optional[int] = ..., worker_boot_id: _Optional[str] = ..., request_id: _Optional[str] = ..., attempt_ordinal: _Optional[int] = ..., operation_key: _Optional[str] = ..., receipt_id: _Optional[str] = ..., content_digest: _Optional[bytes] = ...) -> None: ...

class ProgressOpen(_message.Message):
    __slots__ = ("record_owner_epoch", "control_stream_epoch", "worker_boot_id", "request_id")
    RECORD_OWNER_EPOCH_FIELD_NUMBER: _ClassVar[int]
    CONTROL_STREAM_EPOCH_FIELD_NUMBER: _ClassVar[int]
    WORKER_BOOT_ID_FIELD_NUMBER: _ClassVar[int]
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    record_owner_epoch: int
    control_stream_epoch: int
    worker_boot_id: str
    request_id: str
    def __init__(self, record_owner_epoch: _Optional[int] = ..., control_stream_epoch: _Optional[int] = ..., worker_boot_id: _Optional[str] = ..., request_id: _Optional[str] = ...) -> None: ...

class AttemptProgress(_message.Message):
    __slots__ = ("record_owner_epoch", "control_stream_epoch", "worker_boot_id", "request_id", "attempt_ordinal", "seq", "content_type", "data", "placement_id")
    RECORD_OWNER_EPOCH_FIELD_NUMBER: _ClassVar[int]
    CONTROL_STREAM_EPOCH_FIELD_NUMBER: _ClassVar[int]
    WORKER_BOOT_ID_FIELD_NUMBER: _ClassVar[int]
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    ATTEMPT_ORDINAL_FIELD_NUMBER: _ClassVar[int]
    SEQ_FIELD_NUMBER: _ClassVar[int]
    CONTENT_TYPE_FIELD_NUMBER: _ClassVar[int]
    DATA_FIELD_NUMBER: _ClassVar[int]
    PLACEMENT_ID_FIELD_NUMBER: _ClassVar[int]
    record_owner_epoch: int
    control_stream_epoch: int
    worker_boot_id: str
    request_id: str
    attempt_ordinal: int
    seq: int
    content_type: str
    data: bytes
    placement_id: str
    def __init__(self, record_owner_epoch: _Optional[int] = ..., control_stream_epoch: _Optional[int] = ..., worker_boot_id: _Optional[str] = ..., request_id: _Optional[str] = ..., attempt_ordinal: _Optional[int] = ..., seq: _Optional[int] = ..., content_type: _Optional[str] = ..., data: _Optional[bytes] = ..., placement_id: _Optional[str] = ...) -> None: ...

class InvocationSpec(_message.Message):
    __slots__ = ("environment_digest", "payload_digest", "inputs", "outputs", "deadline_unix_ms", "serving", "job", "capture", "attention_kernel")
    ENVIRONMENT_DIGEST_FIELD_NUMBER: _ClassVar[int]
    PAYLOAD_DIGEST_FIELD_NUMBER: _ClassVar[int]
    INPUTS_FIELD_NUMBER: _ClassVar[int]
    OUTPUTS_FIELD_NUMBER: _ClassVar[int]
    DEADLINE_UNIX_MS_FIELD_NUMBER: _ClassVar[int]
    SERVING_FIELD_NUMBER: _ClassVar[int]
    JOB_FIELD_NUMBER: _ClassVar[int]
    CAPTURE_FIELD_NUMBER: _ClassVar[int]
    ATTENTION_KERNEL_FIELD_NUMBER: _ClassVar[int]
    environment_digest: str
    payload_digest: str
    inputs: _containers.RepeatedCompositeFieldContainer[InputBinding]
    outputs: _containers.RepeatedCompositeFieldContainer[OutputBinding]
    deadline_unix_ms: int
    serving: ServingInvocationSpec
    job: JobInvocationSpec
    capture: ActivationCapture
    attention_kernel: str
    def __init__(self, environment_digest: _Optional[str] = ..., payload_digest: _Optional[str] = ..., inputs: _Optional[_Iterable[_Union[InputBinding, _Mapping]]] = ..., outputs: _Optional[_Iterable[_Union[OutputBinding, _Mapping]]] = ..., deadline_unix_ms: _Optional[int] = ..., serving: _Optional[_Union[ServingInvocationSpec, _Mapping]] = ..., job: _Optional[_Union[JobInvocationSpec, _Mapping]] = ..., capture: _Optional[_Union[ActivationCapture, _Mapping]] = ..., attention_kernel: _Optional[str] = ...) -> None: ...

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
    __slots__ = ("entrypoint_binding_digest", "attempt_binding_id", "bindings_digest")
    ENTRYPOINT_BINDING_DIGEST_FIELD_NUMBER: _ClassVar[int]
    ATTEMPT_BINDING_ID_FIELD_NUMBER: _ClassVar[int]
    BINDINGS_DIGEST_FIELD_NUMBER: _ClassVar[int]
    entrypoint_binding_digest: str
    attempt_binding_id: str
    bindings_digest: str
    def __init__(self, entrypoint_binding_digest: _Optional[str] = ..., attempt_binding_id: _Optional[str] = ..., bindings_digest: _Optional[str] = ...) -> None: ...

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

class CatalogModelSource(_message.Message):
    __slots__ = ("repository",)
    REPOSITORY_FIELD_NUMBER: _ClassVar[int]
    repository: str
    def __init__(self, repository: _Optional[str] = ...) -> None: ...

class InputAccess(_message.Message):
    __slots__ = ("input_id", "url", "native_tree", "catalog_model")
    INPUT_ID_FIELD_NUMBER: _ClassVar[int]
    URL_FIELD_NUMBER: _ClassVar[int]
    NATIVE_TREE_FIELD_NUMBER: _ClassVar[int]
    CATALOG_MODEL_FIELD_NUMBER: _ClassVar[int]
    input_id: str
    url: str
    native_tree: NativeByteRetentionRequest
    catalog_model: CatalogModelSource
    def __init__(self, input_id: _Optional[str] = ..., url: _Optional[str] = ..., native_tree: _Optional[_Union[NativeByteRetentionRequest, _Mapping]] = ..., catalog_model: _Optional[_Union[CatalogModelSource, _Mapping]] = ...) -> None: ...

class OutputAccess(_message.Message):
    __slots__ = ("output_id", "url")
    OUTPUT_ID_FIELD_NUMBER: _ClassVar[int]
    URL_FIELD_NUMBER: _ClassVar[int]
    output_id: str
    url: str
    def __init__(self, output_id: _Optional[str] = ..., url: _Optional[str] = ...) -> None: ...

class DeliveryAccessCredential(_message.Message):
    __slots__ = ("issuer", "key_id", "credential_epoch", "expires_at_unix", "token")
    ISSUER_FIELD_NUMBER: _ClassVar[int]
    KEY_ID_FIELD_NUMBER: _ClassVar[int]
    CREDENTIAL_EPOCH_FIELD_NUMBER: _ClassVar[int]
    EXPIRES_AT_UNIX_FIELD_NUMBER: _ClassVar[int]
    TOKEN_FIELD_NUMBER: _ClassVar[int]
    issuer: str
    key_id: str
    credential_epoch: int
    expires_at_unix: int
    token: bytes
    def __init__(self, issuer: _Optional[str] = ..., key_id: _Optional[str] = ..., credential_epoch: _Optional[int] = ..., expires_at_unix: _Optional[int] = ..., token: _Optional[bytes] = ...) -> None: ...

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
    __slots__ = ("jobs_in_flight", "jobs_available", "free_disk_bytes", "orchestration_in_flight", "orchestration_available")
    JOBS_IN_FLIGHT_FIELD_NUMBER: _ClassVar[int]
    JOBS_AVAILABLE_FIELD_NUMBER: _ClassVar[int]
    FREE_DISK_BYTES_FIELD_NUMBER: _ClassVar[int]
    ORCHESTRATION_IN_FLIGHT_FIELD_NUMBER: _ClassVar[int]
    ORCHESTRATION_AVAILABLE_FIELD_NUMBER: _ClassVar[int]
    jobs_in_flight: int
    jobs_available: int
    free_disk_bytes: int
    orchestration_in_flight: int
    orchestration_available: int
    def __init__(self, jobs_in_flight: _Optional[int] = ..., jobs_available: _Optional[int] = ..., free_disk_bytes: _Optional[int] = ..., orchestration_in_flight: _Optional[int] = ..., orchestration_available: _Optional[int] = ...) -> None: ...

class HeldAttempt(_message.Message):
    __slots__ = ("request_id", "attempt_ordinal", "kind", "state", "invocation_spec_digest", "placement_id", "executor_epoch", "outcome_id", "outcome_digest", "lane_id", "queue_position", "overtaken", "overtake_budget", "plan_digest")
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    ATTEMPT_ORDINAL_FIELD_NUMBER: _ClassVar[int]
    KIND_FIELD_NUMBER: _ClassVar[int]
    STATE_FIELD_NUMBER: _ClassVar[int]
    INVOCATION_SPEC_DIGEST_FIELD_NUMBER: _ClassVar[int]
    PLACEMENT_ID_FIELD_NUMBER: _ClassVar[int]
    EXECUTOR_EPOCH_FIELD_NUMBER: _ClassVar[int]
    OUTCOME_ID_FIELD_NUMBER: _ClassVar[int]
    OUTCOME_DIGEST_FIELD_NUMBER: _ClassVar[int]
    LANE_ID_FIELD_NUMBER: _ClassVar[int]
    QUEUE_POSITION_FIELD_NUMBER: _ClassVar[int]
    OVERTAKEN_FIELD_NUMBER: _ClassVar[int]
    OVERTAKE_BUDGET_FIELD_NUMBER: _ClassVar[int]
    PLAN_DIGEST_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    attempt_ordinal: int
    kind: AttemptKind
    state: AttemptState
    invocation_spec_digest: bytes
    placement_id: str
    executor_epoch: int
    outcome_id: str
    outcome_digest: bytes
    lane_id: str
    queue_position: int
    overtaken: int
    overtake_budget: int
    plan_digest: bytes
    def __init__(self, request_id: _Optional[str] = ..., attempt_ordinal: _Optional[int] = ..., kind: _Optional[_Union[AttemptKind, str]] = ..., state: _Optional[_Union[AttemptState, str]] = ..., invocation_spec_digest: _Optional[bytes] = ..., placement_id: _Optional[str] = ..., executor_epoch: _Optional[int] = ..., outcome_id: _Optional[str] = ..., outcome_digest: _Optional[bytes] = ..., lane_id: _Optional[str] = ..., queue_position: _Optional[int] = ..., overtaken: _Optional[int] = ..., overtake_budget: _Optional[int] = ..., plan_digest: _Optional[bytes] = ...) -> None: ...

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
    __slots__ = ("output_id", "digest", "length", "mime_type", "native_tree")
    OUTPUT_ID_FIELD_NUMBER: _ClassVar[int]
    DIGEST_FIELD_NUMBER: _ClassVar[int]
    LENGTH_FIELD_NUMBER: _ClassVar[int]
    MIME_TYPE_FIELD_NUMBER: _ClassVar[int]
    NATIVE_TREE_FIELD_NUMBER: _ClassVar[int]
    output_id: str
    digest: bytes
    length: int
    mime_type: str
    native_tree: NativeByteTreeRef
    def __init__(self, output_id: _Optional[str] = ..., digest: _Optional[bytes] = ..., length: _Optional[int] = ..., mime_type: _Optional[str] = ..., native_tree: _Optional[_Union[NativeByteTreeRef, _Mapping]] = ...) -> None: ...

class AttemptMetrics(_message.Message):
    __slots__ = ("runtime_ms", "queue_ms", "peak_device_memory_bytes", "rss_at_end_bytes", "output_count", "input_tokens", "output_tokens", "device_lease_ms", "device_count", "handler_ms", "finalization_ms", "unverified_fields", "post_ms")
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
    POST_MS_FIELD_NUMBER: _ClassVar[int]
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
    post_ms: int
    def __init__(self, runtime_ms: _Optional[int] = ..., queue_ms: _Optional[int] = ..., peak_device_memory_bytes: _Optional[int] = ..., rss_at_end_bytes: _Optional[int] = ..., output_count: _Optional[int] = ..., input_tokens: _Optional[int] = ..., output_tokens: _Optional[int] = ..., device_lease_ms: _Optional[int] = ..., device_count: _Optional[int] = ..., handler_ms: _Optional[int] = ..., finalization_ms: _Optional[int] = ..., unverified_fields: _Optional[_Iterable[str]] = ..., post_ms: _Optional[int] = ...) -> None: ...

class TriageBundleRef(_message.Message):
    __slots__ = ("subject_id", "write_receipt_digest", "length")
    SUBJECT_ID_FIELD_NUMBER: _ClassVar[int]
    WRITE_RECEIPT_DIGEST_FIELD_NUMBER: _ClassVar[int]
    LENGTH_FIELD_NUMBER: _ClassVar[int]
    subject_id: str
    write_receipt_digest: bytes
    length: int
    def __init__(self, subject_id: _Optional[str] = ..., write_receipt_digest: _Optional[bytes] = ..., length: _Optional[int] = ...) -> None: ...

class NativeArtifactTransfer(_message.Message):
    __slots__ = ("record_owner_epoch", "control_stream_epoch", "worker_boot_id", "effect_id", "source", "manifest", "command_id", "offset", "limit", "grant", "grant_revision", "server_time_unix", "byte_source")
    RECORD_OWNER_EPOCH_FIELD_NUMBER: _ClassVar[int]
    CONTROL_STREAM_EPOCH_FIELD_NUMBER: _ClassVar[int]
    WORKER_BOOT_ID_FIELD_NUMBER: _ClassVar[int]
    EFFECT_ID_FIELD_NUMBER: _ClassVar[int]
    SOURCE_FIELD_NUMBER: _ClassVar[int]
    MANIFEST_FIELD_NUMBER: _ClassVar[int]
    COMMAND_ID_FIELD_NUMBER: _ClassVar[int]
    OFFSET_FIELD_NUMBER: _ClassVar[int]
    LIMIT_FIELD_NUMBER: _ClassVar[int]
    GRANT_FIELD_NUMBER: _ClassVar[int]
    GRANT_REVISION_FIELD_NUMBER: _ClassVar[int]
    SERVER_TIME_UNIX_FIELD_NUMBER: _ClassVar[int]
    BYTE_SOURCE_FIELD_NUMBER: _ClassVar[int]
    record_owner_epoch: int
    control_stream_epoch: int
    worker_boot_id: str
    effect_id: str
    source: DerivedRetentionRequest
    manifest: Ref
    command_id: int
    offset: int
    limit: int
    grant: WeightsUploadGrant
    grant_revision: int
    server_time_unix: int
    byte_source: NativeByteRetentionRequest
    def __init__(self, record_owner_epoch: _Optional[int] = ..., control_stream_epoch: _Optional[int] = ..., worker_boot_id: _Optional[str] = ..., effect_id: _Optional[str] = ..., source: _Optional[_Union[DerivedRetentionRequest, _Mapping]] = ..., manifest: _Optional[_Union[Ref, _Mapping]] = ..., command_id: _Optional[int] = ..., offset: _Optional[int] = ..., limit: _Optional[int] = ..., grant: _Optional[_Union[WeightsUploadGrant, _Mapping]] = ..., grant_revision: _Optional[int] = ..., server_time_unix: _Optional[int] = ..., byte_source: _Optional[_Union[NativeByteRetentionRequest, _Mapping]] = ...) -> None: ...

class NativeArtifactTransferStatus(_message.Message):
    __slots__ = ("record_owner_epoch", "control_stream_epoch", "worker_boot_id", "effect_id", "source", "manifest", "command_id", "objects", "next_offset", "has_more", "closure_digest", "object_id", "grant_revision", "outcome", "transferred_bytes", "checksum_sha256", "http_status", "safe_code", "safe_detail", "byte_source")
    RECORD_OWNER_EPOCH_FIELD_NUMBER: _ClassVar[int]
    CONTROL_STREAM_EPOCH_FIELD_NUMBER: _ClassVar[int]
    WORKER_BOOT_ID_FIELD_NUMBER: _ClassVar[int]
    EFFECT_ID_FIELD_NUMBER: _ClassVar[int]
    SOURCE_FIELD_NUMBER: _ClassVar[int]
    MANIFEST_FIELD_NUMBER: _ClassVar[int]
    COMMAND_ID_FIELD_NUMBER: _ClassVar[int]
    OBJECTS_FIELD_NUMBER: _ClassVar[int]
    NEXT_OFFSET_FIELD_NUMBER: _ClassVar[int]
    HAS_MORE_FIELD_NUMBER: _ClassVar[int]
    CLOSURE_DIGEST_FIELD_NUMBER: _ClassVar[int]
    OBJECT_ID_FIELD_NUMBER: _ClassVar[int]
    GRANT_REVISION_FIELD_NUMBER: _ClassVar[int]
    OUTCOME_FIELD_NUMBER: _ClassVar[int]
    TRANSFERRED_BYTES_FIELD_NUMBER: _ClassVar[int]
    CHECKSUM_SHA256_FIELD_NUMBER: _ClassVar[int]
    HTTP_STATUS_FIELD_NUMBER: _ClassVar[int]
    SAFE_CODE_FIELD_NUMBER: _ClassVar[int]
    SAFE_DETAIL_FIELD_NUMBER: _ClassVar[int]
    BYTE_SOURCE_FIELD_NUMBER: _ClassVar[int]
    record_owner_epoch: int
    control_stream_epoch: int
    worker_boot_id: str
    effect_id: str
    source: DerivedRetentionRequest
    manifest: Ref
    command_id: int
    objects: _containers.RepeatedCompositeFieldContainer[WeightsObjectRef]
    next_offset: int
    has_more: bool
    closure_digest: bytes
    object_id: str
    grant_revision: int
    outcome: WeightsUploadOutcome
    transferred_bytes: int
    checksum_sha256: str
    http_status: int
    safe_code: str
    safe_detail: str
    byte_source: NativeByteRetentionRequest
    def __init__(self, record_owner_epoch: _Optional[int] = ..., control_stream_epoch: _Optional[int] = ..., worker_boot_id: _Optional[str] = ..., effect_id: _Optional[str] = ..., source: _Optional[_Union[DerivedRetentionRequest, _Mapping]] = ..., manifest: _Optional[_Union[Ref, _Mapping]] = ..., command_id: _Optional[int] = ..., objects: _Optional[_Iterable[_Union[WeightsObjectRef, _Mapping]]] = ..., next_offset: _Optional[int] = ..., has_more: bool = ..., closure_digest: _Optional[bytes] = ..., object_id: _Optional[str] = ..., grant_revision: _Optional[int] = ..., outcome: _Optional[_Union[WeightsUploadOutcome, str]] = ..., transferred_bytes: _Optional[int] = ..., checksum_sha256: _Optional[str] = ..., http_status: _Optional[int] = ..., safe_code: _Optional[str] = ..., safe_detail: _Optional[str] = ..., byte_source: _Optional[_Union[NativeByteRetentionRequest, _Mapping]] = ...) -> None: ...

class NativeByteTreeRef(_message.Message):
    __slots__ = ("producer_root_id", "receipt_digest", "manifest", "content_bytes")
    PRODUCER_ROOT_ID_FIELD_NUMBER: _ClassVar[int]
    RECEIPT_DIGEST_FIELD_NUMBER: _ClassVar[int]
    MANIFEST_FIELD_NUMBER: _ClassVar[int]
    CONTENT_BYTES_FIELD_NUMBER: _ClassVar[int]
    producer_root_id: str
    receipt_digest: bytes
    manifest: Ref
    content_bytes: int
    def __init__(self, producer_root_id: _Optional[str] = ..., receipt_digest: _Optional[bytes] = ..., manifest: _Optional[_Union[Ref, _Mapping]] = ..., content_bytes: _Optional[int] = ...) -> None: ...

class NativeByteRetentionRequest(_message.Message):
    __slots__ = ("source", "retention_id")
    SOURCE_FIELD_NUMBER: _ClassVar[int]
    RETENTION_ID_FIELD_NUMBER: _ClassVar[int]
    source: NativeByteTreeRef
    retention_id: str
    def __init__(self, source: _Optional[_Union[NativeByteTreeRef, _Mapping]] = ..., retention_id: _Optional[str] = ...) -> None: ...

class NativeByteRetentionCall(_message.Message):
    __slots__ = ("claim", "request")
    CLAIM_FIELD_NUMBER: _ClassVar[int]
    REQUEST_FIELD_NUMBER: _ClassVar[int]
    claim: Claim
    request: NativeByteRetentionRequest
    def __init__(self, claim: _Optional[_Union[Claim, _Mapping]] = ..., request: _Optional[_Union[NativeByteRetentionRequest, _Mapping]] = ...) -> None: ...

class NativeByteRetentionResult(_message.Message):
    __slots__ = ("source", "retention_id", "released")
    SOURCE_FIELD_NUMBER: _ClassVar[int]
    RETENTION_ID_FIELD_NUMBER: _ClassVar[int]
    RELEASED_FIELD_NUMBER: _ClassVar[int]
    source: NativeByteTreeRef
    retention_id: str
    released: bool
    def __init__(self, source: _Optional[_Union[NativeByteTreeRef, _Mapping]] = ..., retention_id: _Optional[str] = ..., released: bool = ...) -> None: ...

class ChildByteResultGrant(_message.Message):
    __slots__ = ("output_id", "source", "retention_id")
    OUTPUT_ID_FIELD_NUMBER: _ClassVar[int]
    SOURCE_FIELD_NUMBER: _ClassVar[int]
    RETENTION_ID_FIELD_NUMBER: _ClassVar[int]
    output_id: str
    source: NativeByteTreeRef
    retention_id: str
    def __init__(self, output_id: _Optional[str] = ..., source: _Optional[_Union[NativeByteTreeRef, _Mapping]] = ..., retention_id: _Optional[str] = ...) -> None: ...

class InputTreeImportHeader(_message.Message):
    __slots__ = ("claim", "request_id", "input_id", "manifest", "manifest_canonical_bytes", "content_bytes")
    CLAIM_FIELD_NUMBER: _ClassVar[int]
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    INPUT_ID_FIELD_NUMBER: _ClassVar[int]
    MANIFEST_FIELD_NUMBER: _ClassVar[int]
    MANIFEST_CANONICAL_BYTES_FIELD_NUMBER: _ClassVar[int]
    CONTENT_BYTES_FIELD_NUMBER: _ClassVar[int]
    claim: Claim
    request_id: str
    input_id: str
    manifest: Ref
    manifest_canonical_bytes: bytes
    content_bytes: int
    def __init__(self, claim: _Optional[_Union[Claim, _Mapping]] = ..., request_id: _Optional[str] = ..., input_id: _Optional[str] = ..., manifest: _Optional[_Union[Ref, _Mapping]] = ..., manifest_canonical_bytes: _Optional[bytes] = ..., content_bytes: _Optional[int] = ...) -> None: ...

class InputTreeImportBlob(_message.Message):
    __slots__ = ("object", "offset", "data")
    OBJECT_FIELD_NUMBER: _ClassVar[int]
    OFFSET_FIELD_NUMBER: _ClassVar[int]
    DATA_FIELD_NUMBER: _ClassVar[int]
    object: Ref
    offset: int
    data: bytes
    def __init__(self, object: _Optional[_Union[Ref, _Mapping]] = ..., offset: _Optional[int] = ..., data: _Optional[bytes] = ...) -> None: ...

class InputTreeImportCommit(_message.Message):
    __slots__ = ("abort",)
    ABORT_FIELD_NUMBER: _ClassVar[int]
    abort: bool
    def __init__(self, abort: bool = ...) -> None: ...

class InputTreeImportFrame(_message.Message):
    __slots__ = ("header", "blob", "commit")
    HEADER_FIELD_NUMBER: _ClassVar[int]
    BLOB_FIELD_NUMBER: _ClassVar[int]
    COMMIT_FIELD_NUMBER: _ClassVar[int]
    header: InputTreeImportHeader
    blob: InputTreeImportBlob
    commit: InputTreeImportCommit
    def __init__(self, header: _Optional[_Union[InputTreeImportHeader, _Mapping]] = ..., blob: _Optional[_Union[InputTreeImportBlob, _Mapping]] = ..., commit: _Optional[_Union[InputTreeImportCommit, _Mapping]] = ...) -> None: ...

class NativeByteReadCall(_message.Message):
    __slots__ = ("claim", "source", "object", "offset")
    CLAIM_FIELD_NUMBER: _ClassVar[int]
    SOURCE_FIELD_NUMBER: _ClassVar[int]
    OBJECT_FIELD_NUMBER: _ClassVar[int]
    OFFSET_FIELD_NUMBER: _ClassVar[int]
    claim: Claim
    source: NativeByteRetentionRequest
    object: Ref
    offset: int
    def __init__(self, claim: _Optional[_Union[Claim, _Mapping]] = ..., source: _Optional[_Union[NativeByteRetentionRequest, _Mapping]] = ..., object: _Optional[_Union[Ref, _Mapping]] = ..., offset: _Optional[int] = ...) -> None: ...

class NativeByteReadChunk(_message.Message):
    __slots__ = ("offset", "data")
    OFFSET_FIELD_NUMBER: _ClassVar[int]
    DATA_FIELD_NUMBER: _ClassVar[int]
    offset: int
    data: bytes
    def __init__(self, offset: _Optional[int] = ..., data: _Optional[bytes] = ...) -> None: ...

class ActivationCapture(_message.Message):
    __slots__ = ("components", "steps")
    COMPONENTS_FIELD_NUMBER: _ClassVar[int]
    STEPS_FIELD_NUMBER: _ClassVar[int]
    components: _containers.RepeatedScalarFieldContainer[str]
    steps: _containers.RepeatedScalarFieldContainer[int]
    def __init__(self, components: _Optional[_Iterable[str]] = ..., steps: _Optional[_Iterable[int]] = ...) -> None: ...

class ExecutionEnvironment(_message.Message):
    __slots__ = ("runtime_version", "worker_image_digest", "accelerator", "driver", "cuda", "worker_boot_id", "execution_lane", "execution_contract_digest", "kernel_symbol")
    RUNTIME_VERSION_FIELD_NUMBER: _ClassVar[int]
    WORKER_IMAGE_DIGEST_FIELD_NUMBER: _ClassVar[int]
    ACCELERATOR_FIELD_NUMBER: _ClassVar[int]
    DRIVER_FIELD_NUMBER: _ClassVar[int]
    CUDA_FIELD_NUMBER: _ClassVar[int]
    WORKER_BOOT_ID_FIELD_NUMBER: _ClassVar[int]
    EXECUTION_LANE_FIELD_NUMBER: _ClassVar[int]
    EXECUTION_CONTRACT_DIGEST_FIELD_NUMBER: _ClassVar[int]
    KERNEL_SYMBOL_FIELD_NUMBER: _ClassVar[int]
    runtime_version: str
    worker_image_digest: bytes
    accelerator: str
    driver: str
    cuda: str
    worker_boot_id: str
    execution_lane: str
    execution_contract_digest: bytes
    kernel_symbol: str
    def __init__(self, runtime_version: _Optional[str] = ..., worker_image_digest: _Optional[bytes] = ..., accelerator: _Optional[str] = ..., driver: _Optional[str] = ..., cuda: _Optional[str] = ..., worker_boot_id: _Optional[str] = ..., execution_lane: _Optional[str] = ..., execution_contract_digest: _Optional[bytes] = ..., kernel_symbol: _Optional[str] = ...) -> None: ...

class ActivationCaptureResult(_message.Message):
    __slots__ = ("output_id", "content_digest")
    OUTPUT_ID_FIELD_NUMBER: _ClassVar[int]
    CONTENT_DIGEST_FIELD_NUMBER: _ClassVar[int]
    output_id: str
    content_digest: bytes
    def __init__(self, output_id: _Optional[str] = ..., content_digest: _Optional[bytes] = ...) -> None: ...

class ExecutionObservation(_message.Message):
    __slots__ = ("environment", "capture")
    ENVIRONMENT_FIELD_NUMBER: _ClassVar[int]
    CAPTURE_FIELD_NUMBER: _ClassVar[int]
    environment: ExecutionEnvironment
    capture: ActivationCaptureResult
    def __init__(self, environment: _Optional[_Union[ExecutionEnvironment, _Mapping]] = ..., capture: _Optional[_Union[ActivationCaptureResult, _Mapping]] = ...) -> None: ...

class RuntimeRevision(_message.Message):
    __slots__ = ("wheel_digest", "wheel_length", "runtime_version", "python_abi", "provenance_digest", "channel")
    WHEEL_DIGEST_FIELD_NUMBER: _ClassVar[int]
    WHEEL_LENGTH_FIELD_NUMBER: _ClassVar[int]
    RUNTIME_VERSION_FIELD_NUMBER: _ClassVar[int]
    PYTHON_ABI_FIELD_NUMBER: _ClassVar[int]
    PROVENANCE_DIGEST_FIELD_NUMBER: _ClassVar[int]
    CHANNEL_FIELD_NUMBER: _ClassVar[int]
    wheel_digest: bytes
    wheel_length: int
    runtime_version: str
    python_abi: str
    provenance_digest: bytes
    channel: str
    def __init__(self, wheel_digest: _Optional[bytes] = ..., wheel_length: _Optional[int] = ..., runtime_version: _Optional[str] = ..., python_abi: _Optional[str] = ..., provenance_digest: _Optional[bytes] = ..., channel: _Optional[str] = ...) -> None: ...
