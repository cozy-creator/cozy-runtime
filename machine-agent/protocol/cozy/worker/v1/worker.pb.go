// cozy.worker.v1 — the RecordOwner<->worker wire contract. Schema only; no behavior.
//
// Authority: tracker-v2/worker-protocol/01-wire-schema.md (schema laws), 02-lifecycle.md
// (behavior), 03-versioning-conformance.md (versioning), rev-2-dynamic-serving.md (THIS
// revision). Issue: tracker/tensorhub/th-024.
//
// ORIENTATION: the WORKER (cozy-runtime's torch-free machine control process) HOSTS this
// service; the RECORDOWNER (the Cozy daemon, a pod's PodRecordOwner, or a
// tensorhub orchestrator-shard via th-007) DIALS it. The orientation is IDENTICAL local and
// remote; only channel establishment differs: a Unix socket (Windows: loopback) co-resident,
// TLS over a private network or authenticated overlay remotely. A worker control port is NEVER
// required to be publicly exposed for topology symmetry.
//
// VERSIONING: the MAJOR is the package path (`cozy.worker.v1`). The MINOR is `wire_minor`, a
// release train number. Each peer advertises its [minimum_wire_minor, wire_minor] range through
// ProtocolInfo, and its Claim carries its minor. Claim, ProtocolInfo, status, snapshots,
// keepalive, release and cancellation are never refused over a wire minor. Preparation and
// execution need the peer at or above WIRE_MINIMUM; below it only that operation fails with
// `capability_unavailable`, naming the component to update, and other work and rentals continue.
// Components release independently. An additive change raises WIRE_MINOR alone; a sender relying
// on it checks the peer's minor or a capability and falls back when the peer lacks it, and a peer
// without a new RPC answers UNIMPLEMENTED for that call only. WIRE_MINIMUM is the oldest deployed
// peer's minor: it rises only after a stated deprecation window, once no deployed peer still
// sends or reads what is removed, and never to equal WIRE_MINOR. After the v1 freeze, breaking
// changes need a new major.
// R7 IS AN AUTHORING RULE ONLY: `reserved` numbers and names are compiler-enforced
// tombstones against reuse; ordinary proto3 decoders do not refuse them on the wire and no
// runtime polices them. All generated bindings and fixtures move together.
//
// IDENTITY IS CANONICAL BYTES, PROTOBUF IS TRANSPORT: no digest is ever computed over
// protobuf-marshaled bytes. A meaning-fencing digest is the SHA-256 of a DOCUMENT's exact
// canonical bytes (RFC 8785-shaped canonical JSON under tfs-013's writer). Where the fence
// requires byte agreement the canonical bytes travel IN the message as a `bytes` field instead
// of a structured body; the receiver recomputes sha256 over the resident bytes BEFORE parsing a
// single field. DIGEST CLASSES: (a) document identity over canonical JSON bytes; (b) content
// digest over opaque artifact bytes. On the wire a class-(a)/(b) digest is raw 32-byte `bytes`;
// inside a canonical document the same digest is spelled `"sha256:<64 lowercase hex>"`.
// BYTES NAMING LAW: a `bytes` field named `digest`, `*_digest`, or repeated `*_digests` IS a
// 32-byte SHA-256; every other `bytes` field is an opaque payload (base64 in documents).
//
// THE CONTROL STREAM CARRIES CONTROL, NOT CONTENT. No frame moves an object's bytes. Every
// content-bearing field is bounded by a constant, and every one of those constants is <=
// `MaxInlineControlBytes` (4 MiB) PER TRANSFER -- not per chunk. The distinction is the whole
// rule: a 4 MiB frame is fine, and a 4 MiB frame sent 2,048 times to move 8 GiB is the defect,
// because the total it moves is bounded by nothing. Anything larger than the ceiling travels
// as a presigned URL to the process that holds the bytes, which is what `WeightsUploadGrant`
// and `DeliveryGrant.outputs` already do.
//
// The rule has NO carve-out and is mechanically enforced on the BOUND: tensorhub's
// `scripts/fence.py` convicts any content-bearing `Max*Bytes` constant above the ceiling,
// whatever the field carrying it is called. th-094 retired the last exemption, local package
// upload, which moved 1 GiB in 1 MiB frames.
//
// DOCUMENT VERSIONS. The canonical `format` tag is the message's full name plus `/N`. A reader
// verifies the digest over the carried bytes and their canonical encoding, then reads only the
// fields it consumes: an unknown key is IGNORED and an absent key means its default (an absent
// collection is empty). A key whose omission would change results is a capability: senders
// include it only for peers that advertise it, so ignoring an unknown key is always safe. A
// reader never re-serializes a parsed document to compare identities. `/N` changes only when a
// consumed field is removed or retyped.
//
// THE ENVELOPE (every message, fields 1-3): the ownership + boot fence, checked BEFORE any body
// field is read, in this order:
//   1 `record_owner_epoch`     - durable monotonic AUTHORITY epoch, minted by whatever
//                                grants ownership. Older than the accepted claim => dropped.
//   2 `control_stream_epoch`   - worker-minted, incremented for EVERY accepted control
//                                stream; a frame from a superseded stream is dropped.
//   3 `worker_boot_id`         - minted at worker-process boot, never reused; a frame
//                                addressed to a dead boot is dropped. An id, not an epoch:
//                                it need only DIFFER, never be greater.
// FIELD 4 IS RESERVED IN EVERY MESSAGE (#446): the old global `executor_incarnation` is GONE.
// The worker BOOT id is machine truth; each placement's EXECUTOR epoch is placement truth
// (`PlacementStatus.executor_epoch`) and rides attempt-scoped facts, not the envelope. The
// attempt fence triple is (request_id, attempt_ordinal, invocation_spec_digest).
//
// PLACEMENTS (#481; renamed from DEPLOYMENTS — tensorhub's "deployment" is an id-less semantic
// tuple and the collision was real). `placement_id` is the RecordOwner-minted routing + journal
// key for one hosted assignment. A placement's fully resolved immutable definition lives directly
// inside PlacementSet. Its identity is `(placement_set_digest, placement_id)`; there is no
// independent PlacementSpec document or digest. The worker never resolves a mutable release id,
// so two workers handed the same set converge to the same bytes or fault typed. One placement
// per package; a placement holds every construction its slots bind.
//
// TWO AXES, NOT ONE ENUM (#473/#482): a placement's convergence is MaterializationState x
// ServingState. A single phase cannot represent staged-on-disk x draining independently, which
// is exactly the state an outgoing spec holds under fallback-retention (#474). Machine
// lifecycle is `WorkerPhase`, pulled out of the placement enum; the old `IntakeState` is retired
// with both of its uses. "prepared"/"warming"/"ready" are retired as state words.
//
// ONE ADMISSION GATE (#472e/#482/#486c): per-placement `attempt_credits` are DELETED — N
// counters over ONE serialized device advertise N x the real capacity, and the defect is
// arithmetic. Execution capacity is a WORKER property (`admission_epoch` +
// `admission_state` + `available_attempt_slots`); dispatchability is a PLACEMENT property (the
// serving axis + dispatchable_binding_digests). Only `admission_epoch` FENCES; the other two are
// the GATE — "not accepting" / "full" — not a fence against a stale actor. Per-placement
// `readiness_epoch` is DELETED, not renamed: one fence, not two to keep consistent.
//
// EVERY OFFER GETS A JOURNALED OUTCOME (#472f/#480b): an AttemptOffer receives either
// AttemptAccepted or a JOURNALED AttemptOutcome(REFUSED) — never silence and never an
// unjournaled decline (there is no AttemptDeclined message to add). A pre-execution refusal
// consumes the attempt_ordinal and zero billed execution budget; the billing fact is the
// STRUCTURAL bit AttemptOutcomeBody.execution_started, never a cause-code allowlist.
//
// ONEOF-LAST RULE (transport determinism only, never identity): every oneof carries the highest
// field numbers in its message.
//
// ENUM VALUE NAMING: every value is prefixed with its enum's SCREAMING_SNAKE type name. proto3
// scopes enum values at PACKAGE level, so the unprefixed spellings in the rev document
// (OPEN/FAILED/DRAINING/OFFLINE) would collide across MaterializationState, ServingState,
// WorkerPhase and AdmissionState and fail to compile. NUMBERS are the normative part.
//
// THINNING ("no writer, no field"): the purge lane is a reservation — private-artifact purge
// (cr-013/tfs-014, post-M9) rides Control as a durable command/acknowledgement pair when it
// lands, holding oneof slot 14 on both frames. `AppliedAdapter` (cr-010) and `DrainPolicy`
// (th-007) stay reservations. The retired desired-state artifact delegation remains absent;
// ArtifactSink host durability and transfers instead use the explicit host-only frames below.
//
// THE HOST APPENDS, NEVER RE-AUTHORS (proto-025). A pod's supervisor is the
// pod-resident half of the HOST role the Cozy daemon performs in-process locally: it downloads
// and verifies, asks the Runtime's loopback preparation for PlacementSet bytes, moves objects,
// and keeps the pod ledger. That role has its own lanes so the host never rewrites
// a frame in flight: `PodHost` (owner <-> host, a second service on the pod listener),
// `RuntimeWeights` (host <-> Runtime, loopback), and a host-authored SECOND snapshot document
// beside the worker's, which passes through BYTE-IDENTICAL. The owner's transfer lanes
// (weights transfer, model source files and preparation, local package abort) still
// ride RecordOwnerFrame / WorkerFrame slots on WorkerControl, which the Host routes itself.
//
// DEVICE LANES (proto-024). The serialized resource inside one worker is a DEVICE
// LANE — a set of envelope-local device ordinals, one attempt seat, one ledger — not the worker
// (cr-066). The one-counter-per-serialized-resource principle behind the admission gate (#472e)
// therefore puts the seats PER LANE: `ObservedWorkerState.lanes[]`, with the worker-level
// `available_attempt_slots` provably their sum. A lane carries K >= 1 ordinals. K == 1 is every
// worker shipped so far and is byte-identical on the wire but for the new repeated field. K > 1
// is a GROUP lane: ONE executor sealed to all K devices (cr-018's sequence-parallel door:
// `gpu_count` is never request concurrency). The
// RecordOwner ASSIGNS a group by pinning a placement to K ordinals (`DesiredPlacementSet.
// device_pins`, beside the digest-fenced document, never inside it: where a placement sits on
// THIS worker is not part of what it serves); the worker validates the pin against its envelope
// and the package's declared construction and refuses typed. `PlacementStatus.device_lane_id`
// reports the lane a placement landed on. Nothing at 22 is removed or renumbered.
//
// THE ATTEMPT QUEUE (proto-026; gpu-hot.md §3-§6). The worker ORDERS its lane; the
// RecordOwner OFFERS depth. Acceptance is queue ADMISSION — journaled, hydrated, queued on a
// lane — and binds neither a plan nor an executor epoch: both bind at DEVICE ENTRY. An attempt
// on a lane is QUEUED -> RUNNING -> DEVICE_RELEASED -> OUTCOME_PENDING_ACK; the device seat is
// the RUNNING held attempt on the lane, never a count, and the post phase (encode, digest,
// write, outcome) runs beside the next occupant. Every accepted attempt appears in
// `held_attempts` from admission to ack with its lane, worker order, overtake count and budget
// (`HeldAttempt` 10-14); `DeviceLane.available_attempt_slots` becomes the offers the lane will
// still ADMIT TO ITS QUEUE now; `AttemptMetrics.post_ms` and an honest `device_lease_ms` split
// the two phases. `admission_epoch` no longer bumps on executor respawn: a queued attempt is
// bound to no executor. The four `AttemptAccepted` slots that bound plan and epoch at
// acceptance (8, 9, 10, 12) are RESERVED — a PRE-LAUNCH HARD CUT, no dual reading, no shim; an
// owner below 23 is below the floor. RESIDENCY
// (residency-aware-routing.md §2/§6, same minor): two observed SETS cross the wire and nothing
// else — `DeviceLane.resident_placement_ids` (executor holds device bytes on this lane now) and
// `ObservedWorkerState.held_manifests` (TensorFS manifests the verified store holds complete,
// placed or not), both also on `WorkerSnapshotBody/1`. No bytes, no headroom, no plan: the
// orchestrator routes on counts and membership; VRAM arithmetic stays where it is measured.
//
// THE PACKAGE WIRE (xs-018/xs-019/xs-017). What a pod is told about a published
// package is the hub's own release material and nothing derived: `application`, the
// interface-declared entrypoint model-slot paths, the placed image's pinned inventory, and the
// release's hash-pinned requirements export — all hub-known before the pod exists. Runtime
// preparation receives NO files: wheels come from the package indexes under --require-hashes,
// the package interface rides as the hub release's bytes, and model manifests/objects
// are resolved from the local TensorFS Store the privileged host admitted them to (proto-030).
// `LocalDownloadFile` and `LocalDownloadKind` are DELETED with the per-file handoff; the
// local-package plane keeps its explicitly supplied wheels — the only files with no index
// source — and loses its `kind` rows: the project wheel is the row that MEASURES as the
// package's own distribution/release. `PackageSelection.project_wheel` and `Environment.wheels`
// are deleted: a pod that never retains wheel bytes cannot re-measure WheelFacts, so a
// published install's identity is its locked requirements export
// (`Environment.locked_requirements`) over the image inventory, not a wheel list.
//
// THE STORAGE CUT (proto-031). DownloadDelegation no longer reports held manifests:
// TensorFS computes the missing closure locally and asks only for those bytes. Preparation names
// one immutable `install_root`, never an environment overlay. Local editable wheels stage below
// that install namespace. The proto-026 residency sets on ObservedWorkerState and
// WorkerSnapshotBody remain; the orchestrator still routes on them.
//
// THE EXECUTION LIFECYCLE (proto-061). A preparation may carry the hub release's
// PackageInterface/1 bytes; without them Runtime obtains the interface itself. Placement
// identity is the package release alone, stable as models are added. DISPATCHABLE means the
// executor started and imported the package; models load on demand, and
// `PlacementStatus.loaded_binding_digests` reports the constructions built in the live executor.
// `AttemptMetrics` carries measured working memory and its shape cell.

// Code generated by protoc-gen-go. DO NOT EDIT.
// versions:
// 	protoc-gen-go v1.36.11
// 	protoc        v7.35.1
// source: cozy/worker/v1/worker.proto

package workerprotov1

import (
	protoreflect "google.golang.org/protobuf/reflect/protoreflect"
	protoimpl "google.golang.org/protobuf/runtime/protoimpl"
	reflect "reflect"
	sync "sync"
	unsafe "unsafe"
)

const (
	// Verify that this generated code is sufficiently up-to-date.
	_ = protoimpl.EnforceVersion(20 - protoimpl.MinVersion)
	// Verify that runtime/protoimpl is sufficiently up-to-date.
	_ = protoimpl.EnforceVersion(protoimpl.MaxVersion - 20)
)

// Wire 72: the logs a machine keeps. Readers name an unknown value only to a newer Host; an
// older one answers NOT_FOUND for it.
type MachineLog int32

const (
	MachineLog_MACHINE_LOG_UNSPECIFIED MachineLog = 0
	// TensorFS's transport decisions, one line each: `<unix seconds> <kind> <subject> <detail>`
	// for every hedge, lane grant, win and pull walk. TensorFS rotates it to one older file.
	MachineLog_MACHINE_LOG_TENSORFS_TRANSPORT MachineLog = 1
)

// Enum value maps for MachineLog.
var (
	MachineLog_name = map[int32]string{
		0: "MACHINE_LOG_UNSPECIFIED",
		1: "MACHINE_LOG_TENSORFS_TRANSPORT",
	}
	MachineLog_value = map[string]int32{
		"MACHINE_LOG_UNSPECIFIED":        0,
		"MACHINE_LOG_TENSORFS_TRANSPORT": 1,
	}
)

func (x MachineLog) Enum() *MachineLog {
	p := new(MachineLog)
	*p = x
	return p
}

func (x MachineLog) String() string {
	return protoimpl.X.EnumStringOf(x.Descriptor(), protoreflect.EnumNumber(x))
}

func (MachineLog) Descriptor() protoreflect.EnumDescriptor {
	return file_cozy_worker_v1_worker_proto_enumTypes[0].Descriptor()
}

func (MachineLog) Type() protoreflect.EnumType {
	return &file_cozy_worker_v1_worker_proto_enumTypes[0]
}

func (x MachineLog) Number() protoreflect.EnumNumber {
	return protoreflect.EnumNumber(x)
}

// Deprecated: Use MachineLog.Descriptor instead.
func (MachineLog) EnumDescriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{0}
}

type RunProductOp int32

const (
	RunProductOp_RUN_PRODUCT_OP_UNSPECIFIED RunProductOp = 0
	RunProductOp_RUN_PRODUCT_OP_SET         RunProductOp = 1
	RunProductOp_RUN_PRODUCT_OP_APPEND      RunProductOp = 2
)

// Enum value maps for RunProductOp.
var (
	RunProductOp_name = map[int32]string{
		0: "RUN_PRODUCT_OP_UNSPECIFIED",
		1: "RUN_PRODUCT_OP_SET",
		2: "RUN_PRODUCT_OP_APPEND",
	}
	RunProductOp_value = map[string]int32{
		"RUN_PRODUCT_OP_UNSPECIFIED": 0,
		"RUN_PRODUCT_OP_SET":         1,
		"RUN_PRODUCT_OP_APPEND":      2,
	}
)

func (x RunProductOp) Enum() *RunProductOp {
	p := new(RunProductOp)
	*p = x
	return p
}

func (x RunProductOp) String() string {
	return protoimpl.X.EnumStringOf(x.Descriptor(), protoreflect.EnumNumber(x))
}

func (RunProductOp) Descriptor() protoreflect.EnumDescriptor {
	return file_cozy_worker_v1_worker_proto_enumTypes[1].Descriptor()
}

func (RunProductOp) Type() protoreflect.EnumType {
	return &file_cozy_worker_v1_worker_proto_enumTypes[1]
}

func (x RunProductOp) Number() protoreflect.EnumNumber {
	return protoreflect.EnumNumber(x)
}

// Deprecated: Use RunProductOp.Descriptor instead.
func (RunProductOp) EnumDescriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{1}
}

type MachineExecutionAction int32

const (
	MachineExecutionAction_MACHINE_EXECUTION_ACTION_UNSPECIFIED MachineExecutionAction = 0
	MachineExecutionAction_MACHINE_EXECUTION_ACTION_PAUSE       MachineExecutionAction = 1
	MachineExecutionAction_MACHINE_EXECUTION_ACTION_RESUME      MachineExecutionAction = 2
	MachineExecutionAction_MACHINE_EXECUTION_ACTION_CANCEL      MachineExecutionAction = 3
	// Settle one sent publication from MachineExecutionControl.publication. Runtime refuses it
	// while its own publication authority still answers.
	MachineExecutionAction_MACHINE_EXECUTION_ACTION_RECONCILE_PUBLICATION MachineExecutionAction = 4
	// Answer one memo.lookup from MachineExecutionControl.memo.
	MachineExecutionAction_MACHINE_EXECUTION_ACTION_ANSWER_MEMO MachineExecutionAction = 5
)

// Enum value maps for MachineExecutionAction.
var (
	MachineExecutionAction_name = map[int32]string{
		0: "MACHINE_EXECUTION_ACTION_UNSPECIFIED",
		1: "MACHINE_EXECUTION_ACTION_PAUSE",
		2: "MACHINE_EXECUTION_ACTION_RESUME",
		3: "MACHINE_EXECUTION_ACTION_CANCEL",
		4: "MACHINE_EXECUTION_ACTION_RECONCILE_PUBLICATION",
		5: "MACHINE_EXECUTION_ACTION_ANSWER_MEMO",
	}
	MachineExecutionAction_value = map[string]int32{
		"MACHINE_EXECUTION_ACTION_UNSPECIFIED":           0,
		"MACHINE_EXECUTION_ACTION_PAUSE":                 1,
		"MACHINE_EXECUTION_ACTION_RESUME":                2,
		"MACHINE_EXECUTION_ACTION_CANCEL":                3,
		"MACHINE_EXECUTION_ACTION_RECONCILE_PUBLICATION": 4,
		"MACHINE_EXECUTION_ACTION_ANSWER_MEMO":           5,
	}
)

func (x MachineExecutionAction) Enum() *MachineExecutionAction {
	p := new(MachineExecutionAction)
	*p = x
	return p
}

func (x MachineExecutionAction) String() string {
	return protoimpl.X.EnumStringOf(x.Descriptor(), protoreflect.EnumNumber(x))
}

func (MachineExecutionAction) Descriptor() protoreflect.EnumDescriptor {
	return file_cozy_worker_v1_worker_proto_enumTypes[2].Descriptor()
}

func (MachineExecutionAction) Type() protoreflect.EnumType {
	return &file_cozy_worker_v1_worker_proto_enumTypes[2]
}

func (x MachineExecutionAction) Number() protoreflect.EnumNumber {
	return protoreflect.EnumNumber(x)
}

// Deprecated: Use MachineExecutionAction.Descriptor instead.
func (MachineExecutionAction) EnumDescriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{2}
}

type ChildCallState int32

const (
	ChildCallState_CHILD_CALL_STATE_UNSPECIFIED ChildCallState = 0
	ChildCallState_CHILD_CALL_STATE_PENDING     ChildCallState = 1 // owner durably accepted; child may be queued or running
	ChildCallState_CHILD_CALL_STATE_SUCCEEDED   ChildCallState = 2
	ChildCallState_CHILD_CALL_STATE_REFUSED     ChildCallState = 3
	ChildCallState_CHILD_CALL_STATE_FAILED      ChildCallState = 4
	ChildCallState_CHILD_CALL_STATE_CANCELED    ChildCallState = 5
)

// Enum value maps for ChildCallState.
var (
	ChildCallState_name = map[int32]string{
		0: "CHILD_CALL_STATE_UNSPECIFIED",
		1: "CHILD_CALL_STATE_PENDING",
		2: "CHILD_CALL_STATE_SUCCEEDED",
		3: "CHILD_CALL_STATE_REFUSED",
		4: "CHILD_CALL_STATE_FAILED",
		5: "CHILD_CALL_STATE_CANCELED",
	}
	ChildCallState_value = map[string]int32{
		"CHILD_CALL_STATE_UNSPECIFIED": 0,
		"CHILD_CALL_STATE_PENDING":     1,
		"CHILD_CALL_STATE_SUCCEEDED":   2,
		"CHILD_CALL_STATE_REFUSED":     3,
		"CHILD_CALL_STATE_FAILED":      4,
		"CHILD_CALL_STATE_CANCELED":    5,
	}
)

func (x ChildCallState) Enum() *ChildCallState {
	p := new(ChildCallState)
	*p = x
	return p
}

func (x ChildCallState) String() string {
	return protoimpl.X.EnumStringOf(x.Descriptor(), protoreflect.EnumNumber(x))
}

func (ChildCallState) Descriptor() protoreflect.EnumDescriptor {
	return file_cozy_worker_v1_worker_proto_enumTypes[3].Descriptor()
}

func (ChildCallState) Type() protoreflect.EnumType {
	return &file_cozy_worker_v1_worker_proto_enumTypes[3]
}

func (x ChildCallState) Number() protoreflect.EnumNumber {
	return protoreflect.EnumNumber(x)
}

// Deprecated: Use ChildCallState.Descriptor instead.
func (ChildCallState) EnumDescriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{3}
}

// Fixed native source services share the ordinary parent-call identity and
// control transport. They are never package builds or a generic privileged RPC registry.
// RESOLVE performs metadata only. The owner durably records the exact selection before
// EXECUTE can move source payloads. Reattachment preserves service_id across attempts.
type NativeSourcePhase int32

const (
	NativeSourcePhase_NATIVE_SOURCE_PHASE_UNSPECIFIED NativeSourcePhase = 0
	NativeSourcePhase_NATIVE_SOURCE_PHASE_RESOLVE     NativeSourcePhase = 1
	NativeSourcePhase_NATIVE_SOURCE_PHASE_EXECUTE     NativeSourcePhase = 2
	NativeSourcePhase_NATIVE_SOURCE_PHASE_CANCEL      NativeSourcePhase = 3
)

// Enum value maps for NativeSourcePhase.
var (
	NativeSourcePhase_name = map[int32]string{
		0: "NATIVE_SOURCE_PHASE_UNSPECIFIED",
		1: "NATIVE_SOURCE_PHASE_RESOLVE",
		2: "NATIVE_SOURCE_PHASE_EXECUTE",
		3: "NATIVE_SOURCE_PHASE_CANCEL",
	}
	NativeSourcePhase_value = map[string]int32{
		"NATIVE_SOURCE_PHASE_UNSPECIFIED": 0,
		"NATIVE_SOURCE_PHASE_RESOLVE":     1,
		"NATIVE_SOURCE_PHASE_EXECUTE":     2,
		"NATIVE_SOURCE_PHASE_CANCEL":      3,
	}
)

func (x NativeSourcePhase) Enum() *NativeSourcePhase {
	p := new(NativeSourcePhase)
	*p = x
	return p
}

func (x NativeSourcePhase) String() string {
	return protoimpl.X.EnumStringOf(x.Descriptor(), protoreflect.EnumNumber(x))
}

func (NativeSourcePhase) Descriptor() protoreflect.EnumDescriptor {
	return file_cozy_worker_v1_worker_proto_enumTypes[4].Descriptor()
}

func (NativeSourcePhase) Type() protoreflect.EnumType {
	return &file_cozy_worker_v1_worker_proto_enumTypes[4]
}

func (x NativeSourcePhase) Number() protoreflect.EnumNumber {
	return protoreflect.EnumNumber(x)
}

// Deprecated: Use NativeSourcePhase.Descriptor instead.
func (NativeSourcePhase) EnumDescriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{4}
}

type NativeSourceOperation int32

const (
	NativeSourceOperation_NATIVE_SOURCE_OPERATION_UNSPECIFIED  NativeSourceOperation = 0
	NativeSourceOperation_NATIVE_SOURCE_OPERATION_HUGGINGFACE  NativeSourceOperation = 1
	NativeSourceOperation_NATIVE_SOURCE_OPERATION_CIVITAI      NativeSourceOperation = 2
	NativeSourceOperation_NATIVE_SOURCE_OPERATION_CONVERT      NativeSourceOperation = 3
	NativeSourceOperation_NATIVE_SOURCE_OPERATION_SOURCE_FILES NativeSourceOperation = 4 // bounded readonly ordinary-source view
	NativeSourceOperation_NATIVE_SOURCE_OPERATION_COMMIT_FILE  NativeSourceOperation = 5 // explicitly complete one own pending file
)

// Enum value maps for NativeSourceOperation.
var (
	NativeSourceOperation_name = map[int32]string{
		0: "NATIVE_SOURCE_OPERATION_UNSPECIFIED",
		1: "NATIVE_SOURCE_OPERATION_HUGGINGFACE",
		2: "NATIVE_SOURCE_OPERATION_CIVITAI",
		3: "NATIVE_SOURCE_OPERATION_CONVERT",
		4: "NATIVE_SOURCE_OPERATION_SOURCE_FILES",
		5: "NATIVE_SOURCE_OPERATION_COMMIT_FILE",
	}
	NativeSourceOperation_value = map[string]int32{
		"NATIVE_SOURCE_OPERATION_UNSPECIFIED":  0,
		"NATIVE_SOURCE_OPERATION_HUGGINGFACE":  1,
		"NATIVE_SOURCE_OPERATION_CIVITAI":      2,
		"NATIVE_SOURCE_OPERATION_CONVERT":      3,
		"NATIVE_SOURCE_OPERATION_SOURCE_FILES": 4,
		"NATIVE_SOURCE_OPERATION_COMMIT_FILE":  5,
	}
)

func (x NativeSourceOperation) Enum() *NativeSourceOperation {
	p := new(NativeSourceOperation)
	*p = x
	return p
}

func (x NativeSourceOperation) String() string {
	return protoimpl.X.EnumStringOf(x.Descriptor(), protoreflect.EnumNumber(x))
}

func (NativeSourceOperation) Descriptor() protoreflect.EnumDescriptor {
	return file_cozy_worker_v1_worker_proto_enumTypes[5].Descriptor()
}

func (NativeSourceOperation) Type() protoreflect.EnumType {
	return &file_cozy_worker_v1_worker_proto_enumTypes[5]
}

func (x NativeSourceOperation) Number() protoreflect.EnumNumber {
	return protoreflect.EnumNumber(x)
}

// Deprecated: Use NativeSourceOperation.Descriptor instead.
func (NativeSourceOperation) EnumDescriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{5}
}

type NativeSourceState int32

const (
	NativeSourceState_NATIVE_SOURCE_STATE_UNSPECIFIED NativeSourceState = 0
	NativeSourceState_NATIVE_SOURCE_STATE_RESOLVED    NativeSourceState = 1
	NativeSourceState_NATIVE_SOURCE_STATE_SUCCEEDED   NativeSourceState = 2
	NativeSourceState_NATIVE_SOURCE_STATE_FAILED      NativeSourceState = 3
	NativeSourceState_NATIVE_SOURCE_STATE_CANCELED    NativeSourceState = 4
)

// Enum value maps for NativeSourceState.
var (
	NativeSourceState_name = map[int32]string{
		0: "NATIVE_SOURCE_STATE_UNSPECIFIED",
		1: "NATIVE_SOURCE_STATE_RESOLVED",
		2: "NATIVE_SOURCE_STATE_SUCCEEDED",
		3: "NATIVE_SOURCE_STATE_FAILED",
		4: "NATIVE_SOURCE_STATE_CANCELED",
	}
	NativeSourceState_value = map[string]int32{
		"NATIVE_SOURCE_STATE_UNSPECIFIED": 0,
		"NATIVE_SOURCE_STATE_RESOLVED":    1,
		"NATIVE_SOURCE_STATE_SUCCEEDED":   2,
		"NATIVE_SOURCE_STATE_FAILED":      3,
		"NATIVE_SOURCE_STATE_CANCELED":    4,
	}
)

func (x NativeSourceState) Enum() *NativeSourceState {
	p := new(NativeSourceState)
	*p = x
	return p
}

func (x NativeSourceState) String() string {
	return protoimpl.X.EnumStringOf(x.Descriptor(), protoreflect.EnumNumber(x))
}

func (NativeSourceState) Descriptor() protoreflect.EnumDescriptor {
	return file_cozy_worker_v1_worker_proto_enumTypes[6].Descriptor()
}

func (NativeSourceState) Type() protoreflect.EnumType {
	return &file_cozy_worker_v1_worker_proto_enumTypes[6]
}

func (x NativeSourceState) Number() protoreflect.EnumNumber {
	return protoreflect.EnumNumber(x)
}

// Deprecated: Use NativeSourceState.Descriptor instead.
func (NativeSourceState) EnumDescriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{6}
}

type Posture int32

const (
	Posture_POSTURE_UNSPECIFIED Posture = 0
	Posture_POSTURE_ACCEPTING   Posture = 1
	Posture_POSTURE_DRAINING    Posture = 2
)

// Enum value maps for Posture.
var (
	Posture_name = map[int32]string{
		0: "POSTURE_UNSPECIFIED",
		1: "POSTURE_ACCEPTING",
		2: "POSTURE_DRAINING",
	}
	Posture_value = map[string]int32{
		"POSTURE_UNSPECIFIED": 0,
		"POSTURE_ACCEPTING":   1,
		"POSTURE_DRAINING":    2,
	}
)

func (x Posture) Enum() *Posture {
	p := new(Posture)
	*p = x
	return p
}

func (x Posture) String() string {
	return protoimpl.X.EnumStringOf(x.Descriptor(), protoreflect.EnumNumber(x))
}

func (Posture) Descriptor() protoreflect.EnumDescriptor {
	return file_cozy_worker_v1_worker_proto_enumTypes[7].Descriptor()
}

func (Posture) Type() protoreflect.EnumType {
	return &file_cozy_worker_v1_worker_proto_enumTypes[7]
}

func (x Posture) Number() protoreflect.EnumNumber {
	return protoreflect.EnumNumber(x)
}

// Deprecated: Use Posture.Descriptor instead.
func (Posture) EnumDescriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{7}
}

// Machine lifecycle, pulled out of the placement enum (#482).
type WorkerPhase int32

const (
	WorkerPhase_WORKER_PHASE_UNSPECIFIED WorkerPhase = 0
	WorkerPhase_WORKER_PHASE_BOOTING     WorkerPhase = 1
	WorkerPhase_WORKER_PHASE_ONLINE      WorkerPhase = 2
	WorkerPhase_WORKER_PHASE_DRAINING    WorkerPhase = 3
	WorkerPhase_WORKER_PHASE_FAILED      WorkerPhase = 4
)

// Enum value maps for WorkerPhase.
var (
	WorkerPhase_name = map[int32]string{
		0: "WORKER_PHASE_UNSPECIFIED",
		1: "WORKER_PHASE_BOOTING",
		2: "WORKER_PHASE_ONLINE",
		3: "WORKER_PHASE_DRAINING",
		4: "WORKER_PHASE_FAILED",
	}
	WorkerPhase_value = map[string]int32{
		"WORKER_PHASE_UNSPECIFIED": 0,
		"WORKER_PHASE_BOOTING":     1,
		"WORKER_PHASE_ONLINE":      2,
		"WORKER_PHASE_DRAINING":    3,
		"WORKER_PHASE_FAILED":      4,
	}
)

func (x WorkerPhase) Enum() *WorkerPhase {
	p := new(WorkerPhase)
	*p = x
	return p
}

func (x WorkerPhase) String() string {
	return protoimpl.X.EnumStringOf(x.Descriptor(), protoreflect.EnumNumber(x))
}

func (WorkerPhase) Descriptor() protoreflect.EnumDescriptor {
	return file_cozy_worker_v1_worker_proto_enumTypes[8].Descriptor()
}

func (WorkerPhase) Type() protoreflect.EnumType {
	return &file_cozy_worker_v1_worker_proto_enumTypes[8]
}

func (x WorkerPhase) Number() protoreflect.EnumNumber {
	return protoreflect.EnumNumber(x)
}

// Deprecated: Use WorkerPhase.Descriptor instead.
func (WorkerPhase) EnumDescriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{8}
}

// Axis 1 of a placement's convergence: what is on disk.
type MaterializationState int32

const (
	MaterializationState_MATERIALIZATION_STATE_UNSPECIFIED   MaterializationState = 0
	MaterializationState_MATERIALIZATION_STATE_ABSENT        MaterializationState = 1
	MaterializationState_MATERIALIZATION_STATE_MATERIALIZING MaterializationState = 2
	MaterializationState_MATERIALIZATION_STATE_STAGED        MaterializationState = 3 // asserts the installed receipt digest MATCHED
	MaterializationState_MATERIALIZATION_STATE_FAILED        MaterializationState = 4
)

// Enum value maps for MaterializationState.
var (
	MaterializationState_name = map[int32]string{
		0: "MATERIALIZATION_STATE_UNSPECIFIED",
		1: "MATERIALIZATION_STATE_ABSENT",
		2: "MATERIALIZATION_STATE_MATERIALIZING",
		3: "MATERIALIZATION_STATE_STAGED",
		4: "MATERIALIZATION_STATE_FAILED",
	}
	MaterializationState_value = map[string]int32{
		"MATERIALIZATION_STATE_UNSPECIFIED":   0,
		"MATERIALIZATION_STATE_ABSENT":        1,
		"MATERIALIZATION_STATE_MATERIALIZING": 2,
		"MATERIALIZATION_STATE_STAGED":        3,
		"MATERIALIZATION_STATE_FAILED":        4,
	}
)

func (x MaterializationState) Enum() *MaterializationState {
	p := new(MaterializationState)
	*p = x
	return p
}

func (x MaterializationState) String() string {
	return protoimpl.X.EnumStringOf(x.Descriptor(), protoreflect.EnumNumber(x))
}

func (MaterializationState) Descriptor() protoreflect.EnumDescriptor {
	return file_cozy_worker_v1_worker_proto_enumTypes[9].Descriptor()
}

func (MaterializationState) Type() protoreflect.EnumType {
	return &file_cozy_worker_v1_worker_proto_enumTypes[9]
}

func (x MaterializationState) Number() protoreflect.EnumNumber {
	return protoreflect.EnumNumber(x)
}

// Deprecated: Use MaterializationState.Descriptor instead.
func (MaterializationState) EnumDescriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{9}
}

// Axis 2 of a placement's convergence: what it will take. "prepared"/"warming"/"ready" are
// RETIRED as state words (#482) — "warm" had five meanings across these documents (page cache,
// resident weights, a live process, an accepting placement, a model that has run once) and the
// axes now name exactly one thing each.
type ServingState int32

const (
	ServingState_SERVING_STATE_UNSPECIFIED  ServingState = 0
	ServingState_SERVING_STATE_OFFLINE      ServingState = 1
	ServingState_SERVING_STATE_ACTIVATING   ServingState = 2
	ServingState_SERVING_STATE_DISPATCHABLE ServingState = 3 // executor started and package imported; models load on demand
	ServingState_SERVING_STATE_DRAINING     ServingState = 4
)

// Enum value maps for ServingState.
var (
	ServingState_name = map[int32]string{
		0: "SERVING_STATE_UNSPECIFIED",
		1: "SERVING_STATE_OFFLINE",
		2: "SERVING_STATE_ACTIVATING",
		3: "SERVING_STATE_DISPATCHABLE",
		4: "SERVING_STATE_DRAINING",
	}
	ServingState_value = map[string]int32{
		"SERVING_STATE_UNSPECIFIED":  0,
		"SERVING_STATE_OFFLINE":      1,
		"SERVING_STATE_ACTIVATING":   2,
		"SERVING_STATE_DISPATCHABLE": 3,
		"SERVING_STATE_DRAINING":     4,
	}
)

func (x ServingState) Enum() *ServingState {
	p := new(ServingState)
	*p = x
	return p
}

func (x ServingState) String() string {
	return protoimpl.X.EnumStringOf(x.Descriptor(), protoreflect.EnumNumber(x))
}

func (ServingState) Descriptor() protoreflect.EnumDescriptor {
	return file_cozy_worker_v1_worker_proto_enumTypes[10].Descriptor()
}

func (ServingState) Type() protoreflect.EnumType {
	return &file_cozy_worker_v1_worker_proto_enumTypes[10]
}

func (x ServingState) Number() protoreflect.EnumNumber {
	return protoreflect.EnumNumber(x)
}

// Deprecated: Use ServingState.Descriptor instead.
func (ServingState) EnumDescriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{10}
}

// The worker-level admission fence (§6). CLOSED is STRUCTURAL; OPEN with zero
// available_attempt_slots is TRANSIENT saturation (#486c). Both refuse under
// CAUSE_CODE_NO_CAPACITY, and a RecordOwner backs off differently for each.
type AdmissionState int32

const (
	AdmissionState_ADMISSION_STATE_UNSPECIFIED AdmissionState = 0
	AdmissionState_ADMISSION_STATE_OPEN        AdmissionState = 1
	AdmissionState_ADMISSION_STATE_CLOSED      AdmissionState = 2
)

// Enum value maps for AdmissionState.
var (
	AdmissionState_name = map[int32]string{
		0: "ADMISSION_STATE_UNSPECIFIED",
		1: "ADMISSION_STATE_OPEN",
		2: "ADMISSION_STATE_CLOSED",
	}
	AdmissionState_value = map[string]int32{
		"ADMISSION_STATE_UNSPECIFIED": 0,
		"ADMISSION_STATE_OPEN":        1,
		"ADMISSION_STATE_CLOSED":      2,
	}
)

func (x AdmissionState) Enum() *AdmissionState {
	p := new(AdmissionState)
	*p = x
	return p
}

func (x AdmissionState) String() string {
	return protoimpl.X.EnumStringOf(x.Descriptor(), protoreflect.EnumNumber(x))
}

func (AdmissionState) Descriptor() protoreflect.EnumDescriptor {
	return file_cozy_worker_v1_worker_proto_enumTypes[11].Descriptor()
}

func (AdmissionState) Type() protoreflect.EnumType {
	return &file_cozy_worker_v1_worker_proto_enumTypes[11]
}

func (x AdmissionState) Number() protoreflect.EnumNumber {
	return protoreflect.EnumNumber(x)
}

// Deprecated: Use AdmissionState.Descriptor instead.
func (AdmissionState) EnumDescriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{11}
}

type AttemptKind int32

const (
	AttemptKind_ATTEMPT_KIND_UNSPECIFIED AttemptKind = 0
	AttemptKind_ATTEMPT_KIND_SERVING     AttemptKind = 1
	AttemptKind_ATTEMPT_KIND_JOB         AttemptKind = 2
)

// Enum value maps for AttemptKind.
var (
	AttemptKind_name = map[int32]string{
		0: "ATTEMPT_KIND_UNSPECIFIED",
		1: "ATTEMPT_KIND_SERVING",
		2: "ATTEMPT_KIND_JOB",
	}
	AttemptKind_value = map[string]int32{
		"ATTEMPT_KIND_UNSPECIFIED": 0,
		"ATTEMPT_KIND_SERVING":     1,
		"ATTEMPT_KIND_JOB":         2,
	}
)

func (x AttemptKind) Enum() *AttemptKind {
	p := new(AttemptKind)
	*p = x
	return p
}

func (x AttemptKind) String() string {
	return protoimpl.X.EnumStringOf(x.Descriptor(), protoreflect.EnumNumber(x))
}

func (AttemptKind) Descriptor() protoreflect.EnumDescriptor {
	return file_cozy_worker_v1_worker_proto_enumTypes[12].Descriptor()
}

func (AttemptKind) Type() protoreflect.EnumType {
	return &file_cozy_worker_v1_worker_proto_enumTypes[12]
}

func (x AttemptKind) Number() protoreflect.EnumNumber {
	return protoreflect.EnumNumber(x)
}

// Deprecated: Use AttemptKind.Descriptor instead.
func (AttemptKind) EnumDescriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{12}
}

// The lane order (proto-026, gpu-hot.md §3): QUEUED -> RUNNING -> DEVICE_RELEASED ->
// OUTCOME_PENDING_ACK. Numbers are the normative part; they are not in phase order.
type AttemptState int32

const (
	AttemptState_ATTEMPT_STATE_UNSPECIFIED AttemptState = 0
	AttemptState_ATTEMPT_STATE_RUNNING     AttemptState = 1 // the DEVICE phase: plan priced, executor bound,
	// on the device — THE seat of its lane
	AttemptState_ATTEMPT_STATE_OUTCOME_PENDING_ACK AttemptState = 2 // was TERMINAL_PENDING_ACK
	// #507d: the attempt is HELD but its journal record is not durable. The frozen wire could
	// only say this as RUNNING + a LOCAL_SAFETY fault compound — honest, but compound.
	AttemptState_ATTEMPT_STATE_HELD_UNDURABLE AttemptState = 3
	AttemptState_ATTEMPT_STATE_QUEUED         AttemptState = 4 // accepted into the lane queue, hydrated, bound to
	// no executor (executor_epoch 0, plan_digest empty)
	AttemptState_ATTEMPT_STATE_DEVICE_RELEASED AttemptState = 5 // the device phase is over (executor replied, bytes
)

// Enum value maps for AttemptState.
var (
	AttemptState_name = map[int32]string{
		0: "ATTEMPT_STATE_UNSPECIFIED",
		1: "ATTEMPT_STATE_RUNNING",
		2: "ATTEMPT_STATE_OUTCOME_PENDING_ACK",
		3: "ATTEMPT_STATE_HELD_UNDURABLE",
		4: "ATTEMPT_STATE_QUEUED",
		5: "ATTEMPT_STATE_DEVICE_RELEASED",
	}
	AttemptState_value = map[string]int32{
		"ATTEMPT_STATE_UNSPECIFIED":         0,
		"ATTEMPT_STATE_RUNNING":             1,
		"ATTEMPT_STATE_OUTCOME_PENDING_ACK": 2,
		"ATTEMPT_STATE_HELD_UNDURABLE":      3,
		"ATTEMPT_STATE_QUEUED":              4,
		"ATTEMPT_STATE_DEVICE_RELEASED":     5,
	}
)

func (x AttemptState) Enum() *AttemptState {
	p := new(AttemptState)
	*p = x
	return p
}

func (x AttemptState) String() string {
	return protoimpl.X.EnumStringOf(x.Descriptor(), protoreflect.EnumNumber(x))
}

func (AttemptState) Descriptor() protoreflect.EnumDescriptor {
	return file_cozy_worker_v1_worker_proto_enumTypes[13].Descriptor()
}

func (AttemptState) Type() protoreflect.EnumType {
	return &file_cozy_worker_v1_worker_proto_enumTypes[13]
}

func (x AttemptState) Number() protoreflect.EnumNumber {
	return protoreflect.EnumNumber(x)
}

// Deprecated: Use AttemptState.Descriptor instead.
func (AttemptState) EnumDescriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{13}
}

type OutcomeStatus int32

const (
	OutcomeStatus_OUTCOME_STATUS_UNSPECIFIED OutcomeStatus = 0
	OutcomeStatus_OUTCOME_STATUS_SUCCEEDED   OutcomeStatus = 1
	OutcomeStatus_OUTCOME_STATUS_REFUSED     OutcomeStatus = 2
	OutcomeStatus_OUTCOME_STATUS_FAILED      OutcomeStatus = 3
	OutcomeStatus_OUTCOME_STATUS_CANCELED    OutcomeStatus = 4
	OutcomeStatus_OUTCOME_STATUS_ABANDONED   OutcomeStatus = 5
)

// Enum value maps for OutcomeStatus.
var (
	OutcomeStatus_name = map[int32]string{
		0: "OUTCOME_STATUS_UNSPECIFIED",
		1: "OUTCOME_STATUS_SUCCEEDED",
		2: "OUTCOME_STATUS_REFUSED",
		3: "OUTCOME_STATUS_FAILED",
		4: "OUTCOME_STATUS_CANCELED",
		5: "OUTCOME_STATUS_ABANDONED",
	}
	OutcomeStatus_value = map[string]int32{
		"OUTCOME_STATUS_UNSPECIFIED": 0,
		"OUTCOME_STATUS_SUCCEEDED":   1,
		"OUTCOME_STATUS_REFUSED":     2,
		"OUTCOME_STATUS_FAILED":      3,
		"OUTCOME_STATUS_CANCELED":    4,
		"OUTCOME_STATUS_ABANDONED":   5,
	}
)

func (x OutcomeStatus) Enum() *OutcomeStatus {
	p := new(OutcomeStatus)
	*p = x
	return p
}

func (x OutcomeStatus) String() string {
	return protoimpl.X.EnumStringOf(x.Descriptor(), protoreflect.EnumNumber(x))
}

func (OutcomeStatus) Descriptor() protoreflect.EnumDescriptor {
	return file_cozy_worker_v1_worker_proto_enumTypes[14].Descriptor()
}

func (OutcomeStatus) Type() protoreflect.EnumType {
	return &file_cozy_worker_v1_worker_proto_enumTypes[14]
}

func (x OutcomeStatus) Number() protoreflect.EnumNumber {
	return protoreflect.EnumNumber(x)
}

// Deprecated: Use OutcomeStatus.Descriptor instead.
func (OutcomeStatus) EnumDescriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{14}
}

// Retryability is a RecordOwner PROJECTION over (status, cause, origin), never a wire fact. The
// REFUSED projection SPLITS by cause (#480b): author/runtime causes settle refused and are never
// re-dispatched; WORKER pre-execution causes consume the ordinal, leave the budget
// untouched, and are immediately re-dispatchable as attempt_ordinal + 1 elsewhere.
type CauseCode int32

const (
	CauseCode_CAUSE_CODE_UNSPECIFIED CauseCode = 0
	// REFUSED — a judgment about the WORK (origin author/runtime): settle refused
	CauseCode_CAUSE_CODE_INVALID_REQUEST       CauseCode = 1
	CauseCode_CAUSE_CODE_UNSUPPORTED_INPUT     CauseCode = 2
	CauseCode_CAUSE_CODE_LOCAL_SAFETY          CauseCode = 3
	CauseCode_CAUSE_CODE_PROTOCOL              CauseCode = 4
	CauseCode_CAUSE_CODE_CONSTRAINT_INFEASIBLE CauseCode = 5
	// FAILED
	CauseCode_CAUSE_CODE_AUTHOR_EXCEPTION       CauseCode = 6
	CauseCode_CAUSE_CODE_EXECUTOR_FAULT         CauseCode = 7
	CauseCode_CAUSE_CODE_GRANT_EXPIRED          CauseCode = 8
	CauseCode_CAUSE_CODE_ARTIFACT_UNFETCHABLE   CauseCode = 9
	CauseCode_CAUSE_CODE_CAPABILITY_UNAVAILABLE CauseCode = 10
	// CANCELED
	CauseCode_CAUSE_CODE_CLIENT_CANCEL     CauseCode = 11
	CauseCode_CAUSE_CODE_DEADLINE_EXPIRED  CauseCode = 12
	CauseCode_CAUSE_CODE_DRAIN_CANCEL      CauseCode = 13
	CauseCode_CAUSE_CODE_POLICY_CANCEL     CauseCode = 14
	CauseCode_CAUSE_CODE_SUPERSEDED_CANCEL CauseCode = 15
	// ABANDONED
	CauseCode_CAUSE_CODE_EXECUTOR_INVALIDATED CauseCode = 16
	// REFUSED — pre-execution, origin WORKER (rev-2 §6/§7). Zero billed execution budget;
	// execution_started is false; the ordinal is consumed and the next one may go elsewhere now.
	CauseCode_CAUSE_CODE_NO_CAPACITY                CauseCode = 17 // admission CLOSED, or OPEN with no free slot
	CauseCode_CAUSE_CODE_ADMISSION_EPOCH_STALE      CauseCode = 18 // the echoed epoch is not current
	CauseCode_CAUSE_CODE_UNKNOWN_PLACEMENT          CauseCode = 19
	CauseCode_CAUSE_CODE_PLACEMENT_NOT_DISPATCHABLE CauseCode = 20
)

// Enum value maps for CauseCode.
var (
	CauseCode_name = map[int32]string{
		0:  "CAUSE_CODE_UNSPECIFIED",
		1:  "CAUSE_CODE_INVALID_REQUEST",
		2:  "CAUSE_CODE_UNSUPPORTED_INPUT",
		3:  "CAUSE_CODE_LOCAL_SAFETY",
		4:  "CAUSE_CODE_PROTOCOL",
		5:  "CAUSE_CODE_CONSTRAINT_INFEASIBLE",
		6:  "CAUSE_CODE_AUTHOR_EXCEPTION",
		7:  "CAUSE_CODE_EXECUTOR_FAULT",
		8:  "CAUSE_CODE_GRANT_EXPIRED",
		9:  "CAUSE_CODE_ARTIFACT_UNFETCHABLE",
		10: "CAUSE_CODE_CAPABILITY_UNAVAILABLE",
		11: "CAUSE_CODE_CLIENT_CANCEL",
		12: "CAUSE_CODE_DEADLINE_EXPIRED",
		13: "CAUSE_CODE_DRAIN_CANCEL",
		14: "CAUSE_CODE_POLICY_CANCEL",
		15: "CAUSE_CODE_SUPERSEDED_CANCEL",
		16: "CAUSE_CODE_EXECUTOR_INVALIDATED",
		17: "CAUSE_CODE_NO_CAPACITY",
		18: "CAUSE_CODE_ADMISSION_EPOCH_STALE",
		19: "CAUSE_CODE_UNKNOWN_PLACEMENT",
		20: "CAUSE_CODE_PLACEMENT_NOT_DISPATCHABLE",
	}
	CauseCode_value = map[string]int32{
		"CAUSE_CODE_UNSPECIFIED":                0,
		"CAUSE_CODE_INVALID_REQUEST":            1,
		"CAUSE_CODE_UNSUPPORTED_INPUT":          2,
		"CAUSE_CODE_LOCAL_SAFETY":               3,
		"CAUSE_CODE_PROTOCOL":                   4,
		"CAUSE_CODE_CONSTRAINT_INFEASIBLE":      5,
		"CAUSE_CODE_AUTHOR_EXCEPTION":           6,
		"CAUSE_CODE_EXECUTOR_FAULT":             7,
		"CAUSE_CODE_GRANT_EXPIRED":              8,
		"CAUSE_CODE_ARTIFACT_UNFETCHABLE":       9,
		"CAUSE_CODE_CAPABILITY_UNAVAILABLE":     10,
		"CAUSE_CODE_CLIENT_CANCEL":              11,
		"CAUSE_CODE_DEADLINE_EXPIRED":           12,
		"CAUSE_CODE_DRAIN_CANCEL":               13,
		"CAUSE_CODE_POLICY_CANCEL":              14,
		"CAUSE_CODE_SUPERSEDED_CANCEL":          15,
		"CAUSE_CODE_EXECUTOR_INVALIDATED":       16,
		"CAUSE_CODE_NO_CAPACITY":                17,
		"CAUSE_CODE_ADMISSION_EPOCH_STALE":      18,
		"CAUSE_CODE_UNKNOWN_PLACEMENT":          19,
		"CAUSE_CODE_PLACEMENT_NOT_DISPATCHABLE": 20,
	}
)

func (x CauseCode) Enum() *CauseCode {
	p := new(CauseCode)
	*p = x
	return p
}

func (x CauseCode) String() string {
	return protoimpl.X.EnumStringOf(x.Descriptor(), protoreflect.EnumNumber(x))
}

func (CauseCode) Descriptor() protoreflect.EnumDescriptor {
	return file_cozy_worker_v1_worker_proto_enumTypes[15].Descriptor()
}

func (CauseCode) Type() protoreflect.EnumType {
	return &file_cozy_worker_v1_worker_proto_enumTypes[15]
}

func (x CauseCode) Number() protoreflect.EnumNumber {
	return protoreflect.EnumNumber(x)
}

// Deprecated: Use CauseCode.Descriptor instead.
func (CauseCode) EnumDescriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{15}
}

type CauseOrigin int32

const (
	CauseOrigin_CAUSE_ORIGIN_UNSPECIFIED  CauseOrigin = 0
	CauseOrigin_CAUSE_ORIGIN_AUTHOR       CauseOrigin = 1
	CauseOrigin_CAUSE_ORIGIN_RUNTIME      CauseOrigin = 2
	CauseOrigin_CAUSE_ORIGIN_EXECUTOR     CauseOrigin = 3
	CauseOrigin_CAUSE_ORIGIN_WORKER       CauseOrigin = 4
	CauseOrigin_CAUSE_ORIGIN_INFRA        CauseOrigin = 5
	CauseOrigin_CAUSE_ORIGIN_CLIENT       CauseOrigin = 6
	CauseOrigin_CAUSE_ORIGIN_RECORD_OWNER CauseOrigin = 7
)

// Enum value maps for CauseOrigin.
var (
	CauseOrigin_name = map[int32]string{
		0: "CAUSE_ORIGIN_UNSPECIFIED",
		1: "CAUSE_ORIGIN_AUTHOR",
		2: "CAUSE_ORIGIN_RUNTIME",
		3: "CAUSE_ORIGIN_EXECUTOR",
		4: "CAUSE_ORIGIN_WORKER",
		5: "CAUSE_ORIGIN_INFRA",
		6: "CAUSE_ORIGIN_CLIENT",
		7: "CAUSE_ORIGIN_RECORD_OWNER",
	}
	CauseOrigin_value = map[string]int32{
		"CAUSE_ORIGIN_UNSPECIFIED":  0,
		"CAUSE_ORIGIN_AUTHOR":       1,
		"CAUSE_ORIGIN_RUNTIME":      2,
		"CAUSE_ORIGIN_EXECUTOR":     3,
		"CAUSE_ORIGIN_WORKER":       4,
		"CAUSE_ORIGIN_INFRA":        5,
		"CAUSE_ORIGIN_CLIENT":       6,
		"CAUSE_ORIGIN_RECORD_OWNER": 7,
	}
)

func (x CauseOrigin) Enum() *CauseOrigin {
	p := new(CauseOrigin)
	*p = x
	return p
}

func (x CauseOrigin) String() string {
	return protoimpl.X.EnumStringOf(x.Descriptor(), protoreflect.EnumNumber(x))
}

func (CauseOrigin) Descriptor() protoreflect.EnumDescriptor {
	return file_cozy_worker_v1_worker_proto_enumTypes[16].Descriptor()
}

func (CauseOrigin) Type() protoreflect.EnumType {
	return &file_cozy_worker_v1_worker_proto_enumTypes[16]
}

func (x CauseOrigin) Number() protoreflect.EnumNumber {
	return protoreflect.EnumNumber(x)
}

// Deprecated: Use CauseOrigin.Descriptor instead.
func (CauseOrigin) EnumDescriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{16}
}

type CancelReason int32

const (
	CancelReason_CANCEL_REASON_UNSPECIFIED CancelReason = 0
	CancelReason_CANCEL_REASON_CLIENT      CancelReason = 1
	CancelReason_CANCEL_REASON_DRAIN       CancelReason = 2
	CancelReason_CANCEL_REASON_SUPERSEDED  CancelReason = 3
	CancelReason_CANCEL_REASON_POLICY      CancelReason = 4
	CancelReason_CANCEL_REASON_DEADLINE    CancelReason = 5
)

// Enum value maps for CancelReason.
var (
	CancelReason_name = map[int32]string{
		0: "CANCEL_REASON_UNSPECIFIED",
		1: "CANCEL_REASON_CLIENT",
		2: "CANCEL_REASON_DRAIN",
		3: "CANCEL_REASON_SUPERSEDED",
		4: "CANCEL_REASON_POLICY",
		5: "CANCEL_REASON_DEADLINE",
	}
	CancelReason_value = map[string]int32{
		"CANCEL_REASON_UNSPECIFIED": 0,
		"CANCEL_REASON_CLIENT":      1,
		"CANCEL_REASON_DRAIN":       2,
		"CANCEL_REASON_SUPERSEDED":  3,
		"CANCEL_REASON_POLICY":      4,
		"CANCEL_REASON_DEADLINE":    5,
	}
)

func (x CancelReason) Enum() *CancelReason {
	p := new(CancelReason)
	*p = x
	return p
}

func (x CancelReason) String() string {
	return protoimpl.X.EnumStringOf(x.Descriptor(), protoreflect.EnumNumber(x))
}

func (CancelReason) Descriptor() protoreflect.EnumDescriptor {
	return file_cozy_worker_v1_worker_proto_enumTypes[17].Descriptor()
}

func (CancelReason) Type() protoreflect.EnumType {
	return &file_cozy_worker_v1_worker_proto_enumTypes[17]
}

func (x CancelReason) Number() protoreflect.EnumNumber {
	return protoreflect.EnumNumber(x)
}

// Deprecated: Use CancelReason.Descriptor instead.
func (CancelReason) EnumDescriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{17}
}

type ClaimRejection int32

const (
	ClaimRejection_CLAIM_REJECTION_UNSPECIFIED              ClaimRejection = 0
	ClaimRejection_CLAIM_REJECTION_UNAUTHENTICATED          ClaimRejection = 1 // proof failure; checked before readiness/state
	ClaimRejection_CLAIM_REJECTION_STALE_RECORD_OWNER_EPOCH ClaimRejection = 2 // epoch older than the accepted claim
	ClaimRejection_CLAIM_REJECTION_EPOCH_HELD               ClaimRejection = 3 // equal epoch, different record_owner_id
	ClaimRejection_CLAIM_REJECTION_WORKER_ID_MISMATCH       ClaimRejection = 4
	ClaimRejection_CLAIM_REJECTION_RELEASE_ID_MISMATCH      ClaimRejection = 5
	// The worker cannot yet promise durable ownership: either its journal is unavailable or its
	// post-bind readiness barrier is still closed. In both cases no ownership state changes and an
	// authenticated caller may retry. BootFailure(DISK_SHAPE) remains the boot-fatal form.
	ClaimRejection_CLAIM_REJECTION_UNDURABLE ClaimRejection = 7
)

// Enum value maps for ClaimRejection.
var (
	ClaimRejection_name = map[int32]string{
		0: "CLAIM_REJECTION_UNSPECIFIED",
		1: "CLAIM_REJECTION_UNAUTHENTICATED",
		2: "CLAIM_REJECTION_STALE_RECORD_OWNER_EPOCH",
		3: "CLAIM_REJECTION_EPOCH_HELD",
		4: "CLAIM_REJECTION_WORKER_ID_MISMATCH",
		5: "CLAIM_REJECTION_RELEASE_ID_MISMATCH",
		7: "CLAIM_REJECTION_UNDURABLE",
	}
	ClaimRejection_value = map[string]int32{
		"CLAIM_REJECTION_UNSPECIFIED":              0,
		"CLAIM_REJECTION_UNAUTHENTICATED":          1,
		"CLAIM_REJECTION_STALE_RECORD_OWNER_EPOCH": 2,
		"CLAIM_REJECTION_EPOCH_HELD":               3,
		"CLAIM_REJECTION_WORKER_ID_MISMATCH":       4,
		"CLAIM_REJECTION_RELEASE_ID_MISMATCH":      5,
		"CLAIM_REJECTION_UNDURABLE":                7,
	}
)

func (x ClaimRejection) Enum() *ClaimRejection {
	p := new(ClaimRejection)
	*p = x
	return p
}

func (x ClaimRejection) String() string {
	return protoimpl.X.EnumStringOf(x.Descriptor(), protoreflect.EnumNumber(x))
}

func (ClaimRejection) Descriptor() protoreflect.EnumDescriptor {
	return file_cozy_worker_v1_worker_proto_enumTypes[18].Descriptor()
}

func (ClaimRejection) Type() protoreflect.EnumType {
	return &file_cozy_worker_v1_worker_proto_enumTypes[18]
}

func (x ClaimRejection) Number() protoreflect.EnumNumber {
	return protoreflect.EnumNumber(x)
}

// Deprecated: Use ClaimRejection.Descriptor instead.
func (ClaimRejection) EnumDescriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{18}
}

type FaultKind int32

const (
	FaultKind_FAULT_KIND_UNSPECIFIED         FaultKind = 0
	FaultKind_FAULT_KIND_BINDING_UNAVAILABLE FaultKind = 1
	FaultKind_FAULT_KIND_BINDING_DEGRADED    FaultKind = 2
	FaultKind_FAULT_KIND_HARDWARE_UNSUITABLE FaultKind = 3 // this worker's actual platform cannot install or
	// run the selected package; the previous spec
	// keeps serving
	FaultKind_FAULT_KIND_ARTIFACT_FETCH_FAILED     FaultKind = 4
	FaultKind_FAULT_KIND_CREDENTIAL_UNAPPLIED      FaultKind = 6
	FaultKind_FAULT_KIND_LOCAL_SAFETY_REFUSAL      FaultKind = 7
	FaultKind_FAULT_KIND_EXECUTOR_POISONED         FaultKind = 8
	FaultKind_FAULT_KIND_CONFIG_REFUSED            FaultKind = 9
	FaultKind_FAULT_KIND_UNKNOWN_PLACEMENT         FaultKind = 10
	FaultKind_FAULT_KIND_PLACEMENT_SET_UNSUPPORTED FaultKind = 11 // an unsupportable set (two placements for one package);
	// the desired state is UNAPPLIED
	FaultKind_FAULT_KIND_PLACEMENT_SET_DIGEST_MISMATCH FaultKind = 12 // recompute over
	// placement_set_canonical_bytes disagreed with
	// placement_set_digest; UNAPPLIED, before any
	// field was parsed (§4)
	FaultKind_FAULT_KIND_ARTIFACT_DIGEST_MISMATCH FaultKind = 14 // fetched bytes whose digest is not the named
	// subject's; discarded, never installed
	FaultKind_FAULT_KIND_FALLBACK_PIN_MISSING FaultKind = 15 // #485c: the rollback pin is gone (GC'd or never
)

// Enum value maps for FaultKind.
var (
	FaultKind_name = map[int32]string{
		0:  "FAULT_KIND_UNSPECIFIED",
		1:  "FAULT_KIND_BINDING_UNAVAILABLE",
		2:  "FAULT_KIND_BINDING_DEGRADED",
		3:  "FAULT_KIND_HARDWARE_UNSUITABLE",
		4:  "FAULT_KIND_ARTIFACT_FETCH_FAILED",
		6:  "FAULT_KIND_CREDENTIAL_UNAPPLIED",
		7:  "FAULT_KIND_LOCAL_SAFETY_REFUSAL",
		8:  "FAULT_KIND_EXECUTOR_POISONED",
		9:  "FAULT_KIND_CONFIG_REFUSED",
		10: "FAULT_KIND_UNKNOWN_PLACEMENT",
		11: "FAULT_KIND_PLACEMENT_SET_UNSUPPORTED",
		12: "FAULT_KIND_PLACEMENT_SET_DIGEST_MISMATCH",
		14: "FAULT_KIND_ARTIFACT_DIGEST_MISMATCH",
		15: "FAULT_KIND_FALLBACK_PIN_MISSING",
	}
	FaultKind_value = map[string]int32{
		"FAULT_KIND_UNSPECIFIED":                   0,
		"FAULT_KIND_BINDING_UNAVAILABLE":           1,
		"FAULT_KIND_BINDING_DEGRADED":              2,
		"FAULT_KIND_HARDWARE_UNSUITABLE":           3,
		"FAULT_KIND_ARTIFACT_FETCH_FAILED":         4,
		"FAULT_KIND_CREDENTIAL_UNAPPLIED":          6,
		"FAULT_KIND_LOCAL_SAFETY_REFUSAL":          7,
		"FAULT_KIND_EXECUTOR_POISONED":             8,
		"FAULT_KIND_CONFIG_REFUSED":                9,
		"FAULT_KIND_UNKNOWN_PLACEMENT":             10,
		"FAULT_KIND_PLACEMENT_SET_UNSUPPORTED":     11,
		"FAULT_KIND_PLACEMENT_SET_DIGEST_MISMATCH": 12,
		"FAULT_KIND_ARTIFACT_DIGEST_MISMATCH":      14,
		"FAULT_KIND_FALLBACK_PIN_MISSING":          15,
	}
)

func (x FaultKind) Enum() *FaultKind {
	p := new(FaultKind)
	*p = x
	return p
}

func (x FaultKind) String() string {
	return protoimpl.X.EnumStringOf(x.Descriptor(), protoreflect.EnumNumber(x))
}

func (FaultKind) Descriptor() protoreflect.EnumDescriptor {
	return file_cozy_worker_v1_worker_proto_enumTypes[19].Descriptor()
}

func (FaultKind) Type() protoreflect.EnumType {
	return &file_cozy_worker_v1_worker_proto_enumTypes[19]
}

func (x FaultKind) Number() protoreflect.EnumNumber {
	return protoreflect.EnumNumber(x)
}

// Deprecated: Use FaultKind.Descriptor instead.
func (FaultKind) EnumDescriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{19}
}

type BootFailureReason int32

const (
	BootFailureReason_BOOT_FAILURE_REASON_UNSPECIFIED      BootFailureReason = 0
	BootFailureReason_BOOT_FAILURE_REASON_HARDWARE_VERDICT BootFailureReason = 1
	BootFailureReason_BOOT_FAILURE_REASON_ARTIFACT_FAULT   BootFailureReason = 2 // was IMAGE_FAULT; a native worker has no image
	BootFailureReason_BOOT_FAILURE_REASON_DRIVER_FAULT     BootFailureReason = 3
	BootFailureReason_BOOT_FAILURE_REASON_DISK_SHAPE       BootFailureReason = 4
	BootFailureReason_BOOT_FAILURE_REASON_CONFIG_INVALID   BootFailureReason = 5
	BootFailureReason_BOOT_FAILURE_REASON_OTHER_FATAL      BootFailureReason = 6
)

// Enum value maps for BootFailureReason.
var (
	BootFailureReason_name = map[int32]string{
		0: "BOOT_FAILURE_REASON_UNSPECIFIED",
		1: "BOOT_FAILURE_REASON_HARDWARE_VERDICT",
		2: "BOOT_FAILURE_REASON_ARTIFACT_FAULT",
		3: "BOOT_FAILURE_REASON_DRIVER_FAULT",
		4: "BOOT_FAILURE_REASON_DISK_SHAPE",
		5: "BOOT_FAILURE_REASON_CONFIG_INVALID",
		6: "BOOT_FAILURE_REASON_OTHER_FATAL",
	}
	BootFailureReason_value = map[string]int32{
		"BOOT_FAILURE_REASON_UNSPECIFIED":      0,
		"BOOT_FAILURE_REASON_HARDWARE_VERDICT": 1,
		"BOOT_FAILURE_REASON_ARTIFACT_FAULT":   2,
		"BOOT_FAILURE_REASON_DRIVER_FAULT":     3,
		"BOOT_FAILURE_REASON_DISK_SHAPE":       4,
		"BOOT_FAILURE_REASON_CONFIG_INVALID":   5,
		"BOOT_FAILURE_REASON_OTHER_FATAL":      6,
	}
)

func (x BootFailureReason) Enum() *BootFailureReason {
	p := new(BootFailureReason)
	*p = x
	return p
}

func (x BootFailureReason) String() string {
	return protoimpl.X.EnumStringOf(x.Descriptor(), protoreflect.EnumNumber(x))
}

func (BootFailureReason) Descriptor() protoreflect.EnumDescriptor {
	return file_cozy_worker_v1_worker_proto_enumTypes[20].Descriptor()
}

func (BootFailureReason) Type() protoreflect.EnumType {
	return &file_cozy_worker_v1_worker_proto_enumTypes[20]
}

func (x BootFailureReason) Number() protoreflect.EnumNumber {
	return protoreflect.EnumNumber(x)
}

// Deprecated: Use BootFailureReason.Descriptor instead.
func (BootFailureReason) EnumDescriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{20}
}

type CheckpointFaultCode int32

const (
	CheckpointFaultCode_CHECKPOINT_FAULT_CODE_UNSPECIFIED       CheckpointFaultCode = 0
	CheckpointFaultCode_CHECKPOINT_FAULT_CODE_IDENTITY_CONFLICT CheckpointFaultCode = 1
	CheckpointFaultCode_CHECKPOINT_FAULT_CODE_UNKNOWN_ATTEMPT   CheckpointFaultCode = 2
	CheckpointFaultCode_CHECKPOINT_FAULT_CODE_NOT_JOB_MODE      CheckpointFaultCode = 3
	CheckpointFaultCode_CHECKPOINT_FAULT_CODE_STALE_SEQUENCE    CheckpointFaultCode = 4
	CheckpointFaultCode_CHECKPOINT_FAULT_CODE_QUOTA_EXCEEDED    CheckpointFaultCode = 5
)

// Enum value maps for CheckpointFaultCode.
var (
	CheckpointFaultCode_name = map[int32]string{
		0: "CHECKPOINT_FAULT_CODE_UNSPECIFIED",
		1: "CHECKPOINT_FAULT_CODE_IDENTITY_CONFLICT",
		2: "CHECKPOINT_FAULT_CODE_UNKNOWN_ATTEMPT",
		3: "CHECKPOINT_FAULT_CODE_NOT_JOB_MODE",
		4: "CHECKPOINT_FAULT_CODE_STALE_SEQUENCE",
		5: "CHECKPOINT_FAULT_CODE_QUOTA_EXCEEDED",
	}
	CheckpointFaultCode_value = map[string]int32{
		"CHECKPOINT_FAULT_CODE_UNSPECIFIED":       0,
		"CHECKPOINT_FAULT_CODE_IDENTITY_CONFLICT": 1,
		"CHECKPOINT_FAULT_CODE_UNKNOWN_ATTEMPT":   2,
		"CHECKPOINT_FAULT_CODE_NOT_JOB_MODE":      3,
		"CHECKPOINT_FAULT_CODE_STALE_SEQUENCE":    4,
		"CHECKPOINT_FAULT_CODE_QUOTA_EXCEEDED":    5,
	}
)

func (x CheckpointFaultCode) Enum() *CheckpointFaultCode {
	p := new(CheckpointFaultCode)
	*p = x
	return p
}

func (x CheckpointFaultCode) String() string {
	return protoimpl.X.EnumStringOf(x.Descriptor(), protoreflect.EnumNumber(x))
}

func (CheckpointFaultCode) Descriptor() protoreflect.EnumDescriptor {
	return file_cozy_worker_v1_worker_proto_enumTypes[21].Descriptor()
}

func (CheckpointFaultCode) Type() protoreflect.EnumType {
	return &file_cozy_worker_v1_worker_proto_enumTypes[21]
}

func (x CheckpointFaultCode) Number() protoreflect.EnumNumber {
	return protoreflect.EnumNumber(x)
}

// Deprecated: Use CheckpointFaultCode.Descriptor instead.
func (CheckpointFaultCode) EnumDescriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{21}
}

type PrepareStage int32

const (
	PrepareStage_PREPARE_STAGE_UNSPECIFIED PrepareStage = 0
	PrepareStage_PREPARE_STAGE_RESOLVED    PrepareStage = 1 // authority verified, plan bounded, intent journaled
	PrepareStage_PREPARE_STAGE_DOWNLOADING PrepareStage = 2
	PrepareStage_PREPARE_STAGE_PREPARING   PrepareStage = 3 // verified local files handed to Runtime preparation
	PrepareStage_PREPARE_STAGE_PREPARED    PrepareStage = 4 // terminal; placement_set present
	PrepareStage_PREPARE_STAGE_REFUSED     PrepareStage = 5 // terminal; safe_code present
)

// Enum value maps for PrepareStage.
var (
	PrepareStage_name = map[int32]string{
		0: "PREPARE_STAGE_UNSPECIFIED",
		1: "PREPARE_STAGE_RESOLVED",
		2: "PREPARE_STAGE_DOWNLOADING",
		3: "PREPARE_STAGE_PREPARING",
		4: "PREPARE_STAGE_PREPARED",
		5: "PREPARE_STAGE_REFUSED",
	}
	PrepareStage_value = map[string]int32{
		"PREPARE_STAGE_UNSPECIFIED": 0,
		"PREPARE_STAGE_RESOLVED":    1,
		"PREPARE_STAGE_DOWNLOADING": 2,
		"PREPARE_STAGE_PREPARING":   3,
		"PREPARE_STAGE_PREPARED":    4,
		"PREPARE_STAGE_REFUSED":     5,
	}
)

func (x PrepareStage) Enum() *PrepareStage {
	p := new(PrepareStage)
	*p = x
	return p
}

func (x PrepareStage) String() string {
	return protoimpl.X.EnumStringOf(x.Descriptor(), protoreflect.EnumNumber(x))
}

func (PrepareStage) Descriptor() protoreflect.EnumDescriptor {
	return file_cozy_worker_v1_worker_proto_enumTypes[22].Descriptor()
}

func (PrepareStage) Type() protoreflect.EnumType {
	return &file_cozy_worker_v1_worker_proto_enumTypes[22]
}

func (x PrepareStage) Number() protoreflect.EnumNumber {
	return protoreflect.EnumNumber(x)
}

// Deprecated: Use PrepareStage.Descriptor instead.
func (PrepareStage) EnumDescriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{22}
}

type WeightsHostStage int32

const (
	WeightsHostStage_WEIGHTS_HOST_STAGE_UNSPECIFIED WeightsHostStage = 0
	WeightsHostStage_WEIGHTS_HOST_STAGE_INTENT      WeightsHostStage = 1
	WeightsHostStage_WEIGHTS_HOST_STAGE_RECEIPT     WeightsHostStage = 2
	WeightsHostStage_WEIGHTS_HOST_STAGE_CHECKPOINT  WeightsHostStage = 3
)

// Enum value maps for WeightsHostStage.
var (
	WeightsHostStage_name = map[int32]string{
		0: "WEIGHTS_HOST_STAGE_UNSPECIFIED",
		1: "WEIGHTS_HOST_STAGE_INTENT",
		2: "WEIGHTS_HOST_STAGE_RECEIPT",
		3: "WEIGHTS_HOST_STAGE_CHECKPOINT",
	}
	WeightsHostStage_value = map[string]int32{
		"WEIGHTS_HOST_STAGE_UNSPECIFIED": 0,
		"WEIGHTS_HOST_STAGE_INTENT":      1,
		"WEIGHTS_HOST_STAGE_RECEIPT":     2,
		"WEIGHTS_HOST_STAGE_CHECKPOINT":  3,
	}
)

func (x WeightsHostStage) Enum() *WeightsHostStage {
	p := new(WeightsHostStage)
	*p = x
	return p
}

func (x WeightsHostStage) String() string {
	return protoimpl.X.EnumStringOf(x.Descriptor(), protoreflect.EnumNumber(x))
}

func (WeightsHostStage) Descriptor() protoreflect.EnumDescriptor {
	return file_cozy_worker_v1_worker_proto_enumTypes[23].Descriptor()
}

func (WeightsHostStage) Type() protoreflect.EnumType {
	return &file_cozy_worker_v1_worker_proto_enumTypes[23]
}

func (x WeightsHostStage) Number() protoreflect.EnumNumber {
	return protoreflect.EnumNumber(x)
}

// Deprecated: Use WeightsHostStage.Descriptor instead.
func (WeightsHostStage) EnumDescriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{23}
}

type WeightsHostOutcome int32

const (
	WeightsHostOutcome_WEIGHTS_HOST_OUTCOME_UNSPECIFIED WeightsHostOutcome = 0
	WeightsHostOutcome_WEIGHTS_HOST_OUTCOME_RECORDED    WeightsHostOutcome = 1
	WeightsHostOutcome_WEIGHTS_HOST_OUTCOME_REPLAYED    WeightsHostOutcome = 2
	WeightsHostOutcome_WEIGHTS_HOST_OUTCOME_REFUSED     WeightsHostOutcome = 3
)

// Enum value maps for WeightsHostOutcome.
var (
	WeightsHostOutcome_name = map[int32]string{
		0: "WEIGHTS_HOST_OUTCOME_UNSPECIFIED",
		1: "WEIGHTS_HOST_OUTCOME_RECORDED",
		2: "WEIGHTS_HOST_OUTCOME_REPLAYED",
		3: "WEIGHTS_HOST_OUTCOME_REFUSED",
	}
	WeightsHostOutcome_value = map[string]int32{
		"WEIGHTS_HOST_OUTCOME_UNSPECIFIED": 0,
		"WEIGHTS_HOST_OUTCOME_RECORDED":    1,
		"WEIGHTS_HOST_OUTCOME_REPLAYED":    2,
		"WEIGHTS_HOST_OUTCOME_REFUSED":     3,
	}
)

func (x WeightsHostOutcome) Enum() *WeightsHostOutcome {
	p := new(WeightsHostOutcome)
	*p = x
	return p
}

func (x WeightsHostOutcome) String() string {
	return protoimpl.X.EnumStringOf(x.Descriptor(), protoreflect.EnumNumber(x))
}

func (WeightsHostOutcome) Descriptor() protoreflect.EnumDescriptor {
	return file_cozy_worker_v1_worker_proto_enumTypes[24].Descriptor()
}

func (WeightsHostOutcome) Type() protoreflect.EnumType {
	return &file_cozy_worker_v1_worker_proto_enumTypes[24]
}

func (x WeightsHostOutcome) Number() protoreflect.EnumNumber {
	return protoreflect.EnumNumber(x)
}

// Deprecated: Use WeightsHostOutcome.Descriptor instead.
func (WeightsHostOutcome) EnumDescriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{24}
}

type WeightsHostRefusal int32

const (
	WeightsHostRefusal_WEIGHTS_HOST_REFUSAL_UNSPECIFIED       WeightsHostRefusal = 0
	WeightsHostRefusal_WEIGHTS_HOST_REFUSAL_UNKNOWN_ATTEMPT   WeightsHostRefusal = 1
	WeightsHostRefusal_WEIGHTS_HOST_REFUSAL_INTENT_CONFLICT   WeightsHostRefusal = 2
	WeightsHostRefusal_WEIGHTS_HOST_REFUSAL_STALE_WRITER      WeightsHostRefusal = 3
	WeightsHostRefusal_WEIGHTS_HOST_REFUSAL_RECEIPT_CONFLICT  WeightsHostRefusal = 4
	WeightsHostRefusal_WEIGHTS_HOST_REFUSAL_INVENTORY_INVALID WeightsHostRefusal = 5
)

// Enum value maps for WeightsHostRefusal.
var (
	WeightsHostRefusal_name = map[int32]string{
		0: "WEIGHTS_HOST_REFUSAL_UNSPECIFIED",
		1: "WEIGHTS_HOST_REFUSAL_UNKNOWN_ATTEMPT",
		2: "WEIGHTS_HOST_REFUSAL_INTENT_CONFLICT",
		3: "WEIGHTS_HOST_REFUSAL_STALE_WRITER",
		4: "WEIGHTS_HOST_REFUSAL_RECEIPT_CONFLICT",
		5: "WEIGHTS_HOST_REFUSAL_INVENTORY_INVALID",
	}
	WeightsHostRefusal_value = map[string]int32{
		"WEIGHTS_HOST_REFUSAL_UNSPECIFIED":       0,
		"WEIGHTS_HOST_REFUSAL_UNKNOWN_ATTEMPT":   1,
		"WEIGHTS_HOST_REFUSAL_INTENT_CONFLICT":   2,
		"WEIGHTS_HOST_REFUSAL_STALE_WRITER":      3,
		"WEIGHTS_HOST_REFUSAL_RECEIPT_CONFLICT":  4,
		"WEIGHTS_HOST_REFUSAL_INVENTORY_INVALID": 5,
	}
)

func (x WeightsHostRefusal) Enum() *WeightsHostRefusal {
	p := new(WeightsHostRefusal)
	*p = x
	return p
}

func (x WeightsHostRefusal) String() string {
	return protoimpl.X.EnumStringOf(x.Descriptor(), protoreflect.EnumNumber(x))
}

func (WeightsHostRefusal) Descriptor() protoreflect.EnumDescriptor {
	return file_cozy_worker_v1_worker_proto_enumTypes[25].Descriptor()
}

func (WeightsHostRefusal) Type() protoreflect.EnumType {
	return &file_cozy_worker_v1_worker_proto_enumTypes[25]
}

func (x WeightsHostRefusal) Number() protoreflect.EnumNumber {
	return protoreflect.EnumNumber(x)
}

// Deprecated: Use WeightsHostRefusal.Descriptor instead.
func (WeightsHostRefusal) EnumDescriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{25}
}

type WeightsTransactionState int32

const (
	WeightsTransactionState_WEIGHTS_TRANSACTION_STATE_UNSPECIFIED WeightsTransactionState = 0
	WeightsTransactionState_WEIGHTS_TRANSACTION_STATE_INTENT      WeightsTransactionState = 1
	WeightsTransactionState_WEIGHTS_TRANSACTION_STATE_RECEIPT     WeightsTransactionState = 2
)

// Enum value maps for WeightsTransactionState.
var (
	WeightsTransactionState_name = map[int32]string{
		0: "WEIGHTS_TRANSACTION_STATE_UNSPECIFIED",
		1: "WEIGHTS_TRANSACTION_STATE_INTENT",
		2: "WEIGHTS_TRANSACTION_STATE_RECEIPT",
	}
	WeightsTransactionState_value = map[string]int32{
		"WEIGHTS_TRANSACTION_STATE_UNSPECIFIED": 0,
		"WEIGHTS_TRANSACTION_STATE_INTENT":      1,
		"WEIGHTS_TRANSACTION_STATE_RECEIPT":     2,
	}
)

func (x WeightsTransactionState) Enum() *WeightsTransactionState {
	p := new(WeightsTransactionState)
	*p = x
	return p
}

func (x WeightsTransactionState) String() string {
	return protoimpl.X.EnumStringOf(x.Descriptor(), protoreflect.EnumNumber(x))
}

func (WeightsTransactionState) Descriptor() protoreflect.EnumDescriptor {
	return file_cozy_worker_v1_worker_proto_enumTypes[26].Descriptor()
}

func (WeightsTransactionState) Type() protoreflect.EnumType {
	return &file_cozy_worker_v1_worker_proto_enumTypes[26]
}

func (x WeightsTransactionState) Number() protoreflect.EnumNumber {
	return protoreflect.EnumNumber(x)
}

// Deprecated: Use WeightsTransactionState.Descriptor instead.
func (WeightsTransactionState) EnumDescriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{26}
}

type WeightsUploadOutcome int32

const (
	WeightsUploadOutcome_WEIGHTS_UPLOAD_OUTCOME_UNSPECIFIED     WeightsUploadOutcome = 0
	WeightsUploadOutcome_WEIGHTS_UPLOAD_OUTCOME_UPLOADED        WeightsUploadOutcome = 1
	WeightsUploadOutcome_WEIGHTS_UPLOAD_OUTCOME_ALREADY_PRESENT WeightsUploadOutcome = 2
	WeightsUploadOutcome_WEIGHTS_UPLOAD_OUTCOME_REFUSED         WeightsUploadOutcome = 3
)

// Enum value maps for WeightsUploadOutcome.
var (
	WeightsUploadOutcome_name = map[int32]string{
		0: "WEIGHTS_UPLOAD_OUTCOME_UNSPECIFIED",
		1: "WEIGHTS_UPLOAD_OUTCOME_UPLOADED",
		2: "WEIGHTS_UPLOAD_OUTCOME_ALREADY_PRESENT",
		3: "WEIGHTS_UPLOAD_OUTCOME_REFUSED",
	}
	WeightsUploadOutcome_value = map[string]int32{
		"WEIGHTS_UPLOAD_OUTCOME_UNSPECIFIED":     0,
		"WEIGHTS_UPLOAD_OUTCOME_UPLOADED":        1,
		"WEIGHTS_UPLOAD_OUTCOME_ALREADY_PRESENT": 2,
		"WEIGHTS_UPLOAD_OUTCOME_REFUSED":         3,
	}
)

func (x WeightsUploadOutcome) Enum() *WeightsUploadOutcome {
	p := new(WeightsUploadOutcome)
	*p = x
	return p
}

func (x WeightsUploadOutcome) String() string {
	return protoimpl.X.EnumStringOf(x.Descriptor(), protoreflect.EnumNumber(x))
}

func (WeightsUploadOutcome) Descriptor() protoreflect.EnumDescriptor {
	return file_cozy_worker_v1_worker_proto_enumTypes[27].Descriptor()
}

func (WeightsUploadOutcome) Type() protoreflect.EnumType {
	return &file_cozy_worker_v1_worker_proto_enumTypes[27]
}

func (x WeightsUploadOutcome) Number() protoreflect.EnumNumber {
	return protoreflect.EnumNumber(x)
}

// Deprecated: Use WeightsUploadOutcome.Descriptor instead.
func (WeightsUploadOutcome) EnumDescriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{27}
}

type WeightsUploadRefusal int32

const (
	WeightsUploadRefusal_WEIGHTS_UPLOAD_REFUSAL_UNSPECIFIED         WeightsUploadRefusal = 0
	WeightsUploadRefusal_WEIGHTS_UPLOAD_REFUSAL_UNKNOWN_TRANSACTION WeightsUploadRefusal = 1
	WeightsUploadRefusal_WEIGHTS_UPLOAD_REFUSAL_STALE_WRITER        WeightsUploadRefusal = 2
	WeightsUploadRefusal_WEIGHTS_UPLOAD_REFUSAL_UNKNOWN_OBJECT      WeightsUploadRefusal = 3
	WeightsUploadRefusal_WEIGHTS_UPLOAD_REFUSAL_SOURCE_REF_MISMATCH WeightsUploadRefusal = 4
	WeightsUploadRefusal_WEIGHTS_UPLOAD_REFUSAL_SOURCE_UNAVAILABLE  WeightsUploadRefusal = 5
	WeightsUploadRefusal_WEIGHTS_UPLOAD_REFUSAL_GRANT_REFUSED       WeightsUploadRefusal = 6
	WeightsUploadRefusal_WEIGHTS_UPLOAD_REFUSAL_TRANSFER_FAILED     WeightsUploadRefusal = 7
)

// Enum value maps for WeightsUploadRefusal.
var (
	WeightsUploadRefusal_name = map[int32]string{
		0: "WEIGHTS_UPLOAD_REFUSAL_UNSPECIFIED",
		1: "WEIGHTS_UPLOAD_REFUSAL_UNKNOWN_TRANSACTION",
		2: "WEIGHTS_UPLOAD_REFUSAL_STALE_WRITER",
		3: "WEIGHTS_UPLOAD_REFUSAL_UNKNOWN_OBJECT",
		4: "WEIGHTS_UPLOAD_REFUSAL_SOURCE_REF_MISMATCH",
		5: "WEIGHTS_UPLOAD_REFUSAL_SOURCE_UNAVAILABLE",
		6: "WEIGHTS_UPLOAD_REFUSAL_GRANT_REFUSED",
		7: "WEIGHTS_UPLOAD_REFUSAL_TRANSFER_FAILED",
	}
	WeightsUploadRefusal_value = map[string]int32{
		"WEIGHTS_UPLOAD_REFUSAL_UNSPECIFIED":         0,
		"WEIGHTS_UPLOAD_REFUSAL_UNKNOWN_TRANSACTION": 1,
		"WEIGHTS_UPLOAD_REFUSAL_STALE_WRITER":        2,
		"WEIGHTS_UPLOAD_REFUSAL_UNKNOWN_OBJECT":      3,
		"WEIGHTS_UPLOAD_REFUSAL_SOURCE_REF_MISMATCH": 4,
		"WEIGHTS_UPLOAD_REFUSAL_SOURCE_UNAVAILABLE":  5,
		"WEIGHTS_UPLOAD_REFUSAL_GRANT_REFUSED":       6,
		"WEIGHTS_UPLOAD_REFUSAL_TRANSFER_FAILED":     7,
	}
)

func (x WeightsUploadRefusal) Enum() *WeightsUploadRefusal {
	p := new(WeightsUploadRefusal)
	*p = x
	return p
}

func (x WeightsUploadRefusal) String() string {
	return protoimpl.X.EnumStringOf(x.Descriptor(), protoreflect.EnumNumber(x))
}

func (WeightsUploadRefusal) Descriptor() protoreflect.EnumDescriptor {
	return file_cozy_worker_v1_worker_proto_enumTypes[28].Descriptor()
}

func (WeightsUploadRefusal) Type() protoreflect.EnumType {
	return &file_cozy_worker_v1_worker_proto_enumTypes[28]
}

func (x WeightsUploadRefusal) Number() protoreflect.EnumNumber {
	return protoreflect.EnumNumber(x)
}

// Deprecated: Use WeightsUploadRefusal.Descriptor instead.
func (WeightsUploadRefusal) EnumDescriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{28}
}

type WeightsTransferState int32

const (
	WeightsTransferState_WEIGHTS_TRANSFER_STATE_UNSPECIFIED     WeightsTransferState = 0
	WeightsTransferState_WEIGHTS_TRANSFER_STATE_ACCEPTED        WeightsTransferState = 1
	WeightsTransferState_WEIGHTS_TRANSFER_STATE_UPLOADING       WeightsTransferState = 2
	WeightsTransferState_WEIGHTS_TRANSFER_STATE_UPLOADED        WeightsTransferState = 3
	WeightsTransferState_WEIGHTS_TRANSFER_STATE_ALREADY_PRESENT WeightsTransferState = 4
	WeightsTransferState_WEIGHTS_TRANSFER_STATE_HELD            WeightsTransferState = 5
	WeightsTransferState_WEIGHTS_TRANSFER_STATE_FAILED          WeightsTransferState = 6
)

// Enum value maps for WeightsTransferState.
var (
	WeightsTransferState_name = map[int32]string{
		0: "WEIGHTS_TRANSFER_STATE_UNSPECIFIED",
		1: "WEIGHTS_TRANSFER_STATE_ACCEPTED",
		2: "WEIGHTS_TRANSFER_STATE_UPLOADING",
		3: "WEIGHTS_TRANSFER_STATE_UPLOADED",
		4: "WEIGHTS_TRANSFER_STATE_ALREADY_PRESENT",
		5: "WEIGHTS_TRANSFER_STATE_HELD",
		6: "WEIGHTS_TRANSFER_STATE_FAILED",
	}
	WeightsTransferState_value = map[string]int32{
		"WEIGHTS_TRANSFER_STATE_UNSPECIFIED":     0,
		"WEIGHTS_TRANSFER_STATE_ACCEPTED":        1,
		"WEIGHTS_TRANSFER_STATE_UPLOADING":       2,
		"WEIGHTS_TRANSFER_STATE_UPLOADED":        3,
		"WEIGHTS_TRANSFER_STATE_ALREADY_PRESENT": 4,
		"WEIGHTS_TRANSFER_STATE_HELD":            5,
		"WEIGHTS_TRANSFER_STATE_FAILED":          6,
	}
)

func (x WeightsTransferState) Enum() *WeightsTransferState {
	p := new(WeightsTransferState)
	*p = x
	return p
}

func (x WeightsTransferState) String() string {
	return protoimpl.X.EnumStringOf(x.Descriptor(), protoreflect.EnumNumber(x))
}

func (WeightsTransferState) Descriptor() protoreflect.EnumDescriptor {
	return file_cozy_worker_v1_worker_proto_enumTypes[29].Descriptor()
}

func (WeightsTransferState) Type() protoreflect.EnumType {
	return &file_cozy_worker_v1_worker_proto_enumTypes[29]
}

func (x WeightsTransferState) Number() protoreflect.EnumNumber {
	return protoreflect.EnumNumber(x)
}

// Deprecated: Use WeightsTransferState.Descriptor instead.
func (WeightsTransferState) EnumDescriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{29}
}

type ModelSourceProvider int32

const (
	ModelSourceProvider_MODEL_SOURCE_PROVIDER_UNSPECIFIED  ModelSourceProvider = 0
	ModelSourceProvider_MODEL_SOURCE_PROVIDER_HUGGING_FACE ModelSourceProvider = 1
	ModelSourceProvider_MODEL_SOURCE_PROVIDER_CIVITAI      ModelSourceProvider = 2
)

// Enum value maps for ModelSourceProvider.
var (
	ModelSourceProvider_name = map[int32]string{
		0: "MODEL_SOURCE_PROVIDER_UNSPECIFIED",
		1: "MODEL_SOURCE_PROVIDER_HUGGING_FACE",
		2: "MODEL_SOURCE_PROVIDER_CIVITAI",
	}
	ModelSourceProvider_value = map[string]int32{
		"MODEL_SOURCE_PROVIDER_UNSPECIFIED":  0,
		"MODEL_SOURCE_PROVIDER_HUGGING_FACE": 1,
		"MODEL_SOURCE_PROVIDER_CIVITAI":      2,
	}
)

func (x ModelSourceProvider) Enum() *ModelSourceProvider {
	p := new(ModelSourceProvider)
	*p = x
	return p
}

func (x ModelSourceProvider) String() string {
	return protoimpl.X.EnumStringOf(x.Descriptor(), protoreflect.EnumNumber(x))
}

func (ModelSourceProvider) Descriptor() protoreflect.EnumDescriptor {
	return file_cozy_worker_v1_worker_proto_enumTypes[30].Descriptor()
}

func (ModelSourceProvider) Type() protoreflect.EnumType {
	return &file_cozy_worker_v1_worker_proto_enumTypes[30]
}

func (x ModelSourceProvider) Number() protoreflect.EnumNumber {
	return protoreflect.EnumNumber(x)
}

// Deprecated: Use ModelSourceProvider.Descriptor instead.
func (ModelSourceProvider) EnumDescriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{30}
}

type ModelSourceFileState int32

const (
	ModelSourceFileState_MODEL_SOURCE_FILE_STATE_UNSPECIFIED ModelSourceFileState = 0
	ModelSourceFileState_MODEL_SOURCE_FILE_STATE_ACCEPTED    ModelSourceFileState = 1
	ModelSourceFileState_MODEL_SOURCE_FILE_STATE_DOWNLOADING ModelSourceFileState = 2
	ModelSourceFileState_MODEL_SOURCE_FILE_STATE_VERIFIED    ModelSourceFileState = 3
	ModelSourceFileState_MODEL_SOURCE_FILE_STATE_FAILED      ModelSourceFileState = 4
	// All conversion consumers are complete/recoverable. The physical carrier may have
	// been released, so this MUST NOT be used as LocalModelSourceFile.verified.
	ModelSourceFileState_MODEL_SOURCE_FILE_STATE_CONVERTED ModelSourceFileState = 5
)

// Enum value maps for ModelSourceFileState.
var (
	ModelSourceFileState_name = map[int32]string{
		0: "MODEL_SOURCE_FILE_STATE_UNSPECIFIED",
		1: "MODEL_SOURCE_FILE_STATE_ACCEPTED",
		2: "MODEL_SOURCE_FILE_STATE_DOWNLOADING",
		3: "MODEL_SOURCE_FILE_STATE_VERIFIED",
		4: "MODEL_SOURCE_FILE_STATE_FAILED",
		5: "MODEL_SOURCE_FILE_STATE_CONVERTED",
	}
	ModelSourceFileState_value = map[string]int32{
		"MODEL_SOURCE_FILE_STATE_UNSPECIFIED": 0,
		"MODEL_SOURCE_FILE_STATE_ACCEPTED":    1,
		"MODEL_SOURCE_FILE_STATE_DOWNLOADING": 2,
		"MODEL_SOURCE_FILE_STATE_VERIFIED":    3,
		"MODEL_SOURCE_FILE_STATE_FAILED":      4,
		"MODEL_SOURCE_FILE_STATE_CONVERTED":   5,
	}
)

func (x ModelSourceFileState) Enum() *ModelSourceFileState {
	p := new(ModelSourceFileState)
	*p = x
	return p
}

func (x ModelSourceFileState) String() string {
	return protoimpl.X.EnumStringOf(x.Descriptor(), protoreflect.EnumNumber(x))
}

func (ModelSourceFileState) Descriptor() protoreflect.EnumDescriptor {
	return file_cozy_worker_v1_worker_proto_enumTypes[31].Descriptor()
}

func (ModelSourceFileState) Type() protoreflect.EnumType {
	return &file_cozy_worker_v1_worker_proto_enumTypes[31]
}

func (x ModelSourceFileState) Number() protoreflect.EnumNumber {
	return protoreflect.EnumNumber(x)
}

// Deprecated: Use ModelSourceFileState.Descriptor instead.
func (ModelSourceFileState) EnumDescriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{31}
}

type ModelSourcePrepareOutcome int32

const (
	ModelSourcePrepareOutcome_MODEL_SOURCE_PREPARE_OUTCOME_UNSPECIFIED ModelSourcePrepareOutcome = 0
	ModelSourcePrepareOutcome_MODEL_SOURCE_PREPARE_OUTCOME_PREPARED    ModelSourcePrepareOutcome = 1
	ModelSourcePrepareOutcome_MODEL_SOURCE_PREPARE_OUTCOME_REPLAYED    ModelSourcePrepareOutcome = 2
	ModelSourcePrepareOutcome_MODEL_SOURCE_PREPARE_OUTCOME_REFUSED     ModelSourcePrepareOutcome = 3
	ModelSourcePrepareOutcome_MODEL_SOURCE_PREPARE_OUTCOME_INCOMPLETE  ModelSourcePrepareOutcome = 4
)

// Enum value maps for ModelSourcePrepareOutcome.
var (
	ModelSourcePrepareOutcome_name = map[int32]string{
		0: "MODEL_SOURCE_PREPARE_OUTCOME_UNSPECIFIED",
		1: "MODEL_SOURCE_PREPARE_OUTCOME_PREPARED",
		2: "MODEL_SOURCE_PREPARE_OUTCOME_REPLAYED",
		3: "MODEL_SOURCE_PREPARE_OUTCOME_REFUSED",
		4: "MODEL_SOURCE_PREPARE_OUTCOME_INCOMPLETE",
	}
	ModelSourcePrepareOutcome_value = map[string]int32{
		"MODEL_SOURCE_PREPARE_OUTCOME_UNSPECIFIED": 0,
		"MODEL_SOURCE_PREPARE_OUTCOME_PREPARED":    1,
		"MODEL_SOURCE_PREPARE_OUTCOME_REPLAYED":    2,
		"MODEL_SOURCE_PREPARE_OUTCOME_REFUSED":     3,
		"MODEL_SOURCE_PREPARE_OUTCOME_INCOMPLETE":  4,
	}
)

func (x ModelSourcePrepareOutcome) Enum() *ModelSourcePrepareOutcome {
	p := new(ModelSourcePrepareOutcome)
	*p = x
	return p
}

func (x ModelSourcePrepareOutcome) String() string {
	return protoimpl.X.EnumStringOf(x.Descriptor(), protoreflect.EnumNumber(x))
}

func (ModelSourcePrepareOutcome) Descriptor() protoreflect.EnumDescriptor {
	return file_cozy_worker_v1_worker_proto_enumTypes[32].Descriptor()
}

func (ModelSourcePrepareOutcome) Type() protoreflect.EnumType {
	return &file_cozy_worker_v1_worker_proto_enumTypes[32]
}

func (x ModelSourcePrepareOutcome) Number() protoreflect.EnumNumber {
	return protoreflect.EnumNumber(x)
}

// Deprecated: Use ModelSourcePrepareOutcome.Descriptor instead.
func (ModelSourcePrepareOutcome) EnumDescriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{32}
}

type LocalPackageFileState int32

const (
	LocalPackageFileState_LOCAL_PACKAGE_FILE_STATE_UNSPECIFIED LocalPackageFileState = 0
	LocalPackageFileState_LOCAL_PACKAGE_FILE_STATE_RECEIVING   LocalPackageFileState = 1
	LocalPackageFileState_LOCAL_PACKAGE_FILE_STATE_VERIFIED    LocalPackageFileState = 2
	LocalPackageFileState_LOCAL_PACKAGE_FILE_STATE_REFUSED     LocalPackageFileState = 3
)

// Enum value maps for LocalPackageFileState.
var (
	LocalPackageFileState_name = map[int32]string{
		0: "LOCAL_PACKAGE_FILE_STATE_UNSPECIFIED",
		1: "LOCAL_PACKAGE_FILE_STATE_RECEIVING",
		2: "LOCAL_PACKAGE_FILE_STATE_VERIFIED",
		3: "LOCAL_PACKAGE_FILE_STATE_REFUSED",
	}
	LocalPackageFileState_value = map[string]int32{
		"LOCAL_PACKAGE_FILE_STATE_UNSPECIFIED": 0,
		"LOCAL_PACKAGE_FILE_STATE_RECEIVING":   1,
		"LOCAL_PACKAGE_FILE_STATE_VERIFIED":    2,
		"LOCAL_PACKAGE_FILE_STATE_REFUSED":     3,
	}
)

func (x LocalPackageFileState) Enum() *LocalPackageFileState {
	p := new(LocalPackageFileState)
	*p = x
	return p
}

func (x LocalPackageFileState) String() string {
	return protoimpl.X.EnumStringOf(x.Descriptor(), protoreflect.EnumNumber(x))
}

func (LocalPackageFileState) Descriptor() protoreflect.EnumDescriptor {
	return file_cozy_worker_v1_worker_proto_enumTypes[33].Descriptor()
}

func (LocalPackageFileState) Type() protoreflect.EnumType {
	return &file_cozy_worker_v1_worker_proto_enumTypes[33]
}

func (x LocalPackageFileState) Number() protoreflect.EnumNumber {
	return protoreflect.EnumNumber(x)
}

// Deprecated: Use LocalPackageFileState.Descriptor instead.
func (LocalPackageFileState) EnumDescriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{33}
}

type WeightsFinalizeDisposition int32

const (
	WeightsFinalizeDisposition_WEIGHTS_FINALIZE_DISPOSITION_UNSPECIFIED         WeightsFinalizeDisposition = 0
	WeightsFinalizeDisposition_WEIGHTS_FINALIZE_DISPOSITION_ADOPT               WeightsFinalizeDisposition = 1
	WeightsFinalizeDisposition_WEIGHTS_FINALIZE_DISPOSITION_ABANDON             WeightsFinalizeDisposition = 2
	WeightsFinalizeDisposition_WEIGHTS_FINALIZE_DISPOSITION_ABANDON_UNCOMMITTED WeightsFinalizeDisposition = 3
)

// Enum value maps for WeightsFinalizeDisposition.
var (
	WeightsFinalizeDisposition_name = map[int32]string{
		0: "WEIGHTS_FINALIZE_DISPOSITION_UNSPECIFIED",
		1: "WEIGHTS_FINALIZE_DISPOSITION_ADOPT",
		2: "WEIGHTS_FINALIZE_DISPOSITION_ABANDON",
		3: "WEIGHTS_FINALIZE_DISPOSITION_ABANDON_UNCOMMITTED",
	}
	WeightsFinalizeDisposition_value = map[string]int32{
		"WEIGHTS_FINALIZE_DISPOSITION_UNSPECIFIED":         0,
		"WEIGHTS_FINALIZE_DISPOSITION_ADOPT":               1,
		"WEIGHTS_FINALIZE_DISPOSITION_ABANDON":             2,
		"WEIGHTS_FINALIZE_DISPOSITION_ABANDON_UNCOMMITTED": 3,
	}
)

func (x WeightsFinalizeDisposition) Enum() *WeightsFinalizeDisposition {
	p := new(WeightsFinalizeDisposition)
	*p = x
	return p
}

func (x WeightsFinalizeDisposition) String() string {
	return protoimpl.X.EnumStringOf(x.Descriptor(), protoreflect.EnumNumber(x))
}

func (WeightsFinalizeDisposition) Descriptor() protoreflect.EnumDescriptor {
	return file_cozy_worker_v1_worker_proto_enumTypes[34].Descriptor()
}

func (WeightsFinalizeDisposition) Type() protoreflect.EnumType {
	return &file_cozy_worker_v1_worker_proto_enumTypes[34]
}

func (x WeightsFinalizeDisposition) Number() protoreflect.EnumNumber {
	return protoreflect.EnumNumber(x)
}

// Deprecated: Use WeightsFinalizeDisposition.Descriptor instead.
func (WeightsFinalizeDisposition) EnumDescriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{34}
}

type WeightsFinalizeOutcome int32

const (
	WeightsFinalizeOutcome_WEIGHTS_FINALIZE_OUTCOME_UNSPECIFIED WeightsFinalizeOutcome = 0
	WeightsFinalizeOutcome_WEIGHTS_FINALIZE_OUTCOME_ADOPTED     WeightsFinalizeOutcome = 1
	WeightsFinalizeOutcome_WEIGHTS_FINALIZE_OUTCOME_ABANDONED   WeightsFinalizeOutcome = 2
)

// Enum value maps for WeightsFinalizeOutcome.
var (
	WeightsFinalizeOutcome_name = map[int32]string{
		0: "WEIGHTS_FINALIZE_OUTCOME_UNSPECIFIED",
		1: "WEIGHTS_FINALIZE_OUTCOME_ADOPTED",
		2: "WEIGHTS_FINALIZE_OUTCOME_ABANDONED",
	}
	WeightsFinalizeOutcome_value = map[string]int32{
		"WEIGHTS_FINALIZE_OUTCOME_UNSPECIFIED": 0,
		"WEIGHTS_FINALIZE_OUTCOME_ADOPTED":     1,
		"WEIGHTS_FINALIZE_OUTCOME_ABANDONED":   2,
	}
)

func (x WeightsFinalizeOutcome) Enum() *WeightsFinalizeOutcome {
	p := new(WeightsFinalizeOutcome)
	*p = x
	return p
}

func (x WeightsFinalizeOutcome) String() string {
	return protoimpl.X.EnumStringOf(x.Descriptor(), protoreflect.EnumNumber(x))
}

func (WeightsFinalizeOutcome) Descriptor() protoreflect.EnumDescriptor {
	return file_cozy_worker_v1_worker_proto_enumTypes[35].Descriptor()
}

func (WeightsFinalizeOutcome) Type() protoreflect.EnumType {
	return &file_cozy_worker_v1_worker_proto_enumTypes[35]
}

func (x WeightsFinalizeOutcome) Number() protoreflect.EnumNumber {
	return protoreflect.EnumNumber(x)
}

// Deprecated: Use WeightsFinalizeOutcome.Descriptor instead.
func (WeightsFinalizeOutcome) EnumDescriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{35}
}

type MachineLogQuery struct {
	state         protoimpl.MessageState `protogen:"open.v1"`
	Claim         *Claim                 `protobuf:"bytes,1,opt,name=claim,proto3" json:"claim,omitempty"`
	Log           MachineLog             `protobuf:"varint,2,opt,name=log,proto3,enum=cozy.worker.v1.MachineLog" json:"log,omitempty"`
	TailBytes     uint64                 `protobuf:"varint,3,opt,name=tail_bytes,json=tailBytes,proto3" json:"tail_bytes,omitempty"` // at most the newest this many bytes, from a line start; 0: all kept
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *MachineLogQuery) Reset() {
	*x = MachineLogQuery{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[0]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *MachineLogQuery) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*MachineLogQuery) ProtoMessage() {}

func (x *MachineLogQuery) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[0]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use MachineLogQuery.ProtoReflect.Descriptor instead.
func (*MachineLogQuery) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{0}
}

func (x *MachineLogQuery) GetClaim() *Claim {
	if x != nil {
		return x.Claim
	}
	return nil
}

func (x *MachineLogQuery) GetLog() MachineLog {
	if x != nil {
		return x.Log
	}
	return MachineLog_MACHINE_LOG_UNSPECIFIED
}

func (x *MachineLogQuery) GetTailBytes() uint64 {
	if x != nil {
		return x.TailBytes
	}
	return 0
}

// Consecutive pieces of the log, oldest first; the stream ends after the last. A log the
// machine has not written yet is an empty stream.
type MachineLogChunk struct {
	state         protoimpl.MessageState `protogen:"open.v1"`
	Data          []byte                 `protobuf:"bytes,1,opt,name=data,proto3" json:"data,omitempty"` // nonempty, <=MaxMachineLogChunkBytes
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *MachineLogChunk) Reset() {
	*x = MachineLogChunk{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[1]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *MachineLogChunk) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*MachineLogChunk) ProtoMessage() {}

func (x *MachineLogChunk) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[1]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use MachineLogChunk.ProtoReflect.Descriptor instead.
func (*MachineLogChunk) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{1}
}

func (x *MachineLogChunk) GetData() []byte {
	if x != nil {
		return x.Data
	}
	return nil
}

type ProtocolInfoRequest struct {
	state         protoimpl.MessageState `protogen:"open.v1"`
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *ProtocolInfoRequest) Reset() {
	*x = ProtocolInfoRequest{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[2]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *ProtocolInfoRequest) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*ProtocolInfoRequest) ProtoMessage() {}

func (x *ProtocolInfoRequest) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[2]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use ProtocolInfoRequest.ProtoReflect.Descriptor instead.
func (*ProtocolInfoRequest) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{2}
}

// One explicit authenticated idle reset. The Host validates the current Claim,
// rental lifetime, and a nonempty request_id of at most MaxRentalKeepaliveRequestIDBytes.
// The same owner/boot/request_id returns its original receipt without extending it.
// A new user command supplies a new request_id. There is no duration override.
type KeepRentalAliveRequest struct {
	state         protoimpl.MessageState `protogen:"open.v1"`
	Claim         *Claim                 `protobuf:"bytes,1,opt,name=claim,proto3" json:"claim,omitempty"`
	RequestId     string                 `protobuf:"bytes,2,opt,name=request_id,json=requestId,proto3" json:"request_id,omitempty"`
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *KeepRentalAliveRequest) Reset() {
	*x = KeepRentalAliveRequest{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[3]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *KeepRentalAliveRequest) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*KeepRentalAliveRequest) ProtoMessage() {}

func (x *KeepRentalAliveRequest) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[3]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use KeepRentalAliveRequest.ProtoReflect.Descriptor instead.
func (*KeepRentalAliveRequest) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{3}
}

func (x *KeepRentalAliveRequest) GetClaim() *Claim {
	if x != nil {
		return x.Claim
	}
	return nil
}

func (x *KeepRentalAliveRequest) GetRequestId() string {
	if x != nil {
		return x.RequestId
	}
	return ""
}

// Host-clock acknowledgement. idle_deadline_unix_ms is exactly
// acknowledged_at_unix_ms + RentalIdleTimeoutSeconds * 1000. Clients only
// project/persist this acknowledged deadline; lost/failed RPCs do not renew locally.
type KeepRentalAliveResult struct {
	state                protoimpl.MessageState `protogen:"open.v1"`
	RequestId            string                 `protobuf:"bytes,1,opt,name=request_id,json=requestId,proto3" json:"request_id,omitempty"`
	WorkerId             string                 `protobuf:"bytes,2,opt,name=worker_id,json=workerId,proto3" json:"worker_id,omitempty"`
	WorkerBootId         string                 `protobuf:"bytes,3,opt,name=worker_boot_id,json=workerBootId,proto3" json:"worker_boot_id,omitempty"`
	AcknowledgedAtUnixMs int64                  `protobuf:"varint,4,opt,name=acknowledged_at_unix_ms,json=acknowledgedAtUnixMs,proto3" json:"acknowledged_at_unix_ms,omitempty"`
	IdleDeadlineUnixMs   int64                  `protobuf:"varint,5,opt,name=idle_deadline_unix_ms,json=idleDeadlineUnixMs,proto3" json:"idle_deadline_unix_ms,omitempty"`
	unknownFields        protoimpl.UnknownFields
	sizeCache            protoimpl.SizeCache
}

func (x *KeepRentalAliveResult) Reset() {
	*x = KeepRentalAliveResult{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[4]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *KeepRentalAliveResult) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*KeepRentalAliveResult) ProtoMessage() {}

func (x *KeepRentalAliveResult) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[4]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use KeepRentalAliveResult.ProtoReflect.Descriptor instead.
func (*KeepRentalAliveResult) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{4}
}

func (x *KeepRentalAliveResult) GetRequestId() string {
	if x != nil {
		return x.RequestId
	}
	return ""
}

func (x *KeepRentalAliveResult) GetWorkerId() string {
	if x != nil {
		return x.WorkerId
	}
	return ""
}

func (x *KeepRentalAliveResult) GetWorkerBootId() string {
	if x != nil {
		return x.WorkerBootId
	}
	return ""
}

func (x *KeepRentalAliveResult) GetAcknowledgedAtUnixMs() int64 {
	if x != nil {
		return x.AcknowledgedAtUnixMs
	}
	return 0
}

func (x *KeepRentalAliveResult) GetIdleDeadlineUnixMs() int64 {
	if x != nil {
		return x.IdleDeadlineUnixMs
	}
	return 0
}

// DOCUMENT SHAPE: cozy.worker.v1.MachineExecutionCapture/2. Installation handles
// select retained environments, not content identities. Runtime supplies interface
// metadata after installation. No client filesystem paths or credentials belong here.
type MachineExecutionCapture struct {
	state              protoimpl.MessageState    `protogen:"open.v1"`
	Bindings           []*MachineCallableBinding `protobuf:"bytes,3,rep,name=bindings,proto3" json:"bindings,omitempty"`                                // sorted by caller/module/export
	ModelDefaults      []*MachineModelDefault    `protobuf:"bytes,5,rep,name=model_defaults,json=modelDefaults,proto3" json:"model_defaults,omitempty"` // sorted by callee/entrypoint/parameter
	RootInstallationId string                    `protobuf:"bytes,6,opt,name=root_installation_id,json=rootInstallationId,proto3" json:"root_installation_id,omitempty"`
	InstalledPackages  []*InstalledPackage       `protobuf:"bytes,7,rep,name=installed_packages,json=installedPackages,proto3" json:"installed_packages,omitempty"` // 1..128, sorted by installation_id
	// Sorted by key, <=128.
	DeferredInstallations []*DeferredInstallation `protobuf:"bytes,8,rep,name=deferred_installations,json=deferredInstallations,proto3" json:"deferred_installations,omitempty"`
	// Wire 70: explicit root/captured-callable selections, sorted by parameter. The
	// capture digest retains them across retries and descendants. Overrides name only
	// callables in this captured graph; an adapter stack keeps its authored order.
	ModelChoices  []*ModelChoice `protobuf:"bytes,10,rep,name=model_choices,json=modelChoices,proto3" json:"model_choices,omitempty"`
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *MachineExecutionCapture) Reset() {
	*x = MachineExecutionCapture{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[5]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *MachineExecutionCapture) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*MachineExecutionCapture) ProtoMessage() {}

func (x *MachineExecutionCapture) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[5]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use MachineExecutionCapture.ProtoReflect.Descriptor instead.
func (*MachineExecutionCapture) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{5}
}

func (x *MachineExecutionCapture) GetBindings() []*MachineCallableBinding {
	if x != nil {
		return x.Bindings
	}
	return nil
}

func (x *MachineExecutionCapture) GetModelDefaults() []*MachineModelDefault {
	if x != nil {
		return x.ModelDefaults
	}
	return nil
}

func (x *MachineExecutionCapture) GetRootInstallationId() string {
	if x != nil {
		return x.RootInstallationId
	}
	return ""
}

func (x *MachineExecutionCapture) GetInstalledPackages() []*InstalledPackage {
	if x != nil {
		return x.InstalledPackages
	}
	return nil
}

func (x *MachineExecutionCapture) GetDeferredInstallations() []*DeferredInstallation {
	if x != nil {
		return x.DeferredInstallations
	}
	return nil
}

func (x *MachineExecutionCapture) GetModelChoices() []*ModelChoice {
	if x != nil {
		return x.ModelChoices
	}
	return nil
}

// A published callee Runtime installs on its first selection instead of before acceptance.
// Bindings and model defaults name it by key. The installed interface is authoritative once
// prepared; preparation's own package_interface is the release's, for verifying the capture.
type DeferredInstallation struct {
	state         protoimpl.MessageState    `protogen:"open.v1"`
	Key           string                    `protobuf:"bytes,1,opt,name=key,proto3" json:"key,omitempty"` // "org/name@release", unique within the capture
	Package       string                    `protobuf:"bytes,2,opt,name=package,proto3" json:"package,omitempty"`
	Release       string                    `protobuf:"bytes,3,opt,name=release,proto3" json:"release,omitempty"`
	Preparation   *PreparePackageSetRequest `protobuf:"bytes,4,opt,name=preparation,proto3" json:"preparation,omitempty"` // this callee's download set and facts; install_root empty
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *DeferredInstallation) Reset() {
	*x = DeferredInstallation{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[6]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *DeferredInstallation) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*DeferredInstallation) ProtoMessage() {}

func (x *DeferredInstallation) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[6]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use DeferredInstallation.ProtoReflect.Descriptor instead.
func (*DeferredInstallation) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{6}
}

func (x *DeferredInstallation) GetKey() string {
	if x != nil {
		return x.Key
	}
	return ""
}

func (x *DeferredInstallation) GetPackage() string {
	if x != nil {
		return x.Package
	}
	return ""
}

func (x *DeferredInstallation) GetRelease() string {
	if x != nil {
		return x.Release
	}
	return ""
}

func (x *DeferredInstallation) GetPreparation() *PreparePackageSetRequest {
	if x != nil {
		return x.Preparation
	}
	return nil
}

// Frozen selection metadata, never a future call or a producer receipt. An
// explicit Model argument bypasses this row. Unavailable defaults do not prevent
// importing a library or supplying an override. Every imported inference Model
// slot has one row when this feature is used; no alias is resolved after accept.
type MachineModelDefault struct {
	state                protoimpl.MessageState     `protogen:"open.v1"`
	Entrypoint           string                     `protobuf:"bytes,2,opt,name=entrypoint,proto3" json:"entrypoint,omitempty"`
	Parameter            string                     `protobuf:"bytes,3,opt,name=parameter,proto3" json:"parameter,omitempty"`
	PublicOrigin         string                     `protobuf:"bytes,4,opt,name=public_origin,json=publicOrigin,proto3" json:"public_origin,omitempty"`          // exact configured anonymous catalog origin; no credentials
	Rungs                []*MachineModelDefaultRung `protobuf:"bytes,5,rep,name=rungs,proto3" json:"rungs,omitempty"`                                            // authored/owner ladder order, unique (GPU, count) selectors
	UnavailableCode      string                     `protobuf:"bytes,6,opt,name=unavailable_code,json=unavailableCode,proto3" json:"unavailable_code,omitempty"` // exclusive with origin/rungs; bounded safe reason
	CalleeInstallationId string                     `protobuf:"bytes,7,opt,name=callee_installation_id,json=calleeInstallationId,proto3" json:"callee_installation_id,omitempty"`
	CalleeDeferredKey    string                     `protobuf:"bytes,8,opt,name=callee_deferred_key,json=calleeDeferredKey,proto3" json:"callee_deferred_key,omitempty"` // exclusive with callee_installation_id
	unknownFields        protoimpl.UnknownFields
	sizeCache            protoimpl.SizeCache
}

func (x *MachineModelDefault) Reset() {
	*x = MachineModelDefault{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[7]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *MachineModelDefault) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*MachineModelDefault) ProtoMessage() {}

func (x *MachineModelDefault) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[7]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use MachineModelDefault.ProtoReflect.Descriptor instead.
func (*MachineModelDefault) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{7}
}

func (x *MachineModelDefault) GetEntrypoint() string {
	if x != nil {
		return x.Entrypoint
	}
	return ""
}

func (x *MachineModelDefault) GetParameter() string {
	if x != nil {
		return x.Parameter
	}
	return ""
}

func (x *MachineModelDefault) GetPublicOrigin() string {
	if x != nil {
		return x.PublicOrigin
	}
	return ""
}

func (x *MachineModelDefault) GetRungs() []*MachineModelDefaultRung {
	if x != nil {
		return x.Rungs
	}
	return nil
}

func (x *MachineModelDefault) GetUnavailableCode() string {
	if x != nil {
		return x.UnavailableCode
	}
	return ""
}

func (x *MachineModelDefault) GetCalleeInstallationId() string {
	if x != nil {
		return x.CalleeInstallationId
	}
	return ""
}

func (x *MachineModelDefault) GetCalleeDeferredKey() string {
	if x != nil {
		return x.CalleeDeferredKey
	}
	return ""
}

type MachineModelDefaultRung struct {
	state protoimpl.MessageState `protogen:"open.v1"`
	// Zero preserves earlier captures: count was unspecified, never implicitly one.
	// Positive counts select the exact execution group, not the whole machine width.
	Gpus          uint32 `protobuf:"varint,4,opt,name=gpus,proto3" json:"gpus,omitempty"`
	Gpu           string `protobuf:"bytes,1,opt,name=gpu,proto3" json:"gpu,omitempty"`               // existing GPU selector, or "*"
	Repository    string `protobuf:"bytes,2,opt,name=repository,proto3" json:"repository,omitempty"` // org/name; native catalog membership remains mandatory
	Manifest      *Ref   `protobuf:"bytes,3,opt,name=manifest,proto3" json:"manifest,omitempty"`     // immutable checkpoint, never a release/lane lookup at use time
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *MachineModelDefaultRung) Reset() {
	*x = MachineModelDefaultRung{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[8]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *MachineModelDefaultRung) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*MachineModelDefaultRung) ProtoMessage() {}

func (x *MachineModelDefaultRung) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[8]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use MachineModelDefaultRung.ProtoReflect.Descriptor instead.
func (*MachineModelDefaultRung) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{8}
}

func (x *MachineModelDefaultRung) GetGpus() uint32 {
	if x != nil {
		return x.Gpus
	}
	return 0
}

func (x *MachineModelDefaultRung) GetGpu() string {
	if x != nil {
		return x.Gpu
	}
	return ""
}

func (x *MachineModelDefaultRung) GetRepository() string {
	if x != nil {
		return x.Repository
	}
	return ""
}

func (x *MachineModelDefaultRung) GetManifest() *Ref {
	if x != nil {
		return x.Manifest
	}
	return nil
}

type InstalledPackage struct {
	state            protoimpl.MessageState `protogen:"open.v1"`
	InstallationId   string                 `protobuf:"bytes,1,opt,name=installation_id,json=installationId,proto3" json:"installation_id,omitempty"` // opaque lifetime handle, never derived from package contents
	Package          string                 `protobuf:"bytes,2,opt,name=package,proto3" json:"package,omitempty"`
	Release          string                 `protobuf:"bytes,3,opt,name=release,proto3" json:"release,omitempty"`
	PackageInterface []byte                 `protobuf:"bytes,4,opt,name=package_interface,json=packageInterface,proto3" json:"package_interface,omitempty"` // discovered in this installation, no cross-SDK digest check
	unknownFields    protoimpl.UnknownFields
	sizeCache        protoimpl.SizeCache
}

func (x *InstalledPackage) Reset() {
	*x = InstalledPackage{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[9]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *InstalledPackage) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*InstalledPackage) ProtoMessage() {}

func (x *InstalledPackage) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[9]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use InstalledPackage.ProtoReflect.Descriptor instead.
func (*InstalledPackage) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{9}
}

func (x *InstalledPackage) GetInstallationId() string {
	if x != nil {
		return x.InstallationId
	}
	return ""
}

func (x *InstalledPackage) GetPackage() string {
	if x != nil {
		return x.Package
	}
	return ""
}

func (x *InstalledPackage) GetRelease() string {
	if x != nil {
		return x.Release
	}
	return ""
}

func (x *InstalledPackage) GetPackageInterface() []byte {
	if x != nil {
		return x.PackageInterface
	}
	return nil
}

type MachineCallableBinding struct {
	state                protoimpl.MessageState `protogen:"open.v1"`
	Module               string                 `protobuf:"bytes,3,opt,name=module,proto3" json:"module,omitempty"`
	Export               string                 `protobuf:"bytes,4,opt,name=export,proto3" json:"export,omitempty"`
	Entrypoint           string                 `protobuf:"bytes,6,opt,name=entrypoint,proto3" json:"entrypoint,omitempty"`
	CallerInstallationId string                 `protobuf:"bytes,7,opt,name=caller_installation_id,json=callerInstallationId,proto3" json:"caller_installation_id,omitempty"`
	CalleeInstallationId string                 `protobuf:"bytes,8,opt,name=callee_installation_id,json=calleeInstallationId,proto3" json:"callee_installation_id,omitempty"`
	CalleeDeferredKey    string                 `protobuf:"bytes,9,opt,name=callee_deferred_key,json=calleeDeferredKey,proto3" json:"callee_deferred_key,omitempty"`  // exclusive with callee_installation_id
	CallerDeferredKey    string                 `protobuf:"bytes,10,opt,name=caller_deferred_key,json=callerDeferredKey,proto3" json:"caller_deferred_key,omitempty"` // a deferred callee's own imports; exclusive with caller_installation_id
	unknownFields        protoimpl.UnknownFields
	sizeCache            protoimpl.SizeCache
}

func (x *MachineCallableBinding) Reset() {
	*x = MachineCallableBinding{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[10]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *MachineCallableBinding) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*MachineCallableBinding) ProtoMessage() {}

func (x *MachineCallableBinding) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[10]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use MachineCallableBinding.ProtoReflect.Descriptor instead.
func (*MachineCallableBinding) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{10}
}

func (x *MachineCallableBinding) GetModule() string {
	if x != nil {
		return x.Module
	}
	return ""
}

func (x *MachineCallableBinding) GetExport() string {
	if x != nil {
		return x.Export
	}
	return ""
}

func (x *MachineCallableBinding) GetEntrypoint() string {
	if x != nil {
		return x.Entrypoint
	}
	return ""
}

func (x *MachineCallableBinding) GetCallerInstallationId() string {
	if x != nil {
		return x.CallerInstallationId
	}
	return ""
}

func (x *MachineCallableBinding) GetCalleeInstallationId() string {
	if x != nil {
		return x.CalleeInstallationId
	}
	return ""
}

func (x *MachineCallableBinding) GetCalleeDeferredKey() string {
	if x != nil {
		return x.CalleeDeferredKey
	}
	return ""
}

func (x *MachineCallableBinding) GetCallerDeferredKey() string {
	if x != nil {
		return x.CallerDeferredKey
	}
	return ""
}

type MachineExecutionWorkspaceQuery struct {
	state protoimpl.MessageState `protogen:"open.v1"`
	Claim *Claim                 `protobuf:"bytes,1,opt,name=claim,proto3" json:"claim,omitempty"`
	// The machine describes this published release as it reads it from its own Hub (the newest
	// release when `release` is empty), in MachineExecutionWorkspace.described_release.
	Describe      *PackageSelection `protobuf:"bytes,2,opt,name=describe,proto3" json:"describe,omitempty"`
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *MachineExecutionWorkspaceQuery) Reset() {
	*x = MachineExecutionWorkspaceQuery{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[11]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *MachineExecutionWorkspaceQuery) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*MachineExecutionWorkspaceQuery) ProtoMessage() {}

func (x *MachineExecutionWorkspaceQuery) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[11]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use MachineExecutionWorkspaceQuery.ProtoReflect.Descriptor instead.
func (*MachineExecutionWorkspaceQuery) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{11}
}

func (x *MachineExecutionWorkspaceQuery) GetClaim() *Claim {
	if x != nil {
		return x.Claim
	}
	return nil
}

func (x *MachineExecutionWorkspaceQuery) GetDescribe() *PackageSelection {
	if x != nil {
		return x.Describe
	}
	return nil
}

type MachineExecutionWorkspace struct {
	state                protoimpl.MessageState `protogen:"open.v1"`
	WorkerId             string                 `protobuf:"bytes,1,opt,name=worker_id,json=workerId,proto3" json:"worker_id,omitempty"`
	WorkerBootId         string                 `protobuf:"bytes,2,opt,name=worker_boot_id,json=workerBootId,proto3" json:"worker_boot_id,omitempty"`
	ExecutionWorkspaceId string                 `protobuf:"bytes,3,opt,name=execution_workspace_id,json=executionWorkspaceId,proto3" json:"execution_workspace_id,omitempty"` // exact Runtime journal lifetime, never a path or credential
	// Additive observations measured by Runtime at boot. Readers never substitute their own probe.
	Devices              []*MachineDevice  `protobuf:"bytes,4,rep,name=devices,proto3" json:"devices,omitempty"`                                                          // every device Runtime schedules, in ordinal order
	AcceleratorBackend   string            `protobuf:"bytes,5,opt,name=accelerator_backend,json=acceleratorBackend,proto3" json:"accelerator_backend,omitempty"`          // "cuda" | "none"; empty from a Runtime that does not report
	ExecutorUidIsolation bool              `protobuf:"varint,6,opt,name=executor_uid_isolation,json=executorUidIsolation,proto3" json:"executor_uid_isolation,omitempty"` // package executors run under a uid separate from Runtime's
	DescribedRelease     *DescribedRelease `protobuf:"bytes,20,opt,name=described_release,json=describedRelease,proto3" json:"described_release,omitempty"`               // the query's `describe`, answered
	// Wire 65: the run output log. Its execution events carry `product` and the terminal
	// `outcome`. A Host forwards this field unchanged, so it is the capability Creator reads.
	RunOutputLog bool `protobuf:"varint,21,opt,name=run_output_log,json=runOutputLog,proto3" json:"run_output_log,omitempty"`
	// Submit accepts a ReleaseRoot naming installation_id and resolves its open Model slots,
	// org-relative defaults under ReleaseRoot.owner. Without it a controller submits unpublished
	// code as a capture with its Models resolved.
	ReleaseRootOwner bool `protobuf:"varint,22,opt,name=release_root_owner,json=releaseRootOwner,proto3" json:"release_root_owner,omitempty"`
	// Wire 68: close a submission key before re-placement or canceling unknown acceptance.
	SubmissionClose bool `protobuf:"varint,23,opt,name=submission_close,json=submissionClose,proto3" json:"submission_close,omitempty"`
	// Wire 70: ordered adapter stacks and explicitly qualified captured-callable Model
	// overrides. Required only by those selections; ordinary base-only roots remain valid.
	// A Host forwards this Runtime capability unchanged.
	ModelOverrides bool `protobuf:"varint,24,opt,name=model_overrides,json=modelOverrides,proto3" json:"model_overrides,omitempty"`
	unknownFields  protoimpl.UnknownFields
	sizeCache      protoimpl.SizeCache
}

func (x *MachineExecutionWorkspace) Reset() {
	*x = MachineExecutionWorkspace{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[12]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *MachineExecutionWorkspace) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*MachineExecutionWorkspace) ProtoMessage() {}

func (x *MachineExecutionWorkspace) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[12]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use MachineExecutionWorkspace.ProtoReflect.Descriptor instead.
func (*MachineExecutionWorkspace) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{12}
}

func (x *MachineExecutionWorkspace) GetWorkerId() string {
	if x != nil {
		return x.WorkerId
	}
	return ""
}

func (x *MachineExecutionWorkspace) GetWorkerBootId() string {
	if x != nil {
		return x.WorkerBootId
	}
	return ""
}

func (x *MachineExecutionWorkspace) GetExecutionWorkspaceId() string {
	if x != nil {
		return x.ExecutionWorkspaceId
	}
	return ""
}

func (x *MachineExecutionWorkspace) GetDevices() []*MachineDevice {
	if x != nil {
		return x.Devices
	}
	return nil
}

func (x *MachineExecutionWorkspace) GetAcceleratorBackend() string {
	if x != nil {
		return x.AcceleratorBackend
	}
	return ""
}

func (x *MachineExecutionWorkspace) GetExecutorUidIsolation() bool {
	if x != nil {
		return x.ExecutorUidIsolation
	}
	return false
}

func (x *MachineExecutionWorkspace) GetDescribedRelease() *DescribedRelease {
	if x != nil {
		return x.DescribedRelease
	}
	return nil
}

func (x *MachineExecutionWorkspace) GetRunOutputLog() bool {
	if x != nil {
		return x.RunOutputLog
	}
	return false
}

func (x *MachineExecutionWorkspace) GetReleaseRootOwner() bool {
	if x != nil {
		return x.ReleaseRootOwner
	}
	return false
}

func (x *MachineExecutionWorkspace) GetSubmissionClose() bool {
	if x != nil {
		return x.SubmissionClose
	}
	return false
}

func (x *MachineExecutionWorkspace) GetModelOverrides() bool {
	if x != nil {
		return x.ModelOverrides
	}
	return false
}

// A published release as the machine read it: the exact release and its interface.
type DescribedRelease struct {
	state            protoimpl.MessageState `protogen:"open.v1"`
	Package          string                 `protobuf:"bytes,1,opt,name=package,proto3" json:"package,omitempty"`
	Release          string                 `protobuf:"bytes,2,opt,name=release,proto3" json:"release,omitempty"`
	PackageInterface []byte                 `protobuf:"bytes,3,opt,name=package_interface,json=packageInterface,proto3" json:"package_interface,omitempty"` // canonical PackageInterface/1 bytes
	unknownFields    protoimpl.UnknownFields
	sizeCache        protoimpl.SizeCache
}

func (x *DescribedRelease) Reset() {
	*x = DescribedRelease{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[13]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *DescribedRelease) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*DescribedRelease) ProtoMessage() {}

func (x *DescribedRelease) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[13]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use DescribedRelease.ProtoReflect.Descriptor instead.
func (*DescribedRelease) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{13}
}

func (x *DescribedRelease) GetPackage() string {
	if x != nil {
		return x.Package
	}
	return ""
}

func (x *DescribedRelease) GetRelease() string {
	if x != nil {
		return x.Release
	}
	return ""
}

func (x *DescribedRelease) GetPackageInterface() []byte {
	if x != nil {
		return x.PackageInterface
	}
	return nil
}

// One measured accelerator. MachineExecutionGpu.ordinals name these ordinals.
type MachineDevice struct {
	state         protoimpl.MessageState `protogen:"open.v1"`
	Ordinal       uint32                 `protobuf:"varint,1,opt,name=ordinal,proto3" json:"ordinal,omitempty"`
	Name          string                 `protobuf:"bytes,2,opt,name=name,proto3" json:"name,omitempty"`
	Uuid          string                 `protobuf:"bytes,3,opt,name=uuid,proto3" json:"uuid,omitempty"`
	MemoryBytes   uint64                 `protobuf:"varint,4,opt,name=memory_bytes,json=memoryBytes,proto3" json:"memory_bytes,omitempty"`
	PciBusId      string                 `protobuf:"bytes,5,opt,name=pci_bus_id,json=pciBusId,proto3" json:"pci_bus_id,omitempty"`
	DriverVersion string                 `protobuf:"bytes,6,opt,name=driver_version,json=driverVersion,proto3" json:"driver_version,omitempty"`
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *MachineDevice) Reset() {
	*x = MachineDevice{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[14]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *MachineDevice) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*MachineDevice) ProtoMessage() {}

func (x *MachineDevice) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[14]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use MachineDevice.ProtoReflect.Descriptor instead.
func (*MachineDevice) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{14}
}

func (x *MachineDevice) GetOrdinal() uint32 {
	if x != nil {
		return x.Ordinal
	}
	return 0
}

func (x *MachineDevice) GetName() string {
	if x != nil {
		return x.Name
	}
	return ""
}

func (x *MachineDevice) GetUuid() string {
	if x != nil {
		return x.Uuid
	}
	return ""
}

func (x *MachineDevice) GetMemoryBytes() uint64 {
	if x != nil {
		return x.MemoryBytes
	}
	return 0
}

func (x *MachineDevice) GetPciBusId() string {
	if x != nil {
		return x.PciBusId
	}
	return ""
}

func (x *MachineDevice) GetDriverVersion() string {
	if x != nil {
		return x.DriverVersion
	}
	return ""
}

// Closing is durable even when submission has not arrived. An existing execution is
// returned, not silently canceled: the caller must reconcile/control it before re-placement.
type MachineSubmissionClose struct {
	state                        protoimpl.MessageState `protogen:"open.v1"`
	Claim                        *Claim                 `protobuf:"bytes,1,opt,name=claim,proto3" json:"claim,omitempty"`
	SubmissionId                 string                 `protobuf:"bytes,2,opt,name=submission_id,json=submissionId,proto3" json:"submission_id,omitempty"`
	RequestId                    string                 `protobuf:"bytes,3,opt,name=request_id,json=requestId,proto3" json:"request_id,omitempty"`
	ExpectedExecutionWorkspaceId string                 `protobuf:"bytes,4,opt,name=expected_execution_workspace_id,json=expectedExecutionWorkspaceId,proto3" json:"expected_execution_workspace_id,omitempty"`
	unknownFields                protoimpl.UnknownFields
	sizeCache                    protoimpl.SizeCache
}

func (x *MachineSubmissionClose) Reset() {
	*x = MachineSubmissionClose{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[15]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *MachineSubmissionClose) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*MachineSubmissionClose) ProtoMessage() {}

func (x *MachineSubmissionClose) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[15]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use MachineSubmissionClose.ProtoReflect.Descriptor instead.
func (*MachineSubmissionClose) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{15}
}

func (x *MachineSubmissionClose) GetClaim() *Claim {
	if x != nil {
		return x.Claim
	}
	return nil
}

func (x *MachineSubmissionClose) GetSubmissionId() string {
	if x != nil {
		return x.SubmissionId
	}
	return ""
}

func (x *MachineSubmissionClose) GetRequestId() string {
	if x != nil {
		return x.RequestId
	}
	return ""
}

func (x *MachineSubmissionClose) GetExpectedExecutionWorkspaceId() string {
	if x != nil {
		return x.ExpectedExecutionWorkspaceId
	}
	return ""
}

type MachineSubmissionClosure struct {
	state                protoimpl.MessageState   `protogen:"open.v1"`
	SubmissionId         string                   `protobuf:"bytes,1,opt,name=submission_id,json=submissionId,proto3" json:"submission_id,omitempty"`
	RequestId            string                   `protobuf:"bytes,2,opt,name=request_id,json=requestId,proto3" json:"request_id,omitempty"`
	ExecutionWorkspaceId string                   `protobuf:"bytes,3,opt,name=execution_workspace_id,json=executionWorkspaceId,proto3" json:"execution_workspace_id,omitempty"`
	Receipt              *MachineExecutionReceipt `protobuf:"bytes,4,opt,name=receipt,proto3" json:"receipt,omitempty"` // absent proves no acceptance and prevents later acceptance
	unknownFields        protoimpl.UnknownFields
	sizeCache            protoimpl.SizeCache
}

func (x *MachineSubmissionClosure) Reset() {
	*x = MachineSubmissionClosure{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[16]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *MachineSubmissionClosure) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*MachineSubmissionClosure) ProtoMessage() {}

func (x *MachineSubmissionClosure) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[16]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use MachineSubmissionClosure.ProtoReflect.Descriptor instead.
func (*MachineSubmissionClosure) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{16}
}

func (x *MachineSubmissionClosure) GetSubmissionId() string {
	if x != nil {
		return x.SubmissionId
	}
	return ""
}

func (x *MachineSubmissionClosure) GetRequestId() string {
	if x != nil {
		return x.RequestId
	}
	return ""
}

func (x *MachineSubmissionClosure) GetExecutionWorkspaceId() string {
	if x != nil {
		return x.ExecutionWorkspaceId
	}
	return ""
}

func (x *MachineSubmissionClosure) GetReceipt() *MachineExecutionReceipt {
	if x != nil {
		return x.Receipt
	}
	return nil
}

type MachineExecutionSubmit struct {
	state                 protoimpl.MessageState `protogen:"open.v1"`
	Claim                 *Claim                 `protobuf:"bytes,1,opt,name=claim,proto3" json:"claim,omitempty"`                                   // authentication now, never a continuing observer-liveness lease
	SubmissionId          string                 `protobuf:"bytes,2,opt,name=submission_id,json=submissionId,proto3" json:"submission_id,omitempty"` // <=256 bytes; identical resubmission returns one receipt
	CaptureDigest         []byte                 `protobuf:"bytes,3,opt,name=capture_digest,json=captureDigest,proto3" json:"capture_digest,omitempty"`
	CaptureCanonicalBytes []byte                 `protobuf:"bytes,4,opt,name=capture_canonical_bytes,json=captureCanonicalBytes,proto3" json:"capture_canonical_bytes,omitempty"` // <=1 MiB; verify digest before closed parsing
	Offer                 *AttemptOffer          `protobuf:"bytes,5,opt,name=offer,proto3" json:"offer,omitempty"`                                                                // first ordinal; existing immutable InvocationSpec/outcome codecs
	PreparedState         *DesiredWorkerState    `protobuf:"bytes,8,opt,name=prepared_state,json=preparedState,proto3" json:"prepared_state,omitempty"`                           // exact initial binding/capacity evidence
	// prepared_state is scoped to this execution; it cannot replace another accepted
	// execution or become authority to buy capacity. Runtime retains/revalidates its
	// own immutable preparation paths; client paths are not restart authority.
	PayloadCanonicalBytes []byte `protobuf:"bytes,9,opt,name=payload_canonical_bytes,json=payloadCanonicalBytes,proto3" json:"payload_canonical_bytes,omitempty"` // bounded typed arguments; hash must equal InvocationSpec.payload_digest
	// Runtime retains payload and native input custody before acceptance. No later
	// attempt may depend on a laptop file URL or an expired original delivery grant.
	PublicationAuthorizationId string `protobuf:"bytes,10,opt,name=publication_authorization_id,json=publicationAuthorizationId,proto3" json:"publication_authorization_id,omitempty"` // optional canonical nonzero UUID
	// Empty grants no Hub publication authority. Otherwise this exact independently
	// authorized, machine-bound grant is immutable for the root and its descendants.
	// Runtime must persist it before acceptance and cannot borrow another root's
	// grant. No token/key, authority URL or script content belongs in this field.
	// Changing it requires a new submission, never an idempotent replay or resume.
	ExpectedExecutionWorkspaceId string `protobuf:"bytes,11,opt,name=expected_execution_workspace_id,json=expectedExecutionWorkspaceId,proto3" json:"expected_execution_workspace_id,omitempty"` // required on first submit and every replay
	// The RecordOwner durably pins this identity BEFORE sending the first submit and
	// never refreshes it after an ambiguous reply. Runtime checks it in the SAME journal
	// transaction that finds or accepts the execution. Replacement refuses typed as
	// execution_workspace_changed; an empty identity refuses as execution_workspace_required.
	// Neither refusal proves that an earlier submit did not execute in another journal.
	SourceCredentials []*SourceCredential `protobuf:"bytes,12,rep,name=source_credentials,json=sourceCredentials,proto3" json:"source_credentials,omitempty"` // memory-only; see SourceCredential
	// A root named by its published release. The offer carries only
	// request_id; there is no capture or prepared_state: Runtime installs, resolves, prepares
	// and mints them, and the receipt names what it minted. The machine installs every release
	// the root's closure names itself, reading its own Hub. Until then Submit answers
	// UNAVAILABLE with cozy-error-code release_root_preparing (the message is progress).
	ReleaseRoot *ReleaseRoot `protobuf:"bytes,13,opt,name=release_root,json=releaseRoot,proto3" json:"release_root,omitempty"`
	// The record owner answers this execution's memo.lookup events.
	// False: a local miss computes at once; Runtime never waits on the owner.
	OwnerMemo bool `protobuf:"varint,14,opt,name=owner_memo,json=ownerMemo,proto3" json:"owner_memo,omitempty"`
	// The account that owns the run. Unpublished code (local/) has no org, so the org-relative
	// Model defaults of it and of every unpublished callee resolve under this account. Empty:
	// such a default is refused. A published package's org is its own.
	Account string `protobuf:"bytes,15,opt,name=account,proto3" json:"account,omitempty"`
	// Wire 69: explicit Hub scope for a captured root and its descendants. This selects
	// an existing scoped grant; no authority is conferred by the URL. Retained before
	// acceptance and immutable on replay. Release-root submissions use ReleaseRoot.hub.
	Hub           string `protobuf:"bytes,16,opt,name=hub,proto3" json:"hub,omitempty"`
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *MachineExecutionSubmit) Reset() {
	*x = MachineExecutionSubmit{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[17]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *MachineExecutionSubmit) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*MachineExecutionSubmit) ProtoMessage() {}

func (x *MachineExecutionSubmit) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[17]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use MachineExecutionSubmit.ProtoReflect.Descriptor instead.
func (*MachineExecutionSubmit) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{17}
}

func (x *MachineExecutionSubmit) GetClaim() *Claim {
	if x != nil {
		return x.Claim
	}
	return nil
}

func (x *MachineExecutionSubmit) GetSubmissionId() string {
	if x != nil {
		return x.SubmissionId
	}
	return ""
}

func (x *MachineExecutionSubmit) GetCaptureDigest() []byte {
	if x != nil {
		return x.CaptureDigest
	}
	return nil
}

func (x *MachineExecutionSubmit) GetCaptureCanonicalBytes() []byte {
	if x != nil {
		return x.CaptureCanonicalBytes
	}
	return nil
}

func (x *MachineExecutionSubmit) GetOffer() *AttemptOffer {
	if x != nil {
		return x.Offer
	}
	return nil
}

func (x *MachineExecutionSubmit) GetPreparedState() *DesiredWorkerState {
	if x != nil {
		return x.PreparedState
	}
	return nil
}

func (x *MachineExecutionSubmit) GetPayloadCanonicalBytes() []byte {
	if x != nil {
		return x.PayloadCanonicalBytes
	}
	return nil
}

func (x *MachineExecutionSubmit) GetPublicationAuthorizationId() string {
	if x != nil {
		return x.PublicationAuthorizationId
	}
	return ""
}

func (x *MachineExecutionSubmit) GetExpectedExecutionWorkspaceId() string {
	if x != nil {
		return x.ExpectedExecutionWorkspaceId
	}
	return ""
}

func (x *MachineExecutionSubmit) GetSourceCredentials() []*SourceCredential {
	if x != nil {
		return x.SourceCredentials
	}
	return nil
}

func (x *MachineExecutionSubmit) GetReleaseRoot() *ReleaseRoot {
	if x != nil {
		return x.ReleaseRoot
	}
	return nil
}

func (x *MachineExecutionSubmit) GetOwnerMemo() bool {
	if x != nil {
		return x.OwnerMemo
	}
	return false
}

func (x *MachineExecutionSubmit) GetAccount() string {
	if x != nil {
		return x.Account
	}
	return ""
}

func (x *MachineExecutionSubmit) GetHub() string {
	if x != nil {
		return x.Hub
	}
	return ""
}

type ReleaseRoot struct {
	state   protoimpl.MessageState `protogen:"open.v1"`
	Package string                 `protobuf:"bytes,1,opt,name=package,proto3" json:"package,omitempty"`
	// Exactly one of release and installation_id names the root's code.
	Release         string             `protobuf:"bytes,2,opt,name=release,proto3" json:"release,omitempty"`
	Entrypoint      string             `protobuf:"bytes,3,opt,name=entrypoint,proto3" json:"entrypoint,omitempty"`
	Models          []*ModelChoice     `protobuf:"bytes,6,rep,name=models,proto3" json:"models,omitempty"`                              // explicit Model choices only, sorted by parameter
	Inputs          []*InputBinding    `protobuf:"bytes,7,rep,name=inputs,proto3" json:"inputs,omitempty"`                              // byte inputs other than payload, as InvocationSpec.inputs
	InputAccess     []*InputAccess     `protobuf:"bytes,8,rep,name=input_access,json=inputAccess,proto3" json:"input_access,omitempty"` // their native trees, as DeliveryGrant.inputs
	DeadlineUnixMs  uint64             `protobuf:"varint,9,opt,name=deadline_unix_ms,json=deadlineUnixMs,proto3" json:"deadline_unix_ms,omitempty"`
	AttentionKernel string             `protobuf:"bytes,10,opt,name=attention_kernel,json=attentionKernel,proto3" json:"attention_kernel,omitempty"`
	Capture         *ActivationCapture `protobuf:"bytes,11,opt,name=capture,proto3" json:"capture,omitempty"`
	// The entrypoint names a published job: Runtime mints its
	// JobInvocationSpec and directive from the job's declaration, as it does for child jobs.
	Job bool `protobuf:"varint,13,opt,name=job,proto3" json:"job,omitempty"`
	// `--upload-to`: each weights output is sent to model://<weights_destination>. Empty: the
	// job's weights stay in the scratch grant.
	WeightsDestination string `protobuf:"bytes,14,opt,name=weights_destination,json=weightsDestination,proto3" json:"weights_destination,omitempty"`
	// The owner's scratch publication grant for this job ("org/_job-<request>").
	PublicationGrant string `protobuf:"bytes,15,opt,name=publication_grant,json=publicationGrant,proto3" json:"publication_grant,omitempty"`
	// A captured installation this machine already holds (an unpublished package the
	// controller prepared on it with PreparePackageSet) instead of a published release. The
	// machine resolves its slots, installs its published callees and mints the offer exactly
	// as for a release. Absent here: Submit refuses with cozy-error-code
	// release_root_installation_absent, and the controller prepares it and asks again.
	InstallationId string `protobuf:"bytes,16,opt,name=installation_id,json=installationId,proto3" json:"installation_id,omitempty"`
	// The account an unpublished installation's org-relative Model references name: it has no
	// org of its own. A published release's org is its package's. Sent only to a machine whose
	// workspace advertises release_root_owner.
	Owner string `protobuf:"bytes,17,opt,name=owner,proto3" json:"owner,omitempty"`
	// Wire 67: the Tensorhub origin this run came from. The machine reads the release, its callees,
	// its package index and its Models there, with its registration at that Hub. Empty: the
	// machine's default Hub. A Runtime before 67 ignores it and reads its default Hub, so a
	// controller sends another Hub's run only to a machine whose range reaches 67.
	Hub           string `protobuf:"bytes,18,opt,name=hub,proto3" json:"hub,omitempty"`
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *ReleaseRoot) Reset() {
	*x = ReleaseRoot{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[18]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *ReleaseRoot) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*ReleaseRoot) ProtoMessage() {}

func (x *ReleaseRoot) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[18]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use ReleaseRoot.ProtoReflect.Descriptor instead.
func (*ReleaseRoot) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{18}
}

func (x *ReleaseRoot) GetPackage() string {
	if x != nil {
		return x.Package
	}
	return ""
}

func (x *ReleaseRoot) GetRelease() string {
	if x != nil {
		return x.Release
	}
	return ""
}

func (x *ReleaseRoot) GetEntrypoint() string {
	if x != nil {
		return x.Entrypoint
	}
	return ""
}

func (x *ReleaseRoot) GetModels() []*ModelChoice {
	if x != nil {
		return x.Models
	}
	return nil
}

func (x *ReleaseRoot) GetInputs() []*InputBinding {
	if x != nil {
		return x.Inputs
	}
	return nil
}

func (x *ReleaseRoot) GetInputAccess() []*InputAccess {
	if x != nil {
		return x.InputAccess
	}
	return nil
}

func (x *ReleaseRoot) GetDeadlineUnixMs() uint64 {
	if x != nil {
		return x.DeadlineUnixMs
	}
	return 0
}

func (x *ReleaseRoot) GetAttentionKernel() string {
	if x != nil {
		return x.AttentionKernel
	}
	return ""
}

func (x *ReleaseRoot) GetCapture() *ActivationCapture {
	if x != nil {
		return x.Capture
	}
	return nil
}

func (x *ReleaseRoot) GetJob() bool {
	if x != nil {
		return x.Job
	}
	return false
}

func (x *ReleaseRoot) GetWeightsDestination() string {
	if x != nil {
		return x.WeightsDestination
	}
	return ""
}

func (x *ReleaseRoot) GetPublicationGrant() string {
	if x != nil {
		return x.PublicationGrant
	}
	return ""
}

func (x *ReleaseRoot) GetInstallationId() string {
	if x != nil {
		return x.InstallationId
	}
	return ""
}

func (x *ReleaseRoot) GetOwner() string {
	if x != nil {
		return x.Owner
	}
	return ""
}

func (x *ReleaseRoot) GetHub() string {
	if x != nil {
		return x.Hub
	}
	return ""
}

// An explicit Model for one slot. A manifest pins the checkpoint; otherwise Runtime reads the
// repository's release (newest when empty) and takes `lane`, or the slot's default ladder lanes.
// A provider `source` instead has the machine resolve it (headers first), download only the
// members its reviewed profiles need, convert them and bind the result; never with repository
// or manifest.
type ModelChoice struct {
	state protoimpl.MessageState `protogen:"open.v1"`
	// A bare parameter or the root's full slot path selects a root slot. Wire 70 also
	// accepts <entrypoint>.models.<parameter> for a captured callable in the root package,
	// or <package>/<entrypoint>.models.<parameter> for a captured external callable.
	// These are exact targets, never suffix matches or broadcasts. Unknown/uncaptured
	// targets refuse the operation. An adapter-only choice retains the selected base.
	Parameter     string                `protobuf:"bytes,1,opt,name=parameter,proto3" json:"parameter,omitempty"`
	Repository    string                `protobuf:"bytes,2,opt,name=repository,proto3" json:"repository,omitempty"`
	Release       string                `protobuf:"bytes,3,opt,name=release,proto3" json:"release,omitempty"`
	Lane          string                `protobuf:"bytes,4,opt,name=lane,proto3" json:"lane,omitempty"`
	Manifest      *Ref                  `protobuf:"bytes,5,opt,name=manifest,proto3" json:"manifest,omitempty"`
	Source        string                `protobuf:"bytes,6,opt,name=source,proto3" json:"source,omitempty"`     // immutable hf://org/repo@revision[/member] or civitai://version
	Profiles      []string              `protobuf:"bytes,7,rep,name=profiles,proto3" json:"profiles,omitempty"` // reviewed TensorFS profiles; empty: the machine selects one
	Adapters      []*DownloadAdapterRef `protobuf:"bytes,8,rep,name=adapters,proto3" json:"adapters,omitempty"` // Wire 70: ordered; order/scale are execution identity
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *ModelChoice) Reset() {
	*x = ModelChoice{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[19]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *ModelChoice) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*ModelChoice) ProtoMessage() {}

func (x *ModelChoice) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[19]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use ModelChoice.ProtoReflect.Descriptor instead.
func (*ModelChoice) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{19}
}

func (x *ModelChoice) GetParameter() string {
	if x != nil {
		return x.Parameter
	}
	return ""
}

func (x *ModelChoice) GetRepository() string {
	if x != nil {
		return x.Repository
	}
	return ""
}

func (x *ModelChoice) GetRelease() string {
	if x != nil {
		return x.Release
	}
	return ""
}

func (x *ModelChoice) GetLane() string {
	if x != nil {
		return x.Lane
	}
	return ""
}

func (x *ModelChoice) GetManifest() *Ref {
	if x != nil {
		return x.Manifest
	}
	return nil
}

func (x *ModelChoice) GetSource() string {
	if x != nil {
		return x.Source
	}
	return ""
}

func (x *ModelChoice) GetProfiles() []string {
	if x != nil {
		return x.Profiles
	}
	return nil
}

func (x *ModelChoice) GetAdapters() []*DownloadAdapterRef {
	if x != nil {
		return x.Adapters
	}
	return nil
}

// The owner's credential for one provider's native source calls, for this execution and its
// descendants. Runtime holds the value in memory only: never its journal, events, logs or
// diagnostics. It presents it only as NativeSourceCommand.credential for that provider. An
// identical resubmission replaces the values, which is how an owner supplies them again.
type SourceCredential struct {
	state         protoimpl.MessageState `protogen:"open.v1"`
	Provider      NativeSourceOperation  `protobuf:"varint,1,opt,name=provider,proto3,enum=cozy.worker.v1.NativeSourceOperation" json:"provider,omitempty"` // HUGGINGFACE or CIVITAI
	Credential    string                 `protobuf:"bytes,2,opt,name=credential,proto3" json:"credential,omitempty"`                                        // `bearer <token>`; empty presents none
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *SourceCredential) Reset() {
	*x = SourceCredential{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[20]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *SourceCredential) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*SourceCredential) ProtoMessage() {}

func (x *SourceCredential) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[20]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use SourceCredential.ProtoReflect.Descriptor instead.
func (*SourceCredential) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{20}
}

func (x *SourceCredential) GetProvider() NativeSourceOperation {
	if x != nil {
		return x.Provider
	}
	return NativeSourceOperation_NATIVE_SOURCE_OPERATION_UNSPECIFIED
}

func (x *SourceCredential) GetCredential() string {
	if x != nil {
		return x.Credential
	}
	return ""
}

type MachineExecutionReceipt struct {
	state                      protoimpl.MessageState `protogen:"open.v1"`
	RequestId                  string                 `protobuf:"bytes,1,opt,name=request_id,json=requestId,proto3" json:"request_id,omitempty"`
	SubmissionId               string                 `protobuf:"bytes,2,opt,name=submission_id,json=submissionId,proto3" json:"submission_id,omitempty"`
	CaptureDigest              []byte                 `protobuf:"bytes,3,opt,name=capture_digest,json=captureDigest,proto3" json:"capture_digest,omitempty"`
	InvocationSpecDigest       []byte                 `protobuf:"bytes,4,opt,name=invocation_spec_digest,json=invocationSpecDigest,proto3" json:"invocation_spec_digest,omitempty"`
	AcceptedAtMs               uint64                 `protobuf:"varint,5,opt,name=accepted_at_ms,json=acceptedAtMs,proto3" json:"accepted_at_ms,omitempty"`
	WorkerId                   string                 `protobuf:"bytes,6,opt,name=worker_id,json=workerId,proto3" json:"worker_id,omitempty"`
	WorkerBootId               string                 `protobuf:"bytes,7,opt,name=worker_boot_id,json=workerBootId,proto3" json:"worker_boot_id,omitempty"`                                           // acceptance boot; a later boot may reopen the same execution journal
	ExecutionWorkspaceId       string                 `protobuf:"bytes,8,opt,name=execution_workspace_id,json=executionWorkspaceId,proto3" json:"execution_workspace_id,omitempty"`                   // Runtime journal lifetime identity, not a filesystem path
	PublicationAuthorizationId string                 `protobuf:"bytes,9,opt,name=publication_authorization_id,json=publicationAuthorizationId,proto3" json:"publication_authorization_id,omitempty"` // exact accepted scope reference, or empty
	Number                     uint64                 `protobuf:"varint,10,opt,name=number,proto3" json:"number,omitempty"`                                                                           // wire 66: the run's number on this machine; 0 from an older Runtime
	unknownFields              protoimpl.UnknownFields
	sizeCache                  protoimpl.SizeCache
}

func (x *MachineExecutionReceipt) Reset() {
	*x = MachineExecutionReceipt{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[21]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *MachineExecutionReceipt) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*MachineExecutionReceipt) ProtoMessage() {}

func (x *MachineExecutionReceipt) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[21]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use MachineExecutionReceipt.ProtoReflect.Descriptor instead.
func (*MachineExecutionReceipt) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{21}
}

func (x *MachineExecutionReceipt) GetRequestId() string {
	if x != nil {
		return x.RequestId
	}
	return ""
}

func (x *MachineExecutionReceipt) GetSubmissionId() string {
	if x != nil {
		return x.SubmissionId
	}
	return ""
}

func (x *MachineExecutionReceipt) GetCaptureDigest() []byte {
	if x != nil {
		return x.CaptureDigest
	}
	return nil
}

func (x *MachineExecutionReceipt) GetInvocationSpecDigest() []byte {
	if x != nil {
		return x.InvocationSpecDigest
	}
	return nil
}

func (x *MachineExecutionReceipt) GetAcceptedAtMs() uint64 {
	if x != nil {
		return x.AcceptedAtMs
	}
	return 0
}

func (x *MachineExecutionReceipt) GetWorkerId() string {
	if x != nil {
		return x.WorkerId
	}
	return ""
}

func (x *MachineExecutionReceipt) GetWorkerBootId() string {
	if x != nil {
		return x.WorkerBootId
	}
	return ""
}

func (x *MachineExecutionReceipt) GetExecutionWorkspaceId() string {
	if x != nil {
		return x.ExecutionWorkspaceId
	}
	return ""
}

func (x *MachineExecutionReceipt) GetPublicationAuthorizationId() string {
	if x != nil {
		return x.PublicationAuthorizationId
	}
	return ""
}

func (x *MachineExecutionReceipt) GetNumber() uint64 {
	if x != nil {
		return x.Number
	}
	return 0
}

type MachineExecutionQuery struct {
	state                        protoimpl.MessageState `protogen:"open.v1"`
	Claim                        *Claim                 `protobuf:"bytes,1,opt,name=claim,proto3" json:"claim,omitempty"`
	RequestId                    string                 `protobuf:"bytes,2,opt,name=request_id,json=requestId,proto3" json:"request_id,omitempty"`
	ExpectedExecutionWorkspaceId string                 `protobuf:"bytes,3,opt,name=expected_execution_workspace_id,json=expectedExecutionWorkspaceId,proto3" json:"expected_execution_workspace_id,omitempty"` // a replaced/lost execution journal refuses
	unknownFields                protoimpl.UnknownFields
	sizeCache                    protoimpl.SizeCache
}

func (x *MachineExecutionQuery) Reset() {
	*x = MachineExecutionQuery{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[22]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *MachineExecutionQuery) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*MachineExecutionQuery) ProtoMessage() {}

func (x *MachineExecutionQuery) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[22]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use MachineExecutionQuery.ProtoReflect.Descriptor instead.
func (*MachineExecutionQuery) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{22}
}

func (x *MachineExecutionQuery) GetClaim() *Claim {
	if x != nil {
		return x.Claim
	}
	return nil
}

func (x *MachineExecutionQuery) GetRequestId() string {
	if x != nil {
		return x.RequestId
	}
	return ""
}

func (x *MachineExecutionQuery) GetExpectedExecutionWorkspaceId() string {
	if x != nil {
		return x.ExpectedExecutionWorkspaceId
	}
	return ""
}

type MachineExecutionState struct {
	state                protoimpl.MessageState `protogen:"open.v1"`
	RequestId            string                 `protobuf:"bytes,1,opt,name=request_id,json=requestId,proto3" json:"request_id,omitempty"`
	AttemptOrdinal       uint64                 `protobuf:"varint,2,opt,name=attempt_ordinal,json=attemptOrdinal,proto3" json:"attempt_ordinal,omitempty"`
	Generation           uint64                 `protobuf:"varint,3,opt,name=generation,proto3" json:"generation,omitempty"`
	State                string                 `protobuf:"bytes,4,opt,name=state,proto3" json:"state,omitempty"` // Runtime's bounded execution state, including queued/running/terminal
	Collected            bool                   `protobuf:"varint,5,opt,name=collected,proto3" json:"collected,omitempty"`
	Sequence             uint64                 `protobuf:"varint,6,opt,name=sequence,proto3" json:"sequence,omitempty"`
	WorkerId             string                 `protobuf:"bytes,7,opt,name=worker_id,json=workerId,proto3" json:"worker_id,omitempty"`
	WorkerBootId         string                 `protobuf:"bytes,8,opt,name=worker_boot_id,json=workerBootId,proto3" json:"worker_boot_id,omitempty"` // current boot, distinct from the receipt's acceptance boot
	ExecutionWorkspaceId string                 `protobuf:"bytes,9,opt,name=execution_workspace_id,json=executionWorkspaceId,proto3" json:"execution_workspace_id,omitempty"`
	Gpu                  *MachineExecutionGpu   `protobuf:"bytes,10,opt,name=gpu,proto3" json:"gpu,omitempty"` // additive observation; absent reads as phase "none"
	// Providers this unfinished execution was given a credential for whose value Runtime no longer
	// holds (a restart). Their source calls wait until the owner resubmits with source_credentials.
	AwaitingSourceCredentials []NativeSourceOperation `protobuf:"varint,11,rep,packed,name=awaiting_source_credentials,json=awaitingSourceCredentials,proto3,enum=cozy.worker.v1.NativeSourceOperation" json:"awaiting_source_credentials,omitempty"`
	// Wire 66; zero or absent from an older Runtime.
	Number        uint64                  `protobuf:"varint,12,opt,name=number,proto3" json:"number,omitempty"` // the run's number on this machine; 0 for a child call
	AcceptedAtMs  uint64                  `protobuf:"varint,13,opt,name=accepted_at_ms,json=acceptedAtMs,proto3" json:"accepted_at_ms,omitempty"`
	FinishedAtMs  uint64                  `protobuf:"varint,14,opt,name=finished_at_ms,json=finishedAtMs,proto3" json:"finished_at_ms,omitempty"` // when it reached its terminal state; 0 while open
	Target        *MachineExecutionTarget `protobuf:"bytes,15,opt,name=target,proto3" json:"target,omitempty"`
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *MachineExecutionState) Reset() {
	*x = MachineExecutionState{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[23]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *MachineExecutionState) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*MachineExecutionState) ProtoMessage() {}

func (x *MachineExecutionState) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[23]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use MachineExecutionState.ProtoReflect.Descriptor instead.
func (*MachineExecutionState) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{23}
}

func (x *MachineExecutionState) GetRequestId() string {
	if x != nil {
		return x.RequestId
	}
	return ""
}

func (x *MachineExecutionState) GetAttemptOrdinal() uint64 {
	if x != nil {
		return x.AttemptOrdinal
	}
	return 0
}

func (x *MachineExecutionState) GetGeneration() uint64 {
	if x != nil {
		return x.Generation
	}
	return 0
}

func (x *MachineExecutionState) GetState() string {
	if x != nil {
		return x.State
	}
	return ""
}

func (x *MachineExecutionState) GetCollected() bool {
	if x != nil {
		return x.Collected
	}
	return false
}

func (x *MachineExecutionState) GetSequence() uint64 {
	if x != nil {
		return x.Sequence
	}
	return 0
}

func (x *MachineExecutionState) GetWorkerId() string {
	if x != nil {
		return x.WorkerId
	}
	return ""
}

func (x *MachineExecutionState) GetWorkerBootId() string {
	if x != nil {
		return x.WorkerBootId
	}
	return ""
}

func (x *MachineExecutionState) GetExecutionWorkspaceId() string {
	if x != nil {
		return x.ExecutionWorkspaceId
	}
	return ""
}

func (x *MachineExecutionState) GetGpu() *MachineExecutionGpu {
	if x != nil {
		return x.Gpu
	}
	return nil
}

func (x *MachineExecutionState) GetAwaitingSourceCredentials() []NativeSourceOperation {
	if x != nil {
		return x.AwaitingSourceCredentials
	}
	return nil
}

func (x *MachineExecutionState) GetNumber() uint64 {
	if x != nil {
		return x.Number
	}
	return 0
}

func (x *MachineExecutionState) GetAcceptedAtMs() uint64 {
	if x != nil {
		return x.AcceptedAtMs
	}
	return 0
}

func (x *MachineExecutionState) GetFinishedAtMs() uint64 {
	if x != nil {
		return x.FinishedAtMs
	}
	return 0
}

func (x *MachineExecutionState) GetTarget() *MachineExecutionTarget {
	if x != nil {
		return x.Target
	}
	return nil
}

// What a run executes, for people and listings. Never an identity.
type MachineExecutionTarget struct {
	state          protoimpl.MessageState `protogen:"open.v1"`
	Package        string                 `protobuf:"bytes,1,opt,name=package,proto3" json:"package,omitempty"` // org/name
	Release        string                 `protobuf:"bytes,2,opt,name=release,proto3" json:"release,omitempty"`
	Entrypoint     string                 `protobuf:"bytes,3,opt,name=entrypoint,proto3" json:"entrypoint,omitempty"` // the function or job; empty when the machine cannot name it
	InstallationId string                 `protobuf:"bytes,4,opt,name=installation_id,json=installationId,proto3" json:"installation_id,omitempty"`
	Job            bool                   `protobuf:"varint,5,opt,name=job,proto3" json:"job,omitempty"`
	unknownFields  protoimpl.UnknownFields
	sizeCache      protoimpl.SizeCache
}

func (x *MachineExecutionTarget) Reset() {
	*x = MachineExecutionTarget{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[24]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *MachineExecutionTarget) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*MachineExecutionTarget) ProtoMessage() {}

func (x *MachineExecutionTarget) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[24]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use MachineExecutionTarget.ProtoReflect.Descriptor instead.
func (*MachineExecutionTarget) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{24}
}

func (x *MachineExecutionTarget) GetPackage() string {
	if x != nil {
		return x.Package
	}
	return ""
}

func (x *MachineExecutionTarget) GetRelease() string {
	if x != nil {
		return x.Release
	}
	return ""
}

func (x *MachineExecutionTarget) GetEntrypoint() string {
	if x != nil {
		return x.Entrypoint
	}
	return ""
}

func (x *MachineExecutionTarget) GetInstallationId() string {
	if x != nil {
		return x.InstallationId
	}
	return ""
}

func (x *MachineExecutionTarget) GetJob() bool {
	if x != nil {
		return x.Job
	}
	return false
}

// Runtime's GPU scheduler view of one execution and its descendant calls. Observation only:
// nothing is decided from it, and a reader tolerates an unknown phase.
type MachineExecutionGpu struct {
	state         protoimpl.MessageState `protogen:"open.v1"`
	Phase         string                 `protobuf:"bytes,1,opt,name=phase,proto3" json:"phase,omitempty"`                          // none | waiting | granted | executing | holding
	Width         uint32                 `protobuf:"varint,2,opt,name=width,proto3" json:"width,omitempty"`                         // GPUs the waiting or granted call needs
	Ordinals      []uint32               `protobuf:"varint,3,rep,packed,name=ordinals,proto3" json:"ordinals,omitempty"`            // the granted devices, or the root's held lease when holding
	BlockedBy     []string               `protobuf:"bytes,4,rep,name=blocked_by,json=blockedBy,proto3" json:"blocked_by,omitempty"` // root request ids holding the GPUs a waiting call needs
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *MachineExecutionGpu) Reset() {
	*x = MachineExecutionGpu{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[25]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *MachineExecutionGpu) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*MachineExecutionGpu) ProtoMessage() {}

func (x *MachineExecutionGpu) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[25]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use MachineExecutionGpu.ProtoReflect.Descriptor instead.
func (*MachineExecutionGpu) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{25}
}

func (x *MachineExecutionGpu) GetPhase() string {
	if x != nil {
		return x.Phase
	}
	return ""
}

func (x *MachineExecutionGpu) GetWidth() uint32 {
	if x != nil {
		return x.Width
	}
	return 0
}

func (x *MachineExecutionGpu) GetOrdinals() []uint32 {
	if x != nil {
		return x.Ordinals
	}
	return nil
}

func (x *MachineExecutionGpu) GetBlockedBy() []string {
	if x != nil {
		return x.BlockedBy
	}
	return nil
}

type MachineExecutionEventsQuery struct {
	state     protoimpl.MessageState `protogen:"open.v1"`
	Execution *MachineExecutionQuery `protobuf:"bytes,1,opt,name=execution,proto3" json:"execution,omitempty"`
	After     uint64                 `protobuf:"varint,2,opt,name=after,proto3" json:"after,omitempty"`
	Limit     uint32                 `protobuf:"varint,3,opt,name=limit,proto3" json:"limit,omitempty"` // 0 means 256; at most 256, bounded to 256 KiB per page
	// When no event follows `after`, answer once one does or the caller goes away. An observer
	// holds one such read open instead of polling.
	Wait          bool `protobuf:"varint,4,opt,name=wait,proto3" json:"wait,omitempty"`
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *MachineExecutionEventsQuery) Reset() {
	*x = MachineExecutionEventsQuery{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[26]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *MachineExecutionEventsQuery) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*MachineExecutionEventsQuery) ProtoMessage() {}

func (x *MachineExecutionEventsQuery) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[26]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use MachineExecutionEventsQuery.ProtoReflect.Descriptor instead.
func (*MachineExecutionEventsQuery) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{26}
}

func (x *MachineExecutionEventsQuery) GetExecution() *MachineExecutionQuery {
	if x != nil {
		return x.Execution
	}
	return nil
}

func (x *MachineExecutionEventsQuery) GetAfter() uint64 {
	if x != nil {
		return x.After
	}
	return 0
}

func (x *MachineExecutionEventsQuery) GetLimit() uint32 {
	if x != nil {
		return x.Limit
	}
	return 0
}

func (x *MachineExecutionEventsQuery) GetWait() bool {
	if x != nil {
		return x.Wait
	}
	return false
}

type MachineExecutionEvent struct {
	state              protoimpl.MessageState `protogen:"open.v1"`
	Sequence           uint64                 `protobuf:"varint,1,opt,name=sequence,proto3" json:"sequence,omitempty"`
	AttemptOrdinal     uint64                 `protobuf:"varint,2,opt,name=attempt_ordinal,json=attemptOrdinal,proto3" json:"attempt_ordinal,omitempty"`
	AtMs               uint64                 `protobuf:"varint,3,opt,name=at_ms,json=atMs,proto3" json:"at_ms,omitempty"`
	Kind               string                 `protobuf:"bytes,4,opt,name=kind,proto3" json:"kind,omitempty"`
	BodyCanonicalBytes []byte                 `protobuf:"bytes,5,opt,name=body_canonical_bytes,json=bodyCanonicalBytes,proto3" json:"body_canonical_bytes,omitempty"` // bounded canonical JSON metadata, <=64 KiB
	// Kind "product": one product entered an output of this run. body_canonical_bytes holds the
	// same RunProduct as its canonical document.
	Product *RunProduct `protobuf:"bytes,6,opt,name=product,proto3" json:"product,omitempty"`
	// Kind "outcome": the exact terminal of the current attempt, the last entry of the run's log.
	// It replaces CollectMachineExecution; a page that carries it may exceed 256 KiB.
	Outcome       *AttemptOutcome `protobuf:"bytes,7,opt,name=outcome,proto3" json:"outcome,omitempty"`
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *MachineExecutionEvent) Reset() {
	*x = MachineExecutionEvent{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[27]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *MachineExecutionEvent) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*MachineExecutionEvent) ProtoMessage() {}

func (x *MachineExecutionEvent) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[27]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use MachineExecutionEvent.ProtoReflect.Descriptor instead.
func (*MachineExecutionEvent) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{27}
}

func (x *MachineExecutionEvent) GetSequence() uint64 {
	if x != nil {
		return x.Sequence
	}
	return 0
}

func (x *MachineExecutionEvent) GetAttemptOrdinal() uint64 {
	if x != nil {
		return x.AttemptOrdinal
	}
	return 0
}

func (x *MachineExecutionEvent) GetAtMs() uint64 {
	if x != nil {
		return x.AtMs
	}
	return 0
}

func (x *MachineExecutionEvent) GetKind() string {
	if x != nil {
		return x.Kind
	}
	return ""
}

func (x *MachineExecutionEvent) GetBodyCanonicalBytes() []byte {
	if x != nil {
		return x.BodyCanonicalBytes
	}
	return nil
}

func (x *MachineExecutionEvent) GetProduct() *RunProduct {
	if x != nil {
		return x.Product
	}
	return nil
}

func (x *MachineExecutionEvent) GetOutcome() *AttemptOutcome {
	if x != nil {
		return x.Outcome
	}
	return nil
}

// The run output log's unit of incremental delivery. An output is an asset field of the
// function's declared result: a single output's product is replaced by each SET, a list output
// grows by one APPEND per product. The run's result is the fold of its products. The bytes
// were committed and are held by the log before the entry is journaled; the owner reads them
// with ReadByteTreeObject from `source` (or each part's source). Before wire 66 they are held
// until the owner acknowledges the terminal; from 66 until the store's GC evicts them, so any
// authorized client can fetch them later. A SET is released as soon as a newer SET of its
// output supersedes it.
type RunProduct struct {
	state     protoimpl.MessageState `protogen:"open.v1"`
	Output    string                 `protobuf:"bytes,1,opt,name=output,proto3" json:"output,omitempty"` // declared result field path, e.g. "video", "references"
	Op        RunProductOp           `protobuf:"varint,2,opt,name=op,proto3,enum=cozy.worker.v1.RunProductOp" json:"op,omitempty"`
	Index     uint32                 `protobuf:"varint,3,opt,name=index,proto3" json:"index,omitempty"`    // APPEND: position in the list output; SET: 0
	Content   *Ref                   `protobuf:"bytes,4,opt,name=content,proto3" json:"content,omitempty"` // sha256 and length of the product's bytes
	MediaType string                 `protobuf:"bytes,5,opt,name=media_type,json=mediaType,proto3" json:"media_type,omitempty"`
	Label     string                 `protobuf:"bytes,6,opt,name=label,proto3" json:"label,omitempty"` // for people, e.g. "Video (segment 3 of 9)"; <=256 bytes
	// Exactly one of source and parts. With parts the bytes are the parts concatenated in order.
	Source        *NativeByteRetentionRequest `protobuf:"bytes,7,opt,name=source,proto3" json:"source,omitempty"`
	Parts         []*RunProductPart           `protobuf:"bytes,8,rep,name=parts,proto3" json:"parts,omitempty"`
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *RunProduct) Reset() {
	*x = RunProduct{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[28]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *RunProduct) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*RunProduct) ProtoMessage() {}

func (x *RunProduct) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[28]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use RunProduct.ProtoReflect.Descriptor instead.
func (*RunProduct) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{28}
}

func (x *RunProduct) GetOutput() string {
	if x != nil {
		return x.Output
	}
	return ""
}

func (x *RunProduct) GetOp() RunProductOp {
	if x != nil {
		return x.Op
	}
	return RunProductOp_RUN_PRODUCT_OP_UNSPECIFIED
}

func (x *RunProduct) GetIndex() uint32 {
	if x != nil {
		return x.Index
	}
	return 0
}

func (x *RunProduct) GetContent() *Ref {
	if x != nil {
		return x.Content
	}
	return nil
}

func (x *RunProduct) GetMediaType() string {
	if x != nil {
		return x.MediaType
	}
	return ""
}

func (x *RunProduct) GetLabel() string {
	if x != nil {
		return x.Label
	}
	return ""
}

func (x *RunProduct) GetSource() *NativeByteRetentionRequest {
	if x != nil {
		return x.Source
	}
	return nil
}

func (x *RunProduct) GetParts() []*RunProductPart {
	if x != nil {
		return x.Parts
	}
	return nil
}

type RunProductPart struct {
	state         protoimpl.MessageState      `protogen:"open.v1"`
	Content       *Ref                        `protobuf:"bytes,1,opt,name=content,proto3" json:"content,omitempty"`
	Source        *NativeByteRetentionRequest `protobuf:"bytes,2,opt,name=source,proto3" json:"source,omitempty"`
	DurationUs    uint64                      `protobuf:"varint,3,opt,name=duration_us,json=durationUs,proto3" json:"duration_us,omitempty"` // media time this part adds; 0 for an initialization segment
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *RunProductPart) Reset() {
	*x = RunProductPart{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[29]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *RunProductPart) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*RunProductPart) ProtoMessage() {}

func (x *RunProductPart) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[29]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use RunProductPart.ProtoReflect.Descriptor instead.
func (*RunProductPart) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{29}
}

func (x *RunProductPart) GetContent() *Ref {
	if x != nil {
		return x.Content
	}
	return nil
}

func (x *RunProductPart) GetSource() *NativeByteRetentionRequest {
	if x != nil {
		return x.Source
	}
	return nil
}

func (x *RunProductPart) GetDurationUs() uint64 {
	if x != nil {
		return x.DurationUs
	}
	return 0
}

type MachineExecutionEventPage struct {
	state            protoimpl.MessageState   `protogen:"open.v1"`
	Events           []*MachineExecutionEvent `protobuf:"bytes,1,rep,name=events,proto3" json:"events,omitempty"`
	NextAfter        uint64                   `protobuf:"varint,2,opt,name=next_after,json=nextAfter,proto3" json:"next_after,omitempty"`
	HeadSequence     uint64                   `protobuf:"varint,3,opt,name=head_sequence,json=headSequence,proto3" json:"head_sequence,omitempty"`
	CompactedThrough uint64                   `protobuf:"varint,4,opt,name=compacted_through,json=compactedThrough,proto3" json:"compacted_through,omitempty"` // explicit lossy progress gaps; terminal history is retained
	unknownFields    protoimpl.UnknownFields
	sizeCache        protoimpl.SizeCache
}

func (x *MachineExecutionEventPage) Reset() {
	*x = MachineExecutionEventPage{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[30]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *MachineExecutionEventPage) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*MachineExecutionEventPage) ProtoMessage() {}

func (x *MachineExecutionEventPage) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[30]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use MachineExecutionEventPage.ProtoReflect.Descriptor instead.
func (*MachineExecutionEventPage) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{30}
}

func (x *MachineExecutionEventPage) GetEvents() []*MachineExecutionEvent {
	if x != nil {
		return x.Events
	}
	return nil
}

func (x *MachineExecutionEventPage) GetNextAfter() uint64 {
	if x != nil {
		return x.NextAfter
	}
	return 0
}

func (x *MachineExecutionEventPage) GetHeadSequence() uint64 {
	if x != nil {
		return x.HeadSequence
	}
	return 0
}

func (x *MachineExecutionEventPage) GetCompactedThrough() uint64 {
	if x != nil {
		return x.CompactedThrough
	}
	return 0
}

type MachineExecutionControl struct {
	state              protoimpl.MessageState            `protogen:"open.v1"`
	Execution          *MachineExecutionQuery            `protobuf:"bytes,1,opt,name=execution,proto3" json:"execution,omitempty"`
	CommandId          string                            `protobuf:"bytes,2,opt,name=command_id,json=commandId,proto3" json:"command_id,omitempty"` // idempotent; same ID with changed contents refuses
	ExpectedGeneration uint64                            `protobuf:"varint,3,opt,name=expected_generation,json=expectedGeneration,proto3" json:"expected_generation,omitempty"`
	Action             MachineExecutionAction            `protobuf:"varint,4,opt,name=action,proto3,enum=cozy.worker.v1.MachineExecutionAction" json:"action,omitempty"`
	Publication        *MachinePublicationReconciliation `protobuf:"bytes,5,opt,name=publication,proto3" json:"publication,omitempty"` // RECONCILE_PUBLICATION only
	Memo               *MachineMemoAnswer                `protobuf:"bytes,6,opt,name=memo,proto3" json:"memo,omitempty"`               // ANSWER_MEMO only
	unknownFields      protoimpl.UnknownFields
	sizeCache          protoimpl.SizeCache
}

func (x *MachineExecutionControl) Reset() {
	*x = MachineExecutionControl{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[31]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *MachineExecutionControl) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*MachineExecutionControl) ProtoMessage() {}

func (x *MachineExecutionControl) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[31]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use MachineExecutionControl.ProtoReflect.Descriptor instead.
func (*MachineExecutionControl) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{31}
}

func (x *MachineExecutionControl) GetExecution() *MachineExecutionQuery {
	if x != nil {
		return x.Execution
	}
	return nil
}

func (x *MachineExecutionControl) GetCommandId() string {
	if x != nil {
		return x.CommandId
	}
	return ""
}

func (x *MachineExecutionControl) GetExpectedGeneration() uint64 {
	if x != nil {
		return x.ExpectedGeneration
	}
	return 0
}

func (x *MachineExecutionControl) GetAction() MachineExecutionAction {
	if x != nil {
		return x.Action
	}
	return MachineExecutionAction_MACHINE_EXECUTION_ACTION_UNSPECIFIED
}

func (x *MachineExecutionControl) GetPublication() *MachinePublicationReconciliation {
	if x != nil {
		return x.Publication
	}
	return nil
}

func (x *MachineExecutionControl) GetMemo() *MachineMemoAnswer {
	if x != nil {
		return x.Memo
	}
	return nil
}

// The record owner's answer to one memo.lookup, from its own completed results. Runtime
// accepts a hit only for the same operation and computation digest, and only a result that
// names durable custody the operation's own completion would name (an upload's committed
// checkpoint in the call's destination). Anything else is refused typed and computes.
type MachineMemoAnswer struct {
	state                protoimpl.MessageState `protogen:"open.v1"`
	LookupSequence       uint64                 `protobuf:"varint,1,opt,name=lookup_sequence,json=lookupSequence,proto3" json:"lookup_sequence,omitempty"`                    // the memo.lookup event answered
	ComputationDigest    []byte                 `protobuf:"bytes,2,opt,name=computation_digest,json=computationDigest,proto3" json:"computation_digest,omitempty"`            // 32 bytes; equals the lookup's
	ResultCanonicalBytes []byte                 `protobuf:"bytes,3,opt,name=result_canonical_bytes,json=resultCanonicalBytes,proto3" json:"result_canonical_bytes,omitempty"` // empty is a miss; else the recorded result, <= 48 KiB
	unknownFields        protoimpl.UnknownFields
	sizeCache            protoimpl.SizeCache
}

func (x *MachineMemoAnswer) Reset() {
	*x = MachineMemoAnswer{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[32]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *MachineMemoAnswer) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*MachineMemoAnswer) ProtoMessage() {}

func (x *MachineMemoAnswer) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[32]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use MachineMemoAnswer.ProtoReflect.Descriptor instead.
func (*MachineMemoAnswer) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{32}
}

func (x *MachineMemoAnswer) GetLookupSequence() uint64 {
	if x != nil {
		return x.LookupSequence
	}
	return 0
}

func (x *MachineMemoAnswer) GetComputationDigest() []byte {
	if x != nil {
		return x.ComputationDigest
	}
	return nil
}

func (x *MachineMemoAnswer) GetResultCanonicalBytes() []byte {
	if x != nil {
		return x.ResultCanonicalBytes
	}
	return nil
}

// The owner's read of one sent publication's Hub finalization status, forwarded unmodified.
// Runtime verifies it names this call's operation, checkpoint and objects before settling:
// completed commits, failed or an absent publication/finalization (404) is uncommitted, and
// a queued or running finalization is refused as pending.
type MachinePublicationReconciliation struct {
	state         protoimpl.MessageState `protogen:"open.v1"`
	CallIndex     uint32                 `protobuf:"varint,1,opt,name=call_index,json=callIndex,proto3" json:"call_index,omitempty"`    // the publication effect of this execution
	HttpStatus    uint32                 `protobuf:"varint,2,opt,name=http_status,json=httpStatus,proto3" json:"http_status,omitempty"` // Hub's status for GET .../publications/{operation}/finalization
	Finalization  []byte                 `protobuf:"bytes,3,opt,name=finalization,proto3" json:"finalization,omitempty"`                // Hub's exact response body
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *MachinePublicationReconciliation) Reset() {
	*x = MachinePublicationReconciliation{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[33]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *MachinePublicationReconciliation) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*MachinePublicationReconciliation) ProtoMessage() {}

func (x *MachinePublicationReconciliation) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[33]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use MachinePublicationReconciliation.ProtoReflect.Descriptor instead.
func (*MachinePublicationReconciliation) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{33}
}

func (x *MachinePublicationReconciliation) GetCallIndex() uint32 {
	if x != nil {
		return x.CallIndex
	}
	return 0
}

func (x *MachinePublicationReconciliation) GetHttpStatus() uint32 {
	if x != nil {
		return x.HttpStatus
	}
	return 0
}

func (x *MachinePublicationReconciliation) GetFinalization() []byte {
	if x != nil {
		return x.Finalization
	}
	return nil
}

type MachineExecutionCollect struct {
	state          protoimpl.MessageState `protogen:"open.v1"`
	Execution      *MachineExecutionQuery `protobuf:"bytes,1,opt,name=execution,proto3" json:"execution,omitempty"`
	AttemptOrdinal uint64                 `protobuf:"varint,2,opt,name=attempt_ordinal,json=attemptOrdinal,proto3" json:"attempt_ordinal,omitempty"` // zero selects the current attempt
	unknownFields  protoimpl.UnknownFields
	sizeCache      protoimpl.SizeCache
}

func (x *MachineExecutionCollect) Reset() {
	*x = MachineExecutionCollect{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[34]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *MachineExecutionCollect) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*MachineExecutionCollect) ProtoMessage() {}

func (x *MachineExecutionCollect) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[34]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use MachineExecutionCollect.ProtoReflect.Descriptor instead.
func (*MachineExecutionCollect) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{34}
}

func (x *MachineExecutionCollect) GetExecution() *MachineExecutionQuery {
	if x != nil {
		return x.Execution
	}
	return nil
}

func (x *MachineExecutionCollect) GetAttemptOrdinal() uint64 {
	if x != nil {
		return x.AttemptOrdinal
	}
	return 0
}

type MachineExecutionCollectionAck struct {
	state         protoimpl.MessageState `protogen:"open.v1"`
	Execution     *MachineExecutionQuery `protobuf:"bytes,1,opt,name=execution,proto3" json:"execution,omitempty"`
	Outcome       *AttemptOutcomeAck     `protobuf:"bytes,2,opt,name=outcome,proto3" json:"outcome,omitempty"` // exact terminal identity, after recipient custody
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *MachineExecutionCollectionAck) Reset() {
	*x = MachineExecutionCollectionAck{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[35]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *MachineExecutionCollectionAck) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*MachineExecutionCollectionAck) ProtoMessage() {}

func (x *MachineExecutionCollectionAck) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[35]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use MachineExecutionCollectionAck.ProtoReflect.Descriptor instead.
func (*MachineExecutionCollectionAck) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{35}
}

func (x *MachineExecutionCollectionAck) GetExecution() *MachineExecutionQuery {
	if x != nil {
		return x.Execution
	}
	return nil
}

func (x *MachineExecutionCollectionAck) GetOutcome() *AttemptOutcomeAck {
	if x != nil {
		return x.Outcome
	}
	return nil
}

type MachineExecutionTriageQuery struct {
	state          protoimpl.MessageState `protogen:"open.v1"`
	Execution      *MachineExecutionQuery `protobuf:"bytes,1,opt,name=execution,proto3" json:"execution,omitempty"`
	AttemptOrdinal uint64                 `protobuf:"varint,2,opt,name=attempt_ordinal,json=attemptOrdinal,proto3" json:"attempt_ordinal,omitempty"` // zero selects the current attempt
	unknownFields  protoimpl.UnknownFields
	sizeCache      protoimpl.SizeCache
}

func (x *MachineExecutionTriageQuery) Reset() {
	*x = MachineExecutionTriageQuery{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[36]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *MachineExecutionTriageQuery) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*MachineExecutionTriageQuery) ProtoMessage() {}

func (x *MachineExecutionTriageQuery) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[36]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use MachineExecutionTriageQuery.ProtoReflect.Descriptor instead.
func (*MachineExecutionTriageQuery) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{36}
}

func (x *MachineExecutionTriageQuery) GetExecution() *MachineExecutionQuery {
	if x != nil {
		return x.Execution
	}
	return nil
}

func (x *MachineExecutionTriageQuery) GetAttemptOrdinal() uint64 {
	if x != nil {
		return x.AttemptOrdinal
	}
	return 0
}

// NOT_FOUND when the outcome names no bundle or the bundle is no longer retained.
type MachineExecutionTriage struct {
	state                protoimpl.MessageState `protogen:"open.v1"`
	Bundle               *TriageBundleRef       `protobuf:"bytes,1,opt,name=bundle,proto3" json:"bundle,omitempty"`                                                           // exactly the reference the attempt's outcome carries
	BundleCanonicalBytes []byte                 `protobuf:"bytes,2,opt,name=bundle_canonical_bytes,json=bundleCanonicalBytes,proto3" json:"bundle_canonical_bytes,omitempty"` // <= 1 MiB; sha256 equals bundle.write_receipt_digest
	unknownFields        protoimpl.UnknownFields
	sizeCache            protoimpl.SizeCache
}

func (x *MachineExecutionTriage) Reset() {
	*x = MachineExecutionTriage{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[37]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *MachineExecutionTriage) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*MachineExecutionTriage) ProtoMessage() {}

func (x *MachineExecutionTriage) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[37]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use MachineExecutionTriage.ProtoReflect.Descriptor instead.
func (*MachineExecutionTriage) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{37}
}

func (x *MachineExecutionTriage) GetBundle() *TriageBundleRef {
	if x != nil {
		return x.Bundle
	}
	return nil
}

func (x *MachineExecutionTriage) GetBundleCanonicalBytes() []byte {
	if x != nil {
		return x.BundleCanonicalBytes
	}
	return nil
}

// Wire 66. One page of this machine's runs by `number`: the execution journal's sequence of root
// executions, assigned at acceptance and never reused within that journal. Child calls have no
// number and are not listed. Oldest first after `after_number`, or newest first before
// `before_number` (0: from the newest). The next page's cursor is the last row's number.
type MachineExecutionListQuery struct {
	state        protoimpl.MessageState `protogen:"open.v1"`
	Claim        *Claim                 `protobuf:"bytes,1,opt,name=claim,proto3" json:"claim,omitempty"`
	AfterNumber  uint64                 `protobuf:"varint,2,opt,name=after_number,json=afterNumber,proto3" json:"after_number,omitempty"`
	NewestFirst  bool                   `protobuf:"varint,3,opt,name=newest_first,json=newestFirst,proto3" json:"newest_first,omitempty"`
	BeforeNumber uint64                 `protobuf:"varint,4,opt,name=before_number,json=beforeNumber,proto3" json:"before_number,omitempty"` // newest_first only
	Limit        uint32                 `protobuf:"varint,5,opt,name=limit,proto3" json:"limit,omitempty"`                                   // 0 means 64; at most 256
	States       []string               `protobuf:"bytes,6,rep,name=states,proto3" json:"states,omitempty"`                                  // only runs in these MachineExecutionState.state values; empty: all
	// Oldest first only: when no listed run follows after_number, answer once one is accepted or
	// the caller goes away. A follower holds one such read open instead of polling.
	Wait          bool `protobuf:"varint,7,opt,name=wait,proto3" json:"wait,omitempty"`
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *MachineExecutionListQuery) Reset() {
	*x = MachineExecutionListQuery{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[38]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *MachineExecutionListQuery) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*MachineExecutionListQuery) ProtoMessage() {}

func (x *MachineExecutionListQuery) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[38]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use MachineExecutionListQuery.ProtoReflect.Descriptor instead.
func (*MachineExecutionListQuery) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{38}
}

func (x *MachineExecutionListQuery) GetClaim() *Claim {
	if x != nil {
		return x.Claim
	}
	return nil
}

func (x *MachineExecutionListQuery) GetAfterNumber() uint64 {
	if x != nil {
		return x.AfterNumber
	}
	return 0
}

func (x *MachineExecutionListQuery) GetNewestFirst() bool {
	if x != nil {
		return x.NewestFirst
	}
	return false
}

func (x *MachineExecutionListQuery) GetBeforeNumber() uint64 {
	if x != nil {
		return x.BeforeNumber
	}
	return 0
}

func (x *MachineExecutionListQuery) GetLimit() uint32 {
	if x != nil {
		return x.Limit
	}
	return 0
}

func (x *MachineExecutionListQuery) GetStates() []string {
	if x != nil {
		return x.States
	}
	return nil
}

func (x *MachineExecutionListQuery) GetWait() bool {
	if x != nil {
		return x.Wait
	}
	return false
}

type MachineExecutionList struct {
	state                protoimpl.MessageState   `protogen:"open.v1"`
	Executions           []*MachineExecutionState `protobuf:"bytes,1,rep,name=executions,proto3" json:"executions,omitempty"`                                                   // in the query's order
	HeadNumber           uint64                   `protobuf:"varint,2,opt,name=head_number,json=headNumber,proto3" json:"head_number,omitempty"`                                // the newest number assigned; 0: none yet
	ExecutionWorkspaceId string                   `protobuf:"bytes,3,opt,name=execution_workspace_id,json=executionWorkspaceId,proto3" json:"execution_workspace_id,omitempty"` // the journal lifetime its numbers belong to
	unknownFields        protoimpl.UnknownFields
	sizeCache            protoimpl.SizeCache
}

func (x *MachineExecutionList) Reset() {
	*x = MachineExecutionList{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[39]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *MachineExecutionList) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*MachineExecutionList) ProtoMessage() {}

func (x *MachineExecutionList) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[39]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use MachineExecutionList.ProtoReflect.Descriptor instead.
func (*MachineExecutionList) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{39}
}

func (x *MachineExecutionList) GetExecutions() []*MachineExecutionState {
	if x != nil {
		return x.Executions
	}
	return nil
}

func (x *MachineExecutionList) GetHeadNumber() uint64 {
	if x != nil {
		return x.HeadNumber
	}
	return 0
}

func (x *MachineExecutionList) GetExecutionWorkspaceId() string {
	if x != nil {
		return x.ExecutionWorkspaceId
	}
	return ""
}

type PackageListQuery struct {
	state         protoimpl.MessageState `protogen:"open.v1"`
	Claim         *Claim                 `protobuf:"bytes,1,opt,name=claim,proto3" json:"claim,omitempty"`
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *PackageListQuery) Reset() {
	*x = PackageListQuery{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[40]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *PackageListQuery) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*PackageListQuery) ProtoMessage() {}

func (x *PackageListQuery) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[40]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use PackageListQuery.ProtoReflect.Descriptor instead.
func (*PackageListQuery) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{40}
}

func (x *PackageListQuery) GetClaim() *Claim {
	if x != nil {
		return x.Claim
	}
	return nil
}

type PackageList struct {
	state         protoimpl.MessageState `protogen:"open.v1"`
	Packages      []*MachinePackage      `protobuf:"bytes,1,rep,name=packages,proto3" json:"packages,omitempty"` // sorted by package, release, installation_id
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *PackageList) Reset() {
	*x = PackageList{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[41]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *PackageList) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*PackageList) ProtoMessage() {}

func (x *PackageList) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[41]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use PackageList.ProtoReflect.Descriptor instead.
func (*PackageList) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{41}
}

func (x *PackageList) GetPackages() []*MachinePackage {
	if x != nil {
		return x.Packages
	}
	return nil
}

// One installed environment this machine holds.
type MachinePackage struct {
	state          protoimpl.MessageState `protogen:"open.v1"`
	InstallationId string                 `protobuf:"bytes,1,opt,name=installation_id,json=installationId,proto3" json:"installation_id,omitempty"`
	Package        string                 `protobuf:"bytes,2,opt,name=package,proto3" json:"package,omitempty"` // org/name; local/<name> for unpublished code
	Release        string                 `protobuf:"bytes,3,opt,name=release,proto3" json:"release,omitempty"`
	Origin         string                 `protobuf:"bytes,4,opt,name=origin,proto3" json:"origin,omitempty"` // release | local | runtime; readers tolerate others
	InstalledAtMs  uint64                 `protobuf:"varint,5,opt,name=installed_at_ms,json=installedAtMs,proto3" json:"installed_at_ms,omitempty"`
	Sdk            []*ImageDistribution   `protobuf:"bytes,6,rep,name=sdk,proto3" json:"sdk,omitempty"`                 // the Cozy SDK installed in it (cozy-runtime, tensorfs)
	Entrypoints    []string               `protobuf:"bytes,7,rep,name=entrypoints,proto3" json:"entrypoints,omitempty"` // its interface's functions and jobs, sorted
	unknownFields  protoimpl.UnknownFields
	sizeCache      protoimpl.SizeCache
}

func (x *MachinePackage) Reset() {
	*x = MachinePackage{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[42]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *MachinePackage) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*MachinePackage) ProtoMessage() {}

func (x *MachinePackage) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[42]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use MachinePackage.ProtoReflect.Descriptor instead.
func (*MachinePackage) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{42}
}

func (x *MachinePackage) GetInstallationId() string {
	if x != nil {
		return x.InstallationId
	}
	return ""
}

func (x *MachinePackage) GetPackage() string {
	if x != nil {
		return x.Package
	}
	return ""
}

func (x *MachinePackage) GetRelease() string {
	if x != nil {
		return x.Release
	}
	return ""
}

func (x *MachinePackage) GetOrigin() string {
	if x != nil {
		return x.Origin
	}
	return ""
}

func (x *MachinePackage) GetInstalledAtMs() uint64 {
	if x != nil {
		return x.InstalledAtMs
	}
	return 0
}

func (x *MachinePackage) GetSdk() []*ImageDistribution {
	if x != nil {
		return x.Sdk
	}
	return nil
}

func (x *MachinePackage) GetEntrypoints() []string {
	if x != nil {
		return x.Entrypoints
	}
	return nil
}

type ModelListQuery struct {
	state         protoimpl.MessageState `protogen:"open.v1"`
	Claim         *Claim                 `protobuf:"bytes,1,opt,name=claim,proto3" json:"claim,omitempty"`
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *ModelListQuery) Reset() {
	*x = ModelListQuery{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[43]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *ModelListQuery) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*ModelListQuery) ProtoMessage() {}

func (x *ModelListQuery) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[43]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use ModelListQuery.ProtoReflect.Descriptor instead.
func (*ModelListQuery) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{43}
}

func (x *ModelListQuery) GetClaim() *Claim {
	if x != nil {
		return x.Claim
	}
	return nil
}

// The weights this machine holds: its TensorFS store's repository rows and usage, as
// `tfs repo list` and `tfs repo usage` report them.
type ModelList struct {
	state         protoimpl.MessageState `protogen:"open.v1"`
	Models        []*MachineModel        `protobuf:"bytes,1,rep,name=models,proto3" json:"models,omitempty"`             // sorted by repository, version, lane
	Repositories  []*RepositoryUsage     `protobuf:"bytes,2,rep,name=repositories,proto3" json:"repositories,omitempty"` // sorted by repository
	Store         *StoreUsage            `protobuf:"bytes,3,opt,name=store,proto3" json:"store,omitempty"`
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *ModelList) Reset() {
	*x = ModelList{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[44]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *ModelList) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*ModelList) ProtoMessage() {}

func (x *ModelList) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[44]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use ModelList.ProtoReflect.Descriptor instead.
func (*ModelList) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{44}
}

func (x *ModelList) GetModels() []*MachineModel {
	if x != nil {
		return x.Models
	}
	return nil
}

func (x *ModelList) GetRepositories() []*RepositoryUsage {
	if x != nil {
		return x.Repositories
	}
	return nil
}

func (x *ModelList) GetStore() *StoreUsage {
	if x != nil {
		return x.Store
	}
	return nil
}

type MachineModel struct {
	state           protoimpl.MessageState `protogen:"open.v1"`
	Kind            string                 `protobuf:"bytes,1,opt,name=kind,proto3" json:"kind,omitempty"`                                              // release | local; readers tolerate others
	Repository      string                 `protobuf:"bytes,2,opt,name=repository,proto3" json:"repository,omitempty"`                                  // org/name; a local alias is local/<name>
	Version         string                 `protobuf:"bytes,3,opt,name=version,proto3" json:"version,omitempty"`                                        // release rows
	Lane            string                 `protobuf:"bytes,4,opt,name=lane,proto3" json:"lane,omitempty"`                                              // release rows
	SourceSelection string                 `protobuf:"bytes,5,opt,name=source_selection,json=sourceSelection,proto3" json:"source_selection,omitempty"` // local rows: the source it was made from
	Manifest        *Ref                   `protobuf:"bytes,6,opt,name=manifest,proto3" json:"manifest,omitempty"`
	Complete        bool                   `protobuf:"varint,7,opt,name=complete,proto3" json:"complete,omitempty"` // the store holds the manifest's whole closure
	unknownFields   protoimpl.UnknownFields
	sizeCache       protoimpl.SizeCache
}

func (x *MachineModel) Reset() {
	*x = MachineModel{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[45]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *MachineModel) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*MachineModel) ProtoMessage() {}

func (x *MachineModel) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[45]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use MachineModel.ProtoReflect.Descriptor instead.
func (*MachineModel) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{45}
}

func (x *MachineModel) GetKind() string {
	if x != nil {
		return x.Kind
	}
	return ""
}

func (x *MachineModel) GetRepository() string {
	if x != nil {
		return x.Repository
	}
	return ""
}

func (x *MachineModel) GetVersion() string {
	if x != nil {
		return x.Version
	}
	return ""
}

func (x *MachineModel) GetLane() string {
	if x != nil {
		return x.Lane
	}
	return ""
}

func (x *MachineModel) GetSourceSelection() string {
	if x != nil {
		return x.SourceSelection
	}
	return ""
}

func (x *MachineModel) GetManifest() *Ref {
	if x != nil {
		return x.Manifest
	}
	return nil
}

func (x *MachineModel) GetComplete() bool {
	if x != nil {
		return x.Complete
	}
	return false
}

type RepositoryUsage struct {
	state         protoimpl.MessageState `protogen:"open.v1"`
	Repository    string                 `protobuf:"bytes,1,opt,name=repository,proto3" json:"repository,omitempty"`
	BytesTotal    uint64                 `protobuf:"varint,2,opt,name=bytes_total,json=bytesTotal,proto3" json:"bytes_total,omitempty"`    // everything its retained checkpoints reach
	BytesUnique   uint64                 `protobuf:"varint,3,opt,name=bytes_unique,json=bytesUnique,proto3" json:"bytes_unique,omitempty"` // what no other repository reaches: removing it frees this
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *RepositoryUsage) Reset() {
	*x = RepositoryUsage{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[46]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *RepositoryUsage) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*RepositoryUsage) ProtoMessage() {}

func (x *RepositoryUsage) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[46]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use RepositoryUsage.ProtoReflect.Descriptor instead.
func (*RepositoryUsage) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{46}
}

func (x *RepositoryUsage) GetRepository() string {
	if x != nil {
		return x.Repository
	}
	return ""
}

func (x *RepositoryUsage) GetBytesTotal() uint64 {
	if x != nil {
		return x.BytesTotal
	}
	return 0
}

func (x *RepositoryUsage) GetBytesUnique() uint64 {
	if x != nil {
		return x.BytesUnique
	}
	return 0
}

type StoreUsage struct {
	state             protoimpl.MessageState `protogen:"open.v1"`
	BytesTotal        uint64                 `protobuf:"varint,1,opt,name=bytes_total,json=bytesTotal,proto3" json:"bytes_total,omitempty"` // the union every repository reaches
	BytesUniqueSum    uint64                 `protobuf:"varint,2,opt,name=bytes_unique_sum,json=bytesUniqueSum,proto3" json:"bytes_unique_sum,omitempty"`
	BytesUnreferenced uint64                 `protobuf:"varint,3,opt,name=bytes_unreferenced,json=bytesUnreferenced,proto3" json:"bytes_unreferenced,omitempty"` // verified objects no retained manifest reaches
	Filesystem        *MachineFilesystem     `protobuf:"bytes,4,opt,name=filesystem,proto3" json:"filesystem,omitempty"`
	unknownFields     protoimpl.UnknownFields
	sizeCache         protoimpl.SizeCache
}

func (x *StoreUsage) Reset() {
	*x = StoreUsage{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[47]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *StoreUsage) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*StoreUsage) ProtoMessage() {}

func (x *StoreUsage) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[47]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use StoreUsage.ProtoReflect.Descriptor instead.
func (*StoreUsage) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{47}
}

func (x *StoreUsage) GetBytesTotal() uint64 {
	if x != nil {
		return x.BytesTotal
	}
	return 0
}

func (x *StoreUsage) GetBytesUniqueSum() uint64 {
	if x != nil {
		return x.BytesUniqueSum
	}
	return 0
}

func (x *StoreUsage) GetBytesUnreferenced() uint64 {
	if x != nil {
		return x.BytesUnreferenced
	}
	return 0
}

func (x *StoreUsage) GetFilesystem() *MachineFilesystem {
	if x != nil {
		return x.Filesystem
	}
	return nil
}

type MachineFilesystem struct {
	state          protoimpl.MessageState `protogen:"open.v1"`
	Path           string                 `protobuf:"bytes,1,opt,name=path,proto3" json:"path,omitempty"`
	TotalBytes     uint64                 `protobuf:"varint,2,opt,name=total_bytes,json=totalBytes,proto3" json:"total_bytes,omitempty"`
	AvailableBytes uint64                 `protobuf:"varint,3,opt,name=available_bytes,json=availableBytes,proto3" json:"available_bytes,omitempty"`
	unknownFields  protoimpl.UnknownFields
	sizeCache      protoimpl.SizeCache
}

func (x *MachineFilesystem) Reset() {
	*x = MachineFilesystem{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[48]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *MachineFilesystem) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*MachineFilesystem) ProtoMessage() {}

func (x *MachineFilesystem) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[48]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use MachineFilesystem.ProtoReflect.Descriptor instead.
func (*MachineFilesystem) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{48}
}

func (x *MachineFilesystem) GetPath() string {
	if x != nil {
		return x.Path
	}
	return ""
}

func (x *MachineFilesystem) GetTotalBytes() uint64 {
	if x != nil {
		return x.TotalBytes
	}
	return 0
}

func (x *MachineFilesystem) GetAvailableBytes() uint64 {
	if x != nil {
		return x.AvailableBytes
	}
	return 0
}

type DescribeMachineQuery struct {
	state         protoimpl.MessageState `protogen:"open.v1"`
	Claim         *Claim                 `protobuf:"bytes,1,opt,name=claim,proto3" json:"claim,omitempty"`
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *DescribeMachineQuery) Reset() {
	*x = DescribeMachineQuery{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[49]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *DescribeMachineQuery) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*DescribeMachineQuery) ProtoMessage() {}

func (x *DescribeMachineQuery) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[49]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use DescribeMachineQuery.ProtoReflect.Descriptor instead.
func (*DescribeMachineQuery) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{49}
}

func (x *DescribeMachineQuery) GetClaim() *Claim {
	if x != nil {
		return x.Claim
	}
	return nil
}

// Wire 66: what this machine is. Replaces ProtocolInfo, GetMachineExecutionWorkspace's
// inventory and ClaimAck.resources for new clients. A Host fills `host` and forwards its
// Runtime's `runtime`; a Runtime answers `runtime` alone.
type MachineDescription struct {
	state         protoimpl.MessageState `protogen:"open.v1"`
	WorkerId      string                 `protobuf:"bytes,1,opt,name=worker_id,json=workerId,proto3" json:"worker_id,omitempty"`
	WorkerBootId  string                 `protobuf:"bytes,2,opt,name=worker_boot_id,json=workerBootId,proto3" json:"worker_boot_id,omitempty"`
	Host          *MachineHost           `protobuf:"bytes,3,opt,name=host,proto3" json:"host,omitempty"`
	Runtime       *MachineRuntime        `protobuf:"bytes,4,opt,name=runtime,proto3" json:"runtime,omitempty"`
	RuntimeAbsent string                 `protobuf:"bytes,5,opt,name=runtime_absent,json=runtimeAbsent,proto3" json:"runtime_absent,omitempty"` // why `runtime` is absent, for people; empty when present
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *MachineDescription) Reset() {
	*x = MachineDescription{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[50]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *MachineDescription) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*MachineDescription) ProtoMessage() {}

func (x *MachineDescription) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[50]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use MachineDescription.ProtoReflect.Descriptor instead.
func (*MachineDescription) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{50}
}

func (x *MachineDescription) GetWorkerId() string {
	if x != nil {
		return x.WorkerId
	}
	return ""
}

func (x *MachineDescription) GetWorkerBootId() string {
	if x != nil {
		return x.WorkerBootId
	}
	return ""
}

func (x *MachineDescription) GetHost() *MachineHost {
	if x != nil {
		return x.Host
	}
	return nil
}

func (x *MachineDescription) GetRuntime() *MachineRuntime {
	if x != nil {
		return x.Runtime
	}
	return nil
}

func (x *MachineDescription) GetRuntimeAbsent() string {
	if x != nil {
		return x.RuntimeAbsent
	}
	return ""
}

type MachineHost struct {
	state              protoimpl.MessageState `protogen:"open.v1"`
	Version            string                 `protobuf:"bytes,1,opt,name=version,proto3" json:"version,omitempty"` // the Host's release
	WireMinor          uint32                 `protobuf:"varint,2,opt,name=wire_minor,json=wireMinor,proto3" json:"wire_minor,omitempty"`
	MinimumWireMinor   uint32                 `protobuf:"varint,3,opt,name=minimum_wire_minor,json=minimumWireMinor,proto3" json:"minimum_wire_minor,omitempty"`
	Platform           string                 `protobuf:"bytes,4,opt,name=platform,proto3" json:"platform,omitempty"` // os/arch, as WorkerResources.platform
	OsRelease          string                 `protobuf:"bytes,5,opt,name=os_release,json=osRelease,proto3" json:"os_release,omitempty"`
	Hostname           string                 `protobuf:"bytes,6,opt,name=hostname,proto3" json:"hostname,omitempty"`
	Phase              string                 `protobuf:"bytes,7,opt,name=phase,proto3" json:"phase,omitempty"`                                                          // booting | ready | idle | releasing; readers tolerate others
	IdleDeadlineUnixMs uint64                 `protobuf:"varint,8,opt,name=idle_deadline_unix_ms,json=idleDeadlineUnixMs,proto3" json:"idle_deadline_unix_ms,omitempty"` // 0: no idle release pending
	Hubs               []*MachineHub          `protobuf:"bytes,9,rep,name=hubs,proto3" json:"hubs,omitempty"`
	Filesystems        []*MachineFilesystem   `protobuf:"bytes,10,rep,name=filesystems,proto3" json:"filesystems,omitempty"`
	StartedAtUnixMs    uint64                 `protobuf:"varint,11,opt,name=started_at_unix_ms,json=startedAtUnixMs,proto3" json:"started_at_unix_ms,omitempty"`
	unknownFields      protoimpl.UnknownFields
	sizeCache          protoimpl.SizeCache
}

func (x *MachineHost) Reset() {
	*x = MachineHost{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[51]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *MachineHost) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*MachineHost) ProtoMessage() {}

func (x *MachineHost) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[51]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use MachineHost.ProtoReflect.Descriptor instead.
func (*MachineHost) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{51}
}

func (x *MachineHost) GetVersion() string {
	if x != nil {
		return x.Version
	}
	return ""
}

func (x *MachineHost) GetWireMinor() uint32 {
	if x != nil {
		return x.WireMinor
	}
	return 0
}

func (x *MachineHost) GetMinimumWireMinor() uint32 {
	if x != nil {
		return x.MinimumWireMinor
	}
	return 0
}

func (x *MachineHost) GetPlatform() string {
	if x != nil {
		return x.Platform
	}
	return ""
}

func (x *MachineHost) GetOsRelease() string {
	if x != nil {
		return x.OsRelease
	}
	return ""
}

func (x *MachineHost) GetHostname() string {
	if x != nil {
		return x.Hostname
	}
	return ""
}

func (x *MachineHost) GetPhase() string {
	if x != nil {
		return x.Phase
	}
	return ""
}

func (x *MachineHost) GetIdleDeadlineUnixMs() uint64 {
	if x != nil {
		return x.IdleDeadlineUnixMs
	}
	return 0
}

func (x *MachineHost) GetHubs() []*MachineHub {
	if x != nil {
		return x.Hubs
	}
	return nil
}

func (x *MachineHost) GetFilesystems() []*MachineFilesystem {
	if x != nil {
		return x.Filesystems
	}
	return nil
}

func (x *MachineHost) GetStartedAtUnixMs() uint64 {
	if x != nil {
		return x.StartedAtUnixMs
	}
	return 0
}

type MachineHub struct {
	state         protoimpl.MessageState `protogen:"open.v1"`
	Origin        string                 `protobuf:"bytes,1,opt,name=origin,proto3" json:"origin,omitempty"`                        // the Hub's https origin
	MachineId     string                 `protobuf:"bytes,2,opt,name=machine_id,json=machineId,proto3" json:"machine_id,omitempty"` // this machine's worker id at that Hub
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *MachineHub) Reset() {
	*x = MachineHub{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[52]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *MachineHub) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*MachineHub) ProtoMessage() {}

func (x *MachineHub) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[52]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use MachineHub.ProtoReflect.Descriptor instead.
func (*MachineHub) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{52}
}

func (x *MachineHub) GetOrigin() string {
	if x != nil {
		return x.Origin
	}
	return ""
}

func (x *MachineHub) GetMachineId() string {
	if x != nil {
		return x.MachineId
	}
	return ""
}

type MachineRuntime struct {
	state                protoimpl.MessageState `protogen:"open.v1"`
	Version              string                 `protobuf:"bytes,1,opt,name=version,proto3" json:"version,omitempty"` // the cozy-runtime release
	WireMinor            uint32                 `protobuf:"varint,2,opt,name=wire_minor,json=wireMinor,proto3" json:"wire_minor,omitempty"`
	MinimumWireMinor     uint32                 `protobuf:"varint,3,opt,name=minimum_wire_minor,json=minimumWireMinor,proto3" json:"minimum_wire_minor,omitempty"`
	TensorfsVersion      string                 `protobuf:"bytes,4,opt,name=tensorfs_version,json=tensorfsVersion,proto3" json:"tensorfs_version,omitempty"`
	PythonVersion        string                 `protobuf:"bytes,5,opt,name=python_version,json=pythonVersion,proto3" json:"python_version,omitempty"`
	TorchVersion         string                 `protobuf:"bytes,6,opt,name=torch_version,json=torchVersion,proto3" json:"torch_version,omitempty"` // empty: no torch in the Runtime's environment
	UvVersion            string                 `protobuf:"bytes,7,opt,name=uv_version,json=uvVersion,proto3" json:"uv_version,omitempty"`
	AcceleratorBackend   string                 `protobuf:"bytes,8,opt,name=accelerator_backend,json=acceleratorBackend,proto3" json:"accelerator_backend,omitempty"` // cuda | none; empty when unmeasured
	ExecutorUidIsolation bool                   `protobuf:"varint,9,opt,name=executor_uid_isolation,json=executorUidIsolation,proto3" json:"executor_uid_isolation,omitempty"`
	Devices              []*MachineDevice       `protobuf:"bytes,10,rep,name=devices,proto3" json:"devices,omitempty"`
	Resources            *WorkerResources       `protobuf:"bytes,11,opt,name=resources,proto3" json:"resources,omitempty"` // backend_version is the CUDA toolkit torch was built for
	ExecutionWorkspaceId string                 `protobuf:"bytes,12,opt,name=execution_workspace_id,json=executionWorkspaceId,proto3" json:"execution_workspace_id,omitempty"`
	StartedAtUnixMs      uint64                 `protobuf:"varint,13,opt,name=started_at_unix_ms,json=startedAtUnixMs,proto3" json:"started_at_unix_ms,omitempty"`
	Store                *MachineFilesystem     `protobuf:"bytes,14,opt,name=store,proto3" json:"store,omitempty"`               // the TensorFS store's filesystem
	Interpreters         []*PythonInterpreter   `protobuf:"bytes,15,rep,name=interpreters,proto3" json:"interpreters,omitempty"` // the Pythons packages may install against
	// Wire 71: optional current/recent model preparation observations, using the same exact
	// selections and counters as PrepareEvent. Readers may sample without waiting for a
	// preparation lock; multiple observers see the same native work. Empty on an older peer
	// means telemetry unavailable and never refuses preparation or execution.
	PreparationProgress []*PrepareModelProgress `protobuf:"bytes,16,rep,name=preparation_progress,json=preparationProgress,proto3" json:"preparation_progress,omitempty"`
	unknownFields       protoimpl.UnknownFields
	sizeCache           protoimpl.SizeCache
}

func (x *MachineRuntime) Reset() {
	*x = MachineRuntime{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[53]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *MachineRuntime) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*MachineRuntime) ProtoMessage() {}

func (x *MachineRuntime) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[53]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use MachineRuntime.ProtoReflect.Descriptor instead.
func (*MachineRuntime) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{53}
}

func (x *MachineRuntime) GetVersion() string {
	if x != nil {
		return x.Version
	}
	return ""
}

func (x *MachineRuntime) GetWireMinor() uint32 {
	if x != nil {
		return x.WireMinor
	}
	return 0
}

func (x *MachineRuntime) GetMinimumWireMinor() uint32 {
	if x != nil {
		return x.MinimumWireMinor
	}
	return 0
}

func (x *MachineRuntime) GetTensorfsVersion() string {
	if x != nil {
		return x.TensorfsVersion
	}
	return ""
}

func (x *MachineRuntime) GetPythonVersion() string {
	if x != nil {
		return x.PythonVersion
	}
	return ""
}

func (x *MachineRuntime) GetTorchVersion() string {
	if x != nil {
		return x.TorchVersion
	}
	return ""
}

func (x *MachineRuntime) GetUvVersion() string {
	if x != nil {
		return x.UvVersion
	}
	return ""
}

func (x *MachineRuntime) GetAcceleratorBackend() string {
	if x != nil {
		return x.AcceleratorBackend
	}
	return ""
}

func (x *MachineRuntime) GetExecutorUidIsolation() bool {
	if x != nil {
		return x.ExecutorUidIsolation
	}
	return false
}

func (x *MachineRuntime) GetDevices() []*MachineDevice {
	if x != nil {
		return x.Devices
	}
	return nil
}

func (x *MachineRuntime) GetResources() *WorkerResources {
	if x != nil {
		return x.Resources
	}
	return nil
}

func (x *MachineRuntime) GetExecutionWorkspaceId() string {
	if x != nil {
		return x.ExecutionWorkspaceId
	}
	return ""
}

func (x *MachineRuntime) GetStartedAtUnixMs() uint64 {
	if x != nil {
		return x.StartedAtUnixMs
	}
	return 0
}

func (x *MachineRuntime) GetStore() *MachineFilesystem {
	if x != nil {
		return x.Store
	}
	return nil
}

func (x *MachineRuntime) GetInterpreters() []*PythonInterpreter {
	if x != nil {
		return x.Interpreters
	}
	return nil
}

func (x *MachineRuntime) GetPreparationProgress() []*PrepareModelProgress {
	if x != nil {
		return x.PreparationProgress
	}
	return nil
}

// No ownership, credential, readiness or machine state. Callers record the range and gate
// preparation and execution on it; a missing RPC or disjoint range never refuses the connection
// or Claim. Generated constants are the authority. PodHost answers the intersection of its own
// range and its Runtime's, or its own range when they share no minor; it then refuses
// preparation and execution itself, naming the Runtime update.
type ProtocolInfoResult struct {
	state            protoimpl.MessageState `protogen:"open.v1"`
	WireMinor        uint32                 `protobuf:"varint,1,opt,name=wire_minor,json=wireMinor,proto3" json:"wire_minor,omitempty"`
	MinimumWireMinor uint32                 `protobuf:"varint,2,opt,name=minimum_wire_minor,json=minimumWireMinor,proto3" json:"minimum_wire_minor,omitempty"`
	unknownFields    protoimpl.UnknownFields
	sizeCache        protoimpl.SizeCache
}

func (x *ProtocolInfoResult) Reset() {
	*x = ProtocolInfoResult{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[54]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *ProtocolInfoResult) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*ProtocolInfoResult) ProtoMessage() {}

func (x *ProtocolInfoResult) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[54]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use ProtocolInfoResult.ProtoReflect.Descriptor instead.
func (*ProtocolInfoResult) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{54}
}

func (x *ProtocolInfoResult) GetWireMinor() uint32 {
	if x != nil {
		return x.WireMinor
	}
	return 0
}

func (x *ProtocolInfoResult) GetMinimumWireMinor() uint32 {
	if x != nil {
		return x.MinimumWireMinor
	}
	return 0
}

// The three preparations name the same logical sets the retiring DesiredWorkerState modes
// carried; the difference is who sends the result to the worker.
// (xs-019) Beside the signed set ride the facts prepare consumes, every one of them
// hub-known from the release record and the registered image before the pod exists. No package
// file or model file travels; the pod installs from the indexes and resolves model manifests
// from its local TensorFS Store. The release's interface document may travel (field 10).
type PreparePackageSetCall struct {
	state          protoimpl.MessageState `protogen:"open.v1"`
	Claim          *Claim                 `protobuf:"bytes,1,opt,name=claim,proto3" json:"claim,omitempty"`
	PackageSet     *DesiredPackageSet     `protobuf:"bytes,2,opt,name=package_set,json=packageSet,proto3" json:"package_set,omitempty"`
	Application    string                 `protobuf:"bytes,3,opt,name=application,proto3" json:"application,omitempty"`                               // the release interface's application; nonempty
	ModelSlotPaths []string               `protobuf:"bytes,4,rep,name=model_slot_paths,json=modelSlotPaths,proto3" json:"model_slot_paths,omitempty"` // every interface-declared entrypoint model-slot path;
	// sorted unique; <= MaxModelSlotPaths
	ImageInventory     *ImageInventory `protobuf:"bytes,5,opt,name=image_inventory,json=imageInventory,proto3" json:"image_inventory,omitempty"`             // the placed image's exact pinned inventory
	LockedRequirements []byte          `protobuf:"bytes,6,opt,name=locked_requirements,json=lockedRequirements,proto3" json:"locked_requirements,omitempty"` // hash-pinned requirements export of the release's
	// Declared PEP 440 Python range and optional exact captured version.
	// Worker intersects these with its installed rolling supported window before install.
	PythonRequires string `protobuf:"bytes,8,opt,name=python_requires,json=pythonRequires,proto3" json:"python_requires,omitempty"`
	PythonVersion  string `protobuf:"bytes,9,opt,name=python_version,json=pythonVersion,proto3" json:"python_version,omitempty"`
	// The hub release's PackageInterface/1 canonical bytes. Empty: Runtime obtains them itself.
	PackageInterface []byte `protobuf:"bytes,10,opt,name=package_interface,json=packageInterface,proto3" json:"package_interface,omitempty"`
	// Wire 67: the Tensorhub origin the set is read and its Models fetched at; empty: the machine's
	// default Hub.
	Hub           string `protobuf:"bytes,11,opt,name=hub,proto3" json:"hub,omitempty"`
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *PreparePackageSetCall) Reset() {
	*x = PreparePackageSetCall{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[55]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *PreparePackageSetCall) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*PreparePackageSetCall) ProtoMessage() {}

func (x *PreparePackageSetCall) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[55]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use PreparePackageSetCall.ProtoReflect.Descriptor instead.
func (*PreparePackageSetCall) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{55}
}

func (x *PreparePackageSetCall) GetClaim() *Claim {
	if x != nil {
		return x.Claim
	}
	return nil
}

func (x *PreparePackageSetCall) GetPackageSet() *DesiredPackageSet {
	if x != nil {
		return x.PackageSet
	}
	return nil
}

func (x *PreparePackageSetCall) GetApplication() string {
	if x != nil {
		return x.Application
	}
	return ""
}

func (x *PreparePackageSetCall) GetModelSlotPaths() []string {
	if x != nil {
		return x.ModelSlotPaths
	}
	return nil
}

func (x *PreparePackageSetCall) GetImageInventory() *ImageInventory {
	if x != nil {
		return x.ImageInventory
	}
	return nil
}

func (x *PreparePackageSetCall) GetLockedRequirements() []byte {
	if x != nil {
		return x.LockedRequirements
	}
	return nil
}

func (x *PreparePackageSetCall) GetPythonRequires() string {
	if x != nil {
		return x.PythonRequires
	}
	return ""
}

func (x *PreparePackageSetCall) GetPythonVersion() string {
	if x != nil {
		return x.PythonVersion
	}
	return ""
}

func (x *PreparePackageSetCall) GetPackageInterface() []byte {
	if x != nil {
		return x.PackageInterface
	}
	return nil
}

func (x *PreparePackageSetCall) GetHub() string {
	if x != nil {
		return x.Hub
	}
	return ""
}

type PrepareLocalPackageCall struct {
	state           protoimpl.MessageState  `protogen:"open.v1"`
	Claim           *Claim                  `protobuf:"bytes,1,opt,name=claim,proto3" json:"claim,omitempty"`
	LocalPackageSet *DesiredLocalPackageSet `protobuf:"bytes,2,opt,name=local_package_set,json=localPackageSet,proto3" json:"local_package_set,omitempty"`
	// Wire 69: select this machine's scoped Hub grant for account-index dependencies.
	// Empty never borrows an attached persistent-machine grant; a rental may use its own.
	Hub           string `protobuf:"bytes,3,opt,name=hub,proto3" json:"hub,omitempty"`
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *PrepareLocalPackageCall) Reset() {
	*x = PrepareLocalPackageCall{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[56]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *PrepareLocalPackageCall) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*PrepareLocalPackageCall) ProtoMessage() {}

func (x *PrepareLocalPackageCall) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[56]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use PrepareLocalPackageCall.ProtoReflect.Descriptor instead.
func (*PrepareLocalPackageCall) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{56}
}

func (x *PrepareLocalPackageCall) GetClaim() *Claim {
	if x != nil {
		return x.Claim
	}
	return nil
}

func (x *PrepareLocalPackageCall) GetLocalPackageSet() *DesiredLocalPackageSet {
	if x != nil {
		return x.LocalPackageSet
	}
	return nil
}

func (x *PrepareLocalPackageCall) GetHub() string {
	if x != nil {
		return x.Hub
	}
	return ""
}

type PreparePrivatePlacementCall struct {
	state               protoimpl.MessageState      `protogen:"open.v1"`
	Claim               *Claim                      `protobuf:"bytes,1,opt,name=claim,proto3" json:"claim,omitempty"`
	PrivatePlacementSet *DesiredPrivatePlacementSet `protobuf:"bytes,2,opt,name=private_placement_set,json=privatePlacementSet,proto3" json:"private_placement_set,omitempty"`
	unknownFields       protoimpl.UnknownFields
	sizeCache           protoimpl.SizeCache
}

func (x *PreparePrivatePlacementCall) Reset() {
	*x = PreparePrivatePlacementCall{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[57]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *PreparePrivatePlacementCall) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*PreparePrivatePlacementCall) ProtoMessage() {}

func (x *PreparePrivatePlacementCall) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[57]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use PreparePrivatePlacementCall.ProtoReflect.Descriptor instead.
func (*PreparePrivatePlacementCall) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{57}
}

func (x *PreparePrivatePlacementCall) GetClaim() *Claim {
	if x != nil {
		return x.Claim
	}
	return nil
}

func (x *PreparePrivatePlacementCall) GetPrivatePlacementSet() *DesiredPrivatePlacementSet {
	if x != nil {
		return x.PrivatePlacementSet
	}
	return nil
}

// One preparation's observable progress. Byte counters are monotonic within a call; the
// terminal PREPARED event carries the exact DesiredPlacementSet the owner sends next, and the
// terminal REFUSED event carries one bounded safe code and detail.
type PrepareEvent struct {
	state            protoimpl.MessageState  `protogen:"open.v1"`
	Stage            PrepareStage            `protobuf:"varint,1,opt,name=stage,proto3,enum=cozy.worker.v1.PrepareStage" json:"stage,omitempty"`
	TotalBytes       uint64                  `protobuf:"varint,2,opt,name=total_bytes,json=totalBytes,proto3" json:"total_bytes,omitempty"`                   // bytes the bounded plan must land; 0 until RESOLVED
	TransferredBytes uint64                  `protobuf:"varint,3,opt,name=transferred_bytes,json=transferredBytes,proto3" json:"transferred_bytes,omitempty"` // verified so far
	PlacementSet     *DesiredPlacementSet    `protobuf:"bytes,4,opt,name=placement_set,json=placementSet,proto3" json:"placement_set,omitempty"`              // PREPARED only
	SafeCode         string                  `protobuf:"bytes,5,opt,name=safe_code,json=safeCode,proto3" json:"safe_code,omitempty"`                          // REFUSED only
	SafeDetail       string                  `protobuf:"bytes,6,opt,name=safe_detail,json=safeDetail,proto3" json:"safe_detail,omitempty"`                    // REFUSED only; bounded <= 1024 bytes, sanitized
	ModelProgress    []*PrepareModelProgress `protobuf:"bytes,7,rep,name=model_progress,json=modelProgress,proto3" json:"model_progress,omitempty"`           // complete current model rows;
	// keep completed rows through PREPARED. Optional telemetry.
	InstalledPackage *InstalledPackage `protobuf:"bytes,8,opt,name=installed_package,json=installedPackage,proto3" json:"installed_package,omitempty"` // PREPARED; worker-installed callable metadata
	unknownFields    protoimpl.UnknownFields
	sizeCache        protoimpl.SizeCache
}

func (x *PrepareEvent) Reset() {
	*x = PrepareEvent{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[58]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *PrepareEvent) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*PrepareEvent) ProtoMessage() {}

func (x *PrepareEvent) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[58]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use PrepareEvent.ProtoReflect.Descriptor instead.
func (*PrepareEvent) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{58}
}

func (x *PrepareEvent) GetStage() PrepareStage {
	if x != nil {
		return x.Stage
	}
	return PrepareStage_PREPARE_STAGE_UNSPECIFIED
}

func (x *PrepareEvent) GetTotalBytes() uint64 {
	if x != nil {
		return x.TotalBytes
	}
	return 0
}

func (x *PrepareEvent) GetTransferredBytes() uint64 {
	if x != nil {
		return x.TransferredBytes
	}
	return 0
}

func (x *PrepareEvent) GetPlacementSet() *DesiredPlacementSet {
	if x != nil {
		return x.PlacementSet
	}
	return nil
}

func (x *PrepareEvent) GetSafeCode() string {
	if x != nil {
		return x.SafeCode
	}
	return ""
}

func (x *PrepareEvent) GetSafeDetail() string {
	if x != nil {
		return x.SafeDetail
	}
	return ""
}

func (x *PrepareEvent) GetModelProgress() []*PrepareModelProgress {
	if x != nil {
		return x.ModelProgress
	}
	return nil
}

func (x *PrepareEvent) GetInstalledPackage() *InstalledPackage {
	if x != nil {
		return x.InstalledPackage
	}
	return nil
}

// Per-model availability, including objects held before this preparation. Counters are
// observations, not custody or billing authority. Different manifests may share objects:
// summing these rows does not measure unique network traffic. Zero total means unknown.
type PrepareModelProgress struct {
	state            protoimpl.MessageState `protogen:"open.v1"`
	Model            *DownloadModelRef      `protobuf:"bytes,1,opt,name=model,proto3" json:"model,omitempty"`                                                // exact existing selection; no second identity scheme
	TotalBytes       uint64                 `protobuf:"varint,2,opt,name=total_bytes,json=totalBytes,proto3" json:"total_bytes,omitempty"`                   // native TensorFS plan's declared object bytes
	TransferredBytes uint64                 `protobuf:"varint,3,opt,name=transferred_bytes,json=transferredBytes,proto3" json:"transferred_bytes,omitempty"` // held + newly verified origin/cache bytes for this model
	OriginBytes      uint64                 `protobuf:"varint,4,opt,name=origin_bytes,json=originBytes,proto3" json:"origin_bytes,omitempty"`                // newly fetched from origin during this preparation
	CachedBytes      uint64                 `protobuf:"varint,5,opt,name=cached_bytes,json=cachedBytes,proto3" json:"cached_bytes,omitempty"`                // newly fetched from cache during this preparation
	// Verified bytes newly published to the mounted cache for this exact runtime closure.
	// A lower bound when replication is incomplete; never inferred from origin/held bytes.
	CacheWrittenBytes uint64 `protobuf:"varint,6,opt,name=cache_written_bytes,json=cacheWrittenBytes,proto3" json:"cache_written_bytes,omitempty"`
	unknownFields     protoimpl.UnknownFields
	sizeCache         protoimpl.SizeCache
}

func (x *PrepareModelProgress) Reset() {
	*x = PrepareModelProgress{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[59]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *PrepareModelProgress) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*PrepareModelProgress) ProtoMessage() {}

func (x *PrepareModelProgress) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[59]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use PrepareModelProgress.ProtoReflect.Descriptor instead.
func (*PrepareModelProgress) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{59}
}

func (x *PrepareModelProgress) GetModel() *DownloadModelRef {
	if x != nil {
		return x.Model
	}
	return nil
}

func (x *PrepareModelProgress) GetTotalBytes() uint64 {
	if x != nil {
		return x.TotalBytes
	}
	return 0
}

func (x *PrepareModelProgress) GetTransferredBytes() uint64 {
	if x != nil {
		return x.TransferredBytes
	}
	return 0
}

func (x *PrepareModelProgress) GetOriginBytes() uint64 {
	if x != nil {
		return x.OriginBytes
	}
	return 0
}

func (x *PrepareModelProgress) GetCachedBytes() uint64 {
	if x != nil {
		return x.CachedBytes
	}
	return 0
}

func (x *PrepareModelProgress) GetCacheWrittenBytes() uint64 {
	if x != nil {
		return x.CacheWrittenBytes
	}
	return 0
}

type ModelSourceFileCall struct {
	state         protoimpl.MessageState  `protogen:"open.v1"`
	Claim         *Claim                  `protobuf:"bytes,1,opt,name=claim,proto3" json:"claim,omitempty"`
	Request       *ModelSourceFileRequest `protobuf:"bytes,2,opt,name=request,proto3" json:"request,omitempty"`
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *ModelSourceFileCall) Reset() {
	*x = ModelSourceFileCall{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[60]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *ModelSourceFileCall) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*ModelSourceFileCall) ProtoMessage() {}

func (x *ModelSourceFileCall) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[60]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use ModelSourceFileCall.ProtoReflect.Descriptor instead.
func (*ModelSourceFileCall) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{60}
}

func (x *ModelSourceFileCall) GetClaim() *Claim {
	if x != nil {
		return x.Claim
	}
	return nil
}

func (x *ModelSourceFileCall) GetRequest() *ModelSourceFileRequest {
	if x != nil {
		return x.Request
	}
	return nil
}

type ModelSourcePrepareCall struct {
	state         protoimpl.MessageState     `protogen:"open.v1"`
	Claim         *Claim                     `protobuf:"bytes,1,opt,name=claim,proto3" json:"claim,omitempty"`
	Request       *ModelSourcePrepareRequest `protobuf:"bytes,2,opt,name=request,proto3" json:"request,omitempty"`
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *ModelSourcePrepareCall) Reset() {
	*x = ModelSourcePrepareCall{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[61]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *ModelSourcePrepareCall) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*ModelSourcePrepareCall) ProtoMessage() {}

func (x *ModelSourcePrepareCall) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[61]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use ModelSourcePrepareCall.ProtoReflect.Descriptor instead.
func (*ModelSourcePrepareCall) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{61}
}

func (x *ModelSourcePrepareCall) GetClaim() *Claim {
	if x != nil {
		return x.Claim
	}
	return nil
}

func (x *ModelSourcePrepareCall) GetRequest() *ModelSourcePrepareRequest {
	if x != nil {
		return x.Request
	}
	return nil
}

// Permanent owner disposition, never a pause. Host journals the tombstone,
// cancels outstanding fetches, drains entered conversion, then releases this
// operation's native roots. Other source operations are unaffected.
type ModelSourceReleaseCall struct {
	state                 protoimpl.MessageState `protogen:"open.v1"`
	Claim                 *Claim                 `protobuf:"bytes,1,opt,name=claim,proto3" json:"claim,omitempty"`
	OperationId           string                 `protobuf:"bytes,2,opt,name=operation_id,json=operationId,proto3" json:"operation_id,omitempty"`
	SourceSelectionDigest []byte                 `protobuf:"bytes,3,opt,name=source_selection_digest,json=sourceSelectionDigest,proto3" json:"source_selection_digest,omitempty"`
	unknownFields         protoimpl.UnknownFields
	sizeCache             protoimpl.SizeCache
}

func (x *ModelSourceReleaseCall) Reset() {
	*x = ModelSourceReleaseCall{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[62]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *ModelSourceReleaseCall) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*ModelSourceReleaseCall) ProtoMessage() {}

func (x *ModelSourceReleaseCall) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[62]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use ModelSourceReleaseCall.ProtoReflect.Descriptor instead.
func (*ModelSourceReleaseCall) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{62}
}

func (x *ModelSourceReleaseCall) GetClaim() *Claim {
	if x != nil {
		return x.Claim
	}
	return nil
}

func (x *ModelSourceReleaseCall) GetOperationId() string {
	if x != nil {
		return x.OperationId
	}
	return ""
}

func (x *ModelSourceReleaseCall) GetSourceSelectionDigest() []byte {
	if x != nil {
		return x.SourceSelectionDigest
	}
	return nil
}

// Loopback-only Host -> Runtime. The Host has already checked signed ownership
// and exact source selection; Runtime serializes against native preparation.
type ReleaseModelSourceRequest struct {
	state         protoimpl.MessageState `protogen:"open.v1"`
	OperationId   string                 `protobuf:"bytes,1,opt,name=operation_id,json=operationId,proto3" json:"operation_id,omitempty"`
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *ReleaseModelSourceRequest) Reset() {
	*x = ReleaseModelSourceRequest{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[63]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *ReleaseModelSourceRequest) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*ReleaseModelSourceRequest) ProtoMessage() {}

func (x *ReleaseModelSourceRequest) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[63]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use ReleaseModelSourceRequest.ProtoReflect.Descriptor instead.
func (*ReleaseModelSourceRequest) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{63}
}

func (x *ReleaseModelSourceRequest) GetOperationId() string {
	if x != nil {
		return x.OperationId
	}
	return ""
}

// Independent result custody preserves the original native receipt and
// manifest. The RecordOwner assigns retention_id; it never enters author payloads.
// tensorfs_receipt_digest hashes the NATIVE canonical TensorFS receipt, not its
// enclosing cozy.worker.v1.WeightsReceipt document. Runtime verifies it natively.
// Read-only actual numerical platform identity. It excludes caller package code,
// machine UUID, custody/attempt clocks, paths, and credentials. Missing proof
// refuses this probe; callers may execute without cross-request result reuse.
type NumericalEnvironmentRequest struct {
	state         protoimpl.MessageState `protogen:"open.v1"`
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *NumericalEnvironmentRequest) Reset() {
	*x = NumericalEnvironmentRequest{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[64]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *NumericalEnvironmentRequest) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*NumericalEnvironmentRequest) ProtoMessage() {}

func (x *NumericalEnvironmentRequest) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[64]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use NumericalEnvironmentRequest.ProtoReflect.Descriptor instead.
func (*NumericalEnvironmentRequest) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{64}
}

// Owner control revision orders non-releasing source pause/resume. A paused
// acknowledgement proves entered source work drained; native roots stay retained.
type ModelSourceControlCall struct {
	state                 protoimpl.MessageState `protogen:"open.v1"`
	Claim                 *Claim                 `protobuf:"bytes,1,opt,name=claim,proto3" json:"claim,omitempty"`
	OperationId           string                 `protobuf:"bytes,2,opt,name=operation_id,json=operationId,proto3" json:"operation_id,omitempty"`
	SourceSelectionDigest []byte                 `protobuf:"bytes,3,opt,name=source_selection_digest,json=sourceSelectionDigest,proto3" json:"source_selection_digest,omitempty"`
	ControlRevision       uint64                 `protobuf:"varint,4,opt,name=control_revision,json=controlRevision,proto3" json:"control_revision,omitempty"`
	Paused                bool                   `protobuf:"varint,5,opt,name=paused,proto3" json:"paused,omitempty"`
	unknownFields         protoimpl.UnknownFields
	sizeCache             protoimpl.SizeCache
}

func (x *ModelSourceControlCall) Reset() {
	*x = ModelSourceControlCall{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[65]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *ModelSourceControlCall) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*ModelSourceControlCall) ProtoMessage() {}

func (x *ModelSourceControlCall) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[65]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use ModelSourceControlCall.ProtoReflect.Descriptor instead.
func (*ModelSourceControlCall) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{65}
}

func (x *ModelSourceControlCall) GetClaim() *Claim {
	if x != nil {
		return x.Claim
	}
	return nil
}

func (x *ModelSourceControlCall) GetOperationId() string {
	if x != nil {
		return x.OperationId
	}
	return ""
}

func (x *ModelSourceControlCall) GetSourceSelectionDigest() []byte {
	if x != nil {
		return x.SourceSelectionDigest
	}
	return nil
}

func (x *ModelSourceControlCall) GetControlRevision() uint64 {
	if x != nil {
		return x.ControlRevision
	}
	return 0
}

func (x *ModelSourceControlCall) GetPaused() bool {
	if x != nil {
		return x.Paused
	}
	return false
}

type ModelSourceControlResult struct {
	state                 protoimpl.MessageState `protogen:"open.v1"`
	OperationId           string                 `protobuf:"bytes,1,opt,name=operation_id,json=operationId,proto3" json:"operation_id,omitempty"`
	SourceSelectionDigest []byte                 `protobuf:"bytes,2,opt,name=source_selection_digest,json=sourceSelectionDigest,proto3" json:"source_selection_digest,omitempty"`
	ControlRevision       uint64                 `protobuf:"varint,3,opt,name=control_revision,json=controlRevision,proto3" json:"control_revision,omitempty"`
	Paused                bool                   `protobuf:"varint,4,opt,name=paused,proto3" json:"paused,omitempty"`
	unknownFields         protoimpl.UnknownFields
	sizeCache             protoimpl.SizeCache
}

func (x *ModelSourceControlResult) Reset() {
	*x = ModelSourceControlResult{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[66]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *ModelSourceControlResult) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*ModelSourceControlResult) ProtoMessage() {}

func (x *ModelSourceControlResult) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[66]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use ModelSourceControlResult.ProtoReflect.Descriptor instead.
func (*ModelSourceControlResult) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{66}
}

func (x *ModelSourceControlResult) GetOperationId() string {
	if x != nil {
		return x.OperationId
	}
	return ""
}

func (x *ModelSourceControlResult) GetSourceSelectionDigest() []byte {
	if x != nil {
		return x.SourceSelectionDigest
	}
	return nil
}

func (x *ModelSourceControlResult) GetControlRevision() uint64 {
	if x != nil {
		return x.ControlRevision
	}
	return 0
}

func (x *ModelSourceControlResult) GetPaused() bool {
	if x != nil {
		return x.Paused
	}
	return false
}

type NumericalEnvironmentCall struct {
	state         protoimpl.MessageState `protogen:"open.v1"`
	Claim         *Claim                 `protobuf:"bytes,1,opt,name=claim,proto3" json:"claim,omitempty"`
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *NumericalEnvironmentCall) Reset() {
	*x = NumericalEnvironmentCall{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[67]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *NumericalEnvironmentCall) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*NumericalEnvironmentCall) ProtoMessage() {}

func (x *NumericalEnvironmentCall) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[67]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use NumericalEnvironmentCall.ProtoReflect.Descriptor instead.
func (*NumericalEnvironmentCall) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{67}
}

func (x *NumericalEnvironmentCall) GetClaim() *Claim {
	if x != nil {
		return x.Claim
	}
	return nil
}

type NumericalEnvironmentResult struct {
	state         protoimpl.MessageState `protogen:"open.v1"`
	Digest        []byte                 `protobuf:"bytes,1,opt,name=digest,proto3" json:"digest,omitempty"`
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *NumericalEnvironmentResult) Reset() {
	*x = NumericalEnvironmentResult{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[68]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *NumericalEnvironmentResult) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*NumericalEnvironmentResult) ProtoMessage() {}

func (x *NumericalEnvironmentResult) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[68]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use NumericalEnvironmentResult.ProtoReflect.Descriptor instead.
func (*NumericalEnvironmentResult) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{68}
}

func (x *NumericalEnvironmentResult) GetDigest() []byte {
	if x != nil {
		return x.Digest
	}
	return nil
}

type DerivedRetentionCall struct {
	state         protoimpl.MessageState   `protogen:"open.v1"`
	Claim         *Claim                   `protobuf:"bytes,1,opt,name=claim,proto3" json:"claim,omitempty"`
	Request       *DerivedRetentionRequest `protobuf:"bytes,2,opt,name=request,proto3" json:"request,omitempty"`
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *DerivedRetentionCall) Reset() {
	*x = DerivedRetentionCall{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[69]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *DerivedRetentionCall) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*DerivedRetentionCall) ProtoMessage() {}

func (x *DerivedRetentionCall) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[69]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use DerivedRetentionCall.ProtoReflect.Descriptor instead.
func (*DerivedRetentionCall) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{69}
}

func (x *DerivedRetentionCall) GetClaim() *Claim {
	if x != nil {
		return x.Claim
	}
	return nil
}

func (x *DerivedRetentionCall) GetRequest() *DerivedRetentionRequest {
	if x != nil {
		return x.Request
	}
	return nil
}

type DerivedRetentionRequest struct {
	state                 protoimpl.MessageState `protogen:"open.v1"`
	WeightsTransactionId  string                 `protobuf:"bytes,1,opt,name=weights_transaction_id,json=weightsTransactionId,proto3" json:"weights_transaction_id,omitempty"`
	TensorfsReceiptDigest []byte                 `protobuf:"bytes,2,opt,name=tensorfs_receipt_digest,json=tensorfsReceiptDigest,proto3" json:"tensorfs_receipt_digest,omitempty"`
	RetentionId           string                 `protobuf:"bytes,3,opt,name=retention_id,json=retentionId,proto3" json:"retention_id,omitempty"`
	unknownFields         protoimpl.UnknownFields
	sizeCache             protoimpl.SizeCache
}

func (x *DerivedRetentionRequest) Reset() {
	*x = DerivedRetentionRequest{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[70]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *DerivedRetentionRequest) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*DerivedRetentionRequest) ProtoMessage() {}

func (x *DerivedRetentionRequest) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[70]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use DerivedRetentionRequest.ProtoReflect.Descriptor instead.
func (*DerivedRetentionRequest) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{70}
}

func (x *DerivedRetentionRequest) GetWeightsTransactionId() string {
	if x != nil {
		return x.WeightsTransactionId
	}
	return ""
}

func (x *DerivedRetentionRequest) GetTensorfsReceiptDigest() []byte {
	if x != nil {
		return x.TensorfsReceiptDigest
	}
	return nil
}

func (x *DerivedRetentionRequest) GetRetentionId() string {
	if x != nil {
		return x.RetentionId
	}
	return ""
}

type DerivedRetentionResult struct {
	state                 protoimpl.MessageState `protogen:"open.v1"`
	WeightsTransactionId  string                 `protobuf:"bytes,1,opt,name=weights_transaction_id,json=weightsTransactionId,proto3" json:"weights_transaction_id,omitempty"`
	TensorfsReceiptDigest []byte                 `protobuf:"bytes,2,opt,name=tensorfs_receipt_digest,json=tensorfsReceiptDigest,proto3" json:"tensorfs_receipt_digest,omitempty"`
	RetentionId           string                 `protobuf:"bytes,3,opt,name=retention_id,json=retentionId,proto3" json:"retention_id,omitempty"`
	Manifest              *Ref                   `protobuf:"bytes,4,opt,name=manifest,proto3" json:"manifest,omitempty"` // absent only when release won before retention existed
	Released              bool                   `protobuf:"varint,5,opt,name=released,proto3" json:"released,omitempty"`
	unknownFields         protoimpl.UnknownFields
	sizeCache             protoimpl.SizeCache
}

func (x *DerivedRetentionResult) Reset() {
	*x = DerivedRetentionResult{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[71]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *DerivedRetentionResult) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*DerivedRetentionResult) ProtoMessage() {}

func (x *DerivedRetentionResult) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[71]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use DerivedRetentionResult.ProtoReflect.Descriptor instead.
func (*DerivedRetentionResult) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{71}
}

func (x *DerivedRetentionResult) GetWeightsTransactionId() string {
	if x != nil {
		return x.WeightsTransactionId
	}
	return ""
}

func (x *DerivedRetentionResult) GetTensorfsReceiptDigest() []byte {
	if x != nil {
		return x.TensorfsReceiptDigest
	}
	return nil
}

func (x *DerivedRetentionResult) GetRetentionId() string {
	if x != nil {
		return x.RetentionId
	}
	return ""
}

func (x *DerivedRetentionResult) GetManifest() *Ref {
	if x != nil {
		return x.Manifest
	}
	return nil
}

func (x *DerivedRetentionResult) GetReleased() bool {
	if x != nil {
		return x.Released
	}
	return false
}

// Release the original retained result only after dependent requests have
// independently acquired custody. This does not rewrite its first finalization
// decision or native receipt, and does not release independent retention roots.
type DerivedResultReleaseCall struct {
	state         protoimpl.MessageState       `protogen:"open.v1"`
	Claim         *Claim                       `protobuf:"bytes,1,opt,name=claim,proto3" json:"claim,omitempty"`
	Request       *DerivedResultReleaseRequest `protobuf:"bytes,2,opt,name=request,proto3" json:"request,omitempty"`
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *DerivedResultReleaseCall) Reset() {
	*x = DerivedResultReleaseCall{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[72]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *DerivedResultReleaseCall) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*DerivedResultReleaseCall) ProtoMessage() {}

func (x *DerivedResultReleaseCall) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[72]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use DerivedResultReleaseCall.ProtoReflect.Descriptor instead.
func (*DerivedResultReleaseCall) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{72}
}

func (x *DerivedResultReleaseCall) GetClaim() *Claim {
	if x != nil {
		return x.Claim
	}
	return nil
}

func (x *DerivedResultReleaseCall) GetRequest() *DerivedResultReleaseRequest {
	if x != nil {
		return x.Request
	}
	return nil
}

type DerivedResultReleaseRequest struct {
	state                 protoimpl.MessageState `protogen:"open.v1"`
	WeightsTransactionId  string                 `protobuf:"bytes,1,opt,name=weights_transaction_id,json=weightsTransactionId,proto3" json:"weights_transaction_id,omitempty"`
	TensorfsReceiptDigest []byte                 `protobuf:"bytes,2,opt,name=tensorfs_receipt_digest,json=tensorfsReceiptDigest,proto3" json:"tensorfs_receipt_digest,omitempty"`
	unknownFields         protoimpl.UnknownFields
	sizeCache             protoimpl.SizeCache
}

func (x *DerivedResultReleaseRequest) Reset() {
	*x = DerivedResultReleaseRequest{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[73]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *DerivedResultReleaseRequest) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*DerivedResultReleaseRequest) ProtoMessage() {}

func (x *DerivedResultReleaseRequest) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[73]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use DerivedResultReleaseRequest.ProtoReflect.Descriptor instead.
func (*DerivedResultReleaseRequest) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{73}
}

func (x *DerivedResultReleaseRequest) GetWeightsTransactionId() string {
	if x != nil {
		return x.WeightsTransactionId
	}
	return ""
}

func (x *DerivedResultReleaseRequest) GetTensorfsReceiptDigest() []byte {
	if x != nil {
		return x.TensorfsReceiptDigest
	}
	return nil
}

type DerivedResultReleaseResult struct {
	state                 protoimpl.MessageState `protogen:"open.v1"`
	WeightsTransactionId  string                 `protobuf:"bytes,1,opt,name=weights_transaction_id,json=weightsTransactionId,proto3" json:"weights_transaction_id,omitempty"`
	TensorfsReceiptDigest []byte                 `protobuf:"bytes,2,opt,name=tensorfs_receipt_digest,json=tensorfsReceiptDigest,proto3" json:"tensorfs_receipt_digest,omitempty"`
	Released              bool                   `protobuf:"varint,3,opt,name=released,proto3" json:"released,omitempty"`
	unknownFields         protoimpl.UnknownFields
	sizeCache             protoimpl.SizeCache
}

func (x *DerivedResultReleaseResult) Reset() {
	*x = DerivedResultReleaseResult{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[74]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *DerivedResultReleaseResult) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*DerivedResultReleaseResult) ProtoMessage() {}

func (x *DerivedResultReleaseResult) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[74]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use DerivedResultReleaseResult.ProtoReflect.Descriptor instead.
func (*DerivedResultReleaseResult) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{74}
}

func (x *DerivedResultReleaseResult) GetWeightsTransactionId() string {
	if x != nil {
		return x.WeightsTransactionId
	}
	return ""
}

func (x *DerivedResultReleaseResult) GetTensorfsReceiptDigest() []byte {
	if x != nil {
		return x.TensorfsReceiptDigest
	}
	return nil
}

func (x *DerivedResultReleaseResult) GetReleased() bool {
	if x != nil {
		return x.Released
	}
	return false
}

// The trusted RecordOwner computes this key from the frozen operation, its
// content inputs/parameters and measured numerical environment. No result bytes
// or cache namespace are supplied: the Host verifies its own successful terminal.
type RecordOperationResultCall struct {
	state                protoimpl.MessageState `protogen:"open.v1"`
	Claim                *Claim                 `protobuf:"bytes,1,opt,name=claim,proto3" json:"claim,omitempty"`
	ComputationDigest    []byte                 `protobuf:"bytes,2,opt,name=computation_digest,json=computationDigest,proto3" json:"computation_digest,omitempty"`
	RequestId            string                 `protobuf:"bytes,3,opt,name=request_id,json=requestId,proto3" json:"request_id,omitempty"`
	AttemptOrdinal       uint64                 `protobuf:"varint,4,opt,name=attempt_ordinal,json=attemptOrdinal,proto3" json:"attempt_ordinal,omitempty"`
	InvocationSpecDigest []byte                 `protobuf:"bytes,5,opt,name=invocation_spec_digest,json=invocationSpecDigest,proto3" json:"invocation_spec_digest,omitempty"`
	OutcomeId            string                 `protobuf:"bytes,6,opt,name=outcome_id,json=outcomeId,proto3" json:"outcome_id,omitempty"`
	OutcomeDigest        []byte                 `protobuf:"bytes,7,opt,name=outcome_digest,json=outcomeDigest,proto3" json:"outcome_digest,omitempty"`
	unknownFields        protoimpl.UnknownFields
	sizeCache            protoimpl.SizeCache
}

func (x *RecordOperationResultCall) Reset() {
	*x = RecordOperationResultCall{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[75]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *RecordOperationResultCall) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*RecordOperationResultCall) ProtoMessage() {}

func (x *RecordOperationResultCall) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[75]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use RecordOperationResultCall.ProtoReflect.Descriptor instead.
func (*RecordOperationResultCall) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{75}
}

func (x *RecordOperationResultCall) GetClaim() *Claim {
	if x != nil {
		return x.Claim
	}
	return nil
}

func (x *RecordOperationResultCall) GetComputationDigest() []byte {
	if x != nil {
		return x.ComputationDigest
	}
	return nil
}

func (x *RecordOperationResultCall) GetRequestId() string {
	if x != nil {
		return x.RequestId
	}
	return ""
}

func (x *RecordOperationResultCall) GetAttemptOrdinal() uint64 {
	if x != nil {
		return x.AttemptOrdinal
	}
	return 0
}

func (x *RecordOperationResultCall) GetInvocationSpecDigest() []byte {
	if x != nil {
		return x.InvocationSpecDigest
	}
	return nil
}

func (x *RecordOperationResultCall) GetOutcomeId() string {
	if x != nil {
		return x.OutcomeId
	}
	return ""
}

func (x *RecordOperationResultCall) GetOutcomeDigest() []byte {
	if x != nil {
		return x.OutcomeDigest
	}
	return nil
}

type RecordOperationResultResult struct {
	state             protoimpl.MessageState `protogen:"open.v1"`
	ComputationDigest []byte                 `protobuf:"bytes,1,opt,name=computation_digest,json=computationDigest,proto3" json:"computation_digest,omitempty"`
	Recorded          bool                   `protobuf:"varint,2,opt,name=recorded,proto3" json:"recorded,omitempty"`
	unknownFields     protoimpl.UnknownFields
	sizeCache         protoimpl.SizeCache
}

func (x *RecordOperationResultResult) Reset() {
	*x = RecordOperationResultResult{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[76]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *RecordOperationResultResult) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*RecordOperationResultResult) ProtoMessage() {}

func (x *RecordOperationResultResult) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[76]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use RecordOperationResultResult.ProtoReflect.Descriptor instead.
func (*RecordOperationResultResult) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{76}
}

func (x *RecordOperationResultResult) GetComputationDigest() []byte {
	if x != nil {
		return x.ComputationDigest
	}
	return nil
}

func (x *RecordOperationResultResult) GetRecorded() bool {
	if x != nil {
		return x.Recorded
	}
	return false
}

// consumer_request_id is an ownership recipient, never part of the computation
// key. A hit acquires its independent native roots before returning. Replay is
// bound to that recipient even if the cache entry is subsequently evicted.
type LookupOperationCall struct {
	state             protoimpl.MessageState `protogen:"open.v1"`
	Claim             *Claim                 `protobuf:"bytes,1,opt,name=claim,proto3" json:"claim,omitempty"`
	ComputationDigest []byte                 `protobuf:"bytes,2,opt,name=computation_digest,json=computationDigest,proto3" json:"computation_digest,omitempty"`
	ConsumerRequestId string                 `protobuf:"bytes,3,opt,name=consumer_request_id,json=consumerRequestId,proto3" json:"consumer_request_id,omitempty"`
	unknownFields     protoimpl.UnknownFields
	sizeCache         protoimpl.SizeCache
}

func (x *LookupOperationCall) Reset() {
	*x = LookupOperationCall{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[77]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *LookupOperationCall) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*LookupOperationCall) ProtoMessage() {}

func (x *LookupOperationCall) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[77]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use LookupOperationCall.ProtoReflect.Descriptor instead.
func (*LookupOperationCall) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{77}
}

func (x *LookupOperationCall) GetClaim() *Claim {
	if x != nil {
		return x.Claim
	}
	return nil
}

func (x *LookupOperationCall) GetComputationDigest() []byte {
	if x != nil {
		return x.ComputationDigest
	}
	return nil
}

func (x *LookupOperationCall) GetConsumerRequestId() string {
	if x != nil {
		return x.ConsumerRequestId
	}
	return ""
}

type LookupOperationResult struct {
	state             protoimpl.MessageState       `protogen:"open.v1"`
	ComputationDigest []byte                       `protobuf:"bytes,1,opt,name=computation_digest,json=computationDigest,proto3" json:"computation_digest,omitempty"`
	Found             bool                         `protobuf:"varint,2,opt,name=found,proto3" json:"found,omitempty"`
	Source            *AttemptOutcome              `protobuf:"bytes,3,opt,name=source,proto3" json:"source,omitempty"`
	Retentions        []*DerivedRetentionResult    `protobuf:"bytes,4,rep,name=retentions,proto3" json:"retentions,omitempty"`
	ConsumerRequestId string                       `protobuf:"bytes,5,opt,name=consumer_request_id,json=consumerRequestId,proto3" json:"consumer_request_id,omitempty"`
	ByteRetentions    []*NativeByteRetentionResult `protobuf:"bytes,7,rep,name=byte_retentions,json=byteRetentions,proto3" json:"byte_retentions,omitempty"` // combined recipient count <=32
	unknownFields     protoimpl.UnknownFields
	sizeCache         protoimpl.SizeCache
}

func (x *LookupOperationResult) Reset() {
	*x = LookupOperationResult{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[78]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *LookupOperationResult) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*LookupOperationResult) ProtoMessage() {}

func (x *LookupOperationResult) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[78]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use LookupOperationResult.ProtoReflect.Descriptor instead.
func (*LookupOperationResult) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{78}
}

func (x *LookupOperationResult) GetComputationDigest() []byte {
	if x != nil {
		return x.ComputationDigest
	}
	return nil
}

func (x *LookupOperationResult) GetFound() bool {
	if x != nil {
		return x.Found
	}
	return false
}

func (x *LookupOperationResult) GetSource() *AttemptOutcome {
	if x != nil {
		return x.Source
	}
	return nil
}

func (x *LookupOperationResult) GetRetentions() []*DerivedRetentionResult {
	if x != nil {
		return x.Retentions
	}
	return nil
}

func (x *LookupOperationResult) GetConsumerRequestId() string {
	if x != nil {
		return x.ConsumerRequestId
	}
	return ""
}

func (x *LookupOperationResult) GetByteRetentions() []*NativeByteRetentionResult {
	if x != nil {
		return x.ByteRetentions
	}
	return nil
}

// Explicit workspace maintenance releases only unused cache-owned roots.
// Live request/lookup roots and active native readers remain authoritative.
type PruneOperationCacheCall struct {
	state         protoimpl.MessageState `protogen:"open.v1"`
	Claim         *Claim                 `protobuf:"bytes,1,opt,name=claim,proto3" json:"claim,omitempty"`
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *PruneOperationCacheCall) Reset() {
	*x = PruneOperationCacheCall{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[79]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *PruneOperationCacheCall) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*PruneOperationCacheCall) ProtoMessage() {}

func (x *PruneOperationCacheCall) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[79]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use PruneOperationCacheCall.ProtoReflect.Descriptor instead.
func (*PruneOperationCacheCall) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{79}
}

func (x *PruneOperationCacheCall) GetClaim() *Claim {
	if x != nil {
		return x.Claim
	}
	return nil
}

type PruneOperationCacheResult struct {
	state          protoimpl.MessageState `protogen:"open.v1"`
	RemovedEntries uint32                 `protobuf:"varint,1,opt,name=removed_entries,json=removedEntries,proto3" json:"removed_entries,omitempty"`
	ReclaimedBytes uint64                 `protobuf:"varint,2,opt,name=reclaimed_bytes,json=reclaimedBytes,proto3" json:"reclaimed_bytes,omitempty"`
	StoreBusy      bool                   `protobuf:"varint,3,opt,name=store_busy,json=storeBusy,proto3" json:"store_busy,omitempty"` // root pruning completed; native byte collection deferred
	unknownFields  protoimpl.UnknownFields
	sizeCache      protoimpl.SizeCache
}

func (x *PruneOperationCacheResult) Reset() {
	*x = PruneOperationCacheResult{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[80]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *PruneOperationCacheResult) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*PruneOperationCacheResult) ProtoMessage() {}

func (x *PruneOperationCacheResult) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[80]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use PruneOperationCacheResult.ProtoReflect.Descriptor instead.
func (*PruneOperationCacheResult) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{80}
}

func (x *PruneOperationCacheResult) GetRemovedEntries() uint32 {
	if x != nil {
		return x.RemovedEntries
	}
	return 0
}

func (x *PruneOperationCacheResult) GetReclaimedBytes() uint64 {
	if x != nil {
		return x.ReclaimedBytes
	}
	return 0
}

func (x *PruneOperationCacheResult) GetStoreBusy() bool {
	if x != nil {
		return x.StoreBusy
	}
	return false
}

type CollectStoreGarbageRequest struct {
	state         protoimpl.MessageState `protogen:"open.v1"`
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *CollectStoreGarbageRequest) Reset() {
	*x = CollectStoreGarbageRequest{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[81]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *CollectStoreGarbageRequest) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*CollectStoreGarbageRequest) ProtoMessage() {}

func (x *CollectStoreGarbageRequest) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[81]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use CollectStoreGarbageRequest.ProtoReflect.Descriptor instead.
func (*CollectStoreGarbageRequest) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{81}
}

type CollectStoreGarbageResult struct {
	state          protoimpl.MessageState `protogen:"open.v1"`
	ReclaimedBytes uint64                 `protobuf:"varint,1,opt,name=reclaimed_bytes,json=reclaimedBytes,proto3" json:"reclaimed_bytes,omitempty"`
	StoreBusy      bool                   `protobuf:"varint,2,opt,name=store_busy,json=storeBusy,proto3" json:"store_busy,omitempty"`
	unknownFields  protoimpl.UnknownFields
	sizeCache      protoimpl.SizeCache
}

func (x *CollectStoreGarbageResult) Reset() {
	*x = CollectStoreGarbageResult{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[82]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *CollectStoreGarbageResult) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*CollectStoreGarbageResult) ProtoMessage() {}

func (x *CollectStoreGarbageResult) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[82]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use CollectStoreGarbageResult.ProtoReflect.Descriptor instead.
func (*CollectStoreGarbageResult) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{82}
}

func (x *CollectStoreGarbageResult) GetReclaimedBytes() uint64 {
	if x != nil {
		return x.ReclaimedBytes
	}
	return 0
}

func (x *CollectStoreGarbageResult) GetStoreBusy() bool {
	if x != nil {
		return x.StoreBusy
	}
	return false
}

type ReleaseModelSourceResult struct {
	state         protoimpl.MessageState `protogen:"open.v1"`
	OperationId   string                 `protobuf:"bytes,1,opt,name=operation_id,json=operationId,proto3" json:"operation_id,omitempty"`
	Released      bool                   `protobuf:"varint,2,opt,name=released,proto3" json:"released,omitempty"`
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *ReleaseModelSourceResult) Reset() {
	*x = ReleaseModelSourceResult{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[83]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *ReleaseModelSourceResult) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*ReleaseModelSourceResult) ProtoMessage() {}

func (x *ReleaseModelSourceResult) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[83]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use ReleaseModelSourceResult.ProtoReflect.Descriptor instead.
func (*ReleaseModelSourceResult) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{83}
}

func (x *ReleaseModelSourceResult) GetOperationId() string {
	if x != nil {
		return x.OperationId
	}
	return ""
}

func (x *ReleaseModelSourceResult) GetReleased() bool {
	if x != nil {
		return x.Released
	}
	return false
}

// The signed owner adopts retained source work into a different ordinary request
// on this pod. Both operations must declare the same exact source selection;
// TensorFS independently validates the old checkpoint chain and newly derived
// profile plan, creates a new operation chain, and retains its roots before the
// old owner may release. No provider reads or incomplete conversion starts here.
type ModelSourceAdoptCall struct {
	state           protoimpl.MessageState     `protogen:"open.v1"`
	Claim           *Claim                     `protobuf:"bytes,1,opt,name=claim,proto3" json:"claim,omitempty"`
	FromOperationId string                     `protobuf:"bytes,2,opt,name=from_operation_id,json=fromOperationId,proto3" json:"from_operation_id,omitempty"`
	Request         *ModelSourcePrepareRequest `protobuf:"bytes,3,opt,name=request,proto3" json:"request,omitempty"`
	unknownFields   protoimpl.UnknownFields
	sizeCache       protoimpl.SizeCache
}

func (x *ModelSourceAdoptCall) Reset() {
	*x = ModelSourceAdoptCall{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[84]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *ModelSourceAdoptCall) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*ModelSourceAdoptCall) ProtoMessage() {}

func (x *ModelSourceAdoptCall) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[84]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use ModelSourceAdoptCall.ProtoReflect.Descriptor instead.
func (*ModelSourceAdoptCall) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{84}
}

func (x *ModelSourceAdoptCall) GetClaim() *Claim {
	if x != nil {
		return x.Claim
	}
	return nil
}

func (x *ModelSourceAdoptCall) GetFromOperationId() string {
	if x != nil {
		return x.FromOperationId
	}
	return ""
}

func (x *ModelSourceAdoptCall) GetRequest() *ModelSourcePrepareRequest {
	if x != nil {
		return x.Request
	}
	return nil
}

type CheckpointPageCall struct {
	state         protoimpl.MessageState `protogen:"open.v1"`
	Claim         *Claim                 `protobuf:"bytes,1,opt,name=claim,proto3" json:"claim,omitempty"`
	Request       *CheckpointPageRequest `protobuf:"bytes,2,opt,name=request,proto3" json:"request,omitempty"`
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *CheckpointPageCall) Reset() {
	*x = CheckpointPageCall{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[85]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *CheckpointPageCall) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*CheckpointPageCall) ProtoMessage() {}

func (x *CheckpointPageCall) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[85]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use CheckpointPageCall.ProtoReflect.Descriptor instead.
func (*CheckpointPageCall) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{85}
}

func (x *CheckpointPageCall) GetClaim() *Claim {
	if x != nil {
		return x.Claim
	}
	return nil
}

func (x *CheckpointPageCall) GetRequest() *CheckpointPageRequest {
	if x != nil {
		return x.Request
	}
	return nil
}

type CheckpointTransferCall struct {
	state         protoimpl.MessageState     `protogen:"open.v1"`
	Claim         *Claim                     `protobuf:"bytes,1,opt,name=claim,proto3" json:"claim,omitempty"`
	Request       *CheckpointTransferRequest `protobuf:"bytes,2,opt,name=request,proto3" json:"request,omitempty"`
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *CheckpointTransferCall) Reset() {
	*x = CheckpointTransferCall{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[86]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *CheckpointTransferCall) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*CheckpointTransferCall) ProtoMessage() {}

func (x *CheckpointTransferCall) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[86]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use CheckpointTransferCall.ProtoReflect.Descriptor instead.
func (*CheckpointTransferCall) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{86}
}

func (x *CheckpointTransferCall) GetClaim() *Claim {
	if x != nil {
		return x.Claim
	}
	return nil
}

func (x *CheckpointTransferCall) GetRequest() *CheckpointTransferRequest {
	if x != nil {
		return x.Request
	}
	return nil
}

// Creator uploads private wheels directly over this dedicated PodHost
// byte stream. Code never enters WorkerControl or a Tensorhub publication/store.
// Header is first and occurs exactly once. Host reserves the exact file and
// replies with its durable received_bytes before accepting chunks. Each chunk
// starts at that exact offset, is at most 1 MiB, and receives a durable progress
// response. VERIFIED follows complete file digest verification. Disconnect keeps
// the prefix; replay the header to learn the resume offset. Existing operation
// file/count/aggregate limits and PrepareLocalPackage remain the custody contract.
type LocalPackageUploadFrame struct {
	state protoimpl.MessageState `protogen:"open.v1"`
	// Types that are valid to be assigned to Body:
	//
	//	*LocalPackageUploadFrame_Header
	//	*LocalPackageUploadFrame_Chunk
	Body          isLocalPackageUploadFrame_Body `protobuf_oneof:"body"`
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *LocalPackageUploadFrame) Reset() {
	*x = LocalPackageUploadFrame{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[87]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *LocalPackageUploadFrame) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*LocalPackageUploadFrame) ProtoMessage() {}

func (x *LocalPackageUploadFrame) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[87]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use LocalPackageUploadFrame.ProtoReflect.Descriptor instead.
func (*LocalPackageUploadFrame) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{87}
}

func (x *LocalPackageUploadFrame) GetBody() isLocalPackageUploadFrame_Body {
	if x != nil {
		return x.Body
	}
	return nil
}

func (x *LocalPackageUploadFrame) GetHeader() *LocalPackageUploadHeader {
	if x != nil {
		if x, ok := x.Body.(*LocalPackageUploadFrame_Header); ok {
			return x.Header
		}
	}
	return nil
}

func (x *LocalPackageUploadFrame) GetChunk() *LocalPackageUploadChunk {
	if x != nil {
		if x, ok := x.Body.(*LocalPackageUploadFrame_Chunk); ok {
			return x.Chunk
		}
	}
	return nil
}

type isLocalPackageUploadFrame_Body interface {
	isLocalPackageUploadFrame_Body()
}

type LocalPackageUploadFrame_Header struct {
	Header *LocalPackageUploadHeader `protobuf:"bytes,1,opt,name=header,proto3,oneof"`
}

type LocalPackageUploadFrame_Chunk struct {
	Chunk *LocalPackageUploadChunk `protobuf:"bytes,2,opt,name=chunk,proto3,oneof"`
}

func (*LocalPackageUploadFrame_Header) isLocalPackageUploadFrame_Body() {}

func (*LocalPackageUploadFrame_Chunk) isLocalPackageUploadFrame_Body() {}

type LocalPackageUploadHeader struct {
	state         protoimpl.MessageState `protogen:"open.v1"`
	Claim         *Claim                 `protobuf:"bytes,1,opt,name=claim,proto3" json:"claim,omitempty"`
	OperationId   string                 `protobuf:"bytes,2,opt,name=operation_id,json=operationId,proto3" json:"operation_id,omitempty"`
	File          *LocalPackageFileRef   `protobuf:"bytes,4,opt,name=file,proto3" json:"file,omitempty"`
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *LocalPackageUploadHeader) Reset() {
	*x = LocalPackageUploadHeader{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[88]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *LocalPackageUploadHeader) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*LocalPackageUploadHeader) ProtoMessage() {}

func (x *LocalPackageUploadHeader) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[88]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use LocalPackageUploadHeader.ProtoReflect.Descriptor instead.
func (*LocalPackageUploadHeader) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{88}
}

func (x *LocalPackageUploadHeader) GetClaim() *Claim {
	if x != nil {
		return x.Claim
	}
	return nil
}

func (x *LocalPackageUploadHeader) GetOperationId() string {
	if x != nil {
		return x.OperationId
	}
	return ""
}

func (x *LocalPackageUploadHeader) GetFile() *LocalPackageFileRef {
	if x != nil {
		return x.File
	}
	return nil
}

type LocalPackageUploadChunk struct {
	state         protoimpl.MessageState `protogen:"open.v1"`
	Offset        uint64                 `protobuf:"varint,1,opt,name=offset,proto3" json:"offset,omitempty"`
	Data          []byte                 `protobuf:"bytes,2,opt,name=data,proto3" json:"data,omitempty"`
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *LocalPackageUploadChunk) Reset() {
	*x = LocalPackageUploadChunk{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[89]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *LocalPackageUploadChunk) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*LocalPackageUploadChunk) ProtoMessage() {}

func (x *LocalPackageUploadChunk) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[89]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use LocalPackageUploadChunk.ProtoReflect.Descriptor instead.
func (*LocalPackageUploadChunk) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{89}
}

func (x *LocalPackageUploadChunk) GetOffset() uint64 {
	if x != nil {
		return x.Offset
	}
	return 0
}

func (x *LocalPackageUploadChunk) GetData() []byte {
	if x != nil {
		return x.Data
	}
	return nil
}

// The loopback form of PreparePackageSetCall. The retired per-file handoff
// (`LocalDownloadFile` with `LocalDownloadKind` PROJECT_WHEEL/DEPENDENCY_WHEEL/MODEL_MANIFEST/
// MODEL_OBJECT/PACKAGE_INTERFACE) is DELETED, not renamed; the names must not return. Runtime
// preparation no longer receives files whose final owner is TensorFS or the package
// index/install materializer.
type PreparePackageSetRequest struct {
	state              protoimpl.MessageState `protogen:"open.v1"`
	DownloadDelegation []byte                 `protobuf:"bytes,1,opt,name=download_delegation,json=downloadDelegation,proto3" json:"download_delegation,omitempty"` // exact canonical DownloadDelegation/1 bytes
	Application        string                 `protobuf:"bytes,4,opt,name=application,proto3" json:"application,omitempty"`                                         // as PreparePackageSetCall fields 3-6
	ModelSlotPaths     []string               `protobuf:"bytes,5,rep,name=model_slot_paths,json=modelSlotPaths,proto3" json:"model_slot_paths,omitempty"`
	ImageInventory     *ImageInventory        `protobuf:"bytes,6,opt,name=image_inventory,json=imageInventory,proto3" json:"image_inventory,omitempty"`
	LockedRequirements []byte                 `protobuf:"bytes,7,opt,name=locked_requirements,json=lockedRequirements,proto3" json:"locked_requirements,omitempty"`
	InstallRoot        string                 `protobuf:"bytes,8,opt,name=install_root,json=installRoot,proto3" json:"install_root,omitempty"` // absolute immutable install destination
	// The delegation's other half. The same 64-byte Ed25519 signature by the rental
	// Creator key that DesiredPackageSet.download_delegation_signature carries, forwarded
	// unchanged by the pod host. Runtime's TensorFS presents `delegation <payload> <signature>`
	// as its hub credential on the tensorfs closure and presign routes; the payload alone does
	// not authenticate.
	DownloadDelegationSignature []byte `protobuf:"bytes,9,opt,name=download_delegation_signature,json=downloadDelegationSignature,proto3" json:"download_delegation_signature,omitempty"`
	// Declared PEP 440 Python range and optional exact captured version.
	// Worker intersects these with its installed rolling supported window before install.
	PythonRequires string `protobuf:"bytes,10,opt,name=python_requires,json=pythonRequires,proto3" json:"python_requires,omitempty"`
	PythonVersion  string `protobuf:"bytes,11,opt,name=python_version,json=pythonVersion,proto3" json:"python_version,omitempty"`
	// PreparePackageSetCall.package_interface, forwarded unchanged by PodHost.
	PackageInterface []byte `protobuf:"bytes,12,opt,name=package_interface,json=packageInterface,proto3" json:"package_interface,omitempty"`
	// Additive: the Host is still landing this set's models in the Store. Runtime installs the
	// release, reads its interface and starts that installation's warm phase (executor start,
	// kernel probes); it reads no model and answers installed_package only. The Host calls
	// again without it once every model is verified in the Store.
	ModelsLanding bool `protobuf:"varint,13,opt,name=models_landing,json=modelsLanding,proto3" json:"models_landing,omitempty"`
	// Wire 67: as PreparePackageSetCall.hub.
	Hub           string `protobuf:"bytes,14,opt,name=hub,proto3" json:"hub,omitempty"`
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *PreparePackageSetRequest) Reset() {
	*x = PreparePackageSetRequest{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[90]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *PreparePackageSetRequest) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*PreparePackageSetRequest) ProtoMessage() {}

func (x *PreparePackageSetRequest) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[90]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use PreparePackageSetRequest.ProtoReflect.Descriptor instead.
func (*PreparePackageSetRequest) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{90}
}

func (x *PreparePackageSetRequest) GetDownloadDelegation() []byte {
	if x != nil {
		return x.DownloadDelegation
	}
	return nil
}

func (x *PreparePackageSetRequest) GetApplication() string {
	if x != nil {
		return x.Application
	}
	return ""
}

func (x *PreparePackageSetRequest) GetModelSlotPaths() []string {
	if x != nil {
		return x.ModelSlotPaths
	}
	return nil
}

func (x *PreparePackageSetRequest) GetImageInventory() *ImageInventory {
	if x != nil {
		return x.ImageInventory
	}
	return nil
}

func (x *PreparePackageSetRequest) GetLockedRequirements() []byte {
	if x != nil {
		return x.LockedRequirements
	}
	return nil
}

func (x *PreparePackageSetRequest) GetInstallRoot() string {
	if x != nil {
		return x.InstallRoot
	}
	return ""
}

func (x *PreparePackageSetRequest) GetDownloadDelegationSignature() []byte {
	if x != nil {
		return x.DownloadDelegationSignature
	}
	return nil
}

func (x *PreparePackageSetRequest) GetPythonRequires() string {
	if x != nil {
		return x.PythonRequires
	}
	return ""
}

func (x *PreparePackageSetRequest) GetPythonVersion() string {
	if x != nil {
		return x.PythonVersion
	}
	return ""
}

func (x *PreparePackageSetRequest) GetPackageInterface() []byte {
	if x != nil {
		return x.PackageInterface
	}
	return nil
}

func (x *PreparePackageSetRequest) GetModelsLanding() bool {
	if x != nil {
		return x.ModelsLanding
	}
	return false
}

func (x *PreparePackageSetRequest) GetHub() string {
	if x != nil {
		return x.Hub
	}
	return ""
}

// The placed worker image's pinned inventory (`tensorhub.image_inventory/1` facts). The list
// the image was built from describes reuse opportunities. Package dependencies are
// installed into their own isolated environment; an inventory version is not a
// constraint on that environment. Native/platform compatibility is checked separately.
type ImageInventory struct {
	state         protoimpl.MessageState `protogen:"open.v1"`
	Profile       string                 `protobuf:"bytes,1,opt,name=profile,proto3" json:"profile,omitempty"`             // the registered base image profile
	Python        string                 `protobuf:"bytes,2,opt,name=python,proto3" json:"python,omitempty"`               // exact interpreter version the image ships
	Distributions []*ImageDistribution   `protobuf:"bytes,3,rep,name=distributions,proto3" json:"distributions,omitempty"` // sorted unique by distribution;
	// <= MaxImageInventoryDistributions
	Interpreters  []*PythonInterpreter `protobuf:"bytes,4,rep,name=interpreters,proto3" json:"interpreters,omitempty"` // installed execution versions, sorted unique
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *ImageInventory) Reset() {
	*x = ImageInventory{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[91]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *ImageInventory) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*ImageInventory) ProtoMessage() {}

func (x *ImageInventory) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[91]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use ImageInventory.ProtoReflect.Descriptor instead.
func (*ImageInventory) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{91}
}

func (x *ImageInventory) GetProfile() string {
	if x != nil {
		return x.Profile
	}
	return ""
}

func (x *ImageInventory) GetPython() string {
	if x != nil {
		return x.Python
	}
	return ""
}

func (x *ImageInventory) GetDistributions() []*ImageDistribution {
	if x != nil {
		return x.Distributions
	}
	return nil
}

func (x *ImageInventory) GetInterpreters() []*PythonInterpreter {
	if x != nil {
		return x.Interpreters
	}
	return nil
}

// Observed standard CPython execution interpreter; paths remain worker-local.
type PythonInterpreter struct {
	state         protoimpl.MessageState `protogen:"open.v1"`
	Version       string                 `protobuf:"bytes,1,opt,name=version,proto3" json:"version,omitempty"` // exact major.minor.patch
	Abi           string                 `protobuf:"bytes,2,opt,name=abi,proto3" json:"abi,omitempty"`         // cp312, cp313, cp314
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *PythonInterpreter) Reset() {
	*x = PythonInterpreter{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[92]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *PythonInterpreter) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*PythonInterpreter) ProtoMessage() {}

func (x *PythonInterpreter) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[92]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use PythonInterpreter.ProtoReflect.Descriptor instead.
func (*PythonInterpreter) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{92}
}

func (x *PythonInterpreter) GetVersion() string {
	if x != nil {
		return x.Version
	}
	return ""
}

func (x *PythonInterpreter) GetAbi() string {
	if x != nil {
		return x.Abi
	}
	return ""
}

type ImageDistribution struct {
	state         protoimpl.MessageState `protogen:"open.v1"`
	Distribution  string                 `protobuf:"bytes,1,opt,name=distribution,proto3" json:"distribution,omitempty"` // normalized lowercase distribution name
	Version       string                 `protobuf:"bytes,2,opt,name=version,proto3" json:"version,omitempty"`           // the exact == pin the image promises
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *ImageDistribution) Reset() {
	*x = ImageDistribution{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[93]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *ImageDistribution) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*ImageDistribution) ProtoMessage() {}

func (x *ImageDistribution) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[93]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use ImageDistribution.ProtoReflect.Descriptor instead.
func (*ImageDistribution) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{93}
}

func (x *ImageDistribution) GetDistribution() string {
	if x != nil {
		return x.Distribution
	}
	return ""
}

func (x *ImageDistribution) GetVersion() string {
	if x != nil {
		return x.Version
	}
	return ""
}

type PreparePackageSetResult struct {
	state            protoimpl.MessageState `protogen:"open.v1"`
	PlacementSet     *DesiredPlacementSet   `protobuf:"bytes,1,opt,name=placement_set,json=placementSet,proto3" json:"placement_set,omitempty"`
	InstalledPackage *InstalledPackage      `protobuf:"bytes,2,opt,name=installed_package,json=installedPackage,proto3" json:"installed_package,omitempty"`
	unknownFields    protoimpl.UnknownFields
	sizeCache        protoimpl.SizeCache
}

func (x *PreparePackageSetResult) Reset() {
	*x = PreparePackageSetResult{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[94]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *PreparePackageSetResult) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*PreparePackageSetResult) ProtoMessage() {}

func (x *PreparePackageSetResult) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[94]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use PreparePackageSetResult.ProtoReflect.Descriptor instead.
func (*PreparePackageSetResult) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{94}
}

func (x *PreparePackageSetResult) GetPlacementSet() *DesiredPlacementSet {
	if x != nil {
		return x.PlacementSet
	}
	return nil
}

func (x *PreparePackageSetResult) GetInstalledPackage() *InstalledPackage {
	if x != nil {
		return x.InstalledPackage
	}
	return nil
}

// Loopback-only source or wheel installation. Host resolves each operation-scoped
// carrier to a local path. Source archives contain the project and its relocated
// editable dependencies; uv owns dependency resolution and ordinary wheel integrity.
type PrepareLocalPackageRequest struct {
	state       protoimpl.MessageState `protogen:"open.v1"`
	OperationId string                 `protobuf:"bytes,1,opt,name=operation_id,json=operationId,proto3" json:"operation_id,omitempty"`
	Package     *DevelopmentPackage    `protobuf:"bytes,2,opt,name=package,proto3" json:"package,omitempty"`
	// Replays use operation_id and the retained installation, never package contents.
	Files       []*LocalPackageFile `protobuf:"bytes,5,rep,name=files,proto3" json:"files,omitempty"`                                // sorted unique by filename; <= MaxLocalPackageFiles
	InstallRoot string              `protobuf:"bytes,6,opt,name=install_root,json=installRoot,proto3" json:"install_root,omitempty"` // absolute immutable install destination
	// Optional registry dependency selection, byte-identical to the
	// DesiredLocalPackageSet input; <= MaxLockedRequirementsBytes. Standard hashed
	// requirements rows, using the published installer grammar. No credentials,
	// floating versions or local paths; no distribution may also be in wheels.
	DependencyRequirements []byte `protobuf:"bytes,7,opt,name=dependency_requirements,json=dependencyRequirements,proto3" json:"dependency_requirements,omitempty"`
	// Declared PEP 440 Python range and optional exact captured version.
	// Worker intersects these with its installed rolling supported window before install.
	PythonRequires string `protobuf:"bytes,8,opt,name=python_requires,json=pythonRequires,proto3" json:"python_requires,omitempty"`
	PythonVersion  string `protobuf:"bytes,9,opt,name=python_version,json=pythonVersion,proto3" json:"python_version,omitempty"`
	SourceArchive  string `protobuf:"bytes,10,opt,name=source_archive,json=sourceArchive,proto3" json:"source_archive,omitempty"` // filename in files; extracted project for ordinary uv sync
	// Wire 69: the explicit scoped Hub selected by PrepareLocalPackageCall.
	Hub           string `protobuf:"bytes,11,opt,name=hub,proto3" json:"hub,omitempty"`
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *PrepareLocalPackageRequest) Reset() {
	*x = PrepareLocalPackageRequest{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[95]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *PrepareLocalPackageRequest) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*PrepareLocalPackageRequest) ProtoMessage() {}

func (x *PrepareLocalPackageRequest) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[95]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use PrepareLocalPackageRequest.ProtoReflect.Descriptor instead.
func (*PrepareLocalPackageRequest) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{95}
}

func (x *PrepareLocalPackageRequest) GetOperationId() string {
	if x != nil {
		return x.OperationId
	}
	return ""
}

func (x *PrepareLocalPackageRequest) GetPackage() *DevelopmentPackage {
	if x != nil {
		return x.Package
	}
	return nil
}

func (x *PrepareLocalPackageRequest) GetFiles() []*LocalPackageFile {
	if x != nil {
		return x.Files
	}
	return nil
}

func (x *PrepareLocalPackageRequest) GetInstallRoot() string {
	if x != nil {
		return x.InstallRoot
	}
	return ""
}

func (x *PrepareLocalPackageRequest) GetDependencyRequirements() []byte {
	if x != nil {
		return x.DependencyRequirements
	}
	return nil
}

func (x *PrepareLocalPackageRequest) GetPythonRequires() string {
	if x != nil {
		return x.PythonRequires
	}
	return ""
}

func (x *PrepareLocalPackageRequest) GetPythonVersion() string {
	if x != nil {
		return x.PythonVersion
	}
	return ""
}

func (x *PrepareLocalPackageRequest) GetSourceArchive() string {
	if x != nil {
		return x.SourceArchive
	}
	return ""
}

func (x *PrepareLocalPackageRequest) GetHub() string {
	if x != nil {
		return x.Hub
	}
	return ""
}

type LocalPackageFile struct {
	state         protoimpl.MessageState `protogen:"open.v1"`
	Digest        []byte                 `protobuf:"bytes,1,opt,name=digest,proto3" json:"digest,omitempty"` // normal wheel integrity only; empty for source_archive
	Filename      string                 `protobuf:"bytes,2,opt,name=filename,proto3" json:"filename,omitempty"`
	Length        uint64                 `protobuf:"varint,3,opt,name=length,proto3" json:"length,omitempty"`
	Path          string                 `protobuf:"bytes,4,opt,name=path,proto3" json:"path,omitempty"` // absolute verified supervisor-owned file
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *LocalPackageFile) Reset() {
	*x = LocalPackageFile{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[96]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *LocalPackageFile) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*LocalPackageFile) ProtoMessage() {}

func (x *LocalPackageFile) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[96]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use LocalPackageFile.ProtoReflect.Descriptor instead.
func (*LocalPackageFile) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{96}
}

func (x *LocalPackageFile) GetDigest() []byte {
	if x != nil {
		return x.Digest
	}
	return nil
}

func (x *LocalPackageFile) GetFilename() string {
	if x != nil {
		return x.Filename
	}
	return ""
}

func (x *LocalPackageFile) GetLength() uint64 {
	if x != nil {
		return x.Length
	}
	return 0
}

func (x *LocalPackageFile) GetPath() string {
	if x != nil {
		return x.Path
	}
	return ""
}

// Loopback-only model binding for an already installed package. TensorFS owns
// model custody; installation_id selects Runtime's retained environment.
type PreparePrivatePlacementRequest struct {
	state              protoimpl.MessageState `protogen:"open.v1"`
	OperationId        string                 `protobuf:"bytes,1,opt,name=operation_id,json=operationId,proto3" json:"operation_id,omitempty"`
	DownloadDelegation []byte                 `protobuf:"bytes,3,opt,name=download_delegation,json=downloadDelegation,proto3" json:"download_delegation,omitempty"` // exact model-only canonical DownloadDelegation/1 bytes
	// The same 64-byte Ed25519 signature by the rental Creator key that
	// DesiredPrivatePlacementSet.download_delegation_signature carries, forwarded unchanged by the
	// pod host so Runtime's TensorFS can present `delegation <payload> <signature>` to the hub's
	// tensorfs routes.
	DownloadDelegationSignature []byte `protobuf:"bytes,5,opt,name=download_delegation_signature,json=downloadDelegationSignature,proto3" json:"download_delegation_signature,omitempty"`
	// Native bindings are already retained in the worker's Store and never cause a registry
	// query or download. download_delegation may independently bind other slots in the same
	// set. The combined selection must name each declared slot at most once.
	NativeModels []*NativeModelBinding `protobuf:"bytes,6,rep,name=native_models,json=nativeModels,proto3" json:"native_models,omitempty"`
	// Required in the native arm: exact existing outer PodHost Claim. Runtime verifies
	// its current owner/epoch and proof before every native custody lookup or cached reply.
	Claim          *Claim `protobuf:"bytes,7,opt,name=claim,proto3" json:"claim,omitempty"`
	InstallationId string `protobuf:"bytes,8,opt,name=installation_id,json=installationId,proto3" json:"installation_id,omitempty"`
	// Wire 70: caller selections resolved by Runtime before returning the fixed serving
	// binding. These override the corresponding base slot without changing its custody.
	ModelChoices      []*ModelChoice      `protobuf:"bytes,9,rep,name=model_choices,json=modelChoices,proto3" json:"model_choices,omitempty"`
	SourceCredentials []*SourceCredential `protobuf:"bytes,10,rep,name=source_credentials,json=sourceCredentials,proto3" json:"source_credentials,omitempty"` // memory-only, never retained or logged
	Hub               string              `protobuf:"bytes,11,opt,name=hub,proto3" json:"hub,omitempty"`                                                      // selects an existing scoped grant, never confers authority
	Owner             string              `protobuf:"bytes,12,opt,name=owner,proto3" json:"owner,omitempty"`                                                  // account for unpublished org-relative model defaults
	unknownFields     protoimpl.UnknownFields
	sizeCache         protoimpl.SizeCache
}

func (x *PreparePrivatePlacementRequest) Reset() {
	*x = PreparePrivatePlacementRequest{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[97]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *PreparePrivatePlacementRequest) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*PreparePrivatePlacementRequest) ProtoMessage() {}

func (x *PreparePrivatePlacementRequest) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[97]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use PreparePrivatePlacementRequest.ProtoReflect.Descriptor instead.
func (*PreparePrivatePlacementRequest) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{97}
}

func (x *PreparePrivatePlacementRequest) GetOperationId() string {
	if x != nil {
		return x.OperationId
	}
	return ""
}

func (x *PreparePrivatePlacementRequest) GetDownloadDelegation() []byte {
	if x != nil {
		return x.DownloadDelegation
	}
	return nil
}

func (x *PreparePrivatePlacementRequest) GetDownloadDelegationSignature() []byte {
	if x != nil {
		return x.DownloadDelegationSignature
	}
	return nil
}

func (x *PreparePrivatePlacementRequest) GetNativeModels() []*NativeModelBinding {
	if x != nil {
		return x.NativeModels
	}
	return nil
}

func (x *PreparePrivatePlacementRequest) GetClaim() *Claim {
	if x != nil {
		return x.Claim
	}
	return nil
}

func (x *PreparePrivatePlacementRequest) GetInstallationId() string {
	if x != nil {
		return x.InstallationId
	}
	return ""
}

func (x *PreparePrivatePlacementRequest) GetModelChoices() []*ModelChoice {
	if x != nil {
		return x.ModelChoices
	}
	return nil
}

func (x *PreparePrivatePlacementRequest) GetSourceCredentials() []*SourceCredential {
	if x != nil {
		return x.SourceCredentials
	}
	return nil
}

func (x *PreparePrivatePlacementRequest) GetHub() string {
	if x != nil {
		return x.Hub
	}
	return ""
}

func (x *PreparePrivatePlacementRequest) GetOwner() string {
	if x != nil {
		return x.Owner
	}
	return ""
}

type LocalModelSourceFile struct {
	state         protoimpl.MessageState `protogen:"open.v1"`
	Member        string                 `protobuf:"bytes,1,opt,name=member,proto3" json:"member,omitempty"`                     // safe relative provider member, <= MaxModelSourceMemberBytes
	ObjectId      string                 `protobuf:"bytes,2,opt,name=object_id,json=objectId,proto3" json:"object_id,omitempty"` // lowercase sha256:<64 hex>
	Length        uint64                 `protobuf:"varint,3,opt,name=length,proto3" json:"length,omitempty"`
	Path          string                 `protobuf:"bytes,4,opt,name=path,proto3" json:"path,omitempty"`                               // absolute supervisor-owned payload destination; may be absent until verified
	HeaderPath    string                 `protobuf:"bytes,5,opt,name=header_path,json=headerPath,proto3" json:"header_path,omitempty"` // absolute supervisor-owned bounded header file; loopback only
	Verified      bool                   `protobuf:"varint,6,opt,name=verified,proto3" json:"verified,omitempty"`                      // this exact full payload has passed supervisor digest/length admission
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *LocalModelSourceFile) Reset() {
	*x = LocalModelSourceFile{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[98]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *LocalModelSourceFile) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*LocalModelSourceFile) ProtoMessage() {}

func (x *LocalModelSourceFile) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[98]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use LocalModelSourceFile.ProtoReflect.Descriptor instead.
func (*LocalModelSourceFile) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{98}
}

func (x *LocalModelSourceFile) GetMember() string {
	if x != nil {
		return x.Member
	}
	return ""
}

func (x *LocalModelSourceFile) GetObjectId() string {
	if x != nil {
		return x.ObjectId
	}
	return ""
}

func (x *LocalModelSourceFile) GetLength() uint64 {
	if x != nil {
		return x.Length
	}
	return 0
}

func (x *LocalModelSourceFile) GetPath() string {
	if x != nil {
		return x.Path
	}
	return ""
}

func (x *LocalModelSourceFile) GetHeaderPath() string {
	if x != nil {
		return x.HeaderPath
	}
	return ""
}

func (x *LocalModelSourceFile) GetVerified() bool {
	if x != nil {
		return x.Verified
	}
	return false
}

type ModelSourceProfile struct {
	state         protoimpl.MessageState `protogen:"open.v1"`
	Slot          string                 `protobuf:"bytes,1,opt,name=slot,proto3" json:"slot,omitempty"`
	Profile       string                 `protobuf:"bytes,2,opt,name=profile,proto3" json:"profile,omitempty"`
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *ModelSourceProfile) Reset() {
	*x = ModelSourceProfile{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[99]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *ModelSourceProfile) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*ModelSourceProfile) ProtoMessage() {}

func (x *ModelSourceProfile) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[99]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use ModelSourceProfile.ProtoReflect.Descriptor instead.
func (*ModelSourceProfile) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{99}
}

func (x *ModelSourceProfile) GetSlot() string {
	if x != nil {
		return x.Slot
	}
	return ""
}

func (x *ModelSourceProfile) GetProfile() string {
	if x != nil {
		return x.Profile
	}
	return ""
}

type PreparedModelSource struct {
	state         protoimpl.MessageState `protogen:"open.v1"`
	Slot          string                 `protobuf:"bytes,1,opt,name=slot,proto3" json:"slot,omitempty"`
	Profile       string                 `protobuf:"bytes,2,opt,name=profile,proto3" json:"profile,omitempty"`
	Manifest      *Ref                   `protobuf:"bytes,3,opt,name=manifest,proto3" json:"manifest,omitempty"`
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *PreparedModelSource) Reset() {
	*x = PreparedModelSource{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[100]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *PreparedModelSource) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*PreparedModelSource) ProtoMessage() {}

func (x *PreparedModelSource) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[100]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use PreparedModelSource.ProtoReflect.Descriptor instead.
func (*PreparedModelSource) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{100}
}

func (x *PreparedModelSource) GetSlot() string {
	if x != nil {
		return x.Slot
	}
	return ""
}

func (x *PreparedModelSource) GetProfile() string {
	if x != nil {
		return x.Profile
	}
	return ""
}

func (x *PreparedModelSource) GetManifest() *Ref {
	if x != nil {
		return x.Manifest
	}
	return nil
}

// A TensorFS-authored conversion checkpoint, not a new manifest or a publication receipt.
// The head binds the plan and a chain of immutable object refs. bytes is cumulative LOCAL
// checkpointed progress; it becomes remotely durable only after the record owner confirms
// custody of the head and its entire referenced prefix. A replacement pod receives only an
// acknowledged durable head and restores it through ordinary TensorFS admission.
type ModelSourceCheckpoint struct {
	state         protoimpl.MessageState `protogen:"open.v1"`
	Slot          string                 `protobuf:"bytes,1,opt,name=slot,proto3" json:"slot,omitempty"`
	Head          *Ref                   `protobuf:"bytes,2,opt,name=head,proto3" json:"head,omitempty"`
	PlanDigest    []byte                 `protobuf:"bytes,3,opt,name=plan_digest,json=planDigest,proto3" json:"plan_digest,omitempty"` // exact SHA-256, 32 bytes
	Index         uint64                 `protobuf:"varint,4,opt,name=index,proto3" json:"index,omitempty"`
	Bytes         uint64                 `protobuf:"varint,5,opt,name=bytes,proto3" json:"bytes,omitempty"`
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *ModelSourceCheckpoint) Reset() {
	*x = ModelSourceCheckpoint{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[101]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *ModelSourceCheckpoint) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*ModelSourceCheckpoint) ProtoMessage() {}

func (x *ModelSourceCheckpoint) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[101]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use ModelSourceCheckpoint.ProtoReflect.Descriptor instead.
func (*ModelSourceCheckpoint) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{101}
}

func (x *ModelSourceCheckpoint) GetSlot() string {
	if x != nil {
		return x.Slot
	}
	return ""
}

func (x *ModelSourceCheckpoint) GetHead() *Ref {
	if x != nil {
		return x.Head
	}
	return nil
}

func (x *ModelSourceCheckpoint) GetPlanDigest() []byte {
	if x != nil {
		return x.PlanDigest
	}
	return nil
}

func (x *ModelSourceCheckpoint) GetIndex() uint64 {
	if x != nil {
		return x.Index
	}
	return 0
}

func (x *ModelSourceCheckpoint) GetBytes() uint64 {
	if x != nil {
		return x.Bytes
	}
	return 0
}

// Source and derived-output authority remain disjoint even though their immutable Link
// transport is identical. The authenticated host validates exactly one subject before
// forwarding; Runtime validates the corresponding operation/slot/plan against TensorFS.
type SourceCheckpointSubject struct {
	state                 protoimpl.MessageState `protogen:"open.v1"`
	OperationId           string                 `protobuf:"bytes,1,opt,name=operation_id,json=operationId,proto3" json:"operation_id,omitempty"`
	SourceSelectionDigest []byte                 `protobuf:"bytes,2,opt,name=source_selection_digest,json=sourceSelectionDigest,proto3" json:"source_selection_digest,omitempty"`
	Slot                  string                 `protobuf:"bytes,3,opt,name=slot,proto3" json:"slot,omitempty"`
	unknownFields         protoimpl.UnknownFields
	sizeCache             protoimpl.SizeCache
}

func (x *SourceCheckpointSubject) Reset() {
	*x = SourceCheckpointSubject{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[102]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *SourceCheckpointSubject) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*SourceCheckpointSubject) ProtoMessage() {}

func (x *SourceCheckpointSubject) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[102]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use SourceCheckpointSubject.ProtoReflect.Descriptor instead.
func (*SourceCheckpointSubject) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{102}
}

func (x *SourceCheckpointSubject) GetOperationId() string {
	if x != nil {
		return x.OperationId
	}
	return ""
}

func (x *SourceCheckpointSubject) GetSourceSelectionDigest() []byte {
	if x != nil {
		return x.SourceSelectionDigest
	}
	return nil
}

func (x *SourceCheckpointSubject) GetSlot() string {
	if x != nil {
		return x.Slot
	}
	return ""
}

type WeightsCheckpointSubject struct {
	state                     protoimpl.MessageState `protogen:"open.v1"`
	RequestId                 string                 `protobuf:"bytes,1,opt,name=request_id,json=requestId,proto3" json:"request_id,omitempty"`
	InvocationSpecDigest      []byte                 `protobuf:"bytes,2,opt,name=invocation_spec_digest,json=invocationSpecDigest,proto3" json:"invocation_spec_digest,omitempty"`
	OutputSlot                string                 `protobuf:"bytes,3,opt,name=output_slot,json=outputSlot,proto3" json:"output_slot,omitempty"`
	WeightsTransactionId      string                 `protobuf:"bytes,4,opt,name=weights_transaction_id,json=weightsTransactionId,proto3" json:"weights_transaction_id,omitempty"` // supplied by Runtime, never derived by the host
	WriterEpoch               uint64                 `protobuf:"varint,5,opt,name=writer_epoch,json=writerEpoch,proto3" json:"writer_epoch,omitempty"`
	TensorfsDeclarationDigest []byte                 `protobuf:"bytes,6,opt,name=tensorfs_declaration_digest,json=tensorfsDeclarationDigest,proto3" json:"tensorfs_declaration_digest,omitempty"`
	unknownFields             protoimpl.UnknownFields
	sizeCache                 protoimpl.SizeCache
}

func (x *WeightsCheckpointSubject) Reset() {
	*x = WeightsCheckpointSubject{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[103]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *WeightsCheckpointSubject) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*WeightsCheckpointSubject) ProtoMessage() {}

func (x *WeightsCheckpointSubject) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[103]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use WeightsCheckpointSubject.ProtoReflect.Descriptor instead.
func (*WeightsCheckpointSubject) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{103}
}

func (x *WeightsCheckpointSubject) GetRequestId() string {
	if x != nil {
		return x.RequestId
	}
	return ""
}

func (x *WeightsCheckpointSubject) GetInvocationSpecDigest() []byte {
	if x != nil {
		return x.InvocationSpecDigest
	}
	return nil
}

func (x *WeightsCheckpointSubject) GetOutputSlot() string {
	if x != nil {
		return x.OutputSlot
	}
	return ""
}

func (x *WeightsCheckpointSubject) GetWeightsTransactionId() string {
	if x != nil {
		return x.WeightsTransactionId
	}
	return ""
}

func (x *WeightsCheckpointSubject) GetWriterEpoch() uint64 {
	if x != nil {
		return x.WriterEpoch
	}
	return 0
}

func (x *WeightsCheckpointSubject) GetTensorfsDeclarationDigest() []byte {
	if x != nil {
		return x.TensorfsDeclarationDigest
	}
	return nil
}

type CheckpointSubject struct {
	state protoimpl.MessageState `protogen:"open.v1"`
	// Types that are valid to be assigned to Kind:
	//
	//	*CheckpointSubject_Source
	//	*CheckpointSubject_Weights
	Kind          isCheckpointSubject_Kind `protobuf_oneof:"kind"`
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *CheckpointSubject) Reset() {
	*x = CheckpointSubject{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[104]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *CheckpointSubject) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*CheckpointSubject) ProtoMessage() {}

func (x *CheckpointSubject) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[104]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use CheckpointSubject.ProtoReflect.Descriptor instead.
func (*CheckpointSubject) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{104}
}

func (x *CheckpointSubject) GetKind() isCheckpointSubject_Kind {
	if x != nil {
		return x.Kind
	}
	return nil
}

func (x *CheckpointSubject) GetSource() *SourceCheckpointSubject {
	if x != nil {
		if x, ok := x.Kind.(*CheckpointSubject_Source); ok {
			return x.Source
		}
	}
	return nil
}

func (x *CheckpointSubject) GetWeights() *WeightsCheckpointSubject {
	if x != nil {
		if x, ok := x.Kind.(*CheckpointSubject_Weights); ok {
			return x.Weights
		}
	}
	return nil
}

type isCheckpointSubject_Kind interface {
	isCheckpointSubject_Kind()
}

type CheckpointSubject_Source struct {
	Source *SourceCheckpointSubject `protobuf:"bytes,1,opt,name=source,proto3,oneof"`
}

type CheckpointSubject_Weights struct {
	Weights *WeightsCheckpointSubject `protobuf:"bytes,2,opt,name=weights,proto3,oneof"`
}

func (*CheckpointSubject_Source) isCheckpointSubject_Kind() {}

func (*CheckpointSubject_Weights) isCheckpointSubject_Kind() {}

type CheckpointObject struct {
	state         protoimpl.MessageState `protogen:"open.v1"`
	Ref           *Ref                   `protobuf:"bytes,1,opt,name=ref,proto3" json:"ref,omitempty"`
	Manifest      bool                   `protobuf:"varint,2,opt,name=manifest,proto3" json:"manifest,omitempty"` // false is a blob; true is the manifest namespace
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *CheckpointObject) Reset() {
	*x = CheckpointObject{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[105]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *CheckpointObject) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*CheckpointObject) ProtoMessage() {}

func (x *CheckpointObject) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[105]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use CheckpointObject.ProtoReflect.Descriptor instead.
func (*CheckpointObject) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{105}
}

func (x *CheckpointObject) GetRef() *Ref {
	if x != nil {
		return x.Ref
	}
	return nil
}

func (x *CheckpointObject) GetManifest() bool {
	if x != nil {
		return x.Manifest
	}
	return false
}

// Inspect ONE immutable checkpoint Link, in bounded pages. TensorFS verifies operation,
// slot and plan against the Link; no caller-provided object inventory becomes authority.
type CheckpointPageRequest struct {
	state              protoimpl.MessageState `protogen:"open.v1"`
	RecordOwnerEpoch   uint64                 `protobuf:"varint,1,opt,name=record_owner_epoch,json=recordOwnerEpoch,proto3" json:"record_owner_epoch,omitempty"`
	ControlStreamEpoch uint64                 `protobuf:"varint,2,opt,name=control_stream_epoch,json=controlStreamEpoch,proto3" json:"control_stream_epoch,omitempty"`
	WorkerBootId       string                 `protobuf:"bytes,3,opt,name=worker_boot_id,json=workerBootId,proto3" json:"worker_boot_id,omitempty"`
	Subject            *CheckpointSubject     `protobuf:"bytes,5,opt,name=subject,proto3" json:"subject,omitempty"`
	PlanDigest         []byte                 `protobuf:"bytes,8,opt,name=plan_digest,json=planDigest,proto3" json:"plan_digest,omitempty"`
	Head               *Ref                   `protobuf:"bytes,9,opt,name=head,proto3" json:"head,omitempty"`
	Offset             uint32                 `protobuf:"varint,10,opt,name=offset,proto3" json:"offset,omitempty"`
	Limit              uint32                 `protobuf:"varint,11,opt,name=limit,proto3" json:"limit,omitempty"` // 1..MaxCheckpointObjects
	unknownFields      protoimpl.UnknownFields
	sizeCache          protoimpl.SizeCache
}

func (x *CheckpointPageRequest) Reset() {
	*x = CheckpointPageRequest{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[106]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *CheckpointPageRequest) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*CheckpointPageRequest) ProtoMessage() {}

func (x *CheckpointPageRequest) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[106]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use CheckpointPageRequest.ProtoReflect.Descriptor instead.
func (*CheckpointPageRequest) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{106}
}

func (x *CheckpointPageRequest) GetRecordOwnerEpoch() uint64 {
	if x != nil {
		return x.RecordOwnerEpoch
	}
	return 0
}

func (x *CheckpointPageRequest) GetControlStreamEpoch() uint64 {
	if x != nil {
		return x.ControlStreamEpoch
	}
	return 0
}

func (x *CheckpointPageRequest) GetWorkerBootId() string {
	if x != nil {
		return x.WorkerBootId
	}
	return ""
}

func (x *CheckpointPageRequest) GetSubject() *CheckpointSubject {
	if x != nil {
		return x.Subject
	}
	return nil
}

func (x *CheckpointPageRequest) GetPlanDigest() []byte {
	if x != nil {
		return x.PlanDigest
	}
	return nil
}

func (x *CheckpointPageRequest) GetHead() *Ref {
	if x != nil {
		return x.Head
	}
	return nil
}

func (x *CheckpointPageRequest) GetOffset() uint32 {
	if x != nil {
		return x.Offset
	}
	return 0
}

func (x *CheckpointPageRequest) GetLimit() uint32 {
	if x != nil {
		return x.Limit
	}
	return 0
}

type CheckpointPageResult struct {
	state              protoimpl.MessageState `protogen:"open.v1"`
	RecordOwnerEpoch   uint64                 `protobuf:"varint,1,opt,name=record_owner_epoch,json=recordOwnerEpoch,proto3" json:"record_owner_epoch,omitempty"`
	ControlStreamEpoch uint64                 `protobuf:"varint,2,opt,name=control_stream_epoch,json=controlStreamEpoch,proto3" json:"control_stream_epoch,omitempty"`
	WorkerBootId       string                 `protobuf:"bytes,3,opt,name=worker_boot_id,json=workerBootId,proto3" json:"worker_boot_id,omitempty"`
	Subject            *CheckpointSubject     `protobuf:"bytes,5,opt,name=subject,proto3" json:"subject,omitempty"`
	PlanDigest         []byte                 `protobuf:"bytes,8,opt,name=plan_digest,json=planDigest,proto3" json:"plan_digest,omitempty"`
	Head               *Ref                   `protobuf:"bytes,9,opt,name=head,proto3" json:"head,omitempty"`
	Index              uint64                 `protobuf:"varint,10,opt,name=index,proto3" json:"index,omitempty"`
	Bytes              uint64                 `protobuf:"varint,11,opt,name=bytes,proto3" json:"bytes,omitempty"`      // local cumulative checkpointed bytes, not remote custody
	Previous           *Ref                   `protobuf:"bytes,12,opt,name=previous,proto3" json:"previous,omitempty"` // absent at the root Link
	Progress           *Ref                   `protobuf:"bytes,13,opt,name=progress,proto3" json:"progress,omitempty"` // the immutable source journal or derived accepted-part snapshot
	Objects            []*CheckpointObject    `protobuf:"bytes,14,rep,name=objects,proto3" json:"objects,omitempty"`   // <= MaxCheckpointObjects
	NextOffset         uint32                 `protobuf:"varint,15,opt,name=next_offset,json=nextOffset,proto3" json:"next_offset,omitempty"`
	HasMore            bool                   `protobuf:"varint,16,opt,name=has_more,json=hasMore,proto3" json:"has_more,omitempty"`
	SafeCode           string                 `protobuf:"bytes,17,opt,name=safe_code,json=safeCode,proto3" json:"safe_code,omitempty"` // nonempty means refused; no partial success then
	SafeDetail         string                 `protobuf:"bytes,18,opt,name=safe_detail,json=safeDetail,proto3" json:"safe_detail,omitempty"`
	unknownFields      protoimpl.UnknownFields
	sizeCache          protoimpl.SizeCache
}

func (x *CheckpointPageResult) Reset() {
	*x = CheckpointPageResult{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[107]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *CheckpointPageResult) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*CheckpointPageResult) ProtoMessage() {}

func (x *CheckpointPageResult) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[107]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use CheckpointPageResult.ProtoReflect.Descriptor instead.
func (*CheckpointPageResult) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{107}
}

func (x *CheckpointPageResult) GetRecordOwnerEpoch() uint64 {
	if x != nil {
		return x.RecordOwnerEpoch
	}
	return 0
}

func (x *CheckpointPageResult) GetControlStreamEpoch() uint64 {
	if x != nil {
		return x.ControlStreamEpoch
	}
	return 0
}

func (x *CheckpointPageResult) GetWorkerBootId() string {
	if x != nil {
		return x.WorkerBootId
	}
	return ""
}

func (x *CheckpointPageResult) GetSubject() *CheckpointSubject {
	if x != nil {
		return x.Subject
	}
	return nil
}

func (x *CheckpointPageResult) GetPlanDigest() []byte {
	if x != nil {
		return x.PlanDigest
	}
	return nil
}

func (x *CheckpointPageResult) GetHead() *Ref {
	if x != nil {
		return x.Head
	}
	return nil
}

func (x *CheckpointPageResult) GetIndex() uint64 {
	if x != nil {
		return x.Index
	}
	return 0
}

func (x *CheckpointPageResult) GetBytes() uint64 {
	if x != nil {
		return x.Bytes
	}
	return 0
}

func (x *CheckpointPageResult) GetPrevious() *Ref {
	if x != nil {
		return x.Previous
	}
	return nil
}

func (x *CheckpointPageResult) GetProgress() *Ref {
	if x != nil {
		return x.Progress
	}
	return nil
}

func (x *CheckpointPageResult) GetObjects() []*CheckpointObject {
	if x != nil {
		return x.Objects
	}
	return nil
}

func (x *CheckpointPageResult) GetNextOffset() uint32 {
	if x != nil {
		return x.NextOffset
	}
	return 0
}

func (x *CheckpointPageResult) GetHasMore() bool {
	if x != nil {
		return x.HasMore
	}
	return false
}

func (x *CheckpointPageResult) GetSafeCode() string {
	if x != nil {
		return x.SafeCode
	}
	return ""
}

func (x *CheckpointPageResult) GetSafeDetail() string {
	if x != nil {
		return x.SafeDetail
	}
	return ""
}

// One grant for one object under an exact checkpoint Link. A GET may first restore the
// named head itself, after which TensorFS validates that Link before exposing its members.
// Every other object must be a member, predecessor or progress ref of the admitted Link.
// Capabilities stay in memory; host/owner ledgers retain only identity and measured results.
type CheckpointTransferRequest struct {
	state              protoimpl.MessageState `protogen:"open.v1"`
	RecordOwnerEpoch   uint64                 `protobuf:"varint,1,opt,name=record_owner_epoch,json=recordOwnerEpoch,proto3" json:"record_owner_epoch,omitempty"`
	ControlStreamEpoch uint64                 `protobuf:"varint,2,opt,name=control_stream_epoch,json=controlStreamEpoch,proto3" json:"control_stream_epoch,omitempty"`
	WorkerBootId       string                 `protobuf:"bytes,3,opt,name=worker_boot_id,json=workerBootId,proto3" json:"worker_boot_id,omitempty"`
	Subject            *CheckpointSubject     `protobuf:"bytes,5,opt,name=subject,proto3" json:"subject,omitempty"`
	PlanDigest         []byte                 `protobuf:"bytes,8,opt,name=plan_digest,json=planDigest,proto3" json:"plan_digest,omitempty"`
	Head               *Ref                   `protobuf:"bytes,9,opt,name=head,proto3" json:"head,omitempty"`
	Object             *CheckpointObject      `protobuf:"bytes,10,opt,name=object,proto3" json:"object,omitempty"`
	TransferId         string                 `protobuf:"bytes,11,opt,name=transfer_id,json=transferId,proto3" json:"transfer_id,omitempty"`
	GrantRevision      uint64                 `protobuf:"varint,12,opt,name=grant_revision,json=grantRevision,proto3" json:"grant_revision,omitempty"`
	// Types that are valid to be assigned to Decision:
	//
	//	*CheckpointTransferRequest_UploadGrant
	//	*CheckpointTransferRequest_DownloadUrl
	Decision      isCheckpointTransferRequest_Decision `protobuf_oneof:"decision"`
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *CheckpointTransferRequest) Reset() {
	*x = CheckpointTransferRequest{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[108]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *CheckpointTransferRequest) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*CheckpointTransferRequest) ProtoMessage() {}

func (x *CheckpointTransferRequest) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[108]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use CheckpointTransferRequest.ProtoReflect.Descriptor instead.
func (*CheckpointTransferRequest) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{108}
}

func (x *CheckpointTransferRequest) GetRecordOwnerEpoch() uint64 {
	if x != nil {
		return x.RecordOwnerEpoch
	}
	return 0
}

func (x *CheckpointTransferRequest) GetControlStreamEpoch() uint64 {
	if x != nil {
		return x.ControlStreamEpoch
	}
	return 0
}

func (x *CheckpointTransferRequest) GetWorkerBootId() string {
	if x != nil {
		return x.WorkerBootId
	}
	return ""
}

func (x *CheckpointTransferRequest) GetSubject() *CheckpointSubject {
	if x != nil {
		return x.Subject
	}
	return nil
}

func (x *CheckpointTransferRequest) GetPlanDigest() []byte {
	if x != nil {
		return x.PlanDigest
	}
	return nil
}

func (x *CheckpointTransferRequest) GetHead() *Ref {
	if x != nil {
		return x.Head
	}
	return nil
}

func (x *CheckpointTransferRequest) GetObject() *CheckpointObject {
	if x != nil {
		return x.Object
	}
	return nil
}

func (x *CheckpointTransferRequest) GetTransferId() string {
	if x != nil {
		return x.TransferId
	}
	return ""
}

func (x *CheckpointTransferRequest) GetGrantRevision() uint64 {
	if x != nil {
		return x.GrantRevision
	}
	return 0
}

func (x *CheckpointTransferRequest) GetDecision() isCheckpointTransferRequest_Decision {
	if x != nil {
		return x.Decision
	}
	return nil
}

func (x *CheckpointTransferRequest) GetUploadGrant() *WeightsUploadGrant {
	if x != nil {
		if x, ok := x.Decision.(*CheckpointTransferRequest_UploadGrant); ok {
			return x.UploadGrant
		}
	}
	return nil
}

func (x *CheckpointTransferRequest) GetDownloadUrl() string {
	if x != nil {
		if x, ok := x.Decision.(*CheckpointTransferRequest_DownloadUrl); ok {
			return x.DownloadUrl
		}
	}
	return ""
}

type isCheckpointTransferRequest_Decision interface {
	isCheckpointTransferRequest_Decision()
}

type CheckpointTransferRequest_UploadGrant struct {
	UploadGrant *WeightsUploadGrant `protobuf:"bytes,13,opt,name=upload_grant,json=uploadGrant,proto3,oneof"`
}

type CheckpointTransferRequest_DownloadUrl struct {
	DownloadUrl string `protobuf:"bytes,14,opt,name=download_url,json=downloadUrl,proto3,oneof"` // one scoped GET; digest and length come from object
}

func (*CheckpointTransferRequest_UploadGrant) isCheckpointTransferRequest_Decision() {}

func (*CheckpointTransferRequest_DownloadUrl) isCheckpointTransferRequest_Decision() {}

type CheckpointTransferStatus struct {
	state              protoimpl.MessageState `protogen:"open.v1"`
	RecordOwnerEpoch   uint64                 `protobuf:"varint,1,opt,name=record_owner_epoch,json=recordOwnerEpoch,proto3" json:"record_owner_epoch,omitempty"`
	ControlStreamEpoch uint64                 `protobuf:"varint,2,opt,name=control_stream_epoch,json=controlStreamEpoch,proto3" json:"control_stream_epoch,omitempty"`
	WorkerBootId       string                 `protobuf:"bytes,3,opt,name=worker_boot_id,json=workerBootId,proto3" json:"worker_boot_id,omitempty"`
	Subject            *CheckpointSubject     `protobuf:"bytes,5,opt,name=subject,proto3" json:"subject,omitempty"`
	PlanDigest         []byte                 `protobuf:"bytes,8,opt,name=plan_digest,json=planDigest,proto3" json:"plan_digest,omitempty"`
	Head               *Ref                   `protobuf:"bytes,9,opt,name=head,proto3" json:"head,omitempty"`
	Object             *CheckpointObject      `protobuf:"bytes,10,opt,name=object,proto3" json:"object,omitempty"`
	TransferId         string                 `protobuf:"bytes,11,opt,name=transfer_id,json=transferId,proto3" json:"transfer_id,omitempty"`
	GrantRevision      uint64                 `protobuf:"varint,12,opt,name=grant_revision,json=grantRevision,proto3" json:"grant_revision,omitempty"`
	State              WeightsTransferState   `protobuf:"varint,13,opt,name=state,proto3,enum=cozy.worker.v1.WeightsTransferState" json:"state,omitempty"` // GET success is HELD (verified local admission)
	TransferredBytes   uint64                 `protobuf:"varint,14,opt,name=transferred_bytes,json=transferredBytes,proto3" json:"transferred_bytes,omitempty"`
	HttpStatus         uint32                 `protobuf:"varint,15,opt,name=http_status,json=httpStatus,proto3" json:"http_status,omitempty"`
	ChecksumSha256     string                 `protobuf:"bytes,16,opt,name=checksum_sha256,json=checksumSha256,proto3" json:"checksum_sha256,omitempty"` // digest proven by local admission/read verification
	SafeCode           string                 `protobuf:"bytes,17,opt,name=safe_code,json=safeCode,proto3" json:"safe_code,omitempty"`
	SafeDetail         string                 `protobuf:"bytes,18,opt,name=safe_detail,json=safeDetail,proto3" json:"safe_detail,omitempty"`
	unknownFields      protoimpl.UnknownFields
	sizeCache          protoimpl.SizeCache
}

func (x *CheckpointTransferStatus) Reset() {
	*x = CheckpointTransferStatus{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[109]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *CheckpointTransferStatus) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*CheckpointTransferStatus) ProtoMessage() {}

func (x *CheckpointTransferStatus) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[109]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use CheckpointTransferStatus.ProtoReflect.Descriptor instead.
func (*CheckpointTransferStatus) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{109}
}

func (x *CheckpointTransferStatus) GetRecordOwnerEpoch() uint64 {
	if x != nil {
		return x.RecordOwnerEpoch
	}
	return 0
}

func (x *CheckpointTransferStatus) GetControlStreamEpoch() uint64 {
	if x != nil {
		return x.ControlStreamEpoch
	}
	return 0
}

func (x *CheckpointTransferStatus) GetWorkerBootId() string {
	if x != nil {
		return x.WorkerBootId
	}
	return ""
}

func (x *CheckpointTransferStatus) GetSubject() *CheckpointSubject {
	if x != nil {
		return x.Subject
	}
	return nil
}

func (x *CheckpointTransferStatus) GetPlanDigest() []byte {
	if x != nil {
		return x.PlanDigest
	}
	return nil
}

func (x *CheckpointTransferStatus) GetHead() *Ref {
	if x != nil {
		return x.Head
	}
	return nil
}

func (x *CheckpointTransferStatus) GetObject() *CheckpointObject {
	if x != nil {
		return x.Object
	}
	return nil
}

func (x *CheckpointTransferStatus) GetTransferId() string {
	if x != nil {
		return x.TransferId
	}
	return ""
}

func (x *CheckpointTransferStatus) GetGrantRevision() uint64 {
	if x != nil {
		return x.GrantRevision
	}
	return 0
}

func (x *CheckpointTransferStatus) GetState() WeightsTransferState {
	if x != nil {
		return x.State
	}
	return WeightsTransferState_WEIGHTS_TRANSFER_STATE_UNSPECIFIED
}

func (x *CheckpointTransferStatus) GetTransferredBytes() uint64 {
	if x != nil {
		return x.TransferredBytes
	}
	return 0
}

func (x *CheckpointTransferStatus) GetHttpStatus() uint32 {
	if x != nil {
		return x.HttpStatus
	}
	return 0
}

func (x *CheckpointTransferStatus) GetChecksumSha256() string {
	if x != nil {
		return x.ChecksumSha256
	}
	return ""
}

func (x *CheckpointTransferStatus) GetSafeCode() string {
	if x != nil {
		return x.SafeCode
	}
	return ""
}

func (x *CheckpointTransferStatus) GetSafeDetail() string {
	if x != nil {
		return x.SafeDetail
	}
	return ""
}

type PrepareModelSourceRequest struct {
	state                 protoimpl.MessageState   `protogen:"open.v1"`
	OperationId           string                   `protobuf:"bytes,1,opt,name=operation_id,json=operationId,proto3" json:"operation_id,omitempty"`
	SourceSelectionDigest []byte                   `protobuf:"bytes,2,opt,name=source_selection_digest,json=sourceSelectionDigest,proto3" json:"source_selection_digest,omitempty"`
	Profiles              []*ModelSourceProfile    `protobuf:"bytes,3,rep,name=profiles,proto3" json:"profiles,omitempty"`                                                         // sorted unique by slot; <= MaxModelSourceProfiles
	Files                 []*LocalModelSourceFile  `protobuf:"bytes,4,rep,name=files,proto3" json:"files,omitempty"`                                                               // sorted unique by member; <= MaxModelSourceFiles
	SourceUri             string                   `protobuf:"bytes,5,opt,name=source_uri,json=sourceUri,proto3" json:"source_uri,omitempty"`                                      // credential-free pinned hf:// or civitai:// provenance
	DeclaredLicense       string                   `protobuf:"bytes,6,opt,name=declared_license,json=declaredLicense,proto3" json:"declared_license,omitempty"`                    // bounded printable provenance; empty means undeclared
	Checkpoints           []*ModelSourceCheckpoint `protobuf:"bytes,7,rep,name=checkpoints,proto3" json:"checkpoints,omitempty"`                                                   // sorted unique by slot; admitted restore heads
	AdoptFromOperationId  string                   `protobuf:"bytes,8,opt,name=adopt_from_operation_id,json=adoptFromOperationId,proto3" json:"adopt_from_operation_id,omitempty"` // signed Host adoption only; never an executor choice
	unknownFields         protoimpl.UnknownFields
	sizeCache             protoimpl.SizeCache
}

func (x *PrepareModelSourceRequest) Reset() {
	*x = PrepareModelSourceRequest{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[110]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *PrepareModelSourceRequest) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*PrepareModelSourceRequest) ProtoMessage() {}

func (x *PrepareModelSourceRequest) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[110]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use PrepareModelSourceRequest.ProtoReflect.Descriptor instead.
func (*PrepareModelSourceRequest) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{110}
}

func (x *PrepareModelSourceRequest) GetOperationId() string {
	if x != nil {
		return x.OperationId
	}
	return ""
}

func (x *PrepareModelSourceRequest) GetSourceSelectionDigest() []byte {
	if x != nil {
		return x.SourceSelectionDigest
	}
	return nil
}

func (x *PrepareModelSourceRequest) GetProfiles() []*ModelSourceProfile {
	if x != nil {
		return x.Profiles
	}
	return nil
}

func (x *PrepareModelSourceRequest) GetFiles() []*LocalModelSourceFile {
	if x != nil {
		return x.Files
	}
	return nil
}

func (x *PrepareModelSourceRequest) GetSourceUri() string {
	if x != nil {
		return x.SourceUri
	}
	return ""
}

func (x *PrepareModelSourceRequest) GetDeclaredLicense() string {
	if x != nil {
		return x.DeclaredLicense
	}
	return ""
}

func (x *PrepareModelSourceRequest) GetCheckpoints() []*ModelSourceCheckpoint {
	if x != nil {
		return x.Checkpoints
	}
	return nil
}

func (x *PrepareModelSourceRequest) GetAdoptFromOperationId() string {
	if x != nil {
		return x.AdoptFromOperationId
	}
	return ""
}

type PrepareModelSourceResult struct {
	state         protoimpl.MessageState    `protogen:"open.v1"`
	Outcome       ModelSourcePrepareOutcome `protobuf:"varint,1,opt,name=outcome,proto3,enum=cozy.worker.v1.ModelSourcePrepareOutcome" json:"outcome,omitempty"`
	Sources       []*PreparedModelSource    `protobuf:"bytes,2,rep,name=sources,proto3" json:"sources,omitempty"` // sorted unique completed slots; exact set when complete
	SafeCode      string                    `protobuf:"bytes,3,opt,name=safe_code,json=safeCode,proto3" json:"safe_code,omitempty"`
	SafeDetail    string                    `protobuf:"bytes,4,opt,name=safe_detail,json=safeDetail,proto3" json:"safe_detail,omitempty"`
	Checkpoints   []*ModelSourceCheckpoint  `protobuf:"bytes,5,rep,name=checkpoints,proto3" json:"checkpoints,omitempty"`                       // sorted unique by slot, <= MaxModelSourceProfiles
	SpentMembers  []string                  `protobuf:"bytes,6,rep,name=spent_members,json=spentMembers,proto3" json:"spent_members,omitempty"` // sorted unique verified carriers no remaining conversion uses
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *PrepareModelSourceResult) Reset() {
	*x = PrepareModelSourceResult{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[111]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *PrepareModelSourceResult) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*PrepareModelSourceResult) ProtoMessage() {}

func (x *PrepareModelSourceResult) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[111]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use PrepareModelSourceResult.ProtoReflect.Descriptor instead.
func (*PrepareModelSourceResult) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{111}
}

func (x *PrepareModelSourceResult) GetOutcome() ModelSourcePrepareOutcome {
	if x != nil {
		return x.Outcome
	}
	return ModelSourcePrepareOutcome_MODEL_SOURCE_PREPARE_OUTCOME_UNSPECIFIED
}

func (x *PrepareModelSourceResult) GetSources() []*PreparedModelSource {
	if x != nil {
		return x.Sources
	}
	return nil
}

func (x *PrepareModelSourceResult) GetSafeCode() string {
	if x != nil {
		return x.SafeCode
	}
	return ""
}

func (x *PrepareModelSourceResult) GetSafeDetail() string {
	if x != nil {
		return x.SafeDetail
	}
	return ""
}

func (x *PrepareModelSourceResult) GetCheckpoints() []*ModelSourceCheckpoint {
	if x != nil {
		return x.Checkpoints
	}
	return nil
}

func (x *PrepareModelSourceResult) GetSpentMembers() []string {
	if x != nil {
		return x.SpentMembers
	}
	return nil
}

type RecordOwnerFrame struct {
	state protoimpl.MessageState `protogen:"open.v1"`
	// Types that are valid to be assigned to Msg:
	//
	//	*RecordOwnerFrame_Claim
	//	*RecordOwnerFrame_DesiredState
	//	*RecordOwnerFrame_SnapshotAck
	Msg           isRecordOwnerFrame_Msg `protobuf_oneof:"msg"`
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *RecordOwnerFrame) Reset() {
	*x = RecordOwnerFrame{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[112]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *RecordOwnerFrame) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*RecordOwnerFrame) ProtoMessage() {}

func (x *RecordOwnerFrame) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[112]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use RecordOwnerFrame.ProtoReflect.Descriptor instead.
func (*RecordOwnerFrame) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{112}
}

func (x *RecordOwnerFrame) GetMsg() isRecordOwnerFrame_Msg {
	if x != nil {
		return x.Msg
	}
	return nil
}

func (x *RecordOwnerFrame) GetClaim() *Claim {
	if x != nil {
		if x, ok := x.Msg.(*RecordOwnerFrame_Claim); ok {
			return x.Claim
		}
	}
	return nil
}

func (x *RecordOwnerFrame) GetDesiredState() *DesiredWorkerState {
	if x != nil {
		if x, ok := x.Msg.(*RecordOwnerFrame_DesiredState); ok {
			return x.DesiredState
		}
	}
	return nil
}

func (x *RecordOwnerFrame) GetSnapshotAck() *SnapshotAck {
	if x != nil {
		if x, ok := x.Msg.(*RecordOwnerFrame_SnapshotAck); ok {
			return x.SnapshotAck
		}
	}
	return nil
}

type isRecordOwnerFrame_Msg interface {
	isRecordOwnerFrame_Msg()
}

type RecordOwnerFrame_Claim struct {
	Claim *Claim `protobuf:"bytes,5,opt,name=claim,proto3,oneof"`
}

type RecordOwnerFrame_DesiredState struct {
	DesiredState *DesiredWorkerState `protobuf:"bytes,6,opt,name=desired_state,json=desiredState,proto3,oneof"`
}

type RecordOwnerFrame_SnapshotAck struct {
	SnapshotAck *SnapshotAck `protobuf:"bytes,11,opt,name=snapshot_ack,json=snapshotAck,proto3,oneof"`
}

func (*RecordOwnerFrame_Claim) isRecordOwnerFrame_Msg() {}

func (*RecordOwnerFrame_DesiredState) isRecordOwnerFrame_Msg() {}

func (*RecordOwnerFrame_SnapshotAck) isRecordOwnerFrame_Msg() {}

type WorkerFrame struct {
	state protoimpl.MessageState `protogen:"open.v1"`
	// Types that are valid to be assigned to Msg:
	//
	//	*WorkerFrame_ClaimAck
	//	*WorkerFrame_ObservedState
	//	*WorkerFrame_BootFailure
	//	*WorkerFrame_Snapshot
	Msg           isWorkerFrame_Msg `protobuf_oneof:"msg"`
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *WorkerFrame) Reset() {
	*x = WorkerFrame{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[113]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *WorkerFrame) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*WorkerFrame) ProtoMessage() {}

func (x *WorkerFrame) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[113]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use WorkerFrame.ProtoReflect.Descriptor instead.
func (*WorkerFrame) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{113}
}

func (x *WorkerFrame) GetMsg() isWorkerFrame_Msg {
	if x != nil {
		return x.Msg
	}
	return nil
}

func (x *WorkerFrame) GetClaimAck() *ClaimAck {
	if x != nil {
		if x, ok := x.Msg.(*WorkerFrame_ClaimAck); ok {
			return x.ClaimAck
		}
	}
	return nil
}

func (x *WorkerFrame) GetObservedState() *ObservedWorkerState {
	if x != nil {
		if x, ok := x.Msg.(*WorkerFrame_ObservedState); ok {
			return x.ObservedState
		}
	}
	return nil
}

func (x *WorkerFrame) GetBootFailure() *BootFailure {
	if x != nil {
		if x, ok := x.Msg.(*WorkerFrame_BootFailure); ok {
			return x.BootFailure
		}
	}
	return nil
}

func (x *WorkerFrame) GetSnapshot() *WorkerSnapshot {
	if x != nil {
		if x, ok := x.Msg.(*WorkerFrame_Snapshot); ok {
			return x.Snapshot
		}
	}
	return nil
}

type isWorkerFrame_Msg interface {
	isWorkerFrame_Msg()
}

type WorkerFrame_ClaimAck struct {
	ClaimAck *ClaimAck `protobuf:"bytes,5,opt,name=claim_ack,json=claimAck,proto3,oneof"`
}

type WorkerFrame_ObservedState struct {
	ObservedState *ObservedWorkerState `protobuf:"bytes,6,opt,name=observed_state,json=observedState,proto3,oneof"`
}

type WorkerFrame_BootFailure struct {
	// Valid as the stream's ONLY substantive reply: the boot-fatal verdict INSTEAD OF ClaimAck.
	BootFailure *BootFailure `protobuf:"bytes,9,opt,name=boot_failure,json=bootFailure,proto3,oneof"`
}

type WorkerFrame_Snapshot struct {
	Snapshot *WorkerSnapshot `protobuf:"bytes,12,opt,name=snapshot,proto3,oneof"`
}

func (*WorkerFrame_ClaimAck) isWorkerFrame_Msg() {}

func (*WorkerFrame_ObservedState) isWorkerFrame_Msg() {}

func (*WorkerFrame_BootFailure) isWorkerFrame_Msg() {}

func (*WorkerFrame_Snapshot) isWorkerFrame_Msg() {}

// An ordinary package call, not a graph or a second request protocol.
// The parent request owns the index; this attempt only re-establishes it. The
// owner resolves the exact target from the parent's frozen interface dependency
// join and durably accepts an ordinary child request before dispatching it.
// intent_digest = SHA256(JCS({module, export,
// request: <decoded request_canonical_bytes>})). It omits parent/attempt clocks;
// the full parent identity below independently fences transport and replay.
// When capture is present, include capture: <its canonical object> in
// that intent preimage. Absence preserves the pre-capture intent identity.
type ChildCallRequest struct {
	state                      protoimpl.MessageState `protogen:"open.v1"`
	RecordOwnerEpoch           uint64                 `protobuf:"varint,1,opt,name=record_owner_epoch,json=recordOwnerEpoch,proto3" json:"record_owner_epoch,omitempty"`
	ControlStreamEpoch         uint64                 `protobuf:"varint,2,opt,name=control_stream_epoch,json=controlStreamEpoch,proto3" json:"control_stream_epoch,omitempty"`
	WorkerBootId               string                 `protobuf:"bytes,3,opt,name=worker_boot_id,json=workerBootId,proto3" json:"worker_boot_id,omitempty"`
	ParentRequestId            string                 `protobuf:"bytes,4,opt,name=parent_request_id,json=parentRequestId,proto3" json:"parent_request_id,omitempty"`
	ParentAttemptOrdinal       uint64                 `protobuf:"varint,5,opt,name=parent_attempt_ordinal,json=parentAttemptOrdinal,proto3" json:"parent_attempt_ordinal,omitempty"`
	ParentInvocationSpecDigest []byte                 `protobuf:"bytes,6,opt,name=parent_invocation_spec_digest,json=parentInvocationSpecDigest,proto3" json:"parent_invocation_spec_digest,omitempty"`
	// Monotone uint32 lifetime index, never reused within the parent request.
	// At most MaxActiveChildCalls reserved/nonterminal calls; completed calls no longer
	// spend that concurrency bound. Exhausting uint32 refuses rather than wrapping.
	CallIndex             uint32             `protobuf:"varint,7,opt,name=call_index,json=callIndex,proto3" json:"call_index,omitempty"`
	Module                string             `protobuf:"bytes,9,opt,name=module,proto3" json:"module,omitempty"`
	Export                string             `protobuf:"bytes,10,opt,name=export,proto3" json:"export,omitempty"`
	RequestCanonicalBytes []byte             `protobuf:"bytes,11,opt,name=request_canonical_bytes,json=requestCanonicalBytes,proto3" json:"request_canonical_bytes,omitempty"` // bounded 48 KiB; typed refs, never inline artifacts
	IntentDigest          []byte             `protobuf:"bytes,12,opt,name=intent_digest,json=intentDigest,proto3" json:"intent_digest,omitempty"`
	Capture               *ActivationCapture `protobuf:"bytes,13,opt,name=capture,proto3" json:"capture,omitempty"` // present options participate in intent_digest
	unknownFields         protoimpl.UnknownFields
	sizeCache             protoimpl.SizeCache
}

func (x *ChildCallRequest) Reset() {
	*x = ChildCallRequest{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[114]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *ChildCallRequest) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*ChildCallRequest) ProtoMessage() {}

func (x *ChildCallRequest) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[114]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use ChildCallRequest.ProtoReflect.Descriptor instead.
func (*ChildCallRequest) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{114}
}

func (x *ChildCallRequest) GetRecordOwnerEpoch() uint64 {
	if x != nil {
		return x.RecordOwnerEpoch
	}
	return 0
}

func (x *ChildCallRequest) GetControlStreamEpoch() uint64 {
	if x != nil {
		return x.ControlStreamEpoch
	}
	return 0
}

func (x *ChildCallRequest) GetWorkerBootId() string {
	if x != nil {
		return x.WorkerBootId
	}
	return ""
}

func (x *ChildCallRequest) GetParentRequestId() string {
	if x != nil {
		return x.ParentRequestId
	}
	return ""
}

func (x *ChildCallRequest) GetParentAttemptOrdinal() uint64 {
	if x != nil {
		return x.ParentAttemptOrdinal
	}
	return 0
}

func (x *ChildCallRequest) GetParentInvocationSpecDigest() []byte {
	if x != nil {
		return x.ParentInvocationSpecDigest
	}
	return nil
}

func (x *ChildCallRequest) GetCallIndex() uint32 {
	if x != nil {
		return x.CallIndex
	}
	return 0
}

func (x *ChildCallRequest) GetModule() string {
	if x != nil {
		return x.Module
	}
	return ""
}

func (x *ChildCallRequest) GetExport() string {
	if x != nil {
		return x.Export
	}
	return ""
}

func (x *ChildCallRequest) GetRequestCanonicalBytes() []byte {
	if x != nil {
		return x.RequestCanonicalBytes
	}
	return nil
}

func (x *ChildCallRequest) GetIntentDigest() []byte {
	if x != nil {
		return x.IntentDigest
	}
	return nil
}

func (x *ChildCallRequest) GetCapture() *ActivationCapture {
	if x != nil {
		return x.Capture
	}
	return nil
}

type ChildCallResult struct {
	state                      protoimpl.MessageState `protogen:"open.v1"`
	RecordOwnerEpoch           uint64                 `protobuf:"varint,1,opt,name=record_owner_epoch,json=recordOwnerEpoch,proto3" json:"record_owner_epoch,omitempty"`
	ControlStreamEpoch         uint64                 `protobuf:"varint,2,opt,name=control_stream_epoch,json=controlStreamEpoch,proto3" json:"control_stream_epoch,omitempty"`
	WorkerBootId               string                 `protobuf:"bytes,3,opt,name=worker_boot_id,json=workerBootId,proto3" json:"worker_boot_id,omitempty"`
	ParentRequestId            string                 `protobuf:"bytes,4,opt,name=parent_request_id,json=parentRequestId,proto3" json:"parent_request_id,omitempty"`
	ParentAttemptOrdinal       uint64                 `protobuf:"varint,5,opt,name=parent_attempt_ordinal,json=parentAttemptOrdinal,proto3" json:"parent_attempt_ordinal,omitempty"`
	ParentInvocationSpecDigest []byte                 `protobuf:"bytes,6,opt,name=parent_invocation_spec_digest,json=parentInvocationSpecDigest,proto3" json:"parent_invocation_spec_digest,omitempty"`
	CallIndex                  uint32                 `protobuf:"varint,7,opt,name=call_index,json=callIndex,proto3" json:"call_index,omitempty"`
	IntentDigest               []byte                 `protobuf:"bytes,8,opt,name=intent_digest,json=intentDigest,proto3" json:"intent_digest,omitempty"`
	ChildRequestId             string                 `protobuf:"bytes,9,opt,name=child_request_id,json=childRequestId,proto3" json:"child_request_id,omitempty"` // absent only for refusal before durable acceptance
	State                      ChildCallState         `protobuf:"varint,10,opt,name=state,proto3,enum=cozy.worker.v1.ChildCallState" json:"state,omitempty"`
	ResultCanonicalBytes       []byte                 `protobuf:"bytes,11,opt,name=result_canonical_bytes,json=resultCanonicalBytes,proto3" json:"result_canonical_bytes,omitempty"` // SUCCEEDED only; bounded 48 KiB
	SafeCode                   string                 `protobuf:"bytes,12,opt,name=safe_code,json=safeCode,proto3" json:"safe_code,omitempty"`
	SafeDetail                 string                 `protobuf:"bytes,13,opt,name=safe_detail,json=safeDetail,proto3" json:"safe_detail,omitempty"` // bounded 1024 bytes; no remote traceback
	// At most MaxChildArtifactGrants TOTAL, including optional runtime.capture.
	// Grants are visible only after independent native retention; no payload bytes.
	ByteResultGrants []*ChildByteResultGrant `protobuf:"bytes,14,rep,name=byte_result_grants,json=byteResultGrants,proto3" json:"byte_result_grants,omitempty"`
	Observation      *ExecutionObservation   `protobuf:"bytes,15,opt,name=observation,proto3" json:"observation,omitempty"` // observer facts, no result DTO wrapper
	unknownFields    protoimpl.UnknownFields
	sizeCache        protoimpl.SizeCache
}

func (x *ChildCallResult) Reset() {
	*x = ChildCallResult{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[115]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *ChildCallResult) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*ChildCallResult) ProtoMessage() {}

func (x *ChildCallResult) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[115]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use ChildCallResult.ProtoReflect.Descriptor instead.
func (*ChildCallResult) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{115}
}

func (x *ChildCallResult) GetRecordOwnerEpoch() uint64 {
	if x != nil {
		return x.RecordOwnerEpoch
	}
	return 0
}

func (x *ChildCallResult) GetControlStreamEpoch() uint64 {
	if x != nil {
		return x.ControlStreamEpoch
	}
	return 0
}

func (x *ChildCallResult) GetWorkerBootId() string {
	if x != nil {
		return x.WorkerBootId
	}
	return ""
}

func (x *ChildCallResult) GetParentRequestId() string {
	if x != nil {
		return x.ParentRequestId
	}
	return ""
}

func (x *ChildCallResult) GetParentAttemptOrdinal() uint64 {
	if x != nil {
		return x.ParentAttemptOrdinal
	}
	return 0
}

func (x *ChildCallResult) GetParentInvocationSpecDigest() []byte {
	if x != nil {
		return x.ParentInvocationSpecDigest
	}
	return nil
}

func (x *ChildCallResult) GetCallIndex() uint32 {
	if x != nil {
		return x.CallIndex
	}
	return 0
}

func (x *ChildCallResult) GetIntentDigest() []byte {
	if x != nil {
		return x.IntentDigest
	}
	return nil
}

func (x *ChildCallResult) GetChildRequestId() string {
	if x != nil {
		return x.ChildRequestId
	}
	return ""
}

func (x *ChildCallResult) GetState() ChildCallState {
	if x != nil {
		return x.State
	}
	return ChildCallState_CHILD_CALL_STATE_UNSPECIFIED
}

func (x *ChildCallResult) GetResultCanonicalBytes() []byte {
	if x != nil {
		return x.ResultCanonicalBytes
	}
	return nil
}

func (x *ChildCallResult) GetSafeCode() string {
	if x != nil {
		return x.SafeCode
	}
	return ""
}

func (x *ChildCallResult) GetSafeDetail() string {
	if x != nil {
		return x.SafeDetail
	}
	return ""
}

func (x *ChildCallResult) GetByteResultGrants() []*ChildByteResultGrant {
	if x != nil {
		return x.ByteResultGrants
	}
	return nil
}

func (x *ChildCallResult) GetObservation() *ExecutionObservation {
	if x != nil {
		return x.Observation
	}
	return nil
}

type NativeSourceMember struct {
	state         protoimpl.MessageState `protogen:"open.v1"`
	Member        string                 `protobuf:"bytes,1,opt,name=member,proto3" json:"member,omitempty"`
	Object        *Ref                   `protobuf:"bytes,2,opt,name=object,proto3" json:"object,omitempty"`
	Url           string                 `protobuf:"bytes,3,opt,name=url,proto3" json:"url,omitempty"` // ephemeral delivery authority; never content identity or public history
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *NativeSourceMember) Reset() {
	*x = NativeSourceMember{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[116]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *NativeSourceMember) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*NativeSourceMember) ProtoMessage() {}

func (x *NativeSourceMember) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[116]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use NativeSourceMember.ProtoReflect.Descriptor instead.
func (*NativeSourceMember) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{116}
}

func (x *NativeSourceMember) GetMember() string {
	if x != nil {
		return x.Member
	}
	return ""
}

func (x *NativeSourceMember) GetObject() *Ref {
	if x != nil {
		return x.Object
	}
	return nil
}

func (x *NativeSourceMember) GetUrl() string {
	if x != nil {
		return x.Url
	}
	return ""
}

type NativeSourceSelection struct {
	state           protoimpl.MessageState `protogen:"open.v1"`
	Canonical       string                 `protobuf:"bytes,1,opt,name=canonical,proto3" json:"canonical,omitempty"`
	SelectionDigest []byte                 `protobuf:"bytes,2,opt,name=selection_digest,json=selectionDigest,proto3" json:"selection_digest,omitempty"` // accepted provider provenance
	ContentManifest *Ref                   `protobuf:"bytes,3,opt,name=content_manifest,json=contentManifest,proto3" json:"content_manifest,omitempty"` // ordinary-file content identity, independent of origin
	Members         []*NativeSourceMember  `protobuf:"bytes,4,rep,name=members,proto3" json:"members,omitempty"`                                        // exact sorted roster; <=4096 and frame byte cap
	AllowedHosts    []string               `protobuf:"bytes,5,rep,name=allowed_hosts,json=allowedHosts,proto3" json:"allowed_hosts,omitempty"`
	CredentialHosts []string               `protobuf:"bytes,6,rep,name=credential_hosts,json=credentialHosts,proto3" json:"credential_hosts,omitempty"`
	unknownFields   protoimpl.UnknownFields
	sizeCache       protoimpl.SizeCache
}

func (x *NativeSourceSelection) Reset() {
	*x = NativeSourceSelection{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[117]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *NativeSourceSelection) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*NativeSourceSelection) ProtoMessage() {}

func (x *NativeSourceSelection) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[117]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use NativeSourceSelection.ProtoReflect.Descriptor instead.
func (*NativeSourceSelection) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{117}
}

func (x *NativeSourceSelection) GetCanonical() string {
	if x != nil {
		return x.Canonical
	}
	return ""
}

func (x *NativeSourceSelection) GetSelectionDigest() []byte {
	if x != nil {
		return x.SelectionDigest
	}
	return nil
}

func (x *NativeSourceSelection) GetContentManifest() *Ref {
	if x != nil {
		return x.ContentManifest
	}
	return nil
}

func (x *NativeSourceSelection) GetMembers() []*NativeSourceMember {
	if x != nil {
		return x.Members
	}
	return nil
}

func (x *NativeSourceSelection) GetAllowedHosts() []string {
	if x != nil {
		return x.AllowedHosts
	}
	return nil
}

func (x *NativeSourceSelection) GetCredentialHosts() []string {
	if x != nil {
		return x.CredentialHosts
	}
	return nil
}

type NativeSourceCommand struct {
	state              protoimpl.MessageState `protogen:"open.v1"`
	RecordOwnerEpoch   uint64                 `protobuf:"varint,1,opt,name=record_owner_epoch,json=recordOwnerEpoch,proto3" json:"record_owner_epoch,omitempty"`
	ControlStreamEpoch uint64                 `protobuf:"varint,2,opt,name=control_stream_epoch,json=controlStreamEpoch,proto3" json:"control_stream_epoch,omitempty"`
	WorkerBootId       string                 `protobuf:"bytes,3,opt,name=worker_boot_id,json=workerBootId,proto3" json:"worker_boot_id,omitempty"`
	ParentCall         *ChildCallRequest      `protobuf:"bytes,4,opt,name=parent_call,json=parentCall,proto3" json:"parent_call,omitempty"` // exact original call; current attempt independently fenced
	ServiceId          string                 `protobuf:"bytes,5,opt,name=service_id,json=serviceId,proto3" json:"service_id,omitempty"`
	Operation          NativeSourceOperation  `protobuf:"varint,6,opt,name=operation,proto3,enum=cozy.worker.v1.NativeSourceOperation" json:"operation,omitempty"`
	Phase              NativeSourcePhase      `protobuf:"varint,7,opt,name=phase,proto3,enum=cozy.worker.v1.NativeSourcePhase" json:"phase,omitempty"`
	Selection          *NativeSourceSelection `protobuf:"bytes,8,opt,name=selection,proto3" json:"selection,omitempty"`   // EXECUTE downloads only; immutable accepted pin
	Credential         string                 `protobuf:"bytes,9,opt,name=credential,proto3" json:"credential,omitempty"` // scoped provider credential; never logs, receipts or workspace rows
	unknownFields      protoimpl.UnknownFields
	sizeCache          protoimpl.SizeCache
}

func (x *NativeSourceCommand) Reset() {
	*x = NativeSourceCommand{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[118]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *NativeSourceCommand) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*NativeSourceCommand) ProtoMessage() {}

func (x *NativeSourceCommand) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[118]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use NativeSourceCommand.ProtoReflect.Descriptor instead.
func (*NativeSourceCommand) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{118}
}

func (x *NativeSourceCommand) GetRecordOwnerEpoch() uint64 {
	if x != nil {
		return x.RecordOwnerEpoch
	}
	return 0
}

func (x *NativeSourceCommand) GetControlStreamEpoch() uint64 {
	if x != nil {
		return x.ControlStreamEpoch
	}
	return 0
}

func (x *NativeSourceCommand) GetWorkerBootId() string {
	if x != nil {
		return x.WorkerBootId
	}
	return ""
}

func (x *NativeSourceCommand) GetParentCall() *ChildCallRequest {
	if x != nil {
		return x.ParentCall
	}
	return nil
}

func (x *NativeSourceCommand) GetServiceId() string {
	if x != nil {
		return x.ServiceId
	}
	return ""
}

func (x *NativeSourceCommand) GetOperation() NativeSourceOperation {
	if x != nil {
		return x.Operation
	}
	return NativeSourceOperation_NATIVE_SOURCE_OPERATION_UNSPECIFIED
}

func (x *NativeSourceCommand) GetPhase() NativeSourcePhase {
	if x != nil {
		return x.Phase
	}
	return NativeSourcePhase_NATIVE_SOURCE_PHASE_UNSPECIFIED
}

func (x *NativeSourceCommand) GetSelection() *NativeSourceSelection {
	if x != nil {
		return x.Selection
	}
	return nil
}

func (x *NativeSourceCommand) GetCredential() string {
	if x != nil {
		return x.Credential
	}
	return ""
}

type NativeSourceStatus struct {
	state                          protoimpl.MessageState `protogen:"open.v1"`
	RecordOwnerEpoch               uint64                 `protobuf:"varint,1,opt,name=record_owner_epoch,json=recordOwnerEpoch,proto3" json:"record_owner_epoch,omitempty"`
	ControlStreamEpoch             uint64                 `protobuf:"varint,2,opt,name=control_stream_epoch,json=controlStreamEpoch,proto3" json:"control_stream_epoch,omitempty"`
	WorkerBootId                   string                 `protobuf:"bytes,3,opt,name=worker_boot_id,json=workerBootId,proto3" json:"worker_boot_id,omitempty"`
	ParentRequestId                string                 `protobuf:"bytes,4,opt,name=parent_request_id,json=parentRequestId,proto3" json:"parent_request_id,omitempty"`
	ParentAttemptOrdinal           uint64                 `protobuf:"varint,5,opt,name=parent_attempt_ordinal,json=parentAttemptOrdinal,proto3" json:"parent_attempt_ordinal,omitempty"`
	ParentInvocationSpecDigest     []byte                 `protobuf:"bytes,6,opt,name=parent_invocation_spec_digest,json=parentInvocationSpecDigest,proto3" json:"parent_invocation_spec_digest,omitempty"`
	CallIndex                      uint32                 `protobuf:"varint,7,opt,name=call_index,json=callIndex,proto3" json:"call_index,omitempty"`
	IntentDigest                   []byte                 `protobuf:"bytes,8,opt,name=intent_digest,json=intentDigest,proto3" json:"intent_digest,omitempty"`
	ServiceId                      string                 `protobuf:"bytes,9,opt,name=service_id,json=serviceId,proto3" json:"service_id,omitempty"`
	State                          NativeSourceState      `protobuf:"varint,10,opt,name=state,proto3,enum=cozy.worker.v1.NativeSourceState" json:"state,omitempty"`
	Selection                      *NativeSourceSelection `protobuf:"bytes,11,opt,name=selection,proto3" json:"selection,omitempty"`                                                     // RESOLVED only
	ResultCanonicalBytes           []byte                 `protobuf:"bytes,12,opt,name=result_canonical_bytes,json=resultCanonicalBytes,proto3" json:"result_canonical_bytes,omitempty"` // SUCCEEDED only, <=48KiB
	SafeCode                       string                 `protobuf:"bytes,13,opt,name=safe_code,json=safeCode,proto3" json:"safe_code,omitempty"`
	SafeDetail                     string                 `protobuf:"bytes,14,opt,name=safe_detail,json=safeDetail,proto3" json:"safe_detail,omitempty"`                                                        // bounded, excludes credential/URL/remote traceback
	NativeReceiptCanonicalBytes    []byte                 `protobuf:"bytes,15,opt,name=native_receipt_canonical_bytes,json=nativeReceiptCanonicalBytes,proto3" json:"native_receipt_canonical_bytes,omitempty"` // actual native result, bounded by existing receipt cap
	ComputationDigest              []byte                 `protobuf:"bytes,16,opt,name=computation_digest,json=computationDigest,proto3" json:"computation_digest,omitempty"`
	MemoHit                        bool                   `protobuf:"varint,17,opt,name=memo_hit,json=memoHit,proto3" json:"memo_hit,omitempty"`                                                                           // observation, excluded from semantic result
	ByteOutput                     *NativeByteTreeRef     `protobuf:"bytes,18,opt,name=byte_output,json=byteOutput,proto3" json:"byte_output,omitempty"`                                                                   // SOURCE_FILES or COMMIT_FILE/SUCCEEDED; parent-owned bytes
	ByteOutputAttemptOrdinal       uint64                 `protobuf:"varint,19,opt,name=byte_output_attempt_ordinal,json=byteOutputAttemptOrdinal,proto3" json:"byte_output_attempt_ordinal,omitempty"`                    // original producer attempt, possibly before reply correlation
	ByteOutputInvocationSpecDigest []byte                 `protobuf:"bytes,20,opt,name=byte_output_invocation_spec_digest,json=byteOutputInvocationSpecDigest,proto3" json:"byte_output_invocation_spec_digest,omitempty"` // exact original parent producer spec; 32 bytes
	unknownFields                  protoimpl.UnknownFields
	sizeCache                      protoimpl.SizeCache
}

func (x *NativeSourceStatus) Reset() {
	*x = NativeSourceStatus{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[119]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *NativeSourceStatus) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*NativeSourceStatus) ProtoMessage() {}

func (x *NativeSourceStatus) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[119]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use NativeSourceStatus.ProtoReflect.Descriptor instead.
func (*NativeSourceStatus) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{119}
}

func (x *NativeSourceStatus) GetRecordOwnerEpoch() uint64 {
	if x != nil {
		return x.RecordOwnerEpoch
	}
	return 0
}

func (x *NativeSourceStatus) GetControlStreamEpoch() uint64 {
	if x != nil {
		return x.ControlStreamEpoch
	}
	return 0
}

func (x *NativeSourceStatus) GetWorkerBootId() string {
	if x != nil {
		return x.WorkerBootId
	}
	return ""
}

func (x *NativeSourceStatus) GetParentRequestId() string {
	if x != nil {
		return x.ParentRequestId
	}
	return ""
}

func (x *NativeSourceStatus) GetParentAttemptOrdinal() uint64 {
	if x != nil {
		return x.ParentAttemptOrdinal
	}
	return 0
}

func (x *NativeSourceStatus) GetParentInvocationSpecDigest() []byte {
	if x != nil {
		return x.ParentInvocationSpecDigest
	}
	return nil
}

func (x *NativeSourceStatus) GetCallIndex() uint32 {
	if x != nil {
		return x.CallIndex
	}
	return 0
}

func (x *NativeSourceStatus) GetIntentDigest() []byte {
	if x != nil {
		return x.IntentDigest
	}
	return nil
}

func (x *NativeSourceStatus) GetServiceId() string {
	if x != nil {
		return x.ServiceId
	}
	return ""
}

func (x *NativeSourceStatus) GetState() NativeSourceState {
	if x != nil {
		return x.State
	}
	return NativeSourceState_NATIVE_SOURCE_STATE_UNSPECIFIED
}

func (x *NativeSourceStatus) GetSelection() *NativeSourceSelection {
	if x != nil {
		return x.Selection
	}
	return nil
}

func (x *NativeSourceStatus) GetResultCanonicalBytes() []byte {
	if x != nil {
		return x.ResultCanonicalBytes
	}
	return nil
}

func (x *NativeSourceStatus) GetSafeCode() string {
	if x != nil {
		return x.SafeCode
	}
	return ""
}

func (x *NativeSourceStatus) GetSafeDetail() string {
	if x != nil {
		return x.SafeDetail
	}
	return ""
}

func (x *NativeSourceStatus) GetNativeReceiptCanonicalBytes() []byte {
	if x != nil {
		return x.NativeReceiptCanonicalBytes
	}
	return nil
}

func (x *NativeSourceStatus) GetComputationDigest() []byte {
	if x != nil {
		return x.ComputationDigest
	}
	return nil
}

func (x *NativeSourceStatus) GetMemoHit() bool {
	if x != nil {
		return x.MemoHit
	}
	return false
}

func (x *NativeSourceStatus) GetByteOutput() *NativeByteTreeRef {
	if x != nil {
		return x.ByteOutput
	}
	return nil
}

func (x *NativeSourceStatus) GetByteOutputAttemptOrdinal() uint64 {
	if x != nil {
		return x.ByteOutputAttemptOrdinal
	}
	return 0
}

func (x *NativeSourceStatus) GetByteOutputInvocationSpecDigest() []byte {
	if x != nil {
		return x.ByteOutputInvocationSpecDigest
	}
	return nil
}

type Claim struct {
	state              protoimpl.MessageState `protogen:"open.v1"`
	RecordOwnerEpoch   uint64                 `protobuf:"varint,1,opt,name=record_owner_epoch,json=recordOwnerEpoch,proto3" json:"record_owner_epoch,omitempty"`       // envelope; the claimed authority epoch
	ControlStreamEpoch uint64                 `protobuf:"varint,2,opt,name=control_stream_epoch,json=controlStreamEpoch,proto3" json:"control_stream_epoch,omitempty"` // 0 on Claim (none minted yet)
	WorkerBootId       string                 `protobuf:"bytes,3,opt,name=worker_boot_id,json=workerBootId,proto3" json:"worker_boot_id,omitempty"`                    // empty on a fresh dial; set when re-dialing a known boot
	RecordOwnerId      string                 `protobuf:"bytes,5,opt,name=record_owner_id,json=recordOwnerId,proto3" json:"record_owner_id,omitempty"`                 // stable identity of the claiming RecordOwner
	WorkerId           string                 `protobuf:"bytes,6,opt,name=worker_id,json=workerId,proto3" json:"worker_id,omitempty"`                                  // the worker identity the RecordOwner expects to be claiming
	WireMinor          uint32                 `protobuf:"varint,7,opt,name=wire_minor,json=wireMinor,proto3" json:"wire_minor,omitempty"`                              // highest minor the RECORDOWNER implements
	Proof              []byte                 `protobuf:"bytes,8,opt,name=proof,proto3" json:"proof,omitempty"`                                                        // 64-byte Ed25519 signature over canonical ClaimProof/1 on
	unknownFields      protoimpl.UnknownFields
	sizeCache          protoimpl.SizeCache
}

func (x *Claim) Reset() {
	*x = Claim{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[120]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *Claim) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*Claim) ProtoMessage() {}

func (x *Claim) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[120]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use Claim.ProtoReflect.Descriptor instead.
func (*Claim) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{120}
}

func (x *Claim) GetRecordOwnerEpoch() uint64 {
	if x != nil {
		return x.RecordOwnerEpoch
	}
	return 0
}

func (x *Claim) GetControlStreamEpoch() uint64 {
	if x != nil {
		return x.ControlStreamEpoch
	}
	return 0
}

func (x *Claim) GetWorkerBootId() string {
	if x != nil {
		return x.WorkerBootId
	}
	return ""
}

func (x *Claim) GetRecordOwnerId() string {
	if x != nil {
		return x.RecordOwnerId
	}
	return ""
}

func (x *Claim) GetWorkerId() string {
	if x != nil {
		return x.WorkerId
	}
	return ""
}

func (x *Claim) GetWireMinor() uint32 {
	if x != nil {
		return x.WireMinor
	}
	return 0
}

func (x *Claim) GetProof() []byte {
	if x != nil {
		return x.Proof
	}
	return nil
}

// DOCUMENT SHAPE (not a wire message): canonical form `cozy.worker.v1.ClaimProof/1`.
// The Cozy daemon signs these exact reconstructed bytes with the rental key before its first
// Claim. There is no challenge, timestamp, jti, or application request-signature grammar: epoch,
// boot, worker, and exact TLS leaf bind replay to this worker lifetime.
type ClaimProof struct {
	state                      protoimpl.MessageState `protogen:"open.v1"`
	RecordOwnerEpoch           uint64                 `protobuf:"varint,1,opt,name=record_owner_epoch,json=recordOwnerEpoch,proto3" json:"record_owner_epoch,omitempty"`
	WorkerBootId               string                 `protobuf:"bytes,2,opt,name=worker_boot_id,json=workerBootId,proto3" json:"worker_boot_id,omitempty"`
	WorkerId                   string                 `protobuf:"bytes,3,opt,name=worker_id,json=workerId,proto3" json:"worker_id,omitempty"`
	WorkerTlsCertificateDigest []byte                 `protobuf:"bytes,4,opt,name=worker_tls_certificate_digest,json=workerTlsCertificateDigest,proto3" json:"worker_tls_certificate_digest,omitempty"`
	unknownFields              protoimpl.UnknownFields
	sizeCache                  protoimpl.SizeCache
}

func (x *ClaimProof) Reset() {
	*x = ClaimProof{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[121]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *ClaimProof) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*ClaimProof) ProtoMessage() {}

func (x *ClaimProof) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[121]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use ClaimProof.ProtoReflect.Descriptor instead.
func (*ClaimProof) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{121}
}

func (x *ClaimProof) GetRecordOwnerEpoch() uint64 {
	if x != nil {
		return x.RecordOwnerEpoch
	}
	return 0
}

func (x *ClaimProof) GetWorkerBootId() string {
	if x != nil {
		return x.WorkerBootId
	}
	return ""
}

func (x *ClaimProof) GetWorkerId() string {
	if x != nil {
		return x.WorkerId
	}
	return ""
}

func (x *ClaimProof) GetWorkerTlsCertificateDigest() []byte {
	if x != nil {
		return x.WorkerTlsCertificateDigest
	}
	return nil
}

// Claim proof is evaluated before readiness or ownership mutation. A bad proof answers
// UNAUTHENTICATED even before ready. A valid proof while the worker's post-bind readiness barrier
// is closed answers UNDURABLE without minting a control_stream_epoch. On acceptance the worker
// fences the previous stream (if any), increments control_stream_epoch, and answers with its
// identity + one WorkerSnapshot. DISPATCH STAYS CLOSED until the RecordOwner's SnapshotAck
// (02 §6) — admission_state reports CLOSED until then.
type ClaimAck struct {
	state              protoimpl.MessageState `protogen:"open.v1"`
	RecordOwnerEpoch   uint64                 `protobuf:"varint,1,opt,name=record_owner_epoch,json=recordOwnerEpoch,proto3" json:"record_owner_epoch,omitempty"`       // echo of the accepted epoch
	ControlStreamEpoch uint64                 `protobuf:"varint,2,opt,name=control_stream_epoch,json=controlStreamEpoch,proto3" json:"control_stream_epoch,omitempty"` // newly minted for this stream
	WorkerBootId       string                 `protobuf:"bytes,3,opt,name=worker_boot_id,json=workerBootId,proto3" json:"worker_boot_id,omitempty"`                    // this boot's identity
	Accepted           bool                   `protobuf:"varint,5,opt,name=accepted,proto3" json:"accepted,omitempty"`                                                 // false => `rejection` set; stream then closes
	Rejection          ClaimRejection         `protobuf:"varint,6,opt,name=rejection,proto3,enum=cozy.worker.v1.ClaimRejection" json:"rejection,omitempty"`
	WireMinor          uint32                 `protobuf:"varint,7,opt,name=wire_minor,json=wireMinor,proto3" json:"wire_minor,omitempty"` // highest minor the WORKER implements
	WorkerId           string                 `protobuf:"bytes,8,opt,name=worker_id,json=workerId,proto3" json:"worker_id,omitempty"`     // CREDENTIAL subject; cross-checked vs the worker's server
	// identity (02 §4)
	WorkerInstanceId string `protobuf:"bytes,9,opt,name=worker_instance_id,json=workerInstanceId,proto3" json:"worker_instance_id,omitempty"` // ONE provisioned instance lifetime; outcome-replay
	// authorization keys on this, never worker_id
	WorkerReleaseId      string `protobuf:"bytes,10,opt,name=worker_release_id,json=workerReleaseId,proto3" json:"worker_release_id,omitempty"`
	ControlRuntimeDigest string `protobuf:"bytes,11,opt,name=control_runtime_digest,json=controlRuntimeDigest,proto3" json:"control_runtime_digest,omitempty"` // class (b), `sha256:<hex>`: exact image-owned control
	// Runtime wheel bytes measured from the installed wheelhouse.
	// It is NOT the OCI image identity: an image cannot bake its
	// own manifest digest without a self-reference. The provisioner
	// independently verifies the selected immutable OCI identity;
	// a RecordOwner needs BOTH facts to admit the machine.
	GitCommit     string           `protobuf:"bytes,12,opt,name=git_commit,json=gitCommit,proto3" json:"git_commit,omitempty"`
	Resources     *WorkerResources `protobuf:"bytes,13,opt,name=resources,proto3" json:"resources,omitempty"` // torch-free worker statics; never re-sent mid-stream.
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *ClaimAck) Reset() {
	*x = ClaimAck{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[122]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *ClaimAck) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*ClaimAck) ProtoMessage() {}

func (x *ClaimAck) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[122]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use ClaimAck.ProtoReflect.Descriptor instead.
func (*ClaimAck) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{122}
}

func (x *ClaimAck) GetRecordOwnerEpoch() uint64 {
	if x != nil {
		return x.RecordOwnerEpoch
	}
	return 0
}

func (x *ClaimAck) GetControlStreamEpoch() uint64 {
	if x != nil {
		return x.ControlStreamEpoch
	}
	return 0
}

func (x *ClaimAck) GetWorkerBootId() string {
	if x != nil {
		return x.WorkerBootId
	}
	return ""
}

func (x *ClaimAck) GetAccepted() bool {
	if x != nil {
		return x.Accepted
	}
	return false
}

func (x *ClaimAck) GetRejection() ClaimRejection {
	if x != nil {
		return x.Rejection
	}
	return ClaimRejection_CLAIM_REJECTION_UNSPECIFIED
}

func (x *ClaimAck) GetWireMinor() uint32 {
	if x != nil {
		return x.WireMinor
	}
	return 0
}

func (x *ClaimAck) GetWorkerId() string {
	if x != nil {
		return x.WorkerId
	}
	return ""
}

func (x *ClaimAck) GetWorkerInstanceId() string {
	if x != nil {
		return x.WorkerInstanceId
	}
	return ""
}

func (x *ClaimAck) GetWorkerReleaseId() string {
	if x != nil {
		return x.WorkerReleaseId
	}
	return ""
}

func (x *ClaimAck) GetControlRuntimeDigest() string {
	if x != nil {
		return x.ControlRuntimeDigest
	}
	return ""
}

func (x *ClaimAck) GetGitCommit() string {
	if x != nil {
		return x.GitCommit
	}
	return ""
}

func (x *ClaimAck) GetResources() *WorkerResources {
	if x != nil {
		return x.Resources
	}
	return nil
}

// Authenticated boot-fatal delivery: the worker accepted a stream but cannot serve.
type BootFailure struct {
	state                protoimpl.MessageState `protogen:"open.v1"`
	RecordOwnerEpoch     uint64                 `protobuf:"varint,1,opt,name=record_owner_epoch,json=recordOwnerEpoch,proto3" json:"record_owner_epoch,omitempty"`
	ControlStreamEpoch   uint64                 `protobuf:"varint,2,opt,name=control_stream_epoch,json=controlStreamEpoch,proto3" json:"control_stream_epoch,omitempty"`
	WorkerBootId         string                 `protobuf:"bytes,3,opt,name=worker_boot_id,json=workerBootId,proto3" json:"worker_boot_id,omitempty"`
	WorkerId             string                 `protobuf:"bytes,5,opt,name=worker_id,json=workerId,proto3" json:"worker_id,omitempty"`
	WorkerInstanceId     string                 `protobuf:"bytes,6,opt,name=worker_instance_id,json=workerInstanceId,proto3" json:"worker_instance_id,omitempty"`
	Reason               BootFailureReason      `protobuf:"varint,7,opt,name=reason,proto3,enum=cozy.worker.v1.BootFailureReason" json:"reason,omitempty"`
	Detail               string                 `protobuf:"bytes,8,opt,name=detail,proto3" json:"detail,omitempty"`                                                            // bounded <= 1024 bytes, sanitized
	Resources            *WorkerResources       `protobuf:"bytes,9,opt,name=resources,proto3" json:"resources,omitempty"`                                                      // whatever was measurable
	ControlRuntimeDigest string                 `protobuf:"bytes,10,opt,name=control_runtime_digest,json=controlRuntimeDigest,proto3" json:"control_runtime_digest,omitempty"` // same measured control Runtime provenance as ClaimAck 11
	WorkerReleaseId      string                 `protobuf:"bytes,11,opt,name=worker_release_id,json=workerReleaseId,proto3" json:"worker_release_id,omitempty"`
	unknownFields        protoimpl.UnknownFields
	sizeCache            protoimpl.SizeCache
}

func (x *BootFailure) Reset() {
	*x = BootFailure{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[123]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *BootFailure) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*BootFailure) ProtoMessage() {}

func (x *BootFailure) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[123]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use BootFailure.ProtoReflect.Descriptor instead.
func (*BootFailure) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{123}
}

func (x *BootFailure) GetRecordOwnerEpoch() uint64 {
	if x != nil {
		return x.RecordOwnerEpoch
	}
	return 0
}

func (x *BootFailure) GetControlStreamEpoch() uint64 {
	if x != nil {
		return x.ControlStreamEpoch
	}
	return 0
}

func (x *BootFailure) GetWorkerBootId() string {
	if x != nil {
		return x.WorkerBootId
	}
	return ""
}

func (x *BootFailure) GetWorkerId() string {
	if x != nil {
		return x.WorkerId
	}
	return ""
}

func (x *BootFailure) GetWorkerInstanceId() string {
	if x != nil {
		return x.WorkerInstanceId
	}
	return ""
}

func (x *BootFailure) GetReason() BootFailureReason {
	if x != nil {
		return x.Reason
	}
	return BootFailureReason_BOOT_FAILURE_REASON_UNSPECIFIED
}

func (x *BootFailure) GetDetail() string {
	if x != nil {
		return x.Detail
	}
	return ""
}

func (x *BootFailure) GetResources() *WorkerResources {
	if x != nil {
		return x.Resources
	}
	return nil
}

func (x *BootFailure) GetControlRuntimeDigest() string {
	if x != nil {
		return x.ControlRuntimeDigest
	}
	return ""
}

func (x *BootFailure) GetWorkerReleaseId() string {
	if x != nil {
		return x.WorkerReleaseId
	}
	return ""
}

type WorkerSnapshot struct {
	state                              protoimpl.MessageState `protogen:"open.v1"`
	RecordOwnerEpoch                   uint64                 `protobuf:"varint,1,opt,name=record_owner_epoch,json=recordOwnerEpoch,proto3" json:"record_owner_epoch,omitempty"`
	ControlStreamEpoch                 uint64                 `protobuf:"varint,2,opt,name=control_stream_epoch,json=controlStreamEpoch,proto3" json:"control_stream_epoch,omitempty"`
	WorkerBootId                       string                 `protobuf:"bytes,3,opt,name=worker_boot_id,json=workerBootId,proto3" json:"worker_boot_id,omitempty"`
	SnapshotId                         string                 `protobuf:"bytes,5,opt,name=snapshot_id,json=snapshotId,proto3" json:"snapshot_id,omitempty"`                                                                               // worker-minted; names this exact snapshot
	SnapshotDigest                     []byte                 `protobuf:"bytes,6,opt,name=snapshot_digest,json=snapshotDigest,proto3" json:"snapshot_digest,omitempty"`                                                                   // class (a): sha256 over EXACTLY the bytes in 7
	SnapshotCanonicalBytes             []byte                 `protobuf:"bytes,7,opt,name=snapshot_canonical_bytes,json=snapshotCanonicalBytes,proto3" json:"snapshot_canonical_bytes,omitempty"`                                         // the WorkerSnapshotBody DOCUMENT, canonical JSON bytes
	AcceptedPlacementSetCanonicalBytes []byte                 `protobuf:"bytes,8,opt,name=accepted_placement_set_canonical_bytes,json=acceptedPlacementSetCanonicalBytes,proto3" json:"accepted_placement_set_canonical_bytes,omitempty"` // the EXACT accepted set bytes as journaled
	// — never a re-serialization; its digest lives INSIDE 7
	// THE HOST'S OWN DOCUMENT (proto-025). A canonical document under a digest cannot be appended
	// to -- any addition re-authors it -- so a host standing between the RecordOwner and the
	// worker gets a second document beside the worker's, and 6/7 pass through BYTE-IDENTICAL.
	// Both empty when no host stands between them (local execution).
	HostSnapshotDigest         []byte `protobuf:"bytes,9,opt,name=host_snapshot_digest,json=hostSnapshotDigest,proto3" json:"host_snapshot_digest,omitempty"`                            // class (a): sha256 over EXACTLY the bytes in 10
	HostSnapshotCanonicalBytes []byte `protobuf:"bytes,10,opt,name=host_snapshot_canonical_bytes,json=hostSnapshotCanonicalBytes,proto3" json:"host_snapshot_canonical_bytes,omitempty"` // the HostSnapshotBody DOCUMENT, canonical JSON
	unknownFields              protoimpl.UnknownFields
	sizeCache                  protoimpl.SizeCache
}

func (x *WorkerSnapshot) Reset() {
	*x = WorkerSnapshot{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[124]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *WorkerSnapshot) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*WorkerSnapshot) ProtoMessage() {}

func (x *WorkerSnapshot) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[124]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use WorkerSnapshot.ProtoReflect.Descriptor instead.
func (*WorkerSnapshot) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{124}
}

func (x *WorkerSnapshot) GetRecordOwnerEpoch() uint64 {
	if x != nil {
		return x.RecordOwnerEpoch
	}
	return 0
}

func (x *WorkerSnapshot) GetControlStreamEpoch() uint64 {
	if x != nil {
		return x.ControlStreamEpoch
	}
	return 0
}

func (x *WorkerSnapshot) GetWorkerBootId() string {
	if x != nil {
		return x.WorkerBootId
	}
	return ""
}

func (x *WorkerSnapshot) GetSnapshotId() string {
	if x != nil {
		return x.SnapshotId
	}
	return ""
}

func (x *WorkerSnapshot) GetSnapshotDigest() []byte {
	if x != nil {
		return x.SnapshotDigest
	}
	return nil
}

func (x *WorkerSnapshot) GetSnapshotCanonicalBytes() []byte {
	if x != nil {
		return x.SnapshotCanonicalBytes
	}
	return nil
}

func (x *WorkerSnapshot) GetAcceptedPlacementSetCanonicalBytes() []byte {
	if x != nil {
		return x.AcceptedPlacementSetCanonicalBytes
	}
	return nil
}

func (x *WorkerSnapshot) GetHostSnapshotDigest() []byte {
	if x != nil {
		return x.HostSnapshotDigest
	}
	return nil
}

func (x *WorkerSnapshot) GetHostSnapshotCanonicalBytes() []byte {
	if x != nil {
		return x.HostSnapshotCanonicalBytes
	}
	return nil
}

// DOCUMENT SHAPE (not a wire message): canonical form `cozy.worker.v1.WorkerSnapshotBody/1`, the
// subject of WorkerSnapshot.snapshot_digest.
//
// NAMING (implementation resolution, rev §5): the rev document calls both the wire message and
// the document `WorkerSnapshot`, which one proto package cannot express. The document takes the
// `Body` suffix, exactly as AttemptOutcome/AttemptOutcomeBody already does, so the wire message
// keeps the name the rev's proto sketch and WorkerFrame slot 12 give it.
//
// BOUNDEDNESS IS STRUCTURAL, NOT A PROMISE (gpu-hot.md §3):
// the post phase is bounded ONE attempt per lane, so a lane's device takes its next occupant
// only while at most one DEVICE_RELEASED attempt on it is unacked — a RecordOwner that stops
// acking idles its own device rather than growing the worker's held set. Queue room is a
// host-bytes fact (§4) and does not move with acks.
type WorkerSnapshotBody struct {
	state                        protoimpl.MessageState `protogen:"open.v1"`
	AcceptedDesiredStateRevision uint64                 `protobuf:"varint,1,opt,name=accepted_desired_state_revision,json=acceptedDesiredStateRevision,proto3" json:"accepted_desired_state_revision,omitempty"` // durably ACCEPTED intent
	AcceptedPlacementSetDigest   []byte                 `protobuf:"bytes,2,opt,name=accepted_placement_set_digest,json=acceptedPlacementSetDigest,proto3" json:"accepted_placement_set_digest,omitempty"`        // class (a); equals sha256 of the bytes the
	// enclosing WorkerSnapshot carries in field 8
	WorkerPhase           WorkerPhase        `protobuf:"varint,4,opt,name=worker_phase,json=workerPhase,proto3,enum=cozy.worker.v1.WorkerPhase" json:"worker_phase,omitempty"` // machine lifecycle
	Placements            []*PlacementStatus `protobuf:"bytes,5,rep,name=placements,proto3" json:"placements,omitempty"`                                                       // sorted by placement_id
	ConvergedRevision     uint64             `protobuf:"varint,6,opt,name=converged_revision,json=convergedRevision,proto3" json:"converged_revision,omitempty"`               // advances ONLY when observed satisfies accepted
	AdmissionEpoch        uint64             `protobuf:"varint,7,opt,name=admission_epoch,json=admissionEpoch,proto3" json:"admission_epoch,omitempty"`
	AdmissionState        AdmissionState     `protobuf:"varint,8,opt,name=admission_state,json=admissionState,proto3,enum=cozy.worker.v1.AdmissionState" json:"admission_state,omitempty"`
	AvailableAttemptSlots uint32             `protobuf:"varint,9,opt,name=available_attempt_slots,json=availableAttemptSlots,proto3" json:"available_attempt_slots,omitempty"` // Σ lanes[].available_attempt_slots: offers the
	// worker will still admit to its queues now
	HeldAttempts []*HeldAttempt `protobuf:"bytes,10,rep,name=held_attempts,json=heldAttempts,proto3" json:"held_attempts,omitempty"` // every accepted attempt from admission to ack
	// (QUEUED, RUNNING, DEVICE_RELEASED,
	// OUTCOME_PENDING_ACK); sorted by
	// (request_id, attempt_ordinal)
	Lanes []*DeviceLane `protobuf:"bytes,12,rep,name=lanes,proto3" json:"lanes,omitempty"` // proto-024: the lanes as of this snapshot,
	// sorted by lane_id — the per-lane room a
	// reconnecting RecordOwner reconciles from,
	// beside the worker-level sum in 9; each row
	// carries its resident_placement_ids (proto-026)
	HeldManifests []string `protobuf:"bytes,13,rep,name=held_manifests,json=heldManifests,proto3" json:"held_manifests,omitempty"` // proto-026 residency: the same set as
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *WorkerSnapshotBody) Reset() {
	*x = WorkerSnapshotBody{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[125]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *WorkerSnapshotBody) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*WorkerSnapshotBody) ProtoMessage() {}

func (x *WorkerSnapshotBody) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[125]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use WorkerSnapshotBody.ProtoReflect.Descriptor instead.
func (*WorkerSnapshotBody) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{125}
}

func (x *WorkerSnapshotBody) GetAcceptedDesiredStateRevision() uint64 {
	if x != nil {
		return x.AcceptedDesiredStateRevision
	}
	return 0
}

func (x *WorkerSnapshotBody) GetAcceptedPlacementSetDigest() []byte {
	if x != nil {
		return x.AcceptedPlacementSetDigest
	}
	return nil
}

func (x *WorkerSnapshotBody) GetWorkerPhase() WorkerPhase {
	if x != nil {
		return x.WorkerPhase
	}
	return WorkerPhase_WORKER_PHASE_UNSPECIFIED
}

func (x *WorkerSnapshotBody) GetPlacements() []*PlacementStatus {
	if x != nil {
		return x.Placements
	}
	return nil
}

func (x *WorkerSnapshotBody) GetConvergedRevision() uint64 {
	if x != nil {
		return x.ConvergedRevision
	}
	return 0
}

func (x *WorkerSnapshotBody) GetAdmissionEpoch() uint64 {
	if x != nil {
		return x.AdmissionEpoch
	}
	return 0
}

func (x *WorkerSnapshotBody) GetAdmissionState() AdmissionState {
	if x != nil {
		return x.AdmissionState
	}
	return AdmissionState_ADMISSION_STATE_UNSPECIFIED
}

func (x *WorkerSnapshotBody) GetAvailableAttemptSlots() uint32 {
	if x != nil {
		return x.AvailableAttemptSlots
	}
	return 0
}

func (x *WorkerSnapshotBody) GetHeldAttempts() []*HeldAttempt {
	if x != nil {
		return x.HeldAttempts
	}
	return nil
}

func (x *WorkerSnapshotBody) GetLanes() []*DeviceLane {
	if x != nil {
		return x.Lanes
	}
	return nil
}

func (x *WorkerSnapshotBody) GetHeldManifests() []string {
	if x != nil {
		return x.HeldManifests
	}
	return nil
}

// DOCUMENT SHAPE (not a wire message): canonical form `cozy.worker.v1.HostSnapshotBody/1`, the
// subject of WorkerSnapshot.host_snapshot_digest. Authored by the pod host from its ledger at
// every worker snapshot; never by the worker.
//
// held_outcomes are the ledger's outcomes pending ack that the worker's own held_attempts does
// NOT name -- the survivor set after a child replacement. Each replays as an AttemptOutcome
// after the barrier, exactly like the worker's own. A live child's held_attempts always covers
// the ledger's, so this list is empty until a child is replaced. weights_transactions announce
// the receipt frames the host replays after the barrier. Lanes that moved to PodHost have no
// post-barrier replay and therefore no row here: the owner asks the host.
type HostSnapshotBody struct {
	state        protoimpl.MessageState `protogen:"open.v1"`
	HeldOutcomes []*HeldAttempt         `protobuf:"bytes,1,rep,name=held_outcomes,json=heldOutcomes,proto3" json:"held_outcomes,omitempty"` // sorted by (request_id, attempt_ordinal); state is
	// always OUTCOME_PENDING_ACK
	WeightsTransactions     []*WeightsTransactionStatus `protobuf:"bytes,2,rep,name=weights_transactions,json=weightsTransactions,proto3" json:"weights_transactions,omitempty"`                // sorted by weights_transaction_id
	RetainedDesiredRevision uint64                      `protobuf:"varint,3,opt,name=retained_desired_revision,json=retainedDesiredRevision,proto3" json:"retained_desired_revision,omitempty"` // Host ledger high-water, not Runtime acceptance or convergence.
	unknownFields           protoimpl.UnknownFields
	sizeCache               protoimpl.SizeCache
}

func (x *HostSnapshotBody) Reset() {
	*x = HostSnapshotBody{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[126]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *HostSnapshotBody) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*HostSnapshotBody) ProtoMessage() {}

func (x *HostSnapshotBody) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[126]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use HostSnapshotBody.ProtoReflect.Descriptor instead.
func (*HostSnapshotBody) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{126}
}

func (x *HostSnapshotBody) GetHeldOutcomes() []*HeldAttempt {
	if x != nil {
		return x.HeldOutcomes
	}
	return nil
}

func (x *HostSnapshotBody) GetWeightsTransactions() []*WeightsTransactionStatus {
	if x != nil {
		return x.WeightsTransactions
	}
	return nil
}

func (x *HostSnapshotBody) GetRetainedDesiredRevision() uint64 {
	if x != nil {
		return x.RetainedDesiredRevision
	}
	return 0
}

type SnapshotAck struct {
	state              protoimpl.MessageState `protogen:"open.v1"`
	RecordOwnerEpoch   uint64                 `protobuf:"varint,1,opt,name=record_owner_epoch,json=recordOwnerEpoch,proto3" json:"record_owner_epoch,omitempty"`
	ControlStreamEpoch uint64                 `protobuf:"varint,2,opt,name=control_stream_epoch,json=controlStreamEpoch,proto3" json:"control_stream_epoch,omitempty"`
	WorkerBootId       string                 `protobuf:"bytes,3,opt,name=worker_boot_id,json=workerBootId,proto3" json:"worker_boot_id,omitempty"`
	SnapshotId         string                 `protobuf:"bytes,5,opt,name=snapshot_id,json=snapshotId,proto3" json:"snapshot_id,omitempty"`             // echo of the exact snapshot durably reconciled
	SnapshotDigest     []byte                 `protobuf:"bytes,6,opt,name=snapshot_digest,json=snapshotDigest,proto3" json:"snapshot_digest,omitempty"` // echo; COMPARED, never recomputed by the worker. An ack
	// naming a different id/digest is NOT an ack — the barrier
	// stays closed and dispatch never opens.
	HostSnapshotDigest []byte `protobuf:"bytes,7,opt,name=host_snapshot_digest,json=hostSnapshotDigest,proto3" json:"host_snapshot_digest,omitempty"` // echo of WorkerSnapshot 9; the host compares it before
	unknownFields      protoimpl.UnknownFields
	sizeCache          protoimpl.SizeCache
}

func (x *SnapshotAck) Reset() {
	*x = SnapshotAck{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[127]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *SnapshotAck) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*SnapshotAck) ProtoMessage() {}

func (x *SnapshotAck) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[127]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use SnapshotAck.ProtoReflect.Descriptor instead.
func (*SnapshotAck) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{127}
}

func (x *SnapshotAck) GetRecordOwnerEpoch() uint64 {
	if x != nil {
		return x.RecordOwnerEpoch
	}
	return 0
}

func (x *SnapshotAck) GetControlStreamEpoch() uint64 {
	if x != nil {
		return x.ControlStreamEpoch
	}
	return 0
}

func (x *SnapshotAck) GetWorkerBootId() string {
	if x != nil {
		return x.WorkerBootId
	}
	return ""
}

func (x *SnapshotAck) GetSnapshotId() string {
	if x != nil {
		return x.SnapshotId
	}
	return ""
}

func (x *SnapshotAck) GetSnapshotDigest() []byte {
	if x != nil {
		return x.SnapshotDigest
	}
	return nil
}

func (x *SnapshotAck) GetHostSnapshotDigest() []byte {
	if x != nil {
		return x.HostSnapshotDigest
	}
	return nil
}

type DesiredWorkerState struct {
	state              protoimpl.MessageState `protogen:"open.v1"`
	RecordOwnerEpoch   uint64                 `protobuf:"varint,1,opt,name=record_owner_epoch,json=recordOwnerEpoch,proto3" json:"record_owner_epoch,omitempty"`
	ControlStreamEpoch uint64                 `protobuf:"varint,2,opt,name=control_stream_epoch,json=controlStreamEpoch,proto3" json:"control_stream_epoch,omitempty"`
	WorkerBootId       string                 `protobuf:"bytes,3,opt,name=worker_boot_id,json=workerBootId,proto3" json:"worker_boot_id,omitempty"`
	Revision           uint64                 `protobuf:"varint,5,opt,name=revision,proto3" json:"revision,omitempty"` // RecordOwner-owned, monotonic; older ignored, equal
	// idempotent, newer applies; equal revision + different
	// bytes is a protocol error (02 §3 — now byte-checkable
	// rather than claim-checkable)
	Posture      Posture `protobuf:"varint,6,opt,name=posture,proto3,enum=cozy.worker.v1.Posture" json:"posture,omitempty"`
	WireMinor    uint32  `protobuf:"varint,7,opt,name=wire_minor,json=wireMinor,proto3" json:"wire_minor,omitempty"`            // the minor the RecordOwner speaks on this stream; echoed
	DrainGraceMs uint64  `protobuf:"varint,8,opt,name=drain_grace_ms,json=drainGraceMs,proto3" json:"drain_grace_ms,omitempty"` // draining posture only: grace before forceful; 0 = the
	// worker's runtime-constant default
	//
	// Types that are valid to be assigned to Mode:
	//
	//	*DesiredWorkerState_Job
	//	*DesiredWorkerState_PlacementSet
	Mode isDesiredWorkerState_Mode `protobuf_oneof:"mode"`
	// A supervisor-owned Runtime/TensorFS replacement target. The worker must reject a
	// malformed or unsupported revision before staging bytes.
	RuntimeRevision *RuntimeRevision `protobuf:"bytes,19,opt,name=runtime_revision,json=runtimeRevision,proto3" json:"runtime_revision,omitempty"`
	unknownFields   protoimpl.UnknownFields
	sizeCache       protoimpl.SizeCache
}

func (x *DesiredWorkerState) Reset() {
	*x = DesiredWorkerState{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[128]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *DesiredWorkerState) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*DesiredWorkerState) ProtoMessage() {}

func (x *DesiredWorkerState) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[128]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use DesiredWorkerState.ProtoReflect.Descriptor instead.
func (*DesiredWorkerState) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{128}
}

func (x *DesiredWorkerState) GetRecordOwnerEpoch() uint64 {
	if x != nil {
		return x.RecordOwnerEpoch
	}
	return 0
}

func (x *DesiredWorkerState) GetControlStreamEpoch() uint64 {
	if x != nil {
		return x.ControlStreamEpoch
	}
	return 0
}

func (x *DesiredWorkerState) GetWorkerBootId() string {
	if x != nil {
		return x.WorkerBootId
	}
	return ""
}

func (x *DesiredWorkerState) GetRevision() uint64 {
	if x != nil {
		return x.Revision
	}
	return 0
}

func (x *DesiredWorkerState) GetPosture() Posture {
	if x != nil {
		return x.Posture
	}
	return Posture_POSTURE_UNSPECIFIED
}

func (x *DesiredWorkerState) GetWireMinor() uint32 {
	if x != nil {
		return x.WireMinor
	}
	return 0
}

func (x *DesiredWorkerState) GetDrainGraceMs() uint64 {
	if x != nil {
		return x.DrainGraceMs
	}
	return 0
}

func (x *DesiredWorkerState) GetMode() isDesiredWorkerState_Mode {
	if x != nil {
		return x.Mode
	}
	return nil
}

func (x *DesiredWorkerState) GetJob() *JobDirective {
	if x != nil {
		if x, ok := x.Mode.(*DesiredWorkerState_Job); ok {
			return x.Job
		}
	}
	return nil
}

func (x *DesiredWorkerState) GetPlacementSet() *DesiredPlacementSet {
	if x != nil {
		if x, ok := x.Mode.(*DesiredWorkerState_PlacementSet); ok {
			return x.PlacementSet
		}
	}
	return nil
}

func (x *DesiredWorkerState) GetRuntimeRevision() *RuntimeRevision {
	if x != nil {
		return x.RuntimeRevision
	}
	return nil
}

type isDesiredWorkerState_Mode interface {
	isDesiredWorkerState_Mode()
}

type DesiredWorkerState_Job struct {
	Job *JobDirective `protobuf:"bytes,13,opt,name=job,proto3,oneof"`
}

type DesiredWorkerState_PlacementSet struct {
	PlacementSet *DesiredPlacementSet `protobuf:"bytes,15,opt,name=placement_set,json=placementSet,proto3,oneof"`
}

func (*DesiredWorkerState_Job) isDesiredWorkerState_Mode() {}

func (*DesiredWorkerState_PlacementSet) isDesiredWorkerState_Mode() {}

type DesiredPackageSet struct {
	state                       protoimpl.MessageState `protogen:"open.v1"`
	DownloadDelegation          []byte                 `protobuf:"bytes,1,opt,name=download_delegation,json=downloadDelegation,proto3" json:"download_delegation,omitempty"`                              // exact canonical DownloadDelegation/1 bytes
	DownloadDelegationSignature []byte                 `protobuf:"bytes,2,opt,name=download_delegation_signature,json=downloadDelegationSignature,proto3" json:"download_delegation_signature,omitempty"` // 64-byte Ed25519 signature by the rental Creator key
	unknownFields               protoimpl.UnknownFields
	sizeCache                   protoimpl.SizeCache
}

func (x *DesiredPackageSet) Reset() {
	*x = DesiredPackageSet{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[129]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *DesiredPackageSet) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*DesiredPackageSet) ProtoMessage() {}

func (x *DesiredPackageSet) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[129]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use DesiredPackageSet.ProtoReflect.Descriptor instead.
func (*DesiredPackageSet) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{129}
}

func (x *DesiredPackageSet) GetDownloadDelegation() []byte {
	if x != nil {
		return x.DownloadDelegation
	}
	return nil
}

func (x *DesiredPackageSet) GetDownloadDelegationSignature() []byte {
	if x != nil {
		return x.DownloadDelegationSignature
	}
	return nil
}

// An operation-scoped private install. Code transfers directly to the worker.
// Source files have no fingerprint; ordinary wheel artifacts retain uv integrity.
type DesiredLocalPackageSet struct {
	state       protoimpl.MessageState `protogen:"open.v1"`
	OperationId string                 `protobuf:"bytes,1,opt,name=operation_id,json=operationId,proto3" json:"operation_id,omitempty"`
	Package     *DevelopmentPackage    `protobuf:"bytes,2,opt,name=package,proto3" json:"package,omitempty"`
	Files       []*LocalPackageFileRef `protobuf:"bytes,3,rep,name=files,proto3" json:"files,omitempty"` // sorted uniquely by filename
	// Optional ordinary locked requirements for wheel-only installs.
	DependencyRequirements []byte `protobuf:"bytes,4,opt,name=dependency_requirements,json=dependencyRequirements,proto3" json:"dependency_requirements,omitempty"` // <= MaxLockedRequirementsBytes
	// Declared PEP 440 Python range and optional exact captured version.
	// Worker intersects these with its installed rolling supported window before install.
	PythonRequires string `protobuf:"bytes,5,opt,name=python_requires,json=pythonRequires,proto3" json:"python_requires,omitempty"`
	PythonVersion  string `protobuf:"bytes,6,opt,name=python_version,json=pythonVersion,proto3" json:"python_version,omitempty"`
	SourceArchive  string `protobuf:"bytes,7,opt,name=source_archive,json=sourceArchive,proto3" json:"source_archive,omitempty"` // filename in files; root contains pyproject.toml and uv.lock
	unknownFields  protoimpl.UnknownFields
	sizeCache      protoimpl.SizeCache
}

func (x *DesiredLocalPackageSet) Reset() {
	*x = DesiredLocalPackageSet{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[130]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *DesiredLocalPackageSet) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*DesiredLocalPackageSet) ProtoMessage() {}

func (x *DesiredLocalPackageSet) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[130]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use DesiredLocalPackageSet.ProtoReflect.Descriptor instead.
func (*DesiredLocalPackageSet) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{130}
}

func (x *DesiredLocalPackageSet) GetOperationId() string {
	if x != nil {
		return x.OperationId
	}
	return ""
}

func (x *DesiredLocalPackageSet) GetPackage() *DevelopmentPackage {
	if x != nil {
		return x.Package
	}
	return nil
}

func (x *DesiredLocalPackageSet) GetFiles() []*LocalPackageFileRef {
	if x != nil {
		return x.Files
	}
	return nil
}

func (x *DesiredLocalPackageSet) GetDependencyRequirements() []byte {
	if x != nil {
		return x.DependencyRequirements
	}
	return nil
}

func (x *DesiredLocalPackageSet) GetPythonRequires() string {
	if x != nil {
		return x.PythonRequires
	}
	return ""
}

func (x *DesiredLocalPackageSet) GetPythonVersion() string {
	if x != nil {
		return x.PythonVersion
	}
	return ""
}

func (x *DesiredLocalPackageSet) GetSourceArchive() string {
	if x != nil {
		return x.SourceArchive
	}
	return ""
}

// Bind selected models to an existing installation. The model-only delegation
// authorizes downloads; it is independent of package content.
type DesiredPrivatePlacementSet struct {
	state                       protoimpl.MessageState `protogen:"open.v1"`
	OperationId                 string                 `protobuf:"bytes,1,opt,name=operation_id,json=operationId,proto3" json:"operation_id,omitempty"`
	DownloadDelegation          []byte                 `protobuf:"bytes,3,opt,name=download_delegation,json=downloadDelegation,proto3" json:"download_delegation,omitempty"`
	DownloadDelegationSignature []byte                 `protobuf:"bytes,4,opt,name=download_delegation_signature,json=downloadDelegationSignature,proto3" json:"download_delegation_signature,omitempty"` // 64-byte Ed25519 signature by rental Creator key
	// These retained slots and the download set must be disjoint.
	NativeModels   []*NativeModelBinding `protobuf:"bytes,5,rep,name=native_models,json=nativeModels,proto3" json:"native_models,omitempty"`
	InstallationId string                `protobuf:"bytes,6,opt,name=installation_id,json=installationId,proto3" json:"installation_id,omitempty"`
	// Wire 70: forwarded unchanged to Runtime's PreparePrivatePlacementRequest. A sender
	// relying on these fields requires a wire-70 Host and Runtime.model_overrides.
	ModelChoices      []*ModelChoice      `protobuf:"bytes,7,rep,name=model_choices,json=modelChoices,proto3" json:"model_choices,omitempty"`
	SourceCredentials []*SourceCredential `protobuf:"bytes,8,rep,name=source_credentials,json=sourceCredentials,proto3" json:"source_credentials,omitempty"` // memory-only, excluded from identity
	Hub               string              `protobuf:"bytes,9,opt,name=hub,proto3" json:"hub,omitempty"`
	Owner             string              `protobuf:"bytes,10,opt,name=owner,proto3" json:"owner,omitempty"`
	unknownFields     protoimpl.UnknownFields
	sizeCache         protoimpl.SizeCache
}

func (x *DesiredPrivatePlacementSet) Reset() {
	*x = DesiredPrivatePlacementSet{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[131]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *DesiredPrivatePlacementSet) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*DesiredPrivatePlacementSet) ProtoMessage() {}

func (x *DesiredPrivatePlacementSet) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[131]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use DesiredPrivatePlacementSet.ProtoReflect.Descriptor instead.
func (*DesiredPrivatePlacementSet) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{131}
}

func (x *DesiredPrivatePlacementSet) GetOperationId() string {
	if x != nil {
		return x.OperationId
	}
	return ""
}

func (x *DesiredPrivatePlacementSet) GetDownloadDelegation() []byte {
	if x != nil {
		return x.DownloadDelegation
	}
	return nil
}

func (x *DesiredPrivatePlacementSet) GetDownloadDelegationSignature() []byte {
	if x != nil {
		return x.DownloadDelegationSignature
	}
	return nil
}

func (x *DesiredPrivatePlacementSet) GetNativeModels() []*NativeModelBinding {
	if x != nil {
		return x.NativeModels
	}
	return nil
}

func (x *DesiredPrivatePlacementSet) GetInstallationId() string {
	if x != nil {
		return x.InstallationId
	}
	return ""
}

func (x *DesiredPrivatePlacementSet) GetModelChoices() []*ModelChoice {
	if x != nil {
		return x.ModelChoices
	}
	return nil
}

func (x *DesiredPrivatePlacementSet) GetSourceCredentials() []*SourceCredential {
	if x != nil {
		return x.SourceCredentials
	}
	return nil
}

func (x *DesiredPrivatePlacementSet) GetHub() string {
	if x != nil {
		return x.Hub
	}
	return ""
}

func (x *DesiredPrivatePlacementSet) GetOwner() string {
	if x != nil {
		return x.Owner
	}
	return ""
}

// A private model already retained for this owner. Sorted uniquely by full slot
// path, 1..MaxModelSlotPaths rows. Runtime verifies the exact live retention before using
// any cached preparation; release of an equal retention ID cannot replay a stale success.
// The label is provenance/display only; manifest and retention are the byte authority.
type NativeModelBinding struct {
	state         protoimpl.MessageState   `protogen:"open.v1"`
	Slot          string                   `protobuf:"bytes,1,opt,name=slot,proto3" json:"slot,omitempty"`         // full declared <entrypoint>.models.<parameter> path
	Model         string                   `protobuf:"bytes,2,opt,name=model,proto3" json:"model,omitempty"`       // existing private producer/output label, never a fetch address
	Manifest      *Ref                     `protobuf:"bytes,3,opt,name=manifest,proto3" json:"manifest,omitempty"` // exact SHA-256 and positive canonical manifest length
	Retention     *DerivedRetentionRequest `protobuf:"bytes,4,opt,name=retention,proto3" json:"retention,omitempty"`
	Adapters      []*DownloadAdapterRef    `protobuf:"bytes,5,rep,name=adapters,proto3" json:"adapters,omitempty"` // ordinary adapter checkpoints; base custody stays native.
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *NativeModelBinding) Reset() {
	*x = NativeModelBinding{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[132]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *NativeModelBinding) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*NativeModelBinding) ProtoMessage() {}

func (x *NativeModelBinding) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[132]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use NativeModelBinding.ProtoReflect.Descriptor instead.
func (*NativeModelBinding) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{132}
}

func (x *NativeModelBinding) GetSlot() string {
	if x != nil {
		return x.Slot
	}
	return ""
}

func (x *NativeModelBinding) GetModel() string {
	if x != nil {
		return x.Model
	}
	return ""
}

func (x *NativeModelBinding) GetManifest() *Ref {
	if x != nil {
		return x.Manifest
	}
	return nil
}

func (x *NativeModelBinding) GetRetention() *DerivedRetentionRequest {
	if x != nil {
		return x.Retention
	}
	return nil
}

func (x *NativeModelBinding) GetAdapters() []*DownloadAdapterRef {
	if x != nil {
		return x.Adapters
	}
	return nil
}

type LocalPackageFileRef struct {
	state         protoimpl.MessageState `protogen:"open.v1"`
	Digest        []byte                 `protobuf:"bytes,1,opt,name=digest,proto3" json:"digest,omitempty"` // ordinary wheel integrity only; empty for source_archive
	Filename      string                 `protobuf:"bytes,2,opt,name=filename,proto3" json:"filename,omitempty"`
	Length        uint64                 `protobuf:"varint,4,opt,name=length,proto3" json:"length,omitempty"`
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *LocalPackageFileRef) Reset() {
	*x = LocalPackageFileRef{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[133]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *LocalPackageFileRef) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*LocalPackageFileRef) ProtoMessage() {}

func (x *LocalPackageFileRef) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[133]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use LocalPackageFileRef.ProtoReflect.Descriptor instead.
func (*LocalPackageFileRef) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{133}
}

func (x *LocalPackageFileRef) GetDigest() []byte {
	if x != nil {
		return x.Digest
	}
	return nil
}

func (x *LocalPackageFileRef) GetFilename() string {
	if x != nil {
		return x.Filename
	}
	return ""
}

func (x *LocalPackageFileRef) GetLength() uint64 {
	if x != nil {
		return x.Length
	}
	return 0
}

// The serving-mode desired state, GENUINELY content-addressed (§4; was DeploymentSetDirective).
// The one-digester ECHO is replaced by a byte fence in the exact shape InvocationSpec already
// uses, and the duplicate structured representation is DELETED — there is no second place for
// the set to disagree with itself.
//
// ORDER OF OPERATIONS, NON-NEGOTIABLE: recompute sha256(placement_set_canonical_bytes) and
// compare BEFORE parsing a single field; on match, journal the EXACT accepted bytes and their
// digest (never a re-serialization, never a structured projection); only then parse under
// unknown-field refusal. Recovery, status attribution, and the snapshot all quote the journaled
// bytes. A mismatch is FAULT_KIND_PLACEMENT_SET_DIGEST_MISMATCH, UNAPPLIED, with
// accepted_desired_state_revision NOT advancing and the previous set still serving.
type DesiredPlacementSet struct {
	state                      protoimpl.MessageState `protogen:"open.v1"`
	PlacementSetDigest         []byte                 `protobuf:"bytes,1,opt,name=placement_set_digest,json=placementSetDigest,proto3" json:"placement_set_digest,omitempty"`                           // class (a): sha256 over EXACTLY the bytes in 3
	PlacementSetCanonicalBytes []byte                 `protobuf:"bytes,3,opt,name=placement_set_canonical_bytes,json=placementSetCanonicalBytes,proto3" json:"placement_set_canonical_bytes,omitempty"` // the PlacementSet DOCUMENT, canonical JSON bytes
	DevicePins                 []*PlacementDevicePin  `protobuf:"bytes,4,rep,name=device_pins,json=devicePins,proto3" json:"device_pins,omitempty"`                                                     // proto-024: sorted unique by placement_id;
	// absent for a placement ⇒ the worker assigns a 1-device
	// lane by measured fit. OUTSIDE the digest on purpose.
	// One unchanged CPU orchestration parent may coexist with serving children.
	// Operational, outside PlacementSet bytes/digest, like device_pins. Requires
	// orchestration=true, device_count=0, no device request/memory and no nested parent;
	// Runtime also verifies its actual frozen job interface has no Model/Weights capability.
	OrchestrationParent *JobDirective `protobuf:"bytes,5,opt,name=orchestration_parent,json=orchestrationParent,proto3" json:"orchestration_parent,omitempty"`
	// Machine executions only (MachineExecutionSubmit.prepared_state): the exact width of the
	// device group Runtime grants the offered placement; Runtime picks the ordinals. Zero lets
	// Runtime choose the widest degree every model slot declares. Operational and outside the
	// digest, like device_pins.
	ExecutionGpus uint32 `protobuf:"varint,6,opt,name=execution_gpus,json=executionGpus,proto3" json:"execution_gpus,omitempty"`
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *DesiredPlacementSet) Reset() {
	*x = DesiredPlacementSet{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[134]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *DesiredPlacementSet) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*DesiredPlacementSet) ProtoMessage() {}

func (x *DesiredPlacementSet) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[134]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use DesiredPlacementSet.ProtoReflect.Descriptor instead.
func (*DesiredPlacementSet) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{134}
}

func (x *DesiredPlacementSet) GetPlacementSetDigest() []byte {
	if x != nil {
		return x.PlacementSetDigest
	}
	return nil
}

func (x *DesiredPlacementSet) GetPlacementSetCanonicalBytes() []byte {
	if x != nil {
		return x.PlacementSetCanonicalBytes
	}
	return nil
}

func (x *DesiredPlacementSet) GetDevicePins() []*PlacementDevicePin {
	if x != nil {
		return x.DevicePins
	}
	return nil
}

func (x *DesiredPlacementSet) GetOrchestrationParent() *JobDirective {
	if x != nil {
		return x.OrchestrationParent
	}
	return nil
}

func (x *DesiredPlacementSet) GetExecutionGpus() uint32 {
	if x != nil {
		return x.ExecutionGpus
	}
	return 0
}

// Where a placement sits on THIS worker (proto-024). Ordinals are envelope-local (the worker's
// `--devices` order) and sorted unique; K > 1 pins the placement to a GROUP lane whose ONE
// executor is sealed to all K devices and whose construction must be declared for degree K.
// The pin is part of neither placement-set identity nor invocation identity; re-pinning is a
// new desired-state revision over the SAME bytes. A pin the worker cannot honour — an ordinal
// outside the envelope, a device another placement holds, a fit that fails, a package with no
// construction declared for K — is a typed placement fault (FAULT_KIND_CONFIG_REFUSED, reason
// `device_pin_infeasible` / `device_group_unsupported`); other placements are untouched.
type PlacementDevicePin struct {
	state          protoimpl.MessageState `protogen:"open.v1"`
	PlacementId    string                 `protobuf:"bytes,1,opt,name=placement_id,json=placementId,proto3" json:"placement_id,omitempty"`                  // must name one placement in placement_set_canonical_bytes
	DeviceOrdinals []uint32               `protobuf:"varint,2,rep,packed,name=device_ordinals,json=deviceOrdinals,proto3" json:"device_ordinals,omitempty"` // K >= 1, sorted unique, envelope-local
	unknownFields  protoimpl.UnknownFields
	sizeCache      protoimpl.SizeCache
}

func (x *PlacementDevicePin) Reset() {
	*x = PlacementDevicePin{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[135]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *PlacementDevicePin) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*PlacementDevicePin) ProtoMessage() {}

func (x *PlacementDevicePin) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[135]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use PlacementDevicePin.ProtoReflect.Descriptor instead.
func (*PlacementDevicePin) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{135}
}

func (x *PlacementDevicePin) GetPlacementId() string {
	if x != nil {
		return x.PlacementId
	}
	return ""
}

func (x *PlacementDevicePin) GetDeviceOrdinals() []uint32 {
	if x != nil {
		return x.DeviceOrdinals
	}
	return nil
}

// DOCUMENT SHAPE (not a wire message): canonical form `cozy.worker.v1.PlacementSet/1`.
// Placements sorted by placement_id. One placement per package; a placement holds every
// construction its slots bind.
type PlacementSet struct {
	state         protoimpl.MessageState `protogen:"open.v1"`
	Placements    []*Placement           `protobuf:"bytes,1,rep,name=placements,proto3" json:"placements,omitempty"`
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *PlacementSet) Reset() {
	*x = PlacementSet{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[136]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *PlacementSet) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*PlacementSet) ProtoMessage() {}

func (x *PlacementSet) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[136]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use PlacementSet.ProtoReflect.Descriptor instead.
func (*PlacementSet) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{136}
}

func (x *PlacementSet) GetPlacements() []*Placement {
	if x != nil {
		return x.Placements
	}
	return nil
}

// DOCUMENT SHAPE: canonical `cozy.worker.v1.DownloadDelegation/1`. Creator signs the exact
// canonical bytes. Replay is intentionally valid until expiry; Tensorhub keeps no nonce state.
type DownloadDelegation struct {
	state         protoimpl.MessageState `protogen:"open.v1"`
	ExpiresAtUnix uint64                 `protobuf:"varint,1,opt,name=expires_at_unix,json=expiresAtUnix,proto3" json:"expires_at_unix,omitempty"`
	Models        []*DownloadModelRef    `protobuf:"bytes,2,rep,name=models,proto3" json:"models,omitempty"`     // sorted unique by (package,slot,model,release,manifest)
	Packages      []*DownloadPackageRef  `protobuf:"bytes,3,rep,name=packages,proto3" json:"packages,omitempty"` // sorted unique by (package,release); empty only for
	// DesiredPrivatePlacementSet model binding
	RentalId                   string `protobuf:"bytes,4,opt,name=rental_id,json=rentalId,proto3" json:"rental_id,omitempty"`
	WorkerBootId               string `protobuf:"bytes,5,opt,name=worker_boot_id,json=workerBootId,proto3" json:"worker_boot_id,omitempty"`
	WorkerId                   string `protobuf:"bytes,6,opt,name=worker_id,json=workerId,proto3" json:"worker_id,omitempty"`
	WorkerTlsCertificateDigest []byte `protobuf:"bytes,7,opt,name=worker_tls_certificate_digest,json=workerTlsCertificateDigest,proto3" json:"worker_tls_certificate_digest,omitempty"` // sha256 of exact readiness-pinned leaf DER
	unknownFields              protoimpl.UnknownFields
	sizeCache                  protoimpl.SizeCache
}

func (x *DownloadDelegation) Reset() {
	*x = DownloadDelegation{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[137]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *DownloadDelegation) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*DownloadDelegation) ProtoMessage() {}

func (x *DownloadDelegation) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[137]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use DownloadDelegation.ProtoReflect.Descriptor instead.
func (*DownloadDelegation) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{137}
}

func (x *DownloadDelegation) GetExpiresAtUnix() uint64 {
	if x != nil {
		return x.ExpiresAtUnix
	}
	return 0
}

func (x *DownloadDelegation) GetModels() []*DownloadModelRef {
	if x != nil {
		return x.Models
	}
	return nil
}

func (x *DownloadDelegation) GetPackages() []*DownloadPackageRef {
	if x != nil {
		return x.Packages
	}
	return nil
}

func (x *DownloadDelegation) GetRentalId() string {
	if x != nil {
		return x.RentalId
	}
	return ""
}

func (x *DownloadDelegation) GetWorkerBootId() string {
	if x != nil {
		return x.WorkerBootId
	}
	return ""
}

func (x *DownloadDelegation) GetWorkerId() string {
	if x != nil {
		return x.WorkerId
	}
	return ""
}

func (x *DownloadDelegation) GetWorkerTlsCertificateDigest() []byte {
	if x != nil {
		return x.WorkerTlsCertificateDigest
	}
	return nil
}

type DownloadModelRef struct {
	state    protoimpl.MessageState `protogen:"open.v1"`
	Manifest string                 `protobuf:"bytes,1,opt,name=manifest,proto3" json:"manifest,omitempty"` // sha256:<64 lowercase hex>
	Model    string                 `protobuf:"bytes,2,opt,name=model,proto3" json:"model,omitempty"`       // org/name
	Release  string                 `protobuf:"bytes,3,opt,name=release,proto3" json:"release,omitempty"`   // immutable author release
	Package  string                 `protobuf:"bytes,4,opt,name=package,proto3" json:"package,omitempty"`   // one package named by this delegation
	Slot     string                 `protobuf:"bytes,5,opt,name=slot,proto3" json:"slot,omitempty"`         // exact package-interface model-slot path; unique per package
	Lane     string                 `protobuf:"bytes,6,opt,name=lane,proto3" json:"lane,omitempty"`         // the release lane the manifest was resolved from (e.g.
	// bf16, fp8). The manifest is identity; the lane is the
	// repository pointer the worker's placement row spells,
	// so a pod names the same lane the owner resolved.
	Adapters      []*DownloadAdapterRef `protobuf:"bytes,7,rep,name=adapters,proto3" json:"adapters,omitempty"` // ORDERED per-request stack for this model slot.
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *DownloadModelRef) Reset() {
	*x = DownloadModelRef{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[138]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *DownloadModelRef) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*DownloadModelRef) ProtoMessage() {}

func (x *DownloadModelRef) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[138]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use DownloadModelRef.ProtoReflect.Descriptor instead.
func (*DownloadModelRef) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{138}
}

func (x *DownloadModelRef) GetManifest() string {
	if x != nil {
		return x.Manifest
	}
	return ""
}

func (x *DownloadModelRef) GetModel() string {
	if x != nil {
		return x.Model
	}
	return ""
}

func (x *DownloadModelRef) GetRelease() string {
	if x != nil {
		return x.Release
	}
	return ""
}

func (x *DownloadModelRef) GetPackage() string {
	if x != nil {
		return x.Package
	}
	return ""
}

func (x *DownloadModelRef) GetSlot() string {
	if x != nil {
		return x.Slot
	}
	return ""
}

func (x *DownloadModelRef) GetLane() string {
	if x != nil {
		return x.Lane
	}
	return ""
}

func (x *DownloadModelRef) GetAdapters() []*DownloadAdapterRef {
	if x != nil {
		return x.Adapters
	}
	return nil
}

// A model selection used as an adapter. Download delegations contain exact immutable
// catalog checkpoints; ModelChoice may instead ask the machine to resolve a provider
// source. The selected base component is explicit; no executable adapter code is sent.
type DownloadAdapterRef struct {
	state           protoimpl.MessageState `protogen:"open.v1"`
	Component       string                 `protobuf:"bytes,1,opt,name=component,proto3" json:"component,omitempty"` // empty: infer one compatible base component, refuse if ambiguous
	Model           string                 `protobuf:"bytes,2,opt,name=model,proto3" json:"model,omitempty"`
	Release         string                 `protobuf:"bytes,3,opt,name=release,proto3" json:"release,omitempty"`
	Lane            string                 `protobuf:"bytes,4,opt,name=lane,proto3" json:"lane,omitempty"`
	Manifest        string                 `protobuf:"bytes,5,opt,name=manifest,proto3" json:"manifest,omitempty"`
	SourceComponent string                 `protobuf:"bytes,6,opt,name=source_component,json=sourceComponent,proto3" json:"source_component,omitempty"` // empty selects "adapter"
	Scale           string                 `protobuf:"bytes,7,opt,name=scale,proto3" json:"scale,omitempty"`                                            // finite canonical decimal string; empty means 1; zero/negative are valid.
	// Wire 70: immutable provider selection, resolved by the machine using normal model
	// source preparation. Exclusive with model/release/lane/manifest. These choice fields
	// are absent from materialized download delegations, whose manifests are already pinned.
	Source        string   `protobuf:"bytes,8,opt,name=source,proto3" json:"source,omitempty"`
	Profiles      []string `protobuf:"bytes,9,rep,name=profiles,proto3" json:"profiles,omitempty"`
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *DownloadAdapterRef) Reset() {
	*x = DownloadAdapterRef{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[139]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *DownloadAdapterRef) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*DownloadAdapterRef) ProtoMessage() {}

func (x *DownloadAdapterRef) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[139]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use DownloadAdapterRef.ProtoReflect.Descriptor instead.
func (*DownloadAdapterRef) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{139}
}

func (x *DownloadAdapterRef) GetComponent() string {
	if x != nil {
		return x.Component
	}
	return ""
}

func (x *DownloadAdapterRef) GetModel() string {
	if x != nil {
		return x.Model
	}
	return ""
}

func (x *DownloadAdapterRef) GetRelease() string {
	if x != nil {
		return x.Release
	}
	return ""
}

func (x *DownloadAdapterRef) GetLane() string {
	if x != nil {
		return x.Lane
	}
	return ""
}

func (x *DownloadAdapterRef) GetManifest() string {
	if x != nil {
		return x.Manifest
	}
	return ""
}

func (x *DownloadAdapterRef) GetSourceComponent() string {
	if x != nil {
		return x.SourceComponent
	}
	return ""
}

func (x *DownloadAdapterRef) GetScale() string {
	if x != nil {
		return x.Scale
	}
	return ""
}

func (x *DownloadAdapterRef) GetSource() string {
	if x != nil {
		return x.Source
	}
	return ""
}

func (x *DownloadAdapterRef) GetProfiles() []string {
	if x != nil {
		return x.Profiles
	}
	return nil
}

type DownloadPackageRef struct {
	state         protoimpl.MessageState `protogen:"open.v1"`
	Package       string                 `protobuf:"bytes,1,opt,name=package,proto3" json:"package,omitempty"` // org/name
	Release       string                 `protobuf:"bytes,2,opt,name=release,proto3" json:"release,omitempty"` // exact semver release
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *DownloadPackageRef) Reset() {
	*x = DownloadPackageRef{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[140]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *DownloadPackageRef) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*DownloadPackageRef) ProtoMessage() {}

func (x *DownloadPackageRef) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[140]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use DownloadPackageRef.ProtoReflect.Descriptor instead.
func (*DownloadPackageRef) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{140}
}

func (x *DownloadPackageRef) GetPackage() string {
	if x != nil {
		return x.Package
	}
	return ""
}

func (x *DownloadPackageRef) GetRelease() string {
	if x != nil {
		return x.Release
	}
	return ""
}

type Placement struct {
	state       protoimpl.MessageState `protogen:"open.v1"`
	PlacementId string                 `protobuf:"bytes,1,opt,name=placement_id,json=placementId,proto3" json:"placement_id,omitempty"` // "package-" + hex(sha256(package \0 release))[:24]:
	// stable as models are added. Routing + journal key; NEVER
	// part of invocation identity (InvocationSpec digests
	// exclude it)
	//
	// Types that are valid to be assigned to PackageMode:
	//
	//	*Placement_Package
	//	*Placement_Development
	PackageMode    isPlacement_PackageMode `protobuf_oneof:"package_mode"`
	BindingsDigest []byte                  `protobuf:"bytes,6,opt,name=bindings_digest,json=bindingsDigest,proto3" json:"bindings_digest,omitempty"` // class (a): sha256(canonical JSON of exactly
	// {entrypoints:<field 8>,models:<field 7>})
	Models           []*Model      `protobuf:"bytes,7,rep,name=models,proto3" json:"models,omitempty"`                                              // sorted unique by id; [] for a weightless package
	Entrypoints      []*Entrypoint `protobuf:"bytes,8,rep,name=entrypoints,proto3" json:"entrypoints,omitempty"`                                    // sorted unique by name
	InstallationId   string        `protobuf:"bytes,12,opt,name=installation_id,json=installationId,proto3" json:"installation_id,omitempty"`       // worker-owned installed environment lifetime
	PackageInterface []byte        `protobuf:"bytes,13,opt,name=package_interface,json=packageInterface,proto3" json:"package_interface,omitempty"` // interface discovered in that installed environment
	unknownFields    protoimpl.UnknownFields
	sizeCache        protoimpl.SizeCache
}

func (x *Placement) Reset() {
	*x = Placement{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[141]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *Placement) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*Placement) ProtoMessage() {}

func (x *Placement) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[141]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use Placement.ProtoReflect.Descriptor instead.
func (*Placement) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{141}
}

func (x *Placement) GetPlacementId() string {
	if x != nil {
		return x.PlacementId
	}
	return ""
}

func (x *Placement) GetPackageMode() isPlacement_PackageMode {
	if x != nil {
		return x.PackageMode
	}
	return nil
}

func (x *Placement) GetPackage() *PackageSelection {
	if x != nil {
		if x, ok := x.PackageMode.(*Placement_Package); ok {
			return x.Package
		}
	}
	return nil
}

func (x *Placement) GetDevelopment() *DevelopmentPackage {
	if x != nil {
		if x, ok := x.PackageMode.(*Placement_Development); ok {
			return x.Development
		}
	}
	return nil
}

func (x *Placement) GetBindingsDigest() []byte {
	if x != nil {
		return x.BindingsDigest
	}
	return nil
}

func (x *Placement) GetModels() []*Model {
	if x != nil {
		return x.Models
	}
	return nil
}

func (x *Placement) GetEntrypoints() []*Entrypoint {
	if x != nil {
		return x.Entrypoints
	}
	return nil
}

func (x *Placement) GetInstallationId() string {
	if x != nil {
		return x.InstallationId
	}
	return ""
}

func (x *Placement) GetPackageInterface() []byte {
	if x != nil {
		return x.PackageInterface
	}
	return nil
}

type isPlacement_PackageMode interface {
	isPlacement_PackageMode()
}

type Placement_Package struct {
	Package *PackageSelection `protobuf:"bytes,2,opt,name=package,proto3,oneof"` // installed published release
}

type Placement_Development struct {
	Development *DevelopmentPackage `protobuf:"bytes,11,opt,name=development,proto3,oneof"` // explicit source-checkout execution
}

func (*Placement_Package) isPlacement_PackageMode() {}

func (*Placement_Development) isPlacement_PackageMode() {}

// Exact immutable bytes. Nested placement facts use this one reference shape instead of generic
// artifact subjects, role strings, or digest-only pointers.
type Ref struct {
	state         protoimpl.MessageState `protogen:"open.v1"`
	Digest        []byte                 `protobuf:"bytes,1,opt,name=digest,proto3" json:"digest,omitempty"`
	Length        uint64                 `protobuf:"varint,2,opt,name=length,proto3" json:"length,omitempty"`
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *Ref) Reset() {
	*x = Ref{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[142]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *Ref) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*Ref) ProtoMessage() {}

func (x *Ref) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[142]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use Ref.ProtoReflect.Descriptor instead.
func (*Ref) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{142}
}

func (x *Ref) GetDigest() []byte {
	if x != nil {
		return x.Digest
	}
	return nil
}

func (x *Ref) GetLength() uint64 {
	if x != nil {
		return x.Length
	}
	return 0
}

// A published package is selected by repository and version. uv verifies its
// downloaded wheel artifacts; execution uses Runtime's installed environment.
type PackageSelection struct {
	state         protoimpl.MessageState `protogen:"open.v1"`
	Package       string                 `protobuf:"bytes,1,opt,name=package,proto3" json:"package,omitempty"` // exact org/name package identity
	Release       string                 `protobuf:"bytes,2,opt,name=release,proto3" json:"release,omitempty"` // exact semantic release
	Hub           string                 `protobuf:"bytes,5,opt,name=hub,proto3" json:"hub,omitempty"`         // Wire 67: the Tensorhub origin it is read at; empty: the default Hub
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *PackageSelection) Reset() {
	*x = PackageSelection{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[143]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *PackageSelection) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*PackageSelection) ProtoMessage() {}

func (x *PackageSelection) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[143]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use PackageSelection.ProtoReflect.Descriptor instead.
func (*PackageSelection) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{143}
}

func (x *PackageSelection) GetPackage() string {
	if x != nil {
		return x.Package
	}
	return ""
}

func (x *PackageSelection) GetRelease() string {
	if x != nil {
		return x.Release
	}
	return ""
}

func (x *PackageSelection) GetHub() string {
	if x != nil {
		return x.Hub
	}
	return ""
}

// Source sync selects an installed generation. Updates do not mutate environments
// retained by running requests. The handle is lifecycle plumbing, not a fingerprint.
type DevelopmentPackage struct {
	state          protoimpl.MessageState `protogen:"open.v1"`
	Package        string                 `protobuf:"bytes,1,opt,name=package,proto3" json:"package,omitempty"` // exact org/name package identity
	Release        string                 `protobuf:"bytes,2,opt,name=release,proto3" json:"release,omitempty"` // exact semantic release
	InstallationId string                 `protobuf:"bytes,6,opt,name=installation_id,json=installationId,proto3" json:"installation_id,omitempty"`
	unknownFields  protoimpl.UnknownFields
	sizeCache      protoimpl.SizeCache
}

func (x *DevelopmentPackage) Reset() {
	*x = DevelopmentPackage{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[144]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *DevelopmentPackage) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*DevelopmentPackage) ProtoMessage() {}

func (x *DevelopmentPackage) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[144]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use DevelopmentPackage.ProtoReflect.Descriptor instead.
func (*DevelopmentPackage) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{144}
}

func (x *DevelopmentPackage) GetPackage() string {
	if x != nil {
		return x.Package
	}
	return ""
}

func (x *DevelopmentPackage) GetRelease() string {
	if x != nil {
		return x.Release
	}
	return ""
}

func (x *DevelopmentPackage) GetInstallationId() string {
	if x != nil {
		return x.InstallationId
	}
	return ""
}

type Model struct {
	state protoimpl.MessageState `protogen:"open.v1"`
	Id    string                 `protobuf:"bytes,1,opt,name=id,proto3" json:"id,omitempty"` // "model-" + hex(sha256(slot_path))[:16]; stable as
	// other models are added
	Repo          string `protobuf:"bytes,2,opt,name=repo,proto3" json:"repo,omitempty"`         // exact org/name repository identity
	Version       string `protobuf:"bytes,3,opt,name=version,proto3" json:"version,omitempty"`   // exact model release
	Lane          string `protobuf:"bytes,4,opt,name=lane,proto3" json:"lane,omitempty"`         // exact selected lane
	Manifest      *Ref   `protobuf:"bytes,5,opt,name=manifest,proto3" json:"manifest,omitempty"` // exact TensorFS Manifest; no redundant header reference
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *Model) Reset() {
	*x = Model{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[145]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *Model) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*Model) ProtoMessage() {}

func (x *Model) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[145]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use Model.ProtoReflect.Descriptor instead.
func (*Model) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{145}
}

func (x *Model) GetId() string {
	if x != nil {
		return x.Id
	}
	return ""
}

func (x *Model) GetRepo() string {
	if x != nil {
		return x.Repo
	}
	return ""
}

func (x *Model) GetVersion() string {
	if x != nil {
		return x.Version
	}
	return ""
}

func (x *Model) GetLane() string {
	if x != nil {
		return x.Lane
	}
	return ""
}

func (x *Model) GetManifest() *Ref {
	if x != nil {
		return x.Manifest
	}
	return nil
}

type Entrypoint struct {
	state                   protoimpl.MessageState `protogen:"open.v1"`
	Name                    string                 `protobuf:"bytes,1,opt,name=name,proto3" json:"name,omitempty"`
	EntrypointBindingDigest []byte                 `protobuf:"bytes,2,opt,name=entrypoint_binding_digest,json=entrypointBindingDigest,proto3" json:"entrypoint_binding_digest,omitempty"` // sha256(canonical JSON of exactly
	// {name:<field 1>,slots:<field 3>})
	Slots         []*Slot `protobuf:"bytes,3,rep,name=slots,proto3" json:"slots,omitempty"` // sorted unique by slot; [] for a weightless entrypoint
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *Entrypoint) Reset() {
	*x = Entrypoint{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[146]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *Entrypoint) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*Entrypoint) ProtoMessage() {}

func (x *Entrypoint) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[146]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use Entrypoint.ProtoReflect.Descriptor instead.
func (*Entrypoint) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{146}
}

func (x *Entrypoint) GetName() string {
	if x != nil {
		return x.Name
	}
	return ""
}

func (x *Entrypoint) GetEntrypointBindingDigest() []byte {
	if x != nil {
		return x.EntrypointBindingDigest
	}
	return nil
}

func (x *Entrypoint) GetSlots() []*Slot {
	if x != nil {
		return x.Slots
	}
	return nil
}

type Slot struct {
	state                     protoimpl.MessageState `protogen:"open.v1"`
	Slot                      string                 `protobuf:"bytes,1,opt,name=slot,proto3" json:"slot,omitempty"`
	ReferenceModelId          string                 `protobuf:"bytes,2,opt,name=reference_model_id,json=referenceModelId,proto3" json:"reference_model_id,omitempty"`                            // must name one Placement.models id
	Components                []*Component           `protobuf:"bytes,4,rep,name=components,proto3" json:"components,omitempty"`                                                                  // ORDERED MCC destination sequence; never sorted
	ModelConstructionContract *Ref                   `protobuf:"bytes,5,opt,name=model_construction_contract,json=modelConstructionContract,proto3" json:"model_construction_contract,omitempty"` // required for qualified publication; omitted when
	// DevelopmentPackage derives it from source + local model
	Stamps        []*Stamp        `protobuf:"bytes,6,rep,name=stamps,proto3" json:"stamps,omitempty"`     // sorted unique by (component,key)
	Adapters      []*ModelAdapter `protobuf:"bytes,7,rep,name=adapters,proto3" json:"adapters,omitempty"` // ORDERED; omitted for an unadapted model slot.
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *Slot) Reset() {
	*x = Slot{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[147]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *Slot) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*Slot) ProtoMessage() {}

func (x *Slot) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[147]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use Slot.ProtoReflect.Descriptor instead.
func (*Slot) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{147}
}

func (x *Slot) GetSlot() string {
	if x != nil {
		return x.Slot
	}
	return ""
}

func (x *Slot) GetReferenceModelId() string {
	if x != nil {
		return x.ReferenceModelId
	}
	return ""
}

func (x *Slot) GetComponents() []*Component {
	if x != nil {
		return x.Components
	}
	return nil
}

func (x *Slot) GetModelConstructionContract() *Ref {
	if x != nil {
		return x.ModelConstructionContract
	}
	return nil
}

func (x *Slot) GetStamps() []*Stamp {
	if x != nil {
		return x.Stamps
	}
	return nil
}

func (x *Slot) GetAdapters() []*ModelAdapter {
	if x != nil {
		return x.Adapters
	}
	return nil
}

type ModelAdapter struct {
	state           protoimpl.MessageState `protogen:"open.v1"`
	Component       string                 `protobuf:"bytes,1,opt,name=component,proto3" json:"component,omitempty"`                                    // one exact Slot.components target, never inferred.
	ModelId         string                 `protobuf:"bytes,2,opt,name=model_id,json=modelId,proto3" json:"model_id,omitempty"`                         // ordinary Placement.models row for the adapter snapshot.
	SourceComponent string                 `protobuf:"bytes,3,opt,name=source_component,json=sourceComponent,proto3" json:"source_component,omitempty"` // canonical component holding A/B and optional alpha.
	Scale           string                 `protobuf:"bytes,4,opt,name=scale,proto3" json:"scale,omitempty"`                                            // finite canonical decimal string, retained in identity.
	unknownFields   protoimpl.UnknownFields
	sizeCache       protoimpl.SizeCache
}

func (x *ModelAdapter) Reset() {
	*x = ModelAdapter{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[148]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *ModelAdapter) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*ModelAdapter) ProtoMessage() {}

func (x *ModelAdapter) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[148]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use ModelAdapter.ProtoReflect.Descriptor instead.
func (*ModelAdapter) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{148}
}

func (x *ModelAdapter) GetComponent() string {
	if x != nil {
		return x.Component
	}
	return ""
}

func (x *ModelAdapter) GetModelId() string {
	if x != nil {
		return x.ModelId
	}
	return ""
}

func (x *ModelAdapter) GetSourceComponent() string {
	if x != nil {
		return x.SourceComponent
	}
	return ""
}

func (x *ModelAdapter) GetScale() string {
	if x != nil {
		return x.Scale
	}
	return ""
}

type Component struct {
	state         protoimpl.MessageState `protogen:"open.v1"`
	Component     string                 `protobuf:"bytes,1,opt,name=component,proto3" json:"component,omitempty"`
	ModelId       string                 `protobuf:"bytes,2,opt,name=model_id,json=modelId,proto3" json:"model_id,omitempty"` // must name one Placement.models id
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *Component) Reset() {
	*x = Component{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[149]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *Component) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*Component) ProtoMessage() {}

func (x *Component) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[149]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use Component.ProtoReflect.Descriptor instead.
func (*Component) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{149}
}

func (x *Component) GetComponent() string {
	if x != nil {
		return x.Component
	}
	return ""
}

func (x *Component) GetModelId() string {
	if x != nil {
		return x.ModelId
	}
	return ""
}

type Stamp struct {
	state         protoimpl.MessageState `protogen:"open.v1"`
	Component     string                 `protobuf:"bytes,1,opt,name=component,proto3" json:"component,omitempty"` // must name one Slot.components component
	Key           string                 `protobuf:"bytes,2,opt,name=key,proto3" json:"key,omitempty"`
	Values        []string               `protobuf:"bytes,3,rep,name=values,proto3" json:"values,omitempty"` // sorted unique, nonempty
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *Stamp) Reset() {
	*x = Stamp{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[150]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *Stamp) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*Stamp) ProtoMessage() {}

func (x *Stamp) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[150]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use Stamp.ProtoReflect.Descriptor instead.
func (*Stamp) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{150}
}

func (x *Stamp) GetComponent() string {
	if x != nil {
		return x.Component
	}
	return ""
}

func (x *Stamp) GetKey() string {
	if x != nil {
		return x.Key
	}
	return ""
}

func (x *Stamp) GetValues() []string {
	if x != nil {
		return x.Values
	}
	return nil
}

type JobDirective struct {
	state               protoimpl.MessageState `protogen:"open.v1"`
	JobDescriptorId     string                 `protobuf:"bytes,2,opt,name=job_descriptor_id,json=jobDescriptorId,proto3" json:"job_descriptor_id,omitempty"`
	ResourceCaps        *ResourceCaps          `protobuf:"bytes,3,opt,name=resource_caps,json=resourceCaps,proto3" json:"resource_caps,omitempty"`
	PublicationContract *PublicationContract   `protobuf:"bytes,4,opt,name=publication_contract,json=publicationContract,proto3" json:"publication_contract,omitempty"`
	DeviceCount         uint32                 `protobuf:"varint,6,opt,name=device_count,json=deviceCount,proto3" json:"device_count,omitempty"`
	// Exactly one CPU orchestration slot may coexist with the serial
	// ordinary job envelope. The selected parent must have frozen interface
	// dependencies, no Model/Weights capabilities, and no device grant.
	Orchestration bool `protobuf:"varint,7,opt,name=orchestration,proto3" json:"orchestration,omitempty"`
	// A complete desired state while an ordinary child runs includes its unchanged
	// CPU parent explicitly. Must have orchestration=true and no nested parent;
	// top-level orchestration=true cannot contain this field. This is an admitted
	// job set, never a program or a list of future calls.
	OrchestrationParent *JobDirective `protobuf:"bytes,8,opt,name=orchestration_parent,json=orchestrationParent,proto3" json:"orchestration_parent,omitempty"`
	InstallationId      string        `protobuf:"bytes,9,opt,name=installation_id,json=installationId,proto3" json:"installation_id,omitempty"`
	unknownFields       protoimpl.UnknownFields
	sizeCache           protoimpl.SizeCache
}

func (x *JobDirective) Reset() {
	*x = JobDirective{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[151]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *JobDirective) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*JobDirective) ProtoMessage() {}

func (x *JobDirective) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[151]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use JobDirective.ProtoReflect.Descriptor instead.
func (*JobDirective) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{151}
}

func (x *JobDirective) GetJobDescriptorId() string {
	if x != nil {
		return x.JobDescriptorId
	}
	return ""
}

func (x *JobDirective) GetResourceCaps() *ResourceCaps {
	if x != nil {
		return x.ResourceCaps
	}
	return nil
}

func (x *JobDirective) GetPublicationContract() *PublicationContract {
	if x != nil {
		return x.PublicationContract
	}
	return nil
}

func (x *JobDirective) GetDeviceCount() uint32 {
	if x != nil {
		return x.DeviceCount
	}
	return 0
}

func (x *JobDirective) GetOrchestration() bool {
	if x != nil {
		return x.Orchestration
	}
	return false
}

func (x *JobDirective) GetOrchestrationParent() *JobDirective {
	if x != nil {
		return x.OrchestrationParent
	}
	return nil
}

func (x *JobDirective) GetInstallationId() string {
	if x != nil {
		return x.InstallationId
	}
	return ""
}

// CONVERGENCE FACTS ON THE WIRE (§8; was Report). `applied_revision` is RETIRED as dishonest —
// it advanced on ACCEPTANCE, so a RecordOwner reading it learned only that its own message
// arrived. Acceptance and convergence are now two separate, checkable facts:
// converged_revision < accepted_desired_state_revision is the normal, visible state of a
// convergence in progress or a latched failure — never an error, always readable.
type ObservedWorkerState struct {
	state              protoimpl.MessageState `protogen:"open.v1"`
	RecordOwnerEpoch   uint64                 `protobuf:"varint,1,opt,name=record_owner_epoch,json=recordOwnerEpoch,proto3" json:"record_owner_epoch,omitempty"`
	ControlStreamEpoch uint64                 `protobuf:"varint,2,opt,name=control_stream_epoch,json=controlStreamEpoch,proto3" json:"control_stream_epoch,omitempty"`
	WorkerBootId       string                 `protobuf:"bytes,3,opt,name=worker_boot_id,json=workerBootId,proto3" json:"worker_boot_id,omitempty"`
	AppliedWireMinor   uint32                 `protobuf:"varint,8,opt,name=applied_wire_minor,json=appliedWireMinor,proto3" json:"applied_wire_minor,omitempty"` // stale echo is a visible skew fact, never a refusal
	HeldAttempts       []*HeldAttempt         `protobuf:"bytes,9,rep,name=held_attempts,json=heldAttempts,proto3" json:"held_attempts,omitempty"`                // every accepted attempt from admission to ack:
	// QUEUED, RUNNING, DEVICE_RELEASED, OUTCOME_PENDING_ACK
	// (proto-026: a queued attempt is no longer absent)
	Faults      []*Fault           `protobuf:"bytes,10,rep,name=faults,proto3" json:"faults,omitempty"`                              // MACHINE-scope faults; placement faults ride their status
	Activity    []*ActivityEvent   `protobuf:"bytes,11,rep,name=activity,proto3" json:"activity,omitempty"`                          // the non-attempt activity lane (durable)
	JobCapacity *JobCapacity       `protobuf:"bytes,13,opt,name=job_capacity,json=jobCapacity,proto3" json:"job_capacity,omitempty"` // job mode only
	Placements  []*PlacementStatus `protobuf:"bytes,15,rep,name=placements,proto3" json:"placements,omitempty"`                      // serving mode only; sorted by placement_id
	// The ONE worker-level admission gate (§6). The RecordOwner dispatches against the epoch it
	// last saw; the worker admits only if the echoed epoch is CURRENT, admission_state is OPEN,
	// and a slot is free. A stale epoch refuses DETERMINISTICALLY — same input, same verdict, no
	// race window. Only the epoch FENCES: state and slots say "not accepting" / "full", which is a
	// gate, not a fence against a stale actor.
	AdmissionEpoch uint64 `protobuf:"varint,16,opt,name=admission_epoch,json=admissionEpoch,proto3" json:"admission_epoch,omitempty"` // bumps whenever admission MEANING changes: window
	// resize, phase change, cutover, lane change. NOT on
	// executor respawn (proto-026, gpu-hot D10): a queued
	// attempt is bound to no executor, so a respawn
	// invalidates nothing it used to; a RUNNING attempt dies
	// with its own executor_epoch as before. record_owner_epoch,
	// control_stream_epoch and executor_epoch are unchanged.
	AdmissionState AdmissionState `protobuf:"varint,17,opt,name=admission_state,json=admissionState,proto3,enum=cozy.worker.v1.AdmissionState" json:"admission_state,omitempty"` // CLOSED is STRUCTURAL (pre-snapshot-barrier,
	// draining, mid-cutover); OPEN with zero slots is TRANSIENT
	// saturation — a RecordOwner backs off differently for each
	AvailableAttemptSlots        uint32 `protobuf:"varint,18,opt,name=available_attempt_slots,json=availableAttemptSlots,proto3" json:"available_attempt_slots,omitempty"`                        // offers this worker will still admit to its lane
	AcceptedDesiredStateRevision uint64 `protobuf:"varint,19,opt,name=accepted_desired_state_revision,json=acceptedDesiredStateRevision,proto3" json:"accepted_desired_state_revision,omitempty"` // durably ACCEPTED intent (advances on
	// acceptance — which is all it ever honestly meant)
	AcceptedPlacementSetDigest []byte `protobuf:"bytes,20,opt,name=accepted_placement_set_digest,json=acceptedPlacementSetDigest,proto3" json:"accepted_placement_set_digest,omitempty"` // class (a): the accepted set's digest
	ConvergedRevision          uint64 `protobuf:"varint,21,opt,name=converged_revision,json=convergedRevision,proto3" json:"converged_revision,omitempty"`                               // advances ONLY when observed satisfies accepted. It may
	// NEVER advance past a placement whose serving state is not
	// DISPATCHABLE for a desired spec: the field is a derived
	// fact, and a worker that advances it while observed state
	// disagrees is in breach, not merely optimistic.
	WorkerPhase WorkerPhase   `protobuf:"varint,22,opt,name=worker_phase,json=workerPhase,proto3,enum=cozy.worker.v1.WorkerPhase" json:"worker_phase,omitempty"` // machine lifecycle, out of the placement enum
	Lanes       []*DeviceLane `protobuf:"bytes,26,rep,name=lanes,proto3" json:"lanes,omitempty"`                                                                 // proto-024: every lane, sorted by lane_id; serving mode.
	// A 1-device worker reports exactly one lane. Any lane
	// change (a group forming or dissolving) bumps
	// admission_epoch; structural CLOSED stays a worker fact.
	HeldManifests []string `protobuf:"bytes,27,rep,name=held_manifests,json=heldManifests,proto3" json:"held_manifests,omitempty"` // proto-026 residency (residency-aware-routing.md §2):
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *ObservedWorkerState) Reset() {
	*x = ObservedWorkerState{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[152]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *ObservedWorkerState) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*ObservedWorkerState) ProtoMessage() {}

func (x *ObservedWorkerState) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[152]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use ObservedWorkerState.ProtoReflect.Descriptor instead.
func (*ObservedWorkerState) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{152}
}

func (x *ObservedWorkerState) GetRecordOwnerEpoch() uint64 {
	if x != nil {
		return x.RecordOwnerEpoch
	}
	return 0
}

func (x *ObservedWorkerState) GetControlStreamEpoch() uint64 {
	if x != nil {
		return x.ControlStreamEpoch
	}
	return 0
}

func (x *ObservedWorkerState) GetWorkerBootId() string {
	if x != nil {
		return x.WorkerBootId
	}
	return ""
}

func (x *ObservedWorkerState) GetAppliedWireMinor() uint32 {
	if x != nil {
		return x.AppliedWireMinor
	}
	return 0
}

func (x *ObservedWorkerState) GetHeldAttempts() []*HeldAttempt {
	if x != nil {
		return x.HeldAttempts
	}
	return nil
}

func (x *ObservedWorkerState) GetFaults() []*Fault {
	if x != nil {
		return x.Faults
	}
	return nil
}

func (x *ObservedWorkerState) GetActivity() []*ActivityEvent {
	if x != nil {
		return x.Activity
	}
	return nil
}

func (x *ObservedWorkerState) GetJobCapacity() *JobCapacity {
	if x != nil {
		return x.JobCapacity
	}
	return nil
}

func (x *ObservedWorkerState) GetPlacements() []*PlacementStatus {
	if x != nil {
		return x.Placements
	}
	return nil
}

func (x *ObservedWorkerState) GetAdmissionEpoch() uint64 {
	if x != nil {
		return x.AdmissionEpoch
	}
	return 0
}

func (x *ObservedWorkerState) GetAdmissionState() AdmissionState {
	if x != nil {
		return x.AdmissionState
	}
	return AdmissionState_ADMISSION_STATE_UNSPECIFIED
}

func (x *ObservedWorkerState) GetAvailableAttemptSlots() uint32 {
	if x != nil {
		return x.AvailableAttemptSlots
	}
	return 0
}

func (x *ObservedWorkerState) GetAcceptedDesiredStateRevision() uint64 {
	if x != nil {
		return x.AcceptedDesiredStateRevision
	}
	return 0
}

func (x *ObservedWorkerState) GetAcceptedPlacementSetDigest() []byte {
	if x != nil {
		return x.AcceptedPlacementSetDigest
	}
	return nil
}

func (x *ObservedWorkerState) GetConvergedRevision() uint64 {
	if x != nil {
		return x.ConvergedRevision
	}
	return 0
}

func (x *ObservedWorkerState) GetWorkerPhase() WorkerPhase {
	if x != nil {
		return x.WorkerPhase
	}
	return WorkerPhase_WORKER_PHASE_UNSPECIFIED
}

func (x *ObservedWorkerState) GetLanes() []*DeviceLane {
	if x != nil {
		return x.Lanes
	}
	return nil
}

func (x *ObservedWorkerState) GetHeldManifests() []string {
	if x != nil {
		return x.HeldManifests
	}
	return nil
}

// ONE serialized resource inside the worker (proto-024/cr-066): a set of envelope-local device
// ordinals, one attempt seat, one ledger, and the placements resident on those devices. Law 8
// (one active attempt per PHYSICAL device) is a lane fact. K == 1 is a device lane; K > 1 is a
// GROUP lane — one executor sealed to all K, the ledger accounting per device, the
// whole executor torn down as a unit (executor_epoch bumps once, every device's rows clear).
// Ordinals are positions in the worker's `--devices` envelope, sorted unique, disjoint across
// lanes.
//
// The lane holds a QUEUE of accepted attempts (proto-026, gpu-hot.md §4) and orders it itself:
// resident tenant first, then co-fitting, then the rest, within an overtake budget each attempt
// carries on HeldAttempt. The device SEAT is not a count here — it is the RUNNING held attempt
// whose lane_id names this lane.
type DeviceLane struct {
	state                 protoimpl.MessageState `protogen:"open.v1"`
	LaneId                string                 `protobuf:"bytes,1,opt,name=lane_id,json=laneId,proto3" json:"lane_id,omitempty"`                                                 // worker-minted, opaque, stable while the lane exists
	DeviceOrdinals        []uint32               `protobuf:"varint,2,rep,packed,name=device_ordinals,json=deviceOrdinals,proto3" json:"device_ordinals,omitempty"`                 // K >= 1
	AvailableAttemptSlots uint32                 `protobuf:"varint,3,opt,name=available_attempt_slots,json=availableAttemptSlots,proto3" json:"available_attempt_slots,omitempty"` // offers this lane will still ADMIT TO ITS QUEUE now:
	// the depth the lane can account for in host bytes
	// (declared input + output bounds against the host
	// budget, gpu-hot.md §4) minus attempts held. Never a
	// constant; never the seat.
	PlacementIds         []string `protobuf:"bytes,4,rep,name=placement_ids,json=placementIds,proto3" json:"placement_ids,omitempty"`                           // sorted; placements assigned to this lane
	ResidentPlacementIds []string `protobuf:"bytes,5,rep,name=resident_placement_ids,json=residentPlacementIds,proto3" json:"resident_placement_ids,omitempty"` // proto-026 residency: placements whose executor
	unknownFields        protoimpl.UnknownFields
	sizeCache            protoimpl.SizeCache
}

func (x *DeviceLane) Reset() {
	*x = DeviceLane{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[153]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *DeviceLane) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*DeviceLane) ProtoMessage() {}

func (x *DeviceLane) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[153]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use DeviceLane.ProtoReflect.Descriptor instead.
func (*DeviceLane) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{153}
}

func (x *DeviceLane) GetLaneId() string {
	if x != nil {
		return x.LaneId
	}
	return ""
}

func (x *DeviceLane) GetDeviceOrdinals() []uint32 {
	if x != nil {
		return x.DeviceOrdinals
	}
	return nil
}

func (x *DeviceLane) GetAvailableAttemptSlots() uint32 {
	if x != nil {
		return x.AvailableAttemptSlots
	}
	return 0
}

func (x *DeviceLane) GetPlacementIds() []string {
	if x != nil {
		return x.PlacementIds
	}
	return nil
}

func (x *DeviceLane) GetResidentPlacementIds() []string {
	if x != nil {
		return x.ResidentPlacementIds
	}
	return nil
}

// PER-PLACEMENT TRUTH, on TWO AXES (§8, #473/#482). The frozen wire's single IntakeState cannot
// express "staged on disk but offline" — the exact state an outgoing spec holds under
// fallback-retention — and a single phase cannot represent staged x draining independently.
//
// The axes make #474's fallback-retention expressible: outgoing sits at (STAGED, OFFLINE),
// restorable without a refetch, while incoming sits at (STAGED, ACTIVATING). An activation
// failure restores outgoing to (STAGED, DISPATCHABLE) and LATCHES the failure for that desired
// revision — the worker does not evict-and-retry forever, and converged_revision stays behind
// with the fault typed on the placement.
//
// REPLACEMENT REQUIRES A LIVE FALLBACK (#485c): a placement whose rollback pin is gone — GC'd
// under a bytes/age cap, or never materialized — CANNOT accept a new spec until it
// re-materializes one. Fallback capability is a PRECONDITION of replacement, not a best-effort
// nicety; a cap that evicts the pin thereby PAUSES replacement, recorded as
// FAULT_KIND_FALLBACK_PIN_MISSING, and forcing past it is an explicit administrative act.
type PlacementStatus struct {
	state                        protoimpl.MessageState    `protogen:"open.v1"`
	PlacementId                  string                    `protobuf:"bytes,1,opt,name=placement_id,json=placementId,proto3" json:"placement_id,omitempty"`
	ExecutorEpoch                uint64                    `protobuf:"varint,4,opt,name=executor_epoch,json=executorEpoch,proto3" json:"executor_epoch,omitempty"` // THIS placement's executor fence; bumps on its respawn
	DispatchableBindingDigests   [][]byte                  `protobuf:"bytes,6,rep,name=dispatchable_binding_digests,json=dispatchableBindingDigests,proto3" json:"dispatchable_binding_digests,omitempty"`
	MaterializableBindingDigests [][]byte                  `protobuf:"bytes,7,rep,name=materializable_binding_digests,json=materializableBindingDigests,proto3" json:"materializable_binding_digests,omitempty"` // DISJOINT from dispatchable
	Faults                       []*Fault                  `protobuf:"bytes,8,rep,name=faults,proto3" json:"faults,omitempty"`                                                                                   // placement-scoped
	Accelerator                  *AcceleratorQualification `protobuf:"bytes,9,opt,name=accelerator,proto3" json:"accelerator,omitempty"`                                                                         // absent = present-but-unqualified
	PlacementSetDigest           []byte                    `protobuf:"bytes,11,opt,name=placement_set_digest,json=placementSetDigest,proto3" json:"placement_set_digest,omitempty"`                              // class (a): parent desired-set identity. Together with
	// placement_id, identifies what this placement holds.
	Materialization MaterializationState `protobuf:"varint,12,opt,name=materialization,proto3,enum=cozy.worker.v1.MaterializationState" json:"materialization,omitempty"` // STAGED asserts the
	// exact overlay installed and imported successfully; it is
	// unspeakable otherwise
	Serving                            ServingState `protobuf:"varint,13,opt,name=serving,proto3,enum=cozy.worker.v1.ServingState" json:"serving,omitempty"`
	RetainedFallbackPlacementSetDigest []byte       `protobuf:"bytes,14,opt,name=retained_fallback_placement_set_digest,json=retainedFallbackPlacementSetDigest,proto3" json:"retained_fallback_placement_set_digest,omitempty"` // class (a): predecessor set kept for restore
	// (#474/#485c). Empty means replacement is PAUSED.
	Acquisition  *PlacementAcquisitionObservation `protobuf:"bytes,15,opt,name=acquisition,proto3" json:"acquisition,omitempty"`                         // OBSERVATION ONLY for this
	DeviceLaneId string                           `protobuf:"bytes,19,opt,name=device_lane_id,json=deviceLaneId,proto3" json:"device_lane_id,omitempty"` // proto-024: the ObservedWorkerState.lanes[] entry this
	// placement's executor is sealed to; empty while
	// OFFLINE/ABSENT or before the worker assigned a lane
	LoadedBindingDigests [][]byte `protobuf:"bytes,20,rep,name=loaded_binding_digests,json=loadedBindingDigests,proto3" json:"loaded_binding_digests,omitempty"` // sorted unique, a subset of
	// dispatchable_binding_digests: constructions built in the
	// live executor (resident or parked)
	InstallationId string `protobuf:"bytes,21,opt,name=installation_id,json=installationId,proto3" json:"installation_id,omitempty"`
	unknownFields  protoimpl.UnknownFields
	sizeCache      protoimpl.SizeCache
}

func (x *PlacementStatus) Reset() {
	*x = PlacementStatus{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[154]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *PlacementStatus) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*PlacementStatus) ProtoMessage() {}

func (x *PlacementStatus) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[154]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use PlacementStatus.ProtoReflect.Descriptor instead.
func (*PlacementStatus) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{154}
}

func (x *PlacementStatus) GetPlacementId() string {
	if x != nil {
		return x.PlacementId
	}
	return ""
}

func (x *PlacementStatus) GetExecutorEpoch() uint64 {
	if x != nil {
		return x.ExecutorEpoch
	}
	return 0
}

func (x *PlacementStatus) GetDispatchableBindingDigests() [][]byte {
	if x != nil {
		return x.DispatchableBindingDigests
	}
	return nil
}

func (x *PlacementStatus) GetMaterializableBindingDigests() [][]byte {
	if x != nil {
		return x.MaterializableBindingDigests
	}
	return nil
}

func (x *PlacementStatus) GetFaults() []*Fault {
	if x != nil {
		return x.Faults
	}
	return nil
}

func (x *PlacementStatus) GetAccelerator() *AcceleratorQualification {
	if x != nil {
		return x.Accelerator
	}
	return nil
}

func (x *PlacementStatus) GetPlacementSetDigest() []byte {
	if x != nil {
		return x.PlacementSetDigest
	}
	return nil
}

func (x *PlacementStatus) GetMaterialization() MaterializationState {
	if x != nil {
		return x.Materialization
	}
	return MaterializationState_MATERIALIZATION_STATE_UNSPECIFIED
}

func (x *PlacementStatus) GetServing() ServingState {
	if x != nil {
		return x.Serving
	}
	return ServingState_SERVING_STATE_UNSPECIFIED
}

func (x *PlacementStatus) GetRetainedFallbackPlacementSetDigest() []byte {
	if x != nil {
		return x.RetainedFallbackPlacementSetDigest
	}
	return nil
}

func (x *PlacementStatus) GetAcquisition() *PlacementAcquisitionObservation {
	if x != nil {
		return x.Acquisition
	}
	return nil
}

func (x *PlacementStatus) GetDeviceLaneId() string {
	if x != nil {
		return x.DeviceLaneId
	}
	return ""
}

func (x *PlacementStatus) GetLoadedBindingDigests() [][]byte {
	if x != nil {
		return x.LoadedBindingDigests
	}
	return nil
}

func (x *PlacementStatus) GetInstallationId() string {
	if x != nil {
		return x.InstallationId
	}
	return ""
}

// Cozy-visible proof of concurrent package/model acquisition and cache reuse. Both legs use
// one process-relative monotonic clock and therefore one origin; they reset with the worker
// process and MUST NOT be compared across worker_boot_id. A leg is absent before it starts. Once
// present its start is non-zero, counters are cumulative and never decrease, and end is zero
// while in progress or >= start when terminal. The two terminal intervals prove overlap exactly.
//
// Scope is deliberately disjoint. `package` counts the desired package overlay bytes needed
// above the base worker image (including its project/custom wheels), excluding model bytes.
// `model` counts the object bytes reached from Placement.models[].manifest. At terminal,
// downloaded_bytes + reused_bytes is the exact reached
// content length for that leg. Downloaded bytes were newly fetched into verified local holdings;
// reused bytes were already verified and held locally. Partial observations carry progress so
// far; terminal counters persist as described on PlacementStatus.
type PlacementAcquisitionObservation struct {
	state         protoimpl.MessageState     `protogen:"open.v1"`
	Package       *AcquisitionLegObservation `protobuf:"bytes,1,opt,name=package,proto3" json:"package,omitempty"`
	Model         *AcquisitionLegObservation `protobuf:"bytes,2,opt,name=model,proto3" json:"model,omitempty"`
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *PlacementAcquisitionObservation) Reset() {
	*x = PlacementAcquisitionObservation{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[155]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *PlacementAcquisitionObservation) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*PlacementAcquisitionObservation) ProtoMessage() {}

func (x *PlacementAcquisitionObservation) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[155]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use PlacementAcquisitionObservation.ProtoReflect.Descriptor instead.
func (*PlacementAcquisitionObservation) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{155}
}

func (x *PlacementAcquisitionObservation) GetPackage() *AcquisitionLegObservation {
	if x != nil {
		return x.Package
	}
	return nil
}

func (x *PlacementAcquisitionObservation) GetModel() *AcquisitionLegObservation {
	if x != nil {
		return x.Model
	}
	return nil
}

type AcquisitionLegObservation struct {
	state              protoimpl.MessageState `protogen:"open.v1"`
	StartedMonotonicNs uint64                 `protobuf:"varint,1,opt,name=started_monotonic_ns,json=startedMonotonicNs,proto3" json:"started_monotonic_ns,omitempty"` // non-zero; one process-relative clock for both legs
	EndedMonotonicNs   uint64                 `protobuf:"varint,2,opt,name=ended_monotonic_ns,json=endedMonotonicNs,proto3" json:"ended_monotonic_ns,omitempty"`       // 0 while in progress; terminal iff >= start
	DownloadedBytes    uint64                 `protobuf:"varint,3,opt,name=downloaded_bytes,json=downloadedBytes,proto3" json:"downloaded_bytes,omitempty"`            // cumulative unique content bytes newly fetched this leg
	ReusedBytes        uint64                 `protobuf:"varint,4,opt,name=reused_bytes,json=reusedBytes,proto3" json:"reused_bytes,omitempty"`                        // cumulative verified bytes already held before this leg
	unknownFields      protoimpl.UnknownFields
	sizeCache          protoimpl.SizeCache
}

func (x *AcquisitionLegObservation) Reset() {
	*x = AcquisitionLegObservation{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[156]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *AcquisitionLegObservation) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*AcquisitionLegObservation) ProtoMessage() {}

func (x *AcquisitionLegObservation) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[156]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use AcquisitionLegObservation.ProtoReflect.Descriptor instead.
func (*AcquisitionLegObservation) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{156}
}

func (x *AcquisitionLegObservation) GetStartedMonotonicNs() uint64 {
	if x != nil {
		return x.StartedMonotonicNs
	}
	return 0
}

func (x *AcquisitionLegObservation) GetEndedMonotonicNs() uint64 {
	if x != nil {
		return x.EndedMonotonicNs
	}
	return 0
}

func (x *AcquisitionLegObservation) GetDownloadedBytes() uint64 {
	if x != nil {
		return x.DownloadedBytes
	}
	return 0
}

func (x *AcquisitionLegObservation) GetReusedBytes() uint64 {
	if x != nil {
		return x.ReusedBytes
	}
	return 0
}

// Executor-probed capability facts (#430: MEASURED, never inferred from chip generation or OS
// version). Vocabulary IS the AcceleratorFacts record's (#422) — one facts schema, no second
// wire vocabulary. The torch-free worker NEVER fabricates these (it cannot measure them); until
// an executor qualifies the device, the accelerator is PRESENT-BUT-UNQUALIFIED and this message
// is absent from PlacementStatus.
type AcceleratorQualification struct {
	state        protoimpl.MessageState `protogen:"open.v1"`
	Qualified    bool                   `protobuf:"varint,1,opt,name=qualified,proto3" json:"qualified,omitempty"`
	Capabilities []string               `protobuf:"bytes,2,rep,name=capabilities,proto3" json:"capabilities,omitempty"` // sorted facts vocabulary (bf16, fp8_compute,
	// pinned_h2d_streams, non_blocking_safe,
	// attention_upcast_required); presence = measured true
	RecommendedMaxBytes uint64   `protobuf:"varint,3,opt,name=recommended_max_bytes,json=recommendedMaxBytes,proto3" json:"recommended_max_bytes,omitempty"` // the working-set envelope the ledger prices against
	PinnedCapacityBytes uint64   `protobuf:"varint,4,opt,name=pinned_capacity_bytes,json=pinnedCapacityBytes,proto3" json:"pinned_capacity_bytes,omitempty"`
	H2DGbpsMeasured     uint32   `protobuf:"varint,5,opt,name=h2d_gbps_measured,json=h2dGbpsMeasured,proto3" json:"h2d_gbps_measured,omitempty"`
	D2HGbpsMeasured     uint32   `protobuf:"varint,6,opt,name=d2h_gbps_measured,json=d2hGbpsMeasured,proto3" json:"d2h_gbps_measured,omitempty"`
	TorchVersion        string   `protobuf:"bytes,7,opt,name=torch_version,json=torchVersion,proto3" json:"torch_version,omitempty"` // TELEMETRY ONLY
	Unreadable          []string `protobuf:"bytes,8,rep,name=unreadable,proto3" json:"unreadable,omitempty"`                         // tri-state honesty
	unknownFields       protoimpl.UnknownFields
	sizeCache           protoimpl.SizeCache
}

func (x *AcceleratorQualification) Reset() {
	*x = AcceleratorQualification{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[157]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *AcceleratorQualification) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*AcceleratorQualification) ProtoMessage() {}

func (x *AcceleratorQualification) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[157]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use AcceleratorQualification.ProtoReflect.Descriptor instead.
func (*AcceleratorQualification) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{157}
}

func (x *AcceleratorQualification) GetQualified() bool {
	if x != nil {
		return x.Qualified
	}
	return false
}

func (x *AcceleratorQualification) GetCapabilities() []string {
	if x != nil {
		return x.Capabilities
	}
	return nil
}

func (x *AcceleratorQualification) GetRecommendedMaxBytes() uint64 {
	if x != nil {
		return x.RecommendedMaxBytes
	}
	return 0
}

func (x *AcceleratorQualification) GetPinnedCapacityBytes() uint64 {
	if x != nil {
		return x.PinnedCapacityBytes
	}
	return 0
}

func (x *AcceleratorQualification) GetH2DGbpsMeasured() uint32 {
	if x != nil {
		return x.H2DGbpsMeasured
	}
	return 0
}

func (x *AcceleratorQualification) GetD2HGbpsMeasured() uint32 {
	if x != nil {
		return x.D2HGbpsMeasured
	}
	return 0
}

func (x *AcceleratorQualification) GetTorchVersion() string {
	if x != nil {
		return x.TorchVersion
	}
	return ""
}

func (x *AcceleratorQualification) GetUnreadable() []string {
	if x != nil {
		return x.Unreadable
	}
	return nil
}

type ActivityEvent struct {
	state         protoimpl.MessageState `protogen:"open.v1"`
	Seq           uint64                 `protobuf:"varint,1,opt,name=seq,proto3" json:"seq,omitempty"`
	Kind          string                 `protobuf:"bytes,2,opt,name=kind,proto3" json:"kind,omitempty"`
	Step          string                 `protobuf:"bytes,3,opt,name=step,proto3" json:"step,omitempty"` // bounded <= 256 bytes
	AtUnixMs      uint64                 `protobuf:"varint,4,opt,name=at_unix_ms,json=atUnixMs,proto3" json:"at_unix_ms,omitempty"`
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *ActivityEvent) Reset() {
	*x = ActivityEvent{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[158]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *ActivityEvent) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*ActivityEvent) ProtoMessage() {}

func (x *ActivityEvent) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[158]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use ActivityEvent.ProtoReflect.Descriptor instead.
func (*ActivityEvent) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{158}
}

func (x *ActivityEvent) GetSeq() uint64 {
	if x != nil {
		return x.Seq
	}
	return 0
}

func (x *ActivityEvent) GetKind() string {
	if x != nil {
		return x.Kind
	}
	return ""
}

func (x *ActivityEvent) GetStep() string {
	if x != nil {
		return x.Step
	}
	return ""
}

func (x *ActivityEvent) GetAtUnixMs() uint64 {
	if x != nil {
		return x.AtUnixMs
	}
	return 0
}

// Was StartAttempt (#481): receiving an OFFER begins VALIDATION, not execution, so refusing one
// is an ordinary outcome rather than an exception. EVERY offer receives either AttemptAccepted or
// a JOURNALED AttemptOutcome(REFUSED) — never silence.
type AttemptOffer struct {
	state              protoimpl.MessageState `protogen:"open.v1"`
	RecordOwnerEpoch   uint64                 `protobuf:"varint,1,opt,name=record_owner_epoch,json=recordOwnerEpoch,proto3" json:"record_owner_epoch,omitempty"`
	ControlStreamEpoch uint64                 `protobuf:"varint,2,opt,name=control_stream_epoch,json=controlStreamEpoch,proto3" json:"control_stream_epoch,omitempty"`
	WorkerBootId       string                 `protobuf:"bytes,3,opt,name=worker_boot_id,json=workerBootId,proto3" json:"worker_boot_id,omitempty"`
	RequestId          string                 `protobuf:"bytes,5,opt,name=request_id,json=requestId,proto3" json:"request_id,omitempty"`                 // fence (a)
	AttemptOrdinal     uint64                 `protobuf:"varint,6,opt,name=attempt_ordinal,json=attemptOrdinal,proto3" json:"attempt_ordinal,omitempty"` // fence (b); RecordOwner-bumped only. EVERY offer consumes
	// its ordinal — what a pre-execution refusal does not
	// consume is BUDGET (#480b; 02 §6's "only a proven decline
	// is non-consuming" is retired with the decline).
	InvocationSpecDigest []byte `protobuf:"bytes,7,opt,name=invocation_spec_digest,json=invocationSpecDigest,proto3" json:"invocation_spec_digest,omitempty"` // fence (c), class (a): sha256 over EXACTLY the bytes in 9;
	// the receiver RECOMPUTES and refuses on mismatch
	Grant                        *DeliveryGrant `protobuf:"bytes,8,opt,name=grant,proto3" json:"grant,omitempty"`                                                                                       // expiring, refreshable; OUTSIDE the digest
	InvocationSpecCanonicalBytes []byte         `protobuf:"bytes,9,opt,name=invocation_spec_canonical_bytes,json=invocationSpecCanonicalBytes,proto3" json:"invocation_spec_canonical_bytes,omitempty"` // the InvocationSpec DOCUMENT, canonical JSON
	// bytes; immutable for the attempt; parsed under
	// unknown-field REFUSAL
	PlacementId string `protobuf:"bytes,10,opt,name=placement_id,json=placementId,proto3" json:"placement_id,omitempty"` // the target placement (serving mode; REQUIRED there, empty
	// in job mode); unknown id refuses
	// FAULT_KIND_UNKNOWN_PLACEMENT as a JOURNALED REFUSED
	// outcome
	AdmissionEpoch uint64 `protobuf:"varint,12,opt,name=admission_epoch,json=admissionEpoch,proto3" json:"admission_epoch,omitempty"` // the epoch the RecordOwner OBSERVED when it
	unknownFields  protoimpl.UnknownFields
	sizeCache      protoimpl.SizeCache
}

func (x *AttemptOffer) Reset() {
	*x = AttemptOffer{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[159]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *AttemptOffer) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*AttemptOffer) ProtoMessage() {}

func (x *AttemptOffer) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[159]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use AttemptOffer.ProtoReflect.Descriptor instead.
func (*AttemptOffer) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{159}
}

func (x *AttemptOffer) GetRecordOwnerEpoch() uint64 {
	if x != nil {
		return x.RecordOwnerEpoch
	}
	return 0
}

func (x *AttemptOffer) GetControlStreamEpoch() uint64 {
	if x != nil {
		return x.ControlStreamEpoch
	}
	return 0
}

func (x *AttemptOffer) GetWorkerBootId() string {
	if x != nil {
		return x.WorkerBootId
	}
	return ""
}

func (x *AttemptOffer) GetRequestId() string {
	if x != nil {
		return x.RequestId
	}
	return ""
}

func (x *AttemptOffer) GetAttemptOrdinal() uint64 {
	if x != nil {
		return x.AttemptOrdinal
	}
	return 0
}

func (x *AttemptOffer) GetInvocationSpecDigest() []byte {
	if x != nil {
		return x.InvocationSpecDigest
	}
	return nil
}

func (x *AttemptOffer) GetGrant() *DeliveryGrant {
	if x != nil {
		return x.Grant
	}
	return nil
}

func (x *AttemptOffer) GetInvocationSpecCanonicalBytes() []byte {
	if x != nil {
		return x.InvocationSpecCanonicalBytes
	}
	return nil
}

func (x *AttemptOffer) GetPlacementId() string {
	if x != nil {
		return x.PlacementId
	}
	return ""
}

func (x *AttemptOffer) GetAdmissionEpoch() uint64 {
	if x != nil {
		return x.AdmissionEpoch
	}
	return 0
}

// ACCEPTANCE IS QUEUE ADMISSION (proto-026, gpu-hot D2): "journaled, hydrated, queued on
// lane_id". It binds neither a plan nor an executor epoch — both bind at DEVICE ENTRY and are
// OBSERVED on HeldAttempt (plan_digest from RUNNING on; executor_epoch, 0 while QUEUED). Plan
// facts ride the outcome body (AttemptMetrics).
type AttemptAccepted struct {
	state                protoimpl.MessageState `protogen:"open.v1"`
	RecordOwnerEpoch     uint64                 `protobuf:"varint,1,opt,name=record_owner_epoch,json=recordOwnerEpoch,proto3" json:"record_owner_epoch,omitempty"`
	ControlStreamEpoch   uint64                 `protobuf:"varint,2,opt,name=control_stream_epoch,json=controlStreamEpoch,proto3" json:"control_stream_epoch,omitempty"`
	WorkerBootId         string                 `protobuf:"bytes,3,opt,name=worker_boot_id,json=workerBootId,proto3" json:"worker_boot_id,omitempty"`
	RequestId            string                 `protobuf:"bytes,5,opt,name=request_id,json=requestId,proto3" json:"request_id,omitempty"`
	AttemptOrdinal       uint64                 `protobuf:"varint,6,opt,name=attempt_ordinal,json=attemptOrdinal,proto3" json:"attempt_ordinal,omitempty"`
	InvocationSpecDigest []byte                 `protobuf:"bytes,7,opt,name=invocation_spec_digest,json=invocationSpecDigest,proto3" json:"invocation_spec_digest,omitempty"` // echo
	PlacementId          string                 `protobuf:"bytes,11,opt,name=placement_id,json=placementId,proto3" json:"placement_id,omitempty"`                             // echo
	LaneId               string                 `protobuf:"bytes,13,opt,name=lane_id,json=laneId,proto3" json:"lane_id,omitempty"`                                            // proto-026: the DeviceLane the attempt was admitted to;
	unknownFields        protoimpl.UnknownFields
	sizeCache            protoimpl.SizeCache
}

func (x *AttemptAccepted) Reset() {
	*x = AttemptAccepted{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[160]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *AttemptAccepted) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*AttemptAccepted) ProtoMessage() {}

func (x *AttemptAccepted) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[160]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use AttemptAccepted.ProtoReflect.Descriptor instead.
func (*AttemptAccepted) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{160}
}

func (x *AttemptAccepted) GetRecordOwnerEpoch() uint64 {
	if x != nil {
		return x.RecordOwnerEpoch
	}
	return 0
}

func (x *AttemptAccepted) GetControlStreamEpoch() uint64 {
	if x != nil {
		return x.ControlStreamEpoch
	}
	return 0
}

func (x *AttemptAccepted) GetWorkerBootId() string {
	if x != nil {
		return x.WorkerBootId
	}
	return ""
}

func (x *AttemptAccepted) GetRequestId() string {
	if x != nil {
		return x.RequestId
	}
	return ""
}

func (x *AttemptAccepted) GetAttemptOrdinal() uint64 {
	if x != nil {
		return x.AttemptOrdinal
	}
	return 0
}

func (x *AttemptAccepted) GetInvocationSpecDigest() []byte {
	if x != nil {
		return x.InvocationSpecDigest
	}
	return nil
}

func (x *AttemptAccepted) GetPlacementId() string {
	if x != nil {
		return x.PlacementId
	}
	return ""
}

func (x *AttemptAccepted) GetLaneId() string {
	if x != nil {
		return x.LaneId
	}
	return ""
}

// No placement_id here: the fence triple resolves against the journaled acceptance, which
// records the placement.
type CancelAttempt struct {
	state                protoimpl.MessageState `protogen:"open.v1"`
	RecordOwnerEpoch     uint64                 `protobuf:"varint,1,opt,name=record_owner_epoch,json=recordOwnerEpoch,proto3" json:"record_owner_epoch,omitempty"`
	ControlStreamEpoch   uint64                 `protobuf:"varint,2,opt,name=control_stream_epoch,json=controlStreamEpoch,proto3" json:"control_stream_epoch,omitempty"`
	WorkerBootId         string                 `protobuf:"bytes,3,opt,name=worker_boot_id,json=workerBootId,proto3" json:"worker_boot_id,omitempty"`
	RequestId            string                 `protobuf:"bytes,5,opt,name=request_id,json=requestId,proto3" json:"request_id,omitempty"`
	AttemptOrdinal       uint64                 `protobuf:"varint,6,opt,name=attempt_ordinal,json=attemptOrdinal,proto3" json:"attempt_ordinal,omitempty"`
	Reason               CancelReason           `protobuf:"varint,7,opt,name=reason,proto3,enum=cozy.worker.v1.CancelReason" json:"reason,omitempty"`
	GraceMs              uint64                 `protobuf:"varint,8,opt,name=grace_ms,json=graceMs,proto3" json:"grace_ms,omitempty"`                                         // operator override; 0 = worker default. Never author-set.
	InvocationSpecDigest []byte                 `protobuf:"bytes,9,opt,name=invocation_spec_digest,json=invocationSpecDigest,proto3" json:"invocation_spec_digest,omitempty"` // full-triple fence; stale/mismatched cancel is DROPPED
	unknownFields        protoimpl.UnknownFields
	sizeCache            protoimpl.SizeCache
}

func (x *CancelAttempt) Reset() {
	*x = CancelAttempt{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[161]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *CancelAttempt) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*CancelAttempt) ProtoMessage() {}

func (x *CancelAttempt) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[161]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use CancelAttempt.ProtoReflect.Descriptor instead.
func (*CancelAttempt) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{161}
}

func (x *CancelAttempt) GetRecordOwnerEpoch() uint64 {
	if x != nil {
		return x.RecordOwnerEpoch
	}
	return 0
}

func (x *CancelAttempt) GetControlStreamEpoch() uint64 {
	if x != nil {
		return x.ControlStreamEpoch
	}
	return 0
}

func (x *CancelAttempt) GetWorkerBootId() string {
	if x != nil {
		return x.WorkerBootId
	}
	return ""
}

func (x *CancelAttempt) GetRequestId() string {
	if x != nil {
		return x.RequestId
	}
	return ""
}

func (x *CancelAttempt) GetAttemptOrdinal() uint64 {
	if x != nil {
		return x.AttemptOrdinal
	}
	return 0
}

func (x *CancelAttempt) GetReason() CancelReason {
	if x != nil {
		return x.Reason
	}
	return CancelReason_CANCEL_REASON_UNSPECIFIED
}

func (x *CancelAttempt) GetGraceMs() uint64 {
	if x != nil {
		return x.GraceMs
	}
	return 0
}

func (x *CancelAttempt) GetInvocationSpecDigest() []byte {
	if x != nil {
		return x.InvocationSpecDigest
	}
	return nil
}

// The envelope around a journaled AttemptOutcomeBody document — the ATTEMPT OUTCOME (was
// AttemptTerminal; "terminal" reads as a device and the thing is an outcome, #481). One
// immutable outcome per attempt; the customer-visible settlement is the RecordOwner's
// RequestClosure and never crosses this wire. Body travels as canonical bytes; routing copies
// must agree with it. journal-before-send applies unchanged, INCLUDING to refusals: the outcome
// is persisted before it is sent and replays until AttemptOutcomeAck.
type AttemptOutcome struct {
	state                 protoimpl.MessageState `protogen:"open.v1"`
	RecordOwnerEpoch      uint64                 `protobuf:"varint,1,opt,name=record_owner_epoch,json=recordOwnerEpoch,proto3" json:"record_owner_epoch,omitempty"`
	ControlStreamEpoch    uint64                 `protobuf:"varint,2,opt,name=control_stream_epoch,json=controlStreamEpoch,proto3" json:"control_stream_epoch,omitempty"`
	WorkerBootId          string                 `protobuf:"bytes,3,opt,name=worker_boot_id,json=workerBootId,proto3" json:"worker_boot_id,omitempty"`
	RequestId             string                 `protobuf:"bytes,5,opt,name=request_id,json=requestId,proto3" json:"request_id,omitempty"`                                        // routing copy; must agree with the document
	AttemptOrdinal        uint64                 `protobuf:"varint,6,opt,name=attempt_ordinal,json=attemptOrdinal,proto3" json:"attempt_ordinal,omitempty"`                        // routing copy
	InvocationSpecDigest  []byte                 `protobuf:"bytes,7,opt,name=invocation_spec_digest,json=invocationSpecDigest,proto3" json:"invocation_spec_digest,omitempty"`     // routing copy
	OutcomeId             string                 `protobuf:"bytes,8,opt,name=outcome_id,json=outcomeId,proto3" json:"outcome_id,omitempty"`                                        // WORKER-minted observation id; stable across replays
	OutcomeDigest         []byte                 `protobuf:"bytes,9,opt,name=outcome_digest,json=outcomeDigest,proto3" json:"outcome_digest,omitempty"`                            // class (a): sha256 over EXACTLY the bytes in 10
	OutcomeCanonicalBytes []byte                 `protobuf:"bytes,10,opt,name=outcome_canonical_bytes,json=outcomeCanonicalBytes,proto3" json:"outcome_canonical_bytes,omitempty"` // the AttemptOutcomeBody DOCUMENT, canonical JSON bytes
	PlacementId           string                 `protobuf:"bytes,11,opt,name=placement_id,json=placementId,proto3" json:"placement_id,omitempty"`                                 // routing
	unknownFields         protoimpl.UnknownFields
	sizeCache             protoimpl.SizeCache
}

func (x *AttemptOutcome) Reset() {
	*x = AttemptOutcome{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[162]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *AttemptOutcome) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*AttemptOutcome) ProtoMessage() {}

func (x *AttemptOutcome) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[162]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use AttemptOutcome.ProtoReflect.Descriptor instead.
func (*AttemptOutcome) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{162}
}

func (x *AttemptOutcome) GetRecordOwnerEpoch() uint64 {
	if x != nil {
		return x.RecordOwnerEpoch
	}
	return 0
}

func (x *AttemptOutcome) GetControlStreamEpoch() uint64 {
	if x != nil {
		return x.ControlStreamEpoch
	}
	return 0
}

func (x *AttemptOutcome) GetWorkerBootId() string {
	if x != nil {
		return x.WorkerBootId
	}
	return ""
}

func (x *AttemptOutcome) GetRequestId() string {
	if x != nil {
		return x.RequestId
	}
	return ""
}

func (x *AttemptOutcome) GetAttemptOrdinal() uint64 {
	if x != nil {
		return x.AttemptOrdinal
	}
	return 0
}

func (x *AttemptOutcome) GetInvocationSpecDigest() []byte {
	if x != nil {
		return x.InvocationSpecDigest
	}
	return nil
}

func (x *AttemptOutcome) GetOutcomeId() string {
	if x != nil {
		return x.OutcomeId
	}
	return ""
}

func (x *AttemptOutcome) GetOutcomeDigest() []byte {
	if x != nil {
		return x.OutcomeDigest
	}
	return nil
}

func (x *AttemptOutcome) GetOutcomeCanonicalBytes() []byte {
	if x != nil {
		return x.OutcomeCanonicalBytes
	}
	return nil
}

func (x *AttemptOutcome) GetPlacementId() string {
	if x != nil {
		return x.PlacementId
	}
	return ""
}

// DOCUMENT SHAPE (not a wire message): canonical form
// `cozy.worker.v1.AttemptOutcomeBody/1` (#480c/#481/th-049).
type AttemptOutcomeBody struct {
	state                protoimpl.MessageState `protogen:"open.v1"`
	RequestId            string                 `protobuf:"bytes,1,opt,name=request_id,json=requestId,proto3" json:"request_id,omitempty"`
	AttemptOrdinal       uint64                 `protobuf:"varint,2,opt,name=attempt_ordinal,json=attemptOrdinal,proto3" json:"attempt_ordinal,omitempty"`
	InvocationSpecDigest string                 `protobuf:"bytes,3,opt,name=invocation_spec_digest,json=invocationSpecDigest,proto3" json:"invocation_spec_digest,omitempty"` // sha256:<hex> spelling inside the document
	Status               OutcomeStatus          `protobuf:"varint,4,opt,name=status,proto3,enum=cozy.worker.v1.OutcomeStatus" json:"status,omitempty"`                        // closed NEUTRAL vocabulary; retryability is a RecordOwner
	// PROJECTION over (status, cause, origin), unrepresentable
	// as a worker wire fact
	OutputManifest   *OutputManifest  `protobuf:"bytes,5,opt,name=output_manifest,json=outputManifest,proto3" json:"output_manifest,omitempty"`
	Metrics          *AttemptMetrics  `protobuf:"bytes,6,opt,name=metrics,proto3" json:"metrics,omitempty"`
	TriageBundle     *TriageBundleRef `protobuf:"bytes,7,opt,name=triage_bundle,json=triageBundle,proto3" json:"triage_bundle,omitempty"`               // observation only
	SafeMessage      string           `protobuf:"bytes,8,opt,name=safe_message,json=safeMessage,proto3" json:"safe_message,omitempty"`                  // bounded <= 4096 bytes, sanitized
	Cause            *OutcomeCause    `protobuf:"bytes,9,opt,name=cause,proto3" json:"cause,omitempty"`                                                 // REQUIRED
	Result           *ResultEnvelope  `protobuf:"bytes,10,opt,name=result,proto3" json:"result,omitempty"`                                              // the typed function result (success outcomes)
	ExecutionStarted bool             `protobuf:"varint,11,opt,name=execution_started,json=executionStarted,proto3" json:"execution_started,omitempty"` // THE BILLING FACT, structural rather than derived from a
	// cause-code allowlist (#480c): false => the author never
	// ran and zero execution budget was billed, so the
	// RecordOwner may immediately mint attempt_ordinal + 1 and
	// dispatch it elsewhere with no cooldown and no budget draw
	WeightsReceipts []*WeightsReceiptRef  `protobuf:"bytes,12,rep,name=weights_receipts,json=weightsReceipts,proto3" json:"weights_receipts,omitempty"` // committed job outputs only; sorted by slot
	Observation     *ExecutionObservation `protobuf:"bytes,13,opt,name=observation,proto3" json:"observation,omitempty"`                                // exact outcome-owned observation
	unknownFields   protoimpl.UnknownFields
	sizeCache       protoimpl.SizeCache
}

func (x *AttemptOutcomeBody) Reset() {
	*x = AttemptOutcomeBody{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[163]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *AttemptOutcomeBody) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*AttemptOutcomeBody) ProtoMessage() {}

func (x *AttemptOutcomeBody) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[163]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use AttemptOutcomeBody.ProtoReflect.Descriptor instead.
func (*AttemptOutcomeBody) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{163}
}

func (x *AttemptOutcomeBody) GetRequestId() string {
	if x != nil {
		return x.RequestId
	}
	return ""
}

func (x *AttemptOutcomeBody) GetAttemptOrdinal() uint64 {
	if x != nil {
		return x.AttemptOrdinal
	}
	return 0
}

func (x *AttemptOutcomeBody) GetInvocationSpecDigest() string {
	if x != nil {
		return x.InvocationSpecDigest
	}
	return ""
}

func (x *AttemptOutcomeBody) GetStatus() OutcomeStatus {
	if x != nil {
		return x.Status
	}
	return OutcomeStatus_OUTCOME_STATUS_UNSPECIFIED
}

func (x *AttemptOutcomeBody) GetOutputManifest() *OutputManifest {
	if x != nil {
		return x.OutputManifest
	}
	return nil
}

func (x *AttemptOutcomeBody) GetMetrics() *AttemptMetrics {
	if x != nil {
		return x.Metrics
	}
	return nil
}

func (x *AttemptOutcomeBody) GetTriageBundle() *TriageBundleRef {
	if x != nil {
		return x.TriageBundle
	}
	return nil
}

func (x *AttemptOutcomeBody) GetSafeMessage() string {
	if x != nil {
		return x.SafeMessage
	}
	return ""
}

func (x *AttemptOutcomeBody) GetCause() *OutcomeCause {
	if x != nil {
		return x.Cause
	}
	return nil
}

func (x *AttemptOutcomeBody) GetResult() *ResultEnvelope {
	if x != nil {
		return x.Result
	}
	return nil
}

func (x *AttemptOutcomeBody) GetExecutionStarted() bool {
	if x != nil {
		return x.ExecutionStarted
	}
	return false
}

func (x *AttemptOutcomeBody) GetWeightsReceipts() []*WeightsReceiptRef {
	if x != nil {
		return x.WeightsReceipts
	}
	return nil
}

func (x *AttemptOutcomeBody) GetObservation() *ExecutionObservation {
	if x != nil {
		return x.Observation
	}
	return nil
}

// Carried exact bytes, not a structured duplicate. The receiver hashes 2, compares 1, then opens
// the canonical WeightsReceipt document. This reference is nested in AttemptOutcomeBody/1 and
// therefore has no format tag of its own.
type WeightsReceiptRef struct {
	state                        protoimpl.MessageState `protogen:"open.v1"`
	WeightsReceiptDigest         []byte                 `protobuf:"bytes,1,opt,name=weights_receipt_digest,json=weightsReceiptDigest,proto3" json:"weights_receipt_digest,omitempty"`
	WeightsReceiptCanonicalBytes []byte                 `protobuf:"bytes,2,opt,name=weights_receipt_canonical_bytes,json=weightsReceiptCanonicalBytes,proto3" json:"weights_receipt_canonical_bytes,omitempty"`
	unknownFields                protoimpl.UnknownFields
	sizeCache                    protoimpl.SizeCache
}

func (x *WeightsReceiptRef) Reset() {
	*x = WeightsReceiptRef{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[164]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *WeightsReceiptRef) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*WeightsReceiptRef) ProtoMessage() {}

func (x *WeightsReceiptRef) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[164]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use WeightsReceiptRef.ProtoReflect.Descriptor instead.
func (*WeightsReceiptRef) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{164}
}

func (x *WeightsReceiptRef) GetWeightsReceiptDigest() []byte {
	if x != nil {
		return x.WeightsReceiptDigest
	}
	return nil
}

func (x *WeightsReceiptRef) GetWeightsReceiptCanonicalBytes() []byte {
	if x != nil {
		return x.WeightsReceiptCanonicalBytes
	}
	return nil
}

// DOCUMENT SHAPE: canonical `cozy.worker.v1.WeightsReceipt/1`. TensorFS receipt bytes remain
// opaque to this protocol package; Runtime proves their digest before authoring this document.
type WeightsReceipt struct {
	state                         protoimpl.MessageState `protogen:"open.v1"`
	OwnerAuthorityScope           string                 `protobuf:"bytes,1,opt,name=owner_authority_scope,json=ownerAuthorityScope,proto3" json:"owner_authority_scope,omitempty"`
	RequestId                     string                 `protobuf:"bytes,2,opt,name=request_id,json=requestId,proto3" json:"request_id,omitempty"`
	InvocationSpecDigest          string                 `protobuf:"bytes,3,opt,name=invocation_spec_digest,json=invocationSpecDigest,proto3" json:"invocation_spec_digest,omitempty"`
	OutputSlot                    string                 `protobuf:"bytes,4,opt,name=output_slot,json=outputSlot,proto3" json:"output_slot,omitempty"`
	WeightsTransactionId          string                 `protobuf:"bytes,5,opt,name=weights_transaction_id,json=weightsTransactionId,proto3" json:"weights_transaction_id,omitempty"`
	TensorfsReceiptDigest         string                 `protobuf:"bytes,6,opt,name=tensorfs_receipt_digest,json=tensorfsReceiptDigest,proto3" json:"tensorfs_receipt_digest,omitempty"`
	TensorfsReceiptCanonicalBytes []byte                 `protobuf:"bytes,7,opt,name=tensorfs_receipt_canonical_bytes,json=tensorfsReceiptCanonicalBytes,proto3" json:"tensorfs_receipt_canonical_bytes,omitempty"`
	unknownFields                 protoimpl.UnknownFields
	sizeCache                     protoimpl.SizeCache
}

func (x *WeightsReceipt) Reset() {
	*x = WeightsReceipt{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[165]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *WeightsReceipt) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*WeightsReceipt) ProtoMessage() {}

func (x *WeightsReceipt) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[165]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use WeightsReceipt.ProtoReflect.Descriptor instead.
func (*WeightsReceipt) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{165}
}

func (x *WeightsReceipt) GetOwnerAuthorityScope() string {
	if x != nil {
		return x.OwnerAuthorityScope
	}
	return ""
}

func (x *WeightsReceipt) GetRequestId() string {
	if x != nil {
		return x.RequestId
	}
	return ""
}

func (x *WeightsReceipt) GetInvocationSpecDigest() string {
	if x != nil {
		return x.InvocationSpecDigest
	}
	return ""
}

func (x *WeightsReceipt) GetOutputSlot() string {
	if x != nil {
		return x.OutputSlot
	}
	return ""
}

func (x *WeightsReceipt) GetWeightsTransactionId() string {
	if x != nil {
		return x.WeightsTransactionId
	}
	return ""
}

func (x *WeightsReceipt) GetTensorfsReceiptDigest() string {
	if x != nil {
		return x.TensorfsReceiptDigest
	}
	return ""
}

func (x *WeightsReceipt) GetTensorfsReceiptCanonicalBytes() []byte {
	if x != nil {
		return x.TensorfsReceiptCanonicalBytes
	}
	return nil
}

// One produced TensorFS ObjectRef plus an opaque transaction-scoped reader token. source_ref uses
// `<namespace>:<token>` printable ASCII with no slash, backslash, URI, or filesystem semantics.
type WeightsObjectSource struct {
	state         protoimpl.MessageState `protogen:"open.v1"`
	ObjectId      string                 `protobuf:"bytes,1,opt,name=object_id,json=objectId,proto3" json:"object_id,omitempty"`
	Length        uint64                 `protobuf:"varint,2,opt,name=length,proto3" json:"length,omitempty"`
	SourceRef     string                 `protobuf:"bytes,3,opt,name=source_ref,json=sourceRef,proto3" json:"source_ref,omitempty"`
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *WeightsObjectSource) Reset() {
	*x = WeightsObjectSource{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[166]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *WeightsObjectSource) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*WeightsObjectSource) ProtoMessage() {}

func (x *WeightsObjectSource) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[166]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use WeightsObjectSource.ProtoReflect.Descriptor instead.
func (*WeightsObjectSource) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{166}
}

func (x *WeightsObjectSource) GetObjectId() string {
	if x != nil {
		return x.ObjectId
	}
	return ""
}

func (x *WeightsObjectSource) GetLength() uint64 {
	if x != nil {
		return x.Length
	}
	return 0
}

func (x *WeightsObjectSource) GetSourceRef() string {
	if x != nil {
		return x.SourceRef
	}
	return ""
}

// Runtime commits semantic intent in its per-Store workspace before
// native writes, then sends this observation. requested_writer_epoch is the positive
// Runtime-assigned epoch; Host must project it exactly and never allocate another.
// The attempt ordinal is routing only and does not enter transaction identity.
// The TensorFS declaration is carried directly, or by reference above the inline bound: it is
// not wrapped in a second Worker Protocol document. Its digest and length are the durable
// conflict/replay fence.
type WeightsIntentFrame struct {
	state                             protoimpl.MessageState `protogen:"open.v1"`
	RecordOwnerEpoch                  uint64                 `protobuf:"varint,1,opt,name=record_owner_epoch,json=recordOwnerEpoch,proto3" json:"record_owner_epoch,omitempty"`
	ControlStreamEpoch                uint64                 `protobuf:"varint,2,opt,name=control_stream_epoch,json=controlStreamEpoch,proto3" json:"control_stream_epoch,omitempty"`
	WorkerBootId                      string                 `protobuf:"bytes,3,opt,name=worker_boot_id,json=workerBootId,proto3" json:"worker_boot_id,omitempty"`
	RequestId                         string                 `protobuf:"bytes,5,opt,name=request_id,json=requestId,proto3" json:"request_id,omitempty"`
	AttemptOrdinal                    uint64                 `protobuf:"varint,6,opt,name=attempt_ordinal,json=attemptOrdinal,proto3" json:"attempt_ordinal,omitempty"`
	InvocationSpecDigest              []byte                 `protobuf:"bytes,7,opt,name=invocation_spec_digest,json=invocationSpecDigest,proto3" json:"invocation_spec_digest,omitempty"`
	OutputSlot                        string                 `protobuf:"bytes,8,opt,name=output_slot,json=outputSlot,proto3" json:"output_slot,omitempty"`
	TensorfsDeclarationDigest         []byte                 `protobuf:"bytes,9,opt,name=tensorfs_declaration_digest,json=tensorfsDeclarationDigest,proto3" json:"tensorfs_declaration_digest,omitempty"`
	RequestedWriterEpoch              uint64                 `protobuf:"varint,10,opt,name=requested_writer_epoch,json=requestedWriterEpoch,proto3" json:"requested_writer_epoch,omitempty"`
	TensorfsDeclarationCanonicalBytes []byte                 `protobuf:"bytes,11,opt,name=tensorfs_declaration_canonical_bytes,json=tensorfsDeclarationCanonicalBytes,proto3" json:"tensorfs_declaration_canonical_bytes,omitempty"`
	// Created only by Runtime. The host validates and durably binds this supplied ID to
	// owner/request/spec/slot, refuses conflicting bindings, and echoes it; it never hashes
	// a second transaction ID. Required lowercase sha256:<64 hex>.
	WeightsTransactionId string `protobuf:"bytes,12,opt,name=weights_transaction_id,json=weightsTransactionId,proto3" json:"weights_transaction_id,omitempty"`
	// Exact byte length of the declaration. One over MaxWeightsDeclarationBytes rides by
	// reference: field 11 is empty, and this length with the digest names the declaration
	// Runtime's workspace journals. Required.
	TensorfsDeclarationLength uint64 `protobuf:"varint,13,opt,name=tensorfs_declaration_length,json=tensorfsDeclarationLength,proto3" json:"tensorfs_declaration_length,omitempty"`
	unknownFields             protoimpl.UnknownFields
	sizeCache                 protoimpl.SizeCache
}

func (x *WeightsIntentFrame) Reset() {
	*x = WeightsIntentFrame{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[167]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *WeightsIntentFrame) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*WeightsIntentFrame) ProtoMessage() {}

func (x *WeightsIntentFrame) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[167]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use WeightsIntentFrame.ProtoReflect.Descriptor instead.
func (*WeightsIntentFrame) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{167}
}

func (x *WeightsIntentFrame) GetRecordOwnerEpoch() uint64 {
	if x != nil {
		return x.RecordOwnerEpoch
	}
	return 0
}

func (x *WeightsIntentFrame) GetControlStreamEpoch() uint64 {
	if x != nil {
		return x.ControlStreamEpoch
	}
	return 0
}

func (x *WeightsIntentFrame) GetWorkerBootId() string {
	if x != nil {
		return x.WorkerBootId
	}
	return ""
}

func (x *WeightsIntentFrame) GetRequestId() string {
	if x != nil {
		return x.RequestId
	}
	return ""
}

func (x *WeightsIntentFrame) GetAttemptOrdinal() uint64 {
	if x != nil {
		return x.AttemptOrdinal
	}
	return 0
}

func (x *WeightsIntentFrame) GetInvocationSpecDigest() []byte {
	if x != nil {
		return x.InvocationSpecDigest
	}
	return nil
}

func (x *WeightsIntentFrame) GetOutputSlot() string {
	if x != nil {
		return x.OutputSlot
	}
	return ""
}

func (x *WeightsIntentFrame) GetTensorfsDeclarationDigest() []byte {
	if x != nil {
		return x.TensorfsDeclarationDigest
	}
	return nil
}

func (x *WeightsIntentFrame) GetRequestedWriterEpoch() uint64 {
	if x != nil {
		return x.RequestedWriterEpoch
	}
	return 0
}

func (x *WeightsIntentFrame) GetTensorfsDeclarationCanonicalBytes() []byte {
	if x != nil {
		return x.TensorfsDeclarationCanonicalBytes
	}
	return nil
}

func (x *WeightsIntentFrame) GetWeightsTransactionId() string {
	if x != nil {
		return x.WeightsTransactionId
	}
	return ""
}

func (x *WeightsIntentFrame) GetTensorfsDeclarationLength() uint64 {
	if x != nil {
		return x.TensorfsDeclarationLength
	}
	return 0
}

// Runtime's workspace readiness reply and internal intent/receipt observation result.
// The Host only forwards Runtime-authored readiness replies; it never
// injects this message to authorize native writes. REFUSED grants no writer authority.
type WeightsHostAck struct {
	state                     protoimpl.MessageState `protogen:"open.v1"`
	RecordOwnerEpoch          uint64                 `protobuf:"varint,1,opt,name=record_owner_epoch,json=recordOwnerEpoch,proto3" json:"record_owner_epoch,omitempty"`
	ControlStreamEpoch        uint64                 `protobuf:"varint,2,opt,name=control_stream_epoch,json=controlStreamEpoch,proto3" json:"control_stream_epoch,omitempty"`
	WorkerBootId              string                 `protobuf:"bytes,3,opt,name=worker_boot_id,json=workerBootId,proto3" json:"worker_boot_id,omitempty"`
	RequestId                 string                 `protobuf:"bytes,5,opt,name=request_id,json=requestId,proto3" json:"request_id,omitempty"`
	AttemptOrdinal            uint64                 `protobuf:"varint,6,opt,name=attempt_ordinal,json=attemptOrdinal,proto3" json:"attempt_ordinal,omitempty"`
	InvocationSpecDigest      []byte                 `protobuf:"bytes,7,opt,name=invocation_spec_digest,json=invocationSpecDigest,proto3" json:"invocation_spec_digest,omitempty"`
	OutputSlot                string                 `protobuf:"bytes,8,opt,name=output_slot,json=outputSlot,proto3" json:"output_slot,omitempty"`
	WeightsTransactionId      string                 `protobuf:"bytes,9,opt,name=weights_transaction_id,json=weightsTransactionId,proto3" json:"weights_transaction_id,omitempty"`
	WriterEpoch               uint64                 `protobuf:"varint,10,opt,name=writer_epoch,json=writerEpoch,proto3" json:"writer_epoch,omitempty"`
	TensorfsDeclarationDigest []byte                 `protobuf:"bytes,11,opt,name=tensorfs_declaration_digest,json=tensorfsDeclarationDigest,proto3" json:"tensorfs_declaration_digest,omitempty"`
	Stage                     WeightsHostStage       `protobuf:"varint,12,opt,name=stage,proto3,enum=cozy.worker.v1.WeightsHostStage" json:"stage,omitempty"`
	Outcome                   WeightsHostOutcome     `protobuf:"varint,13,opt,name=outcome,proto3,enum=cozy.worker.v1.WeightsHostOutcome" json:"outcome,omitempty"`
	Refusal                   WeightsHostRefusal     `protobuf:"varint,14,opt,name=refusal,proto3,enum=cozy.worker.v1.WeightsHostRefusal" json:"refusal,omitempty"`
	WeightsReceipt            *WeightsReceiptRef     `protobuf:"bytes,15,opt,name=weights_receipt,json=weightsReceipt,proto3" json:"weights_receipt,omitempty"`
	Manifest                  *Ref                   `protobuf:"bytes,16,opt,name=manifest,proto3" json:"manifest,omitempty"` // present exactly with weights_receipt; the produced Manifest identity
	// downstream nodes consume without parsing opaque TensorFS receipt bytes
	Checkpoint    *CheckpointRef `protobuf:"bytes,17,opt,name=checkpoint,proto3" json:"checkpoint,omitempty"` // INTENT: natively validated restore head; CHECKPOINT: exact
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *WeightsHostAck) Reset() {
	*x = WeightsHostAck{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[168]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *WeightsHostAck) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*WeightsHostAck) ProtoMessage() {}

func (x *WeightsHostAck) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[168]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use WeightsHostAck.ProtoReflect.Descriptor instead.
func (*WeightsHostAck) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{168}
}

func (x *WeightsHostAck) GetRecordOwnerEpoch() uint64 {
	if x != nil {
		return x.RecordOwnerEpoch
	}
	return 0
}

func (x *WeightsHostAck) GetControlStreamEpoch() uint64 {
	if x != nil {
		return x.ControlStreamEpoch
	}
	return 0
}

func (x *WeightsHostAck) GetWorkerBootId() string {
	if x != nil {
		return x.WorkerBootId
	}
	return ""
}

func (x *WeightsHostAck) GetRequestId() string {
	if x != nil {
		return x.RequestId
	}
	return ""
}

func (x *WeightsHostAck) GetAttemptOrdinal() uint64 {
	if x != nil {
		return x.AttemptOrdinal
	}
	return 0
}

func (x *WeightsHostAck) GetInvocationSpecDigest() []byte {
	if x != nil {
		return x.InvocationSpecDigest
	}
	return nil
}

func (x *WeightsHostAck) GetOutputSlot() string {
	if x != nil {
		return x.OutputSlot
	}
	return ""
}

func (x *WeightsHostAck) GetWeightsTransactionId() string {
	if x != nil {
		return x.WeightsTransactionId
	}
	return ""
}

func (x *WeightsHostAck) GetWriterEpoch() uint64 {
	if x != nil {
		return x.WriterEpoch
	}
	return 0
}

func (x *WeightsHostAck) GetTensorfsDeclarationDigest() []byte {
	if x != nil {
		return x.TensorfsDeclarationDigest
	}
	return nil
}

func (x *WeightsHostAck) GetStage() WeightsHostStage {
	if x != nil {
		return x.Stage
	}
	return WeightsHostStage_WEIGHTS_HOST_STAGE_UNSPECIFIED
}

func (x *WeightsHostAck) GetOutcome() WeightsHostOutcome {
	if x != nil {
		return x.Outcome
	}
	return WeightsHostOutcome_WEIGHTS_HOST_OUTCOME_UNSPECIFIED
}

func (x *WeightsHostAck) GetRefusal() WeightsHostRefusal {
	if x != nil {
		return x.Refusal
	}
	return WeightsHostRefusal_WEIGHTS_HOST_REFUSAL_UNSPECIFIED
}

func (x *WeightsHostAck) GetWeightsReceipt() *WeightsReceiptRef {
	if x != nil {
		return x.WeightsReceipt
	}
	return nil
}

func (x *WeightsHostAck) GetManifest() *Ref {
	if x != nil {
		return x.Manifest
	}
	return nil
}

func (x *WeightsHostAck) GetCheckpoint() *CheckpointRef {
	if x != nil {
		return x.Checkpoint
	}
	return nil
}

// Fixed-size TensorFS Link facts. No object inventory or model-growing role list rides here.
type CheckpointRef struct {
	state         protoimpl.MessageState `protogen:"open.v1"`
	Head          *Ref                   `protobuf:"bytes,1,opt,name=head,proto3" json:"head,omitempty"`
	PlanDigest    []byte                 `protobuf:"bytes,2,opt,name=plan_digest,json=planDigest,proto3" json:"plan_digest,omitempty"`
	Index         uint64                 `protobuf:"varint,3,opt,name=index,proto3" json:"index,omitempty"`
	Bytes         uint64                 `protobuf:"varint,4,opt,name=bytes,proto3" json:"bytes,omitempty"` // cumulative locally accepted bytes; not a remote durability receipt
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *CheckpointRef) Reset() {
	*x = CheckpointRef{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[169]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *CheckpointRef) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*CheckpointRef) ProtoMessage() {}

func (x *CheckpointRef) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[169]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use CheckpointRef.ProtoReflect.Descriptor instead.
func (*CheckpointRef) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{169}
}

func (x *CheckpointRef) GetHead() *Ref {
	if x != nil {
		return x.Head
	}
	return nil
}

func (x *CheckpointRef) GetPlanDigest() []byte {
	if x != nil {
		return x.PlanDigest
	}
	return nil
}

func (x *CheckpointRef) GetIndex() uint64 {
	if x != nil {
		return x.Index
	}
	return 0
}

func (x *CheckpointRef) GetBytes() uint64 {
	if x != nil {
		return x.Bytes
	}
	return 0
}

// Runtime -> host -> authenticated owner, journaled under the existing weights intent before
// ACK. The owner banks this Link closure with CheckpointPage/Transfer and retains its own
// remotely acknowledged prefix separately from this locally observed progress.
type WeightsCheckpointFrame struct {
	state                     protoimpl.MessageState `protogen:"open.v1"`
	RecordOwnerEpoch          uint64                 `protobuf:"varint,1,opt,name=record_owner_epoch,json=recordOwnerEpoch,proto3" json:"record_owner_epoch,omitempty"`
	ControlStreamEpoch        uint64                 `protobuf:"varint,2,opt,name=control_stream_epoch,json=controlStreamEpoch,proto3" json:"control_stream_epoch,omitempty"`
	WorkerBootId              string                 `protobuf:"bytes,3,opt,name=worker_boot_id,json=workerBootId,proto3" json:"worker_boot_id,omitempty"`
	RequestId                 string                 `protobuf:"bytes,5,opt,name=request_id,json=requestId,proto3" json:"request_id,omitempty"`
	AttemptOrdinal            uint64                 `protobuf:"varint,6,opt,name=attempt_ordinal,json=attemptOrdinal,proto3" json:"attempt_ordinal,omitempty"`
	InvocationSpecDigest      []byte                 `protobuf:"bytes,7,opt,name=invocation_spec_digest,json=invocationSpecDigest,proto3" json:"invocation_spec_digest,omitempty"`
	OutputSlot                string                 `protobuf:"bytes,8,opt,name=output_slot,json=outputSlot,proto3" json:"output_slot,omitempty"`
	WeightsTransactionId      string                 `protobuf:"bytes,9,opt,name=weights_transaction_id,json=weightsTransactionId,proto3" json:"weights_transaction_id,omitempty"`
	WriterEpoch               uint64                 `protobuf:"varint,10,opt,name=writer_epoch,json=writerEpoch,proto3" json:"writer_epoch,omitempty"`
	TensorfsDeclarationDigest []byte                 `protobuf:"bytes,11,opt,name=tensorfs_declaration_digest,json=tensorfsDeclarationDigest,proto3" json:"tensorfs_declaration_digest,omitempty"`
	Checkpoint                *CheckpointRef         `protobuf:"bytes,12,opt,name=checkpoint,proto3" json:"checkpoint,omitempty"`
	unknownFields             protoimpl.UnknownFields
	sizeCache                 protoimpl.SizeCache
}

func (x *WeightsCheckpointFrame) Reset() {
	*x = WeightsCheckpointFrame{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[170]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *WeightsCheckpointFrame) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*WeightsCheckpointFrame) ProtoMessage() {}

func (x *WeightsCheckpointFrame) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[170]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use WeightsCheckpointFrame.ProtoReflect.Descriptor instead.
func (*WeightsCheckpointFrame) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{170}
}

func (x *WeightsCheckpointFrame) GetRecordOwnerEpoch() uint64 {
	if x != nil {
		return x.RecordOwnerEpoch
	}
	return 0
}

func (x *WeightsCheckpointFrame) GetControlStreamEpoch() uint64 {
	if x != nil {
		return x.ControlStreamEpoch
	}
	return 0
}

func (x *WeightsCheckpointFrame) GetWorkerBootId() string {
	if x != nil {
		return x.WorkerBootId
	}
	return ""
}

func (x *WeightsCheckpointFrame) GetRequestId() string {
	if x != nil {
		return x.RequestId
	}
	return ""
}

func (x *WeightsCheckpointFrame) GetAttemptOrdinal() uint64 {
	if x != nil {
		return x.AttemptOrdinal
	}
	return 0
}

func (x *WeightsCheckpointFrame) GetInvocationSpecDigest() []byte {
	if x != nil {
		return x.InvocationSpecDigest
	}
	return nil
}

func (x *WeightsCheckpointFrame) GetOutputSlot() string {
	if x != nil {
		return x.OutputSlot
	}
	return ""
}

func (x *WeightsCheckpointFrame) GetWeightsTransactionId() string {
	if x != nil {
		return x.WeightsTransactionId
	}
	return ""
}

func (x *WeightsCheckpointFrame) GetWriterEpoch() uint64 {
	if x != nil {
		return x.WriterEpoch
	}
	return 0
}

func (x *WeightsCheckpointFrame) GetTensorfsDeclarationDigest() []byte {
	if x != nil {
		return x.TensorfsDeclarationDigest
	}
	return nil
}

func (x *WeightsCheckpointFrame) GetCheckpoint() *CheckpointRef {
	if x != nil {
		return x.Checkpoint
	}
	return nil
}

// A current intent is registered and externally observable BEFORE writer ACK. The owner
// restores its banked Link under that intent's exact authority, then calls Ready once.
// An absent checkpoint explicitly chooses no external restore; verified local progress stays.
// Same intent/head replays; changed head conflicts. Cancellation tombstones even while waiting.
type WeightsIntentReadyRequest struct {
	state              protoimpl.MessageState    `protogen:"open.v1"`
	RecordOwnerEpoch   uint64                    `protobuf:"varint,1,opt,name=record_owner_epoch,json=recordOwnerEpoch,proto3" json:"record_owner_epoch,omitempty"`
	ControlStreamEpoch uint64                    `protobuf:"varint,2,opt,name=control_stream_epoch,json=controlStreamEpoch,proto3" json:"control_stream_epoch,omitempty"`
	WorkerBootId       string                    `protobuf:"bytes,3,opt,name=worker_boot_id,json=workerBootId,proto3" json:"worker_boot_id,omitempty"`
	Weights            *WeightsCheckpointSubject `protobuf:"bytes,5,opt,name=weights,proto3" json:"weights,omitempty"`
	AttemptOrdinal     uint64                    `protobuf:"varint,6,opt,name=attempt_ordinal,json=attemptOrdinal,proto3" json:"attempt_ordinal,omitempty"`
	Checkpoint         *CheckpointRef            `protobuf:"bytes,7,opt,name=checkpoint,proto3" json:"checkpoint,omitempty"`
	unknownFields      protoimpl.UnknownFields
	sizeCache          protoimpl.SizeCache
}

func (x *WeightsIntentReadyRequest) Reset() {
	*x = WeightsIntentReadyRequest{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[171]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *WeightsIntentReadyRequest) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*WeightsIntentReadyRequest) ProtoMessage() {}

func (x *WeightsIntentReadyRequest) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[171]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use WeightsIntentReadyRequest.ProtoReflect.Descriptor instead.
func (*WeightsIntentReadyRequest) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{171}
}

func (x *WeightsIntentReadyRequest) GetRecordOwnerEpoch() uint64 {
	if x != nil {
		return x.RecordOwnerEpoch
	}
	return 0
}

func (x *WeightsIntentReadyRequest) GetControlStreamEpoch() uint64 {
	if x != nil {
		return x.ControlStreamEpoch
	}
	return 0
}

func (x *WeightsIntentReadyRequest) GetWorkerBootId() string {
	if x != nil {
		return x.WorkerBootId
	}
	return ""
}

func (x *WeightsIntentReadyRequest) GetWeights() *WeightsCheckpointSubject {
	if x != nil {
		return x.Weights
	}
	return nil
}

func (x *WeightsIntentReadyRequest) GetAttemptOrdinal() uint64 {
	if x != nil {
		return x.AttemptOrdinal
	}
	return 0
}

func (x *WeightsIntentReadyRequest) GetCheckpoint() *CheckpointRef {
	if x != nil {
		return x.Checkpoint
	}
	return nil
}

// Loopback only: the host adds the exact TensorFS-authored declaration already held by its
// current intent. Runtime verifies its digest and native checkpoint closure before success.
// This never opens/fences/imports a writer. The host journals Ready and releases its existing
// INTENT ACK only after this synchronous verdict. Complete request <= MaxInlineControlBytes.
type ValidateWeightsCheckpointRequest struct {
	state                             protoimpl.MessageState     `protogen:"open.v1"`
	Intent                            *WeightsIntentReadyRequest `protobuf:"bytes,1,opt,name=intent,proto3" json:"intent,omitempty"`
	TensorfsDeclarationCanonicalBytes []byte                     `protobuf:"bytes,2,opt,name=tensorfs_declaration_canonical_bytes,json=tensorfsDeclarationCanonicalBytes,proto3" json:"tensorfs_declaration_canonical_bytes,omitempty"` // <= MaxWeightsDeclarationBytes
	unknownFields                     protoimpl.UnknownFields
	sizeCache                         protoimpl.SizeCache
}

func (x *ValidateWeightsCheckpointRequest) Reset() {
	*x = ValidateWeightsCheckpointRequest{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[172]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *ValidateWeightsCheckpointRequest) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*ValidateWeightsCheckpointRequest) ProtoMessage() {}

func (x *ValidateWeightsCheckpointRequest) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[172]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use ValidateWeightsCheckpointRequest.ProtoReflect.Descriptor instead.
func (*ValidateWeightsCheckpointRequest) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{172}
}

func (x *ValidateWeightsCheckpointRequest) GetIntent() *WeightsIntentReadyRequest {
	if x != nil {
		return x.Intent
	}
	return nil
}

func (x *ValidateWeightsCheckpointRequest) GetTensorfsDeclarationCanonicalBytes() []byte {
	if x != nil {
		return x.TensorfsDeclarationCanonicalBytes
	}
	return nil
}

type ValidateWeightsCheckpointResult struct {
	state         protoimpl.MessageState    `protogen:"open.v1"`
	Weights       *WeightsCheckpointSubject `protobuf:"bytes,1,opt,name=weights,proto3" json:"weights,omitempty"`
	Checkpoint    *CheckpointRef            `protobuf:"bytes,2,opt,name=checkpoint,proto3" json:"checkpoint,omitempty"`
	Valid         bool                      `protobuf:"varint,3,opt,name=valid,proto3" json:"valid,omitempty"`
	SafeCode      string                    `protobuf:"bytes,4,opt,name=safe_code,json=safeCode,proto3" json:"safe_code,omitempty"`
	SafeDetail    string                    `protobuf:"bytes,5,opt,name=safe_detail,json=safeDetail,proto3" json:"safe_detail,omitempty"`
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *ValidateWeightsCheckpointResult) Reset() {
	*x = ValidateWeightsCheckpointResult{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[173]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *ValidateWeightsCheckpointResult) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*ValidateWeightsCheckpointResult) ProtoMessage() {}

func (x *ValidateWeightsCheckpointResult) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[173]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use ValidateWeightsCheckpointResult.ProtoReflect.Descriptor instead.
func (*ValidateWeightsCheckpointResult) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{173}
}

func (x *ValidateWeightsCheckpointResult) GetWeights() *WeightsCheckpointSubject {
	if x != nil {
		return x.Weights
	}
	return nil
}

func (x *ValidateWeightsCheckpointResult) GetCheckpoint() *CheckpointRef {
	if x != nil {
		return x.Checkpoint
	}
	return nil
}

func (x *ValidateWeightsCheckpointResult) GetValid() bool {
	if x != nil {
		return x.Valid
	}
	return false
}

func (x *ValidateWeightsCheckpointResult) GetSafeCode() string {
	if x != nil {
		return x.SafeCode
	}
	return ""
}

func (x *ValidateWeightsCheckpointResult) GetSafeDetail() string {
	if x != nil {
		return x.SafeDetail
	}
	return ""
}

// The Host refused to record ONE weights transaction frame (intent, checkpoint or receipt):
// a stale writer, a conflicting intent or receipt, or an invalid inventory. The refused frame
// is not forwarded and the transaction has no Host custody; the control stream and every other
// transaction and attempt continue. The owner's later WeightsTransfer of that receipt fails
// alone. Runtime stops the transaction's work and fails its attempt promptly with the detail
// instead of reporting an output the owner cannot collect.
type WeightsTransactionRefused struct {
	state                protoimpl.MessageState `protogen:"open.v1"`
	RecordOwnerEpoch     uint64                 `protobuf:"varint,1,opt,name=record_owner_epoch,json=recordOwnerEpoch,proto3" json:"record_owner_epoch,omitempty"`
	ControlStreamEpoch   uint64                 `protobuf:"varint,2,opt,name=control_stream_epoch,json=controlStreamEpoch,proto3" json:"control_stream_epoch,omitempty"`
	WorkerBootId         string                 `protobuf:"bytes,3,opt,name=worker_boot_id,json=workerBootId,proto3" json:"worker_boot_id,omitempty"`
	RequestId            string                 `protobuf:"bytes,5,opt,name=request_id,json=requestId,proto3" json:"request_id,omitempty"`
	AttemptOrdinal       uint64                 `protobuf:"varint,6,opt,name=attempt_ordinal,json=attemptOrdinal,proto3" json:"attempt_ordinal,omitempty"`
	InvocationSpecDigest []byte                 `protobuf:"bytes,7,opt,name=invocation_spec_digest,json=invocationSpecDigest,proto3" json:"invocation_spec_digest,omitempty"`
	OutputSlot           string                 `protobuf:"bytes,8,opt,name=output_slot,json=outputSlot,proto3" json:"output_slot,omitempty"`
	WeightsTransactionId string                 `protobuf:"bytes,9,opt,name=weights_transaction_id,json=weightsTransactionId,proto3" json:"weights_transaction_id,omitempty"`
	WriterEpoch          uint64                 `protobuf:"varint,10,opt,name=writer_epoch,json=writerEpoch,proto3" json:"writer_epoch,omitempty"`
	Stage                WeightsHostStage       `protobuf:"varint,11,opt,name=stage,proto3,enum=cozy.worker.v1.WeightsHostStage" json:"stage,omitempty"`
	Refusal              WeightsHostRefusal     `protobuf:"varint,12,opt,name=refusal,proto3,enum=cozy.worker.v1.WeightsHostRefusal" json:"refusal,omitempty"`
	SafeDetail           string                 `protobuf:"bytes,13,opt,name=safe_detail,json=safeDetail,proto3" json:"safe_detail,omitempty"` // printable ASCII, <= 1024 bytes, naming the refusal and its recovery
	unknownFields        protoimpl.UnknownFields
	sizeCache            protoimpl.SizeCache
}

func (x *WeightsTransactionRefused) Reset() {
	*x = WeightsTransactionRefused{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[174]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *WeightsTransactionRefused) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*WeightsTransactionRefused) ProtoMessage() {}

func (x *WeightsTransactionRefused) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[174]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use WeightsTransactionRefused.ProtoReflect.Descriptor instead.
func (*WeightsTransactionRefused) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{174}
}

func (x *WeightsTransactionRefused) GetRecordOwnerEpoch() uint64 {
	if x != nil {
		return x.RecordOwnerEpoch
	}
	return 0
}

func (x *WeightsTransactionRefused) GetControlStreamEpoch() uint64 {
	if x != nil {
		return x.ControlStreamEpoch
	}
	return 0
}

func (x *WeightsTransactionRefused) GetWorkerBootId() string {
	if x != nil {
		return x.WorkerBootId
	}
	return ""
}

func (x *WeightsTransactionRefused) GetRequestId() string {
	if x != nil {
		return x.RequestId
	}
	return ""
}

func (x *WeightsTransactionRefused) GetAttemptOrdinal() uint64 {
	if x != nil {
		return x.AttemptOrdinal
	}
	return 0
}

func (x *WeightsTransactionRefused) GetInvocationSpecDigest() []byte {
	if x != nil {
		return x.InvocationSpecDigest
	}
	return nil
}

func (x *WeightsTransactionRefused) GetOutputSlot() string {
	if x != nil {
		return x.OutputSlot
	}
	return ""
}

func (x *WeightsTransactionRefused) GetWeightsTransactionId() string {
	if x != nil {
		return x.WeightsTransactionId
	}
	return ""
}

func (x *WeightsTransactionRefused) GetWriterEpoch() uint64 {
	if x != nil {
		return x.WriterEpoch
	}
	return 0
}

func (x *WeightsTransactionRefused) GetStage() WeightsHostStage {
	if x != nil {
		return x.Stage
	}
	return WeightsHostStage_WEIGHTS_HOST_STAGE_UNSPECIFIED
}

func (x *WeightsTransactionRefused) GetRefusal() WeightsHostRefusal {
	if x != nil {
		return x.Refusal
	}
	return WeightsHostRefusal_WEIGHTS_HOST_REFUSAL_UNSPECIFIED
}

func (x *WeightsTransactionRefused) GetSafeDetail() string {
	if x != nil {
		return x.SafeDetail
	}
	return ""
}

// Runtime sends the exact committed receipt and exact local source inventory together. The
// supervisor validates all routing copies and writer_epoch, commits both before ACK, and then
// may forward this frame to the external RecordOwner. A stale child cannot publish a late receipt.
type WeightsReceiptFrame struct {
	state                     protoimpl.MessageState `protogen:"open.v1"`
	RecordOwnerEpoch          uint64                 `protobuf:"varint,1,opt,name=record_owner_epoch,json=recordOwnerEpoch,proto3" json:"record_owner_epoch,omitempty"`
	ControlStreamEpoch        uint64                 `protobuf:"varint,2,opt,name=control_stream_epoch,json=controlStreamEpoch,proto3" json:"control_stream_epoch,omitempty"`
	WorkerBootId              string                 `protobuf:"bytes,3,opt,name=worker_boot_id,json=workerBootId,proto3" json:"worker_boot_id,omitempty"`
	RequestId                 string                 `protobuf:"bytes,5,opt,name=request_id,json=requestId,proto3" json:"request_id,omitempty"`
	AttemptOrdinal            uint64                 `protobuf:"varint,6,opt,name=attempt_ordinal,json=attemptOrdinal,proto3" json:"attempt_ordinal,omitempty"`
	InvocationSpecDigest      []byte                 `protobuf:"bytes,7,opt,name=invocation_spec_digest,json=invocationSpecDigest,proto3" json:"invocation_spec_digest,omitempty"`
	OutputSlot                string                 `protobuf:"bytes,8,opt,name=output_slot,json=outputSlot,proto3" json:"output_slot,omitempty"`
	WeightsTransactionId      string                 `protobuf:"bytes,9,opt,name=weights_transaction_id,json=weightsTransactionId,proto3" json:"weights_transaction_id,omitempty"`
	WriterEpoch               uint64                 `protobuf:"varint,10,opt,name=writer_epoch,json=writerEpoch,proto3" json:"writer_epoch,omitempty"`
	TensorfsDeclarationDigest []byte                 `protobuf:"bytes,11,opt,name=tensorfs_declaration_digest,json=tensorfsDeclarationDigest,proto3" json:"tensorfs_declaration_digest,omitempty"`
	WeightsReceipt            *WeightsReceiptRef     `protobuf:"bytes,12,opt,name=weights_receipt,json=weightsReceipt,proto3" json:"weights_receipt,omitempty"`
	Objects                   []*WeightsObjectSource `protobuf:"bytes,13,rep,name=objects,proto3" json:"objects,omitempty"` // sorted uniquely by object_id; committed in the
	// same host-ledger transaction as weights_receipt
	Manifest      *Ref `protobuf:"bytes,14,opt,name=manifest,proto3" json:"manifest,omitempty"` // exact produced TensorFS Manifest; it is also one row in objects
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *WeightsReceiptFrame) Reset() {
	*x = WeightsReceiptFrame{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[175]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *WeightsReceiptFrame) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*WeightsReceiptFrame) ProtoMessage() {}

func (x *WeightsReceiptFrame) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[175]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use WeightsReceiptFrame.ProtoReflect.Descriptor instead.
func (*WeightsReceiptFrame) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{175}
}

func (x *WeightsReceiptFrame) GetRecordOwnerEpoch() uint64 {
	if x != nil {
		return x.RecordOwnerEpoch
	}
	return 0
}

func (x *WeightsReceiptFrame) GetControlStreamEpoch() uint64 {
	if x != nil {
		return x.ControlStreamEpoch
	}
	return 0
}

func (x *WeightsReceiptFrame) GetWorkerBootId() string {
	if x != nil {
		return x.WorkerBootId
	}
	return ""
}

func (x *WeightsReceiptFrame) GetRequestId() string {
	if x != nil {
		return x.RequestId
	}
	return ""
}

func (x *WeightsReceiptFrame) GetAttemptOrdinal() uint64 {
	if x != nil {
		return x.AttemptOrdinal
	}
	return 0
}

func (x *WeightsReceiptFrame) GetInvocationSpecDigest() []byte {
	if x != nil {
		return x.InvocationSpecDigest
	}
	return nil
}

func (x *WeightsReceiptFrame) GetOutputSlot() string {
	if x != nil {
		return x.OutputSlot
	}
	return ""
}

func (x *WeightsReceiptFrame) GetWeightsTransactionId() string {
	if x != nil {
		return x.WeightsTransactionId
	}
	return ""
}

func (x *WeightsReceiptFrame) GetWriterEpoch() uint64 {
	if x != nil {
		return x.WriterEpoch
	}
	return 0
}

func (x *WeightsReceiptFrame) GetTensorfsDeclarationDigest() []byte {
	if x != nil {
		return x.TensorfsDeclarationDigest
	}
	return nil
}

func (x *WeightsReceiptFrame) GetWeightsReceipt() *WeightsReceiptRef {
	if x != nil {
		return x.WeightsReceipt
	}
	return nil
}

func (x *WeightsReceiptFrame) GetObjects() []*WeightsObjectSource {
	if x != nil {
		return x.Objects
	}
	return nil
}

func (x *WeightsReceiptFrame) GetManifest() *Ref {
	if x != nil {
		return x.Manifest
	}
	return nil
}

// Compact supervisor-ledger state included in WorkerSnapshotBody/1. A stateless replacement
// Runtime does not author these rows; pod-supervisor injects them before digesting the external
// snapshot. It MUST NOT put declarations, receipt bytes, or object inventory here. After the
// snapshot ACK, the supervisor replays each RECEIPT transaction's exact WeightsReceiptFrame.
type WeightsTransactionStatus struct {
	state                     protoimpl.MessageState  `protogen:"open.v1"`
	WeightsTransactionId      string                  `protobuf:"bytes,1,opt,name=weights_transaction_id,json=weightsTransactionId,proto3" json:"weights_transaction_id,omitempty"`
	RequestId                 string                  `protobuf:"bytes,2,opt,name=request_id,json=requestId,proto3" json:"request_id,omitempty"`
	AttemptOrdinal            uint64                  `protobuf:"varint,3,opt,name=attempt_ordinal,json=attemptOrdinal,proto3" json:"attempt_ordinal,omitempty"`
	InvocationSpecDigest      string                  `protobuf:"bytes,4,opt,name=invocation_spec_digest,json=invocationSpecDigest,proto3" json:"invocation_spec_digest,omitempty"`
	OutputSlot                string                  `protobuf:"bytes,5,opt,name=output_slot,json=outputSlot,proto3" json:"output_slot,omitempty"`
	WriterEpoch               uint64                  `protobuf:"varint,6,opt,name=writer_epoch,json=writerEpoch,proto3" json:"writer_epoch,omitempty"`
	State                     WeightsTransactionState `protobuf:"varint,7,opt,name=state,proto3,enum=cozy.worker.v1.WeightsTransactionState" json:"state,omitempty"`
	TensorfsDeclarationDigest []byte                  `protobuf:"bytes,8,opt,name=tensorfs_declaration_digest,json=tensorfsDeclarationDigest,proto3" json:"tensorfs_declaration_digest,omitempty"`
	WeightsReceiptDigest      []byte                  `protobuf:"bytes,9,opt,name=weights_receipt_digest,json=weightsReceiptDigest,proto3" json:"weights_receipt_digest,omitempty"` // absent in INTENT; exact replay fence in RECEIPT
	Checkpoint                *CheckpointRef          `protobuf:"bytes,10,opt,name=checkpoint,proto3" json:"checkpoint,omitempty"`                                                  // latest locally recorded progress, not remote custody
	IntentReady               bool                    `protobuf:"varint,11,opt,name=intent_ready,json=intentReady,proto3" json:"intent_ready,omitempty"`                            // the owner has finished restore (or explicitly chose no restore)
	unknownFields             protoimpl.UnknownFields
	sizeCache                 protoimpl.SizeCache
}

func (x *WeightsTransactionStatus) Reset() {
	*x = WeightsTransactionStatus{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[176]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *WeightsTransactionStatus) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*WeightsTransactionStatus) ProtoMessage() {}

func (x *WeightsTransactionStatus) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[176]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use WeightsTransactionStatus.ProtoReflect.Descriptor instead.
func (*WeightsTransactionStatus) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{176}
}

func (x *WeightsTransactionStatus) GetWeightsTransactionId() string {
	if x != nil {
		return x.WeightsTransactionId
	}
	return ""
}

func (x *WeightsTransactionStatus) GetRequestId() string {
	if x != nil {
		return x.RequestId
	}
	return ""
}

func (x *WeightsTransactionStatus) GetAttemptOrdinal() uint64 {
	if x != nil {
		return x.AttemptOrdinal
	}
	return 0
}

func (x *WeightsTransactionStatus) GetInvocationSpecDigest() string {
	if x != nil {
		return x.InvocationSpecDigest
	}
	return ""
}

func (x *WeightsTransactionStatus) GetOutputSlot() string {
	if x != nil {
		return x.OutputSlot
	}
	return ""
}

func (x *WeightsTransactionStatus) GetWriterEpoch() uint64 {
	if x != nil {
		return x.WriterEpoch
	}
	return 0
}

func (x *WeightsTransactionStatus) GetState() WeightsTransactionState {
	if x != nil {
		return x.State
	}
	return WeightsTransactionState_WEIGHTS_TRANSACTION_STATE_UNSPECIFIED
}

func (x *WeightsTransactionStatus) GetTensorfsDeclarationDigest() []byte {
	if x != nil {
		return x.TensorfsDeclarationDigest
	}
	return nil
}

func (x *WeightsTransactionStatus) GetWeightsReceiptDigest() []byte {
	if x != nil {
		return x.WeightsReceiptDigest
	}
	return nil
}

func (x *WeightsTransactionStatus) GetCheckpoint() *CheckpointRef {
	if x != nil {
		return x.Checkpoint
	}
	return nil
}

func (x *WeightsTransactionStatus) GetIntentReady() bool {
	if x != nil {
		return x.IntentReady
	}
	return false
}

type WeightsObjectRef struct {
	state         protoimpl.MessageState `protogen:"open.v1"`
	ObjectId      string                 `protobuf:"bytes,1,opt,name=object_id,json=objectId,proto3" json:"object_id,omitempty"`
	Length        uint64                 `protobuf:"varint,2,opt,name=length,proto3" json:"length,omitempty"`
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *WeightsObjectRef) Reset() {
	*x = WeightsObjectRef{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[177]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *WeightsObjectRef) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*WeightsObjectRef) ProtoMessage() {}

func (x *WeightsObjectRef) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[177]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use WeightsObjectRef.ProtoReflect.Descriptor instead.
func (*WeightsObjectRef) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{177}
}

func (x *WeightsObjectRef) GetObjectId() string {
	if x != nil {
		return x.ObjectId
	}
	return ""
}

func (x *WeightsObjectRef) GetLength() uint64 {
	if x != nil {
		return x.Length
	}
	return 0
}

type WeightsUploadHeader struct {
	state         protoimpl.MessageState `protogen:"open.v1"`
	Name          string                 `protobuf:"bytes,1,opt,name=name,proto3" json:"name,omitempty"`
	Value         string                 `protobuf:"bytes,2,opt,name=value,proto3" json:"value,omitempty"`
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *WeightsUploadHeader) Reset() {
	*x = WeightsUploadHeader{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[178]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *WeightsUploadHeader) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*WeightsUploadHeader) ProtoMessage() {}

func (x *WeightsUploadHeader) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[178]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use WeightsUploadHeader.ProtoReflect.Descriptor instead.
func (*WeightsUploadHeader) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{178}
}

func (x *WeightsUploadHeader) GetName() string {
	if x != nil {
		return x.Name
	}
	return ""
}

func (x *WeightsUploadHeader) GetValue() string {
	if x != nil {
		return x.Value
	}
	return ""
}

// A th-059 upload capability: ONE object, ONE PUT, ONE short expiry, pinned to the object's own
// digest by its signed checksum header and to first-write-wins by `if-none-match: *`. url
// includes the scoped query credential, is memory-only, and MUST NOT reach a log or the host
// ledger.
//
// The supervisor validates it against the durable receipt and then hands it to the process that
// HOLDS the bytes, which is Runtime. Relaying 8 GiB back through the supervisor 4 MiB at a time
// bought nothing: the capability is already scoped to one object, Runtime cannot widen it, and
// the object store -- not the relay -- is what enforces the signed checksum. Runtime already
// spends exactly this kind of capability for job outputs (`DeliveryGrant.outputs`).
type WeightsUploadGrant struct {
	state           protoimpl.MessageState `protogen:"open.v1"`
	ObjectId        string                 `protobuf:"bytes,1,opt,name=object_id,json=objectId,proto3" json:"object_id,omitempty"`
	Length          uint64                 `protobuf:"varint,2,opt,name=length,proto3" json:"length,omitempty"`
	Url             string                 `protobuf:"bytes,3,opt,name=url,proto3" json:"url,omitempty"`
	RequiredHeaders []*WeightsUploadHeader `protobuf:"bytes,4,rep,name=required_headers,json=requiredHeaders,proto3" json:"required_headers,omitempty"`
	ExpiresAtUnix   uint64                 `protobuf:"varint,5,opt,name=expires_at_unix,json=expiresAtUnix,proto3" json:"expires_at_unix,omitempty"`
	unknownFields   protoimpl.UnknownFields
	sizeCache       protoimpl.SizeCache
}

func (x *WeightsUploadGrant) Reset() {
	*x = WeightsUploadGrant{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[179]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *WeightsUploadGrant) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*WeightsUploadGrant) ProtoMessage() {}

func (x *WeightsUploadGrant) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[179]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use WeightsUploadGrant.ProtoReflect.Descriptor instead.
func (*WeightsUploadGrant) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{179}
}

func (x *WeightsUploadGrant) GetObjectId() string {
	if x != nil {
		return x.ObjectId
	}
	return ""
}

func (x *WeightsUploadGrant) GetLength() uint64 {
	if x != nil {
		return x.Length
	}
	return 0
}

func (x *WeightsUploadGrant) GetUrl() string {
	if x != nil {
		return x.Url
	}
	return ""
}

func (x *WeightsUploadGrant) GetRequiredHeaders() []*WeightsUploadHeader {
	if x != nil {
		return x.RequiredHeaders
	}
	return nil
}

func (x *WeightsUploadGrant) GetExpiresAtUnix() uint64 {
	if x != nil {
		return x.ExpiresAtUnix
	}
	return 0
}

// pod-supervisor -> Runtime. The supervisor has already bound this transfer to the durable
// receipt inventory and its ledger; this frame delegates the one thing it cannot do better than
// Runtime, which is stream bytes it does not have. Runtime PUTs to the granted URL and answers
// with WeightsUploadResult. It is the supervisor's ledger, not this frame, that decides what
// was written -- custody is recorded before the writer opens and again when it closes.
type WeightsUploadRequest struct {
	state                protoimpl.MessageState `protogen:"open.v1"`
	RecordOwnerEpoch     uint64                 `protobuf:"varint,1,opt,name=record_owner_epoch,json=recordOwnerEpoch,proto3" json:"record_owner_epoch,omitempty"`
	ControlStreamEpoch   uint64                 `protobuf:"varint,2,opt,name=control_stream_epoch,json=controlStreamEpoch,proto3" json:"control_stream_epoch,omitempty"`
	WorkerBootId         string                 `protobuf:"bytes,3,opt,name=worker_boot_id,json=workerBootId,proto3" json:"worker_boot_id,omitempty"`
	WeightsTransactionId string                 `protobuf:"bytes,5,opt,name=weights_transaction_id,json=weightsTransactionId,proto3" json:"weights_transaction_id,omitempty"`
	WriterEpoch          uint64                 `protobuf:"varint,6,opt,name=writer_epoch,json=writerEpoch,proto3" json:"writer_epoch,omitempty"`
	ObjectId             string                 `protobuf:"bytes,7,opt,name=object_id,json=objectId,proto3" json:"object_id,omitempty"`
	SourceRef            string                 `protobuf:"bytes,8,opt,name=source_ref,json=sourceRef,proto3" json:"source_ref,omitempty"`
	Length               uint64                 `protobuf:"varint,9,opt,name=length,proto3" json:"length,omitempty"`
	OperationId          string                 `protobuf:"bytes,10,opt,name=operation_id,json=operationId,proto3" json:"operation_id,omitempty"`
	GrantRevision        uint64                 `protobuf:"varint,11,opt,name=grant_revision,json=grantRevision,proto3" json:"grant_revision,omitempty"`
	Grant                *WeightsUploadGrant    `protobuf:"bytes,12,opt,name=grant,proto3" json:"grant,omitempty"`
	unknownFields        protoimpl.UnknownFields
	sizeCache            protoimpl.SizeCache
}

func (x *WeightsUploadRequest) Reset() {
	*x = WeightsUploadRequest{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[180]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *WeightsUploadRequest) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*WeightsUploadRequest) ProtoMessage() {}

func (x *WeightsUploadRequest) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[180]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use WeightsUploadRequest.ProtoReflect.Descriptor instead.
func (*WeightsUploadRequest) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{180}
}

func (x *WeightsUploadRequest) GetRecordOwnerEpoch() uint64 {
	if x != nil {
		return x.RecordOwnerEpoch
	}
	return 0
}

func (x *WeightsUploadRequest) GetControlStreamEpoch() uint64 {
	if x != nil {
		return x.ControlStreamEpoch
	}
	return 0
}

func (x *WeightsUploadRequest) GetWorkerBootId() string {
	if x != nil {
		return x.WorkerBootId
	}
	return ""
}

func (x *WeightsUploadRequest) GetWeightsTransactionId() string {
	if x != nil {
		return x.WeightsTransactionId
	}
	return ""
}

func (x *WeightsUploadRequest) GetWriterEpoch() uint64 {
	if x != nil {
		return x.WriterEpoch
	}
	return 0
}

func (x *WeightsUploadRequest) GetObjectId() string {
	if x != nil {
		return x.ObjectId
	}
	return ""
}

func (x *WeightsUploadRequest) GetSourceRef() string {
	if x != nil {
		return x.SourceRef
	}
	return ""
}

func (x *WeightsUploadRequest) GetLength() uint64 {
	if x != nil {
		return x.Length
	}
	return 0
}

func (x *WeightsUploadRequest) GetOperationId() string {
	if x != nil {
		return x.OperationId
	}
	return ""
}

func (x *WeightsUploadRequest) GetGrantRevision() uint64 {
	if x != nil {
		return x.GrantRevision
	}
	return 0
}

func (x *WeightsUploadRequest) GetGrant() *WeightsUploadGrant {
	if x != nil {
		return x.Grant
	}
	return nil
}

// Runtime -> pod-supervisor. One terminal answer per WeightsUploadRequest. checksum_sha256 is
// what Runtime measured while streaming; the supervisor compares it to the durable ObjectRef,
// and the store independently enforced the same digest through the grant's signed checksum
// header, so neither party is trusted alone.
type WeightsUploadResult struct {
	state                protoimpl.MessageState `protogen:"open.v1"`
	RecordOwnerEpoch     uint64                 `protobuf:"varint,1,opt,name=record_owner_epoch,json=recordOwnerEpoch,proto3" json:"record_owner_epoch,omitempty"`
	ControlStreamEpoch   uint64                 `protobuf:"varint,2,opt,name=control_stream_epoch,json=controlStreamEpoch,proto3" json:"control_stream_epoch,omitempty"`
	WorkerBootId         string                 `protobuf:"bytes,3,opt,name=worker_boot_id,json=workerBootId,proto3" json:"worker_boot_id,omitempty"`
	WeightsTransactionId string                 `protobuf:"bytes,5,opt,name=weights_transaction_id,json=weightsTransactionId,proto3" json:"weights_transaction_id,omitempty"`
	WriterEpoch          uint64                 `protobuf:"varint,6,opt,name=writer_epoch,json=writerEpoch,proto3" json:"writer_epoch,omitempty"`
	ObjectId             string                 `protobuf:"bytes,7,opt,name=object_id,json=objectId,proto3" json:"object_id,omitempty"`
	OperationId          string                 `protobuf:"bytes,8,opt,name=operation_id,json=operationId,proto3" json:"operation_id,omitempty"`
	GrantRevision        uint64                 `protobuf:"varint,9,opt,name=grant_revision,json=grantRevision,proto3" json:"grant_revision,omitempty"`
	Outcome              WeightsUploadOutcome   `protobuf:"varint,10,opt,name=outcome,proto3,enum=cozy.worker.v1.WeightsUploadOutcome" json:"outcome,omitempty"`
	TransferredBytes     uint64                 `protobuf:"varint,11,opt,name=transferred_bytes,json=transferredBytes,proto3" json:"transferred_bytes,omitempty"`
	HttpStatus           uint32                 `protobuf:"varint,12,opt,name=http_status,json=httpStatus,proto3" json:"http_status,omitempty"`
	Etag                 string                 `protobuf:"bytes,13,opt,name=etag,proto3" json:"etag,omitempty"`
	ChecksumSha256       string                 `protobuf:"bytes,14,opt,name=checksum_sha256,json=checksumSha256,proto3" json:"checksum_sha256,omitempty"` // lowercase sha256:<64 hex> measured while streaming
	Refusal              WeightsUploadRefusal   `protobuf:"varint,15,opt,name=refusal,proto3,enum=cozy.worker.v1.WeightsUploadRefusal" json:"refusal,omitempty"`
	SafeDetail           string                 `protobuf:"bytes,16,opt,name=safe_detail,json=safeDetail,proto3" json:"safe_detail,omitempty"`
	unknownFields        protoimpl.UnknownFields
	sizeCache            protoimpl.SizeCache
}

func (x *WeightsUploadResult) Reset() {
	*x = WeightsUploadResult{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[181]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *WeightsUploadResult) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*WeightsUploadResult) ProtoMessage() {}

func (x *WeightsUploadResult) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[181]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use WeightsUploadResult.ProtoReflect.Descriptor instead.
func (*WeightsUploadResult) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{181}
}

func (x *WeightsUploadResult) GetRecordOwnerEpoch() uint64 {
	if x != nil {
		return x.RecordOwnerEpoch
	}
	return 0
}

func (x *WeightsUploadResult) GetControlStreamEpoch() uint64 {
	if x != nil {
		return x.ControlStreamEpoch
	}
	return 0
}

func (x *WeightsUploadResult) GetWorkerBootId() string {
	if x != nil {
		return x.WorkerBootId
	}
	return ""
}

func (x *WeightsUploadResult) GetWeightsTransactionId() string {
	if x != nil {
		return x.WeightsTransactionId
	}
	return ""
}

func (x *WeightsUploadResult) GetWriterEpoch() uint64 {
	if x != nil {
		return x.WriterEpoch
	}
	return 0
}

func (x *WeightsUploadResult) GetObjectId() string {
	if x != nil {
		return x.ObjectId
	}
	return ""
}

func (x *WeightsUploadResult) GetOperationId() string {
	if x != nil {
		return x.OperationId
	}
	return ""
}

func (x *WeightsUploadResult) GetGrantRevision() uint64 {
	if x != nil {
		return x.GrantRevision
	}
	return 0
}

func (x *WeightsUploadResult) GetOutcome() WeightsUploadOutcome {
	if x != nil {
		return x.Outcome
	}
	return WeightsUploadOutcome_WEIGHTS_UPLOAD_OUTCOME_UNSPECIFIED
}

func (x *WeightsUploadResult) GetTransferredBytes() uint64 {
	if x != nil {
		return x.TransferredBytes
	}
	return 0
}

func (x *WeightsUploadResult) GetHttpStatus() uint32 {
	if x != nil {
		return x.HttpStatus
	}
	return 0
}

func (x *WeightsUploadResult) GetEtag() string {
	if x != nil {
		return x.Etag
	}
	return ""
}

func (x *WeightsUploadResult) GetChecksumSha256() string {
	if x != nil {
		return x.ChecksumSha256
	}
	return ""
}

func (x *WeightsUploadResult) GetRefusal() WeightsUploadRefusal {
	if x != nil {
		return x.Refusal
	}
	return WeightsUploadRefusal_WEIGHTS_UPLOAD_REFUSAL_UNSPECIFIED
}

func (x *WeightsUploadResult) GetSafeDetail() string {
	if x != nil {
		return x.SafeDetail
	}
	return ""
}

// Creator resolves and pins provider metadata. One selected file capability at a time reaches
// pod-supervisor; Runtime performs TensorFS's bounded profile/header proof before all payloads
// arrive, then converts only operations whose physical carriers are verified. URL is memory-only; the ledger
// stores only stable identity/revision and measured progress. A higher capability_revision may
// refresh access but cannot change member/ObjectRef/source selection. Broad account tokens are not
// representable.
type ModelSourceFileRequest struct {
	state                 protoimpl.MessageState `protogen:"open.v1"`
	RecordOwnerEpoch      uint64                 `protobuf:"varint,1,opt,name=record_owner_epoch,json=recordOwnerEpoch,proto3" json:"record_owner_epoch,omitempty"`
	ControlStreamEpoch    uint64                 `protobuf:"varint,2,opt,name=control_stream_epoch,json=controlStreamEpoch,proto3" json:"control_stream_epoch,omitempty"`
	WorkerBootId          string                 `protobuf:"bytes,3,opt,name=worker_boot_id,json=workerBootId,proto3" json:"worker_boot_id,omitempty"`
	OperationId           string                 `protobuf:"bytes,5,opt,name=operation_id,json=operationId,proto3" json:"operation_id,omitempty"`
	SourceSelectionDigest []byte                 `protobuf:"bytes,6,opt,name=source_selection_digest,json=sourceSelectionDigest,proto3" json:"source_selection_digest,omitempty"`
	Member                string                 `protobuf:"bytes,7,opt,name=member,proto3" json:"member,omitempty"`
	ObjectId              string                 `protobuf:"bytes,8,opt,name=object_id,json=objectId,proto3" json:"object_id,omitempty"` // lowercase sha256:<64 hex>
	Length                uint64                 `protobuf:"varint,9,opt,name=length,proto3" json:"length,omitempty"`
	Provider              ModelSourceProvider    `protobuf:"varint,10,opt,name=provider,proto3,enum=cozy.worker.v1.ModelSourceProvider" json:"provider,omitempty"`
	// Empty declares header metadata only and returns ACCEPTED without downloading. The
	// owner declares every roster header first, restores an acknowledged checkpoint, then
	// grants nonempty URLs only for the physical carriers TensorFS still needs.
	Url                string `protobuf:"bytes,11,opt,name=url,proto3" json:"url,omitempty"`                                             // memory-only access; never echoed in status/snapshot/loopback
	ExpiresAtUnix      uint64 `protobuf:"varint,12,opt,name=expires_at_unix,json=expiresAtUnix,proto3" json:"expires_at_unix,omitempty"` // zero only for an immutable public URL
	CapabilityRevision uint64 `protobuf:"varint,13,opt,name=capability_revision,json=capabilityRevision,proto3" json:"capability_revision,omitempty"`
	// Exact safetensors 8-byte LE length + header JSON, or full JSON for an index carrier.
	// One bounded header per file frame; never the model's entire header corpus in one frame.
	// <= MaxModelSourceHeaderBytes AND the complete frame <= MaxInlineControlBytes.
	Header        []byte `protobuf:"bytes,14,opt,name=header,proto3" json:"header,omitempty"`
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *ModelSourceFileRequest) Reset() {
	*x = ModelSourceFileRequest{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[182]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *ModelSourceFileRequest) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*ModelSourceFileRequest) ProtoMessage() {}

func (x *ModelSourceFileRequest) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[182]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use ModelSourceFileRequest.ProtoReflect.Descriptor instead.
func (*ModelSourceFileRequest) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{182}
}

func (x *ModelSourceFileRequest) GetRecordOwnerEpoch() uint64 {
	if x != nil {
		return x.RecordOwnerEpoch
	}
	return 0
}

func (x *ModelSourceFileRequest) GetControlStreamEpoch() uint64 {
	if x != nil {
		return x.ControlStreamEpoch
	}
	return 0
}

func (x *ModelSourceFileRequest) GetWorkerBootId() string {
	if x != nil {
		return x.WorkerBootId
	}
	return ""
}

func (x *ModelSourceFileRequest) GetOperationId() string {
	if x != nil {
		return x.OperationId
	}
	return ""
}

func (x *ModelSourceFileRequest) GetSourceSelectionDigest() []byte {
	if x != nil {
		return x.SourceSelectionDigest
	}
	return nil
}

func (x *ModelSourceFileRequest) GetMember() string {
	if x != nil {
		return x.Member
	}
	return ""
}

func (x *ModelSourceFileRequest) GetObjectId() string {
	if x != nil {
		return x.ObjectId
	}
	return ""
}

func (x *ModelSourceFileRequest) GetLength() uint64 {
	if x != nil {
		return x.Length
	}
	return 0
}

func (x *ModelSourceFileRequest) GetProvider() ModelSourceProvider {
	if x != nil {
		return x.Provider
	}
	return ModelSourceProvider_MODEL_SOURCE_PROVIDER_UNSPECIFIED
}

func (x *ModelSourceFileRequest) GetUrl() string {
	if x != nil {
		return x.Url
	}
	return ""
}

func (x *ModelSourceFileRequest) GetExpiresAtUnix() uint64 {
	if x != nil {
		return x.ExpiresAtUnix
	}
	return 0
}

func (x *ModelSourceFileRequest) GetCapabilityRevision() uint64 {
	if x != nil {
		return x.CapabilityRevision
	}
	return 0
}

func (x *ModelSourceFileRequest) GetHeader() []byte {
	if x != nil {
		return x.Header
	}
	return nil
}

type ModelSourceFileStatus struct {
	state                 protoimpl.MessageState `protogen:"open.v1"`
	RecordOwnerEpoch      uint64                 `protobuf:"varint,1,opt,name=record_owner_epoch,json=recordOwnerEpoch,proto3" json:"record_owner_epoch,omitempty"`
	ControlStreamEpoch    uint64                 `protobuf:"varint,2,opt,name=control_stream_epoch,json=controlStreamEpoch,proto3" json:"control_stream_epoch,omitempty"`
	WorkerBootId          string                 `protobuf:"bytes,3,opt,name=worker_boot_id,json=workerBootId,proto3" json:"worker_boot_id,omitempty"`
	OperationId           string                 `protobuf:"bytes,5,opt,name=operation_id,json=operationId,proto3" json:"operation_id,omitempty"`
	SourceSelectionDigest []byte                 `protobuf:"bytes,6,opt,name=source_selection_digest,json=sourceSelectionDigest,proto3" json:"source_selection_digest,omitempty"`
	Member                string                 `protobuf:"bytes,7,opt,name=member,proto3" json:"member,omitempty"`
	ObjectId              string                 `protobuf:"bytes,8,opt,name=object_id,json=objectId,proto3" json:"object_id,omitempty"`
	Length                uint64                 `protobuf:"varint,9,opt,name=length,proto3" json:"length,omitempty"`
	CapabilityRevision    uint64                 `protobuf:"varint,10,opt,name=capability_revision,json=capabilityRevision,proto3" json:"capability_revision,omitempty"`
	State                 ModelSourceFileState   `protobuf:"varint,11,opt,name=state,proto3,enum=cozy.worker.v1.ModelSourceFileState" json:"state,omitempty"`
	TransferredBytes      uint64                 `protobuf:"varint,12,opt,name=transferred_bytes,json=transferredBytes,proto3" json:"transferred_bytes,omitempty"`
	Attempts              uint32                 `protobuf:"varint,13,opt,name=attempts,proto3" json:"attempts,omitempty"`
	SafeCode              string                 `protobuf:"bytes,14,opt,name=safe_code,json=safeCode,proto3" json:"safe_code,omitempty"`
	SafeDetail            string                 `protobuf:"bytes,15,opt,name=safe_detail,json=safeDetail,proto3" json:"safe_detail,omitempty"`
	unknownFields         protoimpl.UnknownFields
	sizeCache             protoimpl.SizeCache
}

func (x *ModelSourceFileStatus) Reset() {
	*x = ModelSourceFileStatus{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[183]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *ModelSourceFileStatus) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*ModelSourceFileStatus) ProtoMessage() {}

func (x *ModelSourceFileStatus) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[183]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use ModelSourceFileStatus.ProtoReflect.Descriptor instead.
func (*ModelSourceFileStatus) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{183}
}

func (x *ModelSourceFileStatus) GetRecordOwnerEpoch() uint64 {
	if x != nil {
		return x.RecordOwnerEpoch
	}
	return 0
}

func (x *ModelSourceFileStatus) GetControlStreamEpoch() uint64 {
	if x != nil {
		return x.ControlStreamEpoch
	}
	return 0
}

func (x *ModelSourceFileStatus) GetWorkerBootId() string {
	if x != nil {
		return x.WorkerBootId
	}
	return ""
}

func (x *ModelSourceFileStatus) GetOperationId() string {
	if x != nil {
		return x.OperationId
	}
	return ""
}

func (x *ModelSourceFileStatus) GetSourceSelectionDigest() []byte {
	if x != nil {
		return x.SourceSelectionDigest
	}
	return nil
}

func (x *ModelSourceFileStatus) GetMember() string {
	if x != nil {
		return x.Member
	}
	return ""
}

func (x *ModelSourceFileStatus) GetObjectId() string {
	if x != nil {
		return x.ObjectId
	}
	return ""
}

func (x *ModelSourceFileStatus) GetLength() uint64 {
	if x != nil {
		return x.Length
	}
	return 0
}

func (x *ModelSourceFileStatus) GetCapabilityRevision() uint64 {
	if x != nil {
		return x.CapabilityRevision
	}
	return 0
}

func (x *ModelSourceFileStatus) GetState() ModelSourceFileState {
	if x != nil {
		return x.State
	}
	return ModelSourceFileState_MODEL_SOURCE_FILE_STATE_UNSPECIFIED
}

func (x *ModelSourceFileStatus) GetTransferredBytes() uint64 {
	if x != nil {
		return x.TransferredBytes
	}
	return 0
}

func (x *ModelSourceFileStatus) GetAttempts() uint32 {
	if x != nil {
		return x.Attempts
	}
	return 0
}

func (x *ModelSourceFileStatus) GetSafeCode() string {
	if x != nil {
		return x.SafeCode
	}
	return ""
}

func (x *ModelSourceFileStatus) GetSafeDetail() string {
	if x != nil {
		return x.SafeDetail
	}
	return ""
}

// The complete selected roster and its headers are fixed before conversion. Runtime advances
// every descriptor-reviewed profile over the currently verified physical carriers and returns
// INCOMPLETE until every target is available. Pinned source URI and declared license are
// credential-free provenance, not authorization. This replaces the pre-release full-file barrier;
// deploy the generated schema and all consumers together.
type ModelSourcePrepareRequest struct {
	state                 protoimpl.MessageState   `protogen:"open.v1"`
	RecordOwnerEpoch      uint64                   `protobuf:"varint,1,opt,name=record_owner_epoch,json=recordOwnerEpoch,proto3" json:"record_owner_epoch,omitempty"`
	ControlStreamEpoch    uint64                   `protobuf:"varint,2,opt,name=control_stream_epoch,json=controlStreamEpoch,proto3" json:"control_stream_epoch,omitempty"`
	WorkerBootId          string                   `protobuf:"bytes,3,opt,name=worker_boot_id,json=workerBootId,proto3" json:"worker_boot_id,omitempty"`
	OperationId           string                   `protobuf:"bytes,5,opt,name=operation_id,json=operationId,proto3" json:"operation_id,omitempty"`
	SourceSelectionDigest []byte                   `protobuf:"bytes,6,opt,name=source_selection_digest,json=sourceSelectionDigest,proto3" json:"source_selection_digest,omitempty"`
	Profiles              []*ModelSourceProfile    `protobuf:"bytes,7,rep,name=profiles,proto3" json:"profiles,omitempty"`                                      // sorted unique by slot
	SourceUri             string                   `protobuf:"bytes,8,opt,name=source_uri,json=sourceUri,proto3" json:"source_uri,omitempty"`                   // credential-free pinned provenance
	DeclaredLicense       string                   `protobuf:"bytes,9,opt,name=declared_license,json=declaredLicense,proto3" json:"declared_license,omitempty"` // bounded printable provenance; empty means undeclared
	Checkpoints           []*ModelSourceCheckpoint `protobuf:"bytes,10,rep,name=checkpoints,proto3" json:"checkpoints,omitempty"`                               // sorted unique acknowledged durable heads
	unknownFields         protoimpl.UnknownFields
	sizeCache             protoimpl.SizeCache
}

func (x *ModelSourcePrepareRequest) Reset() {
	*x = ModelSourcePrepareRequest{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[184]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *ModelSourcePrepareRequest) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*ModelSourcePrepareRequest) ProtoMessage() {}

func (x *ModelSourcePrepareRequest) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[184]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use ModelSourcePrepareRequest.ProtoReflect.Descriptor instead.
func (*ModelSourcePrepareRequest) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{184}
}

func (x *ModelSourcePrepareRequest) GetRecordOwnerEpoch() uint64 {
	if x != nil {
		return x.RecordOwnerEpoch
	}
	return 0
}

func (x *ModelSourcePrepareRequest) GetControlStreamEpoch() uint64 {
	if x != nil {
		return x.ControlStreamEpoch
	}
	return 0
}

func (x *ModelSourcePrepareRequest) GetWorkerBootId() string {
	if x != nil {
		return x.WorkerBootId
	}
	return ""
}

func (x *ModelSourcePrepareRequest) GetOperationId() string {
	if x != nil {
		return x.OperationId
	}
	return ""
}

func (x *ModelSourcePrepareRequest) GetSourceSelectionDigest() []byte {
	if x != nil {
		return x.SourceSelectionDigest
	}
	return nil
}

func (x *ModelSourcePrepareRequest) GetProfiles() []*ModelSourceProfile {
	if x != nil {
		return x.Profiles
	}
	return nil
}

func (x *ModelSourcePrepareRequest) GetSourceUri() string {
	if x != nil {
		return x.SourceUri
	}
	return ""
}

func (x *ModelSourcePrepareRequest) GetDeclaredLicense() string {
	if x != nil {
		return x.DeclaredLicense
	}
	return ""
}

func (x *ModelSourcePrepareRequest) GetCheckpoints() []*ModelSourceCheckpoint {
	if x != nil {
		return x.Checkpoints
	}
	return nil
}

type ModelSourcePrepared struct {
	state                 protoimpl.MessageState    `protogen:"open.v1"`
	RecordOwnerEpoch      uint64                    `protobuf:"varint,1,opt,name=record_owner_epoch,json=recordOwnerEpoch,proto3" json:"record_owner_epoch,omitempty"`
	ControlStreamEpoch    uint64                    `protobuf:"varint,2,opt,name=control_stream_epoch,json=controlStreamEpoch,proto3" json:"control_stream_epoch,omitempty"`
	WorkerBootId          string                    `protobuf:"bytes,3,opt,name=worker_boot_id,json=workerBootId,proto3" json:"worker_boot_id,omitempty"`
	OperationId           string                    `protobuf:"bytes,5,opt,name=operation_id,json=operationId,proto3" json:"operation_id,omitempty"`
	SourceSelectionDigest []byte                    `protobuf:"bytes,6,opt,name=source_selection_digest,json=sourceSelectionDigest,proto3" json:"source_selection_digest,omitempty"`
	Outcome               ModelSourcePrepareOutcome `protobuf:"varint,7,opt,name=outcome,proto3,enum=cozy.worker.v1.ModelSourcePrepareOutcome" json:"outcome,omitempty"`
	Sources               []*PreparedModelSource    `protobuf:"bytes,8,rep,name=sources,proto3" json:"sources,omitempty"` // sorted unique completed slots; exact set when complete
	SafeCode              string                    `protobuf:"bytes,9,opt,name=safe_code,json=safeCode,proto3" json:"safe_code,omitempty"`
	SafeDetail            string                    `protobuf:"bytes,10,opt,name=safe_detail,json=safeDetail,proto3" json:"safe_detail,omitempty"`
	Checkpoints           []*ModelSourceCheckpoint  `protobuf:"bytes,11,rep,name=checkpoints,proto3" json:"checkpoints,omitempty"` // sorted unique locally checkpointed progress
	unknownFields         protoimpl.UnknownFields
	sizeCache             protoimpl.SizeCache
}

func (x *ModelSourcePrepared) Reset() {
	*x = ModelSourcePrepared{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[185]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *ModelSourcePrepared) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*ModelSourcePrepared) ProtoMessage() {}

func (x *ModelSourcePrepared) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[185]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use ModelSourcePrepared.ProtoReflect.Descriptor instead.
func (*ModelSourcePrepared) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{185}
}

func (x *ModelSourcePrepared) GetRecordOwnerEpoch() uint64 {
	if x != nil {
		return x.RecordOwnerEpoch
	}
	return 0
}

func (x *ModelSourcePrepared) GetControlStreamEpoch() uint64 {
	if x != nil {
		return x.ControlStreamEpoch
	}
	return 0
}

func (x *ModelSourcePrepared) GetWorkerBootId() string {
	if x != nil {
		return x.WorkerBootId
	}
	return ""
}

func (x *ModelSourcePrepared) GetOperationId() string {
	if x != nil {
		return x.OperationId
	}
	return ""
}

func (x *ModelSourcePrepared) GetSourceSelectionDigest() []byte {
	if x != nil {
		return x.SourceSelectionDigest
	}
	return nil
}

func (x *ModelSourcePrepared) GetOutcome() ModelSourcePrepareOutcome {
	if x != nil {
		return x.Outcome
	}
	return ModelSourcePrepareOutcome_MODEL_SOURCE_PREPARE_OUTCOME_UNSPECIFIED
}

func (x *ModelSourcePrepared) GetSources() []*PreparedModelSource {
	if x != nil {
		return x.Sources
	}
	return nil
}

func (x *ModelSourcePrepared) GetSafeCode() string {
	if x != nil {
		return x.SafeCode
	}
	return ""
}

func (x *ModelSourcePrepared) GetSafeDetail() string {
	if x != nil {
		return x.SafeDetail
	}
	return ""
}

func (x *ModelSourcePrepared) GetCheckpoints() []*ModelSourceCheckpoint {
	if x != nil {
		return x.Checkpoints
	}
	return nil
}

// Durable progress/terminal observation for one exact uploaded file. Exact replay of an already
// verified prefix returns the same received byte count; changed identity under one operation
// refuses. A RECEIVING status carrying a safe code is a resumable stall: the owner replays the
// upload header to learn the resume offset.
type LocalPackageFileStatus struct {
	state              protoimpl.MessageState `protogen:"open.v1"`
	RecordOwnerEpoch   uint64                 `protobuf:"varint,1,opt,name=record_owner_epoch,json=recordOwnerEpoch,proto3" json:"record_owner_epoch,omitempty"`
	ControlStreamEpoch uint64                 `protobuf:"varint,2,opt,name=control_stream_epoch,json=controlStreamEpoch,proto3" json:"control_stream_epoch,omitempty"`
	WorkerBootId       string                 `protobuf:"bytes,3,opt,name=worker_boot_id,json=workerBootId,proto3" json:"worker_boot_id,omitempty"`
	OperationId        string                 `protobuf:"bytes,5,opt,name=operation_id,json=operationId,proto3" json:"operation_id,omitempty"`
	Digest             []byte                 `protobuf:"bytes,7,opt,name=digest,proto3" json:"digest,omitempty"`
	Filename           string                 `protobuf:"bytes,8,opt,name=filename,proto3" json:"filename,omitempty"`
	Length             uint64                 `protobuf:"varint,10,opt,name=length,proto3" json:"length,omitempty"`
	State              LocalPackageFileState  `protobuf:"varint,11,opt,name=state,proto3,enum=cozy.worker.v1.LocalPackageFileState" json:"state,omitempty"`
	ReceivedBytes      uint64                 `protobuf:"varint,12,opt,name=received_bytes,json=receivedBytes,proto3" json:"received_bytes,omitempty"`
	SafeCode           string                 `protobuf:"bytes,13,opt,name=safe_code,json=safeCode,proto3" json:"safe_code,omitempty"`
	SafeDetail         string                 `protobuf:"bytes,14,opt,name=safe_detail,json=safeDetail,proto3" json:"safe_detail,omitempty"`
	unknownFields      protoimpl.UnknownFields
	sizeCache          protoimpl.SizeCache
}

func (x *LocalPackageFileStatus) Reset() {
	*x = LocalPackageFileStatus{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[186]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *LocalPackageFileStatus) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*LocalPackageFileStatus) ProtoMessage() {}

func (x *LocalPackageFileStatus) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[186]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use LocalPackageFileStatus.ProtoReflect.Descriptor instead.
func (*LocalPackageFileStatus) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{186}
}

func (x *LocalPackageFileStatus) GetRecordOwnerEpoch() uint64 {
	if x != nil {
		return x.RecordOwnerEpoch
	}
	return 0
}

func (x *LocalPackageFileStatus) GetControlStreamEpoch() uint64 {
	if x != nil {
		return x.ControlStreamEpoch
	}
	return 0
}

func (x *LocalPackageFileStatus) GetWorkerBootId() string {
	if x != nil {
		return x.WorkerBootId
	}
	return ""
}

func (x *LocalPackageFileStatus) GetOperationId() string {
	if x != nil {
		return x.OperationId
	}
	return ""
}

func (x *LocalPackageFileStatus) GetDigest() []byte {
	if x != nil {
		return x.Digest
	}
	return nil
}

func (x *LocalPackageFileStatus) GetFilename() string {
	if x != nil {
		return x.Filename
	}
	return ""
}

func (x *LocalPackageFileStatus) GetLength() uint64 {
	if x != nil {
		return x.Length
	}
	return 0
}

func (x *LocalPackageFileStatus) GetState() LocalPackageFileState {
	if x != nil {
		return x.State
	}
	return LocalPackageFileState_LOCAL_PACKAGE_FILE_STATE_UNSPECIFIED
}

func (x *LocalPackageFileStatus) GetReceivedBytes() uint64 {
	if x != nil {
		return x.ReceivedBytes
	}
	return 0
}

func (x *LocalPackageFileStatus) GetSafeCode() string {
	if x != nil {
		return x.SafeCode
	}
	return ""
}

func (x *LocalPackageFileStatus) GetSafeDetail() string {
	if x != nil {
		return x.SafeDetail
	}
	return ""
}

// Durable owner->worker request. This is a typed protobuf frame, not a canonical document. The
// record owner journals the decision fields before send; Runtime journals the result fields
// before reply. Replay keys on the request tuple and compares the typed fields.
type WeightsFinalizeRequest struct {
	state                protoimpl.MessageState     `protogen:"open.v1"`
	RecordOwnerEpoch     uint64                     `protobuf:"varint,1,opt,name=record_owner_epoch,json=recordOwnerEpoch,proto3" json:"record_owner_epoch,omitempty"`
	ControlStreamEpoch   uint64                     `protobuf:"varint,2,opt,name=control_stream_epoch,json=controlStreamEpoch,proto3" json:"control_stream_epoch,omitempty"`
	WorkerBootId         string                     `protobuf:"bytes,3,opt,name=worker_boot_id,json=workerBootId,proto3" json:"worker_boot_id,omitempty"`
	RequestId            string                     `protobuf:"bytes,5,opt,name=request_id,json=requestId,proto3" json:"request_id,omitempty"`
	InvocationSpecDigest []byte                     `protobuf:"bytes,6,opt,name=invocation_spec_digest,json=invocationSpecDigest,proto3" json:"invocation_spec_digest,omitempty"`
	OutputSlot           string                     `protobuf:"bytes,7,opt,name=output_slot,json=outputSlot,proto3" json:"output_slot,omitempty"`
	Disposition          WeightsFinalizeDisposition `protobuf:"varint,8,opt,name=disposition,proto3,enum=cozy.worker.v1.WeightsFinalizeDisposition" json:"disposition,omitempty"`
	WeightsReceiptDigest []byte                     `protobuf:"bytes,9,opt,name=weights_receipt_digest,json=weightsReceiptDigest,proto3" json:"weights_receipt_digest,omitempty"`
	ScratchRootId        string                     `protobuf:"bytes,10,opt,name=scratch_root_id,json=scratchRootId,proto3" json:"scratch_root_id,omitempty"`
	OwnerAuthorityScope  string                     `protobuf:"bytes,11,opt,name=owner_authority_scope,json=ownerAuthorityScope,proto3" json:"owner_authority_scope,omitempty"`
	// Exact already-owned InvocationSpec/1 bytes, required and SHA-bound to field6.
	// Runtime re-validates canonical form and the declared model output even after its
	// ephemeral attempt history is gone. This is existing authority, never a new identity.
	// The complete request must remain within MaxInlineControlBytes.
	InvocationSpecCanonicalBytes []byte `protobuf:"bytes,12,opt,name=invocation_spec_canonical_bytes,json=invocationSpecCanonicalBytes,proto3" json:"invocation_spec_canonical_bytes,omitempty"`
	unknownFields                protoimpl.UnknownFields
	sizeCache                    protoimpl.SizeCache
}

func (x *WeightsFinalizeRequest) Reset() {
	*x = WeightsFinalizeRequest{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[187]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *WeightsFinalizeRequest) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*WeightsFinalizeRequest) ProtoMessage() {}

func (x *WeightsFinalizeRequest) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[187]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use WeightsFinalizeRequest.ProtoReflect.Descriptor instead.
func (*WeightsFinalizeRequest) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{187}
}

func (x *WeightsFinalizeRequest) GetRecordOwnerEpoch() uint64 {
	if x != nil {
		return x.RecordOwnerEpoch
	}
	return 0
}

func (x *WeightsFinalizeRequest) GetControlStreamEpoch() uint64 {
	if x != nil {
		return x.ControlStreamEpoch
	}
	return 0
}

func (x *WeightsFinalizeRequest) GetWorkerBootId() string {
	if x != nil {
		return x.WorkerBootId
	}
	return ""
}

func (x *WeightsFinalizeRequest) GetRequestId() string {
	if x != nil {
		return x.RequestId
	}
	return ""
}

func (x *WeightsFinalizeRequest) GetInvocationSpecDigest() []byte {
	if x != nil {
		return x.InvocationSpecDigest
	}
	return nil
}

func (x *WeightsFinalizeRequest) GetOutputSlot() string {
	if x != nil {
		return x.OutputSlot
	}
	return ""
}

func (x *WeightsFinalizeRequest) GetDisposition() WeightsFinalizeDisposition {
	if x != nil {
		return x.Disposition
	}
	return WeightsFinalizeDisposition_WEIGHTS_FINALIZE_DISPOSITION_UNSPECIFIED
}

func (x *WeightsFinalizeRequest) GetWeightsReceiptDigest() []byte {
	if x != nil {
		return x.WeightsReceiptDigest
	}
	return nil
}

func (x *WeightsFinalizeRequest) GetScratchRootId() string {
	if x != nil {
		return x.ScratchRootId
	}
	return ""
}

func (x *WeightsFinalizeRequest) GetOwnerAuthorityScope() string {
	if x != nil {
		return x.OwnerAuthorityScope
	}
	return ""
}

func (x *WeightsFinalizeRequest) GetInvocationSpecCanonicalBytes() []byte {
	if x != nil {
		return x.InvocationSpecCanonicalBytes
	}
	return nil
}

// Durable worker->owner result. A commit-first ABANDON_UNCOMMITTED race carries the released
// receipt as evidence; other outcomes may omit it. No extra acknowledgement exists.
type WeightsFinalizeResult struct {
	state                protoimpl.MessageState `protogen:"open.v1"`
	RecordOwnerEpoch     uint64                 `protobuf:"varint,1,opt,name=record_owner_epoch,json=recordOwnerEpoch,proto3" json:"record_owner_epoch,omitempty"`
	ControlStreamEpoch   uint64                 `protobuf:"varint,2,opt,name=control_stream_epoch,json=controlStreamEpoch,proto3" json:"control_stream_epoch,omitempty"`
	WorkerBootId         string                 `protobuf:"bytes,3,opt,name=worker_boot_id,json=workerBootId,proto3" json:"worker_boot_id,omitempty"`
	RequestId            string                 `protobuf:"bytes,5,opt,name=request_id,json=requestId,proto3" json:"request_id,omitempty"`
	InvocationSpecDigest []byte                 `protobuf:"bytes,6,opt,name=invocation_spec_digest,json=invocationSpecDigest,proto3" json:"invocation_spec_digest,omitempty"`
	OutputSlot           string                 `protobuf:"bytes,7,opt,name=output_slot,json=outputSlot,proto3" json:"output_slot,omitempty"`
	Outcome              WeightsFinalizeOutcome `protobuf:"varint,8,opt,name=outcome,proto3,enum=cozy.worker.v1.WeightsFinalizeOutcome" json:"outcome,omitempty"`
	WeightsReceipt       *WeightsReceiptRef     `protobuf:"bytes,9,opt,name=weights_receipt,json=weightsReceipt,proto3" json:"weights_receipt,omitempty"`
	OwnerAuthorityScope  string                 `protobuf:"bytes,10,opt,name=owner_authority_scope,json=ownerAuthorityScope,proto3" json:"owner_authority_scope,omitempty"`
	unknownFields        protoimpl.UnknownFields
	sizeCache            protoimpl.SizeCache
}

func (x *WeightsFinalizeResult) Reset() {
	*x = WeightsFinalizeResult{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[188]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *WeightsFinalizeResult) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*WeightsFinalizeResult) ProtoMessage() {}

func (x *WeightsFinalizeResult) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[188]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use WeightsFinalizeResult.ProtoReflect.Descriptor instead.
func (*WeightsFinalizeResult) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{188}
}

func (x *WeightsFinalizeResult) GetRecordOwnerEpoch() uint64 {
	if x != nil {
		return x.RecordOwnerEpoch
	}
	return 0
}

func (x *WeightsFinalizeResult) GetControlStreamEpoch() uint64 {
	if x != nil {
		return x.ControlStreamEpoch
	}
	return 0
}

func (x *WeightsFinalizeResult) GetWorkerBootId() string {
	if x != nil {
		return x.WorkerBootId
	}
	return ""
}

func (x *WeightsFinalizeResult) GetRequestId() string {
	if x != nil {
		return x.RequestId
	}
	return ""
}

func (x *WeightsFinalizeResult) GetInvocationSpecDigest() []byte {
	if x != nil {
		return x.InvocationSpecDigest
	}
	return nil
}

func (x *WeightsFinalizeResult) GetOutputSlot() string {
	if x != nil {
		return x.OutputSlot
	}
	return ""
}

func (x *WeightsFinalizeResult) GetOutcome() WeightsFinalizeOutcome {
	if x != nil {
		return x.Outcome
	}
	return WeightsFinalizeOutcome_WEIGHTS_FINALIZE_OUTCOME_UNSPECIFIED
}

func (x *WeightsFinalizeResult) GetWeightsReceipt() *WeightsReceiptRef {
	if x != nil {
		return x.WeightsReceipt
	}
	return nil
}

func (x *WeightsFinalizeResult) GetOwnerAuthorityScope() string {
	if x != nil {
		return x.OwnerAuthorityScope
	}
	return ""
}

type ResultEnvelope struct {
	state protoimpl.MessageState `protogen:"open.v1"`
	// lands with cr-010 (post-launch), keyed by
	// resolved_adapter_plan_id
	ResultSchemaDigest []byte                 `protobuf:"bytes,1,opt,name=result_schema_digest,json=resultSchemaDigest,proto3" json:"result_schema_digest,omitempty"`
	InlineResult       []byte                 `protobuf:"bytes,2,opt,name=inline_result,json=inlineResult,proto3" json:"inline_result,omitempty"` // canonical typed result bytes, <= 4 MiB; XOR with 3
	ResultBlob         *OutputEntry           `protobuf:"bytes,3,opt,name=result_blob,json=resultBlob,proto3" json:"result_blob,omitempty"`       // by immutable blob receipt when large; XOR with 2
	Adjustments        []*AdjustmentRow       `protobuf:"bytes,4,rep,name=adjustments,proto3" json:"adjustments,omitempty"`
	CheckpointRef      string                 `protobuf:"bytes,5,opt,name=checkpoint_ref,json=checkpointRef,proto3" json:"checkpoint_ref,omitempty"`
	RetainedModels     []*RetainedModelResult `protobuf:"bytes,7,rep,name=retained_models,json=retainedModels,proto3" json:"retained_models,omitempty"` // <=32, sorted by result_pointer
	unknownFields      protoimpl.UnknownFields
	sizeCache          protoimpl.SizeCache
}

func (x *ResultEnvelope) Reset() {
	*x = ResultEnvelope{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[189]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *ResultEnvelope) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*ResultEnvelope) ProtoMessage() {}

func (x *ResultEnvelope) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[189]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use ResultEnvelope.ProtoReflect.Descriptor instead.
func (*ResultEnvelope) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{189}
}

func (x *ResultEnvelope) GetResultSchemaDigest() []byte {
	if x != nil {
		return x.ResultSchemaDigest
	}
	return nil
}

func (x *ResultEnvelope) GetInlineResult() []byte {
	if x != nil {
		return x.InlineResult
	}
	return nil
}

func (x *ResultEnvelope) GetResultBlob() *OutputEntry {
	if x != nil {
		return x.ResultBlob
	}
	return nil
}

func (x *ResultEnvelope) GetAdjustments() []*AdjustmentRow {
	if x != nil {
		return x.Adjustments
	}
	return nil
}

func (x *ResultEnvelope) GetCheckpointRef() string {
	if x != nil {
		return x.CheckpointRef
	}
	return ""
}

func (x *ResultEnvelope) GetRetainedModels() []*RetainedModelResult {
	if x != nil {
		return x.RetainedModels
	}
	return nil
}

// A schema-declared ModelArtifact returned directly or borrowed from a child.
// This entry is inside the exact hashed terminal, never an unbound post-hoc lookup.
type RetainedModelResult struct {
	state                       protoimpl.MessageState   `protogen:"open.v1"`
	ResultPointer               string                   `protobuf:"bytes,1,opt,name=result_pointer,json=resultPointer,proto3" json:"result_pointer,omitempty"`                                               // RFC6901 JSON pointer, <=1024 UTF-8 bytes; empty means root
	ModelArtifactCanonicalBytes []byte                   `protobuf:"bytes,2,opt,name=model_artifact_canonical_bytes,json=modelArtifactCanonicalBytes,proto3" json:"model_artifact_canonical_bytes,omitempty"` // <=4096 bytes, exact closed ModelArtifact value
	Retention                   *DerivedRetentionRequest `protobuf:"bytes,3,opt,name=retention,proto3" json:"retention,omitempty"`                                                                            // actual producer transaction/receipt and root-held lease
	unknownFields               protoimpl.UnknownFields
	sizeCache                   protoimpl.SizeCache
}

func (x *RetainedModelResult) Reset() {
	*x = RetainedModelResult{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[190]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *RetainedModelResult) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*RetainedModelResult) ProtoMessage() {}

func (x *RetainedModelResult) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[190]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use RetainedModelResult.ProtoReflect.Descriptor instead.
func (*RetainedModelResult) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{190}
}

func (x *RetainedModelResult) GetResultPointer() string {
	if x != nil {
		return x.ResultPointer
	}
	return ""
}

func (x *RetainedModelResult) GetModelArtifactCanonicalBytes() []byte {
	if x != nil {
		return x.ModelArtifactCanonicalBytes
	}
	return nil
}

func (x *RetainedModelResult) GetRetention() *DerivedRetentionRequest {
	if x != nil {
		return x.Retention
	}
	return nil
}

type AdjustmentRow struct {
	state         protoimpl.MessageState `protogen:"open.v1"`
	Field         string                 `protobuf:"bytes,1,opt,name=field,proto3" json:"field,omitempty"`
	Requested     string                 `protobuf:"bytes,2,opt,name=requested,proto3" json:"requested,omitempty"`
	Applied       string                 `protobuf:"bytes,3,opt,name=applied,proto3" json:"applied,omitempty"`
	Reason        string                 `protobuf:"bytes,4,opt,name=reason,proto3" json:"reason,omitempty"` // adjusted | clamp | warn, plus bounded detail
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *AdjustmentRow) Reset() {
	*x = AdjustmentRow{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[191]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *AdjustmentRow) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*AdjustmentRow) ProtoMessage() {}

func (x *AdjustmentRow) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[191]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use AdjustmentRow.ProtoReflect.Descriptor instead.
func (*AdjustmentRow) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{191}
}

func (x *AdjustmentRow) GetField() string {
	if x != nil {
		return x.Field
	}
	return ""
}

func (x *AdjustmentRow) GetRequested() string {
	if x != nil {
		return x.Requested
	}
	return ""
}

func (x *AdjustmentRow) GetApplied() string {
	if x != nil {
		return x.Applied
	}
	return ""
}

func (x *AdjustmentRow) GetReason() string {
	if x != nil {
		return x.Reason
	}
	return ""
}

type OutcomeCause struct {
	state         protoimpl.MessageState `protogen:"open.v1"`
	Code          CauseCode              `protobuf:"varint,1,opt,name=code,proto3,enum=cozy.worker.v1.CauseCode" json:"code,omitempty"`
	Origin        CauseOrigin            `protobuf:"varint,2,opt,name=origin,proto3,enum=cozy.worker.v1.CauseOrigin" json:"origin,omitempty"`
	Detail        string                 `protobuf:"bytes,3,opt,name=detail,proto3" json:"detail,omitempty"` // bounded <= 1024 bytes, sanitized
	Shortfall     *ResourceShortfall     `protobuf:"bytes,4,opt,name=shortfall,proto3" json:"shortfall,omitempty"`
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *OutcomeCause) Reset() {
	*x = OutcomeCause{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[192]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *OutcomeCause) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*OutcomeCause) ProtoMessage() {}

func (x *OutcomeCause) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[192]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use OutcomeCause.ProtoReflect.Descriptor instead.
func (*OutcomeCause) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{192}
}

func (x *OutcomeCause) GetCode() CauseCode {
	if x != nil {
		return x.Code
	}
	return CauseCode_CAUSE_CODE_UNSPECIFIED
}

func (x *OutcomeCause) GetOrigin() CauseOrigin {
	if x != nil {
		return x.Origin
	}
	return CauseOrigin_CAUSE_ORIGIN_UNSPECIFIED
}

func (x *OutcomeCause) GetDetail() string {
	if x != nil {
		return x.Detail
	}
	return ""
}

func (x *OutcomeCause) GetShortfall() *ResourceShortfall {
	if x != nil {
		return x.Shortfall
	}
	return nil
}

type ResourceShortfall struct {
	state          protoimpl.MessageState `protogen:"open.v1"`
	Resource       string                 `protobuf:"bytes,1,opt,name=resource,proto3" json:"resource,omitempty"` // device_memory | host_ram | pinned | disk | ...
	Scope          string                 `protobuf:"bytes,2,opt,name=scope,proto3" json:"scope,omitempty"`
	NeededBytes    uint64                 `protobuf:"varint,3,opt,name=needed_bytes,json=neededBytes,proto3" json:"needed_bytes,omitempty"`
	AvailableBytes uint64                 `protobuf:"varint,4,opt,name=available_bytes,json=availableBytes,proto3" json:"available_bytes,omitempty"`
	RequestShape   string                 `protobuf:"bytes,5,opt,name=request_shape,json=requestShape,proto3" json:"request_shape,omitempty"`
	EvidenceClass  string                 `protobuf:"bytes,6,opt,name=evidence_class,json=evidenceClass,proto3" json:"evidence_class,omitempty"` // measured | fitted | prior
	unknownFields  protoimpl.UnknownFields
	sizeCache      protoimpl.SizeCache
}

func (x *ResourceShortfall) Reset() {
	*x = ResourceShortfall{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[193]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *ResourceShortfall) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*ResourceShortfall) ProtoMessage() {}

func (x *ResourceShortfall) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[193]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use ResourceShortfall.ProtoReflect.Descriptor instead.
func (*ResourceShortfall) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{193}
}

func (x *ResourceShortfall) GetResource() string {
	if x != nil {
		return x.Resource
	}
	return ""
}

func (x *ResourceShortfall) GetScope() string {
	if x != nil {
		return x.Scope
	}
	return ""
}

func (x *ResourceShortfall) GetNeededBytes() uint64 {
	if x != nil {
		return x.NeededBytes
	}
	return 0
}

func (x *ResourceShortfall) GetAvailableBytes() uint64 {
	if x != nil {
		return x.AvailableBytes
	}
	return 0
}

func (x *ResourceShortfall) GetRequestShape() string {
	if x != nil {
		return x.RequestShape
	}
	return ""
}

func (x *ResourceShortfall) GetEvidenceClass() string {
	if x != nil {
		return x.EvidenceClass
	}
	return ""
}

type AttemptOutcomeAck struct {
	state                protoimpl.MessageState `protogen:"open.v1"`
	RecordOwnerEpoch     uint64                 `protobuf:"varint,1,opt,name=record_owner_epoch,json=recordOwnerEpoch,proto3" json:"record_owner_epoch,omitempty"`
	ControlStreamEpoch   uint64                 `protobuf:"varint,2,opt,name=control_stream_epoch,json=controlStreamEpoch,proto3" json:"control_stream_epoch,omitempty"`
	WorkerBootId         string                 `protobuf:"bytes,3,opt,name=worker_boot_id,json=workerBootId,proto3" json:"worker_boot_id,omitempty"`
	RequestId            string                 `protobuf:"bytes,5,opt,name=request_id,json=requestId,proto3" json:"request_id,omitempty"`
	AttemptOrdinal       uint64                 `protobuf:"varint,6,opt,name=attempt_ordinal,json=attemptOrdinal,proto3" json:"attempt_ordinal,omitempty"`
	InvocationSpecDigest []byte                 `protobuf:"bytes,7,opt,name=invocation_spec_digest,json=invocationSpecDigest,proto3" json:"invocation_spec_digest,omitempty"`
	OutcomeId            string                 `protobuf:"bytes,8,opt,name=outcome_id,json=outcomeId,proto3" json:"outcome_id,omitempty"`             // echo; a mismatched ack is NOT an ack — replay continues
	OutcomeDigest        []byte                 `protobuf:"bytes,9,opt,name=outcome_digest,json=outcomeDigest,proto3" json:"outcome_digest,omitempty"` // echo; compared, never recomputed
	// Close this attempt while retaining its request-scoped writer identity,
	// epochs and artifact custody for pause/retry. Does not authorize further execution.
	// The owner sends false only after permanent abandonment or required final custody
	// and disposition have settled. Native roots are still controlled by their explicit
	// finalization/source-release operations; this flag never adopts or deletes bytes.
	RetainWork    bool `protobuf:"varint,10,opt,name=retain_work,json=retainWork,proto3" json:"retain_work,omitempty"`
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *AttemptOutcomeAck) Reset() {
	*x = AttemptOutcomeAck{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[194]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *AttemptOutcomeAck) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*AttemptOutcomeAck) ProtoMessage() {}

func (x *AttemptOutcomeAck) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[194]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use AttemptOutcomeAck.ProtoReflect.Descriptor instead.
func (*AttemptOutcomeAck) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{194}
}

func (x *AttemptOutcomeAck) GetRecordOwnerEpoch() uint64 {
	if x != nil {
		return x.RecordOwnerEpoch
	}
	return 0
}

func (x *AttemptOutcomeAck) GetControlStreamEpoch() uint64 {
	if x != nil {
		return x.ControlStreamEpoch
	}
	return 0
}

func (x *AttemptOutcomeAck) GetWorkerBootId() string {
	if x != nil {
		return x.WorkerBootId
	}
	return ""
}

func (x *AttemptOutcomeAck) GetRequestId() string {
	if x != nil {
		return x.RequestId
	}
	return ""
}

func (x *AttemptOutcomeAck) GetAttemptOrdinal() uint64 {
	if x != nil {
		return x.AttemptOrdinal
	}
	return 0
}

func (x *AttemptOutcomeAck) GetInvocationSpecDigest() []byte {
	if x != nil {
		return x.InvocationSpecDigest
	}
	return nil
}

func (x *AttemptOutcomeAck) GetOutcomeId() string {
	if x != nil {
		return x.OutcomeId
	}
	return ""
}

func (x *AttemptOutcomeAck) GetOutcomeDigest() []byte {
	if x != nil {
		return x.OutcomeDigest
	}
	return nil
}

func (x *AttemptOutcomeAck) GetRetainWork() bool {
	if x != nil {
		return x.RetainWork
	}
	return false
}

type JobCheckpointRequest struct {
	state              protoimpl.MessageState `protogen:"open.v1"`
	RecordOwnerEpoch   uint64                 `protobuf:"varint,1,opt,name=record_owner_epoch,json=recordOwnerEpoch,proto3" json:"record_owner_epoch,omitempty"`
	ControlStreamEpoch uint64                 `protobuf:"varint,2,opt,name=control_stream_epoch,json=controlStreamEpoch,proto3" json:"control_stream_epoch,omitempty"`
	WorkerBootId       string                 `protobuf:"bytes,3,opt,name=worker_boot_id,json=workerBootId,proto3" json:"worker_boot_id,omitempty"`
	RequestId          string                 `protobuf:"bytes,5,opt,name=request_id,json=requestId,proto3" json:"request_id,omitempty"`
	AttemptOrdinal     uint64                 `protobuf:"varint,6,opt,name=attempt_ordinal,json=attemptOrdinal,proto3" json:"attempt_ordinal,omitempty"`
	OperationKey       string                 `protobuf:"bytes,7,opt,name=operation_key,json=operationKey,proto3" json:"operation_key,omitempty"`    // worker-minted, attempt-scoped idempotency key
	LogicalKey         string                 `protobuf:"bytes,8,opt,name=logical_key,json=logicalKey,proto3" json:"logical_key,omitempty"`          // author-stable checkpoint slot
	ContentDigest      []byte                 `protobuf:"bytes,9,opt,name=content_digest,json=contentDigest,proto3" json:"content_digest,omitempty"` // class (b): the checkpoint artifact's bytes
	Seq                uint64                 `protobuf:"varint,10,opt,name=seq,proto3" json:"seq,omitempty"`                                        // ordering only, never identity
	Artifact           *OutputEntry           `protobuf:"bytes,11,opt,name=artifact,proto3" json:"artifact,omitempty"`
	unknownFields      protoimpl.UnknownFields
	sizeCache          protoimpl.SizeCache
}

func (x *JobCheckpointRequest) Reset() {
	*x = JobCheckpointRequest{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[195]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *JobCheckpointRequest) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*JobCheckpointRequest) ProtoMessage() {}

func (x *JobCheckpointRequest) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[195]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use JobCheckpointRequest.ProtoReflect.Descriptor instead.
func (*JobCheckpointRequest) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{195}
}

func (x *JobCheckpointRequest) GetRecordOwnerEpoch() uint64 {
	if x != nil {
		return x.RecordOwnerEpoch
	}
	return 0
}

func (x *JobCheckpointRequest) GetControlStreamEpoch() uint64 {
	if x != nil {
		return x.ControlStreamEpoch
	}
	return 0
}

func (x *JobCheckpointRequest) GetWorkerBootId() string {
	if x != nil {
		return x.WorkerBootId
	}
	return ""
}

func (x *JobCheckpointRequest) GetRequestId() string {
	if x != nil {
		return x.RequestId
	}
	return ""
}

func (x *JobCheckpointRequest) GetAttemptOrdinal() uint64 {
	if x != nil {
		return x.AttemptOrdinal
	}
	return 0
}

func (x *JobCheckpointRequest) GetOperationKey() string {
	if x != nil {
		return x.OperationKey
	}
	return ""
}

func (x *JobCheckpointRequest) GetLogicalKey() string {
	if x != nil {
		return x.LogicalKey
	}
	return ""
}

func (x *JobCheckpointRequest) GetContentDigest() []byte {
	if x != nil {
		return x.ContentDigest
	}
	return nil
}

func (x *JobCheckpointRequest) GetSeq() uint64 {
	if x != nil {
		return x.Seq
	}
	return 0
}

func (x *JobCheckpointRequest) GetArtifact() *OutputEntry {
	if x != nil {
		return x.Artifact
	}
	return nil
}

type ProgressOpen struct {
	state              protoimpl.MessageState `protogen:"open.v1"`
	RecordOwnerEpoch   uint64                 `protobuf:"varint,1,opt,name=record_owner_epoch,json=recordOwnerEpoch,proto3" json:"record_owner_epoch,omitempty"`
	ControlStreamEpoch uint64                 `protobuf:"varint,2,opt,name=control_stream_epoch,json=controlStreamEpoch,proto3" json:"control_stream_epoch,omitempty"` // binds the watch to the fenced control stream
	WorkerBootId       string                 `protobuf:"bytes,3,opt,name=worker_boot_id,json=workerBootId,proto3" json:"worker_boot_id,omitempty"`
	RequestId          string                 `protobuf:"bytes,5,opt,name=request_id,json=requestId,proto3" json:"request_id,omitempty"` // empty = all attempts on this stream
	unknownFields      protoimpl.UnknownFields
	sizeCache          protoimpl.SizeCache
}

func (x *ProgressOpen) Reset() {
	*x = ProgressOpen{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[196]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *ProgressOpen) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*ProgressOpen) ProtoMessage() {}

func (x *ProgressOpen) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[196]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use ProgressOpen.ProtoReflect.Descriptor instead.
func (*ProgressOpen) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{196}
}

func (x *ProgressOpen) GetRecordOwnerEpoch() uint64 {
	if x != nil {
		return x.RecordOwnerEpoch
	}
	return 0
}

func (x *ProgressOpen) GetControlStreamEpoch() uint64 {
	if x != nil {
		return x.ControlStreamEpoch
	}
	return 0
}

func (x *ProgressOpen) GetWorkerBootId() string {
	if x != nil {
		return x.WorkerBootId
	}
	return ""
}

func (x *ProgressOpen) GetRequestId() string {
	if x != nil {
		return x.RequestId
	}
	return ""
}

type AttemptProgress struct {
	state              protoimpl.MessageState `protogen:"open.v1"`
	RecordOwnerEpoch   uint64                 `protobuf:"varint,1,opt,name=record_owner_epoch,json=recordOwnerEpoch,proto3" json:"record_owner_epoch,omitempty"`
	ControlStreamEpoch uint64                 `protobuf:"varint,2,opt,name=control_stream_epoch,json=controlStreamEpoch,proto3" json:"control_stream_epoch,omitempty"`
	WorkerBootId       string                 `protobuf:"bytes,3,opt,name=worker_boot_id,json=workerBootId,proto3" json:"worker_boot_id,omitempty"`
	RequestId          string                 `protobuf:"bytes,5,opt,name=request_id,json=requestId,proto3" json:"request_id,omitempty"`
	AttemptOrdinal     uint64                 `protobuf:"varint,6,opt,name=attempt_ordinal,json=attemptOrdinal,proto3" json:"attempt_ordinal,omitempty"`
	Seq                uint64                 `protobuf:"varint,7,opt,name=seq,proto3" json:"seq,omitempty"` // starts at 1, strictly increasing per (request_id,
	// attempt_ordinal); gaps are VISIBLE and the only loss
	// allowed
	ContentType   string `protobuf:"bytes,8,opt,name=content_type,json=contentType,proto3" json:"content_type,omitempty"`
	Data          []byte `protobuf:"bytes,9,opt,name=data,proto3" json:"data,omitempty"`                                   // bounded <= 65536 bytes; NEVER an outcome
	PlacementId   string `protobuf:"bytes,10,opt,name=placement_id,json=placementId,proto3" json:"placement_id,omitempty"` // routing
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *AttemptProgress) Reset() {
	*x = AttemptProgress{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[197]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *AttemptProgress) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*AttemptProgress) ProtoMessage() {}

func (x *AttemptProgress) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[197]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use AttemptProgress.ProtoReflect.Descriptor instead.
func (*AttemptProgress) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{197}
}

func (x *AttemptProgress) GetRecordOwnerEpoch() uint64 {
	if x != nil {
		return x.RecordOwnerEpoch
	}
	return 0
}

func (x *AttemptProgress) GetControlStreamEpoch() uint64 {
	if x != nil {
		return x.ControlStreamEpoch
	}
	return 0
}

func (x *AttemptProgress) GetWorkerBootId() string {
	if x != nil {
		return x.WorkerBootId
	}
	return ""
}

func (x *AttemptProgress) GetRequestId() string {
	if x != nil {
		return x.RequestId
	}
	return ""
}

func (x *AttemptProgress) GetAttemptOrdinal() uint64 {
	if x != nil {
		return x.AttemptOrdinal
	}
	return 0
}

func (x *AttemptProgress) GetSeq() uint64 {
	if x != nil {
		return x.Seq
	}
	return 0
}

func (x *AttemptProgress) GetContentType() string {
	if x != nil {
		return x.ContentType
	}
	return ""
}

func (x *AttemptProgress) GetData() []byte {
	if x != nil {
		return x.Data
	}
	return nil
}

func (x *AttemptProgress) GetPlacementId() string {
	if x != nil {
		return x.PlacementId
	}
	return ""
}

// DOCUMENT SHAPE (not a wire message): the subject of `invocation_spec_digest`. Canonical form
// `cozy.worker.v1.InvocationSpec/1`. No human model/adapter ref is spellable. Readers ignore
// unknown keys; a result-changing key is sent only to peers advertising its capability.
// placement_id is DELIBERATELY absent: the same invocation is the same work wherever it routes.
type InvocationSpec struct {
	state          protoimpl.MessageState `protogen:"open.v1"`
	PayloadDigest  string                 `protobuf:"bytes,4,opt,name=payload_digest,json=payloadDigest,proto3" json:"payload_digest,omitempty"`       // class (a): the canonical typed-argument document
	Inputs         []*InputBinding        `protobuf:"bytes,5,rep,name=inputs,proto3" json:"inputs,omitempty"`                                          // ORDERED input identities — INSIDE the digest
	Outputs        []*OutputBinding       `protobuf:"bytes,6,rep,name=outputs,proto3" json:"outputs,omitempty"`                                        // output ids/kinds/limits — INSIDE the digest
	DeadlineUnixMs uint64                 `protobuf:"varint,7,opt,name=deadline_unix_ms,json=deadlineUnixMs,proto3" json:"deadline_unix_ms,omitempty"` // absolute attempt deadline; worker-enforced; 0 = none
	// Types that are valid to be assigned to Spec:
	//
	//	*InvocationSpec_Serving
	//	*InvocationSpec_Job
	Spec    isInvocationSpec_Spec `protobuf_oneof:"spec"`
	Capture *ActivationCapture    `protobuf:"bytes,10,opt,name=capture,proto3" json:"capture,omitempty"` // immutable requested capture semantics
	// The execution-path override (cr-125). The attention kernel this attempt
	// pins, by `attention.BY_NAME`; empty leaves the runtime's own selection alone. INSIDE
	// the digest: it changes what the attempt computes. A pin the worker cannot honour is a
	// typed refusal, never a quiet fallback, so a non-empty value here is a fact about what ran.
	AttentionKernel string `protobuf:"bytes,11,opt,name=attention_kernel,json=attentionKernel,proto3" json:"attention_kernel,omitempty"`
	InstallationId  string `protobuf:"bytes,12,opt,name=installation_id,json=installationId,proto3" json:"installation_id,omitempty"` // routing/lifetime only; never an operation memoization input
	unknownFields   protoimpl.UnknownFields
	sizeCache       protoimpl.SizeCache
}

func (x *InvocationSpec) Reset() {
	*x = InvocationSpec{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[198]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *InvocationSpec) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*InvocationSpec) ProtoMessage() {}

func (x *InvocationSpec) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[198]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use InvocationSpec.ProtoReflect.Descriptor instead.
func (*InvocationSpec) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{198}
}

func (x *InvocationSpec) GetPayloadDigest() string {
	if x != nil {
		return x.PayloadDigest
	}
	return ""
}

func (x *InvocationSpec) GetInputs() []*InputBinding {
	if x != nil {
		return x.Inputs
	}
	return nil
}

func (x *InvocationSpec) GetOutputs() []*OutputBinding {
	if x != nil {
		return x.Outputs
	}
	return nil
}

func (x *InvocationSpec) GetDeadlineUnixMs() uint64 {
	if x != nil {
		return x.DeadlineUnixMs
	}
	return 0
}

func (x *InvocationSpec) GetSpec() isInvocationSpec_Spec {
	if x != nil {
		return x.Spec
	}
	return nil
}

func (x *InvocationSpec) GetServing() *ServingInvocationSpec {
	if x != nil {
		if x, ok := x.Spec.(*InvocationSpec_Serving); ok {
			return x.Serving
		}
	}
	return nil
}

func (x *InvocationSpec) GetJob() *JobInvocationSpec {
	if x != nil {
		if x, ok := x.Spec.(*InvocationSpec_Job); ok {
			return x.Job
		}
	}
	return nil
}

func (x *InvocationSpec) GetCapture() *ActivationCapture {
	if x != nil {
		return x.Capture
	}
	return nil
}

func (x *InvocationSpec) GetAttentionKernel() string {
	if x != nil {
		return x.AttentionKernel
	}
	return ""
}

func (x *InvocationSpec) GetInstallationId() string {
	if x != nil {
		return x.InstallationId
	}
	return ""
}

type isInvocationSpec_Spec interface {
	isInvocationSpec_Spec()
}

type InvocationSpec_Serving struct {
	Serving *ServingInvocationSpec `protobuf:"bytes,8,opt,name=serving,proto3,oneof"`
}

type InvocationSpec_Job struct {
	Job *JobInvocationSpec `protobuf:"bytes,9,opt,name=job,proto3,oneof"`
}

func (*InvocationSpec_Serving) isInvocationSpec_Spec() {}

func (*InvocationSpec_Job) isInvocationSpec_Spec() {}

type InputBinding struct {
	state         protoimpl.MessageState `protogen:"open.v1"`
	InputId       string                 `protobuf:"bytes,1,opt,name=input_id,json=inputId,proto3" json:"input_id,omitempty"` // the payload field path, list index included
	Digest        string                 `protobuf:"bytes,2,opt,name=digest,proto3" json:"digest,omitempty"`                  // class (b), sha256:<hex>: the input asset's content
	Length        uint64                 `protobuf:"varint,3,opt,name=length,proto3" json:"length,omitempty"`
	KindMime      string                 `protobuf:"bytes,4,opt,name=kind_mime,json=kindMime,proto3" json:"kind_mime,omitempty"`
	Order         uint32                 `protobuf:"varint,5,opt,name=order,proto3" json:"order,omitempty"`
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *InputBinding) Reset() {
	*x = InputBinding{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[199]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *InputBinding) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*InputBinding) ProtoMessage() {}

func (x *InputBinding) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[199]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use InputBinding.ProtoReflect.Descriptor instead.
func (*InputBinding) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{199}
}

func (x *InputBinding) GetInputId() string {
	if x != nil {
		return x.InputId
	}
	return ""
}

func (x *InputBinding) GetDigest() string {
	if x != nil {
		return x.Digest
	}
	return ""
}

func (x *InputBinding) GetLength() uint64 {
	if x != nil {
		return x.Length
	}
	return 0
}

func (x *InputBinding) GetKindMime() string {
	if x != nil {
		return x.KindMime
	}
	return ""
}

func (x *InputBinding) GetOrder() uint32 {
	if x != nil {
		return x.Order
	}
	return 0
}

type OutputBinding struct {
	state         protoimpl.MessageState `protogen:"open.v1"`
	OutputId      string                 `protobuf:"bytes,1,opt,name=output_id,json=outputId,proto3" json:"output_id,omitempty"`
	MimeType      string                 `protobuf:"bytes,2,opt,name=mime_type,json=mimeType,proto3" json:"mime_type,omitempty"`
	MaxBytes      uint64                 `protobuf:"varint,3,opt,name=max_bytes,json=maxBytes,proto3" json:"max_bytes,omitempty"`
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *OutputBinding) Reset() {
	*x = OutputBinding{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[200]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *OutputBinding) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*OutputBinding) ProtoMessage() {}

func (x *OutputBinding) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[200]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use OutputBinding.ProtoReflect.Descriptor instead.
func (*OutputBinding) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{200}
}

func (x *OutputBinding) GetOutputId() string {
	if x != nil {
		return x.OutputId
	}
	return ""
}

func (x *OutputBinding) GetMimeType() string {
	if x != nil {
		return x.MimeType
	}
	return ""
}

func (x *OutputBinding) GetMaxBytes() uint64 {
	if x != nil {
		return x.MaxBytes
	}
	return 0
}

type ServingInvocationSpec struct {
	state                   protoimpl.MessageState `protogen:"open.v1"`
	EntrypointBindingDigest string                 `protobuf:"bytes,1,opt,name=entrypoint_binding_digest,json=entrypointBindingDigest,proto3" json:"entrypoint_binding_digest,omitempty"` // class (a): selected entrypoint binding identity
	AttemptBindingId        string                 `protobuf:"bytes,2,opt,name=attempt_binding_id,json=attemptBindingId,proto3" json:"attempt_binding_id,omitempty"`                      // equal to field 1 when no adapters
	// REQUIRED: exact activated Placement.bindings_digest over canonical
	// {entrypoints,models}. Pins the selected model manifests as well as the entrypoint.
	// This is per placement, not the digest of unrelated co-fitting placements.
	BindingsDigest string `protobuf:"bytes,3,opt,name=bindings_digest,json=bindingsDigest,proto3" json:"bindings_digest,omitempty"`
	unknownFields  protoimpl.UnknownFields
	sizeCache      protoimpl.SizeCache
}

func (x *ServingInvocationSpec) Reset() {
	*x = ServingInvocationSpec{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[201]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *ServingInvocationSpec) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*ServingInvocationSpec) ProtoMessage() {}

func (x *ServingInvocationSpec) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[201]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use ServingInvocationSpec.ProtoReflect.Descriptor instead.
func (*ServingInvocationSpec) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{201}
}

func (x *ServingInvocationSpec) GetEntrypointBindingDigest() string {
	if x != nil {
		return x.EntrypointBindingDigest
	}
	return ""
}

func (x *ServingInvocationSpec) GetAttemptBindingId() string {
	if x != nil {
		return x.AttemptBindingId
	}
	return ""
}

func (x *ServingInvocationSpec) GetBindingsDigest() string {
	if x != nil {
		return x.BindingsDigest
	}
	return ""
}

type JobInvocationSpec struct {
	state               protoimpl.MessageState `protogen:"open.v1"`
	JobDescriptorId     string                 `protobuf:"bytes,2,opt,name=job_descriptor_id,json=jobDescriptorId,proto3" json:"job_descriptor_id,omitempty"`
	PublicationContract *PublicationContract   `protobuf:"bytes,3,opt,name=publication_contract,json=publicationContract,proto3" json:"publication_contract,omitempty"`
	InstallationId      string                 `protobuf:"bytes,4,opt,name=installation_id,json=installationId,proto3" json:"installation_id,omitempty"`
	unknownFields       protoimpl.UnknownFields
	sizeCache           protoimpl.SizeCache
}

func (x *JobInvocationSpec) Reset() {
	*x = JobInvocationSpec{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[202]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *JobInvocationSpec) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*JobInvocationSpec) ProtoMessage() {}

func (x *JobInvocationSpec) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[202]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use JobInvocationSpec.ProtoReflect.Descriptor instead.
func (*JobInvocationSpec) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{202}
}

func (x *JobInvocationSpec) GetJobDescriptorId() string {
	if x != nil {
		return x.JobDescriptorId
	}
	return ""
}

func (x *JobInvocationSpec) GetPublicationContract() *PublicationContract {
	if x != nil {
		return x.PublicationContract
	}
	return nil
}

func (x *JobInvocationSpec) GetInstallationId() string {
	if x != nil {
		return x.InstallationId
	}
	return ""
}

// Refreshable ACCESS, strictly PER-ATTEMPT: urls, token, expiry — and the invocation it serves.
// Nothing else. Pre-attempt placement acquisition does not ride this per-attempt grant.
type DeliveryGrant struct {
	state                protoimpl.MessageState    `protogen:"open.v1"`
	InvocationSpecDigest []byte                    `protobuf:"bytes,1,opt,name=invocation_spec_digest,json=invocationSpecDigest,proto3" json:"invocation_spec_digest,omitempty"` // the subject this grant serves; revalidated on refresh
	Credential           *DeliveryAccessCredential `protobuf:"bytes,2,opt,name=credential,proto3" json:"credential,omitempty"`                                                   // per-attempt scoped capability token
	FileBaseUrl          string                    `protobuf:"bytes,3,opt,name=file_base_url,json=fileBaseUrl,proto3" json:"file_base_url,omitempty"`
	ExpiresAtUnix        uint64                    `protobuf:"varint,4,opt,name=expires_at_unix,json=expiresAtUnix,proto3" json:"expires_at_unix,omitempty"`
	Inputs               []*InputAccess            `protobuf:"bytes,5,rep,name=inputs,proto3" json:"inputs,omitempty"`   // sorted by input_id; ids must match the spec's set
	Outputs              []*OutputAccess           `protobuf:"bytes,6,rep,name=outputs,proto3" json:"outputs,omitempty"` // sorted by output_id
	unknownFields        protoimpl.UnknownFields
	sizeCache            protoimpl.SizeCache
}

func (x *DeliveryGrant) Reset() {
	*x = DeliveryGrant{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[203]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *DeliveryGrant) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*DeliveryGrant) ProtoMessage() {}

func (x *DeliveryGrant) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[203]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use DeliveryGrant.ProtoReflect.Descriptor instead.
func (*DeliveryGrant) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{203}
}

func (x *DeliveryGrant) GetInvocationSpecDigest() []byte {
	if x != nil {
		return x.InvocationSpecDigest
	}
	return nil
}

func (x *DeliveryGrant) GetCredential() *DeliveryAccessCredential {
	if x != nil {
		return x.Credential
	}
	return nil
}

func (x *DeliveryGrant) GetFileBaseUrl() string {
	if x != nil {
		return x.FileBaseUrl
	}
	return ""
}

func (x *DeliveryGrant) GetExpiresAtUnix() uint64 {
	if x != nil {
		return x.ExpiresAtUnix
	}
	return 0
}

func (x *DeliveryGrant) GetInputs() []*InputAccess {
	if x != nil {
		return x.Inputs
	}
	return nil
}

func (x *DeliveryGrant) GetOutputs() []*OutputAccess {
	if x != nil {
		return x.Outputs
	}
	return nil
}

// Access provenance only; digest and length remain exclusively in InputBinding.
type CatalogModelSource struct {
	state         protoimpl.MessageState `protogen:"open.v1"`
	Repository    string                 `protobuf:"bytes,1,opt,name=repository,proto3" json:"repository,omitempty"` // canonical org/name; no release, lane, query or fragment
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *CatalogModelSource) Reset() {
	*x = CatalogModelSource{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[204]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *CatalogModelSource) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*CatalogModelSource) ProtoMessage() {}

func (x *CatalogModelSource) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[204]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use CatalogModelSource.ProtoReflect.Descriptor instead.
func (*CatalogModelSource) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{204}
}

func (x *CatalogModelSource) GetRepository() string {
	if x != nil {
		return x.Repository
	}
	return ""
}

type InputAccess struct {
	state   protoimpl.MessageState `protogen:"open.v1"`
	InputId string                 `protobuf:"bytes,1,opt,name=input_id,json=inputId,proto3" json:"input_id,omitempty"`
	Url     string                 `protobuf:"bytes,2,opt,name=url,proto3" json:"url,omitempty"` // presigned download URL
	// Exactly one of url and native_tree. The paired InputBinding names
	// the payload member's digest/length for a file, or the manifest for a Tree.
	// The source content_bytes is a quota fact, never an object byte length.
	NativeTree *NativeByteRetentionRequest `protobuf:"bytes,3,opt,name=native_tree,json=nativeTree,proto3" json:"native_tree,omitempty"`
	// Only model:<parameter>, with its existing model:// URL and exact
	// model-manifest InputBinding. Mutually exclusive with native_tree. Runtime
	// verifies native repository membership once, then holds the exact checkpoint.
	CatalogModel  *CatalogModelSource `protobuf:"bytes,4,opt,name=catalog_model,json=catalogModel,proto3" json:"catalog_model,omitempty"`
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *InputAccess) Reset() {
	*x = InputAccess{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[205]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *InputAccess) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*InputAccess) ProtoMessage() {}

func (x *InputAccess) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[205]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use InputAccess.ProtoReflect.Descriptor instead.
func (*InputAccess) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{205}
}

func (x *InputAccess) GetInputId() string {
	if x != nil {
		return x.InputId
	}
	return ""
}

func (x *InputAccess) GetUrl() string {
	if x != nil {
		return x.Url
	}
	return ""
}

func (x *InputAccess) GetNativeTree() *NativeByteRetentionRequest {
	if x != nil {
		return x.NativeTree
	}
	return nil
}

func (x *InputAccess) GetCatalogModel() *CatalogModelSource {
	if x != nil {
		return x.CatalogModel
	}
	return nil
}

type OutputAccess struct {
	state         protoimpl.MessageState `protogen:"open.v1"`
	OutputId      string                 `protobuf:"bytes,1,opt,name=output_id,json=outputId,proto3" json:"output_id,omitempty"`
	Url           string                 `protobuf:"bytes,2,opt,name=url,proto3" json:"url,omitempty"` // presigned upload URL
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *OutputAccess) Reset() {
	*x = OutputAccess{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[206]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *OutputAccess) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*OutputAccess) ProtoMessage() {}

func (x *OutputAccess) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[206]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use OutputAccess.ProtoReflect.Descriptor instead.
func (*OutputAccess) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{206}
}

func (x *OutputAccess) GetOutputId() string {
	if x != nil {
		return x.OutputId
	}
	return ""
}

func (x *OutputAccess) GetUrl() string {
	if x != nil {
		return x.Url
	}
	return ""
}

type DeliveryAccessCredential struct {
	state           protoimpl.MessageState `protogen:"open.v1"`
	Issuer          string                 `protobuf:"bytes,1,opt,name=issuer,proto3" json:"issuer,omitempty"`
	KeyId           string                 `protobuf:"bytes,2,opt,name=key_id,json=keyId,proto3" json:"key_id,omitempty"`
	CredentialEpoch uint64                 `protobuf:"varint,3,opt,name=credential_epoch,json=credentialEpoch,proto3" json:"credential_epoch,omitempty"`
	ExpiresAtUnix   uint64                 `protobuf:"varint,4,opt,name=expires_at_unix,json=expiresAtUnix,proto3" json:"expires_at_unix,omitempty"`
	Token           []byte                 `protobuf:"bytes,5,opt,name=token,proto3" json:"token,omitempty"` // STRUCTURALLY SECRET; never in a digest, a log, or an
	unknownFields   protoimpl.UnknownFields
	sizeCache       protoimpl.SizeCache
}

func (x *DeliveryAccessCredential) Reset() {
	*x = DeliveryAccessCredential{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[207]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *DeliveryAccessCredential) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*DeliveryAccessCredential) ProtoMessage() {}

func (x *DeliveryAccessCredential) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[207]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use DeliveryAccessCredential.ProtoReflect.Descriptor instead.
func (*DeliveryAccessCredential) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{207}
}

func (x *DeliveryAccessCredential) GetIssuer() string {
	if x != nil {
		return x.Issuer
	}
	return ""
}

func (x *DeliveryAccessCredential) GetKeyId() string {
	if x != nil {
		return x.KeyId
	}
	return ""
}

func (x *DeliveryAccessCredential) GetCredentialEpoch() uint64 {
	if x != nil {
		return x.CredentialEpoch
	}
	return 0
}

func (x *DeliveryAccessCredential) GetExpiresAtUnix() uint64 {
	if x != nil {
		return x.ExpiresAtUnix
	}
	return 0
}

func (x *DeliveryAccessCredential) GetToken() []byte {
	if x != nil {
		return x.Token
	}
	return nil
}

type ResourceCaps struct {
	state                protoimpl.MessageState `protogen:"open.v1"`
	DeviceRequired       bool                   `protobuf:"varint,1,opt,name=device_required,json=deviceRequired,proto3" json:"device_required,omitempty"`
	MaxDeviceMemoryBytes uint64                 `protobuf:"varint,2,opt,name=max_device_memory_bytes,json=maxDeviceMemoryBytes,proto3" json:"max_device_memory_bytes,omitempty"`
	MaxRssBytes          uint64                 `protobuf:"varint,3,opt,name=max_rss_bytes,json=maxRssBytes,proto3" json:"max_rss_bytes,omitempty"`
	MaxDiskBytes         uint64                 `protobuf:"varint,4,opt,name=max_disk_bytes,json=maxDiskBytes,proto3" json:"max_disk_bytes,omitempty"`
	unknownFields        protoimpl.UnknownFields
	sizeCache            protoimpl.SizeCache
}

func (x *ResourceCaps) Reset() {
	*x = ResourceCaps{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[208]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *ResourceCaps) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*ResourceCaps) ProtoMessage() {}

func (x *ResourceCaps) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[208]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use ResourceCaps.ProtoReflect.Descriptor instead.
func (*ResourceCaps) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{208}
}

func (x *ResourceCaps) GetDeviceRequired() bool {
	if x != nil {
		return x.DeviceRequired
	}
	return false
}

func (x *ResourceCaps) GetMaxDeviceMemoryBytes() uint64 {
	if x != nil {
		return x.MaxDeviceMemoryBytes
	}
	return 0
}

func (x *ResourceCaps) GetMaxRssBytes() uint64 {
	if x != nil {
		return x.MaxRssBytes
	}
	return 0
}

func (x *ResourceCaps) GetMaxDiskBytes() uint64 {
	if x != nil {
		return x.MaxDiskBytes
	}
	return 0
}

type PublicationContract struct {
	state         protoimpl.MessageState `protogen:"open.v1"`
	Outputs       []*OutputBinding       `protobuf:"bytes,1,rep,name=outputs,proto3" json:"outputs,omitempty"` // sorted by output_id
	GrantId       string                 `protobuf:"bytes,2,opt,name=grant_id,json=grantId,proto3" json:"grant_id,omitempty"`
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *PublicationContract) Reset() {
	*x = PublicationContract{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[209]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *PublicationContract) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*PublicationContract) ProtoMessage() {}

func (x *PublicationContract) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[209]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use PublicationContract.ProtoReflect.Descriptor instead.
func (*PublicationContract) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{209}
}

func (x *PublicationContract) GetOutputs() []*OutputBinding {
	if x != nil {
		return x.Outputs
	}
	return nil
}

func (x *PublicationContract) GetGrantId() string {
	if x != nil {
		return x.GrantId
	}
	return ""
}

// Torch-free worker-measurable STATICS (NVML / sysctl / OS facts), sent once at ClaimAck and
// never re-sent. Accelerator-neutral (#446): backend + memory_model use the AcceleratorFacts
// vocabulary (#422/#423). The worker never fabricates capability facts — a discovered
// accelerator is PRESENT-BUT-UNQUALIFIED until a placement executor qualifies it
// (PlacementStatus.accelerator). torch_version is gone from here: a torch-free process cannot
// honestly report one.
type WorkerResources struct {
	state                  protoimpl.MessageState `protogen:"open.v1"`
	Platform               string                 `protobuf:"bytes,1,opt,name=platform,proto3" json:"platform,omitempty"`                          // os/arch, e.g. linux/amd64, windows/amd64, darwin/arm64
	Backend                string                 `protobuf:"bytes,2,opt,name=backend,proto3" json:"backend,omitempty"`                            // cuda | mps; "" = no accelerator discovered
	MemoryModel            string                 `protobuf:"bytes,3,opt,name=memory_model,json=memoryModel,proto3" json:"memory_model,omitempty"` // discrete | unified — the ledger's PRICING MODEL selector
	DeviceCount            uint32                 `protobuf:"varint,4,opt,name=device_count,json=deviceCount,proto3" json:"device_count,omitempty"`
	DeviceName             string                 `protobuf:"bytes,5,opt,name=device_name,json=deviceName,proto3" json:"device_name,omitempty"`
	DeviceMemoryTotalBytes uint64                 `protobuf:"varint,6,opt,name=device_memory_total_bytes,json=deviceMemoryTotalBytes,proto3" json:"device_memory_total_bytes,omitempty"` // per device
	DriverVersion          string                 `protobuf:"bytes,7,opt,name=driver_version,json=driverVersion,proto3" json:"driver_version,omitempty"`
	BackendVersion         string                 `protobuf:"bytes,8,opt,name=backend_version,json=backendVersion,proto3" json:"backend_version,omitempty"` // CUDA toolkit version; macOS/Metal version on mps
	HostRamTotalBytes      uint64                 `protobuf:"varint,9,opt,name=host_ram_total_bytes,json=hostRamTotalBytes,proto3" json:"host_ram_total_bytes,omitempty"`
	VcpuCount              uint32                 `protobuf:"varint,10,opt,name=vcpu_count,json=vcpuCount,proto3" json:"vcpu_count,omitempty"`
	DiskTotalBytes         uint64                 `protobuf:"varint,11,opt,name=disk_total_bytes,json=diskTotalBytes,proto3" json:"disk_total_bytes,omitempty"`
	PowerCapWatts          uint32                 `protobuf:"varint,12,opt,name=power_cap_watts,json=powerCapWatts,proto3" json:"power_cap_watts,omitempty"`
	PartitionProfile       string                 `protobuf:"bytes,13,opt,name=partition_profile,json=partitionProfile,proto3" json:"partition_profile,omitempty"` // device partition (NVIDIA MIG on cuda); empty = whole
	// device
	Interconnect  string   `protobuf:"bytes,14,opt,name=interconnect,proto3" json:"interconnect,omitempty"`
	PeerAccess    bool     `protobuf:"varint,15,opt,name=peer_access,json=peerAccess,proto3" json:"peer_access,omitempty"`
	Unreadable    []string `protobuf:"bytes,16,rep,name=unreadable,proto3" json:"unreadable,omitempty"` // tri-state honesty: unreadable is distinct from zero
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *WorkerResources) Reset() {
	*x = WorkerResources{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[210]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *WorkerResources) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*WorkerResources) ProtoMessage() {}

func (x *WorkerResources) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[210]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use WorkerResources.ProtoReflect.Descriptor instead.
func (*WorkerResources) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{210}
}

func (x *WorkerResources) GetPlatform() string {
	if x != nil {
		return x.Platform
	}
	return ""
}

func (x *WorkerResources) GetBackend() string {
	if x != nil {
		return x.Backend
	}
	return ""
}

func (x *WorkerResources) GetMemoryModel() string {
	if x != nil {
		return x.MemoryModel
	}
	return ""
}

func (x *WorkerResources) GetDeviceCount() uint32 {
	if x != nil {
		return x.DeviceCount
	}
	return 0
}

func (x *WorkerResources) GetDeviceName() string {
	if x != nil {
		return x.DeviceName
	}
	return ""
}

func (x *WorkerResources) GetDeviceMemoryTotalBytes() uint64 {
	if x != nil {
		return x.DeviceMemoryTotalBytes
	}
	return 0
}

func (x *WorkerResources) GetDriverVersion() string {
	if x != nil {
		return x.DriverVersion
	}
	return ""
}

func (x *WorkerResources) GetBackendVersion() string {
	if x != nil {
		return x.BackendVersion
	}
	return ""
}

func (x *WorkerResources) GetHostRamTotalBytes() uint64 {
	if x != nil {
		return x.HostRamTotalBytes
	}
	return 0
}

func (x *WorkerResources) GetVcpuCount() uint32 {
	if x != nil {
		return x.VcpuCount
	}
	return 0
}

func (x *WorkerResources) GetDiskTotalBytes() uint64 {
	if x != nil {
		return x.DiskTotalBytes
	}
	return 0
}

func (x *WorkerResources) GetPowerCapWatts() uint32 {
	if x != nil {
		return x.PowerCapWatts
	}
	return 0
}

func (x *WorkerResources) GetPartitionProfile() string {
	if x != nil {
		return x.PartitionProfile
	}
	return ""
}

func (x *WorkerResources) GetInterconnect() string {
	if x != nil {
		return x.Interconnect
	}
	return ""
}

func (x *WorkerResources) GetPeerAccess() bool {
	if x != nil {
		return x.PeerAccess
	}
	return false
}

func (x *WorkerResources) GetUnreadable() []string {
	if x != nil {
		return x.Unreadable
	}
	return nil
}

type JobCapacity struct {
	state protoimpl.MessageState `protogen:"open.v1"`
	// #446); jobs_available IS the credit
	JobsInFlight           uint32 `protobuf:"varint,1,opt,name=jobs_in_flight,json=jobsInFlight,proto3" json:"jobs_in_flight,omitempty"`
	JobsAvailable          uint32 `protobuf:"varint,2,opt,name=jobs_available,json=jobsAvailable,proto3" json:"jobs_available,omitempty"`
	FreeDiskBytes          uint64 `protobuf:"varint,4,opt,name=free_disk_bytes,json=freeDiskBytes,proto3" json:"free_disk_bytes,omitempty"`
	OrchestrationInFlight  uint32 `protobuf:"varint,5,opt,name=orchestration_in_flight,json=orchestrationInFlight,proto3" json:"orchestration_in_flight,omitempty"`
	OrchestrationAvailable uint32 `protobuf:"varint,6,opt,name=orchestration_available,json=orchestrationAvailable,proto3" json:"orchestration_available,omitempty"`
	unknownFields          protoimpl.UnknownFields
	sizeCache              protoimpl.SizeCache
}

func (x *JobCapacity) Reset() {
	*x = JobCapacity{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[211]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *JobCapacity) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*JobCapacity) ProtoMessage() {}

func (x *JobCapacity) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[211]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use JobCapacity.ProtoReflect.Descriptor instead.
func (*JobCapacity) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{211}
}

func (x *JobCapacity) GetJobsInFlight() uint32 {
	if x != nil {
		return x.JobsInFlight
	}
	return 0
}

func (x *JobCapacity) GetJobsAvailable() uint32 {
	if x != nil {
		return x.JobsAvailable
	}
	return 0
}

func (x *JobCapacity) GetFreeDiskBytes() uint64 {
	if x != nil {
		return x.FreeDiskBytes
	}
	return 0
}

func (x *JobCapacity) GetOrchestrationInFlight() uint32 {
	if x != nil {
		return x.OrchestrationInFlight
	}
	return 0
}

func (x *JobCapacity) GetOrchestrationAvailable() uint32 {
	if x != nil {
		return x.OrchestrationAvailable
	}
	return 0
}

// Was ActiveAttempt (#481): the set holds unacked OUTCOMES too, and those are not active. At
// (proto-026) it holds QUEUED attempts too: every accepted attempt appears here from
// admission to ack, so the orchestrator's records are exact from the observed queue and a
// queued attempt is never absent. Lane, order and fairness are worker facts the orchestrator
// RENDERS, never sets: the worker orders its lane (gpu-hot.md §4), the orchestrator offers depth.
type HeldAttempt struct {
	state                protoimpl.MessageState `protogen:"open.v1"`
	RequestId            string                 `protobuf:"bytes,1,opt,name=request_id,json=requestId,proto3" json:"request_id,omitempty"`
	AttemptOrdinal       uint64                 `protobuf:"varint,2,opt,name=attempt_ordinal,json=attemptOrdinal,proto3" json:"attempt_ordinal,omitempty"`
	Kind                 AttemptKind            `protobuf:"varint,3,opt,name=kind,proto3,enum=cozy.worker.v1.AttemptKind" json:"kind,omitempty"`
	State                AttemptState           `protobuf:"varint,4,opt,name=state,proto3,enum=cozy.worker.v1.AttemptState" json:"state,omitempty"`
	InvocationSpecDigest []byte                 `protobuf:"bytes,5,opt,name=invocation_spec_digest,json=invocationSpecDigest,proto3" json:"invocation_spec_digest,omitempty"`
	PlacementId          string                 `protobuf:"bytes,6,opt,name=placement_id,json=placementId,proto3" json:"placement_id,omitempty"`        // the executing placement; empty in job mode
	ExecutorEpoch        uint64                 `protobuf:"varint,7,opt,name=executor_epoch,json=executorEpoch,proto3" json:"executor_epoch,omitempty"` // the epoch it runs/ran under; 0 while QUEUED — a queued
	// attempt is bound to no executor and survives a respawn
	OutcomeId     string `protobuf:"bytes,8,opt,name=outcome_id,json=outcomeId,proto3" json:"outcome_id,omitempty"`             // set iff state is OUTCOME_PENDING_ACK
	OutcomeDigest []byte `protobuf:"bytes,9,opt,name=outcome_digest,json=outcomeDigest,proto3" json:"outcome_digest,omitempty"` // class (a); set iff state is OUTCOME_PENDING_ACK
	LaneId        string `protobuf:"bytes,10,opt,name=lane_id,json=laneId,proto3" json:"lane_id,omitempty"`                     // proto-026: the DeviceLane this attempt is queued on,
	// running on, or ran on; empty in job mode
	QueuePosition uint32 `protobuf:"varint,11,opt,name=queue_position,json=queuePosition,proto3" json:"queue_position,omitempty"` // the worker's CURRENT order among this lane's QUEUED
	// attempts: 0 = next on the device. Meaningful only while
	// QUEUED; 0 otherwise. Re-derived at every device entry.
	Overtaken uint32 `protobuf:"varint,12,opt,name=overtaken,proto3" json:"overtaken,omitempty"` // times a later-admitted attempt entered the device ahead
	// of this one; monotone for the attempt's life
	OvertakeBudget uint32 `protobuf:"varint,13,opt,name=overtake_budget,json=overtakeBudget,proto3" json:"overtake_budget,omitempty"` // = the attempt's queue position at ADMISSION, the most it
	// may be overtaken; spent => picked before any class.
	// A count the queue itself yields, never a constant or env.
	PlanDigest    []byte `protobuf:"bytes,14,opt,name=plan_digest,json=planDigest,proto3" json:"plan_digest,omitempty"` // class (a): AttemptPlan (cr-007), bound at DEVICE ENTRY;
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *HeldAttempt) Reset() {
	*x = HeldAttempt{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[212]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *HeldAttempt) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*HeldAttempt) ProtoMessage() {}

func (x *HeldAttempt) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[212]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use HeldAttempt.ProtoReflect.Descriptor instead.
func (*HeldAttempt) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{212}
}

func (x *HeldAttempt) GetRequestId() string {
	if x != nil {
		return x.RequestId
	}
	return ""
}

func (x *HeldAttempt) GetAttemptOrdinal() uint64 {
	if x != nil {
		return x.AttemptOrdinal
	}
	return 0
}

func (x *HeldAttempt) GetKind() AttemptKind {
	if x != nil {
		return x.Kind
	}
	return AttemptKind_ATTEMPT_KIND_UNSPECIFIED
}

func (x *HeldAttempt) GetState() AttemptState {
	if x != nil {
		return x.State
	}
	return AttemptState_ATTEMPT_STATE_UNSPECIFIED
}

func (x *HeldAttempt) GetInvocationSpecDigest() []byte {
	if x != nil {
		return x.InvocationSpecDigest
	}
	return nil
}

func (x *HeldAttempt) GetPlacementId() string {
	if x != nil {
		return x.PlacementId
	}
	return ""
}

func (x *HeldAttempt) GetExecutorEpoch() uint64 {
	if x != nil {
		return x.ExecutorEpoch
	}
	return 0
}

func (x *HeldAttempt) GetOutcomeId() string {
	if x != nil {
		return x.OutcomeId
	}
	return ""
}

func (x *HeldAttempt) GetOutcomeDigest() []byte {
	if x != nil {
		return x.OutcomeDigest
	}
	return nil
}

func (x *HeldAttempt) GetLaneId() string {
	if x != nil {
		return x.LaneId
	}
	return ""
}

func (x *HeldAttempt) GetQueuePosition() uint32 {
	if x != nil {
		return x.QueuePosition
	}
	return 0
}

func (x *HeldAttempt) GetOvertaken() uint32 {
	if x != nil {
		return x.Overtaken
	}
	return 0
}

func (x *HeldAttempt) GetOvertakeBudget() uint32 {
	if x != nil {
		return x.OvertakeBudget
	}
	return 0
}

func (x *HeldAttempt) GetPlanDigest() []byte {
	if x != nil {
		return x.PlanDigest
	}
	return nil
}

type Fault struct {
	state   protoimpl.MessageState `protogen:"open.v1"`
	Kind    FaultKind              `protobuf:"varint,1,opt,name=kind,proto3,enum=cozy.worker.v1.FaultKind" json:"kind,omitempty"`
	Subject string                 `protobuf:"bytes,2,opt,name=subject,proto3" json:"subject,omitempty"` // the placement_id, subject digest, or machine facet
	Reason  string                 `protobuf:"bytes,3,opt,name=reason,proto3" json:"reason,omitempty"`   // bounded <= 1024 bytes
	Detail  string                 `protobuf:"bytes,4,opt,name=detail,proto3" json:"detail,omitempty"`   // bounded <= 1024 bytes
	// The DesiredWorkerState revision this fault refuses: Runtime's verdict on an unapplied
	// revision, which never advances accepted_desired_state_revision. 0: not such a refusal.
	DesiredStateRevision uint64 `protobuf:"varint,5,opt,name=desired_state_revision,json=desiredStateRevision,proto3" json:"desired_state_revision,omitempty"`
	unknownFields        protoimpl.UnknownFields
	sizeCache            protoimpl.SizeCache
}

func (x *Fault) Reset() {
	*x = Fault{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[213]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *Fault) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*Fault) ProtoMessage() {}

func (x *Fault) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[213]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use Fault.ProtoReflect.Descriptor instead.
func (*Fault) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{213}
}

func (x *Fault) GetKind() FaultKind {
	if x != nil {
		return x.Kind
	}
	return FaultKind_FAULT_KIND_UNSPECIFIED
}

func (x *Fault) GetSubject() string {
	if x != nil {
		return x.Subject
	}
	return ""
}

func (x *Fault) GetReason() string {
	if x != nil {
		return x.Reason
	}
	return ""
}

func (x *Fault) GetDetail() string {
	if x != nil {
		return x.Detail
	}
	return ""
}

func (x *Fault) GetDesiredStateRevision() uint64 {
	if x != nil {
		return x.DesiredStateRevision
	}
	return 0
}

type OutputManifest struct {
	state                    protoimpl.MessageState `protogen:"open.v1"`
	PublicationReceiptDigest string                 `protobuf:"bytes,1,opt,name=publication_receipt_digest,json=publicationReceiptDigest,proto3" json:"publication_receipt_digest,omitempty"` // class (a), sha256:<hex>: the PublicationReceipt's
	// canonical digest (#420). Was `manifest_id` — the frozen
	// proto's own comment said what it carries and the field
	// name claimed otherwise (#481/#486).
	Outputs       []*OutputEntry `protobuf:"bytes,3,rep,name=outputs,proto3" json:"outputs,omitempty"` // sorted by output_id
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *OutputManifest) Reset() {
	*x = OutputManifest{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[214]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *OutputManifest) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*OutputManifest) ProtoMessage() {}

func (x *OutputManifest) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[214]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use OutputManifest.ProtoReflect.Descriptor instead.
func (*OutputManifest) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{214}
}

func (x *OutputManifest) GetPublicationReceiptDigest() string {
	if x != nil {
		return x.PublicationReceiptDigest
	}
	return ""
}

func (x *OutputManifest) GetOutputs() []*OutputEntry {
	if x != nil {
		return x.Outputs
	}
	return nil
}

type OutputEntry struct {
	state    protoimpl.MessageState `protogen:"open.v1"`
	OutputId string                 `protobuf:"bytes,1,opt,name=output_id,json=outputId,proto3" json:"output_id,omitempty"`
	Digest   []byte                 `protobuf:"bytes,2,opt,name=digest,proto3" json:"digest,omitempty"` // class (b)
	Length   uint64                 `protobuf:"varint,3,opt,name=length,proto3" json:"length,omitempty"`
	MimeType string                 `protobuf:"bytes,4,opt,name=mime_type,json=mimeType,proto3" json:"mime_type,omitempty"`
	// Optional only for ordinary byte outputs committed in native custody.
	// digest/length remain the actual file, or actual manifest for a Tree output.
	NativeTree    *NativeByteTreeRef `protobuf:"bytes,5,opt,name=native_tree,json=nativeTree,proto3" json:"native_tree,omitempty"`
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *OutputEntry) Reset() {
	*x = OutputEntry{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[215]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *OutputEntry) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*OutputEntry) ProtoMessage() {}

func (x *OutputEntry) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[215]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use OutputEntry.ProtoReflect.Descriptor instead.
func (*OutputEntry) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{215}
}

func (x *OutputEntry) GetOutputId() string {
	if x != nil {
		return x.OutputId
	}
	return ""
}

func (x *OutputEntry) GetDigest() []byte {
	if x != nil {
		return x.Digest
	}
	return nil
}

func (x *OutputEntry) GetLength() uint64 {
	if x != nil {
		return x.Length
	}
	return 0
}

func (x *OutputEntry) GetMimeType() string {
	if x != nil {
		return x.MimeType
	}
	return ""
}

func (x *OutputEntry) GetNativeTree() *NativeByteTreeRef {
	if x != nil {
		return x.NativeTree
	}
	return nil
}

type AttemptMetrics struct {
	state                 protoimpl.MessageState `protogen:"open.v1"`
	RuntimeMs             uint64                 `protobuf:"varint,1,opt,name=runtime_ms,json=runtimeMs,proto3" json:"runtime_ms,omitempty"`
	QueueMs               uint64                 `protobuf:"varint,2,opt,name=queue_ms,json=queueMs,proto3" json:"queue_ms,omitempty"`
	PeakDeviceMemoryBytes uint64                 `protobuf:"varint,3,opt,name=peak_device_memory_bytes,json=peakDeviceMemoryBytes,proto3" json:"peak_device_memory_bytes,omitempty"`
	RssAtEndBytes         uint64                 `protobuf:"varint,4,opt,name=rss_at_end_bytes,json=rssAtEndBytes,proto3" json:"rss_at_end_bytes,omitempty"`
	OutputCount           uint32                 `protobuf:"varint,5,opt,name=output_count,json=outputCount,proto3" json:"output_count,omitempty"`
	InputTokens           uint64                 `protobuf:"varint,6,opt,name=input_tokens,json=inputTokens,proto3" json:"input_tokens,omitempty"` // analytics at launch, never settlement input
	OutputTokens          uint64                 `protobuf:"varint,7,opt,name=output_tokens,json=outputTokens,proto3" json:"output_tokens,omitempty"`
	DeviceLeaseMs         uint64                 `protobuf:"varint,8,opt,name=device_lease_ms,json=deviceLeaseMs,proto3" json:"device_lease_ms,omitempty"` // worker-attested: device ENTRY -> device RELEASE (the
	// executor's invoke reply, gpu-hot.md §3). The encode is
	// EXCLUDED at minor 23 — it is post_ms — so this is the
	// device phase and nothing else.
	DeviceCount      uint32   `protobuf:"varint,9,opt,name=device_count,json=deviceCount,proto3" json:"device_count,omitempty"`
	HandlerMs        uint64   `protobuf:"varint,10,opt,name=handler_ms,json=handlerMs,proto3" json:"handler_ms,omitempty"`
	FinalizationMs   uint64   `protobuf:"varint,11,opt,name=finalization_ms,json=finalizationMs,proto3" json:"finalization_ms,omitempty"`
	UnverifiedFields []string `protobuf:"bytes,12,rep,name=unverified_fields,json=unverifiedFields,proto3" json:"unverified_fields,omitempty"`
	PostMs           uint64   `protobuf:"varint,13,opt,name=post_ms,json=postMs,proto3" json:"post_ms,omitempty"` // proto-026: device release -> outcome sent — the post
	// phase (encode, digest, write + fsync, outcome build);
	// CPU, no device, no executor
	WorkingPeakDeviceBytes uint64 `protobuf:"varint,14,opt,name=working_peak_device_bytes,json=workingPeakDeviceBytes,proto3" json:"working_peak_device_bytes,omitempty"` // max over ranks of (allocator peak -
	// allocated at device entry)
	ShapeCell     string `protobuf:"bytes,15,opt,name=shape_cell,json=shapeCell,proto3" json:"shape_cell,omitempty"` // Runtime plan.shape_cell(features)
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *AttemptMetrics) Reset() {
	*x = AttemptMetrics{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[216]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *AttemptMetrics) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*AttemptMetrics) ProtoMessage() {}

func (x *AttemptMetrics) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[216]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use AttemptMetrics.ProtoReflect.Descriptor instead.
func (*AttemptMetrics) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{216}
}

func (x *AttemptMetrics) GetRuntimeMs() uint64 {
	if x != nil {
		return x.RuntimeMs
	}
	return 0
}

func (x *AttemptMetrics) GetQueueMs() uint64 {
	if x != nil {
		return x.QueueMs
	}
	return 0
}

func (x *AttemptMetrics) GetPeakDeviceMemoryBytes() uint64 {
	if x != nil {
		return x.PeakDeviceMemoryBytes
	}
	return 0
}

func (x *AttemptMetrics) GetRssAtEndBytes() uint64 {
	if x != nil {
		return x.RssAtEndBytes
	}
	return 0
}

func (x *AttemptMetrics) GetOutputCount() uint32 {
	if x != nil {
		return x.OutputCount
	}
	return 0
}

func (x *AttemptMetrics) GetInputTokens() uint64 {
	if x != nil {
		return x.InputTokens
	}
	return 0
}

func (x *AttemptMetrics) GetOutputTokens() uint64 {
	if x != nil {
		return x.OutputTokens
	}
	return 0
}

func (x *AttemptMetrics) GetDeviceLeaseMs() uint64 {
	if x != nil {
		return x.DeviceLeaseMs
	}
	return 0
}

func (x *AttemptMetrics) GetDeviceCount() uint32 {
	if x != nil {
		return x.DeviceCount
	}
	return 0
}

func (x *AttemptMetrics) GetHandlerMs() uint64 {
	if x != nil {
		return x.HandlerMs
	}
	return 0
}

func (x *AttemptMetrics) GetFinalizationMs() uint64 {
	if x != nil {
		return x.FinalizationMs
	}
	return 0
}

func (x *AttemptMetrics) GetUnverifiedFields() []string {
	if x != nil {
		return x.UnverifiedFields
	}
	return nil
}

func (x *AttemptMetrics) GetPostMs() uint64 {
	if x != nil {
		return x.PostMs
	}
	return 0
}

func (x *AttemptMetrics) GetWorkingPeakDeviceBytes() uint64 {
	if x != nil {
		return x.WorkingPeakDeviceBytes
	}
	return 0
}

func (x *AttemptMetrics) GetShapeCell() string {
	if x != nil {
		return x.ShapeCell
	}
	return ""
}

type TriageBundleRef struct {
	state              protoimpl.MessageState `protogen:"open.v1"`
	SubjectId          string                 `protobuf:"bytes,1,opt,name=subject_id,json=subjectId,proto3" json:"subject_id,omitempty"`
	WriteReceiptDigest []byte                 `protobuf:"bytes,4,opt,name=write_receipt_digest,json=writeReceiptDigest,proto3" json:"write_receipt_digest,omitempty"` // class (a)
	Length             uint64                 `protobuf:"varint,3,opt,name=length,proto3" json:"length,omitempty"`                                                    // bounded <= 1048576 bytes
	unknownFields      protoimpl.UnknownFields
	sizeCache          protoimpl.SizeCache
}

func (x *TriageBundleRef) Reset() {
	*x = TriageBundleRef{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[217]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *TriageBundleRef) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*TriageBundleRef) ProtoMessage() {}

func (x *TriageBundleRef) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[217]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use TriageBundleRef.ProtoReflect.Descriptor instead.
func (*TriageBundleRef) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{217}
}

func (x *TriageBundleRef) GetSubjectId() string {
	if x != nil {
		return x.SubjectId
	}
	return ""
}

func (x *TriageBundleRef) GetWriteReceiptDigest() []byte {
	if x != nil {
		return x.WriteReceiptDigest
	}
	return nil
}

func (x *TriageBundleRef) GetLength() uint64 {
	if x != nil {
		return x.Length
	}
	return 0
}

// A machine reads a package at its own Hub once and keeps the answer: its newest release, and
// each installation's Model resolutions (the owner's bindings, each lane's checkpoint). A warm
// run reads nothing. The owner's rebinding or new release is the change it cannot see, so the
// controller that made it names the package here.
type ForgetPackageCall struct {
	state         protoimpl.MessageState `protogen:"open.v1"`
	Claim         *Claim                 `protobuf:"bytes,1,opt,name=claim,proto3" json:"claim,omitempty"`
	Package       string                 `protobuf:"bytes,2,opt,name=package,proto3" json:"package,omitempty"` // org/name
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *ForgetPackageCall) Reset() {
	*x = ForgetPackageCall{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[218]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *ForgetPackageCall) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*ForgetPackageCall) ProtoMessage() {}

func (x *ForgetPackageCall) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[218]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use ForgetPackageCall.ProtoReflect.Descriptor instead.
func (*ForgetPackageCall) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{218}
}

func (x *ForgetPackageCall) GetClaim() *Claim {
	if x != nil {
		return x.Claim
	}
	return nil
}

func (x *ForgetPackageCall) GetPackage() string {
	if x != nil {
		return x.Package
	}
	return ""
}

type ForgetPackageResult struct {
	state         protoimpl.MessageState `protogen:"open.v1"`
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *ForgetPackageResult) Reset() {
	*x = ForgetPackageResult{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[219]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *ForgetPackageResult) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*ForgetPackageResult) ProtoMessage() {}

func (x *ForgetPackageResult) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[219]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use ForgetPackageResult.ProtoReflect.Descriptor instead.
func (*ForgetPackageResult) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{219}
}

// A NativeArtifactTransfer over the machine connection. The claim authorizes it; the
// command's epoch and stream fields are unused there.
type NativeArtifactTransferCall struct {
	state         protoimpl.MessageState  `protogen:"open.v1"`
	Claim         *Claim                  `protobuf:"bytes,1,opt,name=claim,proto3" json:"claim,omitempty"`
	Request       *NativeArtifactTransfer `protobuf:"bytes,2,opt,name=request,proto3" json:"request,omitempty"`
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *NativeArtifactTransferCall) Reset() {
	*x = NativeArtifactTransferCall{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[220]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *NativeArtifactTransferCall) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*NativeArtifactTransferCall) ProtoMessage() {}

func (x *NativeArtifactTransferCall) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[220]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use NativeArtifactTransferCall.ProtoReflect.Descriptor instead.
func (*NativeArtifactTransferCall) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{220}
}

func (x *NativeArtifactTransferCall) GetClaim() *Claim {
	if x != nil {
		return x.Claim
	}
	return nil
}

func (x *NativeArtifactTransferCall) GetRequest() *NativeArtifactTransfer {
	if x != nil {
		return x.Request
	}
	return nil
}

// A claimed owner may inspect/upload objects only inside its independently
// retained native artifact. No package attempt, author path or raw object inventory
// is manufactured. Inventory is computed from the exact native manifest closure.
type NativeArtifactTransfer struct {
	state              protoimpl.MessageState   `protogen:"open.v1"`
	RecordOwnerEpoch   uint64                   `protobuf:"varint,1,opt,name=record_owner_epoch,json=recordOwnerEpoch,proto3" json:"record_owner_epoch,omitempty"`
	ControlStreamEpoch uint64                   `protobuf:"varint,2,opt,name=control_stream_epoch,json=controlStreamEpoch,proto3" json:"control_stream_epoch,omitempty"`
	WorkerBootId       string                   `protobuf:"bytes,3,opt,name=worker_boot_id,json=workerBootId,proto3" json:"worker_boot_id,omitempty"`
	EffectId           string                   `protobuf:"bytes,4,opt,name=effect_id,json=effectId,proto3" json:"effect_id,omitempty"`
	Source             *DerivedRetentionRequest `protobuf:"bytes,5,opt,name=source,proto3" json:"source,omitempty"`
	Manifest           *Ref                     `protobuf:"bytes,6,opt,name=manifest,proto3" json:"manifest,omitempty"`
	CommandId          uint64                   `protobuf:"varint,7,opt,name=command_id,json=commandId,proto3" json:"command_id,omitempty"`
	Offset             uint32                   `protobuf:"varint,8,opt,name=offset,proto3" json:"offset,omitempty"` // inventory page, no payload offset
	Limit              uint32                   `protobuf:"varint,9,opt,name=limit,proto3" json:"limit,omitempty"`   // inventory page <=128
	Grant              *WeightsUploadGrant      `protobuf:"bytes,10,opt,name=grant,proto3" json:"grant,omitempty"`   // absent means inventory, present means one object upload
	GrantRevision      uint64                   `protobuf:"varint,11,opt,name=grant_revision,json=grantRevision,proto3" json:"grant_revision,omitempty"`
	ServerTimeUnix     int64                    `protobuf:"varint,12,opt,name=server_time_unix,json=serverTimeUnix,proto3" json:"server_time_unix,omitempty"` // grant signer's clock for relative expiry validation
	// Exactly one of source and byte_source is present. The manifest above
	// must equal byte_source.source.manifest; ordinary files are never CozyTensors.
	ByteSource    *NativeByteRetentionRequest `protobuf:"bytes,13,opt,name=byte_source,json=byteSource,proto3" json:"byte_source,omitempty"`
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *NativeArtifactTransfer) Reset() {
	*x = NativeArtifactTransfer{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[221]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *NativeArtifactTransfer) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*NativeArtifactTransfer) ProtoMessage() {}

func (x *NativeArtifactTransfer) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[221]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use NativeArtifactTransfer.ProtoReflect.Descriptor instead.
func (*NativeArtifactTransfer) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{221}
}

func (x *NativeArtifactTransfer) GetRecordOwnerEpoch() uint64 {
	if x != nil {
		return x.RecordOwnerEpoch
	}
	return 0
}

func (x *NativeArtifactTransfer) GetControlStreamEpoch() uint64 {
	if x != nil {
		return x.ControlStreamEpoch
	}
	return 0
}

func (x *NativeArtifactTransfer) GetWorkerBootId() string {
	if x != nil {
		return x.WorkerBootId
	}
	return ""
}

func (x *NativeArtifactTransfer) GetEffectId() string {
	if x != nil {
		return x.EffectId
	}
	return ""
}

func (x *NativeArtifactTransfer) GetSource() *DerivedRetentionRequest {
	if x != nil {
		return x.Source
	}
	return nil
}

func (x *NativeArtifactTransfer) GetManifest() *Ref {
	if x != nil {
		return x.Manifest
	}
	return nil
}

func (x *NativeArtifactTransfer) GetCommandId() uint64 {
	if x != nil {
		return x.CommandId
	}
	return 0
}

func (x *NativeArtifactTransfer) GetOffset() uint32 {
	if x != nil {
		return x.Offset
	}
	return 0
}

func (x *NativeArtifactTransfer) GetLimit() uint32 {
	if x != nil {
		return x.Limit
	}
	return 0
}

func (x *NativeArtifactTransfer) GetGrant() *WeightsUploadGrant {
	if x != nil {
		return x.Grant
	}
	return nil
}

func (x *NativeArtifactTransfer) GetGrantRevision() uint64 {
	if x != nil {
		return x.GrantRevision
	}
	return 0
}

func (x *NativeArtifactTransfer) GetServerTimeUnix() int64 {
	if x != nil {
		return x.ServerTimeUnix
	}
	return 0
}

func (x *NativeArtifactTransfer) GetByteSource() *NativeByteRetentionRequest {
	if x != nil {
		return x.ByteSource
	}
	return nil
}

type NativeArtifactTransferStatus struct {
	state              protoimpl.MessageState      `protogen:"open.v1"`
	RecordOwnerEpoch   uint64                      `protobuf:"varint,1,opt,name=record_owner_epoch,json=recordOwnerEpoch,proto3" json:"record_owner_epoch,omitempty"`
	ControlStreamEpoch uint64                      `protobuf:"varint,2,opt,name=control_stream_epoch,json=controlStreamEpoch,proto3" json:"control_stream_epoch,omitempty"`
	WorkerBootId       string                      `protobuf:"bytes,3,opt,name=worker_boot_id,json=workerBootId,proto3" json:"worker_boot_id,omitempty"`
	EffectId           string                      `protobuf:"bytes,4,opt,name=effect_id,json=effectId,proto3" json:"effect_id,omitempty"`
	Source             *DerivedRetentionRequest    `protobuf:"bytes,5,opt,name=source,proto3" json:"source,omitempty"`
	Manifest           *Ref                        `protobuf:"bytes,6,opt,name=manifest,proto3" json:"manifest,omitempty"`
	CommandId          uint64                      `protobuf:"varint,7,opt,name=command_id,json=commandId,proto3" json:"command_id,omitempty"`
	Objects            []*WeightsObjectRef         `protobuf:"bytes,8,rep,name=objects,proto3" json:"objects,omitempty"` // sorted verified native closure page, <=128
	NextOffset         uint32                      `protobuf:"varint,9,opt,name=next_offset,json=nextOffset,proto3" json:"next_offset,omitempty"`
	HasMore            bool                        `protobuf:"varint,10,opt,name=has_more,json=hasMore,proto3" json:"has_more,omitempty"`
	ClosureDigest      []byte                      `protobuf:"bytes,11,opt,name=closure_digest,json=closureDigest,proto3" json:"closure_digest,omitempty"` // canonical ordered object-ID/length inventory
	ObjectId           string                      `protobuf:"bytes,12,opt,name=object_id,json=objectId,proto3" json:"object_id,omitempty"`
	GrantRevision      uint64                      `protobuf:"varint,13,opt,name=grant_revision,json=grantRevision,proto3" json:"grant_revision,omitempty"`
	Outcome            WeightsUploadOutcome        `protobuf:"varint,14,opt,name=outcome,proto3,enum=cozy.worker.v1.WeightsUploadOutcome" json:"outcome,omitempty"`
	TransferredBytes   uint64                      `protobuf:"varint,15,opt,name=transferred_bytes,json=transferredBytes,proto3" json:"transferred_bytes,omitempty"`
	ChecksumSha256     string                      `protobuf:"bytes,16,opt,name=checksum_sha256,json=checksumSha256,proto3" json:"checksum_sha256,omitempty"`
	HttpStatus         uint32                      `protobuf:"varint,17,opt,name=http_status,json=httpStatus,proto3" json:"http_status,omitempty"`
	SafeCode           string                      `protobuf:"bytes,18,opt,name=safe_code,json=safeCode,proto3" json:"safe_code,omitempty"`
	SafeDetail         string                      `protobuf:"bytes,19,opt,name=safe_detail,json=safeDetail,proto3" json:"safe_detail,omitempty"`
	ByteSource         *NativeByteRetentionRequest `protobuf:"bytes,20,opt,name=byte_source,json=byteSource,proto3" json:"byte_source,omitempty"` // exact echoed ordinary-tree source
	unknownFields      protoimpl.UnknownFields
	sizeCache          protoimpl.SizeCache
}

func (x *NativeArtifactTransferStatus) Reset() {
	*x = NativeArtifactTransferStatus{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[222]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *NativeArtifactTransferStatus) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*NativeArtifactTransferStatus) ProtoMessage() {}

func (x *NativeArtifactTransferStatus) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[222]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use NativeArtifactTransferStatus.ProtoReflect.Descriptor instead.
func (*NativeArtifactTransferStatus) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{222}
}

func (x *NativeArtifactTransferStatus) GetRecordOwnerEpoch() uint64 {
	if x != nil {
		return x.RecordOwnerEpoch
	}
	return 0
}

func (x *NativeArtifactTransferStatus) GetControlStreamEpoch() uint64 {
	if x != nil {
		return x.ControlStreamEpoch
	}
	return 0
}

func (x *NativeArtifactTransferStatus) GetWorkerBootId() string {
	if x != nil {
		return x.WorkerBootId
	}
	return ""
}

func (x *NativeArtifactTransferStatus) GetEffectId() string {
	if x != nil {
		return x.EffectId
	}
	return ""
}

func (x *NativeArtifactTransferStatus) GetSource() *DerivedRetentionRequest {
	if x != nil {
		return x.Source
	}
	return nil
}

func (x *NativeArtifactTransferStatus) GetManifest() *Ref {
	if x != nil {
		return x.Manifest
	}
	return nil
}

func (x *NativeArtifactTransferStatus) GetCommandId() uint64 {
	if x != nil {
		return x.CommandId
	}
	return 0
}

func (x *NativeArtifactTransferStatus) GetObjects() []*WeightsObjectRef {
	if x != nil {
		return x.Objects
	}
	return nil
}

func (x *NativeArtifactTransferStatus) GetNextOffset() uint32 {
	if x != nil {
		return x.NextOffset
	}
	return 0
}

func (x *NativeArtifactTransferStatus) GetHasMore() bool {
	if x != nil {
		return x.HasMore
	}
	return false
}

func (x *NativeArtifactTransferStatus) GetClosureDigest() []byte {
	if x != nil {
		return x.ClosureDigest
	}
	return nil
}

func (x *NativeArtifactTransferStatus) GetObjectId() string {
	if x != nil {
		return x.ObjectId
	}
	return ""
}

func (x *NativeArtifactTransferStatus) GetGrantRevision() uint64 {
	if x != nil {
		return x.GrantRevision
	}
	return 0
}

func (x *NativeArtifactTransferStatus) GetOutcome() WeightsUploadOutcome {
	if x != nil {
		return x.Outcome
	}
	return WeightsUploadOutcome_WEIGHTS_UPLOAD_OUTCOME_UNSPECIFIED
}

func (x *NativeArtifactTransferStatus) GetTransferredBytes() uint64 {
	if x != nil {
		return x.TransferredBytes
	}
	return 0
}

func (x *NativeArtifactTransferStatus) GetChecksumSha256() string {
	if x != nil {
		return x.ChecksumSha256
	}
	return ""
}

func (x *NativeArtifactTransferStatus) GetHttpStatus() uint32 {
	if x != nil {
		return x.HttpStatus
	}
	return 0
}

func (x *NativeArtifactTransferStatus) GetSafeCode() string {
	if x != nil {
		return x.SafeCode
	}
	return ""
}

func (x *NativeArtifactTransferStatus) GetSafeDetail() string {
	if x != nil {
		return x.SafeDetail
	}
	return ""
}

func (x *NativeArtifactTransferStatus) GetByteSource() *NativeByteRetentionRequest {
	if x != nil {
		return x.ByteSource
	}
	return nil
}

// A native ordinary-file tree. The receipt identifies its original producer
// root even when the currently held donor is an independent recipient/cache root.
// FileAsset outputs use the single member "payload"; Tree outputs use their actual
// ordinary-file manifest. content_bytes is capacity, NEVER manifest byte length.
type NativeByteTreeRef struct {
	state          protoimpl.MessageState `protogen:"open.v1"`
	ProducerRootId string                 `protobuf:"bytes,1,opt,name=producer_root_id,json=producerRootId,proto3" json:"producer_root_id,omitempty"` // exact sha256:<hex> native producer root identity
	ReceiptDigest  []byte                 `protobuf:"bytes,2,opt,name=receipt_digest,json=receiptDigest,proto3" json:"receipt_digest,omitempty"`      // SHA-256 of the actual immutable native receipt
	Manifest       *Ref                   `protobuf:"bytes,3,opt,name=manifest,proto3" json:"manifest,omitempty"`
	ContentBytes   uint64                 `protobuf:"varint,4,opt,name=content_bytes,json=contentBytes,proto3" json:"content_bytes,omitempty"`
	unknownFields  protoimpl.UnknownFields
	sizeCache      protoimpl.SizeCache
}

func (x *NativeByteTreeRef) Reset() {
	*x = NativeByteTreeRef{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[223]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *NativeByteTreeRef) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*NativeByteTreeRef) ProtoMessage() {}

func (x *NativeByteTreeRef) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[223]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use NativeByteTreeRef.ProtoReflect.Descriptor instead.
func (*NativeByteTreeRef) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{223}
}

func (x *NativeByteTreeRef) GetProducerRootId() string {
	if x != nil {
		return x.ProducerRootId
	}
	return ""
}

func (x *NativeByteTreeRef) GetReceiptDigest() []byte {
	if x != nil {
		return x.ReceiptDigest
	}
	return nil
}

func (x *NativeByteTreeRef) GetManifest() *Ref {
	if x != nil {
		return x.Manifest
	}
	return nil
}

func (x *NativeByteTreeRef) GetContentBytes() uint64 {
	if x != nil {
		return x.ContentBytes
	}
	return 0
}

type NativeByteRetentionRequest struct {
	state         protoimpl.MessageState `protogen:"open.v1"`
	Source        *NativeByteTreeRef     `protobuf:"bytes,1,opt,name=source,proto3" json:"source,omitempty"`
	RetentionId   string                 `protobuf:"bytes,2,opt,name=retention_id,json=retentionId,proto3" json:"retention_id,omitempty"` // exact sha256:<hex>; immutable consumer obligation
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *NativeByteRetentionRequest) Reset() {
	*x = NativeByteRetentionRequest{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[224]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *NativeByteRetentionRequest) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*NativeByteRetentionRequest) ProtoMessage() {}

func (x *NativeByteRetentionRequest) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[224]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use NativeByteRetentionRequest.ProtoReflect.Descriptor instead.
func (*NativeByteRetentionRequest) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{224}
}

func (x *NativeByteRetentionRequest) GetSource() *NativeByteTreeRef {
	if x != nil {
		return x.Source
	}
	return nil
}

func (x *NativeByteRetentionRequest) GetRetentionId() string {
	if x != nil {
		return x.RetentionId
	}
	return ""
}

type NativeByteRetentionCall struct {
	state         protoimpl.MessageState      `protogen:"open.v1"`
	Claim         *Claim                      `protobuf:"bytes,1,opt,name=claim,proto3" json:"claim,omitempty"`
	Request       *NativeByteRetentionRequest `protobuf:"bytes,2,opt,name=request,proto3" json:"request,omitempty"`
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *NativeByteRetentionCall) Reset() {
	*x = NativeByteRetentionCall{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[225]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *NativeByteRetentionCall) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*NativeByteRetentionCall) ProtoMessage() {}

func (x *NativeByteRetentionCall) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[225]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use NativeByteRetentionCall.ProtoReflect.Descriptor instead.
func (*NativeByteRetentionCall) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{225}
}

func (x *NativeByteRetentionCall) GetClaim() *Claim {
	if x != nil {
		return x.Claim
	}
	return nil
}

func (x *NativeByteRetentionCall) GetRequest() *NativeByteRetentionRequest {
	if x != nil {
		return x.Request
	}
	return nil
}

type NativeByteRetentionResult struct {
	state         protoimpl.MessageState `protogen:"open.v1"`
	Source        *NativeByteTreeRef     `protobuf:"bytes,1,opt,name=source,proto3" json:"source,omitempty"`
	RetentionId   string                 `protobuf:"bytes,2,opt,name=retention_id,json=retentionId,proto3" json:"retention_id,omitempty"`
	Released      bool                   `protobuf:"varint,3,opt,name=released,proto3" json:"released,omitempty"`
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *NativeByteRetentionResult) Reset() {
	*x = NativeByteRetentionResult{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[226]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *NativeByteRetentionResult) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*NativeByteRetentionResult) ProtoMessage() {}

func (x *NativeByteRetentionResult) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[226]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use NativeByteRetentionResult.ProtoReflect.Descriptor instead.
func (*NativeByteRetentionResult) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{226}
}

func (x *NativeByteRetentionResult) GetSource() *NativeByteTreeRef {
	if x != nil {
		return x.Source
	}
	return nil
}

func (x *NativeByteRetentionResult) GetRetentionId() string {
	if x != nil {
		return x.RetentionId
	}
	return ""
}

func (x *NativeByteRetentionResult) GetReleased() bool {
	if x != nil {
		return x.Released
	}
	return false
}

// The surrounding ChildCallResult owns recipient request/ordinal/spec/call/intent.
// output_id names a declared result field, or the explicitly requested capture slot.
type ChildByteResultGrant struct {
	state         protoimpl.MessageState `protogen:"open.v1"`
	OutputId      string                 `protobuf:"bytes,1,opt,name=output_id,json=outputId,proto3" json:"output_id,omitempty"`
	Source        *NativeByteTreeRef     `protobuf:"bytes,2,opt,name=source,proto3" json:"source,omitempty"`
	RetentionId   string                 `protobuf:"bytes,3,opt,name=retention_id,json=retentionId,proto3" json:"retention_id,omitempty"`
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *ChildByteResultGrant) Reset() {
	*x = ChildByteResultGrant{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[227]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *ChildByteResultGrant) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*ChildByteResultGrant) ProtoMessage() {}

func (x *ChildByteResultGrant) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[227]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use ChildByteResultGrant.ProtoReflect.Descriptor instead.
func (*ChildByteResultGrant) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{227}
}

func (x *ChildByteResultGrant) GetOutputId() string {
	if x != nil {
		return x.OutputId
	}
	return ""
}

func (x *ChildByteResultGrant) GetSource() *NativeByteTreeRef {
	if x != nil {
		return x.Source
	}
	return nil
}

func (x *ChildByteResultGrant) GetRetentionId() string {
	if x != nil {
		return x.RetentionId
	}
	return ""
}

// Payloads use a dedicated data RPC, never WorkerControl. The claimed owner must
// hold this exact native retention; object must be a verified member of its closure.
// A receiver verifies contiguous offsets, exact requested length and final SHA-256.
// One root input file/tree. The canonical native file manifest is the
// complete object authority, not a caller-authored path list beside it. Exactly
// one header comes first and one commit last. Each declared unique object is sent
// in contiguous chunks; a disconnected object restarts (no offset resume promise).
// Runtime validates capacity and native-commits the full closure before replying.
// Header replay is bound to owner/request/input_id/manifest/content_bytes; a
// changed subject refuses. Root acceptance acquires its own semantic recipient
// before the intake recipient can be released. No producer attempt is invented.
type InputTreeImportHeader struct {
	state                  protoimpl.MessageState `protogen:"open.v1"`
	Claim                  *Claim                 `protobuf:"bytes,1,opt,name=claim,proto3" json:"claim,omitempty"`
	RequestId              string                 `protobuf:"bytes,2,opt,name=request_id,json=requestId,proto3" json:"request_id,omitempty"` // existing root request, <=128 printable bytes
	InputId                string                 `protobuf:"bytes,3,opt,name=input_id,json=inputId,proto3" json:"input_id,omitempty"`       // exact typed input field, <=1024 bytes
	Manifest               *Ref                   `protobuf:"bytes,4,opt,name=manifest,proto3" json:"manifest,omitempty"`
	ManifestCanonicalBytes []byte                 `protobuf:"bytes,5,opt,name=manifest_canonical_bytes,json=manifestCanonicalBytes,proto3" json:"manifest_canonical_bytes,omitempty"` // <= MaxInputTreeManifestBytes
	ContentBytes           uint64                 `protobuf:"varint,6,opt,name=content_bytes,json=contentBytes,proto3" json:"content_bytes,omitempty"`                                // sum of file lengths; Runtime owns admission capacity
	unknownFields          protoimpl.UnknownFields
	sizeCache              protoimpl.SizeCache
}

func (x *InputTreeImportHeader) Reset() {
	*x = InputTreeImportHeader{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[228]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *InputTreeImportHeader) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*InputTreeImportHeader) ProtoMessage() {}

func (x *InputTreeImportHeader) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[228]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use InputTreeImportHeader.ProtoReflect.Descriptor instead.
func (*InputTreeImportHeader) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{228}
}

func (x *InputTreeImportHeader) GetClaim() *Claim {
	if x != nil {
		return x.Claim
	}
	return nil
}

func (x *InputTreeImportHeader) GetRequestId() string {
	if x != nil {
		return x.RequestId
	}
	return ""
}

func (x *InputTreeImportHeader) GetInputId() string {
	if x != nil {
		return x.InputId
	}
	return ""
}

func (x *InputTreeImportHeader) GetManifest() *Ref {
	if x != nil {
		return x.Manifest
	}
	return nil
}

func (x *InputTreeImportHeader) GetManifestCanonicalBytes() []byte {
	if x != nil {
		return x.ManifestCanonicalBytes
	}
	return nil
}

func (x *InputTreeImportHeader) GetContentBytes() uint64 {
	if x != nil {
		return x.ContentBytes
	}
	return 0
}

type InputTreeImportBlob struct {
	state         protoimpl.MessageState `protogen:"open.v1"`
	Object        *Ref                   `protobuf:"bytes,1,opt,name=object,proto3" json:"object,omitempty"`  // exact declared blob; duplicate complete objects need not be repeated
	Offset        uint64                 `protobuf:"varint,2,opt,name=offset,proto3" json:"offset,omitempty"` // contiguous within this stream, starts at zero for each object
	Data          []byte                 `protobuf:"bytes,3,opt,name=data,proto3" json:"data,omitempty"`      // <= MaxInputTreeChunkBytes; empty only for a declared empty object
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *InputTreeImportBlob) Reset() {
	*x = InputTreeImportBlob{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[229]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *InputTreeImportBlob) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*InputTreeImportBlob) ProtoMessage() {}

func (x *InputTreeImportBlob) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[229]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use InputTreeImportBlob.ProtoReflect.Descriptor instead.
func (*InputTreeImportBlob) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{229}
}

func (x *InputTreeImportBlob) GetObject() *Ref {
	if x != nil {
		return x.Object
	}
	return nil
}

func (x *InputTreeImportBlob) GetOffset() uint64 {
	if x != nil {
		return x.Offset
	}
	return 0
}

func (x *InputTreeImportBlob) GetData() []byte {
	if x != nil {
		return x.Data
	}
	return nil
}

type InputTreeImportCommit struct {
	state protoimpl.MessageState `protogen:"open.v1"`
	// Fence this exact intake permanently. No objects are required. Reconciles a
	// native commit that beat the abort, then releases only the intake recipient.
	// The result has released=true; source is absent if no native commit occurred.
	Abort         bool `protobuf:"varint,1,opt,name=abort,proto3" json:"abort,omitempty"`
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *InputTreeImportCommit) Reset() {
	*x = InputTreeImportCommit{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[230]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *InputTreeImportCommit) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*InputTreeImportCommit) ProtoMessage() {}

func (x *InputTreeImportCommit) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[230]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use InputTreeImportCommit.ProtoReflect.Descriptor instead.
func (*InputTreeImportCommit) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{230}
}

func (x *InputTreeImportCommit) GetAbort() bool {
	if x != nil {
		return x.Abort
	}
	return false
}

type InputTreeImportFrame struct {
	state protoimpl.MessageState `protogen:"open.v1"`
	// Types that are valid to be assigned to Body:
	//
	//	*InputTreeImportFrame_Header
	//	*InputTreeImportFrame_Blob
	//	*InputTreeImportFrame_Commit
	Body          isInputTreeImportFrame_Body `protobuf_oneof:"body"`
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *InputTreeImportFrame) Reset() {
	*x = InputTreeImportFrame{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[231]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *InputTreeImportFrame) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*InputTreeImportFrame) ProtoMessage() {}

func (x *InputTreeImportFrame) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[231]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use InputTreeImportFrame.ProtoReflect.Descriptor instead.
func (*InputTreeImportFrame) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{231}
}

func (x *InputTreeImportFrame) GetBody() isInputTreeImportFrame_Body {
	if x != nil {
		return x.Body
	}
	return nil
}

func (x *InputTreeImportFrame) GetHeader() *InputTreeImportHeader {
	if x != nil {
		if x, ok := x.Body.(*InputTreeImportFrame_Header); ok {
			return x.Header
		}
	}
	return nil
}

func (x *InputTreeImportFrame) GetBlob() *InputTreeImportBlob {
	if x != nil {
		if x, ok := x.Body.(*InputTreeImportFrame_Blob); ok {
			return x.Blob
		}
	}
	return nil
}

func (x *InputTreeImportFrame) GetCommit() *InputTreeImportCommit {
	if x != nil {
		if x, ok := x.Body.(*InputTreeImportFrame_Commit); ok {
			return x.Commit
		}
	}
	return nil
}

type isInputTreeImportFrame_Body interface {
	isInputTreeImportFrame_Body()
}

type InputTreeImportFrame_Header struct {
	Header *InputTreeImportHeader `protobuf:"bytes,1,opt,name=header,proto3,oneof"`
}

type InputTreeImportFrame_Blob struct {
	Blob *InputTreeImportBlob `protobuf:"bytes,2,opt,name=blob,proto3,oneof"`
}

type InputTreeImportFrame_Commit struct {
	Commit *InputTreeImportCommit `protobuf:"bytes,3,opt,name=commit,proto3,oneof"`
}

func (*InputTreeImportFrame_Header) isInputTreeImportFrame_Body() {}

func (*InputTreeImportFrame_Blob) isInputTreeImportFrame_Body() {}

func (*InputTreeImportFrame_Commit) isInputTreeImportFrame_Body() {}

type NativeByteReadCall struct {
	state         protoimpl.MessageState      `protogen:"open.v1"`
	Claim         *Claim                      `protobuf:"bytes,1,opt,name=claim,proto3" json:"claim,omitempty"`
	Source        *NativeByteRetentionRequest `protobuf:"bytes,2,opt,name=source,proto3" json:"source,omitempty"`
	Object        *Ref                        `protobuf:"bytes,3,opt,name=object,proto3" json:"object,omitempty"`
	Offset        uint64                      `protobuf:"varint,4,opt,name=offset,proto3" json:"offset,omitempty"` // bounded by object.length; exact suffix read for interrupted I/O
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *NativeByteReadCall) Reset() {
	*x = NativeByteReadCall{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[232]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *NativeByteReadCall) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*NativeByteReadCall) ProtoMessage() {}

func (x *NativeByteReadCall) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[232]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use NativeByteReadCall.ProtoReflect.Descriptor instead.
func (*NativeByteReadCall) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{232}
}

func (x *NativeByteReadCall) GetClaim() *Claim {
	if x != nil {
		return x.Claim
	}
	return nil
}

func (x *NativeByteReadCall) GetSource() *NativeByteRetentionRequest {
	if x != nil {
		return x.Source
	}
	return nil
}

func (x *NativeByteReadCall) GetObject() *Ref {
	if x != nil {
		return x.Object
	}
	return nil
}

func (x *NativeByteReadCall) GetOffset() uint64 {
	if x != nil {
		return x.Offset
	}
	return 0
}

type NativeByteReadChunk struct {
	state         protoimpl.MessageState `protogen:"open.v1"`
	Offset        uint64                 `protobuf:"varint,1,opt,name=offset,proto3" json:"offset,omitempty"`
	Data          []byte                 `protobuf:"bytes,2,opt,name=data,proto3" json:"data,omitempty"` // nonempty, <=MaxNativeByteReadChunkBytes
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *NativeByteReadChunk) Reset() {
	*x = NativeByteReadChunk{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[233]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *NativeByteReadChunk) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*NativeByteReadChunk) ProtoMessage() {}

func (x *NativeByteReadChunk) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[233]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use NativeByteReadChunk.ProtoReflect.Descriptor instead.
func (*NativeByteReadChunk) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{233}
}

func (x *NativeByteReadChunk) GetOffset() uint64 {
	if x != nil {
		return x.Offset
	}
	return 0
}

func (x *NativeByteReadChunk) GetData() []byte {
	if x != nil {
		return x.Data
	}
	return nil
}

// cr-113 request option: never part of a package's payload or interface.
type ActivationCapture struct {
	state         protoimpl.MessageState `protogen:"open.v1"`
	Components    []string               `protobuf:"bytes,1,rep,name=components,proto3" json:"components,omitempty"` // nonempty, unique declared component names
	Steps         []uint32               `protobuf:"varint,2,rep,packed,name=steps,proto3" json:"steps,omitempty"`   // ascending unique, includes 0; empty means the complete schedule
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *ActivationCapture) Reset() {
	*x = ActivationCapture{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[234]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *ActivationCapture) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*ActivationCapture) ProtoMessage() {}

func (x *ActivationCapture) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[234]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use ActivationCapture.ProtoReflect.Descriptor instead.
func (*ActivationCapture) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{234}
}

func (x *ActivationCapture) GetComponents() []string {
	if x != nil {
		return x.Components
	}
	return nil
}

func (x *ActivationCapture) GetSteps() []uint32 {
	if x != nil {
		return x.Steps
	}
	return nil
}

// Observed facts from this attempt's worker/executor, never copied from a lane promise.
type ExecutionEnvironment struct {
	state                   protoimpl.MessageState `protogen:"open.v1"`
	RuntimeVersion          string                 `protobuf:"bytes,1,opt,name=runtime_version,json=runtimeVersion,proto3" json:"runtime_version,omitempty"`
	WorkerImageDigest       []byte                 `protobuf:"bytes,2,opt,name=worker_image_digest,json=workerImageDigest,proto3" json:"worker_image_digest,omitempty"` // absent for a local process with no image identity
	Accelerator             string                 `protobuf:"bytes,3,opt,name=accelerator,proto3" json:"accelerator,omitempty"`                                        // actual accelerator model, or CPU for CPU execution
	Driver                  string                 `protobuf:"bytes,4,opt,name=driver,proto3" json:"driver,omitempty"`
	Cuda                    string                 `protobuf:"bytes,5,opt,name=cuda,proto3" json:"cuda,omitempty"`
	WorkerBootId            string                 `protobuf:"bytes,6,opt,name=worker_boot_id,json=workerBootId,proto3" json:"worker_boot_id,omitempty"`
	ExecutionLane           string                 `protobuf:"bytes,7,opt,name=execution_lane,json=executionLane,proto3" json:"execution_lane,omitempty"`                                 // eager or compiled, only when observed
	ExecutionContractDigest []byte                 `protobuf:"bytes,8,opt,name=execution_contract_digest,json=executionContractDigest,proto3" json:"execution_contract_digest,omitempty"` // absent on a contract-less execution
	KernelSymbol            string                 `protobuf:"bytes,9,opt,name=kernel_symbol,json=kernelSymbol,proto3" json:"kernel_symbol,omitempty"`                                    // actual cr-109 attestation; empty without an observation
	unknownFields           protoimpl.UnknownFields
	sizeCache               protoimpl.SizeCache
}

func (x *ExecutionEnvironment) Reset() {
	*x = ExecutionEnvironment{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[235]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *ExecutionEnvironment) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*ExecutionEnvironment) ProtoMessage() {}

func (x *ExecutionEnvironment) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[235]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use ExecutionEnvironment.ProtoReflect.Descriptor instead.
func (*ExecutionEnvironment) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{235}
}

func (x *ExecutionEnvironment) GetRuntimeVersion() string {
	if x != nil {
		return x.RuntimeVersion
	}
	return ""
}

func (x *ExecutionEnvironment) GetWorkerImageDigest() []byte {
	if x != nil {
		return x.WorkerImageDigest
	}
	return nil
}

func (x *ExecutionEnvironment) GetAccelerator() string {
	if x != nil {
		return x.Accelerator
	}
	return ""
}

func (x *ExecutionEnvironment) GetDriver() string {
	if x != nil {
		return x.Driver
	}
	return ""
}

func (x *ExecutionEnvironment) GetCuda() string {
	if x != nil {
		return x.Cuda
	}
	return ""
}

func (x *ExecutionEnvironment) GetWorkerBootId() string {
	if x != nil {
		return x.WorkerBootId
	}
	return ""
}

func (x *ExecutionEnvironment) GetExecutionLane() string {
	if x != nil {
		return x.ExecutionLane
	}
	return ""
}

func (x *ExecutionEnvironment) GetExecutionContractDigest() []byte {
	if x != nil {
		return x.ExecutionContractDigest
	}
	return nil
}

func (x *ExecutionEnvironment) GetKernelSymbol() string {
	if x != nil {
		return x.KernelSymbol
	}
	return ""
}

type ActivationCaptureResult struct {
	state         protoimpl.MessageState `protogen:"open.v1"`
	OutputId      string                 `protobuf:"bytes,1,opt,name=output_id,json=outputId,proto3" json:"output_id,omitempty"`                // reserved runtime.capture; exact OutputEntry/parent grant lookup
	ContentDigest []byte                 `protobuf:"bytes,2,opt,name=content_digest,json=contentDigest,proto3" json:"content_digest,omitempty"` // SHA256(capture.json || sketches.f32), NOT the Tree manifest
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *ActivationCaptureResult) Reset() {
	*x = ActivationCaptureResult{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[236]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *ActivationCaptureResult) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*ActivationCaptureResult) ProtoMessage() {}

func (x *ActivationCaptureResult) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[236]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use ActivationCaptureResult.ProtoReflect.Descriptor instead.
func (*ActivationCaptureResult) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{236}
}

func (x *ActivationCaptureResult) GetOutputId() string {
	if x != nil {
		return x.OutputId
	}
	return ""
}

func (x *ActivationCaptureResult) GetContentDigest() []byte {
	if x != nil {
		return x.ContentDigest
	}
	return nil
}

type ExecutionObservation struct {
	state         protoimpl.MessageState   `protogen:"open.v1"`
	Environment   *ExecutionEnvironment    `protobuf:"bytes,1,opt,name=environment,proto3" json:"environment,omitempty"`
	Capture       *ActivationCaptureResult `protobuf:"bytes,2,opt,name=capture,proto3" json:"capture,omitempty"`
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *ExecutionObservation) Reset() {
	*x = ExecutionObservation{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[237]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *ExecutionObservation) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*ExecutionObservation) ProtoMessage() {}

func (x *ExecutionObservation) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[237]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use ExecutionObservation.ProtoReflect.Descriptor instead.
func (*ExecutionObservation) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{237}
}

func (x *ExecutionObservation) GetEnvironment() *ExecutionEnvironment {
	if x != nil {
		return x.Environment
	}
	return nil
}

func (x *ExecutionObservation) GetCapture() *ActivationCaptureResult {
	if x != nil {
		return x.Capture
	}
	return nil
}

// RuntimeRevision is the image-owned Runtime/TensorFS software revision a worker should
// converge to. It is metadata only: the desired state never carries wheel bytes or URLs.
// The outer DesiredWorkerState.revision is the monotone fence and the worker must persist
// this nested document before it acknowledges that desired revision.
type RuntimeRevision struct {
	state            protoimpl.MessageState `protogen:"open.v1"`
	WheelDigest      []byte                 `protobuf:"bytes,1,opt,name=wheel_digest,json=wheelDigest,proto3" json:"wheel_digest,omitempty"`                // SHA-256, exactly 32 bytes
	WheelLength      uint64                 `protobuf:"varint,2,opt,name=wheel_length,json=wheelLength,proto3" json:"wheel_length,omitempty"`               // bounded before staging
	RuntimeVersion   string                 `protobuf:"bytes,3,opt,name=runtime_version,json=runtimeVersion,proto3" json:"runtime_version,omitempty"`       // normalized cozy-runtime distribution version
	PythonAbi        string                 `protobuf:"bytes,4,opt,name=python_abi,json=pythonAbi,proto3" json:"python_abi,omitempty"`                      // cpNNN, matched to the immutable base image
	ProvenanceDigest []byte                 `protobuf:"bytes,5,opt,name=provenance_digest,json=provenanceDigest,proto3" json:"provenance_digest,omitempty"` // canonical build/provenance record digest
	Channel          string                 `protobuf:"bytes,6,opt,name=channel,proto3" json:"channel,omitempty"`                                           // development or production
	unknownFields    protoimpl.UnknownFields
	sizeCache        protoimpl.SizeCache
}

func (x *RuntimeRevision) Reset() {
	*x = RuntimeRevision{}
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[238]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *RuntimeRevision) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*RuntimeRevision) ProtoMessage() {}

func (x *RuntimeRevision) ProtoReflect() protoreflect.Message {
	mi := &file_cozy_worker_v1_worker_proto_msgTypes[238]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use RuntimeRevision.ProtoReflect.Descriptor instead.
func (*RuntimeRevision) Descriptor() ([]byte, []int) {
	return file_cozy_worker_v1_worker_proto_rawDescGZIP(), []int{238}
}

func (x *RuntimeRevision) GetWheelDigest() []byte {
	if x != nil {
		return x.WheelDigest
	}
	return nil
}

func (x *RuntimeRevision) GetWheelLength() uint64 {
	if x != nil {
		return x.WheelLength
	}
	return 0
}

func (x *RuntimeRevision) GetRuntimeVersion() string {
	if x != nil {
		return x.RuntimeVersion
	}
	return ""
}

func (x *RuntimeRevision) GetPythonAbi() string {
	if x != nil {
		return x.PythonAbi
	}
	return ""
}

func (x *RuntimeRevision) GetProvenanceDigest() []byte {
	if x != nil {
		return x.ProvenanceDigest
	}
	return nil
}

func (x *RuntimeRevision) GetChannel() string {
	if x != nil {
		return x.Channel
	}
	return ""
}

var File_cozy_worker_v1_worker_proto protoreflect.FileDescriptor

const file_cozy_worker_v1_worker_proto_rawDesc = "" +
	"\n" +
	"\x1bcozy/worker/v1/worker.proto\x12\x0ecozy.worker.v1\"\x8b\x01\n" +
	"\x0fMachineLogQuery\x12+\n" +
	"\x05claim\x18\x01 \x01(\v2\x15.cozy.worker.v1.ClaimR\x05claim\x12,\n" +
	"\x03log\x18\x02 \x01(\x0e2\x1a.cozy.worker.v1.MachineLogR\x03log\x12\x1d\n" +
	"\n" +
	"tail_bytes\x18\x03 \x01(\x04R\ttailBytes\"%\n" +
	"\x0fMachineLogChunk\x12\x12\n" +
	"\x04data\x18\x01 \x01(\fR\x04data\"\x15\n" +
	"\x13ProtocolInfoRequest\"d\n" +
	"\x16KeepRentalAliveRequest\x12+\n" +
	"\x05claim\x18\x01 \x01(\v2\x15.cozy.worker.v1.ClaimR\x05claim\x12\x1d\n" +
	"\n" +
	"request_id\x18\x02 \x01(\tR\trequestId\"\xe3\x01\n" +
	"\x15KeepRentalAliveResult\x12\x1d\n" +
	"\n" +
	"request_id\x18\x01 \x01(\tR\trequestId\x12\x1b\n" +
	"\tworker_id\x18\x02 \x01(\tR\bworkerId\x12$\n" +
	"\x0eworker_boot_id\x18\x03 \x01(\tR\fworkerBootId\x125\n" +
	"\x17acknowledged_at_unix_ms\x18\x04 \x01(\x03R\x14acknowledgedAtUnixMs\x121\n" +
	"\x15idle_deadline_unix_ms\x18\x05 \x01(\x03R\x12idleDeadlineUnixMs\"\xa9\x04\n" +
	"\x17MachineExecutionCapture\x12B\n" +
	"\bbindings\x18\x03 \x03(\v2&.cozy.worker.v1.MachineCallableBindingR\bbindings\x12J\n" +
	"\x0emodel_defaults\x18\x05 \x03(\v2#.cozy.worker.v1.MachineModelDefaultR\rmodelDefaults\x120\n" +
	"\x14root_installation_id\x18\x06 \x01(\tR\x12rootInstallationId\x12O\n" +
	"\x12installed_packages\x18\a \x03(\v2 .cozy.worker.v1.InstalledPackageR\x11installedPackages\x12[\n" +
	"\x16deferred_installations\x18\b \x03(\v2$.cozy.worker.v1.DeferredInstallationR\x15deferredInstallations\x12@\n" +
	"\rmodel_choices\x18\n" +
	" \x03(\v2\x1b.cozy.worker.v1.ModelChoiceR\fmodelChoicesJ\x04\b\x01\x10\x02J\x04\b\x02\x10\x03J\x04\b\x04\x10\x05J\x04\b\t\x10\n" +
	"R\x14root_revision_digestR\trevisionsR\x13published_revisionsR\x0ecatalog_origin\"\xa8\x01\n" +
	"\x14DeferredInstallation\x12\x10\n" +
	"\x03key\x18\x01 \x01(\tR\x03key\x12\x18\n" +
	"\apackage\x18\x02 \x01(\tR\apackage\x12\x18\n" +
	"\arelease\x18\x03 \x01(\tR\arelease\x12J\n" +
	"\vpreparation\x18\x04 \x01(\v2(.cozy.worker.v1.PreparePackageSetRequestR\vpreparation\"\xe6\x02\n" +
	"\x13MachineModelDefault\x12\x1e\n" +
	"\n" +
	"entrypoint\x18\x02 \x01(\tR\n" +
	"entrypoint\x12\x1c\n" +
	"\tparameter\x18\x03 \x01(\tR\tparameter\x12#\n" +
	"\rpublic_origin\x18\x04 \x01(\tR\fpublicOrigin\x12=\n" +
	"\x05rungs\x18\x05 \x03(\v2'.cozy.worker.v1.MachineModelDefaultRungR\x05rungs\x12)\n" +
	"\x10unavailable_code\x18\x06 \x01(\tR\x0funavailableCode\x124\n" +
	"\x16callee_installation_id\x18\a \x01(\tR\x14calleeInstallationId\x12.\n" +
	"\x13callee_deferred_key\x18\b \x01(\tR\x11calleeDeferredKeyJ\x04\b\x01\x10\x02R\x16callee_revision_digest\"\x90\x01\n" +
	"\x17MachineModelDefaultRung\x12\x12\n" +
	"\x04gpus\x18\x04 \x01(\rR\x04gpus\x12\x10\n" +
	"\x03gpu\x18\x01 \x01(\tR\x03gpu\x12\x1e\n" +
	"\n" +
	"repository\x18\x02 \x01(\tR\n" +
	"repository\x12/\n" +
	"\bmanifest\x18\x03 \x01(\v2\x13.cozy.worker.v1.RefR\bmanifest\"\x9c\x01\n" +
	"\x10InstalledPackage\x12'\n" +
	"\x0finstallation_id\x18\x01 \x01(\tR\x0einstallationId\x12\x18\n" +
	"\apackage\x18\x02 \x01(\tR\apackage\x12\x18\n" +
	"\arelease\x18\x03 \x01(\tR\arelease\x12+\n" +
	"\x11package_interface\x18\x04 \x01(\fR\x10packageInterface\"\x88\x03\n" +
	"\x16MachineCallableBinding\x12\x16\n" +
	"\x06module\x18\x03 \x01(\tR\x06module\x12\x16\n" +
	"\x06export\x18\x04 \x01(\tR\x06export\x12\x1e\n" +
	"\n" +
	"entrypoint\x18\x06 \x01(\tR\n" +
	"entrypoint\x124\n" +
	"\x16caller_installation_id\x18\a \x01(\tR\x14callerInstallationId\x124\n" +
	"\x16callee_installation_id\x18\b \x01(\tR\x14calleeInstallationId\x12.\n" +
	"\x13callee_deferred_key\x18\t \x01(\tR\x11calleeDeferredKey\x12.\n" +
	"\x13caller_deferred_key\x18\n" +
	" \x01(\tR\x11callerDeferredKeyJ\x04\b\x01\x10\x02J\x04\b\x02\x10\x03J\x04\b\x05\x10\x06R\x16caller_revision_digestR\x10interface_digestR\x16callee_revision_digest\"\xa1\x01\n" +
	"\x1eMachineExecutionWorkspaceQuery\x12+\n" +
	"\x05claim\x18\x01 \x01(\v2\x15.cozy.worker.v1.ClaimR\x05claim\x12<\n" +
	"\bdescribe\x18\x02 \x01(\v2 .cozy.worker.v1.PackageSelectionR\bdescribeJ\x04\b\x03\x10\x04R\x0ecatalog_origin\"\xc1\x06\n" +
	"\x19MachineExecutionWorkspace\x12\x1b\n" +
	"\tworker_id\x18\x01 \x01(\tR\bworkerId\x12$\n" +
	"\x0eworker_boot_id\x18\x02 \x01(\tR\fworkerBootId\x124\n" +
	"\x16execution_workspace_id\x18\x03 \x01(\tR\x14executionWorkspaceId\x127\n" +
	"\adevices\x18\x04 \x03(\v2\x1d.cozy.worker.v1.MachineDeviceR\adevices\x12/\n" +
	"\x13accelerator_backend\x18\x05 \x01(\tR\x12acceleratorBackend\x124\n" +
	"\x16executor_uid_isolation\x18\x06 \x01(\bR\x14executorUidIsolation\x12M\n" +
	"\x11described_release\x18\x14 \x01(\v2 .cozy.worker.v1.DescribedReleaseR\x10describedRelease\x12$\n" +
	"\x0erun_output_log\x18\x15 \x01(\bR\frunOutputLog\x12,\n" +
	"\x12release_root_owner\x18\x16 \x01(\bR\x10releaseRootOwner\x12)\n" +
	"\x10submission_close\x18\x17 \x01(\bR\x0fsubmissionClose\x12'\n" +
	"\x0fmodel_overrides\x18\x18 \x01(\bR\x0emodelOverridesJ\x04\b\a\x10\x14R\x15cpu_slot_model_inputsR\x14exact_execution_gpusR owner_publication_reconciliationR\x12source_credentialsR\x16deferred_installationsR\x17resolves_model_defaultsR\rrelease_rootsR\x12input_object_reuseR\n" +
	"event_waitR\vmemo_lookupR\x14release_root_sourcesR\x11release_root_jobsR\x15release_root_installs\"s\n" +
	"\x10DescribedRelease\x12\x18\n" +
	"\apackage\x18\x01 \x01(\tR\apackage\x12\x18\n" +
	"\arelease\x18\x02 \x01(\tR\arelease\x12+\n" +
	"\x11package_interface\x18\x03 \x01(\fR\x10packageInterface\"\xb9\x01\n" +
	"\rMachineDevice\x12\x18\n" +
	"\aordinal\x18\x01 \x01(\rR\aordinal\x12\x12\n" +
	"\x04name\x18\x02 \x01(\tR\x04name\x12\x12\n" +
	"\x04uuid\x18\x03 \x01(\tR\x04uuid\x12!\n" +
	"\fmemory_bytes\x18\x04 \x01(\x04R\vmemoryBytes\x12\x1c\n" +
	"\n" +
	"pci_bus_id\x18\x05 \x01(\tR\bpciBusId\x12%\n" +
	"\x0edriver_version\x18\x06 \x01(\tR\rdriverVersion\"\xd0\x01\n" +
	"\x16MachineSubmissionClose\x12+\n" +
	"\x05claim\x18\x01 \x01(\v2\x15.cozy.worker.v1.ClaimR\x05claim\x12#\n" +
	"\rsubmission_id\x18\x02 \x01(\tR\fsubmissionId\x12\x1d\n" +
	"\n" +
	"request_id\x18\x03 \x01(\tR\trequestId\x12E\n" +
	"\x1fexpected_execution_workspace_id\x18\x04 \x01(\tR\x1cexpectedExecutionWorkspaceId\"\xd7\x01\n" +
	"\x18MachineSubmissionClosure\x12#\n" +
	"\rsubmission_id\x18\x01 \x01(\tR\fsubmissionId\x12\x1d\n" +
	"\n" +
	"request_id\x18\x02 \x01(\tR\trequestId\x124\n" +
	"\x16execution_workspace_id\x18\x03 \x01(\tR\x14executionWorkspaceId\x12A\n" +
	"\areceipt\x18\x04 \x01(\v2'.cozy.worker.v1.MachineExecutionReceiptR\areceipt\"\x8f\x06\n" +
	"\x16MachineExecutionSubmit\x12+\n" +
	"\x05claim\x18\x01 \x01(\v2\x15.cozy.worker.v1.ClaimR\x05claim\x12#\n" +
	"\rsubmission_id\x18\x02 \x01(\tR\fsubmissionId\x12%\n" +
	"\x0ecapture_digest\x18\x03 \x01(\fR\rcaptureDigest\x126\n" +
	"\x17capture_canonical_bytes\x18\x04 \x01(\fR\x15captureCanonicalBytes\x122\n" +
	"\x05offer\x18\x05 \x01(\v2\x1c.cozy.worker.v1.AttemptOfferR\x05offer\x12I\n" +
	"\x0eprepared_state\x18\b \x01(\v2\".cozy.worker.v1.DesiredWorkerStateR\rpreparedState\x126\n" +
	"\x17payload_canonical_bytes\x18\t \x01(\fR\x15payloadCanonicalBytes\x12@\n" +
	"\x1cpublication_authorization_id\x18\n" +
	" \x01(\tR\x1apublicationAuthorizationId\x12E\n" +
	"\x1fexpected_execution_workspace_id\x18\v \x01(\tR\x1cexpectedExecutionWorkspaceId\x12O\n" +
	"\x12source_credentials\x18\f \x03(\v2 .cozy.worker.v1.SourceCredentialR\x11sourceCredentials\x12>\n" +
	"\frelease_root\x18\r \x01(\v2\x1b.cozy.worker.v1.ReleaseRootR\vreleaseRoot\x12\x1d\n" +
	"\n" +
	"owner_memo\x18\x0e \x01(\bR\townerMemo\x12\x18\n" +
	"\aaccount\x18\x0f \x01(\tR\aaccount\x12\x10\n" +
	"\x03hub\x18\x10 \x01(\tR\x03hubJ\x04\b\x06\x10\aJ\x04\b\a\x10\bR\fmax_attemptsR\x0eretry_delay_ms\"\x99\x05\n" +
	"\vReleaseRoot\x12\x18\n" +
	"\apackage\x18\x01 \x01(\tR\apackage\x12\x18\n" +
	"\arelease\x18\x02 \x01(\tR\arelease\x12\x1e\n" +
	"\n" +
	"entrypoint\x18\x03 \x01(\tR\n" +
	"entrypoint\x123\n" +
	"\x06models\x18\x06 \x03(\v2\x1b.cozy.worker.v1.ModelChoiceR\x06models\x124\n" +
	"\x06inputs\x18\a \x03(\v2\x1c.cozy.worker.v1.InputBindingR\x06inputs\x12>\n" +
	"\finput_access\x18\b \x03(\v2\x1b.cozy.worker.v1.InputAccessR\vinputAccess\x12(\n" +
	"\x10deadline_unix_ms\x18\t \x01(\x04R\x0edeadlineUnixMs\x12)\n" +
	"\x10attention_kernel\x18\n" +
	" \x01(\tR\x0fattentionKernel\x12;\n" +
	"\acapture\x18\v \x01(\v2!.cozy.worker.v1.ActivationCaptureR\acapture\x12\x10\n" +
	"\x03job\x18\r \x01(\bR\x03job\x12/\n" +
	"\x13weights_destination\x18\x0e \x01(\tR\x12weightsDestination\x12+\n" +
	"\x11publication_grant\x18\x0f \x01(\tR\x10publicationGrant\x12'\n" +
	"\x0finstallation_id\x18\x10 \x01(\tR\x0einstallationId\x12\x14\n" +
	"\x05owner\x18\x11 \x01(\tR\x05owner\x12\x10\n" +
	"\x03hub\x18\x12 \x01(\tR\x03hubJ\x04\b\x04\x10\x05J\x04\b\x05\x10\x06J\x04\b\f\x10\rR\acalleesR\rinstallationsR\x0ecatalog_origin\"\x9e\x02\n" +
	"\vModelChoice\x12\x1c\n" +
	"\tparameter\x18\x01 \x01(\tR\tparameter\x12\x1e\n" +
	"\n" +
	"repository\x18\x02 \x01(\tR\n" +
	"repository\x12\x18\n" +
	"\arelease\x18\x03 \x01(\tR\arelease\x12\x12\n" +
	"\x04lane\x18\x04 \x01(\tR\x04lane\x12/\n" +
	"\bmanifest\x18\x05 \x01(\v2\x13.cozy.worker.v1.RefR\bmanifest\x12\x16\n" +
	"\x06source\x18\x06 \x01(\tR\x06source\x12\x1a\n" +
	"\bprofiles\x18\a \x03(\tR\bprofiles\x12>\n" +
	"\badapters\x18\b \x03(\v2\".cozy.worker.v1.DownloadAdapterRefR\badapters\"u\n" +
	"\x10SourceCredential\x12A\n" +
	"\bprovider\x18\x01 \x01(\x0e2%.cozy.worker.v1.NativeSourceOperationR\bprovider\x12\x1e\n" +
	"\n" +
	"credential\x18\x02 \x01(\tR\n" +
	"credential\"\xb3\x03\n" +
	"\x17MachineExecutionReceipt\x12\x1d\n" +
	"\n" +
	"request_id\x18\x01 \x01(\tR\trequestId\x12#\n" +
	"\rsubmission_id\x18\x02 \x01(\tR\fsubmissionId\x12%\n" +
	"\x0ecapture_digest\x18\x03 \x01(\fR\rcaptureDigest\x124\n" +
	"\x16invocation_spec_digest\x18\x04 \x01(\fR\x14invocationSpecDigest\x12$\n" +
	"\x0eaccepted_at_ms\x18\x05 \x01(\x04R\facceptedAtMs\x12\x1b\n" +
	"\tworker_id\x18\x06 \x01(\tR\bworkerId\x12$\n" +
	"\x0eworker_boot_id\x18\a \x01(\tR\fworkerBootId\x124\n" +
	"\x16execution_workspace_id\x18\b \x01(\tR\x14executionWorkspaceId\x12@\n" +
	"\x1cpublication_authorization_id\x18\t \x01(\tR\x1apublicationAuthorizationId\x12\x16\n" +
	"\x06number\x18\n" +
	" \x01(\x04R\x06number\"\xaa\x01\n" +
	"\x15MachineExecutionQuery\x12+\n" +
	"\x05claim\x18\x01 \x01(\v2\x15.cozy.worker.v1.ClaimR\x05claim\x12\x1d\n" +
	"\n" +
	"request_id\x18\x02 \x01(\tR\trequestId\x12E\n" +
	"\x1fexpected_execution_workspace_id\x18\x03 \x01(\tR\x1cexpectedExecutionWorkspaceId\"\x8a\x05\n" +
	"\x15MachineExecutionState\x12\x1d\n" +
	"\n" +
	"request_id\x18\x01 \x01(\tR\trequestId\x12'\n" +
	"\x0fattempt_ordinal\x18\x02 \x01(\x04R\x0eattemptOrdinal\x12\x1e\n" +
	"\n" +
	"generation\x18\x03 \x01(\x04R\n" +
	"generation\x12\x14\n" +
	"\x05state\x18\x04 \x01(\tR\x05state\x12\x1c\n" +
	"\tcollected\x18\x05 \x01(\bR\tcollected\x12\x1a\n" +
	"\bsequence\x18\x06 \x01(\x04R\bsequence\x12\x1b\n" +
	"\tworker_id\x18\a \x01(\tR\bworkerId\x12$\n" +
	"\x0eworker_boot_id\x18\b \x01(\tR\fworkerBootId\x124\n" +
	"\x16execution_workspace_id\x18\t \x01(\tR\x14executionWorkspaceId\x125\n" +
	"\x03gpu\x18\n" +
	" \x01(\v2#.cozy.worker.v1.MachineExecutionGpuR\x03gpu\x12e\n" +
	"\x1bawaiting_source_credentials\x18\v \x03(\x0e2%.cozy.worker.v1.NativeSourceOperationR\x19awaitingSourceCredentials\x12\x16\n" +
	"\x06number\x18\f \x01(\x04R\x06number\x12$\n" +
	"\x0eaccepted_at_ms\x18\r \x01(\x04R\facceptedAtMs\x12$\n" +
	"\x0efinished_at_ms\x18\x0e \x01(\x04R\ffinishedAtMs\x12>\n" +
	"\x06target\x18\x0f \x01(\v2&.cozy.worker.v1.MachineExecutionTargetR\x06target\"\xa7\x01\n" +
	"\x16MachineExecutionTarget\x12\x18\n" +
	"\apackage\x18\x01 \x01(\tR\apackage\x12\x18\n" +
	"\arelease\x18\x02 \x01(\tR\arelease\x12\x1e\n" +
	"\n" +
	"entrypoint\x18\x03 \x01(\tR\n" +
	"entrypoint\x12'\n" +
	"\x0finstallation_id\x18\x04 \x01(\tR\x0einstallationId\x12\x10\n" +
	"\x03job\x18\x05 \x01(\bR\x03job\"|\n" +
	"\x13MachineExecutionGpu\x12\x14\n" +
	"\x05phase\x18\x01 \x01(\tR\x05phase\x12\x14\n" +
	"\x05width\x18\x02 \x01(\rR\x05width\x12\x1a\n" +
	"\bordinals\x18\x03 \x03(\rR\bordinals\x12\x1d\n" +
	"\n" +
	"blocked_by\x18\x04 \x03(\tR\tblockedBy\"\xa2\x01\n" +
	"\x1bMachineExecutionEventsQuery\x12C\n" +
	"\texecution\x18\x01 \x01(\v2%.cozy.worker.v1.MachineExecutionQueryR\texecution\x12\x14\n" +
	"\x05after\x18\x02 \x01(\x04R\x05after\x12\x14\n" +
	"\x05limit\x18\x03 \x01(\rR\x05limit\x12\x12\n" +
	"\x04wait\x18\x04 \x01(\bR\x04wait\"\xa7\x02\n" +
	"\x15MachineExecutionEvent\x12\x1a\n" +
	"\bsequence\x18\x01 \x01(\x04R\bsequence\x12'\n" +
	"\x0fattempt_ordinal\x18\x02 \x01(\x04R\x0eattemptOrdinal\x12\x13\n" +
	"\x05at_ms\x18\x03 \x01(\x04R\x04atMs\x12\x12\n" +
	"\x04kind\x18\x04 \x01(\tR\x04kind\x120\n" +
	"\x14body_canonical_bytes\x18\x05 \x01(\fR\x12bodyCanonicalBytes\x124\n" +
	"\aproduct\x18\x06 \x01(\v2\x1a.cozy.worker.v1.RunProductR\aproduct\x128\n" +
	"\aoutcome\x18\a \x01(\v2\x1e.cozy.worker.v1.AttemptOutcomeR\aoutcome\"\xc6\x02\n" +
	"\n" +
	"RunProduct\x12\x16\n" +
	"\x06output\x18\x01 \x01(\tR\x06output\x12,\n" +
	"\x02op\x18\x02 \x01(\x0e2\x1c.cozy.worker.v1.RunProductOpR\x02op\x12\x14\n" +
	"\x05index\x18\x03 \x01(\rR\x05index\x12-\n" +
	"\acontent\x18\x04 \x01(\v2\x13.cozy.worker.v1.RefR\acontent\x12\x1d\n" +
	"\n" +
	"media_type\x18\x05 \x01(\tR\tmediaType\x12\x14\n" +
	"\x05label\x18\x06 \x01(\tR\x05label\x12B\n" +
	"\x06source\x18\a \x01(\v2*.cozy.worker.v1.NativeByteRetentionRequestR\x06source\x124\n" +
	"\x05parts\x18\b \x03(\v2\x1e.cozy.worker.v1.RunProductPartR\x05parts\"\xa4\x01\n" +
	"\x0eRunProductPart\x12-\n" +
	"\acontent\x18\x01 \x01(\v2\x13.cozy.worker.v1.RefR\acontent\x12B\n" +
	"\x06source\x18\x02 \x01(\v2*.cozy.worker.v1.NativeByteRetentionRequestR\x06source\x12\x1f\n" +
	"\vduration_us\x18\x03 \x01(\x04R\n" +
	"durationUs\"\xcb\x01\n" +
	"\x19MachineExecutionEventPage\x12=\n" +
	"\x06events\x18\x01 \x03(\v2%.cozy.worker.v1.MachineExecutionEventR\x06events\x12\x1d\n" +
	"\n" +
	"next_after\x18\x02 \x01(\x04R\tnextAfter\x12#\n" +
	"\rhead_sequence\x18\x03 \x01(\x04R\fheadSequence\x12+\n" +
	"\x11compacted_through\x18\x04 \x01(\x04R\x10compactedThrough\"\xf9\x02\n" +
	"\x17MachineExecutionControl\x12C\n" +
	"\texecution\x18\x01 \x01(\v2%.cozy.worker.v1.MachineExecutionQueryR\texecution\x12\x1d\n" +
	"\n" +
	"command_id\x18\x02 \x01(\tR\tcommandId\x12/\n" +
	"\x13expected_generation\x18\x03 \x01(\x04R\x12expectedGeneration\x12>\n" +
	"\x06action\x18\x04 \x01(\x0e2&.cozy.worker.v1.MachineExecutionActionR\x06action\x12R\n" +
	"\vpublication\x18\x05 \x01(\v20.cozy.worker.v1.MachinePublicationReconciliationR\vpublication\x125\n" +
	"\x04memo\x18\x06 \x01(\v2!.cozy.worker.v1.MachineMemoAnswerR\x04memo\"\xa1\x01\n" +
	"\x11MachineMemoAnswer\x12'\n" +
	"\x0flookup_sequence\x18\x01 \x01(\x04R\x0elookupSequence\x12-\n" +
	"\x12computation_digest\x18\x02 \x01(\fR\x11computationDigest\x124\n" +
	"\x16result_canonical_bytes\x18\x03 \x01(\fR\x14resultCanonicalBytes\"\x86\x01\n" +
	" MachinePublicationReconciliation\x12\x1d\n" +
	"\n" +
	"call_index\x18\x01 \x01(\rR\tcallIndex\x12\x1f\n" +
	"\vhttp_status\x18\x02 \x01(\rR\n" +
	"httpStatus\x12\"\n" +
	"\ffinalization\x18\x03 \x01(\fR\ffinalization\"\x87\x01\n" +
	"\x17MachineExecutionCollect\x12C\n" +
	"\texecution\x18\x01 \x01(\v2%.cozy.worker.v1.MachineExecutionQueryR\texecution\x12'\n" +
	"\x0fattempt_ordinal\x18\x02 \x01(\x04R\x0eattemptOrdinal\"\xa1\x01\n" +
	"\x1dMachineExecutionCollectionAck\x12C\n" +
	"\texecution\x18\x01 \x01(\v2%.cozy.worker.v1.MachineExecutionQueryR\texecution\x12;\n" +
	"\aoutcome\x18\x02 \x01(\v2!.cozy.worker.v1.AttemptOutcomeAckR\aoutcome\"\x8b\x01\n" +
	"\x1bMachineExecutionTriageQuery\x12C\n" +
	"\texecution\x18\x01 \x01(\v2%.cozy.worker.v1.MachineExecutionQueryR\texecution\x12'\n" +
	"\x0fattempt_ordinal\x18\x02 \x01(\x04R\x0eattemptOrdinal\"\x87\x01\n" +
	"\x16MachineExecutionTriage\x127\n" +
	"\x06bundle\x18\x01 \x01(\v2\x1f.cozy.worker.v1.TriageBundleRefR\x06bundle\x124\n" +
	"\x16bundle_canonical_bytes\x18\x02 \x01(\fR\x14bundleCanonicalBytes\"\xf5\x01\n" +
	"\x19MachineExecutionListQuery\x12+\n" +
	"\x05claim\x18\x01 \x01(\v2\x15.cozy.worker.v1.ClaimR\x05claim\x12!\n" +
	"\fafter_number\x18\x02 \x01(\x04R\vafterNumber\x12!\n" +
	"\fnewest_first\x18\x03 \x01(\bR\vnewestFirst\x12#\n" +
	"\rbefore_number\x18\x04 \x01(\x04R\fbeforeNumber\x12\x14\n" +
	"\x05limit\x18\x05 \x01(\rR\x05limit\x12\x16\n" +
	"\x06states\x18\x06 \x03(\tR\x06states\x12\x12\n" +
	"\x04wait\x18\a \x01(\bR\x04wait\"\xb4\x01\n" +
	"\x14MachineExecutionList\x12E\n" +
	"\n" +
	"executions\x18\x01 \x03(\v2%.cozy.worker.v1.MachineExecutionStateR\n" +
	"executions\x12\x1f\n" +
	"\vhead_number\x18\x02 \x01(\x04R\n" +
	"headNumber\x124\n" +
	"\x16execution_workspace_id\x18\x03 \x01(\tR\x14executionWorkspaceId\"?\n" +
	"\x10PackageListQuery\x12+\n" +
	"\x05claim\x18\x01 \x01(\v2\x15.cozy.worker.v1.ClaimR\x05claim\"I\n" +
	"\vPackageList\x12:\n" +
	"\bpackages\x18\x01 \x03(\v2\x1e.cozy.worker.v1.MachinePackageR\bpackages\"\x84\x02\n" +
	"\x0eMachinePackage\x12'\n" +
	"\x0finstallation_id\x18\x01 \x01(\tR\x0einstallationId\x12\x18\n" +
	"\apackage\x18\x02 \x01(\tR\apackage\x12\x18\n" +
	"\arelease\x18\x03 \x01(\tR\arelease\x12\x16\n" +
	"\x06origin\x18\x04 \x01(\tR\x06origin\x12&\n" +
	"\x0finstalled_at_ms\x18\x05 \x01(\x04R\rinstalledAtMs\x123\n" +
	"\x03sdk\x18\x06 \x03(\v2!.cozy.worker.v1.ImageDistributionR\x03sdk\x12 \n" +
	"\ventrypoints\x18\a \x03(\tR\ventrypoints\"=\n" +
	"\x0eModelListQuery\x12+\n" +
	"\x05claim\x18\x01 \x01(\v2\x15.cozy.worker.v1.ClaimR\x05claim\"\xb8\x01\n" +
	"\tModelList\x124\n" +
	"\x06models\x18\x01 \x03(\v2\x1c.cozy.worker.v1.MachineModelR\x06models\x12C\n" +
	"\frepositories\x18\x02 \x03(\v2\x1f.cozy.worker.v1.RepositoryUsageR\frepositories\x120\n" +
	"\x05store\x18\x03 \x01(\v2\x1a.cozy.worker.v1.StoreUsageR\x05store\"\xe8\x01\n" +
	"\fMachineModel\x12\x12\n" +
	"\x04kind\x18\x01 \x01(\tR\x04kind\x12\x1e\n" +
	"\n" +
	"repository\x18\x02 \x01(\tR\n" +
	"repository\x12\x18\n" +
	"\aversion\x18\x03 \x01(\tR\aversion\x12\x12\n" +
	"\x04lane\x18\x04 \x01(\tR\x04lane\x12)\n" +
	"\x10source_selection\x18\x05 \x01(\tR\x0fsourceSelection\x12/\n" +
	"\bmanifest\x18\x06 \x01(\v2\x13.cozy.worker.v1.RefR\bmanifest\x12\x1a\n" +
	"\bcomplete\x18\a \x01(\bR\bcomplete\"u\n" +
	"\x0fRepositoryUsage\x12\x1e\n" +
	"\n" +
	"repository\x18\x01 \x01(\tR\n" +
	"repository\x12\x1f\n" +
	"\vbytes_total\x18\x02 \x01(\x04R\n" +
	"bytesTotal\x12!\n" +
	"\fbytes_unique\x18\x03 \x01(\x04R\vbytesUnique\"\xc9\x01\n" +
	"\n" +
	"StoreUsage\x12\x1f\n" +
	"\vbytes_total\x18\x01 \x01(\x04R\n" +
	"bytesTotal\x12(\n" +
	"\x10bytes_unique_sum\x18\x02 \x01(\x04R\x0ebytesUniqueSum\x12-\n" +
	"\x12bytes_unreferenced\x18\x03 \x01(\x04R\x11bytesUnreferenced\x12A\n" +
	"\n" +
	"filesystem\x18\x04 \x01(\v2!.cozy.worker.v1.MachineFilesystemR\n" +
	"filesystem\"q\n" +
	"\x11MachineFilesystem\x12\x12\n" +
	"\x04path\x18\x01 \x01(\tR\x04path\x12\x1f\n" +
	"\vtotal_bytes\x18\x02 \x01(\x04R\n" +
	"totalBytes\x12'\n" +
	"\x0favailable_bytes\x18\x03 \x01(\x04R\x0eavailableBytes\"C\n" +
	"\x14DescribeMachineQuery\x12+\n" +
	"\x05claim\x18\x01 \x01(\v2\x15.cozy.worker.v1.ClaimR\x05claim\"\xe9\x01\n" +
	"\x12MachineDescription\x12\x1b\n" +
	"\tworker_id\x18\x01 \x01(\tR\bworkerId\x12$\n" +
	"\x0eworker_boot_id\x18\x02 \x01(\tR\fworkerBootId\x12/\n" +
	"\x04host\x18\x03 \x01(\v2\x1b.cozy.worker.v1.MachineHostR\x04host\x128\n" +
	"\aruntime\x18\x04 \x01(\v2\x1e.cozy.worker.v1.MachineRuntimeR\aruntime\x12%\n" +
	"\x0eruntime_absent\x18\x05 \x01(\tR\rruntimeAbsent\"\xb6\x03\n" +
	"\vMachineHost\x12\x18\n" +
	"\aversion\x18\x01 \x01(\tR\aversion\x12\x1d\n" +
	"\n" +
	"wire_minor\x18\x02 \x01(\rR\twireMinor\x12,\n" +
	"\x12minimum_wire_minor\x18\x03 \x01(\rR\x10minimumWireMinor\x12\x1a\n" +
	"\bplatform\x18\x04 \x01(\tR\bplatform\x12\x1d\n" +
	"\n" +
	"os_release\x18\x05 \x01(\tR\tosRelease\x12\x1a\n" +
	"\bhostname\x18\x06 \x01(\tR\bhostname\x12\x14\n" +
	"\x05phase\x18\a \x01(\tR\x05phase\x121\n" +
	"\x15idle_deadline_unix_ms\x18\b \x01(\x04R\x12idleDeadlineUnixMs\x12.\n" +
	"\x04hubs\x18\t \x03(\v2\x1a.cozy.worker.v1.MachineHubR\x04hubs\x12C\n" +
	"\vfilesystems\x18\n" +
	" \x03(\v2!.cozy.worker.v1.MachineFilesystemR\vfilesystems\x12+\n" +
	"\x12started_at_unix_ms\x18\v \x01(\x04R\x0fstartedAtUnixMs\"C\n" +
	"\n" +
	"MachineHub\x12\x16\n" +
	"\x06origin\x18\x01 \x01(\tR\x06origin\x12\x1d\n" +
	"\n" +
	"machine_id\x18\x02 \x01(\tR\tmachineId\"\xa8\x06\n" +
	"\x0eMachineRuntime\x12\x18\n" +
	"\aversion\x18\x01 \x01(\tR\aversion\x12\x1d\n" +
	"\n" +
	"wire_minor\x18\x02 \x01(\rR\twireMinor\x12,\n" +
	"\x12minimum_wire_minor\x18\x03 \x01(\rR\x10minimumWireMinor\x12)\n" +
	"\x10tensorfs_version\x18\x04 \x01(\tR\x0ftensorfsVersion\x12%\n" +
	"\x0epython_version\x18\x05 \x01(\tR\rpythonVersion\x12#\n" +
	"\rtorch_version\x18\x06 \x01(\tR\ftorchVersion\x12\x1d\n" +
	"\n" +
	"uv_version\x18\a \x01(\tR\tuvVersion\x12/\n" +
	"\x13accelerator_backend\x18\b \x01(\tR\x12acceleratorBackend\x124\n" +
	"\x16executor_uid_isolation\x18\t \x01(\bR\x14executorUidIsolation\x127\n" +
	"\adevices\x18\n" +
	" \x03(\v2\x1d.cozy.worker.v1.MachineDeviceR\adevices\x12=\n" +
	"\tresources\x18\v \x01(\v2\x1f.cozy.worker.v1.WorkerResourcesR\tresources\x124\n" +
	"\x16execution_workspace_id\x18\f \x01(\tR\x14executionWorkspaceId\x12+\n" +
	"\x12started_at_unix_ms\x18\r \x01(\x04R\x0fstartedAtUnixMs\x127\n" +
	"\x05store\x18\x0e \x01(\v2!.cozy.worker.v1.MachineFilesystemR\x05store\x12E\n" +
	"\finterpreters\x18\x0f \x03(\v2!.cozy.worker.v1.PythonInterpreterR\finterpreters\x12W\n" +
	"\x14preparation_progress\x18\x10 \x03(\v2$.cozy.worker.v1.PrepareModelProgressR\x13preparationProgress\"\xd4\x02\n" +
	"\x12ProtocolInfoResult\x12\x1d\n" +
	"\n" +
	"wire_minor\x18\x01 \x01(\rR\twireMinor\x12,\n" +
	"\x12minimum_wire_minor\x18\x02 \x01(\rR\x10minimumWireMinorJ\x04\b\x03\x10\n" +
	"R\x1bsupports_mixed_model_inputsR'supports_model_materialization_recoveryR!supports_local_installation_reuseR\x19supports_rental_keepaliveR$supports_weights_transaction_refusalR!supports_machine_execution_triageR\x1esupports_prepare_while_landing\"\x8c\x04\n" +
	"\x15PreparePackageSetCall\x12+\n" +
	"\x05claim\x18\x01 \x01(\v2\x15.cozy.worker.v1.ClaimR\x05claim\x12B\n" +
	"\vpackage_set\x18\x02 \x01(\v2!.cozy.worker.v1.DesiredPackageSetR\n" +
	"packageSet\x12 \n" +
	"\vapplication\x18\x03 \x01(\tR\vapplication\x12(\n" +
	"\x10model_slot_paths\x18\x04 \x03(\tR\x0emodelSlotPaths\x12G\n" +
	"\x0fimage_inventory\x18\x05 \x01(\v2\x1e.cozy.worker.v1.ImageInventoryR\x0eimageInventory\x12/\n" +
	"\x13locked_requirements\x18\x06 \x01(\fR\x12lockedRequirements\x12'\n" +
	"\x0fpython_requires\x18\b \x01(\tR\x0epythonRequires\x12%\n" +
	"\x0epython_version\x18\t \x01(\tR\rpythonVersion\x12+\n" +
	"\x11package_interface\x18\n" +
	" \x01(\fR\x10packageInterface\x12\x10\n" +
	"\x03hub\x18\v \x01(\tR\x03hubJ\x04\b\a\x10\bR'supports_model_materialization_recovery\"\xac\x01\n" +
	"\x17PrepareLocalPackageCall\x12+\n" +
	"\x05claim\x18\x01 \x01(\v2\x15.cozy.worker.v1.ClaimR\x05claim\x12R\n" +
	"\x11local_package_set\x18\x02 \x01(\v2&.cozy.worker.v1.DesiredLocalPackageSetR\x0flocalPackageSet\x12\x10\n" +
	"\x03hub\x18\x03 \x01(\tR\x03hub\"\xd9\x01\n" +
	"\x1bPreparePrivatePlacementCall\x12+\n" +
	"\x05claim\x18\x01 \x01(\v2\x15.cozy.worker.v1.ClaimR\x05claim\x12^\n" +
	"\x15private_placement_set\x18\x02 \x01(\v2*.cozy.worker.v1.DesiredPrivatePlacementSetR\x13privatePlacementSetJ\x04\b\x03\x10\x04R'supports_model_materialization_recovery\"\xb4\x03\n" +
	"\fPrepareEvent\x122\n" +
	"\x05stage\x18\x01 \x01(\x0e2\x1c.cozy.worker.v1.PrepareStageR\x05stage\x12\x1f\n" +
	"\vtotal_bytes\x18\x02 \x01(\x04R\n" +
	"totalBytes\x12+\n" +
	"\x11transferred_bytes\x18\x03 \x01(\x04R\x10transferredBytes\x12H\n" +
	"\rplacement_set\x18\x04 \x01(\v2#.cozy.worker.v1.DesiredPlacementSetR\fplacementSet\x12\x1b\n" +
	"\tsafe_code\x18\x05 \x01(\tR\bsafeCode\x12\x1f\n" +
	"\vsafe_detail\x18\x06 \x01(\tR\n" +
	"safeDetail\x12K\n" +
	"\x0emodel_progress\x18\a \x03(\v2$.cozy.worker.v1.PrepareModelProgressR\rmodelProgress\x12M\n" +
	"\x11installed_package\x18\b \x01(\v2 .cozy.worker.v1.InstalledPackageR\x10installedPackage\"\x92\x02\n" +
	"\x14PrepareModelProgress\x126\n" +
	"\x05model\x18\x01 \x01(\v2 .cozy.worker.v1.DownloadModelRefR\x05model\x12\x1f\n" +
	"\vtotal_bytes\x18\x02 \x01(\x04R\n" +
	"totalBytes\x12+\n" +
	"\x11transferred_bytes\x18\x03 \x01(\x04R\x10transferredBytes\x12!\n" +
	"\forigin_bytes\x18\x04 \x01(\x04R\voriginBytes\x12!\n" +
	"\fcached_bytes\x18\x05 \x01(\x04R\vcachedBytes\x12.\n" +
	"\x13cache_written_bytes\x18\x06 \x01(\x04R\x11cacheWrittenBytes\"\x84\x01\n" +
	"\x13ModelSourceFileCall\x12+\n" +
	"\x05claim\x18\x01 \x01(\v2\x15.cozy.worker.v1.ClaimR\x05claim\x12@\n" +
	"\arequest\x18\x02 \x01(\v2&.cozy.worker.v1.ModelSourceFileRequestR\arequest\"\x8a\x01\n" +
	"\x16ModelSourcePrepareCall\x12+\n" +
	"\x05claim\x18\x01 \x01(\v2\x15.cozy.worker.v1.ClaimR\x05claim\x12C\n" +
	"\arequest\x18\x02 \x01(\v2).cozy.worker.v1.ModelSourcePrepareRequestR\arequest\"\xa0\x01\n" +
	"\x16ModelSourceReleaseCall\x12+\n" +
	"\x05claim\x18\x01 \x01(\v2\x15.cozy.worker.v1.ClaimR\x05claim\x12!\n" +
	"\foperation_id\x18\x02 \x01(\tR\voperationId\x126\n" +
	"\x17source_selection_digest\x18\x03 \x01(\fR\x15sourceSelectionDigest\">\n" +
	"\x19ReleaseModelSourceRequest\x12!\n" +
	"\foperation_id\x18\x01 \x01(\tR\voperationId\"\x1d\n" +
	"\x1bNumericalEnvironmentRequest\"\xe3\x01\n" +
	"\x16ModelSourceControlCall\x12+\n" +
	"\x05claim\x18\x01 \x01(\v2\x15.cozy.worker.v1.ClaimR\x05claim\x12!\n" +
	"\foperation_id\x18\x02 \x01(\tR\voperationId\x126\n" +
	"\x17source_selection_digest\x18\x03 \x01(\fR\x15sourceSelectionDigest\x12)\n" +
	"\x10control_revision\x18\x04 \x01(\x04R\x0fcontrolRevision\x12\x16\n" +
	"\x06paused\x18\x05 \x01(\bR\x06paused\"\xb8\x01\n" +
	"\x18ModelSourceControlResult\x12!\n" +
	"\foperation_id\x18\x01 \x01(\tR\voperationId\x126\n" +
	"\x17source_selection_digest\x18\x02 \x01(\fR\x15sourceSelectionDigest\x12)\n" +
	"\x10control_revision\x18\x03 \x01(\x04R\x0fcontrolRevision\x12\x16\n" +
	"\x06paused\x18\x04 \x01(\bR\x06paused\"G\n" +
	"\x18NumericalEnvironmentCall\x12+\n" +
	"\x05claim\x18\x01 \x01(\v2\x15.cozy.worker.v1.ClaimR\x05claim\"4\n" +
	"\x1aNumericalEnvironmentResult\x12\x16\n" +
	"\x06digest\x18\x01 \x01(\fR\x06digest\"\x86\x01\n" +
	"\x14DerivedRetentionCall\x12+\n" +
	"\x05claim\x18\x01 \x01(\v2\x15.cozy.worker.v1.ClaimR\x05claim\x12A\n" +
	"\arequest\x18\x02 \x01(\v2'.cozy.worker.v1.DerivedRetentionRequestR\arequest\"\xaa\x01\n" +
	"\x17DerivedRetentionRequest\x124\n" +
	"\x16weights_transaction_id\x18\x01 \x01(\tR\x14weightsTransactionId\x126\n" +
	"\x17tensorfs_receipt_digest\x18\x02 \x01(\fR\x15tensorfsReceiptDigest\x12!\n" +
	"\fretention_id\x18\x03 \x01(\tR\vretentionId\"\xf6\x01\n" +
	"\x16DerivedRetentionResult\x124\n" +
	"\x16weights_transaction_id\x18\x01 \x01(\tR\x14weightsTransactionId\x126\n" +
	"\x17tensorfs_receipt_digest\x18\x02 \x01(\fR\x15tensorfsReceiptDigest\x12!\n" +
	"\fretention_id\x18\x03 \x01(\tR\vretentionId\x12/\n" +
	"\bmanifest\x18\x04 \x01(\v2\x13.cozy.worker.v1.RefR\bmanifest\x12\x1a\n" +
	"\breleased\x18\x05 \x01(\bR\breleased\"\x8e\x01\n" +
	"\x18DerivedResultReleaseCall\x12+\n" +
	"\x05claim\x18\x01 \x01(\v2\x15.cozy.worker.v1.ClaimR\x05claim\x12E\n" +
	"\arequest\x18\x02 \x01(\v2+.cozy.worker.v1.DerivedResultReleaseRequestR\arequest\"\x8b\x01\n" +
	"\x1bDerivedResultReleaseRequest\x124\n" +
	"\x16weights_transaction_id\x18\x01 \x01(\tR\x14weightsTransactionId\x126\n" +
	"\x17tensorfs_receipt_digest\x18\x02 \x01(\fR\x15tensorfsReceiptDigest\"\xa6\x01\n" +
	"\x1aDerivedResultReleaseResult\x124\n" +
	"\x16weights_transaction_id\x18\x01 \x01(\tR\x14weightsTransactionId\x126\n" +
	"\x17tensorfs_receipt_digest\x18\x02 \x01(\fR\x15tensorfsReceiptDigest\x12\x1a\n" +
	"\breleased\x18\x03 \x01(\bR\breleased\"\xbb\x02\n" +
	"\x19RecordOperationResultCall\x12+\n" +
	"\x05claim\x18\x01 \x01(\v2\x15.cozy.worker.v1.ClaimR\x05claim\x12-\n" +
	"\x12computation_digest\x18\x02 \x01(\fR\x11computationDigest\x12\x1d\n" +
	"\n" +
	"request_id\x18\x03 \x01(\tR\trequestId\x12'\n" +
	"\x0fattempt_ordinal\x18\x04 \x01(\x04R\x0eattemptOrdinal\x124\n" +
	"\x16invocation_spec_digest\x18\x05 \x01(\fR\x14invocationSpecDigest\x12\x1d\n" +
	"\n" +
	"outcome_id\x18\x06 \x01(\tR\toutcomeId\x12%\n" +
	"\x0eoutcome_digest\x18\a \x01(\fR\routcomeDigest\"h\n" +
	"\x1bRecordOperationResultResult\x12-\n" +
	"\x12computation_digest\x18\x01 \x01(\fR\x11computationDigest\x12\x1a\n" +
	"\brecorded\x18\x02 \x01(\bR\brecorded\"\xa1\x01\n" +
	"\x13LookupOperationCall\x12+\n" +
	"\x05claim\x18\x01 \x01(\v2\x15.cozy.worker.v1.ClaimR\x05claim\x12-\n" +
	"\x12computation_digest\x18\x02 \x01(\fR\x11computationDigest\x12.\n" +
	"\x13consumer_request_id\x18\x03 \x01(\tR\x11consumerRequestId\"\xe0\x02\n" +
	"\x15LookupOperationResult\x12-\n" +
	"\x12computation_digest\x18\x01 \x01(\fR\x11computationDigest\x12\x14\n" +
	"\x05found\x18\x02 \x01(\bR\x05found\x126\n" +
	"\x06source\x18\x03 \x01(\v2\x1e.cozy.worker.v1.AttemptOutcomeR\x06source\x12F\n" +
	"\n" +
	"retentions\x18\x04 \x03(\v2&.cozy.worker.v1.DerivedRetentionResultR\n" +
	"retentions\x12.\n" +
	"\x13consumer_request_id\x18\x05 \x01(\tR\x11consumerRequestId\x12R\n" +
	"\x0fbyte_retentions\x18\a \x03(\v2).cozy.worker.v1.NativeByteRetentionResultR\x0ebyteRetentions\"F\n" +
	"\x17PruneOperationCacheCall\x12+\n" +
	"\x05claim\x18\x01 \x01(\v2\x15.cozy.worker.v1.ClaimR\x05claim\"\x8c\x01\n" +
	"\x19PruneOperationCacheResult\x12'\n" +
	"\x0fremoved_entries\x18\x01 \x01(\rR\x0eremovedEntries\x12'\n" +
	"\x0freclaimed_bytes\x18\x02 \x01(\x04R\x0ereclaimedBytes\x12\x1d\n" +
	"\n" +
	"store_busy\x18\x03 \x01(\bR\tstoreBusy\"\x1c\n" +
	"\x1aCollectStoreGarbageRequest\"c\n" +
	"\x19CollectStoreGarbageResult\x12'\n" +
	"\x0freclaimed_bytes\x18\x01 \x01(\x04R\x0ereclaimedBytes\x12\x1d\n" +
	"\n" +
	"store_busy\x18\x02 \x01(\bR\tstoreBusy\"Y\n" +
	"\x18ReleaseModelSourceResult\x12!\n" +
	"\foperation_id\x18\x01 \x01(\tR\voperationId\x12\x1a\n" +
	"\breleased\x18\x02 \x01(\bR\breleased\"\xb4\x01\n" +
	"\x14ModelSourceAdoptCall\x12+\n" +
	"\x05claim\x18\x01 \x01(\v2\x15.cozy.worker.v1.ClaimR\x05claim\x12*\n" +
	"\x11from_operation_id\x18\x02 \x01(\tR\x0ffromOperationId\x12C\n" +
	"\arequest\x18\x03 \x01(\v2).cozy.worker.v1.ModelSourcePrepareRequestR\arequest\"\x82\x01\n" +
	"\x12CheckpointPageCall\x12+\n" +
	"\x05claim\x18\x01 \x01(\v2\x15.cozy.worker.v1.ClaimR\x05claim\x12?\n" +
	"\arequest\x18\x02 \x01(\v2%.cozy.worker.v1.CheckpointPageRequestR\arequest\"\x8a\x01\n" +
	"\x16CheckpointTransferCall\x12+\n" +
	"\x05claim\x18\x01 \x01(\v2\x15.cozy.worker.v1.ClaimR\x05claim\x12C\n" +
	"\arequest\x18\x02 \x01(\v2).cozy.worker.v1.CheckpointTransferRequestR\arequest\"\xa6\x01\n" +
	"\x17LocalPackageUploadFrame\x12B\n" +
	"\x06header\x18\x01 \x01(\v2(.cozy.worker.v1.LocalPackageUploadHeaderH\x00R\x06header\x12?\n" +
	"\x05chunk\x18\x02 \x01(\v2'.cozy.worker.v1.LocalPackageUploadChunkH\x00R\x05chunkB\x06\n" +
	"\x04body\"\xb8\x01\n" +
	"\x18LocalPackageUploadHeader\x12+\n" +
	"\x05claim\x18\x01 \x01(\v2\x15.cozy.worker.v1.ClaimR\x05claim\x12!\n" +
	"\foperation_id\x18\x02 \x01(\tR\voperationId\x127\n" +
	"\x04file\x18\x04 \x01(\v2#.cozy.worker.v1.LocalPackageFileRefR\x04fileJ\x04\b\x03\x10\x04R\rsource_digest\"E\n" +
	"\x17LocalPackageUploadChunk\x12\x16\n" +
	"\x06offset\x18\x01 \x01(\x04R\x06offset\x12\x12\n" +
	"\x04data\x18\x02 \x01(\fR\x04data\"\xd3\x04\n" +
	"\x18PreparePackageSetRequest\x12/\n" +
	"\x13download_delegation\x18\x01 \x01(\fR\x12downloadDelegation\x12 \n" +
	"\vapplication\x18\x04 \x01(\tR\vapplication\x12(\n" +
	"\x10model_slot_paths\x18\x05 \x03(\tR\x0emodelSlotPaths\x12G\n" +
	"\x0fimage_inventory\x18\x06 \x01(\v2\x1e.cozy.worker.v1.ImageInventoryR\x0eimageInventory\x12/\n" +
	"\x13locked_requirements\x18\a \x01(\fR\x12lockedRequirements\x12!\n" +
	"\finstall_root\x18\b \x01(\tR\vinstallRoot\x12B\n" +
	"\x1ddownload_delegation_signature\x18\t \x01(\fR\x1bdownloadDelegationSignature\x12'\n" +
	"\x0fpython_requires\x18\n" +
	" \x01(\tR\x0epythonRequires\x12%\n" +
	"\x0epython_version\x18\v \x01(\tR\rpythonVersion\x12+\n" +
	"\x11package_interface\x18\f \x01(\fR\x10packageInterface\x12%\n" +
	"\x0emodels_landing\x18\r \x01(\bR\rmodelsLanding\x12\x10\n" +
	"\x03hub\x18\x0e \x01(\tR\x03hubJ\x04\b\x02\x10\x03J\x04\b\x03\x10\x04R\x05filesR\x10environment_root\"\xd2\x01\n" +
	"\x0eImageInventory\x12\x18\n" +
	"\aprofile\x18\x01 \x01(\tR\aprofile\x12\x16\n" +
	"\x06python\x18\x02 \x01(\tR\x06python\x12G\n" +
	"\rdistributions\x18\x03 \x03(\v2!.cozy.worker.v1.ImageDistributionR\rdistributions\x12E\n" +
	"\finterpreters\x18\x04 \x03(\v2!.cozy.worker.v1.PythonInterpreterR\finterpreters\"?\n" +
	"\x11PythonInterpreter\x12\x18\n" +
	"\aversion\x18\x01 \x01(\tR\aversion\x12\x10\n" +
	"\x03abi\x18\x02 \x01(\tR\x03abi\"Q\n" +
	"\x11ImageDistribution\x12\"\n" +
	"\fdistribution\x18\x01 \x01(\tR\fdistribution\x12\x18\n" +
	"\aversion\x18\x02 \x01(\tR\aversion\"\xb2\x01\n" +
	"\x17PreparePackageSetResult\x12H\n" +
	"\rplacement_set\x18\x01 \x01(\v2#.cozy.worker.v1.DesiredPlacementSetR\fplacementSet\x12M\n" +
	"\x11installed_package\x18\x02 \x01(\v2 .cozy.worker.v1.InstalledPackageR\x10installedPackage\"\xc0\x03\n" +
	"\x1aPrepareLocalPackageRequest\x12!\n" +
	"\foperation_id\x18\x01 \x01(\tR\voperationId\x12<\n" +
	"\apackage\x18\x02 \x01(\v2\".cozy.worker.v1.DevelopmentPackageR\apackage\x126\n" +
	"\x05files\x18\x05 \x03(\v2 .cozy.worker.v1.LocalPackageFileR\x05files\x12!\n" +
	"\finstall_root\x18\x06 \x01(\tR\vinstallRoot\x127\n" +
	"\x17dependency_requirements\x18\a \x01(\fR\x16dependencyRequirements\x12'\n" +
	"\x0fpython_requires\x18\b \x01(\tR\x0epythonRequires\x12%\n" +
	"\x0epython_version\x18\t \x01(\tR\rpythonVersion\x12%\n" +
	"\x0esource_archive\x18\n" +
	" \x01(\tR\rsourceArchive\x12\x10\n" +
	"\x03hub\x18\v \x01(\tR\x03hubJ\x04\b\x03\x10\x04J\x04\b\x04\x10\x05R\x10environment_rootR\x06wheels\"r\n" +
	"\x10LocalPackageFile\x12\x16\n" +
	"\x06digest\x18\x01 \x01(\fR\x06digest\x12\x1a\n" +
	"\bfilename\x18\x02 \x01(\tR\bfilename\x12\x16\n" +
	"\x06length\x18\x03 \x01(\x04R\x06length\x12\x12\n" +
	"\x04path\x18\x04 \x01(\tR\x04path\"\xbc\x04\n" +
	"\x1ePreparePrivatePlacementRequest\x12!\n" +
	"\foperation_id\x18\x01 \x01(\tR\voperationId\x12/\n" +
	"\x13download_delegation\x18\x03 \x01(\fR\x12downloadDelegation\x12B\n" +
	"\x1ddownload_delegation_signature\x18\x05 \x01(\fR\x1bdownloadDelegationSignature\x12G\n" +
	"\rnative_models\x18\x06 \x03(\v2\".cozy.worker.v1.NativeModelBindingR\fnativeModels\x12+\n" +
	"\x05claim\x18\a \x01(\v2\x15.cozy.worker.v1.ClaimR\x05claim\x12'\n" +
	"\x0finstallation_id\x18\b \x01(\tR\x0einstallationId\x12@\n" +
	"\rmodel_choices\x18\t \x03(\v2\x1b.cozy.worker.v1.ModelChoiceR\fmodelChoices\x12O\n" +
	"\x12source_credentials\x18\n" +
	" \x03(\v2 .cozy.worker.v1.SourceCredentialR\x11sourceCredentials\x12\x10\n" +
	"\x03hub\x18\v \x01(\tR\x03hub\x12\x14\n" +
	"\x05owner\x18\f \x01(\tR\x05ownerJ\x04\b\x04\x10\x05J\x04\b\x02\x10\x03R\x05filesR\x15local_revision_digest\"\xb4\x01\n" +
	"\x14LocalModelSourceFile\x12\x16\n" +
	"\x06member\x18\x01 \x01(\tR\x06member\x12\x1b\n" +
	"\tobject_id\x18\x02 \x01(\tR\bobjectId\x12\x16\n" +
	"\x06length\x18\x03 \x01(\x04R\x06length\x12\x12\n" +
	"\x04path\x18\x04 \x01(\tR\x04path\x12\x1f\n" +
	"\vheader_path\x18\x05 \x01(\tR\n" +
	"headerPath\x12\x1a\n" +
	"\bverified\x18\x06 \x01(\bR\bverified\"B\n" +
	"\x12ModelSourceProfile\x12\x12\n" +
	"\x04slot\x18\x01 \x01(\tR\x04slot\x12\x18\n" +
	"\aprofile\x18\x02 \x01(\tR\aprofile\"\x9f\x01\n" +
	"\x13PreparedModelSource\x12\x12\n" +
	"\x04slot\x18\x01 \x01(\tR\x04slot\x12\x18\n" +
	"\aprofile\x18\x02 \x01(\tR\aprofile\x12/\n" +
	"\bmanifest\x18\x03 \x01(\v2\x13.cozy.worker.v1.RefR\bmanifestJ\x04\b\x04\x10\x05R#checkpoint_evidence_canonical_bytes\"\xa1\x01\n" +
	"\x15ModelSourceCheckpoint\x12\x12\n" +
	"\x04slot\x18\x01 \x01(\tR\x04slot\x12'\n" +
	"\x04head\x18\x02 \x01(\v2\x13.cozy.worker.v1.RefR\x04head\x12\x1f\n" +
	"\vplan_digest\x18\x03 \x01(\fR\n" +
	"planDigest\x12\x14\n" +
	"\x05index\x18\x04 \x01(\x04R\x05index\x12\x14\n" +
	"\x05bytes\x18\x05 \x01(\x04R\x05bytes\"\x88\x01\n" +
	"\x17SourceCheckpointSubject\x12!\n" +
	"\foperation_id\x18\x01 \x01(\tR\voperationId\x126\n" +
	"\x17source_selection_digest\x18\x02 \x01(\fR\x15sourceSelectionDigest\x12\x12\n" +
	"\x04slot\x18\x03 \x01(\tR\x04slot\"\xa9\x02\n" +
	"\x18WeightsCheckpointSubject\x12\x1d\n" +
	"\n" +
	"request_id\x18\x01 \x01(\tR\trequestId\x124\n" +
	"\x16invocation_spec_digest\x18\x02 \x01(\fR\x14invocationSpecDigest\x12\x1f\n" +
	"\voutput_slot\x18\x03 \x01(\tR\n" +
	"outputSlot\x124\n" +
	"\x16weights_transaction_id\x18\x04 \x01(\tR\x14weightsTransactionId\x12!\n" +
	"\fwriter_epoch\x18\x05 \x01(\x04R\vwriterEpoch\x12>\n" +
	"\x1btensorfs_declaration_digest\x18\x06 \x01(\fR\x19tensorfsDeclarationDigest\"\xa4\x01\n" +
	"\x11CheckpointSubject\x12A\n" +
	"\x06source\x18\x01 \x01(\v2'.cozy.worker.v1.SourceCheckpointSubjectH\x00R\x06source\x12D\n" +
	"\aweights\x18\x02 \x01(\v2(.cozy.worker.v1.WeightsCheckpointSubjectH\x00R\aweightsB\x06\n" +
	"\x04kind\"U\n" +
	"\x10CheckpointObject\x12%\n" +
	"\x03ref\x18\x01 \x01(\v2\x13.cozy.worker.v1.RefR\x03ref\x12\x1a\n" +
	"\bmanifest\x18\x02 \x01(\bR\bmanifest\"\xe4\x02\n" +
	"\x15CheckpointPageRequest\x12,\n" +
	"\x12record_owner_epoch\x18\x01 \x01(\x04R\x10recordOwnerEpoch\x120\n" +
	"\x14control_stream_epoch\x18\x02 \x01(\x04R\x12controlStreamEpoch\x12$\n" +
	"\x0eworker_boot_id\x18\x03 \x01(\tR\fworkerBootId\x12;\n" +
	"\asubject\x18\x05 \x01(\v2!.cozy.worker.v1.CheckpointSubjectR\asubject\x12\x1f\n" +
	"\vplan_digest\x18\b \x01(\fR\n" +
	"planDigest\x12'\n" +
	"\x04head\x18\t \x01(\v2\x13.cozy.worker.v1.RefR\x04head\x12\x16\n" +
	"\x06offset\x18\n" +
	" \x01(\rR\x06offset\x12\x14\n" +
	"\x05limit\x18\v \x01(\rR\x05limitJ\x04\b\x04\x10\x05J\x04\b\x06\x10\aJ\x04\b\a\x10\b\"\xf9\x04\n" +
	"\x14CheckpointPageResult\x12,\n" +
	"\x12record_owner_epoch\x18\x01 \x01(\x04R\x10recordOwnerEpoch\x120\n" +
	"\x14control_stream_epoch\x18\x02 \x01(\x04R\x12controlStreamEpoch\x12$\n" +
	"\x0eworker_boot_id\x18\x03 \x01(\tR\fworkerBootId\x12;\n" +
	"\asubject\x18\x05 \x01(\v2!.cozy.worker.v1.CheckpointSubjectR\asubject\x12\x1f\n" +
	"\vplan_digest\x18\b \x01(\fR\n" +
	"planDigest\x12'\n" +
	"\x04head\x18\t \x01(\v2\x13.cozy.worker.v1.RefR\x04head\x12\x14\n" +
	"\x05index\x18\n" +
	" \x01(\x04R\x05index\x12\x14\n" +
	"\x05bytes\x18\v \x01(\x04R\x05bytes\x12/\n" +
	"\bprevious\x18\f \x01(\v2\x13.cozy.worker.v1.RefR\bprevious\x12/\n" +
	"\bprogress\x18\r \x01(\v2\x13.cozy.worker.v1.RefR\bprogress\x12:\n" +
	"\aobjects\x18\x0e \x03(\v2 .cozy.worker.v1.CheckpointObjectR\aobjects\x12\x1f\n" +
	"\vnext_offset\x18\x0f \x01(\rR\n" +
	"nextOffset\x12\x19\n" +
	"\bhas_more\x18\x10 \x01(\bR\ahasMore\x12\x1b\n" +
	"\tsafe_code\x18\x11 \x01(\tR\bsafeCode\x12\x1f\n" +
	"\vsafe_detail\x18\x12 \x01(\tR\n" +
	"safeDetailJ\x04\b\x04\x10\x05J\x04\b\x06\x10\aJ\x04\b\a\x10\b\"\xb6\x04\n" +
	"\x19CheckpointTransferRequest\x12,\n" +
	"\x12record_owner_epoch\x18\x01 \x01(\x04R\x10recordOwnerEpoch\x120\n" +
	"\x14control_stream_epoch\x18\x02 \x01(\x04R\x12controlStreamEpoch\x12$\n" +
	"\x0eworker_boot_id\x18\x03 \x01(\tR\fworkerBootId\x12;\n" +
	"\asubject\x18\x05 \x01(\v2!.cozy.worker.v1.CheckpointSubjectR\asubject\x12\x1f\n" +
	"\vplan_digest\x18\b \x01(\fR\n" +
	"planDigest\x12'\n" +
	"\x04head\x18\t \x01(\v2\x13.cozy.worker.v1.RefR\x04head\x128\n" +
	"\x06object\x18\n" +
	" \x01(\v2 .cozy.worker.v1.CheckpointObjectR\x06object\x12\x1f\n" +
	"\vtransfer_id\x18\v \x01(\tR\n" +
	"transferId\x12%\n" +
	"\x0egrant_revision\x18\f \x01(\x04R\rgrantRevision\x12G\n" +
	"\fupload_grant\x18\r \x01(\v2\".cozy.worker.v1.WeightsUploadGrantH\x00R\vuploadGrant\x12#\n" +
	"\fdownload_url\x18\x0e \x01(\tH\x00R\vdownloadUrlB\n" +
	"\n" +
	"\bdecisionJ\x04\b\x04\x10\x05J\x04\b\x06\x10\aJ\x04\b\a\x10\b\"\xac\x05\n" +
	"\x18CheckpointTransferStatus\x12,\n" +
	"\x12record_owner_epoch\x18\x01 \x01(\x04R\x10recordOwnerEpoch\x120\n" +
	"\x14control_stream_epoch\x18\x02 \x01(\x04R\x12controlStreamEpoch\x12$\n" +
	"\x0eworker_boot_id\x18\x03 \x01(\tR\fworkerBootId\x12;\n" +
	"\asubject\x18\x05 \x01(\v2!.cozy.worker.v1.CheckpointSubjectR\asubject\x12\x1f\n" +
	"\vplan_digest\x18\b \x01(\fR\n" +
	"planDigest\x12'\n" +
	"\x04head\x18\t \x01(\v2\x13.cozy.worker.v1.RefR\x04head\x128\n" +
	"\x06object\x18\n" +
	" \x01(\v2 .cozy.worker.v1.CheckpointObjectR\x06object\x12\x1f\n" +
	"\vtransfer_id\x18\v \x01(\tR\n" +
	"transferId\x12%\n" +
	"\x0egrant_revision\x18\f \x01(\x04R\rgrantRevision\x12:\n" +
	"\x05state\x18\r \x01(\x0e2$.cozy.worker.v1.WeightsTransferStateR\x05state\x12+\n" +
	"\x11transferred_bytes\x18\x0e \x01(\x04R\x10transferredBytes\x12\x1f\n" +
	"\vhttp_status\x18\x0f \x01(\rR\n" +
	"httpStatus\x12'\n" +
	"\x0fchecksum_sha256\x18\x10 \x01(\tR\x0echecksumSha256\x12\x1b\n" +
	"\tsafe_code\x18\x11 \x01(\tR\bsafeCode\x12\x1f\n" +
	"\vsafe_detail\x18\x12 \x01(\tR\n" +
	"safeDetailJ\x04\b\x04\x10\x05J\x04\b\x06\x10\aJ\x04\b\a\x10\b\"\xbc\x03\n" +
	"\x19PrepareModelSourceRequest\x12!\n" +
	"\foperation_id\x18\x01 \x01(\tR\voperationId\x126\n" +
	"\x17source_selection_digest\x18\x02 \x01(\fR\x15sourceSelectionDigest\x12>\n" +
	"\bprofiles\x18\x03 \x03(\v2\".cozy.worker.v1.ModelSourceProfileR\bprofiles\x12:\n" +
	"\x05files\x18\x04 \x03(\v2$.cozy.worker.v1.LocalModelSourceFileR\x05files\x12\x1d\n" +
	"\n" +
	"source_uri\x18\x05 \x01(\tR\tsourceUri\x12)\n" +
	"\x10declared_license\x18\x06 \x01(\tR\x0fdeclaredLicense\x12G\n" +
	"\vcheckpoints\x18\a \x03(\v2%.cozy.worker.v1.ModelSourceCheckpointR\vcheckpoints\x125\n" +
	"\x17adopt_from_operation_id\x18\b \x01(\tR\x14adoptFromOperationId\"\xca\x02\n" +
	"\x18PrepareModelSourceResult\x12C\n" +
	"\aoutcome\x18\x01 \x01(\x0e2).cozy.worker.v1.ModelSourcePrepareOutcomeR\aoutcome\x12=\n" +
	"\asources\x18\x02 \x03(\v2#.cozy.worker.v1.PreparedModelSourceR\asources\x12\x1b\n" +
	"\tsafe_code\x18\x03 \x01(\tR\bsafeCode\x12\x1f\n" +
	"\vsafe_detail\x18\x04 \x01(\tR\n" +
	"safeDetail\x12G\n" +
	"\vcheckpoints\x18\x05 \x03(\v2%.cozy.worker.v1.ModelSourceCheckpointR\vcheckpoints\x12#\n" +
	"\rspent_members\x18\x06 \x03(\tR\fspentMembers\"\x98\x06\n" +
	"\x10RecordOwnerFrame\x12-\n" +
	"\x05claim\x18\x05 \x01(\v2\x15.cozy.worker.v1.ClaimH\x00R\x05claim\x12I\n" +
	"\rdesired_state\x18\x06 \x01(\v2\".cozy.worker.v1.DesiredWorkerStateH\x00R\fdesiredState\x12@\n" +
	"\fsnapshot_ack\x18\v \x01(\v2\x1b.cozy.worker.v1.SnapshotAckH\x00R\vsnapshotAckB\x05\n" +
	"\x03msgJ\x04\b\x0e\x10\x0fJ\x04\b\x10\x10\x11J\x04\b\x12\x10\x13J\x04\b\x18\x10\x19J\x04\b\x13\x10\x14J\x04\b\x15\x10\x16J\x04\b\x1a\x10\x1bJ\x04\b\x1b\x10\x1cJ\x04\b\a\x10\bJ\x04\b\b\x10\tJ\x04\b\t\x10\n" +
	"J\x04\b\n" +
	"\x10\vJ\x04\b\x11\x10\x12J\x04\b\x14\x10\x15J\x04\b\x16\x10\x17J\x04\b\x17\x10\x18J\x04\b\x19\x10\x1aJ\x04\b\x1c\x10\x1dJ\x04\b\x1d\x10\x1eJ\x04\b\x1e\x10\x1fJ\x04\b\x1f\x10 R\x15artifact_grant_updateR\x10ensure_artifactsR\x1aprivate_package_file_chunkR\x10weights_host_ackR\x14weights_read_requestR\x16weights_upload_requestR\x1blocal_package_fetch_requestR\rattempt_offerR\x0ecancel_attemptR\voutcome_ackR\x12checkpoint_receiptR\x18weights_finalize_requestR\x18weights_transfer_requestR\x19model_source_file_requestR\x1cmodel_source_prepare_requestR\x13local_package_abortR\x11child_call_resultR\x15native_source_commandR\x18native_artifact_transferR\x1bweights_transaction_refused\"\xda\x06\n" +
	"\vWorkerFrame\x127\n" +
	"\tclaim_ack\x18\x05 \x01(\v2\x18.cozy.worker.v1.ClaimAckH\x00R\bclaimAck\x12L\n" +
	"\x0eobserved_state\x18\x06 \x01(\v2#.cozy.worker.v1.ObservedWorkerStateH\x00R\robservedState\x12@\n" +
	"\fboot_failure\x18\t \x01(\v2\x1b.cozy.worker.v1.BootFailureH\x00R\vbootFailure\x12<\n" +
	"\bsnapshot\x18\f \x01(\v2\x1e.cozy.worker.v1.WorkerSnapshotH\x00R\bsnapshotB\x05\n" +
	"\x03msgJ\x04\b\r\x10\x0eJ\x04\b\x0e\x10\x0fJ\x04\b\x0f\x10\x10J\x04\b\x13\x10\x14J\x04\b\x17\x10\x18J\x04\b\x19\x10\x1aJ\x04\b\a\x10\bJ\x04\b\b\x10\tJ\x04\b\n" +
	"\x10\vJ\x04\b\v\x10\fJ\x04\b\x10\x10\x11J\x04\b\x11\x10\x12J\x04\b\x12\x10\x13J\x04\b\x14\x10\x15J\x04\b\x15\x10\x16J\x04\b\x16\x10\x17J\x04\b\x18\x10\x19J\x04\b\x1a\x10\x1bJ\x04\b\x1b\x10\x1cJ\x04\b\x1c\x10\x1dJ\x04\b\x1d\x10\x1eJ\x04\b\x1e\x10\x1fJ\x04\b\x1f\x10 R\x13weights_read_resultR\x15weights_upload_resultR\x19local_package_file_statusR\x10attempt_acceptedR\x0fattempt_outcomeR\x12checkpoint_requestR\x0echeckpoint_ackR\x17weights_finalize_resultR\x0eweights_intentR\x0fweights_receiptR\x17weights_transfer_statusR\x18model_source_file_statusR\x15model_source_preparedR\x1alocal_package_abort_statusR\x12weights_checkpointR\x13weights_transactionR\x12child_call_requestR\x11child_call_cancelR\x14native_source_statusR\x1fnative_artifact_transfer_status\"\xbe\x04\n" +
	"\x10ChildCallRequest\x12,\n" +
	"\x12record_owner_epoch\x18\x01 \x01(\x04R\x10recordOwnerEpoch\x120\n" +
	"\x14control_stream_epoch\x18\x02 \x01(\x04R\x12controlStreamEpoch\x12$\n" +
	"\x0eworker_boot_id\x18\x03 \x01(\tR\fworkerBootId\x12*\n" +
	"\x11parent_request_id\x18\x04 \x01(\tR\x0fparentRequestId\x124\n" +
	"\x16parent_attempt_ordinal\x18\x05 \x01(\x04R\x14parentAttemptOrdinal\x12A\n" +
	"\x1dparent_invocation_spec_digest\x18\x06 \x01(\fR\x1aparentInvocationSpecDigest\x12\x1d\n" +
	"\n" +
	"call_index\x18\a \x01(\rR\tcallIndex\x12\x16\n" +
	"\x06module\x18\t \x01(\tR\x06module\x12\x16\n" +
	"\x06export\x18\n" +
	" \x01(\tR\x06export\x126\n" +
	"\x17request_canonical_bytes\x18\v \x01(\fR\x15requestCanonicalBytes\x12#\n" +
	"\rintent_digest\x18\f \x01(\fR\fintentDigest\x12;\n" +
	"\acapture\x18\r \x01(\v2!.cozy.worker.v1.ActivationCaptureR\acaptureJ\x04\b\b\x10\tR\x10interface_digest\"\xf0\x05\n" +
	"\x0fChildCallResult\x12,\n" +
	"\x12record_owner_epoch\x18\x01 \x01(\x04R\x10recordOwnerEpoch\x120\n" +
	"\x14control_stream_epoch\x18\x02 \x01(\x04R\x12controlStreamEpoch\x12$\n" +
	"\x0eworker_boot_id\x18\x03 \x01(\tR\fworkerBootId\x12*\n" +
	"\x11parent_request_id\x18\x04 \x01(\tR\x0fparentRequestId\x124\n" +
	"\x16parent_attempt_ordinal\x18\x05 \x01(\x04R\x14parentAttemptOrdinal\x12A\n" +
	"\x1dparent_invocation_spec_digest\x18\x06 \x01(\fR\x1aparentInvocationSpecDigest\x12\x1d\n" +
	"\n" +
	"call_index\x18\a \x01(\rR\tcallIndex\x12#\n" +
	"\rintent_digest\x18\b \x01(\fR\fintentDigest\x12(\n" +
	"\x10child_request_id\x18\t \x01(\tR\x0echildRequestId\x124\n" +
	"\x05state\x18\n" +
	" \x01(\x0e2\x1e.cozy.worker.v1.ChildCallStateR\x05state\x124\n" +
	"\x16result_canonical_bytes\x18\v \x01(\fR\x14resultCanonicalBytes\x12\x1b\n" +
	"\tsafe_code\x18\f \x01(\tR\bsafeCode\x12\x1f\n" +
	"\vsafe_detail\x18\r \x01(\tR\n" +
	"safeDetail\x12R\n" +
	"\x12byte_result_grants\x18\x0e \x03(\v2$.cozy.worker.v1.ChildByteResultGrantR\x10byteResultGrants\x12F\n" +
	"\vobservation\x18\x0f \x01(\v2$.cozy.worker.v1.ExecutionObservationR\vobservation\"k\n" +
	"\x12NativeSourceMember\x12\x16\n" +
	"\x06member\x18\x01 \x01(\tR\x06member\x12+\n" +
	"\x06object\x18\x02 \x01(\v2\x13.cozy.worker.v1.RefR\x06object\x12\x10\n" +
	"\x03url\x18\x03 \x01(\tR\x03url\"\xae\x02\n" +
	"\x15NativeSourceSelection\x12\x1c\n" +
	"\tcanonical\x18\x01 \x01(\tR\tcanonical\x12)\n" +
	"\x10selection_digest\x18\x02 \x01(\fR\x0fselectionDigest\x12>\n" +
	"\x10content_manifest\x18\x03 \x01(\v2\x13.cozy.worker.v1.RefR\x0fcontentManifest\x12<\n" +
	"\amembers\x18\x04 \x03(\v2\".cozy.worker.v1.NativeSourceMemberR\amembers\x12#\n" +
	"\rallowed_hosts\x18\x05 \x03(\tR\fallowedHosts\x12)\n" +
	"\x10credential_hosts\x18\x06 \x03(\tR\x0fcredentialHosts\"\xe0\x03\n" +
	"\x13NativeSourceCommand\x12,\n" +
	"\x12record_owner_epoch\x18\x01 \x01(\x04R\x10recordOwnerEpoch\x120\n" +
	"\x14control_stream_epoch\x18\x02 \x01(\x04R\x12controlStreamEpoch\x12$\n" +
	"\x0eworker_boot_id\x18\x03 \x01(\tR\fworkerBootId\x12A\n" +
	"\vparent_call\x18\x04 \x01(\v2 .cozy.worker.v1.ChildCallRequestR\n" +
	"parentCall\x12\x1d\n" +
	"\n" +
	"service_id\x18\x05 \x01(\tR\tserviceId\x12C\n" +
	"\toperation\x18\x06 \x01(\x0e2%.cozy.worker.v1.NativeSourceOperationR\toperation\x127\n" +
	"\x05phase\x18\a \x01(\x0e2!.cozy.worker.v1.NativeSourcePhaseR\x05phase\x12C\n" +
	"\tselection\x18\b \x01(\v2%.cozy.worker.v1.NativeSourceSelectionR\tselection\x12\x1e\n" +
	"\n" +
	"credential\x18\t \x01(\tR\n" +
	"credential\"\xf2\a\n" +
	"\x12NativeSourceStatus\x12,\n" +
	"\x12record_owner_epoch\x18\x01 \x01(\x04R\x10recordOwnerEpoch\x120\n" +
	"\x14control_stream_epoch\x18\x02 \x01(\x04R\x12controlStreamEpoch\x12$\n" +
	"\x0eworker_boot_id\x18\x03 \x01(\tR\fworkerBootId\x12*\n" +
	"\x11parent_request_id\x18\x04 \x01(\tR\x0fparentRequestId\x124\n" +
	"\x16parent_attempt_ordinal\x18\x05 \x01(\x04R\x14parentAttemptOrdinal\x12A\n" +
	"\x1dparent_invocation_spec_digest\x18\x06 \x01(\fR\x1aparentInvocationSpecDigest\x12\x1d\n" +
	"\n" +
	"call_index\x18\a \x01(\rR\tcallIndex\x12#\n" +
	"\rintent_digest\x18\b \x01(\fR\fintentDigest\x12\x1d\n" +
	"\n" +
	"service_id\x18\t \x01(\tR\tserviceId\x127\n" +
	"\x05state\x18\n" +
	" \x01(\x0e2!.cozy.worker.v1.NativeSourceStateR\x05state\x12C\n" +
	"\tselection\x18\v \x01(\v2%.cozy.worker.v1.NativeSourceSelectionR\tselection\x124\n" +
	"\x16result_canonical_bytes\x18\f \x01(\fR\x14resultCanonicalBytes\x12\x1b\n" +
	"\tsafe_code\x18\r \x01(\tR\bsafeCode\x12\x1f\n" +
	"\vsafe_detail\x18\x0e \x01(\tR\n" +
	"safeDetail\x12C\n" +
	"\x1enative_receipt_canonical_bytes\x18\x0f \x01(\fR\x1bnativeReceiptCanonicalBytes\x12-\n" +
	"\x12computation_digest\x18\x10 \x01(\fR\x11computationDigest\x12\x19\n" +
	"\bmemo_hit\x18\x11 \x01(\bR\amemoHit\x12B\n" +
	"\vbyte_output\x18\x12 \x01(\v2!.cozy.worker.v1.NativeByteTreeRefR\n" +
	"byteOutput\x12=\n" +
	"\x1bbyte_output_attempt_ordinal\x18\x13 \x01(\x04R\x18byteOutputAttemptOrdinal\x12J\n" +
	"\"byte_output_invocation_spec_digest\x18\x14 \x01(\fR\x1ebyteOutputInvocationSpecDigest\"\xa7\x02\n" +
	"\x05Claim\x12,\n" +
	"\x12record_owner_epoch\x18\x01 \x01(\x04R\x10recordOwnerEpoch\x120\n" +
	"\x14control_stream_epoch\x18\x02 \x01(\x04R\x12controlStreamEpoch\x12$\n" +
	"\x0eworker_boot_id\x18\x03 \x01(\tR\fworkerBootId\x12&\n" +
	"\x0frecord_owner_id\x18\x05 \x01(\tR\rrecordOwnerId\x12\x1b\n" +
	"\tworker_id\x18\x06 \x01(\tR\bworkerId\x12\x1d\n" +
	"\n" +
	"wire_minor\x18\a \x01(\rR\twireMinor\x12\x14\n" +
	"\x05proof\x18\b \x01(\fR\x05proofJ\x04\b\x04\x10\x05J\x04\b\t\x10\n" +
	"R\x12wire_schema_digest\"\xc0\x01\n" +
	"\n" +
	"ClaimProof\x12,\n" +
	"\x12record_owner_epoch\x18\x01 \x01(\x04R\x10recordOwnerEpoch\x12$\n" +
	"\x0eworker_boot_id\x18\x02 \x01(\tR\fworkerBootId\x12\x1b\n" +
	"\tworker_id\x18\x03 \x01(\tR\bworkerId\x12A\n" +
	"\x1dworker_tls_certificate_digest\x18\x04 \x01(\fR\x1aworkerTlsCertificateDigest\"\xd2\x04\n" +
	"\bClaimAck\x12,\n" +
	"\x12record_owner_epoch\x18\x01 \x01(\x04R\x10recordOwnerEpoch\x120\n" +
	"\x14control_stream_epoch\x18\x02 \x01(\x04R\x12controlStreamEpoch\x12$\n" +
	"\x0eworker_boot_id\x18\x03 \x01(\tR\fworkerBootId\x12\x1a\n" +
	"\baccepted\x18\x05 \x01(\bR\baccepted\x12<\n" +
	"\trejection\x18\x06 \x01(\x0e2\x1e.cozy.worker.v1.ClaimRejectionR\trejection\x12\x1d\n" +
	"\n" +
	"wire_minor\x18\a \x01(\rR\twireMinor\x12\x1b\n" +
	"\tworker_id\x18\b \x01(\tR\bworkerId\x12,\n" +
	"\x12worker_instance_id\x18\t \x01(\tR\x10workerInstanceId\x12*\n" +
	"\x11worker_release_id\x18\n" +
	" \x01(\tR\x0fworkerReleaseId\x124\n" +
	"\x16control_runtime_digest\x18\v \x01(\tR\x14controlRuntimeDigest\x12\x1d\n" +
	"\n" +
	"git_commit\x18\f \x01(\tR\tgitCommit\x12=\n" +
	"\tresources\x18\r \x01(\v2\x1f.cozy.worker.v1.WorkerResourcesR\tresourcesJ\x04\b\x04\x10\x05J\x04\b\x0e\x10\x0fJ\x04\b\x0f\x10\x10R\x12wire_schema_digestR\x16claim_survives_restart\"\xd8\x03\n" +
	"\vBootFailure\x12,\n" +
	"\x12record_owner_epoch\x18\x01 \x01(\x04R\x10recordOwnerEpoch\x120\n" +
	"\x14control_stream_epoch\x18\x02 \x01(\x04R\x12controlStreamEpoch\x12$\n" +
	"\x0eworker_boot_id\x18\x03 \x01(\tR\fworkerBootId\x12\x1b\n" +
	"\tworker_id\x18\x05 \x01(\tR\bworkerId\x12,\n" +
	"\x12worker_instance_id\x18\x06 \x01(\tR\x10workerInstanceId\x129\n" +
	"\x06reason\x18\a \x01(\x0e2!.cozy.worker.v1.BootFailureReasonR\x06reason\x12\x16\n" +
	"\x06detail\x18\b \x01(\tR\x06detail\x12=\n" +
	"\tresources\x18\t \x01(\v2\x1f.cozy.worker.v1.WorkerResourcesR\tresources\x124\n" +
	"\x16control_runtime_digest\x18\n" +
	" \x01(\tR\x14controlRuntimeDigest\x12*\n" +
	"\x11worker_release_id\x18\v \x01(\tR\x0fworkerReleaseIdJ\x04\b\x04\x10\x05\"\xe9\x03\n" +
	"\x0eWorkerSnapshot\x12,\n" +
	"\x12record_owner_epoch\x18\x01 \x01(\x04R\x10recordOwnerEpoch\x120\n" +
	"\x14control_stream_epoch\x18\x02 \x01(\x04R\x12controlStreamEpoch\x12$\n" +
	"\x0eworker_boot_id\x18\x03 \x01(\tR\fworkerBootId\x12\x1f\n" +
	"\vsnapshot_id\x18\x05 \x01(\tR\n" +
	"snapshotId\x12'\n" +
	"\x0fsnapshot_digest\x18\x06 \x01(\fR\x0esnapshotDigest\x128\n" +
	"\x18snapshot_canonical_bytes\x18\a \x01(\fR\x16snapshotCanonicalBytes\x12R\n" +
	"&accepted_placement_set_canonical_bytes\x18\b \x01(\fR\"acceptedPlacementSetCanonicalBytes\x120\n" +
	"\x14host_snapshot_digest\x18\t \x01(\fR\x12hostSnapshotDigest\x12A\n" +
	"\x1dhost_snapshot_canonical_bytes\x18\n" +
	" \x01(\fR\x1ahostSnapshotCanonicalBytesJ\x04\b\x04\x10\x05\"\xc8\x05\n" +
	"\x12WorkerSnapshotBody\x12E\n" +
	"\x1faccepted_desired_state_revision\x18\x01 \x01(\x04R\x1cacceptedDesiredStateRevision\x12A\n" +
	"\x1daccepted_placement_set_digest\x18\x02 \x01(\fR\x1aacceptedPlacementSetDigest\x12>\n" +
	"\fworker_phase\x18\x04 \x01(\x0e2\x1b.cozy.worker.v1.WorkerPhaseR\vworkerPhase\x12?\n" +
	"\n" +
	"placements\x18\x05 \x03(\v2\x1f.cozy.worker.v1.PlacementStatusR\n" +
	"placements\x12-\n" +
	"\x12converged_revision\x18\x06 \x01(\x04R\x11convergedRevision\x12'\n" +
	"\x0fadmission_epoch\x18\a \x01(\x04R\x0eadmissionEpoch\x12G\n" +
	"\x0fadmission_state\x18\b \x01(\x0e2\x1e.cozy.worker.v1.AdmissionStateR\x0eadmissionState\x126\n" +
	"\x17available_attempt_slots\x18\t \x01(\rR\x15availableAttemptSlots\x12@\n" +
	"\rheld_attempts\x18\n" +
	" \x03(\v2\x1b.cozy.worker.v1.HeldAttemptR\fheldAttempts\x120\n" +
	"\x05lanes\x18\f \x03(\v2\x1a.cozy.worker.v1.DeviceLaneR\x05lanes\x12%\n" +
	"\x0eheld_manifests\x18\r \x03(\tR\rheldManifestsJ\x04\b\x03\x10\x04J\x04\b\v\x10\fR\x11journal_highwaterR\x14weights_transactions\"\xed\x01\n" +
	"\x10HostSnapshotBody\x12@\n" +
	"\rheld_outcomes\x18\x01 \x03(\v2\x1b.cozy.worker.v1.HeldAttemptR\fheldOutcomes\x12[\n" +
	"\x14weights_transactions\x18\x02 \x03(\v2(.cozy.worker.v1.WeightsTransactionStatusR\x13weightsTransactions\x12:\n" +
	"\x19retained_desired_revision\x18\x03 \x01(\x04R\x17retainedDesiredRevision\"\x95\x02\n" +
	"\vSnapshotAck\x12,\n" +
	"\x12record_owner_epoch\x18\x01 \x01(\x04R\x10recordOwnerEpoch\x120\n" +
	"\x14control_stream_epoch\x18\x02 \x01(\x04R\x12controlStreamEpoch\x12$\n" +
	"\x0eworker_boot_id\x18\x03 \x01(\tR\fworkerBootId\x12\x1f\n" +
	"\vsnapshot_id\x18\x05 \x01(\tR\n" +
	"snapshotId\x12'\n" +
	"\x0fsnapshot_digest\x18\x06 \x01(\fR\x0esnapshotDigest\x120\n" +
	"\x14host_snapshot_digest\x18\a \x01(\fR\x12hostSnapshotDigestJ\x04\b\x04\x10\x05\"\xe7\x04\n" +
	"\x12DesiredWorkerState\x12,\n" +
	"\x12record_owner_epoch\x18\x01 \x01(\x04R\x10recordOwnerEpoch\x120\n" +
	"\x14control_stream_epoch\x18\x02 \x01(\x04R\x12controlStreamEpoch\x12$\n" +
	"\x0eworker_boot_id\x18\x03 \x01(\tR\fworkerBootId\x12\x1a\n" +
	"\brevision\x18\x05 \x01(\x04R\brevision\x121\n" +
	"\aposture\x18\x06 \x01(\x0e2\x17.cozy.worker.v1.PostureR\aposture\x12\x1d\n" +
	"\n" +
	"wire_minor\x18\a \x01(\rR\twireMinor\x12$\n" +
	"\x0edrain_grace_ms\x18\b \x01(\x04R\fdrainGraceMs\x120\n" +
	"\x03job\x18\r \x01(\v2\x1c.cozy.worker.v1.JobDirectiveH\x00R\x03job\x12J\n" +
	"\rplacement_set\x18\x0f \x01(\v2#.cozy.worker.v1.DesiredPlacementSetH\x00R\fplacementSet\x12J\n" +
	"\x10runtime_revision\x18\x13 \x01(\v2\x1f.cozy.worker.v1.RuntimeRevisionR\x0fruntimeRevisionB\x06\n" +
	"\x04modeJ\x04\b\x04\x10\x05J\x04\b\t\x10\n" +
	"J\x04\b\n" +
	"\x10\vJ\x04\b\v\x10\fJ\x04\b\f\x10\rJ\x04\b\x10\x10\x11J\x04\b\x11\x10\x12J\x04\b\x12\x10\x13R\vpackage_setR\x11local_package_setR\x15private_placement_set\"\x88\x01\n" +
	"\x11DesiredPackageSet\x12/\n" +
	"\x13download_delegation\x18\x01 \x01(\fR\x12downloadDelegation\x12B\n" +
	"\x1ddownload_delegation_signature\x18\x02 \x01(\fR\x1bdownloadDelegationSignature\"\xe4\x02\n" +
	"\x16DesiredLocalPackageSet\x12!\n" +
	"\foperation_id\x18\x01 \x01(\tR\voperationId\x12<\n" +
	"\apackage\x18\x02 \x01(\v2\".cozy.worker.v1.DevelopmentPackageR\apackage\x129\n" +
	"\x05files\x18\x03 \x03(\v2#.cozy.worker.v1.LocalPackageFileRefR\x05files\x127\n" +
	"\x17dependency_requirements\x18\x04 \x01(\fR\x16dependencyRequirements\x12'\n" +
	"\x0fpython_requires\x18\x05 \x01(\tR\x0epythonRequires\x12%\n" +
	"\x0epython_version\x18\x06 \x01(\tR\rpythonVersion\x12%\n" +
	"\x0esource_archive\x18\a \x01(\tR\rsourceArchive\"\xfe\x03\n" +
	"\x1aDesiredPrivatePlacementSet\x12!\n" +
	"\foperation_id\x18\x01 \x01(\tR\voperationId\x12/\n" +
	"\x13download_delegation\x18\x03 \x01(\fR\x12downloadDelegation\x12B\n" +
	"\x1ddownload_delegation_signature\x18\x04 \x01(\fR\x1bdownloadDelegationSignature\x12G\n" +
	"\rnative_models\x18\x05 \x03(\v2\".cozy.worker.v1.NativeModelBindingR\fnativeModels\x12'\n" +
	"\x0finstallation_id\x18\x06 \x01(\tR\x0einstallationId\x12@\n" +
	"\rmodel_choices\x18\a \x03(\v2\x1b.cozy.worker.v1.ModelChoiceR\fmodelChoices\x12O\n" +
	"\x12source_credentials\x18\b \x03(\v2 .cozy.worker.v1.SourceCredentialR\x11sourceCredentials\x12\x10\n" +
	"\x03hub\x18\t \x01(\tR\x03hub\x12\x14\n" +
	"\x05owner\x18\n" +
	" \x01(\tR\x05ownerJ\x04\b\x02\x10\x03R\x15local_revision_digest\"\xf6\x01\n" +
	"\x12NativeModelBinding\x12\x12\n" +
	"\x04slot\x18\x01 \x01(\tR\x04slot\x12\x14\n" +
	"\x05model\x18\x02 \x01(\tR\x05model\x12/\n" +
	"\bmanifest\x18\x03 \x01(\v2\x13.cozy.worker.v1.RefR\bmanifest\x12E\n" +
	"\tretention\x18\x04 \x01(\v2'.cozy.worker.v1.DerivedRetentionRequestR\tretention\x12>\n" +
	"\badapters\x18\x05 \x03(\v2\".cozy.worker.v1.DownloadAdapterRefR\badapters\"m\n" +
	"\x13LocalPackageFileRef\x12\x16\n" +
	"\x06digest\x18\x01 \x01(\fR\x06digest\x12\x1a\n" +
	"\bfilename\x18\x02 \x01(\tR\bfilename\x12\x16\n" +
	"\x06length\x18\x04 \x01(\x04R\x06lengthJ\x04\b\x03\x10\x04R\x04kind\"\xcd\x02\n" +
	"\x13DesiredPlacementSet\x120\n" +
	"\x14placement_set_digest\x18\x01 \x01(\fR\x12placementSetDigest\x12A\n" +
	"\x1dplacement_set_canonical_bytes\x18\x03 \x01(\fR\x1aplacementSetCanonicalBytes\x12C\n" +
	"\vdevice_pins\x18\x04 \x03(\v2\".cozy.worker.v1.PlacementDevicePinR\n" +
	"devicePins\x12O\n" +
	"\x14orchestration_parent\x18\x05 \x01(\v2\x1c.cozy.worker.v1.JobDirectiveR\x13orchestrationParent\x12%\n" +
	"\x0eexecution_gpus\x18\x06 \x01(\rR\rexecutionGpusJ\x04\b\x02\x10\x03\"`\n" +
	"\x12PlacementDevicePin\x12!\n" +
	"\fplacement_id\x18\x01 \x01(\tR\vplacementId\x12'\n" +
	"\x0fdevice_ordinals\x18\x02 \x03(\rR\x0edeviceOrdinals\"I\n" +
	"\fPlacementSet\x129\n" +
	"\n" +
	"placements\x18\x01 \x03(\v2\x19.cozy.worker.v1.PlacementR\n" +
	"placements\"\xef\x02\n" +
	"\x12DownloadDelegation\x12&\n" +
	"\x0fexpires_at_unix\x18\x01 \x01(\x04R\rexpiresAtUnix\x128\n" +
	"\x06models\x18\x02 \x03(\v2 .cozy.worker.v1.DownloadModelRefR\x06models\x12>\n" +
	"\bpackages\x18\x03 \x03(\v2\".cozy.worker.v1.DownloadPackageRefR\bpackages\x12\x1b\n" +
	"\trental_id\x18\x04 \x01(\tR\brentalId\x12$\n" +
	"\x0eworker_boot_id\x18\x05 \x01(\tR\fworkerBootId\x12\x1b\n" +
	"\tworker_id\x18\x06 \x01(\tR\bworkerId\x12A\n" +
	"\x1dworker_tls_certificate_digest\x18\a \x01(\fR\x1aworkerTlsCertificateDigestJ\x04\b\b\x10\tR\x0eheld_manifests\"\xe0\x01\n" +
	"\x10DownloadModelRef\x12\x1a\n" +
	"\bmanifest\x18\x01 \x01(\tR\bmanifest\x12\x14\n" +
	"\x05model\x18\x02 \x01(\tR\x05model\x12\x18\n" +
	"\arelease\x18\x03 \x01(\tR\arelease\x12\x18\n" +
	"\apackage\x18\x04 \x01(\tR\apackage\x12\x12\n" +
	"\x04slot\x18\x05 \x01(\tR\x04slot\x12\x12\n" +
	"\x04lane\x18\x06 \x01(\tR\x04lane\x12>\n" +
	"\badapters\x18\a \x03(\v2\".cozy.worker.v1.DownloadAdapterRefR\badapters\"\x87\x02\n" +
	"\x12DownloadAdapterRef\x12\x1c\n" +
	"\tcomponent\x18\x01 \x01(\tR\tcomponent\x12\x14\n" +
	"\x05model\x18\x02 \x01(\tR\x05model\x12\x18\n" +
	"\arelease\x18\x03 \x01(\tR\arelease\x12\x12\n" +
	"\x04lane\x18\x04 \x01(\tR\x04lane\x12\x1a\n" +
	"\bmanifest\x18\x05 \x01(\tR\bmanifest\x12)\n" +
	"\x10source_component\x18\x06 \x01(\tR\x0fsourceComponent\x12\x14\n" +
	"\x05scale\x18\a \x01(\tR\x05scale\x12\x16\n" +
	"\x06source\x18\b \x01(\tR\x06source\x12\x1a\n" +
	"\bprofiles\x18\t \x03(\tR\bprofiles\"^\n" +
	"\x12DownloadPackageRef\x12\x18\n" +
	"\apackage\x18\x01 \x01(\tR\apackage\x12\x18\n" +
	"\arelease\x18\x02 \x01(\tR\areleaseJ\x04\b\x03\x10\x04R\x0erelease_digest\"\x9a\x04\n" +
	"\tPlacement\x12!\n" +
	"\fplacement_id\x18\x01 \x01(\tR\vplacementId\x12<\n" +
	"\apackage\x18\x02 \x01(\v2 .cozy.worker.v1.PackageSelectionH\x00R\apackage\x12F\n" +
	"\vdevelopment\x18\v \x01(\v2\".cozy.worker.v1.DevelopmentPackageH\x00R\vdevelopment\x12'\n" +
	"\x0fbindings_digest\x18\x06 \x01(\fR\x0ebindingsDigest\x12-\n" +
	"\x06models\x18\a \x03(\v2\x15.cozy.worker.v1.ModelR\x06models\x12<\n" +
	"\ventrypoints\x18\b \x03(\v2\x1a.cozy.worker.v1.EntrypointR\ventrypoints\x12'\n" +
	"\x0finstallation_id\x18\f \x01(\tR\x0einstallationId\x12+\n" +
	"\x11package_interface\x18\r \x01(\fR\x10packageInterfaceB\x0e\n" +
	"\fpackage_modeJ\x04\b\x03\x10\x04J\x04\b\x04\x10\x05J\x04\b\x05\x10\x06J\x04\b\t\x10\n" +
	"J\x04\b\n" +
	"\x10\vR\x12environment_digestR\x1aenvironment_receipt_digestR\rqualificationR\venvironment\"5\n" +
	"\x03Ref\x12\x16\n" +
	"\x06digest\x18\x01 \x01(\fR\x06digest\x12\x16\n" +
	"\x06length\x18\x02 \x01(\x04R\x06length\"\x83\x01\n" +
	"\x10PackageSelection\x12\x18\n" +
	"\apackage\x18\x01 \x01(\tR\apackage\x12\x18\n" +
	"\arelease\x18\x02 \x01(\tR\arelease\x12\x10\n" +
	"\x03hub\x18\x05 \x01(\tR\x03hubJ\x04\b\x03\x10\x04J\x04\b\x04\x10\x05R\x0erelease_digestR\rproject_wheel\"\xb8\x01\n" +
	"\x12DevelopmentPackage\x12\x18\n" +
	"\apackage\x18\x01 \x01(\tR\apackage\x12\x18\n" +
	"\arelease\x18\x02 \x01(\tR\arelease\x12'\n" +
	"\x0finstallation_id\x18\x06 \x01(\tR\x0einstallationIdJ\x04\b\x03\x10\x04J\x04\b\x04\x10\x05J\x04\b\x05\x10\x06R\rsource_digestR\rproject_wheelR\x15local_revision_digest\"\x8a\x01\n" +
	"\x05Model\x12\x0e\n" +
	"\x02id\x18\x01 \x01(\tR\x02id\x12\x12\n" +
	"\x04repo\x18\x02 \x01(\tR\x04repo\x12\x18\n" +
	"\aversion\x18\x03 \x01(\tR\aversion\x12\x12\n" +
	"\x04lane\x18\x04 \x01(\tR\x04lane\x12/\n" +
	"\bmanifest\x18\x05 \x01(\v2\x13.cozy.worker.v1.RefR\bmanifest\"\x88\x01\n" +
	"\n" +
	"Entrypoint\x12\x12\n" +
	"\x04name\x18\x01 \x01(\tR\x04name\x12:\n" +
	"\x19entrypoint_binding_digest\x18\x02 \x01(\fR\x17entrypointBindingDigest\x12*\n" +
	"\x05slots\x18\x03 \x03(\v2\x14.cozy.worker.v1.SlotR\x05slots\"\xcf\x02\n" +
	"\x04Slot\x12\x12\n" +
	"\x04slot\x18\x01 \x01(\tR\x04slot\x12,\n" +
	"\x12reference_model_id\x18\x02 \x01(\tR\x10referenceModelId\x129\n" +
	"\n" +
	"components\x18\x04 \x03(\v2\x19.cozy.worker.v1.ComponentR\n" +
	"components\x12S\n" +
	"\x1bmodel_construction_contract\x18\x05 \x01(\v2\x13.cozy.worker.v1.RefR\x19modelConstructionContract\x12-\n" +
	"\x06stamps\x18\x06 \x03(\v2\x15.cozy.worker.v1.StampR\x06stamps\x128\n" +
	"\badapters\x18\a \x03(\v2\x1c.cozy.worker.v1.ModelAdapterR\badaptersJ\x04\b\x03\x10\x04R\x06config\"\x88\x01\n" +
	"\fModelAdapter\x12\x1c\n" +
	"\tcomponent\x18\x01 \x01(\tR\tcomponent\x12\x19\n" +
	"\bmodel_id\x18\x02 \x01(\tR\amodelId\x12)\n" +
	"\x10source_component\x18\x03 \x01(\tR\x0fsourceComponent\x12\x14\n" +
	"\x05scale\x18\x04 \x01(\tR\x05scale\"D\n" +
	"\tComponent\x12\x1c\n" +
	"\tcomponent\x18\x01 \x01(\tR\tcomponent\x12\x19\n" +
	"\bmodel_id\x18\x02 \x01(\tR\amodelId\"O\n" +
	"\x05Stamp\x12\x1c\n" +
	"\tcomponent\x18\x01 \x01(\tR\tcomponent\x12\x10\n" +
	"\x03key\x18\x02 \x01(\tR\x03key\x12\x16\n" +
	"\x06values\x18\x03 \x03(\tR\x06values\"\xc3\x03\n" +
	"\fJobDirective\x12*\n" +
	"\x11job_descriptor_id\x18\x02 \x01(\tR\x0fjobDescriptorId\x12A\n" +
	"\rresource_caps\x18\x03 \x01(\v2\x1c.cozy.worker.v1.ResourceCapsR\fresourceCaps\x12V\n" +
	"\x14publication_contract\x18\x04 \x01(\v2#.cozy.worker.v1.PublicationContractR\x13publicationContract\x12!\n" +
	"\fdevice_count\x18\x06 \x01(\rR\vdeviceCount\x12$\n" +
	"\rorchestration\x18\a \x01(\bR\rorchestration\x12O\n" +
	"\x14orchestration_parent\x18\b \x01(\v2\x1c.cozy.worker.v1.JobDirectiveR\x13orchestrationParent\x12'\n" +
	"\x0finstallation_id\x18\t \x01(\tR\x0einstallationIdJ\x04\b\x01\x10\x02J\x04\b\x05\x10\x06R\bbuild_idR\x13reclaim_on_terminal\"\xec\b\n" +
	"\x13ObservedWorkerState\x12,\n" +
	"\x12record_owner_epoch\x18\x01 \x01(\x04R\x10recordOwnerEpoch\x120\n" +
	"\x14control_stream_epoch\x18\x02 \x01(\x04R\x12controlStreamEpoch\x12$\n" +
	"\x0eworker_boot_id\x18\x03 \x01(\tR\fworkerBootId\x12,\n" +
	"\x12applied_wire_minor\x18\b \x01(\rR\x10appliedWireMinor\x12@\n" +
	"\rheld_attempts\x18\t \x03(\v2\x1b.cozy.worker.v1.HeldAttemptR\fheldAttempts\x12-\n" +
	"\x06faults\x18\n" +
	" \x03(\v2\x15.cozy.worker.v1.FaultR\x06faults\x129\n" +
	"\bactivity\x18\v \x03(\v2\x1d.cozy.worker.v1.ActivityEventR\bactivity\x12>\n" +
	"\fjob_capacity\x18\r \x01(\v2\x1b.cozy.worker.v1.JobCapacityR\vjobCapacity\x12?\n" +
	"\n" +
	"placements\x18\x0f \x03(\v2\x1f.cozy.worker.v1.PlacementStatusR\n" +
	"placements\x12'\n" +
	"\x0fadmission_epoch\x18\x10 \x01(\x04R\x0eadmissionEpoch\x12G\n" +
	"\x0fadmission_state\x18\x11 \x01(\x0e2\x1e.cozy.worker.v1.AdmissionStateR\x0eadmissionState\x126\n" +
	"\x17available_attempt_slots\x18\x12 \x01(\rR\x15availableAttemptSlots\x12E\n" +
	"\x1faccepted_desired_state_revision\x18\x13 \x01(\x04R\x1cacceptedDesiredStateRevision\x12A\n" +
	"\x1daccepted_placement_set_digest\x18\x14 \x01(\fR\x1aacceptedPlacementSetDigest\x12-\n" +
	"\x12converged_revision\x18\x15 \x01(\x04R\x11convergedRevision\x12>\n" +
	"\fworker_phase\x18\x16 \x01(\x0e2\x1b.cozy.worker.v1.WorkerPhaseR\vworkerPhase\x120\n" +
	"\x05lanes\x18\x1a \x03(\v2\x1a.cozy.worker.v1.DeviceLaneR\x05lanes\x12%\n" +
	"\x0eheld_manifests\x18\x1b \x03(\tR\rheldManifestsJ\x04\b\x04\x10\x05J\x04\b\x05\x10\x06J\x04\b\x06\x10\aJ\x04\b\a\x10\bJ\x04\b\f\x10\rJ\x04\b\x0e\x10\x0fJ\x04\b\x17\x10\x18J\x04\b\x18\x10\x19J\x04\b\x19\x10\x1aR\x19applied_artifact_grant_idR\x16applied_grant_revisionR\x0fartifact_intent\"\xe1\x01\n" +
	"\n" +
	"DeviceLane\x12\x17\n" +
	"\alane_id\x18\x01 \x01(\tR\x06laneId\x12'\n" +
	"\x0fdevice_ordinals\x18\x02 \x03(\rR\x0edeviceOrdinals\x126\n" +
	"\x17available_attempt_slots\x18\x03 \x01(\rR\x15availableAttemptSlots\x12#\n" +
	"\rplacement_ids\x18\x04 \x03(\tR\fplacementIds\x124\n" +
	"\x16resident_placement_ids\x18\x05 \x03(\tR\x14residentPlacementIds\"\xa4\a\n" +
	"\x0fPlacementStatus\x12!\n" +
	"\fplacement_id\x18\x01 \x01(\tR\vplacementId\x12%\n" +
	"\x0eexecutor_epoch\x18\x04 \x01(\x04R\rexecutorEpoch\x12@\n" +
	"\x1cdispatchable_binding_digests\x18\x06 \x03(\fR\x1adispatchableBindingDigests\x12D\n" +
	"\x1ematerializable_binding_digests\x18\a \x03(\fR\x1cmaterializableBindingDigests\x12-\n" +
	"\x06faults\x18\b \x03(\v2\x15.cozy.worker.v1.FaultR\x06faults\x12J\n" +
	"\vaccelerator\x18\t \x01(\v2(.cozy.worker.v1.AcceleratorQualificationR\vaccelerator\x120\n" +
	"\x14placement_set_digest\x18\v \x01(\fR\x12placementSetDigest\x12N\n" +
	"\x0fmaterialization\x18\f \x01(\x0e2$.cozy.worker.v1.MaterializationStateR\x0fmaterialization\x126\n" +
	"\aserving\x18\r \x01(\x0e2\x1c.cozy.worker.v1.ServingStateR\aserving\x12R\n" +
	"&retained_fallback_placement_set_digest\x18\x0e \x01(\fR\"retainedFallbackPlacementSetDigest\x12Q\n" +
	"\vacquisition\x18\x0f \x01(\v2/.cozy.worker.v1.PlacementAcquisitionObservationR\vacquisition\x12$\n" +
	"\x0edevice_lane_id\x18\x13 \x01(\tR\fdeviceLaneId\x124\n" +
	"\x16loaded_binding_digests\x18\x14 \x03(\fR\x14loadedBindingDigests\x12'\n" +
	"\x0finstallation_id\x18\x15 \x01(\tR\x0einstallationIdJ\x04\b\x02\x10\x03J\x04\b\x03\x10\x04J\x04\b\x05\x10\x06J\x04\b\x10\x10\x11J\x04\b\x11\x10\x12J\x04\b\x12\x10\x13R\x17package_revision_digestR\x12environment_digestR\rconfig_digest\"\xa7\x01\n" +
	"\x1fPlacementAcquisitionObservation\x12C\n" +
	"\apackage\x18\x01 \x01(\v2).cozy.worker.v1.AcquisitionLegObservationR\apackage\x12?\n" +
	"\x05model\x18\x02 \x01(\v2).cozy.worker.v1.AcquisitionLegObservationR\x05model\"\xc9\x01\n" +
	"\x19AcquisitionLegObservation\x120\n" +
	"\x14started_monotonic_ns\x18\x01 \x01(\x04R\x12startedMonotonicNs\x12,\n" +
	"\x12ended_monotonic_ns\x18\x02 \x01(\x04R\x10endedMonotonicNs\x12)\n" +
	"\x10downloaded_bytes\x18\x03 \x01(\x04R\x0fdownloadedBytes\x12!\n" +
	"\freused_bytes\x18\x04 \x01(\x04R\vreusedBytes\"\xe1\x02\n" +
	"\x18AcceleratorQualification\x12\x1c\n" +
	"\tqualified\x18\x01 \x01(\bR\tqualified\x12\"\n" +
	"\fcapabilities\x18\x02 \x03(\tR\fcapabilities\x122\n" +
	"\x15recommended_max_bytes\x18\x03 \x01(\x04R\x13recommendedMaxBytes\x122\n" +
	"\x15pinned_capacity_bytes\x18\x04 \x01(\x04R\x13pinnedCapacityBytes\x12*\n" +
	"\x11h2d_gbps_measured\x18\x05 \x01(\rR\x0fh2dGbpsMeasured\x12*\n" +
	"\x11d2h_gbps_measured\x18\x06 \x01(\rR\x0fd2hGbpsMeasured\x12#\n" +
	"\rtorch_version\x18\a \x01(\tR\ftorchVersion\x12\x1e\n" +
	"\n" +
	"unreadable\x18\b \x03(\tR\n" +
	"unreadable\"g\n" +
	"\rActivityEvent\x12\x10\n" +
	"\x03seq\x18\x01 \x01(\x04R\x03seq\x12\x12\n" +
	"\x04kind\x18\x02 \x01(\tR\x04kind\x12\x12\n" +
	"\x04step\x18\x03 \x01(\tR\x04step\x12\x1c\n" +
	"\n" +
	"at_unix_ms\x18\x04 \x01(\x04R\batUnixMs\"\xe0\x03\n" +
	"\fAttemptOffer\x12,\n" +
	"\x12record_owner_epoch\x18\x01 \x01(\x04R\x10recordOwnerEpoch\x120\n" +
	"\x14control_stream_epoch\x18\x02 \x01(\x04R\x12controlStreamEpoch\x12$\n" +
	"\x0eworker_boot_id\x18\x03 \x01(\tR\fworkerBootId\x12\x1d\n" +
	"\n" +
	"request_id\x18\x05 \x01(\tR\trequestId\x12'\n" +
	"\x0fattempt_ordinal\x18\x06 \x01(\x04R\x0eattemptOrdinal\x124\n" +
	"\x16invocation_spec_digest\x18\a \x01(\fR\x14invocationSpecDigest\x123\n" +
	"\x05grant\x18\b \x01(\v2\x1d.cozy.worker.v1.DeliveryGrantR\x05grant\x12E\n" +
	"\x1finvocation_spec_canonical_bytes\x18\t \x01(\fR\x1cinvocationSpecCanonicalBytes\x12!\n" +
	"\fplacement_id\x18\n" +
	" \x01(\tR\vplacementId\x12'\n" +
	"\x0fadmission_epoch\x18\f \x01(\x04R\x0eadmissionEpochJ\x04\b\x04\x10\x05\"\xad\x03\n" +
	"\x0fAttemptAccepted\x12,\n" +
	"\x12record_owner_epoch\x18\x01 \x01(\x04R\x10recordOwnerEpoch\x120\n" +
	"\x14control_stream_epoch\x18\x02 \x01(\x04R\x12controlStreamEpoch\x12$\n" +
	"\x0eworker_boot_id\x18\x03 \x01(\tR\fworkerBootId\x12\x1d\n" +
	"\n" +
	"request_id\x18\x05 \x01(\tR\trequestId\x12'\n" +
	"\x0fattempt_ordinal\x18\x06 \x01(\x04R\x0eattemptOrdinal\x124\n" +
	"\x16invocation_spec_digest\x18\a \x01(\fR\x14invocationSpecDigest\x12!\n" +
	"\fplacement_id\x18\v \x01(\tR\vplacementId\x12\x17\n" +
	"\alane_id\x18\r \x01(\tR\x06laneIdJ\x04\b\x04\x10\x05J\x04\b\b\x10\tJ\x04\b\t\x10\n" +
	"J\x04\b\n" +
	"\x10\vJ\x04\b\f\x10\rR\vplan_digestR\x19model_construction_digestR\x04planR\x0eexecutor_epoch\"\xea\x02\n" +
	"\rCancelAttempt\x12,\n" +
	"\x12record_owner_epoch\x18\x01 \x01(\x04R\x10recordOwnerEpoch\x120\n" +
	"\x14control_stream_epoch\x18\x02 \x01(\x04R\x12controlStreamEpoch\x12$\n" +
	"\x0eworker_boot_id\x18\x03 \x01(\tR\fworkerBootId\x12\x1d\n" +
	"\n" +
	"request_id\x18\x05 \x01(\tR\trequestId\x12'\n" +
	"\x0fattempt_ordinal\x18\x06 \x01(\x04R\x0eattemptOrdinal\x124\n" +
	"\x06reason\x18\a \x01(\x0e2\x1c.cozy.worker.v1.CancelReasonR\x06reason\x12\x19\n" +
	"\bgrace_ms\x18\b \x01(\x04R\agraceMs\x124\n" +
	"\x16invocation_spec_digest\x18\t \x01(\fR\x14invocationSpecDigestJ\x04\b\x04\x10\x05\"\xbb\x03\n" +
	"\x0eAttemptOutcome\x12,\n" +
	"\x12record_owner_epoch\x18\x01 \x01(\x04R\x10recordOwnerEpoch\x120\n" +
	"\x14control_stream_epoch\x18\x02 \x01(\x04R\x12controlStreamEpoch\x12$\n" +
	"\x0eworker_boot_id\x18\x03 \x01(\tR\fworkerBootId\x12\x1d\n" +
	"\n" +
	"request_id\x18\x05 \x01(\tR\trequestId\x12'\n" +
	"\x0fattempt_ordinal\x18\x06 \x01(\x04R\x0eattemptOrdinal\x124\n" +
	"\x16invocation_spec_digest\x18\a \x01(\fR\x14invocationSpecDigest\x12\x1d\n" +
	"\n" +
	"outcome_id\x18\b \x01(\tR\toutcomeId\x12%\n" +
	"\x0eoutcome_digest\x18\t \x01(\fR\routcomeDigest\x126\n" +
	"\x17outcome_canonical_bytes\x18\n" +
	" \x01(\fR\x15outcomeCanonicalBytes\x12!\n" +
	"\fplacement_id\x18\v \x01(\tR\vplacementIdJ\x04\b\x04\x10\x05\"\xe4\x05\n" +
	"\x12AttemptOutcomeBody\x12\x1d\n" +
	"\n" +
	"request_id\x18\x01 \x01(\tR\trequestId\x12'\n" +
	"\x0fattempt_ordinal\x18\x02 \x01(\x04R\x0eattemptOrdinal\x124\n" +
	"\x16invocation_spec_digest\x18\x03 \x01(\tR\x14invocationSpecDigest\x125\n" +
	"\x06status\x18\x04 \x01(\x0e2\x1d.cozy.worker.v1.OutcomeStatusR\x06status\x12G\n" +
	"\x0foutput_manifest\x18\x05 \x01(\v2\x1e.cozy.worker.v1.OutputManifestR\x0eoutputManifest\x128\n" +
	"\ametrics\x18\x06 \x01(\v2\x1e.cozy.worker.v1.AttemptMetricsR\ametrics\x12D\n" +
	"\rtriage_bundle\x18\a \x01(\v2\x1f.cozy.worker.v1.TriageBundleRefR\ftriageBundle\x12!\n" +
	"\fsafe_message\x18\b \x01(\tR\vsafeMessage\x122\n" +
	"\x05cause\x18\t \x01(\v2\x1c.cozy.worker.v1.OutcomeCauseR\x05cause\x126\n" +
	"\x06result\x18\n" +
	" \x01(\v2\x1e.cozy.worker.v1.ResultEnvelopeR\x06result\x12+\n" +
	"\x11execution_started\x18\v \x01(\bR\x10executionStarted\x12L\n" +
	"\x10weights_receipts\x18\f \x03(\v2!.cozy.worker.v1.WeightsReceiptRefR\x0fweightsReceipts\x12F\n" +
	"\vobservation\x18\r \x01(\v2$.cozy.worker.v1.ExecutionObservationR\vobservation\"\x90\x01\n" +
	"\x11WeightsReceiptRef\x124\n" +
	"\x16weights_receipt_digest\x18\x01 \x01(\fR\x14weightsReceiptDigest\x12E\n" +
	"\x1fweights_receipt_canonical_bytes\x18\x02 \x01(\fR\x1cweightsReceiptCanonicalBytes\"\x9c\x03\n" +
	"\x0eWeightsReceipt\x122\n" +
	"\x15owner_authority_scope\x18\x01 \x01(\tR\x13ownerAuthorityScope\x12\x1d\n" +
	"\n" +
	"request_id\x18\x02 \x01(\tR\trequestId\x124\n" +
	"\x16invocation_spec_digest\x18\x03 \x01(\tR\x14invocationSpecDigest\x12\x1f\n" +
	"\voutput_slot\x18\x04 \x01(\tR\n" +
	"outputSlot\x124\n" +
	"\x16weights_transaction_id\x18\x05 \x01(\tR\x14weightsTransactionId\x126\n" +
	"\x17tensorfs_receipt_digest\x18\x06 \x01(\tR\x15tensorfsReceiptDigest\x12G\n" +
	" tensorfs_receipt_canonical_bytes\x18\a \x01(\fR\x1dtensorfsReceiptCanonicalBytesJ\x04\b\b\x10\tR#checkpoint_evidence_canonical_bytes\"i\n" +
	"\x13WeightsObjectSource\x12\x1b\n" +
	"\tobject_id\x18\x01 \x01(\tR\bobjectId\x12\x16\n" +
	"\x06length\x18\x02 \x01(\x04R\x06length\x12\x1d\n" +
	"\n" +
	"source_ref\x18\x03 \x01(\tR\tsourceRef\"\xfc\x04\n" +
	"\x12WeightsIntentFrame\x12,\n" +
	"\x12record_owner_epoch\x18\x01 \x01(\x04R\x10recordOwnerEpoch\x120\n" +
	"\x14control_stream_epoch\x18\x02 \x01(\x04R\x12controlStreamEpoch\x12$\n" +
	"\x0eworker_boot_id\x18\x03 \x01(\tR\fworkerBootId\x12\x1d\n" +
	"\n" +
	"request_id\x18\x05 \x01(\tR\trequestId\x12'\n" +
	"\x0fattempt_ordinal\x18\x06 \x01(\x04R\x0eattemptOrdinal\x124\n" +
	"\x16invocation_spec_digest\x18\a \x01(\fR\x14invocationSpecDigest\x12\x1f\n" +
	"\voutput_slot\x18\b \x01(\tR\n" +
	"outputSlot\x12>\n" +
	"\x1btensorfs_declaration_digest\x18\t \x01(\fR\x19tensorfsDeclarationDigest\x124\n" +
	"\x16requested_writer_epoch\x18\n" +
	" \x01(\x04R\x14requestedWriterEpoch\x12O\n" +
	"$tensorfs_declaration_canonical_bytes\x18\v \x01(\fR!tensorfsDeclarationCanonicalBytes\x124\n" +
	"\x16weights_transaction_id\x18\f \x01(\tR\x14weightsTransactionId\x12>\n" +
	"\x1btensorfs_declaration_length\x18\r \x01(\x04R\x19tensorfsDeclarationLengthJ\x04\b\x04\x10\x05\"\xc4\x06\n" +
	"\x0eWeightsHostAck\x12,\n" +
	"\x12record_owner_epoch\x18\x01 \x01(\x04R\x10recordOwnerEpoch\x120\n" +
	"\x14control_stream_epoch\x18\x02 \x01(\x04R\x12controlStreamEpoch\x12$\n" +
	"\x0eworker_boot_id\x18\x03 \x01(\tR\fworkerBootId\x12\x1d\n" +
	"\n" +
	"request_id\x18\x05 \x01(\tR\trequestId\x12'\n" +
	"\x0fattempt_ordinal\x18\x06 \x01(\x04R\x0eattemptOrdinal\x124\n" +
	"\x16invocation_spec_digest\x18\a \x01(\fR\x14invocationSpecDigest\x12\x1f\n" +
	"\voutput_slot\x18\b \x01(\tR\n" +
	"outputSlot\x124\n" +
	"\x16weights_transaction_id\x18\t \x01(\tR\x14weightsTransactionId\x12!\n" +
	"\fwriter_epoch\x18\n" +
	" \x01(\x04R\vwriterEpoch\x12>\n" +
	"\x1btensorfs_declaration_digest\x18\v \x01(\fR\x19tensorfsDeclarationDigest\x126\n" +
	"\x05stage\x18\f \x01(\x0e2 .cozy.worker.v1.WeightsHostStageR\x05stage\x12<\n" +
	"\aoutcome\x18\r \x01(\x0e2\".cozy.worker.v1.WeightsHostOutcomeR\aoutcome\x12<\n" +
	"\arefusal\x18\x0e \x01(\x0e2\".cozy.worker.v1.WeightsHostRefusalR\arefusal\x12J\n" +
	"\x0fweights_receipt\x18\x0f \x01(\v2!.cozy.worker.v1.WeightsReceiptRefR\x0eweightsReceipt\x12/\n" +
	"\bmanifest\x18\x10 \x01(\v2\x13.cozy.worker.v1.RefR\bmanifest\x12=\n" +
	"\n" +
	"checkpoint\x18\x11 \x01(\v2\x1d.cozy.worker.v1.CheckpointRefR\n" +
	"checkpointJ\x04\b\x04\x10\x05\"\x85\x01\n" +
	"\rCheckpointRef\x12'\n" +
	"\x04head\x18\x01 \x01(\v2\x13.cozy.worker.v1.RefR\x04head\x12\x1f\n" +
	"\vplan_digest\x18\x02 \x01(\fR\n" +
	"planDigest\x12\x14\n" +
	"\x05index\x18\x03 \x01(\x04R\x05index\x12\x14\n" +
	"\x05bytes\x18\x04 \x01(\x04R\x05bytes\"\x9b\x04\n" +
	"\x16WeightsCheckpointFrame\x12,\n" +
	"\x12record_owner_epoch\x18\x01 \x01(\x04R\x10recordOwnerEpoch\x120\n" +
	"\x14control_stream_epoch\x18\x02 \x01(\x04R\x12controlStreamEpoch\x12$\n" +
	"\x0eworker_boot_id\x18\x03 \x01(\tR\fworkerBootId\x12\x1d\n" +
	"\n" +
	"request_id\x18\x05 \x01(\tR\trequestId\x12'\n" +
	"\x0fattempt_ordinal\x18\x06 \x01(\x04R\x0eattemptOrdinal\x124\n" +
	"\x16invocation_spec_digest\x18\a \x01(\fR\x14invocationSpecDigest\x12\x1f\n" +
	"\voutput_slot\x18\b \x01(\tR\n" +
	"outputSlot\x124\n" +
	"\x16weights_transaction_id\x18\t \x01(\tR\x14weightsTransactionId\x12!\n" +
	"\fwriter_epoch\x18\n" +
	" \x01(\x04R\vwriterEpoch\x12>\n" +
	"\x1btensorfs_declaration_digest\x18\v \x01(\fR\x19tensorfsDeclarationDigest\x12=\n" +
	"\n" +
	"checkpoint\x18\f \x01(\v2\x1d.cozy.worker.v1.CheckpointRefR\n" +
	"checkpointJ\x04\b\x04\x10\x05\"\xd3\x02\n" +
	"\x19WeightsIntentReadyRequest\x12,\n" +
	"\x12record_owner_epoch\x18\x01 \x01(\x04R\x10recordOwnerEpoch\x120\n" +
	"\x14control_stream_epoch\x18\x02 \x01(\x04R\x12controlStreamEpoch\x12$\n" +
	"\x0eworker_boot_id\x18\x03 \x01(\tR\fworkerBootId\x12B\n" +
	"\aweights\x18\x05 \x01(\v2(.cozy.worker.v1.WeightsCheckpointSubjectR\aweights\x12'\n" +
	"\x0fattempt_ordinal\x18\x06 \x01(\x04R\x0eattemptOrdinal\x12=\n" +
	"\n" +
	"checkpoint\x18\a \x01(\v2\x1d.cozy.worker.v1.CheckpointRefR\n" +
	"checkpointJ\x04\b\x04\x10\x05\"\xb6\x01\n" +
	" ValidateWeightsCheckpointRequest\x12A\n" +
	"\x06intent\x18\x01 \x01(\v2).cozy.worker.v1.WeightsIntentReadyRequestR\x06intent\x12O\n" +
	"$tensorfs_declaration_canonical_bytes\x18\x02 \x01(\fR!tensorfsDeclarationCanonicalBytes\"\xf8\x01\n" +
	"\x1fValidateWeightsCheckpointResult\x12B\n" +
	"\aweights\x18\x01 \x01(\v2(.cozy.worker.v1.WeightsCheckpointSubjectR\aweights\x12=\n" +
	"\n" +
	"checkpoint\x18\x02 \x01(\v2\x1d.cozy.worker.v1.CheckpointRefR\n" +
	"checkpoint\x12\x14\n" +
	"\x05valid\x18\x03 \x01(\bR\x05valid\x12\x1b\n" +
	"\tsafe_code\x18\x04 \x01(\tR\bsafeCode\x12\x1f\n" +
	"\vsafe_detail\x18\x05 \x01(\tR\n" +
	"safeDetail\"\xb6\x04\n" +
	"\x19WeightsTransactionRefused\x12,\n" +
	"\x12record_owner_epoch\x18\x01 \x01(\x04R\x10recordOwnerEpoch\x120\n" +
	"\x14control_stream_epoch\x18\x02 \x01(\x04R\x12controlStreamEpoch\x12$\n" +
	"\x0eworker_boot_id\x18\x03 \x01(\tR\fworkerBootId\x12\x1d\n" +
	"\n" +
	"request_id\x18\x05 \x01(\tR\trequestId\x12'\n" +
	"\x0fattempt_ordinal\x18\x06 \x01(\x04R\x0eattemptOrdinal\x124\n" +
	"\x16invocation_spec_digest\x18\a \x01(\fR\x14invocationSpecDigest\x12\x1f\n" +
	"\voutput_slot\x18\b \x01(\tR\n" +
	"outputSlot\x124\n" +
	"\x16weights_transaction_id\x18\t \x01(\tR\x14weightsTransactionId\x12!\n" +
	"\fwriter_epoch\x18\n" +
	" \x01(\x04R\vwriterEpoch\x126\n" +
	"\x05stage\x18\v \x01(\x0e2 .cozy.worker.v1.WeightsHostStageR\x05stage\x12<\n" +
	"\arefusal\x18\f \x01(\x0e2\".cozy.worker.v1.WeightsHostRefusalR\arefusal\x12\x1f\n" +
	"\vsafe_detail\x18\r \x01(\tR\n" +
	"safeDetailJ\x04\b\x04\x10\x05\"\x95\x05\n" +
	"\x13WeightsReceiptFrame\x12,\n" +
	"\x12record_owner_epoch\x18\x01 \x01(\x04R\x10recordOwnerEpoch\x120\n" +
	"\x14control_stream_epoch\x18\x02 \x01(\x04R\x12controlStreamEpoch\x12$\n" +
	"\x0eworker_boot_id\x18\x03 \x01(\tR\fworkerBootId\x12\x1d\n" +
	"\n" +
	"request_id\x18\x05 \x01(\tR\trequestId\x12'\n" +
	"\x0fattempt_ordinal\x18\x06 \x01(\x04R\x0eattemptOrdinal\x124\n" +
	"\x16invocation_spec_digest\x18\a \x01(\fR\x14invocationSpecDigest\x12\x1f\n" +
	"\voutput_slot\x18\b \x01(\tR\n" +
	"outputSlot\x124\n" +
	"\x16weights_transaction_id\x18\t \x01(\tR\x14weightsTransactionId\x12!\n" +
	"\fwriter_epoch\x18\n" +
	" \x01(\x04R\vwriterEpoch\x12>\n" +
	"\x1btensorfs_declaration_digest\x18\v \x01(\fR\x19tensorfsDeclarationDigest\x12J\n" +
	"\x0fweights_receipt\x18\f \x01(\v2!.cozy.worker.v1.WeightsReceiptRefR\x0eweightsReceipt\x12=\n" +
	"\aobjects\x18\r \x03(\v2#.cozy.worker.v1.WeightsObjectSourceR\aobjects\x12/\n" +
	"\bmanifest\x18\x0e \x01(\v2\x13.cozy.worker.v1.RefR\bmanifestJ\x04\b\x04\x10\x05\"\xa9\x04\n" +
	"\x18WeightsTransactionStatus\x124\n" +
	"\x16weights_transaction_id\x18\x01 \x01(\tR\x14weightsTransactionId\x12\x1d\n" +
	"\n" +
	"request_id\x18\x02 \x01(\tR\trequestId\x12'\n" +
	"\x0fattempt_ordinal\x18\x03 \x01(\x04R\x0eattemptOrdinal\x124\n" +
	"\x16invocation_spec_digest\x18\x04 \x01(\tR\x14invocationSpecDigest\x12\x1f\n" +
	"\voutput_slot\x18\x05 \x01(\tR\n" +
	"outputSlot\x12!\n" +
	"\fwriter_epoch\x18\x06 \x01(\x04R\vwriterEpoch\x12=\n" +
	"\x05state\x18\a \x01(\x0e2'.cozy.worker.v1.WeightsTransactionStateR\x05state\x12>\n" +
	"\x1btensorfs_declaration_digest\x18\b \x01(\fR\x19tensorfsDeclarationDigest\x124\n" +
	"\x16weights_receipt_digest\x18\t \x01(\fR\x14weightsReceiptDigest\x12=\n" +
	"\n" +
	"checkpoint\x18\n" +
	" \x01(\v2\x1d.cozy.worker.v1.CheckpointRefR\n" +
	"checkpoint\x12!\n" +
	"\fintent_ready\x18\v \x01(\bR\vintentReady\"G\n" +
	"\x10WeightsObjectRef\x12\x1b\n" +
	"\tobject_id\x18\x01 \x01(\tR\bobjectId\x12\x16\n" +
	"\x06length\x18\x02 \x01(\x04R\x06length\"?\n" +
	"\x13WeightsUploadHeader\x12\x12\n" +
	"\x04name\x18\x01 \x01(\tR\x04name\x12\x14\n" +
	"\x05value\x18\x02 \x01(\tR\x05value\"\xd3\x01\n" +
	"\x12WeightsUploadGrant\x12\x1b\n" +
	"\tobject_id\x18\x01 \x01(\tR\bobjectId\x12\x16\n" +
	"\x06length\x18\x02 \x01(\x04R\x06length\x12\x10\n" +
	"\x03url\x18\x03 \x01(\tR\x03url\x12N\n" +
	"\x10required_headers\x18\x04 \x03(\v2#.cozy.worker.v1.WeightsUploadHeaderR\x0frequiredHeaders\x12&\n" +
	"\x0fexpires_at_unix\x18\x05 \x01(\x04R\rexpiresAtUnix\"\xd3\x03\n" +
	"\x14WeightsUploadRequest\x12,\n" +
	"\x12record_owner_epoch\x18\x01 \x01(\x04R\x10recordOwnerEpoch\x120\n" +
	"\x14control_stream_epoch\x18\x02 \x01(\x04R\x12controlStreamEpoch\x12$\n" +
	"\x0eworker_boot_id\x18\x03 \x01(\tR\fworkerBootId\x124\n" +
	"\x16weights_transaction_id\x18\x05 \x01(\tR\x14weightsTransactionId\x12!\n" +
	"\fwriter_epoch\x18\x06 \x01(\x04R\vwriterEpoch\x12\x1b\n" +
	"\tobject_id\x18\a \x01(\tR\bobjectId\x12\x1d\n" +
	"\n" +
	"source_ref\x18\b \x01(\tR\tsourceRef\x12\x16\n" +
	"\x06length\x18\t \x01(\x04R\x06length\x12!\n" +
	"\foperation_id\x18\n" +
	" \x01(\tR\voperationId\x12%\n" +
	"\x0egrant_revision\x18\v \x01(\x04R\rgrantRevision\x128\n" +
	"\x05grant\x18\f \x01(\v2\".cozy.worker.v1.WeightsUploadGrantR\x05grantJ\x04\b\x04\x10\x05\"\x8d\x05\n" +
	"\x13WeightsUploadResult\x12,\n" +
	"\x12record_owner_epoch\x18\x01 \x01(\x04R\x10recordOwnerEpoch\x120\n" +
	"\x14control_stream_epoch\x18\x02 \x01(\x04R\x12controlStreamEpoch\x12$\n" +
	"\x0eworker_boot_id\x18\x03 \x01(\tR\fworkerBootId\x124\n" +
	"\x16weights_transaction_id\x18\x05 \x01(\tR\x14weightsTransactionId\x12!\n" +
	"\fwriter_epoch\x18\x06 \x01(\x04R\vwriterEpoch\x12\x1b\n" +
	"\tobject_id\x18\a \x01(\tR\bobjectId\x12!\n" +
	"\foperation_id\x18\b \x01(\tR\voperationId\x12%\n" +
	"\x0egrant_revision\x18\t \x01(\x04R\rgrantRevision\x12>\n" +
	"\aoutcome\x18\n" +
	" \x01(\x0e2$.cozy.worker.v1.WeightsUploadOutcomeR\aoutcome\x12+\n" +
	"\x11transferred_bytes\x18\v \x01(\x04R\x10transferredBytes\x12\x1f\n" +
	"\vhttp_status\x18\f \x01(\rR\n" +
	"httpStatus\x12\x12\n" +
	"\x04etag\x18\r \x01(\tR\x04etag\x12'\n" +
	"\x0fchecksum_sha256\x18\x0e \x01(\tR\x0echecksumSha256\x12>\n" +
	"\arefusal\x18\x0f \x01(\x0e2$.cozy.worker.v1.WeightsUploadRefusalR\arefusal\x12\x1f\n" +
	"\vsafe_detail\x18\x10 \x01(\tR\n" +
	"safeDetailJ\x04\b\x04\x10\x05\"\x90\x04\n" +
	"\x16ModelSourceFileRequest\x12,\n" +
	"\x12record_owner_epoch\x18\x01 \x01(\x04R\x10recordOwnerEpoch\x120\n" +
	"\x14control_stream_epoch\x18\x02 \x01(\x04R\x12controlStreamEpoch\x12$\n" +
	"\x0eworker_boot_id\x18\x03 \x01(\tR\fworkerBootId\x12!\n" +
	"\foperation_id\x18\x05 \x01(\tR\voperationId\x126\n" +
	"\x17source_selection_digest\x18\x06 \x01(\fR\x15sourceSelectionDigest\x12\x16\n" +
	"\x06member\x18\a \x01(\tR\x06member\x12\x1b\n" +
	"\tobject_id\x18\b \x01(\tR\bobjectId\x12\x16\n" +
	"\x06length\x18\t \x01(\x04R\x06length\x12?\n" +
	"\bprovider\x18\n" +
	" \x01(\x0e2#.cozy.worker.v1.ModelSourceProviderR\bprovider\x12\x10\n" +
	"\x03url\x18\v \x01(\tR\x03url\x12&\n" +
	"\x0fexpires_at_unix\x18\f \x01(\x04R\rexpiresAtUnix\x12/\n" +
	"\x13capability_revision\x18\r \x01(\x04R\x12capabilityRevision\x12\x16\n" +
	"\x06header\x18\x0e \x01(\fR\x06headerJ\x04\b\x04\x10\x05\"\xbf\x04\n" +
	"\x15ModelSourceFileStatus\x12,\n" +
	"\x12record_owner_epoch\x18\x01 \x01(\x04R\x10recordOwnerEpoch\x120\n" +
	"\x14control_stream_epoch\x18\x02 \x01(\x04R\x12controlStreamEpoch\x12$\n" +
	"\x0eworker_boot_id\x18\x03 \x01(\tR\fworkerBootId\x12!\n" +
	"\foperation_id\x18\x05 \x01(\tR\voperationId\x126\n" +
	"\x17source_selection_digest\x18\x06 \x01(\fR\x15sourceSelectionDigest\x12\x16\n" +
	"\x06member\x18\a \x01(\tR\x06member\x12\x1b\n" +
	"\tobject_id\x18\b \x01(\tR\bobjectId\x12\x16\n" +
	"\x06length\x18\t \x01(\x04R\x06length\x12/\n" +
	"\x13capability_revision\x18\n" +
	" \x01(\x04R\x12capabilityRevision\x12:\n" +
	"\x05state\x18\v \x01(\x0e2$.cozy.worker.v1.ModelSourceFileStateR\x05state\x12+\n" +
	"\x11transferred_bytes\x18\f \x01(\x04R\x10transferredBytes\x12\x1a\n" +
	"\battempts\x18\r \x01(\rR\battempts\x12\x1b\n" +
	"\tsafe_code\x18\x0e \x01(\tR\bsafeCode\x12\x1f\n" +
	"\vsafe_detail\x18\x0f \x01(\tR\n" +
	"safeDetailJ\x04\b\x04\x10\x05\"\xd5\x03\n" +
	"\x19ModelSourcePrepareRequest\x12,\n" +
	"\x12record_owner_epoch\x18\x01 \x01(\x04R\x10recordOwnerEpoch\x120\n" +
	"\x14control_stream_epoch\x18\x02 \x01(\x04R\x12controlStreamEpoch\x12$\n" +
	"\x0eworker_boot_id\x18\x03 \x01(\tR\fworkerBootId\x12!\n" +
	"\foperation_id\x18\x05 \x01(\tR\voperationId\x126\n" +
	"\x17source_selection_digest\x18\x06 \x01(\fR\x15sourceSelectionDigest\x12>\n" +
	"\bprofiles\x18\a \x03(\v2\".cozy.worker.v1.ModelSourceProfileR\bprofiles\x12\x1d\n" +
	"\n" +
	"source_uri\x18\b \x01(\tR\tsourceUri\x12)\n" +
	"\x10declared_license\x18\t \x01(\tR\x0fdeclaredLicense\x12G\n" +
	"\vcheckpoints\x18\n" +
	" \x03(\v2%.cozy.worker.v1.ModelSourceCheckpointR\vcheckpointsJ\x04\b\x04\x10\x05\"\x87\x04\n" +
	"\x13ModelSourcePrepared\x12,\n" +
	"\x12record_owner_epoch\x18\x01 \x01(\x04R\x10recordOwnerEpoch\x120\n" +
	"\x14control_stream_epoch\x18\x02 \x01(\x04R\x12controlStreamEpoch\x12$\n" +
	"\x0eworker_boot_id\x18\x03 \x01(\tR\fworkerBootId\x12!\n" +
	"\foperation_id\x18\x05 \x01(\tR\voperationId\x126\n" +
	"\x17source_selection_digest\x18\x06 \x01(\fR\x15sourceSelectionDigest\x12C\n" +
	"\aoutcome\x18\a \x01(\x0e2).cozy.worker.v1.ModelSourcePrepareOutcomeR\aoutcome\x12=\n" +
	"\asources\x18\b \x03(\v2#.cozy.worker.v1.PreparedModelSourceR\asources\x12\x1b\n" +
	"\tsafe_code\x18\t \x01(\tR\bsafeCode\x12\x1f\n" +
	"\vsafe_detail\x18\n" +
	" \x01(\tR\n" +
	"safeDetail\x12G\n" +
	"\vcheckpoints\x18\v \x03(\v2%.cozy.worker.v1.ModelSourceCheckpointR\vcheckpointsJ\x04\b\x04\x10\x05\"\xd6\x03\n" +
	"\x16LocalPackageFileStatus\x12,\n" +
	"\x12record_owner_epoch\x18\x01 \x01(\x04R\x10recordOwnerEpoch\x120\n" +
	"\x14control_stream_epoch\x18\x02 \x01(\x04R\x12controlStreamEpoch\x12$\n" +
	"\x0eworker_boot_id\x18\x03 \x01(\tR\fworkerBootId\x12!\n" +
	"\foperation_id\x18\x05 \x01(\tR\voperationId\x12\x16\n" +
	"\x06digest\x18\a \x01(\fR\x06digest\x12\x1a\n" +
	"\bfilename\x18\b \x01(\tR\bfilename\x12\x16\n" +
	"\x06length\x18\n" +
	" \x01(\x04R\x06length\x12;\n" +
	"\x05state\x18\v \x01(\x0e2%.cozy.worker.v1.LocalPackageFileStateR\x05state\x12%\n" +
	"\x0ereceived_bytes\x18\f \x01(\x04R\rreceivedBytes\x12\x1b\n" +
	"\tsafe_code\x18\r \x01(\tR\bsafeCode\x12\x1f\n" +
	"\vsafe_detail\x18\x0e \x01(\tR\n" +
	"safeDetailJ\x04\b\x04\x10\x05J\x04\b\x06\x10\aJ\x04\b\t\x10\n" +
	"R\rsource_digestR\x04kind\"\xc1\x04\n" +
	"\x16WeightsFinalizeRequest\x12,\n" +
	"\x12record_owner_epoch\x18\x01 \x01(\x04R\x10recordOwnerEpoch\x120\n" +
	"\x14control_stream_epoch\x18\x02 \x01(\x04R\x12controlStreamEpoch\x12$\n" +
	"\x0eworker_boot_id\x18\x03 \x01(\tR\fworkerBootId\x12\x1d\n" +
	"\n" +
	"request_id\x18\x05 \x01(\tR\trequestId\x124\n" +
	"\x16invocation_spec_digest\x18\x06 \x01(\fR\x14invocationSpecDigest\x12\x1f\n" +
	"\voutput_slot\x18\a \x01(\tR\n" +
	"outputSlot\x12L\n" +
	"\vdisposition\x18\b \x01(\x0e2*.cozy.worker.v1.WeightsFinalizeDispositionR\vdisposition\x124\n" +
	"\x16weights_receipt_digest\x18\t \x01(\fR\x14weightsReceiptDigest\x12&\n" +
	"\x0fscratch_root_id\x18\n" +
	" \x01(\tR\rscratchRootId\x122\n" +
	"\x15owner_authority_scope\x18\v \x01(\tR\x13ownerAuthorityScope\x12E\n" +
	"\x1finvocation_spec_canonical_bytes\x18\f \x01(\fR\x1cinvocationSpecCanonicalBytesJ\x04\b\x04\x10\x05\"\xdb\x03\n" +
	"\x15WeightsFinalizeResult\x12,\n" +
	"\x12record_owner_epoch\x18\x01 \x01(\x04R\x10recordOwnerEpoch\x120\n" +
	"\x14control_stream_epoch\x18\x02 \x01(\x04R\x12controlStreamEpoch\x12$\n" +
	"\x0eworker_boot_id\x18\x03 \x01(\tR\fworkerBootId\x12\x1d\n" +
	"\n" +
	"request_id\x18\x05 \x01(\tR\trequestId\x124\n" +
	"\x16invocation_spec_digest\x18\x06 \x01(\fR\x14invocationSpecDigest\x12\x1f\n" +
	"\voutput_slot\x18\a \x01(\tR\n" +
	"outputSlot\x12@\n" +
	"\aoutcome\x18\b \x01(\x0e2&.cozy.worker.v1.WeightsFinalizeOutcomeR\aoutcome\x12J\n" +
	"\x0fweights_receipt\x18\t \x01(\v2!.cozy.worker.v1.WeightsReceiptRefR\x0eweightsReceipt\x122\n" +
	"\x15owner_authority_scope\x18\n" +
	" \x01(\tR\x13ownerAuthorityScopeJ\x04\b\x04\x10\x05\"\xe1\x02\n" +
	"\x0eResultEnvelope\x120\n" +
	"\x14result_schema_digest\x18\x01 \x01(\fR\x12resultSchemaDigest\x12#\n" +
	"\rinline_result\x18\x02 \x01(\fR\finlineResult\x12<\n" +
	"\vresult_blob\x18\x03 \x01(\v2\x1b.cozy.worker.v1.OutputEntryR\n" +
	"resultBlob\x12?\n" +
	"\vadjustments\x18\x04 \x03(\v2\x1d.cozy.worker.v1.AdjustmentRowR\vadjustments\x12%\n" +
	"\x0echeckpoint_ref\x18\x05 \x01(\tR\rcheckpointRef\x12L\n" +
	"\x0fretained_models\x18\a \x03(\v2#.cozy.worker.v1.RetainedModelResultR\x0eretainedModelsJ\x04\b\x06\x10\a\"\xc8\x01\n" +
	"\x13RetainedModelResult\x12%\n" +
	"\x0eresult_pointer\x18\x01 \x01(\tR\rresultPointer\x12C\n" +
	"\x1emodel_artifact_canonical_bytes\x18\x02 \x01(\fR\x1bmodelArtifactCanonicalBytes\x12E\n" +
	"\tretention\x18\x03 \x01(\v2'.cozy.worker.v1.DerivedRetentionRequestR\tretention\"u\n" +
	"\rAdjustmentRow\x12\x14\n" +
	"\x05field\x18\x01 \x01(\tR\x05field\x12\x1c\n" +
	"\trequested\x18\x02 \x01(\tR\trequested\x12\x18\n" +
	"\aapplied\x18\x03 \x01(\tR\aapplied\x12\x16\n" +
	"\x06reason\x18\x04 \x01(\tR\x06reason\"\xcb\x01\n" +
	"\fOutcomeCause\x12-\n" +
	"\x04code\x18\x01 \x01(\x0e2\x19.cozy.worker.v1.CauseCodeR\x04code\x123\n" +
	"\x06origin\x18\x02 \x01(\x0e2\x1b.cozy.worker.v1.CauseOriginR\x06origin\x12\x16\n" +
	"\x06detail\x18\x03 \x01(\tR\x06detail\x12?\n" +
	"\tshortfall\x18\x04 \x01(\v2!.cozy.worker.v1.ResourceShortfallR\tshortfall\"\xdd\x01\n" +
	"\x11ResourceShortfall\x12\x1a\n" +
	"\bresource\x18\x01 \x01(\tR\bresource\x12\x14\n" +
	"\x05scope\x18\x02 \x01(\tR\x05scope\x12!\n" +
	"\fneeded_bytes\x18\x03 \x01(\x04R\vneededBytes\x12'\n" +
	"\x0favailable_bytes\x18\x04 \x01(\x04R\x0eavailableBytes\x12#\n" +
	"\rrequest_shape\x18\x05 \x01(\tR\frequestShape\x12%\n" +
	"\x0eevidence_class\x18\x06 \x01(\tR\revidenceClass\"\x84\x03\n" +
	"\x11AttemptOutcomeAck\x12,\n" +
	"\x12record_owner_epoch\x18\x01 \x01(\x04R\x10recordOwnerEpoch\x120\n" +
	"\x14control_stream_epoch\x18\x02 \x01(\x04R\x12controlStreamEpoch\x12$\n" +
	"\x0eworker_boot_id\x18\x03 \x01(\tR\fworkerBootId\x12\x1d\n" +
	"\n" +
	"request_id\x18\x05 \x01(\tR\trequestId\x12'\n" +
	"\x0fattempt_ordinal\x18\x06 \x01(\x04R\x0eattemptOrdinal\x124\n" +
	"\x16invocation_spec_digest\x18\a \x01(\fR\x14invocationSpecDigest\x12\x1d\n" +
	"\n" +
	"outcome_id\x18\b \x01(\tR\toutcomeId\x12%\n" +
	"\x0eoutcome_digest\x18\t \x01(\fR\routcomeDigest\x12\x1f\n" +
	"\vretain_work\x18\n" +
	" \x01(\bR\n" +
	"retainWorkJ\x04\b\x04\x10\x05\"\xa2\x03\n" +
	"\x14JobCheckpointRequest\x12,\n" +
	"\x12record_owner_epoch\x18\x01 \x01(\x04R\x10recordOwnerEpoch\x120\n" +
	"\x14control_stream_epoch\x18\x02 \x01(\x04R\x12controlStreamEpoch\x12$\n" +
	"\x0eworker_boot_id\x18\x03 \x01(\tR\fworkerBootId\x12\x1d\n" +
	"\n" +
	"request_id\x18\x05 \x01(\tR\trequestId\x12'\n" +
	"\x0fattempt_ordinal\x18\x06 \x01(\x04R\x0eattemptOrdinal\x12#\n" +
	"\roperation_key\x18\a \x01(\tR\foperationKey\x12\x1f\n" +
	"\vlogical_key\x18\b \x01(\tR\n" +
	"logicalKey\x12%\n" +
	"\x0econtent_digest\x18\t \x01(\fR\rcontentDigest\x12\x10\n" +
	"\x03seq\x18\n" +
	" \x01(\x04R\x03seq\x127\n" +
	"\bartifact\x18\v \x01(\v2\x1b.cozy.worker.v1.OutputEntryR\bartifactJ\x04\b\x04\x10\x05\"\xb9\x01\n" +
	"\fProgressOpen\x12,\n" +
	"\x12record_owner_epoch\x18\x01 \x01(\x04R\x10recordOwnerEpoch\x120\n" +
	"\x14control_stream_epoch\x18\x02 \x01(\x04R\x12controlStreamEpoch\x12$\n" +
	"\x0eworker_boot_id\x18\x03 \x01(\tR\fworkerBootId\x12\x1d\n" +
	"\n" +
	"request_id\x18\x05 \x01(\tR\trequestIdJ\x04\b\x04\x10\x05\"\xd1\x02\n" +
	"\x0fAttemptProgress\x12,\n" +
	"\x12record_owner_epoch\x18\x01 \x01(\x04R\x10recordOwnerEpoch\x120\n" +
	"\x14control_stream_epoch\x18\x02 \x01(\x04R\x12controlStreamEpoch\x12$\n" +
	"\x0eworker_boot_id\x18\x03 \x01(\tR\fworkerBootId\x12\x1d\n" +
	"\n" +
	"request_id\x18\x05 \x01(\tR\trequestId\x12'\n" +
	"\x0fattempt_ordinal\x18\x06 \x01(\x04R\x0eattemptOrdinal\x12\x10\n" +
	"\x03seq\x18\a \x01(\x04R\x03seq\x12!\n" +
	"\fcontent_type\x18\b \x01(\tR\vcontentType\x12\x12\n" +
	"\x04data\x18\t \x01(\fR\x04data\x12!\n" +
	"\fplacement_id\x18\n" +
	" \x01(\tR\vplacementIdJ\x04\b\x04\x10\x05\"\xc5\x04\n" +
	"\x0eInvocationSpec\x12%\n" +
	"\x0epayload_digest\x18\x04 \x01(\tR\rpayloadDigest\x124\n" +
	"\x06inputs\x18\x05 \x03(\v2\x1c.cozy.worker.v1.InputBindingR\x06inputs\x127\n" +
	"\aoutputs\x18\x06 \x03(\v2\x1d.cozy.worker.v1.OutputBindingR\aoutputs\x12(\n" +
	"\x10deadline_unix_ms\x18\a \x01(\x04R\x0edeadlineUnixMs\x12A\n" +
	"\aserving\x18\b \x01(\v2%.cozy.worker.v1.ServingInvocationSpecH\x00R\aserving\x125\n" +
	"\x03job\x18\t \x01(\v2!.cozy.worker.v1.JobInvocationSpecH\x00R\x03job\x12;\n" +
	"\acapture\x18\n" +
	" \x01(\v2!.cozy.worker.v1.ActivationCaptureR\acapture\x12)\n" +
	"\x10attention_kernel\x18\v \x01(\tR\x0fattentionKernel\x12'\n" +
	"\x0finstallation_id\x18\f \x01(\tR\x0einstallationIdB\x06\n" +
	"\x04specJ\x04\b\x01\x10\x02J\x04\b\x03\x10\x04J\x04\b\x02\x10\x03R\x17package_revision_digestR\x12package_release_idR\rconfig_digestR\x12environment_digest\"\x8c\x01\n" +
	"\fInputBinding\x12\x19\n" +
	"\binput_id\x18\x01 \x01(\tR\ainputId\x12\x16\n" +
	"\x06digest\x18\x02 \x01(\tR\x06digest\x12\x16\n" +
	"\x06length\x18\x03 \x01(\x04R\x06length\x12\x1b\n" +
	"\tkind_mime\x18\x04 \x01(\tR\bkindMime\x12\x14\n" +
	"\x05order\x18\x05 \x01(\rR\x05order\"f\n" +
	"\rOutputBinding\x12\x1b\n" +
	"\toutput_id\x18\x01 \x01(\tR\boutputId\x12\x1b\n" +
	"\tmime_type\x18\x02 \x01(\tR\bmimeType\x12\x1b\n" +
	"\tmax_bytes\x18\x03 \x01(\x04R\bmaxBytes\"\xaa\x01\n" +
	"\x15ServingInvocationSpec\x12:\n" +
	"\x19entrypoint_binding_digest\x18\x01 \x01(\tR\x17entrypointBindingDigest\x12,\n" +
	"\x12attempt_binding_id\x18\x02 \x01(\tR\x10attemptBindingId\x12'\n" +
	"\x0fbindings_digest\x18\x03 \x01(\tR\x0ebindingsDigest\"\xd0\x01\n" +
	"\x11JobInvocationSpec\x12*\n" +
	"\x11job_descriptor_id\x18\x02 \x01(\tR\x0fjobDescriptorId\x12V\n" +
	"\x14publication_contract\x18\x03 \x01(\v2#.cozy.worker.v1.PublicationContractR\x13publicationContract\x12'\n" +
	"\x0finstallation_id\x18\x04 \x01(\tR\x0einstallationIdJ\x04\b\x01\x10\x02R\bbuild_id\"\xc8\x02\n" +
	"\rDeliveryGrant\x124\n" +
	"\x16invocation_spec_digest\x18\x01 \x01(\fR\x14invocationSpecDigest\x12H\n" +
	"\n" +
	"credential\x18\x02 \x01(\v2(.cozy.worker.v1.DeliveryAccessCredentialR\n" +
	"credential\x12\"\n" +
	"\rfile_base_url\x18\x03 \x01(\tR\vfileBaseUrl\x12&\n" +
	"\x0fexpires_at_unix\x18\x04 \x01(\x04R\rexpiresAtUnix\x123\n" +
	"\x06inputs\x18\x05 \x03(\v2\x1b.cozy.worker.v1.InputAccessR\x06inputs\x126\n" +
	"\aoutputs\x18\x06 \x03(\v2\x1c.cozy.worker.v1.OutputAccessR\aoutputs\"4\n" +
	"\x12CatalogModelSource\x12\x1e\n" +
	"\n" +
	"repository\x18\x01 \x01(\tR\n" +
	"repository\"\xd0\x01\n" +
	"\vInputAccess\x12\x19\n" +
	"\binput_id\x18\x01 \x01(\tR\ainputId\x12\x10\n" +
	"\x03url\x18\x02 \x01(\tR\x03url\x12K\n" +
	"\vnative_tree\x18\x03 \x01(\v2*.cozy.worker.v1.NativeByteRetentionRequestR\n" +
	"nativeTree\x12G\n" +
	"\rcatalog_model\x18\x04 \x01(\v2\".cozy.worker.v1.CatalogModelSourceR\fcatalogModel\"=\n" +
	"\fOutputAccess\x12\x1b\n" +
	"\toutput_id\x18\x01 \x01(\tR\boutputId\x12\x10\n" +
	"\x03url\x18\x02 \x01(\tR\x03url\"\xb2\x01\n" +
	"\x18DeliveryAccessCredential\x12\x16\n" +
	"\x06issuer\x18\x01 \x01(\tR\x06issuer\x12\x15\n" +
	"\x06key_id\x18\x02 \x01(\tR\x05keyId\x12)\n" +
	"\x10credential_epoch\x18\x03 \x01(\x04R\x0fcredentialEpoch\x12&\n" +
	"\x0fexpires_at_unix\x18\x04 \x01(\x04R\rexpiresAtUnix\x12\x14\n" +
	"\x05token\x18\x05 \x01(\fR\x05token\"\xb8\x01\n" +
	"\fResourceCaps\x12'\n" +
	"\x0fdevice_required\x18\x01 \x01(\bR\x0edeviceRequired\x125\n" +
	"\x17max_device_memory_bytes\x18\x02 \x01(\x04R\x14maxDeviceMemoryBytes\x12\"\n" +
	"\rmax_rss_bytes\x18\x03 \x01(\x04R\vmaxRssBytes\x12$\n" +
	"\x0emax_disk_bytes\x18\x04 \x01(\x04R\fmaxDiskBytes\"i\n" +
	"\x13PublicationContract\x127\n" +
	"\aoutputs\x18\x01 \x03(\v2\x1d.cozy.worker.v1.OutputBindingR\aoutputs\x12\x19\n" +
	"\bgrant_id\x18\x02 \x01(\tR\agrantId\"\xed\x04\n" +
	"\x0fWorkerResources\x12\x1a\n" +
	"\bplatform\x18\x01 \x01(\tR\bplatform\x12\x18\n" +
	"\abackend\x18\x02 \x01(\tR\abackend\x12!\n" +
	"\fmemory_model\x18\x03 \x01(\tR\vmemoryModel\x12!\n" +
	"\fdevice_count\x18\x04 \x01(\rR\vdeviceCount\x12\x1f\n" +
	"\vdevice_name\x18\x05 \x01(\tR\n" +
	"deviceName\x129\n" +
	"\x19device_memory_total_bytes\x18\x06 \x01(\x04R\x16deviceMemoryTotalBytes\x12%\n" +
	"\x0edriver_version\x18\a \x01(\tR\rdriverVersion\x12'\n" +
	"\x0fbackend_version\x18\b \x01(\tR\x0ebackendVersion\x12/\n" +
	"\x14host_ram_total_bytes\x18\t \x01(\x04R\x11hostRamTotalBytes\x12\x1d\n" +
	"\n" +
	"vcpu_count\x18\n" +
	" \x01(\rR\tvcpuCount\x12(\n" +
	"\x10disk_total_bytes\x18\v \x01(\x04R\x0ediskTotalBytes\x12&\n" +
	"\x0fpower_cap_watts\x18\f \x01(\rR\rpowerCapWatts\x12+\n" +
	"\x11partition_profile\x18\r \x01(\tR\x10partitionProfile\x12\"\n" +
	"\finterconnect\x18\x0e \x01(\tR\finterconnect\x12\x1f\n" +
	"\vpeer_access\x18\x0f \x01(\bR\n" +
	"peerAccess\x12\x1e\n" +
	"\n" +
	"unreadable\x18\x10 \x03(\tR\n" +
	"unreadable\"\xf9\x01\n" +
	"\vJobCapacity\x12$\n" +
	"\x0ejobs_in_flight\x18\x01 \x01(\rR\fjobsInFlight\x12%\n" +
	"\x0ejobs_available\x18\x02 \x01(\rR\rjobsAvailable\x12&\n" +
	"\x0ffree_disk_bytes\x18\x04 \x01(\x04R\rfreeDiskBytes\x126\n" +
	"\x17orchestration_in_flight\x18\x05 \x01(\rR\x15orchestrationInFlight\x127\n" +
	"\x17orchestration_available\x18\x06 \x01(\rR\x16orchestrationAvailableJ\x04\b\x03\x10\x04\"\xa8\x04\n" +
	"\vHeldAttempt\x12\x1d\n" +
	"\n" +
	"request_id\x18\x01 \x01(\tR\trequestId\x12'\n" +
	"\x0fattempt_ordinal\x18\x02 \x01(\x04R\x0eattemptOrdinal\x12/\n" +
	"\x04kind\x18\x03 \x01(\x0e2\x1b.cozy.worker.v1.AttemptKindR\x04kind\x122\n" +
	"\x05state\x18\x04 \x01(\x0e2\x1c.cozy.worker.v1.AttemptStateR\x05state\x124\n" +
	"\x16invocation_spec_digest\x18\x05 \x01(\fR\x14invocationSpecDigest\x12!\n" +
	"\fplacement_id\x18\x06 \x01(\tR\vplacementId\x12%\n" +
	"\x0eexecutor_epoch\x18\a \x01(\x04R\rexecutorEpoch\x12\x1d\n" +
	"\n" +
	"outcome_id\x18\b \x01(\tR\toutcomeId\x12%\n" +
	"\x0eoutcome_digest\x18\t \x01(\fR\routcomeDigest\x12\x17\n" +
	"\alane_id\x18\n" +
	" \x01(\tR\x06laneId\x12%\n" +
	"\x0equeue_position\x18\v \x01(\rR\rqueuePosition\x12\x1c\n" +
	"\tovertaken\x18\f \x01(\rR\tovertaken\x12'\n" +
	"\x0fovertake_budget\x18\r \x01(\rR\x0eovertakeBudget\x12\x1f\n" +
	"\vplan_digest\x18\x0e \x01(\fR\n" +
	"planDigest\"\xb6\x01\n" +
	"\x05Fault\x12-\n" +
	"\x04kind\x18\x01 \x01(\x0e2\x19.cozy.worker.v1.FaultKindR\x04kind\x12\x18\n" +
	"\asubject\x18\x02 \x01(\tR\asubject\x12\x16\n" +
	"\x06reason\x18\x03 \x01(\tR\x06reason\x12\x16\n" +
	"\x06detail\x18\x04 \x01(\tR\x06detail\x124\n" +
	"\x16desired_state_revision\x18\x05 \x01(\x04R\x14desiredStateRevision\"\x8b\x01\n" +
	"\x0eOutputManifest\x12<\n" +
	"\x1apublication_receipt_digest\x18\x01 \x01(\tR\x18publicationReceiptDigest\x125\n" +
	"\aoutputs\x18\x03 \x03(\v2\x1b.cozy.worker.v1.OutputEntryR\aoutputsJ\x04\b\x02\x10\x03\"\xbb\x01\n" +
	"\vOutputEntry\x12\x1b\n" +
	"\toutput_id\x18\x01 \x01(\tR\boutputId\x12\x16\n" +
	"\x06digest\x18\x02 \x01(\fR\x06digest\x12\x16\n" +
	"\x06length\x18\x03 \x01(\x04R\x06length\x12\x1b\n" +
	"\tmime_type\x18\x04 \x01(\tR\bmimeType\x12B\n" +
	"\vnative_tree\x18\x05 \x01(\v2!.cozy.worker.v1.NativeByteTreeRefR\n" +
	"nativeTree\"\xca\x04\n" +
	"\x0eAttemptMetrics\x12\x1d\n" +
	"\n" +
	"runtime_ms\x18\x01 \x01(\x04R\truntimeMs\x12\x19\n" +
	"\bqueue_ms\x18\x02 \x01(\x04R\aqueueMs\x127\n" +
	"\x18peak_device_memory_bytes\x18\x03 \x01(\x04R\x15peakDeviceMemoryBytes\x12'\n" +
	"\x10rss_at_end_bytes\x18\x04 \x01(\x04R\rrssAtEndBytes\x12!\n" +
	"\foutput_count\x18\x05 \x01(\rR\voutputCount\x12!\n" +
	"\finput_tokens\x18\x06 \x01(\x04R\vinputTokens\x12#\n" +
	"\routput_tokens\x18\a \x01(\x04R\foutputTokens\x12&\n" +
	"\x0fdevice_lease_ms\x18\b \x01(\x04R\rdeviceLeaseMs\x12!\n" +
	"\fdevice_count\x18\t \x01(\rR\vdeviceCount\x12\x1d\n" +
	"\n" +
	"handler_ms\x18\n" +
	" \x01(\x04R\thandlerMs\x12'\n" +
	"\x0ffinalization_ms\x18\v \x01(\x04R\x0efinalizationMs\x12+\n" +
	"\x11unverified_fields\x18\f \x03(\tR\x10unverifiedFields\x12\x17\n" +
	"\apost_ms\x18\r \x01(\x04R\x06postMs\x129\n" +
	"\x19working_peak_device_bytes\x18\x0e \x01(\x04R\x16workingPeakDeviceBytes\x12\x1d\n" +
	"\n" +
	"shape_cell\x18\x0f \x01(\tR\tshapeCell\"\x80\x01\n" +
	"\x0fTriageBundleRef\x12\x1d\n" +
	"\n" +
	"subject_id\x18\x01 \x01(\tR\tsubjectId\x120\n" +
	"\x14write_receipt_digest\x18\x04 \x01(\fR\x12writeReceiptDigest\x12\x16\n" +
	"\x06length\x18\x03 \x01(\x04R\x06lengthJ\x04\b\x02\x10\x03\"Z\n" +
	"\x11ForgetPackageCall\x12+\n" +
	"\x05claim\x18\x01 \x01(\v2\x15.cozy.worker.v1.ClaimR\x05claim\x12\x18\n" +
	"\apackage\x18\x02 \x01(\tR\apackage\"\x15\n" +
	"\x13ForgetPackageResult\"\x8b\x01\n" +
	"\x1aNativeArtifactTransferCall\x12+\n" +
	"\x05claim\x18\x01 \x01(\v2\x15.cozy.worker.v1.ClaimR\x05claim\x12@\n" +
	"\arequest\x18\x02 \x01(\v2&.cozy.worker.v1.NativeArtifactTransferR\arequest\"\xd2\x04\n" +
	"\x16NativeArtifactTransfer\x12,\n" +
	"\x12record_owner_epoch\x18\x01 \x01(\x04R\x10recordOwnerEpoch\x120\n" +
	"\x14control_stream_epoch\x18\x02 \x01(\x04R\x12controlStreamEpoch\x12$\n" +
	"\x0eworker_boot_id\x18\x03 \x01(\tR\fworkerBootId\x12\x1b\n" +
	"\teffect_id\x18\x04 \x01(\tR\beffectId\x12?\n" +
	"\x06source\x18\x05 \x01(\v2'.cozy.worker.v1.DerivedRetentionRequestR\x06source\x12/\n" +
	"\bmanifest\x18\x06 \x01(\v2\x13.cozy.worker.v1.RefR\bmanifest\x12\x1d\n" +
	"\n" +
	"command_id\x18\a \x01(\x04R\tcommandId\x12\x16\n" +
	"\x06offset\x18\b \x01(\rR\x06offset\x12\x14\n" +
	"\x05limit\x18\t \x01(\rR\x05limit\x128\n" +
	"\x05grant\x18\n" +
	" \x01(\v2\".cozy.worker.v1.WeightsUploadGrantR\x05grant\x12%\n" +
	"\x0egrant_revision\x18\v \x01(\x04R\rgrantRevision\x12(\n" +
	"\x10server_time_unix\x18\f \x01(\x03R\x0eserverTimeUnix\x12K\n" +
	"\vbyte_source\x18\r \x01(\v2*.cozy.worker.v1.NativeByteRetentionRequestR\n" +
	"byteSource\"\xf7\x06\n" +
	"\x1cNativeArtifactTransferStatus\x12,\n" +
	"\x12record_owner_epoch\x18\x01 \x01(\x04R\x10recordOwnerEpoch\x120\n" +
	"\x14control_stream_epoch\x18\x02 \x01(\x04R\x12controlStreamEpoch\x12$\n" +
	"\x0eworker_boot_id\x18\x03 \x01(\tR\fworkerBootId\x12\x1b\n" +
	"\teffect_id\x18\x04 \x01(\tR\beffectId\x12?\n" +
	"\x06source\x18\x05 \x01(\v2'.cozy.worker.v1.DerivedRetentionRequestR\x06source\x12/\n" +
	"\bmanifest\x18\x06 \x01(\v2\x13.cozy.worker.v1.RefR\bmanifest\x12\x1d\n" +
	"\n" +
	"command_id\x18\a \x01(\x04R\tcommandId\x12:\n" +
	"\aobjects\x18\b \x03(\v2 .cozy.worker.v1.WeightsObjectRefR\aobjects\x12\x1f\n" +
	"\vnext_offset\x18\t \x01(\rR\n" +
	"nextOffset\x12\x19\n" +
	"\bhas_more\x18\n" +
	" \x01(\bR\ahasMore\x12%\n" +
	"\x0eclosure_digest\x18\v \x01(\fR\rclosureDigest\x12\x1b\n" +
	"\tobject_id\x18\f \x01(\tR\bobjectId\x12%\n" +
	"\x0egrant_revision\x18\r \x01(\x04R\rgrantRevision\x12>\n" +
	"\aoutcome\x18\x0e \x01(\x0e2$.cozy.worker.v1.WeightsUploadOutcomeR\aoutcome\x12+\n" +
	"\x11transferred_bytes\x18\x0f \x01(\x04R\x10transferredBytes\x12'\n" +
	"\x0fchecksum_sha256\x18\x10 \x01(\tR\x0echecksumSha256\x12\x1f\n" +
	"\vhttp_status\x18\x11 \x01(\rR\n" +
	"httpStatus\x12\x1b\n" +
	"\tsafe_code\x18\x12 \x01(\tR\bsafeCode\x12\x1f\n" +
	"\vsafe_detail\x18\x13 \x01(\tR\n" +
	"safeDetail\x12K\n" +
	"\vbyte_source\x18\x14 \x01(\v2*.cozy.worker.v1.NativeByteRetentionRequestR\n" +
	"byteSource\"\xba\x01\n" +
	"\x11NativeByteTreeRef\x12(\n" +
	"\x10producer_root_id\x18\x01 \x01(\tR\x0eproducerRootId\x12%\n" +
	"\x0ereceipt_digest\x18\x02 \x01(\fR\rreceiptDigest\x12/\n" +
	"\bmanifest\x18\x03 \x01(\v2\x13.cozy.worker.v1.RefR\bmanifest\x12#\n" +
	"\rcontent_bytes\x18\x04 \x01(\x04R\fcontentBytes\"z\n" +
	"\x1aNativeByteRetentionRequest\x129\n" +
	"\x06source\x18\x01 \x01(\v2!.cozy.worker.v1.NativeByteTreeRefR\x06source\x12!\n" +
	"\fretention_id\x18\x02 \x01(\tR\vretentionId\"\x8c\x01\n" +
	"\x17NativeByteRetentionCall\x12+\n" +
	"\x05claim\x18\x01 \x01(\v2\x15.cozy.worker.v1.ClaimR\x05claim\x12D\n" +
	"\arequest\x18\x02 \x01(\v2*.cozy.worker.v1.NativeByteRetentionRequestR\arequest\"\x95\x01\n" +
	"\x19NativeByteRetentionResult\x129\n" +
	"\x06source\x18\x01 \x01(\v2!.cozy.worker.v1.NativeByteTreeRefR\x06source\x12!\n" +
	"\fretention_id\x18\x02 \x01(\tR\vretentionId\x12\x1a\n" +
	"\breleased\x18\x03 \x01(\bR\breleased\"\x91\x01\n" +
	"\x14ChildByteResultGrant\x12\x1b\n" +
	"\toutput_id\x18\x01 \x01(\tR\boutputId\x129\n" +
	"\x06source\x18\x02 \x01(\v2!.cozy.worker.v1.NativeByteTreeRefR\x06source\x12!\n" +
	"\fretention_id\x18\x03 \x01(\tR\vretentionId\"\x8e\x02\n" +
	"\x15InputTreeImportHeader\x12+\n" +
	"\x05claim\x18\x01 \x01(\v2\x15.cozy.worker.v1.ClaimR\x05claim\x12\x1d\n" +
	"\n" +
	"request_id\x18\x02 \x01(\tR\trequestId\x12\x19\n" +
	"\binput_id\x18\x03 \x01(\tR\ainputId\x12/\n" +
	"\bmanifest\x18\x04 \x01(\v2\x13.cozy.worker.v1.RefR\bmanifest\x128\n" +
	"\x18manifest_canonical_bytes\x18\x05 \x01(\fR\x16manifestCanonicalBytes\x12#\n" +
	"\rcontent_bytes\x18\x06 \x01(\x04R\fcontentBytes\"n\n" +
	"\x13InputTreeImportBlob\x12+\n" +
	"\x06object\x18\x01 \x01(\v2\x13.cozy.worker.v1.RefR\x06object\x12\x16\n" +
	"\x06offset\x18\x02 \x01(\x04R\x06offset\x12\x12\n" +
	"\x04data\x18\x03 \x01(\fR\x04data\"-\n" +
	"\x15InputTreeImportCommit\x12\x14\n" +
	"\x05abort\x18\x01 \x01(\bR\x05abort\"\xdb\x01\n" +
	"\x14InputTreeImportFrame\x12?\n" +
	"\x06header\x18\x01 \x01(\v2%.cozy.worker.v1.InputTreeImportHeaderH\x00R\x06header\x129\n" +
	"\x04blob\x18\x02 \x01(\v2#.cozy.worker.v1.InputTreeImportBlobH\x00R\x04blob\x12?\n" +
	"\x06commit\x18\x03 \x01(\v2%.cozy.worker.v1.InputTreeImportCommitH\x00R\x06commitB\x06\n" +
	"\x04body\"\xca\x01\n" +
	"\x12NativeByteReadCall\x12+\n" +
	"\x05claim\x18\x01 \x01(\v2\x15.cozy.worker.v1.ClaimR\x05claim\x12B\n" +
	"\x06source\x18\x02 \x01(\v2*.cozy.worker.v1.NativeByteRetentionRequestR\x06source\x12+\n" +
	"\x06object\x18\x03 \x01(\v2\x13.cozy.worker.v1.RefR\x06object\x12\x16\n" +
	"\x06offset\x18\x04 \x01(\x04R\x06offset\"A\n" +
	"\x13NativeByteReadChunk\x12\x16\n" +
	"\x06offset\x18\x01 \x01(\x04R\x06offset\x12\x12\n" +
	"\x04data\x18\x02 \x01(\fR\x04data\"I\n" +
	"\x11ActivationCapture\x12\x1e\n" +
	"\n" +
	"components\x18\x01 \x03(\tR\n" +
	"components\x12\x14\n" +
	"\x05steps\x18\x02 \x03(\rR\x05steps\"\xeb\x02\n" +
	"\x14ExecutionEnvironment\x12'\n" +
	"\x0fruntime_version\x18\x01 \x01(\tR\x0eruntimeVersion\x12.\n" +
	"\x13worker_image_digest\x18\x02 \x01(\fR\x11workerImageDigest\x12 \n" +
	"\vaccelerator\x18\x03 \x01(\tR\vaccelerator\x12\x16\n" +
	"\x06driver\x18\x04 \x01(\tR\x06driver\x12\x12\n" +
	"\x04cuda\x18\x05 \x01(\tR\x04cuda\x12$\n" +
	"\x0eworker_boot_id\x18\x06 \x01(\tR\fworkerBootId\x12%\n" +
	"\x0eexecution_lane\x18\a \x01(\tR\rexecutionLane\x12:\n" +
	"\x19execution_contract_digest\x18\b \x01(\fR\x17executionContractDigest\x12#\n" +
	"\rkernel_symbol\x18\t \x01(\tR\fkernelSymbol\"]\n" +
	"\x17ActivationCaptureResult\x12\x1b\n" +
	"\toutput_id\x18\x01 \x01(\tR\boutputId\x12%\n" +
	"\x0econtent_digest\x18\x02 \x01(\fR\rcontentDigest\"\xa1\x01\n" +
	"\x14ExecutionObservation\x12F\n" +
	"\venvironment\x18\x01 \x01(\v2$.cozy.worker.v1.ExecutionEnvironmentR\venvironment\x12A\n" +
	"\acapture\x18\x02 \x01(\v2'.cozy.worker.v1.ActivationCaptureResultR\acapture\"\xe6\x01\n" +
	"\x0fRuntimeRevision\x12!\n" +
	"\fwheel_digest\x18\x01 \x01(\fR\vwheelDigest\x12!\n" +
	"\fwheel_length\x18\x02 \x01(\x04R\vwheelLength\x12'\n" +
	"\x0fruntime_version\x18\x03 \x01(\tR\x0eruntimeVersion\x12\x1d\n" +
	"\n" +
	"python_abi\x18\x04 \x01(\tR\tpythonAbi\x12+\n" +
	"\x11provenance_digest\x18\x05 \x01(\fR\x10provenanceDigest\x12\x18\n" +
	"\achannel\x18\x06 \x01(\tR\achannel*M\n" +
	"\n" +
	"MachineLog\x12\x1b\n" +
	"\x17MACHINE_LOG_UNSPECIFIED\x10\x00\x12\"\n" +
	"\x1eMACHINE_LOG_TENSORFS_TRANSPORT\x10\x01*a\n" +
	"\fRunProductOp\x12\x1e\n" +
	"\x1aRUN_PRODUCT_OP_UNSPECIFIED\x10\x00\x12\x16\n" +
	"\x12RUN_PRODUCT_OP_SET\x10\x01\x12\x19\n" +
	"\x15RUN_PRODUCT_OP_APPEND\x10\x02*\x8e\x02\n" +
	"\x16MachineExecutionAction\x12(\n" +
	"$MACHINE_EXECUTION_ACTION_UNSPECIFIED\x10\x00\x12\"\n" +
	"\x1eMACHINE_EXECUTION_ACTION_PAUSE\x10\x01\x12#\n" +
	"\x1fMACHINE_EXECUTION_ACTION_RESUME\x10\x02\x12#\n" +
	"\x1fMACHINE_EXECUTION_ACTION_CANCEL\x10\x03\x122\n" +
	".MACHINE_EXECUTION_ACTION_RECONCILE_PUBLICATION\x10\x04\x12(\n" +
	"$MACHINE_EXECUTION_ACTION_ANSWER_MEMO\x10\x05*\xca\x01\n" +
	"\x0eChildCallState\x12 \n" +
	"\x1cCHILD_CALL_STATE_UNSPECIFIED\x10\x00\x12\x1c\n" +
	"\x18CHILD_CALL_STATE_PENDING\x10\x01\x12\x1e\n" +
	"\x1aCHILD_CALL_STATE_SUCCEEDED\x10\x02\x12\x1c\n" +
	"\x18CHILD_CALL_STATE_REFUSED\x10\x03\x12\x1b\n" +
	"\x17CHILD_CALL_STATE_FAILED\x10\x04\x12\x1d\n" +
	"\x19CHILD_CALL_STATE_CANCELED\x10\x05*\x9a\x01\n" +
	"\x11NativeSourcePhase\x12#\n" +
	"\x1fNATIVE_SOURCE_PHASE_UNSPECIFIED\x10\x00\x12\x1f\n" +
	"\x1bNATIVE_SOURCE_PHASE_RESOLVE\x10\x01\x12\x1f\n" +
	"\x1bNATIVE_SOURCE_PHASE_EXECUTE\x10\x02\x12\x1e\n" +
	"\x1aNATIVE_SOURCE_PHASE_CANCEL\x10\x03*\x86\x02\n" +
	"\x15NativeSourceOperation\x12'\n" +
	"#NATIVE_SOURCE_OPERATION_UNSPECIFIED\x10\x00\x12'\n" +
	"#NATIVE_SOURCE_OPERATION_HUGGINGFACE\x10\x01\x12#\n" +
	"\x1fNATIVE_SOURCE_OPERATION_CIVITAI\x10\x02\x12#\n" +
	"\x1fNATIVE_SOURCE_OPERATION_CONVERT\x10\x03\x12(\n" +
	"$NATIVE_SOURCE_OPERATION_SOURCE_FILES\x10\x04\x12'\n" +
	"#NATIVE_SOURCE_OPERATION_COMMIT_FILE\x10\x05*\xbf\x01\n" +
	"\x11NativeSourceState\x12#\n" +
	"\x1fNATIVE_SOURCE_STATE_UNSPECIFIED\x10\x00\x12 \n" +
	"\x1cNATIVE_SOURCE_STATE_RESOLVED\x10\x01\x12!\n" +
	"\x1dNATIVE_SOURCE_STATE_SUCCEEDED\x10\x02\x12\x1e\n" +
	"\x1aNATIVE_SOURCE_STATE_FAILED\x10\x03\x12 \n" +
	"\x1cNATIVE_SOURCE_STATE_CANCELED\x10\x04*O\n" +
	"\aPosture\x12\x17\n" +
	"\x13POSTURE_UNSPECIFIED\x10\x00\x12\x15\n" +
	"\x11POSTURE_ACCEPTING\x10\x01\x12\x14\n" +
	"\x10POSTURE_DRAINING\x10\x02*\x92\x01\n" +
	"\vWorkerPhase\x12\x1c\n" +
	"\x18WORKER_PHASE_UNSPECIFIED\x10\x00\x12\x18\n" +
	"\x14WORKER_PHASE_BOOTING\x10\x01\x12\x17\n" +
	"\x13WORKER_PHASE_ONLINE\x10\x02\x12\x19\n" +
	"\x15WORKER_PHASE_DRAINING\x10\x03\x12\x17\n" +
	"\x13WORKER_PHASE_FAILED\x10\x04*\xcc\x01\n" +
	"\x14MaterializationState\x12%\n" +
	"!MATERIALIZATION_STATE_UNSPECIFIED\x10\x00\x12 \n" +
	"\x1cMATERIALIZATION_STATE_ABSENT\x10\x01\x12'\n" +
	"#MATERIALIZATION_STATE_MATERIALIZING\x10\x02\x12 \n" +
	"\x1cMATERIALIZATION_STATE_STAGED\x10\x03\x12 \n" +
	"\x1cMATERIALIZATION_STATE_FAILED\x10\x04*\xa2\x01\n" +
	"\fServingState\x12\x1d\n" +
	"\x19SERVING_STATE_UNSPECIFIED\x10\x00\x12\x19\n" +
	"\x15SERVING_STATE_OFFLINE\x10\x01\x12\x1c\n" +
	"\x18SERVING_STATE_ACTIVATING\x10\x02\x12\x1e\n" +
	"\x1aSERVING_STATE_DISPATCHABLE\x10\x03\x12\x1a\n" +
	"\x16SERVING_STATE_DRAINING\x10\x04*g\n" +
	"\x0eAdmissionState\x12\x1f\n" +
	"\x1bADMISSION_STATE_UNSPECIFIED\x10\x00\x12\x18\n" +
	"\x14ADMISSION_STATE_OPEN\x10\x01\x12\x1a\n" +
	"\x16ADMISSION_STATE_CLOSED\x10\x02*[\n" +
	"\vAttemptKind\x12\x1c\n" +
	"\x18ATTEMPT_KIND_UNSPECIFIED\x10\x00\x12\x18\n" +
	"\x14ATTEMPT_KIND_SERVING\x10\x01\x12\x14\n" +
	"\x10ATTEMPT_KIND_JOB\x10\x02*\xce\x01\n" +
	"\fAttemptState\x12\x1d\n" +
	"\x19ATTEMPT_STATE_UNSPECIFIED\x10\x00\x12\x19\n" +
	"\x15ATTEMPT_STATE_RUNNING\x10\x01\x12%\n" +
	"!ATTEMPT_STATE_OUTCOME_PENDING_ACK\x10\x02\x12 \n" +
	"\x1cATTEMPT_STATE_HELD_UNDURABLE\x10\x03\x12\x18\n" +
	"\x14ATTEMPT_STATE_QUEUED\x10\x04\x12!\n" +
	"\x1dATTEMPT_STATE_DEVICE_RELEASED\x10\x05*\xbf\x01\n" +
	"\rOutcomeStatus\x12\x1e\n" +
	"\x1aOUTCOME_STATUS_UNSPECIFIED\x10\x00\x12\x1c\n" +
	"\x18OUTCOME_STATUS_SUCCEEDED\x10\x01\x12\x1a\n" +
	"\x16OUTCOME_STATUS_REFUSED\x10\x02\x12\x19\n" +
	"\x15OUTCOME_STATUS_FAILED\x10\x03\x12\x1b\n" +
	"\x17OUTCOME_STATUS_CANCELED\x10\x04\x12\x1c\n" +
	"\x18OUTCOME_STATUS_ABANDONED\x10\x05*\xbf\x05\n" +
	"\tCauseCode\x12\x1a\n" +
	"\x16CAUSE_CODE_UNSPECIFIED\x10\x00\x12\x1e\n" +
	"\x1aCAUSE_CODE_INVALID_REQUEST\x10\x01\x12 \n" +
	"\x1cCAUSE_CODE_UNSUPPORTED_INPUT\x10\x02\x12\x1b\n" +
	"\x17CAUSE_CODE_LOCAL_SAFETY\x10\x03\x12\x17\n" +
	"\x13CAUSE_CODE_PROTOCOL\x10\x04\x12$\n" +
	" CAUSE_CODE_CONSTRAINT_INFEASIBLE\x10\x05\x12\x1f\n" +
	"\x1bCAUSE_CODE_AUTHOR_EXCEPTION\x10\x06\x12\x1d\n" +
	"\x19CAUSE_CODE_EXECUTOR_FAULT\x10\a\x12\x1c\n" +
	"\x18CAUSE_CODE_GRANT_EXPIRED\x10\b\x12#\n" +
	"\x1fCAUSE_CODE_ARTIFACT_UNFETCHABLE\x10\t\x12%\n" +
	"!CAUSE_CODE_CAPABILITY_UNAVAILABLE\x10\n" +
	"\x12\x1c\n" +
	"\x18CAUSE_CODE_CLIENT_CANCEL\x10\v\x12\x1f\n" +
	"\x1bCAUSE_CODE_DEADLINE_EXPIRED\x10\f\x12\x1b\n" +
	"\x17CAUSE_CODE_DRAIN_CANCEL\x10\r\x12\x1c\n" +
	"\x18CAUSE_CODE_POLICY_CANCEL\x10\x0e\x12 \n" +
	"\x1cCAUSE_CODE_SUPERSEDED_CANCEL\x10\x0f\x12#\n" +
	"\x1fCAUSE_CODE_EXECUTOR_INVALIDATED\x10\x10\x12\x1a\n" +
	"\x16CAUSE_CODE_NO_CAPACITY\x10\x11\x12$\n" +
	" CAUSE_CODE_ADMISSION_EPOCH_STALE\x10\x12\x12 \n" +
	"\x1cCAUSE_CODE_UNKNOWN_PLACEMENT\x10\x13\x12)\n" +
	"%CAUSE_CODE_PLACEMENT_NOT_DISPATCHABLE\x10\x14*\xe2\x01\n" +
	"\vCauseOrigin\x12\x1c\n" +
	"\x18CAUSE_ORIGIN_UNSPECIFIED\x10\x00\x12\x17\n" +
	"\x13CAUSE_ORIGIN_AUTHOR\x10\x01\x12\x18\n" +
	"\x14CAUSE_ORIGIN_RUNTIME\x10\x02\x12\x19\n" +
	"\x15CAUSE_ORIGIN_EXECUTOR\x10\x03\x12\x17\n" +
	"\x13CAUSE_ORIGIN_WORKER\x10\x04\x12\x16\n" +
	"\x12CAUSE_ORIGIN_INFRA\x10\x05\x12\x17\n" +
	"\x13CAUSE_ORIGIN_CLIENT\x10\x06\x12\x1d\n" +
	"\x19CAUSE_ORIGIN_RECORD_OWNER\x10\a*\xb4\x01\n" +
	"\fCancelReason\x12\x1d\n" +
	"\x19CANCEL_REASON_UNSPECIFIED\x10\x00\x12\x18\n" +
	"\x14CANCEL_REASON_CLIENT\x10\x01\x12\x17\n" +
	"\x13CANCEL_REASON_DRAIN\x10\x02\x12\x1c\n" +
	"\x18CANCEL_REASON_SUPERSEDED\x10\x03\x12\x18\n" +
	"\x14CANCEL_REASON_POLICY\x10\x04\x12\x1a\n" +
	"\x16CANCEL_REASON_DEADLINE\x10\x05*\x98\x03\n" +
	"\x0eClaimRejection\x12\x1f\n" +
	"\x1bCLAIM_REJECTION_UNSPECIFIED\x10\x00\x12#\n" +
	"\x1fCLAIM_REJECTION_UNAUTHENTICATED\x10\x01\x12,\n" +
	"(CLAIM_REJECTION_STALE_RECORD_OWNER_EPOCH\x10\x02\x12\x1e\n" +
	"\x1aCLAIM_REJECTION_EPOCH_HELD\x10\x03\x12&\n" +
	"\"CLAIM_REJECTION_WORKER_ID_MISMATCH\x10\x04\x12'\n" +
	"#CLAIM_REJECTION_RELEASE_ID_MISMATCH\x10\x05\x12\x1d\n" +
	"\x19CLAIM_REJECTION_UNDURABLE\x10\a\"\x04\b\x06\x10\x06\"\x04\b\b\x10\b\"\x04\b\t\x10\t*!CLAIM_REJECTION_UNSUPPORTED_MINOR*&CLAIM_REJECTION_SCHEMA_DIGEST_MISMATCH*%CLAIM_REJECTION_PROTOCOL_INCOMPATIBLE*\x80\x05\n" +
	"\tFaultKind\x12\x1a\n" +
	"\x16FAULT_KIND_UNSPECIFIED\x10\x00\x12\"\n" +
	"\x1eFAULT_KIND_BINDING_UNAVAILABLE\x10\x01\x12\x1f\n" +
	"\x1bFAULT_KIND_BINDING_DEGRADED\x10\x02\x12\"\n" +
	"\x1eFAULT_KIND_HARDWARE_UNSUITABLE\x10\x03\x12$\n" +
	" FAULT_KIND_ARTIFACT_FETCH_FAILED\x10\x04\x12#\n" +
	"\x1fFAULT_KIND_CREDENTIAL_UNAPPLIED\x10\x06\x12#\n" +
	"\x1fFAULT_KIND_LOCAL_SAFETY_REFUSAL\x10\a\x12 \n" +
	"\x1cFAULT_KIND_EXECUTOR_POISONED\x10\b\x12\x1d\n" +
	"\x19FAULT_KIND_CONFIG_REFUSED\x10\t\x12 \n" +
	"\x1cFAULT_KIND_UNKNOWN_PLACEMENT\x10\n" +
	"\x12(\n" +
	"$FAULT_KIND_PLACEMENT_SET_UNSUPPORTED\x10\v\x12,\n" +
	"(FAULT_KIND_PLACEMENT_SET_DIGEST_MISMATCH\x10\f\x12'\n" +
	"#FAULT_KIND_ARTIFACT_DIGEST_MISMATCH\x10\x0e\x12#\n" +
	"\x1fFAULT_KIND_FALLBACK_PIN_MISSING\x10\x0f\"\x04\b\x05\x10\x05\"\x04\b\r\x10\r\"\x04\b\x10\x10\x10*\x18FAULT_KIND_GRANT_EXPIRED*'FAULT_KIND_ENVIRONMENT_RECEIPT_MISMATCH* FAULT_KIND_AUTHORIZATION_EXPIRED*\xa1\x02\n" +
	"\x11BootFailureReason\x12#\n" +
	"\x1fBOOT_FAILURE_REASON_UNSPECIFIED\x10\x00\x12(\n" +
	"$BOOT_FAILURE_REASON_HARDWARE_VERDICT\x10\x01\x12&\n" +
	"\"BOOT_FAILURE_REASON_ARTIFACT_FAULT\x10\x02\x12$\n" +
	" BOOT_FAILURE_REASON_DRIVER_FAULT\x10\x03\x12\"\n" +
	"\x1eBOOT_FAILURE_REASON_DISK_SHAPE\x10\x04\x12&\n" +
	"\"BOOT_FAILURE_REASON_CONFIG_INVALID\x10\x05\x12#\n" +
	"\x1fBOOT_FAILURE_REASON_OTHER_FATAL\x10\x06*\x90\x02\n" +
	"\x13CheckpointFaultCode\x12%\n" +
	"!CHECKPOINT_FAULT_CODE_UNSPECIFIED\x10\x00\x12+\n" +
	"'CHECKPOINT_FAULT_CODE_IDENTITY_CONFLICT\x10\x01\x12)\n" +
	"%CHECKPOINT_FAULT_CODE_UNKNOWN_ATTEMPT\x10\x02\x12&\n" +
	"\"CHECKPOINT_FAULT_CODE_NOT_JOB_MODE\x10\x03\x12(\n" +
	"$CHECKPOINT_FAULT_CODE_STALE_SEQUENCE\x10\x04\x12(\n" +
	"$CHECKPOINT_FAULT_CODE_QUOTA_EXCEEDED\x10\x05*\xbc\x01\n" +
	"\fPrepareStage\x12\x1d\n" +
	"\x19PREPARE_STAGE_UNSPECIFIED\x10\x00\x12\x1a\n" +
	"\x16PREPARE_STAGE_RESOLVED\x10\x01\x12\x1d\n" +
	"\x19PREPARE_STAGE_DOWNLOADING\x10\x02\x12\x1b\n" +
	"\x17PREPARE_STAGE_PREPARING\x10\x03\x12\x1a\n" +
	"\x16PREPARE_STAGE_PREPARED\x10\x04\x12\x19\n" +
	"\x15PREPARE_STAGE_REFUSED\x10\x05*\x98\x01\n" +
	"\x10WeightsHostStage\x12\"\n" +
	"\x1eWEIGHTS_HOST_STAGE_UNSPECIFIED\x10\x00\x12\x1d\n" +
	"\x19WEIGHTS_HOST_STAGE_INTENT\x10\x01\x12\x1e\n" +
	"\x1aWEIGHTS_HOST_STAGE_RECEIPT\x10\x02\x12!\n" +
	"\x1dWEIGHTS_HOST_STAGE_CHECKPOINT\x10\x03*\xa2\x01\n" +
	"\x12WeightsHostOutcome\x12$\n" +
	" WEIGHTS_HOST_OUTCOME_UNSPECIFIED\x10\x00\x12!\n" +
	"\x1dWEIGHTS_HOST_OUTCOME_RECORDED\x10\x01\x12!\n" +
	"\x1dWEIGHTS_HOST_OUTCOME_REPLAYED\x10\x02\x12 \n" +
	"\x1cWEIGHTS_HOST_OUTCOME_REFUSED\x10\x03*\x8c\x02\n" +
	"\x12WeightsHostRefusal\x12$\n" +
	" WEIGHTS_HOST_REFUSAL_UNSPECIFIED\x10\x00\x12(\n" +
	"$WEIGHTS_HOST_REFUSAL_UNKNOWN_ATTEMPT\x10\x01\x12(\n" +
	"$WEIGHTS_HOST_REFUSAL_INTENT_CONFLICT\x10\x02\x12%\n" +
	"!WEIGHTS_HOST_REFUSAL_STALE_WRITER\x10\x03\x12)\n" +
	"%WEIGHTS_HOST_REFUSAL_RECEIPT_CONFLICT\x10\x04\x12*\n" +
	"&WEIGHTS_HOST_REFUSAL_INVENTORY_INVALID\x10\x05*\x91\x01\n" +
	"\x17WeightsTransactionState\x12)\n" +
	"%WEIGHTS_TRANSACTION_STATE_UNSPECIFIED\x10\x00\x12$\n" +
	" WEIGHTS_TRANSACTION_STATE_INTENT\x10\x01\x12%\n" +
	"!WEIGHTS_TRANSACTION_STATE_RECEIPT\x10\x02*\xb3\x01\n" +
	"\x14WeightsUploadOutcome\x12&\n" +
	"\"WEIGHTS_UPLOAD_OUTCOME_UNSPECIFIED\x10\x00\x12#\n" +
	"\x1fWEIGHTS_UPLOAD_OUTCOME_UPLOADED\x10\x01\x12*\n" +
	"&WEIGHTS_UPLOAD_OUTCOME_ALREADY_PRESENT\x10\x02\x12\"\n" +
	"\x1eWEIGHTS_UPLOAD_OUTCOME_REFUSED\x10\x03*\xf7\x02\n" +
	"\x14WeightsUploadRefusal\x12&\n" +
	"\"WEIGHTS_UPLOAD_REFUSAL_UNSPECIFIED\x10\x00\x12.\n" +
	"*WEIGHTS_UPLOAD_REFUSAL_UNKNOWN_TRANSACTION\x10\x01\x12'\n" +
	"#WEIGHTS_UPLOAD_REFUSAL_STALE_WRITER\x10\x02\x12)\n" +
	"%WEIGHTS_UPLOAD_REFUSAL_UNKNOWN_OBJECT\x10\x03\x12.\n" +
	"*WEIGHTS_UPLOAD_REFUSAL_SOURCE_REF_MISMATCH\x10\x04\x12-\n" +
	")WEIGHTS_UPLOAD_REFUSAL_SOURCE_UNAVAILABLE\x10\x05\x12(\n" +
	"$WEIGHTS_UPLOAD_REFUSAL_GRANT_REFUSED\x10\x06\x12*\n" +
	"&WEIGHTS_UPLOAD_REFUSAL_TRANSFER_FAILED\x10\a*\x9e\x02\n" +
	"\x14WeightsTransferState\x12&\n" +
	"\"WEIGHTS_TRANSFER_STATE_UNSPECIFIED\x10\x00\x12#\n" +
	"\x1fWEIGHTS_TRANSFER_STATE_ACCEPTED\x10\x01\x12$\n" +
	" WEIGHTS_TRANSFER_STATE_UPLOADING\x10\x02\x12#\n" +
	"\x1fWEIGHTS_TRANSFER_STATE_UPLOADED\x10\x03\x12*\n" +
	"&WEIGHTS_TRANSFER_STATE_ALREADY_PRESENT\x10\x04\x12\x1f\n" +
	"\x1bWEIGHTS_TRANSFER_STATE_HELD\x10\x05\x12!\n" +
	"\x1dWEIGHTS_TRANSFER_STATE_FAILED\x10\x06*\x87\x01\n" +
	"\x13ModelSourceProvider\x12%\n" +
	"!MODEL_SOURCE_PROVIDER_UNSPECIFIED\x10\x00\x12&\n" +
	"\"MODEL_SOURCE_PROVIDER_HUGGING_FACE\x10\x01\x12!\n" +
	"\x1dMODEL_SOURCE_PROVIDER_CIVITAI\x10\x02*\xff\x01\n" +
	"\x14ModelSourceFileState\x12'\n" +
	"#MODEL_SOURCE_FILE_STATE_UNSPECIFIED\x10\x00\x12$\n" +
	" MODEL_SOURCE_FILE_STATE_ACCEPTED\x10\x01\x12'\n" +
	"#MODEL_SOURCE_FILE_STATE_DOWNLOADING\x10\x02\x12$\n" +
	" MODEL_SOURCE_FILE_STATE_VERIFIED\x10\x03\x12\"\n" +
	"\x1eMODEL_SOURCE_FILE_STATE_FAILED\x10\x04\x12%\n" +
	"!MODEL_SOURCE_FILE_STATE_CONVERTED\x10\x05*\xf6\x01\n" +
	"\x19ModelSourcePrepareOutcome\x12,\n" +
	"(MODEL_SOURCE_PREPARE_OUTCOME_UNSPECIFIED\x10\x00\x12)\n" +
	"%MODEL_SOURCE_PREPARE_OUTCOME_PREPARED\x10\x01\x12)\n" +
	"%MODEL_SOURCE_PREPARE_OUTCOME_REPLAYED\x10\x02\x12(\n" +
	"$MODEL_SOURCE_PREPARE_OUTCOME_REFUSED\x10\x03\x12+\n" +
	"'MODEL_SOURCE_PREPARE_OUTCOME_INCOMPLETE\x10\x04*\xb6\x01\n" +
	"\x15LocalPackageFileState\x12(\n" +
	"$LOCAL_PACKAGE_FILE_STATE_UNSPECIFIED\x10\x00\x12&\n" +
	"\"LOCAL_PACKAGE_FILE_STATE_RECEIVING\x10\x01\x12%\n" +
	"!LOCAL_PACKAGE_FILE_STATE_VERIFIED\x10\x02\x12$\n" +
	" LOCAL_PACKAGE_FILE_STATE_REFUSED\x10\x03*\xd2\x01\n" +
	"\x1aWeightsFinalizeDisposition\x12,\n" +
	"(WEIGHTS_FINALIZE_DISPOSITION_UNSPECIFIED\x10\x00\x12&\n" +
	"\"WEIGHTS_FINALIZE_DISPOSITION_ADOPT\x10\x01\x12(\n" +
	"$WEIGHTS_FINALIZE_DISPOSITION_ABANDON\x10\x02\x124\n" +
	"0WEIGHTS_FINALIZE_DISPOSITION_ABANDON_UNCOMMITTED\x10\x03*\x90\x01\n" +
	"\x16WeightsFinalizeOutcome\x12(\n" +
	"$WEIGHTS_FINALIZE_OUTCOME_UNSPECIFIED\x10\x00\x12$\n" +
	" WEIGHTS_FINALIZE_OUTCOME_ADOPTED\x10\x01\x12&\n" +
	"\"WEIGHTS_FINALIZE_OUTCOME_ABANDONED\x10\x022\xfc\v\n" +
	"\rWorkerControl\x12y\n" +
	"\x1cGetMachineExecutionWorkspace\x12..cozy.worker.v1.MachineExecutionWorkspaceQuery\x1a).cozy.worker.v1.MachineExecutionWorkspace\x12i\n" +
	"\x16SubmitMachineExecution\x12&.cozy.worker.v1.MachineExecutionSubmit\x1a'.cozy.worker.v1.MachineExecutionReceipt\x12j\n" +
	"\x16CloseMachineSubmission\x12&.cozy.worker.v1.MachineSubmissionClose\x1a(.cozy.worker.v1.MachineSubmissionClosure\x12c\n" +
	"\x13GetMachineExecution\x12%.cozy.worker.v1.MachineExecutionQuery\x1a%.cozy.worker.v1.MachineExecutionState\x12t\n" +
	"\x1aListMachineExecutionEvents\x12+.cozy.worker.v1.MachineExecutionEventsQuery\x1a).cozy.worker.v1.MachineExecutionEventPage\x12i\n" +
	"\x17ControlMachineExecution\x12'.cozy.worker.v1.MachineExecutionControl\x1a%.cozy.worker.v1.MachineExecutionState\x12b\n" +
	"\x17CollectMachineExecution\x12'.cozy.worker.v1.MachineExecutionCollect\x1a\x1e.cozy.worker.v1.AttemptOutcome\x12}\n" +
	"%AcknowledgeMachineExecutionCollection\x12-.cozy.worker.v1.MachineExecutionCollectionAck\x1a%.cozy.worker.v1.MachineExecutionState\x12q\n" +
	"\x1aReadMachineExecutionTriage\x12+.cozy.worker.v1.MachineExecutionTriageQuery\x1a&.cozy.worker.v1.MachineExecutionTriage\x12h\n" +
	"\x15ListMachineExecutions\x12).cozy.worker.v1.MachineExecutionListQuery\x1a$.cozy.worker.v1.MachineExecutionList\x12M\n" +
	"\fListPackages\x12 .cozy.worker.v1.PackageListQuery\x1a\x1b.cozy.worker.v1.PackageList\x12G\n" +
	"\n" +
	"ListModels\x12\x1e.cozy.worker.v1.ModelListQuery\x1a\x19.cozy.worker.v1.ModelList\x12[\n" +
	"\x0fDescribeMachine\x12$.cozy.worker.v1.DescribeMachineQuery\x1a\".cozy.worker.v1.MachineDescription\x12L\n" +
	"\aControl\x12 .cozy.worker.v1.RecordOwnerFrame\x1a\x1b.cozy.worker.v1.WorkerFrame(\x010\x01\x12P\n" +
	"\rWatchProgress\x12\x1c.cozy.worker.v1.ProgressOpen\x1a\x1f.cozy.worker.v1.AttemptProgress0\x012\x99\x16\n" +
	"\x12RuntimePreparation\x12W\n" +
	"\fProtocolInfo\x12#.cozy.worker.v1.ProtocolInfoRequest\x1a\".cozy.worker.v1.ProtocolInfoResult\x12o\n" +
	"\x14NumericalEnvironment\x12+.cozy.worker.v1.NumericalEnvironmentRequest\x1a*.cozy.worker.v1.NumericalEnvironmentResult\x12o\n" +
	"\x15RecordOperationResult\x12).cozy.worker.v1.RecordOperationResultCall\x1a+.cozy.worker.v1.RecordOperationResultResult\x12]\n" +
	"\x0fLookupOperation\x12#.cozy.worker.v1.LookupOperationCall\x1a%.cozy.worker.v1.LookupOperationResult\x12i\n" +
	"\x13PruneOperationCache\x12'.cozy.worker.v1.PruneOperationCacheCall\x1a).cozy.worker.v1.PruneOperationCacheResult\x12l\n" +
	"\x1cWorkspaceRetainDerivedResult\x12$.cozy.worker.v1.DerivedRetentionCall\x1a&.cozy.worker.v1.DerivedRetentionResult\x12p\n" +
	" WorkspaceReleaseDerivedRetention\x12$.cozy.worker.v1.DerivedRetentionCall\x1a&.cozy.worker.v1.DerivedRetentionResult\x12u\n" +
	"\x1dWorkspaceReleaseDerivedResult\x12(.cozy.worker.v1.DerivedResultReleaseCall\x1a*.cozy.worker.v1.DerivedResultReleaseResult\x12m\n" +
	"\x17WorkspaceRetainByteTree\x12'.cozy.worker.v1.NativeByteRetentionCall\x1a).cozy.worker.v1.NativeByteRetentionResult\x12n\n" +
	"\x18WorkspaceReleaseByteTree\x12'.cozy.worker.v1.NativeByteRetentionCall\x1a).cozy.worker.v1.NativeByteRetentionResult\x12h\n" +
	"\x1bWorkspaceReadByteTreeObject\x12\".cozy.worker.v1.NativeByteReadCall\x1a#.cozy.worker.v1.NativeByteReadChunk0\x01\x12{\n" +
	"\x1fWorkspaceNativeArtifactTransfer\x12*.cozy.worker.v1.NativeArtifactTransferCall\x1a,.cozy.worker.v1.NativeArtifactTransferStatus\x12`\n" +
	"\x16WorkspaceForgetPackage\x12!.cozy.worker.v1.ForgetPackageCall\x1a#.cozy.worker.v1.ForgetPackageResult\x12d\n" +
	"\x0fImportInputTree\x12$.cozy.worker.v1.InputTreeImportFrame\x1a).cozy.worker.v1.NativeByteRetentionResult(\x01\x12f\n" +
	"\x11PreparePackageSet\x12(.cozy.worker.v1.PreparePackageSetRequest\x1a'.cozy.worker.v1.PreparePackageSetResult\x12i\n" +
	"\x12PrepareModelSource\x12).cozy.worker.v1.PrepareModelSourceRequest\x1a(.cozy.worker.v1.PrepareModelSourceResult\x12i\n" +
	"\x12ReleaseModelSource\x12).cozy.worker.v1.ReleaseModelSourceRequest\x1a(.cozy.worker.v1.ReleaseModelSourceResult\x12f\n" +
	"\x13RetainDerivedResult\x12'.cozy.worker.v1.DerivedRetentionRequest\x1a&.cozy.worker.v1.DerivedRetentionResult\x12j\n" +
	"\x17ReleaseDerivedRetention\x12'.cozy.worker.v1.DerivedRetentionRequest\x1a&.cozy.worker.v1.DerivedRetentionResult\x12o\n" +
	"\x14ReleaseDerivedResult\x12+.cozy.worker.v1.DerivedResultReleaseRequest\x1a*.cozy.worker.v1.DerivedResultReleaseResult\x12l\n" +
	"\x13CollectStoreGarbage\x12*.cozy.worker.v1.CollectStoreGarbageRequest\x1a).cozy.worker.v1.CollectStoreGarbageResult\x12~\n" +
	"\x19ValidateWeightsCheckpoint\x120.cozy.worker.v1.ValidateWeightsCheckpointRequest\x1a/.cozy.worker.v1.ValidateWeightsCheckpointResult\x12]\n" +
	"\x0eCheckpointPage\x12%.cozy.worker.v1.CheckpointPageRequest\x1a$.cozy.worker.v1.CheckpointPageResult\x12i\n" +
	"\x12CheckpointTransfer\x12).cozy.worker.v1.CheckpointTransferRequest\x1a(.cozy.worker.v1.CheckpointTransferStatus\x12j\n" +
	"\x13PrepareLocalPackage\x12*.cozy.worker.v1.PrepareLocalPackageRequest\x1a'.cozy.worker.v1.PreparePackageSetResult\x12r\n" +
	"\x17PreparePrivatePlacement\x12..cozy.worker.v1.PreparePrivatePlacementRequest\x1a'.cozy.worker.v1.PreparePackageSetResult2e\n" +
	"\x0eRuntimeWeights\x12S\n" +
	"\x06Upload\x12$.cozy.worker.v1.WeightsUploadRequest\x1a#.cozy.worker.v1.WeightsUploadResult2\xf3\x1f\n" +
	"\aPodHost\x12`\n" +
	"\x0fKeepRentalAlive\x12&.cozy.worker.v1.KeepRentalAliveRequest\x1a%.cozy.worker.v1.KeepRentalAliveResult\x12y\n" +
	"\x1cGetMachineExecutionWorkspace\x12..cozy.worker.v1.MachineExecutionWorkspaceQuery\x1a).cozy.worker.v1.MachineExecutionWorkspace\x12i\n" +
	"\x16SubmitMachineExecution\x12&.cozy.worker.v1.MachineExecutionSubmit\x1a'.cozy.worker.v1.MachineExecutionReceipt\x12j\n" +
	"\x16CloseMachineSubmission\x12&.cozy.worker.v1.MachineSubmissionClose\x1a(.cozy.worker.v1.MachineSubmissionClosure\x12c\n" +
	"\x13GetMachineExecution\x12%.cozy.worker.v1.MachineExecutionQuery\x1a%.cozy.worker.v1.MachineExecutionState\x12t\n" +
	"\x1aListMachineExecutionEvents\x12+.cozy.worker.v1.MachineExecutionEventsQuery\x1a).cozy.worker.v1.MachineExecutionEventPage\x12i\n" +
	"\x17ControlMachineExecution\x12'.cozy.worker.v1.MachineExecutionControl\x1a%.cozy.worker.v1.MachineExecutionState\x12b\n" +
	"\x17CollectMachineExecution\x12'.cozy.worker.v1.MachineExecutionCollect\x1a\x1e.cozy.worker.v1.AttemptOutcome\x12}\n" +
	"%AcknowledgeMachineExecutionCollection\x12-.cozy.worker.v1.MachineExecutionCollectionAck\x1a%.cozy.worker.v1.MachineExecutionState\x12q\n" +
	"\x1aReadMachineExecutionTriage\x12+.cozy.worker.v1.MachineExecutionTriageQuery\x1a&.cozy.worker.v1.MachineExecutionTriage\x12W\n" +
	"\fProtocolInfo\x12#.cozy.worker.v1.ProtocolInfoRequest\x1a\".cozy.worker.v1.ProtocolInfoResult\x12l\n" +
	"\x14NumericalEnvironment\x12(.cozy.worker.v1.NumericalEnvironmentCall\x1a*.cozy.worker.v1.NumericalEnvironmentResult\x12Z\n" +
	"\x11PreparePackageSet\x12%.cozy.worker.v1.PreparePackageSetCall\x1a\x1c.cozy.worker.v1.PrepareEvent0\x01\x12^\n" +
	"\x13PrepareLocalPackage\x12'.cozy.worker.v1.PrepareLocalPackageCall\x1a\x1c.cozy.worker.v1.PrepareEvent0\x01\x12f\n" +
	"\x17PreparePrivatePlacement\x12+.cozy.worker.v1.PreparePrivatePlacementCall\x1a\x1c.cozy.worker.v1.PrepareEvent0\x01\x12_\n" +
	"\x0fModelSourceFile\x12#.cozy.worker.v1.ModelSourceFileCall\x1a%.cozy.worker.v1.ModelSourceFileStatus0\x01\x12a\n" +
	"\x12ModelSourcePrepare\x12&.cozy.worker.v1.ModelSourcePrepareCall\x1a#.cozy.worker.v1.ModelSourcePrepared\x12f\n" +
	"\x12ModelSourceRelease\x12&.cozy.worker.v1.ModelSourceReleaseCall\x1a(.cozy.worker.v1.ReleaseModelSourceResult\x12f\n" +
	"\x12ModelSourceControl\x12&.cozy.worker.v1.ModelSourceControlCall\x1a(.cozy.worker.v1.ModelSourceControlResult\x12c\n" +
	"\x13RetainDerivedResult\x12$.cozy.worker.v1.DerivedRetentionCall\x1a&.cozy.worker.v1.DerivedRetentionResult\x12g\n" +
	"\x17ReleaseDerivedRetention\x12$.cozy.worker.v1.DerivedRetentionCall\x1a&.cozy.worker.v1.DerivedRetentionResult\x12l\n" +
	"\x14ReleaseDerivedResult\x12(.cozy.worker.v1.DerivedResultReleaseCall\x1a*.cozy.worker.v1.DerivedResultReleaseResult\x12d\n" +
	"\x0eRetainByteTree\x12'.cozy.worker.v1.NativeByteRetentionCall\x1a).cozy.worker.v1.NativeByteRetentionResult\x12e\n" +
	"\x0fReleaseByteTree\x12'.cozy.worker.v1.NativeByteRetentionCall\x1a).cozy.worker.v1.NativeByteRetentionResult\x12_\n" +
	"\x12ReadByteTreeObject\x12\".cozy.worker.v1.NativeByteReadCall\x1a#.cozy.worker.v1.NativeByteReadChunk0\x01\x12r\n" +
	"\x16NativeArtifactTransfer\x12*.cozy.worker.v1.NativeArtifactTransferCall\x1a,.cozy.worker.v1.NativeArtifactTransferStatus\x12W\n" +
	"\rForgetPackage\x12!.cozy.worker.v1.ForgetPackageCall\x1a#.cozy.worker.v1.ForgetPackageResult\x12d\n" +
	"\x0fImportInputTree\x12$.cozy.worker.v1.InputTreeImportFrame\x1a).cozy.worker.v1.NativeByteRetentionResult(\x01\x12o\n" +
	"\x15RecordOperationResult\x12).cozy.worker.v1.RecordOperationResultCall\x1a+.cozy.worker.v1.RecordOperationResultResult\x12]\n" +
	"\x0fLookupOperation\x12#.cozy.worker.v1.LookupOperationCall\x1a%.cozy.worker.v1.LookupOperationResult\x12i\n" +
	"\x13PruneOperationCache\x12'.cozy.worker.v1.PruneOperationCacheCall\x1a).cozy.worker.v1.PruneOperationCacheResult\x12]\n" +
	"\x10ModelSourceAdopt\x12$.cozy.worker.v1.ModelSourceAdoptCall\x1a#.cozy.worker.v1.ModelSourcePrepared\x12Z\n" +
	"\x0eCheckpointPage\x12\".cozy.worker.v1.CheckpointPageCall\x1a$.cozy.worker.v1.CheckpointPageResult\x12f\n" +
	"\x12CheckpointTransfer\x12&.cozy.worker.v1.CheckpointTransferCall\x1a(.cozy.worker.v1.CheckpointTransferStatus\x12i\n" +
	"\x12LocalPackageUpload\x12'.cozy.worker.v1.LocalPackageUploadFrame\x1a&.cozy.worker.v1.LocalPackageFileStatus(\x010\x01\x12h\n" +
	"\x15ListMachineExecutions\x12).cozy.worker.v1.MachineExecutionListQuery\x1a$.cozy.worker.v1.MachineExecutionList\x12M\n" +
	"\fListPackages\x12 .cozy.worker.v1.PackageListQuery\x1a\x1b.cozy.worker.v1.PackageList\x12G\n" +
	"\n" +
	"ListModels\x12\x1e.cozy.worker.v1.ModelListQuery\x1a\x19.cozy.worker.v1.ModelList\x12[\n" +
	"\x0fDescribeMachine\x12$.cozy.worker.v1.DescribeMachineQuery\x1a\".cozy.worker.v1.MachineDescription\x12T\n" +
	"\x0eReadMachineLog\x12\x1f.cozy.worker.v1.MachineLogQuery\x1a\x1f.cozy.worker.v1.MachineLogChunk0\x01B4\n" +
	"\x0ecozy.worker.v1P\x01Z cozy/workerprotov1;workerprotov1b\x06proto3"

var (
	file_cozy_worker_v1_worker_proto_rawDescOnce sync.Once
	file_cozy_worker_v1_worker_proto_rawDescData []byte
)

func file_cozy_worker_v1_worker_proto_rawDescGZIP() []byte {
	file_cozy_worker_v1_worker_proto_rawDescOnce.Do(func() {
		file_cozy_worker_v1_worker_proto_rawDescData = protoimpl.X.CompressGZIP(unsafe.Slice(unsafe.StringData(file_cozy_worker_v1_worker_proto_rawDesc), len(file_cozy_worker_v1_worker_proto_rawDesc)))
	})
	return file_cozy_worker_v1_worker_proto_rawDescData
}

var file_cozy_worker_v1_worker_proto_enumTypes = make([]protoimpl.EnumInfo, 36)
var file_cozy_worker_v1_worker_proto_msgTypes = make([]protoimpl.MessageInfo, 239)
var file_cozy_worker_v1_worker_proto_goTypes = []any{
	(MachineLog)(0),                          // 0: cozy.worker.v1.MachineLog
	(RunProductOp)(0),                        // 1: cozy.worker.v1.RunProductOp
	(MachineExecutionAction)(0),              // 2: cozy.worker.v1.MachineExecutionAction
	(ChildCallState)(0),                      // 3: cozy.worker.v1.ChildCallState
	(NativeSourcePhase)(0),                   // 4: cozy.worker.v1.NativeSourcePhase
	(NativeSourceOperation)(0),               // 5: cozy.worker.v1.NativeSourceOperation
	(NativeSourceState)(0),                   // 6: cozy.worker.v1.NativeSourceState
	(Posture)(0),                             // 7: cozy.worker.v1.Posture
	(WorkerPhase)(0),                         // 8: cozy.worker.v1.WorkerPhase
	(MaterializationState)(0),                // 9: cozy.worker.v1.MaterializationState
	(ServingState)(0),                        // 10: cozy.worker.v1.ServingState
	(AdmissionState)(0),                      // 11: cozy.worker.v1.AdmissionState
	(AttemptKind)(0),                         // 12: cozy.worker.v1.AttemptKind
	(AttemptState)(0),                        // 13: cozy.worker.v1.AttemptState
	(OutcomeStatus)(0),                       // 14: cozy.worker.v1.OutcomeStatus
	(CauseCode)(0),                           // 15: cozy.worker.v1.CauseCode
	(CauseOrigin)(0),                         // 16: cozy.worker.v1.CauseOrigin
	(CancelReason)(0),                        // 17: cozy.worker.v1.CancelReason
	(ClaimRejection)(0),                      // 18: cozy.worker.v1.ClaimRejection
	(FaultKind)(0),                           // 19: cozy.worker.v1.FaultKind
	(BootFailureReason)(0),                   // 20: cozy.worker.v1.BootFailureReason
	(CheckpointFaultCode)(0),                 // 21: cozy.worker.v1.CheckpointFaultCode
	(PrepareStage)(0),                        // 22: cozy.worker.v1.PrepareStage
	(WeightsHostStage)(0),                    // 23: cozy.worker.v1.WeightsHostStage
	(WeightsHostOutcome)(0),                  // 24: cozy.worker.v1.WeightsHostOutcome
	(WeightsHostRefusal)(0),                  // 25: cozy.worker.v1.WeightsHostRefusal
	(WeightsTransactionState)(0),             // 26: cozy.worker.v1.WeightsTransactionState
	(WeightsUploadOutcome)(0),                // 27: cozy.worker.v1.WeightsUploadOutcome
	(WeightsUploadRefusal)(0),                // 28: cozy.worker.v1.WeightsUploadRefusal
	(WeightsTransferState)(0),                // 29: cozy.worker.v1.WeightsTransferState
	(ModelSourceProvider)(0),                 // 30: cozy.worker.v1.ModelSourceProvider
	(ModelSourceFileState)(0),                // 31: cozy.worker.v1.ModelSourceFileState
	(ModelSourcePrepareOutcome)(0),           // 32: cozy.worker.v1.ModelSourcePrepareOutcome
	(LocalPackageFileState)(0),               // 33: cozy.worker.v1.LocalPackageFileState
	(WeightsFinalizeDisposition)(0),          // 34: cozy.worker.v1.WeightsFinalizeDisposition
	(WeightsFinalizeOutcome)(0),              // 35: cozy.worker.v1.WeightsFinalizeOutcome
	(*MachineLogQuery)(nil),                  // 36: cozy.worker.v1.MachineLogQuery
	(*MachineLogChunk)(nil),                  // 37: cozy.worker.v1.MachineLogChunk
	(*ProtocolInfoRequest)(nil),              // 38: cozy.worker.v1.ProtocolInfoRequest
	(*KeepRentalAliveRequest)(nil),           // 39: cozy.worker.v1.KeepRentalAliveRequest
	(*KeepRentalAliveResult)(nil),            // 40: cozy.worker.v1.KeepRentalAliveResult
	(*MachineExecutionCapture)(nil),          // 41: cozy.worker.v1.MachineExecutionCapture
	(*DeferredInstallation)(nil),             // 42: cozy.worker.v1.DeferredInstallation
	(*MachineModelDefault)(nil),              // 43: cozy.worker.v1.MachineModelDefault
	(*MachineModelDefaultRung)(nil),          // 44: cozy.worker.v1.MachineModelDefaultRung
	(*InstalledPackage)(nil),                 // 45: cozy.worker.v1.InstalledPackage
	(*MachineCallableBinding)(nil),           // 46: cozy.worker.v1.MachineCallableBinding
	(*MachineExecutionWorkspaceQuery)(nil),   // 47: cozy.worker.v1.MachineExecutionWorkspaceQuery
	(*MachineExecutionWorkspace)(nil),        // 48: cozy.worker.v1.MachineExecutionWorkspace
	(*DescribedRelease)(nil),                 // 49: cozy.worker.v1.DescribedRelease
	(*MachineDevice)(nil),                    // 50: cozy.worker.v1.MachineDevice
	(*MachineSubmissionClose)(nil),           // 51: cozy.worker.v1.MachineSubmissionClose
	(*MachineSubmissionClosure)(nil),         // 52: cozy.worker.v1.MachineSubmissionClosure
	(*MachineExecutionSubmit)(nil),           // 53: cozy.worker.v1.MachineExecutionSubmit
	(*ReleaseRoot)(nil),                      // 54: cozy.worker.v1.ReleaseRoot
	(*ModelChoice)(nil),                      // 55: cozy.worker.v1.ModelChoice
	(*SourceCredential)(nil),                 // 56: cozy.worker.v1.SourceCredential
	(*MachineExecutionReceipt)(nil),          // 57: cozy.worker.v1.MachineExecutionReceipt
	(*MachineExecutionQuery)(nil),            // 58: cozy.worker.v1.MachineExecutionQuery
	(*MachineExecutionState)(nil),            // 59: cozy.worker.v1.MachineExecutionState
	(*MachineExecutionTarget)(nil),           // 60: cozy.worker.v1.MachineExecutionTarget
	(*MachineExecutionGpu)(nil),              // 61: cozy.worker.v1.MachineExecutionGpu
	(*MachineExecutionEventsQuery)(nil),      // 62: cozy.worker.v1.MachineExecutionEventsQuery
	(*MachineExecutionEvent)(nil),            // 63: cozy.worker.v1.MachineExecutionEvent
	(*RunProduct)(nil),                       // 64: cozy.worker.v1.RunProduct
	(*RunProductPart)(nil),                   // 65: cozy.worker.v1.RunProductPart
	(*MachineExecutionEventPage)(nil),        // 66: cozy.worker.v1.MachineExecutionEventPage
	(*MachineExecutionControl)(nil),          // 67: cozy.worker.v1.MachineExecutionControl
	(*MachineMemoAnswer)(nil),                // 68: cozy.worker.v1.MachineMemoAnswer
	(*MachinePublicationReconciliation)(nil), // 69: cozy.worker.v1.MachinePublicationReconciliation
	(*MachineExecutionCollect)(nil),          // 70: cozy.worker.v1.MachineExecutionCollect
	(*MachineExecutionCollectionAck)(nil),    // 71: cozy.worker.v1.MachineExecutionCollectionAck
	(*MachineExecutionTriageQuery)(nil),      // 72: cozy.worker.v1.MachineExecutionTriageQuery
	(*MachineExecutionTriage)(nil),           // 73: cozy.worker.v1.MachineExecutionTriage
	(*MachineExecutionListQuery)(nil),        // 74: cozy.worker.v1.MachineExecutionListQuery
	(*MachineExecutionList)(nil),             // 75: cozy.worker.v1.MachineExecutionList
	(*PackageListQuery)(nil),                 // 76: cozy.worker.v1.PackageListQuery
	(*PackageList)(nil),                      // 77: cozy.worker.v1.PackageList
	(*MachinePackage)(nil),                   // 78: cozy.worker.v1.MachinePackage
	(*ModelListQuery)(nil),                   // 79: cozy.worker.v1.ModelListQuery
	(*ModelList)(nil),                        // 80: cozy.worker.v1.ModelList
	(*MachineModel)(nil),                     // 81: cozy.worker.v1.MachineModel
	(*RepositoryUsage)(nil),                  // 82: cozy.worker.v1.RepositoryUsage
	(*StoreUsage)(nil),                       // 83: cozy.worker.v1.StoreUsage
	(*MachineFilesystem)(nil),                // 84: cozy.worker.v1.MachineFilesystem
	(*DescribeMachineQuery)(nil),             // 85: cozy.worker.v1.DescribeMachineQuery
	(*MachineDescription)(nil),               // 86: cozy.worker.v1.MachineDescription
	(*MachineHost)(nil),                      // 87: cozy.worker.v1.MachineHost
	(*MachineHub)(nil),                       // 88: cozy.worker.v1.MachineHub
	(*MachineRuntime)(nil),                   // 89: cozy.worker.v1.MachineRuntime
	(*ProtocolInfoResult)(nil),               // 90: cozy.worker.v1.ProtocolInfoResult
	(*PreparePackageSetCall)(nil),            // 91: cozy.worker.v1.PreparePackageSetCall
	(*PrepareLocalPackageCall)(nil),          // 92: cozy.worker.v1.PrepareLocalPackageCall
	(*PreparePrivatePlacementCall)(nil),      // 93: cozy.worker.v1.PreparePrivatePlacementCall
	(*PrepareEvent)(nil),                     // 94: cozy.worker.v1.PrepareEvent
	(*PrepareModelProgress)(nil),             // 95: cozy.worker.v1.PrepareModelProgress
	(*ModelSourceFileCall)(nil),              // 96: cozy.worker.v1.ModelSourceFileCall
	(*ModelSourcePrepareCall)(nil),           // 97: cozy.worker.v1.ModelSourcePrepareCall
	(*ModelSourceReleaseCall)(nil),           // 98: cozy.worker.v1.ModelSourceReleaseCall
	(*ReleaseModelSourceRequest)(nil),        // 99: cozy.worker.v1.ReleaseModelSourceRequest
	(*NumericalEnvironmentRequest)(nil),      // 100: cozy.worker.v1.NumericalEnvironmentRequest
	(*ModelSourceControlCall)(nil),           // 101: cozy.worker.v1.ModelSourceControlCall
	(*ModelSourceControlResult)(nil),         // 102: cozy.worker.v1.ModelSourceControlResult
	(*NumericalEnvironmentCall)(nil),         // 103: cozy.worker.v1.NumericalEnvironmentCall
	(*NumericalEnvironmentResult)(nil),       // 104: cozy.worker.v1.NumericalEnvironmentResult
	(*DerivedRetentionCall)(nil),             // 105: cozy.worker.v1.DerivedRetentionCall
	(*DerivedRetentionRequest)(nil),          // 106: cozy.worker.v1.DerivedRetentionRequest
	(*DerivedRetentionResult)(nil),           // 107: cozy.worker.v1.DerivedRetentionResult
	(*DerivedResultReleaseCall)(nil),         // 108: cozy.worker.v1.DerivedResultReleaseCall
	(*DerivedResultReleaseRequest)(nil),      // 109: cozy.worker.v1.DerivedResultReleaseRequest
	(*DerivedResultReleaseResult)(nil),       // 110: cozy.worker.v1.DerivedResultReleaseResult
	(*RecordOperationResultCall)(nil),        // 111: cozy.worker.v1.RecordOperationResultCall
	(*RecordOperationResultResult)(nil),      // 112: cozy.worker.v1.RecordOperationResultResult
	(*LookupOperationCall)(nil),              // 113: cozy.worker.v1.LookupOperationCall
	(*LookupOperationResult)(nil),            // 114: cozy.worker.v1.LookupOperationResult
	(*PruneOperationCacheCall)(nil),          // 115: cozy.worker.v1.PruneOperationCacheCall
	(*PruneOperationCacheResult)(nil),        // 116: cozy.worker.v1.PruneOperationCacheResult
	(*CollectStoreGarbageRequest)(nil),       // 117: cozy.worker.v1.CollectStoreGarbageRequest
	(*CollectStoreGarbageResult)(nil),        // 118: cozy.worker.v1.CollectStoreGarbageResult
	(*ReleaseModelSourceResult)(nil),         // 119: cozy.worker.v1.ReleaseModelSourceResult
	(*ModelSourceAdoptCall)(nil),             // 120: cozy.worker.v1.ModelSourceAdoptCall
	(*CheckpointPageCall)(nil),               // 121: cozy.worker.v1.CheckpointPageCall
	(*CheckpointTransferCall)(nil),           // 122: cozy.worker.v1.CheckpointTransferCall
	(*LocalPackageUploadFrame)(nil),          // 123: cozy.worker.v1.LocalPackageUploadFrame
	(*LocalPackageUploadHeader)(nil),         // 124: cozy.worker.v1.LocalPackageUploadHeader
	(*LocalPackageUploadChunk)(nil),          // 125: cozy.worker.v1.LocalPackageUploadChunk
	(*PreparePackageSetRequest)(nil),         // 126: cozy.worker.v1.PreparePackageSetRequest
	(*ImageInventory)(nil),                   // 127: cozy.worker.v1.ImageInventory
	(*PythonInterpreter)(nil),                // 128: cozy.worker.v1.PythonInterpreter
	(*ImageDistribution)(nil),                // 129: cozy.worker.v1.ImageDistribution
	(*PreparePackageSetResult)(nil),          // 130: cozy.worker.v1.PreparePackageSetResult
	(*PrepareLocalPackageRequest)(nil),       // 131: cozy.worker.v1.PrepareLocalPackageRequest
	(*LocalPackageFile)(nil),                 // 132: cozy.worker.v1.LocalPackageFile
	(*PreparePrivatePlacementRequest)(nil),   // 133: cozy.worker.v1.PreparePrivatePlacementRequest
	(*LocalModelSourceFile)(nil),             // 134: cozy.worker.v1.LocalModelSourceFile
	(*ModelSourceProfile)(nil),               // 135: cozy.worker.v1.ModelSourceProfile
	(*PreparedModelSource)(nil),              // 136: cozy.worker.v1.PreparedModelSource
	(*ModelSourceCheckpoint)(nil),            // 137: cozy.worker.v1.ModelSourceCheckpoint
	(*SourceCheckpointSubject)(nil),          // 138: cozy.worker.v1.SourceCheckpointSubject
	(*WeightsCheckpointSubject)(nil),         // 139: cozy.worker.v1.WeightsCheckpointSubject
	(*CheckpointSubject)(nil),                // 140: cozy.worker.v1.CheckpointSubject
	(*CheckpointObject)(nil),                 // 141: cozy.worker.v1.CheckpointObject
	(*CheckpointPageRequest)(nil),            // 142: cozy.worker.v1.CheckpointPageRequest
	(*CheckpointPageResult)(nil),             // 143: cozy.worker.v1.CheckpointPageResult
	(*CheckpointTransferRequest)(nil),        // 144: cozy.worker.v1.CheckpointTransferRequest
	(*CheckpointTransferStatus)(nil),         // 145: cozy.worker.v1.CheckpointTransferStatus
	(*PrepareModelSourceRequest)(nil),        // 146: cozy.worker.v1.PrepareModelSourceRequest
	(*PrepareModelSourceResult)(nil),         // 147: cozy.worker.v1.PrepareModelSourceResult
	(*RecordOwnerFrame)(nil),                 // 148: cozy.worker.v1.RecordOwnerFrame
	(*WorkerFrame)(nil),                      // 149: cozy.worker.v1.WorkerFrame
	(*ChildCallRequest)(nil),                 // 150: cozy.worker.v1.ChildCallRequest
	(*ChildCallResult)(nil),                  // 151: cozy.worker.v1.ChildCallResult
	(*NativeSourceMember)(nil),               // 152: cozy.worker.v1.NativeSourceMember
	(*NativeSourceSelection)(nil),            // 153: cozy.worker.v1.NativeSourceSelection
	(*NativeSourceCommand)(nil),              // 154: cozy.worker.v1.NativeSourceCommand
	(*NativeSourceStatus)(nil),               // 155: cozy.worker.v1.NativeSourceStatus
	(*Claim)(nil),                            // 156: cozy.worker.v1.Claim
	(*ClaimProof)(nil),                       // 157: cozy.worker.v1.ClaimProof
	(*ClaimAck)(nil),                         // 158: cozy.worker.v1.ClaimAck
	(*BootFailure)(nil),                      // 159: cozy.worker.v1.BootFailure
	(*WorkerSnapshot)(nil),                   // 160: cozy.worker.v1.WorkerSnapshot
	(*WorkerSnapshotBody)(nil),               // 161: cozy.worker.v1.WorkerSnapshotBody
	(*HostSnapshotBody)(nil),                 // 162: cozy.worker.v1.HostSnapshotBody
	(*SnapshotAck)(nil),                      // 163: cozy.worker.v1.SnapshotAck
	(*DesiredWorkerState)(nil),               // 164: cozy.worker.v1.DesiredWorkerState
	(*DesiredPackageSet)(nil),                // 165: cozy.worker.v1.DesiredPackageSet
	(*DesiredLocalPackageSet)(nil),           // 166: cozy.worker.v1.DesiredLocalPackageSet
	(*DesiredPrivatePlacementSet)(nil),       // 167: cozy.worker.v1.DesiredPrivatePlacementSet
	(*NativeModelBinding)(nil),               // 168: cozy.worker.v1.NativeModelBinding
	(*LocalPackageFileRef)(nil),              // 169: cozy.worker.v1.LocalPackageFileRef
	(*DesiredPlacementSet)(nil),              // 170: cozy.worker.v1.DesiredPlacementSet
	(*PlacementDevicePin)(nil),               // 171: cozy.worker.v1.PlacementDevicePin
	(*PlacementSet)(nil),                     // 172: cozy.worker.v1.PlacementSet
	(*DownloadDelegation)(nil),               // 173: cozy.worker.v1.DownloadDelegation
	(*DownloadModelRef)(nil),                 // 174: cozy.worker.v1.DownloadModelRef
	(*DownloadAdapterRef)(nil),               // 175: cozy.worker.v1.DownloadAdapterRef
	(*DownloadPackageRef)(nil),               // 176: cozy.worker.v1.DownloadPackageRef
	(*Placement)(nil),                        // 177: cozy.worker.v1.Placement
	(*Ref)(nil),                              // 178: cozy.worker.v1.Ref
	(*PackageSelection)(nil),                 // 179: cozy.worker.v1.PackageSelection
	(*DevelopmentPackage)(nil),               // 180: cozy.worker.v1.DevelopmentPackage
	(*Model)(nil),                            // 181: cozy.worker.v1.Model
	(*Entrypoint)(nil),                       // 182: cozy.worker.v1.Entrypoint
	(*Slot)(nil),                             // 183: cozy.worker.v1.Slot
	(*ModelAdapter)(nil),                     // 184: cozy.worker.v1.ModelAdapter
	(*Component)(nil),                        // 185: cozy.worker.v1.Component
	(*Stamp)(nil),                            // 186: cozy.worker.v1.Stamp
	(*JobDirective)(nil),                     // 187: cozy.worker.v1.JobDirective
	(*ObservedWorkerState)(nil),              // 188: cozy.worker.v1.ObservedWorkerState
	(*DeviceLane)(nil),                       // 189: cozy.worker.v1.DeviceLane
	(*PlacementStatus)(nil),                  // 190: cozy.worker.v1.PlacementStatus
	(*PlacementAcquisitionObservation)(nil),  // 191: cozy.worker.v1.PlacementAcquisitionObservation
	(*AcquisitionLegObservation)(nil),        // 192: cozy.worker.v1.AcquisitionLegObservation
	(*AcceleratorQualification)(nil),         // 193: cozy.worker.v1.AcceleratorQualification
	(*ActivityEvent)(nil),                    // 194: cozy.worker.v1.ActivityEvent
	(*AttemptOffer)(nil),                     // 195: cozy.worker.v1.AttemptOffer
	(*AttemptAccepted)(nil),                  // 196: cozy.worker.v1.AttemptAccepted
	(*CancelAttempt)(nil),                    // 197: cozy.worker.v1.CancelAttempt
	(*AttemptOutcome)(nil),                   // 198: cozy.worker.v1.AttemptOutcome
	(*AttemptOutcomeBody)(nil),               // 199: cozy.worker.v1.AttemptOutcomeBody
	(*WeightsReceiptRef)(nil),                // 200: cozy.worker.v1.WeightsReceiptRef
	(*WeightsReceipt)(nil),                   // 201: cozy.worker.v1.WeightsReceipt
	(*WeightsObjectSource)(nil),              // 202: cozy.worker.v1.WeightsObjectSource
	(*WeightsIntentFrame)(nil),               // 203: cozy.worker.v1.WeightsIntentFrame
	(*WeightsHostAck)(nil),                   // 204: cozy.worker.v1.WeightsHostAck
	(*CheckpointRef)(nil),                    // 205: cozy.worker.v1.CheckpointRef
	(*WeightsCheckpointFrame)(nil),           // 206: cozy.worker.v1.WeightsCheckpointFrame
	(*WeightsIntentReadyRequest)(nil),        // 207: cozy.worker.v1.WeightsIntentReadyRequest
	(*ValidateWeightsCheckpointRequest)(nil), // 208: cozy.worker.v1.ValidateWeightsCheckpointRequest
	(*ValidateWeightsCheckpointResult)(nil),  // 209: cozy.worker.v1.ValidateWeightsCheckpointResult
	(*WeightsTransactionRefused)(nil),        // 210: cozy.worker.v1.WeightsTransactionRefused
	(*WeightsReceiptFrame)(nil),              // 211: cozy.worker.v1.WeightsReceiptFrame
	(*WeightsTransactionStatus)(nil),         // 212: cozy.worker.v1.WeightsTransactionStatus
	(*WeightsObjectRef)(nil),                 // 213: cozy.worker.v1.WeightsObjectRef
	(*WeightsUploadHeader)(nil),              // 214: cozy.worker.v1.WeightsUploadHeader
	(*WeightsUploadGrant)(nil),               // 215: cozy.worker.v1.WeightsUploadGrant
	(*WeightsUploadRequest)(nil),             // 216: cozy.worker.v1.WeightsUploadRequest
	(*WeightsUploadResult)(nil),              // 217: cozy.worker.v1.WeightsUploadResult
	(*ModelSourceFileRequest)(nil),           // 218: cozy.worker.v1.ModelSourceFileRequest
	(*ModelSourceFileStatus)(nil),            // 219: cozy.worker.v1.ModelSourceFileStatus
	(*ModelSourcePrepareRequest)(nil),        // 220: cozy.worker.v1.ModelSourcePrepareRequest
	(*ModelSourcePrepared)(nil),              // 221: cozy.worker.v1.ModelSourcePrepared
	(*LocalPackageFileStatus)(nil),           // 222: cozy.worker.v1.LocalPackageFileStatus
	(*WeightsFinalizeRequest)(nil),           // 223: cozy.worker.v1.WeightsFinalizeRequest
	(*WeightsFinalizeResult)(nil),            // 224: cozy.worker.v1.WeightsFinalizeResult
	(*ResultEnvelope)(nil),                   // 225: cozy.worker.v1.ResultEnvelope
	(*RetainedModelResult)(nil),              // 226: cozy.worker.v1.RetainedModelResult
	(*AdjustmentRow)(nil),                    // 227: cozy.worker.v1.AdjustmentRow
	(*OutcomeCause)(nil),                     // 228: cozy.worker.v1.OutcomeCause
	(*ResourceShortfall)(nil),                // 229: cozy.worker.v1.ResourceShortfall
	(*AttemptOutcomeAck)(nil),                // 230: cozy.worker.v1.AttemptOutcomeAck
	(*JobCheckpointRequest)(nil),             // 231: cozy.worker.v1.JobCheckpointRequest
	(*ProgressOpen)(nil),                     // 232: cozy.worker.v1.ProgressOpen
	(*AttemptProgress)(nil),                  // 233: cozy.worker.v1.AttemptProgress
	(*InvocationSpec)(nil),                   // 234: cozy.worker.v1.InvocationSpec
	(*InputBinding)(nil),                     // 235: cozy.worker.v1.InputBinding
	(*OutputBinding)(nil),                    // 236: cozy.worker.v1.OutputBinding
	(*ServingInvocationSpec)(nil),            // 237: cozy.worker.v1.ServingInvocationSpec
	(*JobInvocationSpec)(nil),                // 238: cozy.worker.v1.JobInvocationSpec
	(*DeliveryGrant)(nil),                    // 239: cozy.worker.v1.DeliveryGrant
	(*CatalogModelSource)(nil),               // 240: cozy.worker.v1.CatalogModelSource
	(*InputAccess)(nil),                      // 241: cozy.worker.v1.InputAccess
	(*OutputAccess)(nil),                     // 242: cozy.worker.v1.OutputAccess
	(*DeliveryAccessCredential)(nil),         // 243: cozy.worker.v1.DeliveryAccessCredential
	(*ResourceCaps)(nil),                     // 244: cozy.worker.v1.ResourceCaps
	(*PublicationContract)(nil),              // 245: cozy.worker.v1.PublicationContract
	(*WorkerResources)(nil),                  // 246: cozy.worker.v1.WorkerResources
	(*JobCapacity)(nil),                      // 247: cozy.worker.v1.JobCapacity
	(*HeldAttempt)(nil),                      // 248: cozy.worker.v1.HeldAttempt
	(*Fault)(nil),                            // 249: cozy.worker.v1.Fault
	(*OutputManifest)(nil),                   // 250: cozy.worker.v1.OutputManifest
	(*OutputEntry)(nil),                      // 251: cozy.worker.v1.OutputEntry
	(*AttemptMetrics)(nil),                   // 252: cozy.worker.v1.AttemptMetrics
	(*TriageBundleRef)(nil),                  // 253: cozy.worker.v1.TriageBundleRef
	(*ForgetPackageCall)(nil),                // 254: cozy.worker.v1.ForgetPackageCall
	(*ForgetPackageResult)(nil),              // 255: cozy.worker.v1.ForgetPackageResult
	(*NativeArtifactTransferCall)(nil),       // 256: cozy.worker.v1.NativeArtifactTransferCall
	(*NativeArtifactTransfer)(nil),           // 257: cozy.worker.v1.NativeArtifactTransfer
	(*NativeArtifactTransferStatus)(nil),     // 258: cozy.worker.v1.NativeArtifactTransferStatus
	(*NativeByteTreeRef)(nil),                // 259: cozy.worker.v1.NativeByteTreeRef
	(*NativeByteRetentionRequest)(nil),       // 260: cozy.worker.v1.NativeByteRetentionRequest
	(*NativeByteRetentionCall)(nil),          // 261: cozy.worker.v1.NativeByteRetentionCall
	(*NativeByteRetentionResult)(nil),        // 262: cozy.worker.v1.NativeByteRetentionResult
	(*ChildByteResultGrant)(nil),             // 263: cozy.worker.v1.ChildByteResultGrant
	(*InputTreeImportHeader)(nil),            // 264: cozy.worker.v1.InputTreeImportHeader
	(*InputTreeImportBlob)(nil),              // 265: cozy.worker.v1.InputTreeImportBlob
	(*InputTreeImportCommit)(nil),            // 266: cozy.worker.v1.InputTreeImportCommit
	(*InputTreeImportFrame)(nil),             // 267: cozy.worker.v1.InputTreeImportFrame
	(*NativeByteReadCall)(nil),               // 268: cozy.worker.v1.NativeByteReadCall
	(*NativeByteReadChunk)(nil),              // 269: cozy.worker.v1.NativeByteReadChunk
	(*ActivationCapture)(nil),                // 270: cozy.worker.v1.ActivationCapture
	(*ExecutionEnvironment)(nil),             // 271: cozy.worker.v1.ExecutionEnvironment
	(*ActivationCaptureResult)(nil),          // 272: cozy.worker.v1.ActivationCaptureResult
	(*ExecutionObservation)(nil),             // 273: cozy.worker.v1.ExecutionObservation
	(*RuntimeRevision)(nil),                  // 274: cozy.worker.v1.RuntimeRevision
}
var file_cozy_worker_v1_worker_proto_depIdxs = []int32{
	156, // 0: cozy.worker.v1.MachineLogQuery.claim:type_name -> cozy.worker.v1.Claim
	0,   // 1: cozy.worker.v1.MachineLogQuery.log:type_name -> cozy.worker.v1.MachineLog
	156, // 2: cozy.worker.v1.KeepRentalAliveRequest.claim:type_name -> cozy.worker.v1.Claim
	46,  // 3: cozy.worker.v1.MachineExecutionCapture.bindings:type_name -> cozy.worker.v1.MachineCallableBinding
	43,  // 4: cozy.worker.v1.MachineExecutionCapture.model_defaults:type_name -> cozy.worker.v1.MachineModelDefault
	45,  // 5: cozy.worker.v1.MachineExecutionCapture.installed_packages:type_name -> cozy.worker.v1.InstalledPackage
	42,  // 6: cozy.worker.v1.MachineExecutionCapture.deferred_installations:type_name -> cozy.worker.v1.DeferredInstallation
	55,  // 7: cozy.worker.v1.MachineExecutionCapture.model_choices:type_name -> cozy.worker.v1.ModelChoice
	126, // 8: cozy.worker.v1.DeferredInstallation.preparation:type_name -> cozy.worker.v1.PreparePackageSetRequest
	44,  // 9: cozy.worker.v1.MachineModelDefault.rungs:type_name -> cozy.worker.v1.MachineModelDefaultRung
	178, // 10: cozy.worker.v1.MachineModelDefaultRung.manifest:type_name -> cozy.worker.v1.Ref
	156, // 11: cozy.worker.v1.MachineExecutionWorkspaceQuery.claim:type_name -> cozy.worker.v1.Claim
	179, // 12: cozy.worker.v1.MachineExecutionWorkspaceQuery.describe:type_name -> cozy.worker.v1.PackageSelection
	50,  // 13: cozy.worker.v1.MachineExecutionWorkspace.devices:type_name -> cozy.worker.v1.MachineDevice
	49,  // 14: cozy.worker.v1.MachineExecutionWorkspace.described_release:type_name -> cozy.worker.v1.DescribedRelease
	156, // 15: cozy.worker.v1.MachineSubmissionClose.claim:type_name -> cozy.worker.v1.Claim
	57,  // 16: cozy.worker.v1.MachineSubmissionClosure.receipt:type_name -> cozy.worker.v1.MachineExecutionReceipt
	156, // 17: cozy.worker.v1.MachineExecutionSubmit.claim:type_name -> cozy.worker.v1.Claim
	195, // 18: cozy.worker.v1.MachineExecutionSubmit.offer:type_name -> cozy.worker.v1.AttemptOffer
	164, // 19: cozy.worker.v1.MachineExecutionSubmit.prepared_state:type_name -> cozy.worker.v1.DesiredWorkerState
	56,  // 20: cozy.worker.v1.MachineExecutionSubmit.source_credentials:type_name -> cozy.worker.v1.SourceCredential
	54,  // 21: cozy.worker.v1.MachineExecutionSubmit.release_root:type_name -> cozy.worker.v1.ReleaseRoot
	55,  // 22: cozy.worker.v1.ReleaseRoot.models:type_name -> cozy.worker.v1.ModelChoice
	235, // 23: cozy.worker.v1.ReleaseRoot.inputs:type_name -> cozy.worker.v1.InputBinding
	241, // 24: cozy.worker.v1.ReleaseRoot.input_access:type_name -> cozy.worker.v1.InputAccess
	270, // 25: cozy.worker.v1.ReleaseRoot.capture:type_name -> cozy.worker.v1.ActivationCapture
	178, // 26: cozy.worker.v1.ModelChoice.manifest:type_name -> cozy.worker.v1.Ref
	175, // 27: cozy.worker.v1.ModelChoice.adapters:type_name -> cozy.worker.v1.DownloadAdapterRef
	5,   // 28: cozy.worker.v1.SourceCredential.provider:type_name -> cozy.worker.v1.NativeSourceOperation
	156, // 29: cozy.worker.v1.MachineExecutionQuery.claim:type_name -> cozy.worker.v1.Claim
	61,  // 30: cozy.worker.v1.MachineExecutionState.gpu:type_name -> cozy.worker.v1.MachineExecutionGpu
	5,   // 31: cozy.worker.v1.MachineExecutionState.awaiting_source_credentials:type_name -> cozy.worker.v1.NativeSourceOperation
	60,  // 32: cozy.worker.v1.MachineExecutionState.target:type_name -> cozy.worker.v1.MachineExecutionTarget
	58,  // 33: cozy.worker.v1.MachineExecutionEventsQuery.execution:type_name -> cozy.worker.v1.MachineExecutionQuery
	64,  // 34: cozy.worker.v1.MachineExecutionEvent.product:type_name -> cozy.worker.v1.RunProduct
	198, // 35: cozy.worker.v1.MachineExecutionEvent.outcome:type_name -> cozy.worker.v1.AttemptOutcome
	1,   // 36: cozy.worker.v1.RunProduct.op:type_name -> cozy.worker.v1.RunProductOp
	178, // 37: cozy.worker.v1.RunProduct.content:type_name -> cozy.worker.v1.Ref
	260, // 38: cozy.worker.v1.RunProduct.source:type_name -> cozy.worker.v1.NativeByteRetentionRequest
	65,  // 39: cozy.worker.v1.RunProduct.parts:type_name -> cozy.worker.v1.RunProductPart
	178, // 40: cozy.worker.v1.RunProductPart.content:type_name -> cozy.worker.v1.Ref
	260, // 41: cozy.worker.v1.RunProductPart.source:type_name -> cozy.worker.v1.NativeByteRetentionRequest
	63,  // 42: cozy.worker.v1.MachineExecutionEventPage.events:type_name -> cozy.worker.v1.MachineExecutionEvent
	58,  // 43: cozy.worker.v1.MachineExecutionControl.execution:type_name -> cozy.worker.v1.MachineExecutionQuery
	2,   // 44: cozy.worker.v1.MachineExecutionControl.action:type_name -> cozy.worker.v1.MachineExecutionAction
	69,  // 45: cozy.worker.v1.MachineExecutionControl.publication:type_name -> cozy.worker.v1.MachinePublicationReconciliation
	68,  // 46: cozy.worker.v1.MachineExecutionControl.memo:type_name -> cozy.worker.v1.MachineMemoAnswer
	58,  // 47: cozy.worker.v1.MachineExecutionCollect.execution:type_name -> cozy.worker.v1.MachineExecutionQuery
	58,  // 48: cozy.worker.v1.MachineExecutionCollectionAck.execution:type_name -> cozy.worker.v1.MachineExecutionQuery
	230, // 49: cozy.worker.v1.MachineExecutionCollectionAck.outcome:type_name -> cozy.worker.v1.AttemptOutcomeAck
	58,  // 50: cozy.worker.v1.MachineExecutionTriageQuery.execution:type_name -> cozy.worker.v1.MachineExecutionQuery
	253, // 51: cozy.worker.v1.MachineExecutionTriage.bundle:type_name -> cozy.worker.v1.TriageBundleRef
	156, // 52: cozy.worker.v1.MachineExecutionListQuery.claim:type_name -> cozy.worker.v1.Claim
	59,  // 53: cozy.worker.v1.MachineExecutionList.executions:type_name -> cozy.worker.v1.MachineExecutionState
	156, // 54: cozy.worker.v1.PackageListQuery.claim:type_name -> cozy.worker.v1.Claim
	78,  // 55: cozy.worker.v1.PackageList.packages:type_name -> cozy.worker.v1.MachinePackage
	129, // 56: cozy.worker.v1.MachinePackage.sdk:type_name -> cozy.worker.v1.ImageDistribution
	156, // 57: cozy.worker.v1.ModelListQuery.claim:type_name -> cozy.worker.v1.Claim
	81,  // 58: cozy.worker.v1.ModelList.models:type_name -> cozy.worker.v1.MachineModel
	82,  // 59: cozy.worker.v1.ModelList.repositories:type_name -> cozy.worker.v1.RepositoryUsage
	83,  // 60: cozy.worker.v1.ModelList.store:type_name -> cozy.worker.v1.StoreUsage
	178, // 61: cozy.worker.v1.MachineModel.manifest:type_name -> cozy.worker.v1.Ref
	84,  // 62: cozy.worker.v1.StoreUsage.filesystem:type_name -> cozy.worker.v1.MachineFilesystem
	156, // 63: cozy.worker.v1.DescribeMachineQuery.claim:type_name -> cozy.worker.v1.Claim
	87,  // 64: cozy.worker.v1.MachineDescription.host:type_name -> cozy.worker.v1.MachineHost
	89,  // 65: cozy.worker.v1.MachineDescription.runtime:type_name -> cozy.worker.v1.MachineRuntime
	88,  // 66: cozy.worker.v1.MachineHost.hubs:type_name -> cozy.worker.v1.MachineHub
	84,  // 67: cozy.worker.v1.MachineHost.filesystems:type_name -> cozy.worker.v1.MachineFilesystem
	50,  // 68: cozy.worker.v1.MachineRuntime.devices:type_name -> cozy.worker.v1.MachineDevice
	246, // 69: cozy.worker.v1.MachineRuntime.resources:type_name -> cozy.worker.v1.WorkerResources
	84,  // 70: cozy.worker.v1.MachineRuntime.store:type_name -> cozy.worker.v1.MachineFilesystem
	128, // 71: cozy.worker.v1.MachineRuntime.interpreters:type_name -> cozy.worker.v1.PythonInterpreter
	95,  // 72: cozy.worker.v1.MachineRuntime.preparation_progress:type_name -> cozy.worker.v1.PrepareModelProgress
	156, // 73: cozy.worker.v1.PreparePackageSetCall.claim:type_name -> cozy.worker.v1.Claim
	165, // 74: cozy.worker.v1.PreparePackageSetCall.package_set:type_name -> cozy.worker.v1.DesiredPackageSet
	127, // 75: cozy.worker.v1.PreparePackageSetCall.image_inventory:type_name -> cozy.worker.v1.ImageInventory
	156, // 76: cozy.worker.v1.PrepareLocalPackageCall.claim:type_name -> cozy.worker.v1.Claim
	166, // 77: cozy.worker.v1.PrepareLocalPackageCall.local_package_set:type_name -> cozy.worker.v1.DesiredLocalPackageSet
	156, // 78: cozy.worker.v1.PreparePrivatePlacementCall.claim:type_name -> cozy.worker.v1.Claim
	167, // 79: cozy.worker.v1.PreparePrivatePlacementCall.private_placement_set:type_name -> cozy.worker.v1.DesiredPrivatePlacementSet
	22,  // 80: cozy.worker.v1.PrepareEvent.stage:type_name -> cozy.worker.v1.PrepareStage
	170, // 81: cozy.worker.v1.PrepareEvent.placement_set:type_name -> cozy.worker.v1.DesiredPlacementSet
	95,  // 82: cozy.worker.v1.PrepareEvent.model_progress:type_name -> cozy.worker.v1.PrepareModelProgress
	45,  // 83: cozy.worker.v1.PrepareEvent.installed_package:type_name -> cozy.worker.v1.InstalledPackage
	174, // 84: cozy.worker.v1.PrepareModelProgress.model:type_name -> cozy.worker.v1.DownloadModelRef
	156, // 85: cozy.worker.v1.ModelSourceFileCall.claim:type_name -> cozy.worker.v1.Claim
	218, // 86: cozy.worker.v1.ModelSourceFileCall.request:type_name -> cozy.worker.v1.ModelSourceFileRequest
	156, // 87: cozy.worker.v1.ModelSourcePrepareCall.claim:type_name -> cozy.worker.v1.Claim
	220, // 88: cozy.worker.v1.ModelSourcePrepareCall.request:type_name -> cozy.worker.v1.ModelSourcePrepareRequest
	156, // 89: cozy.worker.v1.ModelSourceReleaseCall.claim:type_name -> cozy.worker.v1.Claim
	156, // 90: cozy.worker.v1.ModelSourceControlCall.claim:type_name -> cozy.worker.v1.Claim
	156, // 91: cozy.worker.v1.NumericalEnvironmentCall.claim:type_name -> cozy.worker.v1.Claim
	156, // 92: cozy.worker.v1.DerivedRetentionCall.claim:type_name -> cozy.worker.v1.Claim
	106, // 93: cozy.worker.v1.DerivedRetentionCall.request:type_name -> cozy.worker.v1.DerivedRetentionRequest
	178, // 94: cozy.worker.v1.DerivedRetentionResult.manifest:type_name -> cozy.worker.v1.Ref
	156, // 95: cozy.worker.v1.DerivedResultReleaseCall.claim:type_name -> cozy.worker.v1.Claim
	109, // 96: cozy.worker.v1.DerivedResultReleaseCall.request:type_name -> cozy.worker.v1.DerivedResultReleaseRequest
	156, // 97: cozy.worker.v1.RecordOperationResultCall.claim:type_name -> cozy.worker.v1.Claim
	156, // 98: cozy.worker.v1.LookupOperationCall.claim:type_name -> cozy.worker.v1.Claim
	198, // 99: cozy.worker.v1.LookupOperationResult.source:type_name -> cozy.worker.v1.AttemptOutcome
	107, // 100: cozy.worker.v1.LookupOperationResult.retentions:type_name -> cozy.worker.v1.DerivedRetentionResult
	262, // 101: cozy.worker.v1.LookupOperationResult.byte_retentions:type_name -> cozy.worker.v1.NativeByteRetentionResult
	156, // 102: cozy.worker.v1.PruneOperationCacheCall.claim:type_name -> cozy.worker.v1.Claim
	156, // 103: cozy.worker.v1.ModelSourceAdoptCall.claim:type_name -> cozy.worker.v1.Claim
	220, // 104: cozy.worker.v1.ModelSourceAdoptCall.request:type_name -> cozy.worker.v1.ModelSourcePrepareRequest
	156, // 105: cozy.worker.v1.CheckpointPageCall.claim:type_name -> cozy.worker.v1.Claim
	142, // 106: cozy.worker.v1.CheckpointPageCall.request:type_name -> cozy.worker.v1.CheckpointPageRequest
	156, // 107: cozy.worker.v1.CheckpointTransferCall.claim:type_name -> cozy.worker.v1.Claim
	144, // 108: cozy.worker.v1.CheckpointTransferCall.request:type_name -> cozy.worker.v1.CheckpointTransferRequest
	124, // 109: cozy.worker.v1.LocalPackageUploadFrame.header:type_name -> cozy.worker.v1.LocalPackageUploadHeader
	125, // 110: cozy.worker.v1.LocalPackageUploadFrame.chunk:type_name -> cozy.worker.v1.LocalPackageUploadChunk
	156, // 111: cozy.worker.v1.LocalPackageUploadHeader.claim:type_name -> cozy.worker.v1.Claim
	169, // 112: cozy.worker.v1.LocalPackageUploadHeader.file:type_name -> cozy.worker.v1.LocalPackageFileRef
	127, // 113: cozy.worker.v1.PreparePackageSetRequest.image_inventory:type_name -> cozy.worker.v1.ImageInventory
	129, // 114: cozy.worker.v1.ImageInventory.distributions:type_name -> cozy.worker.v1.ImageDistribution
	128, // 115: cozy.worker.v1.ImageInventory.interpreters:type_name -> cozy.worker.v1.PythonInterpreter
	170, // 116: cozy.worker.v1.PreparePackageSetResult.placement_set:type_name -> cozy.worker.v1.DesiredPlacementSet
	45,  // 117: cozy.worker.v1.PreparePackageSetResult.installed_package:type_name -> cozy.worker.v1.InstalledPackage
	180, // 118: cozy.worker.v1.PrepareLocalPackageRequest.package:type_name -> cozy.worker.v1.DevelopmentPackage
	132, // 119: cozy.worker.v1.PrepareLocalPackageRequest.files:type_name -> cozy.worker.v1.LocalPackageFile
	168, // 120: cozy.worker.v1.PreparePrivatePlacementRequest.native_models:type_name -> cozy.worker.v1.NativeModelBinding
	156, // 121: cozy.worker.v1.PreparePrivatePlacementRequest.claim:type_name -> cozy.worker.v1.Claim
	55,  // 122: cozy.worker.v1.PreparePrivatePlacementRequest.model_choices:type_name -> cozy.worker.v1.ModelChoice
	56,  // 123: cozy.worker.v1.PreparePrivatePlacementRequest.source_credentials:type_name -> cozy.worker.v1.SourceCredential
	178, // 124: cozy.worker.v1.PreparedModelSource.manifest:type_name -> cozy.worker.v1.Ref
	178, // 125: cozy.worker.v1.ModelSourceCheckpoint.head:type_name -> cozy.worker.v1.Ref
	138, // 126: cozy.worker.v1.CheckpointSubject.source:type_name -> cozy.worker.v1.SourceCheckpointSubject
	139, // 127: cozy.worker.v1.CheckpointSubject.weights:type_name -> cozy.worker.v1.WeightsCheckpointSubject
	178, // 128: cozy.worker.v1.CheckpointObject.ref:type_name -> cozy.worker.v1.Ref
	140, // 129: cozy.worker.v1.CheckpointPageRequest.subject:type_name -> cozy.worker.v1.CheckpointSubject
	178, // 130: cozy.worker.v1.CheckpointPageRequest.head:type_name -> cozy.worker.v1.Ref
	140, // 131: cozy.worker.v1.CheckpointPageResult.subject:type_name -> cozy.worker.v1.CheckpointSubject
	178, // 132: cozy.worker.v1.CheckpointPageResult.head:type_name -> cozy.worker.v1.Ref
	178, // 133: cozy.worker.v1.CheckpointPageResult.previous:type_name -> cozy.worker.v1.Ref
	178, // 134: cozy.worker.v1.CheckpointPageResult.progress:type_name -> cozy.worker.v1.Ref
	141, // 135: cozy.worker.v1.CheckpointPageResult.objects:type_name -> cozy.worker.v1.CheckpointObject
	140, // 136: cozy.worker.v1.CheckpointTransferRequest.subject:type_name -> cozy.worker.v1.CheckpointSubject
	178, // 137: cozy.worker.v1.CheckpointTransferRequest.head:type_name -> cozy.worker.v1.Ref
	141, // 138: cozy.worker.v1.CheckpointTransferRequest.object:type_name -> cozy.worker.v1.CheckpointObject
	215, // 139: cozy.worker.v1.CheckpointTransferRequest.upload_grant:type_name -> cozy.worker.v1.WeightsUploadGrant
	140, // 140: cozy.worker.v1.CheckpointTransferStatus.subject:type_name -> cozy.worker.v1.CheckpointSubject
	178, // 141: cozy.worker.v1.CheckpointTransferStatus.head:type_name -> cozy.worker.v1.Ref
	141, // 142: cozy.worker.v1.CheckpointTransferStatus.object:type_name -> cozy.worker.v1.CheckpointObject
	29,  // 143: cozy.worker.v1.CheckpointTransferStatus.state:type_name -> cozy.worker.v1.WeightsTransferState
	135, // 144: cozy.worker.v1.PrepareModelSourceRequest.profiles:type_name -> cozy.worker.v1.ModelSourceProfile
	134, // 145: cozy.worker.v1.PrepareModelSourceRequest.files:type_name -> cozy.worker.v1.LocalModelSourceFile
	137, // 146: cozy.worker.v1.PrepareModelSourceRequest.checkpoints:type_name -> cozy.worker.v1.ModelSourceCheckpoint
	32,  // 147: cozy.worker.v1.PrepareModelSourceResult.outcome:type_name -> cozy.worker.v1.ModelSourcePrepareOutcome
	136, // 148: cozy.worker.v1.PrepareModelSourceResult.sources:type_name -> cozy.worker.v1.PreparedModelSource
	137, // 149: cozy.worker.v1.PrepareModelSourceResult.checkpoints:type_name -> cozy.worker.v1.ModelSourceCheckpoint
	156, // 150: cozy.worker.v1.RecordOwnerFrame.claim:type_name -> cozy.worker.v1.Claim
	164, // 151: cozy.worker.v1.RecordOwnerFrame.desired_state:type_name -> cozy.worker.v1.DesiredWorkerState
	163, // 152: cozy.worker.v1.RecordOwnerFrame.snapshot_ack:type_name -> cozy.worker.v1.SnapshotAck
	158, // 153: cozy.worker.v1.WorkerFrame.claim_ack:type_name -> cozy.worker.v1.ClaimAck
	188, // 154: cozy.worker.v1.WorkerFrame.observed_state:type_name -> cozy.worker.v1.ObservedWorkerState
	159, // 155: cozy.worker.v1.WorkerFrame.boot_failure:type_name -> cozy.worker.v1.BootFailure
	160, // 156: cozy.worker.v1.WorkerFrame.snapshot:type_name -> cozy.worker.v1.WorkerSnapshot
	270, // 157: cozy.worker.v1.ChildCallRequest.capture:type_name -> cozy.worker.v1.ActivationCapture
	3,   // 158: cozy.worker.v1.ChildCallResult.state:type_name -> cozy.worker.v1.ChildCallState
	263, // 159: cozy.worker.v1.ChildCallResult.byte_result_grants:type_name -> cozy.worker.v1.ChildByteResultGrant
	273, // 160: cozy.worker.v1.ChildCallResult.observation:type_name -> cozy.worker.v1.ExecutionObservation
	178, // 161: cozy.worker.v1.NativeSourceMember.object:type_name -> cozy.worker.v1.Ref
	178, // 162: cozy.worker.v1.NativeSourceSelection.content_manifest:type_name -> cozy.worker.v1.Ref
	152, // 163: cozy.worker.v1.NativeSourceSelection.members:type_name -> cozy.worker.v1.NativeSourceMember
	150, // 164: cozy.worker.v1.NativeSourceCommand.parent_call:type_name -> cozy.worker.v1.ChildCallRequest
	5,   // 165: cozy.worker.v1.NativeSourceCommand.operation:type_name -> cozy.worker.v1.NativeSourceOperation
	4,   // 166: cozy.worker.v1.NativeSourceCommand.phase:type_name -> cozy.worker.v1.NativeSourcePhase
	153, // 167: cozy.worker.v1.NativeSourceCommand.selection:type_name -> cozy.worker.v1.NativeSourceSelection
	6,   // 168: cozy.worker.v1.NativeSourceStatus.state:type_name -> cozy.worker.v1.NativeSourceState
	153, // 169: cozy.worker.v1.NativeSourceStatus.selection:type_name -> cozy.worker.v1.NativeSourceSelection
	259, // 170: cozy.worker.v1.NativeSourceStatus.byte_output:type_name -> cozy.worker.v1.NativeByteTreeRef
	18,  // 171: cozy.worker.v1.ClaimAck.rejection:type_name -> cozy.worker.v1.ClaimRejection
	246, // 172: cozy.worker.v1.ClaimAck.resources:type_name -> cozy.worker.v1.WorkerResources
	20,  // 173: cozy.worker.v1.BootFailure.reason:type_name -> cozy.worker.v1.BootFailureReason
	246, // 174: cozy.worker.v1.BootFailure.resources:type_name -> cozy.worker.v1.WorkerResources
	8,   // 175: cozy.worker.v1.WorkerSnapshotBody.worker_phase:type_name -> cozy.worker.v1.WorkerPhase
	190, // 176: cozy.worker.v1.WorkerSnapshotBody.placements:type_name -> cozy.worker.v1.PlacementStatus
	11,  // 177: cozy.worker.v1.WorkerSnapshotBody.admission_state:type_name -> cozy.worker.v1.AdmissionState
	248, // 178: cozy.worker.v1.WorkerSnapshotBody.held_attempts:type_name -> cozy.worker.v1.HeldAttempt
	189, // 179: cozy.worker.v1.WorkerSnapshotBody.lanes:type_name -> cozy.worker.v1.DeviceLane
	248, // 180: cozy.worker.v1.HostSnapshotBody.held_outcomes:type_name -> cozy.worker.v1.HeldAttempt
	212, // 181: cozy.worker.v1.HostSnapshotBody.weights_transactions:type_name -> cozy.worker.v1.WeightsTransactionStatus
	7,   // 182: cozy.worker.v1.DesiredWorkerState.posture:type_name -> cozy.worker.v1.Posture
	187, // 183: cozy.worker.v1.DesiredWorkerState.job:type_name -> cozy.worker.v1.JobDirective
	170, // 184: cozy.worker.v1.DesiredWorkerState.placement_set:type_name -> cozy.worker.v1.DesiredPlacementSet
	274, // 185: cozy.worker.v1.DesiredWorkerState.runtime_revision:type_name -> cozy.worker.v1.RuntimeRevision
	180, // 186: cozy.worker.v1.DesiredLocalPackageSet.package:type_name -> cozy.worker.v1.DevelopmentPackage
	169, // 187: cozy.worker.v1.DesiredLocalPackageSet.files:type_name -> cozy.worker.v1.LocalPackageFileRef
	168, // 188: cozy.worker.v1.DesiredPrivatePlacementSet.native_models:type_name -> cozy.worker.v1.NativeModelBinding
	55,  // 189: cozy.worker.v1.DesiredPrivatePlacementSet.model_choices:type_name -> cozy.worker.v1.ModelChoice
	56,  // 190: cozy.worker.v1.DesiredPrivatePlacementSet.source_credentials:type_name -> cozy.worker.v1.SourceCredential
	178, // 191: cozy.worker.v1.NativeModelBinding.manifest:type_name -> cozy.worker.v1.Ref
	106, // 192: cozy.worker.v1.NativeModelBinding.retention:type_name -> cozy.worker.v1.DerivedRetentionRequest
	175, // 193: cozy.worker.v1.NativeModelBinding.adapters:type_name -> cozy.worker.v1.DownloadAdapterRef
	171, // 194: cozy.worker.v1.DesiredPlacementSet.device_pins:type_name -> cozy.worker.v1.PlacementDevicePin
	187, // 195: cozy.worker.v1.DesiredPlacementSet.orchestration_parent:type_name -> cozy.worker.v1.JobDirective
	177, // 196: cozy.worker.v1.PlacementSet.placements:type_name -> cozy.worker.v1.Placement
	174, // 197: cozy.worker.v1.DownloadDelegation.models:type_name -> cozy.worker.v1.DownloadModelRef
	176, // 198: cozy.worker.v1.DownloadDelegation.packages:type_name -> cozy.worker.v1.DownloadPackageRef
	175, // 199: cozy.worker.v1.DownloadModelRef.adapters:type_name -> cozy.worker.v1.DownloadAdapterRef
	179, // 200: cozy.worker.v1.Placement.package:type_name -> cozy.worker.v1.PackageSelection
	180, // 201: cozy.worker.v1.Placement.development:type_name -> cozy.worker.v1.DevelopmentPackage
	181, // 202: cozy.worker.v1.Placement.models:type_name -> cozy.worker.v1.Model
	182, // 203: cozy.worker.v1.Placement.entrypoints:type_name -> cozy.worker.v1.Entrypoint
	178, // 204: cozy.worker.v1.Model.manifest:type_name -> cozy.worker.v1.Ref
	183, // 205: cozy.worker.v1.Entrypoint.slots:type_name -> cozy.worker.v1.Slot
	185, // 206: cozy.worker.v1.Slot.components:type_name -> cozy.worker.v1.Component
	178, // 207: cozy.worker.v1.Slot.model_construction_contract:type_name -> cozy.worker.v1.Ref
	186, // 208: cozy.worker.v1.Slot.stamps:type_name -> cozy.worker.v1.Stamp
	184, // 209: cozy.worker.v1.Slot.adapters:type_name -> cozy.worker.v1.ModelAdapter
	244, // 210: cozy.worker.v1.JobDirective.resource_caps:type_name -> cozy.worker.v1.ResourceCaps
	245, // 211: cozy.worker.v1.JobDirective.publication_contract:type_name -> cozy.worker.v1.PublicationContract
	187, // 212: cozy.worker.v1.JobDirective.orchestration_parent:type_name -> cozy.worker.v1.JobDirective
	248, // 213: cozy.worker.v1.ObservedWorkerState.held_attempts:type_name -> cozy.worker.v1.HeldAttempt
	249, // 214: cozy.worker.v1.ObservedWorkerState.faults:type_name -> cozy.worker.v1.Fault
	194, // 215: cozy.worker.v1.ObservedWorkerState.activity:type_name -> cozy.worker.v1.ActivityEvent
	247, // 216: cozy.worker.v1.ObservedWorkerState.job_capacity:type_name -> cozy.worker.v1.JobCapacity
	190, // 217: cozy.worker.v1.ObservedWorkerState.placements:type_name -> cozy.worker.v1.PlacementStatus
	11,  // 218: cozy.worker.v1.ObservedWorkerState.admission_state:type_name -> cozy.worker.v1.AdmissionState
	8,   // 219: cozy.worker.v1.ObservedWorkerState.worker_phase:type_name -> cozy.worker.v1.WorkerPhase
	189, // 220: cozy.worker.v1.ObservedWorkerState.lanes:type_name -> cozy.worker.v1.DeviceLane
	249, // 221: cozy.worker.v1.PlacementStatus.faults:type_name -> cozy.worker.v1.Fault
	193, // 222: cozy.worker.v1.PlacementStatus.accelerator:type_name -> cozy.worker.v1.AcceleratorQualification
	9,   // 223: cozy.worker.v1.PlacementStatus.materialization:type_name -> cozy.worker.v1.MaterializationState
	10,  // 224: cozy.worker.v1.PlacementStatus.serving:type_name -> cozy.worker.v1.ServingState
	191, // 225: cozy.worker.v1.PlacementStatus.acquisition:type_name -> cozy.worker.v1.PlacementAcquisitionObservation
	192, // 226: cozy.worker.v1.PlacementAcquisitionObservation.package:type_name -> cozy.worker.v1.AcquisitionLegObservation
	192, // 227: cozy.worker.v1.PlacementAcquisitionObservation.model:type_name -> cozy.worker.v1.AcquisitionLegObservation
	239, // 228: cozy.worker.v1.AttemptOffer.grant:type_name -> cozy.worker.v1.DeliveryGrant
	17,  // 229: cozy.worker.v1.CancelAttempt.reason:type_name -> cozy.worker.v1.CancelReason
	14,  // 230: cozy.worker.v1.AttemptOutcomeBody.status:type_name -> cozy.worker.v1.OutcomeStatus
	250, // 231: cozy.worker.v1.AttemptOutcomeBody.output_manifest:type_name -> cozy.worker.v1.OutputManifest
	252, // 232: cozy.worker.v1.AttemptOutcomeBody.metrics:type_name -> cozy.worker.v1.AttemptMetrics
	253, // 233: cozy.worker.v1.AttemptOutcomeBody.triage_bundle:type_name -> cozy.worker.v1.TriageBundleRef
	228, // 234: cozy.worker.v1.AttemptOutcomeBody.cause:type_name -> cozy.worker.v1.OutcomeCause
	225, // 235: cozy.worker.v1.AttemptOutcomeBody.result:type_name -> cozy.worker.v1.ResultEnvelope
	200, // 236: cozy.worker.v1.AttemptOutcomeBody.weights_receipts:type_name -> cozy.worker.v1.WeightsReceiptRef
	273, // 237: cozy.worker.v1.AttemptOutcomeBody.observation:type_name -> cozy.worker.v1.ExecutionObservation
	23,  // 238: cozy.worker.v1.WeightsHostAck.stage:type_name -> cozy.worker.v1.WeightsHostStage
	24,  // 239: cozy.worker.v1.WeightsHostAck.outcome:type_name -> cozy.worker.v1.WeightsHostOutcome
	25,  // 240: cozy.worker.v1.WeightsHostAck.refusal:type_name -> cozy.worker.v1.WeightsHostRefusal
	200, // 241: cozy.worker.v1.WeightsHostAck.weights_receipt:type_name -> cozy.worker.v1.WeightsReceiptRef
	178, // 242: cozy.worker.v1.WeightsHostAck.manifest:type_name -> cozy.worker.v1.Ref
	205, // 243: cozy.worker.v1.WeightsHostAck.checkpoint:type_name -> cozy.worker.v1.CheckpointRef
	178, // 244: cozy.worker.v1.CheckpointRef.head:type_name -> cozy.worker.v1.Ref
	205, // 245: cozy.worker.v1.WeightsCheckpointFrame.checkpoint:type_name -> cozy.worker.v1.CheckpointRef
	139, // 246: cozy.worker.v1.WeightsIntentReadyRequest.weights:type_name -> cozy.worker.v1.WeightsCheckpointSubject
	205, // 247: cozy.worker.v1.WeightsIntentReadyRequest.checkpoint:type_name -> cozy.worker.v1.CheckpointRef
	207, // 248: cozy.worker.v1.ValidateWeightsCheckpointRequest.intent:type_name -> cozy.worker.v1.WeightsIntentReadyRequest
	139, // 249: cozy.worker.v1.ValidateWeightsCheckpointResult.weights:type_name -> cozy.worker.v1.WeightsCheckpointSubject
	205, // 250: cozy.worker.v1.ValidateWeightsCheckpointResult.checkpoint:type_name -> cozy.worker.v1.CheckpointRef
	23,  // 251: cozy.worker.v1.WeightsTransactionRefused.stage:type_name -> cozy.worker.v1.WeightsHostStage
	25,  // 252: cozy.worker.v1.WeightsTransactionRefused.refusal:type_name -> cozy.worker.v1.WeightsHostRefusal
	200, // 253: cozy.worker.v1.WeightsReceiptFrame.weights_receipt:type_name -> cozy.worker.v1.WeightsReceiptRef
	202, // 254: cozy.worker.v1.WeightsReceiptFrame.objects:type_name -> cozy.worker.v1.WeightsObjectSource
	178, // 255: cozy.worker.v1.WeightsReceiptFrame.manifest:type_name -> cozy.worker.v1.Ref
	26,  // 256: cozy.worker.v1.WeightsTransactionStatus.state:type_name -> cozy.worker.v1.WeightsTransactionState
	205, // 257: cozy.worker.v1.WeightsTransactionStatus.checkpoint:type_name -> cozy.worker.v1.CheckpointRef
	214, // 258: cozy.worker.v1.WeightsUploadGrant.required_headers:type_name -> cozy.worker.v1.WeightsUploadHeader
	215, // 259: cozy.worker.v1.WeightsUploadRequest.grant:type_name -> cozy.worker.v1.WeightsUploadGrant
	27,  // 260: cozy.worker.v1.WeightsUploadResult.outcome:type_name -> cozy.worker.v1.WeightsUploadOutcome
	28,  // 261: cozy.worker.v1.WeightsUploadResult.refusal:type_name -> cozy.worker.v1.WeightsUploadRefusal
	30,  // 262: cozy.worker.v1.ModelSourceFileRequest.provider:type_name -> cozy.worker.v1.ModelSourceProvider
	31,  // 263: cozy.worker.v1.ModelSourceFileStatus.state:type_name -> cozy.worker.v1.ModelSourceFileState
	135, // 264: cozy.worker.v1.ModelSourcePrepareRequest.profiles:type_name -> cozy.worker.v1.ModelSourceProfile
	137, // 265: cozy.worker.v1.ModelSourcePrepareRequest.checkpoints:type_name -> cozy.worker.v1.ModelSourceCheckpoint
	32,  // 266: cozy.worker.v1.ModelSourcePrepared.outcome:type_name -> cozy.worker.v1.ModelSourcePrepareOutcome
	136, // 267: cozy.worker.v1.ModelSourcePrepared.sources:type_name -> cozy.worker.v1.PreparedModelSource
	137, // 268: cozy.worker.v1.ModelSourcePrepared.checkpoints:type_name -> cozy.worker.v1.ModelSourceCheckpoint
	33,  // 269: cozy.worker.v1.LocalPackageFileStatus.state:type_name -> cozy.worker.v1.LocalPackageFileState
	34,  // 270: cozy.worker.v1.WeightsFinalizeRequest.disposition:type_name -> cozy.worker.v1.WeightsFinalizeDisposition
	35,  // 271: cozy.worker.v1.WeightsFinalizeResult.outcome:type_name -> cozy.worker.v1.WeightsFinalizeOutcome
	200, // 272: cozy.worker.v1.WeightsFinalizeResult.weights_receipt:type_name -> cozy.worker.v1.WeightsReceiptRef
	251, // 273: cozy.worker.v1.ResultEnvelope.result_blob:type_name -> cozy.worker.v1.OutputEntry
	227, // 274: cozy.worker.v1.ResultEnvelope.adjustments:type_name -> cozy.worker.v1.AdjustmentRow
	226, // 275: cozy.worker.v1.ResultEnvelope.retained_models:type_name -> cozy.worker.v1.RetainedModelResult
	106, // 276: cozy.worker.v1.RetainedModelResult.retention:type_name -> cozy.worker.v1.DerivedRetentionRequest
	15,  // 277: cozy.worker.v1.OutcomeCause.code:type_name -> cozy.worker.v1.CauseCode
	16,  // 278: cozy.worker.v1.OutcomeCause.origin:type_name -> cozy.worker.v1.CauseOrigin
	229, // 279: cozy.worker.v1.OutcomeCause.shortfall:type_name -> cozy.worker.v1.ResourceShortfall
	251, // 280: cozy.worker.v1.JobCheckpointRequest.artifact:type_name -> cozy.worker.v1.OutputEntry
	235, // 281: cozy.worker.v1.InvocationSpec.inputs:type_name -> cozy.worker.v1.InputBinding
	236, // 282: cozy.worker.v1.InvocationSpec.outputs:type_name -> cozy.worker.v1.OutputBinding
	237, // 283: cozy.worker.v1.InvocationSpec.serving:type_name -> cozy.worker.v1.ServingInvocationSpec
	238, // 284: cozy.worker.v1.InvocationSpec.job:type_name -> cozy.worker.v1.JobInvocationSpec
	270, // 285: cozy.worker.v1.InvocationSpec.capture:type_name -> cozy.worker.v1.ActivationCapture
	245, // 286: cozy.worker.v1.JobInvocationSpec.publication_contract:type_name -> cozy.worker.v1.PublicationContract
	243, // 287: cozy.worker.v1.DeliveryGrant.credential:type_name -> cozy.worker.v1.DeliveryAccessCredential
	241, // 288: cozy.worker.v1.DeliveryGrant.inputs:type_name -> cozy.worker.v1.InputAccess
	242, // 289: cozy.worker.v1.DeliveryGrant.outputs:type_name -> cozy.worker.v1.OutputAccess
	260, // 290: cozy.worker.v1.InputAccess.native_tree:type_name -> cozy.worker.v1.NativeByteRetentionRequest
	240, // 291: cozy.worker.v1.InputAccess.catalog_model:type_name -> cozy.worker.v1.CatalogModelSource
	236, // 292: cozy.worker.v1.PublicationContract.outputs:type_name -> cozy.worker.v1.OutputBinding
	12,  // 293: cozy.worker.v1.HeldAttempt.kind:type_name -> cozy.worker.v1.AttemptKind
	13,  // 294: cozy.worker.v1.HeldAttempt.state:type_name -> cozy.worker.v1.AttemptState
	19,  // 295: cozy.worker.v1.Fault.kind:type_name -> cozy.worker.v1.FaultKind
	251, // 296: cozy.worker.v1.OutputManifest.outputs:type_name -> cozy.worker.v1.OutputEntry
	259, // 297: cozy.worker.v1.OutputEntry.native_tree:type_name -> cozy.worker.v1.NativeByteTreeRef
	156, // 298: cozy.worker.v1.ForgetPackageCall.claim:type_name -> cozy.worker.v1.Claim
	156, // 299: cozy.worker.v1.NativeArtifactTransferCall.claim:type_name -> cozy.worker.v1.Claim
	257, // 300: cozy.worker.v1.NativeArtifactTransferCall.request:type_name -> cozy.worker.v1.NativeArtifactTransfer
	106, // 301: cozy.worker.v1.NativeArtifactTransfer.source:type_name -> cozy.worker.v1.DerivedRetentionRequest
	178, // 302: cozy.worker.v1.NativeArtifactTransfer.manifest:type_name -> cozy.worker.v1.Ref
	215, // 303: cozy.worker.v1.NativeArtifactTransfer.grant:type_name -> cozy.worker.v1.WeightsUploadGrant
	260, // 304: cozy.worker.v1.NativeArtifactTransfer.byte_source:type_name -> cozy.worker.v1.NativeByteRetentionRequest
	106, // 305: cozy.worker.v1.NativeArtifactTransferStatus.source:type_name -> cozy.worker.v1.DerivedRetentionRequest
	178, // 306: cozy.worker.v1.NativeArtifactTransferStatus.manifest:type_name -> cozy.worker.v1.Ref
	213, // 307: cozy.worker.v1.NativeArtifactTransferStatus.objects:type_name -> cozy.worker.v1.WeightsObjectRef
	27,  // 308: cozy.worker.v1.NativeArtifactTransferStatus.outcome:type_name -> cozy.worker.v1.WeightsUploadOutcome
	260, // 309: cozy.worker.v1.NativeArtifactTransferStatus.byte_source:type_name -> cozy.worker.v1.NativeByteRetentionRequest
	178, // 310: cozy.worker.v1.NativeByteTreeRef.manifest:type_name -> cozy.worker.v1.Ref
	259, // 311: cozy.worker.v1.NativeByteRetentionRequest.source:type_name -> cozy.worker.v1.NativeByteTreeRef
	156, // 312: cozy.worker.v1.NativeByteRetentionCall.claim:type_name -> cozy.worker.v1.Claim
	260, // 313: cozy.worker.v1.NativeByteRetentionCall.request:type_name -> cozy.worker.v1.NativeByteRetentionRequest
	259, // 314: cozy.worker.v1.NativeByteRetentionResult.source:type_name -> cozy.worker.v1.NativeByteTreeRef
	259, // 315: cozy.worker.v1.ChildByteResultGrant.source:type_name -> cozy.worker.v1.NativeByteTreeRef
	156, // 316: cozy.worker.v1.InputTreeImportHeader.claim:type_name -> cozy.worker.v1.Claim
	178, // 317: cozy.worker.v1.InputTreeImportHeader.manifest:type_name -> cozy.worker.v1.Ref
	178, // 318: cozy.worker.v1.InputTreeImportBlob.object:type_name -> cozy.worker.v1.Ref
	264, // 319: cozy.worker.v1.InputTreeImportFrame.header:type_name -> cozy.worker.v1.InputTreeImportHeader
	265, // 320: cozy.worker.v1.InputTreeImportFrame.blob:type_name -> cozy.worker.v1.InputTreeImportBlob
	266, // 321: cozy.worker.v1.InputTreeImportFrame.commit:type_name -> cozy.worker.v1.InputTreeImportCommit
	156, // 322: cozy.worker.v1.NativeByteReadCall.claim:type_name -> cozy.worker.v1.Claim
	260, // 323: cozy.worker.v1.NativeByteReadCall.source:type_name -> cozy.worker.v1.NativeByteRetentionRequest
	178, // 324: cozy.worker.v1.NativeByteReadCall.object:type_name -> cozy.worker.v1.Ref
	271, // 325: cozy.worker.v1.ExecutionObservation.environment:type_name -> cozy.worker.v1.ExecutionEnvironment
	272, // 326: cozy.worker.v1.ExecutionObservation.capture:type_name -> cozy.worker.v1.ActivationCaptureResult
	47,  // 327: cozy.worker.v1.WorkerControl.GetMachineExecutionWorkspace:input_type -> cozy.worker.v1.MachineExecutionWorkspaceQuery
	53,  // 328: cozy.worker.v1.WorkerControl.SubmitMachineExecution:input_type -> cozy.worker.v1.MachineExecutionSubmit
	51,  // 329: cozy.worker.v1.WorkerControl.CloseMachineSubmission:input_type -> cozy.worker.v1.MachineSubmissionClose
	58,  // 330: cozy.worker.v1.WorkerControl.GetMachineExecution:input_type -> cozy.worker.v1.MachineExecutionQuery
	62,  // 331: cozy.worker.v1.WorkerControl.ListMachineExecutionEvents:input_type -> cozy.worker.v1.MachineExecutionEventsQuery
	67,  // 332: cozy.worker.v1.WorkerControl.ControlMachineExecution:input_type -> cozy.worker.v1.MachineExecutionControl
	70,  // 333: cozy.worker.v1.WorkerControl.CollectMachineExecution:input_type -> cozy.worker.v1.MachineExecutionCollect
	71,  // 334: cozy.worker.v1.WorkerControl.AcknowledgeMachineExecutionCollection:input_type -> cozy.worker.v1.MachineExecutionCollectionAck
	72,  // 335: cozy.worker.v1.WorkerControl.ReadMachineExecutionTriage:input_type -> cozy.worker.v1.MachineExecutionTriageQuery
	74,  // 336: cozy.worker.v1.WorkerControl.ListMachineExecutions:input_type -> cozy.worker.v1.MachineExecutionListQuery
	76,  // 337: cozy.worker.v1.WorkerControl.ListPackages:input_type -> cozy.worker.v1.PackageListQuery
	79,  // 338: cozy.worker.v1.WorkerControl.ListModels:input_type -> cozy.worker.v1.ModelListQuery
	85,  // 339: cozy.worker.v1.WorkerControl.DescribeMachine:input_type -> cozy.worker.v1.DescribeMachineQuery
	148, // 340: cozy.worker.v1.WorkerControl.Control:input_type -> cozy.worker.v1.RecordOwnerFrame
	232, // 341: cozy.worker.v1.WorkerControl.WatchProgress:input_type -> cozy.worker.v1.ProgressOpen
	38,  // 342: cozy.worker.v1.RuntimePreparation.ProtocolInfo:input_type -> cozy.worker.v1.ProtocolInfoRequest
	100, // 343: cozy.worker.v1.RuntimePreparation.NumericalEnvironment:input_type -> cozy.worker.v1.NumericalEnvironmentRequest
	111, // 344: cozy.worker.v1.RuntimePreparation.RecordOperationResult:input_type -> cozy.worker.v1.RecordOperationResultCall
	113, // 345: cozy.worker.v1.RuntimePreparation.LookupOperation:input_type -> cozy.worker.v1.LookupOperationCall
	115, // 346: cozy.worker.v1.RuntimePreparation.PruneOperationCache:input_type -> cozy.worker.v1.PruneOperationCacheCall
	105, // 347: cozy.worker.v1.RuntimePreparation.WorkspaceRetainDerivedResult:input_type -> cozy.worker.v1.DerivedRetentionCall
	105, // 348: cozy.worker.v1.RuntimePreparation.WorkspaceReleaseDerivedRetention:input_type -> cozy.worker.v1.DerivedRetentionCall
	108, // 349: cozy.worker.v1.RuntimePreparation.WorkspaceReleaseDerivedResult:input_type -> cozy.worker.v1.DerivedResultReleaseCall
	261, // 350: cozy.worker.v1.RuntimePreparation.WorkspaceRetainByteTree:input_type -> cozy.worker.v1.NativeByteRetentionCall
	261, // 351: cozy.worker.v1.RuntimePreparation.WorkspaceReleaseByteTree:input_type -> cozy.worker.v1.NativeByteRetentionCall
	268, // 352: cozy.worker.v1.RuntimePreparation.WorkspaceReadByteTreeObject:input_type -> cozy.worker.v1.NativeByteReadCall
	256, // 353: cozy.worker.v1.RuntimePreparation.WorkspaceNativeArtifactTransfer:input_type -> cozy.worker.v1.NativeArtifactTransferCall
	254, // 354: cozy.worker.v1.RuntimePreparation.WorkspaceForgetPackage:input_type -> cozy.worker.v1.ForgetPackageCall
	267, // 355: cozy.worker.v1.RuntimePreparation.ImportInputTree:input_type -> cozy.worker.v1.InputTreeImportFrame
	126, // 356: cozy.worker.v1.RuntimePreparation.PreparePackageSet:input_type -> cozy.worker.v1.PreparePackageSetRequest
	146, // 357: cozy.worker.v1.RuntimePreparation.PrepareModelSource:input_type -> cozy.worker.v1.PrepareModelSourceRequest
	99,  // 358: cozy.worker.v1.RuntimePreparation.ReleaseModelSource:input_type -> cozy.worker.v1.ReleaseModelSourceRequest
	106, // 359: cozy.worker.v1.RuntimePreparation.RetainDerivedResult:input_type -> cozy.worker.v1.DerivedRetentionRequest
	106, // 360: cozy.worker.v1.RuntimePreparation.ReleaseDerivedRetention:input_type -> cozy.worker.v1.DerivedRetentionRequest
	109, // 361: cozy.worker.v1.RuntimePreparation.ReleaseDerivedResult:input_type -> cozy.worker.v1.DerivedResultReleaseRequest
	117, // 362: cozy.worker.v1.RuntimePreparation.CollectStoreGarbage:input_type -> cozy.worker.v1.CollectStoreGarbageRequest
	208, // 363: cozy.worker.v1.RuntimePreparation.ValidateWeightsCheckpoint:input_type -> cozy.worker.v1.ValidateWeightsCheckpointRequest
	142, // 364: cozy.worker.v1.RuntimePreparation.CheckpointPage:input_type -> cozy.worker.v1.CheckpointPageRequest
	144, // 365: cozy.worker.v1.RuntimePreparation.CheckpointTransfer:input_type -> cozy.worker.v1.CheckpointTransferRequest
	131, // 366: cozy.worker.v1.RuntimePreparation.PrepareLocalPackage:input_type -> cozy.worker.v1.PrepareLocalPackageRequest
	133, // 367: cozy.worker.v1.RuntimePreparation.PreparePrivatePlacement:input_type -> cozy.worker.v1.PreparePrivatePlacementRequest
	216, // 368: cozy.worker.v1.RuntimeWeights.Upload:input_type -> cozy.worker.v1.WeightsUploadRequest
	39,  // 369: cozy.worker.v1.PodHost.KeepRentalAlive:input_type -> cozy.worker.v1.KeepRentalAliveRequest
	47,  // 370: cozy.worker.v1.PodHost.GetMachineExecutionWorkspace:input_type -> cozy.worker.v1.MachineExecutionWorkspaceQuery
	53,  // 371: cozy.worker.v1.PodHost.SubmitMachineExecution:input_type -> cozy.worker.v1.MachineExecutionSubmit
	51,  // 372: cozy.worker.v1.PodHost.CloseMachineSubmission:input_type -> cozy.worker.v1.MachineSubmissionClose
	58,  // 373: cozy.worker.v1.PodHost.GetMachineExecution:input_type -> cozy.worker.v1.MachineExecutionQuery
	62,  // 374: cozy.worker.v1.PodHost.ListMachineExecutionEvents:input_type -> cozy.worker.v1.MachineExecutionEventsQuery
	67,  // 375: cozy.worker.v1.PodHost.ControlMachineExecution:input_type -> cozy.worker.v1.MachineExecutionControl
	70,  // 376: cozy.worker.v1.PodHost.CollectMachineExecution:input_type -> cozy.worker.v1.MachineExecutionCollect
	71,  // 377: cozy.worker.v1.PodHost.AcknowledgeMachineExecutionCollection:input_type -> cozy.worker.v1.MachineExecutionCollectionAck
	72,  // 378: cozy.worker.v1.PodHost.ReadMachineExecutionTriage:input_type -> cozy.worker.v1.MachineExecutionTriageQuery
	38,  // 379: cozy.worker.v1.PodHost.ProtocolInfo:input_type -> cozy.worker.v1.ProtocolInfoRequest
	103, // 380: cozy.worker.v1.PodHost.NumericalEnvironment:input_type -> cozy.worker.v1.NumericalEnvironmentCall
	91,  // 381: cozy.worker.v1.PodHost.PreparePackageSet:input_type -> cozy.worker.v1.PreparePackageSetCall
	92,  // 382: cozy.worker.v1.PodHost.PrepareLocalPackage:input_type -> cozy.worker.v1.PrepareLocalPackageCall
	93,  // 383: cozy.worker.v1.PodHost.PreparePrivatePlacement:input_type -> cozy.worker.v1.PreparePrivatePlacementCall
	96,  // 384: cozy.worker.v1.PodHost.ModelSourceFile:input_type -> cozy.worker.v1.ModelSourceFileCall
	97,  // 385: cozy.worker.v1.PodHost.ModelSourcePrepare:input_type -> cozy.worker.v1.ModelSourcePrepareCall
	98,  // 386: cozy.worker.v1.PodHost.ModelSourceRelease:input_type -> cozy.worker.v1.ModelSourceReleaseCall
	101, // 387: cozy.worker.v1.PodHost.ModelSourceControl:input_type -> cozy.worker.v1.ModelSourceControlCall
	105, // 388: cozy.worker.v1.PodHost.RetainDerivedResult:input_type -> cozy.worker.v1.DerivedRetentionCall
	105, // 389: cozy.worker.v1.PodHost.ReleaseDerivedRetention:input_type -> cozy.worker.v1.DerivedRetentionCall
	108, // 390: cozy.worker.v1.PodHost.ReleaseDerivedResult:input_type -> cozy.worker.v1.DerivedResultReleaseCall
	261, // 391: cozy.worker.v1.PodHost.RetainByteTree:input_type -> cozy.worker.v1.NativeByteRetentionCall
	261, // 392: cozy.worker.v1.PodHost.ReleaseByteTree:input_type -> cozy.worker.v1.NativeByteRetentionCall
	268, // 393: cozy.worker.v1.PodHost.ReadByteTreeObject:input_type -> cozy.worker.v1.NativeByteReadCall
	256, // 394: cozy.worker.v1.PodHost.NativeArtifactTransfer:input_type -> cozy.worker.v1.NativeArtifactTransferCall
	254, // 395: cozy.worker.v1.PodHost.ForgetPackage:input_type -> cozy.worker.v1.ForgetPackageCall
	267, // 396: cozy.worker.v1.PodHost.ImportInputTree:input_type -> cozy.worker.v1.InputTreeImportFrame
	111, // 397: cozy.worker.v1.PodHost.RecordOperationResult:input_type -> cozy.worker.v1.RecordOperationResultCall
	113, // 398: cozy.worker.v1.PodHost.LookupOperation:input_type -> cozy.worker.v1.LookupOperationCall
	115, // 399: cozy.worker.v1.PodHost.PruneOperationCache:input_type -> cozy.worker.v1.PruneOperationCacheCall
	120, // 400: cozy.worker.v1.PodHost.ModelSourceAdopt:input_type -> cozy.worker.v1.ModelSourceAdoptCall
	121, // 401: cozy.worker.v1.PodHost.CheckpointPage:input_type -> cozy.worker.v1.CheckpointPageCall
	122, // 402: cozy.worker.v1.PodHost.CheckpointTransfer:input_type -> cozy.worker.v1.CheckpointTransferCall
	123, // 403: cozy.worker.v1.PodHost.LocalPackageUpload:input_type -> cozy.worker.v1.LocalPackageUploadFrame
	74,  // 404: cozy.worker.v1.PodHost.ListMachineExecutions:input_type -> cozy.worker.v1.MachineExecutionListQuery
	76,  // 405: cozy.worker.v1.PodHost.ListPackages:input_type -> cozy.worker.v1.PackageListQuery
	79,  // 406: cozy.worker.v1.PodHost.ListModels:input_type -> cozy.worker.v1.ModelListQuery
	85,  // 407: cozy.worker.v1.PodHost.DescribeMachine:input_type -> cozy.worker.v1.DescribeMachineQuery
	36,  // 408: cozy.worker.v1.PodHost.ReadMachineLog:input_type -> cozy.worker.v1.MachineLogQuery
	48,  // 409: cozy.worker.v1.WorkerControl.GetMachineExecutionWorkspace:output_type -> cozy.worker.v1.MachineExecutionWorkspace
	57,  // 410: cozy.worker.v1.WorkerControl.SubmitMachineExecution:output_type -> cozy.worker.v1.MachineExecutionReceipt
	52,  // 411: cozy.worker.v1.WorkerControl.CloseMachineSubmission:output_type -> cozy.worker.v1.MachineSubmissionClosure
	59,  // 412: cozy.worker.v1.WorkerControl.GetMachineExecution:output_type -> cozy.worker.v1.MachineExecutionState
	66,  // 413: cozy.worker.v1.WorkerControl.ListMachineExecutionEvents:output_type -> cozy.worker.v1.MachineExecutionEventPage
	59,  // 414: cozy.worker.v1.WorkerControl.ControlMachineExecution:output_type -> cozy.worker.v1.MachineExecutionState
	198, // 415: cozy.worker.v1.WorkerControl.CollectMachineExecution:output_type -> cozy.worker.v1.AttemptOutcome
	59,  // 416: cozy.worker.v1.WorkerControl.AcknowledgeMachineExecutionCollection:output_type -> cozy.worker.v1.MachineExecutionState
	73,  // 417: cozy.worker.v1.WorkerControl.ReadMachineExecutionTriage:output_type -> cozy.worker.v1.MachineExecutionTriage
	75,  // 418: cozy.worker.v1.WorkerControl.ListMachineExecutions:output_type -> cozy.worker.v1.MachineExecutionList
	77,  // 419: cozy.worker.v1.WorkerControl.ListPackages:output_type -> cozy.worker.v1.PackageList
	80,  // 420: cozy.worker.v1.WorkerControl.ListModels:output_type -> cozy.worker.v1.ModelList
	86,  // 421: cozy.worker.v1.WorkerControl.DescribeMachine:output_type -> cozy.worker.v1.MachineDescription
	149, // 422: cozy.worker.v1.WorkerControl.Control:output_type -> cozy.worker.v1.WorkerFrame
	233, // 423: cozy.worker.v1.WorkerControl.WatchProgress:output_type -> cozy.worker.v1.AttemptProgress
	90,  // 424: cozy.worker.v1.RuntimePreparation.ProtocolInfo:output_type -> cozy.worker.v1.ProtocolInfoResult
	104, // 425: cozy.worker.v1.RuntimePreparation.NumericalEnvironment:output_type -> cozy.worker.v1.NumericalEnvironmentResult
	112, // 426: cozy.worker.v1.RuntimePreparation.RecordOperationResult:output_type -> cozy.worker.v1.RecordOperationResultResult
	114, // 427: cozy.worker.v1.RuntimePreparation.LookupOperation:output_type -> cozy.worker.v1.LookupOperationResult
	116, // 428: cozy.worker.v1.RuntimePreparation.PruneOperationCache:output_type -> cozy.worker.v1.PruneOperationCacheResult
	107, // 429: cozy.worker.v1.RuntimePreparation.WorkspaceRetainDerivedResult:output_type -> cozy.worker.v1.DerivedRetentionResult
	107, // 430: cozy.worker.v1.RuntimePreparation.WorkspaceReleaseDerivedRetention:output_type -> cozy.worker.v1.DerivedRetentionResult
	110, // 431: cozy.worker.v1.RuntimePreparation.WorkspaceReleaseDerivedResult:output_type -> cozy.worker.v1.DerivedResultReleaseResult
	262, // 432: cozy.worker.v1.RuntimePreparation.WorkspaceRetainByteTree:output_type -> cozy.worker.v1.NativeByteRetentionResult
	262, // 433: cozy.worker.v1.RuntimePreparation.WorkspaceReleaseByteTree:output_type -> cozy.worker.v1.NativeByteRetentionResult
	269, // 434: cozy.worker.v1.RuntimePreparation.WorkspaceReadByteTreeObject:output_type -> cozy.worker.v1.NativeByteReadChunk
	258, // 435: cozy.worker.v1.RuntimePreparation.WorkspaceNativeArtifactTransfer:output_type -> cozy.worker.v1.NativeArtifactTransferStatus
	255, // 436: cozy.worker.v1.RuntimePreparation.WorkspaceForgetPackage:output_type -> cozy.worker.v1.ForgetPackageResult
	262, // 437: cozy.worker.v1.RuntimePreparation.ImportInputTree:output_type -> cozy.worker.v1.NativeByteRetentionResult
	130, // 438: cozy.worker.v1.RuntimePreparation.PreparePackageSet:output_type -> cozy.worker.v1.PreparePackageSetResult
	147, // 439: cozy.worker.v1.RuntimePreparation.PrepareModelSource:output_type -> cozy.worker.v1.PrepareModelSourceResult
	119, // 440: cozy.worker.v1.RuntimePreparation.ReleaseModelSource:output_type -> cozy.worker.v1.ReleaseModelSourceResult
	107, // 441: cozy.worker.v1.RuntimePreparation.RetainDerivedResult:output_type -> cozy.worker.v1.DerivedRetentionResult
	107, // 442: cozy.worker.v1.RuntimePreparation.ReleaseDerivedRetention:output_type -> cozy.worker.v1.DerivedRetentionResult
	110, // 443: cozy.worker.v1.RuntimePreparation.ReleaseDerivedResult:output_type -> cozy.worker.v1.DerivedResultReleaseResult
	118, // 444: cozy.worker.v1.RuntimePreparation.CollectStoreGarbage:output_type -> cozy.worker.v1.CollectStoreGarbageResult
	209, // 445: cozy.worker.v1.RuntimePreparation.ValidateWeightsCheckpoint:output_type -> cozy.worker.v1.ValidateWeightsCheckpointResult
	143, // 446: cozy.worker.v1.RuntimePreparation.CheckpointPage:output_type -> cozy.worker.v1.CheckpointPageResult
	145, // 447: cozy.worker.v1.RuntimePreparation.CheckpointTransfer:output_type -> cozy.worker.v1.CheckpointTransferStatus
	130, // 448: cozy.worker.v1.RuntimePreparation.PrepareLocalPackage:output_type -> cozy.worker.v1.PreparePackageSetResult
	130, // 449: cozy.worker.v1.RuntimePreparation.PreparePrivatePlacement:output_type -> cozy.worker.v1.PreparePackageSetResult
	217, // 450: cozy.worker.v1.RuntimeWeights.Upload:output_type -> cozy.worker.v1.WeightsUploadResult
	40,  // 451: cozy.worker.v1.PodHost.KeepRentalAlive:output_type -> cozy.worker.v1.KeepRentalAliveResult
	48,  // 452: cozy.worker.v1.PodHost.GetMachineExecutionWorkspace:output_type -> cozy.worker.v1.MachineExecutionWorkspace
	57,  // 453: cozy.worker.v1.PodHost.SubmitMachineExecution:output_type -> cozy.worker.v1.MachineExecutionReceipt
	52,  // 454: cozy.worker.v1.PodHost.CloseMachineSubmission:output_type -> cozy.worker.v1.MachineSubmissionClosure
	59,  // 455: cozy.worker.v1.PodHost.GetMachineExecution:output_type -> cozy.worker.v1.MachineExecutionState
	66,  // 456: cozy.worker.v1.PodHost.ListMachineExecutionEvents:output_type -> cozy.worker.v1.MachineExecutionEventPage
	59,  // 457: cozy.worker.v1.PodHost.ControlMachineExecution:output_type -> cozy.worker.v1.MachineExecutionState
	198, // 458: cozy.worker.v1.PodHost.CollectMachineExecution:output_type -> cozy.worker.v1.AttemptOutcome
	59,  // 459: cozy.worker.v1.PodHost.AcknowledgeMachineExecutionCollection:output_type -> cozy.worker.v1.MachineExecutionState
	73,  // 460: cozy.worker.v1.PodHost.ReadMachineExecutionTriage:output_type -> cozy.worker.v1.MachineExecutionTriage
	90,  // 461: cozy.worker.v1.PodHost.ProtocolInfo:output_type -> cozy.worker.v1.ProtocolInfoResult
	104, // 462: cozy.worker.v1.PodHost.NumericalEnvironment:output_type -> cozy.worker.v1.NumericalEnvironmentResult
	94,  // 463: cozy.worker.v1.PodHost.PreparePackageSet:output_type -> cozy.worker.v1.PrepareEvent
	94,  // 464: cozy.worker.v1.PodHost.PrepareLocalPackage:output_type -> cozy.worker.v1.PrepareEvent
	94,  // 465: cozy.worker.v1.PodHost.PreparePrivatePlacement:output_type -> cozy.worker.v1.PrepareEvent
	219, // 466: cozy.worker.v1.PodHost.ModelSourceFile:output_type -> cozy.worker.v1.ModelSourceFileStatus
	221, // 467: cozy.worker.v1.PodHost.ModelSourcePrepare:output_type -> cozy.worker.v1.ModelSourcePrepared
	119, // 468: cozy.worker.v1.PodHost.ModelSourceRelease:output_type -> cozy.worker.v1.ReleaseModelSourceResult
	102, // 469: cozy.worker.v1.PodHost.ModelSourceControl:output_type -> cozy.worker.v1.ModelSourceControlResult
	107, // 470: cozy.worker.v1.PodHost.RetainDerivedResult:output_type -> cozy.worker.v1.DerivedRetentionResult
	107, // 471: cozy.worker.v1.PodHost.ReleaseDerivedRetention:output_type -> cozy.worker.v1.DerivedRetentionResult
	110, // 472: cozy.worker.v1.PodHost.ReleaseDerivedResult:output_type -> cozy.worker.v1.DerivedResultReleaseResult
	262, // 473: cozy.worker.v1.PodHost.RetainByteTree:output_type -> cozy.worker.v1.NativeByteRetentionResult
	262, // 474: cozy.worker.v1.PodHost.ReleaseByteTree:output_type -> cozy.worker.v1.NativeByteRetentionResult
	269, // 475: cozy.worker.v1.PodHost.ReadByteTreeObject:output_type -> cozy.worker.v1.NativeByteReadChunk
	258, // 476: cozy.worker.v1.PodHost.NativeArtifactTransfer:output_type -> cozy.worker.v1.NativeArtifactTransferStatus
	255, // 477: cozy.worker.v1.PodHost.ForgetPackage:output_type -> cozy.worker.v1.ForgetPackageResult
	262, // 478: cozy.worker.v1.PodHost.ImportInputTree:output_type -> cozy.worker.v1.NativeByteRetentionResult
	112, // 479: cozy.worker.v1.PodHost.RecordOperationResult:output_type -> cozy.worker.v1.RecordOperationResultResult
	114, // 480: cozy.worker.v1.PodHost.LookupOperation:output_type -> cozy.worker.v1.LookupOperationResult
	116, // 481: cozy.worker.v1.PodHost.PruneOperationCache:output_type -> cozy.worker.v1.PruneOperationCacheResult
	221, // 482: cozy.worker.v1.PodHost.ModelSourceAdopt:output_type -> cozy.worker.v1.ModelSourcePrepared
	143, // 483: cozy.worker.v1.PodHost.CheckpointPage:output_type -> cozy.worker.v1.CheckpointPageResult
	145, // 484: cozy.worker.v1.PodHost.CheckpointTransfer:output_type -> cozy.worker.v1.CheckpointTransferStatus
	222, // 485: cozy.worker.v1.PodHost.LocalPackageUpload:output_type -> cozy.worker.v1.LocalPackageFileStatus
	75,  // 486: cozy.worker.v1.PodHost.ListMachineExecutions:output_type -> cozy.worker.v1.MachineExecutionList
	77,  // 487: cozy.worker.v1.PodHost.ListPackages:output_type -> cozy.worker.v1.PackageList
	80,  // 488: cozy.worker.v1.PodHost.ListModels:output_type -> cozy.worker.v1.ModelList
	86,  // 489: cozy.worker.v1.PodHost.DescribeMachine:output_type -> cozy.worker.v1.MachineDescription
	37,  // 490: cozy.worker.v1.PodHost.ReadMachineLog:output_type -> cozy.worker.v1.MachineLogChunk
	409, // [409:491] is the sub-list for method output_type
	327, // [327:409] is the sub-list for method input_type
	327, // [327:327] is the sub-list for extension type_name
	327, // [327:327] is the sub-list for extension extendee
	0,   // [0:327] is the sub-list for field type_name
}

func init() { file_cozy_worker_v1_worker_proto_init() }
func file_cozy_worker_v1_worker_proto_init() {
	if File_cozy_worker_v1_worker_proto != nil {
		return
	}
	file_cozy_worker_v1_worker_proto_msgTypes[87].OneofWrappers = []any{
		(*LocalPackageUploadFrame_Header)(nil),
		(*LocalPackageUploadFrame_Chunk)(nil),
	}
	file_cozy_worker_v1_worker_proto_msgTypes[104].OneofWrappers = []any{
		(*CheckpointSubject_Source)(nil),
		(*CheckpointSubject_Weights)(nil),
	}
	file_cozy_worker_v1_worker_proto_msgTypes[108].OneofWrappers = []any{
		(*CheckpointTransferRequest_UploadGrant)(nil),
		(*CheckpointTransferRequest_DownloadUrl)(nil),
	}
	file_cozy_worker_v1_worker_proto_msgTypes[112].OneofWrappers = []any{
		(*RecordOwnerFrame_Claim)(nil),
		(*RecordOwnerFrame_DesiredState)(nil),
		(*RecordOwnerFrame_SnapshotAck)(nil),
	}
	file_cozy_worker_v1_worker_proto_msgTypes[113].OneofWrappers = []any{
		(*WorkerFrame_ClaimAck)(nil),
		(*WorkerFrame_ObservedState)(nil),
		(*WorkerFrame_BootFailure)(nil),
		(*WorkerFrame_Snapshot)(nil),
	}
	file_cozy_worker_v1_worker_proto_msgTypes[128].OneofWrappers = []any{
		(*DesiredWorkerState_Job)(nil),
		(*DesiredWorkerState_PlacementSet)(nil),
	}
	file_cozy_worker_v1_worker_proto_msgTypes[141].OneofWrappers = []any{
		(*Placement_Package)(nil),
		(*Placement_Development)(nil),
	}
	file_cozy_worker_v1_worker_proto_msgTypes[198].OneofWrappers = []any{
		(*InvocationSpec_Serving)(nil),
		(*InvocationSpec_Job)(nil),
	}
	file_cozy_worker_v1_worker_proto_msgTypes[231].OneofWrappers = []any{
		(*InputTreeImportFrame_Header)(nil),
		(*InputTreeImportFrame_Blob)(nil),
		(*InputTreeImportFrame_Commit)(nil),
	}
	type x struct{}
	out := protoimpl.TypeBuilder{
		File: protoimpl.DescBuilder{
			GoPackagePath: reflect.TypeOf(x{}).PkgPath(),
			RawDescriptor: unsafe.Slice(unsafe.StringData(file_cozy_worker_v1_worker_proto_rawDesc), len(file_cozy_worker_v1_worker_proto_rawDesc)),
			NumEnums:      36,
			NumMessages:   239,
			NumExtensions: 0,
			NumServices:   4,
		},
		GoTypes:           file_cozy_worker_v1_worker_proto_goTypes,
		DependencyIndexes: file_cozy_worker_v1_worker_proto_depIdxs,
		EnumInfos:         file_cozy_worker_v1_worker_proto_enumTypes,
		MessageInfos:      file_cozy_worker_v1_worker_proto_msgTypes,
	}.Build()
	File_cozy_worker_v1_worker_proto = out.File
	file_cozy_worker_v1_worker_proto_goTypes = nil
	file_cozy_worker_v1_worker_proto_depIdxs = nil
}
