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

// Code generated by protoc-gen-go-grpc. DO NOT EDIT.
// versions:
// - protoc-gen-go-grpc v1.6.0
// - protoc             v7.35.1
// source: cozy/worker/v1/worker.proto

package workerprotov1

import (
	context "context"
	grpc "google.golang.org/grpc"
	codes "google.golang.org/grpc/codes"
	status "google.golang.org/grpc/status"
)

// This is a compile-time assertion to ensure that this generated file
// is compatible with the grpc package it is being compiled against.
// Requires gRPC-Go v1.64.0 or later.
const _ = grpc.SupportPackageIsVersion9

const (
	WorkerControl_GetMachineExecutionWorkspace_FullMethodName          = "/cozy.worker.v1.WorkerControl/GetMachineExecutionWorkspace"
	WorkerControl_SubmitMachineExecution_FullMethodName                = "/cozy.worker.v1.WorkerControl/SubmitMachineExecution"
	WorkerControl_CloseMachineSubmission_FullMethodName                = "/cozy.worker.v1.WorkerControl/CloseMachineSubmission"
	WorkerControl_GetMachineExecution_FullMethodName                   = "/cozy.worker.v1.WorkerControl/GetMachineExecution"
	WorkerControl_ListMachineExecutionEvents_FullMethodName            = "/cozy.worker.v1.WorkerControl/ListMachineExecutionEvents"
	WorkerControl_ControlMachineExecution_FullMethodName               = "/cozy.worker.v1.WorkerControl/ControlMachineExecution"
	WorkerControl_CollectMachineExecution_FullMethodName               = "/cozy.worker.v1.WorkerControl/CollectMachineExecution"
	WorkerControl_AcknowledgeMachineExecutionCollection_FullMethodName = "/cozy.worker.v1.WorkerControl/AcknowledgeMachineExecutionCollection"
	WorkerControl_ReadMachineExecutionTriage_FullMethodName            = "/cozy.worker.v1.WorkerControl/ReadMachineExecutionTriage"
	WorkerControl_ListMachineExecutions_FullMethodName                 = "/cozy.worker.v1.WorkerControl/ListMachineExecutions"
	WorkerControl_ListPackages_FullMethodName                          = "/cozy.worker.v1.WorkerControl/ListPackages"
	WorkerControl_ListModels_FullMethodName                            = "/cozy.worker.v1.WorkerControl/ListModels"
	WorkerControl_DescribeMachine_FullMethodName                       = "/cozy.worker.v1.WorkerControl/DescribeMachine"
	WorkerControl_Control_FullMethodName                               = "/cozy.worker.v1.WorkerControl/Control"
	WorkerControl_WatchProgress_FullMethodName                         = "/cozy.worker.v1.WorkerControl/WatchProgress"
)

// WorkerControlClient is the client API for WorkerControl service.
//
// For semantics around ctx use and closing/ending streaming RPCs, please refer to https://pkg.go.dev/google.golang.org/grpc/?tab=doc#ClientConn.NewStream.
//
// The worker is the gRPC SERVER; the RecordOwner is the CLIENT.
type WorkerControlClient interface {
	// Discover Runtime's durable journal lifetime before freezing a submit.
	// Authenticated like the other machine calls; no continuing Control stream needed.
	GetMachineExecutionWorkspace(ctx context.Context, in *MachineExecutionWorkspaceQuery, opts ...grpc.CallOption) (*MachineExecutionWorkspace, error)
	// Runtime owns accepted work independently of the caller connection.
	// These calls observe/control that authority; they do not establish a second
	// RecordOwner or require a client coordinator inside the worker.
	SubmitMachineExecution(ctx context.Context, in *MachineExecutionSubmit, opts ...grpc.CallOption) (*MachineExecutionReceipt, error)
	// Atomically close a submission key against delayed acceptance, returning any existing receipt.
	CloseMachineSubmission(ctx context.Context, in *MachineSubmissionClose, opts ...grpc.CallOption) (*MachineSubmissionClosure, error)
	GetMachineExecution(ctx context.Context, in *MachineExecutionQuery, opts ...grpc.CallOption) (*MachineExecutionState, error)
	ListMachineExecutionEvents(ctx context.Context, in *MachineExecutionEventsQuery, opts ...grpc.CallOption) (*MachineExecutionEventPage, error)
	ControlMachineExecution(ctx context.Context, in *MachineExecutionControl, opts ...grpc.CallOption) (*MachineExecutionState, error)
	// Retired at wire 65: the terminal entry of ListMachineExecutionEvents carries the outcome.
	CollectMachineExecution(ctx context.Context, in *MachineExecutionCollect, opts ...grpc.CallOption) (*AttemptOutcome, error)
	// The owner holds every product of the log and the terminal. Wire 66: this means delivered, and
	// the products stay until the store's GC evicts them (delivered first); before 66 Runtime
	// released the log's holds here.
	AcknowledgeMachineExecutionCollection(ctx context.Context, in *MachineExecutionCollectionAck, opts ...grpc.CallOption) (*MachineExecutionState, error)
	// The triage bundle one terminal attempt's outcome names, read over the same authenticated
	// seam as the outcome.
	ReadMachineExecutionTriage(ctx context.Context, in *MachineExecutionTriageQuery, opts ...grpc.CallOption) (*MachineExecutionTriage, error)
	// Wire 66: the machine's own reads, authenticated like the calls above. A Runtime before 66
	// answers UNIMPLEMENTED, and its caller names the Runtime update for that read only.
	ListMachineExecutions(ctx context.Context, in *MachineExecutionListQuery, opts ...grpc.CallOption) (*MachineExecutionList, error)
	ListPackages(ctx context.Context, in *PackageListQuery, opts ...grpc.CallOption) (*PackageList, error)
	ListModels(ctx context.Context, in *ModelListQuery, opts ...grpc.CallOption) (*ModelList, error)
	DescribeMachine(ctx context.Context, in *DescribeMachineQuery, opts ...grpc.CallOption) (*MachineDescription, error)
	// Durable control: the RecordOwner's request stream carries RecordOwnerFrames, the worker's
	// response stream carries WorkerFrames. Durable messages are never shed to backpressure.
	// Outcome authority lives only here. gRPC orders each DIRECTION independently — there is no
	// cross-direction causal order; causality is expressed by revisions, snapshot ids, and acks.
	Control(ctx context.Context, opts ...grpc.CallOption) (grpc.BidiStreamingClient[RecordOwnerFrame, WorkerFrame], error)
	// Bounded LOSSY progress, worker->RecordOwner server-streaming. Sequence-numbered, gaps
	// visible, oldest shed first on overflow, no outcome authority. Where the contract promises
	// that progress saturation cannot block control, the RecordOwner opens WatchProgress on a
	// PHYSICALLY SEPARATE HTTP/2 connection — separate streams on one TCP connection do not
	// prove it.
	WatchProgress(ctx context.Context, in *ProgressOpen, opts ...grpc.CallOption) (grpc.ServerStreamingClient[AttemptProgress], error)
}

type workerControlClient struct {
	cc grpc.ClientConnInterface
}

func NewWorkerControlClient(cc grpc.ClientConnInterface) WorkerControlClient {
	return &workerControlClient{cc}
}

func (c *workerControlClient) GetMachineExecutionWorkspace(ctx context.Context, in *MachineExecutionWorkspaceQuery, opts ...grpc.CallOption) (*MachineExecutionWorkspace, error) {
	cOpts := append([]grpc.CallOption{grpc.StaticMethod()}, opts...)
	out := new(MachineExecutionWorkspace)
	err := c.cc.Invoke(ctx, WorkerControl_GetMachineExecutionWorkspace_FullMethodName, in, out, cOpts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (c *workerControlClient) SubmitMachineExecution(ctx context.Context, in *MachineExecutionSubmit, opts ...grpc.CallOption) (*MachineExecutionReceipt, error) {
	cOpts := append([]grpc.CallOption{grpc.StaticMethod()}, opts...)
	out := new(MachineExecutionReceipt)
	err := c.cc.Invoke(ctx, WorkerControl_SubmitMachineExecution_FullMethodName, in, out, cOpts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (c *workerControlClient) CloseMachineSubmission(ctx context.Context, in *MachineSubmissionClose, opts ...grpc.CallOption) (*MachineSubmissionClosure, error) {
	cOpts := append([]grpc.CallOption{grpc.StaticMethod()}, opts...)
	out := new(MachineSubmissionClosure)
	err := c.cc.Invoke(ctx, WorkerControl_CloseMachineSubmission_FullMethodName, in, out, cOpts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (c *workerControlClient) GetMachineExecution(ctx context.Context, in *MachineExecutionQuery, opts ...grpc.CallOption) (*MachineExecutionState, error) {
	cOpts := append([]grpc.CallOption{grpc.StaticMethod()}, opts...)
	out := new(MachineExecutionState)
	err := c.cc.Invoke(ctx, WorkerControl_GetMachineExecution_FullMethodName, in, out, cOpts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (c *workerControlClient) ListMachineExecutionEvents(ctx context.Context, in *MachineExecutionEventsQuery, opts ...grpc.CallOption) (*MachineExecutionEventPage, error) {
	cOpts := append([]grpc.CallOption{grpc.StaticMethod()}, opts...)
	out := new(MachineExecutionEventPage)
	err := c.cc.Invoke(ctx, WorkerControl_ListMachineExecutionEvents_FullMethodName, in, out, cOpts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (c *workerControlClient) ControlMachineExecution(ctx context.Context, in *MachineExecutionControl, opts ...grpc.CallOption) (*MachineExecutionState, error) {
	cOpts := append([]grpc.CallOption{grpc.StaticMethod()}, opts...)
	out := new(MachineExecutionState)
	err := c.cc.Invoke(ctx, WorkerControl_ControlMachineExecution_FullMethodName, in, out, cOpts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (c *workerControlClient) CollectMachineExecution(ctx context.Context, in *MachineExecutionCollect, opts ...grpc.CallOption) (*AttemptOutcome, error) {
	cOpts := append([]grpc.CallOption{grpc.StaticMethod()}, opts...)
	out := new(AttemptOutcome)
	err := c.cc.Invoke(ctx, WorkerControl_CollectMachineExecution_FullMethodName, in, out, cOpts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (c *workerControlClient) AcknowledgeMachineExecutionCollection(ctx context.Context, in *MachineExecutionCollectionAck, opts ...grpc.CallOption) (*MachineExecutionState, error) {
	cOpts := append([]grpc.CallOption{grpc.StaticMethod()}, opts...)
	out := new(MachineExecutionState)
	err := c.cc.Invoke(ctx, WorkerControl_AcknowledgeMachineExecutionCollection_FullMethodName, in, out, cOpts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (c *workerControlClient) ReadMachineExecutionTriage(ctx context.Context, in *MachineExecutionTriageQuery, opts ...grpc.CallOption) (*MachineExecutionTriage, error) {
	cOpts := append([]grpc.CallOption{grpc.StaticMethod()}, opts...)
	out := new(MachineExecutionTriage)
	err := c.cc.Invoke(ctx, WorkerControl_ReadMachineExecutionTriage_FullMethodName, in, out, cOpts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (c *workerControlClient) ListMachineExecutions(ctx context.Context, in *MachineExecutionListQuery, opts ...grpc.CallOption) (*MachineExecutionList, error) {
	cOpts := append([]grpc.CallOption{grpc.StaticMethod()}, opts...)
	out := new(MachineExecutionList)
	err := c.cc.Invoke(ctx, WorkerControl_ListMachineExecutions_FullMethodName, in, out, cOpts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (c *workerControlClient) ListPackages(ctx context.Context, in *PackageListQuery, opts ...grpc.CallOption) (*PackageList, error) {
	cOpts := append([]grpc.CallOption{grpc.StaticMethod()}, opts...)
	out := new(PackageList)
	err := c.cc.Invoke(ctx, WorkerControl_ListPackages_FullMethodName, in, out, cOpts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (c *workerControlClient) ListModels(ctx context.Context, in *ModelListQuery, opts ...grpc.CallOption) (*ModelList, error) {
	cOpts := append([]grpc.CallOption{grpc.StaticMethod()}, opts...)
	out := new(ModelList)
	err := c.cc.Invoke(ctx, WorkerControl_ListModels_FullMethodName, in, out, cOpts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (c *workerControlClient) DescribeMachine(ctx context.Context, in *DescribeMachineQuery, opts ...grpc.CallOption) (*MachineDescription, error) {
	cOpts := append([]grpc.CallOption{grpc.StaticMethod()}, opts...)
	out := new(MachineDescription)
	err := c.cc.Invoke(ctx, WorkerControl_DescribeMachine_FullMethodName, in, out, cOpts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (c *workerControlClient) Control(ctx context.Context, opts ...grpc.CallOption) (grpc.BidiStreamingClient[RecordOwnerFrame, WorkerFrame], error) {
	cOpts := append([]grpc.CallOption{grpc.StaticMethod()}, opts...)
	stream, err := c.cc.NewStream(ctx, &WorkerControl_ServiceDesc.Streams[0], WorkerControl_Control_FullMethodName, cOpts...)
	if err != nil {
		return nil, err
	}
	x := &grpc.GenericClientStream[RecordOwnerFrame, WorkerFrame]{ClientStream: stream}
	return x, nil
}

// This type alias is provided for backwards compatibility with existing code that references the prior non-generic stream type by name.
type WorkerControl_ControlClient = grpc.BidiStreamingClient[RecordOwnerFrame, WorkerFrame]

func (c *workerControlClient) WatchProgress(ctx context.Context, in *ProgressOpen, opts ...grpc.CallOption) (grpc.ServerStreamingClient[AttemptProgress], error) {
	cOpts := append([]grpc.CallOption{grpc.StaticMethod()}, opts...)
	stream, err := c.cc.NewStream(ctx, &WorkerControl_ServiceDesc.Streams[1], WorkerControl_WatchProgress_FullMethodName, cOpts...)
	if err != nil {
		return nil, err
	}
	x := &grpc.GenericClientStream[ProgressOpen, AttemptProgress]{ClientStream: stream}
	if err := x.ClientStream.SendMsg(in); err != nil {
		return nil, err
	}
	if err := x.ClientStream.CloseSend(); err != nil {
		return nil, err
	}
	return x, nil
}

// This type alias is provided for backwards compatibility with existing code that references the prior non-generic stream type by name.
type WorkerControl_WatchProgressClient = grpc.ServerStreamingClient[AttemptProgress]

// WorkerControlServer is the server API for WorkerControl service.
// All implementations must embed UnimplementedWorkerControlServer
// for forward compatibility.
//
// The worker is the gRPC SERVER; the RecordOwner is the CLIENT.
type WorkerControlServer interface {
	// Discover Runtime's durable journal lifetime before freezing a submit.
	// Authenticated like the other machine calls; no continuing Control stream needed.
	GetMachineExecutionWorkspace(context.Context, *MachineExecutionWorkspaceQuery) (*MachineExecutionWorkspace, error)
	// Runtime owns accepted work independently of the caller connection.
	// These calls observe/control that authority; they do not establish a second
	// RecordOwner or require a client coordinator inside the worker.
	SubmitMachineExecution(context.Context, *MachineExecutionSubmit) (*MachineExecutionReceipt, error)
	// Atomically close a submission key against delayed acceptance, returning any existing receipt.
	CloseMachineSubmission(context.Context, *MachineSubmissionClose) (*MachineSubmissionClosure, error)
	GetMachineExecution(context.Context, *MachineExecutionQuery) (*MachineExecutionState, error)
	ListMachineExecutionEvents(context.Context, *MachineExecutionEventsQuery) (*MachineExecutionEventPage, error)
	ControlMachineExecution(context.Context, *MachineExecutionControl) (*MachineExecutionState, error)
	// Retired at wire 65: the terminal entry of ListMachineExecutionEvents carries the outcome.
	CollectMachineExecution(context.Context, *MachineExecutionCollect) (*AttemptOutcome, error)
	// The owner holds every product of the log and the terminal. Wire 66: this means delivered, and
	// the products stay until the store's GC evicts them (delivered first); before 66 Runtime
	// released the log's holds here.
	AcknowledgeMachineExecutionCollection(context.Context, *MachineExecutionCollectionAck) (*MachineExecutionState, error)
	// The triage bundle one terminal attempt's outcome names, read over the same authenticated
	// seam as the outcome.
	ReadMachineExecutionTriage(context.Context, *MachineExecutionTriageQuery) (*MachineExecutionTriage, error)
	// Wire 66: the machine's own reads, authenticated like the calls above. A Runtime before 66
	// answers UNIMPLEMENTED, and its caller names the Runtime update for that read only.
	ListMachineExecutions(context.Context, *MachineExecutionListQuery) (*MachineExecutionList, error)
	ListPackages(context.Context, *PackageListQuery) (*PackageList, error)
	ListModels(context.Context, *ModelListQuery) (*ModelList, error)
	DescribeMachine(context.Context, *DescribeMachineQuery) (*MachineDescription, error)
	// Durable control: the RecordOwner's request stream carries RecordOwnerFrames, the worker's
	// response stream carries WorkerFrames. Durable messages are never shed to backpressure.
	// Outcome authority lives only here. gRPC orders each DIRECTION independently — there is no
	// cross-direction causal order; causality is expressed by revisions, snapshot ids, and acks.
	Control(grpc.BidiStreamingServer[RecordOwnerFrame, WorkerFrame]) error
	// Bounded LOSSY progress, worker->RecordOwner server-streaming. Sequence-numbered, gaps
	// visible, oldest shed first on overflow, no outcome authority. Where the contract promises
	// that progress saturation cannot block control, the RecordOwner opens WatchProgress on a
	// PHYSICALLY SEPARATE HTTP/2 connection — separate streams on one TCP connection do not
	// prove it.
	WatchProgress(*ProgressOpen, grpc.ServerStreamingServer[AttemptProgress]) error
	mustEmbedUnimplementedWorkerControlServer()
}

// UnimplementedWorkerControlServer must be embedded to have
// forward compatible implementations.
//
// NOTE: this should be embedded by value instead of pointer to avoid a nil
// pointer dereference when methods are called.
type UnimplementedWorkerControlServer struct{}

func (UnimplementedWorkerControlServer) GetMachineExecutionWorkspace(context.Context, *MachineExecutionWorkspaceQuery) (*MachineExecutionWorkspace, error) {
	return nil, status.Error(codes.Unimplemented, "method GetMachineExecutionWorkspace not implemented")
}
func (UnimplementedWorkerControlServer) SubmitMachineExecution(context.Context, *MachineExecutionSubmit) (*MachineExecutionReceipt, error) {
	return nil, status.Error(codes.Unimplemented, "method SubmitMachineExecution not implemented")
}
func (UnimplementedWorkerControlServer) CloseMachineSubmission(context.Context, *MachineSubmissionClose) (*MachineSubmissionClosure, error) {
	return nil, status.Error(codes.Unimplemented, "method CloseMachineSubmission not implemented")
}
func (UnimplementedWorkerControlServer) GetMachineExecution(context.Context, *MachineExecutionQuery) (*MachineExecutionState, error) {
	return nil, status.Error(codes.Unimplemented, "method GetMachineExecution not implemented")
}
func (UnimplementedWorkerControlServer) ListMachineExecutionEvents(context.Context, *MachineExecutionEventsQuery) (*MachineExecutionEventPage, error) {
	return nil, status.Error(codes.Unimplemented, "method ListMachineExecutionEvents not implemented")
}
func (UnimplementedWorkerControlServer) ControlMachineExecution(context.Context, *MachineExecutionControl) (*MachineExecutionState, error) {
	return nil, status.Error(codes.Unimplemented, "method ControlMachineExecution not implemented")
}
func (UnimplementedWorkerControlServer) CollectMachineExecution(context.Context, *MachineExecutionCollect) (*AttemptOutcome, error) {
	return nil, status.Error(codes.Unimplemented, "method CollectMachineExecution not implemented")
}
func (UnimplementedWorkerControlServer) AcknowledgeMachineExecutionCollection(context.Context, *MachineExecutionCollectionAck) (*MachineExecutionState, error) {
	return nil, status.Error(codes.Unimplemented, "method AcknowledgeMachineExecutionCollection not implemented")
}
func (UnimplementedWorkerControlServer) ReadMachineExecutionTriage(context.Context, *MachineExecutionTriageQuery) (*MachineExecutionTriage, error) {
	return nil, status.Error(codes.Unimplemented, "method ReadMachineExecutionTriage not implemented")
}
func (UnimplementedWorkerControlServer) ListMachineExecutions(context.Context, *MachineExecutionListQuery) (*MachineExecutionList, error) {
	return nil, status.Error(codes.Unimplemented, "method ListMachineExecutions not implemented")
}
func (UnimplementedWorkerControlServer) ListPackages(context.Context, *PackageListQuery) (*PackageList, error) {
	return nil, status.Error(codes.Unimplemented, "method ListPackages not implemented")
}
func (UnimplementedWorkerControlServer) ListModels(context.Context, *ModelListQuery) (*ModelList, error) {
	return nil, status.Error(codes.Unimplemented, "method ListModels not implemented")
}
func (UnimplementedWorkerControlServer) DescribeMachine(context.Context, *DescribeMachineQuery) (*MachineDescription, error) {
	return nil, status.Error(codes.Unimplemented, "method DescribeMachine not implemented")
}
func (UnimplementedWorkerControlServer) Control(grpc.BidiStreamingServer[RecordOwnerFrame, WorkerFrame]) error {
	return status.Error(codes.Unimplemented, "method Control not implemented")
}
func (UnimplementedWorkerControlServer) WatchProgress(*ProgressOpen, grpc.ServerStreamingServer[AttemptProgress]) error {
	return status.Error(codes.Unimplemented, "method WatchProgress not implemented")
}
func (UnimplementedWorkerControlServer) mustEmbedUnimplementedWorkerControlServer() {}
func (UnimplementedWorkerControlServer) testEmbeddedByValue()                       {}

// UnsafeWorkerControlServer may be embedded to opt out of forward compatibility for this service.
// Use of this interface is not recommended, as added methods to WorkerControlServer will
// result in compilation errors.
type UnsafeWorkerControlServer interface {
	mustEmbedUnimplementedWorkerControlServer()
}

func RegisterWorkerControlServer(s grpc.ServiceRegistrar, srv WorkerControlServer) {
	// If the following call panics, it indicates UnimplementedWorkerControlServer was
	// embedded by pointer and is nil.  This will cause panics if an
	// unimplemented method is ever invoked, so we test this at initialization
	// time to prevent it from happening at runtime later due to I/O.
	if t, ok := srv.(interface{ testEmbeddedByValue() }); ok {
		t.testEmbeddedByValue()
	}
	s.RegisterService(&WorkerControl_ServiceDesc, srv)
}

func _WorkerControl_GetMachineExecutionWorkspace_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(MachineExecutionWorkspaceQuery)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(WorkerControlServer).GetMachineExecutionWorkspace(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: WorkerControl_GetMachineExecutionWorkspace_FullMethodName,
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(WorkerControlServer).GetMachineExecutionWorkspace(ctx, req.(*MachineExecutionWorkspaceQuery))
	}
	return interceptor(ctx, in, info, handler)
}

func _WorkerControl_SubmitMachineExecution_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(MachineExecutionSubmit)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(WorkerControlServer).SubmitMachineExecution(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: WorkerControl_SubmitMachineExecution_FullMethodName,
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(WorkerControlServer).SubmitMachineExecution(ctx, req.(*MachineExecutionSubmit))
	}
	return interceptor(ctx, in, info, handler)
}

func _WorkerControl_CloseMachineSubmission_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(MachineSubmissionClose)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(WorkerControlServer).CloseMachineSubmission(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: WorkerControl_CloseMachineSubmission_FullMethodName,
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(WorkerControlServer).CloseMachineSubmission(ctx, req.(*MachineSubmissionClose))
	}
	return interceptor(ctx, in, info, handler)
}

func _WorkerControl_GetMachineExecution_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(MachineExecutionQuery)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(WorkerControlServer).GetMachineExecution(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: WorkerControl_GetMachineExecution_FullMethodName,
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(WorkerControlServer).GetMachineExecution(ctx, req.(*MachineExecutionQuery))
	}
	return interceptor(ctx, in, info, handler)
}

func _WorkerControl_ListMachineExecutionEvents_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(MachineExecutionEventsQuery)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(WorkerControlServer).ListMachineExecutionEvents(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: WorkerControl_ListMachineExecutionEvents_FullMethodName,
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(WorkerControlServer).ListMachineExecutionEvents(ctx, req.(*MachineExecutionEventsQuery))
	}
	return interceptor(ctx, in, info, handler)
}

func _WorkerControl_ControlMachineExecution_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(MachineExecutionControl)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(WorkerControlServer).ControlMachineExecution(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: WorkerControl_ControlMachineExecution_FullMethodName,
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(WorkerControlServer).ControlMachineExecution(ctx, req.(*MachineExecutionControl))
	}
	return interceptor(ctx, in, info, handler)
}

func _WorkerControl_CollectMachineExecution_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(MachineExecutionCollect)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(WorkerControlServer).CollectMachineExecution(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: WorkerControl_CollectMachineExecution_FullMethodName,
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(WorkerControlServer).CollectMachineExecution(ctx, req.(*MachineExecutionCollect))
	}
	return interceptor(ctx, in, info, handler)
}

func _WorkerControl_AcknowledgeMachineExecutionCollection_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(MachineExecutionCollectionAck)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(WorkerControlServer).AcknowledgeMachineExecutionCollection(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: WorkerControl_AcknowledgeMachineExecutionCollection_FullMethodName,
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(WorkerControlServer).AcknowledgeMachineExecutionCollection(ctx, req.(*MachineExecutionCollectionAck))
	}
	return interceptor(ctx, in, info, handler)
}

func _WorkerControl_ReadMachineExecutionTriage_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(MachineExecutionTriageQuery)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(WorkerControlServer).ReadMachineExecutionTriage(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: WorkerControl_ReadMachineExecutionTriage_FullMethodName,
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(WorkerControlServer).ReadMachineExecutionTriage(ctx, req.(*MachineExecutionTriageQuery))
	}
	return interceptor(ctx, in, info, handler)
}

func _WorkerControl_ListMachineExecutions_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(MachineExecutionListQuery)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(WorkerControlServer).ListMachineExecutions(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: WorkerControl_ListMachineExecutions_FullMethodName,
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(WorkerControlServer).ListMachineExecutions(ctx, req.(*MachineExecutionListQuery))
	}
	return interceptor(ctx, in, info, handler)
}

func _WorkerControl_ListPackages_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(PackageListQuery)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(WorkerControlServer).ListPackages(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: WorkerControl_ListPackages_FullMethodName,
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(WorkerControlServer).ListPackages(ctx, req.(*PackageListQuery))
	}
	return interceptor(ctx, in, info, handler)
}

func _WorkerControl_ListModels_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(ModelListQuery)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(WorkerControlServer).ListModels(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: WorkerControl_ListModels_FullMethodName,
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(WorkerControlServer).ListModels(ctx, req.(*ModelListQuery))
	}
	return interceptor(ctx, in, info, handler)
}

func _WorkerControl_DescribeMachine_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(DescribeMachineQuery)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(WorkerControlServer).DescribeMachine(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: WorkerControl_DescribeMachine_FullMethodName,
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(WorkerControlServer).DescribeMachine(ctx, req.(*DescribeMachineQuery))
	}
	return interceptor(ctx, in, info, handler)
}

func _WorkerControl_Control_Handler(srv interface{}, stream grpc.ServerStream) error {
	return srv.(WorkerControlServer).Control(&grpc.GenericServerStream[RecordOwnerFrame, WorkerFrame]{ServerStream: stream})
}

// This type alias is provided for backwards compatibility with existing code that references the prior non-generic stream type by name.
type WorkerControl_ControlServer = grpc.BidiStreamingServer[RecordOwnerFrame, WorkerFrame]

func _WorkerControl_WatchProgress_Handler(srv interface{}, stream grpc.ServerStream) error {
	m := new(ProgressOpen)
	if err := stream.RecvMsg(m); err != nil {
		return err
	}
	return srv.(WorkerControlServer).WatchProgress(m, &grpc.GenericServerStream[ProgressOpen, AttemptProgress]{ServerStream: stream})
}

// This type alias is provided for backwards compatibility with existing code that references the prior non-generic stream type by name.
type WorkerControl_WatchProgressServer = grpc.ServerStreamingServer[AttemptProgress]

// WorkerControl_ServiceDesc is the grpc.ServiceDesc for WorkerControl service.
// It's only intended for direct use with grpc.RegisterService,
// and not to be introspected or modified (even as a copy)
var WorkerControl_ServiceDesc = grpc.ServiceDesc{
	ServiceName: "cozy.worker.v1.WorkerControl",
	HandlerType: (*WorkerControlServer)(nil),
	Methods: []grpc.MethodDesc{
		{
			MethodName: "GetMachineExecutionWorkspace",
			Handler:    _WorkerControl_GetMachineExecutionWorkspace_Handler,
		},
		{
			MethodName: "SubmitMachineExecution",
			Handler:    _WorkerControl_SubmitMachineExecution_Handler,
		},
		{
			MethodName: "CloseMachineSubmission",
			Handler:    _WorkerControl_CloseMachineSubmission_Handler,
		},
		{
			MethodName: "GetMachineExecution",
			Handler:    _WorkerControl_GetMachineExecution_Handler,
		},
		{
			MethodName: "ListMachineExecutionEvents",
			Handler:    _WorkerControl_ListMachineExecutionEvents_Handler,
		},
		{
			MethodName: "ControlMachineExecution",
			Handler:    _WorkerControl_ControlMachineExecution_Handler,
		},
		{
			MethodName: "CollectMachineExecution",
			Handler:    _WorkerControl_CollectMachineExecution_Handler,
		},
		{
			MethodName: "AcknowledgeMachineExecutionCollection",
			Handler:    _WorkerControl_AcknowledgeMachineExecutionCollection_Handler,
		},
		{
			MethodName: "ReadMachineExecutionTriage",
			Handler:    _WorkerControl_ReadMachineExecutionTriage_Handler,
		},
		{
			MethodName: "ListMachineExecutions",
			Handler:    _WorkerControl_ListMachineExecutions_Handler,
		},
		{
			MethodName: "ListPackages",
			Handler:    _WorkerControl_ListPackages_Handler,
		},
		{
			MethodName: "ListModels",
			Handler:    _WorkerControl_ListModels_Handler,
		},
		{
			MethodName: "DescribeMachine",
			Handler:    _WorkerControl_DescribeMachine_Handler,
		},
	},
	Streams: []grpc.StreamDesc{
		{
			StreamName:    "Control",
			Handler:       _WorkerControl_Control_Handler,
			ServerStreams: true,
			ClientStreams: true,
		},
		{
			StreamName:    "WatchProgress",
			Handler:       _WorkerControl_WatchProgress_Handler,
			ServerStreams: true,
		},
	},
	Metadata: "cozy/worker/v1/worker.proto",
}

const (
	RuntimePreparation_ProtocolInfo_FullMethodName                     = "/cozy.worker.v1.RuntimePreparation/ProtocolInfo"
	RuntimePreparation_NumericalEnvironment_FullMethodName             = "/cozy.worker.v1.RuntimePreparation/NumericalEnvironment"
	RuntimePreparation_RecordOperationResult_FullMethodName            = "/cozy.worker.v1.RuntimePreparation/RecordOperationResult"
	RuntimePreparation_LookupOperation_FullMethodName                  = "/cozy.worker.v1.RuntimePreparation/LookupOperation"
	RuntimePreparation_PruneOperationCache_FullMethodName              = "/cozy.worker.v1.RuntimePreparation/PruneOperationCache"
	RuntimePreparation_WorkspaceRetainDerivedResult_FullMethodName     = "/cozy.worker.v1.RuntimePreparation/WorkspaceRetainDerivedResult"
	RuntimePreparation_WorkspaceReleaseDerivedRetention_FullMethodName = "/cozy.worker.v1.RuntimePreparation/WorkspaceReleaseDerivedRetention"
	RuntimePreparation_WorkspaceReleaseDerivedResult_FullMethodName    = "/cozy.worker.v1.RuntimePreparation/WorkspaceReleaseDerivedResult"
	RuntimePreparation_WorkspaceRetainByteTree_FullMethodName          = "/cozy.worker.v1.RuntimePreparation/WorkspaceRetainByteTree"
	RuntimePreparation_WorkspaceReleaseByteTree_FullMethodName         = "/cozy.worker.v1.RuntimePreparation/WorkspaceReleaseByteTree"
	RuntimePreparation_WorkspaceReadByteTreeObject_FullMethodName      = "/cozy.worker.v1.RuntimePreparation/WorkspaceReadByteTreeObject"
	RuntimePreparation_WorkspaceNativeArtifactTransfer_FullMethodName  = "/cozy.worker.v1.RuntimePreparation/WorkspaceNativeArtifactTransfer"
	RuntimePreparation_WorkspaceForgetPackage_FullMethodName           = "/cozy.worker.v1.RuntimePreparation/WorkspaceForgetPackage"
	RuntimePreparation_ImportInputTree_FullMethodName                  = "/cozy.worker.v1.RuntimePreparation/ImportInputTree"
	RuntimePreparation_PreparePackageSet_FullMethodName                = "/cozy.worker.v1.RuntimePreparation/PreparePackageSet"
	RuntimePreparation_PrepareModelSource_FullMethodName               = "/cozy.worker.v1.RuntimePreparation/PrepareModelSource"
	RuntimePreparation_ReleaseModelSource_FullMethodName               = "/cozy.worker.v1.RuntimePreparation/ReleaseModelSource"
	RuntimePreparation_RetainDerivedResult_FullMethodName              = "/cozy.worker.v1.RuntimePreparation/RetainDerivedResult"
	RuntimePreparation_ReleaseDerivedRetention_FullMethodName          = "/cozy.worker.v1.RuntimePreparation/ReleaseDerivedRetention"
	RuntimePreparation_ReleaseDerivedResult_FullMethodName             = "/cozy.worker.v1.RuntimePreparation/ReleaseDerivedResult"
	RuntimePreparation_CollectStoreGarbage_FullMethodName              = "/cozy.worker.v1.RuntimePreparation/CollectStoreGarbage"
	RuntimePreparation_ValidateWeightsCheckpoint_FullMethodName        = "/cozy.worker.v1.RuntimePreparation/ValidateWeightsCheckpoint"
	RuntimePreparation_CheckpointPage_FullMethodName                   = "/cozy.worker.v1.RuntimePreparation/CheckpointPage"
	RuntimePreparation_CheckpointTransfer_FullMethodName               = "/cozy.worker.v1.RuntimePreparation/CheckpointTransfer"
	RuntimePreparation_PrepareLocalPackage_FullMethodName              = "/cozy.worker.v1.RuntimePreparation/PrepareLocalPackage"
	RuntimePreparation_PreparePrivatePlacement_FullMethodName          = "/cozy.worker.v1.RuntimePreparation/PreparePrivatePlacement"
)

// RuntimePreparationClient is the client API for RuntimePreparation service.
//
// For semantics around ctx use and closing/ending streaming RPCs, please refer to https://pkg.go.dev/google.golang.org/grpc/?tab=doc#ClientConn.NewStream.
//
// Loopback-only pod-supervisor -> Runtime preparation seam. Pod-supervisor never registers this
// service on its external listener. Runtime receives logical refs, hub-known release facts,
// (local packages only) verified local wheel paths, and BOTH halves of the
// signed download delegation, because the party that resolves a model's closure against the hub
// is now Runtime's own TensorFS: no presigned URL, worker TLS credential, or index bearer. The
// anonymous public index directives inside locked_requirements are the other origin that
// crosses — an https URL, never a credential.
type RuntimePreparationClient interface {
	ProtocolInfo(ctx context.Context, in *ProtocolInfoRequest, opts ...grpc.CallOption) (*ProtocolInfoResult, error)
	NumericalEnvironment(ctx context.Context, in *NumericalEnvironmentRequest, opts ...grpc.CallOption) (*NumericalEnvironmentResult, error)
	// The same authenticated workspace custody service is used by the local owner
	// and the remote Host. Identity/custody is independent of a worker's job root.
	RecordOperationResult(ctx context.Context, in *RecordOperationResultCall, opts ...grpc.CallOption) (*RecordOperationResultResult, error)
	LookupOperation(ctx context.Context, in *LookupOperationCall, opts ...grpc.CallOption) (*LookupOperationResult, error)
	PruneOperationCache(ctx context.Context, in *PruneOperationCacheCall, opts ...grpc.CallOption) (*PruneOperationCacheResult, error)
	WorkspaceRetainDerivedResult(ctx context.Context, in *DerivedRetentionCall, opts ...grpc.CallOption) (*DerivedRetentionResult, error)
	WorkspaceReleaseDerivedRetention(ctx context.Context, in *DerivedRetentionCall, opts ...grpc.CallOption) (*DerivedRetentionResult, error)
	WorkspaceReleaseDerivedResult(ctx context.Context, in *DerivedResultReleaseCall, opts ...grpc.CallOption) (*DerivedResultReleaseResult, error)
	// Ordinary byte-tree custody uses the same authenticated Workspace.
	WorkspaceRetainByteTree(ctx context.Context, in *NativeByteRetentionCall, opts ...grpc.CallOption) (*NativeByteRetentionResult, error)
	WorkspaceReleaseByteTree(ctx context.Context, in *NativeByteRetentionCall, opts ...grpc.CallOption) (*NativeByteRetentionResult, error)
	WorkspaceReadByteTreeObject(ctx context.Context, in *NativeByteReadCall, opts ...grpc.CallOption) (grpc.ServerStreamingClient[NativeByteReadChunk], error)
	// A retained artifact's closure page or one granted object upload, for the machine
	// connection: the same command WorkerControl carries, authorized by the call's claim.
	WorkspaceNativeArtifactTransfer(ctx context.Context, in *NativeArtifactTransferCall, opts ...grpc.CallOption) (*NativeArtifactTransferStatus, error)
	// The package's owner state changed: drop what this machine read of it at its Hub.
	WorkspaceForgetPackage(ctx context.Context, in *ForgetPackageCall, opts ...grpc.CallOption) (*ForgetPackageResult, error)
	// Authenticated root input intake; native bytes commit before a receipt.
	ImportInputTree(ctx context.Context, opts ...grpc.CallOption) (grpc.ClientStreamingClient[InputTreeImportFrame, NativeByteRetentionResult], error)
	PreparePackageSet(ctx context.Context, in *PreparePackageSetRequest, opts ...grpc.CallOption) (*PreparePackageSetResult, error)
	PrepareModelSource(ctx context.Context, in *PrepareModelSourceRequest, opts ...grpc.CallOption) (*PrepareModelSourceResult, error)
	ReleaseModelSource(ctx context.Context, in *ReleaseModelSourceRequest, opts ...grpc.CallOption) (*ReleaseModelSourceResult, error)
	RetainDerivedResult(ctx context.Context, in *DerivedRetentionRequest, opts ...grpc.CallOption) (*DerivedRetentionResult, error)
	ReleaseDerivedRetention(ctx context.Context, in *DerivedRetentionRequest, opts ...grpc.CallOption) (*DerivedRetentionResult, error)
	ReleaseDerivedResult(ctx context.Context, in *DerivedResultReleaseRequest, opts ...grpc.CallOption) (*DerivedResultReleaseResult, error)
	CollectStoreGarbage(ctx context.Context, in *CollectStoreGarbageRequest, opts ...grpc.CallOption) (*CollectStoreGarbageResult, error)
	ValidateWeightsCheckpoint(ctx context.Context, in *ValidateWeightsCheckpointRequest, opts ...grpc.CallOption) (*ValidateWeightsCheckpointResult, error)
	CheckpointPage(ctx context.Context, in *CheckpointPageRequest, opts ...grpc.CallOption) (*CheckpointPageResult, error)
	CheckpointTransfer(ctx context.Context, in *CheckpointTransferRequest, opts ...grpc.CallOption) (*CheckpointTransferStatus, error)
	PrepareLocalPackage(ctx context.Context, in *PrepareLocalPackageRequest, opts ...grpc.CallOption) (*PreparePackageSetResult, error)
	PreparePrivatePlacement(ctx context.Context, in *PreparePrivatePlacementRequest, opts ...grpc.CallOption) (*PreparePackageSetResult, error)
}

type runtimePreparationClient struct {
	cc grpc.ClientConnInterface
}

func NewRuntimePreparationClient(cc grpc.ClientConnInterface) RuntimePreparationClient {
	return &runtimePreparationClient{cc}
}

func (c *runtimePreparationClient) ProtocolInfo(ctx context.Context, in *ProtocolInfoRequest, opts ...grpc.CallOption) (*ProtocolInfoResult, error) {
	cOpts := append([]grpc.CallOption{grpc.StaticMethod()}, opts...)
	out := new(ProtocolInfoResult)
	err := c.cc.Invoke(ctx, RuntimePreparation_ProtocolInfo_FullMethodName, in, out, cOpts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (c *runtimePreparationClient) NumericalEnvironment(ctx context.Context, in *NumericalEnvironmentRequest, opts ...grpc.CallOption) (*NumericalEnvironmentResult, error) {
	cOpts := append([]grpc.CallOption{grpc.StaticMethod()}, opts...)
	out := new(NumericalEnvironmentResult)
	err := c.cc.Invoke(ctx, RuntimePreparation_NumericalEnvironment_FullMethodName, in, out, cOpts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (c *runtimePreparationClient) RecordOperationResult(ctx context.Context, in *RecordOperationResultCall, opts ...grpc.CallOption) (*RecordOperationResultResult, error) {
	cOpts := append([]grpc.CallOption{grpc.StaticMethod()}, opts...)
	out := new(RecordOperationResultResult)
	err := c.cc.Invoke(ctx, RuntimePreparation_RecordOperationResult_FullMethodName, in, out, cOpts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (c *runtimePreparationClient) LookupOperation(ctx context.Context, in *LookupOperationCall, opts ...grpc.CallOption) (*LookupOperationResult, error) {
	cOpts := append([]grpc.CallOption{grpc.StaticMethod()}, opts...)
	out := new(LookupOperationResult)
	err := c.cc.Invoke(ctx, RuntimePreparation_LookupOperation_FullMethodName, in, out, cOpts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (c *runtimePreparationClient) PruneOperationCache(ctx context.Context, in *PruneOperationCacheCall, opts ...grpc.CallOption) (*PruneOperationCacheResult, error) {
	cOpts := append([]grpc.CallOption{grpc.StaticMethod()}, opts...)
	out := new(PruneOperationCacheResult)
	err := c.cc.Invoke(ctx, RuntimePreparation_PruneOperationCache_FullMethodName, in, out, cOpts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (c *runtimePreparationClient) WorkspaceRetainDerivedResult(ctx context.Context, in *DerivedRetentionCall, opts ...grpc.CallOption) (*DerivedRetentionResult, error) {
	cOpts := append([]grpc.CallOption{grpc.StaticMethod()}, opts...)
	out := new(DerivedRetentionResult)
	err := c.cc.Invoke(ctx, RuntimePreparation_WorkspaceRetainDerivedResult_FullMethodName, in, out, cOpts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (c *runtimePreparationClient) WorkspaceReleaseDerivedRetention(ctx context.Context, in *DerivedRetentionCall, opts ...grpc.CallOption) (*DerivedRetentionResult, error) {
	cOpts := append([]grpc.CallOption{grpc.StaticMethod()}, opts...)
	out := new(DerivedRetentionResult)
	err := c.cc.Invoke(ctx, RuntimePreparation_WorkspaceReleaseDerivedRetention_FullMethodName, in, out, cOpts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (c *runtimePreparationClient) WorkspaceReleaseDerivedResult(ctx context.Context, in *DerivedResultReleaseCall, opts ...grpc.CallOption) (*DerivedResultReleaseResult, error) {
	cOpts := append([]grpc.CallOption{grpc.StaticMethod()}, opts...)
	out := new(DerivedResultReleaseResult)
	err := c.cc.Invoke(ctx, RuntimePreparation_WorkspaceReleaseDerivedResult_FullMethodName, in, out, cOpts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (c *runtimePreparationClient) WorkspaceRetainByteTree(ctx context.Context, in *NativeByteRetentionCall, opts ...grpc.CallOption) (*NativeByteRetentionResult, error) {
	cOpts := append([]grpc.CallOption{grpc.StaticMethod()}, opts...)
	out := new(NativeByteRetentionResult)
	err := c.cc.Invoke(ctx, RuntimePreparation_WorkspaceRetainByteTree_FullMethodName, in, out, cOpts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (c *runtimePreparationClient) WorkspaceReleaseByteTree(ctx context.Context, in *NativeByteRetentionCall, opts ...grpc.CallOption) (*NativeByteRetentionResult, error) {
	cOpts := append([]grpc.CallOption{grpc.StaticMethod()}, opts...)
	out := new(NativeByteRetentionResult)
	err := c.cc.Invoke(ctx, RuntimePreparation_WorkspaceReleaseByteTree_FullMethodName, in, out, cOpts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (c *runtimePreparationClient) WorkspaceReadByteTreeObject(ctx context.Context, in *NativeByteReadCall, opts ...grpc.CallOption) (grpc.ServerStreamingClient[NativeByteReadChunk], error) {
	cOpts := append([]grpc.CallOption{grpc.StaticMethod()}, opts...)
	stream, err := c.cc.NewStream(ctx, &RuntimePreparation_ServiceDesc.Streams[0], RuntimePreparation_WorkspaceReadByteTreeObject_FullMethodName, cOpts...)
	if err != nil {
		return nil, err
	}
	x := &grpc.GenericClientStream[NativeByteReadCall, NativeByteReadChunk]{ClientStream: stream}
	if err := x.ClientStream.SendMsg(in); err != nil {
		return nil, err
	}
	if err := x.ClientStream.CloseSend(); err != nil {
		return nil, err
	}
	return x, nil
}

// This type alias is provided for backwards compatibility with existing code that references the prior non-generic stream type by name.
type RuntimePreparation_WorkspaceReadByteTreeObjectClient = grpc.ServerStreamingClient[NativeByteReadChunk]

func (c *runtimePreparationClient) WorkspaceNativeArtifactTransfer(ctx context.Context, in *NativeArtifactTransferCall, opts ...grpc.CallOption) (*NativeArtifactTransferStatus, error) {
	cOpts := append([]grpc.CallOption{grpc.StaticMethod()}, opts...)
	out := new(NativeArtifactTransferStatus)
	err := c.cc.Invoke(ctx, RuntimePreparation_WorkspaceNativeArtifactTransfer_FullMethodName, in, out, cOpts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (c *runtimePreparationClient) WorkspaceForgetPackage(ctx context.Context, in *ForgetPackageCall, opts ...grpc.CallOption) (*ForgetPackageResult, error) {
	cOpts := append([]grpc.CallOption{grpc.StaticMethod()}, opts...)
	out := new(ForgetPackageResult)
	err := c.cc.Invoke(ctx, RuntimePreparation_WorkspaceForgetPackage_FullMethodName, in, out, cOpts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (c *runtimePreparationClient) ImportInputTree(ctx context.Context, opts ...grpc.CallOption) (grpc.ClientStreamingClient[InputTreeImportFrame, NativeByteRetentionResult], error) {
	cOpts := append([]grpc.CallOption{grpc.StaticMethod()}, opts...)
	stream, err := c.cc.NewStream(ctx, &RuntimePreparation_ServiceDesc.Streams[1], RuntimePreparation_ImportInputTree_FullMethodName, cOpts...)
	if err != nil {
		return nil, err
	}
	x := &grpc.GenericClientStream[InputTreeImportFrame, NativeByteRetentionResult]{ClientStream: stream}
	return x, nil
}

// This type alias is provided for backwards compatibility with existing code that references the prior non-generic stream type by name.
type RuntimePreparation_ImportInputTreeClient = grpc.ClientStreamingClient[InputTreeImportFrame, NativeByteRetentionResult]

func (c *runtimePreparationClient) PreparePackageSet(ctx context.Context, in *PreparePackageSetRequest, opts ...grpc.CallOption) (*PreparePackageSetResult, error) {
	cOpts := append([]grpc.CallOption{grpc.StaticMethod()}, opts...)
	out := new(PreparePackageSetResult)
	err := c.cc.Invoke(ctx, RuntimePreparation_PreparePackageSet_FullMethodName, in, out, cOpts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (c *runtimePreparationClient) PrepareModelSource(ctx context.Context, in *PrepareModelSourceRequest, opts ...grpc.CallOption) (*PrepareModelSourceResult, error) {
	cOpts := append([]grpc.CallOption{grpc.StaticMethod()}, opts...)
	out := new(PrepareModelSourceResult)
	err := c.cc.Invoke(ctx, RuntimePreparation_PrepareModelSource_FullMethodName, in, out, cOpts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (c *runtimePreparationClient) ReleaseModelSource(ctx context.Context, in *ReleaseModelSourceRequest, opts ...grpc.CallOption) (*ReleaseModelSourceResult, error) {
	cOpts := append([]grpc.CallOption{grpc.StaticMethod()}, opts...)
	out := new(ReleaseModelSourceResult)
	err := c.cc.Invoke(ctx, RuntimePreparation_ReleaseModelSource_FullMethodName, in, out, cOpts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (c *runtimePreparationClient) RetainDerivedResult(ctx context.Context, in *DerivedRetentionRequest, opts ...grpc.CallOption) (*DerivedRetentionResult, error) {
	cOpts := append([]grpc.CallOption{grpc.StaticMethod()}, opts...)
	out := new(DerivedRetentionResult)
	err := c.cc.Invoke(ctx, RuntimePreparation_RetainDerivedResult_FullMethodName, in, out, cOpts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (c *runtimePreparationClient) ReleaseDerivedRetention(ctx context.Context, in *DerivedRetentionRequest, opts ...grpc.CallOption) (*DerivedRetentionResult, error) {
	cOpts := append([]grpc.CallOption{grpc.StaticMethod()}, opts...)
	out := new(DerivedRetentionResult)
	err := c.cc.Invoke(ctx, RuntimePreparation_ReleaseDerivedRetention_FullMethodName, in, out, cOpts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (c *runtimePreparationClient) ReleaseDerivedResult(ctx context.Context, in *DerivedResultReleaseRequest, opts ...grpc.CallOption) (*DerivedResultReleaseResult, error) {
	cOpts := append([]grpc.CallOption{grpc.StaticMethod()}, opts...)
	out := new(DerivedResultReleaseResult)
	err := c.cc.Invoke(ctx, RuntimePreparation_ReleaseDerivedResult_FullMethodName, in, out, cOpts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (c *runtimePreparationClient) CollectStoreGarbage(ctx context.Context, in *CollectStoreGarbageRequest, opts ...grpc.CallOption) (*CollectStoreGarbageResult, error) {
	cOpts := append([]grpc.CallOption{grpc.StaticMethod()}, opts...)
	out := new(CollectStoreGarbageResult)
	err := c.cc.Invoke(ctx, RuntimePreparation_CollectStoreGarbage_FullMethodName, in, out, cOpts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (c *runtimePreparationClient) ValidateWeightsCheckpoint(ctx context.Context, in *ValidateWeightsCheckpointRequest, opts ...grpc.CallOption) (*ValidateWeightsCheckpointResult, error) {
	cOpts := append([]grpc.CallOption{grpc.StaticMethod()}, opts...)
	out := new(ValidateWeightsCheckpointResult)
	err := c.cc.Invoke(ctx, RuntimePreparation_ValidateWeightsCheckpoint_FullMethodName, in, out, cOpts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (c *runtimePreparationClient) CheckpointPage(ctx context.Context, in *CheckpointPageRequest, opts ...grpc.CallOption) (*CheckpointPageResult, error) {
	cOpts := append([]grpc.CallOption{grpc.StaticMethod()}, opts...)
	out := new(CheckpointPageResult)
	err := c.cc.Invoke(ctx, RuntimePreparation_CheckpointPage_FullMethodName, in, out, cOpts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (c *runtimePreparationClient) CheckpointTransfer(ctx context.Context, in *CheckpointTransferRequest, opts ...grpc.CallOption) (*CheckpointTransferStatus, error) {
	cOpts := append([]grpc.CallOption{grpc.StaticMethod()}, opts...)
	out := new(CheckpointTransferStatus)
	err := c.cc.Invoke(ctx, RuntimePreparation_CheckpointTransfer_FullMethodName, in, out, cOpts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (c *runtimePreparationClient) PrepareLocalPackage(ctx context.Context, in *PrepareLocalPackageRequest, opts ...grpc.CallOption) (*PreparePackageSetResult, error) {
	cOpts := append([]grpc.CallOption{grpc.StaticMethod()}, opts...)
	out := new(PreparePackageSetResult)
	err := c.cc.Invoke(ctx, RuntimePreparation_PrepareLocalPackage_FullMethodName, in, out, cOpts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (c *runtimePreparationClient) PreparePrivatePlacement(ctx context.Context, in *PreparePrivatePlacementRequest, opts ...grpc.CallOption) (*PreparePackageSetResult, error) {
	cOpts := append([]grpc.CallOption{grpc.StaticMethod()}, opts...)
	out := new(PreparePackageSetResult)
	err := c.cc.Invoke(ctx, RuntimePreparation_PreparePrivatePlacement_FullMethodName, in, out, cOpts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

// RuntimePreparationServer is the server API for RuntimePreparation service.
// All implementations must embed UnimplementedRuntimePreparationServer
// for forward compatibility.
//
// Loopback-only pod-supervisor -> Runtime preparation seam. Pod-supervisor never registers this
// service on its external listener. Runtime receives logical refs, hub-known release facts,
// (local packages only) verified local wheel paths, and BOTH halves of the
// signed download delegation, because the party that resolves a model's closure against the hub
// is now Runtime's own TensorFS: no presigned URL, worker TLS credential, or index bearer. The
// anonymous public index directives inside locked_requirements are the other origin that
// crosses — an https URL, never a credential.
type RuntimePreparationServer interface {
	ProtocolInfo(context.Context, *ProtocolInfoRequest) (*ProtocolInfoResult, error)
	NumericalEnvironment(context.Context, *NumericalEnvironmentRequest) (*NumericalEnvironmentResult, error)
	// The same authenticated workspace custody service is used by the local owner
	// and the remote Host. Identity/custody is independent of a worker's job root.
	RecordOperationResult(context.Context, *RecordOperationResultCall) (*RecordOperationResultResult, error)
	LookupOperation(context.Context, *LookupOperationCall) (*LookupOperationResult, error)
	PruneOperationCache(context.Context, *PruneOperationCacheCall) (*PruneOperationCacheResult, error)
	WorkspaceRetainDerivedResult(context.Context, *DerivedRetentionCall) (*DerivedRetentionResult, error)
	WorkspaceReleaseDerivedRetention(context.Context, *DerivedRetentionCall) (*DerivedRetentionResult, error)
	WorkspaceReleaseDerivedResult(context.Context, *DerivedResultReleaseCall) (*DerivedResultReleaseResult, error)
	// Ordinary byte-tree custody uses the same authenticated Workspace.
	WorkspaceRetainByteTree(context.Context, *NativeByteRetentionCall) (*NativeByteRetentionResult, error)
	WorkspaceReleaseByteTree(context.Context, *NativeByteRetentionCall) (*NativeByteRetentionResult, error)
	WorkspaceReadByteTreeObject(*NativeByteReadCall, grpc.ServerStreamingServer[NativeByteReadChunk]) error
	// A retained artifact's closure page or one granted object upload, for the machine
	// connection: the same command WorkerControl carries, authorized by the call's claim.
	WorkspaceNativeArtifactTransfer(context.Context, *NativeArtifactTransferCall) (*NativeArtifactTransferStatus, error)
	// The package's owner state changed: drop what this machine read of it at its Hub.
	WorkspaceForgetPackage(context.Context, *ForgetPackageCall) (*ForgetPackageResult, error)
	// Authenticated root input intake; native bytes commit before a receipt.
	ImportInputTree(grpc.ClientStreamingServer[InputTreeImportFrame, NativeByteRetentionResult]) error
	PreparePackageSet(context.Context, *PreparePackageSetRequest) (*PreparePackageSetResult, error)
	PrepareModelSource(context.Context, *PrepareModelSourceRequest) (*PrepareModelSourceResult, error)
	ReleaseModelSource(context.Context, *ReleaseModelSourceRequest) (*ReleaseModelSourceResult, error)
	RetainDerivedResult(context.Context, *DerivedRetentionRequest) (*DerivedRetentionResult, error)
	ReleaseDerivedRetention(context.Context, *DerivedRetentionRequest) (*DerivedRetentionResult, error)
	ReleaseDerivedResult(context.Context, *DerivedResultReleaseRequest) (*DerivedResultReleaseResult, error)
	CollectStoreGarbage(context.Context, *CollectStoreGarbageRequest) (*CollectStoreGarbageResult, error)
	ValidateWeightsCheckpoint(context.Context, *ValidateWeightsCheckpointRequest) (*ValidateWeightsCheckpointResult, error)
	CheckpointPage(context.Context, *CheckpointPageRequest) (*CheckpointPageResult, error)
	CheckpointTransfer(context.Context, *CheckpointTransferRequest) (*CheckpointTransferStatus, error)
	PrepareLocalPackage(context.Context, *PrepareLocalPackageRequest) (*PreparePackageSetResult, error)
	PreparePrivatePlacement(context.Context, *PreparePrivatePlacementRequest) (*PreparePackageSetResult, error)
	mustEmbedUnimplementedRuntimePreparationServer()
}

// UnimplementedRuntimePreparationServer must be embedded to have
// forward compatible implementations.
//
// NOTE: this should be embedded by value instead of pointer to avoid a nil
// pointer dereference when methods are called.
type UnimplementedRuntimePreparationServer struct{}

func (UnimplementedRuntimePreparationServer) ProtocolInfo(context.Context, *ProtocolInfoRequest) (*ProtocolInfoResult, error) {
	return nil, status.Error(codes.Unimplemented, "method ProtocolInfo not implemented")
}
func (UnimplementedRuntimePreparationServer) NumericalEnvironment(context.Context, *NumericalEnvironmentRequest) (*NumericalEnvironmentResult, error) {
	return nil, status.Error(codes.Unimplemented, "method NumericalEnvironment not implemented")
}
func (UnimplementedRuntimePreparationServer) RecordOperationResult(context.Context, *RecordOperationResultCall) (*RecordOperationResultResult, error) {
	return nil, status.Error(codes.Unimplemented, "method RecordOperationResult not implemented")
}
func (UnimplementedRuntimePreparationServer) LookupOperation(context.Context, *LookupOperationCall) (*LookupOperationResult, error) {
	return nil, status.Error(codes.Unimplemented, "method LookupOperation not implemented")
}
func (UnimplementedRuntimePreparationServer) PruneOperationCache(context.Context, *PruneOperationCacheCall) (*PruneOperationCacheResult, error) {
	return nil, status.Error(codes.Unimplemented, "method PruneOperationCache not implemented")
}
func (UnimplementedRuntimePreparationServer) WorkspaceRetainDerivedResult(context.Context, *DerivedRetentionCall) (*DerivedRetentionResult, error) {
	return nil, status.Error(codes.Unimplemented, "method WorkspaceRetainDerivedResult not implemented")
}
func (UnimplementedRuntimePreparationServer) WorkspaceReleaseDerivedRetention(context.Context, *DerivedRetentionCall) (*DerivedRetentionResult, error) {
	return nil, status.Error(codes.Unimplemented, "method WorkspaceReleaseDerivedRetention not implemented")
}
func (UnimplementedRuntimePreparationServer) WorkspaceReleaseDerivedResult(context.Context, *DerivedResultReleaseCall) (*DerivedResultReleaseResult, error) {
	return nil, status.Error(codes.Unimplemented, "method WorkspaceReleaseDerivedResult not implemented")
}
func (UnimplementedRuntimePreparationServer) WorkspaceRetainByteTree(context.Context, *NativeByteRetentionCall) (*NativeByteRetentionResult, error) {
	return nil, status.Error(codes.Unimplemented, "method WorkspaceRetainByteTree not implemented")
}
func (UnimplementedRuntimePreparationServer) WorkspaceReleaseByteTree(context.Context, *NativeByteRetentionCall) (*NativeByteRetentionResult, error) {
	return nil, status.Error(codes.Unimplemented, "method WorkspaceReleaseByteTree not implemented")
}
func (UnimplementedRuntimePreparationServer) WorkspaceReadByteTreeObject(*NativeByteReadCall, grpc.ServerStreamingServer[NativeByteReadChunk]) error {
	return status.Error(codes.Unimplemented, "method WorkspaceReadByteTreeObject not implemented")
}
func (UnimplementedRuntimePreparationServer) WorkspaceNativeArtifactTransfer(context.Context, *NativeArtifactTransferCall) (*NativeArtifactTransferStatus, error) {
	return nil, status.Error(codes.Unimplemented, "method WorkspaceNativeArtifactTransfer not implemented")
}
func (UnimplementedRuntimePreparationServer) WorkspaceForgetPackage(context.Context, *ForgetPackageCall) (*ForgetPackageResult, error) {
	return nil, status.Error(codes.Unimplemented, "method WorkspaceForgetPackage not implemented")
}
func (UnimplementedRuntimePreparationServer) ImportInputTree(grpc.ClientStreamingServer[InputTreeImportFrame, NativeByteRetentionResult]) error {
	return status.Error(codes.Unimplemented, "method ImportInputTree not implemented")
}
func (UnimplementedRuntimePreparationServer) PreparePackageSet(context.Context, *PreparePackageSetRequest) (*PreparePackageSetResult, error) {
	return nil, status.Error(codes.Unimplemented, "method PreparePackageSet not implemented")
}
func (UnimplementedRuntimePreparationServer) PrepareModelSource(context.Context, *PrepareModelSourceRequest) (*PrepareModelSourceResult, error) {
	return nil, status.Error(codes.Unimplemented, "method PrepareModelSource not implemented")
}
func (UnimplementedRuntimePreparationServer) ReleaseModelSource(context.Context, *ReleaseModelSourceRequest) (*ReleaseModelSourceResult, error) {
	return nil, status.Error(codes.Unimplemented, "method ReleaseModelSource not implemented")
}
func (UnimplementedRuntimePreparationServer) RetainDerivedResult(context.Context, *DerivedRetentionRequest) (*DerivedRetentionResult, error) {
	return nil, status.Error(codes.Unimplemented, "method RetainDerivedResult not implemented")
}
func (UnimplementedRuntimePreparationServer) ReleaseDerivedRetention(context.Context, *DerivedRetentionRequest) (*DerivedRetentionResult, error) {
	return nil, status.Error(codes.Unimplemented, "method ReleaseDerivedRetention not implemented")
}
func (UnimplementedRuntimePreparationServer) ReleaseDerivedResult(context.Context, *DerivedResultReleaseRequest) (*DerivedResultReleaseResult, error) {
	return nil, status.Error(codes.Unimplemented, "method ReleaseDerivedResult not implemented")
}
func (UnimplementedRuntimePreparationServer) CollectStoreGarbage(context.Context, *CollectStoreGarbageRequest) (*CollectStoreGarbageResult, error) {
	return nil, status.Error(codes.Unimplemented, "method CollectStoreGarbage not implemented")
}
func (UnimplementedRuntimePreparationServer) ValidateWeightsCheckpoint(context.Context, *ValidateWeightsCheckpointRequest) (*ValidateWeightsCheckpointResult, error) {
	return nil, status.Error(codes.Unimplemented, "method ValidateWeightsCheckpoint not implemented")
}
func (UnimplementedRuntimePreparationServer) CheckpointPage(context.Context, *CheckpointPageRequest) (*CheckpointPageResult, error) {
	return nil, status.Error(codes.Unimplemented, "method CheckpointPage not implemented")
}
func (UnimplementedRuntimePreparationServer) CheckpointTransfer(context.Context, *CheckpointTransferRequest) (*CheckpointTransferStatus, error) {
	return nil, status.Error(codes.Unimplemented, "method CheckpointTransfer not implemented")
}
func (UnimplementedRuntimePreparationServer) PrepareLocalPackage(context.Context, *PrepareLocalPackageRequest) (*PreparePackageSetResult, error) {
	return nil, status.Error(codes.Unimplemented, "method PrepareLocalPackage not implemented")
}
func (UnimplementedRuntimePreparationServer) PreparePrivatePlacement(context.Context, *PreparePrivatePlacementRequest) (*PreparePackageSetResult, error) {
	return nil, status.Error(codes.Unimplemented, "method PreparePrivatePlacement not implemented")
}
func (UnimplementedRuntimePreparationServer) mustEmbedUnimplementedRuntimePreparationServer() {}
func (UnimplementedRuntimePreparationServer) testEmbeddedByValue()                            {}

// UnsafeRuntimePreparationServer may be embedded to opt out of forward compatibility for this service.
// Use of this interface is not recommended, as added methods to RuntimePreparationServer will
// result in compilation errors.
type UnsafeRuntimePreparationServer interface {
	mustEmbedUnimplementedRuntimePreparationServer()
}

func RegisterRuntimePreparationServer(s grpc.ServiceRegistrar, srv RuntimePreparationServer) {
	// If the following call panics, it indicates UnimplementedRuntimePreparationServer was
	// embedded by pointer and is nil.  This will cause panics if an
	// unimplemented method is ever invoked, so we test this at initialization
	// time to prevent it from happening at runtime later due to I/O.
	if t, ok := srv.(interface{ testEmbeddedByValue() }); ok {
		t.testEmbeddedByValue()
	}
	s.RegisterService(&RuntimePreparation_ServiceDesc, srv)
}

func _RuntimePreparation_ProtocolInfo_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(ProtocolInfoRequest)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(RuntimePreparationServer).ProtocolInfo(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: RuntimePreparation_ProtocolInfo_FullMethodName,
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(RuntimePreparationServer).ProtocolInfo(ctx, req.(*ProtocolInfoRequest))
	}
	return interceptor(ctx, in, info, handler)
}

func _RuntimePreparation_NumericalEnvironment_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(NumericalEnvironmentRequest)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(RuntimePreparationServer).NumericalEnvironment(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: RuntimePreparation_NumericalEnvironment_FullMethodName,
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(RuntimePreparationServer).NumericalEnvironment(ctx, req.(*NumericalEnvironmentRequest))
	}
	return interceptor(ctx, in, info, handler)
}

func _RuntimePreparation_RecordOperationResult_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(RecordOperationResultCall)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(RuntimePreparationServer).RecordOperationResult(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: RuntimePreparation_RecordOperationResult_FullMethodName,
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(RuntimePreparationServer).RecordOperationResult(ctx, req.(*RecordOperationResultCall))
	}
	return interceptor(ctx, in, info, handler)
}

func _RuntimePreparation_LookupOperation_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(LookupOperationCall)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(RuntimePreparationServer).LookupOperation(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: RuntimePreparation_LookupOperation_FullMethodName,
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(RuntimePreparationServer).LookupOperation(ctx, req.(*LookupOperationCall))
	}
	return interceptor(ctx, in, info, handler)
}

func _RuntimePreparation_PruneOperationCache_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(PruneOperationCacheCall)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(RuntimePreparationServer).PruneOperationCache(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: RuntimePreparation_PruneOperationCache_FullMethodName,
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(RuntimePreparationServer).PruneOperationCache(ctx, req.(*PruneOperationCacheCall))
	}
	return interceptor(ctx, in, info, handler)
}

func _RuntimePreparation_WorkspaceRetainDerivedResult_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(DerivedRetentionCall)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(RuntimePreparationServer).WorkspaceRetainDerivedResult(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: RuntimePreparation_WorkspaceRetainDerivedResult_FullMethodName,
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(RuntimePreparationServer).WorkspaceRetainDerivedResult(ctx, req.(*DerivedRetentionCall))
	}
	return interceptor(ctx, in, info, handler)
}

func _RuntimePreparation_WorkspaceReleaseDerivedRetention_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(DerivedRetentionCall)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(RuntimePreparationServer).WorkspaceReleaseDerivedRetention(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: RuntimePreparation_WorkspaceReleaseDerivedRetention_FullMethodName,
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(RuntimePreparationServer).WorkspaceReleaseDerivedRetention(ctx, req.(*DerivedRetentionCall))
	}
	return interceptor(ctx, in, info, handler)
}

func _RuntimePreparation_WorkspaceReleaseDerivedResult_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(DerivedResultReleaseCall)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(RuntimePreparationServer).WorkspaceReleaseDerivedResult(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: RuntimePreparation_WorkspaceReleaseDerivedResult_FullMethodName,
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(RuntimePreparationServer).WorkspaceReleaseDerivedResult(ctx, req.(*DerivedResultReleaseCall))
	}
	return interceptor(ctx, in, info, handler)
}

func _RuntimePreparation_WorkspaceRetainByteTree_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(NativeByteRetentionCall)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(RuntimePreparationServer).WorkspaceRetainByteTree(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: RuntimePreparation_WorkspaceRetainByteTree_FullMethodName,
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(RuntimePreparationServer).WorkspaceRetainByteTree(ctx, req.(*NativeByteRetentionCall))
	}
	return interceptor(ctx, in, info, handler)
}

func _RuntimePreparation_WorkspaceReleaseByteTree_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(NativeByteRetentionCall)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(RuntimePreparationServer).WorkspaceReleaseByteTree(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: RuntimePreparation_WorkspaceReleaseByteTree_FullMethodName,
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(RuntimePreparationServer).WorkspaceReleaseByteTree(ctx, req.(*NativeByteRetentionCall))
	}
	return interceptor(ctx, in, info, handler)
}

func _RuntimePreparation_WorkspaceReadByteTreeObject_Handler(srv interface{}, stream grpc.ServerStream) error {
	m := new(NativeByteReadCall)
	if err := stream.RecvMsg(m); err != nil {
		return err
	}
	return srv.(RuntimePreparationServer).WorkspaceReadByteTreeObject(m, &grpc.GenericServerStream[NativeByteReadCall, NativeByteReadChunk]{ServerStream: stream})
}

// This type alias is provided for backwards compatibility with existing code that references the prior non-generic stream type by name.
type RuntimePreparation_WorkspaceReadByteTreeObjectServer = grpc.ServerStreamingServer[NativeByteReadChunk]

func _RuntimePreparation_WorkspaceNativeArtifactTransfer_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(NativeArtifactTransferCall)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(RuntimePreparationServer).WorkspaceNativeArtifactTransfer(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: RuntimePreparation_WorkspaceNativeArtifactTransfer_FullMethodName,
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(RuntimePreparationServer).WorkspaceNativeArtifactTransfer(ctx, req.(*NativeArtifactTransferCall))
	}
	return interceptor(ctx, in, info, handler)
}

func _RuntimePreparation_WorkspaceForgetPackage_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(ForgetPackageCall)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(RuntimePreparationServer).WorkspaceForgetPackage(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: RuntimePreparation_WorkspaceForgetPackage_FullMethodName,
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(RuntimePreparationServer).WorkspaceForgetPackage(ctx, req.(*ForgetPackageCall))
	}
	return interceptor(ctx, in, info, handler)
}

func _RuntimePreparation_ImportInputTree_Handler(srv interface{}, stream grpc.ServerStream) error {
	return srv.(RuntimePreparationServer).ImportInputTree(&grpc.GenericServerStream[InputTreeImportFrame, NativeByteRetentionResult]{ServerStream: stream})
}

// This type alias is provided for backwards compatibility with existing code that references the prior non-generic stream type by name.
type RuntimePreparation_ImportInputTreeServer = grpc.ClientStreamingServer[InputTreeImportFrame, NativeByteRetentionResult]

func _RuntimePreparation_PreparePackageSet_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(PreparePackageSetRequest)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(RuntimePreparationServer).PreparePackageSet(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: RuntimePreparation_PreparePackageSet_FullMethodName,
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(RuntimePreparationServer).PreparePackageSet(ctx, req.(*PreparePackageSetRequest))
	}
	return interceptor(ctx, in, info, handler)
}

func _RuntimePreparation_PrepareModelSource_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(PrepareModelSourceRequest)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(RuntimePreparationServer).PrepareModelSource(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: RuntimePreparation_PrepareModelSource_FullMethodName,
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(RuntimePreparationServer).PrepareModelSource(ctx, req.(*PrepareModelSourceRequest))
	}
	return interceptor(ctx, in, info, handler)
}

func _RuntimePreparation_ReleaseModelSource_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(ReleaseModelSourceRequest)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(RuntimePreparationServer).ReleaseModelSource(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: RuntimePreparation_ReleaseModelSource_FullMethodName,
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(RuntimePreparationServer).ReleaseModelSource(ctx, req.(*ReleaseModelSourceRequest))
	}
	return interceptor(ctx, in, info, handler)
}

func _RuntimePreparation_RetainDerivedResult_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(DerivedRetentionRequest)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(RuntimePreparationServer).RetainDerivedResult(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: RuntimePreparation_RetainDerivedResult_FullMethodName,
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(RuntimePreparationServer).RetainDerivedResult(ctx, req.(*DerivedRetentionRequest))
	}
	return interceptor(ctx, in, info, handler)
}

func _RuntimePreparation_ReleaseDerivedRetention_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(DerivedRetentionRequest)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(RuntimePreparationServer).ReleaseDerivedRetention(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: RuntimePreparation_ReleaseDerivedRetention_FullMethodName,
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(RuntimePreparationServer).ReleaseDerivedRetention(ctx, req.(*DerivedRetentionRequest))
	}
	return interceptor(ctx, in, info, handler)
}

func _RuntimePreparation_ReleaseDerivedResult_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(DerivedResultReleaseRequest)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(RuntimePreparationServer).ReleaseDerivedResult(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: RuntimePreparation_ReleaseDerivedResult_FullMethodName,
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(RuntimePreparationServer).ReleaseDerivedResult(ctx, req.(*DerivedResultReleaseRequest))
	}
	return interceptor(ctx, in, info, handler)
}

func _RuntimePreparation_CollectStoreGarbage_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(CollectStoreGarbageRequest)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(RuntimePreparationServer).CollectStoreGarbage(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: RuntimePreparation_CollectStoreGarbage_FullMethodName,
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(RuntimePreparationServer).CollectStoreGarbage(ctx, req.(*CollectStoreGarbageRequest))
	}
	return interceptor(ctx, in, info, handler)
}

func _RuntimePreparation_ValidateWeightsCheckpoint_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(ValidateWeightsCheckpointRequest)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(RuntimePreparationServer).ValidateWeightsCheckpoint(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: RuntimePreparation_ValidateWeightsCheckpoint_FullMethodName,
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(RuntimePreparationServer).ValidateWeightsCheckpoint(ctx, req.(*ValidateWeightsCheckpointRequest))
	}
	return interceptor(ctx, in, info, handler)
}

func _RuntimePreparation_CheckpointPage_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(CheckpointPageRequest)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(RuntimePreparationServer).CheckpointPage(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: RuntimePreparation_CheckpointPage_FullMethodName,
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(RuntimePreparationServer).CheckpointPage(ctx, req.(*CheckpointPageRequest))
	}
	return interceptor(ctx, in, info, handler)
}

func _RuntimePreparation_CheckpointTransfer_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(CheckpointTransferRequest)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(RuntimePreparationServer).CheckpointTransfer(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: RuntimePreparation_CheckpointTransfer_FullMethodName,
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(RuntimePreparationServer).CheckpointTransfer(ctx, req.(*CheckpointTransferRequest))
	}
	return interceptor(ctx, in, info, handler)
}

func _RuntimePreparation_PrepareLocalPackage_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(PrepareLocalPackageRequest)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(RuntimePreparationServer).PrepareLocalPackage(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: RuntimePreparation_PrepareLocalPackage_FullMethodName,
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(RuntimePreparationServer).PrepareLocalPackage(ctx, req.(*PrepareLocalPackageRequest))
	}
	return interceptor(ctx, in, info, handler)
}

func _RuntimePreparation_PreparePrivatePlacement_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(PreparePrivatePlacementRequest)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(RuntimePreparationServer).PreparePrivatePlacement(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: RuntimePreparation_PreparePrivatePlacement_FullMethodName,
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(RuntimePreparationServer).PreparePrivatePlacement(ctx, req.(*PreparePrivatePlacementRequest))
	}
	return interceptor(ctx, in, info, handler)
}

// RuntimePreparation_ServiceDesc is the grpc.ServiceDesc for RuntimePreparation service.
// It's only intended for direct use with grpc.RegisterService,
// and not to be introspected or modified (even as a copy)
var RuntimePreparation_ServiceDesc = grpc.ServiceDesc{
	ServiceName: "cozy.worker.v1.RuntimePreparation",
	HandlerType: (*RuntimePreparationServer)(nil),
	Methods: []grpc.MethodDesc{
		{
			MethodName: "ProtocolInfo",
			Handler:    _RuntimePreparation_ProtocolInfo_Handler,
		},
		{
			MethodName: "NumericalEnvironment",
			Handler:    _RuntimePreparation_NumericalEnvironment_Handler,
		},
		{
			MethodName: "RecordOperationResult",
			Handler:    _RuntimePreparation_RecordOperationResult_Handler,
		},
		{
			MethodName: "LookupOperation",
			Handler:    _RuntimePreparation_LookupOperation_Handler,
		},
		{
			MethodName: "PruneOperationCache",
			Handler:    _RuntimePreparation_PruneOperationCache_Handler,
		},
		{
			MethodName: "WorkspaceRetainDerivedResult",
			Handler:    _RuntimePreparation_WorkspaceRetainDerivedResult_Handler,
		},
		{
			MethodName: "WorkspaceReleaseDerivedRetention",
			Handler:    _RuntimePreparation_WorkspaceReleaseDerivedRetention_Handler,
		},
		{
			MethodName: "WorkspaceReleaseDerivedResult",
			Handler:    _RuntimePreparation_WorkspaceReleaseDerivedResult_Handler,
		},
		{
			MethodName: "WorkspaceRetainByteTree",
			Handler:    _RuntimePreparation_WorkspaceRetainByteTree_Handler,
		},
		{
			MethodName: "WorkspaceReleaseByteTree",
			Handler:    _RuntimePreparation_WorkspaceReleaseByteTree_Handler,
		},
		{
			MethodName: "WorkspaceNativeArtifactTransfer",
			Handler:    _RuntimePreparation_WorkspaceNativeArtifactTransfer_Handler,
		},
		{
			MethodName: "WorkspaceForgetPackage",
			Handler:    _RuntimePreparation_WorkspaceForgetPackage_Handler,
		},
		{
			MethodName: "PreparePackageSet",
			Handler:    _RuntimePreparation_PreparePackageSet_Handler,
		},
		{
			MethodName: "PrepareModelSource",
			Handler:    _RuntimePreparation_PrepareModelSource_Handler,
		},
		{
			MethodName: "ReleaseModelSource",
			Handler:    _RuntimePreparation_ReleaseModelSource_Handler,
		},
		{
			MethodName: "RetainDerivedResult",
			Handler:    _RuntimePreparation_RetainDerivedResult_Handler,
		},
		{
			MethodName: "ReleaseDerivedRetention",
			Handler:    _RuntimePreparation_ReleaseDerivedRetention_Handler,
		},
		{
			MethodName: "ReleaseDerivedResult",
			Handler:    _RuntimePreparation_ReleaseDerivedResult_Handler,
		},
		{
			MethodName: "CollectStoreGarbage",
			Handler:    _RuntimePreparation_CollectStoreGarbage_Handler,
		},
		{
			MethodName: "ValidateWeightsCheckpoint",
			Handler:    _RuntimePreparation_ValidateWeightsCheckpoint_Handler,
		},
		{
			MethodName: "CheckpointPage",
			Handler:    _RuntimePreparation_CheckpointPage_Handler,
		},
		{
			MethodName: "CheckpointTransfer",
			Handler:    _RuntimePreparation_CheckpointTransfer_Handler,
		},
		{
			MethodName: "PrepareLocalPackage",
			Handler:    _RuntimePreparation_PrepareLocalPackage_Handler,
		},
		{
			MethodName: "PreparePrivatePlacement",
			Handler:    _RuntimePreparation_PreparePrivatePlacement_Handler,
		},
	},
	Streams: []grpc.StreamDesc{
		{
			StreamName:    "WorkspaceReadByteTreeObject",
			Handler:       _RuntimePreparation_WorkspaceReadByteTreeObject_Handler,
			ServerStreams: true,
		},
		{
			StreamName:    "ImportInputTree",
			Handler:       _RuntimePreparation_ImportInputTree_Handler,
			ClientStreams: true,
		},
	},
	Metadata: "cozy/worker/v1/worker.proto",
}

const (
	RuntimeWeights_Upload_FullMethodName = "/cozy.worker.v1.RuntimeWeights/Upload"
)

// RuntimeWeightsClient is the client API for RuntimeWeights service.
//
// For semantics around ctx use and closing/ending streaming RPCs, please refer to https://pkg.go.dev/google.golang.org/grpc/?tab=doc#ClientConn.NewStream.
//
// Loopback-only Runtime-hosted weights seam. pod-supervisor is its only client and dials it on
// the same loopback listener as WorkerControl and RuntimePreparation; it is never registered
// on the external listener. Runtime sends custody observations and outcomes on WorkerControl.
type RuntimeWeightsClient interface {
	// One validated upload grant in, one terminal answer out. Runtime streams the bytes it holds.
	Upload(ctx context.Context, in *WeightsUploadRequest, opts ...grpc.CallOption) (*WeightsUploadResult, error)
}

type runtimeWeightsClient struct {
	cc grpc.ClientConnInterface
}

func NewRuntimeWeightsClient(cc grpc.ClientConnInterface) RuntimeWeightsClient {
	return &runtimeWeightsClient{cc}
}

func (c *runtimeWeightsClient) Upload(ctx context.Context, in *WeightsUploadRequest, opts ...grpc.CallOption) (*WeightsUploadResult, error) {
	cOpts := append([]grpc.CallOption{grpc.StaticMethod()}, opts...)
	out := new(WeightsUploadResult)
	err := c.cc.Invoke(ctx, RuntimeWeights_Upload_FullMethodName, in, out, cOpts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

// RuntimeWeightsServer is the server API for RuntimeWeights service.
// All implementations must embed UnimplementedRuntimeWeightsServer
// for forward compatibility.
//
// Loopback-only Runtime-hosted weights seam. pod-supervisor is its only client and dials it on
// the same loopback listener as WorkerControl and RuntimePreparation; it is never registered
// on the external listener. Runtime sends custody observations and outcomes on WorkerControl.
type RuntimeWeightsServer interface {
	// One validated upload grant in, one terminal answer out. Runtime streams the bytes it holds.
	Upload(context.Context, *WeightsUploadRequest) (*WeightsUploadResult, error)
	mustEmbedUnimplementedRuntimeWeightsServer()
}

// UnimplementedRuntimeWeightsServer must be embedded to have
// forward compatible implementations.
//
// NOTE: this should be embedded by value instead of pointer to avoid a nil
// pointer dereference when methods are called.
type UnimplementedRuntimeWeightsServer struct{}

func (UnimplementedRuntimeWeightsServer) Upload(context.Context, *WeightsUploadRequest) (*WeightsUploadResult, error) {
	return nil, status.Error(codes.Unimplemented, "method Upload not implemented")
}
func (UnimplementedRuntimeWeightsServer) mustEmbedUnimplementedRuntimeWeightsServer() {}
func (UnimplementedRuntimeWeightsServer) testEmbeddedByValue()                        {}

// UnsafeRuntimeWeightsServer may be embedded to opt out of forward compatibility for this service.
// Use of this interface is not recommended, as added methods to RuntimeWeightsServer will
// result in compilation errors.
type UnsafeRuntimeWeightsServer interface {
	mustEmbedUnimplementedRuntimeWeightsServer()
}

func RegisterRuntimeWeightsServer(s grpc.ServiceRegistrar, srv RuntimeWeightsServer) {
	// If the following call panics, it indicates UnimplementedRuntimeWeightsServer was
	// embedded by pointer and is nil.  This will cause panics if an
	// unimplemented method is ever invoked, so we test this at initialization
	// time to prevent it from happening at runtime later due to I/O.
	if t, ok := srv.(interface{ testEmbeddedByValue() }); ok {
		t.testEmbeddedByValue()
	}
	s.RegisterService(&RuntimeWeights_ServiceDesc, srv)
}

func _RuntimeWeights_Upload_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(WeightsUploadRequest)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(RuntimeWeightsServer).Upload(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: RuntimeWeights_Upload_FullMethodName,
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(RuntimeWeightsServer).Upload(ctx, req.(*WeightsUploadRequest))
	}
	return interceptor(ctx, in, info, handler)
}

// RuntimeWeights_ServiceDesc is the grpc.ServiceDesc for RuntimeWeights service.
// It's only intended for direct use with grpc.RegisterService,
// and not to be introspected or modified (even as a copy)
var RuntimeWeights_ServiceDesc = grpc.ServiceDesc{
	ServiceName: "cozy.worker.v1.RuntimeWeights",
	HandlerType: (*RuntimeWeightsServer)(nil),
	Methods: []grpc.MethodDesc{
		{
			MethodName: "Upload",
			Handler:    _RuntimeWeights_Upload_Handler,
		},
	},
	Streams:  []grpc.StreamDesc{},
	Metadata: "cozy/worker/v1/worker.proto",
}

const (
	PodHost_KeepRentalAlive_FullMethodName                       = "/cozy.worker.v1.PodHost/KeepRentalAlive"
	PodHost_GetMachineExecutionWorkspace_FullMethodName          = "/cozy.worker.v1.PodHost/GetMachineExecutionWorkspace"
	PodHost_SubmitMachineExecution_FullMethodName                = "/cozy.worker.v1.PodHost/SubmitMachineExecution"
	PodHost_CloseMachineSubmission_FullMethodName                = "/cozy.worker.v1.PodHost/CloseMachineSubmission"
	PodHost_GetMachineExecution_FullMethodName                   = "/cozy.worker.v1.PodHost/GetMachineExecution"
	PodHost_ListMachineExecutionEvents_FullMethodName            = "/cozy.worker.v1.PodHost/ListMachineExecutionEvents"
	PodHost_ControlMachineExecution_FullMethodName               = "/cozy.worker.v1.PodHost/ControlMachineExecution"
	PodHost_CollectMachineExecution_FullMethodName               = "/cozy.worker.v1.PodHost/CollectMachineExecution"
	PodHost_AcknowledgeMachineExecutionCollection_FullMethodName = "/cozy.worker.v1.PodHost/AcknowledgeMachineExecutionCollection"
	PodHost_ReadMachineExecutionTriage_FullMethodName            = "/cozy.worker.v1.PodHost/ReadMachineExecutionTriage"
	PodHost_ProtocolInfo_FullMethodName                          = "/cozy.worker.v1.PodHost/ProtocolInfo"
	PodHost_NumericalEnvironment_FullMethodName                  = "/cozy.worker.v1.PodHost/NumericalEnvironment"
	PodHost_PreparePackageSet_FullMethodName                     = "/cozy.worker.v1.PodHost/PreparePackageSet"
	PodHost_PrepareLocalPackage_FullMethodName                   = "/cozy.worker.v1.PodHost/PrepareLocalPackage"
	PodHost_PreparePrivatePlacement_FullMethodName               = "/cozy.worker.v1.PodHost/PreparePrivatePlacement"
	PodHost_ModelSourceFile_FullMethodName                       = "/cozy.worker.v1.PodHost/ModelSourceFile"
	PodHost_ModelSourcePrepare_FullMethodName                    = "/cozy.worker.v1.PodHost/ModelSourcePrepare"
	PodHost_ModelSourceRelease_FullMethodName                    = "/cozy.worker.v1.PodHost/ModelSourceRelease"
	PodHost_ModelSourceControl_FullMethodName                    = "/cozy.worker.v1.PodHost/ModelSourceControl"
	PodHost_RetainDerivedResult_FullMethodName                   = "/cozy.worker.v1.PodHost/RetainDerivedResult"
	PodHost_ReleaseDerivedRetention_FullMethodName               = "/cozy.worker.v1.PodHost/ReleaseDerivedRetention"
	PodHost_ReleaseDerivedResult_FullMethodName                  = "/cozy.worker.v1.PodHost/ReleaseDerivedResult"
	PodHost_RetainByteTree_FullMethodName                        = "/cozy.worker.v1.PodHost/RetainByteTree"
	PodHost_ReleaseByteTree_FullMethodName                       = "/cozy.worker.v1.PodHost/ReleaseByteTree"
	PodHost_ReadByteTreeObject_FullMethodName                    = "/cozy.worker.v1.PodHost/ReadByteTreeObject"
	PodHost_NativeArtifactTransfer_FullMethodName                = "/cozy.worker.v1.PodHost/NativeArtifactTransfer"
	PodHost_ForgetPackage_FullMethodName                         = "/cozy.worker.v1.PodHost/ForgetPackage"
	PodHost_ImportInputTree_FullMethodName                       = "/cozy.worker.v1.PodHost/ImportInputTree"
	PodHost_RecordOperationResult_FullMethodName                 = "/cozy.worker.v1.PodHost/RecordOperationResult"
	PodHost_LookupOperation_FullMethodName                       = "/cozy.worker.v1.PodHost/LookupOperation"
	PodHost_PruneOperationCache_FullMethodName                   = "/cozy.worker.v1.PodHost/PruneOperationCache"
	PodHost_ModelSourceAdopt_FullMethodName                      = "/cozy.worker.v1.PodHost/ModelSourceAdopt"
	PodHost_CheckpointPage_FullMethodName                        = "/cozy.worker.v1.PodHost/CheckpointPage"
	PodHost_CheckpointTransfer_FullMethodName                    = "/cozy.worker.v1.PodHost/CheckpointTransfer"
	PodHost_LocalPackageUpload_FullMethodName                    = "/cozy.worker.v1.PodHost/LocalPackageUpload"
	PodHost_ListMachineExecutions_FullMethodName                 = "/cozy.worker.v1.PodHost/ListMachineExecutions"
	PodHost_ListPackages_FullMethodName                          = "/cozy.worker.v1.PodHost/ListPackages"
	PodHost_ListModels_FullMethodName                            = "/cozy.worker.v1.PodHost/ListModels"
	PodHost_DescribeMachine_FullMethodName                       = "/cozy.worker.v1.PodHost/DescribeMachine"
)

// PodHostClient is the client API for PodHost service.
//
// For semantics around ctx use and closing/ending streaming RPCs, please refer to https://pkg.go.dev/google.golang.org/grpc/?tab=doc#ClientConn.NewStream.
//
// ---------------------------------------------------------------------------
// PodHost -- the RecordOwner <-> pod HOST lane. A second service on the pod's external
// listener, served by pod-supervisor beside WorkerControl. It carries what the host does on the
// owner's behalf and nothing that is control: the owner asks the host to PREPARE (download,
// verify, obtain the PlacementSet bytes from the Runtime's loopback preparation), then sends
// those exact bytes ITSELF as DesiredWorkerState{placement_set} on WorkerControl. Remote and
// local are the same three steps -- prepare, placement_set bytes, desired_state -- and a
// multi-GiB materialization never sits on the control stream's read loop, where before 21 it
// head-of-line blocked every offer, ack, and cancel behind it.
//
// AUTHORITY: every call opens with the owner's Claim -- the same message and the same Ed25519
// proof over ClaimProof/1 the owner presents on WorkerControl -- which the host verifies against
// the same control key the Runtime verifies it against. A call whose record_owner_epoch is
// older than the highest Claim the host has forwarded is refused. A host call is scoped to the
// worker lifetime the proof names, never to one control stream: control_stream_epoch is 0 on
// the Claim and on every request carried here, and nonzero refuses. The request's
// record_owner_epoch and worker_boot_id MUST equal the Claim's.
//
// REPLAY: each lane is durable in the host ledger and idempotent on its identity fields. A
// reconnecting owner re-issues the exact call and receives the ledger's answer (the same
// terminal status, or REPLAYED) or attaches to the operation still in flight. Progress is the
// owner's observed signal; no lane needs a wall-clock guess.
//
// A local machine runs the same Host and serves the same PodHost; only the dial address differs.
// ---------------------------------------------------------------------------
type PodHostClient interface {
	// Explicit owner action only. Connection traffic and status reads do not renew idle time.
	KeepRentalAlive(ctx context.Context, in *KeepRentalAliveRequest, opts ...grpc.CallOption) (*KeepRentalAliveResult, error)
	GetMachineExecutionWorkspace(ctx context.Context, in *MachineExecutionWorkspaceQuery, opts ...grpc.CallOption) (*MachineExecutionWorkspace, error)
	// Authenticated forwarding only. Runtime is the sole execution journal/owner.
	SubmitMachineExecution(ctx context.Context, in *MachineExecutionSubmit, opts ...grpc.CallOption) (*MachineExecutionReceipt, error)
	// Atomically close a submission key against delayed acceptance, returning any existing receipt.
	CloseMachineSubmission(ctx context.Context, in *MachineSubmissionClose, opts ...grpc.CallOption) (*MachineSubmissionClosure, error)
	GetMachineExecution(ctx context.Context, in *MachineExecutionQuery, opts ...grpc.CallOption) (*MachineExecutionState, error)
	ListMachineExecutionEvents(ctx context.Context, in *MachineExecutionEventsQuery, opts ...grpc.CallOption) (*MachineExecutionEventPage, error)
	ControlMachineExecution(ctx context.Context, in *MachineExecutionControl, opts ...grpc.CallOption) (*MachineExecutionState, error)
	CollectMachineExecution(ctx context.Context, in *MachineExecutionCollect, opts ...grpc.CallOption) (*AttemptOutcome, error)
	AcknowledgeMachineExecutionCollection(ctx context.Context, in *MachineExecutionCollectionAck, opts ...grpc.CallOption) (*MachineExecutionState, error)
	// Authenticated forwarding only; Runtime holds the bundle bytes.
	ReadMachineExecutionTriage(ctx context.Context, in *MachineExecutionTriageQuery, opts ...grpc.CallOption) (*MachineExecutionTriage, error)
	// Static, read-only compatibility probe over the existing pinned TLS connection.
	// The Host forwards its actual Runtime's loopback result; it does not guess a version.
	ProtocolInfo(ctx context.Context, in *ProtocolInfoRequest, opts ...grpc.CallOption) (*ProtocolInfoResult, error)
	NumericalEnvironment(ctx context.Context, in *NumericalEnvironmentCall, opts ...grpc.CallOption) (*NumericalEnvironmentResult, error)
	PreparePackageSet(ctx context.Context, in *PreparePackageSetCall, opts ...grpc.CallOption) (grpc.ServerStreamingClient[PrepareEvent], error)
	PrepareLocalPackage(ctx context.Context, in *PrepareLocalPackageCall, opts ...grpc.CallOption) (grpc.ServerStreamingClient[PrepareEvent], error)
	PreparePrivatePlacement(ctx context.Context, in *PreparePrivatePlacementCall, opts ...grpc.CallOption) (grpc.ServerStreamingClient[PrepareEvent], error)
	ModelSourceFile(ctx context.Context, in *ModelSourceFileCall, opts ...grpc.CallOption) (grpc.ServerStreamingClient[ModelSourceFileStatus], error)
	ModelSourcePrepare(ctx context.Context, in *ModelSourcePrepareCall, opts ...grpc.CallOption) (*ModelSourcePrepared, error)
	ModelSourceRelease(ctx context.Context, in *ModelSourceReleaseCall, opts ...grpc.CallOption) (*ReleaseModelSourceResult, error)
	ModelSourceControl(ctx context.Context, in *ModelSourceControlCall, opts ...grpc.CallOption) (*ModelSourceControlResult, error)
	RetainDerivedResult(ctx context.Context, in *DerivedRetentionCall, opts ...grpc.CallOption) (*DerivedRetentionResult, error)
	ReleaseDerivedRetention(ctx context.Context, in *DerivedRetentionCall, opts ...grpc.CallOption) (*DerivedRetentionResult, error)
	ReleaseDerivedResult(ctx context.Context, in *DerivedResultReleaseCall, opts ...grpc.CallOption) (*DerivedResultReleaseResult, error)
	RetainByteTree(ctx context.Context, in *NativeByteRetentionCall, opts ...grpc.CallOption) (*NativeByteRetentionResult, error)
	ReleaseByteTree(ctx context.Context, in *NativeByteRetentionCall, opts ...grpc.CallOption) (*NativeByteRetentionResult, error)
	ReadByteTreeObject(ctx context.Context, in *NativeByteReadCall, opts ...grpc.CallOption) (grpc.ServerStreamingClient[NativeByteReadChunk], error)
	// A retained output's closure and granted object uploads, as `cozy run upload` sends them.
	NativeArtifactTransfer(ctx context.Context, in *NativeArtifactTransferCall, opts ...grpc.CallOption) (*NativeArtifactTransferStatus, error)
	// `cozy package bind`, `unbind` or `publish` changed the package: the machine reads it once more.
	ForgetPackage(ctx context.Context, in *ForgetPackageCall, opts ...grpc.CallOption) (*ForgetPackageResult, error)
	// Proxies the bounded stream to Runtime without owning its byte custody.
	ImportInputTree(ctx context.Context, opts ...grpc.CallOption) (grpc.ClientStreamingClient[InputTreeImportFrame, NativeByteRetentionResult], error)
	RecordOperationResult(ctx context.Context, in *RecordOperationResultCall, opts ...grpc.CallOption) (*RecordOperationResultResult, error)
	LookupOperation(ctx context.Context, in *LookupOperationCall, opts ...grpc.CallOption) (*LookupOperationResult, error)
	PruneOperationCache(ctx context.Context, in *PruneOperationCacheCall, opts ...grpc.CallOption) (*PruneOperationCacheResult, error)
	ModelSourceAdopt(ctx context.Context, in *ModelSourceAdoptCall, opts ...grpc.CallOption) (*ModelSourcePrepared, error)
	CheckpointPage(ctx context.Context, in *CheckpointPageCall, opts ...grpc.CallOption) (*CheckpointPageResult, error)
	CheckpointTransfer(ctx context.Context, in *CheckpointTransferCall, opts ...grpc.CallOption) (*CheckpointTransferStatus, error)
	LocalPackageUpload(ctx context.Context, opts ...grpc.CallOption) (grpc.BidiStreamingClient[LocalPackageUploadFrame, LocalPackageFileStatus], error)
	// Wire 66: the machine's truth, read directly. The Host forwards ListMachineExecutions and
	// ListPackages; it answers ListModels from its TensorFS store and DescribeMachine with its own
	// facts beside its Runtime's, so both work while the Runtime is stopped. A Host before 66
	// answers UNIMPLEMENTED; a Runtime before 66 leaves only the reads it serves unanswered.
	ListMachineExecutions(ctx context.Context, in *MachineExecutionListQuery, opts ...grpc.CallOption) (*MachineExecutionList, error)
	ListPackages(ctx context.Context, in *PackageListQuery, opts ...grpc.CallOption) (*PackageList, error)
	ListModels(ctx context.Context, in *ModelListQuery, opts ...grpc.CallOption) (*ModelList, error)
	DescribeMachine(ctx context.Context, in *DescribeMachineQuery, opts ...grpc.CallOption) (*MachineDescription, error)
}

type podHostClient struct {
	cc grpc.ClientConnInterface
}

func NewPodHostClient(cc grpc.ClientConnInterface) PodHostClient {
	return &podHostClient{cc}
}

func (c *podHostClient) KeepRentalAlive(ctx context.Context, in *KeepRentalAliveRequest, opts ...grpc.CallOption) (*KeepRentalAliveResult, error) {
	cOpts := append([]grpc.CallOption{grpc.StaticMethod()}, opts...)
	out := new(KeepRentalAliveResult)
	err := c.cc.Invoke(ctx, PodHost_KeepRentalAlive_FullMethodName, in, out, cOpts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (c *podHostClient) GetMachineExecutionWorkspace(ctx context.Context, in *MachineExecutionWorkspaceQuery, opts ...grpc.CallOption) (*MachineExecutionWorkspace, error) {
	cOpts := append([]grpc.CallOption{grpc.StaticMethod()}, opts...)
	out := new(MachineExecutionWorkspace)
	err := c.cc.Invoke(ctx, PodHost_GetMachineExecutionWorkspace_FullMethodName, in, out, cOpts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (c *podHostClient) SubmitMachineExecution(ctx context.Context, in *MachineExecutionSubmit, opts ...grpc.CallOption) (*MachineExecutionReceipt, error) {
	cOpts := append([]grpc.CallOption{grpc.StaticMethod()}, opts...)
	out := new(MachineExecutionReceipt)
	err := c.cc.Invoke(ctx, PodHost_SubmitMachineExecution_FullMethodName, in, out, cOpts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (c *podHostClient) CloseMachineSubmission(ctx context.Context, in *MachineSubmissionClose, opts ...grpc.CallOption) (*MachineSubmissionClosure, error) {
	cOpts := append([]grpc.CallOption{grpc.StaticMethod()}, opts...)
	out := new(MachineSubmissionClosure)
	err := c.cc.Invoke(ctx, PodHost_CloseMachineSubmission_FullMethodName, in, out, cOpts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (c *podHostClient) GetMachineExecution(ctx context.Context, in *MachineExecutionQuery, opts ...grpc.CallOption) (*MachineExecutionState, error) {
	cOpts := append([]grpc.CallOption{grpc.StaticMethod()}, opts...)
	out := new(MachineExecutionState)
	err := c.cc.Invoke(ctx, PodHost_GetMachineExecution_FullMethodName, in, out, cOpts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (c *podHostClient) ListMachineExecutionEvents(ctx context.Context, in *MachineExecutionEventsQuery, opts ...grpc.CallOption) (*MachineExecutionEventPage, error) {
	cOpts := append([]grpc.CallOption{grpc.StaticMethod()}, opts...)
	out := new(MachineExecutionEventPage)
	err := c.cc.Invoke(ctx, PodHost_ListMachineExecutionEvents_FullMethodName, in, out, cOpts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (c *podHostClient) ControlMachineExecution(ctx context.Context, in *MachineExecutionControl, opts ...grpc.CallOption) (*MachineExecutionState, error) {
	cOpts := append([]grpc.CallOption{grpc.StaticMethod()}, opts...)
	out := new(MachineExecutionState)
	err := c.cc.Invoke(ctx, PodHost_ControlMachineExecution_FullMethodName, in, out, cOpts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (c *podHostClient) CollectMachineExecution(ctx context.Context, in *MachineExecutionCollect, opts ...grpc.CallOption) (*AttemptOutcome, error) {
	cOpts := append([]grpc.CallOption{grpc.StaticMethod()}, opts...)
	out := new(AttemptOutcome)
	err := c.cc.Invoke(ctx, PodHost_CollectMachineExecution_FullMethodName, in, out, cOpts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (c *podHostClient) AcknowledgeMachineExecutionCollection(ctx context.Context, in *MachineExecutionCollectionAck, opts ...grpc.CallOption) (*MachineExecutionState, error) {
	cOpts := append([]grpc.CallOption{grpc.StaticMethod()}, opts...)
	out := new(MachineExecutionState)
	err := c.cc.Invoke(ctx, PodHost_AcknowledgeMachineExecutionCollection_FullMethodName, in, out, cOpts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (c *podHostClient) ReadMachineExecutionTriage(ctx context.Context, in *MachineExecutionTriageQuery, opts ...grpc.CallOption) (*MachineExecutionTriage, error) {
	cOpts := append([]grpc.CallOption{grpc.StaticMethod()}, opts...)
	out := new(MachineExecutionTriage)
	err := c.cc.Invoke(ctx, PodHost_ReadMachineExecutionTriage_FullMethodName, in, out, cOpts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (c *podHostClient) ProtocolInfo(ctx context.Context, in *ProtocolInfoRequest, opts ...grpc.CallOption) (*ProtocolInfoResult, error) {
	cOpts := append([]grpc.CallOption{grpc.StaticMethod()}, opts...)
	out := new(ProtocolInfoResult)
	err := c.cc.Invoke(ctx, PodHost_ProtocolInfo_FullMethodName, in, out, cOpts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (c *podHostClient) NumericalEnvironment(ctx context.Context, in *NumericalEnvironmentCall, opts ...grpc.CallOption) (*NumericalEnvironmentResult, error) {
	cOpts := append([]grpc.CallOption{grpc.StaticMethod()}, opts...)
	out := new(NumericalEnvironmentResult)
	err := c.cc.Invoke(ctx, PodHost_NumericalEnvironment_FullMethodName, in, out, cOpts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (c *podHostClient) PreparePackageSet(ctx context.Context, in *PreparePackageSetCall, opts ...grpc.CallOption) (grpc.ServerStreamingClient[PrepareEvent], error) {
	cOpts := append([]grpc.CallOption{grpc.StaticMethod()}, opts...)
	stream, err := c.cc.NewStream(ctx, &PodHost_ServiceDesc.Streams[0], PodHost_PreparePackageSet_FullMethodName, cOpts...)
	if err != nil {
		return nil, err
	}
	x := &grpc.GenericClientStream[PreparePackageSetCall, PrepareEvent]{ClientStream: stream}
	if err := x.ClientStream.SendMsg(in); err != nil {
		return nil, err
	}
	if err := x.ClientStream.CloseSend(); err != nil {
		return nil, err
	}
	return x, nil
}

// This type alias is provided for backwards compatibility with existing code that references the prior non-generic stream type by name.
type PodHost_PreparePackageSetClient = grpc.ServerStreamingClient[PrepareEvent]

func (c *podHostClient) PrepareLocalPackage(ctx context.Context, in *PrepareLocalPackageCall, opts ...grpc.CallOption) (grpc.ServerStreamingClient[PrepareEvent], error) {
	cOpts := append([]grpc.CallOption{grpc.StaticMethod()}, opts...)
	stream, err := c.cc.NewStream(ctx, &PodHost_ServiceDesc.Streams[1], PodHost_PrepareLocalPackage_FullMethodName, cOpts...)
	if err != nil {
		return nil, err
	}
	x := &grpc.GenericClientStream[PrepareLocalPackageCall, PrepareEvent]{ClientStream: stream}
	if err := x.ClientStream.SendMsg(in); err != nil {
		return nil, err
	}
	if err := x.ClientStream.CloseSend(); err != nil {
		return nil, err
	}
	return x, nil
}

// This type alias is provided for backwards compatibility with existing code that references the prior non-generic stream type by name.
type PodHost_PrepareLocalPackageClient = grpc.ServerStreamingClient[PrepareEvent]

func (c *podHostClient) PreparePrivatePlacement(ctx context.Context, in *PreparePrivatePlacementCall, opts ...grpc.CallOption) (grpc.ServerStreamingClient[PrepareEvent], error) {
	cOpts := append([]grpc.CallOption{grpc.StaticMethod()}, opts...)
	stream, err := c.cc.NewStream(ctx, &PodHost_ServiceDesc.Streams[2], PodHost_PreparePrivatePlacement_FullMethodName, cOpts...)
	if err != nil {
		return nil, err
	}
	x := &grpc.GenericClientStream[PreparePrivatePlacementCall, PrepareEvent]{ClientStream: stream}
	if err := x.ClientStream.SendMsg(in); err != nil {
		return nil, err
	}
	if err := x.ClientStream.CloseSend(); err != nil {
		return nil, err
	}
	return x, nil
}

// This type alias is provided for backwards compatibility with existing code that references the prior non-generic stream type by name.
type PodHost_PreparePrivatePlacementClient = grpc.ServerStreamingClient[PrepareEvent]

func (c *podHostClient) ModelSourceFile(ctx context.Context, in *ModelSourceFileCall, opts ...grpc.CallOption) (grpc.ServerStreamingClient[ModelSourceFileStatus], error) {
	cOpts := append([]grpc.CallOption{grpc.StaticMethod()}, opts...)
	stream, err := c.cc.NewStream(ctx, &PodHost_ServiceDesc.Streams[3], PodHost_ModelSourceFile_FullMethodName, cOpts...)
	if err != nil {
		return nil, err
	}
	x := &grpc.GenericClientStream[ModelSourceFileCall, ModelSourceFileStatus]{ClientStream: stream}
	if err := x.ClientStream.SendMsg(in); err != nil {
		return nil, err
	}
	if err := x.ClientStream.CloseSend(); err != nil {
		return nil, err
	}
	return x, nil
}

// This type alias is provided for backwards compatibility with existing code that references the prior non-generic stream type by name.
type PodHost_ModelSourceFileClient = grpc.ServerStreamingClient[ModelSourceFileStatus]

func (c *podHostClient) ModelSourcePrepare(ctx context.Context, in *ModelSourcePrepareCall, opts ...grpc.CallOption) (*ModelSourcePrepared, error) {
	cOpts := append([]grpc.CallOption{grpc.StaticMethod()}, opts...)
	out := new(ModelSourcePrepared)
	err := c.cc.Invoke(ctx, PodHost_ModelSourcePrepare_FullMethodName, in, out, cOpts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (c *podHostClient) ModelSourceRelease(ctx context.Context, in *ModelSourceReleaseCall, opts ...grpc.CallOption) (*ReleaseModelSourceResult, error) {
	cOpts := append([]grpc.CallOption{grpc.StaticMethod()}, opts...)
	out := new(ReleaseModelSourceResult)
	err := c.cc.Invoke(ctx, PodHost_ModelSourceRelease_FullMethodName, in, out, cOpts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (c *podHostClient) ModelSourceControl(ctx context.Context, in *ModelSourceControlCall, opts ...grpc.CallOption) (*ModelSourceControlResult, error) {
	cOpts := append([]grpc.CallOption{grpc.StaticMethod()}, opts...)
	out := new(ModelSourceControlResult)
	err := c.cc.Invoke(ctx, PodHost_ModelSourceControl_FullMethodName, in, out, cOpts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (c *podHostClient) RetainDerivedResult(ctx context.Context, in *DerivedRetentionCall, opts ...grpc.CallOption) (*DerivedRetentionResult, error) {
	cOpts := append([]grpc.CallOption{grpc.StaticMethod()}, opts...)
	out := new(DerivedRetentionResult)
	err := c.cc.Invoke(ctx, PodHost_RetainDerivedResult_FullMethodName, in, out, cOpts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (c *podHostClient) ReleaseDerivedRetention(ctx context.Context, in *DerivedRetentionCall, opts ...grpc.CallOption) (*DerivedRetentionResult, error) {
	cOpts := append([]grpc.CallOption{grpc.StaticMethod()}, opts...)
	out := new(DerivedRetentionResult)
	err := c.cc.Invoke(ctx, PodHost_ReleaseDerivedRetention_FullMethodName, in, out, cOpts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (c *podHostClient) ReleaseDerivedResult(ctx context.Context, in *DerivedResultReleaseCall, opts ...grpc.CallOption) (*DerivedResultReleaseResult, error) {
	cOpts := append([]grpc.CallOption{grpc.StaticMethod()}, opts...)
	out := new(DerivedResultReleaseResult)
	err := c.cc.Invoke(ctx, PodHost_ReleaseDerivedResult_FullMethodName, in, out, cOpts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (c *podHostClient) RetainByteTree(ctx context.Context, in *NativeByteRetentionCall, opts ...grpc.CallOption) (*NativeByteRetentionResult, error) {
	cOpts := append([]grpc.CallOption{grpc.StaticMethod()}, opts...)
	out := new(NativeByteRetentionResult)
	err := c.cc.Invoke(ctx, PodHost_RetainByteTree_FullMethodName, in, out, cOpts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (c *podHostClient) ReleaseByteTree(ctx context.Context, in *NativeByteRetentionCall, opts ...grpc.CallOption) (*NativeByteRetentionResult, error) {
	cOpts := append([]grpc.CallOption{grpc.StaticMethod()}, opts...)
	out := new(NativeByteRetentionResult)
	err := c.cc.Invoke(ctx, PodHost_ReleaseByteTree_FullMethodName, in, out, cOpts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (c *podHostClient) ReadByteTreeObject(ctx context.Context, in *NativeByteReadCall, opts ...grpc.CallOption) (grpc.ServerStreamingClient[NativeByteReadChunk], error) {
	cOpts := append([]grpc.CallOption{grpc.StaticMethod()}, opts...)
	stream, err := c.cc.NewStream(ctx, &PodHost_ServiceDesc.Streams[4], PodHost_ReadByteTreeObject_FullMethodName, cOpts...)
	if err != nil {
		return nil, err
	}
	x := &grpc.GenericClientStream[NativeByteReadCall, NativeByteReadChunk]{ClientStream: stream}
	if err := x.ClientStream.SendMsg(in); err != nil {
		return nil, err
	}
	if err := x.ClientStream.CloseSend(); err != nil {
		return nil, err
	}
	return x, nil
}

// This type alias is provided for backwards compatibility with existing code that references the prior non-generic stream type by name.
type PodHost_ReadByteTreeObjectClient = grpc.ServerStreamingClient[NativeByteReadChunk]

func (c *podHostClient) NativeArtifactTransfer(ctx context.Context, in *NativeArtifactTransferCall, opts ...grpc.CallOption) (*NativeArtifactTransferStatus, error) {
	cOpts := append([]grpc.CallOption{grpc.StaticMethod()}, opts...)
	out := new(NativeArtifactTransferStatus)
	err := c.cc.Invoke(ctx, PodHost_NativeArtifactTransfer_FullMethodName, in, out, cOpts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (c *podHostClient) ForgetPackage(ctx context.Context, in *ForgetPackageCall, opts ...grpc.CallOption) (*ForgetPackageResult, error) {
	cOpts := append([]grpc.CallOption{grpc.StaticMethod()}, opts...)
	out := new(ForgetPackageResult)
	err := c.cc.Invoke(ctx, PodHost_ForgetPackage_FullMethodName, in, out, cOpts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (c *podHostClient) ImportInputTree(ctx context.Context, opts ...grpc.CallOption) (grpc.ClientStreamingClient[InputTreeImportFrame, NativeByteRetentionResult], error) {
	cOpts := append([]grpc.CallOption{grpc.StaticMethod()}, opts...)
	stream, err := c.cc.NewStream(ctx, &PodHost_ServiceDesc.Streams[5], PodHost_ImportInputTree_FullMethodName, cOpts...)
	if err != nil {
		return nil, err
	}
	x := &grpc.GenericClientStream[InputTreeImportFrame, NativeByteRetentionResult]{ClientStream: stream}
	return x, nil
}

// This type alias is provided for backwards compatibility with existing code that references the prior non-generic stream type by name.
type PodHost_ImportInputTreeClient = grpc.ClientStreamingClient[InputTreeImportFrame, NativeByteRetentionResult]

func (c *podHostClient) RecordOperationResult(ctx context.Context, in *RecordOperationResultCall, opts ...grpc.CallOption) (*RecordOperationResultResult, error) {
	cOpts := append([]grpc.CallOption{grpc.StaticMethod()}, opts...)
	out := new(RecordOperationResultResult)
	err := c.cc.Invoke(ctx, PodHost_RecordOperationResult_FullMethodName, in, out, cOpts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (c *podHostClient) LookupOperation(ctx context.Context, in *LookupOperationCall, opts ...grpc.CallOption) (*LookupOperationResult, error) {
	cOpts := append([]grpc.CallOption{grpc.StaticMethod()}, opts...)
	out := new(LookupOperationResult)
	err := c.cc.Invoke(ctx, PodHost_LookupOperation_FullMethodName, in, out, cOpts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (c *podHostClient) PruneOperationCache(ctx context.Context, in *PruneOperationCacheCall, opts ...grpc.CallOption) (*PruneOperationCacheResult, error) {
	cOpts := append([]grpc.CallOption{grpc.StaticMethod()}, opts...)
	out := new(PruneOperationCacheResult)
	err := c.cc.Invoke(ctx, PodHost_PruneOperationCache_FullMethodName, in, out, cOpts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (c *podHostClient) ModelSourceAdopt(ctx context.Context, in *ModelSourceAdoptCall, opts ...grpc.CallOption) (*ModelSourcePrepared, error) {
	cOpts := append([]grpc.CallOption{grpc.StaticMethod()}, opts...)
	out := new(ModelSourcePrepared)
	err := c.cc.Invoke(ctx, PodHost_ModelSourceAdopt_FullMethodName, in, out, cOpts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (c *podHostClient) CheckpointPage(ctx context.Context, in *CheckpointPageCall, opts ...grpc.CallOption) (*CheckpointPageResult, error) {
	cOpts := append([]grpc.CallOption{grpc.StaticMethod()}, opts...)
	out := new(CheckpointPageResult)
	err := c.cc.Invoke(ctx, PodHost_CheckpointPage_FullMethodName, in, out, cOpts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (c *podHostClient) CheckpointTransfer(ctx context.Context, in *CheckpointTransferCall, opts ...grpc.CallOption) (*CheckpointTransferStatus, error) {
	cOpts := append([]grpc.CallOption{grpc.StaticMethod()}, opts...)
	out := new(CheckpointTransferStatus)
	err := c.cc.Invoke(ctx, PodHost_CheckpointTransfer_FullMethodName, in, out, cOpts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (c *podHostClient) LocalPackageUpload(ctx context.Context, opts ...grpc.CallOption) (grpc.BidiStreamingClient[LocalPackageUploadFrame, LocalPackageFileStatus], error) {
	cOpts := append([]grpc.CallOption{grpc.StaticMethod()}, opts...)
	stream, err := c.cc.NewStream(ctx, &PodHost_ServiceDesc.Streams[6], PodHost_LocalPackageUpload_FullMethodName, cOpts...)
	if err != nil {
		return nil, err
	}
	x := &grpc.GenericClientStream[LocalPackageUploadFrame, LocalPackageFileStatus]{ClientStream: stream}
	return x, nil
}

// This type alias is provided for backwards compatibility with existing code that references the prior non-generic stream type by name.
type PodHost_LocalPackageUploadClient = grpc.BidiStreamingClient[LocalPackageUploadFrame, LocalPackageFileStatus]

func (c *podHostClient) ListMachineExecutions(ctx context.Context, in *MachineExecutionListQuery, opts ...grpc.CallOption) (*MachineExecutionList, error) {
	cOpts := append([]grpc.CallOption{grpc.StaticMethod()}, opts...)
	out := new(MachineExecutionList)
	err := c.cc.Invoke(ctx, PodHost_ListMachineExecutions_FullMethodName, in, out, cOpts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (c *podHostClient) ListPackages(ctx context.Context, in *PackageListQuery, opts ...grpc.CallOption) (*PackageList, error) {
	cOpts := append([]grpc.CallOption{grpc.StaticMethod()}, opts...)
	out := new(PackageList)
	err := c.cc.Invoke(ctx, PodHost_ListPackages_FullMethodName, in, out, cOpts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (c *podHostClient) ListModels(ctx context.Context, in *ModelListQuery, opts ...grpc.CallOption) (*ModelList, error) {
	cOpts := append([]grpc.CallOption{grpc.StaticMethod()}, opts...)
	out := new(ModelList)
	err := c.cc.Invoke(ctx, PodHost_ListModels_FullMethodName, in, out, cOpts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (c *podHostClient) DescribeMachine(ctx context.Context, in *DescribeMachineQuery, opts ...grpc.CallOption) (*MachineDescription, error) {
	cOpts := append([]grpc.CallOption{grpc.StaticMethod()}, opts...)
	out := new(MachineDescription)
	err := c.cc.Invoke(ctx, PodHost_DescribeMachine_FullMethodName, in, out, cOpts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

// PodHostServer is the server API for PodHost service.
// All implementations must embed UnimplementedPodHostServer
// for forward compatibility.
//
// ---------------------------------------------------------------------------
// PodHost -- the RecordOwner <-> pod HOST lane. A second service on the pod's external
// listener, served by pod-supervisor beside WorkerControl. It carries what the host does on the
// owner's behalf and nothing that is control: the owner asks the host to PREPARE (download,
// verify, obtain the PlacementSet bytes from the Runtime's loopback preparation), then sends
// those exact bytes ITSELF as DesiredWorkerState{placement_set} on WorkerControl. Remote and
// local are the same three steps -- prepare, placement_set bytes, desired_state -- and a
// multi-GiB materialization never sits on the control stream's read loop, where before 21 it
// head-of-line blocked every offer, ack, and cancel behind it.
//
// AUTHORITY: every call opens with the owner's Claim -- the same message and the same Ed25519
// proof over ClaimProof/1 the owner presents on WorkerControl -- which the host verifies against
// the same control key the Runtime verifies it against. A call whose record_owner_epoch is
// older than the highest Claim the host has forwarded is refused. A host call is scoped to the
// worker lifetime the proof names, never to one control stream: control_stream_epoch is 0 on
// the Claim and on every request carried here, and nonzero refuses. The request's
// record_owner_epoch and worker_boot_id MUST equal the Claim's.
//
// REPLAY: each lane is durable in the host ledger and idempotent on its identity fields. A
// reconnecting owner re-issues the exact call and receives the ledger's answer (the same
// terminal status, or REPLAYED) or attaches to the operation still in flight. Progress is the
// owner's observed signal; no lane needs a wall-clock guess.
//
// A local machine runs the same Host and serves the same PodHost; only the dial address differs.
// ---------------------------------------------------------------------------
type PodHostServer interface {
	// Explicit owner action only. Connection traffic and status reads do not renew idle time.
	KeepRentalAlive(context.Context, *KeepRentalAliveRequest) (*KeepRentalAliveResult, error)
	GetMachineExecutionWorkspace(context.Context, *MachineExecutionWorkspaceQuery) (*MachineExecutionWorkspace, error)
	// Authenticated forwarding only. Runtime is the sole execution journal/owner.
	SubmitMachineExecution(context.Context, *MachineExecutionSubmit) (*MachineExecutionReceipt, error)
	// Atomically close a submission key against delayed acceptance, returning any existing receipt.
	CloseMachineSubmission(context.Context, *MachineSubmissionClose) (*MachineSubmissionClosure, error)
	GetMachineExecution(context.Context, *MachineExecutionQuery) (*MachineExecutionState, error)
	ListMachineExecutionEvents(context.Context, *MachineExecutionEventsQuery) (*MachineExecutionEventPage, error)
	ControlMachineExecution(context.Context, *MachineExecutionControl) (*MachineExecutionState, error)
	CollectMachineExecution(context.Context, *MachineExecutionCollect) (*AttemptOutcome, error)
	AcknowledgeMachineExecutionCollection(context.Context, *MachineExecutionCollectionAck) (*MachineExecutionState, error)
	// Authenticated forwarding only; Runtime holds the bundle bytes.
	ReadMachineExecutionTriage(context.Context, *MachineExecutionTriageQuery) (*MachineExecutionTriage, error)
	// Static, read-only compatibility probe over the existing pinned TLS connection.
	// The Host forwards its actual Runtime's loopback result; it does not guess a version.
	ProtocolInfo(context.Context, *ProtocolInfoRequest) (*ProtocolInfoResult, error)
	NumericalEnvironment(context.Context, *NumericalEnvironmentCall) (*NumericalEnvironmentResult, error)
	PreparePackageSet(*PreparePackageSetCall, grpc.ServerStreamingServer[PrepareEvent]) error
	PrepareLocalPackage(*PrepareLocalPackageCall, grpc.ServerStreamingServer[PrepareEvent]) error
	PreparePrivatePlacement(*PreparePrivatePlacementCall, grpc.ServerStreamingServer[PrepareEvent]) error
	ModelSourceFile(*ModelSourceFileCall, grpc.ServerStreamingServer[ModelSourceFileStatus]) error
	ModelSourcePrepare(context.Context, *ModelSourcePrepareCall) (*ModelSourcePrepared, error)
	ModelSourceRelease(context.Context, *ModelSourceReleaseCall) (*ReleaseModelSourceResult, error)
	ModelSourceControl(context.Context, *ModelSourceControlCall) (*ModelSourceControlResult, error)
	RetainDerivedResult(context.Context, *DerivedRetentionCall) (*DerivedRetentionResult, error)
	ReleaseDerivedRetention(context.Context, *DerivedRetentionCall) (*DerivedRetentionResult, error)
	ReleaseDerivedResult(context.Context, *DerivedResultReleaseCall) (*DerivedResultReleaseResult, error)
	RetainByteTree(context.Context, *NativeByteRetentionCall) (*NativeByteRetentionResult, error)
	ReleaseByteTree(context.Context, *NativeByteRetentionCall) (*NativeByteRetentionResult, error)
	ReadByteTreeObject(*NativeByteReadCall, grpc.ServerStreamingServer[NativeByteReadChunk]) error
	// A retained output's closure and granted object uploads, as `cozy run upload` sends them.
	NativeArtifactTransfer(context.Context, *NativeArtifactTransferCall) (*NativeArtifactTransferStatus, error)
	// `cozy package bind`, `unbind` or `publish` changed the package: the machine reads it once more.
	ForgetPackage(context.Context, *ForgetPackageCall) (*ForgetPackageResult, error)
	// Proxies the bounded stream to Runtime without owning its byte custody.
	ImportInputTree(grpc.ClientStreamingServer[InputTreeImportFrame, NativeByteRetentionResult]) error
	RecordOperationResult(context.Context, *RecordOperationResultCall) (*RecordOperationResultResult, error)
	LookupOperation(context.Context, *LookupOperationCall) (*LookupOperationResult, error)
	PruneOperationCache(context.Context, *PruneOperationCacheCall) (*PruneOperationCacheResult, error)
	ModelSourceAdopt(context.Context, *ModelSourceAdoptCall) (*ModelSourcePrepared, error)
	CheckpointPage(context.Context, *CheckpointPageCall) (*CheckpointPageResult, error)
	CheckpointTransfer(context.Context, *CheckpointTransferCall) (*CheckpointTransferStatus, error)
	LocalPackageUpload(grpc.BidiStreamingServer[LocalPackageUploadFrame, LocalPackageFileStatus]) error
	// Wire 66: the machine's truth, read directly. The Host forwards ListMachineExecutions and
	// ListPackages; it answers ListModels from its TensorFS store and DescribeMachine with its own
	// facts beside its Runtime's, so both work while the Runtime is stopped. A Host before 66
	// answers UNIMPLEMENTED; a Runtime before 66 leaves only the reads it serves unanswered.
	ListMachineExecutions(context.Context, *MachineExecutionListQuery) (*MachineExecutionList, error)
	ListPackages(context.Context, *PackageListQuery) (*PackageList, error)
	ListModels(context.Context, *ModelListQuery) (*ModelList, error)
	DescribeMachine(context.Context, *DescribeMachineQuery) (*MachineDescription, error)
	mustEmbedUnimplementedPodHostServer()
}

// UnimplementedPodHostServer must be embedded to have
// forward compatible implementations.
//
// NOTE: this should be embedded by value instead of pointer to avoid a nil
// pointer dereference when methods are called.
type UnimplementedPodHostServer struct{}

func (UnimplementedPodHostServer) KeepRentalAlive(context.Context, *KeepRentalAliveRequest) (*KeepRentalAliveResult, error) {
	return nil, status.Error(codes.Unimplemented, "method KeepRentalAlive not implemented")
}
func (UnimplementedPodHostServer) GetMachineExecutionWorkspace(context.Context, *MachineExecutionWorkspaceQuery) (*MachineExecutionWorkspace, error) {
	return nil, status.Error(codes.Unimplemented, "method GetMachineExecutionWorkspace not implemented")
}
func (UnimplementedPodHostServer) SubmitMachineExecution(context.Context, *MachineExecutionSubmit) (*MachineExecutionReceipt, error) {
	return nil, status.Error(codes.Unimplemented, "method SubmitMachineExecution not implemented")
}
func (UnimplementedPodHostServer) CloseMachineSubmission(context.Context, *MachineSubmissionClose) (*MachineSubmissionClosure, error) {
	return nil, status.Error(codes.Unimplemented, "method CloseMachineSubmission not implemented")
}
func (UnimplementedPodHostServer) GetMachineExecution(context.Context, *MachineExecutionQuery) (*MachineExecutionState, error) {
	return nil, status.Error(codes.Unimplemented, "method GetMachineExecution not implemented")
}
func (UnimplementedPodHostServer) ListMachineExecutionEvents(context.Context, *MachineExecutionEventsQuery) (*MachineExecutionEventPage, error) {
	return nil, status.Error(codes.Unimplemented, "method ListMachineExecutionEvents not implemented")
}
func (UnimplementedPodHostServer) ControlMachineExecution(context.Context, *MachineExecutionControl) (*MachineExecutionState, error) {
	return nil, status.Error(codes.Unimplemented, "method ControlMachineExecution not implemented")
}
func (UnimplementedPodHostServer) CollectMachineExecution(context.Context, *MachineExecutionCollect) (*AttemptOutcome, error) {
	return nil, status.Error(codes.Unimplemented, "method CollectMachineExecution not implemented")
}
func (UnimplementedPodHostServer) AcknowledgeMachineExecutionCollection(context.Context, *MachineExecutionCollectionAck) (*MachineExecutionState, error) {
	return nil, status.Error(codes.Unimplemented, "method AcknowledgeMachineExecutionCollection not implemented")
}
func (UnimplementedPodHostServer) ReadMachineExecutionTriage(context.Context, *MachineExecutionTriageQuery) (*MachineExecutionTriage, error) {
	return nil, status.Error(codes.Unimplemented, "method ReadMachineExecutionTriage not implemented")
}
func (UnimplementedPodHostServer) ProtocolInfo(context.Context, *ProtocolInfoRequest) (*ProtocolInfoResult, error) {
	return nil, status.Error(codes.Unimplemented, "method ProtocolInfo not implemented")
}
func (UnimplementedPodHostServer) NumericalEnvironment(context.Context, *NumericalEnvironmentCall) (*NumericalEnvironmentResult, error) {
	return nil, status.Error(codes.Unimplemented, "method NumericalEnvironment not implemented")
}
func (UnimplementedPodHostServer) PreparePackageSet(*PreparePackageSetCall, grpc.ServerStreamingServer[PrepareEvent]) error {
	return status.Error(codes.Unimplemented, "method PreparePackageSet not implemented")
}
func (UnimplementedPodHostServer) PrepareLocalPackage(*PrepareLocalPackageCall, grpc.ServerStreamingServer[PrepareEvent]) error {
	return status.Error(codes.Unimplemented, "method PrepareLocalPackage not implemented")
}
func (UnimplementedPodHostServer) PreparePrivatePlacement(*PreparePrivatePlacementCall, grpc.ServerStreamingServer[PrepareEvent]) error {
	return status.Error(codes.Unimplemented, "method PreparePrivatePlacement not implemented")
}
func (UnimplementedPodHostServer) ModelSourceFile(*ModelSourceFileCall, grpc.ServerStreamingServer[ModelSourceFileStatus]) error {
	return status.Error(codes.Unimplemented, "method ModelSourceFile not implemented")
}
func (UnimplementedPodHostServer) ModelSourcePrepare(context.Context, *ModelSourcePrepareCall) (*ModelSourcePrepared, error) {
	return nil, status.Error(codes.Unimplemented, "method ModelSourcePrepare not implemented")
}
func (UnimplementedPodHostServer) ModelSourceRelease(context.Context, *ModelSourceReleaseCall) (*ReleaseModelSourceResult, error) {
	return nil, status.Error(codes.Unimplemented, "method ModelSourceRelease not implemented")
}
func (UnimplementedPodHostServer) ModelSourceControl(context.Context, *ModelSourceControlCall) (*ModelSourceControlResult, error) {
	return nil, status.Error(codes.Unimplemented, "method ModelSourceControl not implemented")
}
func (UnimplementedPodHostServer) RetainDerivedResult(context.Context, *DerivedRetentionCall) (*DerivedRetentionResult, error) {
	return nil, status.Error(codes.Unimplemented, "method RetainDerivedResult not implemented")
}
func (UnimplementedPodHostServer) ReleaseDerivedRetention(context.Context, *DerivedRetentionCall) (*DerivedRetentionResult, error) {
	return nil, status.Error(codes.Unimplemented, "method ReleaseDerivedRetention not implemented")
}
func (UnimplementedPodHostServer) ReleaseDerivedResult(context.Context, *DerivedResultReleaseCall) (*DerivedResultReleaseResult, error) {
	return nil, status.Error(codes.Unimplemented, "method ReleaseDerivedResult not implemented")
}
func (UnimplementedPodHostServer) RetainByteTree(context.Context, *NativeByteRetentionCall) (*NativeByteRetentionResult, error) {
	return nil, status.Error(codes.Unimplemented, "method RetainByteTree not implemented")
}
func (UnimplementedPodHostServer) ReleaseByteTree(context.Context, *NativeByteRetentionCall) (*NativeByteRetentionResult, error) {
	return nil, status.Error(codes.Unimplemented, "method ReleaseByteTree not implemented")
}
func (UnimplementedPodHostServer) ReadByteTreeObject(*NativeByteReadCall, grpc.ServerStreamingServer[NativeByteReadChunk]) error {
	return status.Error(codes.Unimplemented, "method ReadByteTreeObject not implemented")
}
func (UnimplementedPodHostServer) NativeArtifactTransfer(context.Context, *NativeArtifactTransferCall) (*NativeArtifactTransferStatus, error) {
	return nil, status.Error(codes.Unimplemented, "method NativeArtifactTransfer not implemented")
}
func (UnimplementedPodHostServer) ForgetPackage(context.Context, *ForgetPackageCall) (*ForgetPackageResult, error) {
	return nil, status.Error(codes.Unimplemented, "method ForgetPackage not implemented")
}
func (UnimplementedPodHostServer) ImportInputTree(grpc.ClientStreamingServer[InputTreeImportFrame, NativeByteRetentionResult]) error {
	return status.Error(codes.Unimplemented, "method ImportInputTree not implemented")
}
func (UnimplementedPodHostServer) RecordOperationResult(context.Context, *RecordOperationResultCall) (*RecordOperationResultResult, error) {
	return nil, status.Error(codes.Unimplemented, "method RecordOperationResult not implemented")
}
func (UnimplementedPodHostServer) LookupOperation(context.Context, *LookupOperationCall) (*LookupOperationResult, error) {
	return nil, status.Error(codes.Unimplemented, "method LookupOperation not implemented")
}
func (UnimplementedPodHostServer) PruneOperationCache(context.Context, *PruneOperationCacheCall) (*PruneOperationCacheResult, error) {
	return nil, status.Error(codes.Unimplemented, "method PruneOperationCache not implemented")
}
func (UnimplementedPodHostServer) ModelSourceAdopt(context.Context, *ModelSourceAdoptCall) (*ModelSourcePrepared, error) {
	return nil, status.Error(codes.Unimplemented, "method ModelSourceAdopt not implemented")
}
func (UnimplementedPodHostServer) CheckpointPage(context.Context, *CheckpointPageCall) (*CheckpointPageResult, error) {
	return nil, status.Error(codes.Unimplemented, "method CheckpointPage not implemented")
}
func (UnimplementedPodHostServer) CheckpointTransfer(context.Context, *CheckpointTransferCall) (*CheckpointTransferStatus, error) {
	return nil, status.Error(codes.Unimplemented, "method CheckpointTransfer not implemented")
}
func (UnimplementedPodHostServer) LocalPackageUpload(grpc.BidiStreamingServer[LocalPackageUploadFrame, LocalPackageFileStatus]) error {
	return status.Error(codes.Unimplemented, "method LocalPackageUpload not implemented")
}
func (UnimplementedPodHostServer) ListMachineExecutions(context.Context, *MachineExecutionListQuery) (*MachineExecutionList, error) {
	return nil, status.Error(codes.Unimplemented, "method ListMachineExecutions not implemented")
}
func (UnimplementedPodHostServer) ListPackages(context.Context, *PackageListQuery) (*PackageList, error) {
	return nil, status.Error(codes.Unimplemented, "method ListPackages not implemented")
}
func (UnimplementedPodHostServer) ListModels(context.Context, *ModelListQuery) (*ModelList, error) {
	return nil, status.Error(codes.Unimplemented, "method ListModels not implemented")
}
func (UnimplementedPodHostServer) DescribeMachine(context.Context, *DescribeMachineQuery) (*MachineDescription, error) {
	return nil, status.Error(codes.Unimplemented, "method DescribeMachine not implemented")
}
func (UnimplementedPodHostServer) mustEmbedUnimplementedPodHostServer() {}
func (UnimplementedPodHostServer) testEmbeddedByValue()                 {}

// UnsafePodHostServer may be embedded to opt out of forward compatibility for this service.
// Use of this interface is not recommended, as added methods to PodHostServer will
// result in compilation errors.
type UnsafePodHostServer interface {
	mustEmbedUnimplementedPodHostServer()
}

func RegisterPodHostServer(s grpc.ServiceRegistrar, srv PodHostServer) {
	// If the following call panics, it indicates UnimplementedPodHostServer was
	// embedded by pointer and is nil.  This will cause panics if an
	// unimplemented method is ever invoked, so we test this at initialization
	// time to prevent it from happening at runtime later due to I/O.
	if t, ok := srv.(interface{ testEmbeddedByValue() }); ok {
		t.testEmbeddedByValue()
	}
	s.RegisterService(&PodHost_ServiceDesc, srv)
}

func _PodHost_KeepRentalAlive_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(KeepRentalAliveRequest)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(PodHostServer).KeepRentalAlive(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: PodHost_KeepRentalAlive_FullMethodName,
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(PodHostServer).KeepRentalAlive(ctx, req.(*KeepRentalAliveRequest))
	}
	return interceptor(ctx, in, info, handler)
}

func _PodHost_GetMachineExecutionWorkspace_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(MachineExecutionWorkspaceQuery)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(PodHostServer).GetMachineExecutionWorkspace(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: PodHost_GetMachineExecutionWorkspace_FullMethodName,
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(PodHostServer).GetMachineExecutionWorkspace(ctx, req.(*MachineExecutionWorkspaceQuery))
	}
	return interceptor(ctx, in, info, handler)
}

func _PodHost_SubmitMachineExecution_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(MachineExecutionSubmit)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(PodHostServer).SubmitMachineExecution(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: PodHost_SubmitMachineExecution_FullMethodName,
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(PodHostServer).SubmitMachineExecution(ctx, req.(*MachineExecutionSubmit))
	}
	return interceptor(ctx, in, info, handler)
}

func _PodHost_CloseMachineSubmission_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(MachineSubmissionClose)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(PodHostServer).CloseMachineSubmission(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: PodHost_CloseMachineSubmission_FullMethodName,
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(PodHostServer).CloseMachineSubmission(ctx, req.(*MachineSubmissionClose))
	}
	return interceptor(ctx, in, info, handler)
}

func _PodHost_GetMachineExecution_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(MachineExecutionQuery)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(PodHostServer).GetMachineExecution(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: PodHost_GetMachineExecution_FullMethodName,
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(PodHostServer).GetMachineExecution(ctx, req.(*MachineExecutionQuery))
	}
	return interceptor(ctx, in, info, handler)
}

func _PodHost_ListMachineExecutionEvents_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(MachineExecutionEventsQuery)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(PodHostServer).ListMachineExecutionEvents(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: PodHost_ListMachineExecutionEvents_FullMethodName,
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(PodHostServer).ListMachineExecutionEvents(ctx, req.(*MachineExecutionEventsQuery))
	}
	return interceptor(ctx, in, info, handler)
}

func _PodHost_ControlMachineExecution_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(MachineExecutionControl)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(PodHostServer).ControlMachineExecution(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: PodHost_ControlMachineExecution_FullMethodName,
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(PodHostServer).ControlMachineExecution(ctx, req.(*MachineExecutionControl))
	}
	return interceptor(ctx, in, info, handler)
}

func _PodHost_CollectMachineExecution_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(MachineExecutionCollect)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(PodHostServer).CollectMachineExecution(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: PodHost_CollectMachineExecution_FullMethodName,
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(PodHostServer).CollectMachineExecution(ctx, req.(*MachineExecutionCollect))
	}
	return interceptor(ctx, in, info, handler)
}

func _PodHost_AcknowledgeMachineExecutionCollection_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(MachineExecutionCollectionAck)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(PodHostServer).AcknowledgeMachineExecutionCollection(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: PodHost_AcknowledgeMachineExecutionCollection_FullMethodName,
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(PodHostServer).AcknowledgeMachineExecutionCollection(ctx, req.(*MachineExecutionCollectionAck))
	}
	return interceptor(ctx, in, info, handler)
}

func _PodHost_ReadMachineExecutionTriage_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(MachineExecutionTriageQuery)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(PodHostServer).ReadMachineExecutionTriage(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: PodHost_ReadMachineExecutionTriage_FullMethodName,
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(PodHostServer).ReadMachineExecutionTriage(ctx, req.(*MachineExecutionTriageQuery))
	}
	return interceptor(ctx, in, info, handler)
}

func _PodHost_ProtocolInfo_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(ProtocolInfoRequest)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(PodHostServer).ProtocolInfo(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: PodHost_ProtocolInfo_FullMethodName,
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(PodHostServer).ProtocolInfo(ctx, req.(*ProtocolInfoRequest))
	}
	return interceptor(ctx, in, info, handler)
}

func _PodHost_NumericalEnvironment_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(NumericalEnvironmentCall)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(PodHostServer).NumericalEnvironment(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: PodHost_NumericalEnvironment_FullMethodName,
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(PodHostServer).NumericalEnvironment(ctx, req.(*NumericalEnvironmentCall))
	}
	return interceptor(ctx, in, info, handler)
}

func _PodHost_PreparePackageSet_Handler(srv interface{}, stream grpc.ServerStream) error {
	m := new(PreparePackageSetCall)
	if err := stream.RecvMsg(m); err != nil {
		return err
	}
	return srv.(PodHostServer).PreparePackageSet(m, &grpc.GenericServerStream[PreparePackageSetCall, PrepareEvent]{ServerStream: stream})
}

// This type alias is provided for backwards compatibility with existing code that references the prior non-generic stream type by name.
type PodHost_PreparePackageSetServer = grpc.ServerStreamingServer[PrepareEvent]

func _PodHost_PrepareLocalPackage_Handler(srv interface{}, stream grpc.ServerStream) error {
	m := new(PrepareLocalPackageCall)
	if err := stream.RecvMsg(m); err != nil {
		return err
	}
	return srv.(PodHostServer).PrepareLocalPackage(m, &grpc.GenericServerStream[PrepareLocalPackageCall, PrepareEvent]{ServerStream: stream})
}

// This type alias is provided for backwards compatibility with existing code that references the prior non-generic stream type by name.
type PodHost_PrepareLocalPackageServer = grpc.ServerStreamingServer[PrepareEvent]

func _PodHost_PreparePrivatePlacement_Handler(srv interface{}, stream grpc.ServerStream) error {
	m := new(PreparePrivatePlacementCall)
	if err := stream.RecvMsg(m); err != nil {
		return err
	}
	return srv.(PodHostServer).PreparePrivatePlacement(m, &grpc.GenericServerStream[PreparePrivatePlacementCall, PrepareEvent]{ServerStream: stream})
}

// This type alias is provided for backwards compatibility with existing code that references the prior non-generic stream type by name.
type PodHost_PreparePrivatePlacementServer = grpc.ServerStreamingServer[PrepareEvent]

func _PodHost_ModelSourceFile_Handler(srv interface{}, stream grpc.ServerStream) error {
	m := new(ModelSourceFileCall)
	if err := stream.RecvMsg(m); err != nil {
		return err
	}
	return srv.(PodHostServer).ModelSourceFile(m, &grpc.GenericServerStream[ModelSourceFileCall, ModelSourceFileStatus]{ServerStream: stream})
}

// This type alias is provided for backwards compatibility with existing code that references the prior non-generic stream type by name.
type PodHost_ModelSourceFileServer = grpc.ServerStreamingServer[ModelSourceFileStatus]

func _PodHost_ModelSourcePrepare_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(ModelSourcePrepareCall)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(PodHostServer).ModelSourcePrepare(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: PodHost_ModelSourcePrepare_FullMethodName,
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(PodHostServer).ModelSourcePrepare(ctx, req.(*ModelSourcePrepareCall))
	}
	return interceptor(ctx, in, info, handler)
}

func _PodHost_ModelSourceRelease_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(ModelSourceReleaseCall)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(PodHostServer).ModelSourceRelease(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: PodHost_ModelSourceRelease_FullMethodName,
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(PodHostServer).ModelSourceRelease(ctx, req.(*ModelSourceReleaseCall))
	}
	return interceptor(ctx, in, info, handler)
}

func _PodHost_ModelSourceControl_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(ModelSourceControlCall)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(PodHostServer).ModelSourceControl(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: PodHost_ModelSourceControl_FullMethodName,
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(PodHostServer).ModelSourceControl(ctx, req.(*ModelSourceControlCall))
	}
	return interceptor(ctx, in, info, handler)
}

func _PodHost_RetainDerivedResult_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(DerivedRetentionCall)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(PodHostServer).RetainDerivedResult(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: PodHost_RetainDerivedResult_FullMethodName,
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(PodHostServer).RetainDerivedResult(ctx, req.(*DerivedRetentionCall))
	}
	return interceptor(ctx, in, info, handler)
}

func _PodHost_ReleaseDerivedRetention_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(DerivedRetentionCall)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(PodHostServer).ReleaseDerivedRetention(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: PodHost_ReleaseDerivedRetention_FullMethodName,
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(PodHostServer).ReleaseDerivedRetention(ctx, req.(*DerivedRetentionCall))
	}
	return interceptor(ctx, in, info, handler)
}

func _PodHost_ReleaseDerivedResult_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(DerivedResultReleaseCall)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(PodHostServer).ReleaseDerivedResult(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: PodHost_ReleaseDerivedResult_FullMethodName,
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(PodHostServer).ReleaseDerivedResult(ctx, req.(*DerivedResultReleaseCall))
	}
	return interceptor(ctx, in, info, handler)
}

func _PodHost_RetainByteTree_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(NativeByteRetentionCall)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(PodHostServer).RetainByteTree(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: PodHost_RetainByteTree_FullMethodName,
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(PodHostServer).RetainByteTree(ctx, req.(*NativeByteRetentionCall))
	}
	return interceptor(ctx, in, info, handler)
}

func _PodHost_ReleaseByteTree_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(NativeByteRetentionCall)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(PodHostServer).ReleaseByteTree(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: PodHost_ReleaseByteTree_FullMethodName,
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(PodHostServer).ReleaseByteTree(ctx, req.(*NativeByteRetentionCall))
	}
	return interceptor(ctx, in, info, handler)
}

func _PodHost_ReadByteTreeObject_Handler(srv interface{}, stream grpc.ServerStream) error {
	m := new(NativeByteReadCall)
	if err := stream.RecvMsg(m); err != nil {
		return err
	}
	return srv.(PodHostServer).ReadByteTreeObject(m, &grpc.GenericServerStream[NativeByteReadCall, NativeByteReadChunk]{ServerStream: stream})
}

// This type alias is provided for backwards compatibility with existing code that references the prior non-generic stream type by name.
type PodHost_ReadByteTreeObjectServer = grpc.ServerStreamingServer[NativeByteReadChunk]

func _PodHost_NativeArtifactTransfer_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(NativeArtifactTransferCall)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(PodHostServer).NativeArtifactTransfer(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: PodHost_NativeArtifactTransfer_FullMethodName,
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(PodHostServer).NativeArtifactTransfer(ctx, req.(*NativeArtifactTransferCall))
	}
	return interceptor(ctx, in, info, handler)
}

func _PodHost_ForgetPackage_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(ForgetPackageCall)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(PodHostServer).ForgetPackage(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: PodHost_ForgetPackage_FullMethodName,
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(PodHostServer).ForgetPackage(ctx, req.(*ForgetPackageCall))
	}
	return interceptor(ctx, in, info, handler)
}

func _PodHost_ImportInputTree_Handler(srv interface{}, stream grpc.ServerStream) error {
	return srv.(PodHostServer).ImportInputTree(&grpc.GenericServerStream[InputTreeImportFrame, NativeByteRetentionResult]{ServerStream: stream})
}

// This type alias is provided for backwards compatibility with existing code that references the prior non-generic stream type by name.
type PodHost_ImportInputTreeServer = grpc.ClientStreamingServer[InputTreeImportFrame, NativeByteRetentionResult]

func _PodHost_RecordOperationResult_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(RecordOperationResultCall)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(PodHostServer).RecordOperationResult(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: PodHost_RecordOperationResult_FullMethodName,
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(PodHostServer).RecordOperationResult(ctx, req.(*RecordOperationResultCall))
	}
	return interceptor(ctx, in, info, handler)
}

func _PodHost_LookupOperation_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(LookupOperationCall)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(PodHostServer).LookupOperation(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: PodHost_LookupOperation_FullMethodName,
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(PodHostServer).LookupOperation(ctx, req.(*LookupOperationCall))
	}
	return interceptor(ctx, in, info, handler)
}

func _PodHost_PruneOperationCache_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(PruneOperationCacheCall)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(PodHostServer).PruneOperationCache(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: PodHost_PruneOperationCache_FullMethodName,
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(PodHostServer).PruneOperationCache(ctx, req.(*PruneOperationCacheCall))
	}
	return interceptor(ctx, in, info, handler)
}

func _PodHost_ModelSourceAdopt_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(ModelSourceAdoptCall)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(PodHostServer).ModelSourceAdopt(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: PodHost_ModelSourceAdopt_FullMethodName,
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(PodHostServer).ModelSourceAdopt(ctx, req.(*ModelSourceAdoptCall))
	}
	return interceptor(ctx, in, info, handler)
}

func _PodHost_CheckpointPage_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(CheckpointPageCall)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(PodHostServer).CheckpointPage(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: PodHost_CheckpointPage_FullMethodName,
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(PodHostServer).CheckpointPage(ctx, req.(*CheckpointPageCall))
	}
	return interceptor(ctx, in, info, handler)
}

func _PodHost_CheckpointTransfer_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(CheckpointTransferCall)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(PodHostServer).CheckpointTransfer(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: PodHost_CheckpointTransfer_FullMethodName,
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(PodHostServer).CheckpointTransfer(ctx, req.(*CheckpointTransferCall))
	}
	return interceptor(ctx, in, info, handler)
}

func _PodHost_LocalPackageUpload_Handler(srv interface{}, stream grpc.ServerStream) error {
	return srv.(PodHostServer).LocalPackageUpload(&grpc.GenericServerStream[LocalPackageUploadFrame, LocalPackageFileStatus]{ServerStream: stream})
}

// This type alias is provided for backwards compatibility with existing code that references the prior non-generic stream type by name.
type PodHost_LocalPackageUploadServer = grpc.BidiStreamingServer[LocalPackageUploadFrame, LocalPackageFileStatus]

func _PodHost_ListMachineExecutions_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(MachineExecutionListQuery)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(PodHostServer).ListMachineExecutions(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: PodHost_ListMachineExecutions_FullMethodName,
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(PodHostServer).ListMachineExecutions(ctx, req.(*MachineExecutionListQuery))
	}
	return interceptor(ctx, in, info, handler)
}

func _PodHost_ListPackages_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(PackageListQuery)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(PodHostServer).ListPackages(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: PodHost_ListPackages_FullMethodName,
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(PodHostServer).ListPackages(ctx, req.(*PackageListQuery))
	}
	return interceptor(ctx, in, info, handler)
}

func _PodHost_ListModels_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(ModelListQuery)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(PodHostServer).ListModels(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: PodHost_ListModels_FullMethodName,
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(PodHostServer).ListModels(ctx, req.(*ModelListQuery))
	}
	return interceptor(ctx, in, info, handler)
}

func _PodHost_DescribeMachine_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(DescribeMachineQuery)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(PodHostServer).DescribeMachine(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: PodHost_DescribeMachine_FullMethodName,
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(PodHostServer).DescribeMachine(ctx, req.(*DescribeMachineQuery))
	}
	return interceptor(ctx, in, info, handler)
}

// PodHost_ServiceDesc is the grpc.ServiceDesc for PodHost service.
// It's only intended for direct use with grpc.RegisterService,
// and not to be introspected or modified (even as a copy)
var PodHost_ServiceDesc = grpc.ServiceDesc{
	ServiceName: "cozy.worker.v1.PodHost",
	HandlerType: (*PodHostServer)(nil),
	Methods: []grpc.MethodDesc{
		{
			MethodName: "KeepRentalAlive",
			Handler:    _PodHost_KeepRentalAlive_Handler,
		},
		{
			MethodName: "GetMachineExecutionWorkspace",
			Handler:    _PodHost_GetMachineExecutionWorkspace_Handler,
		},
		{
			MethodName: "SubmitMachineExecution",
			Handler:    _PodHost_SubmitMachineExecution_Handler,
		},
		{
			MethodName: "CloseMachineSubmission",
			Handler:    _PodHost_CloseMachineSubmission_Handler,
		},
		{
			MethodName: "GetMachineExecution",
			Handler:    _PodHost_GetMachineExecution_Handler,
		},
		{
			MethodName: "ListMachineExecutionEvents",
			Handler:    _PodHost_ListMachineExecutionEvents_Handler,
		},
		{
			MethodName: "ControlMachineExecution",
			Handler:    _PodHost_ControlMachineExecution_Handler,
		},
		{
			MethodName: "CollectMachineExecution",
			Handler:    _PodHost_CollectMachineExecution_Handler,
		},
		{
			MethodName: "AcknowledgeMachineExecutionCollection",
			Handler:    _PodHost_AcknowledgeMachineExecutionCollection_Handler,
		},
		{
			MethodName: "ReadMachineExecutionTriage",
			Handler:    _PodHost_ReadMachineExecutionTriage_Handler,
		},
		{
			MethodName: "ProtocolInfo",
			Handler:    _PodHost_ProtocolInfo_Handler,
		},
		{
			MethodName: "NumericalEnvironment",
			Handler:    _PodHost_NumericalEnvironment_Handler,
		},
		{
			MethodName: "ModelSourcePrepare",
			Handler:    _PodHost_ModelSourcePrepare_Handler,
		},
		{
			MethodName: "ModelSourceRelease",
			Handler:    _PodHost_ModelSourceRelease_Handler,
		},
		{
			MethodName: "ModelSourceControl",
			Handler:    _PodHost_ModelSourceControl_Handler,
		},
		{
			MethodName: "RetainDerivedResult",
			Handler:    _PodHost_RetainDerivedResult_Handler,
		},
		{
			MethodName: "ReleaseDerivedRetention",
			Handler:    _PodHost_ReleaseDerivedRetention_Handler,
		},
		{
			MethodName: "ReleaseDerivedResult",
			Handler:    _PodHost_ReleaseDerivedResult_Handler,
		},
		{
			MethodName: "RetainByteTree",
			Handler:    _PodHost_RetainByteTree_Handler,
		},
		{
			MethodName: "ReleaseByteTree",
			Handler:    _PodHost_ReleaseByteTree_Handler,
		},
		{
			MethodName: "NativeArtifactTransfer",
			Handler:    _PodHost_NativeArtifactTransfer_Handler,
		},
		{
			MethodName: "ForgetPackage",
			Handler:    _PodHost_ForgetPackage_Handler,
		},
		{
			MethodName: "RecordOperationResult",
			Handler:    _PodHost_RecordOperationResult_Handler,
		},
		{
			MethodName: "LookupOperation",
			Handler:    _PodHost_LookupOperation_Handler,
		},
		{
			MethodName: "PruneOperationCache",
			Handler:    _PodHost_PruneOperationCache_Handler,
		},
		{
			MethodName: "ModelSourceAdopt",
			Handler:    _PodHost_ModelSourceAdopt_Handler,
		},
		{
			MethodName: "CheckpointPage",
			Handler:    _PodHost_CheckpointPage_Handler,
		},
		{
			MethodName: "CheckpointTransfer",
			Handler:    _PodHost_CheckpointTransfer_Handler,
		},
		{
			MethodName: "ListMachineExecutions",
			Handler:    _PodHost_ListMachineExecutions_Handler,
		},
		{
			MethodName: "ListPackages",
			Handler:    _PodHost_ListPackages_Handler,
		},
		{
			MethodName: "ListModels",
			Handler:    _PodHost_ListModels_Handler,
		},
		{
			MethodName: "DescribeMachine",
			Handler:    _PodHost_DescribeMachine_Handler,
		},
	},
	Streams: []grpc.StreamDesc{
		{
			StreamName:    "PreparePackageSet",
			Handler:       _PodHost_PreparePackageSet_Handler,
			ServerStreams: true,
		},
		{
			StreamName:    "PrepareLocalPackage",
			Handler:       _PodHost_PrepareLocalPackage_Handler,
			ServerStreams: true,
		},
		{
			StreamName:    "PreparePrivatePlacement",
			Handler:       _PodHost_PreparePrivatePlacement_Handler,
			ServerStreams: true,
		},
		{
			StreamName:    "ModelSourceFile",
			Handler:       _PodHost_ModelSourceFile_Handler,
			ServerStreams: true,
		},
		{
			StreamName:    "ReadByteTreeObject",
			Handler:       _PodHost_ReadByteTreeObject_Handler,
			ServerStreams: true,
		},
		{
			StreamName:    "ImportInputTree",
			Handler:       _PodHost_ImportInputTree_Handler,
			ClientStreams: true,
		},
		{
			StreamName:    "LocalPackageUpload",
			Handler:       _PodHost_LocalPackageUpload_Handler,
			ServerStreams: true,
			ClientStreams: true,
		},
	},
	Metadata: "cozy/worker/v1/worker.proto",
}
