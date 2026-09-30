# Disabled event-owned HOST transfer experiment

`tiers.HostTransfer` is a mechanism only. No Planner, fill path or executor calls it.
Public memory work remains reserved for 0.18.91; immutable R17 .90dev is unchanged.

A caller supplies a copy stream, explicit destination allowance, generation/eligibility
validator, executor-lifetime custody set, and an exclusive already-admitted pinned buffer
lease. The lease's full capacity must hold one whole unit up front; no buffer grows, no
allocator cache is treated as a grant, and insufficient/unpinned capacity declines. A later
provider may borrow the retained ring introduced separately by PR997, but no borrower or
second pool is implemented here. Admission and numerical policy stay outside tiers.

Preparation holds original source references and snapshots, preserves one GPU allocation
per original storage group, stages raw bytes across pinned slots and enqueues copies.
It does not rebind a module or execute a forward. Actual demand joins copy-ready on CPU,
revalidates source/generation, and atomically publishes views with original dtype, shape,
stride and offset. Failed publication restores attempted bindings. Mutable/changed sources,
gradient operations, unsupported layouts and insufficient capacity do not acquire authority.

Pinned borrows return only after copy completion. Published GPU backing remains owned
until `finish`, called after the final consumer submission, records a consumer event and
conservatively drains the device. A pure callable does not prove exclusive current-stream
use. Homing is allowed only after that drain; the mechanism does not home or change a
Planner state itself. Returned tensor views still own their real storage independently.

Failed drain/rollback retains all uncertain buffers/backings in the caller's custody set
and raises device_fault. The owner must mark the generation unusable and retain custody
until verified process retirement; it must not clear that set on timeout or socket EOF.
Successful abort drains before releasing, without claiming RSS dropped when a borrow ends.

Guarded CPU tests use real tensors for values and alias geometry and fake CUDA events for
delayed copy/consumer/failure traces. They prove bookkeeping, not GPU performance or pinned
DMA. The detailed design and measured ceilings are in the benchmark output's
generic-overlap-design/host-transfer/DESIGN.md. Any Planner lookahead or enablement needs
separate source review and GPU-owner qualification with unchanged numerical semantics.
