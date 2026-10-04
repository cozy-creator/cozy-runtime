# H3 embedding under a constrained weight budget

Implementation and qualification: [tracker #320](https://github.com/cozy-creator/tracker/issues/320).

The actual MiniMax H3 text conditioner keeps its 151936 × 5120 token embedding in
bf16. That table is 1.449 GiB. Module refinement leaves it as one indivisible leaf,
so it cannot occupy a weight budget below its size.

Only this plain H3 embedding is eligible for selected-row execution. The normal
coarse/resident route stays unchanged. If fine-grain planning still reaches
`BelowFloor`, the table leaves the paid residency set and cursor order. Its ordinary
module forward reads requested rows from the machine's already verified ObjectFiles
descriptors. The reader checks complete source coverage, duplicates its own handles,
and consumes the ReadPlan's exact object offsets. It never reads optional host-tier
holes or opens a store.

One pageable row buffer is bounded by the current host grant and capped at 1 MiB.
With no optional pinned-tier grant, one row is the maximum buffer. Source values
copy directly to the original device output; no complete host or GPU table is made.
Index order, duplicates, output shape, stored bf16 values and requested output dtype
are preserved. The actual H3 module must have `max_norm=None`; mutating embedding
semantics and encoded embedding storage are outside this implementation.

The reader reports actual touched source bytes and its temporary host peak through
the plane modes. Its H2D bytes and copy duration join the existing process copy
counters. A new stage can return to normal table placement when the budget permits.

Eight tiny CPU tests use real read-only descriptors and prove exact bf16/f16 lookup,
split-object rows, repeated/shaped/empty indices, descriptor lifetime, dtype conversion,
coverage refusal, bad indices and closed-reader refusal. This is operator evidence.
It does not establish the full H3 workflow fits in 1 GiB: downstream activations
and other kernels have their own requirements.

Required rental gates:

1. Exact H3-shape embedding output with a 1 GiB CUDA allocator ceiling, bounded host
   buffer and no full table device allocation (diagnostic gate).
2. Actual machine file-backed ObjectFiles grants and real plane refinement selecting
   the degraded path. Record selected-row mode, touched bytes and host/GPU peaks.
3. Original ordinary-CLI H3 conditioning request and a roomier control, with identical
   output values. Report any later activation or recovery-contract blocker separately.
4. Warm transition from constrained to sufficient budget restores normal placement.

The safe OOM-recovery and idle-component changes in Runtime #1159 are required before
combining this path with constrained inference. Whole-forward and collective replay
are not a substitute for a state-preserving allocation recovery contract.

The actual degraded text component streams one slot so speculative double buffering
cannot consume an unknown first-pass activation peak. The remaining H3 bf16 MLP leaf
is250MiB (25600x5120), which fits one slot at1GiB beside measured context/activations.
Driver budgeting uses the actual one-slot floor and recomputes after the table leaves
residency; two slots are an optimization, never a mandatory minimum.

Integration prerequisite: ObjectFiles is exposed only when Load.staged_tiers and
plane.SEALED_HOLES are available. The reviewed B1 machine wiring and a TensorFS plane
with register_sealed(files=...) are required; stock e044plane and current master
without staged_tiers do not activate this path. This affects selected-row recovery,
not baseline operation compatibility.
