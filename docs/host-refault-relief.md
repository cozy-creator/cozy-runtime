# Idle host-cache relief during kernel refault churn

Instantaneous `memory.current - limit` can be zero while the kernel repeatedly
reclaims and refaults active file/code pages. Ordinary debt maintenance remains;
fresh coincident growth in (`memory.events.high` OR `memory.events.max`) and
`workingset_refault_file`, while a tenant is active, also permits one idle
generation's managed cache to be evacuated.

Samples cover every finite-limit ancestor, keyed by directory identity and both
configured limits. Counters remain separate; resets, availability changes or
changed identity establish a new baseline. Missing optional kernel fields leave
ordinary debt maintenance available. No PSI/rate threshold or free-memory margin
is introduced.

The existing zero-sized reclaim reports managed-held bytes without dropping
managed backing (the executor can still synchronize and trim idle allocators).
Those measured bytes size subsequent requests. Existing queue/reuse ordering,
idle checks, expected PID/epoch at dispatch, exact stream eligibility, and strict
held-byte decrease bound the episode. Lost evidence, active work ending, changed
generation, or no progress stops it. One successfully reclaimed generation ends
that episode; later fresh activity can justify another.

This is best-effort cache policy, not admission credit. Shared references and
allocator retention can prevent physical relief. Existing backing rules preserve
mutable, compiled, unsafe-alias and active-scope state. Aggregate ancestor activity
can include unrelated siblings, so evicting a useful idle cache is a bounded tradeoff;
this does not prove the exact refaulting file or guarantee progress when only active
or unreconstructible backing remains. Host admission still observes physical room.

Validation is CPU-only. Real affected-model/stall qualification is separate; no
performance improvement, GPU or multi-GPU qualification follows from these tests.
