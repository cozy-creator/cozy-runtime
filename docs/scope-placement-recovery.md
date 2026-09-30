# Scope placement recovery

R17 Anima1874/1875 failed while entering `render` inside `warm`: a common-unit fill
ran out of memory although its earlier byte estimate fit. The idle SDXL process still
held212MiB, but no measured room request was made for that allocation failure. The outer
warmup had already begun; the scoped render body had not. Repeating warmup would repeat
authored work, while retrying the failed placement need not.

Scope admission now uses one atomic unit transfer per recovery boundary. A complete
store fill is also eligible because `tiers.from_disk(None)` restores every loading unit
on failure. Previously successful sibling placements stay in place and are never
credited as recovery progress. This distinction matters when a sibling reuses an
already charged allocator cache hole.

After an OOM, the baseline is the larger of pre-attempt and post-failure allocatable
room. The planner trims unused allocator pages and requests a measured minimum through
the existing Machine policy if needed. A retry requires at least the failed allocation's
size in additional observed room; an unreported size still requires strict positive
progress. Failed allocation growth followed by releasing those same cached pages is not
new progress. The baseline rises after every retry, so the same capacity cannot justify
an endless replay. There is no fixed margin, retry count, changed numerical mode or
whole-method replay.

Groups retain their all-rank preparation authority, and spare calls cannot reclaim peers.
Non-OOM failures propagate once. Failed device cleanup marks the planner unusable; later
scope admission refuses before touching backing. Failure still releases the unentered
scope's pins. Per-component scope-placement counters distinguish these retries from
authored operation invocations.

CPU/meta tests exercise the production Model scope and tier rollback paths, including
unit and whole-store fills, no gain, repeated failure, cached-page reuse, partial success,
device cleanup, group/spare authority, and the enclosing warmup's RNG/body continuity.
These checks are not GPU acceptance; the original mixed1GiB sequence remains a separate
qualification gate.
