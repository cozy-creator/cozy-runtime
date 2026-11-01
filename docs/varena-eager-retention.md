# Eager demand retention ownership checkpoint

This next P2 slice builds on frozen c05. Its prefetch cutover passed real IO run
24961 but failed standalone full-H3 performance: 24962 took 940.758346 s denoise
versus b222's 899.693466 s (+4.564%), and read 573,365,415,236 versus
544,543,768,388 store bytes (+5.293%). Preserve those results; no old-prefetch
fallback or weaker performance criterion is proposed.

The native API is `Arena.clear_streaming_limit()`: clear only the legacy cache
window, preserving actual process, development and peer limits. It does no driver
work, allocation or reclaim; Python supplies no maximum integer sentinel.

The Runtime prerequisite is an execution-owned eager capability, distinct from
P3's Legacy/Cooperative sharing owner. It is admitted only after the execution
engine commits to an eager call and real owner registries are closed: no queued,
recording, prepared or kept program; no prepared submission; and no unmanaged
Runtime disposable-cache owner. Empty request lists or zero pool bytes cannot
prove that. Existing explicit/common/escaped storage Pins remain authoritative.
Creation must claim ownership before allocating and must reject a conflict with
an admitted eager capability. The guard spans completion/abort, not one snapshot.
No ordinary user-facing implementation switch is introduced.

The first patch is this ownership boundary for review, not policy activation.
After review, the eligible eager cohort can use actual native demand/reclaim and
native standby-LRU/loop-MRU phase intent while losing its streaming-window,
prefix/headroom and `_replan` byte formulas. Coupled host-tier uses of `preferred`
and generic `Request.room` advice must be handled explicitly; graph/cache-owner
retirement, generic split advice and P3 `_room_floor` compatibility remain named
gaps. Unsupported prepared execution cannot silently inherit the changed policy.

Required gates are real native clear/retention behavior at two capacities, actual
Runtime cohort ownership, common/explicit/escaped Pins, source failure and clean
exhaustion, followed by separately named full H3 and SDXL validation. No retention
benefit or GPU result is claimed at this checkpoint.
