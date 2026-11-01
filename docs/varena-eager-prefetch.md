# Eager raw prefetch policy cutover

This isolated P2 slice replaces Runtime's eager next-block and queued-request
speculative admission with `Region.prepare_prefetch`. The native owner admits
only a successful, matching, known entry key with enough physical room. Current
Runtime eager observations enter with an unknown key, so their speculation must
honestly decline. No prior post-hoc cut digest is substituted.

Remove the Python window/headroom budget, provisional preference, driver-free
comparison, cache-release retry and the now-unreachable host-prefetch adapter.
Immutable stored RAW sources keep their existing source/overlay/identity checks,
owned read publication, cancellation, checked producer drain and demand handoff.
Kept-host and transformed sources decline speculation; ordinary demand's host
cache and fill/decode behavior remain unchanged. A stored encoded weight role
that requires no transform is still an immutable RAW byte source.

Metrics distinguish native declines from unsupported source recipes; a native
None is not labeled a capacity shortage. Older supported SDKs without the native
admission method decline optional prefetch and continue demand loading. There is
no old byte-policy fallback.

Source implementation and qualification are in progress. A separate matched full
H3 arm must report quality, denoise time, store/H2D bytes, actual prefetch counters
and cleanup against the frozen b222 cohort. Run24883's old path started1478 reads,
completed1476 and waited20.994s; removing that overlap may regress performance.
Unknown-key declines demonstrate the declared policy, not restored prefetch value.
This does not remove the remaining demand/residency/split policy or finish P2.
