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

Implementation is source-qualified: residency.py loses 106 net lines; AST and paired
Ruff have zero new findings (12 inherited in each source). Paired mypy improves
13 baseline findings to 12, removing the retired host-reader None case. Actual
Runtime/fixture imports and schema roundtrip pass with CUDA uninitialized. An
actual TensorFS file → Runtime declared model CPU smoke reads 8,388,608 bytes,
returns the expected values and closes its owners. These are not GPU admission
or full-model qualifications.

The ordinary GPU fixture uses the executor's real Request/Memory and the existing
Writer/Declared/Weights/Memory.load seam. It separates actual unknown-entry
Runtime loops from a deliberately fixed two-add program with an explicit native
entry key; only the latter may train and then admit a measured prefetch. It also
checks admitted-before-start cancellation, real host-tier pinning/demand,
transformed-source refusal/demand and model-key retirement. No fake Context,
Request or reader is substituted.

A separate matched full H3 arm must report quality, denoise time, store/H2D bytes,
actual prefetch counters and cleanup against the frozen b222 cohort. Run 24883's old path started 1,478 reads,
completed 1,476 and waited 20.994 s; removing that overlap may regress performance.
Unknown-key declines demonstrate the declared policy, not restored prefetch value.
This does not remove the remaining demand/residency/split policy or finish P2.
