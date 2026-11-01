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

## Runtime IO qualification

Run **24961** passed all seven cases on the RTX 4090, PID **47629**, using Runtime
`c05c3ac32`, TensorFS `0e05a2539`, and fixture `d6c3b91eb`.
[Fixture PR2369](https://github.com/cozy-creator/cozy-creator/pull/2369) contains the
complete `eager-prefetch/evidence/2026-11-01.json` receipt.

- The actual unknown-entry denoise loop made one native attempt/decline, admitted
  and started zero prefetches, and read the correct 8 MiB through demand. Its
  parent remained unknown and unpublished.
- The explicitly fixed program's first pass published an 8 MiB working sample
  without prefetching. Its second pass admitted, started and completed one real
  next-block read; total source bytes remained 8 MiB, including demand reuse.
- Queued invalidation canceled one admitted Fill before start: zero new source
  bytes and zero started reads. Subsequent demand returned correct output.
- Kept-host speculation made no native attempt; ordinary demand used 8 MiB of
  retained host bytes with no store-byte growth. A transforming source likewise
  made no native attempt, then its real fill ran twice during demand.
- Model close had already retired the named phase key. Live working bytes
  returned exactly to 17,039,360 with two executor library allocations, pending
  bytes and graph pools zero. Correct outputs and zero resident weight pages are
  mandatory assertions in the successful fixture, not separately serialized fields.

This qualifies the Runtime read/admission/ownership route. The known key belongs
to the fixture's fully specified program, not adaptive H3 execution. The separately
named full H3 performance/quality/transfer arm remains the acceptance gate for this
policy change; no performance improvement is inferred from these counters.
