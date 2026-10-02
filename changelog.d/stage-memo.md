A Model method can keep its results: `@uses_components("text_encoder", memoize=True)`. A repeated
call with the same arguments, weights, code and numerics returns the stored result without staging
the method's components, so a re-rolled prompt skips its text encoders. This is a Runtime
mechanism with no package-specific code; a package adopts it by marking a method.
- Two tiers. Each executor keeps recent results in memory. The machine keeps them on disk in
  TensorFS keyed roots (TensorFS ≥ 0.3.84), shared by every executor and kept across restarts.
  Without keyed roots only the in-memory tier runs.
- The Runtime decides per call whether a result is worth keeping on disk, from what it measured:
  the result must fit `memo.entry_bytes` (512 KiB by default) and cost at least twice a disk read
  to produce. A result that is skipped is never an error.
- The disk tier manages itself: entries expire after `memo.ttl`, the least recently used go first
  past `memo.store_bytes`, low disk drops them before anything else, and nothing is written while
  disk is low. Settings live under `memo:` in `etc/cozy/runtime.yaml`.
- A memoized method must be a function of its arguments. One that draws from the default random
  generator fails with `memo_rng`; a result from a degraded path (a tiled retry) or with
  non-finite values is returned but not kept.
- Each run records `memo.call` per call and `memo.method` per method: hits, misses, skips with
  their reason, bytes stored and time saved.
Measured on SDXL on an 8 GB card: a 318 KB entry, the encode stage 24–29 ms down to 1–2 ms on a
hit, and about 0.25 s less per request under the stage scheduler.
