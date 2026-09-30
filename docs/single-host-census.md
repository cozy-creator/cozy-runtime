# One fresh host-storage census

Planner.facts reports total host backing, resident host backing and cached host
backing. A storage can belong to both resident and cached classes, so their totals
are not additive. The combined total must deduplicate their union by storage pointer.

Previously facts performed three complete local walks. It now visits each held unit
once, collects its cached tensors followed by any HOST-resident slots, and updates
three independent pointer maps using full storage sizes. The combined map preserves
that per-unit order. Meta/device tensors are not counted as host. Hosted components
still contribute host ownership; disk-weight reporting keeps its existing exclusion.

The census is rebuilt for every facts call. There is no persistent cache or stale
request state. Driver/allocator observations and the available-room query still occur
before host enumeration, in the same order. No admission, RPC or residency policy is
changed. Existing individual _host_backings callers remain unchanged.

Guarded CPU tests use registered Torch modules with shared/strided backing, resident
and cache overlap, empty/meta values, changed parameter storage and a device-only
stand-in that must never be queried. A modest registered2048parameter fixture has
65KiB of real tensors; complete old/new facts dictionaries match. Median CPU facts
cost was4.930ms versus2.538ms in one benchmark. This metadata microbenchmark does not
establish a GPU denoising or end-to-end speedup.
