# Same-request streamed graphs and raw stage prefetch

This P2 followup activates the existing streamed recorder for one explicitly
keyed, nonresident component. It is an execution-path change and needs real model
qualification; replacing dormant lookahead code alone would not be a production
policy cutover.

## Execution boundary

The existing eager first step still runs once. When its declared semantic
configuration is representable, `_record_first_streamed` records only the
remaining steps at the existing block boundaries, on the current request's
already advanced state. The existing phase runs determine the ordered native
sequences and their repeat counts. Accepted native work never retries package
execution. The resident whole-call path is unchanged.

The streamed program is not installed as a retained program and has no `again`
sequence. It closes before the eager `after` work. This preserves H3's prohibition
on cross-call graph reuse with hidden request-local derived caches. H3 requires
its explicit twelve-input/semantic-key declaration; an unkeyed closure does not
silently become eligible.

A failed recording unwinds its frames, closes all partial definitions and their
source owners, and reports an explicit preparation refusal. Only then may the
existing eager remainder run. That is not a successful native qualification:
receipts must show `mode=streamed` and the expected native step count.

## Historical performance boundary

The recorder became dormant in Runtime #1948 (`db7f2c99f`) because one-call
recording and extra reads cost more than replay saved. The recorded comparison
was Qwen int8 1 MP on a 3070: eager 45.6–46.8 s versus native 49.3–50.7 s; H3
FL2VA 15 s on a 5090: 716 s versus 737 s. This followup does not erase that evidence.
The new native policy must qualify capture cost, denoise time and H2D on matched
full requests before this path is accepted as an improvement.

## Native raw admission contract

The supported source is the existing immutable `StageSource`: it holds the
regions and serialized input recipes for a detached raw capture. Its existing
refusal of private decode work remains. Quantized arrays can still be raw source
bytes when the graph consumes those arrays directly; this does not enable a
speculative decoder or its scratch allocation.

The link carries actual sequence adjacency, the current stage's semantic
execution key and its ordered graph-definition tokens, with no Runtime byte
ceiling. Native lookahead matches that identity to the active replay before
consulting its last successful measurement. Unknown, invalidated or retired
definitions decline speculation. The first pass cannot predict itself; repeated
execution of the same definitions supplies a real known-key caller.

Native admission checks the whole destination's missing backing against current
driver/managed capacity and uncovered measured demand, without reclaim or a
streaming-limit change. Locks are released before IO. The existing source job
continues to own pins, pending DMA, cancellation, rearm and checked cleanup.

This path replaces the Python `_stage_prefetch_window` byte/window calculation,
`_stage_window_owner`, restore lease and `max_prefetch_bytes` argument. Runtime no
longer sums current/next source reservations, missing bytes or a maximum pair
window to authorize these reads. A linked edge is only a declaration; native
admission may still decline it at actual read time.

Eager queued prefetch and the broader split/reclaim policy remain separate work.
This source cutover requires the paired native raw-stage admission API and real
GPU qualification before it is accepted. It has no legacy stage-budget fallback.

## Required proof

Exercise first-pass refusal then later-pass admission, a fresh graph with the
same semantic key, wrong active-stage identity, pressure changing after linking,
partial source-group failure, canceled/failed reads and pending-DMA cleanup.
Weightless inter-block stages must preserve preferences; encoded private scratch
must remain ineligible. A repeated interval must not prefetch its suffix early.

Full H3 proof is 15 s, 362 frames, 1344×768, 30 steps and audio, with actual native
mode/step counts and preserved quality. Compare capture, denoise and H2D against
the matched eager path. Completion or an eager fallback alone does not pass.
