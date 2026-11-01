# Varena P2 observation foundation

This checkpoint measures execution. It does not complete P2 or replace production
prefetch, reclaim or split policy.

`_Loop.eager` opens a native phase with an unknown entry key and ends it after its
existing caller-stream synchronization. The final key contains the declared loop
key, step implementation, immutable checkpoint/component and fill/layout identity,
declared tensor geometry/dtype, phase range, actual attention choices/artifacts,
Torch dispatch settings and an ordered fingerprint of actual cut/layout choices.
It contains no tensor values, seeds, virtual addresses, object ids or refill
generations. A kernel epoch is checked for changes, not embedded in the key.

Unknown entry permits measurement only. The native record is published only for
a completed eager execution whose full immutable configuration is represented.
Cut retries and caught dispatch failures invalidate the sample even when a smaller
attempt later succeeds. Progress callbacks are bracketed by native allocation/free
counters; actual allocator transitions invalidate the eager sample, while allocation-free
host telemetry remains measurable. This serial guard is not the later asynchronous replay guard. Native allocation failures invalidate it too. Errors,
cancellation and changed dispatch preserve the previous successful sample.
Preparation, capture and replay are recorded separately and never publish a
successful eager watermark.

Native `peak_bytes` measures phase-owned newly allocated charged block capacity,
including its unsafe pending frees. `total_peak_bytes` separately reports total
charged holdings. `backing_peak_bytes` measures the distinct physical segments
containing phase-owned live/pending blocks; native admission uses that measured
backing requirement, with credit only for exclusively phase-owned eager segments
and wholly completed free eager segments. Pre-existing input/model/other buffers remain physical charges;
Runtime does not subtract them from a guessed global peak. Global peak resets used
by the existing sizing code do not reset the native phase accumulator.
`Request.working_phases` carries these observations into the normal memory record.
Keys belong to their live model owners; the last model close forgets the native
record after execution ends. A refused retirement retains its cleanup ownership.

## Deliberate boundaries

- Unkeyed closures, mutable/unrepresented keys, componentless scopes and active
  step caching do not publish reusable samples. Execution still follows its
  existing path. In particular, current H3 FL2VA closes over per-call tensors and
  supplies no loop key. It needs a targeted package declaration of those tensors
  as inputs and the semantic host layout/producer configuration as its key.
  This does not authorize graph reuse of its request-local derived caches.
- Every current Runtime entry key is unknown until actual adaptive cuts finish.
  The new native `Region.prepare_prefetch` admission API is therefore not wired
  over production RAW prefetch yet. Turning it on now would refuse all those
  reads. Existing `_prefetch_block` and native source speculation retain their
  current path; this is a missing cutover, not a permanent dual-policy fallback.
- Before cutover, a real caller must commit a semantic configuration and legal
  split choices before execution. A prior observed choice is not that promise.
  Native admission must reject unknown keys and use its measured reserve and
  physical accounting without Runtime inventing another byte budget.
- Native `PreparedSequence` replay does not call `vt.replay` for every graph.
  `Execution._finish` already proves readout completion with its event sync, and
  the source continuation proves read/graph completion, but neither supplies
  qualified active graph-pool demand. Pool reservation or all mapped pool bytes
  include unused/shared cache and cannot stand in for replay working demand.
  A native stage/pool ownership signal is the next replay integration seam.
- `_SequenceCapture._working_bytes` remains frozen once for a logical recording.
  No capture/preparation peak feeds back into later stage reservations.

Execution, collective order, source/alias ownership and cancellation remain in
Runtime. No extra CUDA synchronization is added to eager execution. Broader
watermark-driven sizing, unified reclaim, committed split choice, H3 integration
and legacy policy deletion remain open and require real CLI qualification.

## Next replay integration seam

`PreparedSequence.submit` already wraps each stage's continuation with TensorFS'
`PhaseController` before starting the native worker. Its native `Phase.start`
runs before input/graph work, `Phase.observe` receives checked completion, and
`rearm` separates successive executions of a reusable stage. This is the stage
policy seam; the Runtime continuation ABI need not grow a Python callback.

The missing input is an allocator-owned capture-footprint token for that graph,
covering the actual captured allocation offsets and backing segments, including
scratch logically freed while recording. `_Graph.begin/end` can bracket its
construction, and the resulting `PreparedRepeat` must retain it with the actual
graph/pool owner. It adds no separate physical pin. A reservation size or current
`pools_bytes` is not that token: they include unused capacity and other graphs
sharing the pool.

Only a successful, complete execution may qualify this footprint for its full
committed stage key. Native stage cancellation, failure, or rearm after an
incomplete execution cannot publish it. Stage start can then use the previously
qualified footprint for the same key; the live graph's already retained backing
must be credited once, not reserved again as future working allocation. Other
live inputs, outputs, native decode scratch and outside allocations keep their
own charges. `Execution._finish` still proves the final output copies complete.

This contract is the next implementation step, not functionality delivered by
the current measurement checkpoint. H3's explicit-input/key migration must keep
its existing refusal of cross-call whole-graph reuse until request-local derived
caches have an explicit rebinding contract.

## Upstream retry lifetime correction

This branch also carries the narrow Python correction from Runtime #1982:
a successful cut retry clears its saved exception before returning, so the
exception's traceback cannot retain the active caller frames and their tensors.
The P2 invalidation still runs before retry cleanup. The corrected body otherwise
matches upstream exactly. The real 8 GB refused-then-successful path remains a
qualification requirement; source equivalence is not GPU evidence.

The lifecycle foundation is Runtime 0.36.15 / `fb5dfa19c`, including #1977's
room reporting, signal handling and embedding gather plus #1974's weight-reclaim
correction. Stock-Torch allocator-setting revert #1981 is inapplicable to the
Varena allocator path. The later INT4 native kernel addition is not imported by
this checkpoint, so its original native binary reuse boundary stays explicit.

## Proof required before merge

Use paired Creator P2 APIs through the ordinary Cozy release gate. Verify that a
successful keyed eager configuration publishes charged demand; a caught failed
larger cut followed by success does not replace it; changed shape/dispatch/split
keys stay distinct; capture/replay/preparation never publish; and final model
retirement removes its records. Native API cases must prove unknown-key refusal
and real no-reclaim RAW admission, including unrelated pre-existing holdings.
Existing graph/capture/model gates remain required. Static checks alone do not
qualify CUDA execution or performance.
