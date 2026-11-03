# H3 sequence: native geometry, nonnested stage execution

Root selected approach B. This document and the independent schedule/assembly module
are source work, not enabled package wiring or GPU qualification.

| Approach | Concrete behavior | Limit |
|---|---|---|
| A: stage admission only | Edges and existing standalone block each select/begin/end under current phase; no outer leaf | Cannot choose a reduced outer window before writes or close later full-hidden demand. Large-B routes need an additional declared preparation or must decline. |
| B: native window/allocation preparation, then stages | Native chooses immutable outer W/inner A from a complete legal menu; preparation leaf ends before every edge/block; actual source hooks precede fresh stage admission | Initial selection is not a future reservation. Source-phase changes, foreign pressure and unknown library resources may still fail later stages without replay. |

## Chosen interface

`native_vae_sequence.variants(B)` describes geometry only, full B first, down to a
one-tile minimum. No room, free-memory number, percentage or fitted byte estimate
enters this code. The root coordinator must close each declaration using real edge
and block query facts and actual physical owner allocations before offering it to
native policy. `resolve` only resolves an exact native-returned ID; it does not admit
anything. Native phase-preservation ranking can select a smaller W/A than the first
menu entry. Accept/report that legal choice and measure actual reloads/performance.

`operations(variant)` is immutable. For W=B, post-quant/prefix run once and every
block processes all tiles before the next block. For W<B, each outer window runs
post-quant/prefix, all 36 blocks and tail once. There is no Python capacity retry or
reexecution of an accepted window. Ragged inner geometry is `min(A, window_count)`;
its actual full/tail GEMM queries must be prepared, including non-balanced ragged
counts. Source hooks bind one actual block at a time, not all 36 weights for metadata.

The fixed returned output is contiguous FP16 `[B,3,28,256,256]`. `OutputAssembly`
returns exact first-axis views for each tail's supplied output, avoiding an extra
join allocation/copy. `Cursor` enforces host invocation order and terminal failure;
it is explicitly not a native admission or GPU completion capability.

## Allocation and lifetime contract

Existing input/source owners and actual query bias/index/Context/Plan holdings are
separate from future allocations. Every buffer owner/view must come from the actual
post-quant, prefix, block and tail contracts; unknown library bytes stay unknown.
Do not sum 36 scratch sets or invent a maximum allowance. Reuse only physically
specified storage whose current views and nonoverlap/lifetime schedule are proven.

Per outer window: post result survives prefix; FP16 prefix hidden survives block 0;
rotary survives block 35. Two supplied FP32 hidden banks ping-pong through the blocks
(block 0 -> A, block 35 -> B). The current block's source views use its existing
source-use lifetime. Shared packed/expanded/attention/cast/workspace storage is
retained through its actual last use. Full pixel output remains live through handoff.

After checked completion of the last block, retire unused hidden A, block scratch
and dead prefix/rotary holdings. Then perform fresh native admission for the real
functional tail allocations: FP32 Y=`window_count*1797*2048*4`, mean/rstd each
`window_count*1797*4`. They are not fake supplied norm outputs. Tail keeps hidden B
and its actual fixed projection/copy storage, then releases the three norm results
at checked completion. Recreating retired local storage for another outer window
uses fresh admission of the fixed chosen geometry, not a new geometry choice.

Bootstrap observations describe only actual bootstrap allocations. Their keys/scopes
must never publish as whole-decoder demand. Every later stage uses its own current
phase/generation receipt. `split_begin` validates generation at entry; its leaf does
not retain an execution-long phase-plan lease. A current-stage minimum refusal is
before that stage's first write; after prior stages wrote, failure remains terminal
and retains the actual asynchronous owners/partial-output diagnostics.

## Minimal shared block factoring for root review

Preserve the standalone `split_layer_scale_block` interface, ordinary module hooks,
all module/kernel/source eligibility checks, and catch/native nesting guards.
Factor metadata inspection/query preparation away from actual hidden/rotary binding.
Expose a private prepared block with real variants/buffer specs; a supplied-buffer
invocation validates actual input/output/scratch spans and current sources, enters
ordinary module hooks exactly once, then executes the unchanged body once.

The supplied path performs fresh current-phase native admission after hooks using a
single fixed geometry and only its true additional allocations. It must not rerun
selection for a different A, allocate another full output, or take a guard-bypass
boolean. Standalone selection/allocation remains its existing path. Author/package
wiring, source-scope entry and complete outer native menu construction remain root
owned until this concrete factoring diff and GPU edge evidence are reviewed.

Provider queries and bootstrap are not numerical warm-up. Unsupported topology,
source layout, artifact/math policy or legal minimum may decline before writes.
Accepted later failures never fall back to `_decode_tiles` or the old chooser.
No guard is removed by this source lane.
