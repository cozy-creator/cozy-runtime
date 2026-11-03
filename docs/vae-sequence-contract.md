# H3 sequence: native geometry, nonnested stage execution

The active H3 tiled decoder uses approach B through `author.decode_vae_tiles`.
The package no longer wraps `_decode_tiles` in an outer retryable cut or runs the
old functional decoder body. Source integration is complete; GPU and full-H3
qualification remain separate gates.

| Approach | Concrete behavior | Limit |
|---|---|---|
| A: stage admission only | Edges and existing standalone block each select/begin/end under current phase; no outer leaf | Cannot choose a reduced outer window before writes or close later full-hidden demand. Large-B routes need an additional declared preparation or must decline. |
| B: native window/allocation preparation, then stages | Native chooses immutable outer W/inner A from a complete legal menu; preparation leaf ends before every edge/block; actual source hooks precede fresh stage admission | Initial selection is not a future reservation. Source-phase changes, foreign pressure and unknown library resources may still fail later stages without replay. |

## Chosen interface

`native_vae_sequence.variants(B)` describes geometry only, full B first, down to a
one-tile minimum. No room, free-memory number, percentage or fitted byte estimate
enters this code. The coordinator closes each declaration using real edge
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

## Shared block interface

Preserve the standalone `split_layer_scale_block` interface, ordinary module hooks,
all module/kernel/source eligibility checks, and catch/native nesting guards.
Factor metadata inspection/query preparation away from actual hidden/rotary binding.
Expose a private prepared block with real variants/buffer specs; a supplied-buffer
invocation validates actual input/output/scratch spans and current sources, enters
ordinary module hooks exactly once, then executes the unchanged body once.

The supplied path performs fresh current-phase native admission after hooks using a
single fixed geometry and only its true additional allocations. It must not rerun
selection for a different A, allocate another full output, or take a guard-bypass
boolean. Standalone selection/allocation remains its existing path. The active
author/package entrypoint enters the real common source scope and constructs the
complete native menu from these queried providers.

Provider queries and bootstrap are not numerical warm-up. Unsupported topology,
source layout, artifact/math policy or legal minimum may decline before writes.
Accepted later failures never fall back to `_decode_tiles` or the old chooser.
Retryable-parent and native nesting guards remain. A globally initialized process
group no longer disqualifies independent VAE calls: the block, its attention module
and its processor must all have no parallel configuration. These bodies perform
no collectives; their native windows need no cross-rank geometry agreement.

## Explicit stream and custody precondition

The coordinator uses Runtime's current owned nonblocking execution stream S.
Executor `_execution_context` selects S; its command lifecycle serializes this call.
Preparation, allocation, input/rotary production, hooks and execution stay on S.
The standalone fixture uses its real `Domain.using()` plus explicit stream context;
`using()` alone does not select S. If a producer uses T, record a real completion event on T
and join it with S.wait_event before binding. Do not infer producer readiness from
Tensor metadata, native admission, or record_stream.

The coordinator retains the supplied invocation, inputs, output/scratch and query
owners through checked S completion, including an error. Existing source hooks keep
their actual Memory._used -> native Use.consume_on(S) last-consumer fences; those
are the source ownership mechanism, not a guessed replacement. Allocation streams
are not silently inferred or redirected by the private supplied-block interface.

The real GPU client includes a T allocation/producer -> event -> S.wait_event -> bind
case, retaining every T owner until S completion before retirement. It uses the
existing Domain teardown order: drain, retire allocator stream bookkeeping, destroy
that owned stream. This case is prepared, not run; ordinary S-only client paths also
keep all inputs and buffers until denoise_loop's checked S synchronization.


## Current Torch capability and stage holdings

The active prefix/tail pin is Torch 2.15.0 git
`d0f2af8b66b20da4baa50644c0cd3f887a2eae2e`; earlier cohorts remain frozen controls.
The exact [CUDA LayerNorm source](https://github.com/pytorch/pytorch/blob/d0f2af8b66b20da4baa50644c0cd3f887a2eae2e/aten/src/ATen/native/cuda/layer_norm_kernel.cu)
uses contiguous borrows for input/affine vectors, allocates Y, mean and rstd, and
views statistics without another allocation. The vectorized forward and fallback
launches use registers/shared memory, with no additional device tensor. The exact
[autocast registration](https://github.com/pytorch/pytorch/blob/d0f2af8b66b20da4baa50644c0cd3f887a2eae2e/aten/src/ATen/autocast_mode.h)
keeps native_layer_norm in FP32. This is source evidence, not CUDA execution proof.
Unknown Torch commits still decline before numerical writes.

All query preparations stay live through the last borrower: actual edge bias,
block bias/index, Context and full/ragged Plans. Their physical tensor payloads are
reported separately from initial allocation choices. Initial choices allocate the
full returned pixel owner plus exact edge buffers, two FP32 hidden banks and one
actual maximum-sized bank for each block scratch role. Every maximum comes from
real queried block specs; it is an actual allocated owner, not capacity padding.
After prefix completion, post buffers and prefix construction scratch retire.
Hidden16 remains through the checked block35 fence because retained block0 invocation
borrows it. At that fence all invocations, block banks, rotary and unused hidden A
retire; hidden B and tail projection buffers remain while the three actual norm
outputs receive fresh native admission. Tail completion releases these norm outputs.
A later outer window allocates its exact full/ragged owners under the same fixed
native-selected geometry and cannot replay or shrink the operation after failure.

## Remaining Ulysses agreement

The H3 transformer supports degrees 2/4/7/8 and uses the upstream sequence _cp_plan.
`attention_ulysses` calls `parallel.ulysses.exchange`; `agree.passing` gathers the
previous proposal plus shape digest, and `agree.agreed` fixes the maximum proposal
for the pass. `ulysses._level` still derives its next proposal from a Python Room
snapshot. It does not implement native legal-choice agreement before collectives.
This is a distinct transformer integration gap, not covered by independent VAE
spread. Full 15s/362-frame/1344x768/30-step/audio and rank2/4 gates remain mandatory.
