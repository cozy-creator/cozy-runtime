# Complete H3 FFN: one declared row composite

Read-only plan, `/root/retention_restart`, 2026-11-01. Reviewed Runtime **9b8d74e69**
and the frozen package **93861c509**. No product/package source changed. This is
separate from the running native-retention comparison.

## Actual path and arithmetic

Frozen `minimax_h3/glue.py:917–944` calls
`split(_FFN(ff), tokens, Cut(1))`. `_FFN` contains the entire two-projection call:

```text
BF16 normed [1,M,Din]
  → act.proj(normed)                  # [1,M,2F], packed [value | gate]
  → int8_rows(..., down, swiglu)      # currently declines for hooked down
  → package swiglu kernel            # BF16 [1,M,F]
  → down                             # BF16 [1,M,Dout]
```

The frozen H3 construction permanently registers `lora.Hook` on its projection
families (`adaln_pruned.py:380–387`). Runtime `int8_linear.prologue:330–354` rejects
any consumer forward hook. Thus the observed hooked int8 path materializes BF16
SwiGLU even when the hook is baked and its normal forward adds no delta. This is
not permission to change the graph-hook eligibility rule or to enable fused
activation quantization as part of this memory-policy patch.

H3's `_silu_product` / `_swiglu_kernel` at `glue.py:208–250` computes
`BF16(value * BF16(gate / (1 + exp(-gate))))`; division and exponential are FP32,
with the package's explicit BF16 round points. `swiglu` at607–634 allocates the
BF16 output and launches this existing compiled kernel. Reuse the same kernel,
constexprs and launcher; do not replace its arithmetic with a new PyTorch chain
or a different fused quantizer. The identity dropout is already skipped by
`_FFN`. The package install validates that topology at1125–1160.

Current widths are Din=Dout=5376 and F=14336 (`glue.py:87`); obtain and validate
actual module dimensions, do not dispatch from the model name. The existing
full-H3 token count109123 is ragged under many useful row chunks.

## Small public declaration, one operation kind

Add one frozen author descriptor in `author/kernels.py`, provisionally named
`SwiGLUFeedForward`:

```python
SwiGLUFeedForward(
    up=act.proj,
    down=ff.net[2],
    launch=act._glue,
    activation=BF16_SWIGLU_ENTRY,
    value_first=True,
)
```

This descriptor carries operator intent and exact executable identities, no
bytes, available-memory number, split count or callback that allocates outputs.
`activation` is the existing public `TritonEntry` for the package kernel, with
its explicit BF16 pointer signature, `FP8=False`, `BLOCK=1024`, warps8 and source
identity. Runtime accepts only the actual compiled `_Launcher` and exact entry
present in that launcher, then invokes its fixed pointer/integer ABI directly.
No arbitrary Python activation callback runs inside the closed allocation scope.

Add an optional `intent=` argument to public `author.split`, defaultNone. The
package's `feed_forward` supplies this descriptor only for its matched
`FusedSwiGLU`, non-FP8 materialized path. `_FFN(ff)` remains the ordinary callable
for unsupported cohorts. Runtime inspects the descriptor **before** installing
legacy `catch.cut` for this call. An unsupported layout/provider/hook declaration
can return an explicit pre-execution exclusion and use the existing path. Once
native selection is accepted, failure is final for this operation: no descent,
repeated FFN callback or retry of already-written output rows.

The package creates/stores its descriptor when its execution plan installs the
matched topology, after the Runtime has replaced/finalized encoded leaves. It can
select the existing BF16 entry from `GLUE.entries`, avoiding another kernel-set
registration or another compiled shader. A
small `feed_forward` change passes it to `split`; no Runtime import of `_FFN`,
`FusedSwiGLU` or another private H3 class. The first package/Runtime pair requires
the new public descriptor API; no unsupported old SDK is silently treated as the
new contract.

## Exact first supported scope

- Actual Runtime NativeLinear for both projections, int8-convrot, BF16, mma.sync
  SM86/SM89 family already served and qualified by its real native opener.
- Contiguous aligned `[1,M,Din]` Tensor input (or2D rows), no PreQuantized `_Tokens`,
  autocast, gradients, global/pre/custom hooks, compiled-call replacement,
  uncommitted enclosing cut, capture/replay or multi-rank execution.
- Both weights/scales/biases are materialized and retained **before** selection.
  Bias is absent or exact BF16 contiguous/aligned; no hidden cast/copy in scope.
- Projection hooks, if present, are exact built-in baked `lora.Hook`. Preserve
  normal module invocation for each up/down chunk so hooks still receive their
  original float input. Unbaked/custom hooks are excluded before execution.
- The first materialized path also requires a hooked down projection, matching
  the current reason `int8_rows` declines. Unhooked/fused-prologue and FP8 paths
  remain separate until their actual arithmetic/allocation contract is qualified.
- Explicit shared operands or a current cached operand for the incoming Tensor
  exclude the composite. The declaration makes the two intermediate reads
  single-use; do not create or retain implicit sibling operand caches within it.

These restrictions are capability boundaries, not a new user knob or a claim
that all H3 paths have been migrated. The normal single-rank int8 H3 path must
actually emit this composite's telemetry in the full-model test.

## One complete allocation menu

For total rowsM and legal chunkm, let tail=`M % m or m`. Use whole then balanced
halving chunks through1, as the current projection chooser does. Query the exact
existing `int8_plan` for BOTH GEMMs at BOTH actual row counts:

```text
up:   M=count, N=2F,   K=Din, ldo=2F
Down: M=count, N=Dout, K=F,   ldo=Dout
```

Each query returns its automatic dispatch kind/config and exact workspace size.
GEMV is a distinct zero-quantization/workspace route and is preserved for rows it
already serves. No forced Identity kernel or guessed Stream-K bytes.

Preallocate all live scratch once after selection:

| Buffer | Logical bytes |
| --- | --- |
| Full output `[M,Dout]` BF16 | `2*M*Dout` |
| Up packed `[m,2F]` BF16 | `4*m*F` |
| Activation `[m,F]` BF16 | `2*m*F` |
| Shared quantized scratch | max of `count*K` for queried up/down full/tail GEMM routes |
| Shared row scales | `4*max(count)` over those GEMM routes |
| Shared native workspace | max exact query bytes across up/down full/tail |

Zero-sized optional Q/scales/workspace allocations are omitted. The Q allocation
is flat; each projection uses its contiguous prefix as `[count,K]`, avoiding a
max-width stride passed as the wrong kernel width. Full/tail chunks reuse the
same buffers and allocate no tail-specific size class. The menu goes to native
`split_select`/`split_begin`; native block-size classes and reusable same-stream
storage decide required physical backing. Input, weights, common tensors and
caller output aliases are outside-live holdings and are not subtracted again.

All scratch is simultaneously allocated, so the declared list is an exact
closed-scope upper bound, not a sum of unrelated liveness estimates. Measure the
actual successful choice after the real parent phase completion, as existing
native leaf scopes do. This does not estimate the whole-phase watermark.

## One executor, normal projection hooks

Factor only the existing projection executor's reusable mechanism into a private
helper: validated exact dispatch, retained raw weight operands and `_gemm` calls
that write supplied output/Q/scales/workspace. Keep admission policy out of that
helper. A composite installs an owned context for one exact leaf invocation at
a time, then calls `up(chunk, out=packed_view)` and
`down(activation_view, out=output_view)` normally. `NativeLinear.linear` consumes
that context instead of starting another native split or legacy retry. Validate
module/input/output identities and the saved stream on every use.

```text
select one FFN chunk; begin one native FFN observation
allocate full output + reusable scratch once
for each row interval on one saved stream:
    quantize input → queried up GEMM (or actual GEMV) into packed buffer
    existing package SwiGLU kernel into activation buffer
    quantize activation → queried down GEMM (or actual GEMV) into output interval
end submitted observation
parent's actual completion publishes or invalidates it
```

Same-stream ordering protects reused scratch; no per-chunk event or synchronize
is added. Capture and asynchronous unknown completion do not publish. If the
minimum cannot fit, native typed exhaustion happens before output allocation.
After selection, an OOM/dispatch error gets the existing no-retry marker and
invalidates the scope. Cancellation never retries a partial composite.

## Exact identity and lifetime

Key the complete FFN menu by both native library/dispatch identities, up/down
encoding and complete shape/stride/dtype/alignment, output shape, model/source/
overlay identity, package activation source+entry constexpr/options+compiled
launcher artifact, row count and every full/tail dispatch choice. Chosen ID is
part of the parent phase fingerprint. Retain weights, launcher and descriptor
through submission; phase completion supplies proof for learning. Model-owned
keys retire with models; componentless proof keys belong to the real Request.

Do not treat a previously completed dynamic-cut sequence as a committed phase
plan. Migrating this FFN makes one complete operation planned before execution;
complete attention and the rest of the phase still need their contracts before
automatic unknown eager phases can admit measured prefetch.

## Concrete patch ownership and gates

Runtime files:
- `author/kernels.py`: one frozen descriptor and documentation.
- `author/_split.py`, `internal/execution/__init__.py`: optional intent dispatch
  before the generic outer cut; existing callers unchanged.
- `internal/execution/native_linear_split.py`: extract direct prepared projection
  execution from its admission wrapper, preserving the qualified standalone path.
- `internal/execution/native_ffn_split.py`: the single complete FFN menu/executor.
- Reuse the existing `_LINEAR_SPLIT` context consumed by `NativeLinear.linear`:
  its `.module`/`.run` seam already permits the composite's supplied-output executor.
  No leaf or hook source change is expected; if implementation needs one, review
  it separately rather than silently expanding the qualified standalone path.

Package files in a separate branch based on93861:
- `minimax_h3/glue.py`: publish the existing BF16 SwiGLU entry as intent metadata,
  install the immutable descriptor on matched FFNs, pass it from `feed_forward`.
  No Triton kernel body, dropout, residual/gate arithmetic or model scope change.
- Package dependency/API floor chosen with release coordination, not guessed.

No new C++ workspace API is needed: the already-qualified exact query is used
for both GEMMs. Native binaries can be reused only after byte/source verification;
package Triton source keys should remain unchanged because kernel bodies and
constexprs are unchanged, but verify rather than assume.

Required real proofs, in order:
1. Whole/tail queries versus original automatic GEMM workspace callbacks for both
   real widths, including a nonzero Stream-K case and GEMV minimum. No mocks.
2. Actual original `_FFN` versus composite at whole/half/quarter and ragged tails,
   exact intermediate BF16 activation comparisons plus existing numerical budget
   for changed GEMM shapes. Actual baked hooks must preserve invocation behavior;
   unbaked/custom/FP8/shared/capture exclusions remain before execution.
3. Two managed capacities force different chosen chunks; complete output quality
   remains within the existing bound. Minimum refuses before any output/scratch;
   all lifetime/accounting and same-PID key cleanup assertions remain strict.
4. Same-stream pending reusable buffers and clean aborted/failed scope retain the
   previous successful sample. No nested standalone-linear observations.
5. Matched full15s/362frame/1344×768/30step/audio H3 shows real FFN composite
   selection counts and observed chunks, denoise, source store+host transfers,
   native eviction reasons and visual quality. Use matching new-package source
   in both arms, compare before/after Runtime explicitly. No shortened smoke or
   model-name type shortcut qualifies this.

The existing full H3 pair and all tested products remain untouched by this plan.

## Fresh-package integration boundary

At implementation kickoff, fresh package master28ff0e75 (H30.2.6) has already
moved the FFN inside an in-place `_Tail.run` outer `author.split(..., out=...)`.
Its previous `feed_forward` and `_Tokens` helpers are gone. A native FFN nested
inside that uncommitted outer cut must decline. The independent Runtime composite
and original938 operator proof can be implemented, but actual current-package
adoption requires a complete Tail contract or a committed outer menu. The old
package source must not be restored over this newer work merely to force a pass.
Root is reviewing that integration direction; no package execution source has
been changed at this checkpoint.

Standalone projection source and AST are pinned before helper extraction under
this task's `.task/standalone-before.json` and `native_linear_split-before.py`.
No native C++ source change or new release version is planned. This is an
unreleased local dev cohort with explicit required API capability until release
coordination assigns a public floor.
