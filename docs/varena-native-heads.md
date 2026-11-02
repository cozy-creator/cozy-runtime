# Next P2 consumer: complete H3 projected heads, initially INT4QK

The first implementation should claim the actual H3 `_Heads` operation **before** its outer legacy `split` begins. It can close the dense INT4QK route using the existing raw-pointer `_attn4` ABI, without a C++ change. Adding an allocator menu only inside `attention_ulysses._call` is insufficient: actual H3 arrives there inside `_Heads`' retryable legacy scope, and `catch.uncommitted_scope()` correctly makes current native consumers ineligible. Removing that exclusion would reintroduce nested policy/retry ownership.

This draft implements that bounded consumer. Actual CPU construction, the baked Rows hook,
unchanged GLUE identities, and targeted static checks pass. GPU qualification is pending;
no full-model performance claim follows from source checks. The accompanying Creator fixture
uses actual package Heads, encoded projections, compiled kernels and real allocator refusals.

## Actual selected paths

The read-only [selected-backends.json](selected-backends.json) hashes the original run receipts and extracts their selected backends and counters.

- Original retention pair **25022/25041**, package `938...`: transformer `kitchen-sol-int4qk`; dense `kitchen-int4qk`, protected prefix and exact paths `kitchen-int8`. The +5,703 innermost calls are exactly +4,798 `dense_step`, +904 `dense_path`, +1 `sparse`. Dense routes dominate that amplification.
- Current Tail control **25112**, package `9e7490c0`, Runtime `1aca1d6c`: transformer `kitchen-sol-producer-int4qk-shared-qkv`, with 900 sparse producer calls and 616 ordinary `_Heads` nested attention calls. The sparse producer owns QKV chunk generation and is a distinct future consumer. Its numerical/statistics recipe must remain intact.
- Current Tail candidate Runtime source is `fe5a783d`; its existing attention implementation is unchanged by the Tail work. Audio remains SDPA and video VAE Kitchen INT8.

Ordinary dense path:

```text
H3 Producer.produce (dense step/path)
  → original H3AttnProcessor
  → split(glue._Heads, Cut(heads))                 current outer byte policy
  → Rows(Q), Rows(K), Rows(V), norm/RoPE, seed
  → Diffusers selected backend
  → attention_ulysses._call
  → execution.attend                             current inner byte policy
  → SOL model policy
  → selected dense _call → execution.attend       another inner byte policy
  → attention_int4.attention → existing _attn4 ABI
```

`attention_ulysses._call` erases the backend descriptor in a lambda and exposes only head locality and row alignment to execution. `cuts.attend` consequently defaults to `scratch="flash"`; `sizing.Attention` has no exact contract for this INT4 buffer set or SOL's workspace. The migration should carry a concrete selected route/implementation identity, not assign another guessed scratch label.

## Closed initial scope and admission

Add a complete author intent at `H3AttnProcessor.__call__`, passed to `split` alongside the actual `_Heads` object. As with `ResidualFFN`, Runtime validates and claims it before entering any legacy scope. The intent supplies the live hidden input, exact Q/K/V modules, norm weights/eps, rotary tables, existing GLUE entries/launcher, optional actual key-mean destination, and selected backend identity. It does not supply guessed byte budgets.

Initially admit a single GPU, eager inference, plain BF16 input, unfused native INT8_CONVROT projections, head dimension128/equal heads, supported known/baked hooks, and the actually selected **dense INT4QK** route. Refuse unknown module overrides, unbaked adapters, capture, active legacy scopes, multirank, incompatible layouts, and other backend routes before execution. The actual H3 int8 lane has plain BF16 attention input: `_fusion_attn_fp8` requires ROWWISE_FP8 projections.

Resolve SOL's route using its existing model policy before acceptance, including dense step/path versus sparse and exact/outlier selection. Refactor a shared route decision from the current policy if necessary; do not duplicate its rules in an unrelated registry. Producer outlier handling and exact paths select Kitchen INT8 and must remain outside this first consumer. Pinning `kitchen-int4qk` directly can use the same operation when its constraints match. Resolve/record the actual backend identity and loaded `_attn4` artifact digest (`prebuilt.artifact("_attn4")`) along with model/fill/layout and GLUE identities.

An accepted operation bypasses the nested generic attention choosers and calls the **same** INT4 ABI. It retains the existing candidate/operand validation and telemetry semantics. A real allocation/kernel refusal propagates with no legacy retry or backend substitution. Pre-execution capability exclusion alone may return to the original `_Heads` callable.

## Complete allocation inventory

Let T be live tokens, H full heads, h a candidate head window, D=128, input width K=5376, b=2, and C=128 when T>1024 else64. Candidate head windows cover whole then smaller exact head groups, retaining all tokens and per-head reductions.

| Storage | Actual owner/shape |
|---|---|
| Full attended output | BF16 `[1,T,H,128]`, allocated once; each ABI writes its strided head view |
| Q/K/V projection outputs | Three BF16 `[T,h*128]` allocations, reused per head window |
| Shared projection quantization | INT8 `[T,K]` and F32 row scales, actually quantized once for unchanged hidden input |
| Projection workspace | Maximum of actual native GEMM workspace queries for every chosen Q/K/V window and tail; no estimate |
| INT4 Q/K codes | Two INT8 `[1,h,T,64]` allocations |
| INT8 V codes | INT8 `[h*128, ceil(T/C)*C]` |
| Q scales | F32 `[1,h,ceil(T/128)*32]` |
| K scales | F32 `[1,h,ceil(T/C)*4]` |
| K sums and V scales | Two F32 `[h,128]` allocations |
| Optional key-mean seed scratch | F32 `[h,128]`; the already-existing carry/seed destination is outside live storage |

The output plus seven INT4 scratch allocations are the existing eight ABI buffers from `attention_int4.py`. Q/K/V are normalized and rotated in place with the existing GLUE kernel. Contiguous rotary/norm representations must be materialized before admission and included as outside live holdings. No hidden `.contiguous()` or dtype conversion may occur inside the accepted interval. All workspace is allocated before output/seed writes.

The native split menu charges individually rounded real allocations, not their unrounded sum. Full/tail GEMM configuration and head geometry are part of choice identity; source/model/selected-backend identity is part of the reusable semantic key. Publication follows the current enclosing phase's completion proof, not Python return. Descriptor preparation is separate from execution, but currently binds and holds the actual call's weights. A future whole-phase inventory must extract metadata without binding every model block up front.

## Projection and hook seam that must remain real

Current `Rows` is not a trivial GEMM wrapper: it prepares readable weights, creates a weight-row view, runs the leaf's normal `linear` path, and then applies supported row hooks. The first implementation must call that actual path, with baked-hook eligibility matching the Tail consumer, rather than silently dropping `_hooked` or calling a copied projection formula.

The existing committed projection helper can launch into a supplied contiguous Q/K/V buffer. Its current `_LINEAR_SPLIT` interception is keyed to a module; `Rows._leaf` makes a shallow window copy, so simply reusing the Tail's module-identity interception will not match. Add a narrow prepared-row interception at that existing `Rows._leaf` boundary, bound to the exact original module, operands, row range and output; preserve the surrounding `readable` and `_hooked` flow. The native plan supplies queried dispatch, quantization buffers and workspace. It must avoid quantizing the same hidden input once per Q/K/V projection or head window; current sharing should remain one quantization of that unchanged input. This seam needs a genuine row-hook fixture before claiming actual H3 eligibility.

There is one concrete pre-existing hook ordering issue: `Rows._hooked` calls `hook.lora_factors()` before testing `hook._cozy_baked`, while normal `lora.Hook.__call__` returns immediately when baked. To make the admitted baked row path closed, move that same baked short-circuit ahead of factor lookup. Keep unbaked behavior unchanged and use a real baked Hook/Rows fixture to prove its factor callback is not entered. Merely assuming a baked Rows callback allocates nothing would leave an undeclared callback inside the accepted interval.

## Explicit remaining gaps

Kitchen's pinned SOL implementation does expose an exact native workspace query, `_C.sol_attn_plan`, and accepts an explicit workspace/output internally. Its ordinary wrapper also creates lengths and optional routing/coarse intermediates; its chunked producer additionally owns statistics, rope packing, norm conversion, callback allocations, and bootstrap passes. This is useful future structure, but the public Python call is not a closed allocation contract. [Pinned Kitchen CUDA source](https://raw.githubusercontent.com/Comfy-Org/comfy-kitchen/f8b7e97a1418d2e529bd5405fbf86312be198ae6/comfy_kitchen/backends/cuda/__init__.py).

Kitchen INT8 likewise exposes its buffers in its pinned Python adapter, including an anchor-index buffer, and invokes `_C.sage_sdpa`; its raw binding/kernel still needs a corresponding ownership audit and actual numerical fixture. Preserve that exact route for H3's exact/prefix calls. [Pinned Kitchen INT8 source](https://raw.githubusercontent.com/Comfy-Org/comfy-kitchen/f8b7e97a1418d2e529bd5405fbf86312be198ae6/comfy_kitchen/sage_attention.py).

Multirank is not merely a local head menu. `parallel.ulysses.exchange` performs padded-head and unequal-row collectives; `agree` fixes one common level per pass. No rank may independently change exchange count or retry after a collective starts. The initial native `_Heads` consumer must decline it; the existing exchange and sparse producer remain unchanged and explicitly unmigrated.

## Real qualification needed

Use the actual package `_Heads` with real encoded Q/K/V, real GLUE norm/RoPE/seed, actual selected INT4 route, and real Runtime Request/phase. Compare ordinary whole execution with admitted whole and constrained head variants, including odd tails and the two CTA-K geometries. Check complete output and seed, input preservation, actual projection hooks, declared-versus-allocated classes, and final zero ownership. Force minimum-capacity refusal before output/seed mutation; after acceptance, verify an error cannot enter legacy retry. The fixed raw `_attn4` ABI should remain unchanged. Full H3 then checks real eligibility, unchanged selected backend/numerical quality, native head choices, and disappearance of legacy `_Heads`/nested attention choices only for the explicitly supported dense INT4 calls.
