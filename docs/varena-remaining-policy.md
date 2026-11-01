# Remaining CUDA sizing and operation retries

This inventory uses completed full H3 run **25499**, 15 seconds and 30 denoising steps.
The source receipt is `research/varena-p2-p3-integration/h3-25499-policy-inventory.json`.
It is a historical trace, not proof that this combined Runtime has run. Its child call
records all counts below; the public parent has an empty cut ledger. Counts are nested
wrapper calls and must not be added as independent GPU operations. No terminal OOM,
recorded same-level retry, or recorded descent occurred at these sites in this run.

## Observed routes and present source

| Recorded site | Calls in 25499 | Current boundary |
| --- | ---: | --- |
| `split:_Heads(1,109123,-1,128):bfloat16` | 900 (899 whole, 1 level 1) | Sparse producer is now implemented by the complete native consumer from 167; its 26-case proof is separate from combined full-model adoption. |
| `norm_out.norm:tokens` | 150, all whole | Exact Torch RMSNorm output+rstd admission from 3043 replaces this estimate where its dtype/layout/SDK guards hold. Six bounded cases qualify the operator. |
| `split:NativeLinear(1,-1,7168):bfloat16` | 50 (48 whole, levels 9 and 10 once each) | Actual H3 private output producers now declare first-use ownership. The general shared/learned-group guard remains. This needs package 5d27, not the earlier fc4 package alone. |
| `token_refiner.*` RMS norms | 9, each once | The same exact RMSNorm adapter is available, but actual integrated adoption must be counted. Do not assume every refiner layout is eligible. |
| `post_quant_conv:bands` | 22, all whole | Still `cuts._conv_bands` → `_conv_need` → `sizing.conv/peak` → `catch.cut`. The module is an actual 1×1×1 Conv3d, but the current autocast/cuDNN workspace is not a declared native allocation contract. |
| `split:TileBatchedVideoVAE._decode_tiles(-1,24,7,16,16):float32` | 21 (20 whole, 1 level 1) | Still generic `author.split`/`cuts.split`, measured-window fitting and Python row selection over the complete decoder. |
| `split:_BlockRows(-1,1797,2048):float16` | 22 (21 whole, 1 level 1) | Still generic batch splitting of the first fused decoder block. |
| `split:_BlockRows(-1,1797,2048):float32` | 770 (769 whole, 1 level 1) | Still generic batch splitting of the later fused blocks and their FP32 residual streams. |
| The two `_BlockRows/.../attention#1` sites | 23 FP16 + 771 FP32 | Still `attention_ulysses._call` → anonymous callable → `cuts.attend` → `sizing.Attention`/`catch.cut`. Actual selected VAE attention is **Kitchen INT8**, not an inferred SDPA/math route. |
| `token_refiner.refiner_blocks.{0,1}.ff:tokens` and `.ff.net.0:tokens` | 4, each once | Still `_ffn_shape`/`sizing.Ffn` and adaptive token replay. The full FFN and activation wrappers are nested. |
| Refiner `attention#1` and nested `attention#1/attention#1` | 4, each once | Still generic backend attention sizing/retry, with a nested wrapper. |

The VAE counts explain the concrete producer boundary: the outer decoder is invoked
22 times after its one batch split. Each invocation visits 36 fused blocks, giving
792 `_BlockRows` calls. One FP16 and one FP32 block call split in two, so there are
794 nested attention calls. This is not evidence of 794 independent decoder passes.
The trace reports `fused.minimaxh3videotransformerblock.85ec19d8887d` applied to all
36 blocks and `vae=kitchen-int8` as the observed backend.

## What Python still decides

`execution/sizing.py:choose` compares predicted demand with `room.free + room.cached +
room.cold`, then uses a 1/64 knee at the finest geometry. `cuts.split` fits a linear plus
quadratic demand curve through two observed windows. Those observations do not declare
a complete current execution and cannot become a committed phase key.

`catch.cut` still chooses levels and may execute a finer level after an allocation
refusal. The integrated CUDA `Request.reclaim` returns zero, so its two old same-level
Python victim-reclaim retries are skipped. This does **not** remove the outer finer-split
operation replay. Native consumers keep their no-retry markers. No additional retry was
observed in 25499, but the routes remain callable.

`attention_ulysses._call` passes the selected kernel through a lambda and does not forward
its identity/allocation contract to the sizing layer. `cuts.attend` therefore uses its
default `scratch="flash"` unless another caller supplies it. Changing that string to
`"sage"` would remain an estimate, not a migration.

`parallel/ulysses.py:_level` still proposes a head group from Python send/gather bytes and
local room. No multi-rank execution occurred in this trace. Its pass-stable collective
agreement must be replaced by an explicit common choice protocol, not removed on the
strength of a single-GPU test.

The noncooperative legacy Room API still has `_room_floor` (peak/backing, pins, and three
largest blocks), `_apply_room`, and `room_for`. The v3 cooperative Memory rejects that
controller. These are a remaining compatibility cutover, not an active second byte
policy to re-enable in the new owner path. Static development ceilings are constraints;
host-tier `_admit`/`_displace` are host-RAM source caching, not CUDA working selection.

The package's `adaln_pruned._CHUNK=32768` packing/output-head windows are another
memory-motivated constant outside the cut ledger. By contrast, the sparse producer's
4096-token chunk is part of its reviewed numerical/reduction ABI; tile geometry and
cross-fade order are also model semantics. Do not relabel all fixed geometry as a memory
heuristic or change it to make a test fit.

## Next necessary producer migration

The next substantial boundary is the actual **fused VAE `_BlockRows` operation**, with
an explicit tile-batch menu before its first normalization or projection. It must retain
both observed residual arms (FP16 first block, FP32 thereafter), FP16 autocast,
FP32 weights/gates/norm arithmetic where the package declares it, and the existing
round points. `minimax_h3.model.VAE_BLOCK` already declares width 2048, 32 heads of 64,
48-channel split-half RoPE, packed value-first SwiGLU and the layer scales. The Runtime's
`fusion._block_forward/_sequential/_attend/_mlp` is the executed body; an unfused upstream
block would be the wrong allocation oracle.

Before that producer can claim closure, close two specific provider contracts:

1. Preserve the actual selected Kitchen INT8 identity through dispatch and use the pinned
   caller-buffer ABI. For D=64 there is no head padding; the actual Q/K/V layouts, quantized
   arrays, Q/K/V scales, anchors, output layout and any joining copy must be declared for
   every supported batch/head window. Existing H3 INT8 code is B=1/D=128-specific and must
   not be reused by guessing its sizes. VAE attention operands are FP16 even when the
   surrounding `_BlockRows` residual is FP32.
2. Establish the actual dense-linear/autocast allocation contract used by Q/K/V/output and
   the packed up/down projections. Include actual FP32-to-FP16 weight/bias copies where
   present, result storage and the selected cuBLAS/cuBLASLt workspace/initialization
   ownership. No workspace is inferred from a successful warm trace. A provider query or
   reviewed explicit-workspace path must establish it before admission.

The complete block inventory must also represent `_layer_scales`' one CUDA integer index,
normalization/RoPE outputs, the FP32 residual lifetimes and the SwiGLU intermediate. Source
binding remains per actual block; inventory must not pin all 36 blocks. The original
Triton entries and selected attention/math remain unchanged. A real fixture should use
the installed fused block at N=1797, both residual dtypes, actual weight dtypes, whole and
smaller tile batches, actual allocator census, complete-output comparisons, minimum refusal
before writes and checked cleanup. No accepted operation may return to the legacy parent
retry.

Only then can the outer tile decoder become a declared sequence over those blocks plus
its real post-quant Conv3d, packing, final normalization/projection and pixel assembly.
Closing only the inner attention while leaving `_BlockRows`' competing parent chooser
would not retire the outer policy. The small refiner FFN/attention routes and multi-rank
agreement remain separate, concrete follow-ups. No generic owner-inventory guard or new
Python reserve formula is proposed here.
