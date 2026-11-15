# Sol Engine H3: what is already ported, and the remaining source gaps

Inspected 2026-11-15. Source research only in this note: no new GPU run, checkpoint
download, deployment, or speedup claim. Owner: Codex `/root/optimize_h3_attention`.

Verified remote heads: NVlabs/Sana `sol-engine`
[`cf67a3acad7511acb85053e56a05e5dd24c82ba0`](https://github.com/NVlabs/Sana/tree/cf67a3acad7511acb85053e56a05e5dd24c82ba0),
its separate `main` head `27303b8de8e3f02f5dc96acf14705a0e04fca355`;
Cozy packages RC [`ba33c915d2a99206c6513f699f761a4c4d320d79`](https://github.com/cozy-creator/packages/tree/ba33c915d2a99206c6513f699f761a4c4d320d79)
and Runtime RC [`99d1efb6950de7341b7440d62b18481b5e0a7312`](https://github.com/cozy-creator/cozy-runtime/tree/99d1efb6950de7341b7440d62b18481b5e0a7312).
The project page links to **sol-engine**, so that branch is the upstream comparison.

## The 3.95× claim is mostly cross-step reuse

NVIDIA's [H3 report](https://nvlabs.github.io/Sana/Sol-Engine/H3/) gives this cumulative
warm, batch-one, eight-GB200 comparison at 1344×768, 124 frames and 50 steps:

| Arm | Wall time | Speedup over reference | Increment over preceding arm |
|---|---:|---:|---:|
| Diffusers reference | 27.205 s | 1.00× | — |
| Fusion and graph capture, including VAE | 19.514 s | 1.394× | 1.394× |
| Plus Sol sparse attention | 17.744 s | 1.534× | 1.100× |
| Plus cross-step residual reuse | 6.883 s | 3.95× | 2.578× |

The incremental ratios are calculated from the reported times. The page's
cache-without-sparsity configuration reports 3.16× and LPIPS 0.210; adding sparsity
reports LPIPS 0.293. These are approximation/quality tradeoffs, not evidence of
bit-identical native execution. The table is a five-second-class workload, not
Cozy's required 362-frame, 30-step regular qualification. It supplies no matched
comparison with the current Cozy implementation. [Reported conditions and decomposition](https://nvlabs.github.io/Sana/Sol-Engine/H3/).

## Hardware-specific implementations, not one megakernel

| Path | Actual implementation and comparison boundary |
|---|---|
| **GB200** | Resident BF16, Ulysses-8 in one NVLink domain; Triton elementwise/relayout kernels, exact AdaLN precomputation, Sol attention, batched/sharded VAE and optional compile. FBC threshold 0.08 and sparse routing are separate additions. [GB200 implementation](https://github.com/NVlabs/Sana/blob/cf67a3acad7511acb85053e56a05e5dd24c82ba0/models/minimax_h3/GB200/README.md#L18-L25) |
| **GB10 / DGX Spark** | Single-device, 480p, pruned FP8 DiT and FP8 conditioner. Uses Triton fusions/quantization and a Triton Sol path on SM121, plus FBC. This is a different memory/precision/workload regime from the GB200 row. Its FP8 linear respects the checkpoint's weight-only `fc2` exception. [GB10](https://github.com/NVlabs/Sana/blob/cf67a3acad7511acb85053e56a05e5dd24c82ba0/models/minimax_h3/GB10/README.md#L3-L25), [FP8 contract](https://github.com/NVlabs/Sana/blob/cf67a3acad7511acb85053e56a05e5dd24c82ba0/models/minimax_h3/GB10/fp8_linear.py#L1-L30) |
| **H100** | Pinned SGLang BF16/FSDP with Ulysses-4, CuTe SM90 Sol and FBC; its published profile explicitly omits `torch.compile`. H3 calls specialized SGLang activation, indexed-modulation and in-place QKNorm/RoPE kernels. [H100 profile](https://github.com/NVlabs/Sana/blob/cf67a3acad7511acb85053e56a05e5dd24c82ba0/models/minimax_h3/H100/README.md#L18-L24), [kernel imports](https://github.com/NVlabs/Sana/blob/cf67a3acad7511acb85053e56a05e5dd24c82ba0/models/minimax_h3/H100/model.py#L32-L43) |
| **RTX 4090** | BF16 SGLang with layerwise offload, regional compile, SM89 Sol and TeaCache. Its controlled table attributes 1951.27→613.71 s to caching, then 504.33 s with Sol. The authors explicitly report one timing sample and no perceptual metric; their “lossless opt” label does not claim byte or video-quality equivalence. [RTX4090 attribution and limits](https://github.com/NVlabs/Sana/blob/cf67a3acad7511acb85053e56a05e5dd24c82ba0/models/minimax_h3/RTX4090/README.md#L15-L40) |

The Sol attention interface dispatches CuTe implementations for SM89, SM90,
SM100 and SM120, otherwise Triton where supported. The host implementation remains
Python/PyTorch around specialized GPU kernels and CUDA-backed collectives/GEMMs;
the inspected paths do not provide a single whole-H3 CUDA megakernel.
[Architecture dispatch](https://github.com/NVlabs/Sana/blob/cf67a3acad7511acb85053e56a05e5dd24c82ba0/techniques/sparse_backends/sol_attn/interface.py#L9-L15),
[backend selection](https://github.com/NVlabs/Sana/blob/cf67a3acad7511acb85053e56a05e5dd24c82ba0/techniques/sparse_backends/sol_attn/interface.py#L72-L91),
[GB200 forward](https://github.com/NVlabs/Sana/blob/cf67a3acad7511acb85053e56a05e5dd24c82ba0/models/minimax_h3/GB200/ulysses_custom.py#L209-L263).

The newer **Sol-H3** directory is another algorithmic profile: FastH3's four
forwards versus 49 for its BF16 Diffusers/SGLang controls, with lossy transport and
sparse attention on multiple GPUs. Optional MXFP8 reports an additional 11.1%
latency reduction but 13.82 dB RGB PSNR/0.540 SSIM versus its own BF16 output. These
must not be presented as implementation-only wins for Cozy's 30-step model or
eight-step PDD adapter. [Explicit settings and quality limits](https://github.com/NVlabs/Sana/blob/cf67a3acad7511acb85053e56a05e5dd24c82ba0/models/minimax_h3/Sol-H3/README.md#L59-L87).

## Source details qualify the “lossless” label

GB200 `fused_qknorm_rope` explicitly retains incoming FP32 rotary tables where
eager casts them to BF16; its own comment says renders are not bit-identical.
The optional compiled GB200 VAE reports maximum absolute deviation 0.021.
GB10 takes a stricter approach: integer-domain BF16 intermediate rounding,
correctly rounded division, and explicit E4M3 round-to-nearest-even. These are
different numerical contracts. [GB200 RoPE](https://github.com/NVlabs/Sana/blob/cf67a3acad7511acb85053e56a05e5dd24c82ba0/models/minimax_h3/GB200/fusions.py#L247-L254),
[VAE compile](https://github.com/NVlabs/Sana/blob/cf67a3acad7511acb85053e56a05e5dd24c82ba0/models/minimax_h3/GB200/vae_shard.py#L30-L41),
[GB10 rounding](https://github.com/NVlabs/Sana/blob/cf67a3acad7511acb85053e56a05e5dd24c82ba0/models/minimax_h3/GB10/fusions.py#L1-L15).

Cozy already ports the stricter rounding, including BF16 rotary tables before
multiplication. Its source separately acknowledges small RMS-reduction differences;
it does not claim universal byte equality to every Torch kernel. Preserve that
contract in a new native implementation. [Cozy provenance/numerics](https://github.com/cozy-creator/packages/blob/ba33c915d2a99206c6513f699f761a4c4d320d79/minimax-h3/minimax_h3/glue.py#L1-L24),
[explicit quantizer rounding](https://github.com/cozy-creator/packages/blob/ba33c915d2a99206c6513f699f761a4c4d320d79/minimax-h3/minimax_h3/glue.py#L149-L163),
[RoPE rounding](https://github.com/cozy-creator/packages/blob/ba33c915d2a99206c6513f699f761a4c4d320d79/minimax-h3/minimax_h3/glue.py#L306-L323).

Also, the page's generic “QKV merge” entry is not the current GB200 execution:
`ulysses_custom.py` deliberately runs three GEMMs because its fused-QKV experiment
lost. Its packed **communication** is distinct from a fused **projection**. H100
does use a fused projection. The H3 implementation uses SwiGLU, despite the page's
generic GELU label. [GB200 decision](https://github.com/NVlabs/Sana/blob/cf67a3acad7511acb85053e56a05e5dd24c82ba0/models/minimax_h3/GB200/ulysses_custom.py#L219-L242),
[H100 projection](https://github.com/NVlabs/Sana/blob/cf67a3acad7511acb85053e56a05e5dd24c82ba0/models/minimax_h3/H100/model.py#L622-L638),
[actual SwiGLU](https://github.com/NVlabs/Sana/blob/cf67a3acad7511acb85053e56a05e5dd24c82ba0/models/minimax_h3/GB200/fusions.py#L116-L124).

## Already present in Cozy's current RC

| Sol mechanism | Cozy source and remaining distinction |
|---|---|
| RMS/AdaLN, residual gate, QKNorm/RoPE, SwiGLU fusion | Explicitly adapted from Sol GB200/GB10, with FP8 epilogues; INT8 down projection consumes SwiGLU directly. [Glue](https://github.com/cozy-creator/packages/blob/ba33c915d2a99206c6513f699f761a4c4d320d79/minimax-h3/minimax_h3/glue.py#L167-L323), [FFN](https://github.com/cozy-creator/packages/blob/ba33c915d2a99206c6513f699f761a4c4d320d79/minimax-h3/minimax_h3/glue.py#L1156-L1173) |
| Reuse one quantized activation across Q/K/V | Current RC includes #920's explicit inference/capture `quantize_once` scope for head windows. This is not a missing three-times-quantization optimization anymore. [Scope](https://github.com/cozy-creator/packages/blob/ba33c915d2a99206c6513f699f761a4c4d320d79/minimax-h3/minimax_h3/glue.py#L1042-L1064) |
| Precomputed modulation | AdaLN-pruned checkpoints already carry exact schedule-owned tables. FULL artifacts still have official projections. Sol's precomputation is therefore not a generic missing optimization; a FULL-only proposal must first prove that lane is the expensive one actually serving. [Artifact-specific contract](https://github.com/cozy-creator/packages/blob/ba33c915d2a99206c6513f699f761a4c4d320d79/minimax-h3/minimax_h3/adaln_pruned.py#L1-L12), [table lookup](https://github.com/cozy-creator/packages/blob/ba33c915d2a99206c6513f699f761a4c4d320d79/minimax-h3/minimax_h3/adaln_pruned.py#L128-L164) |
| Batched/sharded VAE | Spatial tiles already batch; temporal chunks already use `spread` across ranks, with original ordered blending and bounded blockwise decoding. Sol instead shards spatial tiles. Neither generic batching nor multi-GPU decode is absent. [Cozy temporal spread](https://github.com/cozy-creator/packages/blob/ba33c915d2a99206c6513f699f761a4c4d320d79/minimax-h3/minimax_h3/vae_tiles.py#L77-L105), [tile batch/blockwise decode](https://github.com/cozy-creator/packages/blob/ba33c915d2a99206c6513f699f761a4c4d320d79/minimax-h3/minimax_h3/vae_tiles.py#L159-L218) |
| Optional residual reuse | Current `full` uses EasyCache, `draft` uses FBC, `standard` has no cache; Turbo uses its own dense-step recipe. These are explicit quality recipes, not free kernel gains. [Current recipes](https://github.com/cozy-creator/packages/blob/ba33c915d2a99206c6513f699f761a4c4d320d79/minimax-h3/h3.py#L820-L831) |
| Hardware-specific attention | Runtime already imports Sol, Kitchen and FA4 variants; datacenter Blackwell can combine Sol routing with FA4 exact tiles. Changing that route changes the comparison, even at equal bit width. [Sol/FA4 contract](https://github.com/cozy-creator/cozy-runtime/blob/99d1efb6950de7341b7440d62b18481b5e0a7312/src/cozy_runtime/internal/attention_sol_fa4.py#L1-L14) |

The upstream Sol source pin is `8e0db4fa...`. Comparing complete GitHub trees
(both untruncated, 69 entries under `techniques/sparse_backends/sol_attn`) against
`cf67a3a...` finds **no attention-kernel blob changes**. Only `preprocess.py` and
`triton_ref/preprocess.py` changed: they disable autocast around the key
second-moment GEMM. That is a dtype-correctness fix, not a hidden newer attention
schedule to obtain by bumping the pin. [Runtime pin](https://github.com/cozy-creator/cozy-runtime/blob/99d1efb6950de7341b7440d62b18481b5e0a7312/src/cozy_runtime/internal/kernel_sources.py#L206-L212),
[current preprocessing fix](https://github.com/NVlabs/Sana/blob/cf67a3acad7511acb85053e56a05e5dd24c82ba0/techniques/sparse_backends/sol_attn/preprocess.py#L393-L404).

## Concrete implementation candidate: remove an unneeded receive materialization

For an **unquantized, batch-one** Ulysses head group, Runtime allocates `values`
for received Q/K/V and another `operands` tensor, then copies the transposed
receive tensor into `operands`. With batch one, that transpose is a compatible
`[1, total_tokens, local_heads, head_dim]` view. Sol GB200 consumes views of its
received buffer directly. This identifies an avoidable-looking copy/allocation,
not yet proof that removing it is safe or faster.
[Runtime allocation/copy](https://github.com/cozy-creator/cozy-runtime/blob/99d1efb6950de7341b7440d62b18481b5e0a7312/src/cozy_runtime/internal/parallel/ulysses.py#L381-L432),
[Sol receive views](https://github.com/NVlabs/Sana/blob/cf67a3acad7511acb85053e56a05e5dd24c82ba0/models/minimax_h3/GB200/ulysses_custom.py#L241-L242).

The candidate can save one Q/K/V allocation of
`3 × batch × total_tokens × local_heads × head_dim × element_bytes`, and one read
plus one write of those bytes **per eligible head group**. For 38,247 tokens,
seven local heads, D128 and BF16 this is 205,615,872 bytes of storage and twice
that in copy traffic. This is arithmetic from an illustrative five-second shape,
not an observed time saving or the actual grouping of a current Cozy run.

The follow-up must inspect receive-buffer reuse, attention readers, allocator
custody and cross-stream event ordering before changing ownership. Preserve
Runtime's head-group pipeline and memory accounting. The `narrow=True` path
performs an inverse quantized transform and cannot simply skip `turn.widen`.
Hold the kernel, wire precision, routes and group partition fixed for the first
comparison; test uneven shards and failure cleanup. CPU shape/value equality can
screen this candidate, but actual multi-GPU execution is required before a native
correctness/performance claim. [Pipeline lifetime](https://github.com/cozy-creator/cozy-runtime/blob/99d1efb6950de7341b7440d62b18481b5e0a7312/src/cozy_runtime/internal/parallel/ulysses.py#L724-L757),
[quantized transform](https://github.com/cozy-creator/cozy-runtime/blob/99d1efb6950de7341b7440d62b18481b5e0a7312/src/cozy_runtime/internal/parallel/turn.py#L1-L10).

A secondary candidate is one destination-major QKV pack/collective on the NCCL
path: Runtime currently packs each source separately and submits one collective
per source/scale. Sol has a single-pass Triton pack and one variable-length
collective. Its own earlier `stack`/`cat` implementation regressed, so merely
combining collectives is insufficient. Runtime already pipelines and may use a
copy-engine transport, so adopt this only where its actual trace shows exposed
NCCL packing/launch cost. [Runtime submissions](https://github.com/cozy-creator/cozy-runtime/blob/99d1efb6950de7341b7440d62b18481b5e0a7312/src/cozy_runtime/internal/parallel/ulysses.py#L207-L245),
[Sol direct pack and prior regression](https://github.com/NVlabs/Sana/blob/cf67a3acad7511acb85053e56a05e5dd24c82ba0/models/minimax_h3/GB200/relayout.py#L1-L27).

No generic native rewrite follows from this comparison. Preserve the merged
[no-win exact-tile screens](https://github.com/cozy-creator/cozy-runtime/pull/2185),
and investigate the specific redundant materialization before further blind
stage/cache tuning. A useful result still needs the ordinary 15-second H3 request,
unchanged model assets, schedule, precision, audio and output criteria. Distilled
solvers, additional approximation and alternative VAEs require separate quality
decisions and cannot qualify that result.
