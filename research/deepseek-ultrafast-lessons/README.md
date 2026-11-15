# DeepSeek Ultrafast lessons for Cozy Creator

Research date: 2026-11-15. Status: source investigation complete; recommendations are experiments, not measured speedups. No new GPU execution or production changes.

Owner: Codex `/root`. Purpose: evaluate model/hardware specialization, fused CUDA kernels, CUDA graphs, and transferable H3 improvements. Branch: `codex/deepseek-h3-kernel-research-20261115`. Runtime base: `af2a4753cb70598b75405942901e4f8f3b24eda6`. Worktree: `/home/fidika/cozy/.worktrees/cozy-runtime/deepseek-h3-kernel-research-20261115`.

Source snapshots:

- [DeepSeek Ultrafast](https://github.com/basetenlabs/deepseek-ultrafast/tree/76766a97308bfd1658fb055e835d0f92de737dc3)
- Runtime `af2a4753cb70598b75405942901e4f8f3b24eda6`
- Packages `e4b179bbd099f681658dc7ef5a059f484969cf94`
- Creator `b7fb810bdfdfb5a9f8a0d55d3d0c13c42fea4cc9`
- Tracker `4666c6808461d1463b1e191fa62fa9117a5b14b8`

## Findings

Follow-up: [Hyperkernel compared with Baseten, and what a native Cozy execution path would mean](hyperkernel-comparison.md).

Further source studies:

- [Existing native H3 engines: mmh3, h3-rs and h3-hrx](native-h3-engines.md).
- [NVIDIA Sol H3: measured decomposition, existing Cozy ports, and remaining candidates](sol-h3-gap-analysis.md).
- [A native H3 stage within Cozy: capture lifecycle, current interfaces, proposed private boundary, and profiling recipe](cozy-native-h3-stage.md).

The useful pattern is to specialize the execution schedule and data layout for a model, workload, and hardware class. Baseten implements a graph of specialized kernels, with selected fusion and persistence; its source does not establish a single whole-model megakernel. Nor does its benchmark isolate Rust versus Python or CUDA C++ versus a GPU DSL. The [external source study](external-study.md) identifies the launch paths and their restrictions.

Cozy already implements much of this pattern. The next useful H3 investigation is the expensive attention and GEMM paths and the materialized tensors between them. A wholesale language rewrite has no measured justification here.

## How much faster is Baseten than other runtimes?

No controlled Baseten-versus-SGLang/vLLM A/B was found in the repository, its one public issue, or the primary performance reports examined. The closest published same-model, same-GPU-type/count pair is:

| Implementation | Hardware | Reported single-user decode | Conditions |
|---|---|---:|---|
| Baseten Ultrafast | 4×GB300, TP4 | 1,740 tokens/s | Tetris HTML prompt, thinking disabled, real speculative acceptance averaging about 5.6 tokens/step |
| SGLang optimized revision `8394e82d` | 4×GB300, TP4/EP1 | 1,193 tokens/s/user | Random 4,096-token inputs, 1,024-token outputs, simulated acceptance 5.4, median of three runs; reported September 28 |

Sources: [Baseten benchmark](https://github.com/basetenlabs/deepseek-ultrafast/blob/76766a97308bfd1658fb055e835d0f92de737dc3/README.md), [SGLang team report](https://staging.lmsys.org/blog/2026-09-28-deepseek-v41-optimization). The arithmetic ratio is **1.459×, or 45.9% higher reported decode rate**. It is not a measured runtime speedup: prompt/context, sampler, real versus simulated acceptance, timing aggregation, and software/checkpoint revisions are not controlled. This older SGLang result is not a claim about today's best SGLang configuration. Baseten's 2,060 tokens/s result uses eight B300 GPUs; comparing it to four-GPU results would also change hardware resources.

There is stronger evidence for specialization within existing runtimes:

- [SGLang PR #45450](https://github.com/sgl-project/sglang/pull/45450): same 220 AgentX requests on 4×GB300, with synthetic acceptance fixed at 3.51, improve from 694.0 to 995.7 tokens/s/user P50 (+43.5%) with the plugin and execution changes. The P90-rate improvement is +43.9%. The plugin alone at its measured default slots gives +9.8% P50; the host changes alone give +2.3%. The combined result also enables additional plugin slots, so this is not an isolated language or kernel-fusion ablation.
- [vLLM's October 23 report](https://vllm-project.github.io/2026/10/23/deepseek-v41-flash-pd-wide-ep.html): its specialized decode path improves the same model from 575 to 826 P90 tokens/s/user (+44%) on **four B300s**, using AgentX. That is evidence for the vLLM change; it is not a Baseten comparison against either four GB300s or eight B300s.

To measure Baseten's advantage, run all implementations on the same physical four-GB300 allocation or eight-B300 allocation, with a pinned checkpoint/tokenizer, matching precision and sampling, and both a common real prompt corpus and the demo prompt. Report decode, prefill, TTFT, request completion, acceptance length and step time separately, at concurrency 1 and a concurrent workload supported by every engine. Use real acceptance for the user-visible result; a fixed-acceptance control can isolate execution cost but must be labeled synthetic. No rental or such benchmark was performed in this study.

## What Cozy already has

| Mechanism | Current source evidence | Consequence for this proposal |
|---|---|---|
| H3-specific fused kernels | [glue.py](https://github.com/cozy-creator/packages/blob/e4b179bbd099f681658dc7ef5a059f484969cf94/minimax-h3/minimax_h3/glue.py#L165) fuses RMSNorm/modulation, optional residual work and FP8 production; other entries fuse SwiGLU and norm/RoPE | Do not propose these as missing. Measure the remaining boundaries. |
| Quantization in activation producers | [glue.py](https://github.com/cozy-creator/packages/blob/e4b179bbd099f681658dc7ef5a059f484969cf94/minimax-h3/minimax_h3/glue.py#L1118) and [Runtime's public kernel contract](https://github.com/cozy-creator/cozy-runtime/blob/af2a4753cb70598b75405942901e4f8f3b24eda6/src/cozy_runtime/accel/kernels.py#L1) already support prequantized inputs and INT8 activation prologues | Preserve the executed format and rounding when comparing another implementation. |
| Memory-bounded execution | [H3 block tail](https://github.com/cozy-creator/packages/blob/e4b179bbd099f681658dc7ef5a059f484969cf94/minimax-h3/minimax_h3/glue.py#L1055) splits exact per-token FFN work and reuses the attention-output storage | Large fusion must retain bounded working memory; resident datacenter assumptions do not fit every Creator GPU. |
| Hardware-specific selection | [arch.py](https://github.com/cozy-creator/cozy-runtime/blob/af2a4753cb70598b75405942901e4f8f3b24eda6/src/cozy_runtime/internal/arch.py#L1) distinguishes MMA instruction families and precision capabilities; [attention policy](https://github.com/cozy-creator/cozy-runtime/blob/af2a4753cb70598b75405942901e4f8f3b24eda6/docs/attention-selection.md) dispatches by class/site | Extend those decisions where measurements warrant it; do not introduce GPU-brand-name special cases. |
| Native shipping of DSL kernels | [TensorFS build.py](https://github.com/cozy-creator/cozy-creator/blob/b7fb810bdfdfb5a9f8a0d55d3d0c13c42fea4cc9/crates/tensorfs-py/kernels/build.py) compiles Triton to architecture-specific cubins and generates an embedded Rust catalog | A GPU DSL need not be a serving-time interpreter or JIT dependency. |
| Graph replay | [stable Runtime graph source](https://github.com/cozy-creator/cozy-runtime/blob/af2a4753cb70598b75405942901e4f8f3b24eda6/src/cozy_runtime/graphs/__init__.py#L1); [active cr-297 work](https://github.com/cozy-creator/tracker/blob/4666c6808461d1463b1e191fa62fa9117a5b14b8/tracker/cozy-runtime/cr-297-cuda-graphs-first-class.md) | The stable source and the in-progress RC work differ. Coordinate with cr-297 rather than implementing a competing graph system. |

H3's current [denoising loop](https://github.com/cozy-creator/packages/blob/e4b179bbd099f681658dc7ef5a059f484969cf94/minimax-h3/minimax_h3/steps.py#L43) already hoists rotary/text preparation and distinguishes dense, seeding and sparse work. Its [VAE plan](https://github.com/cozy-creator/packages/blob/e4b179bbd099f681658dc7ef5a059f484969cf94/minimax-h3/minimax_h3/model.py#L59) also declares fused transformer blocks. These are existing mechanisms, not proposed improvements.

## What the recent H3 evidence suggests

Selected facts, source paths and SHA-256 hashes are retained in [h3-profile-observations.json](h3-profile-observations.json). These are existing diagnostic runs, reread for this study, not newly run benchmarks.

The cr-297 session `43442514074`, cell `m-h1`, is a 15-second Turbo request on SM89/RTX4090, with the measurement Runtime `b41d433fc122fb6e7c730d3c4172cfd0f4592743`. Its sampled dense call reports 49.45 s wall, 38.36 s of Kitchen attention events and 7.43 s of CUTLASS GEMM events. The sampled sparse call reports 26.79 s wall, 9.96 s of Sol exact-tile events, 3.86 s of dense attention events and 7.72 s of GEMM events. This makes attention and GEMM the first candidates to reprofile on the intended release. It does not establish current timings, hardware saturation, or transferable speedup percentages.

**Measurement correction:** the [measurement helper](https://github.com/cozy-creator/cozy-runtime/blob/b41d433fc122fb6e7c730d3c4172cfd0f4592743/src/cozy_runtime/graphs/profiles.py) sums CUDA event durations. Its dense `busy_ms` exceeds wall time; these are not unioned active intervals. Thus the tracker's “≤4.4% gaps” cannot be used as a rigorous launch-overhead ceiling. The same profile dictionary is also repeated under multiple component records; it must be counted once. A clean trace with interval union and the critical path is needed to quantify idle time.

Existing cr-297 records also show why memory belongs in the optimization:

- Session `43476403923`, forced H3 on the 32 GB class, recorded zero captures and a refusal involving a 12.1 GiB capture allocation. Reducing scratch and intermediate lifetimes can matter even if individual launches are cheap.
- Session `43683255718`, forced H3 on the 96 GB class, recorded zero captures because an overlay component lacked a prepared binding. That is an ownership/integration finding, not a measured graph speedup. cr-297 already owns the follow-up.

## Recommended experiments, in order

### 1. Specialize H3 attention where the trace spends time

**Candidate:** the existing head-dimension-128 dense and sparse exact-tile kernels, including their Q/K/V preparation, protected rows, sequence/window geometry and output layout. Start on one target hardware class, then remeasure another. The current source and receipt already identify these sites; do not replace the whole runtime.

**Experiment:** pin the baseline recipe and precision, profile full-length regular and Turbo requests separately, identify actual tensor-core, bandwidth, occupancy and spill limits, then compare one change at a time: tile/warp schedule, load pipeline, or a producer that writes the consumer's packed layout directly. Keep sparse selection and protected rows identical. Validate both one-GPU and actual multi-GPU execution before changing defaults.

**Why first:** this is the dominant operation family in the available H3 sample. The size of a possible gain is unknown. Lower-bit attention or fewer diffusion steps would be a separate quality-changing experiment; the current [attention policy](https://github.com/cozy-creator/cozy-runtime/blob/af2a4753cb70598b75405942901e4f8f3b24eda6/docs/attention-selection.md) specifically keeps four-bit attention out of automatic cells after detail loss.

### 2. Remove large intermediate writes across the remaining GEMM boundaries

**Candidates:** projection → norm/RoPE → attention packing, and projection → SwiGLU/quantization → down projection. Existing norm/FP8 fusion and INT8 SwiGLU prologues are the baseline, not work to redo. The [FP8 SwiGLU implementation](https://github.com/cozy-creator/packages/blob/e4b179bbd099f681658dc7ef5a059f484969cf94/minimax-h3/minimax_h3/glue.py#L247) deliberately sweeps/recomputes a wide row twice to avoid register spills; that is a specific scheduling tradeoff worth measuring, not proof it is inefficient.

**Experiment:** compare the present implementation with a tiled CUDA/CuTe/Triton alternative that preserves the whole-row scale and eager rounding. Measure bytes moved, spill traffic, peak working memory, operation time and complete video time. A fused epilogue must account for cross-tile reductions; changing a row scale to a tile scale silently changes the computation.

**Expected benefit:** reduced memory traffic or scratch may help both device execution and residency. Do not expect a large whole-model gain from a sub-percent glue kernel merely because its microbenchmark improves.

### 3. Ship a small set of measured execution variants through existing contracts

**Candidate:** model-private schedules for attention/GEMM shapes, GPU instruction class, precision, parallel topology and bounded working-memory windows. Use H3's existing package `Plan` and Runtime's kernel store, native leaves, graphs and Varena ownership. Keep the model-facing author code ordinary; specialize its implementation under the existing interfaces.

Tune during development, select deterministic variants, and ship their fixed sources or prebuilt cubins through the existing package/native wheel paths. TensorFS already demonstrates Triton-to-cubin-to-Rust. Prebuilt code still accepts runtime pointers; CUDA graph instances bind process-local addresses and remain Runtime/Varena-owned. No worker-image qualification, new admission inventory, or promotion mechanism is proposed.

### 4. Optimize communication only where the interconnect and trace justify it

Baseten's tensor-parallel decode uses a different workload and communication pattern from H3's [Ulysses attention](https://github.com/cozy-creator/cozy-runtime/blob/af2a4753cb70598b75405942901e4f8f3b24eda6/docs/attention-selection.md). Its NVLink multicast and small-message reduction techniques are not drop-in replacements for H3's sequence/head exchanges on PCIe cards.

**Experiment:** measure exchange/packing/compute overlap on the actual topology; test direct packing to the exchange layout, fewer copies, and overlap using existing collectives where dependencies permit. Preserve numerical partition behavior and buffer lifetimes. Retain an ordinary path for hardware without the required instruction/interconnect support. PDL is an option only on architectures that support it and where readiness is explicitly correct.

## What not to infer

- **“Handwritten CUDA beats DSLs.”** Both can produce specialized GPU machine code; CUDA or inline PTX is useful when it enables a measured scheduling/instruction choice the other implementation cannot express or compile well. Compare generated code and execution, not source language labels. See the primary DSL documentation in the external study.
- **“One huge kernel must be faster.”** Larger fusion can increase live registers, shared memory and synchronization, reducing occupancy or causing spills. H3 already documents this tradeoff in its wide SwiGLU kernel. Baseten itself selects a multi-kernel pipelined MoE path for important shapes.
- **“The DeepSeek multiplier applies to H3.”** DeepSeek's accepted draft tokens amortize autoregressive decoding. H3 updates large video/audio tensors through denoising steps. Turbo/distillation and step caching change its algorithm/quality contract; they cannot be counted as lossless kernel improvements.
- **“A high utilization percentage proves optimality.”** Device activity does not reveal tensor-core efficiency, bandwidth efficiency, spill costs, or time spent moving data.
- **“Rewrite all orchestration in Rust.”** First isolate CPU gaps on the critical path. Graph replay can remove dispatch overhead without such a rewrite, while H3's measured expensive GPU operations remain.

## Qualification for a useful Creator improvement

Use the ordinary end-user path with the same model assets, runtime/package commits, precision, attention recipe, seed, frame/resolution/audio settings, GPU degree, and memory conditions. The regular H3 workload remains 15 seconds, 362 frames, 1344×768, 30 steps with audio; test Turbo separately at its own schedule. Preserve cold-start and steady-state results separately. Include a memory-constrained card and a resident configuration, without changing request geometry to make a result pass.

Record the kernel and artifact that actually served, denoise/attention/GEMM/decoder/conditioning and complete request time, peak memory, transfer bytes, and output parity. Require repeatable full-model improvement with either exact output preservation for the asserted exact transformation or an explicitly accepted numerical/quality criterion. A change to reductions must not be described as bit-identical without proving it. Inspect pre-codec video/audio and latents where needed; valid media alone is not a quality comparison.

The immediate recommendation is a focused H3 attention/remaining-GEMM-boundary study integrated with the current graph owner. The evidence supports continuing specialization; it does not yet support a claimed H3 speedup, a wholesale megakernel, or abandoning GPU DSLs.
