# H3 on RTX 5090

Owned benchmark: one GPU first, then four GPUs. Use ordinary default-home `cozy run`
on an explicitly selected rental. Keep the local display GPU unused.

The request is FP8-pruned H3, 15 seconds, 1344×768, 24 fps and 30 steps with audio.
Record cold preparation separately from request execution. Repeat with three seeds;
compare each engine on the same hardware, prompt, geometry, schedule and precision.
Report the actual attention implementations and checkpoint-format differences.
Do not shorten the request to make memory fit.

ComfyUI baseline: v0.39.0 (`b0b743566f65daafc423b4fea8a2fbda94b3384a`).
This was the latest published release when checked on 2026-10-08. Refresh the
official latest-release endpoint before a future comparison and record its version.
`setup-comfy.sh` installs its isolated environment and the FP8 DiT, BF16 text encoder,
FP16 video VAE and FP32 audio VAE from the pinned Comfy-Org checkpoint revision.
Run it only on the owned rental, after extracting ComfyUI into `/root/h3-bench/comfy`.
The saved `comfy-v0.39.0.tar.gz` has root-level files: extract it without
`--strip-components`. The buqi archive has a top-level directory and needs
`--strip-components=1` when extracting into its custom-node directory.

Single-GPU uses stock block sparse attention: tau 1, first ten evaluations and
blocks 0/1 dense, exact conditioning rows/KV, with Comfy Kitchen INT8 dense attention.
`comfy-run.py INPUT SIGMAS OUTPUT --world-size 4 --dense` uses the
[buqi multi-GPU node](https://github.com/buqi-code/buqi-minimax-h3-multigpu)
at `a864277f81ee7609972eb64cd1fb086f239d4c99`. Place that checkout in
`comfy/custom_nodes/buqi-minimax-h3-multigpu` and start ComfyUI with
`MINIMAX_SP_DEVICES=0,1,2,3 MINIMAX_SP_STREAM=1`, normal memory management and
`--cuda-device 0,1,2,3`. The plugin rejects sparse attention and model attention
overrides. Its workers do not propagate `--use-ck-attention`, so this baseline
uses native PyTorch dense attention on every rank. Do not claim an algorithm-matched
speedup against sparse Cozy from this run. The BF16 text encoder and both VAEs
remain on ComfyUI's primary GPU; sequence parallelism applies only to the DiT.

Evidence and ownership are under
`/home/fidika/cozy_v2/outputs/h3-5090-benchmark-20261007/`.
Preserve the original video/audio, hashes, execution events and quality-review
receipts. Compare clean warm executions; exclude overlap with installation,
downloads or another engine's retained memory. Quantized attention timings do not
establish quality equivalence with the native BF16 attention reference.

Turbo uses eight Euler evaluations, with the first four dense and the last four
Sol-eligible. The full request remains 1344×768 with audio: a 15-second request
produces 362 frames at 24 fps (15.083 seconds). Keep the PDD node's exact sigma
output, CFG 1, video/audio shifts 12/3, tau 1, dense blocks 0/1 and exact
conditioning rows/KV.

The four-GPU Turbo graph uses unmodified
[MultiStream](https://github.com/martonsagi/Comfy-H3-MultiStream/tree/4e4381ff2cdb061a4cd0065af5b9c3679ded379a)
and [PDD Apply](https://github.com/Jalen-Brunson/ComfyUI-MiniMax-H3-PDD-Acc/tree/311a65dd53832d8a5f8177a9d5fb923c09e35a90).
The adapter is the official Alibaba `MiniMax-H3-FL2VA-Acc-8Step.safetensors` at
revision `335001fb9e5455d68a0caa18ec2e319072150328`, SHA-256
`0b29be7042d883970eb0c20774a9ba03d95669ed80a721bb4d21be8ea0d0a196`.
It is not the differently trained LightX2V Turbo adapter. MultiStream uses host
exchange with eight chunks, normal DynamicVRAM and its pinned weight cache.
The optimized action cohort additionally uses `H3MSTextEncoderCache` with
`cond_cache_entries=0` and `H3MSVAESplitDecode` on four GPUs. Every timed request
recomputes conditioning and all eight sampling evaluations.

| ComfyUI cohort | Warm executions | Warm mean |
| --- | ---: | ---: |
| One 5090, 30 steps, stock Kitchen dense/Sol | 943.413 / 934.858 s | 939.1355 s |
| Four 5090s, 30 steps, buqi dense PyTorch | 857.778 / 858.285 s | 858.0315 s |
| Four 5090s, Turbo 4/4, lighthouse, stock decode | 251.279 / 195.520 s | 223.3995 s |
| Four 5090s, Turbo 4/4, optimized action, affected cuBLAS 13.1.1.3 | 179.180 / 168.143 s | 173.6615 s |
| Four 5090s, Turbo 4/4, optimized action, fixed cuBLAS 13.2.0.9 | 184.584 / 183.259 s | 183.9215 s |

These are queue-free server `execution_start` to `execution_success` timings,
excluding output transfer. The action warm samples are sparring (seed 581204) and
rally racing (581205), after a parkour setup sample (581203). Compare them only
against matching native action requests. The lighthouse Turbo warm samples differ
because the first rebuilt text-encoder state after eviction: conditioning took
58.36 seconds, then 1.10 seconds. Both are reported. The optimized action setup
sample took 257.509 seconds, including 82.7 seconds to fill a 47.97 GiB encoder
weight cache; it is excluded from the warm mean.

The matching native action runs used the ordinary Cozy CLI, H3 1.26.2,
Runtime 0.19.0, TensorFS 0.4.0 and the default four-card model selection.
Sparring took 145.699 seconds and rally racing 144.632 seconds: a 145.1655-second
warm mean. That is 16.41% less execution time than the affected-library Comfy
cohort (1.1963×), and 21.07% less than the fixed-library cohort (1.2670×).
All three native clips passed full audio/video decoding. Native used BF16 Sol
with Sage INT8-QK/FP8-PV dense callbacks; this measured result retains the
precision, fallback and decoder limitations below. It is not a quality-equivalence
or universally fastest-engine claim.

FP8 storage does not mean the MultiStream graph performs every projection in FP8.
Its sliced QKV path dequantizes to BF16, and the checkpoint's 50 `fc2` exceptions
also use BF16 multiplication. With the affected library, output projection and
`fc1` produced 2,400 FP8 fallback warnings per full request. A standalone PyTorch
2.14/cu130 reproducer shows the same failure outside ComfyUI: for each tested FP8
GEMM shape, the GPU
that executes it first succeeds and the other devices fail. Reversing GPU order
with identical matrix sizes reverses the successful device. Allocator, stream,
workspace and heuristic-cache controls did not fix it. Preserve this limitation;
these measurements do not establish the fastest possible ComfyUI configuration
or a precision-matched engine comparison.

This symptom matches NVIDIA's documented
[CUB-9487 issue](https://docs.nvidia.com/cuda/archive/13.0.3/cuda-toolkit-release-notes/index.html#cublas-release-13-0-update-2),
including its process-per-device workaround. The recorded Comfy package is
cuBLAS 13.1.1.3, which belongs to CUDA Toolkit 13.0 Update 3. NVIDIA documents the
fix in [Toolkit 13.1](https://docs.nvidia.com/cuda/archive/13.1.0/cuda-toolkit-release-notes/index.html#cublas-release-13-1),
whose cuBLAS component is 13.2.0.9. The separate, hash-verified library overlay
passed all 32 exact-shape matmul cases; the affected library passed only eight.
The three repeated full action renders then completed with zero FP8 fallback
warnings. Process maps confirmed the fixed libraries in both ComfyUI and its VAE
workers. The original environment remains intact; only the child processes'
`LD_PRELOAD`, `LD_LIBRARY_PATH` and `PYTHONPATH` select the overlay.

The fixed-library cohort used the same prompts, seeds, weights, sigmas and
unmodified plugins. Its setup sample took 255.142 seconds; warm phase means were
1.662 seconds for conditioning, 150.264 for sampling, 17.978 for video decode,
0.267 for audio decode and 14.171 for mux/save. Repairing the launch failures did
not improve total execution time in these two warm samples. Keep both cohorts;
do not attribute their timing difference to a single kernel from this evidence.

The four-GPU FP16 decoder was checked against stock decode using the identical
cached sampler result under cuBLAS 13.1.1.3. Decoded PCM matched exactly; 112 of 362
decoded RGB frames
matched exactly. Encoded-output RGB PSNR was 43.289 dB, with mean absolute channel
difference 0.973/255 and maximum 54/255. This is not bitwise equivalence and is not
a comparison of raw floating-point VAE tensors, and was not repeated under the
fixed library. Do not use cross-engine PSNR:
matching seeds do not establish matching initial noise across engines.

All recorded clips passed full audio/video decoding. Sixteen sampled frames per
action clip were reviewed, which does not certify fine anatomy, contact physics,
continuous motion or subjective audio quality. The rally clip includes
unrequested watermark-like lettering at the lower right; it is preserved.

Turbo graphs, sigmas, adapter provenance, decoder comparison and the standalone
FP8 reproducer are under
`/home/fidika/cozy_v2/outputs/h3-turbo-4x5090-20261007/`.
The matching action inputs, Comfy graphs, original clips/audio, hashes, phase
timings, GPU samples and process handoff receipt are under
`/home/fidika/cozy_v2/outputs/h3-action-quality-20261007/`;
`comfy-action-results.json` is the consolidated Comfy report. GPU handoff stops
only the owned competitor server/workers. Native and rental operations use
ordinary Cozy CLI commands.
The corrected-library report and original clips are preserved separately in
`comfy-fixed-cublas/comfy-fixed-results.json` below the action evidence directory.

A later native precision matrix reused the same four executors across all three
modes. Its BF16 control MP4s matched the pristine controls byte for byte; its
148.0415-second mean is a separate warm cohort from the earlier 145.1655-second
baseline. Each mode used the same sparring/rally prompts and seeds, model,
adapter and eight-evaluation 4/4 schedule, without a text-conditioning output cache.

| Native sparse attention mode | Two-sample warm mean |
| --- | ---: |
| BF16 Sol control | 148.0415 s |
| Kitchen INT8/S32 | 135.6515 s |
| Experimental FP8/F16 exact sparse tiles | 130.9285 s |

The FP8 prototype retains BF16 routing/tail work and FP32 outer state; it does
not turn every operation into FP8. It reduced execution time by 11.56% versus
this BF16 control and 3.48% versus INT8 in these two samples. This is an opt-in
prototype result, not a default-policy or broad quality qualification.

All six native warm clips passed full audio/video decoding and are retained with
hashes and original audio under `native-precision-quality/`. Sixteen matched
frames per clip showed similar overall poses, clothing identities, camera motion
and car trajectories, without an obvious variant-specific gross collapse.
Fine anatomy/contact, intermediate motion and subjective audio remain unverified.

| Candidate versus native BF16 | Sparring RGB PSNR | Rally RGB PSNR |
| --- | ---: | ---: |
| INT8 | 27.74 dB | 27.28 dB |
| FP8/F16 | 26.74 dB | 26.75 dB |

These are numerical differences in decoded H.264 outputs, with no alignment or
resampling, not perceptual quality scores. FP8 showed slightly more RGB drift in
both scenes; audio drift ordering was mixed (zero-shift PCM correlations
0.958–0.980). Sparring was quiet on average in every mode, approximately −42 dBFS
RMS. Retain BF16 as the reference; neither PSNR nor sampled frames establish
quality equivalence. `native-precision-quality/summary.json` records the scope
and per-clip findings.

Final cleanup through the ordinary Cozy CLI recorded `provider_absent=true` and
an empty rental list. Original benchmark media and provenance remain local.
