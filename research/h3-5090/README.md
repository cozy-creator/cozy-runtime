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
| Four 5090s, Turbo 4/4, optimized action cohort | 179.180 / 168.143 s | 173.6615 s |

These are queue-free server `execution_start` to `execution_success` timings,
excluding output transfer. The action warm samples are sparring (seed 581204) and
rally racing (581205), after a parkour setup sample (581203). Compare them only
against matching native action requests. The lighthouse Turbo warm samples differ
because the first rebuilt text-encoder state after eviction: conditioning took
58.36 seconds, then 1.10 seconds. Both are reported. The optimized action setup
sample took 257.509 seconds, including 82.7 seconds to fill a 47.97 GiB encoder
weight cache; it is excluded from the warm mean.

FP8 storage does not mean the MultiStream graph performs every projection in FP8.
Its sliced QKV path dequantizes to BF16, and the checkpoint's 50 `fc2` exceptions
also use BF16 multiplication. Output projection and `fc1` produced 2,400 FP8
fallback warnings per full request. A standalone PyTorch 2.14/cu130 reproducer
shows the same failure outside ComfyUI: for each tested FP8 GEMM shape, the GPU
that executes it first succeeds and the other devices fail. Reversing GPU order
with identical matrix sizes reverses the successful device. Allocator, stream,
workspace and heuristic-cache controls did not fix it. Preserve this limitation;
these measurements do not establish the fastest possible ComfyUI configuration
or a precision-matched engine comparison.

The four-GPU FP16 decoder was checked against stock decode using the identical
cached sampler result. Decoded PCM matched exactly; 112 of 362 decoded RGB frames
matched exactly. Encoded-output RGB PSNR was 43.289 dB, with mean absolute channel
difference 0.973/255 and maximum 54/255. This is not bitwise equivalence and is not
a comparison of raw floating-point VAE tensors. Do not use cross-engine PSNR:
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
