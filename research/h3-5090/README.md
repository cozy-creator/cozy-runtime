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
