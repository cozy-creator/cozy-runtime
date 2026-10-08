# H3 on RTX 5090

Owned benchmark: one GPU first, then four GPUs. Use ordinary default-home `cozy run`
on an explicitly selected rental. Keep the local display GPU unused.

The request is FP8-pruned H3, 15 seconds, 1344×768, 24 fps and 30 steps with audio.
Record cold preparation separately from request execution. Repeat with three seeds;
compare each engine on the same hardware, prompt, geometry, schedule and precision.
Report the actual attention implementations and checkpoint-format differences.
Do not shorten the request to make memory fit.

ComfyUI baseline: v0.39.0 (`b0b743566f65daafc423b4fea8a2fbda94b3384a`).
`setup-comfy.sh` installs its isolated environment and the FP8 DiT, BF16 text encoder,
FP16 video VAE and FP32 audio VAE from the pinned Comfy-Org checkpoint revision.
Run it only on the owned rental, after extracting ComfyUI into `/root/h3-bench/comfy`.

Evidence and ownership are under
`/home/fidika/cozy_v2/outputs/h3-5090-benchmark-20261007/`.
No completed render or performance result is claimed yet.
