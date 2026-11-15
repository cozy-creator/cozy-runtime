# H3 attention scheduling experiment

Owner: Codex `/root/optimize_h3_attention`. Purpose: screen an unchanged-arithmetic
load-pipeline candidate for H3's expensive selected-key exact attention tiles.
Branch: `codex/h3-attention-schedule-20261115`.
Base: Runtime `af2a4753cb70598b75405942901e4f8f3b24eda6` (`origin/master`, fetched 2026-11-15).
Worktree: `/home/fidika/cozy/.worktrees/cozy-runtime/h3-attention-schedule-20261115`.

The cr-297 session `43442514074` SM89 trace attributes 9.96 seconds of CUDA activity
to `sol_exact_kernel` in its sampled sparse H3 call. Activity is summed, not an
interval union. This identifies a candidate; it does not establish a speedup.

The baseline is Runtime's pinned Comfy Kitchen source
`f8b7e97a1418d2e529bd5405fbf86312be198ae6` (0.2.40), specifically
`comfy_kitchen/backends/cuda/sage_attention/sol_attn_exact.cu`.
It uses two asynchronous stages, four warps, 64 query/key rows and head dimension
128. The first candidate changes only the pipeline depth to three. Key tiles,
selected lists, protected-row policy, precision, and reduction order stay fixed.

Ranked, falsifiable possibilities:

1. More outstanding tile loads hide selected-key latency: three stages improve
   paired kernel time with equal output bytes.
2. Shared memory/occupancy limits dominate: three stages regress because each CTA
   needs 48 KiB instead of 32 KiB.
3. Per-tile scale metadata remains exposed: stage depth changes little, and a
   later metadata-prefetch experiment would be needed. It is not implemented here.

The local RTX 4070 Laptop (SM89, 8 GiB) is authorized for bounded kernel screening.
The parent coordinator serializes GPU access. No rental, production code,
default-selection table, installed package, or running worker is changed.

Status: preparing a matched kernel harness. No GPU result or H3 speedup yet.
A microbenchmark can shortlist a candidate; regular H3 still needs 15 seconds,
362 frames, 1344x768, 30 steps and audio, and Turbo needs its separate fixed
eight-step schedule. Full video, multi-GPU, and cross-hardware qualification are
separate from this experiment.
