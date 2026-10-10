# Approved H3 attention defaults

H3 keeps the user-approved attention families and schedules, while preferring the
partition-stable implementations where they are eligible and ready.

| Hardware | Dense kernel | Sparse kernel | Preferred hybrid |
|---|---|---|---|
| Hopper SM90 | SageAttention2 8-bit, fixed K mean | BF16 CuTe Sparse | `sol-attn-sage2-fixed-kmean` |
| Consumer Blackwell SM120 | SageAttention3 NVFP4, fixed means | Kitchen INT8 Sparse | `kitchen-sol-producer-fp4-fixed-means-shared-qkv` |

Turbo remains four dense steps followed by four sparse steps; regular generation
remains ten dense followed by twenty sparse. Protected rows and always-dense
paths are unchanged. Full-dense FP4 remains an explicit choice, not a default.
The same preferred hybrid now applies to both single-GPU and Ulysses execution.

Runtime still owns hardware, dtype, site and parallelism admission. The new
Hopper implementation remains SM90-only and the new Blackwell implementation
remains SM120-only. SM121 and other architectures keep their existing eligible
fallbacks; this preference change does not widen support. Blackwell automatic
selection still requires the reviewed Kitchen wheel to match the platform.
Explicit request pins, including older supported kernels, continue to win and
restore the prepared default afterward.

## Readiness and qualification

Automatic selection can use a later ready recipe while the preferred kernel is
compiling or unavailable. A changed preference does not guarantee identical
results for every cold-start path. Inspect the actual served kernels; an
unavailable preferred kernel must not be reported as though it ran.

Real Diffusers policy tests cover ready no-pin selection at degrees 1, 2, 4 and
8 on SM90 and SM120, existing fallback behavior on other architectures and cold
recipes, explicit overrides, and processor restoration. These are CPU policy
checks, not additional multi-GPU arithmetic proofs.

The full-video evidence uses explicit recipe pins through the ordinary Cozy CLI:
H100 runs 5342/5343 and RTX 5090 runs 5341/5344 each match across actual one and
two GPUs for one nominal 15-second Turbo input, seed 581204. MP4 bytes, all 362
decoded RGB frames, decoded PCM, pre-encoding RGB/audio hashes, and schedules
match within each pair. The measured source is `40efdc79` with the private
Runtime 0.23.12 build and TensorD 0.5.10. This later preference ordering changes
no kernel math. It does not establish equality across all seeds, other kernels,
other hardware, or four/eight-GPU runs.

H3 audio decoding also scopes deterministic cuDNN selection to its existing
block and restores the previous setting. The isolated H100 check added about
0.1295 seconds per warmed audio decode and changed waveform values; internal
algorithms/precision and perceived quality were not independently established.
See [partition evidence and limits](h3-cp-producer-determinism.md).
