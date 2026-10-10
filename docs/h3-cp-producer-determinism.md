# H3 attention across GPU partitions

These recipes keep attention reductions and request state consistent when Ulysses changes the number of GPU partitions. H3 prefers the eligible, ready hybrid on SM90 and SM120. The approved kernel families, Turbo's 4 dense + 4 sparse schedule, and regular generation's 10 dense + 20 sparse schedule are unchanged. Generic attention selection for other models is unchanged.

| Hardware | Dense selection | Dense + sparse selection |
|---|---|---|
| RTX 5090 / SM120 | `sageattention3-fp4-fixed-means` | `kitchen-sol-producer-fp4-fixed-means-shared-qkv` |
| H100 / SM90 | `sageattention-fixed-kmean` | `sol-attn-sage2-fixed-kmean` |

For example, select the RTX hybrid on an ordinary request with `--attention-kernel=base_model/fl2va_dit=kitchen-sol-producer-fp4-fixed-means-shared-qkv`. Select the Hopper hybrid with `--attention-kernel=base_model/fl2va_dit=sol-attn-sage2-fixed-kmean`. The request's model layout supplies the dense/sparse schedule; a kernel selection does not change the number of denoising steps.

## RTX producer path

Dense attention uses SageAttention3 NVFP4 with fixed Q/K mean reductions and the upstream BF16 correction. Sparse attention uses the existing Kitchen INT8 producer, including its RMSNorm, RoPE, protected prefix, scale margin and previous-step statistics.

The multi-GPU path projects each local token shard through the original modules and LoRA hooks. Eligible FP8 projections share one activation quantization across the three original Q/K/V calls, without an additional packed output buffer. Existing fallback semantics remain for other projection types and hooks. Ulysses exchanges the projected tensors into complete token sequences for each rank's head group; Kitchen consumes global 4096-row chunks, and the inverse exchange restores local token ownership before the output projection.

A Diffusers hook retains each component's complete rotary tables before the context-parallel split. It holds references only during the sparse request and avoids a per-block rotary all-gather. Followers borrow producer state from the existing attention-selection hold for the attempt. Restore, new selection, refusal and matching-construction unload clear it; same-step retries reuse the original statistics inputs.

## Hopper path

SageAttention2 keeps its INT8 Q/K and FP8 P/V arithmetic, with a fixed K-mean reduction. The hybrid uses that same dense implementation for both dense steps and the protected prefix of BF16 CuTe sparse steps. CuTe's sparse arithmetic is unchanged. The fixed reduction matched the tested head partitions in isolated H100 tests; the bounded full-video result below covers actual one/two-GPU execution.

## Readiness and fallback

Preferences are ordered choices, not new hardware admissions. The fixed Hopper
hybrid remains SM90-only; the fixed Sage3/Kitchen producer remains SM120-only.
SM121 and other hardware retain their existing eligible fallbacks. Explicit
request pins still win, including older recipes. All-dense FP4 is not promoted.

Automatic selection may serve a later ready recipe while a preferred kernel is
compiling or unavailable. A preference alone therefore does not establish
byte-identical cold-start results. The full GPU proof used explicit recipe pins;
CPU tests cover ordinary no-pin selection when ready, fallback, override and
restore at degrees 1, 2, 4 and 8.

## Evidence and limits

The official H3 audio decode now scopes `torch.backends.cudnn.deterministic=True`
to its existing audio block and restores the previous value on success or error.
It preserves all other backend settings and the published mean/std and decoder
operations. Runtime already disables cuDNN benchmarking at component entry.

H100 runs 5337/5338 had identical final normalized audio latent bytes across one
and two GPUs, but repeated decodes of each unchanged latent produced different
waveforms. Isolated run 5340 reproduced that baseline difference; three decodes
with cuDNN deterministic selection produced identical waveform bytes. The only
observed numerical-setting change was `cudnn.deterministic`. The selected internal
cuDNN algorithms and their precision were not traced.

The two warmed decoder samples averaged 0.1775 seconds before the change and
0.3070 seconds with deterministic selection: about **0.1295 seconds extra per
decode** on that H100. The observed cumulative allocation peak increased by
265 MiB; this is not an isolated workspace measurement. The candidate waveform
also changed: one comparison had relative L2 error 0.001352 and maximum absolute
difference 0.0002022 versus the baseline. This records numerical differences; it does
not establish unchanged perceived quality. These are decoder measurements for
one saved latent. Subsequent full-video checks are recorded below.

Integrated source `40efdc79` with Runtime `0.23.12+dev.h3partition.40efdc79`,
TensorD `0.5.10` and TensorFS `0.6.1` passed actual one/two-GPU full-video checks
for the same nominal 15-second Turbo input, seed 581204. H100 runs 5342/5343 and
RTX 5090 runs 5341/5344 each produced identical MP4 bytes, all 362 decoded RGB
frames, all decoded PCM, pre-encoding RGB/audio hashes and sampler schedules.
The H100 pair used its direct-link route; the RTX pair used host-staged transfers
without peer access. The subsequent preference promotion changes no kernel math.
This is one input at degrees one and two, not a proof for every seed or GPU count.

An earlier private build of the RTX implementation produced byte-identical delivered MP4 files for one 15-second Turbo input across actual one-GPU run 5315 and two-GPU runs 5307/5308. All 362 decoded RGB frames and decoded audio samples also matched. This was a retrospective comparison across different hosts and drivers, so its timings are not a controlled speed comparison. Only the single-GPU run retained pre-encoding diagnostics; equality of every intermediate tensor is not established.

Separate GPU probes covered dense reductions, sparse statistics, FP8 projections with inactive/baked/unbaked LoRA, real two-rank exchange, retries and request cleanup. The faster shared-projection candidate reduced complete sparse-attention time from 213.09 ms to 195.46 ms in three paired samples on the tested two-5090 host. That is an attention-operation measurement, not a full-video speed claim.

The integration preserves newer Runtime copy-engine routing and int64 quantizer offsets. Earlier private RTX evidence remains attached to its original source; the final integrated runs above qualify their own source and observed transports. These results do not establish universal determinism across kernels, inputs or GPU counts, nor unchanged quality against the older approved clip with a different prompt.
