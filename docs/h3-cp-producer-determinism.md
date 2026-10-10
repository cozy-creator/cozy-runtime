# Opt-in H3 attention across GPU partitions

These recipes keep attention reductions and request state consistent when Ulysses changes the number of GPU partitions. They are explicit selections; H3's automatic preferences and its Turbo 4 dense + 4 sparse schedule are unchanged.

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

SageAttention2 keeps its INT8 Q/K and FP8 P/V arithmetic, with a fixed K-mean reduction. The hybrid uses that same dense implementation for both dense steps and the protected prefix of BF16 CuTe sparse steps. CuTe's sparse arithmetic is unchanged. The fixed reduction matched the tested head partitions in isolated H100 tests; full-video Hopper qualification remains pending.

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
one saved latent, and the integrated full-video one/two-GPU check is still pending.

An earlier private build of the RTX implementation produced byte-identical delivered MP4 files for one 15-second Turbo input across actual one-GPU run 5315 and two-GPU runs 5307/5308. All 362 decoded RGB frames and decoded audio samples also matched. This was a retrospective comparison across different hosts and drivers, so its timings are not a controlled speed comparison. Only the single-GPU run retained pre-encoding diagnostics; equality of every intermediate tensor is not established.

Separate GPU probes covered dense reductions, sparse statistics, FP8 projections with inactive/baked/unbaked LoRA, real two-rank exchange, retries and request cleanup. The faster shared-projection candidate reduced complete sparse-attention time from 213.09 ms to 195.46 ms in three paired samples on the tested two-5090 host. That is an attention-operation measurement, not a full-video speed claim.

The current integration preserves newer Runtime copy-engine routing and int64 quantizer offsets. Earlier private RTX evidence does not qualify this integrated head or a newly selected transport. CPU/source checks, actual transport receipts and any integrated GPU evidence must be reported separately. These results do not establish universal determinism across kernels, inputs or GPU counts, nor unchanged quality against the older approved clip with a different prompt.
