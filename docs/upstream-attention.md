# Upstream dense attention adapters (h3a-060)

`internal/attention_upstream.py` calls maintained public libraries. It contains
layout/argument conversion and admission checks; it does not copy their kernels.
These adapters remain explicit experimental choices until full-shape timings and
matched output comparisons qualify their precision on the selected environment.

| Adapter | Precision | Admitted input | Admitted CUDA architecture |
|---|---|---|---|
| `kitchen-int8` | INT8 Q/K/V/P; FP32 accumulation | BF16/FP16, D1–256 | SM90, SM100, SM103, SM120, SM121 |
| `flashinfer-bf16-fp8` | BF16 Q/K, E4M3 V/P per head, BF16 output | BF16, D128 | SM100, SM103 |
| `flash-attn4-fp8` | E4M3 Q/K/V per head, BF16 output (`_flash_attn_fwd`) | BF16, D≤128, equal heads | SM100, SM103 |
| `sageattention` | INT8 Q/K, smoothed K; E4M3 P/V. SM90: per thread, FP32+FP32. SM89/SM120: upstream's SM89 kernel (per thread / per warp), FP32+FP16 | BF16/FP16, D≤128 | SM89, SM90, SM120 (built on the card by upstream's own `setup.py`) |
| `sageattention-fp16pv` | INT8 Q/K per thread, smoothed K; FP16 P/V, FP32 accumulation | BF16/FP16, D≤128 | SM90 (upstream's SM80 kernel built sm_90a; not upstream-qualified there) |
| `sageattention3-fp4` | NVFP4 Q/K/V, per-16 E4M3 microscales, K and per-block Q mean smoothing | BF16/FP16, D64/128, equal heads, scale D^-1/2 | SM120 |

Every adapter is head-local and served through `attention_ulysses`. No entrypoint
silently falls back: each
refuses nonzero dropout, log-sum-exp output and active autograd; Kitchen and
FlashInfer also refuse causal attention. Grouped-query attention requires
`enable_gqa=True`; the FP8 FlashAttention and Sage3 adapters refuse it.
Kitchen forwards the standard `[B,H,Q,K]` boolean/additive mask and optional
softmax scale. The others refuse masks rather than dropping padding metadata.
Sage3 subtracts K's mean in place upstream, so the adapter passes a copy.

Importing this module neither changes a global Diffusers backend nor imports an
optional kernel. The kernels are compiled on the machine from pinned source
(`kernel_sources`); none is a Runtime base dependency. `flashinfer-python` imports as
`flashinfer`.

## Reviewed source contracts

- Kitchen commit `62c5bb4a5f2f818d4be7e2a51f83aaf1286a1243` exposes
  [`comfy_kitchen.int8_attention`](https://github.com/Comfy-Org/comfy-kitchen/blob/62c5bb4a5f2f818d4be7e2a51f83aaf1286a1243/comfy_kitchen/sage_attention.py).
  Its HND input layout requires only transposed views for contiguous inner
  dimensions. Kitchen owns Hadamard rotation, adaptive K centering, per-thread
  Q/K scales, per-channel V scales and integer attention.
- FlashInfer 0.7.0 exposes
  [`cute_dsl_fmha_ragged_prefill`](https://github.com/flashinfer-ai/flashinfer/blob/v0.7.0/flashinfer/attention/cute_dsl/fmha.py),
  whose scales are per-tensor scalars. The adapter quantises V per (batch, head)
  (the FA3-FP8 quantiser), passes `scale_v=1` and multiplies each head's BF16
  output by its descale: exact, head-local and without a host synchronisation.
  It compiles the kernel with `compile_cute_dsl_fmha_kernel` (CuTe JIT, counted as
  preparation) and hands it in, because the library otherwise downloads prebuilt
  artifacts on first use.
- flash-attn-4 4.0.0b32 serves FP8 only on SM100/103 and only through
  `flash_attn.cute.interface._flash_attn_fwd` (`flash_attn_func` takes no descales).
- SageAttention d1a57a54 `sageattention3_blackwell` (`sageattn3`) refuses every card
  but sm_120/121 in its C++ entry.

## Verification

CPU tests exercise refusal semantics, per-card Sage mode selection and
registration. They do not run any CUDA kernel.
The opt-in integration test executes the installed upstream implementation,
compares FP32 SDPA, verifies unchanged input tensors and repeat equality, and
uses a nondefault scale (default for Sage3), unequal lengths, two batches, GQA
where served and a batch-specific Kitchen mask. It is intentionally not a mock of the CUDA library.

```sh
COZY_TEST_ATTENTION_CUDA=1 uv run pytest tests/test_attention_upstream.py
```

Only run that command on an owned device with the reviewed optional libraries.
The small numerical test is an ABI/arithmetic check; long H3 sequences, output
quality, attention timing and end-to-end videos still need qualification.
