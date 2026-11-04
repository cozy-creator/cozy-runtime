# Upstream dense attention adapters (h3a-060)

`internal/attention_upstream.py` calls maintained public libraries. It contains
layout/argument conversion and admission checks; it does not copy their kernels.
These adapters remain explicit experimental choices until full-shape timings and
matched output comparisons qualify their precision on the selected environment.

| Adapter | Precision | Admitted input | Admitted CUDA architecture |
|---|---|---|---|
| `kitchen-int8` | INT8 Q/K/V/P; FP32 accumulation | BF16/FP16, D1–256 | SM90, SM100, SM103, SM120, SM121 |
| `flash-attn4-fp8` | E4M3 Q/K/V per head, BF16 output (`_flash_attn_fwd`) | BF16, D≤128, equal heads | SM100, SM103 |
| `sageattention` | INT8 Q/K, smoothed K; E4M3 P/V. SM90: per thread, FP32+FP32. SM89/SM120: upstream's SM89 kernel (per thread / per warp), FP32+FP16 | BF16/FP16, D≤128 | SM89, SM90, SM120 (built on the card by upstream's own `setup.py`) |
| `sageattention3-fp4` | NVFP4 Q/K/V, per-16 E4M3 microscales, K and per-block Q mean smoothing | BF16/FP16, D64/128, equal heads, scale D^-1/2 | SM120 |

Every adapter is head-local and served through `attention_ulysses`. No entrypoint
silently falls back: each
refuses nonzero dropout, log-sum-exp output and active autograd; Kitchen also
refuses causal attention. Grouped-query attention requires
`enable_gqa=True`; the FP8 FlashAttention and Sage3 adapters refuse it.
Kitchen forwards the standard `[B,H,Q,K]` boolean/additive mask and optional
softmax scale. The others refuse masks rather than dropping padding metadata.
Sage3 subtracts K's mean in place upstream, so the adapter passes a copy.

Importing this module neither changes a global Diffusers backend nor imports an
optional kernel. The kernels are compiled on the machine from pinned source
(`kernel_sources`); none is a Runtime base dependency.

`flash-attn3-fp8`, `sageattention-fp16pv` and `flashinfer-bf16-fp8` are gone. A pin naming one
warns (`attention_kernel_removed`) and runs the normal choice.

## Reviewed source contracts

- Kitchen commit `62c5bb4a5f2f818d4be7e2a51f83aaf1286a1243` exposes
  [`comfy_kitchen.int8_attention`](https://github.com/Comfy-Org/comfy-kitchen/blob/62c5bb4a5f2f818d4be7e2a51f83aaf1286a1243/comfy_kitchen/sage_attention.py).
  Its HND input layout requires only transposed views for contiguous inner
  dimensions. Kitchen owns Hadamard rotation, adaptive K centering, per-thread
  Q/K scales, per-channel V scales and integer attention.
- flash-attn-4 4.0.0b32 serves FP8 only on SM100/103 and only through
  `flash_attn.cute.interface._flash_attn_fwd` (`flash_attn_func` takes no descales).
- SageAttention d1a57a54 `sageattention3_blackwell` (`sageattn3`) refuses every card
  but sm_120/121 in its C++ entry.
