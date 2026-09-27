# Upstream dense attention adapters (h3a-060)

`internal/attention_upstream.py` calls maintained public libraries. It contains
layout/argument conversion and admission checks; it does not copy their kernels.
These adapters remain explicit experimental choices until full-shape timings and
matched output comparisons qualify their precision on the selected environment.

| Adapter | Precision | Admitted input | Admitted CUDA architecture |
|---|---|---|---|
| `kitchen_int8_attention` | INT8 Q/K/V/P; FP32 accumulation | BF16/FP16, D1–256 | SM90, SM100, SM103 |
| `flashinfer_mixed_attention` | BF16 Q/K, E4M3 V/P, BF16 output | BF16, D128 | SM100, SM103 |
| `sageattention` | INT8 Q/K per thread, smoothed K; E4M3 P/V, FP32+FP32 accumulation | BF16/FP16, D≤128 | SM90 |
| `sageattention-fp16pv` | INT8 Q/K per thread, smoothed K; FP16 P/V, FP32 accumulation | BF16/FP16, D≤128 | SM90 (image builds upstream's SM80 kernel sm_90a; not upstream-qualified there) |

Kitchen supports more architectures upstream; this initial adapter deliberately
admits only the Hopper/datacenter Blackwell scope under investigation. Neither
entrypoint silently falls back. Both refuse causal attention, nonzero dropout,
log-sum-exp output, active autograd, and context parallelism. Grouped-query
attention requires `enable_gqa=True` for unequal head counts. Unequal Q/KV
sequence lengths preserve their upstream semantics.
Kitchen forwards the standard `[B,H,Q,K]` boolean/additive mask and optional
softmax scale. FlashInfer refuses masks rather than dropping padding metadata.

The selector/registration policy is separate work in the same tracker issue.
Importing this module neither changes a global Diffusers backend nor imports an
optional kernel. Installation of `comfy-kitchen` or `flashinfer-python` belongs
to the worker environment; neither becomes a Runtime base dependency. The
distribution `flashinfer-python` imports as `flashinfer`, not `flashinfer_python`.

## Reviewed source contracts

- Kitchen commit `62c5bb4a5f2f818d4be7e2a51f83aaf1286a1243` exposes
  [`comfy_kitchen.int8_attention`](https://github.com/Comfy-Org/comfy-kitchen/blob/62c5bb4a5f2f818d4be7e2a51f83aaf1286a1243/comfy_kitchen/sage_attention.py).
  Its HND input layout requires only transposed views for contiguous inner
  dimensions. Kitchen owns Hadamard rotation, adaptive K centering, per-thread
  Q/K scales, per-channel V scales and integer attention.
- FlashInfer commit `15e83b7bb9f32d81d84017e2e715c265fb7253f5` exposes
  [`cute_dsl_fmha_ragged_prefill`](https://github.com/flashinfer-ai/flashinfer/blob/15e83b7bb9f32d81d84017e2e715c265fb7253f5/flashinfer/attention/cute_dsl/fmha.py).
  The adapter retains separate batch segments, passes exact maximum sequence
  lengths, allocates aligned BF16 output and requests no SkipSoftmax, sliding
  window, attention sinks, LSE or PDL. The library owns artifact selection/JIT.

FlashInfer's public scalar scale interface requires a current V absmax and a
host scalar. This adapter therefore synchronizes once and uses ordinary PyTorch
FP32 scaling followed by E4M3 conversion. That can allocate a full FP32 temporary;
it is a transparent initial integration, not a fused quantization speed claim.
Timings must include this reduction, cast, synchronization, layout copies and any
first-call JIT. Q/K retain BF16; there is no previous-step scaling state.
Replacing scalar conversion with an upstream device-scale interface, if one
becomes available, should not silently change this precision recipe.

## Verification

CPU tests exercise refusal semantics and real PyTorch V quantization, including
zero/nonfinite values and input immutability. They do not run either CUDA kernel.
The opt-in integration test executes the installed upstream implementation,
compares FP32 SDPA, verifies unchanged input tensors and repeat equality, and
uses a nondefault scale, unequal lengths, two batches, GQA and a batch-specific
Kitchen mask. It is intentionally not a mock of the CUDA library.

```sh
COZY_TEST_ATTENTION_CUDA=1 uv run pytest tests/test_attention_upstream.py
```

Only run that command on an owned device with the reviewed optional libraries.
The small numerical test is an ABI/arithmetic check; long H3 sequences, output
quality, attention timing and end-to-end videos still need qualification.
