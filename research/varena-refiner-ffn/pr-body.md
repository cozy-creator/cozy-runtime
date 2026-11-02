## Summary

H3's token-refiner FFN currently enters Python FFN sizing and a nested activation chooser. Recognize the complete stock SwiGLU FeedForward, validate its actual int8 projection providers and kernel plans, then select its complete allocation menu in Varena once.

```text
exact FeedForward/SwiGLU wrapper
  complete provider/layout/full-tail plan checks
  native split selection
  supplied up -> SiLU -> multiply -> down
```

The accepted invocation bypasses both legacy choosers and never retries. Other providers/configurations and CPU/MPS retain their current path. Norm, attention and residual addition remain outside this FFN scope.

## Evidence

- **Before:** Runtime78e58f7e calls cuts._ffn/sizing.Ffn for the refiner and nested SwiGLU.
  **After:** accepted calls use actual int8_plan kind/config/workspace for both projections, a full fresh result, all activation buffers, and shared row quantization/scales/workspace. No guessed workspace or Python capacity selection.
- Real Diffusers0.41/Torch2.14.1+cpu checks pass: unchanged CPU output after execution.install, actual NativeLinear provider/bias/stride refusals, exact FP16/BF16/F32 activation at odd row counts, and mutation/hook/dropout refusals. CUDA uninitialized; libcuda unmapped.
- Targeted Ruff, mypy, AST and genuine fixture import checks pass. Full H3 geometry is constructed on meta.

## Proof

Source/host evidence only. A strict-cohort Cozy fixture is prepared for real BF16/FP16 H3-width FFN execution, one-composite/zero-legacy-cut profiling, output comparison, selected-plan inventory, refusal and same-PID key cleanup. GPU execution, pressure/failed-submission cases, actual LoRA and allocation census remain open. Preserve the original performance failure and later15s/362frames/1344x768/30step/audio full-H3 gate.

## Merge Danger

**Door:** two-way

**Blast Radius:** FFN

The structural contract also covers identically shaped stock Diffusers SwiGLU FFNs. Only actual int8.convrot/mma.sync single-device paths adopt it. No merge, install, release, or broad P2 completion requested.
