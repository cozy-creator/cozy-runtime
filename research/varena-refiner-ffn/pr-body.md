## Summary

The H3 token refiner still enters Python FFN sizing and nested activation retries. Add one complete native consumer for its exact Diffusers SwiGLU layout and selected int8 projection plans.

```text
FeedForward wrapper
  validate complete operation and selected providers
  native split selection
  committed up -> SiLU -> multiply -> down
```

## Evidence

- **Before:** current Runtime 78e58f7e enters cuts._ffn and sizing.Ffn for both refiner FFN and nested SwiGLU.
  **After:** implementation in progress; acceptance, complete allocation inventory, and host checks remain pending.

## Proof

Source and host checks precede a separate real GPU fixture. This draft does not qualify GPU execution or full H3 performance. Preserve the later 15 s / 362 frames / 1344x768 / 30 steps / audio gate.

## Merge Danger

**Door:** two-way

**Blast Radius:** refiner

CPU/MPS and unsupported CUDA contracts retain their existing route. No merge or release requested.
