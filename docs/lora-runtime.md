# Generic linear LoRA execution primitive

`cozy_runtime.internal.lora.LinearAdapters` adds ordered low-rank residuals to Runtime-owned linear leaves. It does not load an ecosystem LoRA, resolve a model name, or merge a weight. The TensorFS converter owns dialect normalization; the executor must supply verified ordinary checkpoint tensors.

For each explicitly selected base component, canonical tensors are `<module>.lora_A.weight` with shape `[rank, input]`, `<module>.lora_B.weight` with shape `[output, rank]`, and optional scalar `<module>.alpha`. Missing alpha means rank. Each adapter adds `linear(linear(x, A), B) * user_scale * alpha / rank`, using the leaf compute dtype, in supplied stack order. Strengths are independent, including negative values; they are never normalized. Zero scale validates all supplied tensors and applies an exact no-op without allocating factors.

A `LinearAdapters` binding installs hooks once for a fixed target set, before the fusion planner runs. Runtime already preserves semantic hooks through encoded-leaf restaging (`StreamingFillBackend._install_leaf`). Its fusion planner notices those hooks and retains floating-point activations at the affected leaves, rather than handing only a prequantized carrier to them. This preserves the FP8 base payload without requiring the LoRA to operate on reconstructed FP8 input. The cost of losing activation-quantization fusion at affected leaves is unmeasured.

`request(...)` accepts an ordered sequence of `AdapterInput` values. Each value directs one exact adapter to one explicitly selected component and selected base snapshot; compatible shapes never select FL2VA and Ref2VA automatically. The binding validates every tensor and target before any device copy, enforces the admitted factor byte budget, and checks finite values again after conversion to compute dtype. Device factors are private copies. Attempts have exclusive access to the binding, cancellation checks, and ContextVar-scoped residuals. Completion, exception and cancellation clear those factors and reset the context. Device cleanup synchronizes CUDA; a cleanup failure poisons the binding against reuse. Construction teardown removes only its owned hooks.

The current implementation supports exact `torch.nn.Linear` and the Runtime's encoded linear leaves. It refuses nonlinear modules, custom forwards, aliased or already-owned targets, unknown keys, incomplete pairs, wrong orientation/shape/dtype, nonfinite factors/scales, stale base bindings and non-CPU source tensors. This is deliberate containment, not a promise that every architecture supports every LoRA. Training-task compatibility and inference-contract deltas must be checked by resolution; shape containment proves only that the residual can be applied.

## Integration still required

This primitive is not connected to ordinary `cozy run` yet. It must remain a draft until the caller does all of the following:

- Carry ordered adapter selections per model slot and explicit component through the shared request/binding contract, resolving exact ordinary adapter snapshots.
- Stage and lease verified tensors through TensorFS with bounded source bytes/rank/count, validate inference compatibility, and include adapter factors plus activation scratch in admission. `max_device_bytes` here accounts for factors only; the caller must budget the two low-rank GEMM outputs and accumulation temporaries for the selected token shape.
- Include the immutable target set in construction identity; install hooks after stock/encoded leaves exist and before fusion. Restaging must preserve those hooks and leave compute geometry unchanged.
- Enter each selected model's `request(...)` scope around execution, echo the applied stack per slot, and prevent poisoned bindings from serving any future attempt. Global `Context._adapters` alone is insufficient when different slots have different stacks.
- Deliver the same ordered factors/scales to every Ulysses rank. Low-rank GEMM output can depend on token shape and selected kernel; real 1/2/4-GPU parity is a separate gate.
- Qualify ordinary default-home CLI output, adapted → none → adapted alternation, cancellation, encoded-leaf staging, fusion behavior and actual adapter quality on GPU. CPU arithmetic tests do not establish these claims.

No Runtime version bump, public SDK LoRA loader or wire-schema change is included here.
