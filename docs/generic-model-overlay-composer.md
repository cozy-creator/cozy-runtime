# Generic model-slot overlay composer

`Model.prepare_adapters()` is the Runtime author seam for a generic ordered
adapter stack. A model class opts in with `__adapter_compatibility__`, declaring
adapter kind, family, target components and weight bounds. The default method
refuses with `adapter_composition_unsupported`; Runtime never silently ignores a
requested overlay.

The worker must call this method on the freshly constructed generation after
weights are filled and before optimization, fusion, warm-up or graph capture.
`ModelRegistry.acquire(..., adapters=...)` includes the ordered refs, components,
families and weights in the construction variant identity, so:

- no overlay reuses the pristine base generation;
- the same ordered stack can reuse one prepared composed generation;
- a changed weight or order constructs a distinct generation;
- concurrent requests never mutate one shared base generation.

The current serving Invocation path still needs to carry placement `Slot.adapters`
through package preparation and executor binding before this seam can be reached by
ordinary CLI requests. The protocol already has ordered `Slot.adapters` and
`ModelAdapter` fields, but placement validation and preparation currently omit them.
That transport follow-up must preserve canonical identity and typed incompatibility
refusals.

H3 Turbo remains specialized. Its current `LoRAFactors.accumulate` computes a
weighted update from the original activation during each DiT forward under a
temporary `TurboArming` hook. It is not a generic bake-once composer. H3 needs an
isolated supported composer that materializes/fuses factors and precomputes its
Turbo tables, or it must keep the dedicated `turbo_lora` slot separate.

Quantized bases are not automatically dequantized and requantized. A model-specific
composer must preserve pristine base custody and provide its own numerical
qualification; unsupported combinations refuse only that request.
