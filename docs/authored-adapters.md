# Adapters in Python calls

`with_adapters` selects an ordered LoRA stack for a serving Model argument. The
base and adapters are existing `ModelArtifact` values; their producer, receipt
and manifest identities remain unchanged.

```python
from cozy_runtime.author import LoRA, with_adapters

styled = with_adapters(
    base,
    [
        LoRA(spatial, component="fl2va_dit", scale=0.3),
        LoRA(action, component="fl2va_dit", scale=-0.1),
    ],
)
result = await generate(prompt="A dancer turns", model=styled)
```

An empty stack returns the original artifact. Strength defaults to 1; zero and
negative strengths are preserved. Each adapter names the exact base component
and defaults its source component to `adapter`. Compatibility is established
from the actual tensors during preparation, without guessing from repository
names or treating training-task metadata as a structural prohibition.

The broker only accepts wrappers at signature-declared serving Model positions.
The complete ordered stack joins the child intent; lowering to a handler payload
preserves its ordinary schema while the prepared binding digest identifies the
exact adapter stack. Each source receives a durable native hold owned by the
child before preparation and remains held after temporary preparation leases
close. Existing child completion/collection releases those holds.

This initial authored path consumes retained artifacts from native conversion
or earlier calls. Exact published Tensorhub references are supported by Creator's
`--lora` path; string references are not resolved through Python networking.
Job-only Model arguments remain derive-only and refuse invocation adapters.

Qualification: CPU broker, generated caller, real native custody, TensorFS target
fit and exact cache-selection checks are independent from H3 GPU/video proof.
