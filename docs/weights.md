# Raw weights: `Weights[M]`

A `Weights[M]` argument is the raw checkpoint of `M`'s layout (bare `Weights`: any layout). It is
never constructed. It binds like any model slot (`<fn>.models.<param>`, `defaults=`, a caller's
`ModelArtifact` override) and its interface row carries `"weights": true`. TensorD selects the
exact manifest; the attempt reads it through TensorFS and creates no Python model or GPU tensor.

```python
@invocable
async def compare(ctx: Context, *, original: Weights[SdxlModel], changed: Weights[SdxlModel]) -> Report:
    for component in original.components():
        for key in original.keys(component):
            if original.identity(component, key) == changed.identity(component, key):
                continue  # exact encoded identity; no tensor-value read
            flat = np.empty(original.tensor(component, key).shape, dtype=np.float32).reshape(-1)
            for offset in range(0, flat.size, 1 << 22):
                original.read_values_into(component, key, offset, flat[offset : offset + (1 << 22)])
```

`tensor()` exposes logical dtype/shape, the encoding digest and stored-role geometry.
`identity()` hashes the tensor's complete encoded representation and excludes its manifest, so
inherited tensors keep their identity; it grants no access.

`read_into(component, key, role, byte_offset, buffer)` reads encoded bytes;
`read_values_into(component, key, element_offset, float32_buffer)` reconstructs a flat logical
range (plain BF16/FP16/F32 and FP8-rowwise; the encoded one needs Torch). Buffers are
writable, contiguous and at most 64 MiB. Saturation at quantization time is not recoverable from
stored values.

`ctx.tensorfs_source(weights)` returns the same source as a TensorFS `Derivation` source, and
`derive.quantize_artifact(weights, plan, ...)` takes it directly. Reads end with the attempt;
cancellation refuses them.
