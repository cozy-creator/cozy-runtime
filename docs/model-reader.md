# Read held model weights

`WeightsReader` is an injected job service. It opens an attempt-scoped read lease over
an exact granted model input. Ordinary invocable callers pass `ModelArtifact` values;
the existing owner binding validates their producer/receipt and grants typed model
parameters such as `QuantizationSource` before the function runs.

```python
with reader.open(reference) as original, reader.open(candidate) as changed:
    for component in original.components():
        for key in original.keys(component):
            if original.identity(component, key) == changed.identity(component, key):
                continue  # exact encoded identity; no tensor-value read
            shape = original.tensor(component, key).shape
            values = np.empty(shape, dtype=np.float32)
            flat = values.reshape(-1)
            for offset in range(0, flat.size, 1 << 22):
                original.read_values_into(component, key, offset, flat[offset : offset + (1 << 22)])
```

`tensor()` exposes logical dtype/shape, the exact encoding digest, and stored-role
geometry. `identity()` hashes that tensor's complete encoded representation, including
inline bytes, object slices and all scales. It excludes its containing manifest and
producer, so inherited tensors retain their identity. The identity grants no object
or filesystem access.

`read_part_into(component, key, role, byte_offset, buffer)` reads encoded bytes.
`read_values_into(component, key, element_offset, float32_buffer)` reconstructs a flat
logical range. Both require writable contiguous buffers of at most64MiB; offsets
and the selected role's native geometry are checked before allocation/read. Plain
BF16/FP16/F32 widens to float32. FP8-rowwise (both scale shapes) and MXFP8 reuse the
existing reviewed CPU float32 decoders; those reads require Torch in the callable's
environment. This is storage-value measurement, not model inference or its casting
policy. Quantizer-time saturation is not recoverable from stored values and is not
invented by the reader.

Views close explicitly or with the context manager. Cancellation, attempt completion
and revoked leases refuse reads. The parent releases reader custody independently
of producer/memo retention. Opening a view declares no output, opens no writer and
creates no checkpoint, publication or durable memo record.
