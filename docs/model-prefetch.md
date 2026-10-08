# Prepare a known next model

Package code can send an optional known-next-call hint through Runtime:

```python
from cozy_runtime.author import prefetch

prefetch(render_video)
references = await generate_references(...)
video = await render_video(references=references, ...)
```

The callee must be an existing locked managed callable. Runtime checks the binding,
serializes any requested `models` choices and sends `ModelPrefetch` to TensorD. The
acknowledgment does not promise completion or a particular residency state. The later
ordinary invocation still resolves its own models and performs any required preparation.
Current TensorD handles this hint using the callable and its parent's bindings; it does
not consume the hint's `models` payload. Pass model overrides to the later invocation
itself, where they determine the selected inputs. Selection handling is tracked in
[tracker #345](https://github.com/cozy-creator/tracker/issues/345).

TensorD decides how far to prepare the known call under current machine policy. This can
include obtaining stored model objects through TensorFS, preparing shared CPU buffers,
and asking a Runtime executor to construct or load a retained model on a GPU. The Runtime
executor performs the GPU work through TensorFS's local plane/device APIs. A hint does not
invoke the package's requested inference callable; it can still cause model construction
and GPU activity. It has no fixed preparation-worker count or guaranteed GPU residency.

The internal executor `Prefetch` command is a different operation: it requests CPU-tier
fill/pinning for an already registered construction through that executor's TensorFS
plane. `SealedPrefetch` batches plans for CPU buffers TensorD prepares. Neither name means
that serialized store objects, prepared CPU buffers and GPU tensors are interchangeable.

An owner that does not support the optional hint can decline it; ordinary demand
preparation remains available. Prefetch does not transfer the machine's budget or storage
lifetime authority to package code. For the data and ownership split, see
[Machine and Runtime roles](machine-agent-runtime.md).
