# Prepare a known next model

Runtime 0.18.30 adds an optional ordinary Python hint:

```python
from cozy_runtime.author import prefetch

prefetch(render_video)
references = await generate_references(...)
video = await render_video(references=references, ...)
```

The callee must be an existing locked managed serving callable. Pass
`models={"model": artifact}` to select the same retained `ModelArtifact` override
as the later invocation; omitted slots use its captured defaults. Future media
inputs are unnecessary. No inference runs, and the hint reserves no GPU capacity,
activates no executor, and evicts no resident model. Package installation remains
code/environment installation without model downloads.

The local Runtime selects the exact models using the worker's GPU name and granted
count, then shares bounded CPU/download work. Only one speculative preparation
runs at once; a second worker remains available for demand. A running hint pauses
only when a demanded preparation is waiting for a worker, and resumes when one is
free. Identical model pulls
on this worker share a per-manifest transfer lock, and each pull uses at most eight
streams. TensorFS owns bytes, object verification, cache, credentials, and native
cancellation. Different manifests sharing individual objects are not guaranteed
to share an in-flight network request; they still reuse verified cached objects.

Actual invocation joins unfinished preparation and reuses a finished one with the same
exact selection. Once this worker's store collector has run since, demand revalidates
source custody instead, restoring evicted bytes; each child's admission still retains
and natively verifies its own inputs. A failed hint
does not fail unrelated work; a later demand can retry the same selection. Native
input holds belong to each actual child. Temporary preparation holds and read
leases are released when preparation ends; interrupted temporary holds are cleaned
through the existing workspace journal on restart.

Canceling one parent detaches its interest. Native cancellation occurs only after
the last consumer leaves. Completed verified TensorFS objects survive cancellation
and can be reused; model pull resumes at the object boundary. CPU validation
already in progress finishes within its ordinary construction boundary. Preparation
counts as active work until it stops, while cached metadata does not extend the
rental idle window. Background progress is a log observation (once per hundredth of
its bytes, with `unit: bytes` and a rate) and does not overwrite the current inference
stage or percentage.

Every model pull is two `model fetch` log records on the parent: `start` (model,
manifest, entrypoint, whether it is a prefetch, the step it blocks) and `end` (bytes,
moved bytes, elapsed time, rate). A call held for its callee's preparation writes
`model wait` records (`start`, then `end` with `waited_ms`) naming its step, so a client
times the step without the wait and the wait as the download's.

This optimization currently applies to captured machine-local composition.
Unsupported remote/older workers can decline it and keep ordinary demand loading.
Authored packages importing `prefetch` require `cozy-runtime>=0.18.30`; the worker
requires `tensorfs>=0.3.55` for native cancellation/progress. No network wire change
is involved. Refresh the public dependency lock after TensorFS 0.3.55 publication;
the coordinated private image must install the exact 0.3.55 candidate wheel until
that release is available. Local CPU/HTTP tests do not establish GPU throughput or
a few-second cold compatibility time.
