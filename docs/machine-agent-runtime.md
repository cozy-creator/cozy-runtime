# Machine and Runtime roles

TensorD (the Rust `cozy-machine` executable) owns accepted runs, preparation,
scheduling, machine-wide memory policy, storage lifetime, process supervision and its
durable journal. Local and remote clients reach its `cozy.machine.v1` API. Runtime
supplies the Python author API and executors that construct models and run package code.

TensorFS is a library on both sides: TensorD embeds its store and buffer implementations;
Runtime calls its parsing, read-planning and weight-plane/device APIs inside the executor.
There is no separate TensorFS daemon. TensorD makes model weights available to Runtime;
Runtime turns the available data into the model's tensor views and executes GPU operations.

| Representation | Responsibility |
|---|---|
| Serialized weight objects, manifests, headers and assets | TensorD obtains and retains them through TensorFS, then provides scoped access to the selected model's data. |
| Prepared CPU weight buffers | Runtime describes a layout/read plan; TensorD fills and retains shared buffers through TensorFS. Runtime maps/pins the supplied buffers and can stream from supplied read-only object files. |
| GPU-resident tensor data | Runtime allocates, copies, maps and unmaps through TensorFS's plane/device APIs, and binds PyTorch tensor views. TensorD coordinates budgets and may retain exported GPU allocation handles for reuse. |

A GPU allocation handle retained by TensorD keeps backing memory available; each executor
still attaches/maps that memory and constructs its own Python/PyTorch model. These handles
and shared CPU buffers are not themselves model objects. TensorD's grants remain real
bounds on access and resource use, alongside the data/descriptors it supplies.

TensorD launches `cozy_runtime.internal.executor` through the private bounded JSON/descriptor
channel. The executor imports the selected package, constructs and invokes it, and reports
measured resource use. The private channel's existing names (`ModelSource`, `HostTier`,
`SealedTier`, `DeviceTier`, and budget grants) describe particular data/access exchanges.

A `Load` constructing new model-bearing state must offer `model_sources`; otherwise it
refuses with
`model_sources_ungranted` before that load imports model code, measures hardware or
initializes a device. Headers and declared assets come from TensorD's `ModelSource`
exchange as actual bytes in read-only descriptors, including follower requests relayed
through rank zero. Runtime parses those bytes and plans reads locally. Weight payload
access uses the supplied CPU buffers/object descriptors and the executor weight plane. Missing grants never fall back to opening the binding's
store path. Reusing an already loaded construction retains its existing grants and opens
no store. Weightless loads do not require this offer.

Jobs receive source/output channels only when TensorD sets `RunJob.weights`.
Installing a full TensorFS wheel does not supply that authority. Jobs without storage
still run; `ctx.tensorfs_source(model)` and `ctx.output(slot)` refuse only when called
without their respective grant. Output commit, checkpoint and handle-close semantics
remain TensorFS's existing transaction contract.

Standalone callers and tests can explicitly open their own store and use the internal
`Checkpoint.from_store(store, manifest)` helper. Managed executors instead construct
`Checkpoint(manifest, sourced=...)`, which has no local store lease. This API distinction
is not a same-user sandbox: TensorFS still supports cooperating standalone writers, and
TensorD's owned-store rules retain their existing collection/lifetime scope. See
[tracker #339](https://github.com/cozy-creator/tracker/issues/339).

The Runtime CLI reads/scaffolds package interfaces and reports installed
distribution/build provenance. It does not advertise the retired worker wire
contract. The machine API and journal are documented in the TensorD repository; the former
Python worker database views and capability-report command are retired.
