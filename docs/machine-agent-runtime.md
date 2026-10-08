# Machine and Runtime roles

The Rust machine owns accepted runs, preparation, scheduling, process lifecycle,
software updates and its durable journal. Local and remote clients reach its
`cozy.machine.v1` API. Runtime contains the Python author API and device executor;
it does not expose a Python worker server or worker protobuf facade.

The machine launches `cozy_runtime.internal.executor` through the private bounded
JSON/descriptor channel. The executor imports the selected package, constructs
and invokes it, and consumes machine-granted native TensorFS source/output and
weight-plane handles. TensorFS owns the native storage semantics.

A `Load` constructing new model-bearing state must offer `model_sources`; otherwise it refuses with
`model_sources_ungranted` before that load imports model code, measures hardware or
initializes a device. Headers and declared assets come from the owner's `ModelSource`
exchange, including follower requests relayed through rank zero. Runtime parses these
bytes and plans reads locally. Missing grants never fall back to opening the binding's
store path. Reusing an already loaded construction retains its existing grants and opens
no store. Weightless loads do not require this offer.

Jobs receive source/output channels only when their owner sets `RunJob.weights`.
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
contract. The machine API and journal are documented in cozy-machine; the former
Python worker database views and capability-report command are retired.
