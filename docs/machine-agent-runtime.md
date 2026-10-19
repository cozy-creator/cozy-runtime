# Machine and Runtime roles

TensorD (the Rust `tensord` executable) owns accepted runs, preparation,
scheduling, machine-wide memory policy, storage lifetime, process supervision and its
durable journal. Local and remote clients reach its `cozy.machine.v1` API. Runtime
supplies the Python author API and executors that construct models and run package code.

TensorFS is a library on both sides, and the machine's store is shared: TensorD and the
Runtime each read and write it through TensorFS. There is no separate TensorFS daemon. The
Runtime opens the checkpoints a request names by their header ref, reads configs and assets
from their components, fetches what the store lacks under the request's Hub capability, and
writes and uploads derived weights itself.

| Representation | Responsibility |
|---|---|
| Headers, chunks and assets | The shared store. TensorD may prefetch public releases into it; the Runtime opens what a request names, fetching the rest on first read. |
| Prepared CPU weight buffers | Runtime describes a layout/read plan; TensorD fills and retains shared buffers through TensorFS. Runtime maps/pins the supplied buffers and reads unstaged regions from the store's chunk files. |
| GPU-resident tensor data | Runtime allocates, copies, maps and unmaps through TensorFS's plane/device APIs, and binds PyTorch tensor views. TensorD coordinates budgets and may retain exported GPU allocation handles for reuse. |

A GPU allocation handle retained by TensorD keeps backing memory available; each executor
still attaches/maps that memory and constructs its own Python/PyTorch model. These handles
and shared CPU buffers are not themselves model objects. TensorD's grants remain real
bounds on access and resource use, alongside the data/descriptors it supplies.

TensorD launches `cozy_runtime.internal.executor` through the private bounded JSON/descriptor
channel, with `TENSORFS_HOME` naming its store. The executor imports the selected package,
constructs and invokes it, and reports measured resource use. The channel's exchanges
(`SealedTier`, `DeviceTier`, budget grants, `CheckpointItem`) carry layouts, GPU handles and
results, never weight bytes.

`Load.hub` and `Invoke.hub` carry a request's Hub endpoint, capability and key; TensorFS
trades them for a DPoP token on its first Hub call, so a warm run makes none. Without a hub
the Runtime reads only what the store holds.

An attempt's `Weights[M]` sources are the header refs in `Invoke.sources`; its weights outputs
are `Invoke.weights`. An output is written with TensorFS's `Writer` straight into the store
under its replay key (`weights_store.replay_key`: actor, operation, resolved request, output,
derivation) and kept as the `local/derived` revision of that key, so the same work replays
and an interrupted one resumes. With `Invoke.destination` it is then uploaded under the
request's capability; a failed upload ends the run `upload_failed` with the local checkpoint
listed (`cozy run upload` retries the upload alone).

The Runtime CLI reads/scaffolds package interfaces and reports installed
distribution/build provenance. It does not advertise the retired worker wire
contract. The machine API and journal are documented in the TensorD repository; the former
Python worker database views and capability-report command are retired.
