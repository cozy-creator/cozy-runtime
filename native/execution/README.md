# Prepared resident GPU repeats

Python prepares a deterministic tensor step and its immutable inputs. The Rust ABI3 extension
retains the CUDA graph template, external ATen tensor dependencies, explicit owners and the
capture pool. CUDA clones the source definition into a conditional WHILE body. A device kernel
advances the iteration index and decides whether the GPU repeats. Python is absent from loop
continuation; the host submits one graph for any number of steps.

The first supported operation is `PreparedRepeat.capture(step, state=..., steps=..., owners=...)`.
`ctx.plan(prepared)` submits it and returns an `Execution`; existing string/component plans remain
hints. The callback receives a CUDA int64 index tensor and changes declared state in place. Its
timestep, guidance and noise tables are ordinary immutable GPU tensors captured by the callback.
ATen operations tagged as seeded nondeterminism are refused: native launch does not invoke Torch's
generator replay prologue. No arbitrary Python branches, scalar reads or functions are compiled.

The complete source graph is walked recursively. Only kernel, empty, memcpy, memset and ordinary
child graph nodes are accepted. CUDA instantiation enforces the conditional body's additional
device/context and copy-address constraints. Host callbacks, allocation/free, event, external
semaphore and existing conditional nodes are refused. Device graph launch is not mixed into the
conditional body. Unsupported hardware or graph operations return a typed CUDA failure.

Cancellation is a GPU atomic write from a separate control stream. The boundary kernel observes
it after a complete body. Progress uses a control-stream kernel reading device atomics into an
owned snapshot, followed by ordered D2H and an event; CPU never races a device write to the same
host flag. Observers can stop polling without stopping the GPU. Cancellation latency depends on
the longest body/kernel and scheduling of the control kernel; it is not GPU preemption.

Each definition can be submitted repeatedly after its prior execution finishes. Native epoch
tokens reject stale cancel/wait/close operations. Every execution owns distinct output buffers;
readout completes before a later execution can overwrite state. The GPU index and
control state reset before each graph launch. State tensors remain in place: callers may restore
or replace their contents between runs, but must not resize, change storage, mutate immutable
inputs or access outputs until completion. Concurrent submissions are refused. Close cancels and
joins before destruction. Device-completion failure retains owners until process death. Forked
processes cannot use inherited native handles.

Build through `scripts/build-execution-native.sh INSTALL_PREFIX` with `COZY_CUDA_TOOLKIT` pointing
to the build-time CUDA 13 toolkit. The artifact is `_kernels/_execution.abi3.so`, carried by the
existing native wheel assembly. Eight native SASS targets are included; there is no PTX or serving
JIT. Linux is the initial qualification target. No Varena representation registry or plan engine
is introduced: resident bindings own weights and the future native-fill descriptor supplies the
source/recipe/target identity.

This is a resident execution primitive, not a full H3 pipeline or an oversubscribed native plan.
Streaming, event-based refill, queue policy and the CPU output handoff remain separate work.
