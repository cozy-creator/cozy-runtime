# Prepared resident GPU repeats

Python prepares a deterministic tensor step and its immutable inputs. The Rust ABI3 extension
retains the CUDA graph template, external ATen tensor dependencies, explicit owners and the
capture pool. CUDA clones the source definition into a conditional WHILE body. A device kernel
advances the iteration index and decides whether the GPU repeats. Python is absent from loop
continuation; the host submits one graph for any number of steps.

The public entry point is `from cozy_runtime.compute import PreparedRepeat`.
The first supported operation is `PreparedRepeat.capture(step, state=..., steps=..., owners=...)`.
`ctx.plan(prepared)` submits it and returns an `Execution`; existing string/component plans remain
hints. The callback receives a CUDA int64 index tensor and changes declared state in place. Its
timestep, guidance and noise tables are ordinary immutable GPU tensors captured by the callback.
ATen operations tagged as seeded nondeterminism are refused: native launch does not invoke Torch's
generator replay prologue. No arbitrary Python branches, scalar reads or functions are compiled.
The prepared object is an immutable fixed program: Python branches, precision, attention choices
and kernel implementations are those selected during capture. Changed choices require preparation
of a new program. A higher-level reusable model-plan cache must key those choices and its source,
target representation and transform/LoRA identity; this primitive introduces no such registry.
Supplied Runtime Bindings are checked for validity and identity once per submission, along with
every external tensor's geometry and storage pointers. Non-ATen launchers must supply their native
module and storage owners explicitly.
Node-kind inspection does not discover pointers hidden in custom/native kernel parameters.
ATen dependency tracking and graph-pool ownership cover the supported Torch path; custom launches
without complete explicit buffer/module owners are outside the admitted contract.

Within a normal invocation, `Context.plan` registers one pending prepared-work owner before
calling the provider. The request admits one complete prepared submission, rejects late or
additional submissions, joins before success serialization, and cancels/joins unfinished work on
unwind. Admission locks never span provider callbacks or blocking device work. Reusable graph
definitions are not automatically closed by request completion. Low-level `submit()` outside an
invocation remains caller-owned. Native launch orders the caller's stream after its completion
event; other streams and host readers must await results or establish their own dependency.
Submission during CUDA capture is refused before allocation/launch.

The complete source graph is walked recursively. Only kernel, empty, memcpy, memset and ordinary
child graph nodes are accepted. Copy and memset nodes must address device memory on the selected
device; host-ended copies are rejected because graph pools do not own CPU temporaries. CUDA
instantiation enforces the conditional body's additional context constraints. Host callbacks, allocation/free, event, external
semaphore and existing conditional nodes are refused. Device graph launch is not mixed into the
conditional body. Known graph-layout ineligibility is a capability refusal; actual CUDA failures
retain their driver error code.

Actual CUDA stream capture holds the process-wide capture exclusion. Prepared replay does not:
it retains every external Tensor and original Storage plus supplied Binding owners, so each
dependency's physical pin remains held through GPU completion and owned output readout. The
graph contains no allocation, I/O, host callback or event-dependent work that could wait on a
reclaiming host thread. Unrelated reclamation still performs the native core's conservative
context join; this change makes no claim of overlapping eviction. A future-fill overlap gate
must map/admit its distinct destination and prepare staging, streams and decoder kernels before
launch, then publish through its own producer event. Actual capture remains excluded throughout.

Cancellation is a GPU atomic write from a separate control stream. The boundary kernel observes
it after a complete body. Progress uses a control-stream kernel reading device atomics into an
owned snapshot, followed by ordered D2H and an event; CPU never races a device write to the same
host flag. Observers can stop polling without stopping the GPU. Cancellation latency depends on
the longest body/kernel and scheduling of the control kernel; it is not GPU preemption.
All three control kernels are loaded during preparation with `cudaFuncGetAttributes`. The real
gate caught a lazy-loading serialization: without this preparation cancellation was submitted
promptly but ran only after the loop ended. Stream priority alone did not fix it.

Real default-CLI gate 17019 passed on Torch 2.14.1+cu130 and TensorFS 0.15.2: a 21-node denoiser
body executed 30 steps while the submitter slept, with exactly two Python preparation calls and
zero numerical difference from eager. Reuse, stale handles, owned outputs and eligibility refusals
passed; cancellation stopped a 100000-step run at step 1094, and teardown reclaimed 2 MiB of
native weight backing. This is one tested device/cohort, not a general cancellation latency claim.

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
