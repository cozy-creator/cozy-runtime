# Prepared resident GPU repeats

Python prepares a deterministic tensor step and its immutable inputs. The Rust ABI3 extension
retains the CUDA graph template, external ATen tensor dependencies, explicit owners and the
capture pool. CUDA clones the source definition into a conditional WHILE body. A device kernel
advances the iteration index and decides whether the GPU repeats. Python is absent from loop
continuation; the host submits one graph for any number of steps.

The public entry point is `from cozy_runtime.compute import PreparedRepeat`.
The first supported operation is `PreparedRepeat.capture(step, state=..., steps=..., owners=...)`.
Optional `prefix=` and `suffix=` callbacks record one-time tensor work before and after the loop.
All phases retain their graph and storage owners; the suffix runs only if the loop completed and
cancellation had not won at its GPU entry boundary. A suffix already admitted is joined to completion.
This resident composition keeps its combined weight dependencies pinned; it does not stream weights
between phases or make graphs survive physical eviction.
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

Each supplied Runtime Binding describes dependencies; capture acquires a separate retained Binding
for the prepared definition. The caller may close its original Binding after capture. Definitions
are attached to the existing component graph owner, so idle offload and changed-fill retirement
close their native graph before releasing their own bindings. Active executions hold model-use
claims through completion and owned readout; an offload cannot cancel them to obtain memory.
Definitions for unrelated component roots remain untouched. Re-capture after retirement acquires
new bindings and addresses. Preparation bypasses existing replay wrappers and captures original
tensor operations, never a nested Torch executable launch.

Since Runtime 0.32.2, `owners=(component,)` accepts placed component roots directly. Runtime
resolves their existing placement and retains the complete weight binding; packages never need
private residency attributes. `resident_only=True` admits only weights already on the device.
The idle preparation guard refuses a backing change before any source read instead of paging an
oversized model in to capture it. It does not turn the surrounding inference request into an
overlapping request: later component transitions keep their normal phase priorities.

Capture refuses package forward hooks and per-step LoRA choices. Fixed baked LoRA uses the
attempt's existing fill identity. The existing graph state guard restores module buffers/plain
attributes and rejects a body that changes them. The kernel implementation epoch must stay fixed
through recording and subsequent submissions. Mutable repeat state must be ordinary tensors,
allocated or cloned under `torch.inference_mode(False)`; preparation runs with inference mode
disabled so its saved state remains writable. Immutable inputs may retain their original mode.

Missing residency, unsupported hooks and model-state changes are
`CapabilityError(code="prepared_graph_ineligible")`. A capture-only capacity failure remains a
typed OOM. Both paths drain queued warmup and restore declared state before returning; capture
allocations are reset on failure. An author can continue eagerly after such a preparation refusal,
but must not catch a submission/execution error and replay the request. Hidden Python side effects
outside the declared tensor state and component remain outside the deterministic-body contract.

Within a normal invocation, `Context.plan` registers every pending prepared-work owner before
calling its provider. Submissions are ordered after their predecessors finish; a prior failure
or cancellation prevents later provider calls. Late and recursive submissions are refused.
The request joins before success serialization and cancels/joins every accepted owner on unwind,
including submissions still waiting for a predecessor. Admission locks never span provider
callbacks or blocking device work. Reusable graph
definitions are not automatically closed by request completion. Low-level `submit()` outside an
invocation remains caller-owned. Native launch orders the caller's stream after its completion
event; other streams and host readers must await results or establish their own dependency.
Submission during CUDA capture is refused before allocation/launch.

`PreparedSequence((first, second, ...))` instead prepares one ordered native submission from
distinct stage definitions. Stages communicate through persistent GPU boundary tensors allocated
outside capture: the second graph reads the first graph's actual output address. The Rust worker
advances existing same-process continuation capabilities without Python callbacks or per-stage
Python output copies. The default returned state is the final stage's state, read only after
the entire sequence completes. Explicit selected outputs are also end-of-sequence values, not
snapshots of intermediate stages.

The package author chooses this sequential mechanism or the existing explicit prefix/body/suffix
composite when its shapes permit that optimization. There is no consumer setting or automatic
fusion. Sequential stages still need native host launches/waits; this is not CPU-free disk I/O.
Deferred source capabilities use the same worker to refill their own retained virtual addresses
and run their graph. Native physical pins guard each actual use. The internal raw-stage helper
removes only captured immutable weight owners, then refuses detachment if a real foreign Pin
survived; it is not a production model adapter or a complete generation/LoRA validator.

A sequence takes ownership of its stage definitions. Failed constructor validation does not
consume them. Standalone stage submission is refused while attached; closing or idle-retiring
one stage closes the sequence. Every accepted capability remains owned through cancellation,
including unstarted stages. An uncertain completion retains the actual source/graph chain.
All-resident sequences use existing bounded next-job prefetch; streamed sequences do not yet
speculate into their future weight/workspace window.

`PreparedSequence(stages, repeat=(begin, end, passes))` repeats the half-open range natively,
with the prefix and suffix outside that range executed once. For example,
`(encode, block_a, block_b, sampler, decode)` with `repeat=(1, 4, 30)` runs thirty complete
denoising passes before decode. The private graph iteration index still resets for each stage:
the package must declare a shared GPU timestep tensor, read it from the block graphs and
advance it once in the sampler. First-step variants can be separate prefix definitions, followed
by a repeated range of later-step variants. The runtime does not infer either schedule.

Repeated stages require the existing continuation.v1 optional reusable tail (flag16, size112).
Their definition remains reserved between checked passes; rearm never records a caller/PTDS
event on the worker. Cancellation remains sticky, failed/invalid/unproven inputs cannot rearm,
and checked finish releases the reservation. Source identities and model claims remain fixed
for the complete submitted program. This mechanism still needs actual model adapters, phase
preferences, shared-pool budgeting and real repeated streaming qualification.

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
Control kernels are loaded during preparation with `cudaFuncGetAttributes`. The real
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
private control state resets before each graph launch; the index resets only inside the admitted
program body. State tensors remain in place: callers may restore
or replace their contents and input values between runs. `valid()` checks whether owners, layouts,
fill identities and kernel implementations still match; submission checks again. Unexpected errors
propagate from that query. During execution, callers must not resize tensors, change storage, mutate
inputs or access outputs until completion. Concurrent submissions are refused. Close cancels and
joins before destruction. Device-completion failure retains owners until process death. Forked
processes cannot use inherited native handles.

Preparation joins warmup and its state restoration, then returns inactive Torch cache before
capture asks for a private graph pool, as the `torch.cuda.graph` context does. This does not
discard live tensors, native weight pins or existing graph pools, and never retries submitted
work. `facts().capture_cache` records the observed Torch allocated/reserved bytes at that
boundary and the reserved-byte reduction. Those figures are not transfer or GPU-stall timings.

Real RTX4070 Laptop8GiB runs17944–17955 passed the twelve resident-phase gates on Runtime8c057,
Creator/TensorFS51a: phase parity, changed-input reuse, checked stored loading, zero-launch source
failure, phase-copy refusals and rollback, cancellation and alias release. Reuse17948 submitted the
same definition twice within one invocation; production cross-request caching remains separate.
Final-iteration cancel17954 skipped its suffix; tail17955 joined an already-admitted suffix.
On Runtime237ddc / Creator/TensorFS51a, later real GPU runs17977,17979 and17980 passed final-weight
dtype preparation, public `valid()` changed-input reuse and owner lifetime checks. The dtype gate
restored each representation from the immutable source, retained escaped aliases correctly, and
performed no reads or fills on unchanged warm reuse. These are primitive checks, not qualification
of a complete production model program.

Concurrent result observers and definition retirement share one readout/claim-release boundary.
Close sends cancellation before waiting for the observer lock, so a blocking result waiter cannot
prevent the control write that ends its GPU work. Retirement serializes against definition reuse;
an old execution handle cannot close a later submission. Pressure coordination first cancels and
joins active work, then uses ordinary idle reclamation; allocator recovery never disposes graphs
or invokes Python. Unrelated native eviction may still join the CUDA context conservatively.

Build through `scripts/build-execution-native.sh INSTALL_PREFIX` with `COZY_CUDA_TOOLKIT` pointing
to the build-time CUDA 13 toolkit. The artifact is `_kernels/_execution.abi3.so`, carried by the
existing native wheel assembly. Eight native SASS targets are included; there is no PTX or serving
JIT. Linux is the initial qualification target. No Varena representation registry or plan engine
is introduced: resident bindings own weights and the future native-fill descriptor supplies the
source/recipe/target identity.

This is a resident execution primitive, not a full H3 pipeline or an oversubscribed native plan.
General streamed binding and broader queue policy remain separate work.

## One queued raw weight block

An entrypoint can declare immutable component phases with `@app.entrypoint(phases=...)`. After a
fully prepared resident submission starts, Runtime may preload one raw block from the exact next
queued request in the same construction. The pending request and fill identity are checked before
reuse. Speculation uses spare capacity and never reclaims the current program's pinned dependencies;
if there is insufficient space, the next request loads the block through its ordinary demand path.
This path does not predict workspace for arbitrary eager work or prepare an entire future model.

Real default-CLI runs 18001/18002 on the RTX 4070 Laptop 8 GiB, Runtime d53000ed5 and Creator/TensorFS51a,
copied 64 MiB while the current conditional graph's completion event remained incomplete. The next
request reused the same Region/checkpoint, read zero additional demand bytes and returned the exact
matrix result. Runs 18005/18006 held a real Torch allocation to force capacity refusal: speculation
was skipped, current work completed, and the next request demand-loaded 64 MiB correctly.
Runs 18009/18010 canceled only the queued request while current work continued. That copy had already
completed, so this proves queued-token invalidation and cancellation safety, not active-I/O cancellation.
Each current program performed 300001 finite increments with one host graph launch. These control
iterations are not denoising steps or generated images. Full-model throughput, encoded/transformed
preloads and multiple ranks require separate qualification.

Exact fixture/artifact hashes and ordinary CLI receipts are retained in
`outputs/stock-torch-20261024/queued-preload-d53000ed5-20261025` in the Cozy workspace.

## One native load followed by its dependent repeat

`PreparedRepeat.capture(step, state=..., steps=..., load=slot)` accepts a dedicated
`tensorfs.varena.LoadSlot`. Its initialized U8 capture tensor owns a still-unpublished Varena Fill.
Python captures the known fixed body and seals the slot before any read producer starts. The first
`ctx.plan(prepared)` submission reserves a native epoch, records caller readiness on the caller's
OS thread and passes a checked continuation capsule to TensorFS's existing native worker. That
worker reads the actual stored parts and uses TensorFS's shared decoder for encoded slots.
Default `load_ordering="checked"` commits after checked producer completion, retains the Pin
and calls the native repeat. No Python callback or observer drives this transition.
Loaded submissions are one-shot; arbitrary cold-model capture is not supported by this interface.
Setup allocation, zero initialization and capture are preparation
costs, not hidden parts of the measured stored-byte read.

The two extensions remain separate. `include/cozy_execution_v1.h` is the private C-layout contract:
version/size/capabilities/device/PID, an opaque owner, and retain/release/start/poll/cancel/wait.
No Rust object layout crosses the DSO seam and Runtime does not link a second Varena registry.
The capsule and provider module remain owned through completion. Start/cancel arbitrate through
an atomic gate. Cancel is nonblocking; the native worker's event-query wait loop delivers a
pending GPU cancel after launch initialization. Only public progress observers request snapshot
kernels; internal waiting never generates a millisecond stream of GPU snapshots.

A never-recorded CUDA event is not a promise of future work. While the native read is pending,
composed submission does not promise immediate ordering of later caller-stream work. Call
`Execution.wait()` or rely on request-scope joining before output access. The native launch uses
its explicitly owned stream and the caller-ready event that was already recorded; a worker never
reinterprets the caller's per-thread default stream. Pure resident submission ordering is unchanged.

`Execution.facts()` includes native load bytes and completion timestamps. In checked mode, failed
reads/decodes and cancellation before native start never launch the graph. A checked ordinary failure remains visible to observers but permits
close to release graph owners; `completion_proven` comes from the native job, never from an error
class. Sticky or unproven completion retains actual source, Fill/Pin, graph and module owners.
The existing request scope still owns the complete submission on success, cancellation and unwind.

Source/build work is not GPU qualification. The corresponding real Creator release-gate cell must
prove nonzero stored reads feed the known body while Python is parked, active cancellation while
the native worker waits, zero launches on prelaunch failure/cancel, and final ownership cleanup.

### Event-ordered input

`load_ordering="events"` requires both native capabilities before capture. `"auto"` selects events
only when both are present, otherwise it selects checked ordering before submission. Facts expose
both `load_ordering_requested` and selected `load_ordering`; a fallback is never event-mode proof.
The optional, size-validated `start_after` ABI tail preserves the original checked ABI prefix.

The one-placement source callback runs only after every host read, copy and padding span has been
successfully enqueued. It records owned native-stream dependencies and submits the graph without
waiting for copies on the host. Encoded slots enqueue the existing prepared decoder first. Its
existing device error flag gates the complete program: public index reset, prefix, loop and suffix.
The loading worker still checks completion and ownership, but its return does not drive GPU
continuation. The Fill remains unpublished and physically pinned through composed completion.

Input/preflight/read failure before enqueue commitment means zero graph submissions and no state
mutation. Encoded value rejection can submit one graph but executes none of its authored body;
`body_entry_checked`, `body_entered` and `input_invalid` distinguish this from graph submission.
Cancellation after enqueue commitment is cooperative execution cancellation. Late device or
ownership failure may occur after body entry: discard the result, retain uncertain owners and
retire the executor. Such a failure never establishes a retroactive zero-execution claim.

Event mode is source/build work until its matching raw, encoded, failed-input, cancellation and
alias-lifetime release cells pass on real CUDA. Host issue/completion timestamps alone do not
prove the absence of a GPU bubble; device ordering and the complete result remain separate gates.
