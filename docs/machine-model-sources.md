# Optional machine-owned model sources

The existing device executor can select `Load.descriptor_sources=true` only after its
hello advertises `model_sources.descriptors/1`. Older commands default to false and use
the existing Store/ReadLease provider. The descriptor capability initially covers one
rank; group/NCCL source delegation is a separate gate.

The new provider changes the byte source while retaining existing Checkpoint header
parsing/rows/read-plan, ModelRegistry, Loader/backend fit, model implementations and
generic residency. It requests header/assets/object descriptors through the existing
durable control exchange. No request occurs per step or block.

```text
request: event=request, seq=N, kind=model_source_read,
         manifest=SELECTED_MANIFEST, role=header|asset|object, name=NAME, length=EXPECTED
answer:  event=answer, seq=N, ok=true, sha256=LOWERCASE_HEX, length=BYTES, descriptor=true
         followed by one read-only regular SCM_RIGHTS descriptor
```

Header and asset data use fd bytes, not inline control arrays. Python reuses native
`tensorfs.plan` over the checked header. Its typed plan rows select only required object
fds; a source is built per component while its provider coalesces object descriptors
across components. Header bytes and imported descriptors are held until backend teardown.
The machine must authorize the selected manifest/asset/object and retain its closure.
Readonly-file transport is distinct from connected-socket Handoff and mutable host-tier
memfd transport; numeric fd values from JSON never become authority.

Metadata blobs are verified against the received SHA-256/length. Native SourceDescriptor
duplicates and verifies each immutable object fd, then retains its pageable read-only
inode mapping and closes the transient fd. Object count does not grow the consumer's
open-FD set; provider teardown releases mappings after the last source/weight-set drops.
`Checkpoint` opens no local
Store; backend expect skips local read-lease acquire and component registration uses the
machine source provider. Legacy code still performs its original lease/source operations.

## Qualification boundary

The CPU test runs a real owner TensorFS store, real typed Executor durable socket/fd
exchange, existing Checkpoint metadata/read-plan, and native host-plane fill. It checks
exact bytes, no consumer store/catalog/CUDA, descriptor coalescing, old Load defaults,
readonly transport and malformed-response fd cleanup. It is not real SDXL/GPU inference.

On 2026-10-02, all 34 selected CPU tests passed: five source/transport checks plus the
existing read-lease and cross-version seam checks, using the compiled TensorFS descriptor
dependency. Ruff, compilation and whitespace checks passed. The dependency's native plane
metadata/fill/stub checks and 17 Rust descriptor/legacy-host tests passed separately.
An unqualified group source mode refuses the affected operation before Torch/device
preparation; it does not prevent the peer's baseline operations.

The Rust machine adapter must implement the source kind and authorize each grant; the
TensorFS descriptor-source candidate must be installed as a coherent native package.
Source import currently adds a full object verification read, and one object grant is a
load-time round trip. Both need matched performance evidence before cold-start claims.
The source hook does not make all host allocations machine-owned: the existing executor
can still fill its own pinned tier, allocate staging/activations, and retain model assets.
Those private bytes must remain visible in the machine ledger and budget/recovery policy.
Sole-writer shared host-tier adoption and universal low-memory recovery remain separate.
Mapped virtual ranges, VMA entries and mapped RSS are also consumer resources. The owner
legacy ReadLease still retains its complete descriptor closure and raises the soft FD
limit toward the hard limit; this is known released behavior, not a general-size solution.

After the CPU gates, root coordinates unchanged SDXL through the stock executor and
existing model/codec path, with source mode explicitly recorded and valid saved WebP.
No model/worker code is copied. No GPU or rental was used by this agent.

Diagnostic continuation retains the same matched candidate code and full integrity
checks, emitting aggregate stderr records per component for source RPC/constructor/build,
native registration/host-tier exchanges, and initial pinned registration enqueue. It adds
no per-object logs, per-step control messages, extra waits or stream synchronization.
Pinned enqueue time is not fill completion or CUDA HostReg time; those remain native/GPU
work and must be interpreted with the existing load-stage telemetry. The diagnostic
Runtime wheel is installed only in a separate immutable `profile-env`, preserving the
original candidate and SDK99. All 34 source/read-lease/skew CPU tests, Ruff and compile
checks pass with these timing records.
