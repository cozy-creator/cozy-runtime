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
duplicates, verifies and retains each immutable object fd. `Checkpoint` opens no local
Store; backend expect skips local read-lease acquire and component registration uses the
machine source provider. Legacy code still performs its original lease/source operations.

## Qualification boundary

The CPU test runs a real owner TensorFS store, real typed Executor durable socket/fd
exchange, existing Checkpoint metadata/read-plan, and native host-plane fill. It checks
exact bytes, no consumer store/catalog/CUDA, descriptor coalescing, old Load defaults,
readonly transport and malformed-response fd cleanup. It is not real SDXL/GPU inference.

On 2026-10-02, all 33 selected CPU tests passed: four source/transport checks plus the
existing read-lease and cross-version seam checks, using the compiled TensorFS descriptor
dependency. Ruff, compilation and whitespace checks passed. The dependency's native plane
metadata/fill/stub checks and 17 Rust descriptor/legacy-host tests passed separately.

The Rust machine adapter must implement the source kind and authorize each grant; the
TensorFS descriptor-source candidate must be installed as a coherent native package.
Source import currently adds a full object verification read, and one object grant is a
load-time round trip. Both need matched performance evidence before cold-start claims.
The source hook does not make all host allocations machine-owned: the existing executor
can still fill its own pinned tier, allocate staging/activations, and retain model assets.
Those private bytes must remain visible in the machine ledger and budget/recovery policy.
Sole-writer shared host-tier adoption and universal low-memory recovery remain separate.

After the CPU gates, root coordinates unchanged SDXL through the stock executor and
existing model/codec path, with source mode explicitly recorded and valid saved WebP.
No model/worker code is copied. No GPU or rental was used by this agent.
