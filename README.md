# cozy-runtime (v2)

The worker runtime: author surface (`cozy_runtime.author`), worker, disposable device
executor, and the single-package CLI. Design authority: [tracker-v2](https://github.com/cozy-creator/tracker-v2)
`cozy-runtime.md` + `cozy-runtime-cli.md`; issues in `tracker/cozy-runtime/`.

## Compatibility status

Cozy Runtime is pre-launch `0.x`: its Python author API may still change incompatibly. Package
code has one supported production import root, `cozy_runtime.author`; package tests may also use
`cozy_runtime.author.fakes`. Underscored author modules, `internal`, `cli`, and `protocol` are not
package APIs.

Runtime source master is `0.0.11.dev0` and targets standard CPython 3.12 only. Project
tooling pins `3.12`; no other Python interpreter or ABI is supported.

Proto-014 owns the eventual v1 cut. Before that cut, the author surface will be pruned and a v1
baseline recorded. From v1 onward the distribution major is the Python compatibility major:
minor releases are additive, breaking author changes require a new major, and candidate base
worker images must replay the supported-major package corpus before activation. The protobuf
package major and additive wire minor remain their own worker-control compatibility contract.

**No behavior test suite** (decisions.md #160, #533): CI runs build, fences, and
deterministic contract/arithmetic/staleness checks; everything else is verified by running
the real thing and recording what it did, plus time/memory benchmarks.

## Layout

| package | role | may import |
|---|---|---|
| `cozy_runtime.author` | the whole author surface — the only package code imports | stdlib + msgspec |
| `cozy_runtime.internal` | runtime implementation; owns the ONE config authority | author, protocol, stdlib |
| `cozy_runtime.protocol` | `cozy.worker.v1` — worker-protocol's bindings, copied verbatim into `src/cozy/`, plus canonical document handling | protobuf, internal.canonical |
| `cozy_runtime.cli` | user CLI plus the fixed product-host worker entrypoint | author, internal |

`cozy_runtime.author` also holds the INVOCATION KERNEL (`_invoke.py`): decode -> resolve
-> preflight -> hydrate -> invoke -> validate -> finalize. It is pure surface semantics —
no device, config, environment, protocol or network — so it lives beside the types it
enforces, and `internal` depends on `author`, never the reverse. Every door (`run`, `job`,
a served attempt) is a user experience over that one function, and the per-service fakes
in `cozy_runtime.author.fakes` are capability adapters over it rather than a parallel
runtime.

`internal/config.py` is the only module permitted to read process environment: one read at
the entrypoint, a frozen `RuntimeConfig` passed explicitly thereafter. Second read refuses.
`internal/child_env.py` holds the device-child env projection allowlist as data;
`internal/spawn.py` executes the seal (ERASE -> IMPOSE -> SEAL) around it.
`internal/exits.py` is the one shared exit-code matrix.

## The two processes, and the two RecordOwners

    a real RecordOwner ──gRPC───┐
                                ├──▶ worker (no CUDA) ──seam──▶ executor (CUDA)
    an in-memory RecordOwner ───┘        cozy.worker.v1, either way

Product hosts run `cozy-runtime-worker`, a private no-argument entrypoint that constructs
the worker core directly. On a base worker image it starts as root, consumes the six-value
owner-neutral host handoff, proves the worker and media TLS listeners, publishes readiness,
then opens Claim admission. The fixed handoff also carries the operator-owned direct Tensorhub
API origin as canonical `TENSORHUB_ORIGIN=https://host[:port]`. Package executors alone drop to UID/GID
65533. The removed
wrapper executable has no compatibility alias; placement acquisition now lives in
`internal/placement_materialization.py`.

The handoff's one `COZY_RECORD_OWNER_AUTH_JSON` value is a closed record containing the rental's
Ed25519 control public key and the media bearer's sorted hashes. Claim authenticates the
RecordOwner; media hashes open only media HTTP. Neither reaches an executor. The exact
PlacementSet is the sole package/model selection. Runtime names only its digest to
`POST /v1/worker/downloads/resolve` over its existing per-boot mTLS leaf, then downloads the
returned package wheels and model-weight files directly from their short-lived R2 GET URLs.
The supervisor may project its decoded private CA as `SSL_CERT_FILE`; Runtime accepts no CA
payload environment variable, delegation document, certificate bypass, proxy certificate
header, or authentication fallback.

| module | role |
|---|---|
| `internal/worker/session.py` | the protocol client and the worker entrypoint |
| `internal/worker/control.py` | the worker-owned control transport: gRPC and in-memory hosts |
| `internal/worker/attempts.py` | offer -> prepare -> accept -> terminal, attestation, ledger close |
| `internal/worker/ledger.py` | the worker-scoped byte ledger and the attempt reconciliation |
| `internal/worker/plan.py` | the AttemptPlan, the PlanChooser, the local binding-plan record |
| `internal/worker/downloads.py` | canonical Claim verification and direct Tensorhub package/model download resolve |
| `internal/worker/grants.py` | DeliveryGrant reads/writes + the closed hub-call table |
| `internal/worker/child.py` | spawn/supervise/kill the executor; the two ordinals |
| `internal/residency.py` | component-use admission: materialize a declared set before entry, event-fence, make evictable |
| `internal/worker/observe.py` | typed boot steps, confessions, and the SILENCE-BOUNDED liveness detector |
| `internal/worker/triage.py` | bounded triage assembly and receipt-checked bundle store |
| `internal/worker/refusal.py` | reader-vs-data fault, verbatim upstream remedy, the REF_ORIGIN severity split |
| `internal/witness.py` | the out-of-process postmortem witness: it names the WORKER's own death |
| `internal/local.py` | the ephemeral local worker: one request, no network |
| `internal/worker/records.py` | bounded same-process worker facts; callers own restart recovery |
| `internal/seam.py` | the control-only, byte-accounted worker<->executor channel |
| `internal/trampoline.py` | the exec-first stage: pdeathsig, no_new_privs, oom_score_adj |
| `internal/executor.py` | the disposable device process — the ONE torch importer |
| `internal/fill.py` | the fill plane: a TensorFS `Checkpoint`, the pinned ring, the transaction |

`checks/architecture.py` keeps versioned identities in two disjoint registries.
`DOCUMENT_KINDS` contains only stored or digested documents that cross a process/repository
boundary: PackageDescriptor, EnvironmentReceipt, ModelConstructionContract, and the discriminated
Qualification document. `IMPLEMENTATION_IDS` contains the six encoding-provider names whose exact strings enter
implementation, plan, and generation digests; those names are executable identity, not document
formats. Command requests, planner values, worker records, proof-local state, and benchmark reports
are typed by their command/path or storage table and carry no global format tag.

The worker never imports torch, and that is proved against the RUNNING process
(`scripts/worker-live.py nocuda` reads its `/proc/<pid>/maps` and the device's compute-app
list), not against the import graph.

**One attempt kernel** (§8). A one-shot local run is not a second execution path: it is an
ephemeral local worker whose RecordOwner is a pair of in-process queues, over the same
process boundary, acceptance boundary, freeze point and ledger. The rule is
structural, not a promise — the `one-attempt-kernel` fence proves the invocation kernel has
exactly one caller (the device executor) and the attempt engine exactly one constructor (the
worker session), so a direct-call runner beside them cannot be written without turning
CI red.

**Observability is bounded, or it is not observability.** `author/_observations.py` holds the
ONE record shape, a ring bounded by count AND bytes that sheds oldest first and counts what it
shed, and an O(1) time accumulator (`Attribution`) that makes inference step time attributable
without one retained row per step — 324 ns per step against a ~100 ms handler leg. Progress
frames are never retained; their durable form is that accumulator plus the monotone position
the silence detector reads. A credential-shaped field is REFUSED at the emit boundary, not
redacted downstream — where `token` and `signature`, which are also ordinary telemetry nouns,
are credential-shaped only bare or directly qualified (`access_token`), so `tokens_per_second`
and `output_tokens` are admitted.

**Every terminal attempt commits one bounded triage record** — identity, applied posture,
ledger reconciliation, attribution, confessions, faults, and the capped observation tail —
written and fsynced BEFORE its write receipt is recorded and before the terminal that names
it. The handle is an opaque `subject_id`; there is no path in it and no way to name one. A
killed executor is therefore explainable from a directory, with no live process and no SSH.

**Every granted destination is CONTROL-PLANE-MINTED** (cr-042). The closed `HUB_CALLS` table
in `internal/worker/grants.py` spends a grant minted for one invocation, so nothing the
package produced — and nothing a request body can influence — chooses where bytes land; a
destination with no row is a SECURITY EVENT. Inputs remain local. Outputs use either a local
destination or the exact presigned HTTPS capability in `OutputAccess.url`; the worker streams
with `Content-Length` and `If-None-Match: *`, validates every resolved address, and refuses a
redirect-selected destination. Tensorhub holds the R2 credential; the worker never does.
Armed by `scripts/worker-live.py grants` end to end and `scripts/assets-live.py egress` at
the table.

**The executor's network claim is Python-deep and says so.** `internal/executor.py:main`
installs `sandbox.refuse_network`, so package or job code that dials is refused typed at
the first `NETWORK_EVENTS` spelling. That bounds a BUGGY package (law 15's same-tenant
rung). It is not a kernel bound: the executor runs in the worker's own network namespace
with no seccomp filter, and a C extension or `ctypes` reaches `connect(2)` underneath the
hook. `scripts/lifecycle-live.py --only egress` observes both halves against one reachable
address, so the claim cannot quietly grow. Granted inputs are fetched by the WORKER and
handed over as spool paths; the executor is never given an address.

**Liveness is silence-bounded, never clocked.** Progress means an advancing monotone position;
a wedge is consecutive observations that find the position unchanged. `give_up_after` counts
observations, not seconds — there is no boot timeout and there will not be one — and the
`timing-decides-nothing` fence proves the observation plane cannot reach residency or placement.

**Component residency is the runtime's, and the package has no vocabulary for it.** An
package declares the possible heavyweight set of each public model operation with
`@uses_components` and nothing else. Entering that scope ADMITS and MATERIALIZES the set
before the body runs — freeing non-declared components, LRU over refcount-0 leases whose
CUDA events have fired, until the plan's activation envelope fits ALLOCATABLE capacity —
and leaving it records the fence and makes the bytes evictable. Eviction FREES; there is no
host residence tier, and a re-stage is an ordinary cr-005 fill from the page-cached CAS
mapping. The `package-owns-no-residency` fence proves the vocabulary is unreachable from
package code rather than merely unused.

**The PlanChooser walks a closed ordered rung set against MEASURED free VRAM.** Every
coefficient declares its provenance (`uncalibrated | bound | measured | ledger`); claiming
`measured` without a source refuses. An uncalibrated coefficient may DEEPEN placement —
which changes weight placement only, never the bits — and may never refuse. Capacity splits
by tenancy: shared reschedules, local deepens to the deepest rung with a quantified
confession and exits only below the declared physical floor.

**The ledger** carries one row per byte class, each declaring its KIND (`arithmetic` for the
worker's own bookkeeping, `measured` for a named device or kernel interface,
`unreadable` for a class this worker genuinely cannot read — never a zero standing in for
one) and the METHOD that produced it. Every attempt closes by reconciling every class
against the line taken before its envelope was granted, and the record is recorded.

## The descriptor

`internal/discovery.py` is the isolation boundary: it imports exactly the ONE `App` export
package.toml names, freezes it, and stops. Importing a module never registers anything, so
there is no recursive discovery, no dependency stubbing and no source parser — a red arm
plants a self-registering sibling module and proves describe cannot see it.

`internal/descriptor.py` is the schema home for `PackageDescriptor`/`JobDescriptor`: only
the launch and invocation facts with current readers. It carries no Runtime-global service
inventory, diagnostics, config UI metadata, or candidate topology (cr-004's
ModelConstructionContract owns that), and `_fixed_point` proves over the built document that
no release digest, build timestamp, machine path or mutable binding reached it.
`internal/canonical.py` is the RFC 8785 writer — tensorfs (tfs-013) owns the rules, and
`scripts/verify.py` runs this codec over tensorfs's own vector corpus and reports which
refusals are shared and which are deliberate profile differences.

The descriptor is derived, not committed. `cozy-runtime --json --dir PROJECT describe` writes
the canonical descriptor bytes to stdout without modifying the project. Creator sends those
bytes with the source release; Tensorhub independently derives the descriptor from the installed
project wheel and compares it. `package_descriptor_digest` is SHA-256 over those canonical bytes.

Runtime also owns the descriptor's semantic-version comparison. Tensorhub invokes the narrow
process boundary below with two exact descriptors; it prints one compact JSON result whose
classification is `patch`, `minor`, or `major`. It imports no package code and authors no document.
Runtime classifies interface change only; Tensorhub retains package-version and v0 policy.

    python -m cozy_runtime.cli.descriptor_compat previous.json candidate.json

## Commands

    uv sync
    uv run cozy-runtime version          # tag, commit, wire + python semvers
    uv run cozy-runtime new <name>       # scaffold a minimal valid package
    uv run cozy-runtime describe              # human-readable surface summary
    uv run cozy-runtime --json describe       # canonical publication bytes on stdout
    uv run python checks/architecture.py # architecture fences
    uv run python scripts/redarm.py      # plant each violation, prove it red, restore
    uv run python checks/declared_facts.py # every public Context/service fact has a live writer
    uv run python scripts/verify.py      # live run of both example packages end to end
    uv run python scripts/bench.py       # time + memory benchmarks
    proofs/cli/run.sh # installed CLI exits, Descriptor/1, offline, keys and media proof
    proofs/environment/run.sh # installed environment cold/rebuild/reuse/refusal proof
    proofs/model-contract/run.sh # installed public-CLI MCC cold/rebuild/red proof
    uv run --extra media python scripts/media-live.py        # real FLAC/MP4 output codecs
    uv run --extra media python scripts/media-decode-live.py # bounded image/audio/video inputs
    uv run --extra media python scripts/media-stream-live.py # cr-029 2/8-shot RSS + kill cleanup

The model-contract proof builds a Runtime wheel, installs it into a fresh driver environment,
and invokes only that wheel's public commands from `/` with `PYTHONPATH` unset. Its package and
fake Torch packages live as ordinary, linted projects under `proofs/fixtures/`. They exercise the
weightless control plane only; this proof makes no claim about real Torch, GPU execution, model
loading, or output quality.

The environment proof likewise builds and installs the exact Runtime wheel before invoking its
public command from `/` with `PYTHONPATH` unset. Its base packages are installed once in the
driver, matching the base worker image. The command hashes and inspects only the project/custom
wheels, extracts them into private staging, precompiles that small overlay, and atomically
publishes it. Cold and independent rebuilds produce the same compact receipt.

### Installed base plus package overlay

`internal/overlay.py` is the serving materializer. The image already contains Python,
Torch/CUDA, cozy-runtime, TensorFS, and the registered base-family dependencies. A placement
therefore installs only its deterministic project wheel and optional exact custom wheels. It
never creates a venv, copies Torch, reads an index, resolves a dependency, or retains wheel
archives after extraction.

The image's OCI digest owns image-resident integrity. `WheelhouseManifest/1` is the complete
canonical ownership document for that image: its existing platform target and exact control-wheel
rows travel with every installed normalized distribution/version and every top-level import root's
sorted providers (`python-stdlib` or normalized distribution names). Every wheel fact records
sorted unique import roots: the image rows in `WheelhouseManifest/1`, the project row in
`Placement.package.project_wheel`, and each selected overlay row embedded in `PlacementSet/1`.
Runtime strictly parses those exact facts, rejects
a distribution collision with the image-owned version, measures every arriving wheel's real ZIP
roots, requires custom declarations to equal the measurement, and rejects a root collision with
the manifest's providers or a duplicate project/custom root before extraction or publication.
There is no version-compatibility reader and no hardcoded protected-root list.

Runtime hashes each arriving package wheel once. A worker-boot-local verified-file receipt binds
its digest/length to device, inode, size, mtime and ctime; stable reuse reads that receipt, while a
stat change or worker restart re-hashes before trust. Overlay admission consumes the receipt instead
of hashing the same wheel again. Each verified wheel is extracted once into a root-owned immutable
pool keyed below the base family and extraction format. Installed content is keyed by
`(base_family_digest, sorted wheel digests)` and composed with collision-checked hardlinks, so a
metadata-only environment-spec change shares one compiled content tree and a one-wheel change
shares every unchanged installed inode. The spec-specific receipt remains a separate exact join.

The rented-pod worker stays root and non-dumpable; the exec-first trampoline sets `no_new_privs`
and drops package execution to uid/gid 65533. Runtime keeps control state in memory because its
pod-supervisor owns the durable replay ledger. Receipts, acquisition cache, and worker credentials
remain root-only. Content is root-owned/read-only, and executor scratch/probe state is private to
one incarnation and deleted before the fixed uid is reused. A worker that cannot perform the
identity split refuses before binding. Native/local development keeps its explicit same-user lane.
Base bytecode is baked in the image and package bytecode is compiled once per content identity. The
executor starts with image Python and appends the exact content import root after image-owned paths.

Source-checkout development is an explicit, same-user lane rather than a partially qualified
published package. Creator privately asks Runtime to author a `Placement.development` carrying
package, semantic release, and source digest, then launches Runtime with the matching local path
and facts. No project wheel, release.json, Environment, receipt, or Qualification is invented.
For a modeled entrypoint, Runtime resolves package.toml through its existing binding owner and
joins the exact release/lane through the explicitly granted local TensorFS Repository. The
Placement carries the ordinary exact Manifest/config/component facts but no MCC; at acquisition
the launch Runtime re-reads the source descriptor and TensorFS Manifest/header/config, derives the
same binding locally, and compares every accepted fact before STAGED. Creator owns the source-file
digest and closes its scan race before replacing the whole worker generation after edits; Runtime
does not invent a second source hash. TensorFS logical assets and uploader/catalog stamps refuse
typed in this first lane because the local Repository exposes neither as a directly usable
selection authority.

A writable delegated cgroup-v2 child remains the preferred executor lifecycle boundary. Ordinary
RunPod containers instead expose their provider-owned cgroup-v2 scope read-only. Runtime uses its
existing process-tree boundary there only after positively proving the root-worker/dedicated-UID
split, root-owned container marker, read-only cgroup lineage and migration controls, finite inherited
`memory.max` and `pids.max`, full target-UID `/proc` visibility, and pidfds. Reclaim owns the entire
uid-65533 domain, not merely an environment token or process group, so token scrubbing, `setsid`,
double-fork, worker restart, and PID reuse do not leave an old executor alive. A writable but
undelegated cgroup-v2 host and an unbounded read-only container still refuse before package import.

An overlay contains exactly one project wheel and at most 127 custom wheels. The Runtime bound is
128 total package-owned wheels, counting the project; 128 custom rows plus the project refuses
before any grant or ZIP is opened.

The project wheel is exactly `py3-none-any`: Runtime checks its Tensorhub fact, filename tags,
`WHEEL` tags, `Root-Is-Purelib`, members, and native magic. A project `.so` or a platform-specific
project tag refuses. Every dependency wheel remains a separate exact prebuilt subject selected by
Tensorhub. Runtime accepts no build request and rejects C/C++/CUDA/Rust sources, build manifests,
headers/objects/static archives, and wheel install schemes that would inject scripts or data. It
never invokes pip, a resolver, PEP 517, or a compiler.

Ordinary native dependency and custom-extension wheels keep the semantic Python/ABI/platform,
`abi3`, marker, manylinux-floor, digest, ZIP, RECORD, and ownership checks. Before native bytes are
published, `internal/native_wheel.py` parses bounded ELF64 metadata and refuses a wrong machine,
hostile or escaping RPATH/RUNPATH, invalid `DT_NEEDED`, and any bundled library that shadows the
selected base's Torch, Triton, CUDA, cuBLAS, cuDNN, cuFFT, cuSPARSE, cuSolver, NCCL, or other NVIDIA
families. CUDA-bearing ELF requires bounded `cuobjdump` cubin/PTX inventory; unavailable tooling is
a refusal, not an omitted proof. The hardware qualification seat then joins that evidence to the
actual loader inventory and SM and runs the registered no-argument operator fixture in a separate
process. Missing libraries/symbols, wrong SM/PTX, `no kernel image`, timeout, oversized output, and
result-digest mismatch are named refusals. Tensorhub's networkless VM/cgroup remains the isolation
authority; Runtime additionally applies CPU/address-space/file/descriptor bounds and kills the
whole fixture process group.

PyTorch, Triton, and the CUDA driver continue to own JIT cache keys and byte validation, but those
keys are never treated as a same-UID security boundary. Runtime scopes cache reuse by unpredictable
worker boot plus exact overlay content, then imposes `TORCHINDUCTOR_CACHE_DIR`,
`TRITON_CACHE_DIR`, `TORCH_EXTENSIONS_DIR`, and `CUDA_CACHE_PATH` before executor import. A respawn
within that pod can retain its warm compiler cache; the whole pod scope is deleted after worker
shutdown has proven the executor tree gone. Tensorhub's public placement law supplies one public
trust domain per pod. Private renter cross-org stacking retains its explicitly documented same-UID
risk—Runtime does not relabel worker boot identity as tenant isolation. Cache bytes stay node-local,
never enter an overlay receipt/content identity, and are never published as release artifacts. Every
placement consumes this same overlay path; OCI provenance belongs to Tensorhub qualification
evidence, while the portable receipt remains the exact WheelhouseManifest/project/custom-wheel join.

On a remote placement, `PlacementSet/1` names one exact
`model_object_set` subject. Cozy sends signed logical package/model intent; the worker receives
exact digest/length/HTTPS locations directly from Tensorhub over mTLS. The non-control
convergence lane starts package and model acquisition together. As soon as package overlay and
binding documents are admitted from a standing start, Runtime spawns the executor so interpreter
and Torch import overlap continued TensorFS hydration; model preparation/fill still waits for the
exact model closure. READY remains the join of overlay, TensorFS closure, binding resolution and
device construction. An outgoing placement is never disturbed by this optimization. Same-revision
delegation renewal changes only access; verified complete objects survive and only missing objects
move after renewal.

`PlacementStatus.acquisition` exposes that work as observation only. It retains
package/model start and end values from one process-monotonic clock plus cumulative
downloaded/reused bytes. Partial states are emitted while either leg runs; terminal values
remain attached to that placement spec. These values are never read by admission,
materialization, convergence, fallback, or readiness code. Cold proof checks interval overlap;
warm proof checks zero downloaded package bytes and exact verified reuse, while the real
TensorFS cache proof checks non-empty model downloaded/reused accounting.

The CLI proof builds and installs the current wheel plus an ordinary model-free source package
fixture, changes to `/`, removes `PYTHONPATH`, closes stdin, and invokes only the installed
`cozy-runtime` command. It proves the current process exit matrix, read-only and exact-byte
Descriptor/1 behavior, canonical/local/strict key classification through a real job, typed
mixed-media grants and canonical invocation identity, and `--offline` execution while every
AF_INET/6 connect is refused. Its private companion
`proofs/internal/local_input_acceptance.py` owns transfer and RecordOwner fault plants; those
private hooks are not presented as CLI evidence. The full scenario census and retired claims
are recorded in `PROOF-SCENARIOS.md`.

`cozy-environment-proof REQUEST.json RECEIPT.json` is the trusted th-046 build-seat door.
Its closed local request names the direct PackageSelection, selected Environment and base
WheelhouseManifest, plus every overlay wheel file by digest,
the installed-base Python, a fresh overlay root, and a fresh receipt output. `wheel_files`
contains only the project and optional custom wheels; base wheel bytes do not travel. It
atomically publishes the overlay receipt bytes and prints their compact
digest/length/generation reference. It
does not accept an expected receipt, author release documents, fetch, resolve, or import package
code.

The request is ordinary JSON because it is a process input, not an identity document. Its field
set is closed; every path value is a normalized absolute local path:

```json
{
  "package": {
    "package": "org/package",
    "release": "1.0.0",
    "release_digest": "sha256:<64 lowercase hex>",
    "project_wheel": {"ref": {"digest": "sha256:<64 lowercase hex>", "length": 1234}, "distribution": "package", "filename": "package-1.0.0-py3-none-any.whl", "import_roots": ["package"], "tags": ["py3-none-any"], "version": "1.0.0"}
  },
  "environment": "/proof/environment.json",
  "wheelhouse_manifest": "/proof/wheelhouse-manifest.json",
  "wheel_files": {"sha256:<64 lowercase hex>": "/proof/objects/<digest>"},
  "python": "/opt/conda/bin/python",
  "environment_root": "/proof/fresh-environment"
}
```

An overlay containing native custom-wheel bytes must additionally pass
`cozy-native-wheel-proof REQUEST.json EVIDENCE.json` on every claimed hardware class. The request
does not carry a caller-asserted OCI realization identity: Runtime cannot measure an OCI manifest,
and echoing one is not proof. Tensorhub joins Runtime's measured evidence to provider OCI readback.

```json
{
  "device_index": 0,
  "environment_digest": "sha256:<64 lowercase hex>",
  "environment_root": "/proof/installed-environment",
  "installed_environment_receipt_digest": "sha256:<64 lowercase hex>",
  "wheelhouse_manifest": "/proof/wheelhouse-manifest.json",
  "python": "/opt/conda/bin/python",
  "fixture": "qualified_fixture:run",
  "expected_result_digest": "sha256:<64 lowercase hex>"
}
```

Runtime asks both its running proof interpreter and the selected `python` to report the actual
CPython ABI, OS/CPU, glibc, complete installed distribution/version set, Torch release and CUDA
build. It requires both reports to be identical and to satisfy the exact WheelhouseManifest using
minimum semantics for glibc, CUDA and driver versions. GPU name, driver and SM are measured through
`nvidia-smi`; cubin/PTX compatibility and the operator result are then proven on that device. The
command emits a bounded result DTO containing those measured facts. Tensorhub normalizes the
result into its own rows and alone authors the final two-field Qualification after all required
evidence is accepted.

The receipt-selected, content-keyed overlay then exposes
`cozy-model-contract-proof REQUEST.json OUTPUT_DIR`. This is the only public MCC writer:
it verifies the overlay/base-family join, compares the exact descriptor with the
installed project wheel, and double-derives every descriptor model binding under the
offline construction sandbox. The fresh output directory appears atomically and contains the
result in `result.json` plus exact digest-named files under `contracts/`. Tensorhub
stores those bytes and refs; it never authors or reserializes an MCC.

The request is a closed local process input. Every descriptor model path appears exactly
once; a function with no model is still recorded in the output index with `bindings: []`.
Config and descriptor input refs bind exact local bytes for proof ingestion. Their generic
`digest` is byte integrity and is never compared with semantic `package_descriptor_digest`:

```json
{
  "environment_generation": "/proof/environment/contents/<content-identity>",
  "environment_digest": "sha256:<64 lowercase hex>",
  "installed_environment_receipt": {"digest": "sha256:<64 lowercase hex>", "length": 1234},
  "wheelhouse_manifest": "/proof/wheelhouse-manifest.json",
  "package": {"package": "org/package", "release": "1.0.0", "release_digest": "sha256:<64 lowercase hex>", "project_wheel": {"ref": {"digest": "sha256:<64 lowercase hex>", "length": 1234}, "distribution": "package", "filename": "package-1.0.0-py3-none-any.whl", "import_roots": ["package"], "tags": ["py3-none-any"], "version": "1.0.0"}},
  "package_descriptor": {
    "path": "/proof/package.descriptor.json",
    "digest": "sha256:<64 lowercase hex>",
    "length": 5678
  },
  "bindings": [{
    "path": "generate.models.model",
    "snapshot": "sha256:<64 lowercase hex>",
    "config": {
      "path": "/proof/model-config.json",
      "digest": "sha256:<64 lowercase hex>",
      "length": 9012
    },
    "hardware_variants": ["sm89", "sm90"]
  }]
}
```

A package whose signature declares `MediaDecoder` must install `cozy-runtime[media]`;
`describe` refuses the release if PyAV is absent. The descriptor publishes the hard per-field
codec ceilings and the attempt-wide retained decoded-value ceiling.

Real-GPU drivers, run from a venv with the `corpus` extra (they need an 8 GiB card and the
real CAS, so they are BUILDER-RUN and never CI steps):

    corpus/.venv/bin/python scripts/worker-live.py attempt   # cr-007: the two-process worker
    corpus/.venv/bin/python scripts/slice-live.py all        # cr-008a: one request, no network
    corpus/.venv/bin/python scripts/slice-live.py triage     # cr-011: explain a killed executor from the bundle
    corpus/.venv/bin/python scripts/slice-live.py wedge      # cr-011: silence vs slow, with no clock
    corpus/.venv/bin/python scripts/slice-live.py witness    # cr-011: a real cgroup OOM kill, named
    corpus/.venv/bin/python scripts/pipeline-live.py fixtures  # cr-008b: the config + bundled tokenizers
    corpus/.venv/bin/python scripts/pipeline-live.py stage     # cr-008b: four components, one card
    corpus/.venv/bin/python scripts/pipeline-live.py image     # cr-008b: cold disk -> first image
    corpus/.venv/bin/python scripts/pipeline-live.py arms      # cr-008b: access + physical-floor refusals
    corpus/.venv/bin/python scripts/pipeline-live.py bandwidth # cr-008b: the disk-bandwidth ladder
    corpus/.venv/bin/python scripts/job-live.py all          # cr-009: the @job surface, services and runner
    corpus/.venv/bin/python scripts/quant-live.py arms       # cr-006: the encoding border and its refusals
    corpus/.venv/bin/python scripts/delivery-live.py all     # cr-008c: the delivery ladder, chosen over two real artifacts
    corpus/.venv/bin/python scripts/assets-live.py all       # cr-012: typed input assets, hydration, egress
    corpus/.venv/bin/python scripts/leaf-live.py all         # cr-006: the native-encoded rung — leaf replacement and the ranked chooser

Model-bound `cozy-runtime run` uses the same explicit source-checkout lane: package.toml (or its
`--model` override) selects an exact ref, the local TensorFS Repository resolves its release/lane,
and the normal worker/attempt kernel applies the host's real GPU qualification. It never authors a
flat binding record or pretends the local selection is a Hub Qualification. The focused
`proofs/internal/modeled_development_acceptance.py` proof builds a tiny local release and checks
selection/acquisition plus drift refusals without claiming GPU execution; the existing
`scripts/pipeline-live.py image/arms` runs remain the real-GPU output and floor evidence.

`scripts/leaf-pod.py` is the same rung on REAL H3 weights and is POD-SIDE ONLY: it fetches
735 MiB of a 40.2 GB checkpoint by HTTP range, so it never runs on a dev box.

**TensorFS.** The fill plane reads through tfs-007's `tensorfs` facade — a compiled
extension that is not on an index yet, so the GPU venvs install a locally built wheel:

    cd ../tensorfs && uvx maturin build --release \
        --manifest-path crates/tensorfs-py/Cargo.toml --out /tmp/cozy-wheels
    uv pip install --python corpus/.venv/bin/python /tmp/cozy-wheels/tensorfs-*.whl

When TensorFS publishes, this becomes an ordinary pinned dependency and the step goes away.
The base package installs and type-checks without it, exactly as it does without torch; a
fill without the reader is a typed `reader_unavailable` refusal naming the install, never an
`ImportError` escaping a request.

Two real packages on the author surface (weightless, no GPU), both driven end to end by
`scripts/verify.py`:

| example | what it exercises |
|---|---|
| `proofs/fixtures/tiny-models/tiny_sdxl.py` | the MINIMUM: `loader.load(T)` sugar, one coarse `@uses_components` set, four entrypoints covering both spellings, the derived capability set, `ModelDefault` overlay, typed preflight, the output transaction |
| `proofs/fixtures/tiny-models/tiny_h3.py` | the component-staged path: one `_H3Base` with two task-stamped role classes, role-scoped factories, granular component scopes, an ordered discriminated reference union, a declared `DerivedCache`, and the `@app.job` that exercises JobDescriptor |

`proofs/fixtures/tiny-models/tiny_diffusers.py` stands in for the model library a package brings as
an ordinary dependency (and registers its own construction adapter);
`proofs/internal/tiny_artifacts.py` and
`proofs/fixtures/tiny-models/package.toml` are the RUNTIME's side — resolved artifacts and default
bindings. Author code never names an artifact identifier; a fence proves it.

`run` · `serve` · `job` · `ingest` · `doctor` · `fit` land in
cr-016 over the real attempt kernel.

**Jobs (cr-009).** A job is an attempt class on the same machinery, not a second runtime:
`@app.job` over the one invocation kernel, the same worker/executor split, the same
records, the same output transaction. What differs is stated out loud and is short — a
`JobDirective`/`JobExecutionSpec` instead of the serving pair, BOUNDED ACCEPT-THEN-PREPARE
(serving never accepts-then-cold-loads; a job has nothing to be warm for), the three job
services (`Scratch`, `Checkpoints`, `Budget`), typed input `Tree` fields whose path rides
the value, and a RECYCLE process exit distinct from a death. `corpus/job/` is the live
project: a structural census over the real 6.9 GB cozytensors store, read-only.
