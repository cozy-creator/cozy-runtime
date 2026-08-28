# cozy-runtime (v2)

The worker runtime: author surface (`cozy_runtime.author`), worker, disposable device
executor, and the single-endpoint CLI. Design authority: [tracker-v2](https://github.com/cozy-creator/tracker-v2)
`cozy-runtime.md` + `cozy-runtime-cli.md`; issues in `tracker/cozy-runtime/`.

**No behavior test suite** (decisions.md #160, #533): CI runs build, fences, and
deterministic contract/arithmetic/staleness checks; everything else is verified by running
the real thing and recording what it did, plus time/memory benchmarks.

## Layout

| package | role | may import |
|---|---|---|
| `cozy_runtime.author` | the whole author surface — the only package endpoint code imports | stdlib + msgspec |
| `cozy_runtime.internal` | runtime implementation; owns the ONE config authority | author, protocol, stdlib |
| `cozy_runtime.protocol` | `cozy.worker.v1` (schema rev 9) — worker-protocol's bindings, COPIED VERBATIM into `src/cozy/`, plus the document plane and the schema fence | protobuf, internal.canonical |
| `cozy_runtime.cli` | local RecordOwner adapter over the core | author, internal |

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

| module | role |
|---|---|
| `internal/worker/session.py` | the protocol client and the worker entrypoint |
| `internal/worker/control.py` | the worker-owned control transport: gRPC and in-memory hosts |
| `internal/worker/attempts.py` | offer -> prepare -> accept -> terminal, attestation, ledger close |
| `internal/worker/ledger.py` | the worker-scoped byte ledger and the attempt reconciliation |
| `internal/worker/plan.py` | the AttemptPlan, the PlanChooser, the local binding-plan record |
| `internal/worker/grants.py` | DeliveryGrant reads/writes + the closed hub-call table |
| `internal/worker/child.py` | spawn/supervise/kill the executor; the two ordinals |
| `internal/residency.py` | component-use admission: materialize a declared set before entry, event-fence, make evictable |
| `internal/worker/observe.py` | typed boot steps, confessions, and the SILENCE-BOUNDED liveness detector |
| `internal/worker/triage.py` | the `WorkerTriageBundle` schema, its assembler, and the journal-backed store |
| `internal/worker/refusal.py` | reader-vs-data fault, verbatim upstream remedy, the REF_ORIGIN severity split |
| `internal/witness.py` | the out-of-process postmortem witness: it names the WORKER's own death |
| `internal/local.py` | the ephemeral local worker: one request, no network |
| `internal/journal.py` | the durable, self-verifying attempt/terminal journal |
| `internal/seam.py` | the control-only, byte-accounted worker<->executor channel |
| `internal/trampoline.py` | the exec-first stage: pdeathsig, no_new_privs, oom_score_adj |
| `internal/executor.py` | the disposable device process — the ONE torch importer |
| `internal/fill.py` | the fill plane: a TensorFS `Checkpoint`, the pinned ring, the transaction |

The worker never imports torch, and that is proved against the RUNNING process
(`scripts/worker-live.py nocuda` reads its `/proc/<pid>/maps` and the device's compute-app
list), not against the import graph.

**One attempt kernel** (§8). A one-shot local run is not a second execution path: it is an
ephemeral local worker whose RecordOwner is a pair of in-process queues, over the same
process boundary, journal, acceptance boundary, freeze point and ledger. The rule is
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

**Every terminal attempt commits one `WorkerTriageBundle`** — identity, applied posture,
ledger reconciliation, attribution, confessions, faults, and the capped observation tail —
written and fsynced BEFORE its write receipt is journaled and before the terminal that names
it. The handle is an opaque `subject_id`; there is no path in it and no way to name one. A
killed executor is therefore explainable from a directory, with no live process and no SSH.

**Every granted destination is CONTROL-PLANE-MINTED** (cr-042). The closed `HUB_CALLS` table
in `internal/worker/grants.py` spends a grant minted for one invocation, so nothing the
endpoint produced — and nothing a request body can influence — chooses where bytes land; a
destination with no row is a SECURITY EVENT. Every row is `file://` today.
`OutputAccess.url`'s "presigned upload URL" is the PLANNED public route — tensorhub mints
it, R2 sits behind it, the worker never holds the credential — and lands WITH host
validation of the minted URL: that, not a filesystem-only table, is what keeps the property.
Armed by `scripts/worker-live.py grants` end to end and `scripts/assets-live.py egress` at
the table.

**The executor's network claim is Python-deep and says so.** `internal/executor.py:main`
installs `sandbox.refuse_network`, so endpoint or job code that dials is refused typed at
the first `NETWORK_EVENTS` spelling. That bounds a BUGGY endpoint (law 15's same-tenant
rung). It is not a kernel bound: the executor runs in the worker's own network namespace
with no seccomp filter, and a C extension or `ctypes` reaches `connect(2)` underneath the
hook. `scripts/lifecycle-live.py --only egress` observes both halves against one reachable
address, so the claim cannot quietly grow. Granted inputs are fetched by the WORKER and
handed over as spool paths; the executor is never given an address.

**Liveness is silence-bounded, never clocked.** Progress means an advancing monotone position;
a wedge is consecutive observations that find the position unchanged. `give_up_after` counts
observations, not seconds — there is no boot timeout and there will not be one — and the
`timing-decides-nothing` fence proves the observation plane cannot reach residency or placement.

**Component residency is the runtime's, and the endpoint has no vocabulary for it.** An
endpoint declares the possible heavyweight set of each public model operation with
`@uses_components` and nothing else. Entering that scope ADMITS and MATERIALIZES the set
before the body runs — freeing non-declared components, LRU over refcount-0 leases whose
CUDA events have fired, until the plan's activation envelope fits ALLOCATABLE capacity —
and leaving it records the fence and makes the bytes evictable. Eviction FREES; there is no
host residence tier, and a re-stage is an ordinary cr-005 fill from the page-cached CAS
mapping. The `endpoint-owns-no-residency` fence proves the vocabulary is unreachable from
endpoint code rather than merely unused.

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
against the line taken before its envelope was granted, and the record is journaled.

## The descriptor

`internal/discovery.py` is the isolation boundary: it imports exactly the ONE `App` export
endpoint.toml names, freezes it, and stops. Importing a module never registers anything, so
there is no recursive discovery, no dependency stubbing and no source parser — a red arm
plants a self-registering sibling module and proves describe cannot see it.

`internal/descriptor.py` is the schema home for `EndpointDescriptor`/`JobDescriptor`: only
the launch and invocation facts with current readers. It carries no Runtime-global service
inventory, diagnostics, config UI metadata, or candidate topology (cr-004's
ModelConstructionContract owns that), and `_fixed_point` proves over the built document that
no release digest, build timestamp, machine path or mutable binding reached it.
`internal/canonical.py` is the RFC 8785 writer — tensorfs (tfs-013) owns the rules, and
`scripts/verify.py` runs this codec over tensorfs's own vector corpus and reports which
refusals are shared and which are deliberate profile differences.

`endpoint.descriptor.json` is generated, committed and reviewed. The committed file is a
deterministic pretty projection; `descriptor_digest` is SHA-256 over canonical JSON content,
so whitespace and key order never affect identity. `describe --check` compares meaning and
refuses a stale surface while admitting a harmless reformat. `describe` itself is read-only;
`--write-descriptor` is the normal projection writer, and
`cozy-runtime new` runs it so a fresh endpoint is `--check`-clean out of the box.

## Commands

    uv sync
    uv run cozy-runtime version          # tag, commit, wire + python semvers
    uv run cozy-runtime new <name>       # scaffold a minimal valid endpoint
    uv run cozy-runtime describe         # the surface, without loading weights
    uv run cozy-runtime describe --check # refuse a stale endpoint.descriptor.json (CI)
    uv run python checks/architecture.py # architecture fences
    uv run python scripts/redarm.py      # plant each violation, prove it red, restore
    uv run python checks/declared_facts.py # every public Context/service fact has a live writer
    uv run python checks/wire_schema.py  # vendored worker-schema identity
    uv run python scripts/verify.py      # live run of both example endpoints end to end
    uv run python scripts/bench.py       # time + memory benchmarks
    proofs/cli/run.sh # installed CLI exits, Descriptor/1, offline, keys and media proof
    proofs/environment/run.sh # installed environment cold/rebuild/reuse/refusal proof
    proofs/model-contract/run.sh # installed public-CLI MCC cold/rebuild/red proof
    uv run --extra media python scripts/media-live.py        # real FLAC/MP4 output codecs
    uv run --extra media python scripts/media-decode-live.py # bounded image/audio/video inputs
    uv run --extra media python scripts/media-stream-live.py # cr-029 2/8-shot RSS + kill cleanup

The model-contract proof builds a Runtime wheel, installs it into a fresh driver environment,
and invokes only that wheel's public commands from `/` with `PYTHONPATH` unset. Its endpoint and
fake Torch packages live as ordinary, linted projects under `proofs/fixtures/`. They exercise the
weightless control plane only; this proof makes no claim about real Torch, GPU execution, model
loading, or output quality.

The environment proof likewise builds and installs the exact Runtime wheel before invoking its
public command from `/` with `PYTHONPATH` unset. Its base packages are installed once in the
driver, matching the base worker image. The command hashes and inspects only the project/custom
wheels, extracts them into private staging, precompiles that small overlay, and atomically
publishes it. Cold and independent rebuilds produce the same compact receipt.

### Installed base plus endpoint overlay

`internal/overlay.py` is the serving materializer. The image already contains Python,
Torch/CUDA, cozy-runtime, TensorFS, and the registered base-family dependencies. A placement
therefore installs only its deterministic project wheel and optional exact custom wheels. It
never creates a venv, copies Torch, reads an index, resolves a dependency, or retains wheel
archives after extraction.

The image's OCI digest owns image-resident integrity. `WheelhouseManifest/3` is the complete
canonical ownership document for that image: its existing platform target and exact control-wheel
rows travel with every installed normalized distribution/version and every top-level import root's
sorted providers (`python-stdlib` or normalized distribution names). Every wheel fact records
sorted unique import roots: the five control rows in `WheelhouseManifest/3`, the project row in
`EndpointBundle/2`, and each custom row in `ResolvedWheelSet/3`. Runtime strictly parses those exact bytes, rejects
a distribution collision with the image-owned version, measures every arriving wheel's real ZIP
roots, requires custom declarations to equal the measurement, and rejects a root collision with
the manifest's providers or a duplicate project/custom root before extraction or publication.
There is no `/1` reader and no
hardcoded protected-root list.

Runtime hashes each arriving endpoint wheel once. A worker-boot-local verified-file receipt binds
its digest/length to device, inode, size, mtime and ctime; stable reuse reads that receipt, while a
stat change or worker restart re-hashes before trust. Overlay admission consumes the receipt instead
of hashing the same wheel again. Each verified wheel is extracted once into a root-owned immutable
pool keyed below the base family and extraction format. Installed content is keyed by
`(base_family_digest, sorted wheel digests)` and composed with collision-checked hardlinks, so a
metadata-only environment-spec change shares one compiled content tree and a one-wheel change
shares every unchanged installed inode. The spec-specific receipt remains a separate exact join.

The rented-pod worker stays root and non-dumpable; the exec-first trampoline sets `no_new_privs`
and drops endpoint execution to uid/gid 65533. Receipts, acquisition cache, journal and worker
credentials remain root-only. Content is root-owned/read-only, and executor scratch/probe state is
private to one incarnation and deleted before the fixed uid is reused. A worker that cannot perform
the identity split refuses before binding. Native/local development keeps its explicit same-user
lane. Base bytecode is baked in the image and endpoint bytecode is compiled once per content
identity. The executor starts with image Python and appends the exact content import root after
image-owned paths.

An overlay contains exactly one project wheel and at most 127 custom wheels. The Runtime bound is
128 total endpoint-owned wheels, counting the project; 128 custom rows plus the project refuses
before any grant or ZIP is opened.

On a remote placement, `PlacementSpec/2` names one exact
`model_object_set` subject. The standing grant supplies access only. The non-control
convergence lane starts endpoint and model acquisition together. As soon as endpoint overlay and
binding documents are admitted from a standing start, Runtime spawns the executor so interpreter
and Torch import overlap continued TensorFS hydration; model preparation/fill still waits for the
exact model closure. READY remains the join of overlay, TensorFS closure, binding resolution and
device construction. An outgoing placement is never disturbed by this optimization, and a grant
refresh re-drives the same accepted revision without changing its identity.

Schema rev 7 exposes that work as observation only. `PlacementStatus.acquisition` retains
endpoint/model start and end values from one process-monotonic clock plus cumulative
downloaded/reused bytes. Partial states are emitted while either leg runs; terminal values
remain attached to that placement spec. These values are never read by admission,
materialization, convergence, fallback, or readiness code. Cold proof checks interval overlap;
warm proof checks zero downloaded endpoint bytes and exact verified reuse, while the real
TensorFS cache proof checks non-empty model downloaded/reused accounting.

The CLI proof builds and installs the current wheel plus an ordinary weightless endpoint
fixture, changes to `/`, removes `PYTHONPATH`, closes stdin, and invokes only the installed
`cozy-runtime` command. It proves the current process exit matrix, read-only and exact-byte
Descriptor/1 behavior, canonical/local/strict key classification through a real job, typed
mixed-media grants and canonical invocation identity, and `--offline` execution while every
AF_INET/6 connect is refused. Its private companion
`proofs/internal/local_input_acceptance.py` owns transfer and RecordOwner fault plants; those
private hooks are not presented as CLI evidence. The full scenario census and retired claims
are recorded in `PROOF-SCENARIOS.md`.

`cozy-environment-proof REQUEST.json RECEIPT.json` is the trusted th-046 build-seat door.
Its closed local request names the four exact release documents, every wheel file by digest,
the installed-base Python, a fresh overlay root, and a fresh receipt output. `wheel_files`
contains only the project and optional custom wheels; base wheel bytes do not travel. It
atomically publishes the overlay receipt bytes and prints their compact
digest/length/generation reference. It
does not accept an expected receipt, author release documents, fetch, resolve, or import endpoint
code.

The request is ordinary JSON because it is a process input, not an identity document. Its field
set is closed; every path value is a normalized absolute local path:

```json
{
  "format": "cozy.runtime.EnvironmentProofRequest/1",
  "endpoint_environment_spec": "/proof/endpoint-environment-spec.json",
  "endpoint_bundle": "/proof/endpoint-bundle.json",
  "resolved_wheel_set": "/proof/resolved-wheel-set.json",
  "wheelhouse_manifest": "/proof/wheelhouse-manifest.json",
  "wheel_files": {"sha256:<64 lowercase hex>": "/proof/objects/<digest>"},
  "python": "/opt/conda/bin/python",
  "environment_root": "/proof/fresh-environment"
}
```

The receipt-selected, content-keyed overlay then exposes
`cozy-model-contract-proof REQUEST.json OUTPUT_DIR`. This is the only public MCC writer:
it verifies the overlay/base-family join, compares the exact descriptor with the
installed project wheel, and double-derives every descriptor model binding under the
offline construction sandbox. The fresh output directory appears atomically and contains
`model-construction-proof.json` plus exact digest-named files under `contracts/`. Tensorhub
stores those bytes and refs; it never authors or reserializes an MCC.

The request is a closed local process input. Every descriptor model path appears exactly
once; a function with no model is still recorded in the output index with `bindings: []`.
Config and descriptor input refs bind exact local bytes for proof ingestion. Their generic
`digest` is byte integrity and is never compared with semantic `descriptor_digest`:

```json
{
  "format": "cozy.runtime.ModelConstructionProofRequest/1",
  "environment_generation": "/proof/environment/contents/<content-identity>",
  "environment_spec_digest": "sha256:<64 lowercase hex>",
  "installed_environment_receipt": {"digest": "sha256:<64 lowercase hex>", "length": 1234},
  "project_wheel_digest": "sha256:<64 lowercase hex>",
  "wheelhouse_manifest": "/proof/wheelhouse-manifest.json",
  "endpoint_release_id": "endpoint-release-id",
  "endpoint_descriptor": {
    "path": "/proof/endpoint.descriptor.json",
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

An endpoint whose signature declares `MediaDecoder` must install `cozy-runtime[media]`;
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

There is deliberately no model-bound `cozy-runtime run` GPU qualification. Flat local model
bindings were removed; that command now exits 6 and requires Tensorhub's selected
Plan/RMB/MCC provision closure. `scripts/pipeline-live.py image/arms` retain the real SDXL
output and floor evidence, while `qualification/gpu/README.md` records why the obsolete
CLI-versus-private comparison was retired instead of rebuilding deleted authority.

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

Two real endpoints on the author surface (weightless, no GPU), both driven end to end by
`scripts/verify.py`:

| example | what it exercises |
|---|---|
| `examples/tiny_sdxl.py` | the MINIMUM: `loader.load(T)` sugar, one coarse `@uses_components` set, four entrypoints covering both spellings, the derived capability set, `ModelDefault` overlay, typed preflight, the output transaction |
| `examples/tiny_h3.py` | the component-staged path: one `_H3Base` with two task-stamped role classes, role-scoped factories, granular component scopes, an ordered discriminated reference union, a declared `DerivedCache`, and the `@app.job` that exercises JobDescriptor |

`examples/tiny_diffusers.py` stands in for the model library an endpoint brings as an
ordinary dependency (and registers its own construction adapter); `examples/tiny_artifacts.py`
and `examples/endpoint.toml` are the RUNTIME's side — resolved artifacts and default
bindings. Author code never names an artifact identifier; a fence proves it.

`run` · `serve` · `job` · `pull` · `ingest` · `list` · `rm` · `doctor` · `fit` land in
cr-016 over the real attempt kernel.

**Jobs (cr-009).** A job is an attempt class on the same machinery, not a second runtime:
`@app.job` over the one invocation kernel, the same worker/executor split, the same
journal, the same output transaction. What differs is stated out loud and is short — a
`JobDirective`/`JobExecutionSpec` instead of the serving pair, BOUNDED ACCEPT-THEN-PREPARE
(serving never accepts-then-cold-loads; a job has nothing to be warm for), the three job
services (`Scratch`, `Checkpoints`, `Budget`), typed input `Tree` fields whose path rides
the value, and a RECYCLE process exit distinct from a death. `corpus/job/` is the live
project: a structural census over the real 6.9 GB cozytensors store, read-only.
