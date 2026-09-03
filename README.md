# cozy-runtime (v2)

The worker runtime: author surface (`cozy_runtime.author`), worker, disposable device
executor, and the single-package CLI. Design authority: [tracker-v2](https://github.com/cozy-creator/tracker-v2)
`cozy-runtime.md` + `cozy-runtime-cli.md`; issues in `tracker/cozy-runtime/`.

## Compatibility status

Cozy Runtime is the platform's declared Python author-API channel. Package code has one supported
production import root, `cozy_runtime.author`; package tests may also use
`cozy_runtime.author.fakes`. Underscored author modules, `internal`, `cli`, and `protocol` are not
package APIs. Runtime remains pre-release: `0.0.x` may make a coordinated hardcut rather than carry
a compatibility alias for an unused or dishonest surface. Published packages declare the supported
Runtime range and must be requalified when a hardcut removes something they used.

Runtime release `0.2.5` targets standard CPython 3.12 only. Project
tooling pins `3.12`; no other Python interpreter or ABI is supported.
Wire minor 18 allows an authenticated external RecordOwner to make the existing bounded
WeightsSink reads through pod-supervisor; Runtime still returns the exact range on the pinned
WorkerControl stream without a second byte-plane listener.

The distribution major is the Python compatibility major: compatible additions remain in the
current range, breaking author changes require a new major, and candidate base worker images must
replay the supported-range package corpus before activation. Tensorhub images still pin one exact
Runtime wheel digest; package admission uses the declared range and selects the newest compatible
active Runtime/profile. The protobuf package major and additive wire minor remain their own
worker-control compatibility contract.

**One end-to-end test suite, no mocks.** `uv run pytest` runs against
`examples/marco-polo` through the public CLI installed from a freshly built wheel into a venv
that has nothing else: describe, an exact byte-identical package interface replay, the exit-code
matrix, one served request, and one job through the real two-process worker. Nothing is
stubbed, patched or doubled — a test that passes because a fake agreed with it is worse than
no test. The two legs that SPAWN AN EXECUTOR skip where the host cannot contain one (a
runner whose own cgroup is not delegated) and name that host fact as the reason. Beside them
CI runs the build, ruff, mypy, the reproducible-wheel check and `checks/architecture.py`'s
AST import scans, and nothing else. Hardware is verified by hand on a rented card.

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
| `internal/witness.py` | the out-of-process postmortem witness: it names the WORKER's own death (DECLARED, not yet wired — the pod launcher does not start it) |
| `internal/local.py` | the ephemeral local worker: one request, no network |
| `internal/worker/records.py` | bounded same-process worker facts; callers own restart recovery |
| `internal/seam.py` | the control-only, byte-accounted worker<->executor channel |
| `internal/trampoline.py` | the exec-first stage: pdeathsig, no_new_privs, oom_score_adj |
| `internal/executor.py` | the disposable device process — the ONE torch importer |
| `internal/fill.py` | the fill plane: a TensorFS `Checkpoint`, the pinned ring, the transaction |

`checks/architecture.py` keeps versioned identities in two disjoint registries.
`DOCUMENT_KINDS` contains only stored or digested documents that cross a process/repository
boundary: PackageInterface, EnvironmentReceipt, and PackageEnvironment.
`IMPLEMENTATION_IDS` contains the six encoding-provider names whose exact strings enter
implementation, plan, and generation digests; those names are executable identity, not document
formats. Command requests, planner values, worker records, proof-local state, and benchmark reports
are typed by their command/path or storage table and carry no global format tag.

The worker never imports torch, and that is proved against the RUNNING process
(the worker process's `/proc/<pid>/maps` and the device's compute-app
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
the silence detector reads. Credential-shaped VALUES — signed URLs, bearer and key
spellings — are redacted at the emit boundary and counted. Nothing is refused: observability
may not fail the thing it observes, and field NAMES are not inspected at all, because the
name regex that used to be here false-positived on `output_tokens` and `eos_token`.

Progress names the package's execution stage. `tel.progress(stage_fraction, stage=...,
overall_fraction=...)` keeps resettable within-stage progress distinct from optional monotonic
whole-invocation progress. `tel.step_callback(total, stage=..., overall_range=(start, end))`
adds exact work-unit coordinates and maps counted steps into a package-owned overall range;
`tel.stage(name, overall_range=...)` reports only entry/success boundaries, never an invented
within-stage percentage. Invalid progress is omitted and diagnosed on the same bounded lossy
lane; it cannot route, settle, bill, or otherwise control an attempt.

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
Armed by the worker grant path end to end and the egress hook at
the table.

**The executor's network claim is Python-deep and says so.** `internal/executor.py:main`
installs `sandbox.refuse_network`, so package or job code that dials is refused typed at
the first `NETWORK_EVENTS` spelling. That bounds a BUGGY package (law 15's same-tenant
rung). It is not a kernel bound: the executor runs in the worker's own network namespace
with no seccomp filter, and a C extension or `ctypes` reaches `connect(2)` underneath the
hook. The egress arms observe both halves against one reachable
address, so the claim cannot quietly grow. Granted inputs are fetched by the WORKER and
handed over as spool paths; the executor is never given an address.

**Progress is observed, never promoted into a timer-driven failure.** Advancing monotone
positions and measured process work are reported; an unchanged sample stays `silent`. Process
exit, typed outcomes, explicit cancellation and enclosing caller deadlines decide lifecycle —
and so does a PROVEN wedge, which `internal/liveness.py` defines once for a stream, a child
process and a local boot: still for longer than the subject's own longest observed pause, and
never for less than a floor derived from the sampling cadence. No number of seconds kills
anything on its own. The `timing-decides-nothing` fence proves observation cannot reach
residency or placement.

**Component residency is the runtime's, and the package has no vocabulary for it.** An
package declares the possible heavyweight set of each public model operation with
`@uses_components` and nothing else. Entering that scope ADMITS and MATERIALIZES the set
before the body runs — freeing non-declared components, LRU over refcount-0 leases whose
CUDA events have fired, until the plan's activation envelope fits ALLOCATABLE capacity —
and leaving it records the fence and makes the bytes evictable. Eviction FREES; there is no
host residence tier, and a re-stage is an ordinary cr-005 fill from the page-cached CAS
mapping. The `package-owns-no-residency` fence proves the vocabulary is unreachable from
package code rather than merely unused.

**Anima's CUDA optimization is an exact post-fill plan.** Runtime recognizes only
`anima:AnimaModel` with Diffusers `0.40.0` and the exact Anima Base 1.0 Cosmos config. After
TensorFS has filled the ordinary module tree, it installs fused RMSNorm + split-half RoPE,
direct split-half RoPE matrix production, `addcmul` residuals, native RMSNorm and a
latent-sized zero padding mask. The installation
compares the state-dict tensor schema and every parameter/buffer storage address before and after;
its source/dependency digest enters constructed-generation identity. Any source, class or
config drift refuses typed.

Runtime is ONE PyPI package with two wheel files per release: the pure `py3-none-any`
wheel, and a `cp312-abi3` linux/x86 wheel carrying the same code plus the compiled CUDA
kernels at `cozy_runtime._kernels` (`COZY_RUNTIME_BUILD_KERNELS=1` selects that build;
the default `uv build` stays toolchain-free). The `native/` sources ship only the fused
operations Runtime actually uses, and adding future Cozy-owned kernels extends this one
reviewed boundary. Runtime uses these fused Anima CUDA kernels only for Torch 2.13.0
built with CUDA 13.0 when CUDA is available. CPU, missing-CUDA, other Torch/CUDA pairs,
and wheels without the native extension keep the ordinary eager implementation. A
candidate CUDA image runs a hand-run comparison of eager and optimized real CUDA blocks
at Anima's native token width, checks determinism and storage identity, and measures the
block speedup.

`scripts/build-kernel-wheel.sh` is that second wheel's ONE recipe, run by `publish.yaml`
and by hand alike. It builds inside `manylinux_2_28_x86_64` with CUDA 13.0 and lets
auditwheel certify the platform tag, so nvcc never has to exist on the host and the
release cannot ship a tag the binary does not honour. A wheel's glibc floor is the floor
of the machine that compiled it, which is why the build is manylinux rather than the GPU
worker image's Ubuntu 24.04 base (that toolchain yields an unpublishable
`manylinux_2_39`); the parts that must match the worker image -- nvcc 13.0.88, cp312,
static cudart -- do. A developer box wanting the fused kernels installs that exact file,
never a side channel:

```
scripts/build-kernel-wheel.sh --out dist
uv tool install --force --python 3.12 \
  "cozy-runtime[media,model-execution] @ $PWD/dist/cozy_runtime-<version>-cp312-abi3-manylinux_2_28_x86_64.whl"
```

Every model-specific optimization first benchmarks `torch.compile`/Inductor as a discovery
oracle and records profile identity, graph breaks, generated kernels, compile cost, warm
latency, peak VRAM and output equivalence. Reusable fusion becomes a maintained in-package
kernel; otherwise qualification may produce one exact AOT artifact. Runtime never compiles
on each invocation, and eager execution remains the fallback.

**The PlanChooser walks a closed ordered rung set against MEASURED free VRAM.** Every
coefficient declares its provenance (`uncalibrated | bound | measured | ledger`); claiming
`measured` without a source refuses. An uncalibrated coefficient may DEEPEN placement —
which changes weight placement only, never the bits — and may never refuse. A shortfall
deepens to the deepest rung with a quantified confession rather than refusing.

**The ledger** carries one row per byte class, each declaring its KIND (`arithmetic` for the
worker's own bookkeeping, `measured` for a named device or kernel interface,
`unreadable` for a class this worker genuinely cannot read — never a zero standing in for
one) and the METHOD that produced it. Every attempt closes by reconciling every class
against the line taken before its envelope was granted, and the record is recorded.

## The package interface

`internal/discovery.py` is the isolation boundary: it imports exactly the ONE `App` export
package.toml names, freezes it, and stops. Importing a module never registers anything, so
there is no recursive discovery, no dependency stubbing and no source parser — a red arm
plants a self-registering sibling module and proves describe cannot see it.

`internal/package_interface.py` is the schema home for `PackageInterface`/`JobDescriptor`: only
the launch and invocation facts with current readers. It carries no Runtime-global service
inventory, diagnostics, config UI metadata, or tensor requirements, and
`_fixed_point` proves over the built document that no release digest, build timestamp,
machine path or mutable binding reached it. A slot is `{path, class, component_use,
encoded_leaves}`. Deleted members are simply unknown fields; there is no legacy decoder or
empty-value compatibility shim.
Packages declare no GPU count, VRAM floor, accelerator class, placement, or free-form hardware
predicate. Those facts come from the selected model lane and its concrete tensor schema/encoding,
RecordOwner placement, qualification measurements, and the worker's observed machine. The
interface reader rejects the retired `resources` key as an unknown field; format `/1` has one
current grammar and no compatibility decoder.
An ordinary `@app.job` may opt into source-first model production without introducing another
workflow language. A slot's entire declaration is its class annotation (cr-077); selection defaults live
in `package.toml [bindings."<callable>.models.<param>"]` (the slot path is the one binding
grammar; the class-keyed spelling refuses `binding_class_key_retired`), and the actual
binding is dispatch. The retired `source_profiles=` decorator field refuses with the typed
code `source_profiles_retired`. Each `WeightsOutput` declares only its name and new-byte ceiling;
the produced CozyTensors header is the sole tensor-schema and encoding authority. The
executor still receives only verified model capabilities, never the foreign source credential or
destination authority. Multiple transformations are ordinary code inside one bounded job—there is
no `model_productions` graph, compatibility reader, or second scheduler.
`internal/canonical.py` is the RFC 8785 writer — tensorfs (tfs-013) owns the rules, and
`checks/architecture.py`'s `canonical-numbers` fence replays it over the frozen ES6 vector
corpus in `vectors/es6-numbers.txt`.

The package interface is derived, not committed to the source tree. `cozy-runtime --json --dir
PROJECT describe` writes canonical bytes to stdout. Creator places those exact bytes at
`metadata/package-interface.json` in the ordinary uploaded package file tree; Tensorhub derives
the same interface from the installed project wheel and compares it. `package_interface_digest`
is SHA-256 over those canonical bytes.

## Commands

    uv sync
    uv run cozy-runtime version          # tag, commit, wire + python semvers
    uv run cozy-runtime new <name>       # scaffold a minimal valid package
    uv run cozy-runtime describe              # human-readable surface summary
    uv run cozy-runtime --json describe       # canonical publication bytes on stdout
    uv run python checks/architecture.py # AST import/dependency scans over the boundaries
    uv run pytest                        # the end-to-end suite: two real runs, no mocks

### Per-package uv environments

`internal/package_environment.py` is the rental materializer. The image already contains Python,
cozy-runtime, TensorFS, and—on accelerator profiles—Torch/CUDA. Each placement gets one uv
venv created from that exact base interpreter with system site packages visible. uv installs only
the deterministic project wheel and exact package dependency wheels with offline, no-index,
no-dependency, and no-build semantics. Python, Torch/CUDA, cozy-runtime, and TensorFS are never
installed a second time inside a rental.

Runtime observes the interpreter it will actually execute once per worker boot: CPython ABI,
Linux architecture/glibc, uv, standard-library roots, every installed distribution version for
dependency resolution, and import-root ownership for the protected Python/Torch/Triton/CUDA/
NVIDIA/cozy-runtime/TensorFS families plus msgspec, Runtime's public interface ABI. Packages may
still carry their own incompatible ordinary dependencies when the selected base does not own that
distribution.
Runtime measures each arriving wheel's real ZIP roots and rejects protected distribution/root
shadowing, duplicate package roots, incompatible tags, and unsatisfied requirements before uv
runs. No image inventory document or caller-authored base fact participates.

Runtime hashes each arriving package wheel once. A worker-boot-local verified-file receipt binds
its digest/length to device, inode, size, mtime and ctime; stable reuse reads that receipt, while a
stat change or worker restart re-hashes before trust. Package admission consumes the receipt instead
of hashing the same wheel again. A venv content identity contains the interpreter ABI/platform,
exact uv version, and sorted package-owned wheel digests. It deliberately excludes observed base
packages: a compatible Runtime patch rechecks declared requirements and reuses the venv,
while a real requirement or ABI conflict refuses. Creation is stage-then-rename and no partial venv
is selectable.

The rented-pod worker stays root and non-dumpable; the exec-first trampoline sets `no_new_privs`
and drops package execution to uid/gid 65533. Runtime keeps control state in memory because its
pod-supervisor owns the durable replay ledger. Receipts, acquisition cache, and worker credentials
remain root-only. Content is root-owned/read-only. The worker keeps no scratch tree: an attempt's
spool and a derived interface's handoff live under its `run/` root for exactly one attempt or
call, and bytes a job stages are the owner's `tmp/<request-id>/`, granted per run and removed by
the owner when the request settles. A worker that cannot perform the
identity split refuses before binding. Native/local development keeps its explicit same-user lane.
Base bytecode is baked in the image and uv compiles package bytecode during installation. Any later
bytecode is directed to the worker-boot/content cache outside the read-only venv. The executor starts
with the venv's own Python; it receives no package `PYTHONPATH` or appended import directory.

Local installed packages use the same ordinary venv boundary but not the rental base policy.
Creator resolves the full lock with uv, so two local packages may carry different Python or Torch
families. `prepare-package --environment-python <venv>/bin/python --package-interface
<download>/metadata/package-interface.json` validates that existing venv and the exact published
interface, and recognizes exact non-protected project/dependency versions as contents of that
completed package environment, not as worker-image distributions. A version disagreement or
protected-family/root collision still refuses. It then authors the PlacementSet without installing
another environment. Local `serve`
receives the same `--environment-python`; `--install-root` is reserved for rental
materialization and the two flags are mutually exclusive.

The local boot warm pass is also capacity evidence. A typed device OOM falsifies only that
executor epoch's attempted placement rung and normalized shape cell. Runtime retries the same
synthetic request through the existing component-staged rung and publishes DISPATCHABLE only when
that fallback is viable; if every proven rung fails capacity, activation stays offline with a typed
fault. Other optional warm failures remain degraded observations rather than placement decisions.

Source-checkout development is an explicit, same-user lane rather than a partially qualified
published package. Creator privately asks Runtime to author a `Placement.development` carrying
package, semantic release, and source digest, then launches Runtime with the matching local path
and facts. No project wheel, release.json, Environment, or receipt is invented.
For a modeled entrypoint, Runtime resolves package.toml through its existing binding owner and
joins the exact release/lane through the explicitly granted local TensorFS Repository. The
Placement carries the ordinary exact Manifest/component facts; at acquisition
the launch Runtime re-reads the source interface and TensorFS Manifest/header/config, derives the
same binding locally, and compares every accepted fact before STAGED. Creator owns the source-file
digest and closes its scan race before replacing the whole worker generation after edits; Runtime
does not invent a second source hash. Header-declared model assets travel in the CozyTensors
runtime closure and reach package construction only through Loader's bounded read-only asset
capability; a temporary private read-only tree exists only for third-party path-based constructors.

A writable delegated cgroup-v2 child remains the preferred executor lifecycle boundary. Ordinary
RunPod containers instead expose their provider-owned cgroup-v2 scope read-only. Runtime uses its
existing process-tree boundary there only after positively proving the root-worker/dedicated-UID
split, root-owned container marker, read-only cgroup lineage and migration controls, finite inherited
`memory.max` and `pids.max`, full target-UID `/proc` visibility, and pidfds. Reclaim owns the entire
uid-65533 domain, not merely an environment token or process group, so token scrubbing, `setsid`,
double-fork, worker restart, and PID reuse do not leave an old executor alive. A writable but
undelegated cgroup-v2 host and an unbounded read-only container still refuse before package import.

Each package environment contains exactly one project wheel and at most 128 dependency wheels. The Runtime bound is
129 total package-owned wheels, counting the project; 129 custom rows plus the project refuses
before any grant or ZIP is opened.

The project wheel is exactly `py3-none-any`: Runtime checks its Tensorhub fact, filename tags,
`WHEEL` tags, `Root-Is-Purelib`, members, and native magic. A project `.so` or a platform-specific
project tag refuses. Every dependency wheel remains a separate exact prebuilt subject selected from
the publisher's lock. Runtime accepts no build request and rejects C/C++/CUDA/Rust sources, build manifests,
headers/objects/static archives, and wheel install schemes that would inject scripts or data. It
never invokes pip, a resolver, PEP 517, or a compiler.

The ordinary package lane admits only pure dependency wheels. A dependency wheel containing ELF,
`.so`, `.pyd`, or another native member refuses as
`package_environment_custom_wheel_native_unsupported` before
installation. Shared native libraries belong in the base worker image. A package-specific native
wheel lane remains absent until a real package requires it and can justify actual loader, SM/PTX,
operator, and isolation checks. `internal/native_wheel.py` and its
`cozy-native-wheel-proof REQUEST.json EVIDENCE.json` door are that lane's inspection and
host-qualification half, built and DORMANT: nothing calls them until the lane opens, and they
are not publication authority.

PyTorch, Triton, and the CUDA driver continue to own JIT cache keys and byte validation, but those
keys are never treated as a same-UID security boundary. Runtime scopes cache reuse by unpredictable
worker boot plus exact package content, then imposes `PYTHONPYCACHEPREFIX`,
`TORCHINDUCTOR_CACHE_DIR`,
`TRITON_CACHE_DIR`, `TORCH_EXTENSIONS_DIR`, and `CUDA_CACHE_PATH` before executor import. A respawn
within that pod can retain its warm compiler cache; the whole pod scope is deleted after worker
shutdown has proven the executor tree gone. Tensorhub's public placement law supplies one public
trust domain per pod. Private renter cross-org stacking retains its explicitly documented same-UID
risk—Runtime does not relabel worker boot identity as tenant isolation. Cache bytes stay node-local,
never enter a package receipt/content identity, and are never published as release artifacts. Every
placement consumes this same venv path; OCI provenance belongs to Tensorhub qualification
evidence, while the portable receipt names only the exact project/dependency-wheel environment.

On a remote placement, `PlacementSet/1` names one exact
`model_object_set` subject. Cozy sends signed logical package/model intent; the worker receives
exact digest/length/HTTPS locations directly from Tensorhub over mTLS. The non-control
convergence lane starts package and model acquisition together. As soon as the package venv and
binding documents are admitted from a standing start, Runtime spawns the executor so interpreter
and Torch import overlap continued TensorFS hydration; model preparation/fill still waits for the
exact model closure. READY remains the join of package environment, TensorFS closure, binding resolution and
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

**The fit (model-code-fit §3).** TensorRequirements exist only as transient in-memory rows.
Runtime executes `factory(config)` on the meta device in the package interpreter's private bounded
child (no network, weight bytes, writes, or CUDA), then discards the rows after use. There is no
public derive verb, document, digest, cache, directory, or publication artifact. The serving
`Loader.construct` performs the same census and calls `tensorfs.fit` directly with requirement
objects and the CozyTensors header before reserving a destination. Encoding consent is one boolean
policy input, not duplicated per tensor. A non-`ok` verdict is the typed prepare refusal
`model_fit_refused`. Every PlacementSet writer fills `Slot.components` from the private construction,
never from the header.

On a cold model-body miss, pod-supervisor may call the loopback-only
`CheckPackageSetCompatibility` before the body finishes. Runtime accepts only the verified package
files plus model Manifests, CozyTensors headers, inline config, and header-declared model assets;
tensor-part files are rejected from this call. It constructs under CUDA-hidden CPU/meta, calls the
same TensorFS fit, discards the transient rows, and returns `checkpoint_incompatible` on a definite
mismatch. A resident model skips this extra construction and follows the ordinary load path.

A package whose signature declares `MediaDecoder` must install `cozy-runtime[media]`;
`describe` refuses the release if PyAV is absent. The package interface publishes the hard per-field
codec ceilings and the attempt-wide retained decoded-value ceiling.

A package that prepares, binds, or fills a model selects `cozy-runtime[model-execution]`.
That Runtime capability owns `tensorfs>=0.2,<1` from PyPI; package code does not declare
TensorFS directly. Every approved base worker image already owns the same distribution, so
rental overlay export prunes it and Runtime observes the image-provided facade before
execution.

Real-GPU behaviour is verified by hand on a rented card. `corpus/` holds the adversarial
derivation cases and the real-torch fixtures for those runs; they are driven from a corpus
venv, not from CI.

Model-bound `cozy-runtime run` uses the same explicit source-checkout lane: package.toml (or its
`--model` override) selects an exact ref, the local TensorFS Repository resolves its release/lane,
and the normal worker/attempt kernel applies the host's real GPU checks. It never authors a
flat binding record or pretends the local selection carries Tensorhub's hardware evidence.

On a private rental, an editable checkout is not mounted or watched. Creator builds one exact
ephemeral project wheel and closed dependency-wheel set, and pod-supervisor verifies their bounded
WorkerControl chunks into local files. The loopback-only
`RuntimePreparation.PrepareLocalPackage` method then calls the same wheel/base/venv/interface/
job-plan implementation as a published package. Its resulting Placement uses
`DevelopmentPackage` source identity plus the measured project wheel and Environment, so it cannot
masquerade as a Tensorhub release. Runtime receives no source path, URL, credential, or catalog
authority.

**TensorFS.** The fill plane reads through tfs-007's published `tensorfs` facade. The base
package installs and type-checks without model execution, exactly as it does without torch; a
fill without the reader is a typed `reader_unavailable` refusal naming the install, never an
`ImportError` escaping a request.

`examples/marco-polo` is the weightless reference package — one entrypoint, one job, no model
— and it is what the end-to-end suite builds, installs and runs.

`run` · `serve` · `job` · `ingest` · `doctor` · `hostfacts` land in
cr-016 over the real attempt kernel.

**Jobs (cr-009).** A job is an attempt class on the same machinery, not a second runtime:
`@app.job` over the one invocation kernel, the same worker/executor split, the same
records, the same output transaction. What differs is stated out loud and is short — a
`JobDirective`/`JobExecutionSpec` instead of the serving pair, BOUNDED ACCEPT-THEN-PREPARE
(serving never accepts-then-cold-loads; a job has nothing to be warm for), the job
services (`Scratch`, `Checkpoints`, `Budget`), typed input `Tree` fields whose path rides
the value, and a RECYCLE process exit distinct from a death. `corpus/job/` is the live
project: a structural census over the real 6.9 GB cozytensors store, read-only.

Model-producing jobs declare `WeightsOutput` slots and request `WeightsSink`. The sink accepts
only exact typed `Model` inputs, locks the declaration/output bound, and streams produced values
directly into TensorFS's real copy/create/derive transaction without a model-sized spool copy.
Config declarations use TensorFS's closed copy/add/derive grammar. Added and derived config bytes
are canonicalized by `DerivedWriter.add_config` into the CozyTensors header; their stream length
is an operational read bound, not published metadata, and configs have no media type.
An opened transaction may read exact source-part ranges into caller-owned contiguous writable
buffers (at most one 64 MiB TensorFS grid window); TensorFS validates alias/component/key/role,
offset and length under the same pinned source lease before any added value commits.
The job receives only the exact receipt—never a store, path, publisher or destination. The
worker sends exact declaration intent before TensorFS opens, waits for the host's writer-epoch
ACK, commits the receipt plus sorted ObjectRef/source-ref inventory, and serves bounded object
ranges on the same WorkerControl stream. Receipt frames and replay ACKs also carry the produced
Manifest `Ref`, so downstream jobs never parse TensorFS receipt bytes. Runtime keeps no journal;
pod-supervisor owns durability.
