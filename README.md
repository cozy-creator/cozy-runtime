# cozy-runtime (v2)

Machine execution requires worker wire 59. Before first submission, the authenticated
RecordOwner reads `GetMachineExecutionWorkspace` and durably pins its workspace ID
with the exact request. `expected_execution_workspace_id` is required on every submit
and replay. Runtime compares it inside the existing journal acceptance transaction;
the returned receipt carries that same transaction's identity. A replacement journal
refuses as `execution_workspace_changed` even when worker/boot identity is unchanged.
Clients must retain unresolved work rather than replace the pinned ID and resubmit.

The worker runtime: author surface (`cozy_runtime.author`), worker, disposable device
executor, and the single-package CLI. Design authority: [tracker-v2](https://github.com/cozy-creator/tracker-v2)
`cozy-runtime.md` + `cozy-runtime-cli.md`; issues in `tracker/cozy-runtime/`.

## Compatibility status

Package code uses `cozy_runtime.author`, the shared numerical helpers under
`cozy_runtime.derive`, and TensorFS's scoped `tensorfs.derived` API. Underscored author
modules, Runtime `internal`/`cli`/`protocol`, and administrative TensorFS Store operations
are not package APIs. Package ranges must be requalified for coordinated author API cuts.

Runtime executes packages on the installed standard CPython 3.12, 3.13 and 3.14 window.
The control process and different package executors may use different versions. The native writer cut requires TensorFS 0.3.41; model readback additionally requires
the scoped reader API under TensorFS PR170 (currently qualified with a source-identified dev wheel).
accepted executions receive scoped source/output descriptors while the worker owns Store
access, admission, epochs and the execution journal.

`cozy-runtime --json python-interpreters` reports the packaged rolling policy and actual
installed interpreters. By default discovery uses `uv python find --no-python-downloads`;
operators may set `COZY_PACKAGE_PYTHONS` to a colon-separated finite list of absolute
interpreter paths. Preparation selects the lowest compatible installed version from the
package's `requires-python` range and optional exact captured version before installing its
complete private dependency closure. No Python download occurs during preparation. Content
identity records the actual Python patch and ABI, and retained environments stop reopening
when their Python minor leaves the supported window. Package `requires-python` declarations
may remain broader than this execution policy.

The distribution major is the Python compatibility major: compatible additions remain in the
current range, breaking author changes require a new major, and candidate base worker images must
replay the supported-range package corpus before activation. Tensorhub images still pin one exact
Runtime wheel digest; package admission uses the declared range and selects the newest compatible
active Runtime/profile. The protobuf package major and additive wire minor remain their own
worker-control compatibility contract.

Published dependency declarations allow patch upgrades, including optional extras.
Minimum versions (`>=`) are preferred; major/minor compatibility ranges are permitted.
Exact development and CI versions belong in `uv.lock`; worker images retain their exact
wheel and image digests. The Python ABI bound is independent of dependency version
policy. The `dependency-version-ranges` architecture fence holds this boundary.

**One end-to-end test suite, no mocks.** `uv run pytest` runs against
`examples/marco-polo` through the public CLI installed from a freshly built wheel into a venv
that has nothing else: describe, an exact byte-identical package interface replay, the exit-code
matrix, one served request, and one job through the real two-process worker. Nothing is
stubbed, patched or doubled — a test that passes because a fake agreed with it is worse than
no test. The two legs that SPAWN AN EXECUTOR skip where the host cannot contain one (a
runner whose own cgroup is not delegated) and name that host fact as the reason. Beside them
CI runs the build, ruff, mypy, the reproducible-wheel check and `checks/architecture.py`'s
AST import scans, and nothing else. Hardware is verified by hand on a rented card.

For labelled model-ready input media, see [Ordered media inputs](docs/assets.md).

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
payload environment variable, certificate bypass, proxy certificate header, or
authentication fallback.

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

For Python development edits, reuse a verified published native wheel without running CUDA:

```sh
uv run --script scripts/build-dev-wheel.py \
  --native-wheel /path/to/cozy_runtime-<version>-cp312-abi3-manylinux_2_28_x86_64.whl \
  --native-sha256 <that-wheel-sha256> --out dist/dev
```

This builds the current working tree, including uncommitted Python edits. It checks the native
sources, build configuration and wrapper against the donor's embedded commit; changes require a
normal native build. The result remains one standard Runtime wheel. Its local dev version binds
both input wheels and the build script, and `COMMIT="unknown"` avoids claiming Git HEAD for a
dirty snapshot. `SOURCE_WHEEL_SHA256` identifies the exact Python wheel retained under the
output's `source-wheels/`; `native-provenance.json` records the donor commit and wheel/binary
hashes. Standard `wheel unpack` and `wheel pack` verify and
regenerate RECORD. Install the resulting wheel through the developer pod's coordinated software
update; this command builds an artifact and does not replace a running worker.

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

`internal/static_interface.py` derives the interface from SOURCE (cr-114, decision #713 —
package code is untrusted). `cozy-runtime describe` runs it by default: `ast` over the module
package.toml's `[application] object` names, following module-scope imports inside the tree
(and cozy-runtime's own `cozy_runtime.derive.*` source), a constant folder over literals,
names, integer arithmetic, `tuple(…)`/`list(…)` and one comprehension over a folded
sequence, and a CLOSED vocabulary — `msgspec.Struct` classes with their
`tag`/`tag_field`/`kw_only` keywords and `msgspec.field(name=…)`, `Annotated` carrying
`msgspec.Meta`/`AssetBound`/`ModelDefault`/`Preflight`, `Literal`, `Enum`/`IntEnum`/`StrEnum`,
unions, `list`/`tuple`/`dict`/`Sequence`/`Mapping`/`Set`, the asset classes, `Assets[...]` with
`AssetLimits`, `Tree`, `ModelArtifact`, `Model` subclasses (`encoded_leaves=`/`fusion=`,
`@sequence_parallel`, `@uses_components`), and `@app.entrypoint`/`@app.job` with literal
keywords and `WeightsOutput(...)` tuples. Explicit function registrations and async
`@invocable` exports use the same schema builders, with their declared capabilities and
immutable defaults. A value the package owns as DATA rather than as
code — MiniMax-H3's served denoise steps are its committed task plan's, and decision #705
forbids a second copy in serving code — is spelled `data_values(__file__, "plan.json",
"schedules", "transformer_evaluations")`: the reader and the imported app take it from the
same bytes through the same `cozy_runtime.author` function, confined to the calling module's
own directory and to JSON scalars, so reading data is never running code (cr-115). Each
declaration is rebuilt as a real type
(`msgspec.defstruct`, `enum.Enum`) and rendered by the SAME `internal/schema.py` the import
path uses, so the two readings cannot drift. Nothing of the package is imported, executed or
evaluated — `checks/architecture.py`'s `static-describe-executes-nothing` fence holds the
reader's import list to that — and anything outside the vocabulary (a call or comprehension in
an interface position, a type from outside the tree, a star import, computed registration,
or a function-local import of an interface name) is a typed refusal naming `file:line`
and the construct: `static_unresolvable`, `static_computed`, `static_unsupported`,
`static_module`. Never a guess.

A package may build its surface out of an INSTALLED cozy package's declarations — a `Model`
subclass, a shared request struct; `diagnostics/h3-vae-roundtrip` subclasses MiniMax-H3's
`H3Model` and reuses its request type. That dependency's source ships in its wheel, so the
reader reads it out of the environment it is itself installed in, confined to distributions
declaring a `cozy.application` entry point — the same fact `discovery.py` calls an app. torch
declares none, site-packages is never a root, and a sibling package supplies TYPES only: its
own callables are its own, never scanned as registrations (cr-117).

A generated SCRIPT app — Creator's `app = script_app("<module>")` adapter — is read the same
way: the adapter's one job is fixed, so all that varies is whether the script has a valid
`main`, and `author/_script.py`'s `script_signature` is the ONE definition of that shape, run
by the adapter and the reader over the same parsed tree.

The reader is also what the DEVELOPMENT placement is built from (`cozy-runtime host
--development-project`, `host development-placement`, the worker session's own staging): they
supervise, they do not execute, so they read the source and take their model slots off the
interface document. `cozy-runtime run` still imports — it is about to run the package.

`internal/discovery.py` is the import path and the static reader's ORACLE: it imports exactly
the ONE `App` export package.toml names, freezes it, and stops. `cozy-runtime --conformance
describe` runs both and refuses unless they are byte-identical — and refuses outright unless
the running interpreter is the package's own locked environment (its distribution, at
pyproject's version, resolves there), because importing package code is running it. That is
the check a package's own CI runs from its locked venv; publish runs the static reader alone.

`internal/package_interface.py` is the schema home for `PackageInterface`/`JobDescriptor`: only
the launch and invocation facts with current readers. It carries no Runtime-global service
inventory, diagnostics, config UI metadata, or tensor requirements, and
`_fixed_point` proves over the built document that no release digest, build timestamp,
machine path or mutable binding reached it. A slot is `{path, class, component_use,
encoded_leaves}`. Deleted members are simply unknown fields; there is no legacy decoder or
empty-value compatibility shim.
Packages declare no GPU count or VRAM floor. Authored default ladders associate GPU selectors
with preferred model lanes. Actual resource demand comes from the selected lane's tensor
schema/encoding, qualification measurements and the worker's observed machine; the RecordOwner
owns placement. The
interface reader rejects the retired `resources` key as an unknown field; format `/1` has one
current grammar and no compatibility decoder.
An ordinary `@app.job` may opt into source-first model production without introducing another
workflow language. A slot's capability is its class annotation (cr-077). Authors may attach
an ordered model default ladder through `@invocable(defaults={"model": [...]})` or
`@app.entrypoint(defaults={"model": [...]})`; both publish the same per-slot `default_ladder`
metadata. Each rung names a GPU selector and full `org/model@release/lane` reference, and
one slot's rungs share its model and release. The invocation override wins over the owner's
mutable Hub binding, which wins over the published default. Selection resolves before
dispatch; the executor receives exact artifacts. See [authored defaults](docs/authored-model-defaults.md).
Known-next calls can [prepare their models in the background](docs/model-prefetch.md)
with `prefetch(callee, models={...})` while another model runs.
The existing `package.toml [bindings."<callable>.models.<param>"]` remains the explicit local
development binding; it is not copied into a package's published default. The retired `source_profiles=` decorator field refuses with the typed
code `source_profiles_retired`. Each `WeightsOutput` declares only its name and new-byte ceiling;
the produced CozyTensors header is the sole tensor-schema and encoding authority. The
executor still receives only verified model capabilities, never the foreign source credential or
destination authority. Multiple transformations are ordinary code inside one bounded job—there is
no `model_productions` graph, compatibility reader, or second scheduler.
`internal/canonical.py` is the RFC 8785 writer — tensorfs (tfs-013) owns the rules, and
`checks/architecture.py`'s `canonical-numbers` fence replays it over the frozen ES6 vector
corpus in `vectors/es6-numbers.txt`.

The package interface is derived, not committed to the source tree. `cozy-runtime --json --dir
PROJECT describe` writes canonical bytes to stdout from source alone — no torch, no venv, no
package import. Creator places those exact bytes at `metadata/package-interface.json` in the
ordinary uploaded package file tree, and `cozy package publish` refuses when its own static
describe disagrees with them. `package_interface_digest` is SHA-256 over those canonical bytes.

## Commands

    uv sync
    uv run cozy-runtime version          # tag, commit, wire + python semvers
    uv run cozy-runtime new <name>       # scaffold a minimal valid package
    uv run cozy-runtime describe              # the surface, from source alone
    uv run cozy-runtime --json describe       # canonical publication bytes on stdout
    uv run --locked cozy-runtime --conformance describe   # + import the app here; equal or refuse
    uv run python checks/architecture.py # AST import/dependency scans over the boundaries
    uv run pytest                        # the end-to-end suite: two real runs, no mocks

### Per-package uv environments

Every published package, editable project and unpublished script must use a private
uv environment with its complete selected dependency closure, locally and remotely.
Baked Python distributions are reuse optimizations, not package version constraints.
The accepted migration and mandatory development gates are in
[Package dependency isolation](docs/package-dependency-isolation.md).

The current implementation still contains inherited-system-site and protected-family
policies being replaced under proto-057/cr-186. Do not use those restrictions as a
contract for new code. Preserve atomic generations, immutable byte custody, storage
admission and executor process/credential isolation while removing image-dependent
resolution. Exact captured locks preserve reproducibility; authors declare compatible
ranges. Dependency repair must solve and validate the full closure before dispatch.

A model has exactly three lifecycle methods — `load(loader)`, optional `warm(ctx)`,
`unload(loader)` (model-lifecycle.md, #708). `warm` is the author's one place for post-fill,
pre-serving work: the runtime calls it once per construction fill, after the fill and its
verification and before the placement is DISPATCHABLE, on every worker including dynamic
placements, with the device lease held and real tensors in place — never again on a component
stage or evict, and again only when the construction is rebuilt. Whether to compile at all is
the author's choice; inside `warm` the runtime refuses only the name-renaming
`torch.compile(module)` spelling (the in-place `module.compile()` is allowed), the `Context`
carries the device, the fill's deadline and cancellation and can reserve no package call, and
a raise is a typed construction failure (`warm_failed`, naming the exception) that keeps the
placement OFFLINE. Its wall time is the `warm` leg of the prepare, on the fill's boot record.
There is no entrypoint-level warm pass and no `ctx.boot_warmup`.

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

Releases may offer multiple project wheels; each environment installs one selected variant.
Project and dependency wheels can contain precompiled native code. Runtime checks filename and
`WHEEL` tags against the selected interpreter and host's actual Linux x86-64/glibc compatibility,
including CPython-specific and stable `abi3` tags. An incompatible interpreter or platform refuses;
native bytes labeled with an `any` platform also refuse. `Root-Is-Purelib` determines the normal
wheel install location and does not assert that every member is Python source.

The same local and remote package installer retains bounded archive expansion, safe paths, exact
name/version/tag identity, import ownership, and per-member `RECORD` SHA-256 validation. Inert
source files and headers may accompany prebuilt wheels; installation always passes `--no-build`
and never invokes a compiler or PEP 517 backend. The remote supervisor still owns downloads and
storage admission, and receives one exact selected URL/hash for each distribution.

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
device construction. An outgoing placement is never disturbed by this optimization. Verified
complete objects survive a re-issue of the same revision; only missing objects move.

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
`--conformance describe` refuses the release if PyAV is absent (the static reader has no
environment to check). The package interface publishes the hard per-field
codec ceilings and the attempt-wide retained decoded-value ceiling.

Decoded image inputs can opt into preparation through their argument annotation:

```python
from typing import Annotated
from cozy_runtime.author import AssetLimits, Assets, Image, ImagePreparation

Pictures = Annotated[
    Assets[Image],
    AssetLimits(images=3),
    ImagePreparation(max_edge=4096, max_pixels=16_777_216),
]
```

These example caps are package policy. Preparation preserves aspect ratio and never
upscales. Creator can run the shared Runtime helper before upload; Runtime applies the
same transform to originals sent by other clients. Small inputs keep their original
carrier. Oversized inputs use one fast lossless encoding, or the original if it is
smaller and admissible. Source files remain unchanged. Byte length, MIME and digest
always describe the delivered carrier, so raw and prepared uploads can have different
memo keys while decoding to identical pixels.

The helper checks source byte/pixel limits before decoding and uses an isolated 2 GiB
address-space limit. That process limit is never applied to a GPU executor. The
`image-fit/1` profile qualifies the shared codec closure; missing local capability
leaves preparation to Runtime. Fidelity remains an independent package-interpreted hint.

A package that prepares, binds, or fills a model selects `cozy-runtime[model-execution]`.
That Runtime capability owns `tensorfs>=0.2,<1` from PyPI; package code does not declare
TensorFS directly. Every approved base worker image already owns the same distribution, so
rental overlay export prunes it and Runtime observes the image-provided facade before
execution.

Real-GPU behaviour is verified by hand on a rented card. `corpus/` holds the adversarial
derivation cases and the real-torch fixtures for those runs; they are driven from a corpus
venv, not from CI.

Plain transformer components larger than the assigned weight-residence ceiling use their
existing `_no_split_modules` boundaries. Runtime retains common weights and fills one block
through the verified TensorFS read plan and bounded pinned ring before its forward. The
completion fence must fire before that block returns to meta. During each paged forward,
Runtime disables only autocast weight caching and restores the caller's cache setting on
every exit. Autocast dtype/enabled state remain unchanged; rematerialized FP32 blocks
must not accumulate FP16 cached copies across a multi-tile decode. There is no CPU model copy;
the planner reports nominal model bytes separately from the common-plus-largest-block
working set. Parameters and persistent buffers remain immutable checkpoint destinations;
nonpersistent rotary buffers retain their config-derived initialization.
Paged blocks must expose live returned tensors through registered PyTrees. Returned weight
views refuse without freeing storage; arbitrary hidden state retaining weight views is not
qualified by the no-split declaration or this proof.

With H3's locked dependencies and the candidate Runtime/TensorFS installed, run the actual
package constructor and GPU parity/refusal proof for each transformer:

```bash
python tests/testdata/h3_paging_proof.py --packages-root /path/to/packages --work-dir /path/to/proof text_encoder
python tests/testdata/h3_paging_proof.py --packages-root /path/to/packages --work-dir /path/to/proof fl2va_dit
python tests/testdata/h3_paging_proof.py --packages-root /path/to/packages --work-dir /path/to/proof video_vae
```

The fixture preserves all 50 language/DiT layers while reducing widths. Its all-resident
reference is discarded before paging; each paged forward must reproduce exact outputs and
evict every visited block. It also checks undersized working-set and autograd refusals,
and process poisoning after a failed block forward. The VAE arm uses the actual modular
decode and checks prior autocast-cache settings after normal and exceptional exits.
This proves weight paging, not full-video
activation capacity. H3's full BF16 text and DiT weight working sets are about 2.90 GB and
1.42 GB respectively; video-VAE activation tiling and encoded-leaf eviction remain separate
qualification gaps. No dtype or inference backend is substituted to make a model fit.

Model-bound `cozy-runtime run` uses the same explicit source-checkout lane: package.toml (or its
`--model` override) selects an exact ref, the local TensorFS Repository resolves its release/lane,
and the normal worker/attempt kernel applies the host's real GPU checks. It never authors a
flat binding record or pretends the local selection carries Tensorhub's hardware evidence.

An unpublished package runs from captured source without a Tensorhub package release.
A local script is the single-file form; editable describes an installation mode.
On a rental, an editable checkout is not mounted or watched. Creator builds one exact
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

Model-producing jobs declare `WeightsOutput` slots and use `Context.tensorfs_source`,
`Context.output`, and `Context.adopt_model`. TensorFS owns the typed `Derivation`, `Target`,
`Tensor`, `Part`, and `Config` declarations and the source/output handles. Inspection keeps
the source header's construction order. Native validation locks exact sources, geometry,
order and output bounds before accepting values; inherited roles require no payload copy.

Source/config bytes and streamed additions use bounded TensorFS descriptor frames. Runtime
exchanges only bounded declaration and receipt metadata; it creates no complete-role spool
copy and the executor opens no Store. Added/derived configs are canonicalized into the
CozyTensors header. Completed-role inventories and checkpoints come from the native writer.

Runtime's worker assigns the accepted execution's transaction and writer epoch, admits its
remaining storage needs, and records native checkpoint/receipt facts in its Workspace before
success reaches the executor. `ctx.adopt_model(native_receipt)` returns the verified
`ModelArtifact`. Source and writer handles are fenced with the attempt, while retained native
partial and committed bytes survive process loss for authorized recovery.

Plain PEP 723 producers that declare `[tool.cozy.weights]` use an explicit `ctx: Context`
parameter. Ordinary client scripts can keep their lightweight `ScriptContext` for composing
managed calls. `WeightsReader` provides logical float32 reconstruction over the same TensorFS source handle;
its geometry, encoded identities, byte reads and lease lifetime are all TensorFS-owned.
