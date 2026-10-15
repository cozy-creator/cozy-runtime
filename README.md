# cozy-runtime (v2)

Cozy Runtime supplies the Python author API, model construction and execution helpers,
TensorD-launched executors, and the package CLI. TensorD makes model weights available to
those executors and coordinates machine budgets, storage lifetime and process lifecycle.
TensorFS is a library used inside both TensorD and Runtime.

The Linux x86-64 wheel (glibc 2.28+) also installs TensorD's `tensord` executable.
Worker images and `cozy machine install` take it from there:

```sh
uv pip install cozy-runtime
tensord version --json
```

It is built from [TensorD](https://github.com/cozy-creator/tensord) at the revision recorded
in `tensord.pin`. `task machine` bundles it into `dist`'s Linux wheel
(`MACHINE=<checkout>` builds a local checkout); the publish workflow runs the same task.
The pure Python wheel and source distribution carry no machine executable.
See [Publishing Runtime](docs/releases.md) for explicit source and approved-prebuilt releases.

TensorD owns accepted executions and their durable journal behind `cozy.machine.v1`.
Runtime reports executor facts over the private control/descriptor channel. The retired
Python worker and its protobuf facade are not part of this topology. See
[Machine and Runtime roles](docs/machine-agent-runtime.md) for the weight-data and memory
boundaries, and [the tracker](https://github.com/cozy-creator/tracker) for current work.

## Compatibility status

Package code uses `cozy_runtime.author`, the shared numerical helpers under
`cozy_runtime.derive`, and TensorFS's scoped `tensorfs.derived` API. Underscored author
modules, Runtime `internal`/`cli`/`protocol`, and administrative TensorFS Store operations
are not package APIs. Package ranges must be requalified for coordinated author API cuts.

Runtime executes packages on the installed standard CPython 3.12, 3.13 and 3.14 window.
TensorD embeds its Rust dependencies; each package environment has its own Python closure. Current
Runtime Python/TensorFS bounds are in `pyproject.toml`. TensorFS parses headers and performs
weight-plane operations inside the Runtime executor; managed store operations and shared
buffer lifetime are coordinated by TensorD. Accepted executions receive scoped source/output
descriptors and budget grants. A model source supplies actual metadata and byte access, while
an installed library or a model identifier alone supplies no managed-store authority.

`cozy-runtime --json python-interpreters` reports the packaged rolling policy and actual
installed interpreters. By default discovery uses `uv python find --no-python-downloads`;
operators may set `COZY_PACKAGE_PYTHONS` to a colon-separated finite list of absolute
interpreter paths. TensorD preparation selects the lowest compatible installed version from the
package's `requires-python` range and optional exact captured version before installing its
complete private dependency closure. No Python download occurs during preparation. Content
identity records the actual Python patch and ABI, and retained environments stop reopening
when their Python minor leaves the supported window. Package `requires-python` declarations
may remain broader than this execution policy.

Python package compatibility is expressed through dependency ranges. TensorD's machine API
and the private executor channel use their own operation/capability contracts; a Python
package version is not a machine-wide authority grant.

Published dependency declarations allow patch upgrades, including optional extras.
Minimum versions (`>=`) are preferred; major/minor compatibility ranges are permitted.
Exact development and CI versions belong in `uv.lock`; worker images retain their exact
wheel and image digests. The Python ABI bound is independent of dependency version
policy. The `dependency-version-ranges` architecture fence holds this boundary.

`uv run pytest` runs the current suite, including real process, descriptor, native storage,
and numerical component checks. CI also checks builds, types, lint and architecture fences.
Tests requiring a GPU or host facilities name those requirements; CPU/descriptor proofs do
not establish GPU inference or provider qualification.

For labelled model-ready input media, see [Ordered media inputs](docs/assets.md).

## Layout

| package | role |
|---|---|
| `cozy_runtime.author` | typed author API and invocation services |
| `cozy_runtime.internal` | executor, model loading, weight policy and scoped storage adapters |
| `cozy_runtime.cli` | package CLI |

`cozy_runtime.author` also holds the INVOCATION KERNEL (`_invoke.py`): decode -> resolve
-> preflight -> hydrate -> invoke -> validate -> finalize. It is pure surface semantics —
no device, config, environment, protocol or network — so it lives beside the types it
enforces, and `internal` depends on `author`, never the reverse. Every door (an entrypoint
or a CPU attempt) is a user experience over that one function, and the per-service fakes
in `cozy_runtime.author.fakes` are capability adapters over it rather than a parallel
runtime.

`internal/config.py` is the only module permitted to read process environment: one read at
the entrypoint, a frozen `RuntimeConfig` passed explicitly thereafter. Second read refuses.
`internal/child_env.py` holds the device-child env projection allowlist as data;
`internal/spawn.py` executes the seal (ERASE -> IMPOSE -> SEAL) around it.
`internal/exits.py` is the one shared exit-code matrix.

## The executor and the machine

    TensorD ──control + descriptors──▶ Runtime executor (Python/PyTorch)

TensorD uses TensorFS to obtain verified stored weights and prepare/retain shared CPU
buffers. Runtime constructs the model, plans its weight layout and uses TensorFS's local
plane/device APIs for GPU allocation, copies, mapping and unmapping. TensorD may retain
exported GPU allocation handles between executors; Runtime attaches those allocations and
binds tensor views in its own process. GPU tensors, prepared CPU buffers, and serialized
weight objects are different representations with different lifetimes.

TensorFS has no separate daemon in this arrangement. Runtime also has explicit standalone
Store helpers for tools/tests; managed executors consume TensorD's scoped sources and buffers.

| module | role |
|---|---|
| `internal/seam.py` | bounded control framing and native descriptor transfer |
| `internal/trampoline.py` | exec-first stage and process settings |
| `internal/executor.py` | disposable device process and package invocation |
| `internal/weights_writer.py` | scoped native source/output client and broker |
| `internal/fill.py` | project source headers into model rows and read plans; explicit standalone Store helpers |
| `internal/weights.py`, `internal/plane.py` | executor weight layout, residency policy and TensorFS GPU/CPU buffer operations |

`checks/architecture.py` keeps versioned identities in two disjoint registries.
`DOCUMENT_KINDS` contains only stored or digested documents that cross a process/repository
boundary: PackageInterface, EnvironmentReceipt, and PackageEnvironment.
`IMPLEMENTATION_IDS` contains the six encoding-provider names whose exact strings enter
implementation, plan, and generation digests; those names are executable identity, not document
formats. Command requests, planner values, worker records, proof-local state, and benchmark reports
are typed by their command/path or storage table and carry no global format tag.

**One attempt kernel** (§8). The rule is structural, not a promise: the `one-attempt-kernel`
fence proves the invocation kernel has exactly one caller (the device executor), so a
direct-call runner beside it cannot be written without turning CI red.

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

The machine owns durable run outcomes and output custody. Runtime returns
structured executor facts and consumes the native capabilities it is granted.
The former Python worker triage ledger, destination-grant table and worker
publication service are retired; see the machine API for current control and
custody behavior.

**The executor's network claim is Python-deep and says so.** `internal/executor.py:main`
installs `sandbox.refuse_network`, so package code that dials is refused typed at
the first `NETWORK_EVENTS` spelling. That bounds a BUGGY package (law 15's same-tenant
rung). It is not a kernel bound: the executor runs in the machine's network namespace
with no seccomp filter, and a C extension or `ctypes` reaches `connect(2)` underneath the
hook. The egress arms observe both halves against one reachable
address, so the claim cannot quietly grow. Granted input bytes are supplied by the machine and
handed over as verified paths; the executor is never given an address.

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

**Encoder results are memoized by marking the method.** `@uses_components(..., memoize=True)`
(with `memo_dependencies`/`memo_version` as on `@invocable`) makes a method eligible; the
Runtime looks the call up before the scope opens, so a hit stages nothing. The key is the
method's code and declared helpers, the environment, the components' weights, the
arguments (values: strings, numbers, host tensors, arrays, images, assets, records) and the
numerics. A method drawing from the default RNG refuses `memo_rng`. Results go to the
executor's LRU, and to the machine's disk tier only when small and measured costly to
produce (`memo:` in runtime.yaml; tracker design `pipeline-memoization.md`).

Runtime is ONE PyPI package with two wheel files per release: the pure `py3-none-any`
wheel, and a `cp312-abi3` linux/x86 wheel carrying the same code plus the compiled CUDA
kernels at `cozy_runtime._kernels` (`COZY_RUNTIME_BUILD_KERNELS=1` selects that build;
the default `uv build` stays toolchain-free). The `native/` sources ship generic fused
operations (today one: RMSNorm + split-half RoPE); packages decide where they run. The
anima package installs it in its own post-fill plan and runs eager wherever the extension,
CUDA or its reviewed Torch/CUDA pair is missing.

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
`AssetLimits`, `Tree`, `ModelArtifact`, `Model` subclasses (`encoded_leaves=`,
`@sequence_parallel`, `@uses_components`), `Weights[M]`, and `@app.entrypoint` with literal
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
way. It registers the script's one `main` as an entrypoint; `[tool.cozy.models]` is its
default ladder and `[tool.cozy] accelerator = false` keeps it on CPU.
`author/_script_metadata.py` is the ONE definition of that shape, run by the adapter and the
reader over the same parsed tree. Every attempt may call children: the executor's Start
carries the call interfaces.

The reader is also what the DEVELOPMENT placement is built from (the worker session's own
staging): it supervises and does not execute, so it reads the source and takes its model
slots off the interface document.

`internal/discovery.py` is the import path and the static reader's ORACLE: it imports exactly
the ONE `App` export package.toml names, freezes it, and stops. `cozy-runtime --conformance
describe` runs both and refuses unless they are byte-identical — and refuses outright unless
the running interpreter is the package's own locked environment (its distribution, at
pyproject's version, resolves there), because importing package code is running it. That is
the check a package's own CI runs from its locked venv; publish runs the static reader alone.

`internal/package_interface.py` is the schema home for `PackageInterface`: only
the launch and invocation facts with current readers. It carries no Runtime-global service
inventory, diagnostics, config UI metadata, or tensor requirements, and
`_fixed_point` proves over the built document that no release digest, build timestamp,
machine path or mutable binding reached it. The document has one `entrypoints` list; an entry
writes `publishes`, `internal` and `weights_outputs` when set and `accelerator` only as
`false`, and its memo flag rides `invocable.memoize`. A slot is `{path, class, component_use,
encoded_leaves}`, plus `weights: true` for a `Weights[M]` argument. `descriptor_id` (sha256 of
one entry) is derived by `cozy-runtime describe <function>`, never stored. Deleted members are simply unknown fields; there is no legacy decoder or
empty-value compatibility shim.
Packages declare no GPU count or VRAM floor. Authored default ladders associate GPU selectors
with preferred model lanes. Actual resource demand comes from the selected lane's tensor
schema/encoding, qualification measurements and the worker's observed machine; the RecordOwner
owns placement. The
interface reader rejects the retired `resources` key as an unknown field; format `/1` has one
current grammar and no compatibility decoder.
Any function may produce models without another workflow language. A slot's capability is its
annotation (cr-077): `model: M` is constructed in a warm session; `source: Weights[M]` (bare
`Weights`: any layout) is the raw checkpoint, never constructed, read through
`components()`/`tensor()`/`read_into()`/`read_values_into()` or handed to
`ctx.tensorfs_source` and `derive.quantize_artifact`. Authors may attach
an ordered model default ladder through `@invocable(defaults={"model": [...]})` or
`@app.entrypoint(defaults={"model": [...]})`; both publish the same per-slot `default_ladder`
metadata. Each rung names a GPU selector and a `[org/]model@release/lane` reference, and
one slot's rungs share its model and release. Omitting the org names a model of the package's
own account; the publishing Creator writes that account into the published interface. The invocation override wins over the owner's
mutable Hub binding, which wins over the published default. Selection resolves before
dispatch; the executor receives exact artifacts. See [authored defaults](docs/authored-model-defaults.md).
The same decorators take `components={"model": ("dit", "vae")}`: the slot's `components` names
the components that callable runs (absent means all), so a machine fetches those first. A
binding that selects only some components leaves the others on meta; a method using one
refuses `component_unbound`.
Known-next calls can [prepare their models in the background](docs/model-prefetch.md)
with `prefetch(callee, models={...})` while another model runs, and a request done with its
GPUs can [release them early](docs/gpu-release.md) with `ctx.release_gpus()`.
The existing `package.toml [bindings."<callable>.models.<param>"]` remains the explicit local
development binding; it is not copied into a package's published default.
Each `WeightsOutput` declares only its name and new-byte ceiling;
the produced CozyTensors header is the sole tensor-schema and encoding authority. The
executor still receives only verified model capabilities, never the foreign source credential or
destination authority. Multiple transformations are ordinary code inside one bounded function—there is
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
    uv run --group diffusers pytest tests/test_package_extensions.py   # + the Diffusers integrations

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

A local package's captured wheels are uploaded to TensorD and installed into a
generation through its package installer. The executor imports that installed environment.
Header-declared model assets travel in the CozyTensors runtime closure and reach package
construction only through Loader's bounded read-only asset capability; a temporary private
read-only tree exists only for third-party path-based constructors.

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

These example caps are package policy. Still images use Pillow decoding,
`ImageOps.exif_transpose` (including mirrored orientations), RGB conversion and Lanczos
resampling. Audio and video retain PyAV. Preparation preserves aspect ratio and never
upscales. Creator can run the shared Runtime helper before upload; Runtime applies the
same transform to originals sent by other clients. Small inputs keep their original
carrier. Oversized inputs use one fast lossless encoding, or the original if it is
smaller and admissible. Source files remain unchanged. Byte length, MIME and digest
always describe the delivered carrier, so raw and prepared uploads can have different
memo keys while decoding to identical pixels.

The helper checks source byte/pixel limits before decoding and uses an isolated 2 GiB
address-space limit. That process limit is never applied to a GPU executor. The
`image-fit/1` declares the geometry policy. Derivative filenames hash freshly
produced carrier bytes, so cached images are reused only when the actual output matches.
Missing local capability leaves preparation to Runtime. Fidelity remains an independent
package-interpreted hint.

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

An unpublished package runs from captured source without a Tensorhub package release.
A local script is the single-file form; editable describes an installation mode.
On a rental, an editable checkout is not mounted or watched. Creator builds one exact
ephemeral project wheel and dependency-wheel set. TensorD bounds the upload, verifies
each wheel's digest and length, then invokes its captured-package installer to build an installed
generation. The Python executor receives the installed environment and Runtime grants rather
than controller credentials or a watched source checkout.

**TensorFS.** The fill plane reads through tfs-007's published `tensorfs` facade. The base
package installs and type-checks without model execution, exactly as it does without torch; a
fill without the reader is a typed `reader_unavailable` refusal naming the install, never an
`ImportError` escaping a request.

`examples/marco-polo` is the weightless reference package — two functions, one CPU-only, no
model — and it is what the end-to-end suite builds, installs and runs.

**One registration (xs-040).** `@app.entrypoint` is the only registration and every keyword
works on every function: `weights=`, `publishes=`/`emits_media=`, `accelerator=False` (absent
means the machine's GPU when it has one), `preflight=`, `demand=`, `defaults=`,
`components=`, `memoize=`, `hidden=`, `internal=`. A writer (`weights=`, `publishes=`,
`emits_media=`) is never memoized (`memoized_writer`); TensorFS derivation replay answers a
repeat. `corpus/job/` is a structural census over the real 6.9 GB cozytensors store, read-only.

Model-producing functions declare `WeightsOutput` slots and use `Context.tensorfs_source`,
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
managed calls. `Weights.read_values_into` provides logical float32 reconstruction over the same
TensorFS source handle; its geometry, encoded identities, byte reads and lease lifetime are all
TensorFS-owned.
