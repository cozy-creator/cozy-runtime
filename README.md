# cozy-runtime (v2)

Cozy Runtime supplies the Python author API, model construction and execution helpers,
TensorD-launched executors, and the package CLI. TensorD makes model weights available to
those executors and coordinates machine budgets, storage lifetime and process lifecycle.
TensorFS is a library used inside both TensorD and Runtime.

TensorD is a role of the `cozy` binary (`cozy machine serve`, from
[cozy-creator/cozy](https://github.com/cozy-creator/cozy)); no Runtime wheel carries it.
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
policy.

For media inputs (handles, reads, preparation), see [Media inputs](docs/files.md).

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

**One attempt kernel** (§8). The invocation kernel has exactly one caller (the device
executor).

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
exit, a dead peer, typed outcomes, explicit cancellation and enclosing caller deadlines decide
lifecycle; work that is alive but still is shown, not killed (`tracker/design/stalls.md`). No
number of seconds kills anything on its own. Observation cannot reach residency or placement.

**Component residency is the runtime's, and the package has no vocabulary for it.** An
package declares the possible heavyweight set of each public model operation with
`@uses_components` and nothing else. Entering that scope ADMITS and MATERIALIZES the set
before the body runs — freeing non-declared components, LRU over refcount-0 leases whose
CUDA events have fired, until the plan's activation envelope fits ALLOCATABLE capacity —
and leaving it records the fence and makes the bytes evictable. Eviction FREES; there is no
host residence tier, and a re-stage is an ordinary cr-005 fill from the page-cached CAS
mapping. The vocabulary is unreachable from package code rather than merely unused.

**Encoder results are memoized by marking the method.** `@uses_components(..., memoize=True)`
(with `memo_dependencies`/`memo_version` as on `@invocable`) makes a method eligible; the
Runtime looks the call up before the scope opens, so a hit stages nothing. The key is the
method's code and declared helpers, the environment, the components' weights, the
arguments (values: strings, numbers, host tensors, arrays, images, assets, records) and the
numerics. A method drawing from the default RNG refuses `memo_rng`. Results go to the
executor's LRU, and to the machine's disk tier only when small and measured costly to
produce (`memo:` in runtime.yaml; tracker design `pipeline-memoization.md`).

Runtime's compiled CUDA kernels (`kernels/`) ride its native wheels, one per OS and CPU (Linux
x86_64 and aarch64, Windows x86_64), under `cozy_runtime/_kernels`: the 8-bit op,
FlashAttention-3's forward and RMSNorm + split-half RoPE (`author.kernels.rms_rope_split_half`),
every GPU target's cubins plus PTX, nanobind on the stable ABI, no torch ABI. Each kernel serves
once it passes its probe on the card (`internal/verify.py`). `kernels.yaml` builds and checks the
modules, cached by a hash of everything that shapes their bytes; `task wheel` adds them to the
pure wheel. The pure wheel (macOS, any other platform) carries none.

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
pyproject's `cozy.application` entry point names, following module-scope imports inside the tree
(and cozy-runtime's own `cozy_runtime.derive.*` source), a constant folder over literals,
names, integer arithmetic, `tuple(…)`/`list(…)` and one comprehension over a folded
sequence, and a CLOSED vocabulary — `msgspec.Struct` classes with their
`tag`/`tag_field`/`kw_only` keywords and `msgspec.field(name=…)`, `Annotated` carrying
`msgspec.Meta`/`FileBound`/`ModelDefault`/`Preflight`, `Literal`, `Enum`/`IntEnum`/`StrEnum`,
unions, `list`/`tuple`/`dict`/`Sequence`/`Mapping`/`Set`, the asset classes, `Files[...]` with
`FileLimits`, `Tree`, `ModelArtifact`, `Model` subclasses (`encoded_leaves=`,
`min_bits=`, `phase=`, `@sequence_parallel`, `@uses_components`), `Weights[M]`, and `@app.entrypoint` with literal
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
evaluated, and anything outside the vocabulary (a call or comprehension in
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

A script is an ordinary package (xs-099): its PEP 723 block is its `pyproject.toml` and it
declares `app = App()` and its functions as any package does, so nothing here reads it
differently. Every attempt may call children: the executor's Start carries the call interfaces.

The reader is also what the DEVELOPMENT placement is built from (the worker session's own
staging): it supervises and does not execute, so it reads the source and takes its model
slots off the interface document.

`internal/discovery.py` is the import path and the static reader's ORACLE: it imports exactly
the ONE `App` export that entry point names, freezes it, and stops. `cozy-runtime --conformance
describe` runs both and refuses unless they are byte-identical — and refuses outright unless
the running interpreter is the package's own locked environment (its distribution, at
pyproject's version, resolves there), because importing package code is running it. That is
the check a package's own CI runs from its locked venv; publish runs the static reader alone.

`internal/package_interface.py` is the schema home for `PackageInterface`: only
the launch and invocation facts with current readers. It carries no Runtime-global service
inventory, diagnostics, config UI metadata, or tensor requirements, and
`_fixed_point` proves over the built document that no release digest, build timestamp,
machine path or mutable selection reached it. The document has one `entrypoints` list; an entry
writes `publishes`, `internal` and `weights_outputs` when set and `accelerator` only as
`false`, and its memo flag rides `invocable.memoize`. A slot is `{path, class, component_use,
encoded_leaves}`, plus `default`, `min_bits` and `phase` when declared and `weights: true` for a
`Weights[M]` argument. `descriptor_id` (sha256 of
one entry) is derived by `cozy-runtime describe <function>`, never stored. Deleted members are simply unknown fields; there is no legacy decoder or
empty-value compatibility shim.
Packages declare no GPU count, VRAM floor or lane: the machine picks a slot's lane from its
GPU, the request's `bits` and the slot's `min_bits`/`phase`. Actual resource demand comes from the selected lane's tensor
schema/encoding, qualification measurements and the worker's observed machine; the RecordOwner
owns placement. The
interface reader rejects the retired `resources` key as an unknown field; format `/1` has one
current grammar and no compatibility decoder.
Any function may produce models without another workflow language. A slot's capability is its
annotation (cr-077): `model: M` is constructed in a warm session; `source: Weights[M]` (bare
`Weights`: any layout) is the raw checkpoint, never constructed, read through
`components()`/`tensor()`/`read_into()`/`read_values_into()` or handed to
`ctx.tensorfs_source` and `derive.quantize_artifact`. Authors may name
a slot's default release through `@invocable(defaults={"model": "sdxl@1.0.0"})` or
`@app.entrypoint(defaults=...)`: `[org/]model@release`, never a lane. Omitting the org names a
first-party model, the root namespace's (`sdxl@1.0.0`). A run's `model.<slot>=` wins over the default. Selection resolves before
dispatch; the executor receives exact artifacts. See [authored defaults](docs/authored-model-defaults.md).
The same decorators take `components={"model": ("dit", "vae")}`: the slot's `components` names
the components that callable runs (absent means all), so a machine fetches those first. A
binding that selects only some components leaves the others on meta; a method using one
refuses `component_unbound`.
A call chooses a callee slot's model with `ModelChoice`; None keeps the parent's:

```python
await render(prompt=p, unet=ModelChoice(lora=(Lora("org/style", 0.8),)))
```

`ref` (a name as `cozy run model.<slot>=` takes, or a ModelArtifact) replaces the slot's model and
its LoRAs. Without `ref`, `lora` stacks after the parent's LoRAs for the slot (the request's
`--lora`): the same LoRA again takes the new strength, and strength 0 drops it. TensorD resolves
each name as `--lora` does, composes the call's adapter view and keys its memo by the stack. A
machine whose Start lacks `model_choices` (an older TensorD) refuses such a call:
`cozy machine update`.
Known-next calls can [prepare their models in the background](docs/model-prefetch.md)
with `prefetch(callee, models={...})` while another model runs, and a request done with its
GPUs can [release them early](docs/gpu-release.md) with `ctx.release_gpus()`.
Independent items go as one call each, awaited together (`asyncio.gather`): TensorD runs a
request's calls on every GPU it holds, an idle GPU before a busy one and one call per GPU at a
time, and keeps each executor warm for the request's later calls.
Each `WeightsOutput` declares only its name and new-byte ceiling;
the produced CozyTensors header is the sole tensor-schema and encoding authority. The
executor still receives only verified model capabilities, never the foreign source credential or
destination authority. Multiple transformations are ordinary code inside one bounded function—there is
no `model_productions` graph, compatibility reader, or second scheduler.
`internal/canonical.py` is the RFC 8785 writer — tensorfs (tfs-013) owns the rules.

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

A package whose signature names a video or audio input must install
`cozy-runtime[media]`; `--conformance describe` refuses the release if PyAV is absent (the
static reader has no environment to check). Media inputs are handles read through the
attempt's one budget; see [Media inputs](docs/files.md).

Image generation uses the shared `ImageRequest` output-canvas limit: aspect ratios must be
between **1:16 and 16:1**, inclusive. Explicit ratios outside this range are rejected during
request decoding, before admission or model work. A canvas inferred from a reference is checked
when `target()` resolves it, and `Sizes.fit()` keeps grid rounding within the same envelope.
Models can impose narrower bounds. This output-generation rule does not crop or reject source
assets merely because they have unusual dimensions; an explicit valid output ratio can use
such a reference. Packages that preserve a reference canvas directly must still resolve it
through `ImageRequest.target()` before choosing output dimensions.

A prompt names its production assets (characters, props, environments) as `<name>` when a
package opts into the author library's assets (xs-135): `assets: list[PictureAsset]` (or an
`Asset` subclass with its own media and fields), and `render(prompt, assets, label)` puts each
mention in the model's own words, the label being how the model refers to the Nth asset
(`lambda n, asset: f"Picture {n}"`). Names match in any case with `_`, `-` and space alike; a
mention of no asset stays as written with a warning. `ImageRequest.assets` takes these, and its
first asset sets the default canvas. Nothing below the author library knows the convention: a
package may declare any input shape. Attached files are `File`, `ImageFile`, `VideoFile` and
`AudioFile`; Runtime 0.36.0 renamed them from `*Asset`, and a package importing an old name
fails at import with a remedy.

Shots and upscales have shared interfaces too (xs-136), so a project swaps one renderer for
another: `ShotRequest`/`ShotResult` (`video/shot`: a shot of any length with `beats`, `assets`,
`world_state` and `writes`) and `UpscaleRequest`/`UpscaleResult` (`video/upscale`). A package
subclasses them for its own knobs and declares what it implements on the entrypoint:
`app.entrypoint(shot, implements=Implements("video/shot", max_piece_s=15))`. The interface
document carries it as `implements`, so a caller flags a shot longer than `max_piece_s` with
no beats before anything runs.

Image inputs can separately opt into preparation on a field or an `Files` argument:

```python
from typing import Annotated
from cozy_runtime.author import FileLimits, Files, ImageFile, ImagePreparation

Pictures = Annotated[
    Files[ImageFile],
    FileLimits(images=3),
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

Real-GPU behaviour is verified by hand on a rented card.

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
works on every function: `weights=`, `publishes=`/`emits_media=`, `accelerator=` (False: CPU only;
True or absent: the machine's GPU when it has one), `preflight=`, `demand=`, `defaults=`,
`components=`, `memoize=`, `hidden=`, `internal=`. A writer (`weights=`, `publishes=`,
`emits_media=`) is never memoized (`memoized_writer`); TensorFS derivation replay answers a
repeat.

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

`Weights.read_values_into` provides logical float32 reconstruction over the same
TensorFS source handle; its geometry, encoded identities, byte reads and lease lifetime are all
TensorFS-owned.

## Merging

PRs into `master` merge through the merge queue: `gh pr merge <n>` (the queue squashes; never `--admin`). CI runs there, on the final PR + branch state; nothing runs on a push.
