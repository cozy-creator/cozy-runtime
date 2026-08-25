# cozy-runtime (v2)

The worker runtime: author surface (`cozy_runtime.author`), supervisor, disposable device
executor, and the single-endpoint CLI. Design authority: [tracker-v2](https://github.com/cozy-creator/tracker-v2)
`cozy-runtime.md` + `cozy-runtime-cli.md`; issues in `tracker/cozy-runtime/`.

**No automated tests** (decisions.md #160): verification = running the real thing and
recording what it did, plus time/memory benchmarks.

## Layout

| package | role | may import |
|---|---|---|
| `cozy_runtime.author` | the whole author surface — the only package endpoint code imports | stdlib + msgspec |
| `cozy_runtime.internal` | runtime implementation; owns the ONE config authority | author, protocol, stdlib |
| `cozy_runtime.protocol` | `cozy.worker.v1` — th-024's bindings, COPIED, plus the document plane | protobuf, internal.canonical |
| `cozy_runtime.cli` | local-coordinator adapter over the core | author, internal |

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

## The two processes, and the two coordinators

    a real coordinator ──gRPC───┐
                                ├──▶ supervisor (no CUDA) ──seam──▶ executor (CUDA)
    an in-memory coordinator ───┘        cozy.worker.v1, either way

| module | role |
|---|---|
| `internal/worker/session.py` | the protocol client and the supervisor entrypoint |
| `internal/worker/control.py` | the transport seam: `Link`/`Coordinator`, gRPC and in-memory |
| `internal/worker/attempts.py` | offer -> prepare -> accept -> terminal, attestation, ledger close |
| `internal/worker/ledger.py` | the worker-scoped byte ledger and the attempt reconciliation |
| `internal/worker/plan.py` | the AttemptPlan, the PlanChooser, the local binding-plan record |
| `internal/worker/grants.py` | DeliveryGrant reads/writes + the closed hub-call table |
| `internal/worker/child.py` | spawn/supervise/kill the executor; the two ordinals |
| `internal/local.py` | the ephemeral local supervisor: one request, no network |
| `internal/journal.py` | the durable, self-verifying attempt/terminal journal |
| `internal/seam.py` | the control-only, byte-accounted supervisor<->executor channel |
| `internal/trampoline.py` | the exec-first stage: pdeathsig, no_new_privs, oom_score_adj |
| `internal/executor.py` | the disposable device process — the ONE torch importer |
| `internal/fill.py` | the fill plane: a TensorFS `Checkpoint`, the pinned ring, the transaction |

The supervisor never imports torch, and that is proved against the RUNNING process
(`scripts/worker-live.py nocuda` reads its `/proc/<pid>/maps` and the device's compute-app
list), not against the import graph.

**One attempt kernel** (§8). A one-shot local run is not a second execution path: it is an
ephemeral local supervisor whose coordinator is a pair of in-process queues, over the same
process boundary, journal, acceptance boundary, freeze point and ledger. The rule is
structural, not a promise — the `one-attempt-kernel` fence proves the invocation kernel has
exactly one caller (the device executor) and the attempt engine exactly one constructor (the
supervisor session), so a direct-call runner beside them cannot be written without turning
CI red.

**The ledger** carries one row per byte class, each declaring its KIND (`arithmetic` for the
supervisor's own bookkeeping, `measured` for a named device or kernel interface,
`unreadable` for a class this worker genuinely cannot read — never a zero standing in for
one) and the METHOD that produced it. Every attempt closes by reconciling every class
against the line taken before its envelope was granted, and the record is journaled.

## The descriptor

`internal/discovery.py` is the isolation boundary: it imports exactly the ONE `App` export
endpoint.toml names, freezes it, and stops. Importing a module never registers anything, so
there is no recursive discovery, no dependency stubbing and no source parser — a red arm
plants a self-registering sibling module and proves describe cannot see it.

`internal/descriptor.py` is the schema home for `EndpointDescriptor`/`JobDescriptor`: the
SOURCE-STABLE surface only. It carries no candidate topology (cr-004's
ModelConstructionContract owns that), and `_fixed_point` proves over the built document that
no release digest, build timestamp, machine path or mutable binding reached it.
`internal/canonical.py` is the RFC 8785 writer — tensorfs (tfs-013) owns the rules, and
`scripts/verify.py` runs this codec over tensorfs's own vector corpus and reports which
refusals are shared and which are deliberate profile differences.

`endpoint.descriptor.json` is committed and is a CHECKED CACHE, never authority:
`describe --check` re-derives and refuses a stale one, and CI runs it. `describe` itself is
read-only; `--write-descriptor` is the only writer, and `cozy-runtime new` runs it so a
fresh endpoint is `--check`-clean out of the box.

## Commands

    uv sync
    uv run cozy-runtime version          # tag, commit, wire + python semvers
    uv run cozy-runtime new <name>       # scaffold a minimal valid endpoint
    uv run cozy-runtime describe         # the surface, without loading weights
    uv run cozy-runtime describe --check # refuse a stale endpoint.descriptor.json (CI)
    uv run python scripts/fence.py       # architecture fences
    uv run python scripts/redarm.py      # plant each violation, prove it red, restore
    uv run python scripts/conformance.py # every descriptor-declared fact has a live writer
    uv run python scripts/verify.py      # live run of both example endpoints end to end
    uv run python scripts/bench.py       # time + memory benchmarks

Real-GPU drivers, run from a venv with the `corpus` extra (they need an 8 GiB card and the
real CAS, so they are BUILDER-RUN and never CI steps):

    corpus/.venv/bin/python scripts/worker-live.py attempt   # cr-007: the two-process worker
    corpus/.venv/bin/python scripts/slice-live.py all        # cr-008a: one request, no network

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
