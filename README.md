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
| `cozy_runtime.internal` | runtime implementation; owns the ONE config authority | author, stdlib |
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
`internal/child_env.py` holds the device-child env projection allowlist as data (the seal
ordering itself is cr-007). `internal/exits.py` is the one shared exit-code matrix.

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
