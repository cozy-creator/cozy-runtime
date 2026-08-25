# cozy-runtime (v2)

The worker runtime: author surface (`cozy_runtime.author`), supervisor, disposable device
executor, and the single-endpoint CLI. Design authority: [tracker-v2](https://github.com/cozy-creator/tracker-v2)
`cozy-runtime.md` + `cozy-runtime-cli.md`; issues in `tracker/cozy-runtime/`.

**No automated tests** (decisions.md #160): verification = running the real thing and
recording what it did, plus time/memory benchmarks.

## Layout

| package | role | may import |
|---|---|---|
| `cozy_runtime.author` | the whole author surface — the only package endpoint code imports | stdlib + schema/typing libraries |
| `cozy_runtime.internal` | runtime implementation; owns the ONE config authority | author, stdlib |
| `cozy_runtime.cli` | local-coordinator adapter over the core | author, internal |

`internal/config.py` is the only module permitted to read process environment: one read at
the entrypoint, a frozen `RuntimeConfig` passed explicitly thereafter. Second read refuses.
`internal/child_env.py` holds the device-child env projection allowlist as data (the seal
ordering itself is cr-007). `internal/exits.py` is the one shared exit-code matrix.

## Commands

    uv sync
    uv run cozy-runtime version          # tag, commit, wire + python semvers
    uv run cozy-runtime new <name>       # scaffold a minimal valid endpoint
    uv run python scripts/fence.py       # architecture fences
    uv run python scripts/redarm.py      # plant each violation, prove it red, restore
    uv run python scripts/bench.py       # time + memory benchmarks

`run` · `serve` · `job` · `describe` · `pull` · `ingest` · `list` · `rm` · `doctor` · `fit`
land in cr-016 over the real attempt kernel.
