# Marco/Polo example

This is a real, weightless Cozy package that can be published to Tensorhub or run through
dynamic worker-placement proofs. Its project wheel contains only package code: the base worker image
provides Cozy Runtime and `msgspec`.

The sole public entrypoint has this closed contract:

```json
{"message":"marco"}
```

returns:

```json
{"message":"polo"}
```

Set `[tool.cozy] organization` in `pyproject.toml` to an organization you can publish, then run
Cozy from this directory:

```sh
cozy package publish
```

The default metadata publishes `paul/marco-polo-package@1.0.0`. Cozy derives the name and version
from the built wheel and never adds a `v` prefix.

Cozy invokes `uv build --wheel`; this example declares `uv_build` and its existing flat module
root explicitly. The backend produces a deterministic `py3-none-any` wheel without writing build
output into the source tree. `pyproject.toml`, `uv.lock`, the Runtime-derived canonical descriptor,
and that wheel are the exact publication inputs. The
descriptor is sent directly to Tensorhub and is not committed here. There is no release manifest
and no author-selected worker profile: Tensorhub derives compatible base worker images from the
locked Python/package requirements. The current remote contract admits one placement per worker,
so publishing this fixture under two package repositories can prove sequential same-pod
replacement with `cozy rent revise`; it does not prove simultaneous stacking.

Cozy derives the publication descriptor through this locked, read-only project command:

```sh
uv run --locked --project examples/marco-polo \
  cozy-runtime --json --dir examples/marco-polo describe
```
