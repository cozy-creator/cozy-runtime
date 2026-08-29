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

The default metadata publishes `paul/marco-polo-package@1.0.1`. Cozy derives the name and version
from the built wheel and never adds a `v` prefix.

Cozy invokes `uv build --wheel`; this example declares `uv_build` and its existing flat module
root explicitly. The backend produces a deterministic `py3-none-any` wheel without writing build
output into the source tree. The `[tool.uv.sources]` entry is only a local-development override:
it is not embedded in the project wheel. The base worker image supplies one exact Cozy Runtime
wheel, while the package declares the oldest compatible Runtime API it needs. Arbitrary local or
editable dependencies are not bundled by publication.

`pyproject.toml`, `uv.lock`, the Runtime-derived canonical descriptor, and the project wheel are
the exact publication inputs. The
descriptor is sent directly to Tensorhub and is not committed here. There is no release manifest
and no author-selected worker profile: Tensorhub derives compatible base worker images from the
wheel-declared Python/package requirements and its exact base inventory. The current remote contract
admits one placement per worker, so publishing this fixture under two package repositories can prove
sequential same-pod replacement with `cozy rent revise`; it does not prove simultaneous stacking.

Cozy derives the publication descriptor through this locked, read-only project command:

```sh
uv run --locked --project examples/marco-polo \
  cozy-runtime --json --dir examples/marco-polo describe
```
