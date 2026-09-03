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

After Cozy has authenticated to Tensorhub and resolved your current account, run it from this
directory:

```sh
cozy package publish
```

Cozy derives the repository owner from the current authenticated Tensorhub account and derives
`marco-polo@1.0.4` from the built wheel. Publisher ownership is not project metadata, and Cozy
never adds a `v` prefix to the version.

Cozy invokes `uv build --wheel`; this example declares `uv_build` and its existing flat module
name and root explicitly. The published distribution is `marco-polo`; the Python import
remains `marco_polo_package`. The backend produces a deterministic `py3-none-any` wheel without
writing build output into the source tree. The `[tool.uv.sources]` entry is only a
local-development override: it is not embedded in the project wheel. The base worker image supplies
one exact Cozy Runtime wheel, while the package declares the oldest compatible Runtime API it needs.
Arbitrary local or editable dependencies are not bundled by publication.

`pyproject.toml`, `uv.lock`, the Runtime-derived canonical package interface, and the project wheel are
the exact publication inputs. The
package interface is sent directly to Tensorhub and is not committed here. There is no release manifest
and no author-selected worker profile. Publication is independent of worker images; the selected
worker validates the wheel against its actual Python and base-package inventory at install time. The current remote contract
admits one placement per worker, so publishing this fixture under two package repositories can prove
sequential same-pod replacement with `cozy rent revise`; it does not prove simultaneous stacking.

Cozy derives the publication package interface through this locked, read-only project command:

```sh
uv run --locked --project examples/marco-polo \
  cozy-runtime --json --dir examples/marco-polo describe
```
