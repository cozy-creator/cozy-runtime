# Lightweight managed calls (xs-198)

Owner: `/root/sdk_typed_calls`. Branch: `codex/xs-198-typed-calls-20261115`.
Created from fetched `origin/master` `af2a4753cb70598b75405942901e4f8f3b24eda6`, then
fast-forwarded to the active integration baseline `origin/rc/0.26.0`
`665d06c2bdfcb48cbec840d19b920fc8e354a721`. This document records the implemented interface. Source, type, compile and root-only install
checks pass; the default-home public process qualification remains pending.

## Developer spelling

```python
from cozy_runtime.author import call
from orbit._calls.minimax_h3 import PackageShot, render_shot

result = await call(render_shot, PackageShot(...))
```

`render_shot` is a generated, non-callable `Operation[PackageShot, ShotResult]`. A positional
request supplies the generated type, and named model choices keep the existing per-slot
selection behavior. The operation carries the real Python module/export identity from its
canonical interface. Same-package/private function targets use the same positional request form; `fn(...)` remains
an ordinary local function call. No import rewriting or automatic function proxy is involved.

## Existing implementation and changes

- Static discovery follows `call(fn, ...)` through local helper bodies and resolves the
  callee's source using the caller's `uv.lock`. It emits existing `calls` rows naming the
  distribution and registered function. Private foreign operations are excluded.
- TensorD `invocables` admits the current package's functions and other installed Apps'
  public functions by their real `(module, export)`. Runtime `CallBindings.__getitem__`
  currently imports a foreign implementation and analyzes it at the first call.
- `catalog::Callee` records the interface and package identity but no execution environment.
  `HeldGeneration::application()` switches App metadata while retaining the caller's Python
  environment. Both published and local installation use the same dependency closure.

Generated references can avoid the implementation import by supplying generated request and
result types after checking the admitted call inventory without resolving its lazy binding.
The existing TensorD child admission, request validation, media custody, model choices,
progress, cancellation and `ChildCallError` remain authoritative.

## Distribution and installation requirement

Generate source from the same canonical installed/published interface used by execution;
do not maintain a second schema. Generated declarations must record their distribution and
declaration identity, and source-only describe must resolve them against the existing lock.
The generator must preserve nested structs, defaults, tagged unions, media and model choices.

Changing Orbit's import alone does **not** remove Torch from its installed closure. The bounded installation change is the standard `[project.optional-dependencies].cozy-calls` extra, locked in the existing `uv.lock`: default sync installs the caller, and a separate prepared
environment selects that extra for managed children. TensorD reuses its existing builder and retains the child environment inside the held caller
environment, selecting its Python for that callee. Local cached callee records use the
canonical shared environment path, so they do not retain an evictable generation symlink.
Published carried wheels are filtered by the caller closure before installation. Root-only
Orbit sync installed seven packages: Runtime, TensorFS, msgspec, numpy, pillow, packaging
and av; no H3, Torch, Diffusers or Transformers.

## Qualification

Use a real CPU caller and child through package build/install, CLI, executor processes,
sockets and storage. Verify media in both directions, model override semantics, typed child
failure, private operation refusal and unchanged local calls. Inspect both installed
closures, and statically describe Orbit/H3 without importing callee code. No GPU, model-weight download,
native Rust package or new generic broker is required.


## Current evidence

- Generated H3 declarations from the exact wheel in Orbit's `uv.lock`; imported Orbit in a
  fresh environment with neither H3 nor Torch installed or imported. Typed Prompt defaults
  remain typed prompts. The generated operation is not callable.
- Source-only Orbit describe now reaches `render_shot` through explicit local branches. Its
  old `METHODS[ENDS]` dispatch table hid its child calls from static discovery.
- Runtime strict mypy passed for the six changed modules, and errors-only Ruff passed.
- Creator `cargo check -p tensord --lib` passed (existing warnings).
- `cozy-creator/e2e/cli/typed-calls` holds the real CPU caller/child fixture and public rerun
  commands. No daemon, GPU or rental was started for this work.
- Public default-home CLI/process/media/model/error qualification is still required before
  merge. The coordinator owns candidate installation and the cl-701 describe correction.

## Package cohort integration gate

Orbit requires Runtime `>=0.47.0.dev0,<0.48` for `Operation`. Published H3 `0.13.0rc2`
currently requires Runtime `<0.47`, so the final combined `uv.lock` cannot resolve until
H3's Runtime floor is migrated in the coordinated package cohort. Optional extras share
one uv resolution even though their installed environments are separate. The earlier
root-only sync receipt used the existing 0.46 lock; it proves the group closure distinction,
not a finished 0.47 package installation. The CPU fixture pins both sources to the candidate
SDK for qualification. No dependency cap was relaxed to hide this integration gate.

## Recursive closure

Managed dependencies use the wheel-published `cozy-calls` optional extra, with each edge
requesting `child[cozy-calls]`. An actual three-package uv resolution exported no children
for the default root and middle, legacy leaf and its ordinary dependency when the extra
was selected. The built middle wheel carries `Provides-Extra: cozy-calls` and the onward
`Requires-Dist` edge. A legacy leaf without the extra warns and retains its ordinary
dependencies. There is no second dependency manifest. The CPU fixture now includes
A → B → legacy C → private self call.

## Shared Torch layer

cl-715 deliberately exposes the machine's full Torch layer to every Python environment.
The actual local Torch 2.15.0+cu132 layer includes SymPy, so a SymPy-absence assertion would
misdiagnose correct managed dependency separation. The CPU fixture uses xxhash, verified
absent from that layer, for a real leaf computation and checks its absence in the caller.
It separately reports Torch/SymPy availability and whether Torch was imported. The earlier
minimal-venv UV proof remains a dependency-resolution result, not a claim that production
Python execution is Torch-free. The shared platform layer remains unchanged; Python-free
native execution belongs to xs-153.
