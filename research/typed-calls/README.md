# Lightweight managed calls (xs-198)

Owner: `/root/sdk_typed_calls`. Branch: `codex/xs-198-typed-calls-20261115`.
Created from fetched `origin/master` `af2a4753cb70598b75405942901e4f8f3b24eda6`, then
fast-forwarded to the active integration baseline `origin/rc/0.26.0`
`665d06c2bdfcb48cbec840d19b920fc8e354a721`. This document records the proposed interface;
implementation and process qualification are pending.

## Developer spelling

```python
from cozy_runtime.author import call
from orbit._calls.minimax_h3 import PackageShot, render_shot

result = await call(render_shot, PackageShot(...), model=choice)
```

`render_shot` is a generated, non-callable `Operation[PackageShot, ShotResult]`. A positional
request supplies the generated type, and named model choices keep the existing per-slot
selection behavior. The operation carries the real Python module/export identity from its
canonical interface. Ordinary `call(fn, **arguments)` remains supported; `fn(...)` remains
an ordinary local function call. No import rewriting or automatic function proxy is involved.

## Existing implementation that must change

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

Changing Orbit's import alone does **not** remove Torch from its installed closure. One
possible bounded installation change is the standard `[dependency-groups].cozy-calls` group,
locked in the existing `uv.lock`: default sync installs the caller, and a separate prepared
environment installs that group for managed children. TensorD must retain and select that
child environment from the same held generation. This needs explicit review of local and
published installs, bytecode preparation, environment lifetime, memo identity and child
selection before it can be claimed as a useful closure reduction.

## Qualification

Use a real CPU caller and child through package build/install, CLI, executor processes,
sockets and storage. Verify media in both directions, model override semantics, typed child
failure, private operation refusal and unchanged local calls. Inspect both installed
closures, and statically describe Orbit/H3 without importing callee code. No GPU, weights,
native Rust package or new generic broker is required.
