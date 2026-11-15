# Managed calls and private functions

`await call(target, request, models=..., assets=...)` runs a typed request as its own job on
the machine, with its own models, memo, progress and placement. A plain `fn(...)` remains
an ordinary Python call in this process.

```python
from cozy_runtime.author import ModelChoice, call, prefetch


@app.entrypoint(private=True, defaults={"base_model": _MODEL}, memoize=True)
async def segment(ctx: Context, payload: SegmentInput, base_model: H3, out: Outputs) -> SegmentOutput: ...


@app.entrypoint
async def shot(ctx: Context, payload: ShotInput) -> ShotOutput:
    selected = {"base_model": ModelChoice(ref=lora)}
    prefetch(segment, models=selected)  # optional model preparation hint
    result = await call(segment, SegmentInput(...), models=selected)
```

- The target and request are positional. The request is a `msgspec.Struct`; even a function
  with no request fields takes an explicit empty request struct. There is no dict or
  keyword-request form.
- A generated `Operation[Request, Result]` names another package's public operation without
  importing its implementation. Registered functions of this package (including private
  functions) and Runtime builtins can be named directly. Foreign implementation functions
  are not managed-call targets.
- `models` maps declared slot names to `ModelChoice` or `None`. `None` retains the applicable
  inherited selection or default. Live injected `Model`/`Weights` objects cannot be forwarded.
- `assets` supplies granted `Files` only when the handler separately declares that input.
  Ordinary media fields belong in the request. Files retain the existing custody checks.
- `prefetch(target, models=...)` is a model-only hint: it neither submits a request nor
  transfers media, and does not promise readiness. Pass the same selectors to the later call.
- `private=True` keeps a registered function available only to its own package's code.
  Generated foreign modules never expose private operations; direct CLI requests are refused.
- Calls are reserved in source order and can run concurrently when gathered. Awaiting returns
  the typed result; a failed child raises `ChildCallError` with its original code and identity.
