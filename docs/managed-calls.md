# Managed calls and private functions

`await call(fn, ...)` runs a registered function as its own job on the machine: its own models,
memo, progress and GPU placement. A plain `fn(...)` is an ordinary Python call in this process.

```python
from cozy_runtime.author import call, prefetch


@app.entrypoint(private=True, defaults={"base_model": _MODEL}, memoize=True)
async def segment(ctx: Context, *, payload: SegmentInput, base_model: H3, out: Outputs) -> SegmentOutput: ...


@app.entrypoint
async def shot(ctx: Context, payload: ShotInput) -> ShotOutput:
    prefetch(segment)  # optional: prepare its models while this work runs
    result = await call(segment, payload=SegmentInput(...), base_model=ModelChoice(ref=lora))
```

- `call` takes `fn`'s own parameters as a direct call names them, less what is injected (Context,
  services): its request struct or fields, its `Files`, and a `ModelChoice` (or None, its
  default) per model argument. It returns `fn`'s result type.
- `fn` is a registered function of this package, or a public function of a package this one
  depends on (installed beside it). The Runtime binds it from its own signature.
- `private=True` registers a function only its own package's code calls. The package interface
  keeps it with `internal: true`; `cozy run <package>/<private function>` is refused. TensorD
  enforces this from admitted capture and child journal identities.
- Calls are reserved in the order they are made and run concurrently when gathered.
