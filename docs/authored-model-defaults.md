# Authored model defaults and precision tolerances

A Model argument may name the release a run takes when it names none for that slot:

```python
@app.entrypoint(defaults={"model": "qwen3.5-9b@0.2.0"})
def caption(payload: Request, model: Qwen) -> Caption: ...


@invocable(defaults={"model": "sdxl@1.0.0"})
async def generate(ctx: Context, *, model: SDXL, prompt: str) -> Image: ...
```

- A default is `[org/]model@release`: an explicit release, never a digest, and no lane. The
  machine picks the lane from its GPU and the request's `bits` (tracker design
  `precision-tiers.md`). A pin for one run is `model.<slot>=…`, optionally with `/lane`.
- A default with no org names a first-party model, the root namespace's (`sdxl@1.0.0`), whoever
  publishes the package. Another account's model names its org (`you/my-model@1.0.0`).
- Declare defaults on one registration layer. Defaults on both `invocable` and its App
  registration are refused. A script's `[tool.cozy.models]` takes the same strings.

The Model class declares how far the lane rules may lower its precision:

```python
class Qwen(Model, encoded_leaves="accept", min_bits=4, phase="decode"): ...
```

- `min_bits`: 16, 8 (the default) or 4. The lowest weight precision the rules pick by
  themselves. At 4 they take `nvfp4` for a decode slot, or when 8-bit weights exceed the GPU
  budget.
- `phase`: `"compute"` (the default: diffusion, prefill) or `"decode"` (an LM's token loop,
  bound by weight bytes).
- Subclasses inherit both. Any other value is refused.

Each slot in the package interface carries `default`, `min_bits` (omitted when 8) and
`phase` (omitted when `"compute"`). The import and static readers produce identical bytes.
The static reader never imports package code, and reads class keywords and their inheritance
from source.
