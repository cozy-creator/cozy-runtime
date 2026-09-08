# Authored model defaults

Owner: `/root/input_runtime`; branch: `authored-model-defaults`.
Base: `edccecff8f0d74d414a436c229742a2da88723c8` (`origin/master`).

A package author can supply an ordered default ladder on an injected Model argument:

```python
H3 = [
    {"gpu": "H100", "lane": "paul/minimax-h3@1.0.0-rc.2/fp8-fa3-adaln-pruned"},
    {"gpu": "5090", "lane": "paul/minimax-h3@1.0.0-rc.2/fp8-adaln-pruned"},
]

@invocable(defaults={"model": H3})
async def ref2va(ctx: Context, *, model: H3Model, prompt: str) -> VideoAsset:
    ...
```

The immutable package interface carries the ordered array at
`models[i].default_ladder`, associated with the existing `<function>.models.<argument>`
slot path. It does not reuse `invocable.defaults`, which describes scalar schema defaults.
One slot's rungs must share the same model and release; only the GPU selector and lane vary.
This uses the existing model/release plus GPU-to-lane binding resolver.

Resolution precedence is the invocation override, the owner's mutable Hub override, then
the function's published default. Creator and Hub resolve names before dispatch; Runtime's
executor receives resolved artifacts. Static inspection never imports package code and uses
the same declaration validator as imported description and stored-interface validation.
