# Authored model defaults

Owner: `/root/input_runtime`; branch: `authored-model-defaults`.
Initial base: `edccecff8f0d74d414a436c229742a2da88723c8`.
Rebased base: `12a05e9fb1c7db0c3f9bdc36d3fb091fde3aaf80` (`origin/master`), preserving the native integrity fixes and TensorFS 0.3.26 pin.

Runtime 0.10.0 lets a package author supply an ordered default ladder on an injected Model argument:

```python
H3 = [
    {"gpu": "H100", "lane": "paul/minimax-h3@1.0.0-rc.2/fp8-fa3-adaln-pruned"},
    {"gpu": "5090", "lane": "paul/minimax-h3@1.0.0-rc.2/fp8-adaln-pruned"},
]


@invocable(defaults={"model": H3})
async def ref2va(ctx: Context, *, model: H3Model, prompt: str) -> VideoResult: ...
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

Existing synchronous entrypoints can use the same declaration directly:

```python
@app.entrypoint(defaults={"model": H3})
def ref2va(payload: Request, model: H3Model) -> VideoResult: ...
```

`app.job` also accepts the same keyword. Declare defaults on only one registration layer;
putting nonempty defaults on both `invocable` and its App registration refuses. The package
can share one literal ladder constant between its functions. Model references are metadata,
never executor-side lookups. A ladder admits 1 through 32 GPU selectors, with an optional
catch-all `*` only last. Each target explicitly names a published release and portable lane.

An explicit `gpus` count names the execution group, for example
`{"gpu": "H100", "gpus": 2, "lane": "paul/minimax-h3@1.0.0-rc.2/fp8-pruned"}`.
Counts are positive integers; counts above 1 must be declared by the model's
`sequence_parallel.degrees`. A four-device pool can reserve a two-device H3 group
and run a singleton Qwen child separately. Extra devices do not enlarge a selected
group. Rental purchase matches an explicit offered count; independent workflow
children require the maximum group count, while slots within a callable share one
group. Older published declarations and captures without `gpus` preserve their
unspecified-count behavior until republished. Explicit-count captures require
wire 63; ordinary uncounted calls retain wire 62 compatibility.
