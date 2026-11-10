# Operation memoization identity

Package installation uses ordinary uv environments. Source archives, SDK inventories,
installation IDs and caller scripts are not computation cache keys.

A deterministic function asks for the result cache:

```python
@app.entrypoint(memoize=True)
async def convert(ctx: Context, *, source: Weights, seed: Seed | None = None) -> Converted: ...
```

Its memo key is its code (its package release, or for local code the modules it imports), its
environment, the request (files by content) and the models it runs with. Changing a caller
script or installing the same release again does not change it. A memoized function must stay
deterministic for its inputs.

A request that leaves an optional `Seed` field unset (omitted or null) asks for a new draw, so
it runs without lookup or storage. The interface keeps `"memoize": true` and lists those fields
in `"memo_requires"`.

An operation without a declared version, or whose source is unavailable, executes
normally without a completed-result shortcut. Runtime reports that reason. This is
not an installation or invocation refusal. Partial-work recovery still belongs to the
operation's TensorFS transaction protocol and must use a sound operation identity.

The shared quantization and model-preparation operations declare their own versions
and source dependencies. TensorFS's native source/ingestion memoization retains its
existing operation-specific implementation identity.

This change has not yet been qualified through the regular Creator CLI. The prior
memoization proof does not establish correctness for this new installation contract.
