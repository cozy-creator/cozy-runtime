# Operation memoization identity

Package installation uses ordinary uv environments. Source archives, SDK inventories,
installation IDs and caller scripts are not computation cache keys.

For a custom deterministic operation, declare the implementation contract:

```python
@invocable(
    memoize=True,
    memo_version="convert/1",
    memo_dependencies=(convert_tensor, "my_tools.tensor_layout"),
)
async def convert(ctx, *, source): ...
```

The worker discovers the operation's own function and the declared helpers in the
installed environment. Its memo key includes their normalized Python source and the
operation version, together with the invocation's inputs and numerical conditions.
Changing a caller script or installing the same operation again does not change it.
Changing a function or declared helper does. Module dependencies include only the
named Python modules, never an entire distribution or an ambient client SDK.

The author must list all helpers whose implementation affects the result, and bump
`memo_version` when external behavior changes outside those sources, including native
algorithms, configuration tables or behavior-affecting global state. Arguments should
carry varying values; a memoized function must remain deterministic for its inputs.
The system does not infer arbitrary dynamic Python dependencies.

An operation without a declared version, or whose source is unavailable, executes
normally without a completed-result shortcut. Runtime reports that reason. This is
not an installation or invocation refusal. Partial-work recovery still belongs to the
operation's TensorFS transaction protocol and must use a sound operation identity.

The shared quantization and model-preparation operations declare their own versions
and source dependencies. TensorFS's native source/ingestion memoization retains its
existing operation-specific implementation identity.

This change has not yet been qualified through the regular Creator CLI. The prior
memoization proof does not establish correctness for this new installation contract.
