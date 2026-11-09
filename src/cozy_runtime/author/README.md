# Writing a package

A package is ordinary PyTorch. You write the model and its loops; the Runtime manages memory
underneath (what is on the GPU, when it moves, how an op is cut to fit) and runs it fast on
whatever card it lands on. There is no loop API to learn.

```python
import torch
from cozy_runtime import rng
from cozy_runtime.author import App, Context, Model, Outputs, Telemetry, uses_components

app = App()


class Flow(Model):
    def load(self, loader):
        self.graph = loader.construct(FlowGraph, factory=FlowGraph)  # transformer, vae, ...

    @uses_components("transformer")
    def denoise(self, cond, *, steps, seed, ctx):
        transformer = self.graph.components["transformer"]
        device = torch.device(str(self.device))
        latents = rng.counter_noise((1, 16, 128, 128), seed, 0, dtype=torch.bfloat16, device=device)
        sigmas = torch.linspace(1.0, 0.0, steps + 1).tolist()
        with torch.no_grad():
            for i in range(steps):
                velocity = transformer(latents, sigmas[i], cond)
                latents = latents + (sigmas[i + 1] - sigmas[i]) * velocity
                ctx.progress(i, steps)
        return latents

    @uses_components("vae")
    def decode(self, latents):
        return self.graph.components["vae"].decode(latents)


@app.entrypoint
def generate(ctx: Context, payload: Request, model: Flow, out: Outputs, tel: Telemetry) -> Result:
    cond = model.encode(payload.prompt)
    with tel.stage("generating image"):
        latents = model.denoise(cond, steps=payload.steps, seed=payload.seed, ctx=ctx)
    with tel.stage("decoding image"):
        return Result(out.save_image(model.decode(latents), format="png"))
```

## What the Runtime does for you

- **Memory.** `@uses_components("transformer")` names the components a method runs. The Runtime
  has them on the device for the call and evicts, streams or restores them as room changes. A
  package never moves weights, reads free memory or catches an out-of-memory error.
- **Speed.** Each call of a component (`transformer(...)`) is recorded as a CUDA graph and
  replayed on later calls with the same shapes. Nothing to declare. A component whose forward
  depends on host state says so: `@graph_keys(unet=("attr",))` for a root attribute that steers
  it, `@uncaptured("name")` to always run as written.
- **Cancellation.** Every component call checks for a cancel, so a loop stops within a step.
  Long host-only work can call `ctx.raise_if_cancelled()`.
- **Kernels.** Linear and attention modules are served by the best kernel for the card and the
  weights' lane. Attention is dense by default.

## What you may call

- `ctx.progress(i, n)`: optional. Step `i` (0-based) of `n` is done; the run shows `i+1/n` and the
  step time under the enclosing `tel.stage`. Without it the run still shows model calls.
- `rng.counter_noise(shape, seed, step, *, stream=0, dtype, device)`: noise that depends only on
  its arguments, the same on every device. `rng.noise(shape, seed, ...)` draws a request's noise once.
- `split(fn, x, Cut(dim, ...))`: `fn(x)`, whole or in the fewest windows that fit (a VAE decode).
- `retained(nbytes)`: a store for tensors a later step reads again (a prefix's K/V).
  `state = store.put(k, v)`, `k, v = state.get()`; close the store (or use `with`) after the loop.
- `stepcache.running(policy, n, device)` with `run.step(i)`: step caching for a declared block stack.
- `attention_scope(layout)`: only for a model that owns a sparse attention layout.

All of these work in plain Python too: with no executor, `retained` holds the tensors as they
are, `split` runs whole and `ctx.progress` is the only thing that needs a `Context`.

More: `docs/package-extensions.md` (fused glue, step caching, attention producers, `split`).
