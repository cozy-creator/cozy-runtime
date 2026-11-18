# Writing a package

A package is ordinary PyTorch. You write the model and its loops; the Runtime manages memory
underneath (what is on the GPU, when it moves, how an op is cut to fit) and runs it fast on
whatever card it lands on.

```python
import msgspec
import torch
from cozy_runtime import rng
from cozy_runtime.author import App, Config, Context, Loader, Model, Seed, Telemetry
from cozy_runtime.media import ImageFile

app = App()


class Request(msgspec.Struct):
    prompt: str
    steps: int = 28
    seed: Seed = 0


class Flow(Model[FlowGraph]):
    def load(self, loader: Loader) -> None:
        self.graph = loader.construct(FlowGraph)  # FlowGraph(config): transformer, vae, ...

    def denoise(self, ctx: Context, cond: torch.Tensor, steps: int, seed: int) -> torch.Tensor:
        device = torch.device(str(self.device))
        x = rng.counter_noise((1, 16, 128, 128), seed, 0, dtype=torch.bfloat16, device=device)
        sigmas = torch.linspace(1.0, 0.0, steps + 1).tolist()
        for i in range(steps):
            x = x + (sigmas[i + 1] - sigmas[i]) * self.graph.transformer(x, sigmas[i], cond)
            ctx.progress(i, steps)
        return x


@app.entrypoint(defaults={"model": "acme/flow@1.0.0"})
def generate(ctx: Context, payload: Request, model: Flow, tel: Telemetry) -> Result:
    with tel.stage("generating image"):
        latents = model.denoise(ctx, model.encode(payload.prompt), payload.steps, payload.seed)
    return Result(ImageFile.from_image(model.decode(latents)))
```

## The core

- **`App`**: `@app.entrypoint` publishes a function; `@app.tool` publishes one that is also a canvas
  box. Options: `name`, `defaults`, `memoize`, `on_demand`, `needs`, `accelerator`, `weights`,
  `private`, `components`, `preflight`.
  A constructed `Model` parameter selects the GPU by default; other handlers use CPU.
  Set `accelerator=True` for custom GPU work with only raw `Weights` or ordinary inputs,
  or `accelerator=False` to keep a model handler on CPU. Explicit settings always win.
- **Handler parameters** are injected by type: `Context`, a `msgspec.Struct` request, any `Model`,
  and the services `Outputs`, `Telemetry`, `Adjustments`.
- **`Model`**: `load(loader)` builds the components. `loader.construct(factory)` takes a class or a
  function of the `Config` and fills its tensors; `loader.assets` holds the checkpoint's other
  files. A single file uploaded as it is (an adapter) has its safetensors metadata as its `Config`.
  `unload(loader)` is optional.
- **Errors**: raise `InvalidRequest` (the request is wrong), `UnsupportedInput` (this model can't)
  or `OutputError` (the result is bad). Any other exception fails the run with its first line.

## Calling another package

Depend on it as usual and lock it, then generate typed references from its locked interface:

```sh
cozy-runtime calls example-child my_package/child_calls.py
```

```python
from cozy_runtime.author import call
from .child_calls import Request, generate

result = await call(generate, Request(prompt="a lighthouse"))
```

`generate` is an `Operation[Request, Result]`: the callee's module, export, request and result
types and model slots. Calling it never imports the callee. Model choices go in
`models={"slot": ModelChoice(...)}`, a `Files` input in `assets=`. Describe checks each
reference's request, result and model slots against the locked callee (its docs, labels and
defaults don't matter) and refuses a stale one (`stale_call_reference`): regenerate it.

## What the Runtime does for you

- **Memory.** Each component is put on the device at its first forward, and evicted, streamed or
  restored as room changes. A package never moves weights, reads free memory or catches OOM.
- **Speed.** Each call of a component root (`transformer(...)`) is recorded as a CUDA graph and
  replayed for the same shapes, wherever it is called from.
- **Cancellation.** Every component call checks for a cancel; `ctx.progress(i, n)` does too.
- **Kernels.** Linear and attention modules get the best kernel for the card and the lane.
- **Final images.** Return `ImageFile.from_image(pil_image_or_frame)` in an `ImageFile`
  result field. The SDK snapshots host pixels, then encodes and stores them through the
  ordinary output path. It defaults to lossless WebP, using PNG for a side over 16383 pixels;
  `format="png"` or `format="webp"` keeps an explicit choice. Both keep alpha. Existing
  result schemas, metadata and output bounds still apply. The same value can go directly
  into a child request or `out.publish("image", image)`: the SDK registers it once and
  encodes it when those bytes are needed. Labeled copies share that file, and the original
  can read its encoded bytes after the child or publication consumed it. `image.probe()`
  reads captured geometry even before encoding. `Outputs.save_image` has been removed;
  `Outputs` handles explicit publication, encoded artifacts and bounded audio/video streams.

## Which requests need a model

A function with several model arguments rarely uses all of them for one request. Say which
requests need one with a rule, and the machine fetches it before the function starts only for
those, and never for the rest:

```python
@app.entrypoint(defaults=DEFAULTS, needs={
    "upsampler": "request.prompt_upsampling",
    "edit_model": "size(request.prompt.references) > 0",
})
def generate(ctx: Context, payload: Request, model: Flow, upsampler: Upsampler, edit_model: Edit) -> Result: ...
```

- A rule is one [CEL](https://cel.dev) boolean expression over `request`: the request as JSON,
  each field it leaves out at its default. `&&`, `||`, `!`, comparisons, `size(x)`,
  `x != null`, and `list.exists(e, ...)` / `list.all(e, ...)`. `cozy` checks it against the
  request when the package is read: a field that isn't there refuses the package.
- A model argument with a rule is built when the function first reads it. One with no rule is
  always needed.
- A function that loads no model and calls others names a function it calls:
  `needs={"edit_segment": "request.plate != null"}`.
- Err toward true: a rule that is true for a request that doesn't read the model only fetches
  too much. One that is false for a request that does read it makes that request wait for the
  download when it reads it, and the run warns.
- `ctx.need("edit_model")`: what only the run itself learns. The fetch starts at once, ahead of
  every other download, while the function goes on.

## What a request costs

Say what a request's cost depends on, and how, beside `needs=`. The planner computes it with the
models a request needs, from the request alone, on every machine it might send it to:

```python
@app.tool(label="Shot", named="shot", needs=_SHOT_NEEDS, estimate={
    "size": {"seconds": "sum(request.beats.map(b, double(b.duration)))",
             "steps": "request.turbo ? 8.0 : 40.0"},
    "ms": "measured(machine).warm_ms * pow(size.seconds / measured(machine).request.seconds, 1.05)"
          " * size.steps / 8.0",
})
```

- `size`: each variable a request's cost depends on, named freely, one CEL number over `request`.
  A run's record keeps them as its size; `cozy package benchmarks` writes them as its rows'
  `request`. One that gives no number for a request is left out, and the run says so once.
- `ms`, `cold_ms`, `peak_device_bytes`, `peak_host_bytes`: optional CEL numbers over `request`,
  `size` and `machine` (`gpu`, `gpus`, `memory_bytes`, `lane`, `cpu`, `cores`, `threads`). They
  call three helpers:
  - `measured(machine)`: the package's own `benchmarks.json` row for that machine: `warm_ms`,
    `cold_ms`, `peak_device_bytes`, `peak_host_bytes`, `budget_bytes`, `request` (its size), `gpu`,
    `lane`, `scale`. The row of that card nearest the request; for a card with none, another
    card's, its times scaled by how the two compare; on a card smaller than the row's peak, the
    row measured under the memory budget nearest at or below it; with no GPU, the CPU row.
    `measured(machine, "generate")` reads another function's rows.
  - `pow(x, y)` and `sum(list)`.
- No `ms`: the platform's default, the nearest row scaled. No `size`: the request's
  `megapixels` (or `width` by `height`), `frames`, `steps`, `tokens` and every `duration` summed.
- Every number in `size`, `machine` and a row is a double: `request.steps` is a whole number, so
  write `double(request.steps)` beside them. `cozy` checks each expression when it reads the
  package: a field that isn't there, a mixed whole number, or one that gives no number refuses it.

## Optional

- `@uses_components("vae")` on a Model method: a hint. The Runtime fetches and holds those
  components for the call. Where every public method has one, the model fetches only those.
- `@memoize` on a Model method: its result is stored and reused (an encoder).
- `serving = Serving(min_bits=4, loras=("transformer",))` on a Model class: its serving facts.
- `ctx.progress(i, n)`: step `i` of `n` is done.
- `rng.noise`, `rng.counter_noise`, `rng.generator`: seeded noise, the same on every device.
- `split(fn, x, Cut(dim))`, `retained(nbytes)`: fit a big op, keep tensors across steps.
- `cozy_runtime.image`: `ImageRequest`, `ImageResult`, `Steps`, `Guidance`, `Sizes`, `held`;
  `Control[Literal[...]]`, a control map of the kinds the function has adapters for (others are
  dropped with a row).
- `cozy_runtime.fields`: `Prompt`, `AspectRatio`, `Advanced`, `Label`, `Range`, `Ratios`.
- `cozy_runtime.media`: `ImageFile`, `VideoFile`, `AudioFile`, and video and audio decoding.
- `cozy_runtime.attention.Attention`: an attention call the Runtime picks the kernel for;
  `attn(q, k, v, layout=...)` passes a sparse layout. Calling
  `F.scaled_dot_product_attention` yourself is fine too.
- `cozy_runtime.lm`, `cozy_runtime.vlm`: fast decoding. `model.generate` works too.
- `cozy_runtime.accel`: speed only. `kernels`, `fusion`, `stepcache`, `moe`, `lora`, `rows`,
  `attention` (layouts, `attention_scope`, producers), `hold_dtype`, `graph_keys`, `uncaptured`,
  `placeable`, `sequence_parallel`, `spread`, `concurrently`.
- On several GPUs (`sequence_parallel`) every GPU runs the function, as torchrun runs a script, and
  a `_cp_plan` model splits its work inside its own calls: write it as for one GPU.

Package code runs as plain Python too: with no executor, `retained` holds the tensors, `split`
runs whole and a Model method is an ordinary method.

More: `docs/package-extensions.md`.

The request must be a positional `msgspec.Struct`; keyword request fields and dicts are not
accepted. Same-package registered/private functions and Runtime builtins use the same form.
Handlers with separately injected `Files` take them through `assets=...`. Other packages
require generated operation references, even when their implementation is installed locally.
