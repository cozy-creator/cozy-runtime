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
  box. Options: `name`, `defaults`, `memoize`, `on_demand`, `accelerator`, `weights`, `private`,
  `components`, `preflight`.
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

## Calling another package without importing its implementation

Keep managed-only dependencies in the standard `cozy-calls` optional extra in
`pyproject.toml`, and lock them with the rest of the project:

```toml
[project.optional-dependencies]
cozy-calls = ["example-child[cozy-calls]>=1"]
```

Generate ordinary Python declarations from that locked package's canonical interface:

```sh
cozy-runtime calls example-child my_package/child_calls.py
```

Then use the generated request and non-callable operation reference explicitly:

```python
from cozy_runtime.author import call
from .child_calls import Request, generate

result = await call(generate, Request(prompt="a lighthouse"))
```

Model choices remain named arguments to `call`; returned media and `ChildCallError` retain
their ordinary managed-call behavior. Regenerate declarations after changing the locked
dependency. Ordinary implementation imports and local function calls remain ordinary Python.
Each managed requirement requests its callee’s `cozy-calls` extra too, so ordinary wheel
metadata and uv follow the managed dependency chain recursively. A child without that
extra still supplies its ordinary dependencies. The platform prepares this closure for
child execution separately from the caller.

Python environments continue to expose the machine's shared Torch layer. Managed dependency
separation removes the callee's implementation and its additional dependencies from the
caller; it does not remove that existing platform layer or imply that available Torch was imported.

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
