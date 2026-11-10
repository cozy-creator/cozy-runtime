# Writing a package

A package is ordinary PyTorch. You write the model and its loops; the Runtime manages memory
underneath (what is on the GPU, when it moves, how an op is cut to fit) and runs it fast on
whatever card it lands on. This one is complete and runs:

```python
import logging

import msgspec
import torch
from PIL import Image
from torch import nn
from tqdm import tqdm

from cozy_runtime.author import App, Config, Loader, Model, Outputs
from cozy_runtime.media import ImageFile

app = App()
log = logging.getLogger(__name__)


class Request(msgspec.Struct):
    prompt: str
    steps: int = 28
    seed: int = 0


class Result(msgspec.Struct):
    image: ImageFile


class Graph:
    """The checkpoint's components, built from its config. The Runtime fills their weights."""

    def __init__(self, config: Config) -> None:
        width = config.section("transformer").as_int("width")
        self.text = nn.Embedding(256, width)
        self.transformer = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, width))
        self.vae = nn.Linear(width, 3)
        self.components = {"text": self.text, "transformer": self.transformer, "vae": self.vae}


class Flow(Model[Graph]):
    def load(self, loader: Loader) -> None:
        self.graph = loader.construct(Graph)

    @torch.no_grad()
    def generate(self, prompt: str, steps: int, seed: int) -> Image.Image:
        ids = torch.tensor([list(prompt.encode())], device=self.device)
        cond = self.graph.text(ids).mean(dim=1, keepdim=True)
        noise = torch.randn((1, 64 * 64, cond.shape[-1]), generator=torch.Generator().manual_seed(seed))
        x = noise.to(self.device)
        for _ in tqdm(range(steps)):  # the run's progress
            x = x - self.graph.transformer(x + cond) / steps
        pixels = (self.graph.vae(x).reshape(64, 64, 3).clamp(-1, 1) + 1) * 127.5
        return Image.fromarray(pixels.byte().cpu().numpy())


@app.entrypoint(defaults={"model": "acme/flow@1.0.0"})
def generate(payload: Request, model: Flow, out: Outputs) -> Result:
    log.warning("drawing %r", payload.prompt)  # a line in the run's log
    return Result(out.save_image(model.generate(payload.prompt, payload.steps, payload.seed)))
```

## The core

- **`App`**: `@app.entrypoint` publishes a function; `@app.tool` publishes one that is also a canvas
  box. Options: `name`, `defaults`, `memoize`, `on_demand`, `accelerator`, `weights`, `private`,
  `components`, `preflight`.
- **Handler parameters** are injected by type: a `msgspec.Struct` request, any `Model`, and, when
  you want them, `Context`, `Outputs`, `Telemetry`, `Adjustments`.
- **`Model`**: `load(loader)` builds the components. `loader.construct(factory)` takes a class or a
  function of the `Config` and fills its tensors; `loader.assets` holds the checkpoint's other
  files. `self.device` is the `torch.device` its calls run on. `unload(loader)` is optional.
- **Errors**: raise `InvalidRequest` (the request is wrong), `UnsupportedInput` (this model can't)
  or `OutputError` (the result is bad). Any other exception fails the run with its first line.

## What the Runtime does for you

- **Memory.** Each component is put on the device at its first forward, and evicted, streamed or
  restored as room changes. A package never moves weights, reads free memory or catches OOM.
- **Speed.** Each call of a component root (`transformer(...)`) is recorded as a CUDA graph and
  replayed for the same shapes, wherever it is called from.
- **Progress and logs.** A tqdm bar, a Diffusers pipeline's own included, is the run's progress.
  Each line you `print`, each `logging` record your logger lets through (Python's default level is
  WARNING) and each `warnings.warn` is a line in the run's log.
- **Cancellation.** Every component call and every tqdm step checks for a cancel.
- **Kernels.** Linear and attention modules get the best kernel for the card and the lane.

## Optional

- `@uses_components("vae")` on a Model method: a hint. The Runtime fetches and holds those
  components for the call. Where every public method has one, the model fetches only those.
- `@memoize` on a Model method: its result is stored and reused (an encoder).
- `serving = Serving(min_bits=4, loras=("transformer",))` on a Model class: its serving facts.
- `with tel.stage("decoding image"):` names the work the progress shows; `ctx.progress(i, n)` reports
  step `i` of `n` without tqdm; `tel.log(message, **fields)` logs with fields.
- `rng.noise`, `rng.counter_noise`: seeded noise that is the same on every device.
- `split(fn, x, Cut(dim))`, `retained(nbytes)`: fit a big op, keep tensors across steps.
- `cozy_runtime.image`: `ImageRequest`, `ImageResult`, `Steps`, `Guidance`, `Sizes`, `held`.
- `cozy_runtime.fields`: `Prompt`, `AspectRatio`, `Advanced`, `Label`, `Range`, `Ratios`.
- `cozy_runtime.media`: `ImageFile`, `VideoFile`, `AudioFile`, and video and audio decoding.
- `cozy_runtime.attention.Attention`: an attention call the Runtime picks the kernel for;
  `attn(q, k, v, layout=...)` passes a sparse layout. Calling
  `F.scaled_dot_product_attention` yourself is fine too.
- `cozy_runtime.lm`, `cozy_runtime.vlm`: fast decoding. `model.generate` works too.
- `cozy_runtime.accel`: speed only. `kernels`, `fusion`, `stepcache`, `moe`, `lora`, `rows`,
  `attention` (layouts, `attention_scope`, producers), `hold_dtype`, `graph_keys`, `uncaptured`,
  `placeable`, `sequence_parallel`, `spread`, `concurrently`.

Package code runs as plain Python too: with no executor, `retained` holds the tensors, `split`
runs whole and a Model method is an ordinary method.

More: `docs/package-extensions.md`.
