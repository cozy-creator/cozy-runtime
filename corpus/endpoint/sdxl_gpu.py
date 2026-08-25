"""The real GPU endpoint cr-007 serves: one SDXL UNet, three entrypoints, real bytes.

`corpus/sdxl_real.py` is the MODEL (upstream diffusers, upstream config, 1,680 real
destinations); this file is the ENDPOINT around it — the request schema, the handlers, and
the output transaction the supervisor drives an attempt through. It is deliberately in
`corpus/` and not in `examples/`: the examples are weightless and run anywhere, this one
needs an 8 GiB card and a real CAS.

Three entrypoints, each earning its place in the verification:

  denoise   the ordinary attempt — N real denoising steps through the guarded component
            root, a real PNG out of the final latent, cooperative cancellation between
            steps, and `ctx.boot_warmup` reported in the result so the warm pass is
            OBSERVABLE rather than asserted.
  stubborn  a handler that never asks whether it was cancelled. Real GPU work, no
            cancellation check: the cooperative signal cannot reach it, so the only honest
            way to stop it is to kill the executor. That is the cancellation red arm.
  restyle   a required VIDEO input. The boot warm pass's payload synthesizer can make
            strings, numbers and images at neutral defaults and cannot make a video, so
            this entrypoint must land `skipped` with a typed reason — never `warmed`.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated, Any

import msgspec
from corpus.sdxl_real import SdxlUnetModel

from cozy_runtime.author import (
    Context,
    ImageAsset,
    ImageFrame,
    Outputs,
    Telemetry,
    VideoAsset,
)
from cozy_runtime.author._app import App

# IMPORT TRIPWIRE (cr-007 red arm, deliberate module-scope side effect). Every process
# that imports this endpoint records its own pid here. The supervisor must never appear:
# it is the long-lived no-CUDA process, and endpoint module scope is exactly where a real
# endpoint would touch torch. `scripts/worker-live.py tripwire` reads these files.
_IMPORTS = Path(__file__).resolve().parent / "imports"
_IMPORTS.mkdir(exist_ok=True)
(_IMPORTS / f"{os.getpid()}.pid").write_text(f"{os.getpid()}\n")

app = App()

#: The latent side the request may ask for. 64 is SDXL's 512 px bucket.
LATENT = (48, 64, 80)


class DenoiseInput(msgspec.Struct, forbid_unknown_fields=True):
    steps: Annotated[int, msgspec.Meta(ge=1, le=64)] = 2
    latent: Annotated[int, msgspec.Meta(ge=32, le=96)] = 64
    seed: int = 1005


class DenoiseOutput(msgspec.Struct):
    image: ImageAsset
    steps: int
    mean: float
    std: float
    digest: str
    warm: bool


class StubbornInput(msgspec.Struct, forbid_unknown_fields=True):
    """`stubborn`'s own bound. It exists to run LONG on purpose, so its ceiling is its
    own declaration rather than a widened bound on the ordinary request schema."""

    steps: Annotated[int, msgspec.Meta(ge=1, le=4096)] = 256
    latent: Annotated[int, msgspec.Meta(ge=32, le=96)] = 80
    seed: int = 3


class PairDetail(msgspec.Struct):
    """A NESTED typed value in the result. It exists to prove two things at once: the
    result is serialized exactly once (so `scores` arrives as numbers, not as a string),
    and an output nested under it gets the stable id `detail.thumb`."""

    thumb: ImageAsset
    note: str
    scores: list[float]


class PairOutput(msgspec.Struct):
    preview: ImageAsset
    detail: PairDetail


class BloatOutput(msgspec.Struct):
    blob: str


class RestyleInput(msgspec.Struct, forbid_unknown_fields=True):
    clip: VideoAsset
    strength: float = 0.5


class RestyleOutput(msgspec.Struct):
    frames: int


def _inputs(torch: Any, device: Any, side: int, seed: int) -> tuple[Any, ...]:
    """The SDXL UNet's declared call shape at one latent size. Pure tensor construction —
    no component is touched, so this is module code, not model code (§1.1)."""
    gen = torch.Generator(device=device).manual_seed(seed)
    return (
        torch.randn(1, 4, side, side, generator=gen, device=device, dtype=torch.float16),
        torch.tensor([981], device=device),
        torch.randn(1, 77, 2048, generator=gen, device=device, dtype=torch.float16),
        torch.randn(1, 1280, generator=gen, device=device, dtype=torch.float16),
        torch.tensor(
            [[side * 8, side * 8, 0, 0, side * 8, side * 8]],
            device=device,
            dtype=torch.float16,
        ),
    )


def _preview(torch: Any, latents: Any) -> ImageFrame:
    """The latent's first three channels as RGB. A real image out of real computation —
    there is no VAE in this binding, and inventing one would be a different endpoint."""
    field = latents[0, :3].float()
    low = field.amin()
    span = (field.amax() - low).clamp(min=1e-6)
    pixels = ((field - low) / span * 255).to(torch.uint8).permute(1, 2, 0).contiguous()
    height, width = int(pixels.shape[0]), int(pixels.shape[1])
    return ImageFrame(width, height, bytes(pixels.cpu().numpy().tobytes()))


@app.entrypoint
def denoise(
    ctx: Context, payload: DenoiseInput, model: SdxlUnetModel, out: Outputs, tel: Telemetry
) -> DenoiseOutput:
    import hashlib

    import torch

    view = model.for_request(ctx, seed=payload.seed)
    device = torch.device("cuda", 0)
    latents, timestep, prompt, embeds, ids = _inputs(torch, device, payload.latent, view._seed)
    # A warm case pays the first-call tax on the smallest legal shape it can: the body
    # cheapens off `ctx.boot_warmup` rather than inventing an out-of-contract schedule.
    steps = 1 if ctx.boot_warmup else payload.steps
    tel.log("starting the denoise loop", latent=payload.latent, steps=steps, seed=view._seed)
    on_step = tel.step_callback(steps, stage="denoise")
    with tel.stage("denoise"):
        for step in range(steps):
            ctx.raise_if_cancelled()  # THE cooperative-cancel spelling
            latents = model.denoise(latents, timestep, prompt, embeds, ids)
            on_step(step)
    stats = latents.float()
    tel.metric("latent_std", round(float(stats.std()), 6))
    body = latents.reshape(-1).view(torch.uint8).cpu().numpy().tobytes()
    return DenoiseOutput(
        image=out.save_image(_preview(torch, latents), format="png"),
        steps=steps,
        mean=round(float(stats.mean()), 6),
        std=round(float(stats.std()), 6),
        digest=hashlib.sha256(body).hexdigest(),
        warm=ctx.boot_warmup,
    )


@app.entrypoint
def stubborn(payload: StubbornInput, model: SdxlUnetModel, ctx: Context) -> DenoiseOutput:
    """Real GPU work with NO cancellation check anywhere. The cooperative signal lands and
    nothing reads it — which is exactly the case where `CANCELED` may only be journaled
    after the executor is provably dead."""
    import hashlib

    import torch

    view = model.for_request(ctx, seed=payload.seed)
    device = torch.device("cuda", 0)
    latents, timestep, prompt, embeds, ids = _inputs(torch, device, payload.latent, view._seed)
    for _ in range(payload.steps):
        latents = model.denoise(latents, timestep, prompt, embeds, ids)
    stats = latents.float()
    return DenoiseOutput(
        image=ImageAsset("attempt:none/image/0000"),
        steps=payload.steps,
        mean=round(float(stats.mean()), 6),
        std=round(float(stats.std()), 6),
        digest=hashlib.sha256(b"stubborn").hexdigest(),
        warm=False,
    )


@app.entrypoint
def pair(ctx: Context, payload: DenoiseInput, model: SdxlUnetModel, out: Outputs) -> PairOutput:
    """TWO outputs, one of them nested. Their ids are their field paths (`preview`,
    `detail.thumb`), never their position in a list."""
    import torch

    view = model.for_request(ctx, seed=payload.seed)
    device = torch.device("cuda", 0)
    latents, timestep, prompt, embeds, ids = _inputs(torch, device, payload.latent, view._seed)
    latents = model.denoise(latents, timestep, prompt, embeds, ids)
    frame = _preview(torch, latents)
    small = ImageFrame(8, 8, frame.rgb[: 8 * 8 * 3])
    stats = latents.float()
    return PairOutput(
        preview=out.save_image(frame, format="png"),
        detail=PairDetail(
            thumb=out.save_image(small, format="png"),
            note="latent preview and its thumbnail",
            scores=[round(float(stats.mean()), 6), round(float(stats.std()), 6)],
        ),
    )


@app.entrypoint
def bloat(payload: DenoiseInput) -> BloatOutput:
    """A typed result far past the inline door. The runtime must REFUSE it typed rather
    than mint a blob receipt for bytes nobody wrote."""
    return BloatOutput(blob="x" * (5 * 1024 * 1024))


@app.entrypoint
def restyle(payload: RestyleInput) -> RestyleOutput:
    """A required video input. Warm synthesis cannot produce one and must say so."""
    return RestyleOutput(frames=int(payload.strength * 100))
