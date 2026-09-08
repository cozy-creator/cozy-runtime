"""The real GPU package cr-007 serves: one SDXL UNet, three entrypoints, real bytes.

`corpus/sdxl_real.py` is the MODEL (upstream diffusers, upstream config, 1,680 real
destinations); this file is the PACKAGE around it — the request schema, the handlers, and
the output transaction the worker drives an attempt through. It is deliberately in
`corpus/` and not in `examples/`: the examples are weightless and run anywhere, this one
needs an 8 GiB card and a real CAS.

Three entrypoints, each earning its place in the verification:

  denoise   the ordinary attempt — N real denoising steps through the guarded component
            root, a real PNG out of the final latent, cooperative cancellation between
            steps.
  stubborn  a handler that never asks whether it was cancelled. Real GPU work, no
            cancellation check: the cooperative signal cannot reach it, so the only honest
            way to stop it is to kill the executor. That is the cancellation red arm.
  restyle   a required VIDEO input: the input-asset plane's video arm.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated, Any

import msgspec
from corpus.sdxl_real import SdxlUnetModel

from cozy_runtime.author import (
    App,
    AssetBound,
    Context,
    ImageAsset,
    ImageFrame,
    Outputs,
    Telemetry,
    VideoAsset,
)

# IMPORT TRIPWIRE (cr-007 red arm, deliberate module-scope side effect). Every process
# that imports this package records its own pid here. The worker must never appear:
# it is the long-lived no-CUDA process, and package module scope is exactly where a real
# package would touch torch. `scripts/worker-live.py tripwire` reads these files.
_IMPORTS = Path(__file__).resolve().parent / "imports"
_IMPORTS.mkdir(exist_ok=True)
(_IMPORTS / f"{os.getpid()}.pid").write_text(f"{os.getpid()}\n")

app = App()

#: The latent side the request may ask for. 64 is SDXL's 512 px bucket.
LATENT = (48, 64, 80)

#: Where `leak_mib`'s planted allocation lives. A module global, because that is where a real
#: package leak lives too.
_RETAINED: list[Any] = []


class DenoiseInput(msgspec.Struct, forbid_unknown_fields=True):
    steps: Annotated[int, msgspec.Meta(ge=1, le=64)] = 2
    latent: Annotated[int, msgspec.Meta(ge=32, le=96)] = 64
    seed: int = 1005
    leak_mib: Annotated[int, msgspec.Meta(ge=0, le=512)] = 0
    """THE PLANT (cr-008b's executor-poisoning arm): device bytes this handler holds PAST
    its own terminal, in a module-level list the attempt cannot reach to free.

    It lives in the CORPUS, never in the runtime — a fault switch inside the product is a
    production hook nobody asked for (decisions #249). What it stands in for is ordinary and
    common: a package that caches a tensor on a module global. From the ledger's side it is
    indistinguishable from any other persistent allocation, which is the point."""


class DenoiseOutput(msgspec.Struct):
    image: ImageAsset
    steps: int
    mean: float
    std: float
    digest: str


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


class RetouchInput(msgspec.Struct, forbid_unknown_fields=True):
    """cr-012's typed input surface: ONE required image, and an ordered reference set.

    `photo` carries both bound kinds — a size the field admits and the media types it can
    decode — so a caller learns the limits from `describe` instead of from a refusal.
    `refs` is a LIST with a msgspec count bound, which is the count rule already in the
    descriptor's constraint vocabulary rather than a second spelling of one; its ORDER is
    model input and is preserved into the handler.
    """

    photo: Annotated[
        ImageAsset, AssetBound(max_bytes=4 << 20, media_types=("image/png", "image/jpeg"))
    ]
    refs: Annotated[list[ImageAsset], msgspec.Meta(max_length=3)] = msgspec.field(
        default_factory=list
    )
    scale: Annotated[int, msgspec.Meta(ge=1, le=8)] = 2


class RetouchOutput(msgspec.Struct):
    image: ImageAsset
    source_digest: str
    source_media_type: str
    source_bytes: int
    width: int
    height: int
    ref_order: list[str]


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
    there is no VAE in this binding, and inventing one would be a different package."""
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
    steps = payload.steps
    tel.log("starting the denoise loop", latent=payload.latent, steps=steps, seed=view._seed)
    on_step = tel.step_callback(steps, stage="denoise")
    with tel.stage("denoise"):
        for step in range(steps):
            ctx.raise_if_cancelled()  # THE cooperative-cancel spelling
            latents = model.denoise(latents, timestep, prompt, embeds, ids)
            on_step(step)
    if payload.leak_mib:
        # Retained on a MODULE GLOBAL, so it outlives every frame of this attempt and the
        # allocator cannot give it back when the handler returns.
        _RETAINED.append(torch.empty(payload.leak_mib << 20, dtype=torch.uint8, device=device))
        tel.log("planted a persistent device allocation", mib=payload.leak_mib)
    stats = latents.float()
    tel.metric("latent_std", round(float(stats.std()), 6))
    body = latents.reshape(-1).view(torch.uint8).cpu().numpy().tobytes()
    return DenoiseOutput(
        image=out.save_image(_preview(torch, latents), format="png"),
        steps=steps,
        mean=round(float(stats.mean()), 6),
        std=round(float(stats.std()), 6),
        digest=hashlib.sha256(body).hexdigest(),
    )


@app.entrypoint
def stubborn(payload: StubbornInput, model: SdxlUnetModel, ctx: Context) -> DenoiseOutput:
    """Real GPU work with NO cancellation check anywhere. The cooperative signal lands and
    nothing reads it — which is exactly the case where `CANCELED` may only be recorded
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


@app.entrypoint
def retouch(payload: RetouchInput, out: Outputs) -> RetouchOutput:
    """cr-012's round trip: a digest-verified image in, a transformed image out.

    Weightless on purpose — what is under test is the INPUT plane, and a model would only
    put a UNet between the bytes arriving and the bytes leaving. The handler reads
    `payload.photo.read_bytes()`, which only returns because the runtime already fetched,
    length-checked, digest-checked, sniffed and spooled those bytes under this attempt's
    grant. There is no fetch here, no URL, and no path: `payload.photo` is a value.

    The OUTPUT side is the runtime's own (`out.save_image`), so the two directions of the
    one asset type meet in one handler.
    """
    data = payload.photo.read_bytes()
    width, height = _png_size(data)
    pixels = _png_rgb(data, width, height)
    step = max(int(payload.scale), 1)
    small_w, small_h = max(width // step, 1), max(height // step, 1)
    rows = bytearray()
    for y in range(small_h):
        for x in range(small_w):
            offset = ((y * step) * width + (x * step)) * 3
            rows += pixels[offset : offset + 3]
    return RetouchOutput(
        image=out.save_image(ImageFrame(small_w, small_h, bytes(rows))),
        source_digest=payload.photo.digest,
        source_media_type=payload.photo.media_type,
        source_bytes=payload.photo.size_bytes,
        width=small_w,
        height=small_h,
        ref_order=[ref.digest for ref in payload.refs],
    )


def _png_size(data: bytes) -> tuple[int, int]:
    return int.from_bytes(data[16:20], "big"), int.from_bytes(data[20:24], "big")


def _png_rgb(data: bytes, width: int, height: int) -> bytes:
    """Decode a NON-INTERLACED, filter-0, 8-bit RGB PNG. The corpus writes exactly that."""
    import zlib

    stream = bytearray()
    cursor = 8
    while cursor < len(data):
        length = int.from_bytes(data[cursor : cursor + 4], "big")
        kind = data[cursor + 4 : cursor + 8]
        if kind == b"IDAT":
            stream += data[cursor + 8 : cursor + 8 + length]
        cursor += 12 + length
    raw = zlib.decompress(bytes(stream))
    out = bytearray()
    stride = width * 3
    for y in range(height):
        start = y * (stride + 1)
        if raw[start] != 0:
            raise ValueError("this package decodes filter-0 PNG rows only")
        out += raw[start + 1 : start + 1 + stride]
    return bytes(out)
