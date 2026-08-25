"""A real, runnable endpoint on the cr-002a author surface — SDXL-shaped, weightless.

Modeled line-for-line on `cozy-runtime-sdxl.md` (and the H3 example's preflight arm), with
one substitution: the pipeline is a tiny deterministic renderer instead of diffusers,
because the `Model` CONSTRUCTION boundary (`Loader`, `load(T)`, `@uses_components`) is
cr-002b's, not this issue's. Everything the handler touches is the production surface: real
injection, real `ModelDefault` overlay, real preflight, real output transaction, real
encode. `scripts/verify.py` runs every entrypoint here and prints what happened.

Four entrypoints, each proving something the others do not:
  generate      the common case — bare decorator, derived gpu, Shape presets, ModelDefault
  plan          NO Outputs parameter: structurally cannot save (the capability red arm)
  generate_ref  typed preflight + Settings + an asset request field + handler Adjustments
  caption       the `async def` spelling on its private per-attempt loop, with Egress
"""

import random
from enum import Enum
from typing import Annotated

import msgspec

from cozy_runtime.author import (
    Adjustments,
    App,
    Context,
    Egress,
    ImageAsset,
    ImageFrame,
    Model,
    ModelDefault,
    Outputs,
    Preflight,
    RequestView,
    Settings,
    Shape,
    Telemetry,
    UnsupportedInput,
)

app = App()


class AspectRatio(Enum):
    SQUARE = "1:1"
    LANDSCAPE = "4:3"
    WIDE = "16:9"


# Preset buckets — the ONE canonical table: it feeds the request schema (Shape), the demand
# features, and the handler's own indexing. A copied number elsewhere is a second authority.
_BUCKETS = {
    AspectRatio.SQUARE: (64, 64),
    AspectRatio.LANDSCAPE: (72, 56),
    AspectRatio.WIDE: (84, 48),
}


class Txt2ImgInput(msgspec.Struct, forbid_unknown_fields=True):
    prompt: str
    negative_prompt: str = ""
    # picks WITHIN the deployment's pinned slot policy; None = the binding's default
    model_selector: str | None = None
    aspect_ratio: Annotated[AspectRatio, Shape(pixels=_BUCKETS)] = AspectRatio.SQUARE
    # explicit request -> adapter recipe -> the selected checkpoint's row -> this fallback.
    # The bound lives ON the struct: 51 REJECTS typed, a policy clamp adjusts visibly.
    num_inference_steps: Annotated[ModelDefault[int], msgspec.Meta(ge=1, le=50)] = 30
    guidance_scale: ModelDefault[float] = 5.0
    seed: int | None = None


class ImageOutput(msgspec.Struct):
    image: ImageAsset
    checkpoint: str


class PlanOutput(msgspec.Struct):
    width: int
    height: int
    steps: int


class TinyPipe:
    """Stands in for `StableDiffusionXLPipeline` until cr-002b lands the Model boundary."""

    def __call__(
        self, *, prompt: str, width: int, height: int, steps: int, guidance: float,
        generator: random.Random,
    ) -> ImageFrame:
        jitter = generator.randbytes(3)
        tint = (sum(prompt.encode()) + steps + int(guidance * 10)) & 0xFF
        rows = bytearray()
        for y in range(height):
            for x in range(width):
                rows += bytes(
                    (
                        (x * 255 // max(width - 1, 1)) ^ jitter[0],
                        (y * 255 // max(height - 1, 1)) ^ jitter[1],
                        tint ^ jitter[2],
                    )
                )
        return ImageFrame(width, height, bytes(rows))


class SdxlModel(Model[TinyPipe]):
    """A bare typed parameter is the whole declaration — and its presence derives `gpu`."""

    def __init__(self, checkpoint_ref: str = "test://sdxl-base@1") -> None:
        self.pipe = TinyPipe()
        self.checkpoint_ref = checkpoint_ref

    # cr-002b adds `load(self, loader: Loader)` and `@uses_components("unet", "vae", …)`.
    def generate(
        self, view: RequestView, prompt: str, *, width: int, height: int,
        steps: int, guidance: float,
    ) -> ImageFrame:
        # The model calls its OWN typed pipeline; the view supplies the request-local
        # generator. RequestView has no pipeline passthrough.
        return self.pipe(
            prompt=prompt,
            width=width,
            height=height,
            steps=steps,
            guidance=guidance,
            generator=view.generator,
        )


@app.entrypoint  # bare is the common case: resources= is an override door
def generate(
    ctx: Context, payload: Txt2ImgInput, model: SdxlModel, out: Outputs, tel: Telemetry
) -> ImageOutput:
    ctx.raise_if_cancelled()
    width, height = _BUCKETS[payload.aspect_ratio]
    view = model.for_request(ctx, seed=payload.seed)
    steps = 1 if ctx.boot_warmup else payload.num_inference_steps
    on_step = tel.step_callback(steps, stage="denoise")
    with tel.stage("denoise"):
        for step in range(steps):
            on_step(step)
        frame = model.generate(
            view, payload.prompt, width=width, height=height, steps=steps,
            guidance=payload.guidance_scale,
        )
    return ImageOutput(
        image=out.save_image(frame, format="png"), checkpoint=model.checkpoint_ref
    )


@app.entrypoint
def plan(payload: Txt2ImgInput, model: SdxlModel) -> PlanOutput:
    """No `Outputs` parameter: this handler structurally cannot save, and its derived
    AuthorCapabilitySet is empty — while the runtime still journals and emits telemetry."""
    width, height = _BUCKETS[payload.aspect_ratio]
    return PlanOutput(width=width, height=height, steps=payload.num_inference_steps)


# ---------------------------------------------------------------- preflight arm


class RefInput(msgspec.Struct, forbid_unknown_fields=True):
    prompt: str
    reference: ImageAsset | None = None
    aspect_ratio: Annotated[AspectRatio, Shape(pixels=_BUCKETS)] = AspectRatio.SQUARE
    num_inference_steps: Annotated[ModelDefault[int], msgspec.Meta(ge=1, le=50)] = 24
    guidance_scale: ModelDefault[float] = 4.5
    seed: int | None = None


class RefSettings(msgspec.Struct):
    """Deployment-owned toggle; the AUTHOR declares the schema, the deployment the value."""

    references_enabled: bool = True


class RefFacts(msgspec.Struct, frozen=True):
    """The frozen preflight record: ONLY cross-field facts RequestFeatures cannot express.
    Naming `steps` or `width` here would refuse at describe — two producers of one fact."""

    has_visual_condition: bool


def preflight_ref(payload: RefInput, settings: RefSettings) -> RefFacts:
    # Pure, metadata-only, no Context/Model/bytes. An exception here is a typed pre-entry
    # refusal BEFORE hydration — a refused request downloads nothing.
    if payload.reference is not None and not settings.references_enabled:
        raise UnsupportedInput(
            "reference conditioning is not enabled on this deployment",
            code="references_disabled",
            fields=["reference"],
        )
    return RefFacts(has_visual_condition=payload.reference is not None)


@app.entrypoint(preflight=preflight_ref)
def generate_ref(
    ctx: Context,
    payload: RefInput,
    facts: Preflight[RefFacts],
    model: SdxlModel,
    out: Outputs,
    adj: Adjustments,
    settings: Settings[RefSettings],
) -> ImageOutput:
    # The reference gate ran in preflight — nothing revalidates here.
    width, height = _BUCKETS[payload.aspect_ratio]
    view = model.for_request(ctx, seed=payload.seed)
    steps = payload.num_inference_steps
    if facts.has_visual_condition and steps > 8:
        # The handler's OWN caller-visible adjustment — this is what `Adjustments` is for.
        adj.adjusted("num_inference_steps", steps, 8, "reference conditioning caps steps")
        steps = 8
    frame = model.generate(
        view, payload.prompt, width=width, height=height, steps=steps,
        guidance=payload.guidance_scale,
    )
    return ImageOutput(
        image=out.save_image(frame, format="png"), checkpoint=model.checkpoint_ref
    )


# ---------------------------------------------------------------- async arm


class CaptionInput(msgspec.Struct, forbid_unknown_fields=True):
    prompt: str
    endpoint: str = "https://captions.example/api"


class CaptionOutput(msgspec.Struct):
    caption: str
    loop: str


@app.entrypoint
async def caption(ctx: Context, payload: CaptionInput, net: Egress) -> CaptionOutput:
    """A weightless connector handler: no Model parameter, so `gpu` derives False.

    `async def` buys the I/O tail, never a compute body — it runs on a PRIVATE event loop
    owned by the executor thread for THIS attempt, which never admits a second.
    """
    import asyncio

    response = net.fetch(payload.endpoint, method="POST", body=payload.prompt.encode())
    await asyncio.sleep(0)
    return CaptionOutput(
        caption=response.body.decode(), loop=hex(id(asyncio.get_running_loop()))
    )
