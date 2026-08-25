"""A real, runnable endpoint on the author surface — SDXL-shaped, weightless.

Modeled line-for-line on `cozy-runtime-sdxl.md`, with one substitution: the pipeline comes
from `tiny_diffusers` instead of diffusers, because cozy-runtime defines no model
architectures and every fixture must run on this box with no weights. Everything the
handler and the model touch is the production surface: real construction through
`loader.load(T)`, real census/match/fill, real guarded component roots, real injection,
real `ModelDefault` overlay, real preflight, real output transaction, real encode.

This file is the MINIMUM case: one input struct, one Model class with one coarse method,
one entrypoint — plus three arms the others do not cover.

  generate      the common case — bare decorator, derived gpu, one coarse component set
  plan          NO Outputs parameter: structurally cannot save (the capability arm)
  generate_ref  typed preflight + Settings + an asset request field + handler Adjustments
  caption       the `async def` spelling on its private per-attempt loop, with Egress

`generate` and `generate_ref` both declare `model: SdxlModel`: two independent binding
paths that COALESCE onto one model generation when their construction identities match.
"""

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
    Loader,
    Model,
    ModelDefault,
    Outputs,
    Preflight,
    RequestView,
    Settings,
    Shape,
    Telemetry,
    UnsupportedInput,
    uses_components,
)
from tiny_diffusers import ImageFrameData, TinySdxlPipeline

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


class SdxlModel(Model[TinySdxlPipeline]):
    """A bare typed parameter is the whole declaration — and its presence derives `gpu`.

    No stamp keyword: SDXL has no structural twins, so admission is pure topology
    satisfaction and the deployment binds whatever curated gallery policy it likes.
    """

    pipe: TinySdxlPipeline  # declared state — mypy-strict checked (§1.7)

    def load(self, loader: Loader) -> None:
        # load(T): sugar legal because this pipeline family ships a REGISTERED construction
        # adapter. It is never a from_pretrained duck-type and never a path handoff (§1.1).
        self.pipe = loader.load(TinySdxlPipeline)

    def unload(self, loader: Loader) -> None:
        del self.pipe

    # FOUR heavyweight components — SDXL has two text encoders; a missing name here poisons
    # at first touch. Coarse whole-__call__ scope: safe-and-slower (§1.1).
    @uses_components("text_encoder", "text_encoder_2", "unet", "vae")
    def generate(
        self, view: RequestView, prompt: str, *, negative: str, width: int, height: int,
        steps: int, guidance: float,
    ) -> ImageFrame:
        # The model calls its OWN typed pipeline; the view supplies the request-local
        # generator. RequestView has no pipeline passthrough.
        frame = self.pipe(
            prompt=prompt,
            # CFG-free turbo execution ignores the negative branch entirely
            negative_prompt=(negative if guidance > 1.0 else None),
            width=width,
            height=height,
            steps=steps,
            guidance=guidance,
            generator=view.generator,
        )
        return _render(frame)


def _render(frame: ImageFrameData) -> ImageFrame:
    """Pure pixel packing — no component is touched, so it is module code (§1.1)."""
    rows = bytearray()
    for y in range(frame.height):
        for x in range(frame.width):
            rows += bytes(
                (
                    (x * 255 // max(frame.width - 1, 1)) ^ frame.jitter[0],
                    (y * 255 // max(frame.height - 1, 1)) ^ frame.jitter[1],
                    frame.tint ^ frame.jitter[2],
                )
            )
    return ImageFrame(frame.width, frame.height, bytes(rows))


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
            view, payload.prompt, negative=payload.negative_prompt, width=width,
            height=height, steps=steps, guidance=payload.guidance_scale,
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
        view, payload.prompt, negative="", width=width, height=height, steps=steps,
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
