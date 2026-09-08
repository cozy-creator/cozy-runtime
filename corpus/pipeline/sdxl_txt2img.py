"""cr-008b's package: real SDXL text-to-image, four components, zero memory management.

One entrypoint, `generate`, doing what a product does: tokenize, encode with both towers,
run a real denoising loop with classifier-free guidance, decode through the VAE, and save a
PNG. It touches three public model operations, each declaring its own component set, and it
contains no device, movement, offload, pinning, eviction or backend API — there is no such
surface for it to reach. Whether the UNet was resident when `denoise` was entered is the
runtime's problem, decided before entry, and this file cannot tell.

`steps` and `size` are the request's own knobs because they are what a caller asks for. The
plan that serves them is not: an 8 GiB card cannot hold 6.461 GiB of weights beside a 1024px
VAE decode, and the answer to that is the PlanChooser's ladder, not a smaller request.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

import msgspec
from corpus.sdxl_pipeline import SdxlPipelineModel

from cozy_runtime.author import App, Context, ImageAsset, ImageFrame, Outputs, Telemetry

app = App()


class Txt2ImgInput(msgspec.Struct, forbid_unknown_fields=True):
    prompt: str = "a photograph of an astronaut riding a horse"
    negative_prompt: str = ""
    steps: Annotated[int, msgspec.Meta(ge=1, le=50)] = 20
    size: Annotated[int, msgspec.Meta(ge=256, le=1024)] = 1024
    guidance: float = 5.0
    seed: int = 1005


class Txt2ImgOutput(msgspec.Struct):
    image: ImageAsset
    steps: int
    size: int
    digest: str
    """sha256 of the decoded RGB pixel bytes — the determinism fence for the WHOLE loop."""


#: The tokenizer vocabularies this package BUNDLES. They are the package's own asset,
#: like the model library it imports — not an artifact identifier, not a catalog ref, and
#: not something the construction config may carry (a path in a config refuses, §1.1).
_TOKENIZERS = Path(__file__).resolve().parent


def _finite(torch: Any, value: Any) -> float:
    """The tensor's largest magnitude, with NaN neutralized so the metric is spellable.

    A metric that cannot be serialized is a metric that is not there: the observation emit
    boundary refuses a non-finite value (there is no canonical spelling for one), so the
    NaN FRACTION is reported as its own number and this one stays a real magnitude.
    """
    return round(float(torch.nan_to_num(value, 0.0, 0.0, 0.0).abs().max()), 4)


def _tokenize(root: Path, prompt: str) -> tuple[Any, Any]:
    from transformers import CLIPTokenizer

    out = []
    for name in ("tokenizer", "tokenizer_2"):
        tok = CLIPTokenizer.from_pretrained(str(root / name))
        out.append(
            tok(
                prompt,
                padding="max_length",
                max_length=tok.model_max_length,
                truncation=True,
                return_tensors="pt",
            ).input_ids
        )
    return out[0], out[1]


@app.entrypoint
def generate(
    ctx: Context,
    payload: Txt2ImgInput,
    model: SdxlPipelineModel,
    out: Outputs,
    tel: Telemetry,
) -> Txt2ImgOutput:
    import hashlib

    import torch
    from diffusers import EulerDiscreteScheduler

    device = torch.device("cuda", 0)
    view = model.for_request(ctx, seed=payload.seed)
    steps = payload.steps
    side = payload.size
    latent = side // 8
    with tel.stage("tokenize"):
        ids, ids_2 = _tokenize(_TOKENIZERS, payload.prompt)
        neg, neg_2 = _tokenize(_TOKENIZERS, payload.negative_prompt)

    with tel.stage("encode"):
        prompt, pooled = model.encode(ids.to(device), ids_2.to(device))
        negative, neg_pooled = model.encode(neg.to(device), neg_2.to(device))

    tel.metric("prompt_absmax", _finite(torch, prompt))
    tel.metric("pooled_absmax", _finite(torch, pooled))
    scheduler = EulerDiscreteScheduler.from_config(model.pipe.scheduler_config)
    scheduler.set_timesteps(steps, device=device)
    generator = torch.Generator(device=device).manual_seed(view._seed)
    latents = (
        torch.randn(1, 4, latent, latent, generator=generator, device=device, dtype=torch.float16)
        * scheduler.init_noise_sigma
    )
    time_ids = torch.tensor([[side, side, 0, 0, side, side]], device=device, dtype=torch.float16)
    both = torch.cat([negative, prompt])
    both_pooled = torch.cat([neg_pooled, pooled])
    both_ids = torch.cat([time_ids, time_ids])

    on_step = tel.step_callback(steps, stage="denoise")
    with tel.stage("denoise"):
        for index, timestep in enumerate(scheduler.timesteps):
            ctx.raise_if_cancelled()
            model_input = scheduler.scale_model_input(torch.cat([latents] * 2), timestep)
            noise = model.denoise(model_input, timestep, both, both_pooled, both_ids)
            if index == 0:
                tel.metric("noise_absmax_step0", _finite(torch, noise))
            uncond, cond = noise.chunk(2)
            guided = uncond + payload.guidance * (cond - uncond)
            latents = scheduler.step(guided, timestep, latents).prev_sample
            on_step(index)

    tel.metric("latent_absmax", _finite(torch, latents))
    with tel.stage("decode"):
        image = model.decode(latents)
    tel.metric("image_absmax", _finite(torch, image))
    tel.metric("image_nan_fraction", round(float(torch.isnan(image).float().mean()), 6))
    pixels = ((image / 2 + 0.5).clamp(0, 1)[0] * 255).to(torch.uint8).permute(1, 2, 0).contiguous()
    rgb = bytes(pixels.cpu().numpy().tobytes())
    height, width = int(pixels.shape[0]), int(pixels.shape[1])
    tel.metric("decoded_pixels", float(width * height))
    with tel.stage("encode_png"):
        asset = out.save_image(ImageFrame(width, height, rgb), format="png")
    return Txt2ImgOutput(
        image=asset,
        steps=steps,
        size=side,
        digest=hashlib.sha256(rgb).hexdigest(),
    )


class ConditionOutput(msgspec.Struct):
    image: ImageAsset
    prompt_absmax: float
    digest: str


@app.entrypoint
def condition(
    ctx: Context,
    payload: Txt2ImgInput,
    model: SdxlPipelineModel,
    out: Outputs,
    tel: Telemetry,
) -> ConditionOutput:
    """The SMALL-FOOTPRINT arm: both encoders and the VAE, no UNet.

    It exists because the whole pipeline needs a card nobody else is on, and the component
    contract does not: two scopes, three components, real staging and real eviction, in
    1.68 GiB. Everything it proves about admission holds for the 6.46 GiB case; what it
    cannot prove is what happens when ONE component is most of the card.
    """
    import hashlib

    import torch

    device = torch.device("cuda", 0)
    view = model.for_request(ctx, seed=payload.seed)
    side = payload.size
    with tel.stage("tokenize"):
        ids, ids_2 = _tokenize(_TOKENIZERS, payload.prompt)
    with tel.stage("encode"):
        prompt, _pooled = model.encode(ids.to(device), ids_2.to(device))
    generator = torch.Generator(device=device).manual_seed(view._seed)
    latents = torch.randn(
        1, 4, side // 8, side // 8, generator=generator, device=device, dtype=torch.float16
    )
    with tel.stage("decode"):
        image = model.decode(latents)
    tel.metric("image_nan_fraction", round(float(torch.isnan(image).float().mean()), 6))
    pixels = ((image / 2 + 0.5).clamp(0, 1)[0] * 255).to(torch.uint8).permute(1, 2, 0).contiguous()
    rgb = bytes(pixels.cpu().numpy().tobytes())
    height, width = int(pixels.shape[0]), int(pixels.shape[1])
    return ConditionOutput(
        image=out.save_image(ImageFrame(width, height, rgb), format="png"),
        prompt_absmax=_finite(torch, prompt),
        digest=hashlib.sha256(rgb).hexdigest(),
    )
