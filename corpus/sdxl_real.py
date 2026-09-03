"""The REAL SDXL UNet, as a package: upstream diffusers, upstream config, real bytes.

`corpus/sdxl.py` is the SHAPE of an SDXL pipeline built from `torch.nn` primitives so the
derivation corpus can run anywhere. This file is the other half of that claim: the actual
`diffusers.UNet2DConditionModel` at `stabilityai/stable-diffusion-xl-base-1.0`'s fp16
config — 1,680 destinations, 4.782 GiB — constructed through the same one primitive, so
cr-005's fill moves real ecosystem weights into a real ecosystem module and cr-004's
derive/serve fence gets re-run against bytes instead of a counter.

The package names no path, no repo and no checkpoint. It receives the artifact's immutable
config as a typed read-only capability and hands it to the library's own constructor —
which is what every real package does, and why `Config.mapping()` exists.
"""

from __future__ import annotations

from typing import Any

from cozy_runtime.author import Config, Loader, Model, uses_components


class SdxlUnetPipeline:
    """One heavyweight component behind one denoise call.

    Deliberately the SINGLE-component case: SDXL's four-component whole-pipeline shape is
    already derived in `corpus/sdxl.py`, and what cr-005 needs from this fixture is a real
    5 GB DestinationSet on an 8 GiB card, not a second copy of the tensor-schema exercise.
    """

    def __init__(self, config: Config) -> None:
        import torch
        from diffusers import UNet2DConditionModel

        unet = UNet2DConditionModel.from_config(config.mapping())
        # fp16 is the DESTINATION's compute contract and the checkpoint's stored dtype;
        # matching them is what makes this fill a plain encoding-contract copy (cr-006
        # owns the case where they differ).
        self.components: dict[str, Any] = {"unet": unet.to(torch.float16)}


def build_unet(config: Config) -> SdxlUnetPipeline:
    return SdxlUnetPipeline(config)


class SdxlUnetModel(Model[SdxlUnetPipeline]):
    """No class keyword: this admits by pure tensor-schema satisfaction, like `SdxlModel`."""

    pipe: SdxlUnetPipeline

    def load(self, loader: Loader) -> None:
        self.pipe = loader.construct(SdxlUnetPipeline, factory=build_unet)

    @uses_components("unet")
    def denoise(
        self,
        latents: Any,
        timestep: Any,
        prompt: Any,
        text_embeds: Any,
        time_ids: Any,
    ) -> Any:
        """One denoising step. Reached through the GUARDED component root, so the scope
        check runs on the real 4.78 GiB UNet on the way in."""
        import torch

        with torch.inference_mode():
            return self.pipe.components["unet"](
                latents,
                timestep,
                encoder_hidden_states=prompt,
                added_cond_kwargs={"text_embeds": text_embeds, "time_ids": time_ids},
            ).sample


COMPONENT_USE = {"denoise": ("unet",)}
