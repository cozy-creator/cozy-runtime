"""The REAL four-component SDXL pipeline: two text encoders, the UNet, the VAE.

`corpus/sdxl_real.py` is the single-component fixture cr-005 and cr-007 needed — one 4.78 GiB
DestinationSet on an 8 GiB card. This is cr-008b's fixture and its whole point is the
opposite: a pipeline whose components DO NOT ALL FIT AT ONCE next to their activations, so
the PlanChooser has a real decision to make and the component-use contract has real
staging to do.

    text_encoder     196 destinations   0.229 GiB   CLIPTextModel (ViT-L text tower)
    text_encoder_2   517 destinations   1.294 GiB   CLIPTextModelWithProjection (bigG)
    unet            1680 destinations   4.782 GiB   UNet2DConditionModel
    vae              248 destinations   0.156 GiB   AutoencoderKL
                    ────────────────────────────
                    2641 destinations   6.461 GiB   on a card with 7.99 GiB total

Three public operations, each declaring EXACTLY the components it may touch. The declaration
is a component-use contract, not a phase list: while `encode` runs, either encoder may be
touched and nothing else may. That is what the runtime admits and materializes against, and
what an undeclared access poisons the executor for.

The package names no device, no placement, no offload and no backend — there is no such
surface to name. Whether a component is resident when a method entered is the runtime's
question, answered before entry, and this file cannot observe the answer.
"""

from __future__ import annotations

from typing import Any

from cozy_runtime.author import Config, Loader, Model, uses_components


class SdxlPipeline:
    """The four constructed component roots. Built from the artifact's immutable config."""

    def __init__(self, config: Config) -> None:
        import torch
        from diffusers import AutoencoderKL, UNet2DConditionModel
        from transformers import CLIPTextConfig, CLIPTextModel, CLIPTextModelWithProjection

        mapping = config.mapping()
        # fp16 everywhere: it is the checkpoint's stored dtype AND the destination's compute
        # contract, so every fill stays a plain encoding-contract copy (cr-006 owns decode).
        self.components: dict[str, Any] = {
            "text_encoder": CLIPTextModel(CLIPTextConfig(**mapping["text_encoder"])).to(
                torch.float16
            ),
            "text_encoder_2": CLIPTextModelWithProjection(
                CLIPTextConfig(**mapping["text_encoder_2"])
            ).to(torch.float16),
            "unet": UNet2DConditionModel.from_config(mapping["unet"]).to(torch.float16),
            "vae": AutoencoderKL.from_config(mapping["vae"]).to(torch.float16),
        }
        self.scheduler_config: dict[str, Any] = dict(mapping["scheduler"])
        self.vae_scale: float = float(mapping["vae"]["scaling_factor"])


def build_pipeline(config: Config) -> SdxlPipeline:
    return SdxlPipeline(config)


class SdxlPipelineModel(Model[SdxlPipeline]):
    """Admits by pure tensor-schema satisfaction, like every other model in this corpus."""

    pipe: SdxlPipeline

    def load(self, loader: Loader) -> None:
        self.pipe = loader.construct(SdxlPipeline, factory=build_pipeline)

    @uses_components("text_encoder", "text_encoder_2")
    def encode(self, ids: Any, ids_2: Any) -> tuple[Any, Any]:
        """SDXL's two-tower conditioning. ONE method, BOTH encoders — a composite operation
        declares its components together rather than opening two scopes."""
        import torch

        with torch.inference_mode():
            first = self.pipe.components["text_encoder"](ids, output_hidden_states=True)
            second = self.pipe.components["text_encoder_2"](ids_2, output_hidden_states=True)
            prompt = torch.cat([first.hidden_states[-2], second.hidden_states[-2]], dim=-1)
            return prompt, second.text_embeds

    @uses_components("unet")
    def denoise(
        self, latents: Any, timestep: Any, prompt: Any, text_embeds: Any, time_ids: Any
    ) -> Any:
        import torch

        with torch.inference_mode():
            return self.pipe.components["unet"](
                latents,
                timestep,
                encoder_hidden_states=prompt,
                added_cond_kwargs={"text_embeds": text_embeds, "time_ids": time_ids},
            ).sample

    @uses_components("vae")
    def decode(self, latents: Any) -> Any:
        import torch

        with torch.inference_mode():
            return self.pipe.components["vae"].decode(latents / self.pipe.vae_scale).sample

    @uses_components("unet")
    def touch_undeclared(self) -> Any:
        """THE POISON ARM, and it is deliberately in the fixture rather than in a plant.

        It declares only `unet` and reaches for `vae`. An undeclared component access is a
        typed poison at the guarded root — never a reactive load — and the executor that
        served it is replaced rather than reused.
        """
        return self.pipe.components["vae"].config


COMPONENT_USE = {
    "encode": ("text_encoder", "text_encoder_2"),
    "denoise": ("unet",),
    "decode": ("vae",),
}
