"""Native TensorFS-backed, component-staged Qwen Image 2.1 inference.

Only the maintained upstream architecture and sampling implementation are reused.
Runtime constructs/fills the modules and owns their residency; no upstream weight
loader, device placement, or offload hook participates in serving.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration, Qwen3VLProcessor
from transformers import initialization as transformer_init

from cozy_runtime.author import Config, Loader, Model, uses_components

from .upstream.autoencoder_kl_qwenimage21 import AutoencoderKLQwenImage21
from .upstream.pipeline_qwenimage21 import QwenImage21Pipeline
from .upstream.transformer_qwenimage21 import QwenImage21Transformer2DModel


class QwenImage21Graph:
    """One native construction census, with source checkpoint precision preserved."""

    def __init__(self, config: Config) -> None:
        mapping = config.mapping()
        with transformer_init.no_init_weights():
            self.components: dict[str, Any] = {
                "transformer": QwenImage21Transformer2DModel.from_config(
                    dict(mapping["transformer"])
                ).to(dtype=torch.bfloat16),
                "text_encoder": Qwen3VLForConditionalGeneration(
                    Qwen3VLConfig(**dict(mapping["text_encoder"]))
                ).to(dtype=torch.bfloat16),
                "vae": AutoencoderKLQwenImage21.from_config(dict(mapping["vae"])).to(
                    dtype=torch.float32
                ),
            }
        self.scheduler_config = dict(mapping["scheduler"])


def build_qwen_image21(config: Config) -> QwenImage21Graph:
    return QwenImage21Graph(config)


class _StagedPipeline(QwenImage21Pipeline):
    @property
    def _execution_device(self) -> torch.device:
        # Upstream chooses the first registered module (VAE), which can correctly
        # remain parked on CPU while Runtime admits the denoiser. Follow the active
        # denoising component instead. Encoding supplies its own leased device.
        return next(self.transformer.parameters()).device


class QwenImage21Model(Model[QwenImage21Graph]):
    graph: QwenImage21Graph
    processor: Any

    def load(self, loader: Loader) -> None:
        self.graph = loader.construct(QwenImage21Graph, factory=build_qwen_image21)
        with loader.assets.materialized("processor") as root:
            self.processor = Qwen3VLProcessor.from_pretrained(root, local_files_only=True)

    def _pipeline(self) -> _StagedPipeline:
        from diffusers import FlowMatchEulerDiscreteScheduler

        pipeline = _StagedPipeline(
            **self.graph.components,
            processor=self.processor,
            scheduler=FlowMatchEulerDiscreteScheduler.from_config(self.graph.scheduler_config),
        )
        pipeline.set_progress_bar_config(disable=True)
        return pipeline

    @uses_components("text_encoder")
    def encode(self, prompt: str) -> tuple[Any, Any]:
        encoder = self.graph.components["text_encoder"]
        with torch.inference_mode():
            embeds, mask, _ = self._pipeline().encode_prompt(
                prompt, device=next(encoder.parameters()).device
            )
        return embeds, mask

    @uses_components("transformer")
    def denoise(
        self,
        embeds: Any,
        mask: Any,
        *,
        width: int,
        height: int,
        steps: int,
        seed: int,
        on_step: Callable[[int], None],
        cancel: Callable[[], None],
    ) -> Any:
        transformer = self.graph.components["transformer"]
        device = next(transformer.parameters()).device
        generator = torch.Generator(device=device).manual_seed(seed)
        pipeline = self._pipeline()

        def observed(_pipeline: Any, step: int, _timestep: Any, values: Any) -> Any:
            cancel()
            on_step(step)
            return values

        cancel()
        with torch.inference_mode():
            return pipeline(
                prompt_embeds=embeds.to(device=device, dtype=transformer.dtype),
                prompt_embeds_mask=None if mask is None else mask.to(device=device),
                width=width,
                height=height,
                num_inference_steps=steps,
                generator=generator,
                output_type="latent",
                callback_on_step_end=observed,
            ).images

    @uses_components("vae")
    def decode(self, latents: Any, *, width: int, height: int) -> Any:
        vae = self.graph.components["vae"]
        device = next(vae.parameters()).device
        with torch.inference_mode():
            latents = QwenImage21Pipeline._unpack_latents(latents, height, width, 16)
            latents = latents.to(device=device, dtype=vae.dtype)
            mean = latents.new_tensor(vae.config.latents_mean).view(1, vae.config.z_dim, 1, 1, 1)
            std = latents.new_tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1)
            # Identical to the upstream final decode, with no cast of F32 source weights.
            return vae.decode(latents * std + mean, return_dict=False)[0][:, :, 0]
