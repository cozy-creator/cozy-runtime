"""Native TensorFS-backed, component-staged Qwen Image 2.1 inference.

Only the maintained upstream architecture and sampling implementation are reused.
Runtime constructs/fills the modules and owns their residency; no upstream weight
loader, device placement, or offload hook participates in serving.
"""

from __future__ import annotations

import base64
import zlib
from collections.abc import Callable, Iterable, Mapping
from typing import Any

import torch
from tokenizers import Tokenizer
from transformers import (
    AutoModelForImageTextToText,
    PreTrainedTokenizerFast,
    Qwen2VLImageProcessor,
    Qwen3VLConfig,
    Qwen3VLProcessor,
    Qwen3VLVideoProcessor,
)
from transformers import initialization as transformer_init

from cozy_runtime.author import Config, Loader, Model, uses_components
from cozy_runtime.models._progress import PipelineProgress

from .upstream.autoencoder_kl_qwenimage21 import AutoencoderKLQwenImage21
from .upstream.pipeline_qwenimage21 import QwenImage21Pipeline, calculate_dimensions
from .upstream.transformer_qwenimage21 import QwenImage21Transformer2DModel


def _section(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"Qwen Image {name} configuration must be a mapping")
    return dict(value)


class QwenImage21Graph:
    """One native construction census, with source checkpoint precision preserved."""

    def __init__(self, config: Config) -> None:
        mapping = config.mapping()
        with transformer_init.no_init_weights():
            self.components: dict[str, Any] = {
                "transformer": QwenImage21Transformer2DModel.from_config(
                    _section(mapping["transformer"], "transformer")
                ).to(dtype=torch.bfloat16),
                # Construction dtype leaves derived RoPE buffers in upstream F32.
                # Casting the completed module would round those frequencies to BF16.
                "text_encoder": AutoModelForImageTextToText.from_config(
                    Qwen3VLConfig(**_section(mapping["text_encoder"], "text_encoder")),
                    dtype=torch.bfloat16,
                ),
                "vae": AutoencoderKLQwenImage21.from_config(_section(mapping["vae"], "vae")).to(
                    dtype=torch.float32
                ),
            }
        self.scheduler_config = _section(mapping["scheduler"], "scheduler")
        self.processor_config = _section(mapping.get("processor", {}), "processor")


def build_qwen_image21(config: Config) -> QwenImage21Graph:
    return QwenImage21Graph(config)


def build_processor(config: Mapping[str, Any]) -> Any:
    """Decode the prepared checkpoint's bounded processor bundle without file/network loading."""
    expected = {
        "tokenizer",
        "tokenizer_config",
        "image_processor",
        "video_processor",
        "chat_template",
    }
    if not expected <= set(config):
        raise ValueError("Qwen Image processor config is missing fields")
    encoded = config["tokenizer"]
    if not isinstance(encoded, str) or len(encoded) > 32 << 20:
        raise ValueError("Qwen Image tokenizer exceeds its encoded size limit")
    compressed = base64.b64decode(encoded, altchars=b"-_", validate=True)
    inflater = zlib.decompressobj()
    raw = inflater.decompress(compressed, (32 << 20) + 1)
    if len(raw) > 32 << 20 or not inflater.eof or inflater.unused_data:
        raise ValueError("Qwen Image tokenizer exceeds its size limit or has invalid compression")
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer.from_str(raw.decode("utf-8")),
        **dict(config["tokenizer_config"]),
    )
    return Qwen3VLProcessor(
        tokenizer=tokenizer,
        image_processor=Qwen2VLImageProcessor(**dict(config["image_processor"])),
        video_processor=Qwen3VLVideoProcessor(**dict(config["video_processor"])),
        chat_template=config["chat_template"],
    )


class _StagedPipeline(QwenImage21Pipeline):
    reference_latents: torch.Tensor | None = None
    reference_pad_mask: torch.Tensor | None = None

    def progress_bar[T](
        self, iterable: Iterable[T] | None = None, total: int | None = None
    ) -> PipelineProgress[T]:
        return PipelineProgress(iterable, total)

    @property
    def _execution_device(self) -> torch.device:
        # Upstream chooses the first registered module (VAE), which can correctly
        # remain parked on CPU while Runtime admits the denoiser. Follow the active
        # denoising component instead. Encoding supplies its own leased device.
        return next(self.transformer.parameters()).device

    def encode_prompt(self, *args: Any, **kwargs: Any) -> Any:
        # The denoiser must reuse the vision token positions from the separately
        # leased text encoder. Upstream otherwise rejects precomputed image embeds.
        if self.reference_pad_mask is not None:
            kwargs["image_pad_mask"] = self.reference_pad_mask
        return super().encode_prompt(*args, **kwargs)

    def prepare_latents(self, images: Any, *args: Any, **kwargs: Any) -> Any:
        # References were encoded while the VAE had its own residency lease.
        # Never invoke that parked component from the transformer-only lease.
        latents, image_latents = super().prepare_latents(
            None if self.reference_latents is not None else images, *args, **kwargs
        )
        if self.reference_latents is not None:
            image_latents = self.reference_latents.to(device=latents.device, dtype=latents.dtype)
        return latents, image_latents


def prefix_cache_bytes(transformer: Any, prefix_tokens: int) -> int:
    """Bytes of the post-RoPE K and V the denoiser caches for every prefix token."""
    config = transformer.config
    per_token = int(
        2 * config.num_attention_heads * config.attention_head_dim * transformer.dtype.itemsize
    )
    return int(prefix_tokens) * int(config.num_layers) * per_token


def _prefix_cache_fits(
    transformer: Any, embeds: Any, reference_latents: Any, device: torch.device
) -> bool:
    if device.type != "cuda":
        return True
    prefix = embeds.shape[1] + (0 if reference_latents is None else reference_latents.shape[1])
    free, _total = torch.cuda.mem_get_info(device)
    unused = torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)
    return bool(prefix_cache_bytes(transformer, prefix) < free + unused)


#: Runtime code the encoders read beyond their own bodies: their memo key binds it.
_ENCODER_CODE = (_StagedPipeline, build_processor)


class QwenImage21Model(Model[QwenImage21Graph]):
    graph: QwenImage21Graph
    processor: Any

    def load(self, loader: Loader) -> None:
        self.graph = loader.construct(QwenImage21Graph, factory=build_qwen_image21)
        self.processor = build_processor(self.graph.processor_config)

    def _pipeline(self) -> _StagedPipeline:
        from diffusers import FlowMatchEulerDiscreteScheduler

        return _StagedPipeline(
            **self.graph.components,
            processor=self.processor,
            scheduler=FlowMatchEulerDiscreteScheduler.from_config(self.graph.scheduler_config),
        )

    @uses_components("text_encoder", memoize=True, memo_dependencies=_ENCODER_CODE)
    def encode(self, prompt: str) -> tuple[Any, Any]:
        encoder = self.graph.components["text_encoder"]
        with torch.inference_mode():
            embeds, mask, _ = self._pipeline().encode_prompt(
                prompt, device=next(encoder.parameters()).device
            )
        return embeds, mask

    @uses_components("vae", memoize=True, memo_dependencies=_ENCODER_CODE)
    def encode_reference_images(self, images: list[Any]) -> tuple[list[Any], torch.Tensor]:
        """Encode each reference once, using upstream's 1 MP conditioning size."""
        if not 1 <= len(images) <= 10:
            raise ValueError("Qwen Image requires between one and ten reference images")
        vae = self.graph.components["vae"]
        device = next(vae.parameters()).device
        pipeline = self._pipeline()
        resized, packed = [], []
        with torch.inference_mode():
            for image in images:
                image = image.convert("RGBA")
                width, height, _ = calculate_dimensions(1024 * 1024, image.width / image.height)
                resized.append(pipeline.image_processor.resize(image, width=width, height=height))
                pixels = (
                    pipeline.image_processor.preprocess(image, width=width, height=height)
                    .unsqueeze(2)
                    .to(device=device, dtype=vae.dtype)
                )
                # argmax encoding is deterministic and consumes no sampler RNG.
                encoded = pipeline._encode_vae_image(pixels, generator=None)
                packed.append(
                    pipeline._pack_latents(
                        encoded, 1, pipeline.latent_channels, encoded.shape[3], encoded.shape[4]
                    ).cpu()
                )
        return resized, torch.cat(packed, dim=1)

    @uses_components("text_encoder", memoize=True, memo_dependencies=_ENCODER_CODE)
    def encode_image_prompt(self, prompt: str, images: list[Any]) -> tuple[Any, Any, Any]:
        """Encode the instruction and ordered reference images as one condition."""
        encoder = self.graph.components["text_encoder"]
        with torch.inference_mode():
            encoded: tuple[Any, Any, Any] = self._pipeline().encode_prompt(
                prompt, image=images, device=next(encoder.parameters()).device
            )
        return encoded

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
        reference_images: list[Any] | None = None,
        reference_latents: torch.Tensor | None = None,
        image_pad_mask: torch.Tensor | None = None,
    ) -> Any:
        transformer = self.graph.components["transformer"]
        device = next(transformer.parameters()).device
        pipeline = self._pipeline()
        if reference_images is not None:
            if not reference_images or reference_latents is None or image_pad_mask is None:
                raise ValueError(
                    "Qwen Image references require images, latents and vision token mask"
                )
            pipeline.reference_latents = reference_latents
            pipeline.reference_pad_mask = image_pad_mask.to(device=device)
        elif reference_latents is not None or image_pad_mask is not None:
            raise ValueError("Qwen Image reference tensors require their source images")

        def observed(_pipeline: Any, step: int, _timestep: Any, values: Any) -> Any:
            cancel()
            on_step(step)
            return values

        def run(use_kv_cache: bool) -> Any:
            with torch.inference_mode():
                return pipeline(
                    image=reference_images,
                    prompt_embeds=embeds.to(device=device, dtype=transformer.dtype),
                    prompt_embeds_mask=None if mask is None else mask.to(device=device),
                    width=width,
                    height=height,
                    num_inference_steps=steps,
                    generator=torch.Generator(device=device).manual_seed(seed),
                    output_type="latent",
                    callback_on_step_end=observed,
                    use_kv_cache=use_kv_cache,
                ).images

        cancel()
        # The prefix KV cache (text, vision and reference tokens, every layer) grows by
        # ~2 GiB per 1 MP reference. It is an exact speedup: when the card cannot hold it,
        # recompute the prefix each step instead, slower but the same image.
        if _prefix_cache_fits(transformer, embeds, reference_latents, device):
            try:
                return run(use_kv_cache=True)
            except torch.OutOfMemoryError:
                pass
            torch.cuda.empty_cache()
        return run(use_kv_cache=False)

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
