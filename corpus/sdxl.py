"""A real-torch SDXL fixture: the WHOLE-PIPELINE path, which is the common case.

The `Model` still owns lifecycle and loaded state, but its generation method delegates
straight to the pipeline's `__call__` with ONE conservative component-use set instead of
exposing smaller model methods (§1.1). Architecturally that is the opposite end from H3's
component-staged path, so deriving it is not the same exercise at a different size:
`components` here is a flat table of four heavyweight roots with no role scoping, no
structural variants, and no schedule.

SDXL's SECOND text encoder is load-bearing and is exactly where a three-component
declaration poisons at first touch of the fourth.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn

from cozy_runtime.author import Config, Loader, Model, uses_components


def _clip(width: int, layers: int) -> nn.Module:
    encoder = nn.Module()
    encoder.embeddings = nn.Embedding(4096, width, dtype=torch.bfloat16)  # type: ignore[assignment]
    encoder.layers = nn.ModuleList(  # type: ignore[assignment]
        [
            nn.TransformerEncoderLayer(width, 4, width * 4, batch_first=True, dtype=torch.bfloat16)
            for _ in range(layers)
        ]
    )
    encoder.final_norm = nn.LayerNorm(width, dtype=torch.bfloat16)  # type: ignore[assignment]
    return encoder


def _unet(width: int, blocks: int) -> nn.Module:
    unet = nn.Module()
    unet.time_embedding = nn.Linear(320, width, dtype=torch.bfloat16)  # type: ignore[assignment]
    unet.down = nn.ModuleList(  # type: ignore[assignment]
        [nn.Conv2d(width, width, 3, padding=1, dtype=torch.bfloat16) for _ in range(blocks)]
    )
    unet.mid_attn = nn.MultiheadAttention(width, 8, batch_first=True, dtype=torch.bfloat16)  # type: ignore[assignment]
    unet.up = nn.ModuleList(  # type: ignore[assignment]
        [nn.Conv2d(width, width, 3, padding=1, dtype=torch.bfloat16) for _ in range(blocks)]
    )
    # the rope-style table every real UNet carries and no checkpoint stores
    unet.register_buffer("position_ids", torch.zeros(1, 77, dtype=torch.int64), persistent=False)
    return unet


def _vae(width: int) -> nn.Module:
    vae = nn.Module()
    vae.encoder = nn.Conv2d(3, width, 3, padding=1, dtype=torch.bfloat16)  # type: ignore[assignment]
    # the NaN-prone fp32 island real SDXL VAEs pin
    vae.decoder = nn.Conv2d(width, 3, 3, padding=1, dtype=torch.float32)  # type: ignore[assignment]
    vae.register_buffer("scaling", torch.zeros(1, dtype=torch.float32))
    return vae


class SdxlPipeline:
    """Four heavyweight components behind one `__call__` — no role scoping, no variants."""

    def __init__(self, config: Config) -> None:
        width = config.as_int("width", 64)
        layers = config.as_int("layers", 2)
        blocks = config.as_int("blocks", 2)
        self.components: dict[str, object] = {
            "text_encoder": _clip(width, layers),
            "text_encoder_2": _clip(width * 2, layers),
            "unet": _unet(width, blocks),
            "vae": _vae(width),
        }


def build_sdxl(config: Config) -> SdxlPipeline:
    return SdxlPipeline(config)


class SdxlModel(Model[SdxlPipeline]):
    """No class keyword: SDXL admits by pure tensor-schema satisfaction (§1.1)."""

    pipe: SdxlPipeline

    def load(self, loader: Loader) -> None:
        self.pipe = loader.construct(SdxlPipeline, factory=build_sdxl)

    @uses_components("text_encoder", "text_encoder_2", "unet", "vae")
    def generate(self, steps: int) -> int:
        return steps


#: The conservative whole-pipeline declaration: one method, every component.
COMPONENT_USE: Mapping[str, tuple[str, ...]] = {
    "generate": ("text_encoder", "text_encoder_2", "unet", "vae")
}

#: The planted under-declaration — three components on a four-component pipeline.
UNDER_DECLARED: Mapping[str, tuple[str, ...]] = {"generate": ("text_encoder", "unet", "vae")}
