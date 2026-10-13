"""The exact SDXL UNet (fp16) for compile research; random weights, real config."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

CONFIG: dict[str, Any] = json.loads(Path(__file__).with_name("sdxl_unet_config.json").read_text())

LATENT_HW = 128  # 1024px / 8
TEXT_TOKENS = 77
BATCH = 2  # classifier-free guidance: the shape the pipeline actually runs


def build(
    layers: int | None = None,
    transformer_layers: list[int] | None = None,
    device: str = "cuda",
    dtype: Any = torch.float16,
) -> Any:
    """The real SDXL UNet config. `layers`/`transformer_layers` shrink the DEPTH only —
    every block type, channel width and resolution stays exactly SDXL's, which is what the
    channels_last question is about. Use them only when 5 GiB of fp16 weights will not fit
    beside another tenant, and say so in the measurement's conditions."""
    from diffusers import UNet2DConditionModel

    cfg = {k: v for k, v in CONFIG.items() if not k.startswith("_")}
    if layers is not None:
        cfg["layers_per_block"] = layers
    if transformer_layers is not None:
        cfg["transformer_layers_per_block"] = transformer_layers
    with torch.device("meta"):
        model = UNet2DConditionModel.from_config(cfg)
    model = model.to(dtype).to_empty(device=device)
    generator = torch.Generator(device=device).manual_seed(0)
    with torch.no_grad():
        for tensor in list(model.parameters()) + list(model.buffers()):
            if tensor.is_floating_point():
                tensor.normal_(0.0, 0.02, generator=generator)
            else:
                tensor.zero_()
    return model.eval()


def inputs(
    latent_hw: int = LATENT_HW,
    batch: int = BATCH,
    device: str = "cuda",
    dtype: Any = torch.float16,
    seed: int = 1,
) -> dict[str, Any]:
    generator = torch.Generator(device=device).manual_seed(seed)

    def randn(*shape: int) -> Any:
        return torch.randn(*shape, device=device, dtype=dtype, generator=generator)

    return {
        "sample": randn(batch, 4, latent_hw, latent_hw),
        "timestep": torch.tensor(500.0, device=device),
        "encoder_hidden_states": randn(batch, TEXT_TOKENS, CONFIG["cross_attention_dim"]),
        "added_cond_kwargs": {
            "text_embeds": randn(batch, 1280),
            "time_ids": randn(batch, 6),
        },
        "return_dict": False,
    }
