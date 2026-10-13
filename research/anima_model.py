"""Shared construction of the exact Anima Cosmos transformer for compile research.

Weights are random: Inductor's decisions are a function of shape/dtype/graph, never of
values. Numerical A/B checks compare eager and compiled on the SAME random weights.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

CONFIG: dict[str, Any] = json.loads(
    Path(__file__).with_name("anima_transformer_config.json").read_text()
)

# 1024x1024 output / VAE 8x downsample -> 128x128 latent; patch (1,2,2) -> 64x64 = 4096 tokens.
LATENT_HW = 128
TEXT_TOKENS = 512


def build(num_layers: int | None = None, device: str = "cuda", dtype: Any = torch.bfloat16) -> Any:
    from diffusers.models.transformers.transformer_cosmos import CosmosTransformer3DModel

    cfg = dict(CONFIG)
    cfg.pop("_class_name", None)
    cfg.pop("_diffusers_version", None)
    if num_layers is not None:
        cfg["num_layers"] = num_layers
    with torch.device("meta"):
        model = CosmosTransformer3DModel(**cfg)
    model = model.to(dtype).to_empty(device=device)
    generator = torch.Generator(device=device).manual_seed(0)
    with torch.no_grad():
        for tensor in list(model.parameters()) + list(model.buffers()):
            tensor.normal_(0.0, 0.02, generator=generator)
    return model.eval()


def inputs(
    latent_hw: int = LATENT_HW,
    batch: int = 1,
    device: str = "cuda",
    dtype: Any = torch.bfloat16,
    seed: int = 1,
) -> dict[str, Any]:
    generator = torch.Generator(device=device).manual_seed(seed)

    def randn(*shape: int) -> Any:
        return torch.randn(*shape, device=device, dtype=dtype, generator=generator)

    return {
        "hidden_states": randn(batch, 16, 1, latent_hw, latent_hw),
        "timestep": torch.full((batch,), 500.0, device=device, dtype=dtype),
        "encoder_hidden_states": randn(batch, TEXT_TOKENS, CONFIG["text_embed_dim"]),
        # The Anima plan's `latent_sized_zero_padding_mask`: already latent-sized, so the
        # torchvision resize inside forward is the identity.
        "padding_mask": torch.zeros(batch, 1, latent_hw, latent_hw, device=device, dtype=dtype),
        "return_dict": False,
    }


def apply_fused_plan(model: Any, via: str = "raw", strict: bool = True) -> None:
    """Install the shipping Anima execution plan on a bare transformer (no pipeline).

    `via="raw"` calls the nanobind entry point exactly as the shipping plan does.
    `via="op"` routes the same kernel through the `cozy_runtime::rms_rope_split_half`
    custom operator, which is the only difference Dynamo can see.
    """
    from cozy_runtime.internal import anima_optimization as plan

    if via == "op":
        from cozy_runtime._kernels.ops import operator

        fn = operator()
    else:
        from cozy_runtime._kernels import rms_rope_split_half as fn

    class _Kernels:
        rms_rope_split_half = staticmethod(fn)

    if strict:
        plan._install_transformer_plan(model, anima_kernels=_Kernels, torch=torch)
        return
    # Same installs as `_install_transformer_plan`, minus the 28-block / 112-norm identity
    # fence, so the plan can be studied on a truncated model that still has the exact
    # per-block shape.
    from types import MethodType

    from diffusers.models.normalization import RMSNorm

    rope = plan._RopeMatrix(torch)
    for module in model.modules():
        if type(module) is RMSNorm:
            module.forward = MethodType(plan._native_rms_forward, module)
    for block in model.transformer_blocks:
        plan._install_block_plan(block, anima_kernels=_Kernels, torch=torch, rope=rope)
    model.rope.forward = MethodType(plan._direct_rope_forward, model.rope)
    original_forward = model.forward

    def forward(_self: Any, *args: object, **kwargs: object) -> Any:
        rope.clear()
        try:
            return original_forward(*args, **kwargs)
        finally:
            rope.clear()

    model.forward = MethodType(forward, model)
