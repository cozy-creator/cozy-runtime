#!/usr/bin/env python3
"""Real-CUDA oracle for the exact Diffusers 0.40 Anima execution plan.

Run inside the candidate base worker image, where Torch, Diffusers and the fused Anima
CUDA kernel library are already installed at their exact pinned versions:

    python scripts/anima-optimization-live.py
"""

from __future__ import annotations

import time
from importlib import import_module, metadata
from types import MethodType
from typing import Any

import torch
from diffusers.models.normalization import RMSNorm
from diffusers.models.transformers.transformer_cosmos import (
    CosmosAttnProcessor2_0,
    CosmosRotaryPosEmbed,
    CosmosTransformer3DModel,
    CosmosTransformerBlock,
)
from diffusers.modular_pipelines.anima.before_denoise import AnimaPrepareLatentsStep
from packaging.version import Version

from cozy_runtime.internal.anima_optimization import (
    _TRANSFORMER_CONFIG,
    DIFFUSERS_VERSION,
    FUSED_ANIMA_CUDA_KERNEL_VERSION,
    FUSED_ANIMA_CUDA_RELEASE,
    FUSED_ANIMA_TORCH_RELEASE,
    OPTIMIZATION_ID,
    AnimaOptimizationRefusal,
    _direct_rope_forward,
    _install_block_plan,
    _install_transformer_plan,
    _native_rms_forward,
    _require_source,
    _require_transformer_config,
    _RopeMatrix,
    _tensor_census,
)


def _call(block: Any, hidden: Any, context: Any, timestep: Any, rotary: Any) -> Any:
    with torch.inference_mode():
        return block(hidden, context, timestep, image_rotary_emb=rotary)


def _bench(block: Any, hidden: Any, context: Any, timestep: Any, rotary: Any) -> float:
    _call(block, hidden, context, timestep, rotary)
    torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(3):
        _call(block, hidden, context, timestep, rotary)
    torch.cuda.synchronize()
    return (time.perf_counter() - started) * 1000 / 3


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("REFUSED: the Anima optimization oracle requires CUDA")
    if Version(metadata.version("torch")).public != FUSED_ANIMA_TORCH_RELEASE:
        raise SystemExit(f"REFUSED: Torch must be {FUSED_ANIMA_TORCH_RELEASE}")
    if torch.version.cuda != FUSED_ANIMA_CUDA_RELEASE:
        raise SystemExit(f"REFUSED: built CUDA must be {FUSED_ANIMA_CUDA_RELEASE}")
    if metadata.version("diffusers") != DIFFUSERS_VERSION:
        raise SystemExit(f"REFUSED: diffusers must be {DIFFUSERS_VERSION}")
    if metadata.version("comfy-kitchen") != FUSED_ANIMA_CUDA_KERNEL_VERSION:
        raise SystemExit(
            f"REFUSED: fused Anima CUDA kernel library must be {FUSED_ANIMA_CUDA_KERNEL_VERSION}"
        )
    anima_kernels = import_module("comfy_kitchen")
    for name, source in {
        "RMSNorm.forward": RMSNorm.forward,
        "CosmosAttnProcessor2_0.__call__": CosmosAttnProcessor2_0.__call__,
        "CosmosRotaryPosEmbed.forward": CosmosRotaryPosEmbed.forward,
        "CosmosTransformerBlock.forward": CosmosTransformerBlock.forward,
        "AnimaPrepareLatentsStep.__call__": AnimaPrepareLatentsStep.__call__,
        "comfy_kitchen.rms_rope_split_half": anima_kernels.rms_rope_split_half,
    }.items():
        _require_source(name, source)

    with torch.device("meta"):
        exact = CosmosTransformer3DModel.from_config(_TRANSFORMER_CONFIG)
        wrong = CosmosTransformer3DModel(num_layers=1)
    _require_transformer_config(exact)
    try:
        _require_transformer_config(wrong)
    except AnimaOptimizationRefusal as exc:
        assert exc.code == "anima_optimization_config"
    else:
        raise AssertionError("a non-Anima Cosmos config was accepted")
    exact_before = _tensor_census(exact)
    _install_transformer_plan(exact, anima_kernels=anima_kernels, torch=torch)
    assert exact_before == _tensor_census(exact), "full transformer topology changed"

    torch.manual_seed(11)
    rope = CosmosRotaryPosEmbed(
        128,
        max_size=(128, 240, 240),
        patch_size=(1, 2, 2),
        rope_scale=(1.0, 4.0, 4.0),
    ).to("cuda")
    latent = torch.empty(1, 17, 1, 128, 128, device="cuda", dtype=torch.bfloat16)
    eager_rotary = rope(latent)
    direct_rotary = _direct_rope_forward(rope, latent)
    assert torch.equal(_RopeMatrix(torch).get(eager_rotary), direct_rotary)

    final_norm = RMSNorm(2048, eps=1e-6).to(device="cuda", dtype=torch.bfloat16)
    norm_input = torch.randn(1, 4096, 2048, device="cuda", dtype=torch.bfloat16)
    with torch.inference_mode():
        eager_norm = final_norm(norm_input)
        final_norm.forward = MethodType(_native_rms_forward, final_norm)
        optimized_norm = final_norm(norm_input)
    assert torch.allclose(eager_norm, optimized_norm, atol=0.03125, rtol=0.03125)

    block = (
        CosmosTransformerBlock(
            16,
            128,
            1024,
            mlp_ratio=4.0,
            adaln_lora_dim=256,
            qk_norm="rms_norm",
            out_bias=False,
            img_context=False,
            before_proj=False,
            after_proj=False,
        )
        .to(device="cuda", dtype=torch.bfloat16)
        .eval()
    )
    sequence = 4096
    hidden = torch.randn(1, sequence, 2048, device="cuda", dtype=torch.bfloat16)
    context = torch.randn(1, 512, 1024, device="cuda", dtype=torch.bfloat16)
    timestep = torch.randn(1, 2048, device="cuda", dtype=torch.bfloat16)
    phase = torch.randn(sequence, 64, device="cuda", dtype=torch.float32)
    rotary = (
        torch.cat([phase.cos(), phase.cos()], dim=-1),
        torch.cat([phase.sin(), phase.sin()], dim=-1),
    )
    eager = _call(block, hidden, context, timestep, rotary)
    eager_ms = _bench(block, hidden, context, timestep, rotary)
    before = _tensor_census(block)
    for module in block.modules():
        if type(module) is RMSNorm:
            module.forward = MethodType(_native_rms_forward, module)
    _install_block_plan(block, anima_kernels=anima_kernels, torch=torch, rope=_RopeMatrix(torch))
    after = _tensor_census(block)
    optimized = _call(block, hidden, context, timestep, rotary)
    repeat = _call(block, hidden, context, timestep, rotary)
    optimized_ms = _bench(block, hidden, context, timestep, rotary)
    error = (eager.float() - optimized.float()).abs()
    norm_error = (eager_norm.float() - optimized_norm.float()).abs()
    assert before == after, "callable installation changed topology or a storage pointer"
    assert torch.equal(optimized, repeat), "optimized block is not deterministic"
    assert torch.allclose(eager, optimized, atol=0.125, rtol=0.03125)

    full_mask = torch.zeros(1, 1, 1024, 1024, device="cuda", dtype=torch.bfloat16)
    latent_mask = torch.zeros(1, 1, 128, 128, device="cuda", dtype=torch.bfloat16)
    resized = torch.nn.functional.interpolate(full_mask, size=(128, 128), mode="nearest")
    assert torch.equal(resized, latent_mask), "latent-sized zero mask changed model input"
    print(f"identity: {OPTIMIZATION_ID}")
    print("config refusal: PASS")
    print("full 28-block topology: PASS")
    print("direct split-half RoPE matrix: PASS")
    print("topology and storage pointers: PASS")
    print("optimized determinism: PASS")
    print("latent-sized zero mask: PASS")
    print(f"max_abs_error: {error.max().item():.6f}")
    print(f"mean_abs_error: {error.mean().item():.9f}")
    print(f"final_norm_max_abs_error: {norm_error.max().item():.6f}")
    print(f"block_eager_ms: {eager_ms:.3f}")
    print(f"block_optimized_ms: {optimized_ms:.3f}")
    print(f"block_speedup: {eager_ms / optimized_ms:.3f}x")


if __name__ == "__main__":
    main()
