"""Thin dense-attention adapters; kernels and their numerical algorithms are upstream.

All entrypoints use Diffusers' [batch, sequence, heads, dimension] convention.
Imports are lazy so optional GPU libraries never become Runtime base dependencies.
These are explicit experimental choices, not automatic quality-qualified defaults.
"""

from __future__ import annotations

import math
from importlib import import_module
from typing import Any

KITCHEN_BACKEND = "_cozy_kitchen_int8"
FLASHINFER_BACKEND = "_cozy_flashinfer_bf16_fp8"


def _inference_options(
    dropout_p: float,
    is_causal: bool,
    scale: float | None,
    return_lse: bool,
    parallel_config: Any,
) -> None:
    if dropout_p != 0.0:
        raise ValueError("this inference attention backend does not support dropout")
    if is_causal:
        raise ValueError("this attention adapter currently serves noncausal attention only")
    if scale is not None and not math.isfinite(scale):
        raise ValueError("attention scale must be finite")
    if return_lse:
        raise ValueError("this attention adapter does not return log-sum-exp")
    if parallel_config is not None:
        raise ValueError("this attention adapter has not qualified context parallelism")


def _inputs(query: Any, key: Any, value: Any, *, mixed: bool, enable_gqa: bool) -> None:
    import torch

    if any(not isinstance(x, torch.Tensor) for x in (query, key, value)):
        raise ValueError("attention Q/K/V must be tensors")
    if any(x.ndim != 4 for x in (query, key, value)):
        raise ValueError("attention Q/K/V must have shape [batch, sequence, heads, dimension]")
    if query.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError("attention Q/K/V must be BF16 or FP16")
    if key.dtype != query.dtype or value.dtype != query.dtype:
        raise ValueError("attention Q/K/V must have identical input dtypes")
    if mixed and query.dtype != torch.bfloat16:
        raise ValueError("flashinfer-bf16-fp8 preserves BF16 Q/K and requires BF16 inputs")
    batch, sequence, heads, dim = query.shape
    if min(batch, sequence, heads, dim, *key.shape, *value.shape) <= 0:
        raise ValueError("attention tensor dimensions must be positive")
    if (
        key.shape != value.shape
        or key.shape[0] != batch
        or key.shape[3] != dim
        or heads % key.shape[2]
    ):
        raise ValueError("attention needs matching K/V and compatible batch/head dimensions")
    if heads != key.shape[2] and not enable_gqa:
        raise ValueError("unequal Q/KV head counts require enable_gqa=True")
    if (mixed and dim != 128) or dim > 256:
        raise ValueError("FlashInfer mixed attention needs D128; Kitchen supports D1 through D256")
    if query.device.type != "cuda" or key.device != query.device or value.device != query.device:
        raise ValueError("attention Q/K/V must be on the same CUDA device")
    if getattr(torch.version, "hip", None):
        raise ValueError("these attention adapters currently admit NVIDIA CUDA devices only")
    capability = torch.cuda.get_device_capability(query.device)
    admitted = ((10, 0), (10, 3)) if mixed else ((9, 0), (10, 0), (10, 3))
    if capability not in admitted:
        raise ValueError(f"this attention adapter admits {admitted}, not {capability}")
    if torch.is_grad_enabled() and any(x.requires_grad for x in (query, key, value)):
        raise ValueError("this attention adapter is inference-only")


def _output(output: Any, query: Any) -> Any:
    import torch

    if (
        not isinstance(output, torch.Tensor)
        or output.shape != query.shape
        or output.dtype != query.dtype
        or output.device != query.device
    ):
        raise ValueError("upstream attention must return the query shape, dtype and device")
    return output


def kitchen_int8_attention(
    query: Any,
    key: Any,
    value: Any,
    attn_mask: Any = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: float | None = None,
    enable_gqa: bool = False,
    return_lse: bool = False,
    _parallel_config: Any = None,
) -> Any:
    """Upstream Kitchen INT8 Q/K/V/P with FP32 accumulation, preserving masks/scale.

    Kitchen owns GQA, range reduction, quantization and the kernel. Layout changes
    are views when the input's head dimension is contiguous (including packed QKV).
    Unequal head counts require an explicit ``enable_gqa=True``.
    """
    _inference_options(dropout_p, is_causal, scale, return_lse, _parallel_config)
    _inputs(query, key, value, mixed=False, enable_gqa=enable_gqa)
    kitchen = import_module("comfy_kitchen")
    tensors = [x if x.stride(-1) == 1 else x.contiguous() for x in (query, key, value)]
    output = kitchen.int8_attention(
        *(x.transpose(1, 2) for x in tensors), scale=scale, attn_mask=attn_mask
    )
    # Check the public HND result before attempting its layout conversion.
    return _output(output, query.transpose(1, 2)).transpose(1, 2)


def _quantize_value(value: Any) -> tuple[Any, float]:
    """Current scalar E4M3 scale required by FlashInfer's mixed FMHA public API.

    That API accepts a Python scalar, so ``item`` synchronizes CUDA. Include the
    reduction, conversion and synchronization in complete-call benchmarks. Q/K
    are never cast. This uses PyTorch arithmetic, not another custom GPU kernel.
    """
    import torch

    maximum = float(value.abs().amax().item())
    if not math.isfinite(maximum):
        raise ValueError("FlashInfer V contains non-finite values")
    descale = max(maximum / 448.0, 1e-12)
    scaled = value.to(dtype=torch.float32, copy=True)
    quantized = scaled.div_(descale).clamp_(-448.0, 448.0).to(torch.float8_e4m3fn)
    return quantized.contiguous(), descale


def flashinfer_mixed_attention(
    query: Any,
    key: Any,
    value: Any,
    attn_mask: Any = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: float | None = None,
    enable_gqa: bool = False,
    return_lse: bool = False,
    _parallel_config: Any = None,
) -> Any:
    """Upstream Blackwell CuTe FMHA: BF16 Q/K, E4M3 V/P, BF16 output.

    A batch becomes equal-length ragged segments; sequence boundaries remain
    separate. The public API can JIT a missing variant. No SkipSoftmax, window,
    sink, scale history or context-parallel approximation is enabled.
    """
    _inference_options(dropout_p, is_causal, scale, return_lse, _parallel_config)
    if attn_mask is not None:
        raise ValueError("FlashInfer mixed attention does not support an attention mask")
    _inputs(query, key, value, mixed=True, enable_gqa=enable_gqa)
    import torch

    fmha = import_module("flashinfer.attention.cute_dsl.fmha")
    batch, q_length, q_heads, dim = query.shape
    kv_length, kv_heads = key.shape[1:3]
    if batch * max(q_length, kv_length) >= 2**31:
        raise ValueError("FlashInfer ragged sequence offsets exceed int32")
    q = query.reshape(batch * q_length, q_heads, dim).contiguous()
    k = key.reshape(batch * kv_length, kv_heads, dim).contiguous()
    v, descale = _quantize_value(value.reshape(batch * kv_length, kv_heads, dim))
    output = torch.empty_like(q)
    offsets = torch.arange(batch + 1, device=query.device, dtype=torch.int32)
    fmha.cute_dsl_fmha_ragged_prefill(
        q,
        k,
        v,
        output,
        offsets * q_length,
        offsets * kv_length,
        is_causal=False,
        sm_scale=scale,
        scale_q=1.0,
        scale_k=1.0,
        scale_v=descale,
        scale_o=1.0,
        max_qo_len=q_length,
        max_kv_len=kv_length,
        window_left=-1,
        window_right=-1,
        attention_sinks=None,
        lse=None,
        skip_softmax_threshold_scale_factor=None,
        enable_pdl=False,
    )
    return _output(output.reshape(batch, q_length, q_heads, dim), query)
