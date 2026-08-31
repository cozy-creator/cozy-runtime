"""Cozy's purpose-built CUDA kernels for the Torch 2.13/CUDA 13 worker profile."""

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from importlib import import_module
from typing import Any


def rms_rope_split_half(
    q: Any,
    k: Any,
    freqs: Any,
    q_scale: Any,
    k_scale: Any,
    epsilon: float = 1e-6,
    rot_dim: int = 0,
) -> tuple[Any, Any]:
    """Return paired out-of-place RMS-normalized split-half RoPE tensors."""
    import torch

    tensors = (q, k, freqs, q_scale, k_scale)
    if not all(isinstance(value, torch.Tensor) and value.is_cuda for value in tensors):
        raise ValueError("RMSNorm+RoPE inputs must all be CUDA tensors")
    if any(value.device != q.device for value in tensors[1:]):
        raise ValueError("RMSNorm+RoPE inputs must occupy one CUDA device")
    q_out = torch.empty_like(q)
    k_out = torch.empty_like(k)
    stream = torch.cuda.current_stream(q.device).cuda_stream
    native = import_module("cozy_runtime_cuda_kernels._C")
    native.rms_rope_split_half(
        q,
        k,
        freqs,
        q_scale,
        k_scale,
        q_out,
        k_out,
        epsilon,
        stream,
        rot_dim,
    )
    return q_out, k_out


__all__ = ["rms_rope_split_half"]
