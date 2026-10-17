"""Cozy's purpose-built CUDA kernels for package authors: the cozy-kernels wheel's `_norm`
(nanobind + DLPack over the CUDA runtime, one wheel per OS and CPU). Callers probe
`author.kernels.native()` before use.
"""

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

try:
    from cozy_kernels import _norm

    ABSENT = ""
except ImportError as absent:  # no cozy-kernels wheel for this OS and CPU
    ABSENT = f"no cozy-kernels wheel is installed ({absent})"


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
    _norm.rms_rope_split_half(
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
