"""Preparation-only LoRA arithmetic; storage transactions and cache identity belong to callers.

This is a deterministic rank-ordered reference for a future prepared variant,
not a serving hook or a selected production implementation. A native TensorFS
recipe must include its code/algorithm identity, exact base encoding/parts and
the ordered adapter parts, strengths and alpha values. Do not key by a mutable
release name, round strengths, sort adapters or coalesce repeated adapters.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any

from cozy_runtime.internal.encoding.formats import SPEC_ROWWISE, SPEC_ROWWISE_KEEPDIM

ALGORITHM = "cozy.lora.rowwise-fp8.rank-ordered-f32/1"


@dataclass(frozen=True)
class MergeAdapter:
    """Canonical A[rank, in], B[out, rank]; FP16/BF16 promote losslessly to FP32."""

    a: Any
    b: Any
    strength: float = 1.0
    alpha: float | None = None


@dataclass(frozen=True)
class MergeError:
    """Error of the re-encoded weight against the merged FP32 weight, not video quality."""

    squared_error: float
    squared_reference: float
    max_abs: float
    changed_scale_rows: int


@dataclass(frozen=True)
class MergedRows:
    start: int
    data: Any
    scale: Any
    error: MergeError


def _finite(torch: Any, value: Any, count: int, label: str) -> None:
    for start in range(0, value.numel(), count):
        if not bool(torch.isfinite(value.reshape(-1)[start : start + count]).all()):
            raise ValueError(f"{label} contains nonfinite values")


def merge_rowwise_fp8(
    data: Any,
    scale: Any,
    adapters: Sequence[MergeAdapter],
    *,
    encoding_spec: str,
    scale_policy: str,
    max_scratch_bytes: int = 64 << 20,
    tile_rows: int = 128,
) -> Iterator[MergedRows] | None:
    """Yield new rowwise E4M3 parts, or None to graft the unchanged base parts.

    Inputs remain immutable and must remain alive and unchanged during iteration.
    A caller commits a new TensorFS artifact only after consuming every tile;
    later invalid base rows or arithmetic overflow can still refuse iteration.
    No full decoded weight or output weight is allocated here. Scratch excludes
    caller-owned inputs, retained output tiles and allocator bookkeeping/cache.

    Shape/dtype/spec/coefficient validation precedes allocation. Every adapter's
    factors are checked for finiteness, including zero-strength adapters. None
    means the base payload is inherited without reading or re-encoding it; the
    caller retains responsibility for that unchanged base's existing validity.
    Row scales are always validated. Only explicitly reviewed rowwise encodings
    are accepted: a scalar-scaled source header must not be mislabeled rowwise.

    Scale policy is explicit: preserve retains the original grid and refuses overflow;
    grow only enlarges scales when required; recalibrate recomputes every row grid.
    The latter can introduce base quantization error even with a tiny LoRA update.

    FP32 rank-ordered multiplication/addition avoids changing process-global
    TF32 settings or relying on GEMM's shape-dependent reduction order. It is a
    correctness control; compare preparation cost with strict-FP32 GEMM before
    selecting a production implementation. CUDA remains unqualified until tested.
    """
    import torch

    if type(max_scratch_bytes) is not int or max_scratch_bytes < 16:
        raise ValueError("max_scratch_bytes must be an integer of at least 16")
    if type(tile_rows) is not int or tile_rows < 1:
        raise ValueError("tile_rows must be a positive integer")
    if scale_policy not in ("preserve", "grow", "recalibrate"):
        raise ValueError("scale_policy must explicitly be preserve, grow or recalibrate")
    if encoding_spec not in (SPEC_ROWWISE, SPEC_ROWWISE_KEEPDIM):
        raise ValueError("LoRA merge requires an explicitly reviewed rowwise FP8 encoding")
    if data.ndim != 2 or min(data.shape) < 1 or data.dtype != torch.float8_e4m3fn:
        raise ValueError("rowwise data must be nonempty E4M3 [out, in]")
    out_features, in_features = data.shape
    expected_scale = (out_features,) if encoding_spec == SPEC_ROWWISE else (out_features, 1)
    if scale.shape != expected_scale or scale.dtype != torch.float32:
        raise ValueError("FP32 scale shape disagrees with the exact rowwise encoding")
    if scale.device != data.device or not data.is_contiguous() or not scale.is_contiguous():
        raise ValueError("base parts must be contiguous on one device")
    active: list[tuple[MergeAdapter, float]] = []
    max_rank = 0
    for adapter in adapters:
        a, b = adapter.a, adapter.b
        if a.ndim != 2 or b.ndim != 2 or a.shape[0] < 1:
            raise ValueError("LoRA factors must be nonempty A[rank, in] and B[out, rank]")
        rank = a.shape[0]
        if a.shape[1] != in_features or b.shape != (out_features, rank):
            raise ValueError("LoRA factor orientation or shape does not match the base weight")
        for factor in (a, b):
            if factor.dtype not in (torch.float16, torch.bfloat16, torch.float32):
                raise ValueError("LoRA factors must be FP16, BF16 or FP32")
            if factor.device != data.device or not factor.is_contiguous():
                raise ValueError("LoRA factors must be contiguous on the base device")
        alpha = rank if adapter.alpha is None else adapter.alpha
        if not math.isfinite(adapter.strength) or not math.isfinite(alpha):
            raise ValueError("LoRA strength and alpha must be finite")
        coefficient = adapter.strength * (alpha / rank)
        if not math.isfinite(coefficient) or abs(coefficient) > torch.finfo(torch.float32).max:
            raise ValueError("LoRA coefficient is not representable in FP32")
        if 0 < abs(coefficient) < 2.0**-149:
            raise ValueError("LoRA coefficient underflows FP32")
        if coefficient:
            active.append((adapter, coefficient))
            max_rank = max(max_rank, rank)

    # Conservative tensor scratch bound: three FP32 arithmetic tiles, E4M3
    # output and FP64 statistics; one promoted A and B tile at a time. Factor
    # finiteness checks are separately bounded and precede all those buffers.
    fixed = 4096 + 4 * max_rank * in_features
    per_row = 20 * in_features + 4 * max_rank + 64
    rows = min(tile_rows, out_features, (max_scratch_bytes - fixed) // per_row)
    if active and rows < 1:
        raise ValueError("scratch budget cannot hold one merge row and its FP32 A factor")
    check_elements = min(65536, max_scratch_bytes // 16)
    _finite(torch, scale, check_elements, "base scales")
    for adapter in adapters:
        _finite(torch, adapter.a, check_elements, "LoRA A")
        _finite(torch, adapter.b, check_elements, "LoRA B")
    if not active:
        return None
    return _merge_tiles(torch, data.detach(), scale.detach(), active, rows, scale_policy)


def _sum_squares(value: Any, torch: Any) -> float:
    # Squaring FP32 first can overflow even when both weights are finite.
    return float(value.to(torch.float64).square_().sum())


def _merge_tiles(
    torch: Any,
    data: Any,
    scale: Any,
    active: Sequence[tuple[MergeAdapter, float]],
    rows: int,
    scale_policy: str,
) -> Iterator[MergedRows]:
    for start in range(0, data.shape[0], rows):
        end = min(start + rows, data.shape[0])
        merged = data[start:end].to(torch.float32)
        merged.mul_(scale[start:end].reshape(-1, 1))
        if not bool(torch.isfinite(merged).all()):
            raise ValueError("decoded base weight contains nonfinite values")
        delta, product = torch.empty_like(merged), torch.empty_like(merged)
        for adapter, coefficient in active:
            a = adapter.a.detach().to(torch.float32)
            b = adapter.b[start:end].detach().to(torch.float32)
            delta.zero_()
            for rank in range(a.shape[0]):
                torch.mul(b[:, rank, None], a[rank, None, :], out=product)
                delta.add_(product)
            delta.mul_(coefficient)
            merged.add_(delta)
            del a, b
        del delta, product
        if not bool(torch.isfinite(merged).all()):
            raise ValueError("FP32 LoRA merge overflowed or produced nonfinite values")
        peak = merged.abs().amax(dim=1)
        original_scale = scale[start:end].reshape(-1)
        if scale_policy in ("preserve", "grow") and not bool((original_scale > 0).all()):
            raise ValueError("preserve/grow requires strictly positive original row scales")
        if scale_policy == "preserve":
            scales = original_scale.clone()
        else:
            scales = peak.div(448.0)
            if scale_policy == "grow":
                scales = torch.maximum(original_scale, scales)
            else:
                scales.clamp_min_(torch.finfo(torch.float32).tiny)
                scales[peak == 0] = 1.0
        normalized = merged / scales[:, None]
        if scale_policy == "preserve":
            overflow_rows = int((normalized.abs() > 448.0).any(dim=1).sum())
            if overflow_rows:
                raise ValueError(
                    f"original FP8 scales overflow in {overflow_rows} rows of tile at {start}"
                )
        codes = normalized.clamp_(-448.0, 448.0).to(torch.float8_e4m3fn)
        del normalized
        error = codes.to(torch.float32).mul_(scales[:, None])
        if not bool(torch.isfinite(scales).all()) or not bool(torch.isfinite(error).all()):
            raise ValueError("FP8 re-encoding produced nonfinite reconstructed weights or scales")
        error.sub_(merged)
        stats = MergeError(
            squared_error=_sum_squares(error, torch),
            squared_reference=_sum_squares(merged, torch),
            max_abs=float(error.abs().amax()),
            changed_scale_rows=int((scales != original_scale).sum()),
        )
        if scale.ndim == 2:
            scales = scales.reshape(-1, 1)
        del merged, error, peak
        yield MergedRows(start, codes, scales, stats)
        del codes, scales
