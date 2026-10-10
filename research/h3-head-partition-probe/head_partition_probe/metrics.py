"""Exact metric functions from the qualified row-partition probe."""

from __future__ import annotations

import math
from typing import Any


def new_metric() -> dict[str, Any]:
    return {"elements": 0, "totals": None, "maximum": None, "first": None}


def accumulate(torch: Any, result: dict[str, Any], actual: Any, reference: Any) -> None:
    if actual.shape != reference.shape or actual.dtype != reference.dtype:
        raise ValueError("Comparison requires identical geometry and dtype")
    if result["totals"] is None:
        result["totals"] = torch.zeros(5, dtype=torch.float64, device=actual.device)
        result["maximum"] = torch.zeros((), dtype=torch.float64, device=actual.device)
        result["first"] = torch.full((), math.inf, dtype=torch.float64, device=actual.device)
    columns = math.prod(actual.shape[1:])
    block_rows = max(1, min(512, (8 << 20) // columns))
    for start in range(0, actual.shape[0], block_rows):
        a = actual[start : start + block_rows].contiguous()
        b = reference[start : start + block_rows].contiguous()
        different = (
            (a.view(torch.uint8) != b.view(torch.uint8))
            .reshape(a.numel(), a.element_size())
            .any(dim=1)
        )
        count = different.sum()
        first = different.to(torch.uint8).argmax() + result["elements"]
        result["first"] = torch.minimum(
            result["first"], torch.where(count > 0, first.to(torch.float64), math.inf)
        )
        finite = torch.isfinite(a.float()) & torch.isfinite(b.float())
        delta = torch.where(finite, a.float() - b.float(), 0.0)
        ref = torch.where(finite, b.float(), 0.0)
        values = (
            count,
            (~finite).sum(),
            delta.abs().sum(),
            delta.square().sum(),
            ref.square().sum(),
        )
        result["totals"].add_(torch.stack([value.to(torch.float64) for value in values]))
        result["maximum"] = torch.maximum(result["maximum"], delta.abs().max().to(torch.float64))
        result["elements"] += a.numel()


def finish(result: dict[str, Any]) -> dict[str, Any]:
    if not result["elements"]:
        raise ValueError("No elements were compared")
    import torch

    # One bounded scalar download per completed metric, not per comparison slice.
    values = (
        torch.cat([result["totals"], result["maximum"].reshape(1), result["first"].reshape(1)])
        .cpu()
        .tolist()
    )
    count, nonfinite, absolute, squared, reference, maximum, first = values
    return {
        "elements": result["elements"],
        "different_elements": int(count),
        "first_different_element": int(first) if count else None,
        "nonfinite_elements": int(nonfinite),
        "max_absolute_difference": maximum,
        "absolute_error_sum": absolute,
        "squared_error_sum": squared,
        "reference_squared_sum": reference,
        "bytes_equal": count == 0,
        "mean_absolute_difference": absolute / result["elements"],
        "relative_l2": math.sqrt(squared / reference) if reference else None,
    }


def compared(torch: Any, actual: Any, reference: Any) -> dict[str, Any]:
    result = new_metric()
    accumulate(torch, result, actual, reference)
    return finish(result)


def operand_layout(value: Any) -> dict[str, Any]:
    return {
        "shape": list(value.shape),
        "stride": list(value.stride()),
        "storage_offset": value.storage_offset(),
        "data_alignment_mod256": value.data_ptr() % 256,
        "contiguous": value.is_contiguous(),
    }


def measured(torch: Any, function: Any, operand: Any) -> tuple[Any, float]:
    begin, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    begin.record()
    output = function(operand)
    end.record()
    end.synchronize()
    return output, float(begin.elapsed_time(end))
