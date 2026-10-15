"""Bounded CPU value reconstruction through the existing reviewed decode providers."""

from __future__ import annotations

import math
from typing import Any

from tensorfs.derived import SourceCapability, SourceInspection

from cozy_runtime.author._errors import CapabilityError
from cozy_runtime.internal.encoding.formats import (
    SPEC_PLAIN,
    SPEC_ROWWISE,
    SPEC_ROWWISE_KEEPDIM,
    RowwiseDequant,
)

_ELEMENTS = 1 << 22


def _part(source: Any, component: str, key: str, role: str, offset: int, length: int) -> bytearray:
    data = bytearray(length)
    source.read_part_into(component, key, role, offset, memoryview(data))
    return data


def read_values_into(
    source: SourceCapability,
    structure: SourceInspection,
    component: str,
    key: str,
    offset: int,
    into: memoryview,
) -> None:
    import numpy as np

    from cozy_runtime.derive.safetensors_io import to_f32

    tensor = structure.components.get(component, {}).get(key)
    count = into.nbytes // 4
    if tensor is None:
        raise CapabilityError("model tensor is absent", code="weights_reader_tensor")
    if offset < 0 or offset + count > math.prod(tensor.shape):
        raise CapabilityError("logical read leaves the tensor", code="weights_reader_bounds")
    destination = np.frombuffer(into, dtype=np.float32)
    if tensor.encoding == SPEC_PLAIN:
        size = {"f32": 4, "f16": 2, "bf16": 2}.get(tensor.logical_dtype)
        if size is None:
            raise CapabilityError(
                "logical dtype is not a supported float", code="weights_reader_encoding"
            )
        if set(tensor.parts) != {"value"}:
            raise CapabilityError("plain tensor roles differ", code="weights_reader_encoding")
        for start in range(0, count, _ELEMENTS):
            length = min(_ELEMENTS, count - start)
            raw = _part(source, component, key, "value", (offset + start) * size, length * size)
            destination[start : start + length] = to_f32(
                raw, tensor.logical_dtype.upper(), [length]
            )
        return
    if tensor.encoding not in {SPEC_ROWWISE, SPEC_ROWWISE_KEEPDIM}:
        raise CapabilityError(
            "encoding has no reviewed model reader", code="weights_reader_encoding"
        )
    if len(tensor.shape) != 2 or set(tensor.parts) != {"data", "scale"}:
        raise CapabilityError("encoded tensor geometry differs", code="weights_reader_encoding")
    try:
        import torch
    except ImportError:
        raise CapabilityError(
            "encoded model reads require Torch in the callable environment",
            code="weights_reader_decoder_unavailable",
        ) from None
    provider = RowwiseDequant(tensor.encoding)
    if provider.supports(tuple(tensor.shape)):
        raise CapabilityError("encoding geometry is not reviewed", code="weights_reader_encoding")
    target = torch.from_numpy(destination)
    columns = tensor.shape[1]
    consumed = 0
    while consumed < count:
        position = offset + consumed
        row, column = divmod(position, columns)
        remaining = count - consumed
        whole_rows = min(remaining // columns, _ELEMENTS // columns) if column == 0 else 0
        rows = max(whole_rows, 1)
        width = columns if whole_rows else min(remaining, columns - column, _ELEMENTS)
        payload = _part(source, component, key, "data", position, rows * width)
        scale = _part(source, component, key, "scale", row * 4, rows * 4)
        data_tensor = (
            torch.frombuffer(payload, dtype=torch.uint8)
            .view(torch.float8_e4m3fn)
            .reshape(rows, width)
        )
        scale_tensor = torch.frombuffer(scale, dtype=torch.float32).reshape(rows)
        output = target[consumed : consumed + rows * width].reshape(rows, width)
        # Arithmetic and special values are the serving decoder's, selected by exact
        # reviewed spec digest. Only the bounded source-range slicing is new here.
        provider.decode(torch, {"data": data_tensor, "scale": scale_tensor}, output)
        consumed += rows * width
