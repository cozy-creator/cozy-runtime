"""Bounded CPU value reconstruction of a source's tensors, through the serving decoders."""

from __future__ import annotations

import importlib
import math
from types import ModuleType

import numpy as np
from tensorfs import raw

from cozy_runtime.author._errors import CapabilityError
from cozy_runtime.derive.safetensors_io import to_f32
from cozy_runtime.internal.encoding.formats import RowwiseDequant, params_of
from cozy_runtime.weights_store import Checkpoint, read_into

_ELEMENTS = 1 << 22
_FLOATS = {"F32": "float32", "F16": "float16", "BF16": "bfloat16"}
_WIDTHS = {"F32": 4, "F16": 2, "BF16": 2}
#: (scheme, weight dtype) -> its torch element, for the per-row schemes a window reads.
_ROWWISE = {
    ("fp8.rowwise", "F8_E4M3"): "float8_e4m3fn",
    ("fp8.rowwise", "F8_E5M2"): "float8_e5m2",
    ("int8.rowwise", "I8"): "int8",
}


def _part(
    source: Checkpoint, component: str, key: str, role: str, offset: int, length: int
) -> bytearray:
    data = bytearray(length)
    read_into(source, component, key, role, offset, memoryview(data))
    return data


def read_values_into(
    torch: ModuleType | None,
    source: Checkpoint,
    component: str,
    key: str,
    offset: int,
    into: memoryview,
) -> None:
    """Elements `offset..` of a plain float or per-row (`fp8.rowwise`, `int8.rowwise`) tensor,
    as f32. Plain floats need no torch; a per-row decode does."""
    held = source.components.get(component)
    tensor = None if held is None else held.tensors.get(key)
    count = into.nbytes // 4
    if tensor is None:
        raise CapabilityError("model tensor is absent", code="weights_reader_tensor")
    if offset < 0 or offset + count > math.prod(tensor.shape):
        raise CapabilityError("logical read leaves the tensor", code="weights_reader_bounds")
    if tensor.encoding is None:
        size = _WIDTHS.get(tensor.dtype)
        if size is None:
            raise CapabilityError("logical dtype is not a float", code="weights_reader_encoding")
        values = np.frombuffer(into, dtype=np.float32)
        for start in range(0, count, _ELEMENTS):
            length = min(_ELEMENTS, count - start)
            payload = _part(source, component, key, "", (offset + start) * size, length * size)
            values[start : start + length] = to_f32(payload, tensor.dtype, [length])
        return
    if torch is None:
        try:
            torch = importlib.import_module("torch")
        except ImportError as exc:
            raise CapabilityError(
                "logical model decoder is unavailable", code="weights_reader_decoder_unavailable"
            ) from exc
    target = torch.frombuffer(into, dtype=torch.float32)
    params = dict(tensor.encoding.params)
    parts = raw.parts(source, component, key)
    stored, scaled = parts.get("weight"), parts.get("scale")
    element = _ROWWISE.get((tensor.encoding.scheme, stored.dtype if stored else ""))
    scale_dtype = scaled.dtype if scaled else ""
    if element is None or params or scale_dtype not in _WIDTHS or len(tensor.shape) != 2:
        raise CapabilityError(
            f"no model reader for {tensor.encoding.scheme} {params}", code="weights_reader_encoding"
        )
    provider = RowwiseDequant(tensor.encoding.scheme, params=params_of(params))
    scale_width, scale_element = _WIDTHS[scale_dtype], getattr(torch, _FLOATS[scale_dtype])
    columns = tensor.shape[1]
    consumed = 0
    while consumed < count:
        position = offset + consumed
        row, column = divmod(position, columns)
        remaining = count - consumed
        whole_rows = min(remaining // columns, _ELEMENTS // columns) if column == 0 else 0
        rows = max(whole_rows, 1)
        width = columns if whole_rows else min(remaining, columns - column, _ELEMENTS)
        payload = _part(source, component, key, "weight", position, rows * width)
        scale = _part(source, component, key, "scale", row * scale_width, rows * scale_width)
        weight = torch.frombuffer(payload, dtype=torch.uint8).view(getattr(torch, element))
        provider.decode(
            torch,
            {
                "weight": weight.reshape(rows, width),
                "scale": torch.frombuffer(scale, dtype=scale_element).reshape(rows),
            },
            target[consumed : consumed + rows * width].reshape(rows, width),
        )
        consumed += rows * width
