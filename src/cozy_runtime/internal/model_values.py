"""Bounded CPU value reconstruction of a source's tensors, through the serving decoders."""

from __future__ import annotations

import math
from types import ModuleType

from cozy_runtime.author._errors import CapabilityError
from cozy_runtime.internal.encoding.formats import RowwiseDequant, params_of
from cozy_runtime.tensord_channels import Source

_ELEMENTS = 1 << 22
_FLOATS = {"F32": "float32", "F16": "float16", "BF16": "bfloat16"}
#: (scheme, weight dtype) -> its torch element, for the per-row schemes a window reads.
_ROWWISE = {
    ("fp8-rowwise", "F8_E4M3"): "float8_e4m3fn",
    ("fp8-rowwise", "F8_E5M2"): "float8_e5m2",
    ("int8-rowwise", "I8"): "int8",
}


def _part(
    source: Source, component: str, key: str, role: str, offset: int, length: int
) -> bytearray:
    data = bytearray(length)
    source.read_into(component, key, role, offset, memoryview(data))
    return data


def read_values_into(
    torch: ModuleType, source: Source, component: str, key: str, offset: int, into: memoryview
) -> None:
    """Elements `offset..` of a plain float or per-row (`fp8-rowwise`, `int8-rowwise`) tensor,
    as f32."""
    tensor = source.checkpoint.tensors.get(component, {}).get(key)
    count = into.nbytes // 4
    if tensor is None:
        raise CapabilityError("model tensor is absent", code="weights_reader_tensor")
    if offset < 0 or offset + count > math.prod(tensor.shape):
        raise CapabilityError("logical read leaves the tensor", code="weights_reader_bounds")
    target = torch.frombuffer(into, dtype=torch.float32)
    if tensor.encoding is None:
        name = _FLOATS.get(tensor.dtype)
        if name is None:
            raise CapabilityError("logical dtype is not a float", code="weights_reader_encoding")
        dtype = getattr(torch, name)
        size = torch.empty((), dtype=dtype).element_size()
        for start in range(0, count, _ELEMENTS):
            length = min(_ELEMENTS, count - start)
            raw = _part(source, component, key, "", (offset + start) * size, length * size)
            target[start : start + length].copy_(torch.frombuffer(raw, dtype=dtype))
        return
    params = dict(tensor.encoding.params)
    stored = tensor.parts.get("weight")
    element = _ROWWISE.get((tensor.encoding.scheme, stored.dtype if stored else ""))
    if element is None or params.get("axis") != 0 or len(tensor.shape) != 2:
        raise CapabilityError(
            f"no model reader for {tensor.encoding.scheme} {params}", code="weights_reader_encoding"
        )
    provider = RowwiseDequant(tensor.encoding.scheme, params=params_of(params))
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
        scale = _part(source, component, key, "scale", row * 4, rows * 4)
        weight = torch.frombuffer(payload, dtype=torch.uint8).view(getattr(torch, element))
        provider.decode(
            torch,
            {
                "weight": weight.reshape(rows, width),
                "scale": torch.frombuffer(scale, dtype=torch.float32).reshape(rows),
            },
            target[consumed : consumed + rows * width].reshape(rows, width),
        )
        consumed += rows * width
