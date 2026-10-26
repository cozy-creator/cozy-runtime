"""An internal program's fixed weight use, before model construction.

This is input to the existing representation resolver, not an allocation or reservation.
The package supplies it on its internal GPU call; TensorD transports the closed value.
"""

from __future__ import annotations

from typing import Annotated, Literal

import msgspec

from cozy_runtime.author._errors import InvalidRequest

_Count = Annotated[int, msgspec.Meta(ge=0, le=2**32 - 1)]
_Bytes = Annotated[int, msgspec.Meta(ge=0, le=2**63 - 1)]


class PreparedUse(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    target: Literal["stored", "decoded_float"]
    tensors: tuple[str, ...]
    activation_rows: _Count
    repetitions: _Count
    compute_dtype: Literal["BF16", "F16", "F32"]
    max_resident_bytes: _Bytes
    workspace_bytes: _Bytes


def prepared_use(value: object) -> PreparedUse:
    """Validate and normalize only the marked value; prompt/seed never enter this identity."""
    try:
        row = msgspec.convert(value, type=PreparedUse, strict=True)
    except (TypeError, msgspec.ValidationError) as exc:
        raise InvalidRequest(str(exc), code="prepared_use_invalid") from exc
    try:
        invalid_key = any(
            not key or len(key.encode("utf-8")) > 1024 or "." not in key
            for key in row.tensors
        )
    except UnicodeEncodeError as exc:
        raise InvalidRequest(
            "prepared tensor keys must be valid UTF-8", code="prepared_use_invalid"
        ) from exc
    if (
        len(row.tensors) > 4096
        or invalid_key
        or len(set(row.tensors)) != len(row.tensors)
    ):
        raise InvalidRequest(
            "prepared tensor keys must be unique component-qualified names",
            code="prepared_use_invalid",
        )
    return msgspec.structs.replace(row, tensors=tuple(sorted(row.tensors)))
