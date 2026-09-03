"""TensorRequirements cross into TensorFS only in its closed dtype namespace."""

from __future__ import annotations

import base64

import pytest
import tensorfs

from cozy_runtime.author._loader import canonical_dtype
from cozy_runtime.internal.fill import tensorfs_requirement_dtype

HEADERS = {
    "f32": base64.b64decode(
        "hW1jb3p5dGVuc29ycy8xgYJkdW5ldFgceyJoaWRkZW5fc2l6ZSI6NCwibGF5ZXJzIjoxfYGFc3Rva2VuaXplci92b2NhYi50eHRYIJ5ekBAsaZRV6QOf+QMoTgaJOU3TRbsRRWcG8IeYTS63Bmp0ZXh0L3BsYWlugYJYIJ5ekBAsaZRV6QOf+QMoTgaJOU3TRbsRRWcG8IeYTS63BoGEjAMLAgEABAUIBwYJCoCBg2V2YWx1ZYEAgQCBglggR2b5Mbu3DtQx40s05Pn8XdqBdOlvQ6is152Hme/NQ9IZAgSBgmR1bmV0gYVmd2VpZ2h0AYEBAIGEZXZhbHVlAYEBRAAAAAA="
    ),
    "f16": base64.b64decode(
        "hW1jb3p5dGVuc29ycy8xgYJkdW5ldFgceyJoaWRkZW5fc2l6ZSI6NCwibGF5ZXJzIjoxfYGFc3Rva2VuaXplci92b2NhYi50eHRYIJ5ekBAsaZRV6QOf+QMoTgaJOU3TRbsRRWcG8IeYTS63Bmp0ZXh0L3BsYWlugYJYIJ5ekBAsaZRV6QOf+QMoTgaJOU3TRbsRRWcG8IeYTS63BoGEjAMLAgEABAUIBwYJCoCBg2V2YWx1ZYEAgQCBglggR2b5Mbu3DtQx40s05Pn8XdqBdOlvQ6is152Hme/NQ9IZAgSBgmR1bmV0gYVmd2VpZ2h0AoEBAIGEZXZhbHVlAoEBQgAA"
    ),
    "bf16": base64.b64decode(
        "hW1jb3p5dGVuc29ycy8xgYJkdW5ldFgceyJoaWRkZW5fc2l6ZSI6NCwibGF5ZXJzIjoxfYGFc3Rva2VuaXplci92b2NhYi50eHRYIJ5ekBAsaZRV6QOf+QMoTgaJOU3TRbsRRWcG8IeYTS63Bmp0ZXh0L3BsYWlugYJYIJ5ekBAsaZRV6QOf+QMoTgaJOU3TRbsRRWcG8IeYTS63BoGEjAMLAgEABAUIBwYJCoCBg2V2YWx1ZYEAgQCBglggR2b5Mbu3DtQx40s05Pn8XdqBdOlvQ6is152Hme/NQ9IZAgSBgmR1bmV0gYVmd2VpZ2h0A4EBAIGEZXZhbHVlA4EBQgAA"
    ),
}


@pytest.mark.parametrize(
    ("torch_spelling", "stored", "constraint"),
    [
        ("torch.bfloat16", "bf16", None),
        ("torch.float16", "f16", None),
        ("torch.float32", "f32", "f32"),
    ],
)
def test_torch_names_fit_tensorfs_03_headers(
    torch_spelling: str, stored: str, constraint: str | None
) -> None:
    assert canonical_dtype(torch_spelling) == stored
    assert tensorfs_requirement_dtype(torch_spelling) == constraint
    verdict = tensorfs.fit(
        [tensorfs.TensorRequirement("unet", "weight", [1], constraint)],
        HEADERS[stored],
        custody="canonical",
        encoded_leaves=False,
    )
    assert verdict["ok"]


def test_torch_bfloat16_is_not_a_tensorfs_dtype() -> None:
    with pytest.raises(tensorfs.errors.Refusal) as refused:
        tensorfs.fit(
            [tensorfs.TensorRequirement("unet", "weight", [1], "bfloat16")],
            HEADERS["bf16"],
            custody="canonical",
            encoded_leaves=False,
        )
    assert refused.value.code == "DTYPE_UNKNOWN"
