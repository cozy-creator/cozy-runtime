"""Exhaustive decode-bit and per-call lookup-table lifetime controls."""

import math
import weakref
from typing import Any

import pytest

from cozy_runtime.internal import lora_bake

torch = pytest.importorskip("torch")


@pytest.mark.parametrize("layout", ["flat", "transpose", "stride", "offset", "odd"])
def test_native_decode_keeps_every_code_bit_and_signed_reserved_value(layout: str) -> None:
    raw = torch.arange(1024, dtype=torch.int32).to(torch.uint8)
    data = {
        "flat": raw[:256],
        "transpose": raw[:256].reshape(16, 16).t(),
        "stride": raw[1:769:3],
        "offset": raw[1:257],
        "odd": raw[:257].reshape(1, 257),
    }[layout]
    before = data.clone()
    expected = []
    for byte in data.reshape(-1).tolist():
        code = min(byte & 127, 126)
        exponent, mantissa = code >> 3, code & 7
        magnitude = math.ldexp(1 + mantissa / 8, exponent - 7) if exponent else mantissa / 512
        expected.append(math.copysign(magnitude, -1 if byte & 128 else 1))
    oracle = torch.tensor(expected, dtype=torch.float32).reshape(data.shape)
    for view in (data, data.view(torch.float8_e4m3fn)):
        actual = lora_bake.decode_e4m3(torch, view)
        assert torch.equal(actual.view(torch.int32), oracle.view(torch.int32))
        assert torch.equal(data, before)


@pytest.mark.parametrize("kind", ["rowwise", "mx"])
@pytest.mark.parametrize("statistics", [False, True])
def test_encoding_table_is_built_once_per_bake_and_not_retained(
    monkeypatch: pytest.MonkeyPatch, kind: str, statistics: bool
) -> None:
    monkeypatch.setattr(lora_bake, "CHUNK_ELEMENTS", 3 * 64)
    data = torch.ones((8, 64), dtype=torch.float8_e4m3fn)
    scale = torch.ones(8) if kind == "rowwise" else torch.full((8, 2), 127, dtype=torch.uint8)
    down, up = torch.ones((4, 64)), torch.ones((8, 4))
    original = lora_bake._magnitudes
    held: list[weakref.ReferenceType[Any]] = []

    def counted(control: Any, device: Any) -> Any:
        table = original(control, device)
        held.append(weakref.ref(table))
        return table

    monkeypatch.setattr(lora_bake, "_magnitudes", counted)
    function = lora_bake.bake_rowwise if kind == "rowwise" else lora_bake.bake_mx
    function(torch, data, scale, down, up, 0.25, stats=lora_bake.Stats() if statistics else None)
    assert len(held) == 1 and held[0]() is None
