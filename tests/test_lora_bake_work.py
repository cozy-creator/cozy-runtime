"""Stored-byte, diagnostic work and call-lifetime controls on real CPU tensors."""

from __future__ import annotations

import gc
import weakref
from typing import Any

import pytest

from cozy_runtime.internal import lora_bake

torch = pytest.importorskip("torch")


def _inputs() -> tuple[Any, Any, Any]:
    generator = torch.Generator().manual_seed(17)
    weight = torch.randn(8, 64, generator=generator) * 0.04
    down = (torch.randn(4, 64, generator=generator) * 0.02).to(torch.bfloat16)
    up = (torch.randn(8, 4, generator=generator) * 0.02).to(torch.bfloat16)
    return weight, down, up


def _stored(kind: str, weight: Any) -> tuple[Any, Any]:
    if kind == "plain":
        return weight.to(torch.bfloat16), None
    if kind == "rowwise":
        scale = (weight.abs().amax(dim=1) / 448).clamp(min=1e-12)
        return (weight / scale[:, None]).to(torch.float8_e4m3fn), scale
    scale = torch.full((8, 2), 118, dtype=torch.uint8)
    return (weight * 512).to(torch.float8_e4m3fn), scale


def _bake(
    kind: str,
    data: Any,
    scale: Any,
    down: Any,
    up: Any,
    rounding: lora_bake.Rounding,
    stats: Any = None,
    refit: bool = False,
) -> None:
    kwargs: dict[str, Any] = {
        "rounding": rounding,
        "stats": stats,
        "generator": torch.Generator().manual_seed(99),
    }
    if kind == "plain":
        lora_bake.bake_plain(torch, data, down, up, 0.75, **kwargs)
    elif kind == "rowwise":
        lora_bake.bake_rowwise(
            torch, data, scale, down, up, 0.75, grid="refit" if refit else "keep", **kwargs
        )
    else:
        lora_bake.bake_mx(torch, data, scale, down, up, 0.75, refit=refit, **kwargs)


@pytest.mark.parametrize("kind", ["plain", "rowwise", "mx"])
@pytest.mark.parametrize("rounding", ["nearest", "stochastic", "feedback"])
@pytest.mark.parametrize("refit", [False, True])
def test_statistics_do_not_change_stored_bytes_or_rng_sequence(
    monkeypatch: pytest.MonkeyPatch, kind: str, rounding: lora_bake.Rounding, refit: bool
) -> None:
    # Multiple row chunks exercise the repeated-factor path, including a non-full tail.
    monkeypatch.setattr(lora_bake, "CHUNK_ELEMENTS", 3 * 64)
    weight, down, up = _inputs()
    first, first_scale = _stored(kind, weight)
    second, second_scale = _stored(kind, weight)
    _bake(kind, first, first_scale, down, up, rounding, refit=refit)
    stats = lora_bake.Stats()
    _bake(kind, second, second_scale, down, up, rounding, stats, refit)
    assert torch.equal(first.view(torch.uint8), second.view(torch.uint8))
    if first_scale is not None:
        assert torch.equal(first_scale.view(torch.uint8), second_scale.view(torch.uint8))
    assert stats.layers == 1 and stats.elements == weight.numel()
    assert stats.weight_energy > 0 and stats.delta_energy > 0


@pytest.mark.parametrize("kind", ["plain", "rowwise", "mx"])
@pytest.mark.parametrize("want_stats", [False, True])
def test_only_requested_diagnostics_and_one_call_scoped_factor_cast(
    monkeypatch: pytest.MonkeyPatch, kind: str, want_stats: bool
) -> None:
    monkeypatch.setattr(lora_bake, "CHUNK_ELEMENTS", 3 * 64)
    weight, down, up = _inputs()
    data, scale = _stored(kind, weight)
    calls = {"account": 0, "decode": 0, "down_cast": 0, "saturation": 0}
    copies: list[weakref.ReferenceType[Any]] = []
    original_float, original_sum = torch.Tensor.float, torch.Tensor.sum
    original_account, original_decode = lora_bake._account, lora_bake.decode_e4m3

    def cast(tensor: Any, *args: Any, **kwargs: Any) -> Any:
        result = original_float(tensor, *args, **kwargs)
        if tensor is down:
            calls["down_cast"] += 1
            copies.append(weakref.ref(result))
        return result

    def summed(tensor: Any, *args: Any, **kwargs: Any) -> Any:
        if tensor.dtype == torch.bool:
            calls["saturation"] += 1
        return original_sum(tensor, *args, **kwargs)

    def account(*args: Any, **kwargs: Any) -> Any:
        calls["account"] += 1
        return original_account(*args, **kwargs)

    def decode(*args: Any, **kwargs: Any) -> Any:
        calls["decode"] += 1
        return original_decode(*args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "float", cast)
    monkeypatch.setattr(torch.Tensor, "sum", summed)
    monkeypatch.setattr(lora_bake, "_account", account)
    monkeypatch.setattr(lora_bake, "decode_e4m3", decode)
    stats = lora_bake.Stats() if want_stats else None
    _bake(kind, data, scale, down, up, "stochastic", stats)
    chunks = 3
    assert calls["account"] == (chunks if want_stats else 0)
    assert calls["down_cast"] == 1
    assert calls["decode"] == (0 if kind == "plain" else chunks * (2 if want_stats else 1))
    assert calls["saturation"] == (chunks if want_stats and kind != "plain" else 0)
    gc.collect()
    assert len(copies) == 1 and copies[0]() is None
