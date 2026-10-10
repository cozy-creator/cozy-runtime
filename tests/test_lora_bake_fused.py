"""The fused GPU bake against its eager reference, on CPU under Triton's interpreter.

The launcher test runs this file again with `TRITON_INTERPRET=1`, which Triton reads when
`lora_bake` decorates its kernels; there the interpreted tests run the kernel's own source on
CPU tensors. Stochastic rounding is held to what it promises: every stored value is one of the
two grid neighbours of its target, unbiased, and the LoRA's delta survives as the eager bake's
does; a value already on the grid is a fixed point.
"""

from __future__ import annotations

import os
import subprocess
import sys
from typing import Any

import pytest

from cozy_runtime.internal import lora_bake

torch = pytest.importorskip("torch")

INTERPRETED = os.environ.get("TRITON_INTERPRET") == "1"
interpreted = pytest.mark.skipif(not INTERPRETED, reason="run by the launcher test")
ROWS, COLS, RANK = 96, 160, 6


def test_the_fused_bake_holds_under_tritons_interpreter() -> None:
    if INTERPRETED:
        pytest.skip("the launcher itself")
    pytest.importorskip("triton")
    env = {**os.environ, "TRITON_INTERPRET": "1", "OMP_NUM_THREADS": "1"}
    command = [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", __file__]
    run = subprocess.run(command, env=env, capture_output=True, text=True, check=False)
    assert run.returncode == 0 and " passed" in run.stdout, run.stdout[-4000:] + run.stderr


def _layer(seed: int) -> tuple[Any, Any, Any]:
    generator = torch.Generator().manual_seed(seed)
    weight = torch.randn(ROWS, COLS, generator=generator) * 0.05
    down = torch.randn(RANK, COLS, generator=generator) * 0.01
    up = torch.randn(ROWS, RANK, generator=generator) * 0.01
    return weight, down, up


def _stored(form: int, weight: Any) -> tuple[dict[str, Any], Any]:
    """A stored weight of `form` and its multiplier from payload to weight units (f64)."""
    if form == lora_bake.ROWWISE:
        scale = (weight.abs().amax(dim=1) / 448).clamp(min=1e-12)
        data = (weight / scale[:, None]).to(torch.float8_e4m3fn)
        return {"data": data, "scale": scale}, scale[:, None].double()
    if form == lora_bake.MX:
        grid = torch.randint(115, 121, (ROWS, COLS // 32), dtype=torch.uint8)
        unit = torch.ldexp(torch.ones(grid.shape), grid.int() - 127).repeat_interleave(32, dim=1)
        data = (weight / unit).clamp(-448, 448).to(torch.float8_e4m3fn)
        return {"data": data, "scale": grid}, unit.double()
    dtype = (torch.bfloat16, torch.float16, torch.float32)[form]
    return {"value": weight.to(dtype)}, torch.ones(())


def _values(stored: dict[str, Any], unit: Any) -> Any:
    if "value" in stored:
        return stored["value"].double()
    return lora_bake.decode_e4m3(torch, stored["data"]).double() * unit


def _copy(stored: dict[str, Any]) -> dict[str, Any]:
    return {role: tensor.clone() for role, tensor in stored.items()}


def _bytes(stored: dict[str, Any]) -> Any:
    tensor = stored["data"] if "data" in stored else stored["value"]
    return tensor.view(torch.uint8) if tensor.dtype == torch.float8_e4m3fn else tensor


def _step(form: int, target: Any, unit: Any) -> Any:
    """The grid step at each target, in weight units."""
    if form == lora_bake.BF16:
        exponent = torch.floor(torch.log2(target.abs().clamp(min=2.0**-126)))
        return torch.exp2(exponent - 7)
    payload = (target / unit).abs().clamp(min=2.0**-6)
    return torch.exp2(torch.floor(torch.log2(payload)) - 3) * unit


@interpreted
@pytest.mark.parametrize("form", [lora_bake.BF16, lora_bake.ROWWISE, lora_bake.MX])
def test_a_fused_stochastic_bake_stores_a_neighbour_without_bias(form: int) -> None:
    weight, down, up = _layer(3)
    stored, unit = _stored(form, weight)
    base, delta = _values(stored, unit), up.double() @ down.double()
    target = base + delta
    step = _step(form, target, unit)
    retained, eager_retained, bias = [], [], []
    for seed in range(4):
        fused, eager = _copy(stored), _copy(stored)
        lora_bake.fused(torch, fused, down, up, seed)
        lora_bake.bake(torch, eager, down, up, seed=seed, stats=lora_bake.Stats())
        got = _values(fused, unit)
        assert ((got - target).abs() <= step * (1 + 1e-6)).all()
        bias.append(float(((got - target) / step).mean()))
        retained.append(float(((got - base) * delta).sum() / delta.square().sum()))
        moved = _values(eager, unit) - base
        eager_retained.append(float((moved * delta).sum() / delta.square().sum()))
    assert abs(sum(bias) / len(bias)) < 0.01
    assert abs(sum(retained) / 4 - 1) < 0.05 and abs(sum(eager_retained) / 4 - 1) < 0.05


@interpreted
@pytest.mark.parametrize("form", [lora_bake.F16, lora_bake.F32])
def test_a_fused_nearest_bake_is_the_eager_one(form: int) -> None:
    weight, down, up = _layer(5)
    stored, _ = _stored(form, weight)
    fused, eager = _copy(stored), _copy(stored)
    lora_bake.fused(torch, fused, down, up, 0)
    lora_bake.bake(torch, eager, down, up, seed=0, stats=lora_bake.Stats())
    # One rounding of the same f32 sum: they differ only where the GEMMs' order moves it.
    torch.testing.assert_close(fused["value"].float(), eager["value"].float(), rtol=1e-3, atol=0)


@interpreted
@pytest.mark.parametrize("form", list(range(5)))
def test_values_on_the_grid_are_fixed_points_and_a_seed_names_the_bytes(form: int) -> None:
    weight, down, up = _layer(7)
    stored, _ = _stored(form, weight)
    zero, eager = _copy(stored), _copy(stored)
    lora_bake.fused(torch, zero, down * 0, up, 11)
    lora_bake.bake(torch, eager, down * 0, up, seed=11, stats=lora_bake.Stats())
    assert torch.equal(_bytes(zero), _bytes(eager))
    first, again, other = _copy(stored), _copy(stored), _copy(stored)
    for copy, seed in ((first, 11), (again, 11), (other, 12)):
        lora_bake.fused(torch, copy, down, up, seed)
    assert torch.equal(_bytes(first), _bytes(again))
    assert torch.equal(_bytes(first), _bytes(other)) == (form in (lora_bake.F16, lora_bake.F32))


@interpreted
def test_the_reserved_codes_decode_to_their_signed_maximum() -> None:
    codes = torch.arange(256, dtype=torch.int32).to(torch.uint8).view(8, 32)
    stored = {"data": codes.clone().view(torch.float8_e4m3fn), "scale": torch.ones(8)}
    lora_bake.fused(torch, stored, torch.zeros(1, 32), torch.zeros(8, 1), 0)
    # As the eager bake: the reserved codes saturate and a negative zero adds to a positive one.
    expected = codes.clone()
    for code, stays in ((0x7F, 0x7E), (0xFF, 0xFE), (0x80, 0)):
        expected[codes == code] = stays
    assert torch.equal(stored["data"].view(torch.uint8), expected)
