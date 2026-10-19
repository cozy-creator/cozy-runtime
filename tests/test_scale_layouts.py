"""Stored layouts the producers emit, against the reference decoder's bytes: every row-major
nvfp4 and mxfp8 conformance case re-laid as ComfyUI writes it (block scales in cuBLAS 128x4
tiles, FP4's even element in the high nibble) decodes to the same expected bytes, and a stack of
weights (MoE `[E, out, in]`) decodes as its rows."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import msgspec
import pytest

from cozy_runtime.author._loader import TENSORFS_DTYPES
from cozy_runtime.internal.encoding import launch_providers, params_of
from cozy_runtime.internal.encoding.formats import (
    MX_BLOCK,
    NVFP4_BLOCK,
    Provider,
    _blocked_shape,
    matrix,
    to_blocked,
)
from cozy_runtime.internal.probe import VectorCase, same_bits, vector_cases, vector_tensor

torch = pytest.importorskip("torch")

BLOCKS = {"nvfp4": NVFP4_BLOCK, "mxfp8": MX_BLOCK}
LAYOUTS = [
    {"scale_layout": "cublas_128x4"},
    {"nibble_order": "high_first"},
    {"scale_layout": "cublas_128x4", "nibble_order": "high_first"},
]


def _decoders(case: VectorCase, params: dict[str, object], roles: dict[str, Any]) -> list[Provider]:
    """Every decode route the selector would hand this tensor (roles, params, shape)."""
    return [
        p
        for p in launch_providers()[case.encoding.scheme]
        if p.route == "decoded_float"
        and p.required <= roles.keys() <= p.required | p.optional
        and p.supports(case.shape, params) == ""
    ]


def _decode(case: VectorCase, params: dict[str, object], roles: dict[str, Any]) -> Any:
    """The bytes every route that takes the tensor decodes (they must agree)."""
    outs = [_decode_with(p, case, params, roles) for p in _decoders(case, params, roles)]
    assert outs, (case.key, params)
    for out in outs[1:]:  # bit for bit, NaN as NaN (a route need not canonicalize its NaN)
        nan = out.isnan() & outs[0].isnan()
        assert torch.equal(out[~nan].view(torch.uint8), outs[0][~nan].view(torch.uint8)), case.key
        assert torch.equal(out.isnan(), outs[0].isnan()), case.key
    return outs[0]


def _decode_with(
    provider: Provider, case: VectorCase, params: dict[str, object], roles: dict[str, Any]
) -> Any:
    bound = replace(provider, params=params_of(params))  # type: ignore[type-var]
    dtype = getattr(
        torch, {"f32": "float32", "bf16": "bfloat16", "f16": "float16"}[TENSORFS_DTYPES[case.dtype]]
    )
    out = torch.empty(*case.shape, dtype=dtype)
    bound.decode(torch, roles, out)
    return out


def _relaid(case: VectorCase, layout: dict[str, object]) -> dict[str, Any]:
    roles = {part.role: vector_tensor(torch, part, "cpu") for part in case.parts}
    rows, cols = matrix(case.shape)
    if layout.get("nibble_order") == "high_first":
        weight = roles["weight"].view(torch.uint8)
        roles["weight"] = (weight << 4) | (weight >> 4)
    if layout.get("scale_layout") == "cublas_128x4":  # tiled per matrix of a stack (spec §6)
        height = case.shape[-2]
        blocks = -(-cols // BLOCKS[case.encoding.scheme])
        grids = roles["scale"].view(torch.uint8).reshape(rows // height, height, blocks)
        padded = _blocked_shape(height, blocks)
        tiles = [to_blocked(torch, grid).reshape(padded) for grid in grids]
        stacked = torch.stack(tiles).reshape(*case.shape[:-2], *padded)
        roles["scale"] = stacked.view(roles["scale"].dtype)
    return roles


CASES = [
    c
    for c in vector_cases()
    if c.encoding.scheme in BLOCKS
    and not c.refuse
    and c.encoding.params.get("scale_layout") == "row_major"
    and c.encoding.params.get("nibble_order", "low_first") == "low_first"
]


@pytest.mark.parametrize("case", CASES, ids=lambda c: f"{c.encoding.scheme}-{c.key}")
@pytest.mark.parametrize(
    "layout", LAYOUTS, ids=lambda layout: "+".join(f"{k}={v}" for k, v in layout.items())
)
def test_a_relaid_case_decodes_to_the_reference_bytes(
    case: VectorCase, layout: dict[str, object]
) -> None:
    if "nibble_order" in layout and case.encoding.scheme != "nvfp4":
        pytest.skip("only FP4 packs two elements a byte")
    params = {**case.encoding.params, **layout}
    got = _decode(case, params, _relaid(case, layout))
    assert same_bits(torch, got, case.expect) == 0


@pytest.mark.parametrize(
    "scheme", ["fp8.rowwise", "int8.rowwise", "int8.convrot", "mxfp8", "nvfp4"]
)
def test_a_stack_of_weights_decodes_as_its_rows(scheme: str) -> None:
    case = next(
        c
        for c in vector_cases()
        if c.encoding.scheme == scheme
        and not c.refuse
        and len(c.shape) == 2
        and c.encoding.params.get("scale_layout", "row_major") == "row_major"
    )
    roles = {part.role: vector_tensor(torch, part, "cpu") for part in case.parts}
    once = _decode(case, dict(case.encoding.params), roles)
    stacked = {
        role: torch.stack([t, t]).reshape(2, *t.shape) if t.dim() else t
        for role, t in roles.items()
    }
    deep = msgspec.structs.replace(case, shape=[2, *case.shape])
    twice = _decode(deep, dict(case.encoding.params), stacked)
    assert torch.equal(twice.view(torch.uint8), torch.stack([once, once]).view(torch.uint8))
