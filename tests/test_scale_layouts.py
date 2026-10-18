"""Stored layouts the producers emit, against the reference decoder's bytes: every nvfp4 and
mxfp8 conformance case re-laid as ComfyUI writes it (block scales in cuBLAS 128x4 tiles, FP4's
even element in the high nibble) decodes to the same expected bytes, and a stack of weights
(MoE `[E, out, in]`) decodes as its rows."""

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


def _decoder(scheme: str) -> Provider:
    return next(p for p in launch_providers()[scheme] if p.route == "decoded_float")


def _decode(case: VectorCase, params: dict[str, object], roles: dict[str, Any]) -> Any:
    bound = replace(_decoder(case.encoding.scheme), params=params_of(params))  # type: ignore[type-var]
    assert bound.supports(case.shape, params) == "", params
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
    if layout.get("scale_layout") == "cublas_128x4":
        grid = (
            roles["scale"].view(torch.uint8).reshape(rows, -(-cols // BLOCKS[case.encoding.scheme]))
        )
        padded = _blocked_shape(*grid.shape)
        roles["scale"] = to_blocked(torch, grid).reshape(padded).view(roles["scale"].dtype)
    return roles


CASES = [c for c in vector_cases() if c.encoding.scheme in BLOCKS and not c.refuse]


@pytest.mark.parametrize("case", CASES, ids=lambda c: f"{c.encoding.scheme}-{c.key}")
@pytest.mark.parametrize(
    "layout", LAYOUTS, ids=lambda l: "+".join(f"{k}={v}" for k, v in l.items())
)
def test_a_relaid_case_decodes_to_the_reference_bytes(
    case: VectorCase, layout: dict[str, object]
) -> None:
    if "nibble_order" in layout and case.encoding.scheme != "nvfp4":
        pytest.skip("only FP4 packs two elements a byte")
    params = {**case.encoding.params, **layout}
    got = _decode(case, params, _relaid(case, layout))
    assert same_bits(torch, got, case.expect) == 0


@pytest.mark.parametrize("scheme", ["fp8-rowwise", "int8-rowwise", "mxfp8", "nvfp4"])
def test_a_stack_of_weights_decodes_as_its_rows(scheme: str) -> None:
    case = next(
        c
        for c in vector_cases()
        if c.encoding.scheme == scheme and not c.refuse and len(c.shape) == 2
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
