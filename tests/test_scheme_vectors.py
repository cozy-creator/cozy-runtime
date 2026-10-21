"""The decoders against the conformance cases, on the CPU: every scheme case decodes to its
expected bytes (NaN as NaN), every decode refusal refuses, and every decoder passes its own
domain oracle in each output dtype."""

from __future__ import annotations

from dataclasses import replace

import msgspec
import pytest
import tensorfs

from cozy_runtime.author._loader import TENSORFS_DTYPES
from cozy_runtime.internal.decode_cores import CodeOutsideTable
from cozy_runtime.internal.encoding import Encoded, Provider, RolePart, launch_providers, params_of
from cozy_runtime.internal.encoding.formats import MicroScaledDequant, part_rule
from cozy_runtime.internal.encoding.leaves import quantize_activation_mx
from cozy_runtime.internal.probe import (
    DECODE_SUITES,
    PROBE_PARAMS,
    VectorCase,
    decode_case,
    same_bits,
    vector_cases,
)

torch = pytest.importorskip("torch")

CASES = vector_cases()
DECODERS = [
    provider
    for candidates in launch_providers().values()
    for provider in candidates
    if provider.route == "decoded_float"
]


def _encoded(case: VectorCase) -> Encoded | str:
    """The case as the fill plane builds it, or why the fill plane refuses it."""
    parts = []
    for part in case.parts:
        if part.dtype == "F4":
            packed = (*part.shape[:-1], part.shape[-1] // 2)
            parts.append(RolePart(part.role, "u8", packed, 0))
            continue
        dtype = TENSORFS_DTYPES.get(part.dtype)
        if dtype is None:
            return f"unknown dtype {part.dtype}"
        parts.append(RolePart(part.role, dtype, tuple(part.shape), 0))
    return Encoded(case.encoding.scheme, params_of(case.encoding.params), tuple(parts))


def _takes(provider: Provider, case: VectorCase) -> bool:
    """The selector hands `provider` this tensor: its roles, params and shape."""
    roles = {p.role for p in case.parts}
    return (
        provider.required <= roles <= provider.required | provider.optional
        and not provider.supports(case.shape, case.encoding.params)
    )


@pytest.mark.parametrize("case", [c for c in CASES if not c.refuse], ids=lambda c: c.key)
def test_every_scheme_case_decodes_to_its_expected_bytes(case: VectorCase) -> None:
    providers = [p for p in DECODERS if p.encoding == case.encoding.scheme and _takes(p, case)]
    assert providers, case
    for provider in providers:
        got = decode_case(torch, provider, case, "cpu")
        assert got is not None, f"{provider.name} refused {case.key}"
        assert same_bits(torch, got, case.expect) == 0, (provider.name, case.key)


@pytest.mark.parametrize("case", [c for c in CASES if c.refuse], ids=lambda c: c.key)
def test_every_decode_refusal_refuses(case: VectorCase) -> None:
    encoded = _encoded(case)
    if isinstance(encoded, str):
        return  # the fill plane refuses an unknown dtype before any decoder sees it
    dtype = TENSORFS_DTYPES.get(case.dtype)
    parts = {p.role: p for p in encoded.parts}
    for provider in [p for p in DECODERS if p.encoding == encoded.scheme]:
        refused = (
            dtype is None
            or bool(part_rule(encoded, case.shape, dtype))
            or bool(provider.required - set(parts))
            or bool(set(parts) - provider.required - provider.optional)
            or bool(provider.supports(case.shape, case.encoding.params))
            or _decode_refuses(provider, case)
        )
        assert refused or (case.encoding.scheme, case.key) in UNSEEN, (provider.name, case.key)


#: The fill plane hands an `F4` part over as its packed bytes, so a `U8` part of the same bytes
#: reads alike: only TensorFS's own decoder, which sees the header's dtype, refuses these.
UNSEEN = {("mxfp4", "weight-u8")}


def _decode_refuses(provider: Provider, case: VectorCase) -> bool:
    """A decode refuses the values the spec constrains (tables, indices, zeros, stored shape)."""
    try:
        decode_case(torch, provider, msgspec.structs.replace(case, dtype="F32"), "cpu")
    except (ValueError, CodeOutsideTable, tensorfs.DecodeError):
        return True
    return False


@pytest.mark.parametrize("dtype", ["float32", "bfloat16", "float16"])
def test_every_decoder_passes_its_own_domain_oracle(dtype: str) -> None:
    for listed in DECODERS:
        provider = replace(listed, params=params_of(PROBE_PARAMS.get(listed.encoding, {})))  # type: ignore[type-var]
        sweep = DECODE_SUITES[provider.suite](
            torch, provider, torch.device("cpu"), getattr(torch, dtype)
        )
        assert sweep.passed, (provider.name, dtype, sweep)


def test_the_mx_activation_quantizer_saturates_nothing_and_decodes_back() -> None:
    x = torch.randn(7, 256, generator=torch.Generator().manual_seed(5)) * torch.logspace(-6, 6, 256)
    payload, scales = quantize_activation_mx(torch, x)
    assert payload.dtype == torch.float8_e4m3fn and scales.shape == (7, 256 // 32)
    assert float(payload.float().abs().max()) <= 448.0 and not payload.float().isnan().any()
    back = torch.empty(7, 256, dtype=torch.float32)
    MicroScaledDequant(params=params_of({"block_size": 32, "scale_layout": "row_major"})).decode(
        torch, {"weight": payload, "scale": scales.view(torch.float8_e8m0fnu)}, back
    )
    assert float(((back - x).abs() / x.abs().clamp(min=1e-30)).max()) <= 2.0**-3
