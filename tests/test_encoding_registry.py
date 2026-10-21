"""Providers are keyed by the header's scheme name; parts, params and unknown schemes refuse."""

from __future__ import annotations

import pytest

from cozy_runtime.internal.decode_schemes import ADAPTERS
from cozy_runtime.internal.encoding import (
    Capabilities,
    DeviceFacts,
    Encoded,
    EncodingRefusal,
    RolePart,
    Selector,
    TensorCandidates,
    launch_providers,
    params_of,
)
from cozy_runtime.internal.encoding.adapters import CoreDequant

CPU = DeviceFacts("cpu", "CPU", 0, "", "", 0)


def test_every_scheme_has_its_parts_and_nothing_else() -> None:
    providers = launch_providers()
    expected = {
        "plain": ({"value"}, set()),
        "fp8.tensorwise": ({"weight", "scale"}, {"input_scale"}),
        "int8.tensorwise": ({"weight", "scale"}, {"input_scale"}),
        "fp8.rowwise": ({"weight", "scale"}, set()),
        "int8.rowwise": ({"weight", "scale"}, set()),
        "fp8.columnwise": ({"weight", "scale"}, set()),
        "int8.columnwise": ({"weight", "scale"}, set()),
        "int8.convrot": ({"weight", "scale"}, set()),
        "mxfp8": ({"weight", "scale"}, set()),
        "nvfp4": ({"weight", "scale", "scale_2"}, {"input_scale"}),
        "comfy.asym_w4a8_int8": (
            {"weight", "group_scale", "scale"},
            {"codebook", "correction"},
        ),
        "comfy.w6a8_int8": ({"weight", "group_scale", "scale"}, set()),
        "int4.convrot": ({"weight", "scale"}, set()),
    }
    # The cores' rows (decode_schemes.ADAPTERS) are their own: nvfp4's adds AWQ's pre_quant_scale.
    rows = {a.scheme: (set(a.required), set(a.optional)) for a in ADAPTERS}
    assert set(providers) == set(expected) | set(rows)
    for scheme, candidates in providers.items():
        for provider in candidates:
            assert provider.encoding == scheme and scheme in provider.schemes
            want = rows[scheme] if isinstance(provider, CoreDequant) else expected[scheme]
            assert (set(provider.required), set(provider.optional)) == want, provider.name


def _candidates(encoded: Encoded, shape: tuple[int, ...]) -> TensorCandidates:
    return Selector(launch_providers(), Capabilities(), CPU).candidates("w", encoded, "bf16", shape)


def test_an_unknown_scheme_refuses_by_name() -> None:
    parts = (RolePart("weight", "f8_e4m3fn", (2, 64), 128),)
    with pytest.raises(EncodingRefusal) as refused:
        _candidates(Encoded("acme-int3", (), parts), (2, 64))
    assert refused.value.code == "unknown_encoding" and "acme-int3" in str(refused.value)


def test_parts_and_params_outside_the_scheme_refuse() -> None:
    weight = RolePart("weight", "f8_e4m3fn", (2, 64), 128)
    with pytest.raises(EncodingRefusal) as refused:
        _candidates(Encoded("fp8.rowwise", params_of({}), (weight,)), (2, 64))
    assert refused.value.code == "role_mismatch"
    scale = RolePart("scale", "f32", (2,), 8)
    with pytest.raises(EncodingRefusal) as refused:
        _candidates(Encoded("fp8.rowwise", params_of({"zero": 1}), (weight, scale)), (2, 64))
    assert refused.value.code == "geometry_unsupported" and "zero" in str(refused.value)
    columns = RolePart("scale", "f32", (64,), 256)
    found = _candidates(Encoded("fp8.columnwise", params_of({}), (weight, columns)), (2, 64))
    assert found.routes_offered == {"decoded_float"}, "the leaf takes a scale per row only"
