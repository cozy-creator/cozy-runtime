"""Providers are keyed by the header's scheme name; parts, params and unknown schemes refuse."""

from __future__ import annotations

import pytest

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

CPU = DeviceFacts("cpu", "CPU", 0, "", "", 0)


def test_every_scheme_has_its_parts_and_nothing_else() -> None:
    providers = launch_providers()
    expected = {
        "plain": ({"value"}, set()),
        "fp8.tensorwise": ({"weight", "scale"}, {"input_scale"}),
        "int8.tensorwise": ({"weight", "scale"}, {"input_scale"}),
        "fp8.rowwise": ({"weight", "scale"}, set()),
        "int8.rowwise": ({"weight", "scale"}, set()),
        "int8.convrot": ({"weight", "scale"}, set()),
        "mxfp8": ({"weight", "scale"}, set()),
        "nvfp4": ({"weight", "scale", "scale_2"}, {"input_scale"}),
    }
    assert set(providers) == set(expected)
    for scheme, candidates in providers.items():
        for provider in candidates:
            assert provider.encoding == scheme and scheme in provider.schemes
            assert (set(provider.required), set(provider.optional)) == expected[scheme], (
                provider.name
            )


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
        _candidates(Encoded("fp8.rowwise", params_of({"axis": 0}), (weight,)), (2, 64))
    assert refused.value.code == "role_mismatch"
    scale = RolePart("scale", "f32", (2,), 8)
    with pytest.raises(EncodingRefusal) as refused:
        _candidates(
            Encoded("fp8.rowwise", params_of({"axis": 0, "zero": 1}), (weight, scale)), (2, 64)
        )
    assert refused.value.code == "geometry_unsupported" and "zero" in str(refused.value)
    found = _candidates(Encoded("fp8.rowwise", params_of({"axis": 1}), (weight, scale)), (2, 64))
    assert "native-leaf" in str(found.geometry_refused), "the leaf takes a scale per row only"
