"""`CoreDequant`: a scheme becomes a decode provider through one `Adapter` row, and the probe
qualifies it with its own oracle (`cores-decode`). The rows here express the schemes the cores
already cover (`test_decode_cores.DECODES`), as the scheme units write theirs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any

import pytest

from cozy_runtime.internal import probe
from cozy_runtime.internal.encoding import launch_providers
from cozy_runtime.internal.encoding.adapters import Adapter, CoreDequant
from cozy_runtime.internal.encoding.formats import params_of
from test_decode_cores import CUDA, DECODES

torch = pytest.importorskip("torch")

LAYOUTS = ("row_major", "cublas_128x4")
TENSORWISE = {"scale_dtype": ("F32", "BF16", "F16")}
AXIS = {"axis": (0, 1), "scale_dtype": ("F32", "BF16", "F16")}


def _bytes(rows: int, cols: int, generator: Any) -> Any:
    return torch.randint(0, 256, (rows, cols), generator=generator, dtype=torch.int32).to(
        torch.uint8
    )


def _scales(n: int, generator: Any) -> Any:
    return torch.randn(n, generator=generator) * 0.05


def _scaled(element: str):  # type: ignore[no-untyped-def]
    """Every byte pattern of `element`, with a scale per tensor, row or column."""

    def sample(
        torch: Any, rows: int, cols: int, generator: Any, params: Mapping[str, object]
    ) -> Any:
        flat = "axis" not in params
        count = 1 if flat else (rows if params["axis"] == 0 else cols)
        scale = _scales(count, generator)
        return {
            "weight": _bytes(rows, cols, generator).view(getattr(torch, element)),
            "scale": scale.reshape(()) if flat else scale,
        }

    return sample


def _sample_mxfp8(
    torch: Any, rows: int, cols: int, generator: Any, params: Mapping[str, object]
) -> Any:
    return {
        "weight": _bytes(rows, cols, generator).view(torch.float8_e4m3fn),
        "scale": _bytes(rows, -(-cols // 32), generator).view(torch.float8_e8m0fnu),
    }


def _sample_nvfp4(
    torch: Any, rows: int, cols: int, generator: Any, params: Mapping[str, object]
) -> Any:
    return {
        "weight": _bytes(rows, cols // 2, generator),
        "scale": _bytes(rows, cols // 16, generator).view(torch.float8_e4m3fn),
        "scale_2": _scales(1, generator).reshape(()),
    }


def _any_shape(shape: Any, params: Mapping[str, object]) -> str:
    return ""


SCHEMES = (
    Adapter(
        "fp8.tensorwise",
        frozenset({"weight", "scale"}),
        frozenset({"input_scale"}),
        TENSORWISE,
        DECODES["fp8.tensorwise"],
        {"scale_dtype": "F32"},
        _scaled("float8_e4m3fn"),
        _any_shape,
    ),
    Adapter(
        "fp8.rowwise",
        frozenset({"weight", "scale"}),
        frozenset(),
        AXIS,
        DECODES["fp8.rowwise"],
        {"axis": 0, "scale_dtype": "F32"},
        _scaled("float8_e5m2"),
    ),
    Adapter(
        "int8.tensorwise",
        frozenset({"weight", "scale"}),
        frozenset({"input_scale"}),
        TENSORWISE,
        DECODES["int8.tensorwise"],
        {"scale_dtype": "F32"},
        _scaled("int8"),
        _any_shape,
    ),
    Adapter(
        "int8.rowwise",
        frozenset({"weight", "scale"}),
        frozenset(),
        AXIS,
        DECODES["int8.rowwise"],
        {"axis": 1, "scale_dtype": "F32"},
        _scaled("int8"),
    ),
    Adapter(
        "mxfp8",
        frozenset({"weight", "scale"}),
        frozenset(),
        {"block_size": (32,), "scale_layout": LAYOUTS},
        DECODES["mxfp8"],
        {"block_size": 32, "scale_layout": "row_major"},
        _sample_mxfp8,
    ),
    Adapter(
        "nvfp4",
        frozenset({"weight", "scale", "scale_2"}),
        frozenset({"input_scale"}),
        {
            "block_size": (16,),
            "global_scale": ("multiply", "divide"),
            "nibble_order": ("low_first", "high_first"),
            "scale_layout": LAYOUTS,
        },
        DECODES["nvfp4"],
        {
            "block_size": 16,
            "global_scale": "divide",
            "nibble_order": "high_first",
            "scale_layout": "row_major",
        },
        _sample_nvfp4,
    ),
)


def _provider(scheme: str) -> CoreDequant:
    return CoreDequant(scheme, adapters=SCHEMES)


def test_the_probe_decodes_every_vector_through_the_provider() -> None:
    """`probe.decode_case` binds the case's params and calls `CoreDequant.decode`."""
    checked = 0
    schemes = {a.scheme for a in SCHEMES}
    device = torch.device("cuda" if CUDA else "cpu")
    for case in probe.vector_cases():
        if case.encoding.scheme in schemes and not case.refuse:
            got = probe.decode_case(torch, _provider(case.encoding.scheme), case, device)
            assert got is not None, case.key
            assert probe.same_bits(torch, got, case.expect) == 0, f"{case.component}/{case.key}"
            checked += 1
    assert checked >= 40, checked


def test_a_row_is_a_provider_with_the_schemes_rules() -> None:
    provider = _provider("mxfp8")
    assert (provider.suite, provider.route, provider.device_predicate) == (
        "cores-decode",
        "decoded_float",
        "any",
    )
    assert provider.suite in probe.DECODE_SUITES
    assert provider.required == {"weight", "scale"} and provider.schemes >= {"nvfp4", "mxfp8"}
    good = {"block_size": 32, "scale_layout": "row_major"}
    assert provider.supports((4, 64), good) == ""
    assert "block_size" in provider.supports((4, 64), {**good, "block_size": 16})
    assert "unknown" in provider.supports((4, 64), {**good, "zero": 1})
    assert provider.supports((64,), good) != "", "rank 2 or more"
    assert _provider("fp8.tensorwise").supports((64,), {"scale_dtype": "F32"}) == ""


def test_the_launch_table_serves_every_adapter_row() -> None:
    """`ADAPTERS` is empty until the scheme units add rows; `CoreDequant` serves each one."""
    providers = launch_providers()
    cores = [
        p for candidates in providers.values() for p in candidates if isinstance(p, CoreDequant)
    ]
    assert {p.encoding for p in cores} == set(CoreDequant().schemes)


@pytest.mark.skipif(not CUDA, reason="a CUDA device with Triton")
@pytest.mark.parametrize("dtype", ["float32", "bfloat16", "float16"])
def test_the_probe_oracle_passes_every_row_on_this_gpu(dtype: str) -> None:
    """The probe's `cores-decode` oracle: each row's random parts (every byte pattern) decoded by
    the kernels on the GPU, bit for bit against the eager reference; then the probe's whole
    decode evidence (oracle, spec vectors, bench, route axes)."""
    for adapter in SCHEMES:
        provider = replace(_provider(adapter.scheme), params=params_of(adapter.probe_params))
        sweep = probe._numerics_cores(torch, provider, torch.device("cuda"), getattr(torch, dtype))
        assert sweep.passed and sweep.nan_preserved, (adapter.scheme, sweep)
        evidence = probe._measured(
            torch, _provider(adapter.scheme), torch.device("cuda"), getattr(torch, dtype)
        )
        assert evidence.passed, (adapter.scheme, evidence)
