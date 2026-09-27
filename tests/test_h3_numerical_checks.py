"""H3's resident-weight scan: exact verdicts over stored widths, in few reductions."""

from typing import Any

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")

from cozy_runtime.author import OutputError  # noqa: E402
from cozy_runtime.author.fakes import fake_telemetry  # noqa: E402
from cozy_runtime.models.minimax_h3.official import NumericalChecks  # noqa: E402

_LARGE = 5000 * 4000  # above one 16M-element reduction, so the split path runs


def _module(**tensors: Any) -> Any:
    module = torch.nn.Module()
    for name, tensor in tensors.items():
        module.register_parameter(name, torch.nn.Parameter(tensor, requires_grad=False))
    return module


def _checks() -> tuple[NumericalChecks, Any]:
    telemetry = fake_telemetry()
    return NumericalChecks(telemetry), telemetry


def _verdicts(telemetry: Any) -> list[dict[str, Any]]:
    return [dict(e.fields) for e in telemetry.events if e.name == "h3 numerical check"]


def test_stored_widths_report_exact_absmax_and_reverify_only_after_a_new_fill() -> None:
    generator = torch.Generator().manual_seed(7)
    fp8 = (torch.randn(5000, 4000, generator=generator) * 0.05).to(torch.float8_e4m3fn)
    fp8.view(torch.uint8)[4321, 17] = 0x7E  # 448, e4m3fn's largest finite value
    tensors = {
        "fp8": fp8,
        "e5m2": (torch.randn(64, 64, generator=generator) * 3).to(torch.float8_e5m2),
        "bf16": torch.randn(300, 7, generator=generator).to(torch.bfloat16),
        "fp32": torch.randn(33, generator=generator),
        "complex": torch.randn(9, generator=generator, dtype=torch.complex64),
    }
    module = _module(**tensors)
    checks, telemetry = _checks()

    observation = checks._observe("resident.dit", list(module.named_parameters()), required=True)
    assert len(observation.chunks) <= 2 + len(tensors)

    checks.component("dit", module)
    verdict = _verdicts(telemetry)[-1]
    expected = max(float(t.to(torch.complex64).abs().max()) for t in tensors.values())
    assert expected == 448.0
    assert verdict["status"] == "finite_resident"
    assert verdict["absmax"] == expected
    assert verdict["absmax_tensor"] == "fp8"
    assert verdict["elements"] == sum(t.numel() for t in tensors.values())

    checks.component("dit", module)
    assert _verdicts(telemetry)[-1]["status"] == "verified_fill"
    with torch.no_grad():
        module.fp8.view(torch.uint8)[0, 0] = 0x01  # an in-place re-fill bumps the version
    checks.component("dit", module)
    assert _verdicts(telemetry)[-1]["status"] == "finite_resident"


@pytest.mark.parametrize(
    ("dtype", "pattern"),
    [(torch.float8_e4m3fn, 0x7F), (torch.float8_e4m3fn, 0xFF), (torch.float8_e5m2, 0x7C)],
)
def test_a_single_nonfinite_fp8_value_past_the_first_reduction_refuses(
    dtype: Any, pattern: int
) -> None:
    weight = torch.zeros(_LARGE, dtype=dtype)
    weight.view(torch.uint8)[_LARGE - 3] = pattern
    checks, telemetry = _checks()
    with pytest.raises(OutputError, match=r"non-finite tensor at resident\.dit/w: 1 of") as caught:
        checks.component("dit", _module(w=weight.reshape(5000, 4000)))
    assert caught.value.code == "numerical_nonfinite"
    assert [e.fields["chunk_nonfinite"] for e in telemetry.events if "non-finite" in e.name] == [1]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_wide_nonfinite_values_refuse(value: float) -> None:
    weight = torch.ones(_LARGE, dtype=torch.bfloat16)
    weight[7] = value
    checks, _ = _checks()
    with pytest.raises(OutputError, match="1 of"):
        checks.component("dit", _module(w=weight))
