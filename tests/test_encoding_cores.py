"""`CoreDequant`: each row of `decode_schemes.ADAPTERS` is a decode provider in the launch table,
decodes every one of its scheme's vectors as the probe binds it, and, on a CUDA device, passes the
probe's own `cores-decode` oracle and its whole decode evidence."""

from __future__ import annotations

from dataclasses import replace

import pytest

from cozy_runtime.internal import probe
from cozy_runtime.internal.decode_schemes import ADAPTERS, BY_SCHEME
from cozy_runtime.internal.encoding import launch_providers
from cozy_runtime.internal.encoding.adapters import CoreDequant
from cozy_runtime.internal.encoding.formats import params_of
from test_decode_cores import CUDA

torch = pytest.importorskip("torch")


def test_every_row_is_a_launched_candidate() -> None:
    providers = launch_providers()
    for adapter in ADAPTERS:
        cores = [p for p in providers[adapter.scheme] if isinstance(p, CoreDequant)]
        assert len(cores) == 1, adapter.scheme
        assert (cores[0].required, cores[0].optional) == (adapter.required, adapter.optional)
    assert len(ADAPTERS) >= 33


def test_the_probe_decodes_every_vector_through_the_provider() -> None:
    """`probe.decode_case` binds the case's params and calls `CoreDequant.decode`."""
    device = torch.device("cuda" if CUDA else "cpu")
    checked = set()
    for case in probe.vector_cases():
        if case.encoding.scheme in BY_SCHEME and not case.refuse:
            got = probe.decode_case(torch, CoreDequant(case.encoding.scheme), case, device)
            assert got is not None, case.key
            assert probe.same_bits(torch, got, case.expect) == 0, f"{case.component}/{case.key}"
            checked.add(case.encoding.scheme)
    assert len(checked) >= 30, sorted(checked)


def test_a_row_is_a_provider_with_the_schemes_rules() -> None:
    provider = CoreDequant("fp8.blockwise")
    assert (provider.suite, provider.route, provider.device_predicate) == (
        "cores-decode",
        "decoded_float",
        "any",
    )
    assert provider.suite in probe.DECODE_SUITES
    good = {"block_size": [128, 128]}
    assert provider.supports((256, 512), good) == ""
    assert "block_size" in provider.supports((256, 512), {**good, "block_size": [0, 128]})
    assert "unknown" in provider.supports((256, 512), {**good, "scale_fmt": "ue8m0"})
    assert provider.supports((512,), good) != "", "rank 2 or more"
    assert CoreDequant("gguf.q4_0").supports((64,), {"block_size": 32}) == "", "any shape"
    nvfp4 = CoreDequant("nvfp4")
    assert nvfp4.optional == {"input_scale", "pre_quant_scale"}


@pytest.mark.skipif(not CUDA, reason="a CUDA device with Triton")
@pytest.mark.parametrize("dtype", ["float32", "bfloat16", "float16"])
def test_the_probe_passes_every_row_on_this_gpu(dtype: str) -> None:
    """The `cores-decode` oracle (the row's random parts, every byte pattern, by the kernels on
    the GPU against the eager reference, bit for bit), then the whole decode evidence: oracle,
    spec vectors, bench and route axes."""
    for adapter in ADAPTERS:
        provider = replace(CoreDequant(adapter.scheme), params=params_of(adapter.probe_params))
        target = torch.device("cuda")
        sweep = probe._numerics_cores(torch, provider, target, getattr(torch, dtype))
        assert sweep.passed and sweep.nan_preserved, (adapter.scheme, sweep)
        evidence = probe._measured(
            torch, CoreDequant(adapter.scheme), target, getattr(torch, dtype)
        )
        assert evidence.passed, (adapter.scheme, evidence)
