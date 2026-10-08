"""The admission fixture passes an integer architecture, not a torch device."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import get_type_hints

import pytest

from cozy_runtime.internal import fusion


def test_fixture_architecture_matches_runtime_and_actual_triton_contract() -> None:
    pytest.importorskip("triton")
    from triton.backends.compiler import GPUTarget
    from triton.backends.nvidia.compiler import CUDABackend

    path = (
        Path(__file__).parents[1]
        / "research/h3-fusion-capability/admission/h3_fusion_capability/__init__.py"
    )
    spec = importlib.util.spec_from_file_location("h3_fusion_admission_contract", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert get_type_hints(fusion.expect)["arch"] is int
    for capability, expected in [((8, 9), 89), ((9, 0), 90), ((12, 0), 120), ((12, 3), 123)]:
        arch = module.compile_architecture(capability)
        assert type(arch) is int and arch == expected
        options = CUDABackend(GPUTarget("cuda", arch, 32)).parse_options({"num_warps": 16})
        assert options.arch == f"sm{arch}"
