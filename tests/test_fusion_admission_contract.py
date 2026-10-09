"""The admission fixture passes an integer architecture, not a torch device."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import get_type_hints

import pytest

from cozy_runtime.internal import fusion, fusion_install


def admission_module() -> ModuleType:
    path = (
        Path(__file__).parents[1]
        / "research/h3-fusion-capability/admission/h3_fusion_capability/__init__.py"
    )
    spec = importlib.util.spec_from_file_location("h3_fusion_admission_contract", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fixture_architecture_matches_runtime_and_actual_triton_contract() -> None:
    pytest.importorskip("triton")
    from triton.backends.compiler import GPUTarget
    from triton.backends.nvidia.compiler import CUDABackend

    module = admission_module()
    assert get_type_hints(fusion.expect)["arch"] is int
    for capability, expected in [((8, 9), 89), ((9, 0), 90), ((12, 0), 120), ((12, 3), 123)]:
        arch = module.compile_architecture(capability)
        assert type(arch) is int and arch == expected
        options = CUDABackend(GPUTarget("cuda", arch, 32)).parse_options({"num_warps": 16})
        assert options.arch == f"sm{arch}"


@pytest.mark.parametrize(
    "mode", ["bf16", "encoded", "encoded_inactive", "encoded_baked", "encoded_unbaked"]
)
def test_all_fixture_constructors_use_actual_leaf_and_lora_interfaces(mode: str) -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("diffusers")
    module = admission_module()
    with torch.inference_mode():
        case = module.construct_case(mode, torch.device("cpu"), hidden_size=256, ffn_dim=512)
        fusion_install._validate_transformer(case.model, "fl2va_dit")
        assert len(case.slots) == 5
        assert len(case.hooks) == (5 if mode.startswith("encoded_") else 0)
        for parent, name in case.slots:
            leaf = getattr(parent, name)
            if mode == "bf16":
                assert isinstance(leaf, torch.nn.Linear)
            else:
                assert leaf.encoding == "fp8-rowwise/1"
                assert leaf.provider == "cozy.fp8-rowwise.native-leaf/1"
                assert leaf.out_dtype == torch.bfloat16
                assert tuple(leaf.data.shape) == (leaf.out_features, leaf.in_features)
                assert leaf.scale.numel() == leaf.out_features
            assert fusion_install._rowwise_leaf(leaf) is (mode == "encoded")
