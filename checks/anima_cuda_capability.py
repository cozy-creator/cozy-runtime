#!/usr/bin/env python3
"""Deterministic gate for the optional fused Anima CUDA kernel capability."""

from __future__ import annotations

import hashlib
import inspect
import sys
import tomllib
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from cozy_runtime.internal import anima_optimization as subject

ROOT = Path(__file__).resolve().parents[1]
COMPANION = ROOT / "packages" / "cozy-runtime-cuda-kernels"
UPSTREAM_NATIVE = {
    "rms_rope.cu": "eb44a105487459fd2a8fe1f69f7252c8c3840d5132ecc3598e0b3e591dc7d715",
    "rope_device.cuh": "94f76011a78bf4e8406bdb79b90baee6d234b7ab47bdee02127d1a8c24e2f02c",
    "dtype_dispatch.cuh": "5a155c3bf2123604a0689ad792016b703c45d9676a389381b482ac1f1afb23bc",
    "utils.cuh": "a3482d9911029f017c64b39dae5198f76a4c9585b8ede5c734643bcc18d04a4d",
}


def source_custody() -> None:
    for name, expected in UPSTREAM_NATIVE.items():
        actual = hashlib.sha256((COMPANION / "native" / name).read_bytes()).hexdigest()
        assert actual == expected, f"{name} drifted from its attributed upstream source"
    native_files = {path.name for path in (COMPANION / "native").iterdir() if path.is_file()}
    assert native_files == {*UPSTREAM_NATIVE, "binding.cpp"}
    cmake = (COMPANION / "CMakeLists.txt").read_text()
    assert 'set(CMAKE_CUDA_ARCHITECTURES "75-real;80-real;89;90a-real;100f;120f")' in cmake
    project = tomllib.loads((COMPANION / "pyproject.toml").read_text())
    assert project["build-system"]["requires"] == [
        "scikit-build-core==1.0.3",
        "nanobind==3.0.1",
    ]
    assert project["project"]["name"] == subject.FUSED_ANIMA_CUDA_KERNEL_DISTRIBUTION
    assert project["project"]["version"] == subject.FUSED_ANIMA_CUDA_KERNEL_VERSION
    assert project["project"]["dependencies"] == ["torch==2.13.0"]
    runtime_project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert runtime_project["project"]["optional-dependencies"]["cuda-kernels"] == [
        "cozy-runtime-cuda-kernels==0.1.0+torch2.13cu130; "
        "sys_platform == 'linux' and platform_machine == 'x86_64'"
    ]
    for required in ("LICENSE", "NOTICE", "UPSTREAM.md"):
        assert (COMPANION / required).is_file(), f"companion omitted {required}"

    sys.path.insert(0, str(COMPANION / "src"))
    try:
        import cozy_runtime_cuda_kernels as kernels

        digest = hashlib.sha256(inspect.getsource(kernels.rms_rope_split_half).encode()).hexdigest()
        assert digest == subject._SOURCE_DIGESTS["cozy_runtime_cuda_kernels.rms_rope_split_half"]
    finally:
        sys.path.pop(0)


def probe(
    *,
    torch_version: str | None = "2.13.0+cu130",
    built_cuda: str | None = "13.0",
    cuda_available: bool = True,
    native_version: str | None = "0.1.0+torch2.13cu130",
    native_importable: bool = True,
    through_apply: bool = False,
) -> tuple[object | None, list[str]]:
    imports: list[str] = []
    fake_torch = SimpleNamespace(
        version=SimpleNamespace(cuda=built_cuda),
        cuda=SimpleNamespace(is_available=lambda: cuda_available),
    )
    fake_kernels = SimpleNamespace(rms_rope_split_half=object())

    def installed(distribution: str) -> str | None:
        value = (
            torch_version
            if distribution == "torch"
            else native_version
            if distribution == subject.FUSED_ANIMA_CUDA_KERNEL_DISTRIBUTION
            else None
        )
        return value

    def importing(module: str) -> Any:
        imports.append(module)
        if module == "torch":
            return fake_torch
        if module == subject.FUSED_ANIMA_CUDA_KERNEL_MODULE and native_importable:
            return fake_kernels
        raise ModuleNotFoundError(module)

    with (
        patch.object(subject, "_installed_version", side_effect=installed),
        patch.object(subject, "import_module", side_effect=importing),
    ):
        if through_apply:
            return (
                subject.apply_anima_execution_plan(
                    object(), application="anima:app", model_class="anima:AnimaModel"
                ),
                imports,
            )
        return subject._fused_anima_cuda_capability(), imports


def main() -> None:
    source_custody()
    for label, options, expected_imports in (
        ("torch-absent", {"torch_version": None}, []),
        ("torch-version", {"torch_version": "2.12.0+cu126"}, []),
        ("cpu", {"torch_version": "2.13.0+cpu", "built_cuda": None}, ["torch"]),
        ("cuda-unavailable", {"cuda_available": False}, ["torch"]),
        ("cuda-version", {"built_cuda": "12.6"}, ["torch"]),
        ("native-absent", {"native_version": None}, ["torch"]),
        ("native-version", {"native_version": "0.0.9+torch2.13cu130"}, ["torch"]),
        (
            "native-import",
            {"native_importable": False},
            ["torch", subject.FUSED_ANIMA_CUDA_KERNEL_MODULE],
        ),
    ):
        applied, imports = probe(through_apply=True, **options)
        assert applied is None, f"{label} selected fused kernels"
        assert imports == expected_imports, f"{label} imports = {imports}"

    capability, imports = probe()
    assert capability is not None and imports == ["torch", subject.FUSED_ANIMA_CUDA_KERNEL_MODULE]

    sentinel = subject.AppliedAnimaOptimization("exact", "test", ("fused",))
    with (
        patch.object(subject, "_fused_anima_cuda_capability", return_value=capability) as gate,
        patch.object(subject, "_apply_fused_anima_cuda_plan", return_value=sentinel) as apply,
    ):
        assert (
            subject.apply_anima_execution_plan(
                object(), application="other:app", model_class="anima:AnimaModel"
            )
            is None
        )
        gate.assert_not_called()
        assert (
            subject.apply_anima_execution_plan(
                object(), application="anima:app", model_class="anima:AnimaModel"
            )
            is sentinel
        )
        apply.assert_called_once()

    strict = inspect.getsource(subject._apply_fused_anima_cuda_plan)
    for guard in (
        "_require_source",
        "_require_transformer_config",
        "_tensor_census",
        "anima_optimization_topology_changed",
    ):
        assert guard in strict, f"strict fused plan lost {guard}"
    print("fused Anima CUDA capability: eager fallback arms + exact supported apply passed")


if __name__ == "__main__":
    main()
