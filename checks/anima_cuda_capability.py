#!/usr/bin/env python3
"""Deterministic gate for the optional fused Anima CUDA kernel capability."""

from __future__ import annotations

import inspect
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from cozy_runtime.internal import anima_optimization as subject


def probe(
    *,
    torch_version: str | None = "2.13.0+cu130",
    built_cuda: str | None = "13.0",
    cuda_available: bool = True,
    native_version: str | None = "0.2.31",
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
        value = torch_version if distribution == "torch" else native_version
        return value

    def importing(module: str) -> Any:
        imports.append(module)
        if module == "torch":
            return fake_torch
        if module == "comfy_kitchen" and native_importable:
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
    for label, options, expected_imports in (
        ("torch-absent", {"torch_version": None}, []),
        ("torch-version", {"torch_version": "2.12.0+cu126"}, []),
        ("cpu", {"torch_version": "2.13.0+cpu", "built_cuda": None}, ["torch"]),
        ("cuda-unavailable", {"cuda_available": False}, ["torch"]),
        ("cuda-version", {"built_cuda": "12.6"}, ["torch"]),
        ("native-absent", {"native_version": None}, ["torch"]),
        ("native-version", {"native_version": "0.2.30"}, ["torch"]),
        (
            "native-import",
            {"native_importable": False},
            ["torch", "comfy_kitchen"],
        ),
    ):
        applied, imports = probe(through_apply=True, **options)
        assert applied is None, f"{label} selected fused kernels"
        assert imports == expected_imports, f"{label} imports = {imports}"

    capability, imports = probe()
    assert capability is not None and imports == ["torch", "comfy_kitchen"]

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
