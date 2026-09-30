"""Architecture-only Sage setup target is keyed and reaches the existing builder."""

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import pytest

from cozy_runtime.internal import config, kernel_compile, kernel_sources


def test_sage_architecture_is_explicit_keyed_and_gpu_hidden(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = kernel_sources.Source("sageattention", "pinned", "unused.tar.gz", "a" * 64)
    monkeypatch.setattr(kernel_sources, "registry", lambda _: ({"sageattention": source}, ""))
    monkeypatch.setattr(kernel_sources, "torch_build", lambda: ("2.14.0", "13.0"))
    monkeypatch.setattr(kernel_sources, "compiler", lambda: "qualified-toolchain")
    first = kernel_sources.job("sageattention", 89, tmp_path)
    second = kernel_sources.job("sageattention", 90, tmp_path)
    assert first.key != second.key
    assert first.env == {"CUDA_VISIBLE_DEVICES": "", "PYTORCH_NVML_BASED_CUDA_CHECK": "1"}
    assert first.spec["target_environment"] == {
        "TORCH_CUDA_ARCH_LIST": "8.9",
        "CUDA_VISIBLE_DEVICES": "",
    }
    assert second.spec["target_environment"]["TORCH_CUDA_ARCH_LIST"] == "9.0"


def test_existing_setup_builder_passes_target_over_ambient_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    recipe = kernel_sources.RECIPES["sageattention"]
    tree = tmp_path / "source"
    tree.mkdir()
    monkeypatch.setattr(kernel_sources, "_extract", lambda *_: tree)
    monkeypatch.setattr(kernel_sources, "_units", lambda _: {})
    monkeypatch.setattr(kernel_sources, "_dist_info", lambda *_: None)
    monkeypatch.setattr(
        config,
        "inherited_environment",
        lambda: {
            "CUDA_VISIBLE_DEVICES": "0",
            "TORCH_CUDA_ARCH_LIST": "12.0",
        },
    )
    observed: list[Mapping[str, str]] = []

    def build(
        argv: Sequence[str], *, cwd: Path, env: Mapping[str, str], report: Callable[[float], None]
    ) -> str:
        observed.append(env)
        work = Path(argv[argv.index("--egg-base") + 1])
        info = work / "sageattention.egg-info"
        info.mkdir()
        (info / "PKG-INFO").write_text("Name: sageattention\nVersion: 2.2.0\n")
        site = Path(argv[argv.index("--build-lib") + 1])
        site.mkdir(parents=True)
        return ""

    monkeypatch.setattr(kernel_compile, "run", build)
    monkeypatch.setattr(kernel_compile, "width", lambda _: 1)
    kernel_sources.build_setup(
        {
            "recipe": recipe.name,
            "sources": {"sageattention": {"version": "pinned"}},
            "target_environment": {"TORCH_CUDA_ARCH_LIST": "8.9", "CUDA_VISIBLE_DEVICES": ""},
        },
        tmp_path / "stage",
        lambda _: None,
    )
    assert observed[0]["TORCH_CUDA_ARCH_LIST"] == "8.9"
    assert observed[0]["CUDA_VISIBLE_DEVICES"] == ""
