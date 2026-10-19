"""A missing optional build tool never fails a run: the build that needs it never starts, its
fallback serves, and every run carries one warning per missing tool, with cozy's text
(`platform::tools`), a stable code and the tool. Real Diffusers attention sites, the real
SageAttention recipe, store and executor; PATH lacks the tools."""

from __future__ import annotations

import json
from collections.abc import Iterator
from functools import partial
from pathlib import Path

import pytest

from cozy_runtime.internal import (
    attention,
    attention_upstream,
    kernel_cache,
    kernel_compile,
    kernel_sources,
    tools,
)
from test_attention_wait import CPU, _host, _skip, store

__all__ = ["store"]

NVCC = (
    "nvcc not found: cards without a prebuilt kernel use the slower SDPA/kitchen path; "
    "see `cozy doctor`"
)
CC = (
    "cc not found: Triton kernels and package extensions can't build; those ops use slower "
    "fallbacks; see `cozy doctor`"
)
NINJA = "ninja not found: kernel builds compile one file at a time; see `cozy doctor`"
JOB = kernel_sources.job
#: A kernel the real SageAttention recipe builds, first in the CPU row's cell.
CUDA_TINY = attention.Candidate(
    "tiny-cuda", "_cozy_tiny_cuda", attention_upstream.SDPA, artifact="sageattention",
    import_name="tiny_cuda_kernel",
)  # fmt: skip
needs_attention = pytest.mark.skipif(bool(_skip()), reason=_skip())


@pytest.fixture
def bare(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """A PATH and CUDA home without nvcc, a C/C++ compiler or ninja, and no warning yet."""
    path = tmp_path / "bin"
    path.mkdir()
    monkeypatch.setenv("PATH", str(path))
    monkeypatch.setenv("CUDA_HOME", str(tmp_path / "no-cuda"))
    for name in ("CC", "CXX"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(tools, "_MISSING", {})
    kernel_sources.compiler.cache_clear()
    yield path
    kernel_sources.compiler.cache_clear()


def test_a_tool_is_one_row_with_cozys_text_its_first_code_and_its_name(bare: Path) -> None:
    for code in ("kernel_verify_failed", "kernel_prebuilt_missing"):
        tools.missing("nvcc", code)
    tools.missing("ninja", "kernel_prebuilt_missing")
    assert not tools.triton() and not tools.triton()
    assert tools.warnings() == [
        (NVCC, {"code": "kernel_verify_failed", "tool": "nvcc"}),
        (NINJA, {"code": "kernel_prebuilt_missing", "tool": "ninja"}),
        (CC, {"code": "triton_no_compiler", "tool": "cc"}),
    ]
    (bare / "gcc").write_text("#!/bin/sh\n")
    (bare / "gcc").chmod(0o755)
    assert tools.triton()


def _real_kernel(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "cuda-sources"
    root.mkdir()
    row = {"version": "1", "file": "sageattention.tar.gz", "sha256": "0" * 64}
    (root / "index.json").write_text(json.dumps({"schema": 1, "sources": {"sageattention": row}}))
    kernel_sources.registry.cache_clear()
    monkeypatch.setattr(kernel_sources, "job", partial(JOB, root=root))
    monkeypatch.setitem(attention.BY_NAME, CUDA_TINY.name, CUDA_TINY)
    monkeypatch.setitem(attention.DENSE, "", {16: (CUDA_TINY.name,)})


@needs_attention
def test_a_run_whose_kernel_cannot_build_serves_its_fallback_and_says_each_tool_once(
    store: kernel_cache.Store, bare: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _real_kernel(tmp_path, monkeypatch)
    with _host(tmp_path, monkeypatch) as host:
        [fallback] = host.construction.attention.fallbacks
        assert fallback["preferred"] == "tiny-cuda" and fallback["selected"] == "sdpa"
        assert fallback["reason"] == "AttentionRefusal: tiny-cuda: nvcc and cc not found"
        for request in ("first", "second"):
            assert host.run(request)["observed"] == "sdpa"
            rows = host.replies[-1]["observations"]
            warned = [(r["name"], r["fields"]) for r in rows if "tool" in r.get("fields", {})]
            assert warned == [
                (NVCC, {"code": "kernel_prebuilt_missing", "tool": "nvcc"}),
                (CC, {"code": "kernel_prebuilt_missing", "tool": "cc"}),
            ]
    assert kernel_compile._STARTED == {}  # no doomed build started


@needs_attention
def test_a_build_without_ninja_still_runs_and_a_failed_prebuilt_names_its_code(
    store: kernel_cache.Store, bare: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _real_kernel(tmp_path, monkeypatch)
    for name in ("nvcc", "c++"):
        (bare / name).write_text("#!/bin/sh\necho fake 1.0\n")
        (bare / name).chmod(0o755)
    attention._STATES["tiny-cuda prebuilt"] = kernel_compile.State("prebuilt", "failed")
    assert attention._job(CUDA_TINY, CPU).kind == "setup"
    assert tools.warnings() == [(NINJA, {"code": "kernel_verify_failed", "tool": "ninja"})]
