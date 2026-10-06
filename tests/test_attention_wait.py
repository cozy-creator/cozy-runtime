"""A site waits for the kernel it will run while that kernel compiles, and serves no other in
its place: real Diffusers attention sites, a real source registry, real builder processes and
the real store. The kernel is a registry wheel a CPU can build, named in a model's ladder and
in the ranked table.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
import os
import sys
import threading
import time
from collections.abc import Iterator
from functools import partial
from pathlib import Path
from typing import Any

import pytest

from cozy_runtime.internal import (
    attention,
    attention_upstream,
    kernel_cache,
    kernel_compile,
    kernel_sources,
    liveness,
)
from cozy_runtime.internal.encoding import DeviceFacts
from test_kernel_compile import _held, _registry

CPU = DeviceFacts(kind="cpu", name="cpu", sm=0, driver="", configuration="", index=0)
TINY = attention.Candidate(
    name="tiny",
    backend="_cozy_tiny",
    local=attention_upstream.SDPA,
    min_sm=0,
    dtypes=frozenset(),
    max_head_dim=0,
    artifact=kernel_sources.PYTHON,
    import_name="tiny_kernel",
)
#: A live builder that keeps moving bytes for as long as a test holds its key.
MOVING = "import os, time\nfor _ in range(3000): os.write(2, b'.'); time.sleep(0.02)"


def _skip() -> str:
    if importlib.util.find_spec("torch") is None:
        return "torch is not installed"
    try:
        installed = importlib.metadata.version("diffusers")
    except importlib.metadata.PackageNotFoundError:
        return "diffusers is not installed"
    return "" if installed == "0.40.0" else f"diffusers is {installed}, the proof needs 0.40.0"


pytestmark = pytest.mark.skipif(bool(_skip()), reason=_skip())


def _sites() -> dict[str, object]:
    import torch
    from diffusers.models.transformers.transformer_wan import WanAttention, WanAttnProcessor

    sites = [
        WanAttention(dim=64, heads=2, dim_head=32, processor=WanAttnProcessor()) for _ in range(2)
    ]
    return {"dit": torch.nn.ModuleList(sites)}


@pytest.fixture
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[kernel_cache.Store]:
    """A machine run's store and scratch scope, and `tiny` known to selection."""
    scope = tmp_path / "run"
    scope.mkdir()
    monkeypatch.setenv("TMPDIR", str(scope))
    monkeypatch.setattr("tempfile.tempdir", None)
    monkeypatch.setattr(
        kernel_sources, "job", partial(kernel_sources.job, root=tmp_path / "sources")
    )
    monkeypatch.setattr(liveness, "SAMPLE_SECONDS", 0.05)
    monkeypatch.setattr(kernel_compile, "POLL_SECONDS", 0.01)
    monkeypatch.setitem(attention.BY_NAME, "tiny", TINY)
    for table in ("_READY", "_BROKEN", "_STATES"):
        monkeypatch.setattr(attention, table, {})
    found = kernel_cache.Store(tmp_path / "u1")
    monkeypatch.setattr(kernel_cache, "_MACHINE_STORE", [found])
    path = list(sys.path)
    yield found
    sys.path[:] = path
    sys.modules.pop("tiny_kernel", None)
    importlib.invalidate_caches()


def _ladder(*names: str) -> Any:
    return lambda context: names


def test_a_ladder_site_waits_for_its_kernels_live_build_and_serves_it(
    store: kernel_cache.Store, tmp_path: Path
) -> None:
    _registry(tmp_path / "sources")
    lock, builder = _held(store, kernel_sources.job(kernel_sources.PYTHON, 0), MOVING)
    try:
        threading.Timer(1.0, os.close, [lock]).start()
        started = time.monotonic()
        applied = attention.select(CPU, _sites(), choose=_ladder("tiny", "sdpa"))
        assert applied.hosts == {"dit": {"tiny": 2}}, applied
        assert time.monotonic() - started >= 1.0 and not applied.skipped
        assert (applied.artifacts or {})["tiny"]["artifact"] == kernel_sources.PYTHON
    finally:
        builder.kill()
        builder.wait()


def test_only_a_site_that_runs_the_compiling_ranked_kernel_waits_for_it(
    store: kernel_cache.Store, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _registry(tmp_path / "sources")
    monkeypatch.setattr(attention, "CANDIDATES", (TINY, attention.BY_NAME["sdpa"]))
    job = kernel_sources.job(kernel_sources.PYTHON, 0)
    lock, builder = _held(store, job, MOVING)
    try:
        # What is ready now, and a model that prefers SDPA: neither waits on the build.
        skipped: dict[str, str] = {}
        assert [entry.name for entry in attention.available(CPU, skipped)] == ["sdpa"]
        assert skipped == {}
        other = attention.select(CPU, _sites(), choose=_ladder("sdpa"))
        assert other.hosts == {"dit": {"sdpa": 2}}
        assert kernel_compile.status(store, job).state == "compiling"
        # A site with no preference runs the best ranked kernel: it waits for it.
        threading.Timer(1.0, os.close, [lock]).start()
        started = time.monotonic()
        applied = attention.select(CPU, _sites())
        assert applied.hosts == {"dit": {"tiny": 2}}, applied
        assert time.monotonic() - started >= 1.0 and not applied.skipped
    finally:
        builder.kill()
        builder.wait()


def test_a_kernel_whose_compile_failed_is_skipped_with_why_and_the_next_one_serves(
    store: kernel_cache.Store, tmp_path: Path
) -> None:
    _registry(tmp_path / "sources", corrupt=True)
    applied = attention.select(CPU, _sites(), choose=_ladder("tiny", "sdpa"))
    assert applied.hosts == {"dit": {"sdpa": 2}}
    assert "compile_source_mismatch" in (applied.skipped or {})["tiny"], applied.skipped
    assert attention.evidence(["sdpa"])[1]["state"] == "failed"
