"""A run never waits for a compile: a site whose kernel still compiles serves the best one
ready, and the construction takes the kernel at its next selection once it is built. Real
Diffusers attention sites, a real source registry, real builder processes and the real store;
the kernel is a registry wheel a CPU can build, named in a model's ladder and in the ranked
table.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
import os
import sys
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


def _built(store: kernel_cache.Store, job: kernel_compile.Job) -> None:
    kernel_compile.submit(store, job)  # nobody builds it now: this starts the build
    while kernel_compile.status(store, job).state != "ready":
        time.sleep(kernel_compile.POLL_SECONDS)


@pytest.mark.parametrize("choose", [_ladder("tiny", "sdpa"), None], ids=["ladder", "ranked"])
def test_a_site_serves_what_is_ready_while_its_kernel_compiles_and_takes_it_once_built(
    store: kernel_cache.Store, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, choose: Any
) -> None:
    _registry(tmp_path / "sources")
    monkeypatch.setattr(attention, "CANDIDATES", (TINY, attention.BY_NAME["sdpa"]))
    job = kernel_sources.job(kernel_sources.PYTHON, 0)
    lock, builder = _held(store, job, MOVING)
    try:
        first = attention.select(CPU, _sites(), choose=choose)
        # Selected while the build still runs: nothing waited for it.
        assert kernel_compile.status(store, job).state == "compiling"
        assert first.hosts == {"dit": {"sdpa": 2}} and first.waiting == ("tiny",), first
        assert not attention.compiled(first.waiting, CPU)
    finally:
        os.close(lock)
        builder.kill()
        builder.wait()
    _built(store, job)
    assert attention.compiled(first.waiting, CPU)
    later = attention.select(CPU, _sites(), choose=choose)
    assert later.hosts == {"dit": {"tiny": 2}} and later.waiting == (), later


def test_a_kernel_whose_compile_failed_is_skipped_with_why_and_the_next_one_serves(
    store: kernel_cache.Store, tmp_path: Path
) -> None:
    _registry(tmp_path / "sources", corrupt=True)
    job = kernel_sources.job(kernel_sources.PYTHON, 0)
    kernel_compile.submit(store, job)
    while kernel_compile.status(store, job).state != "failed":
        time.sleep(kernel_compile.POLL_SECONDS)
    applied = attention.select(CPU, _sites(), choose=_ladder("tiny", "sdpa"))
    assert applied.hosts == {"dit": {"sdpa": 2}} and applied.waiting == ()
    assert "compile_source_mismatch" in (applied.skipped or {})["tiny"], applied.skipped
    assert attention.evidence(["sdpa"])[1]["state"] == "failed"
