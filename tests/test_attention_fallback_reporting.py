"""Selection-time fallback reasons survive real CPU execution and compile completion."""

from __future__ import annotations

import importlib.util
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import pytest

from cozy_runtime.internal import attention, execution_evidence, kernel_cache, kernel_sources
from cozy_runtime.internal.executor import Executor
from test_attention_wait import (
    CPU,
    MOVING,
    _finisher,
    _host,
    _ladder,
    _sites,
    _waiting,
    store,
)
from test_kernel_compile import _held, _registry

# Imported fixture retains the real private kernel store and builder subprocess setup.
__all__ = ["store"]

requires_attention_dependencies = pytest.mark.skipif(
    any(importlib.util.find_spec(name) is None for name in ("torch", "diffusers")),
    reason="real attention execution requires Torch and Diffusers",
)


@requires_attention_dependencies
def test_cold_reason_survives_compile_completion_and_next_request_is_clean(
    store: kernel_cache.Store, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _registry(tmp_path / "sources")
    job = kernel_sources.job(kernel_sources.PYTHON, 0)
    lock, builder = _held(store, job, MOVING)
    try:
        with _host(tmp_path, monkeypatch) as host:
            first = host.construction.attention
            [fallback] = first.fallbacks
            assert fallback["component"] == "dit"
            assert fallback["preferred"] == "tiny" and fallback["selected"] == "sdpa"
            assert "compil" in fallback["reason"].lower()
            assert "failed" not in fallback["reason"].lower()
            assert "selected sdpa instead of tiny" in first.line()
            assert first.document()["fallbacks"] == [fallback]
            merged = attention.merge({"base": first.document()})
            assert merged is not None
            assert merged["fallbacks"] == [{**fallback, "component": "base/dit"}]
            host.package.FINISH[:] = [_finisher(store, job, lock, builder)]
            lock = -1
            switched = host.run("switched")
            assert switched["fallbacks"] == [fallback]
            assert "tiny" in switched["observed"] and "sdpa" in switched["observed"]
            assert any(
                row["kernel"] == "tiny" and row["state"] == "ready" for row in switched["kernels"]
            )
            rows = host.replies[-1]["observations"]
            receipt = next(row for row in rows if row["name"] == "attention.fallback")
            assert receipt["fields"] == fallback
            host.package.FINISH[:] = [lambda: None]
            warm = host.run("warm")
            assert warm["observed"] == "tiny" and "fallbacks" not in warm
    finally:
        if lock >= 0:
            os.close(lock)
        if builder.poll() is None:
            builder.kill()
        builder.wait()


@requires_attention_dependencies
def test_explicit_pin_is_not_mislabeled_as_an_automatic_fallback(
    store: kernel_cache.Store, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _registry(tmp_path / "sources")
    job = kernel_sources.job(kernel_sources.PYTHON, 0)
    lock, builder = _held(store, job, MOVING)
    try:
        with _host(tmp_path, monkeypatch) as host:
            assert host.construction.attention.fallbacks
            pinned = host.run("pinned", "sdpa")
            assert pinned["observed"] == "sdpa" and "fallbacks" not in pinned
            assert not any(
                row["name"] == "attention.fallback" for row in host.replies[-1]["observations"]
            )
    finally:
        os.close(lock)
        builder.kill()
        builder.wait()


@requires_attention_dependencies
def test_failed_build_retains_actual_failure_reason(
    store: kernel_cache.Store, tmp_path: Path
) -> None:
    from cozy_runtime.internal import kernel_compile

    _registry(tmp_path / "sources", corrupt=True)
    job = kernel_sources.job(kernel_sources.PYTHON, 0)
    state = kernel_compile.wait(store, job, "tiny")
    assert state.state == "failed"
    selected = attention.select(CPU, _sites(), choose=_ladder("tiny", "sdpa"))
    [fallback] = selected.fallbacks
    assert fallback["preferred"] == "tiny" and fallback["selected"] == "sdpa"
    assert "compile_source_mismatch" in fallback["reason"]
    assert selected.waiting == ()


def test_scoped_pin_does_not_hide_other_components_or_invent_a_served_kernel(
    tmp_path: Path,
) -> None:
    host = Executor(None, tmp_path)  # type: ignore[arg-type]
    row = execution_evidence.AttentionFallback(
        component="dit", preferred="preferred", selected="fallback", reason="still compiling"
    )
    host.models = {"base": object()}
    host.attention = attention.Applied({"dit": {"fallback": 2}}, (), fallbacks=(row,))
    observed = attention.Applied({"base/dit": {"fallback": 2}}, (), pin="vae=sdpa")
    assert host._attention_selection(observed).fallbacks == ({**row, "component": "base/dit"},)
    assert not host._attention_selection(replace(observed, pin="dit=fallback")).fallbacks
    assert not host._attention_selection(
        replace(observed, hosts={"base/dit": {"other": 2}})
    ).fallbacks


@requires_attention_dependencies
def test_explicit_compiling_pin_still_waits_and_never_falls_back(
    store: kernel_cache.Store, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _registry(tmp_path / "sources")
    job = kernel_sources.job(kernel_sources.PYTHON, 0)
    lock, builder = _held(store, job, MOVING)
    try:
        with _host(tmp_path, monkeypatch) as host, ThreadPoolExecutor() as threads:
            future = threads.submit(host.run, "waits", "tiny")
            while not future.done() and not _waiting(host.frames):
                time.sleep(0.01)
            assert not future.done() and not host.held
            _finisher(store, job, lock, builder)()
            lock = -1
            reply = future.result()
            assert reply.get("observed") == "tiny", reply
            assert "fallbacks" not in reply
            assert len(host.held) == 4 and all(names == ["_cozy_tiny"] * 2 for names in host.held)
    finally:
        if lock >= 0:
            os.close(lock)
        if builder.poll() is None:
            builder.kill()
        builder.wait()
