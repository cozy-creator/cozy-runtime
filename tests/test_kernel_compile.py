"""The compile manager through real builder processes, a real registry and real `flock`s.

A registry wheel is the smallest real artifact: `submit` starts the builder, the builder
publishes the unpacked tree whole, and every later submit reads it. Failures stay failed for
the boot that saw them; a wedged compiler is ended by its own stillness, never a clock.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import textwrap
import time
import zipfile
from pathlib import Path

import pytest

from cozy_runtime.internal import kernel_cache, kernel_compile, kernel_sources, liveness


def _registry(root: Path, *, corrupt: bool = False) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    wheel = root / "tiny_kernel-1.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("tiny_kernel/__init__.py", "VALUE = 42\n")
        archive.writestr("tiny_kernel-1.0.dist-info/METADATA", "Name: tiny-kernel\nVersion: 1.0\n")
        archive.writestr("tiny_kernel-1.0.data/purelib/tiny_extra.py", "EXTRA = 1\n")
    digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
    rows = {"tiny-kernel": {"version": "1.0", "file": wheel.name, "sha256": digest}}
    if corrupt:
        rows["tiny-kernel"]["sha256"] = "0" * 64
    (root / "index.json").write_text(json.dumps({"schema": 1, "sources": rows}))
    kernel_sources.registry.cache_clear()
    return root


def _wait(store: kernel_cache.Store, job: kernel_compile.Job) -> kernel_compile.State:
    for _ in range(600):
        state = kernel_compile.status(store, job)
        if state.state in ("ready", "failed"):
            return state
        time.sleep(0.05)
    raise AssertionError(f"still {state}")


@pytest.fixture(autouse=True)
def _boot_scope(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A worker boot's scratch scope: failures recorded here are this boot's."""
    scope = tmp_path / "boot"
    scope.mkdir()
    monkeypatch.setenv("TMPDIR", str(scope))
    monkeypatch.setattr("tempfile.tempdir", None)


def test_a_wheel_artifact_is_built_once_published_whole_and_read_after(tmp_path: Path) -> None:
    root = _registry(tmp_path / "sources")
    store = kernel_cache.Store(tmp_path / "u1")
    job = kernel_sources.job(kernel_sources.PYTHON, 89, root)
    assert json.loads(job.key.inputs)["sources"] == {
        "tiny-kernel": json.loads((root / "index.json").read_text())["sources"]["tiny-kernel"][
            "sha256"
        ]
    }
    first = kernel_compile.submit(store, job)
    assert first.state in ("compiling", "ready")
    state = _wait(store, job)
    assert state.state == "ready" and state.ms > 0, state
    site = Path(state.path) / kernel_cache.SITE
    assert (site / "tiny_kernel" / "__init__.py").read_text() == "VALUE = 42\n"
    assert (site / "tiny_extra.py").exists()  # .data/purelib folds into the root
    assert not list((tmp_path / "u1" / job.key.kernel).glob(".tmp-*"))
    assert kernel_compile.submit(store, job).state == "ready"
    kernel_compile._STARTED[job.key.digest].wait()
    assert store.state(job.key) is None  # the builder's word ends with it


def test_concurrent_submitters_on_one_machine_build_once(tmp_path: Path) -> None:
    root = _registry(tmp_path / "sources")
    own = tmp_path / "u1"
    code = textwrap.dedent(
        f"""
        import json, time
        from pathlib import Path
        from cozy_runtime.internal import kernel_cache, kernel_compile, kernel_sources
        store = kernel_cache.Store(Path({str(own)!r}))
        job = kernel_sources.job(kernel_sources.PYTHON, 89, Path({str(root)!r}))
        kernel_compile.submit(store, job)
        while (state := kernel_compile.status(store, job)).state != "ready":
            time.sleep(0.05)
        print(json.dumps(json.loads((Path(state.path) / "entry.json").read_text())["producer"]))
        """
    )
    processes = [
        subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, env=os.environ)
        for _ in range(4)
    ]
    producers = {p.communicate()[0].decode().strip().splitlines()[-1] for p in processes}
    assert all(p.returncode == 0 for p in processes)
    assert len(producers) == 1  # one builder published; the rest read its entry


def test_a_failure_is_this_boots_and_the_next_boot_retries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _registry(tmp_path / "sources", corrupt=True)
    store = kernel_cache.Store(tmp_path / "u1")
    job = kernel_sources.job(kernel_sources.PYTHON, 89, root)
    kernel_compile.submit(store, job)
    failed = _wait(store, job)
    assert failed.state == "failed" and "compile_source_mismatch" in failed.detail, failed
    assert kernel_compile.submit(store, job).state == "failed"  # never per request
    assert store.entry(job.key) is None
    fresh = tmp_path / "boot2"
    fresh.mkdir()
    monkeypatch.setenv("TMPDIR", str(fresh))
    monkeypatch.setattr("tempfile.tempdir", None)
    kernel_compile._STARTED[job.key.digest].wait()  # its lock frees as it exits
    assert kernel_compile.status(store, job).state == "pending"


def test_a_stale_builder_word_does_not_hold_the_key(tmp_path: Path) -> None:
    """A builder killed outright leaves its progress file; its lock died with it."""
    root = _registry(tmp_path / "sources")
    store = kernel_cache.Store(tmp_path / "u1")
    job = kernel_sources.job(kernel_sources.PYTHON, 89, root)
    store.set_state(job.key, {"pid": 1, "progress": 0.5, "started_unix_ms": 1})
    assert kernel_compile.status(store, job).state == "pending"
    kernel_compile.submit(store, job)
    assert _wait(store, job).state == "ready"


def test_progress_is_ninjas_and_a_silent_idle_compiler_is_ended(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[float] = []
    units = ((1, 2), (1, 4), (2, 2), (2, 4), (3, 4), (4, 4))  # two extensions at once
    lines = "; ".join(f"print('[{d}/{t}] x')" for d, t in units)
    output = kernel_compile.run(
        [sys.executable, "-c", lines], cwd=tmp_path, env=os.environ, report=seen.append
    )
    assert seen == [0.5, 1 / 3, 0.5, 2 / 3, 5 / 6, 1.0] and "[4/4] x" in output
    with pytest.raises(kernel_compile.CompileFailed) as failed:
        kernel_compile.run(
            [sys.executable, "-c", "raise SystemExit(3)"],
            cwd=tmp_path,
            env=os.environ,
            report=seen.append,
        )
    assert failed.value.code == "compile_failed"
    monkeypatch.setattr(liveness, "SAMPLE_SECONDS", 0.05)
    started = time.monotonic()
    with pytest.raises(kernel_compile.CompileFailed) as stalled:
        kernel_compile.run(
            [sys.executable, "-c", "import time; print('[1/9] x', flush=True); time.sleep(600)"],
            cwd=tmp_path,
            env=os.environ,
            report=seen.append,
        )
    assert stalled.value.code == "compile_stalled" and time.monotonic() - started < 60
    assert "no measurable progress" in str(stalled.value)


def test_an_image_without_sources_or_toolkit_says_absent(tmp_path: Path) -> None:
    kernel_sources.registry.cache_clear()
    with pytest.raises(kernel_sources.Absent, match="carries no kernel sources"):
        kernel_sources.job(kernel_sources.PYTHON, 89, tmp_path / "nowhere")
    root = _registry(tmp_path / "sources")
    with pytest.raises(kernel_sources.Absent, match="sageattention"):
        kernel_sources.job("sageattention", 89, root)


def test_learned_artifacts_are_this_machines_and_replay_as_the_same_jobs(tmp_path: Path) -> None:
    root = _registry(tmp_path / "sources")
    store = kernel_cache.Store(tmp_path / "u1")
    job = kernel_sources.job(kernel_sources.PYTHON, 89, root)
    assert kernel_compile.learned(store, "scope") == []
    for _ in range(3):
        kernel_compile.learn(store, "scope", job)
    [again] = kernel_compile.learned(store, "scope")
    assert (again.kind, again.key, dict(again.spec)) == (job.kind, job.key, dict(job.spec))
    assert kernel_compile.learned(store, "another-card") == []
    kernel_compile.submit(store, again)
    assert _wait(store, job).state == "ready"


def test_the_machines_ordered_builds_are_every_executors(tmp_path: Path) -> None:
    """The worker claims its boot builds at once and runs them in its order; an executor in
    its own namespace sees each as compiling while it waits its turn, starts none of its own,
    and reads it from the worker's namespace once built."""
    root = _registry(tmp_path / "sources")
    worker = kernel_cache.Store(tmp_path / "u0")
    executor = kernel_cache.Store(tmp_path / "u64001", tmp_path / "u0")
    job = kernel_sources.job(kernel_sources.PYTHON, 89, root)
    finished: list[tuple[str, str]] = []
    gate = worker.claim(job.key)  # a build ahead of it in the machine's order
    assert gate is not None
    os.close(gate)
    thread = kernel_compile.in_order(
        worker,
        [job],
        dict(os.environ),
        lambda done, state: finished.append((done.key.kernel, state.state)),
    )
    thread.join()
    assert finished == [("kernel-python", "ready")]
    assert kernel_compile.submit(executor, job).state == "ready"
    assert (
        executor.entry(job.key)
        == worker.entry(job.key)
        == tmp_path / "u0" / job.key.kernel / job.key.digest
    )


def test_an_executor_waits_on_the_machines_claim_and_builds_nothing(tmp_path: Path) -> None:
    root = _registry(tmp_path / "sources")
    worker = kernel_cache.Store(tmp_path / "u0")
    executor = kernel_cache.Store(tmp_path / "u64001", tmp_path / "u0")
    job = kernel_sources.job(kernel_sources.PYTHON, 89, root)
    held = worker.claim(job.key)  # the machine's queue holds it, its turn not yet come
    assert held is not None
    try:
        worker.set_state(job.key, {"pid": os.getpid(), "progress": 0.0, "phase": "queued"})
        started = dict(kernel_compile._STARTED)
        state = kernel_compile.submit(executor, job)
        assert state.state == "compiling" and state.detail == "queued", state
        assert kernel_compile._STARTED == started  # no builder of its own
    finally:
        os.close(held)
    assert kernel_compile.status(executor, job).state == "pending"  # the machine's claim ended


def test_a_wide_build_is_as_wide_as_the_cpus_and_memory_this_process_may_use() -> None:
    cpus = len(os.sched_getaffinity(0))
    assert 1 <= kernel_compile.width(1) <= cpus
    assert kernel_compile.width(1 << 60) == 1  # no memory for even one more: still builds
