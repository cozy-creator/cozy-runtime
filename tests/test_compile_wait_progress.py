"""A call waiting on a kernel build moves as the builder's word moves (cl-648).

One long entry (a Triton GEMM) changes no share done; the builder's own CPU, told in its word,
moves the call waiting on it, which a sandboxed executor cannot read from the builder's
processes: run 15052 was ended as wedged 40 s into one. A builder that computes and says
nothing moves it once."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import threading
import zipfile
from pathlib import Path

import pytest

from cozy_runtime.author import _work as work
from cozy_runtime.internal import kernel_cache, kernel_compile, kernel_sources, liveness

SPIN = "import time\nend = time.monotonic() + 4\nwhile time.monotonic() < end: pass\n"


def _registry(root: Path) -> Path:
    """A one-wheel kernel source registry: the build the waiter makes once the lock frees."""
    root.mkdir(parents=True, exist_ok=True)
    wheel = root / "tiny_kernel-1.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("tiny_kernel/__init__.py", "VALUE = 42\n")
        archive.writestr("tiny_kernel-1.0.dist-info/METADATA", "Name: tiny-kernel\nVersion: 1.0\n")
    digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
    rows = {"tiny-kernel": {"version": "1.0", "file": wheel.name, "sha256": digest}}
    (root / "index.json").write_text(json.dumps({"schema": 1, "sources": rows}))
    kernel_sources.registry.cache_clear()
    return root


def test_a_wait_moves_as_its_builders_word_moves_through_one_long_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scope = tmp_path / "boot"  # this boot's failures
    scope.mkdir()
    monkeypatch.setenv("TMPDIR", str(scope))
    monkeypatch.setattr("tempfile.tempdir", None)
    monkeypatch.setattr(liveness, "SAMPLE_SECONDS", 0.2)
    monkeypatch.setattr(kernel_compile, "POLL_SECONDS", 0.05)
    job = kernel_sources.job(kernel_sources.PYTHON, 89, _registry(tmp_path / "sources"))
    for name, telling in (("telling", True), ("silent", False)):
        store = kernel_cache.Store(tmp_path / name)
        word = (
            "from pathlib import Path\nfrom cozy_runtime.internal import kernel_cache, kernel_compile\n"
            f"store = kernel_cache.Store(Path({str(store.own)!r}))\n"
            f"key = kernel_cache.Key.of({job.key.kernel!r}, {json.loads(job.key.inputs)!r})\n"
            "word = kernel_compile.Word(store, key, 0)\nword.start()\n" + SPIN + "word.stop()\n"
        )
        lock = store.claim(job.key)  # a live build that is not this process's
        assert lock is not None
        builder = subprocess.Popen([sys.executable, "-c", word if telling else SPIN])
        store.set_state(job.key, {"pid": builder.pid, "progress": 0.25, "phase": "compiling"})
        before, held = work.done, []

        def release(lock: int = lock, before: int = before, held: list[int] = held) -> None:
            held.append(work.done - before)
            os.close(lock)

        threading.Timer(4.5, release).start()
        assert kernel_compile.wait(store, job).state == "ready"
        assert builder.wait() == 0
        assert held[0] >= 3 if telling else held[0] <= 1, (name, held)
