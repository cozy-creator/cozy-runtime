"""A vanished birth is harmless; unreadable process identity remains a refusal."""

from __future__ import annotations

import errno
import os
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

from cozy_runtime.internal import proctree


def test_proc_birth_can_exit_after_real_directory_snapshot(monkeypatch: pytest.MonkeyPatch) -> None:
    child = subprocess.Popen(
        [sys.executable, "-c", "import sys; sys.stdin.read()"], stdin=subprocess.PIPE
    )
    try:
        entries = list(Path("/proc").iterdir())
        assert any(path.name == str(child.pid) for path in entries)
        assert child.stdin is not None
        child.stdin.close()
        child.wait(timeout=5)
        assert not Path(f"/proc/{child.pid}").exists()
        original = Path.iterdir

        def snapshot(path: Path) -> Iterator[Path]:
            # Every entry is the actual kernel snapshot taken while the child
            # lived; only the interleaving between enumeration and read is fixed.
            return iter(entries) if path == Path("/proc") else original(path)

        monkeypatch.setattr(Path, "iterdir", snapshot)
        assert proctree._full_proc_visibility(os.geteuid(), require_environment=False)
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=5)


@pytest.mark.parametrize("failure", [errno.EACCES, errno.EPERM, errno.EIO])
def test_wrapped_proc_read_errors_are_not_treated_as_vanished_births(
    monkeypatch: pytest.MonkeyPatch, failure: int
) -> None:
    original = proctree._linux_stat

    def unreadable(pid: int) -> list[str]:
        if pid == os.getpid():
            try:
                raise OSError(failure, "controlled unreadable identity")
            except OSError as exc:
                raise ProcessLookupError(pid) from exc
        return original(pid)

    monkeypatch.setattr(proctree, "_linux_stat", unreadable)
    assert not proctree._full_proc_visibility(os.geteuid(), require_environment=False)
