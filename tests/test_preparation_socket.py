"""Preparation uses a fixed socket within the primary worker's accepted path budget."""

import socket
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from cozy_runtime.internal.worker.child import SUN_PATH_MAX, ExecutorGone, socket_path_refusal
from cozy_runtime.internal.config import Credentials, RuntimeConfig
from cozy_runtime.internal.worker.session import Worker, WorkerOptions


def worker(root: Path, home: Path) -> Worker:
    return cast(
        Worker,
        SimpleNamespace(
            root=root,
            options=WorkerOptions(root=root, python=sys.executable),
            config=RuntimeConfig(cozy_home=home, credentials=Credentials()),
            fence=SimpleNamespace(worker_boot_id="fixed-boot"),
            _slot_executor_uid=lambda _: -1,
            fail_machine=lambda *args: None,
            note=lambda *args: None,
        ),
    )


def test_preparation_socket_fits_every_boot_validated_worker_root() -> None:
    with tempfile.TemporaryDirectory(prefix="cps-") as temporary:
        home = Path(temporary)
        length = SUN_PATH_MAX - len(b"/executor.sock") - 1
        root = home / ("x" * (length - len(str(home).encode()) - 1))
        assert not socket_path_refusal(str(root / "executor.sock"))
        assert socket_path_refusal(str(root / "placements/prepare/executor.sock"))
        supervision = Worker._new_preparation_supervision(worker(root, home))
        try:
            assert supervision.root == root / "placements/prepare"
            assert supervision.socket_path == str(root / "p.sock")
            with socket.socket(socket.AF_UNIX) as listener:
                listener.bind(supervision.socket_path)
            Path(supervision.socket_path).unlink()
        finally:
            supervision.close()


def test_unbindable_preparation_refuses_before_creating_state(tmp_path: Path) -> None:
    root = tmp_path / ("x" * SUN_PATH_MAX)
    with pytest.raises(ExecutorGone, match="executor_socket_unbindable"):
        Worker._new_preparation_supervision(worker(root, tmp_path))
    assert not root.exists()
