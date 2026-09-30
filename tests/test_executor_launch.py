"""The long-lived launcher returns failures without losing its next executor."""

from __future__ import annotations

from pathlib import Path

import pytest

from conftest import image_python
from cozy_runtime.internal import accel
from cozy_runtime.internal.worker.child import ExecutorSupervision
from test_device_lanes import _config, _workspace
from test_end_to_end import NO_EXECUTOR


@pytest.mark.skipif(bool(NO_EXECUTOR), reason=NO_EXECUTOR or "")
def test_launch_failure_then_success_preserves_the_launch_thread() -> None:
    with _workspace() as root:
        config = _config(root / "home")
        missing = root / "missing-python"
        supervision = ExecutorSupervision(
            root=root / "worker",
            python=str(missing),
            base_env=config.child_base_env,
            cozy_home=config.cozy_home,
            device_process=lambda _: accel.ProcessMemory("unreadable"),
        )
        try:
            with pytest.raises(FileNotFoundError) as refused:
                supervision.spawn(imposed={"CUDA_VISIBLE_DEVICES": ""})
            assert Path(refused.value.filename) == missing
            assert supervision.current is None and not supervision._owner_path.exists()
            launcher = supervision._launcher
            assert launcher is not None and launcher.is_alive()
            supervision.python = str(image_python())
            executor = supervision.spawn(imposed={"CUDA_VISIBLE_DEVICES": ""})
            assert supervision._launcher is launcher
            with supervision.hold(executor) as current:
                assert current and executor.alive()
            evidence = supervision.retire_current(executor, "launch fixture complete")
            assert evidence is not None and evidence.members == (executor.pid,)
            assert not executor.alive()
            with supervision.hold(executor) as current:
                assert not current
        finally:
            supervision.close()
        assert not launcher.is_alive()
