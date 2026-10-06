"""Current device visibility config and accelerator no-driver behavior."""

from __future__ import annotations
import subprocess
import sys
import pytest
from cozy_runtime.internal.config import parse_visible_devices, sees_no_gpu


@pytest.mark.parametrize(
    "value,want",
    [
        (None, None),
        ("1", ("1",)),
        ("GPU-a", ("GPU-a",)),
        ("1,GPU-a", ("1", "GPU-a")),
        ("7", ("7",)),
        ("", ()),
        (" , ", ()),
    ],
)
def test_selected_device_names_are_parsed_without_boot_inventory(
    value: str | None, want: tuple[str, ...] | None
) -> None:
    assert parse_visible_devices(value) == want


def test_hidden_devices_do_not_ask_the_accelerator_driver() -> None:
    script = """
from cozy_runtime.internal import accel
from cozy_runtime.internal.config import hide_gpus, sees_no_gpu
hide_gpus()
assert sees_no_gpu()
try:
    accel._smi('--query-gpu=index')
except FileNotFoundError as exc:
    assert 'this process sees no GPU' in str(exc)
else:
    raise AssertionError('hidden process asked the GPU driver')
"""
    done = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert done.returncode == 0, done.stdout + done.stderr
    assert sees_no_gpu({"CUDA_VISIBLE_DEVICES": ""})
    assert not sees_no_gpu({})
