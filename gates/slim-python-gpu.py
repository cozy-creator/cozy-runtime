# /// script
# requires-python = ">=3.12"
# dependencies = ["cozy-runtime==0.27.4"]
# ///
"""Run the supplied TensorFS wheel's Torch regressions on an explicitly rented GPU."""

import os
import re
import signal
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Annotated

from cozy_runtime.author import AssetBound, FileAsset, Outputs, Tree


def main(
    *,
    wheel: Annotated[FileAsset, AssetBound(max_bytes=67108864)],
    wheel_name: str,
    tests: Annotated[FileAsset, AssetBound(max_bytes=131072)],
    conftest: Annotated[FileAsset, AssetBound(max_bytes=131072)],
    out: Outputs,
) -> Tree:
    if (
        Path(wheel_name).name != wheel_name
        or not wheel_name.startswith("tensorfs-")
        or not wheel_name.endswith(".whl")
    ):
        raise ValueError("wheel_name must be the TensorFS wheel's original filename")
    root = Path(tempfile.mkdtemp(prefix="slim-python-gpu-"))
    artifact = root / wheel_name
    artifact.write_bytes(wheel.read_bytes())
    (root / "test_slim_varena_gpu.py").write_bytes(tests.read_bytes())
    (root / "conftest.py").write_bytes(conftest.read_bytes())
    logs = root / "logs"
    logs.mkdir()
    environment_vars = os.environ.copy()
    # Keep the assigned CUDA device; do not inherit an outer Python source overlay.
    for key in ("PYTHONPATH", "PYTHONHOME"):
        environment_vars.pop(key, None)

    def run(name: str, command: list[str]) -> str:
        process = subprocess.Popen(
            command,
            cwd=root,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
            env=environment_vars,
        )
        try:
            text, _ = process.communicate(timeout=300)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            text, _ = process.communicate()
            print(text, flush=True)
            raise RuntimeError(f"{name} exceeded the test-harness bound") from None
        (logs / f"{name}.log").write_text(text)
        print(text, flush=True)
        if process.returncode:
            raise RuntimeError(f"{name} failed with exit {process.returncode}")
        return text

    environment = root / "venv"
    run(
        "environment",
        ["uv", "venv", "--python", sys.executable, "--system-site-packages", str(environment)],
    )
    python = str(environment / "bin/python")
    run("install", ["uv", "pip", "install", "--python", python, str(artifact), "pytest"])
    run(
        "device",
        [
            python,
            "-c",
            "import sys, tensorfs, torch; "
            "assert tensorfs.__file__.startswith(sys.prefix); "
            "assert torch.cuda.device_count() == 1; "
            "print(torch.cuda.get_device_properties(0))",
        ],
    )
    result = run(
        "tests",
        [
            python,
            "-m",
            "pytest",
            "--real-gpu",
            "-q",
            "-p",
            "no:cacheprovider",
            "test_slim_varena_gpu.py",
        ],
    )
    counts = {
        kind: int(count)
        for count, kind in re.findall(r"(\d+) (passed|failed|skipped|xfailed|xpassed)", result)
    }
    if counts != {"passed": 2}:
        raise RuntimeError(f"expected two executed CUDA tests, got {counts}")
    return out.save_tree(logs)
