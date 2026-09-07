"""Fresh-process job device proofs; CUDA arms require an actual allocated GPU."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

import pytest

PROBE = Path(__file__).parent / "testdata" / "gpu_job_device.py"
REAL_GPU = os.environ.get("COZY_TEST_GPU_JOB_DEVICE") == "1"


def probe(mode: str, root: Path) -> dict[str, Any]:
    run = subprocess.run(
        [sys.executable, str(PROBE), mode, str(root)], text=True, capture_output=True, timeout=90
    )
    assert run.returncode == 0, run.stdout + run.stderr
    return cast(dict[str, Any], json.loads(run.stdout))


def test_cpu_parent_never_imports_torch_or_opens_cuda(tmp_path: Path) -> None:
    result = probe("cpu", tmp_path)
    assert result["result"] == {"device": "cpu", "values": []}
    assert not result["torch_imported"] and not result["device_initialized"]
    assert result["quiescent"] and result["gpu_count"] == 0


@pytest.mark.skipif(not REAL_GPU, reason="set COZY_TEST_GPU_JOB_DEVICE=1 on a real CUDA worker")
def test_gpu_job_initializes_before_handler_and_proves_completion(tmp_path: Path) -> None:
    result = probe("gpu", tmp_path)
    assert result["result"] == {"device": "cuda:0", "values": [14.0] * 512}
    assert result["torch_imported"] and result["device_initialized"]
    assert result["quiescent"] and result["gpu_count"] == 1
    assert result["peak_vram_bytes"] >= 4096


@pytest.mark.skipif(not REAL_GPU, reason="set COZY_TEST_GPU_JOB_DEVICE=1 on a real CUDA worker")
def test_gpu_job_refuses_cuda_initialized_outside_runtime(tmp_path: Path) -> None:
    result = probe("early", tmp_path)
    assert result["code"] == "cuda_initialized_before_runtime"
    assert result["executions"] == 0 and not result["result_written"]
