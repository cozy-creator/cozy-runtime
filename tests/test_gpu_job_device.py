"""Fresh-process job device proofs; CUDA arms require an actual allocated GPU."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any, cast

import pytest

from cozy_runtime.internal import child_env, jit_cache

PROBE = Path(__file__).parent / "testdata" / "gpu_job_device.py"
REAL_GPU = os.environ.get("COZY_TEST_GPU_JOB_DEVICE") == "1"


def probe(mode: str, root: Path, *, script: Path = PROBE) -> dict[str, Any]:
    # The fresh child gets the same erase/impose boundary and installation-scoped
    # compiler cache as a worker. Cold upstream imports must not invent cache
    # variables after the child's real _capture_seal() snapshot.
    cache = jit_cache.scope(
        root.resolve(),
        "local-" + uuid.uuid4().hex,
        pod_scope=str(root.resolve()),
    )
    visible = "" if mode in {"cpu", "cpu-order"} else os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    env = {name: value for name, value in os.environ.items() if not child_env.erased(name)}
    env.update(child_env.project_child_env({"CUDA_VISIBLE_DEVICES": visible, **cache.environment}))
    run = subprocess.run(
        [sys.executable, str(script), mode, str(root)],
        env=env,
        text=True,
        capture_output=True,
        timeout=90,
    )
    assert run.returncode == 0, run.stdout + run.stderr
    return cast(dict[str, Any], json.loads(run.stdout))


def test_cpu_parent_never_imports_torch_or_opens_cuda(tmp_path: Path) -> None:
    result = probe("cpu", tmp_path)
    assert result["result"] == {"device": "cpu", "values": []}
    assert not result["torch_imported"] and not result["device_initialized"]
    assert result["quiescent"] and result["gpu_count"] == 0


def test_prepare_establishes_device_boundary_before_kernel_imports(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    result = probe("cpu-order", tmp_path, script=PROBE.with_name("gpu_prepare_imports.py"))
    assert result["runtime_device_boundary_before_kernel_import"] is True


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


@pytest.mark.skipif(not REAL_GPU, reason="requires an allocated CUDA worker with Diffusers/TorchAO")
@pytest.mark.parametrize("mode", ["preload", "prepare"])
def test_gpu_prepare_import_order(tmp_path: Path, mode: str) -> None:
    if importlib.util.find_spec("diffusers") is None:
        pytest.skip("Diffusers is not installed")
    # Availability is inspected without importing the optional integration in
    # this process. The fresh interpreter loads the actual libraries.
    probe(mode, tmp_path, script=PROBE.with_name("gpu_prepare_imports.py"))
