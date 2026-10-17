"""Each kernel module the native wheel carries loads before torch, in a fresh interpreter: a
module exporting its static libstdc++ (GNU-unique symbols bind process-wide) crashed torch's
import after it, and every executor with it (0.28.0's blocker). kernels.yaml checks the built
modules the same way on every target; this holds an installed wheel to it."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from cozy_runtime import _kernels

#: The compiled modules installed beside `cozy_runtime._kernels` (none in the pure wheel).
HERE = Path(_kernels.__file__).parent
MODULES = sorted({p.name.split(".")[0] for p in HERE.iterdir() if p.suffix in (".so", ".pyd")})


@pytest.mark.skipif(bool(_kernels.ABSENT), reason="this Runtime wheel carries no kernels")
@pytest.mark.parametrize("module", MODULES or ["_gemm"])
def test_a_kernel_module_loads_before_torch(module: str) -> None:
    pytest.importorskip("torch")
    code = f"import cozy_runtime._kernels.{module}; import torch; print(torch.__version__)"
    run = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True)
    assert run.returncode == 0, f"{module} then torch: exit {run.returncode}\n{run.stderr[-800:]}"
