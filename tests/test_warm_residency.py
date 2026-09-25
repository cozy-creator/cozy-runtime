"""A staged warm must not permanently force a roomy executor to refill every request."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest


def test_native_staged_warm_then_five_resident_requests(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("diffusers")
    with tempfile.TemporaryDirectory(prefix="wr-", dir=tmp_path.parent) as directory:
        result = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).parent / "testdata/warm_restore.py"),
                str(Path(directory) / "proof"),
            ],
            env={
                **os.environ,
                "CUDA_VISIBLE_DEVICES": "",
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
            },
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        facts = json.loads(result.stdout.strip().splitlines()[-1])
        assert facts["requests"] == [{"stages": 0, "evictions": 0}] * 5
        assert len(facts["pids"]) == 1
