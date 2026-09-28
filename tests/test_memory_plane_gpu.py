"""The executor's memory plane on a real card (`testdata/memory_proof.py`).

A diffusers UNet and VAE filled from native TensorFS, served under an allocator cap below
what they needed resident: the UNet pages its blocks, the decode runs out of memory and
recovers by paging then tiling, and the denoiser's output is bit-identical to the resident
run. Below the floor the call refuses typed and the plane stays usable.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


def test_a_capped_process_pages_tiles_and_recovers_in_place(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("diffusers")
    if not torch.cuda.is_available():
        pytest.skip("the memory plane's recovery is proved on a real card")
    child = subprocess.run(
        [sys.executable, str(Path(__file__).parent / "testdata/memory_proof.py"), str(tmp_path)],
        capture_output=True,
        text=True,
        check=True,
        # The allocator setting every executor is sealed with (`WorkerOptions.alloc_conf`).
        env={**os.environ, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
    )
    result = json.loads(child.stdout.splitlines()[-1])
    assert result["phase"] == "passed", child.stdout + child.stderr
    recoveries = result["recoveries"]
    assert any(line.startswith("unet: pages its blocks") for line in recoveries), recoveries
    assert any(line.startswith("vae: out of memory") and "-> tile" in line for line in recoveries)
    assert result["unet_steps"] == 3, "a forward's counters advance once per call"
    assert result["noise_error"] == 0.0
    assert result["poisoned"] == ""
    assert result["floor"].startswith(("device_shortfall", "unet: out of memory")), result["floor"]
    assert result["again_error"] == 0.0
