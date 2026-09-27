"""ONE long-lived executor process holds several constructions (proto-061 D1).

Every arm runs the executor program as a real subprocess over the real seam; a group arm
adds real follower processes over gloo. The CPU arms adapt only the accelerator accounting
(host RAM) and the staging ring, exactly as `testdata/native_parallel_models.py` does; the
CUDA arm adapts nothing.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("tensorfs")

FIXTURE = Path(__file__).parent / "testdata" / "multi_construction.py"


def _run(tmp_path: Path, *args: str) -> dict[str, Any]:
    env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    result = subprocess.run(
        [sys.executable, str(FIXTURE), "--root", str(tmp_path), *args],
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert result.returncode == 0, result.stdout
    return dict(json.loads(result.stdout.strip().splitlines()[-1]))


@pytest.mark.parametrize("degree", [1, 2])
def test_one_process_loads_serves_unloads_and_reloads_two_constructions(
    tmp_path: Path, degree: int
) -> None:
    pytest.importorskip("diffusers")
    facts = _run(tmp_path, "--arm", "two-models", "--degree", str(degree))
    assert len(facts["pids"]) == degree


def test_activate_parks_the_least_recently_used_construction_under_the_ceiling(
    tmp_path: Path,
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("device residency is measured on a CUDA card")
    facts = _run(tmp_path, "--arm", "sized-blocks", "--size", "8192")
    assert len(facts["activate_ms"]) == 3


def test_loads_refused_before_holding_device_bytes_keep_the_process_serving(
    tmp_path: Path,
) -> None:
    facts = _run(tmp_path, "--arm", "refused-loads")
    assert facts["codes"] == ["model_fit_refused", "device_shortfall"]


def test_a_fill_completes_under_a_host_memory_cap_below_two_checkpoints(tmp_path: Path) -> None:
    probe = subprocess.run(
        ["systemd-run", "--user", "--scope", "--quiet", "-p", "MemoryMax=1G", "true"],
        capture_output=True,
        check=False,
    )
    if probe.returncode != 0:
        pytest.skip("no delegated systemd user scope with a memory controller")
    facts = _run(tmp_path, "--arm", "host-budget", "--size", "12288")
    assert facts["peak"] <= facts["limit"] < facts["held"] + 2 * facts["checkpoint"], facts
