"""Runtime's one Ulysses wrapper, through real gloo ranks.

Every rank resolves SDPA through preparation's own path, runs the registered backend at
degree N on identical global Q/K/V (unequal shards) and compares with degree 1. A planted
rank-local wrapper, the silent-corruption shape, must be caught.
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
pytest.importorskip("diffusers")

from cozy_runtime.internal.parallel import links  # noqa: E402

ENV = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "CUDA_VISIBLE_DEVICES": ""}
FIXTURE = Path(__file__).parent / "testdata" / "ulysses_equivalence.py"


def _ranks(root: Path, degree: int, *extra: str) -> list[dict[str, Any]]:
    argv = [sys.executable, str(FIXTURE), str(root), "--degree", str(degree), *extra]
    subprocess.run(argv, env=ENV, check=True)
    return [json.loads((root / f"rank-{rank}.json").read_text()) for rank in range(degree)]


@pytest.mark.parametrize("degree", [2, 4, 7, 8])
def test_the_wrapper_reproduces_one_process_at_degree(tmp_path: Path, degree: int) -> None:
    # 1031 rows split unequally at every degree (258/258/258/257 at 4), H3's 56 heads.
    for row in _ranks(tmp_path, degree):
        assert row["byte_equal"] and row["rel_l2_to_degree1"] == 0.0, row
        assert any(impl in str(row["impl"]) for impl in ("flash", "efficient", "math")), row


def test_a_rank_local_wrapper_is_caught(tmp_path: Path) -> None:
    for row in _ranks(tmp_path, 2, "--rows", "67", "--heads", "4", "--plant"):
        assert not row["byte_equal"] and float(row["rel_l2_to_degree1"]) > 0.1, row


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2, reason="two CUDA GPUs"
)
def test_the_copy_engines_move_the_bytes_nccl_moves(tmp_path: Path) -> None:
    # H3's 56 heads over every GPU of the host, unequal shards, bf16: the copy-engine
    # exchange returns exactly NCCL's attention output on every GPU, and none of its peer
    # copies reaches Python dispatch. Only where this host's measured links allow it.
    degree = torch.cuda.device_count()
    devices = ",".join(str(index) for index in range(degree))
    measured = links.measured(None, devices, degree)
    if not measured.copy_engines:
        pytest.skip(f"this host's links keep NCCL: {measured.why()}")
    env = {
        **{key: value for key, value in ENV.items() if key != "CUDA_VISIBLE_DEVICES"},
        **measured.nccl(),
    }
    argv = [sys.executable, str(FIXTURE), str(tmp_path), "--degree", str(degree), "--cuda"]
    subprocess.run([*argv, "--rows", str(41_983)], env=env, check=True)
    for rank in range(degree):
        row = json.loads((tmp_path / f"rank-{rank}.json").read_text())
        assert row["copy_engine_equal_nccl"], row
        assert row["calibration"]["copy_engine_exact"] == 1.0, row
        assert row["dispatched_peer_writes"] == 0, row
