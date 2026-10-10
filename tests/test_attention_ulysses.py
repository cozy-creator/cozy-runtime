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


def _peer_gpus() -> int:
    count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    pairs = [(a, b) for a in range(count) for b in range(count) if a != b]
    return count if all(torch.cuda.can_device_access_peer(a, b) for a, b in pairs) else 0


@pytest.mark.skipif(_peer_gpus() < 2, reason="two or more CUDA GPUs with peer access")
def test_the_copy_engines_move_the_bytes_nccl_moves(tmp_path: Path) -> None:
    # H3's 56 heads over every GPU of the host, unequal shards, bf16: the copy-engine
    # exchange returns exactly NCCL's attention output on every GPU.
    degree = _peer_gpus()
    env = {key: value for key, value in ENV.items() if key != "CUDA_VISIBLE_DEVICES"}
    argv = [sys.executable, str(FIXTURE), str(tmp_path), "--degree", str(degree), "--cuda"]
    subprocess.run([*argv, "--rows", str(41_983)], env=env, check=True)
    for rank in range(degree):
        row = json.loads((tmp_path / f"rank-{rank}.json").read_text())
        assert row["copy_engine_equal_nccl"], row
        assert row["calibration"]["copy_engine_exact"] == 1.0, row
