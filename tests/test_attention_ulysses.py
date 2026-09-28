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

pytest.importorskip("torch")
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
