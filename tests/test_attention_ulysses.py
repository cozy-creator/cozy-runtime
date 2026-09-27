"""Runtime's one Ulysses wrapper, through real gloo ranks and the real qualification verb.

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

import pytest

from cozy_runtime.cli.main import main as cli
from cozy_runtime.internal import attention_qualify

pytest.importorskip("torch")
pytest.importorskip("diffusers")

ENV = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "CUDA_VISIBLE_DEVICES": ""}


@pytest.mark.parametrize("degree", [2, 4, 7, 8])
def test_the_wrapper_reproduces_one_process_at_degree(degree: int) -> None:
    # 1031 rows split unequally at every degree (258/258/258/257 at 4), H3's 56 heads.
    report = attention_qualify.qualify(degree, [], "cpu", rows=1031, heads=56)
    assert report["ok"] and report["verdicts"] == {"sdpa": ["ready"]}, report["verdicts"]
    assert [rank["kernels"][0]["rank"] for rank in report["ranks"]] == list(range(degree))
    for rank in report["ranks"]:
        [row] = rank["kernels"]
        assert row["byte_equal"] and row["rel_l2_to_degree1"] == 0.0
        assert row["impl"] in {"flash", "efficient", "math"}
        assert rank["kernel_site"]["path"] == ""  # no image site on this host


def test_the_cli_verb_runs_the_same_check_and_refuses_unknown_kernels(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "report.json"
    argv = ["--json", "attention-qualify", "--degree", "2", "--device", "cpu"]
    assert cli([*argv, "--rows", "67", "--heads", "4", "--out", str(out)]) == 0
    assert json.loads(out.read_text())["verdicts"] == {"sdpa": ["ready"]}
    assert '"ok": true' in capsys.readouterr().out
    assert cli([*argv, "--kernels", "no-such-kernel"]) != 0


def test_a_rank_local_wrapper_is_caught(tmp_path: Path) -> None:
    fixture = Path(__file__).parent / "testdata" / "ulysses_rank_local.py"
    subprocess.run([sys.executable, str(fixture), str(tmp_path)], env=ENV, check=True)
    for rank in range(2):
        [row] = json.loads((tmp_path / f"rank-{rank}.json").read_text())["kernels"]
        assert row["status"] == "not_equivalent" and not row["byte_equal"], row
        assert row["rel_l2_to_degree1"] > 0.1
