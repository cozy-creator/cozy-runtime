"""A spread call's result crosses between real gloo ranks under one contract (`wire.p2p_head`)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from cozy_runtime.internal.parallel import wire  # noqa: E402

FIXTURE = Path(__file__).parent / "testdata" / "spread_p2p.py"


def test_any_tensor_crosses_and_a_tuple_answers_nothing(tmp_path: Path) -> None:
    env = {**os.environ, "OMP_NUM_THREADS": "1", "CUDA_VISIBLE_DEVICES": ""}
    subprocess.run([sys.executable, str(FIXTURE), str(tmp_path)], env=env, check=True)
    report = json.loads((tmp_path / "rank-0.json").read_text())
    assert report == {"equal": [True, True], "tuple": None}


@pytest.mark.parametrize("value", [(torch.zeros(1),), 3.0, torch.zeros([1] * 9)])
def test_rank_zero_holds_its_own_result_to_the_same_contract(value: object) -> None:
    with pytest.raises(wire.UncrossableArgument, match="one tensor"):
        wire.p2p_head(value)
