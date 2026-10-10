import copy
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "validator", Path(__file__).resolve().parents[1] / "validate_pair.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def report(start):
    calls = []
    for index, step in enumerate((start, start + 1, start + 1)):
        calls.append(
            {
                "ranks": [
                    {
                        "rank": rank,
                        "pid": rank + 100,
                        "uuid": f"gpu-{rank}",
                        "python_prefix": "generation-a",
                        "step": step,
                        "backend": module.NAME,
                        "runtime": "fixture",
                        "tensorfs": "0.6.1",
                        "capability": [12, 0],
                        "stats_shapes": [[1, 128], [1, 128]],
                        "full_rope_shape": [4225, 96],
                        "local_rope_shape": [2113 if rank == 0 else 2112, 96],
                        "output_shape": [1, 4225, 128],
                        "first": index == 0,
                        "before": [] if index == 0 else [[step - 1]],
                        "input_stats": None if index == 0 else ["mean-a", "scale-a"],
                        "output_stats": ["mean-a", "scale-a"]
                        if index == 0
                        else ["mean-b", "scale-b"],
                        "output_sha256": "initial" if index == 0 else "repeat",
                    }
                    for rank in range(2)
                ]
            }
        )
    return {"passed": True, "start_step": start, "tokens": 4225, "prefix": 257, "calls": calls}


def test_two_invokes_pass_and_new_generation_is_rejected():
    first, second = report(4), report(6)
    assert module.validate(first, second, "fixture")["passed"]
    changed = copy.deepcopy(second)
    for call in changed["calls"]:
        for row in call["ranks"]:
            row["pid"] += 100
            row["python_prefix"] = "generation-b"
    with pytest.raises(AssertionError, match="different executor"):
        module.validate(first, changed, "fixture")


@pytest.mark.parametrize(
    "field,value",
    [
        ("before", [[5]]),
        ("input_stats", ["leaked", "stats"]),
        ("uuid", "gpu-1"),
        ("output_sha256", "different"),
    ],
)
def test_bad_state_or_rank_proof_is_rejected(field, value):
    first, second = report(4), report(6)
    second["calls"][0]["ranks"][0][field] = value
    with pytest.raises(AssertionError):
        module.validate(first, second, "fixture")
