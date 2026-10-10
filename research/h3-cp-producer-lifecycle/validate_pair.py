"""Validate two retained ordinary-CLI reports; never submit or rent."""

import argparse
import hashlib
import json
from pathlib import Path

NAME = "kitchen-sol-producer-fp4-fixed-means-shared-qkv"


def validate(first, second, runtime):
    identities = []
    for report, start in ((first, 4), (second, 6)):
        assert report["passed"] is True and report["start_step"] == start
        assert report["tokens"] == 4225 and report["prefix"] == 257
        calls = report["calls"]
        assert len(calls) == 3
        for call, step in zip(calls, (start, start + 1, start + 1), strict=True):
            ranks = call["ranks"]
            assert [row["rank"] for row in ranks] == [0, 1]
            assert len({row["uuid"] for row in ranks}) == len({row["pid"] for row in ranks}) == 2
            assert ranks[0]["output_sha256"] == ranks[1]["output_sha256"]
            for rank, row in enumerate(ranks):
                assert row["step"] == step and row["backend"] == NAME
                assert row["runtime"] == runtime and row["tensorfs"] == "0.6.1"
                assert row["capability"] == [12, 0]
                assert row["stats_shapes"] == [[1, 128], [1, 128]]
                assert row["full_rope_shape"] == [4225, 96]
                assert row["local_rope_shape"] == [2113 if rank == 0 else 2112, 96]
                assert row["output_shape"] == [1, 4225, 128]
                assert row["python_prefix"]
        baseline = [(row["pid"], row["uuid"], row["python_prefix"]) for row in calls[0]["ranks"]]
        for call in calls:
            assert [
                (row["pid"], row["uuid"], row["python_prefix"]) for row in call["ranks"]
            ] == baseline
        identities.append(baseline)
        for rank in range(2):
            a, b, replay = [call["ranks"][rank] for call in calls]
            assert a["first"] and a["before"] == [] and a["input_stats"] is None
            assert not b["first"] and not replay["first"]
            assert b["input_stats"] == a["output_stats"] == replay["input_stats"]
            assert b["output_sha256"] == replay["output_sha256"]
            assert b["output_stats"] == replay["output_stats"]
    assert identities[0] == identities[1], (
        "A different executor generation cannot prove next-request cleanup"
    )
    return {
        "passed": True,
        "rank_identities": identities[0],
        "invokes": 2,
        "component_calls_per_invoke": 3,
        "scope": (
            "real CP producer state and hook lifecycle with tiny BF16 projections; "
            "not whole H3 determinism or FP8/LoRA performance"
        ),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("first", type=Path)
    parser.add_argument("second", type=Path)
    parser.add_argument("--runtime", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = validate(
        json.loads(args.first.read_text()), json.loads(args.second.read_text()), args.runtime
    )
    result["reports"] = {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (args.first, args.second)
    }
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
