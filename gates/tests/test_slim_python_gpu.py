"""Stdlib protocol checks for child proof output, without importing Torch or Runtime."""

import ast
import json
import re
import unittest
from pathlib import Path


source = Path(__file__).parents[1] / "slim-python-gpu.py"
tree = ast.parse(source.read_text())
nodes = [
    node
    for node in tree.body
    if (
        isinstance(node, ast.FunctionDef)
        and node.name in {"physical_uuid", "verify_failed_request"}
    )
    or (
        isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "FAILURE_MARKER"
            for target in node.targets
        )
    )
]
namespace = {"json": json, "re": re}
exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), namespace)
verify = namespace["verify_failed_request"]
marker = namespace["FAILURE_MARKER"]
uuid = "GPU-78975d79-cc7d-6916-1a7c-338bf5a6167c"


def report(kinds=("returned", "escaping", "success"), gpu=uuid):
    return {
        "cases": [{"kind": kind, "before": 1, "peak": 2, "after": 1} for kind in kinds],
        "gpu": gpu,
    }


class FailedRequestOutputTests(unittest.TestCase):
    def test_accepts_three_cases_and_normalized_actual_uuid(self):
        verify(
            marker + "\n" + json.dumps(report()), "SLIM_DEVICE_UUID " + uuid[4:].replace("-", "")
        )

    def test_rejects_missing_or_repeated_completion(self):
        for output in (json.dumps(report()), marker + "\n" + marker + "\n" + json.dumps(report())):
            with self.subTest(output=output), self.assertRaises(RuntimeError):
                verify(output, "SLIM_DEVICE_UUID " + uuid)

    def test_rejects_missing_duplicate_or_wrong_cases(self):
        for kinds in (
            ("returned", "escaping"),
            ("returned", "returned", "success"),
            ("escaping", "returned", "success"),
        ):
            with self.subTest(kinds=kinds), self.assertRaises(RuntimeError):
                verify(marker + "\n" + json.dumps(report(kinds)), "SLIM_DEVICE_UUID " + uuid)

    def test_rejects_ambiguous_or_embedded_json(self):
        value = json.dumps(report())
        for output in (marker + "\n" + value + "\n" + value, marker + "\nopaque log " + value):
            with self.subTest(output=output), self.assertRaises(RuntimeError):
                verify(output, "SLIM_DEVICE_UUID " + uuid)

    def test_rejects_missing_malformed_or_different_gpu(self):
        for device, gpu in (
            ("", uuid),
            ("SLIM_DEVICE_UUID " + uuid, ""),
            ("SLIM_DEVICE_UUID " + uuid, "not-a-uuid"),
            ("SLIM_DEVICE_UUID " + uuid, "0" * 32),
        ):
            with self.subTest(device=device, gpu=gpu), self.assertRaises(RuntimeError):
                verify(marker + "\n" + json.dumps(report(gpu=gpu)), device)


if __name__ == "__main__":
    unittest.main()
