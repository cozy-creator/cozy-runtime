"""Proof harness schema/source checks only; no claim of executed model/GPU recovery."""

import ast
import json
import runpy
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GENERATOR = runpy.run_path(str(ROOT / "make-slim-lifecycle.py"))
LIFECYCLE = runpy.run_path(str(ROOT / "slim-lifecycle.py"))

SOURCE = """from __future__ import annotations
class Txt2ImgInput:
    prompt: str
    seed: int | None = None
@app.entrypoint(defaults={"model": "sdxl@1.0.0"}, memoize=Memoize(requires=["seed"]))
def generate(ctx, payload, model, out, tel):
    device = model.device
    for index in range(20):
        noise = model.denoise(index)
        on_step(index)
    return out.save(noise)
"""


class LifecycleHarness(unittest.TestCase):
    def test_models_use_public_inline_cli_overrides(self):
        self.assertEqual(
            LIFECYCLE["model_overrides"](["model=repo/name@1", "model.vae=repo/vae@2"]),
            ["model.model=repo/name@1", "model.vae=repo/vae@2"],
        )
        for value in ("", "slot", "=repo/name@1", "slot=", "model.=repo/name@1"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                LIFECYCLE["model_overrides"]([value])

    def test_generator_preserves_computation_and_changes_only_the_proof_seam(self):
        before = ast.parse(SOURCE)
        after = ast.parse(GENERATOR["patch"](SOURCE))
        after.body = [
            node
            for node in after.body
            if not isinstance(node, ast.ImportFrom) or node.module != "_slim_lifecycle"
        ]
        payload = next(node for node in after.body if isinstance(node, ast.ClassDef))
        self.assertEqual(
            [node.target.id for node in payload.body if isinstance(node, ast.AnnAssign)][-2:],
            ["lifecycle_kill", "lifecycle_case"],
        )
        payload.body = payload.body[:-2]
        old = next(node for node in before.body if isinstance(node, ast.FunctionDef))
        generated = next(node for node in after.body if isinstance(node, ast.FunctionDef))
        memo = next(
            keyword for keyword in generated.decorator_list[0].keywords if keyword.arg == "memoize"
        )
        self.assertIs(memo.value.value, False)
        memo.value = next(
            keyword.value for keyword in old.decorator_list[0].keywords if keyword.arg == "memoize"
        )
        loop = next(node for node in generated.body if isinstance(node, ast.For))
        self.assertEqual(loop.body[-1].value.func.id, "_slim_lifecycle_step")
        loop.body.pop()
        self.assertEqual(
            ast.dump(before, include_attributes=False), ast.dump(after, include_attributes=False)
        )

    def test_ambiguous_step_hook_is_refused(self):
        with self.assertRaises(ValueError):
            GENERATOR["patch"](
                SOURCE.replace(
                    "        on_step(index)", "        on_step(index)\n        on_step(index)"
                )
            )

    def test_marker_survives_public_json_log_escaping(self):
        event = {
            "case": "current",
            "pid": 123,
            "uuid": "GPU-abc",
            "cuda_completed": True,
            "kill": True,
        }
        line = "SLIM_LIFECYCLE " + json.dumps(event)
        encoded = json.dumps({"message": line})
        self.assertEqual(LIFECYCLE["markers"]([encoded]), [event])

    def test_memoized_measurements_cannot_claim_actual_execution(self):
        call = {
            "memoized": True,
            "measurements": {
                "execution": {
                    "gpus": [{"gpu": 0, "pid": 123, "uuid": "GPU-abc", "start_us": 1, "end_us": 2}]
                }
            },
        }
        with self.assertRaisesRegex(AssertionError, "memoized"):
            LIFECYCLE["gpu_facts"]({"calls": [call]})


if __name__ == "__main__":
    unittest.main()
