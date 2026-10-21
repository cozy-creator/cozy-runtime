"""Proof harness schema/source checks only; no claim of executed model/GPU recovery."""

import ast
import json
from pathlib import Path
import runpy
import unittest

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
        self.assertEqual(generated.body[-2].value.func.id, "_slim_lifecycle_step")
        self.assertTrue(generated.body[-2].value.keywords[0].value.value)
        generated.body.pop(-2)
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

    def test_public_identity_accepts_cli_envelopes_and_refuses_a_busy_machine(self):
        status = {
            "phase": "ready",
            "live_runs": 0,
            "worker_id": "worker",
            "started_at_ms": 123,
            "version": "0.11.0",
            "commit": "c" * 40,
            "runtime": "0.30.0",
            "tensorfs": "0.13.0",
        }
        self.assertEqual(
            LIFECYCLE["service_identity"]({"status": status}),
            LIFECYCLE["service_identity"]({"machine": {"status": status}}),
        )
        with self.assertRaisesRegex(AssertionError, "other runs"):
            LIFECYCLE["service_identity"]({"status": {**status, "live_runs": 1}})

    def test_missing_or_nonzero_orphan_evidence_cannot_pass_recovery(self):
        gpu = {"gpu": 0, "pid": 123, "uuid": "a" * 32, "start_us": 1, "end_us": 2}
        stats = {
            "paged_in": 7,
            "copies": [{"kind": "disk", "bytes": 3}],
            "leftover": 0,
            "leftover_by_gpu": [[0, 0]],
            "proofs": [{"complete": True, "initial": [1]}],
        }

        def record(trace):
            return {
                "measurements": {
                    "execution": {"gpus": [gpu]},
                    "memory": {"stats": stats, "trace": trace},
                }
            }

        with self.assertRaisesRegex(AssertionError, "placement trace"):
            LIFECYCLE["gpu_facts"](record([]))
        with self.assertRaisesRegex(AssertionError, "orphan bytes"):
            LIFECYCLE["gpu_facts"](record([{"kind": "end", "gpu": 0, "orphan_bytes": 1}]))
        facts = LIFECYCLE["gpu_facts"](record([{"kind": "end", "gpu": 0, "orphan_bytes": 0}]))
        self.assertEqual(
            (facts["initial_pages"], facts["disk_bytes"], facts["end_orphan_bytes"]), (1, 3, 0)
        )

    def test_reuse_and_refill_are_separately_named(self):
        facts = {"initial_pages": 1, "h2d_bytes": 2, "disk_bytes": 3}
        self.assertEqual(
            LIFECYCLE["recovery_path"](facts), "kept-readonly-pages-present-at-request-begin"
        )
        self.assertEqual(
            LIFECYCLE["recovery_path"]({**facts, "initial_pages": 0}), "verified-disk-refill"
        )
        self.assertEqual(
            LIFECYCLE["recovery_path"]({**facts, "initial_pages": 0, "disk_bytes": 0}),
            "diskless-h2d-refill",
        )

    def test_unreadable_oom_counters_are_not_zero_and_increases_fail(self):
        self.assertFalse(LIFECYCLE["oom_observation"]({}, {})["available"])
        first = {"cgroups": [{"path": "owned", "events": {"oom_kill": 4}}]}
        self.assertEqual(
            LIFECYCLE["oom_observation"](first, first)["oom_kill_deltas"], {"owned": 0}
        )
        after = {"cgroups": [{"path": "owned", "events": {"oom_kill": 5}}]}
        with self.assertRaisesRegex(AssertionError, "oom_kill changed"):
            LIFECYCLE["oom_observation"](first, after)


if __name__ == "__main__":
    unittest.main()
