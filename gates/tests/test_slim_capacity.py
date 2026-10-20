"""Source-format parser controls, not a claim of measured GPU cap enforcement."""

import copy
import json
import runpy
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
API = runpy.run_path(str(ROOT / "slim-capacity.py"))
SOURCE = json.loads((ROOT / "tests/fixtures/slim-capacity-source-format.json").read_text())


class SlimCapacity(unittest.TestCase):
    def measured(self, rows, gib, count=1):
        return API["measured_caps"]("\n".join(json.dumps(row) for row in rows), gib << 30, count)

    def test_source_tracing_uses_latest_pool_after_six_to_twelve_update(self):
        rows = SOURCE["events"]
        self.assertEqual(self.measured(rows[:1], 6)[0]["configured_limit_bytes"], 6 << 30)
        self.assertEqual(self.measured(rows[:2], 12)[0]["generation"], "b" * 32)
        with self.assertRaises(ValueError):
            self.measured(rows[:2], 6)

    def test_requested_twenty_four_can_be_clamped_to_smaller_actual_physical_capacity(self):
        (row,) = self.measured(SOURCE["events"], 24)
        self.assertEqual(row["effective_limit_bytes"], 25_000_000_000)
        self.assertLess(row["effective_limit_bytes"], 24 << 30)

    def test_every_assigned_gpu_needs_current_generation_and_measured_capacity(self):
        one = copy.deepcopy(SOURCE["events"][1])
        two = copy.deepcopy(one)
        two["fields"]["gpu"] = "GPU-22222222-2222-2222-2222-222222222222"
        self.assertEqual(len(self.measured([one, two], 12, 2)), 2)
        with self.assertRaisesRegex(ValueError, "covers"):
            self.measured([one], 12, 2)
        two["fields"]["generation"] = "a" * 32
        with self.assertRaisesRegex(ValueError, "same pool generation"):
            self.measured([one, two], 12, 2)
        for field in (
            "physical_bytes",
            "configured_limit_bytes",
            "effective_limit_bytes",
            "generation",
        ):
            malformed = copy.deepcopy(one)
            del malformed["fields"][field]
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.measured([one, malformed], 12)

    def test_public_log_envelopes_keep_actual_tracing_fields(self):
        row = SOURCE["events"][1]
        envelope = json.dumps({"lines": [{"line": json.dumps(row)}]})
        self.assertEqual(API["measured_caps"](envelope, 12 << 30, 1), [row["fields"]])
        self.assertEqual(
            API["measured_caps"]("prefix " + json.dumps(row), 12 << 30, 1), [row["fields"]]
        )
        for text in (
            "",
            '{"event":"ledger","capacity":12884901888}',
            '{"gpu memory limit":12}',
            '{"gpus":2}',
        ):
            with self.subTest(text=text), self.assertRaises(ValueError):
                API["measured_caps"](text, 12 << 30, 1)


if __name__ == "__main__":
    unittest.main()
