"""Actual/source-format parser controls; allocator enforcement is a separate GPU proof."""

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
        envelope = json.dumps({"log": "tensord", "machine": "fixture", "text": json.dumps(row)})
        self.assertEqual(API["measured_caps"](envelope, 12 << 30, 1), [row["fields"]])
        for text in (
            "",
            "prefix " + json.dumps(row),
            '{"event":"ledger","capacity":12884901888}',
            '{"gpu memory limit":12}',
            '{"gpus":2}',
        ):
            with self.subTest(text=text), self.assertRaises(ValueError):
                API["measured_caps"](text, 12 << 30, 1)

    def test_actual_plain_and_cli_enveloped_logs_select_the_live_six_gib_pool(self):
        # Exact capacity rows from next-models-0330/cap-6-tensord{,-json}.log.
        # These prove parsing of measured pool metadata, not the allocator/OOM bar.
        rows = []
        for name in ("slim-capacity-actual-0330.log", "slim-capacity-actual-0330-cli.jsonl"):
            text = (ROOT / "tests/fixtures" / name).read_text()
            measured = API["measured_caps"](text, 6 << 30, 1)
            self.assertEqual(measured[0]["gpu"], "GPU-f268c62c-03a8-4b7a-e718-5fdd28f92678")
            self.assertEqual(measured[0]["generation"], "8a491d92b02a4954867a64934f75736f")
            self.assertEqual(measured[0]["physical_bytes"], 25_655_508_992)
            self.assertEqual(measured[0]["effective_limit_bytes"], 6 << 30)
            rows.append(measured)
        self.assertEqual(rows[0], rows[1])

    def test_plain_capacity_requires_complete_fields_and_cannot_reuse_an_older_good_row(self):
        good = (ROOT / "tests/fixtures/slim-capacity-actual-0330.log").read_text().splitlines()[-1]
        for changed in (
            good.replace(" effective_limit_bytes=6442450944", ""),
            good + " physical_bytes=25655508992",
            good.replace("effective_limit_bytes=6442450944", "effective_limit_bytes=true"),
            good.replace("effective_limit_bytes=6442450944", "effective_limit_bytes=-1"),
        ):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                API["measured_caps"](good + "\n" + changed, 6 << 30, 1)

    def test_native_rows_are_anchored_and_embedded_executor_json_is_opaque(self):
        good = (ROOT / "tests/fixtures/slim-capacity-actual-0330.log").read_text().splitlines()[-1]
        fake = json.dumps(SOURCE["events"][1])
        captured = "2026-10-20T12:50:43.805217Z  INFO tensord::device_executor: " + fake
        for text in (
            "prefix " + good,
            good.replace(" INFO ", " WARN "),
            good.replace("tensord::gpu_service:", "tensord::device_executor:"),
            good.replace("2026-10-20T12:50:43.805217Z", "not-a-timestamp"),
            captured,
            json.dumps({"log": "tensord", "machine": "fixture", "text": captured}),
            json.dumps({"log": "executor", "machine": "fixture", "text": good}),
            json.dumps({"text": fake}),
        ):
            with self.subTest(text=text), self.assertRaisesRegex(ValueError, "covers 0 GPUs"):
                API["measured_caps"](text, 6 << 30, 1)

    def test_executor_message_text_cannot_replace_native_pool_evidence(self):
        row = SOURCE["events"][1]
        captured = {
            "level": "INFO",
            "target": "tensord::device_executor",
            "fields": {"message": json.dumps(row)},
        }
        with self.assertRaisesRegex(ValueError, "covers 0 GPUs"):
            API["measured_caps"](json.dumps(captured), 12 << 30, 1)


if __name__ == "__main__":
    unittest.main()
