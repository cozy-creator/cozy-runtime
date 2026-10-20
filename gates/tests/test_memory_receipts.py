import contextlib
import copy
import io
import json
import runpy
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
API = runpy.run_path(str(ROOT / "memory_receipts.py"))
RAW = json.loads((ROOT / "tests/fixtures/h3-warm-11880.json").read_text())
CONTROLS = (ROOT / "tests/fixtures/0274-controls.txt").read_text()


def positive():
    # Deliberately synthetic passing memory assertions, derived from a real completed GPU
    # receipt. The unmodified fixture retains its actual leftover and is tested as a failure.
    value = copy.deepcopy(RAW)
    measurements = value.get("calls", [value])[0]["measurements"]
    stats = measurements["memory"]["stats"]
    stats.update(leftover=0, paged_in=0, retries=0)
    stats["leftover_by_gpu"] = [[0, 0]]
    stats["steps"] = {"model/transformer_ref": [0, 0]}
    return value


class MemoryReceipts(unittest.TestCase):
    def test_slim_trace_protects_working_pages_and_orphan_order(self):
        def event(sequence, kind, **values):
            row = dict(sequence=sequence, request="r", step=0, page=1, bytes=1,
                       capacity=8, resident=1, orphan_bytes=0, kind=kind,
                       reason="demand", **{"class": "unused"})
            row.update(values)
            return row
        trace = [event(0, "begin"), event(1, "evict"), event(2, "end")]
        self.assertEqual(API["trace_errors"](trace), [])
        trace[1]["class"] = "working"
        self.assertIn("running-page-evicted", API["trace_errors"](trace))
        trace[1]["reason"] = "lower"
        self.assertEqual(API["trace_errors"](trace), [])
        trace[1].update(**{"class": "unused", "orphan_bytes": 1})
        self.assertIn("orphan-not-reclaimed-first", API["trace_errors"](trace))

    def test_slim_trace_rejects_repeated_or_unmeasured_speculation(self):
        begin = dict(sequence=0, request="r", step=0, page=1, bytes=1, capacity=8,
                     resident=1, orphan_bytes=0, kind="begin", reason="demand", **{"class": "unused"})
        fill = dict(begin, sequence=1, kind="fill", reason="speculation", working_set=4, speculative_budget=3)
        end = dict(begin, sequence=2, kind="end")
        self.assertEqual(API["trace_errors"]([begin, fill, end]), [])
        self.assertIn("repeated-speculative-fetch", API["trace_errors"]([begin, fill, dict(fill, sequence=2), dict(end, sequence=3)]))
        del fill["working_set"]
        self.assertIn("unmeasured-speculation", API["trace_errors"]([begin, fill, end]))
        self.assertEqual(API["trace_errors"]([]), ["missing-decision-trace"])

    def record(self, value, cell="24-sdxl4", cold=""):
        with tempfile.TemporaryDirectory() as directory:
            path, summary = Path(directory) / "run.json", Path(directory) / "summary.txt"
            path.write_text(json.dumps(value))
            summary.write_text(cold)
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                code = API["record"](path, cell, str(value["number"]), "completed landed", "", summary)
            return code, out.getvalue()

    def compare(self, controls, candidate):
        with tempfile.TemporaryDirectory() as directory:
            old, new = Path(directory) / "old.txt", Path(directory) / "new.txt"
            old.write_text(controls)
            new.write_text(candidate)
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                code = API["compare"](old, new, ["24"])
            return code, out.getvalue()

    def test_actual_current_receipt_leftover_cannot_pass(self):
        code, text = self.record(RAW)
        self.assertEqual(code, 1)
        self.assertIn("leftover=33554432", text)
        self.assertRegex(text, r"bar=[^ ]*leftover")

    def test_missing_required_gpu_fields_refuse(self):
        for field in ("leftover", "paged_in", "retries", "steps"):
            with self.subTest(field=field):
                value = positive()
                del value.get("calls", [value])[0]["measurements"]["memory"]["stats"][field]
                code, text = self.record(value)
                self.assertEqual(code, 1)
                self.assertIn(f"no-valid-{field}", text)
        value = positive()
        value.get("calls", [value])[0]["measurements"].pop("memory")
        code, text = self.record(value)
        self.assertEqual(code, 1)
        self.assertIn("no-mem-stats", text)

    def test_cpu_orchestrators_do_not_need_memory_but_gpu_calls_do(self):
        value = positive()
        gpu = value.pop("measurements") if "measurements" in value else value.pop("calls")[0]["measurements"]
        value["calls"] = [{"measurements": {"execution": {"gpus": [{"gpu": -1}]}}}, {"measurements": gpu}]
        self.assertEqual(self.record(value)[0], 0)
        value["calls"].pop()
        code, text = self.record(value)
        self.assertEqual(code, 1)
        self.assertIn("no-gpu-execution", text)

    def test_every_executed_gpu_requires_its_leftover_receipt(self):
        value = positive()
        m = value.get("calls", [value])[0]["measurements"]
        m["execution"]["gpus"].append({**m["execution"]["gpus"][0], "gpu": 1})
        code, text = self.record(value)
        self.assertEqual(code, 1)
        self.assertIn("leftover-gpu-coverage", text)
        m["memory"]["stats"]["leftover_by_gpu"].append([1, 0])
        self.assertEqual(self.record(value)[0], 0)
        m["memory"]["stats"]["leftover_by_gpu"][1][1] = 32
        self.assertIn("leftover-gpu-coverage", self.record(value)[1])
        m["memory"]["stats"]["leftover_by_gpu"] = [[0, 0], [1, 0], [2, 0]]
        self.assertIn("leftover-gpu-coverage", self.record(value)[1])

    def test_actual_degree_two_receipt_exposes_missing_gpu_one(self):
        value = json.loads((ROOT / "tests/fixtures/h3-degree2-11916.json").read_text())
        code, text = self.record(value, "2x8-h3")
        self.assertEqual(code, 1)
        self.assertIn("leftover-gpu-coverage", text)

    def test_missing_malformed_and_duplicate_gpu_totals_refuse(self):
        for by_gpu in (None, [], [0], [[False, 0]], [[0, False]], [[0, -1]], [[0, 0], [0, 0]], [[0, 1]]):
            with self.subTest(by_gpu=by_gpu):
                value = positive()
                value.get("calls", [value])[0]["measurements"]["memory"]["stats"]["leftover_by_gpu"] = by_gpu
                self.assertEqual(self.record(value)[0], 1)

    def test_warm_transformer_steps_cannot_default_to_zero(self):
        cold = "24-h3ref5 run=1 completed landed exec=10s poisoned=0 leftover=0 oom_retries=0 bar=ok\n"
        value = positive()
        code, text = self.record(value, "24-h3ref5-warm", cold)
        self.assertEqual(code, 0, text)
        value.get("calls", [value])[0]["measurements"]["memory"]["stats"]["steps"] = {}
        code, text = self.record(value, "24-h3ref5-warm", cold)
        self.assertEqual(code, 1)
        self.assertIn("no-dit-step-stats", text)

    def test_retries_use_current_stats_and_own_cold_pass(self):
        value = positive()
        value.get("calls", [value])[0]["measurements"]["memory"]["stats"]["retries"] = 3
        cold = "24-h3ref5 run=1 completed landed exec=10s poisoned=0 leftover=0 oom_retries=2 bar=ok\n"
        code, text = self.record(value, "24-h3ref5-warm", cold)
        self.assertEqual(code, 1)
        self.assertIn("oom_retries=3", text)
        self.assertIn("more-retries-than-cold", text)
        code, text = self.record(positive(), "24-h3ref5-warm")
        self.assertEqual(code, 1)
        self.assertIn("no-valid-cold", text)

    def test_failed_poisoned_11618_is_never_numeric_control(self):
        old = next(line for line in CONTROLS.splitlines() if "run=11618 " in line) + "\n"
        current = "24-h3ref-warm run=2 completed landed exec=1000s poisoned=0 leftover=0 bar=ok\n"
        code, text = self.compare(old, current)
        self.assertEqual(code, 0)
        self.assertIn("not compared: control failed", text)
        self.assertNotIn("WORSE", text)

    def test_completed_legacy_control_times_compare_without_new_varena_schema(self):
        old = next(line for line in CONTROLS.splitlines() if "run=11616 " in line) + "\n"
        current = "24-h3ref5-warm run=2 completed landed exec=126s poisoned=0 leftover=0 bar=ok\n"
        code, text = self.compare(old, current)
        self.assertEqual(code, 1)
        self.assertIn("WORSE", text)

    def test_missing_and_substitute_controls_are_explicit(self):
        current = "24-h3ref5-warm run=2 completed landed exec=126s poisoned=0 leftover=0 bar=ok\n"
        self.assertIn("missing control; absolute bars remain", self.compare("", current)[1])
        old = "24-h3ref5-warm run=1 completed landed exec=1s poisoned=0 leftover=0 substitute=32 bar=ok\n"
        self.assertIn("control substitute 32", self.compare(old, current)[1])
        for time in ("0s", "nans", "infs"):
            old = f"24-h3ref5-warm run=1 completed landed exec={time} poisoned=0 leftover=0 bar=ok\n"
            self.assertIn("time missing or invalid", self.compare(old, current)[1])
        invalid = current.replace("bar=ok", "bar=no-mem-stats")
        self.assertEqual(self.compare("", invalid)[0], 1)


if __name__ == "__main__":
    unittest.main()
