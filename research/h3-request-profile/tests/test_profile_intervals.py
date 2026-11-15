"""Union and measurement-window semantics for both diagnostic consumers."""

import subprocess
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from profile_intervals import interval_summary
from trace_summary import summarize


class IntervalTests(unittest.TestCase):
    def test_trace_script_can_be_loaded_by_path_without_sibling_on_sys_path(self):
        script = Path(__file__).resolve().parents[1] / "trace_summary.py"
        code = """
import importlib.util
import sys
spec = importlib.util.spec_from_file_location("isolated_trace", sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
assert module.summarize(iter([]))["device_activity"] == []
assert module.interval_summary([(0, 7), (4, 11)])["busy_union_us"] == 11
"""
        subprocess.run([sys.executable, "-I", "-c", code, str(script)], check=True)

    def test_union_patterns(self):
        for intervals, busy, total, span in (
            ([(0, 7), (4, 11)], 11, 14, 11),
            ([(0, 10), (2, 3)], 10, 11, 10),
            ([(0, 10), (0, 10)], 10, 20, 10),
            ([(0, 4), (4, 6)], 6, 6, 6),
            ([(8, 10), (0, 4)], 6, 6, 10),
            ([], 0, 0, 0),
        ):
            with self.subTest(intervals=intervals):
                row = interval_summary(intervals)
                self.assertEqual(row["busy_union_us"], busy)
                self.assertEqual(row["duration_sum_us"], total)
                self.assertEqual(row["span_us"], span)
                self.assertEqual(row["interior_gap_total_us"], span - busy)
                self.assertNotIn("unclassified_inactivity_us", row)

    def test_window_intersection_not_clipping_an_aggregate(self):
        row = interval_summary([(-5, 3), (2, 4), (9, 15), (12, 20), (5, 5)], window=(0, 10))
        self.assertEqual(row["event_count"], 3)
        self.assertEqual(row["duration_sum_us"], 6)
        self.assertEqual(row["busy_union_us"], 5)
        self.assertEqual(row["unclassified_inactivity_us"], 5)

    def test_tails_count_only_when_window_is_explicit(self):
        row = interval_summary([(2, 3), (6, 7)], window=(0, 10))
        self.assertEqual(row["span_us"], 5)
        self.assertEqual(row["interior_gap_total_us"], 3)
        self.assertEqual(row["unclassified_inactivity_us"], 8)
        self.assertEqual(interval_summary([], window=(0, 10))["unclassified_inactivity_us"], 10)
        self.assertEqual(interval_summary([(0, 20)], window=(5, 5))["busy_union_us"], 0)

    def test_invalid_bounds_are_errors(self):
        for interval in ((2, 1), (0, float("inf")), (float("nan"), 1)):
            with self.subTest(interval=interval):
                with self.assertRaises(ValueError):
                    interval_summary([interval])
                with self.assertRaises(ValueError):
                    interval_summary([], window=interval)

    def test_trace_summary_groups_streams_and_devices_in_a_window(self):
        def event(start, duration, device, stream):
            return {
                "ph": "X",
                "cat": "kernel",
                "name": "work",
                "ts": start,
                "dur": duration,
                "args": {"device": device, "stream": stream},
            }

        events = [event(0, 70, 0, 7), event(40, 70, 0, 8), event(10, 90, 1, 7)]
        row = summarize(iter(events), window=(20, 100))
        devices = row["device_activity"]
        self.assertEqual([d["busy_union_us"] for d in devices], [80, 80])
        self.assertEqual([d["duration_sum_us"] for d in devices], [110, 80])
        self.assertEqual([d["busy_union_us"] for d in row["stream_activity"]], [50, 60, 80])
        self.assertEqual([d["duration_sum_us"] for d in row["events_by_name"]], [110, 80])
        self.assertEqual(row["measurement_window"]["duration_us"], 80)


if __name__ == "__main__":
    unittest.main()
