"""Exercise the measurement helper at its real profiled(run) boundary, without a GPU."""

import contextlib
import importlib.util
import os
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SOURCE = Path(os.environ.get("CR297_PROFILE_SOURCE", ROOT / "cr297_profiles.py"))
package = types.ModuleType("profile_diagnostic")
package.__path__ = [str(SOURCE.parent)]
sys.modules[package.__name__] = package
spec = importlib.util.spec_from_file_location("profile_diagnostic.cr297_profiles", SOURCE)
profiles = importlib.util.module_from_spec(spec)
spec.loader.exec_module(profiles)


def event(start, end, *, device=0, stream=7, name="kernel", kind="CUDA"):
    return types.SimpleNamespace(
        name=name,
        key=name,
        device_type=f"DeviceType.{kind}",
        device_index=device,
        thread=stream,
        time_range=types.SimpleNamespace(start=start, end=end),
        device_time=end - start,
        self_device_time_total=end - start,
    )


class ProfileTests(unittest.TestCase):
    def setUp(self):
        profiles.PROFILES.clear()
        profiles._SEEN.clear()
        self.syncs = []
        self.activity_requests = []
        self.annotation_names = []
        self.events = [
            event(1000, 111000, name="cozy.cr297.profiled-call", kind="CPU"),
            event(1000, 71000, stream=7),
            event(41000, 111000, stream=8),
        ]
        self.torch = types.ModuleType("torch")
        self.torch.cuda = types.SimpleNamespace(
            synchronize=lambda device=0: self.syncs.append(device), device_count=lambda: 1
        )
        self.profiler = types.ModuleType("torch.profiler")
        self.profiler.ProfilerActivity = types.SimpleNamespace(CPU="CPU", CUDA="CUDA")
        self.profiler.profile = self.profile
        self.profiler.record_function = self.annotation
        self.torch.profiler = self.profiler
        self.patches = patch.dict(
            sys.modules, {"torch": self.torch, "torch.profiler": self.profiler}
        )
        self.patches.start()
        self.addCleanup(self.patches.stop)

    @contextlib.contextmanager
    def profile(self, *, activities):
        self.activity_requests.append(activities)
        yield types.SimpleNamespace(
            events=lambda: self.events, key_averages=lambda: self.events[1:]
        )

    @contextlib.contextmanager
    def annotation(self, name):
        self.annotation_names.append(name)
        yield

    def run_profile(self, *, at=1):
        with patch.object(profiles.time, "perf_counter", side_effect=[0.0, 0.11]):
            result = profiles.profiled(self.torch, "dense", lambda: "output", at=at)
        self.assertEqual(result, "output")
        return profiles.PROFILES.get("dense")

    def test_overlapping_streams_do_not_exceed_measurement_window(self):
        row = self.run_profile()
        # The legacy field makes this exact test reproduce the old symptom against b41d433fc.
        busy = row["busy_ms"] if "busy_ms" in row else row["device_activity"][0]["busy_union_ms"]
        self.assertEqual(busy, 110.0)
        self.assertEqual(row["device_activity"][0]["aggregate_device_work_ms"], 140.0)

    def test_contains_duplicate_and_disjoint_events(self):
        self.events = [
            self.events[0],
            event(11000, 51000),
            event(21000, 31000),
            event(11000, 51000),
            event(71000, 91000),
        ]
        row = self.run_profile()["device_activity"][0]
        self.assertEqual(row["busy_union_ms"], 60)
        self.assertEqual(row["aggregate_device_work_ms"], 110)
        self.assertEqual(row["observed_span_ms"], 80)
        self.assertEqual(row["unclassified_inactivity_ms"], 50)

    def test_devices_are_separate_and_all_finish_in_window(self):
        self.torch.cuda.device_count = lambda: 3
        self.events = [self.events[0], event(11000, 51000), event(21000, 81000, device=1)]
        row = self.run_profile()
        self.assertEqual([d["busy_union_ms"] for d in row["device_activity"]], [40, 60, 0])
        self.assertEqual(
            [d["unclassified_inactivity_ms"] for d in row["device_activity"]], [70, 50, 110]
        )
        self.assertEqual(self.syncs, [0, 1, 2, 0, 1, 2])
        self.assertEqual(row["measurement_window"]["duration_ms"], 110)
        self.assertEqual(self.annotation_names, ["cozy.cr297.profiled-call"])
        self.assertEqual(self.activity_requests, [["CPU", "CUDA"]])

    def test_window_clips_events_and_separates_profiler_overhead(self):
        self.events = [
            event(10000, 100000, name="cozy.cr297.profiled-call", kind="CPU"),
            event(0, 20000),
            event(90000, 120000),
            event(130000, 140000),
        ]
        row = self.run_profile()
        self.assertEqual(row["profiler_context_wall_ms"], 110)
        self.assertEqual(row["measurement_window"]["duration_ms"], 90)
        activity = row["device_activity"][0]
        self.assertEqual(activity["busy_union_ms"], 20)
        self.assertEqual(activity["aggregate_device_work_ms"], 20)
        self.assertEqual(activity["event_count"], 2)
        self.assertEqual(activity["unclassified_inactivity_ms"], 70)
        self.assertNotIn("wall_ms", row)
        self.assertNotIn("busy_ms", row)

    def test_only_requested_warm_call_profiles(self):
        self.assertIsNone(self.run_profile(at=2))
        measured = self.run_profile(at=2)
        self.assertIs(self.run_profile(at=2), measured)
        self.assertEqual(len(self.activity_requests), 1)

    def test_no_device_activity_is_full_unclassified_window(self):
        self.events = self.events[:1]
        activity = self.run_profile()["device_activity"][0]
        self.assertEqual(activity["event_count"], 0)
        self.assertEqual(activity["busy_union_ms"], 0)
        self.assertEqual(activity["unclassified_inactivity_ms"], 110)

    def test_missing_or_ambiguous_window_retains_output_and_reports_error(self):
        for events in (self.events[1:], [*self.events, self.events[0]]):
            with self.subTest(events=len(events)):
                profiles.PROFILES.clear()
                profiles._SEEN.clear()
                self.events = events
                row = self.run_profile()
                self.assertIn("Expected one measurement annotation", row["error"])
                self.assertNotIn("device_activity", row)

    def test_invalid_cuda_interval_retains_output_and_reports_error(self):
        self.events.append(event(100, float("nan")))
        row = self.run_profile()
        self.assertIn("Interval bounds must be finite", row["error"])
        self.assertNotIn("device_activity", row)

    def test_model_failure_propagates(self):
        def failed():
            raise RuntimeError("model failed")

        with self.assertRaisesRegex(RuntimeError, "model failed"):
            profiles.profiled(self.torch, "dense", failed, at=1)
        self.assertNotIn("dense", profiles.PROFILES)


if __name__ == "__main__":
    unittest.main()
