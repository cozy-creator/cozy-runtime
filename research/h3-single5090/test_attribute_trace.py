import unittest

from attribute_trace import Ranges, attribute, manual_annotations


def event(cat, name, ts, dur, *, pid=1, tid=1, corr=None):
    args = {} if corr is None else {"correlation": corr}
    return dict(ph="X", cat=cat, name=name, ts=ts, dur=dur, pid=pid, tid=tid, args=args)


class AttributionTests(unittest.TestCase):
    def test_async_kernel_uses_launch_range_and_keeps_all_events(self):
        rows = [
            event("user_annotation", "step", 10, 20),
            event("user_annotation", "QKV", 11, 4),
            event("cuda_runtime", "cudaLaunchKernel", 12, 1, corr=99),
            # Device executes after QKV and step's CPU markers have both ended.
            event("kernel", "gemm", 80, 20, pid=0, tid=7, corr=99),
            event("gpu_memcpy", "Memcpy HtoD", 85, 8, pid=0, tid=8, corr=99),
            event("kernel", "unknown", 101, 3, pid=0, tid=7, corr=100),
            event("kernel", "no-id", 102, 2, pid=0, tid=7),
        ]
        result = attribute(lambda: iter(reversed(rows)))
        self.assertEqual(
            result["gpu_attribution_counts"],
            {
                "gpu_without_correlation": 1,
                "missing_cpu_api": 1,
                "annotated": 2,
            },
        )
        gemm = next(row for row in result["groups"] if row["kernel_or_copy"] == "gemm")
        self.assertEqual(gemm["annotations"], ["step", "QKV"])
        self.assertEqual(sum(row["count"] for row in result["groups"]), 4)
        # Overlap stays a duration sum, not a falsely claimed elapsed time.
        self.assertEqual(sum(row["duration_sum_us"] for row in result["groups"]), 33)

    def test_thread_boundaries_and_duplicate_ids_are_explicit(self):
        rows = [
            event("user_annotation", "wrong-thread", 0, 100, tid=2),
            event("cuda_runtime", "launch", 5, 1, corr=1),
            event("cuda_runtime", "launch", 6, 1, corr=2),
            event("cuda_driver", "launch", 6, 1, corr=2),
            event("kernel", "a", 7, 1, corr=1),
            event("kernel", "b", 8, 1, corr=2),
        ]
        result = attribute(lambda: iter(rows))
        self.assertEqual(
            result["gpu_attribution_counts"],
            {
                "api_without_annotation": 1,
                "ambiguous_correlation": 1,
            },
        )
        self.assertTrue(all(not row["annotations"] for row in result["groups"]))

    def test_ranges_handle_nested_adjacent_and_overlapping_intervals(self):
        ranges = Ranges(
            [
                (0, 30, "outer"),
                (1, 5, "left"),
                (5, 8, "right"),
                (2, 20, "overlap"),
                (30, 40, "next"),
            ]
        )
        self.assertEqual(ranges.enclosing(5), ("outer", "overlap", "right"))
        self.assertEqual(ranges.enclosing(25), ("outer",))
        self.assertEqual(ranges.enclosing(30), ("next",))
        self.assertEqual(ranges.enclosing(40), ())

    def test_invalid_gpu_events_are_reported(self):
        rows = [event("kernel", "bad", 1, -1), event("kernel", "nan", float("nan"), 1)]
        self.assertEqual(attribute(lambda: iter(rows))["invalid_gpu_events"], 2)

    def test_manual_unix_ranges_preserve_clock_and_launch_thread(self):
        base = 1_790_857_026_000_000_000
        manual = [
            {
                "label": "QKV",
                "start_unix_ns": base + 11_000,
                "end_unix_ns": base + 15_000,
                "pid": 1,
                "tid": 7,
            }
        ]
        rows = list(manual_annotations(manual, {"baseTimeNanoseconds": base}))
        rows += [
            event("cuda_runtime", "launch", 12, 1, tid=7, corr=42),
            event("kernel", "gemm", 80, 20, pid=0, tid=9, corr=42),
        ]
        result = attribute(lambda: iter(rows))
        self.assertEqual(result["groups"][0]["annotations"], ["QKV"])
        with self.assertRaises(ValueError):
            list(manual_annotations(manual, {}))

    def test_cupti_signed_thread_alias_and_collision(self):
        row = {
            "label": "step",
            "start_unix_ns": 10000,
            "end_unix_ns": 30000,
            "pid": 1,
            "tid": 7,
            "native_tid": 7,
            "python_thread_ident": 0xF0000001,
            "pthread_tid_int32": -268435455,
            "pthread_tid_uint32": 4026531841,
        }
        ranges = list(manual_annotations([row], {"baseTimeNanoseconds": 0}))
        events = [
            *ranges,
            event("cuda_runtime", "launch", 12, 1, tid=-268435455, corr=4),
            event("kernel", "gemm", 40, 3, pid=0, tid=0, corr=4),
        ]
        result = attribute(lambda: iter(events))
        self.assertEqual(result["groups"][0]["annotations"], ["step"])
        collision = dict(row, native_tid=8, tid=8, python_thread_ident=0x1F0000001)
        with self.assertRaises(ValueError):
            list(manual_annotations([row, collision], {"baseTimeNanoseconds": 0}))


if __name__ == "__main__":
    unittest.main()
