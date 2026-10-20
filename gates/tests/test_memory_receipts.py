import contextlib
import copy
import io
import itertools
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


def slim_positive():
    """A current-schema validator fixture, not a claim of another hardware run."""
    value = positive()
    memory = value.get("calls", [value])[0]["measurements"]["memory"]
    base = dict(request="r", rank=0, gpu=0, step=0, page=1, bytes=64,
                capacity=256, resident=64, orphan_bytes=0, reason="demand",
                **{"class": "pinned"})
    memory["trace"] = [dict(base, sequence=0, kind="begin", page=0, bytes=0),
                       dict(base, sequence=1, kind="bind"),
                       dict(base, sequence=2, kind="end", page=0, bytes=0)]
    memory["stats"]["copies"] = []
    return value


def measured_copy():
    # 1 GB in 0.1 s is 10 GB/s, exactly 50% of a 20 GB/s measured ceiling.
    return dict(kind="h2d", gpu=0, bytes=1_000_000_000, seconds=0.1,
                gbps=10.0, ceiling=20.0, pct=50.0, link="Gen4 x16 of Gen4 x16",
                busy_s=0.08, pin_s=0.01, staged=1_000_000_000, cached=0)


def proof_and_trace(sequence, sizes, capacity, initial=()):
    proof = dict(request="r", gpu=0, rank=0, version=1, complete=True,
                 isolated=True, capacity=capacity, initial=list(initial), invalid=[],
                 pages=[dict(page=page, content=f"fingerprint/{page}", payload=size,
                             mapped=max(size, 8)) for page, size in sizes.items()])
    base = dict(request="r", gpu=0, rank=0, page=0, bytes=0, capacity=capacity,
                resident=0, orphan_bytes=0, reason="demand", **{"class": "pinned"})
    trace = [dict(base, sequence=0, kind="begin", step=0, pass_index=0)]
    trace.extend(dict(base, sequence=index + 1, kind="bind", page=page,
                      bytes=max(sizes[page], 8), step=index // 3, pass_index=index // 3)
                 for index, page in enumerate(sequence))
    trace.append(dict(base, sequence=len(trace), kind="end", step=0, pass_index=0))
    return proof, trace


class MemoryReceipts(unittest.TestCase):
    def test_window_bound_counts_repeated_passes_without_calling_it_opt(self):
        proof, trace = proof_and_trace([1, 2, 3] * 3, {1: 4, 2: 4, 3: 4}, 8)
        bound, compulsory = API["window_lower_bound"](proof, trace)
        self.assertEqual((bound, compulsory), (20, 12))
        # Actual 20 B would satisfy the sufficient bound; 24 B is inconclusive.
        for actual, status in ((20, "sufficient"), (24, "inconclusive")):
            errors, measured, label = API["transfer_bound"](
                {"proofs": [proof]}, trace, {0}, {0: actual}, True)
            self.assertEqual(measured, 20)
            self.assertIn(status, label)
            self.assertEqual(bool(errors), actual == 24)

    def test_window_bound_is_below_exact_weighted_opt_for_small_traces(self):
        sizes, capacity = {1: 1, 2: 2, 3: 2}, 3
        for initial in ((), (1,), (2,)):
            for sequence in itertools.product(sizes, repeat=5):
                states = {frozenset(initial): 0}
                for page in sequence:
                    following = {}
                    for cache, prior in states.items():
                        available = tuple(cache | {page})
                        for mask in range(1 << len(available)):
                            kept = frozenset(p for i, p in enumerate(available) if mask & (1 << i))
                            if page not in kept or sum(sizes[p] for p in kept) > capacity:
                                continue
                            cost = prior + (0 if page in cache else sizes[page])
                            following[kept] = min(following.get(kept, cost), cost)
                    states = following
                proof, trace = proof_and_trace(sequence, sizes, capacity, initial)
                bound, _ = API["window_lower_bound"](proof, trace)
                self.assertLessEqual(bound, min(states.values()), (sequence, initial))

    def test_proof_requires_true_payload_initial_state_complete_scope_and_gpu_coverage(self):
        proof, trace = proof_and_trace([1, 2], {1: 2, 2: 3}, 8, [1])
        self.assertEqual(API["window_lower_bound"](proof, trace), (3, 3))
        for mutate, reason in (
            (lambda p: p.update(complete=False), "incomplete-transfer-proof"),
            (lambda p: p["initial"].append(7), "invalid-initial-residency"),
            (lambda p: p["pages"][0].update(payload=9), "invalid-transfer-catalog"),
            (lambda p: p.update(invalid=["registration-changed"]), "incomplete-transfer-proof"),
        ):
            changed = copy.deepcopy(proof)
            mutate(changed)
            errors, _, _ = API["transfer_bound"]({"proofs": [changed]}, trace, {0}, {0: 3}, True)
            self.assertIn(reason, errors)
        proof["isolated"] = False
        errors, _, status = API["transfer_bound"]({"proofs": [proof]}, trace, {0}, {0: 3}, True)
        self.assertEqual((errors, status), (["transfer-proof-not-isolated"], "shared-scope"))
        self.assertEqual(API["transfer_bound"]({"proofs": [proof]}, trace, {0}, {0: 3}, False)[0], [])
        self.assertIn("transfer-proof-gpu-coverage", API["transfer_bound"](
            {"proofs": [proof]}, trace, {0, 1}, {}, True)[0])
        extra = dict(trace[0], request="unaccounted")
        self.assertIn("transfer-proof-trace-coverage", API["transfer_bound"](
            {"proofs": [proof]}, [*trace, extra], {0}, {}, True)[0])

    def test_slim_trace_protects_working_pages_and_orphan_order(self):
        def event(sequence, kind, **values):
            row = dict(sequence=sequence, request="r", step=0, page=1, bytes=1,
                       capacity=8, resident=1, orphan_bytes=0, kind=kind,
                       reason="demand", **{"class": "unused"})
            row.update(values)
            return row
        trace = [event(0, "begin"), event(1, "evict"), event(2, "bind"), event(3, "end")]
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
        bound = dict(begin, sequence=2, kind="bind")
        end = dict(begin, sequence=3, kind="end")
        self.assertEqual(API["trace_errors"]([begin, fill, bound, end]), [])
        self.assertIn("missing-page-bind-evidence", API["trace_errors"]([begin, end]))
        self.assertIn("repeated-speculative-fetch", API["trace_errors"]([begin, fill, dict(fill, sequence=2), dict(bound, sequence=3), dict(end, sequence=4)]))
        del fill["working_set"]
        self.assertIn("unmeasured-speculation", API["trace_errors"]([begin, fill, end]))
        self.assertEqual(API["trace_errors"]([]), ["missing-decision-trace"])

    def test_explicit_lowering_stops_speculation_without_forbidding_demand(self):
        base = dict(request="r", step=0, page=1, bytes=1, capacity=8,
                    resident=1, orphan_bytes=0, reason="demand", **{"class": "working"})
        trace = [dict(base, sequence=0, kind="begin"),
                 dict(base, sequence=1, kind="lower", reason="lower"),
                 dict(base, sequence=2, kind="bind"),
                 dict(base, sequence=3, kind="end")]
        self.assertEqual(API["trace_errors"](trace), [])
        trace[2] = dict(base, sequence=2, kind="fill", reason="speculation",
                        working_set=4, speculative_budget=3)
        self.assertIn("speculation-after-starvation", API["trace_errors"](trace))

    def record(self, value, cell="24-sdxl4", cold="", slim=""):
        with tempfile.TemporaryDirectory() as directory:
            path, summary = Path(directory) / "run.json", Path(directory) / "summary.txt"
            path.write_text(json.dumps(value))
            summary.write_text(cold)
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                code = API["record"](path, cell, str(value["number"]), "completed landed", "", summary, slim)
            return code, out.getvalue()

    def compare(self, controls, candidate, strict=False):
        with tempfile.TemporaryDirectory() as directory:
            old, new = Path(directory) / "old.txt", Path(directory) / "new.txt"
            old.write_text(controls)
            new.write_text(candidate)
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                code = API["compare"](old, new, ["24"], strict)
            return code, out.getvalue()

    def test_strict_performance_gate_requires_a_valid_matching_control(self):
        current = "24-h3ref5 run=2 completed landed exec=100s poisoned=0 leftover=0 bar=ok\n"
        valid = "24-h3ref5 run=1 completed landed exec=101s poisoned=0 leftover=0 bar=ok\n"
        self.assertEqual(self.compare(valid, current, strict=True)[0], 0)
        self.assertEqual(self.compare(valid.replace("101s", "99s"), current, strict=True)[0], 1)
        for controls in ("", valid.replace("completed", "failed"), valid.replace("bar=ok", "bar=poisoned")):
            self.assertEqual(self.compare(controls, current, strict=True)[0], 1)
        self.assertEqual(self.compare(valid, current.replace("bar=ok", "substitute=32 bar=ok"), strict=True)[0], 1)
        self.assertEqual(self.compare("", "", strict=True)[0], 1)

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

    def test_zero_transfer_proves_zero_minimum_without_invented_cache_state(self):
        value = slim_positive()
        code, text = self.record(value, slim="slim")
        self.assertEqual(code, 0, text)
        self.assertIn("transfer_minimum=0B", text)
        self.assertIn("copy_ceiling_pct=-", text)
        del value["calls"][0]["measurements"]["memory"]["stats"]["copies"]
        self.assertIn("missing-copy-evidence", self.record(value, slim="slim")[1])

    def test_real_copy_percent_is_reported_but_does_not_invent_a_minimum(self):
        value = slim_positive()
        memory = value["calls"][0]["measurements"]["memory"]
        memory["stats"].update(paged_in=1_000_000_000, copies=[measured_copy()])
        memory["trace"].insert(1, dict(memory["trace"][1], kind="fill"))
        for index, event in enumerate(memory["trace"]):
            event["sequence"] = index
        code, text = self.record(value, slim="slim")
        self.assertEqual(code, 1)
        self.assertIn("copy_ceiling_pct=h2d@0:50.0%", text)
        self.assertIn("transfer_minimum=unknown", text)
        self.assertIn("missing-transfer-minimum-evidence", text)
        self.assertNotIn("copy-percentage-mismatch", text)

    def test_complete_nonzero_proof_reaches_record_without_claiming_a_minimum(self):
        value = slim_positive()
        memory = value["calls"][0]["measurements"]["memory"]
        proof, trace = proof_and_trace([1], {1: 1_000_000_000}, 2_000_000_000)
        trace.insert(1, dict(trace[1], kind="fill"))
        for index, event in enumerate(trace):
            event["sequence"] = index
        memory["trace"] = trace
        memory["stats"].update(paged_in=1_000_000_000, copies=[measured_copy()], proofs=[proof])
        code, text = self.record(value, slim="slim")
        self.assertEqual(code, 0, text)
        self.assertIn("h2d_window_lower_bound=1000000000B", text)
        self.assertIn("h2d_bound_status=sufficient", text)
        self.assertIn("transfer_minimum=unknown", text)
        proof["complete"] = False
        self.assertIn("incomplete-transfer-proof", self.record(value, slim="slim")[1])

    def test_missing_or_impossible_copy_measurements_cannot_pass(self):
        cases = [
            ("seconds", 0, "invalid-copy-measurement"),
            ("seconds", float("nan"), "invalid-copy-measurement"),
            ("ceiling", 0, "unmeasured-h2d-ceiling"),
            ("ceiling", float("inf"), "invalid-copy-measurement"),
            ("pct", None, "unmeasured-h2d-ceiling"),
            ("pct", 101, "unmeasured-h2d-ceiling"),
            ("pct", 75, "copy-percentage-mismatch"),
            ("gbps", 5, "copy-rate-mismatch"),
            ("bytes", True, "invalid-copy-identity"),
            ("gpu", 1, "invalid-copy-identity"),
            ("staged", 1_000_000_001, "invalid-copy-byte-counts"),
        ]
        for field, value, reason in cases:
            with self.subTest(field=field, value=value):
                row = measured_copy()
                row[field] = value
                errors, _ = API["movement_evidence"]({"copies": [row], "paged_in": 1_000_000_000}, [], {0})
                self.assertIn(reason, errors)

    def test_decimal_rounding_does_not_become_a_false_percentage_failure(self):
        row = measured_copy()
        row.update(bytes=1_000_000, seconds=0.003, gbps=0.33,
                   ceiling=3.33, pct=10.0, staged=1_000_000)
        errors, _ = API["movement_evidence"]({"copies": [row], "paged_in": 1_000_000}, [], {0})
        self.assertEqual(errors, ["missing-transfer-minimum-evidence"])

    def test_cached_reads_are_not_claimed_as_physical_disk_measurements(self):
        row = dict(kind="disk", gpu=-1, bytes=100, seconds=1.0, gbps=0.0,
                   ceiling=0.0, pct=None, cached=100, staged=0)
        errors, facts = API["movement_evidence"]({"copies": [row], "paged_in": 0}, [], {0})
        self.assertEqual(errors, [])
        self.assertEqual(facts["cached_bytes"], 100)
        self.assertEqual(facts["minimum_bytes"], 0)
        row.update(ceiling=1.0, pct=0.0)
        self.assertIn("cached-disk-has-hardware-ceiling",
                      API["movement_evidence"]({"copies": [row], "paged_in": 0}, [], {0})[0])
        row.update(cached=50, ceiling=0.0, pct=None)
        self.assertIn("mixed-disk-copy-unmeasured",
                      API["movement_evidence"]({"copies": [row], "paged_in": 0}, [], {0})[0])

    def test_padding_initialization_is_preserved_without_invented_h2d_traffic(self):
        row = dict(kind="zero", gpu=0, bytes=2_000_000, seconds=0.001, gbps=2.0,
                   ceiling=0.0, pct=None, cached=0, staged=0)
        errors, facts = API["movement_evidence"]({"copies": [row], "paged_in": 0}, [], {0})
        self.assertEqual(errors, [])
        self.assertEqual(facts["h2d_bytes"], 0)
        row.update(ceiling=2.0, pct=100.0)
        self.assertIn("zero-initialization-has-transfer-ceiling",
                      API["movement_evidence"]({"copies": [row], "paged_in": 0}, [], {0})[0])

    def test_disk_and_each_gpu_report_their_measured_percentage(self):
        disk = dict(kind="disk", gpu=-1, bytes=1_000_000_000, seconds=1.0,
                    gbps=1.0, ceiling=2.0, pct=50.0, cached=0, staged=0)
        second = dict(measured_copy(), gpu=1)
        errors, facts = API["movement_evidence"](
            {"copies": [disk, measured_copy(), second], "paged_in": 2_000_000_000}, [], {0, 1})
        self.assertEqual(errors, ["missing-transfer-minimum-evidence"])
        self.assertEqual(facts["percentages"], ["disk@-1:50.0%", "h2d@0:50.0%", "h2d@1:50.0%"])
        self.assertEqual(facts["uncached_disk_bytes"], 1_000_000_000)
        self.assertEqual(facts["h2d_bytes"], 2_000_000_000)

    def test_copy_log_must_match_the_request_counter(self):
        stats = {"copies": [measured_copy()], "paged_in": 1}
        self.assertIn("copy-request-accounting-mismatch",
                      API["movement_evidence"](stats, [], {0})[0])
        stats = {"copies": [], "paged_in": 1}
        errors, facts = API["movement_evidence"](stats, [], {0})
        self.assertIn("copy-request-accounting-mismatch", errors)
        self.assertIsNone(facts["minimum_bytes"])


if __name__ == "__main__":
    unittest.main()
