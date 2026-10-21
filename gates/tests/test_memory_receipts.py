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
FIXTURES = ROOT / "tests/fixtures"
CATALOG = FIXTURES / "catalog-20261021.json"
Step, belady = API["Step"], API["belady"]
GB = 10**9


def fixture(name):
    return json.loads((FIXTURES / f"{name}.json").read_text())


def varena(reads, capacity=3 * GB, w=int(2.5 * GB), prefetched=0):
    """A one-GPU candidate record as TensorD reports it, in S1's `Stats` field names: a 2 GB DiT in two 1 GB blocks over
    three steps; `reads` names each step's origin for both blocks (`Tier` copies them, `Hit` finds them mapped). The
    first `prefetched` bytes of hits were prefetched by this request."""
    trace = [{"step": s, "set": "base_model/transformer_ref", "block": b, "bytes": GB, "origin": origin,
              "speculative": s == 0 and origin == "Hit"} for s, origin in enumerate(reads) for b in (0, 1)]
    copied = sum(r["bytes"] for r in trace if r["origin"] == "Tier")
    stats = {"request": "r", "trace": trace, "stall_ns": copied // 40, "evicted": [0] * 7, "evicted_lowered": 0,
             "speculative_evicted_d": 0, "retries": 0, "cuts": 0, "parks": 0, "lowered": False, "peak": w, "leftover": 0,
             "moved": {"kept": {"bytes": 0, "ns": 0}, "tier": {"bytes": copied, "ns": copied // 20},
                       "store": {"bytes": 0, "ns": 0}, "speculative": prefetched},
             "steps": [{"capacity": capacity, "w": w, "evicted": [0] * 7} for _ in reads]}
    return {"number": 1, "status": "completed", "execution_ms": 1000, "created_at_ms": 1, "settled_at_ms": 2001,
            "calls": [{"measurements": {"execution": {"gpus": [{"gpu": 0, "start_us": 1, "end_us": 2}]},
                                        "memory": {"stats": {"varena": [
                                            {"gpu": 0, "ceiling": {"disk_gbps": 7.0, "h2d_gbps": 25.0}, "stats": stats}]}}}}]}


class Receipts(unittest.TestCase):
    def record(self, value, cell, line="shipping", cap="24", state="completed landed"):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.json"
            path.write_text(json.dumps(value))
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                code = API["record"](path, cell, str(value["number"]), state, line, cap)
            text = out.getvalue()
            (Path(directory) / "summary.txt").write_text(text)
            return code, text, API["receipts"](Path(directory) / "summary.txt")[0]

    def compare(self, text, lines):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "summary.txt"
            path.write_text(text)
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                code = API["compare"](path, lines)
            return code, out.getvalue()

    def test_belady_is_the_textbook_optimum(self):
        pages = [7, 0, 1, 2, 0, 3, 0, 4, 2, 3, 0, 3, 2, 1, 2, 0, 1, 7, 0, 1]  # 3 frames: OPT faults 9 times
        self.assertEqual(belady([Step(3, tuple((str(p), 1) for p in pages))]), 9)

    def test_belady_counts_held_pages_free_and_shrinks_with_capacity(self):
        refs = (("a", 1), ("b", 1))
        self.assertEqual(belady([Step(2, refs)] * 3), 2)
        self.assertEqual(belady([Step(2, refs)] * 3, held=refs), 0)
        # A step with room for one page evicts the one used furthest ahead (b), which costs its copy back later.
        self.assertEqual(belady([Step(2, refs), Step(1, (("a", 1),)), Step(2, refs)], held=refs), 1)
        # A page larger than the capacity streams through every time.
        self.assertEqual(belady([Step(1, (("big", 2),))] * 3), 6)

    def test_shipping_warm_h3_on_24_streams_its_dit_every_step(self):
        code, text, row = self.record(fixture("h3-warm-12665"), "24-h3ref5-warm.1")
        self.assertEqual(code, 1)
        self.assertEqual(row.bar, "dit-reads-per-step")
        self.assertEqual(row.moved, 70.264)  # every copy run's bytes: the record's paged_in
        self.assertIn("dit_step_max=11366756352B", text)
        self.assertIn("ram_hit=100.0%", text)
        self.assertIn("min=-GB", text)  # no page trace on the shipping line

    def test_shipping_cold_h3_meets_the_bar_with_its_retries(self):
        code, text, row = self.record(fixture("h3-cold-12663"), "24-h3ref5")
        self.assertEqual((code, row.bar), (0, "ok"), text)
        self.assertIn("oom_retries=1333 cuts=1 parks=0", text)

    def test_shipping_qwen_retries_park_and_disk_reads_fail_only_a_warm_cell(self):
        code, text, _ = self.record(fixture("qwen4-12709"), "24-qwen4")
        self.assertEqual(code, 0, text)
        self.assertIn("parks=1", text)
        self.assertIn("fill=h2d:3.91GB/s/19.7%,disk:1.72GB/s/50.9% ", text)  # each against its own ceiling
        code, text, row = self.record(fixture("qwen4-12709"), "24-qwen4-warm.2")
        self.assertEqual(row.bar, "warm-oom-retries,store-in-warm")

    def test_every_executed_gpu_needs_its_leftover(self):
        value = fixture("h3-warm-12665")
        value["calls"][0]["measurements"]["execution"]["gpus"].append({"gpu": 1, "start_us": 5, "end_us": 6})
        self.assertIn("leftover-gpu-coverage", self.record(value, "24-h3ref5")[2].bar)

    def test_leftover_poison_and_missing_stats_fail(self):
        value = fixture("h3-cold-12663")
        value["calls"][0]["measurements"]["memory"]["stats"]["leftover_by_gpu"] = [[0, 32]]
        value["reason"] = {"code": "executor_poisoned", "message": "poisoned"}
        self.assertEqual(self.record(value, "24-h3ref5")[2].bar, "poisoned,leftover")
        del value["calls"][0]["measurements"]["memory"]["stats"]["copies"]
        self.assertIn("no-mem-stats(KeyError:'copies')", self.record(value, "24-h3ref5")[2].bar)

    def test_the_kill_victim_must_end_canceled(self):
        value = fixture("h3-cold-12663")
        self.assertEqual(self.record(value, "24-kill-victim")[2].bar, "not-killed")
        value["status"] = "canceled"
        self.assertEqual(self.record(value, "24-kill-victim", state="canceled landed")[0], 0)

    def test_candidate_bytes_against_belady_min(self):
        code, text, row = self.record(varena(["Tier", "Hit", "Hit"]), "24-h3ref5-warm.1", line="candidate")
        self.assertEqual((code, row.bar, row.min, row.moved), (1, "dit-reads-per-step", 2.0, 2.0))
        code, text, row = self.record(varena(["Tier", "Hit", "Hit"]), "24-h3ref5", line="candidate")
        self.assertEqual((code, row.bar), (0, "ok"), text)
        code, text, row = self.record(varena(["Tier", "Tier", "Tier"]), "24-h3ref5", line="candidate")
        self.assertEqual(row.bar, "moved>1.05xMIN")
        self.assertEqual((row.min, row.moved), (2.0, 6.0))
        # A 1.5 GB pass room leaves one block's room: MIN itself streams all 6 GB.
        _, _, row = self.record(varena(["Tier", "Tier", "Tier"], w=int(3.5 * GB)), "24-h3ref5", line="candidate")
        self.assertEqual((row.bar, row.min), ("ok", 6.0))
        warm = varena(["Hit", "Hit", "Hit"])
        self.assertEqual(self.record(warm, "24-h3ref5-warm.1", line="candidate")[2].bar, "ok")

    def test_candidate_hits_prefetched_by_this_request_are_moves_for_min(self):
        # Two prefetched hits at step 0: held before the request when it copied nothing, copied by it otherwise.
        self.assertEqual(self.record(varena(["Hit", "Hit", "Hit"]), "24-h3ref5", line="candidate")[2].min, 0.0)
        row = self.record(varena(["Hit", "Hit", "Hit"], prefetched=2 * GB), "24-h3ref5", line="candidate")[2]
        self.assertEqual(row.min, 2.0)
        row = self.record(varena(["Hit", "Hit", "Hit"], prefetched=GB), "24-h3ref5", line="candidate")[2]
        self.assertEqual(row.min, 1.0)

    def test_candidate_fills_against_their_own_ceilings(self):
        code, text, _ = self.record(varena(["Tier", "Hit", "Hit"]), "24-h3ref5", line="candidate")
        self.assertIn("fill=tier:20.00GB/s/80.0% ", text)  # 2 GB in 0.1 s against the 25 GB/s h2d ceiling

    def test_actual_candidate_record_12857(self):
        # S8's vertical slice: SDXL 1 MP on varena-one, its weights loaded before the request; TensorD's wrapping.
        code, text, row = self.record(fixture("sdxl1-candidate-12857"), "24-sdxl1", line="candidate")
        self.assertEqual((code, row.bar, row.moved, row.min), (0, "ok", 0.0, 0.0), text)
        self.assertIn("leftover=0 ", text)

    def test_candidate_misplaced_evictions_fail(self):
        value = varena(["Tier", "Hit", "Hit"])
        stats = value["calls"][0]["measurements"]["memory"]["stats"]["varena"][0]["stats"]
        stats["evicted"] = [0, 0, 0, 0, 0, 5, 1]
        stats["evicted_lowered"] = 4
        stats["speculative_evicted_d"] = 1
        self.assertEqual(self.record(value, "24-sdxl4", line="candidate")[2].bar,
                         "gpu0-W-evicted,gpu0-P-evicted,gpu0-speculation-evicted-D")

    def test_matched_comparison_uses_medians_and_counts_only_the_line_under_test(self):
        line = "{cell} run={n} completed landed line={line} cap=24 wall=10s exec={e}s moved={m}GB min=-GB bar={bar}\n"
        rows = [line.format(cell=f"24-h3ref5-warm.{k}", n=k, line="shipping", e=e, m=5, bar="ok")
                for k, e in ((1, 90), (2, 91), (3, 89))]
        rows += [line.format(cell=f"24-h3ref5-warm.{k}", n=10 + k, line="candidate", e=e, m=0, bar="ok")
                 for k, e in ((1, 88), (2, 200), (3, 89))]
        code, text = self.compare("".join(rows), ["shipping", "candidate"])
        self.assertEqual(code, 0, text)
        self.assertIn("exec 90.000 -> 89.000; moved 5.000 -> 0.000", text)
        rows[3] = rows[3].replace("exec=88s", "exec=95s")
        code, text = self.compare("".join(rows), ["shipping", "candidate"])
        self.assertEqual(code, 1)
        self.assertIn("WORSE", text)
        missed = rows[0].replace("bar=ok", "bar=dit-reads-per-step")
        code, text = self.compare("".join([missed, *rows[1:]]).replace("exec=95s", "exec=88s"), ["shipping", "candidate"])
        self.assertEqual(code, 0)
        self.assertIn("not compared (shipping missed its bar)", text)
        self.assertEqual(self.compare(missed, ["shipping"])[0], 1)  # a shakedown counts the shipping line's own

    def test_rentals_come_cheapest_first_from_the_catalog(self):
        self.assertEqual(API["rental"](CATALOG, "8"), "1x24 1 8GiB rtx-pro-4000 l4 rtx-4090 rtx-3090")
        self.assertEqual(API["rental"](CATALOG, "2x12"), "2x24 2 12GiB rtx-pro-4000 l4 rtx-4090 rtx-3090")
        self.assertEqual(API["rental"](CATALOG, "real8"), "real8 1 - rtx-3070")
        self.assertEqual(API["rental"](CATALOG, "h100"), "h100 1 - h100-nvl h100-sxm5-80gb")

    def test_the_plan_prices_measured_groups_and_lists_the_rest(self):
        with tempfile.TemporaryDirectory() as directory:
            walls = Path(directory) / "summary.txt"
            walls.write_text("24-h3ref5 run=1 completed landed line=shipping cap=24 wall=1800s exec=1s bar=ok\n"
                             "24-h3ref5-warm.1 run=2 completed landed line=shipping cap=24 wall=1800s exec=1s bar=ok\n")
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                API["plan"](CATALOG, ["shipping", "candidate"], [walls], ["24=h3ref5,sdxl4", "8=h3ref5"])
            text = out.getvalue()
        self.assertIn("1x24 (24 8): rtx-pro-4000 x1 $0.918/h; measured 2.00 h, $1.84; unmeasured: 24:sdxl4 8:h3ref5", text)


if __name__ == "__main__":
    unittest.main()
