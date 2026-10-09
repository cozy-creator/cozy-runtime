"""CPU-only fake CLI tests: no provider, daemon, subprocess shell or CUDA calls."""

import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

spec = importlib.util.spec_from_file_location(
    "batch_matrix", Path(__file__).with_name("batch_matrix.py")
)
batch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(batch)

FAKE = r"""
import hashlib, json, os, pathlib, signal, sys, time
root = pathlib.Path(os.environ['FAKE_COZY_ROOT'])
args = sys.argv[1:]
with (root / 'calls.jsonl').open('a') as f: f.write(json.dumps(args)+'\n')
mode = (root / 'mode').read_text()
state_path = root / 'state.json'
if args[:2] == ['rental', 'show']:
    print(json.dumps({'rental_id':'pr-fixed','state':'ready','running':0,'queued':0,
          'runtime_version':'wrong' if mode=='preflight-bad-runtime' else 'runtime-fixed',
          'tensorfs_version':'0.4.0','accelerator_count':1,'devices':[{'driver':'driver-fixed'}]}))
    sys.exit(0)
if args[1] not in ['show', 'watch']:
    out = pathlib.Path(next(x.split('=',1)[1] for x in args if x.startswith('--out=')))
    out.mkdir(exist_ok=True, parents=True)
    video = out / '5024-video.mp4'
    video.write_bytes(b'fake video bytes; full AV is the separate review helper')
    selector = next(x.split('=',2)[2] for x in args if x.startswith('--attention-kernel='))
    result = {'status':'completed', 'saved':[{'output':'video','path':str(video),
              'bytes':video.stat().st_size,'digest':'sha256:'+hashlib.sha256(video.read_bytes()).hexdigest()}]}
    show = {'request_id':'req-fixed','number':5024,'status':'completed','execution_known':True,
            'execution_ms':280875,'steps':[{'name':'denoise','count':8}],
            'gpus':[{'uuid':'wrong' if mode=='wrong-gpu' else 'GPU-fixed',
                     'attention':{'observed':'fl2va_dit='+selector,'sol':{'sparse':192}}}]}
    show['output'] = [{'name':'video','type':'video','status':'completed','path':str(video),
                      'length':video.stat().st_size, 'sha256':result['saved'][0]['digest']}]
    state_path.write_text(json.dumps({'result':result,'show':show}))
    if mode != 'lost-unknown':
        print(json.dumps({'type':'run.created','request_id':'req-fixed'}),file=sys.stderr,flush=True)
    if mode.startswith('lost'): sys.exit(1)
    if mode == 'stop':
        (root/'STOP').touch()
        def detached(*_):
            (root/'detached').write_text('SIGINT received, no cancel')
            sys.exit(2)
        signal.signal(signal.SIGINT,detached)
        time.sleep(30)
    print(json.dumps(result))
else:
    state = json.loads(state_path.read_text())
    print(json.dumps(state['show' if args[1]=='show' else 'result']))
"""


class BatchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.cli = self.root / "fake-cozy"
        self.cli.write_text("#!" + sys.executable + "\n" + FAKE)
        self.cli.chmod(0o755)
        (self.root / "mode").write_text("success")
        self.input = self.root / "input.json"
        self.input.write_text('{"seed":581204,"duration_s":15}')
        self.row = {
            "id": "kernel--dense4-sol4--sparring",
            "kernel": "kernel",
            "scene": "sparring",
            "schedule": "dense4-sol4",
            "seed": 581204,
            "attention_selector": "selector",
            "input": str(self.input),
            "input_sha256": batch.sha(self.input),
            "expected_steps": 8,
            "output_dir": str(self.root / "output"),
            "run_show": str(self.root / "output" / "show.json"),
            "status": "pending",
            "video": None,
        }
        self.path = self.root / "manifest.json"
        batch.atomic(
            self.path,
            {
                "runs": [self.row],
                "target": "/fixed/package/fl2va_turbo",
                "rental": "owned-rental",
                "gpu_uuid": "GPU-fixed",
                "hub_url": "http://fixture-hub",
                "rental_id": "pr-fixed",
                "runtime_wheel": {"version": "runtime-fixed"},
                "tensorfs_version": "0.4.0",
                "driver": "driver-fixed",
            },
        )
        self.env = patch.dict(os.environ, {"FAKE_COZY_ROOT": str(self.root)})
        self.env.start()
        self.addCleanup(self.env.stop)

    def driver(self):
        manifest = batch.Manifest(self.path)
        return batch.Driver(manifest, str(self.cli), poll=0.005, observer_seconds=5), manifest.data[
            "runs"
        ][0]

    def calls(self):
        p = self.root / "calls.jsonl"
        return [json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []

    def submissions(self):
        return [x for x in self.calls() if x[1] not in {"show", "watch"}]

    def test_success_and_resume_only_after_file_evidence(self):
        driver, row = self.driver()
        driver.run_case(row)
        self.assertEqual(row["execution_ms"], 280875)
        self.assertEqual(row["driver_evidence"]["sparse_calls"], 192)
        calls = self.calls()
        driver, row = self.driver()
        driver.run_case(row)
        self.assertEqual(calls, self.calls())
        self.assertEqual(len(self.submissions()), 1)
        self.assertIn("--await", self.submissions()[0])
        self.assertIn("--tensorhub=http://fixture-hub", self.submissions()[0])
        self.assertFalse(any(x.startswith("--timeout") for x in self.submissions()[0]))
        Path(row["video"]).write_bytes(b"corrupt")
        with self.assertRaises(batch.ReviewRequired):
            driver.run_case(row)
        self.assertEqual(len(self.submissions()), 1)

    def test_lost_ack_with_id_uses_readonly_recovery(self):
        (self.root / "mode").write_text("lost-known")
        driver, row = self.driver()
        with self.assertRaises(batch.ReviewRequired):
            driver.run_case(row)
        driver, row = self.driver()
        driver.run_case(row)
        self.assertEqual(row["status"], "completed")
        self.assertEqual(len(self.submissions()), 1)
        self.assertEqual(
            [x[1] for x in self.calls() if x[0] == "run"][1:], ["show", "watch", "show"]
        )

    def test_lost_ack_without_id_never_resubmits(self):
        (self.root / "mode").write_text("lost-unknown")
        driver, row = self.driver()
        with self.assertRaises(batch.ReviewRequired):
            driver.run_case(row)
        driver, row = self.driver()
        with self.assertRaisesRegex(batch.ReviewRequired, "Uncertain"):
            driver.run_case(row)
        self.assertEqual(len(self.submissions()), 1)

    def test_stop_detaches_and_recovery_is_readonly(self):
        (self.root / "mode").write_text("stop")
        driver, row = self.driver()
        with self.assertRaises(batch.Detached):
            driver.run_case(row)
        self.assertTrue((self.root / "detached").exists())
        self.assertEqual(row["request_id"], "req-fixed")
        (self.root / "STOP").unlink()
        driver, row = self.driver()
        driver.run_case(row)
        self.assertEqual(len(self.submissions()), 1)
        self.assertFalse(any("cancel" in args for args in self.calls()))

    def test_wrong_gpu_and_changed_input_stop(self):
        (self.root / "mode").write_text("wrong-gpu")
        driver, row = self.driver()
        with self.assertRaisesRegex(batch.ReviewRequired, "GPU UUID"):
            driver.run_case(row)
        self.input.write_text("{}")
        driver, row = self.driver()
        with self.assertRaisesRegex(batch.ReviewRequired, "Input differs"):
            driver.run_case(row)
        self.assertEqual(len(self.submissions()), 1)

    def test_manual_warmup_is_adopted_without_submission(self):
        driver, row = self.driver()
        driver.run_case(row)
        manifest = batch.Manifest(self.path)
        saved = manifest.data["runs"][0]
        saved["events_path"] = saved["stderr"]  # Parent's actual manual receipt field names.
        saved["result_path"] = saved["result"]
        saved["source_run_id"] = "kernel--dense4-sol4--sparring"  # A cell, NOT a request id.
        for key in [
            "request_id",
            "run_number",
            "driver_state",
            "driver_identity",
            "driver_evidence",
            "result",
        ]:
            saved.pop(key, None)
        saved["status"] = "submitting"
        saved["events_path"] = str(self.root / "manual-events.jsonl")
        Path(saved["events_path"]).write_text('{"request_id":"req-fixed","type":"run.created"}\n')
        saved.pop("stdout", None)
        saved.pop("stderr", None)
        manifest.save()
        driver, row = self.driver()
        driver.run_case(row)
        self.assertEqual(row["status"], "completed")
        self.assertEqual(len(self.submissions()), 1)

    def test_kernel_major_plan_and_manifest_collision(self):
        manifest = batch.Manifest(self.path)
        extra = dict(self.row, id="kernel--dense8--sparring", schedule="dense8")
        manifest.data["runs"].append(extra)
        manifest.save()
        ordered = batch.plan(manifest)
        self.assertEqual([r["timed"] for r in ordered], [False, True, True])
        self.assertEqual(ordered[0]["id"], "warmup--" + self.row["id"])
        self.assertEqual(len(batch.plan(manifest)), 3)
        self.path.write_text(self.path.read_text() + " ")
        with self.assertRaisesRegex(batch.ReviewRequired, "externally"):
            manifest.save()

    def test_nonfinite_or_boolean_timing_is_not_evidence(self):
        driver, row = self.driver()
        driver.run_case(row)
        show = json.loads(Path(row["run_show"]).read_text())
        for invalid in [True, float("nan"), float("inf"), 0, -1]:
            with self.subTest(execution_ms=invalid):
                show["execution_ms"] = invalid
                with self.assertRaisesRegex(batch.ReviewRequired, "execution_ms"):
                    batch.proof(row, show, "GPU-fixed")

    def test_preflight_failure_does_not_mark_submission_attempted(self):
        (self.root / "mode").write_text("preflight-bad-runtime")
        driver, row = self.driver()
        with self.assertRaisesRegex(batch.ReviewRequired, "preflight mismatch"):
            driver.run_case(row)
        self.assertEqual(row["status"], "pending")
        self.assertNotIn("driver_state", row)
        self.assertEqual(self.submissions(), [])
        self.assertTrue(list(Path(row["output_dir"]).glob("preflight-*.json")))
        (self.root / "mode").write_text("success")
        driver, row = self.driver()
        driver.run_case(row)
        self.assertEqual(len(self.submissions()), 1)

    def test_canonical_output_binds_manual_and_watch_without_saved(self):
        driver, row = self.driver()
        driver.run_case(row)
        show = json.loads(Path(row["run_show"]).read_text())
        row["video"] = None
        evidence = batch.proof(row, show, "GPU-fixed", {"status": "completed"})
        self.assertEqual(evidence["video"], show["output"][0]["path"])
        wrong = self.root / "wrong-video.mp4"
        wrong.write_bytes(b"wrong")
        row["video"] = str(wrong)
        with self.assertRaisesRegex(batch.ReviewRequired, "canonical output"):
            batch.proof(row, show, "GPU-fixed")
        row["video"] = evidence["video"]
        show["output"][0]["length"] += 1
        with self.assertRaisesRegex(batch.ReviewRequired, "canonical output"):
            batch.proof(row, show, "GPU-fixed")
        show.pop("output")
        with self.assertRaisesRegex(batch.ReviewRequired, "canonical video output"):
            batch.proof(row, show, "GPU-fixed")

    def test_pause_then_stop_submits_nothing(self):
        (self.root / "PAUSE").touch()
        (self.root / "STOP").touch()
        driver, row = self.driver()
        with self.assertRaises(batch.Detached):
            driver.run_case(row)
        self.assertEqual(self.calls(), [])


if __name__ == "__main__":
    unittest.main()
