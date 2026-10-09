"""Run the owned H3 matrix through ordinary Cozy CLI; never retry a submission."""

from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import math
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path


class ReviewRequired(RuntimeError):
    pass


class Detached(RuntimeError):
    pass


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    content = (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()
    with temp.open("wb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)
    fd = os.open(path.parent, os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    return content


class Manifest:
    def __init__(self, path):
        self.path = Path(path).resolve()
        self.raw = self.path.read_bytes()
        self.data = json.loads(self.raw)

    def save(self):
        if self.path.read_bytes() != self.raw:
            raise ReviewRequired("Manifest changed externally; stop and reconcile before resuming")
        self.raw = atomic(self.path, self.data)


def plan(manifest):
    """Preserve explicit/manual warmups; add missing mixed-scene warmups only."""
    data = manifest.data
    runs = data["runs"]
    warmups = data.setdefault("warmups", [])
    result = []
    for kernel in dict.fromkeys(r["kernel"] for r in runs):
        group = [r for r in runs if r["kernel"] == kernel]
        for source in (r for r in group if r["schedule"] == "dense4-sol4"):
            matches = [
                r
                for r in warmups
                if (r["kernel"], r["scene"], r["schedule"])
                == (kernel, source["scene"], "dense4-sol4")
            ]
            if len(matches) > 1:
                raise ReviewRequired("Duplicate warmup for " + source["id"])
            if matches:
                row = matches[0]
            else:
                row = copy.deepcopy(source)
                row["id"] = "warmup--" + source["id"]
                directory = manifest.path.parent / "warmups" / row["id"]
                row.update(
                    output_dir=str(directory),
                    run_show=str(directory / "run-show.json"),
                    status="pending",
                    video=None,
                    timed=False,
                    timing_note="Untimed full mixed-schedule warmup; excluded from comparison",
                )
                for field in (
                    "request_id",
                    "run_number",
                    "execution_ms",
                    "driver_evidence",
                    "driver_state",
                    "idempotency_key",
                    "result",
                    "stdout",
                    "stderr",
                ):
                    row.pop(field, None)
                warmups.append(row)
            row["timed"] = False
            result.append(row)
        for row in group:
            row["timed"] = True
            result.append(row)
    ids = [r["id"] for r in result]
    if len(ids) != len(set(ids)) or any(not re.fullmatch(r"[A-Za-z0-9_.-]{1,140}", x) for x in ids):
        raise ReviewRequired("Unsafe or duplicate case id")
    manifest.save()
    return result


def request_id(*paths):
    ids = set()
    for path in paths:
        if not path or not Path(path).exists():
            continue
        for line in Path(path).read_text(errors="replace").splitlines():
            try:
                value = json.loads(line)
            except ValueError:
                continue
            if isinstance(value, dict) and value.get("request_id"):
                ids.add(str(value["request_id"]))
    if len(ids) > 1:
        raise ReviewRequired("CLI logs contain more than one top-level request id")
    return next(iter(ids), None)


def proof(row, show, gpu_uuid, result=None):
    """Only retained evidence can make a case complete; never round its timing."""
    if show.get("status") != "completed" or not show.get("execution_known"):
        raise ReviewRequired("Run has no completed, known execution time")
    elapsed = show.get("execution_ms")
    if (
        isinstance(elapsed, bool)
        or not isinstance(elapsed, (int, float))
        or not math.isfinite(elapsed)
        or elapsed <= 0
    ):
        raise ReviewRequired("Invalid precise execution_ms")
    if row.get("request_id") and show.get("request_id") != row["request_id"]:
        raise ReviewRequired("Run receipt request identity changed")
    steps = [s for s in show.get("steps", []) if s.get("name") == "denoise"]
    if len(steps) != 1 or steps[0].get("count") != row.get("expected_steps", 8):
        raise ReviewRequired("Expected all eight denoising steps")
    gpus = show.get("gpus", [])
    if len(gpus) != 1 or gpus[0].get("uuid") != gpu_uuid:
        raise ReviewRequired("GPU UUID/degree does not match matrix host")
    attention = gpus[0].get("attention", {})
    selector = "fl2va_dit=" + row["attention_selector"]
    if selector not in attention.get("observed", "").split(","):
        raise ReviewRequired("Observed attention selector does not match")
    sparse = attention.get("sol", {}).get("sparse", 0)
    if sparse != {"dense4-sol4": 192, "dense8": 0}[row["schedule"]]:
        raise ReviewRequired("Observed sparse-call count does not match schedule")
    outputs = [
        o for o in show.get("output", []) if o.get("name") == "video" and o.get("type") == "video"
    ]
    if len(outputs) != 1 or outputs[0].get("status") != "completed":
        raise ReviewRequired("Run show has no unique completed canonical video output")
    output = outputs[0]
    expected = str(output.get("sha256", "")).removeprefix("sha256:")
    length = output.get("length")
    if not re.fullmatch(r"[0-9a-f]{64}", expected) or type(length) is not int or length <= 0:
        raise ReviewRequired("Canonical video output lacks a valid SHA256/length")
    saved = [s for s in (result or {}).get("saved", []) if s.get("output") == "video"]
    if len(saved) > 1:
        raise ReviewRequired("Ambiguous video outputs")
    if saved and (
        saved[0].get("digest", "").removeprefix("sha256:") != expected
        or saved[0].get("bytes") != length
    ):
        raise ReviewRequired("CLI export receipt differs from canonical video output")
    candidates = [saved[0].get("path") if saved else None, row.get("video"), output.get("path")]
    video = next((Path(path) for path in candidates if path and Path(path).is_file()), None)
    if video is None:
        raise ReviewRequired("Completed run has no retained local video; recover with watch")
    digest = sha(video)
    if digest != expected or video.stat().st_size != length:
        raise ReviewRequired("Video bytes/hash differ from canonical output receipt")
    previous = row.get("driver_evidence", {})
    if previous.get("video_sha256") and previous["video_sha256"] != digest:
        raise ReviewRequired("Previously verified video changed")
    return {
        "request_id": show["request_id"],
        "run_number": show.get("number"),
        "execution_ms": show["execution_ms"],
        "video": str(video.resolve()),
        "video_sha256": digest,
        "video_bytes": video.stat().st_size,
        "gpu_uuid": gpu_uuid,
        "selector": row["attention_selector"],
        "denoise_steps": steps[0]["count"],
        "sparse_calls": sparse,
    }


class Driver:
    def __init__(self, manifest, cozy="cozy", poll=1.0, observer_seconds=7200):
        self.manifest = manifest
        self.cozy = cozy
        self.poll = poll
        self.observer_seconds = observer_seconds
        self.pause = manifest.path.parent / "PAUSE"
        self.stop = manifest.path.parent / "STOP"

    def boundary(self):
        while self.pause.exists():
            if self.stop.exists():
                raise Detached("STOP present; no new request submitted")
            time.sleep(self.poll)
        if self.stop.exists():
            raise Detached("STOP present; no new request submitted")

    def observed_id(self, row, stdout, stderr):
        found = request_id(stderr, stdout)
        if found and row.get("request_id") != found:
            if row.get("request_id"):
                raise ReviewRequired("Request id changed in CLI output")
            row["request_id"] = found
            self.manifest.save()

    def command(self, row, args, kind):
        out = Path(row["output_dir"])
        out.mkdir(parents=True, exist_ok=True)
        seq = 1
        while (out / f"{kind}-{seq}.command.json").exists():
            seq += 1
        prefix = out / f"{kind}-{seq}"
        stdout = prefix.with_suffix(".stdout.json")
        stderr = prefix.with_suffix(".stderr.jsonl")
        atomic(prefix.with_suffix(".command.json"), {"argv": args, "started_unix_s": time.time()})
        # This durable marker is written BEFORE Popen: any uncertainty requires recovery.
        row.update(stdout=str(stdout), stderr=str(stderr), driver_state=kind)
        if kind == "submit":
            row["status"] = "submitting"
        self.manifest.save()
        process = None
        began = time.monotonic()
        try:
            with stdout.open("x") as so, stderr.open("x") as se:
                process = subprocess.Popen(
                    args, stdin=subprocess.DEVNULL, stdout=so, stderr=se, start_new_session=True
                )
                atomic(
                    prefix.with_suffix(".process.json"),
                    {"pid": process.pid, "argv": args, "started_unix_s": time.time()},
                )
                while process.poll() is None:
                    self.observed_id(row, stdout, stderr)
                    if self.stop.exists() or time.monotonic() - began > self.observer_seconds:
                        raise Detached("Observer detached; accepted work was not canceled")
                    time.sleep(self.poll)
                self.observed_id(row, stdout, stderr)
                atomic(
                    prefix.with_suffix(".exit.json"),
                    {"returncode": process.returncode, "finished_unix_s": time.time()},
                )
        except (KeyboardInterrupt, Detached, ReviewRequired, OSError) as error:
            if process is not None and process.poll() is None:
                process.send_signal(signal.SIGINT)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                self.observed_id(row, stdout, stderr)
            if isinstance(error, (KeyboardInterrupt, Detached)):
                raise Detached(
                    "Detached from owned CLI observer; use show/watch to recover"
                ) from None
            raise
        if process.returncode:
            raise ReviewRequired(f"CLI {kind} failed ({process.returncode}); inspect {stderr}")
        try:
            result = json.loads(stdout.read_text())
        except ValueError as exc:
            raise ReviewRequired(f"CLI {kind} returned no complete JSON; inspect {stdout}") from exc
        return result

    def show(self, row):
        ref = row.get("request_id") or row.get("run_number")
        if not ref:
            raise ReviewRequired("Submission may have been accepted, but no request id is known")
        result = self.command(row, [self.cozy, "run", "show", str(ref), "--json"], "show")
        if not result.get("request_id"):
            raise ReviewRequired("Show returned no request id")
        if row.get("request_id") and row["request_id"] != result["request_id"]:
            raise ReviewRequired("Show returned a different request identity")
        row["request_id"] = result["request_id"]
        atomic(Path(row["run_show"]), result)
        self.manifest.save()
        return result

    def finish(self, row, show, result=None):
        evidence = proof(row, show, self.manifest.data["gpu_uuid"], result)
        evidence["run_show_sha256"] = sha(row["run_show"])
        row.update(
            status="completed",
            driver_state="verified",
            request_id=evidence["request_id"],
            run_number=evidence["run_number"],
            video=evidence["video"],
            execution_ms=evidence["execution_ms"],
            driver_evidence=evidence,
        )
        self.manifest.save()

    def preflight(self, row):
        """Read-only evidence, kept separate from submission intent/recovery markers."""
        data = self.manifest.data
        expected = {
            "rental_id": data["rental_id"],
            "runtime_version": data["runtime_wheel"]["version"],
            "tensorfs_version": data["tensorfs_version"],
            "driver": data["driver"],
        }
        args = [
            self.cozy,
            "rental",
            "show",
            data["rental"],
            "--json",
            "--fields=machine,rental_id,state,running,queued,runtime_version,tensorfs_version,devices,accelerator_count",
        ]
        out = Path(row["output_dir"])
        out.mkdir(parents=True, exist_ok=True)
        receipt = out / f"preflight-{time.time_ns()}.json"
        try:
            result = subprocess.run(
                args, stdin=subprocess.DEVNULL, text=True, capture_output=True, timeout=30
            )
        except subprocess.TimeoutExpired as error:
            atomic(
                receipt,
                {"argv": args, "error": "read-only preflight timed out", "expected": expected},
            )
            raise ReviewRequired("Rental preflight timed out; no submission attempted") from error
        atomic(
            receipt,
            {
                "argv": args,
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "expected": expected,
                "observed_unix_s": time.time(),
            },
        )
        if result.returncode:
            raise ReviewRequired(
                f"Rental preflight failed; inspect {receipt}; no submission attempted"
            )
        actual = json.loads(result.stdout)
        mismatch = [
            key
            for key in ("rental_id", "runtime_version", "tensorfs_version")
            if actual.get(key) != expected[key]
        ]
        devices = actual.get("devices", [])
        if (
            actual.get("state") != "ready"
            or actual.get("accelerator_count") != 1
            or len(devices) != 1
        ):
            mismatch.append("ready single-GPU rental")
        if not devices or any(device.get("driver") != expected["driver"] for device in devices):
            mismatch.append("driver")
        if actual.get("running") != 0 or actual.get("queued") != 0:
            mismatch.append("idle rental")
        if mismatch:
            raise ReviewRequired(
                f"Rental preflight mismatch ({', '.join(mismatch)}); "
                f"inspect {receipt}; no submission attempted"
            )

    def run_case(self, row):
        self.boundary()
        input_hash = sha(row["input"])
        if row.get("input_sha256") and row["input_sha256"] != input_hash:
            raise ReviewRequired("Input differs from pinned manifest hash")
        identity = {
            "target": self.manifest.data["target"],
            "hub_url": self.manifest.data["hub_url"],
            "input_sha256": input_hash,
            "selector": row["attention_selector"],
            "gpu_uuid": self.manifest.data["gpu_uuid"],
            "rental": self.manifest.data["rental"],
            "id": row["id"],
        }
        if row.get("driver_identity") and row["driver_identity"] != identity:
            raise ReviewRequired("Input/target/selector/host changed after case began")
        row["driver_identity"] = identity
        key = hashlib.sha256(
            (str(self.manifest.path) + json.dumps(identity, sort_keys=True)).encode()
        ).hexdigest()
        row.setdefault("idempotency_key", "h3-matrix-" + key[:40])
        self.manifest.save()
        if row.get("status") in {"failed", "canceled", "blocked", "pending_kernel_gate"}:
            raise ReviewRequired("Case is failed/canceled/blocked; owner review required")
        if row.get("status") == "completed" and Path(row["run_show"]).exists():
            previous = row.get("driver_evidence", {}).get("run_show_sha256")
            if previous and previous != sha(row["run_show"]):
                raise ReviewRequired("Previously verified run receipt changed")
            result_path = row.get("result") or row.get("result_path")
            result = (
                json.loads(Path(result_path).read_text())
                if result_path and Path(result_path).is_file()
                else None
            )
            self.finish(row, json.loads(Path(row["run_show"]).read_text()), result)
            return
        known = row.get("request_id") or row.get("run_number")
        attempted = row.get("driver_state") or row.get("status") not in {
            "pending",
            "pending_kernel_gate",
            None,
        }
        if not known:
            found = request_id(
                row.get("stderr"), row.get("stdout"), row.get("events_path"), row.get("result_path")
            )
            if found:
                row["request_id"] = known = found
                self.manifest.save()
        if attempted and not known:
            raise ReviewRequired(
                "Uncertain prior submission: recover its request id; never resubmit"
            )
        if known:
            show = self.show(row)
            if show["status"] in {"failed", "canceled", "blocked"}:
                raise ReviewRequired("Recovered request failed/canceled/blocked")
            if show["status"] == "completed" and row.get("video"):
                self.finish(row, show)
                return
            if show["status"] not in {"completed", "queued", "preparing", "in_progress", "running"}:
                raise ReviewRequired("Unknown recovered status: " + str(show["status"]))
            result = self.command(
                row, [self.cozy, "run", "watch", row["request_id"], "--json"], "watch"
            )
        else:
            self.preflight(row)
            self.boundary()
            out = Path(row["output_dir"])
            out.mkdir(parents=True, exist_ok=True)
            atomic(out / "submission-identity.json", identity)
            args = [
                self.cozy,
                "run",
                self.manifest.data["target"],
                "--tensorhub=" + self.manifest.data["hub_url"],
                "--rental=" + self.manifest.data["rental"],
                "--input=" + row["input"],
                "--out=" + str(out),
                "--idempotency-key=" + row["idempotency_key"],
                "--attention-kernel=fl2va_dit=" + row["attention_selector"],
                "--await",
                "--json",
            ]
            result = self.command(row, args, "submit")
        atomic(Path(row["output_dir"]) / "result.json", result)
        row["result"] = str(Path(row["output_dir"]) / "result.json")
        self.manifest.save()
        self.finish(row, self.show(row), result)


def detach_signal(*_):
    raise KeyboardInterrupt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--cozy", default="cozy")
    parser.add_argument("--observer-seconds", type=float, default=7200)
    parser.add_argument("--poll-seconds", type=float, default=1)
    parser.add_argument("--prepare-only", action="store_true", help="Create plan, never invoke CLI")
    args = parser.parse_args()
    if args.poll_seconds <= 0 or args.observer_seconds <= 0:
        parser.error("Observer and polling intervals must be positive")
    signal.signal(signal.SIGTERM, detach_signal)
    with args.manifest.with_suffix(".batch.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = Manifest(args.manifest)
        rows = plan(manifest)
        if args.prepare_only:
            print(json.dumps({"order": [r["id"] for r in rows]}))
            return 0
        driver = Driver(manifest, args.cozy, args.poll_seconds, args.observer_seconds)
        for row in rows:
            print(
                json.dumps({"case": row["id"], "timed": row["timed"], "status": row["status"]}),
                flush=True,
            )
            driver.run_case(row)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ReviewRequired, Detached, OSError, ValueError, KeyError) as error:
        print(f"Batch stopped: {error}", file=sys.stderr)
        sys.exit(2)
