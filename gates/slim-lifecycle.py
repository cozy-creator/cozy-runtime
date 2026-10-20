#!/usr/bin/env python3
"""Full SDXL executor restart/self-SIGKILL recovery through the public CLI.

Requires a caller-owned rental and make-slim-lifecycle.py's exact-source package.
It rents/ends nothing, signals no foreign PID and never treats cancellation as a kill.
Raw run records preserve initial residency, copies, fills and readonly-reuse evidence.
The caller installs its SDK beforehand; this driver never replaces TensorD. Public
measurements expose final orphan bytes and Torch leftovers, but not native pin counts
or TensorD's PID. The receipt names those unavailable metrics explicitly.
"""

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time
import uuid as identifiers


def documents(text):
    try:
        return [json.loads(text)]
    except ValueError:
        result = []
        for line in text.splitlines():
            try:
                result.append(json.loads(line))
            except ValueError:
                pass
        return result


def strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from strings(child)


def markers(texts):
    found = []
    for text in texts:
        expanded = [text, *(s for doc in documents(text) for s in strings(doc))]
        for value in expanded:
            for line in value.splitlines():
                if "SLIM_LIFECYCLE " not in line:
                    continue
                try:
                    event, _ = json.JSONDecoder().raw_decode(line.split("SLIM_LIFECYCLE ", 1)[1])
                except ValueError:
                    continue
                if isinstance(event, dict) and event not in found:
                    found.append(event)
    return found


def gpu_facts(run):
    rows, memories = [], []
    for call in run.get("calls") or [run]:
        execution = (call.get("measurements") or {}).get("execution") or {}
        gpus = [
            gpu
            for gpu in execution.get("gpus", [])
            if gpu.get("gpu", -1) >= 0
            and gpu.get("start_us", 0) > 0
            and gpu.get("end_us", 0) >= gpu["start_us"]
        ]
        if not gpus:
            continue
        assert not call.get("memoized"), "memoized inference is not execution evidence"
        assert len(gpus) == 1, "lifecycle requires exactly one assigned GPU"
        memory = (call.get("measurements") or {}).get("memory") or {}
        stats = memory.get("stats") or {}
        assert type(stats.get("paged_in")) is int and stats["paged_in"] >= 0, "missing H2D count"
        assert isinstance(stats.get("copies"), list), "missing owned IO rows"
        assert stats.get("leftover") == 0, "successful request retained Torch bytes"
        assert stats.get("leftover_by_gpu") == [[gpus[0]["gpu"], 0]], (
            "missing GPU leftover evidence"
        )
        assert stats.get("proofs"), "missing native request proof metadata"
        assert all(proof.get("complete") is True for proof in stats["proofs"]), (
            "incomplete native request proof"
        )
        trace = memory.get("trace") or stats.get("trace")
        assert isinstance(trace, list) and trace, "missing native placement trace"
        ends = [event for event in trace if event.get("kind") == "end"]
        assert ends and {event.get("gpu") for event in ends} == {gpus[0]["gpu"]}, (
            "missing completed GPU request end"
        )
        assert all(event.get("orphan_bytes") == 0 for event in ends), (
            "completed request retained orphan bytes"
        )
        rows.extend(gpus)
        memories.append(stats)
    assert rows, "no actual GPU inference"
    assert all(row.get("pid", 0) > 0 and row.get("uuid") for row in rows), "missing PID/UUID"
    return {
        "pids": sorted({row["pid"] for row in rows}),
        "uuids": sorted({uuid(row["uuid"]) for row in rows}),
        "h2d_bytes": sum(row.get("paged_in", 0) for row in memories),
        "disk_bytes": sum(
            copy["bytes"] for row in memories for copy in row["copies"] if copy["kind"] == "disk"
        ),
        "end_orphan_bytes": 0,
        "torch_leftover_bytes": 0,
        "initial_pages": sum(
            len(proof.get("initial", [])) for row in memories for proof in row["proofs"]
        ),
    }


def service_identity(value):
    machine = value.get("machine", value)
    status = machine.get("status") or {}
    assert status.get("phase") == "ready", "TensorD must be ready"
    assert status.get("live_runs", 0) == 0, "finish other runs before lifecycle proof"
    identity = {
        key: status.get(key)
        for key in ("worker_id", "version", "commit", "runtime", "tensorfs", "started_at_ms")
    }
    assert type(identity["started_at_ms"]) is int and identity["started_at_ms"] > 0
    assert all(
        isinstance(identity[key], str) and identity[key]
        for key in identity
        if key != "started_at_ms"
    )
    return identity


def recovery_path(facts):
    if facts["initial_pages"]:
        return "kept-readonly-pages-present-at-request-begin"
    if facts["h2d_bytes"] and not facts["disk_bytes"]:
        return "diskless-h2d-refill"
    if facts["h2d_bytes"] and facts["disk_bytes"]:
        return "verified-disk-refill"
    return "no-weight-transfer-reported"


def oom_observation(before, after):
    def counts(snapshot):
        return {
            group["path"]: group["events"]["oom_kill"]
            for group in snapshot.get("cgroups", [])
            if type(group.get("events", {}).get("oom_kill")) is int
        }

    first, last = counts(before), counts(after)
    changes = {path: last[path] - first[path] for path in first.keys() & last.keys()}
    assert all(delta == 0 for delta in changes.values()), f"cgroup oom_kill changed: {changes}"
    return {"available": bool(changes), "oom_kill_deltas": changes}


def uuid(value):
    normalized = str(value).lower().removeprefix("gpu-").replace("-", "")
    assert re.fullmatch(r"[0-9a-f]{32}", normalized), f"invalid physical GPU UUID: {value}"
    return normalized


def model_overrides(values):
    result = []
    for value in values:
        slot, separator, model = value.partition("=")
        if slot.startswith("model."):
            slot = slot[len("model."):]
        if not separator or not slot.strip() or not model.strip():
            raise ValueError("--model requires a nonempty SLOT=MODEL override")
        result.append(f"model.{slot}={model}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--on", required=True)
    parser.add_argument("--tensorhub", default="tensorhub")
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", action="append", default=[])
    parser.add_argument("--bits", type=int, choices=(8, 16), default=8)
    args = parser.parse_args()
    models = model_overrides(args.model)
    assert args.on.startswith("rent-"), "use the explicit caller-owned rental alias, never local"
    assert not args.output.exists(), "use a new output directory so earlier evidence is preserved"
    origin = json.loads((args.package / "lifecycle-source.json").read_text())
    assert origin.get("proof_only") and re.fullmatch(r"[0-9a-f]{40}", origin["packages_commit"])
    payload = json.loads(args.input.read_text())
    assert type(payload.get("seed")) is int, "deterministic comparison requires an explicit seed"
    args.output.mkdir(parents=True)
    hub = ["--tensorhub", args.tensorhub]

    def command(label, words, check=True):
        result = subprocess.run(["cozy", *words, *hub], text=True, capture_output=True)
        (args.output / (label + ".out")).write_text(result.stdout)
        (args.output / (label + ".err")).write_text(result.stderr)
        if check and result.returncode:
            raise RuntimeError(f"{label} failed: {result.stderr[-2000:]}")
        return result

    def machine(label):
        return json.loads(command(label, ["machine", "show", args.on, "--json"]).stdout)

    identity = service_identity(machine("before-lifecycle"))
    assert identity["runtime"] == "0.30.0" and identity["tensorfs"] == "0.13.0"

    def quiescent(label):
        deadline = time.monotonic() + 180
        while True:
            observed = machine(label)
            status = observed.get("machine", observed).get("status") or {}
            if status.get("phase") == "ready" and status.get("live_runs", 0) == 0:
                assert service_identity(observed) == identity, (
                    "TensorD changed during executor recovery"
                )
                return
            assert time.monotonic() < deadline, "machine did not settle after the request"
            time.sleep(2)

    def case(name, kill):
        folder = args.output / name
        folder.mkdir()
        token = identifiers.uuid4().hex
        request = folder / "input.json"
        request.write_text(json.dumps({**payload, "lifecycle_kill": kill, "lifecycle_case": token}))
        words = [
            "run",
            str(args.package.resolve() / "generate"),
            "--input",
            str(request.resolve()),
            "--on",
            args.on,
            "--gpus",
            "1",
            "--bits",
            str(args.bits),
            "--out",
            str((folder / "assets").resolve()),
            "--detach",
            "--json",
        ]
        words.extend(models)
        submitted = command(name + "-submit", words)
        number = next(
            doc["number"]
            for doc in reversed(documents(submitted.stdout))
            if isinstance(doc, dict) and type(doc.get("number")) is int
        )
        watched = command(name + "-watch", ["run", "watch", str(number), "--json"], check=False)
        result = command(name + "-show", ["run", "show", str(number), "--json"])
        run = json.loads(result.stdout)
        (folder / "run.json").write_text(json.dumps(run, indent=2) + "\n")
        quiescent(name + "-quiescent")
        logs = command(
            name + "-tensord", ["machine", "logs", args.on, "tensord", "-n", "1000"], check=False
        )
        command(name + "-memory", ["machine", "logs", args.on, "memory", "-n", "1000"], check=False)
        all_events = markers(
            [watched.stdout, watched.stderr, *strings(run), logs.stdout, logs.stderr]
        )
        events = [
            event
            for event in all_events
            if event.get("case") == token
            and event.get("kill") is kill
            and event.get("cuda_completed") is True
            and event.get("step") == 0
            and type(event.get("pid")) is int
            and event["pid"] > 0
            and event.get("uuid")
        ]
        assert events, f"{name}: first completed CUDA step marker was not observed"
        event = events[-1]
        if kill:
            assert run["status"] == "failed", "self-kill must fail, never complete or cancel"
            reasons = [run.get("reason"), *(call.get("reason") for call in run.get("calls", []))]
            reason = " ".join(str(value) for value in reasons if value)
            assert "cancel" not in reason.lower(), "cancellation is not a kill-9 proof"
            assert re.search(
                r"EOF|executor.*(?:ended|died|lost)|channel.*(?:closed|lost)", reason, re.I
            ), reason
            assert not re.search(r"SIGSEGV|SIGABRT|signal\s*:\s*(?:6|11)\b", reason), reason
            return {
                "run": number,
                "pid": event["pid"],
                "uuid": uuid(event["uuid"]),
                "evidence": "completed CUDA step, self-SIGKILL marker, failed executor/channel",
            }
        assert run["status"] == "completed", run.get("reason")
        completed = [
            row
            for row in all_events
            if row.get("case") == token
            and row.get("complete") is True
            and row.get("step") == -1
            and row.get("pid") == event["pid"]
            and row.get("cuda_completed") is True
        ]
        assert completed, f"{name}: completed model output marker was not observed"
        host = oom_observation(event.get("host", {}), completed[-1].get("host", {}))
        facts = gpu_facts(run)
        facts["recovery_path"] = recovery_path(facts)
        assert facts["uuids"] == [uuid(event["uuid"])]
        assert event["pid"] in facts["pids"]
        expected = {item["sha256"] for item in run.get("output", [])}
        assert expected, "completed model returned no outputs"
        deadline = time.monotonic() + 180
        while True:
            actual = {
                "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
                for path in (folder / "assets").rglob("*")
                if path.is_file() and path.suffix != ".part"
            }
            if expected <= actual:
                break
            assert time.monotonic() < deadline, "outputs did not land"
            time.sleep(2)
        return {
            "run": number,
            **facts,
            "digest": run["result"]["digest"],
            "outputs": sorted(expected),
            "host_oom_observation": host,
            "host_at_output": completed[-1].get("host", {}),
        }

    prime = case("prime", False)
    warm = case("warm-before-kill", False)
    killed = case("kill-after-first-step", True)
    first = case("recovery-first", False)
    second = case("recovery-second", False)
    assert (
        prime["uuids"] == warm["uuids"] == first["uuids"] == second["uuids"] == [killed["uuid"]]
    ), "physical GPU changed"
    assert prime["pids"] == warm["pids"] == [killed["pid"]], (
        "the self-kill did not end the actual warm model executor"
    )
    assert killed["pid"] not in first["pids"], "killed executor was reused"
    assert first["pids"] == second["pids"], "the recovered executor did not serve the next request"
    assert prime["digest"] == warm["digest"] == first["digest"] == second["digest"], (
        "decoded model pixels differ"
    )
    assert prime["outputs"] == warm["outputs"] == first["outputs"] == second["outputs"], (
        "landed model output bytes differ"
    )
    (args.output / "lifecycle.json").write_text(
        json.dumps(
            {
                "source": origin,
                "machine": args.on,
                "tensord_service_identity": identity,
                "unavailable_public_metrics": [
                    "tensord_process_pid",
                    "quiescent_native_pin_counts",
                ],
                "prime": prime,
                "warm": warm,
                "killed": killed,
                "recovery_first": first,
                "recovery_second": second,
            },
            indent=2,
        )
        + "\n"
    )
    print(args.output / "lifecycle.json")


if __name__ == "__main__":
    main()
