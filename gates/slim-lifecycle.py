#!/usr/bin/env python3
"""Full SDXL restart/self-SIGKILL recovery through the supported public CLI.

Requires a caller-owned rental and make-slim-lifecycle.py's exact-source package.
It rents/ends nothing, signals no foreign PID and never treats cancellation as a kill.
Raw run records preserve initial residency, copies, fills and readonly-reuse evidence.
"""

import argparse
import contextlib
import hashlib
import json
import re
import subprocess
import time
import uuid as identifiers
from pathlib import Path


def documents(text):
    try:
        return [json.loads(text)]
    except ValueError:
        result = []
        for line in text.splitlines():
            with contextlib.suppress(ValueError):
                result.append(json.loads(line))
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
        rows.extend(gpus)
        memories.append(stats)
    assert rows, "no actual GPU inference"
    assert all(row.get("pid", 0) > 0 and row.get("uuid") for row in rows), "missing PID/UUID"
    return {
        "pids": sorted({row["pid"] for row in rows}),
        "uuids": sorted({uuid(row["uuid"]) for row in rows}),
        "h2d_bytes": sum(row.get("paged_in", 0) for row in memories),
        "initial_pages": sum(
            len(proof.get("initial", [])) for row in memories for proof in row["proofs"]
        ),
    }


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
    parser.add_argument("--runtime-wheel", required=True)
    parser.add_argument("--tensorfs-wheel", required=True)
    parser.add_argument("--tensord", required=True)
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

    before = machine("before-restart")
    status = before["status"]
    assert status.get("live_runs", 0) == 0, "finish the model matrix before lifecycle proof"
    assert status["runtime"] == "0.30.0" and status["tensorfs"] == "0.13.0"
    started = status["started_at_ms"]
    software = tuple(status[key] for key in ("runtime", "tensorfs", "commit"))
    command(
        "restart",
        [
            "machine",
            "update",
            args.on,
            "--tensord",
            args.tensord,
            "--runtime-wheel",
            args.runtime_wheel,
            "--tensorfs-wheel",
            args.tensorfs_wheel,
            "--json",
        ],
    )
    deadline = time.monotonic() + 180
    while True:
        after = machine("after-restart")
        current = after.get("status") or {}
        if current.get("phase") == "ready" and current.get("started_at_ms", 0) > started:
            assert tuple(current[key] for key in ("runtime", "tensorfs", "commit")) == software
            break
        assert time.monotonic() < deadline, "public machine started_at_ms did not advance"
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
        logs = command(
            name + "-tensord", ["machine", "logs", args.on, "tensord", "-n", "1000"], check=False
        )
        command(name + "-memory", ["machine", "logs", args.on, "memory", "-n", "1000"], check=False)
        events = markers([watched.stdout, watched.stderr, *strings(run), logs.stdout, logs.stderr])
        events = [
            event
            for event in events
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
        facts = gpu_facts(run)
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
        }

    fresh = case("fresh-after-restart", False)
    killed = case("kill-after-first-step", True)
    first = case("recovery-first", False)
    second = case("recovery-second", False)
    assert fresh["uuids"] == first["uuids"] == second["uuids"] == [killed["uuid"]], (
        "physical GPU changed"
    )
    assert killed["pid"] not in first["pids"], "killed executor was reused"
    assert first["pids"] == second["pids"], "the recovered executor did not serve the next request"
    assert fresh["digest"] == first["digest"] == second["digest"], "decoded model pixels differ"
    assert fresh["outputs"] == first["outputs"] == second["outputs"], (
        "landed model output bytes differ"
    )
    (args.output / "lifecycle.json").write_text(
        json.dumps(
            {
                "source": origin,
                "machine": args.on,
                "before_started_at_ms": started,
                "after_started_at_ms": current["started_at_ms"],
                "fresh": fresh,
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
