"""Loopback Comfy client. No Torch imports, CUDA calls, server control or resubmission."""

from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.request
import uuid
from collections.abc import Callable, Iterator
from contextlib import suppress
from pathlib import Path
from typing import Any

import websocket

MAX_ARTIFACT_BYTES = 64 << 30


def write_json(path: Path, value: Any) -> None:
    temp = path.with_suffix(path.suffix + ".pending")
    with temp.open("w") as output:
        json.dump(value, output, indent=2)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    temp.replace(path)


def request_json(url: str, timeout: float, body: Any = None) -> Any:
    request = urllib.request.Request(
        url,
        None if body is None else json.dumps(body).encode(),
        {} if body is None else {"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def output_files(value: Any) -> Iterator[dict[str, Any]]:
    if isinstance(value, dict):
        if isinstance(value.get("filename"), str) and value.get("type") == "output":
            yield value
        else:
            for child in value.values():
                yield from output_files(child)
    elif isinstance(value, list):
        for child in value:
            yield from output_files(child)


def bounded_copy(source: Path, target: Path, remaining: int) -> int:
    """Return the retained-byte change, crediting an observer's prior saved copy."""
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"Artifact is not a regular file: {source}")
    previous = target.stat().st_size if target.exists() else 0
    remaining += previous
    if remaining < 0:
        raise ValueError("Artifacts exceed the declared64GiB bound")
    target.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with source.open("rb") as reader, target.open("wb") as writer:
        while chunk := reader.read(min(1 << 20, remaining - total + 1)):
            total += len(chunk)
            if total > remaining:
                raise ValueError("Artifacts exceed the declared64GiB bound")
            writer.write(chunk)
    return total - previous


def retain_outputs(history: dict[str, Any], output_root: Path, artifacts: Path) -> None:
    root = output_root.resolve()
    seen: set[Path] = set()
    remaining = MAX_ARTIFACT_BYTES - sum(
        path.stat().st_size for path in artifacts.rglob("*") if path.is_file()
    )
    for item in output_files(history.get("outputs", {})):
        source = (root / item.get("subfolder", "") / item["filename"]).resolve()
        if not source.is_relative_to(root):
            raise ValueError("Comfy output path escapes the configured output directory")
        if source not in seen:
            remaining -= bounded_copy(
                source, artifacts / "outputs" / source.relative_to(root), remaining
            )
            seen.add(source)


def observe(
    graph_json: str,
    *,
    port: int,
    timeout_s: int,
    trace: bool,
    output_root: str,
    trace_root: str,
    directory: Path,
    deadline: float,
    check_cancelled: Callable[[], None],
    expected_steps: int = 8,
) -> dict[str, Any]:
    if type(expected_steps) is not int or expected_steps not in (8, 30):
        raise ValueError("H3 trace expects exactly8 Turbo or30 regular denoising steps")
    graph = json.loads(graph_json)
    if not isinstance(graph, dict) or not graph:
        raise ValueError("graph_json must encode a nonempty Comfy API graph")
    directory.mkdir(parents=True, exist_ok=True)
    artifacts = directory / "artifacts"
    artifacts.mkdir(exist_ok=True)
    fingerprint = hashlib.sha256(
        json.dumps(
            {
                "graph": graph,
                "port": port,
                "trace": trace,
                "output_root": output_root,
                "trace_root": trace_root,
                "expected_steps": expected_steps,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    state_file = directory / "state.json"
    if state_file.exists():
        state = json.loads(state_file.read_text())
        if state["fingerprint"] != fingerprint:
            raise ValueError("A resumed observer cannot change its submitted graph or server")
    else:
        state = {
            "fingerprint": fingerprint,
            "prompt_id": str(uuid.uuid4()),
            "client_id": str(uuid.uuid4()),
            "post_attempted": False,
        }
        write_json(state_file, state)
    prompt_id = state["prompt_id"]
    address = f"127.0.0.1:{port}"
    base = "http://" + address
    stop = min(time.monotonic() + timeout_s, deadline - 1)
    started = time.monotonic()
    ws = None
    history = None
    detail = ""
    status = "timed_out"
    request = {
        "prompt": graph,
        "client_id": state["client_id"],
        "prompt_id": prompt_id,
        "extra_data": {"h3_full_trace": trace},
    }
    write_json(artifacts / "request.json", request)
    try:
        # History polling still observes real work if the event socket is unavailable.
        with suppress(OSError, websocket.WebSocketException):
            ws = websocket.create_connection(
                f"ws://{address}/ws?clientId={state['client_id']}",
                timeout=1,
            )
        if not state["post_attempted"]:
            queue = request_json(base + "/queue", min(5, timeout_s))
            if queue["queue_running"] or queue["queue_pending"]:
                raise ValueError("Benchmark submission requires an otherwise idle Comfy queue")
            # Record uncertainty before POST: a dropped response must never submit twice.
            state["post_attempted"] = True
            write_json(state_file, state)
            try:
                accepted = request_json(base + "/prompt", min(10, timeout_s), request)
                if accepted.get("prompt_id") != prompt_id:
                    raise ValueError("Comfy did not preserve the caller-provided prompt UUID")
                write_json(artifacts / "accepted.json", accepted)
            except urllib.error.HTTPError as exc:
                write_json(
                    artifacts / "submission-error.json",
                    {
                        "status": exc.code,
                        "body": exc.read().decode(errors="replace"),
                    },
                )
                return finish(
                    artifacts, prompt_id, "failed", "Comfy rejected the graph", None, started
                )
            except (OSError, ValueError) as exc:
                detail = f"Submission acknowledgement uncertain: {type(exc).__name__}: {exc}"
        next_poll = 0.0
        with (artifacts / "events.jsonl").open("a", buffering=1) as events:
            while time.monotonic() < stop:
                check_cancelled()
                if time.monotonic() >= next_poll:
                    try:
                        found = request_json(
                            base + f"/history/{prompt_id}",
                            min(3, max(0.1, stop - time.monotonic())),
                        )
                        history = found.get(prompt_id)
                        if history is not None:
                            write_json(artifacts / "history.json", found)
                            status = (
                                "success"
                                if history.get("status", {}).get("status_str") == "success"
                                else "failed"
                            )
                            break
                    except (OSError, ValueError):
                        pass
                    next_poll = time.monotonic() + 5
                if ws is None:
                    time.sleep(min(0.25, max(0, stop - time.monotonic())))
                    continue
                try:
                    raw = ws.recv()
                    if not isinstance(raw, str) or not raw:
                        continue
                    event = json.loads(raw)
                    print(
                        json.dumps(
                            {
                                "unix_s": time.time(),
                                "elapsed_s": time.monotonic() - started,
                                "event": event,
                            }
                        ),
                        file=events,
                    )
                    data = event.get("data", {})
                    if data.get("prompt_id") == prompt_id and (
                        event.get("type")
                        in {"execution_success", "execution_error", "execution_interrupted"}
                        or (event.get("type") == "executing" and data.get("node") is None)
                    ):
                        next_poll = 0
                except websocket.WebSocketTimeoutException:
                    continue
                except (OSError, websocket.WebSocketException, ValueError):
                    ws.close()
                    ws = None  # Read-only history polling resumes; never POST again.
        if history is not None:
            retain_outputs(history, Path(output_root), artifacts)
        if status == "success" and trace:
            source = Path(trace_root) / prompt_id
            while not (source / "summary.json").exists() and time.monotonic() < stop:
                check_cancelled()
                time.sleep(0.25)
            if not (source / "summary.json").exists():
                status, detail = (
                    "timed_out",
                    "Render completed; full trace export did not finish before deadline",
                )
            else:
                summary = json.loads((source / "summary.json").read_text())
                dense = 4 if expected_steps == 8 else 10
                wanted = {
                    f"h3.step.{step}.{'dense' if step < dense else 'sparse_eligible'}"
                    for step in range(expected_steps)
                }
                ranges = [
                    item
                    for item in summary.get("ranges", [])
                    if isinstance(item, dict) and str(item.get("label", "")).startswith("h3.step.")
                ]
                if (
                    type(summary.get("expected_steps")) is not int
                    or summary["expected_steps"] != expected_steps
                    or type(summary.get("steps_observed")) is not int
                    or summary["steps_observed"] != expected_steps
                    or len(ranges) != expected_steps
                    or {item["label"] for item in ranges} != wanted
                    or any(
                        type(item.get("start_unix_ns")) is not int
                        or type(item.get("end_unix_ns")) is not int
                        or item["end_unix_ns"] <= item["start_unix_ns"]
                        for item in ranges
                    )
                    or not summary.get("segments")
                ):
                    status, detail = (
                        "failed",
                        f"Incomplete trace: expected {expected_steps} complete step ranges",
                    )
                remaining = MAX_ARTIFACT_BYTES - sum(
                    path.stat().st_size for path in artifacts.rglob("*") if path.is_file()
                )
                for path in sorted(source.rglob("*")):
                    if path.is_file():
                        remaining -= bounded_copy(
                            path, artifacts / "trace" / path.relative_to(source), remaining
                        )
        if history is None:
            detail = (
                detail
                or "Deadline reached while observing the existing prompt; no cancellation sent"
            )
        elif status == "success":
            detail = (
                "Comfy completed; artifacts retained. Profiled times are diagnostic only."
                if trace
                else "Comfy completed; artifacts retained."
            )
        return finish(artifacts, prompt_id, status, detail, history, started)
    finally:
        if ws is not None:
            ws.close()


def finish(
    artifacts: Path,
    prompt_id: str,
    status: str,
    detail: str,
    history: dict[str, Any] | None,
    started: float,
) -> dict[str, Any]:
    times = {
        name: data["timestamp"]
        for name, data in (history or {}).get("status", {}).get("messages", [])
        if "timestamp" in data
    }
    seconds = (
        (times["execution_success"] - times["execution_start"]) / 1000
        if status == "success" and {"execution_success", "execution_start"} <= times.keys()
        else None
    )
    result = {
        "status": status,
        "prompt_id": prompt_id,
        "server_execution_seconds": seconds,
        "observer_wall_seconds": time.monotonic() - started,
        "detail": detail,
        "submission_repeated": False,
        "cancellation_sent": False,
    }
    write_json(artifacts / "summary.json", result)
    return result
