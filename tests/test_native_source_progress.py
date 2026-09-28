"""Native source work reports byte progress, per-stage timings and crash diagnoses."""

from __future__ import annotations

import hashlib
import json
import os
import queue
import signal
import subprocess
import sys
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from cozy_runtime import canonical_json
from cozy_runtime.internal import source_interfaces
from cozy_runtime.internal.worker import workspace_sources
from cozy_runtime.internal.worker.source_calls import SourceCalls, diagnosis
from cozy_runtime.internal.worker.workspace import Workspace
from cozy_runtime.protocol import worker_pb2 as pb
from test_workspace_native_memo import accepted

FIXTURE = Path(__file__).parent / "testdata/native_source"


@contextmanager
def origin(body: bytes, pieces: int, pause: threading.Event, gap: float) -> Iterator[int]:
    """A stand-in Hugging Face origin that sends one carrier in `pieces` spaced writes."""
    digest = hashlib.sha256(body).hexdigest()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args: object) -> None:
            pass

        def do_GET(self) -> None:
            if self.path.startswith("/api/models/"):
                listing = [
                    {
                        "type": "file",
                        "path": "provider/first.safetensors",
                        "size": len(body),
                        "lfs": {"oid": digest, "size": len(body)},
                    }
                ]
                data = json.dumps(listing).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            start, end = map(int, self.headers["Range"].removeprefix("bytes=").split("-"))
            data = body[start : end + 1]
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{len(body)}")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            step = -(-len(data) // pieces)
            try:
                for offset in range(0, len(data), step):
                    if offset:
                        pause.wait(gap)
                    self.wfile.write(data[offset : offset + step])
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        yield server.server_port
    finally:
        pause.set()
        server.shutdown()
        server.server_close()
        thread.join()


def resolved(
    calls: SourceCalls,
    messages: queue.Queue[pb.NativeSourceStatus],
    command: pb.NativeSourceCommand,
) -> pb.NativeSourceCommand:
    command.phase = pb.NATIVE_SOURCE_PHASE_RESOLVE
    calls.handle(command)
    status = messages.get(timeout=60)
    assert status.state == pb.NATIVE_SOURCE_STATE_RESOLVED, status.safe_code
    command.phase = pb.NATIVE_SOURCE_PHASE_EXECUTE
    command.selection.CopyFrom(status.selection)
    return command


def conversion(source: pb.NativeSourceCommand, result: bytes) -> pb.NativeSourceCommand:
    command = pb.NativeSourceCommand()
    command.CopyFrom(source)
    command.operation = pb.NATIVE_SOURCE_OPERATION_CONVERT
    command.ClearField("selection")
    command.parent_call.call_index = 1
    command.parent_call.export = "convert_cozytensors"
    args = {"source": canonical_json.decode(result), "profiles": ["fixture/first/1"]}
    command.parent_call.request_canonical_bytes = canonical_json.encode(args)
    intent = {"module": source_interfaces.MODULE, "export": "convert_cozytensors", "request": args}
    command.parent_call.intent_digest = hashlib.sha256(canonical_json.encode(intent)).digest()
    parent = source.parent_call.parent_request_id
    command.service_id = workspace_sources.identity("owner", parent, 1)
    return command


def stage(frames: list[tuple[str, int, int, dict[str, Any]]], name: str) -> list[dict[str, Any]]:
    return [
        frame
        for *_, frame in frames
        if frame.get("stage") == name or frame.get("fields", {}).get("phase") == name
    ]


def test_download_and_conversion_report_bytes_rate_and_stage_timings(tmp_path: Path) -> None:
    body = (FIXTURE / "first.safetensors").read_bytes()
    with origin(body, 4, threading.Event(), 0.8) as port:
        workspace = Workspace(tmp_path / "store")
        messages: queue.Queue[pb.NativeSourceStatus] = queue.Queue()
        frames: list[tuple[str, int, int, dict[str, Any]]] = []
        calls = SourceCalls(
            workspace,
            lambda: "owner",
            messages.put,
            lambda _: True,
            endpoints={"huggingface": f"http://127.0.0.1:{port}"},
            native_registry=(FIXTURE / "registry.json").read_bytes(),
            progress=lambda *row: frames.append(row),
        )
        source = resolved(calls, messages, accepted(workspace, "ingest"))
        calls.handle(source)
        downloaded = messages.get(timeout=60)
        assert downloaded.state == pb.NATIVE_SOURCE_STATE_SUCCEEDED, downloaded.safe_code
        calls.handle(conversion(source, downloaded.result_canonical_bytes))
        converted = messages.get(timeout=60)
        assert converted.state == pb.NATIVE_SOURCE_STATE_SUCCEEDED, converted.safe_code

    # Every frame reaches the parent attempt's lane under its own call index.
    assert {(request, attempt) for request, attempt, *_ in frames} == {("ingest", 1)}
    download = stage(frames, "Downloading source")
    assert download[0] == {"kind": "progress", "stage": "Downloading source"}
    samples = [frame for frame in download if frame["kind"] == "progress" and "total" in frame]
    assert all(frame["unit"] == "bytes" and frame["total"] == len(body) for frame in samples)
    # TensorFS reports bytes as they become durable: at each 64 MiB checkpoint and when a run
    # ends. This carrier is one short run; a mid-body report is the resume test's subject.
    assert samples[-1]["position"] == len(body)
    ended = download[-1]
    assert ended["kind"] == "log" and ended["value"] == "info"
    fields = ended["fields"]
    assert fields["completed"] and fields["bytes"] == fields["total_bytes"] == len(body)
    assert fields["moved_bytes"] == len(body) and fields["elapsed_ms"] >= 1500
    assert 0 < fields["rate_bytes_per_second"] < len(body)

    convert = stage(frames, "Converting to cozytensors")
    assert convert[0] == {"kind": "progress", "stage": "Converting to cozytensors"}
    final = [frame for frame in convert if frame["kind"] == "progress" and "total" in frame][-1]
    assert final["position"] == final["total"] > 0 and final["unit"] == "bytes"
    assert convert[-1]["kind"] == "log" and convert[-1]["fields"]["completed"]
    indexes = {index for _, _, index, frame in frames if frame in convert}
    assert indexes == {1}


def test_crashed_source_worker_failure_carries_its_signal_and_stderr_tail(tmp_path: Path) -> None:
    body = (FIXTURE / "first.safetensors").read_bytes()
    held = threading.Event()
    with origin(body, 2, held, 60) as port:
        workspace = Workspace(tmp_path / "store")
        messages: queue.Queue[pb.NativeSourceStatus] = queue.Queue()
        frames: list[tuple[str, int, int, dict[str, Any]]] = []
        calls = SourceCalls(
            workspace,
            lambda: "owner",
            messages.put,
            lambda _: True,
            endpoints={"huggingface": f"http://127.0.0.1:{port}"},
            progress=lambda *row: frames.append(row),
        )
        source = resolved(calls, messages, accepted(workspace, "crashing"))
        calls.handle(source)
        while not any("position" in frame for *_, frame in frames):
            assert messages.empty(), "download ended before its first byte sample"
            held.wait(0.05)
        with calls.lock:
            process = calls.processes[source.service_id]
            assert process is not None
        # A native fault: the worker's own fault handler writes every thread's stack.
        os.kill(process.pid, signal.SIGSEGV)
        failed = messages.get(timeout=60)
    assert failed.state == pb.NATIVE_SOURCE_STATE_FAILED
    assert failed.safe_code == "native_source_failed"
    # The fault's headline and the newest stack frames both survive the 1 KiB detail.
    assert failed.safe_detail.startswith(
        "source worker was killed by SIGSEGV; stderr: Fatal Python error: Segmentation fault"
    )
    assert "in materialize" in failed.safe_detail and len(failed.safe_detail) < 1024
    assert calls.failure_detail(source.service_id) == failed.safe_detail
    logged = [frame for *_, frame in frames if frame.get("name") == "native source failed"]
    assert logged[0]["value"] == "error"
    assert "Segmentation fault" in logged[0]["fields"]["detail"]
    ended = stage(frames, "Downloading source")[-1]
    assert ended["kind"] == "log" and not ended["fields"]["completed"]
    assert ended["value"] == "warning" and ended["fields"]["bytes"] < len(body)


def test_a_malformed_source_worker_launch_names_itself_in_the_diagnosis() -> None:
    """A launch that does not decode refuses at the boundary with its own code, and its
    type and frames reach the detail."""
    child = subprocess.run(
        [sys.executable, "-m", "cozy_runtime.internal.worker.source_worker"],
        input=canonical_json.encode({}),
        capture_output=True,
        check=True,
    )
    assert canonical_json.decode(child.stdout.strip()) == {
        "answer": "refused",
        "code": "native_step_malformed",
    }
    detail = diagnosis(0, child.stderr.decode(), "")
    assert "SourceRefusal: Object missing required field `store`" in detail
    assert "source_steps.py:" in detail and " in serve" in detail


def test_diagnosis_redacts_credentials_and_signed_urls() -> None:
    stderr = (
        "thread 'transfer' panicked at src/pull.rs:12:\n"
        "GET https://cdn.example/a/b.safetensors?X-Amz-Signature=deadbeef&X-Amz-Credential=k"
        " failed with authorization: Bearer hf_secretsecret\n"
        "credential=hf_secretsecret token=abcdef123456\n"
    )
    detail = diagnosis(-signal.SIGKILL, stderr, "bearer hf_secretsecret")
    assert detail.startswith("source worker was killed by SIGKILL (the kernel's out-of-memory")
    assert "https://cdn.example/a/b.safetensors?[redacted]" in detail
    for secret in ("hf_secretsecret", "deadbeef", "abcdef123456"):
        assert secret not in detail
    assert diagnosis(3, "", "") == "source worker exited with status 3"
