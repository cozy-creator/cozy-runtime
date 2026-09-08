"""Fixed source commands execute native bytes and reuse them across independent parents."""

from __future__ import annotations

import hashlib
import json
import queue
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import tensorfs

from cozy_runtime import canonical_json
from cozy_runtime.internal.worker import workspace_memo
from cozy_runtime.internal.worker.source_calls import SourceCalls
from cozy_runtime.internal.worker.workspace import Workspace
from cozy_runtime.protocol import worker_pb2 as pb
from test_workspace_native_memo import accepted


def test_native_source_resolve_execute_and_independent_parent_cache(tmp_path: Path) -> None:
    body = b"exact selected source bytes" * 1024
    sha256 = hashlib.sha256(body).hexdigest()
    bodies: list[str] = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args: object) -> None:
            pass

        def do_GET(self) -> None:
            if self.path.startswith("/api/models/"):
                data = json.dumps(
                    [
                        {
                            "type": "file",
                            "path": "weights.safetensors",
                            "size": len(body),
                            "lfs": {"oid": sha256, "size": len(body)},
                        }
                    ]
                ).encode()
                self.send_response(200)
            else:
                bodies.append(self.path)
                start, end = map(int, self.headers["Range"].removeprefix("bytes=").split("-"))
                data = body[start : end + 1]
                self.send_response(206)
                self.send_header("Content-Range", f"bytes {start}-{end}/{len(body)}")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    origin = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=origin.serve_forever)
    thread.start()
    try:
        workspace = Workspace(tmp_path / "store")
        messages: queue.Queue[pb.WorkerFrame] = queue.Queue()
        calls = SourceCalls(
            workspace,
            lambda: "owner",
            messages.put,
            lambda _: True,
            endpoints={"huggingface": f"http://127.0.0.1:{origin.server_port}"},
        )
        original: bytes | None = None
        for parent in ["first-script", "edited-script"]:
            command = accepted(workspace, parent)
            command.phase = pb.NATIVE_SOURCE_PHASE_RESOLVE
            calls.handle(command)
            resolved = messages.get(timeout=30).native_source_status
            assert resolved.state == pb.NATIVE_SOURCE_STATE_RESOLVED, resolved.safe_code
            command.phase = pb.NATIVE_SOURCE_PHASE_EXECUTE
            command.selection.CopyFrom(resolved.selection)
            calls.handle(command)
            result = messages.get(timeout=30).native_source_status
            assert result.state == pb.NATIVE_SOURCE_STATE_SUCCEEDED, result.safe_code
            assert result.memo_hit is (parent == "edited-script")
            if original is None:
                original = result.result_canonical_bytes
            else:
                assert result.result_canonical_bytes == original
            with workspace.locked() as db:
                access = db.execute(
                    "SELECT * FROM source_access WHERE request=? AND state='held'", (parent,)
                ).fetchone()
                assert access is not None
                assert access[
                    "manifest"
                ] == result.selection.content_manifest.digest or not result.HasField("selection")
            assert len(bodies) == 1
            workspace_memo.release_native_result(workspace, "owner", command.service_id)
        tensorfs.gc(str(workspace.store_root))
        assert tensorfs.Store.open(str(workspace.store_root)).contains(sha256)
        assert "http" not in original.decode()
        with workspace.locked() as db:
            for row in db.execute("SELECT accepted,pinned,result FROM native_calls"):
                assert all(b"127.0.0.1" not in value for value in row)
    finally:
        origin.shutdown()
        origin.server_close()
        thread.join()


def test_real_retained_source_conversion_reuses_and_registry_edit_invalidates(
    tmp_path: Path,
) -> None:
    from cozy_runtime.internal import source_interfaces
    from cozy_runtime.internal.worker import workspace_sources

    fixture = Path(__file__).parent / "testdata/native_source"
    body = (fixture / "first.safetensors").read_bytes()
    registry = (fixture / "registry.json").read_bytes()
    digest = hashlib.sha256(body).hexdigest()
    bodies: list[str] = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args: object) -> None:
            pass

        def do_GET(self) -> None:
            if self.path.startswith("/api/models/"):
                data = json.dumps(
                    [
                        {
                            "type": "file",
                            "path": "provider/first.safetensors",
                            "size": len(body),
                            "lfs": {"oid": digest, "size": len(body)},
                        }
                    ]
                ).encode()
                self.send_response(200)
            else:
                bodies.append(self.path)
                start, end = map(int, self.headers["Range"].removeprefix("bytes=").split("-"))
                data = body[start : end + 1]
                self.send_response(206)
                self.send_header("Content-Range", f"bytes {start}-{end}/{len(body)}")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    origin = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=origin.serve_forever)
    thread.start()
    try:
        workspace = Workspace(tmp_path / "store")
        messages: queue.Queue[pb.WorkerFrame] = queue.Queue()
        calls = SourceCalls(
            workspace,
            lambda: "owner",
            messages.put,
            lambda _: True,
            endpoints={"huggingface": f"http://127.0.0.1:{origin.server_port}"},
            native_registry=registry,
        )
        first_model: bytes | None = None
        for index, parent in enumerate(["original", "edited-script", "edited-registry"]):
            if index == 2:
                changed = json.loads(registry)
                changed["entries"][0]["provenance"]["note"] += "; changed registry evidence"
                calls.native_registry = canonical_json.encode(changed)
            source = accepted(workspace, parent)
            source.phase = pb.NATIVE_SOURCE_PHASE_RESOLVE
            calls.handle(source)
            resolution = messages.get(timeout=30).native_source_status
            assert resolution.state == pb.NATIVE_SOURCE_STATE_RESOLVED, resolution.safe_code
            source.phase = pb.NATIVE_SOURCE_PHASE_EXECUTE
            source.selection.CopyFrom(resolution.selection)
            calls.handle(source)
            downloaded = messages.get(timeout=30).native_source_status
            assert downloaded.state == pb.NATIVE_SOURCE_STATE_SUCCEEDED, downloaded.safe_code
            command = pb.NativeSourceCommand()
            command.CopyFrom(source)
            command.operation = pb.NATIVE_SOURCE_OPERATION_CONVERT
            command.ClearField("selection")
            command.parent_call.call_index = 1
            command.parent_call.export = "convert_cozytensors"
            args = {
                "source": canonical_json.decode(downloaded.result_canonical_bytes),
                "profile": "fixture/first/1",
            }
            command.parent_call.request_canonical_bytes = canonical_json.encode(args)
            command.parent_call.intent_digest = hashlib.sha256(
                canonical_json.encode(
                    {
                        "interface_digest": source_interfaces.DIGEST,
                        "module": source_interfaces.MODULE,
                        "export": "convert_cozytensors",
                        "request": args,
                    }
                )
            ).digest()
            command.service_id = workspace_sources.identity("owner", parent, 1)
            calls.handle(command)
            converted = messages.get(timeout=30).native_source_status
            assert converted.state == pb.NATIVE_SOURCE_STATE_SUCCEEDED, converted.safe_code
            assert converted.memo_hit is (index == 1)
            result = canonical_json.decode(converted.result_canonical_bytes)
            native = canonical_json.decode(converted.native_receipt_canonical_bytes)
            assert result["tensorfs_receipt_digest"] == tensorfs.object_id(
                converted.native_receipt_canonical_bytes
            )
            assert result["manifest"]["digest"] == "sha256:" + native["manifest"]["sha256"]
            if first_model is None:
                first_model = converted.result_canonical_bytes
            elif index == 1:
                assert converted.result_canonical_bytes == first_model
            else:
                assert result["manifest"] == canonical_json.decode(first_model)["manifest"]
                assert (
                    result["producer_request_id"]
                    != canonical_json.decode(first_model)["producer_request_id"]
                )
            assert len(bodies) == 1
            store = tensorfs.Store.open(str(workspace.store_root))
            assert store.contains(digest), "converter spent observation deleted retained raw source"
            assert store.manifest(result["manifest"]["digest"])["header"]
    finally:
        origin.shutdown()
        origin.server_close()
        thread.join()


def test_killed_source_process_adopts_durable_prefix_in_a_fresh_script(tmp_path: Path) -> None:
    block = 64 << 20
    chunk = bytes(range(256)) * 4096
    length = block + len(chunk)
    digest = hashlib.sha256(chunk * 65).hexdigest()
    waiting = threading.Event()
    release = threading.Event()
    ranges: list[int] = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args: object) -> None:
            pass

        def do_GET(self) -> None:
            if self.path.startswith("/api/models/"):
                data = json.dumps(
                    [
                        {
                            "type": "file",
                            "path": "weights.safetensors",
                            "size": length,
                            "lfs": {"oid": digest, "size": length},
                        }
                    ]
                ).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            start, end = map(int, self.headers["Range"].removeprefix("bytes=").split("-"))
            ranges.append(start)
            if start == block and not release.is_set():
                waiting.set()
                release.wait(30)
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{length}")
            self.send_header("Content-Length", str(end - start + 1))
            self.end_headers()
            try:
                for _ in range((end - start + 1) // len(chunk)):
                    self.wfile.write(chunk)
            except (BrokenPipeError, ConnectionResetError):
                pass

    origin = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=origin.serve_forever)
    thread.start()
    try:
        workspace = Workspace(tmp_path / "store")
        messages: queue.Queue[pb.WorkerFrame] = queue.Queue()
        calls = SourceCalls(
            workspace,
            lambda: "owner",
            messages.put,
            lambda _: True,
            endpoints={"huggingface": f"http://127.0.0.1:{origin.server_port}"},
        )
        first = accepted(workspace, "crashing-script")
        first.phase = pb.NATIVE_SOURCE_PHASE_RESOLVE
        calls.handle(first)
        resolved = messages.get(timeout=30).native_source_status
        assert resolved.state == pb.NATIVE_SOURCE_STATE_RESOLVED
        first.phase = pb.NATIVE_SOURCE_PHASE_EXECUTE
        first.selection.CopyFrom(resolved.selection)
        calls.handle(first)
        assert waiting.wait(30), "download never reached the second native range"
        # Another metadata control operation can run while the payload transfer waits.
        second = accepted(workspace, "corrected-script")
        second.phase = pb.NATIVE_SOURCE_PHASE_RESOLVE
        calls.handle(second)
        resolved = messages.get(timeout=30).native_source_status
        assert resolved.state == pb.NATIVE_SOURCE_STATE_RESOLVED
        calls.cancel_parent("crashing-script")
        stopped = messages.get(timeout=30).native_source_status
        assert stopped.state == pb.NATIVE_SOURCE_STATE_CANCELED
        release.set()
        before = len(ranges)
        second.phase = pb.NATIVE_SOURCE_PHASE_EXECUTE
        second.selection.CopyFrom(resolved.selection)
        calls.handle(second)
        finished = messages.get(timeout=30).native_source_status
        assert finished.state == pb.NATIVE_SOURCE_STATE_SUCCEEDED, finished.safe_code
        assert not finished.memo_hit
        assert ranges[before:] == [block], "fresh script did not reuse the native durable prefix"
        store = tensorfs.Store.open(str(workspace.store_root))
        assert store.contains(digest)
        assert store.verify(digest)["verdict"] in {"recorded", "valid", "rehashed"}
    finally:
        release.set()
        origin.shutdown()
        origin.server_close()
        thread.join()
