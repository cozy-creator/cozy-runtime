"""Fixed source commands execute native bytes and reuse them across independent parents."""

from __future__ import annotations

import hashlib
import json
import queue
import threading
from collections.abc import Iterator
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
        assert original is not None
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
        with calls.lock:
            process = calls.processes[first.service_id]
            assert process is not None
            process.kill()
        stopped = messages.get(timeout=30).native_source_status
        assert stopped.state == pb.NATIVE_SOURCE_STATE_FAILED
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
        assert store.verify(digest)["verdict"] in {"recorded", "valid", "rehashed", "verified"}
    finally:
        release.set()
        origin.shutdown()
        origin.server_close()
        thread.join()


def test_permanent_ack_releases_native_producer_and_access_but_keeps_cache(tmp_path: Path) -> None:
    from cozy_runtime.internal.worker import workspace_sources
    from cozy_runtime.protocol import documents
    from test_workspace_native_memo import completed

    workspace = Workspace(tmp_path / "store")
    command = accepted(workspace, "completed-parent")
    result, root = completed(workspace, command)
    assert workspace_memo.record_native(workspace, "owner", command.service_id)
    body, digest = documents.identity(
        pb.AttemptOutcomeBody(
            request_id=command.parent_call.parent_request_id,
            attempt_ordinal=1,
            invocation_spec_digest=documents.spell(
                command.parent_call.parent_invocation_spec_digest
            ),
            status=pb.OUTCOME_STATUS_SUCCEEDED,
            execution_started=True,
            result=pb.ResultEnvelope(inline_result=b"{}"),
        )
    )
    workspace.outcome(
        "owner",
        pb.AttemptOutcome(
            request_id=command.parent_call.parent_request_id,
            attempt_ordinal=1,
            invocation_spec_digest=command.parent_call.parent_invocation_spec_digest,
            outcome_id="source-parent-outcome",
            outcome_digest=digest,
            outcome_canonical_bytes=body,
        ),
    )
    # A stop after the parent terminal is accepted against recorded call identity.
    stopped = pb.NativeSourceCommand()
    stopped.CopyFrom(command)
    stopped.phase = pb.NATIVE_SOURCE_PHASE_CANCEL
    messages: queue.Queue[pb.WorkerFrame] = queue.Queue()
    calls = SourceCalls(workspace, lambda: "owner", messages.put, lambda _: True)
    calls.handle(stopped)
    done = threading.Event()
    ack = pb.AttemptOutcomeAck(
        request_id=command.parent_call.parent_request_id,
        attempt_ordinal=1,
        invocation_spec_digest=command.parent_call.parent_invocation_spec_digest,
        outcome_id="source-parent-outcome",
        outcome_digest=digest,
        retain_work=False,
    )
    assert calls.acknowledge(ack, done.set)
    assert done.wait(30), "source finalizer did not complete"
    store = tensorfs.Store.open(str(workspace.store_root))
    released_root = store.tree_root(root["owner"])
    assert released_root is not None and released_root["released"]
    with workspace.locked() as db:
        row = db.execute(
            "SELECT * FROM native_calls WHERE service_id=?", (command.service_id,)
        ).fetchone()
        assert row["state"] == "released"
        assert row["result"] == result
    tensorfs.gc(store.root)
    consumer = accepted(workspace, "later-independent-parent")
    hit = workspace_memo.lookup_native(workspace, "owner", consumer.service_id, b"k" * 32)
    assert hit is not None and hit["result"] == result
    workspace_memo.record_reuse(workspace, "owner", consumer.service_id)
    assert not store.tree_root(workspace_sources.native_owner("owner", consumer.service_id))


def test_native_source_child_stops_when_its_runtime_process_dies(tmp_path: Path) -> None:
    import subprocess
    import sys

    waiting = threading.Event()
    released = threading.Event()
    data = b"retained worker lifetime proof"
    digest = hashlib.sha256(data).hexdigest()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args: object) -> None:
            pass

        def do_GET(self) -> None:
            waiting.set()
            released.wait(30)
            try:
                self.send_response(206)
                self.send_header("Content-Range", f"bytes 0-{len(data) - 1}/{len(data)}")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):
                pass

    origin = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=origin.serve_forever)
    thread.start()
    parent = None
    try:
        store = tensorfs.Store.ensure(tmp_path / "store")
        native_owner = tensorfs.object_id(b"parent-death-source")
        envelope = {
            "store": store.root,
            "operation": "download_huggingface",
            "arguments": {},
            "phase": "execute",
            "native_owner": native_owner,
            "service_id": "source-parent-death",
            "allow_local": True,
            "selection": {
                "members": [
                    [
                        "weights.safetensors",
                        "sha256:" + digest,
                        len(data),
                        f"http://127.0.0.1:{origin.server_port}/weights",
                    ]
                ],
                "allowed_hosts": ["127.0.0.1"],
                "credential_hosts": [],
            },
        }
        wrapper = """
import json,os,subprocess,sys,threading
from cozy_runtime import canonical_json
command=json.load(sys.stdin);command['parent_pid']=os.getpid()
child=subprocess.Popen([sys.executable,'-m','cozy_runtime.internal.worker.source_worker'],stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
child.stdin.write(canonical_json.encode(command));child.stdin.close()
print(child.pid,flush=True)
threading.Event().wait()
"""
        parent = subprocess.Popen(
            [sys.executable, "-c", wrapper],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        assert parent.stdin is not None and parent.stdout is not None
        parent.stdin.write(canonical_json.encode(envelope))
        parent.stdin.close()
        child_pid = int(parent.stdout.readline())
        assert waiting.wait(30)
        parent.kill()
        parent.wait()
        stopped = False
        for _ in range(200):
            try:
                state = Path(f"/proc/{child_pid}/stat").read_text().rsplit(") ", 1)[1].split()[0]
                stopped = state in ("Z", "X")
            except FileNotFoundError:
                stopped = True
            if stopped:
                break
            threading.Event().wait(0.05)
        assert stopped, "source subprocess outlived its Runtime process"
        store.release_tree_root(native_owner)  # kernel writer fence has actually been released
        released_root = store.tree_root(native_owner)
        assert released_root is not None and released_root["released"]
    finally:
        if parent is not None and parent.poll() is None:
            parent.kill()
            parent.wait()
        released.set()
        origin.shutdown()
        origin.server_close()
        thread.join()


def test_stale_parent_cancel_cannot_kill_a_new_source_attempt(tmp_path: Path) -> None:
    class Process:
        kills = 0

        def kill(self) -> None:
            self.kills += 1

    workspace = Workspace(tmp_path / "store")
    calls = SourceCalls(workspace, lambda: "owner", lambda _: None, lambda _: True)
    command = pb.NativeSourceCommand(
        parent_call=pb.ChildCallRequest(
            parent_request_id="parent",
            parent_attempt_ordinal=2,
            parent_invocation_spec_digest=b"n" * 32,
        )
    )
    process = Process()
    calls.parents["service"] = "parent"
    calls.commands["service"] = command
    calls.processes["service"] = process  # type: ignore[assignment]
    calls.cancel_parent("parent", ordinal=1, spec=b"o" * 32)
    calls.cancel_parent("parent", ordinal=2, spec=b"o" * 32)
    calls.cancel_parent("parent", through=1)
    assert process.kills == 0
    calls.cancel_parent("parent", ordinal=2, spec=b"n" * 32)
    assert process.kills == 1


def test_current_source_validation_refusal_reaches_protocol_peer(tmp_path: Path) -> None:
    from concurrent.futures import ThreadPoolExecutor

    import grpc

    from cozy_runtime.internal.worker.session import OwnershipFence, Worker
    from cozy_runtime.protocol import worker_pb2_grpc as rpc

    workspace = Workspace(tmp_path / "store")
    command = accepted(workspace, "refused-source-parent")
    command.record_owner_epoch = 1
    command.control_stream_epoch = 1
    command.worker_boot_id = "boot"
    command.phase = pb.NATIVE_SOURCE_PHASE_RESOLVE
    command.credential = "fake-source-refusal-secret-canary"
    command.parent_call.interface_digest = b"wrong-interface" * 2
    outgoing: queue.Queue[pb.WorkerFrame] = queue.Queue()

    class DispatchWorker(Worker):
        """Bind the real dispatcher to this test's existing control transport."""

        def __init__(self) -> None:
            self.fence = OwnershipFence(
                record_owner_epoch=1,
                control_stream_epoch=1,
                worker_boot_id="boot",
                wire_minor=42,
                dispatch_open=True,
            )
            self.source_calls = SourceCalls(
                workspace, lambda: "owner", outgoing.put, lambda _: True
            )

        def note(self, kind: str, step: str) -> None:
            pass

        def send(self, frame: pb.WorkerFrame) -> None:
            outgoing.put(frame)

    worker = DispatchWorker()

    # The generated grpc base has no type declarations.
    class Peer(rpc.WorkerControlServicer):  # type: ignore[misc]
        def Control(
            self, requests: Iterator[pb.RecordOwnerFrame], context: grpc.ServicerContext
        ) -> Iterator[pb.WorkerFrame]:
            for frame in requests:
                worker._dispatch_current_frame(frame)
                while not outgoing.empty():
                    yield outgoing.get_nowait()

    server = grpc.server(ThreadPoolExecutor(max_workers=2))
    rpc.add_WorkerControlServicer_to_server(Peer(), server)
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()
    try:
        with grpc.insecure_channel(f"127.0.0.1:{port}") as channel:
            frames = list(
                rpc.WorkerControlStub(channel).Control(
                    iter([pb.RecordOwnerFrame(native_source_command=command)]),
                    timeout=5,
                )
            )
        assert len(frames) == 1, "accepted caller was left waiting after source validation refusal"
        result = frames[0].native_source_status
        assert result.state == pb.NATIVE_SOURCE_STATE_FAILED
        assert result.safe_code == "native_source_refused"
        assert result.service_id == command.service_id
        assert result.parent_request_id == command.parent_call.parent_request_id
        assert result.intent_digest == command.parent_call.intent_digest
        assert b"fake-source-refusal-secret-canary" not in frames[0].SerializeToString()
        command.record_owner_epoch = 2
        worker._dispatch_current_frame(pb.RecordOwnerFrame(native_source_command=command))
        assert outgoing.empty(), "stale owner received a new source callback"
    finally:
        server.stop(0).wait()
