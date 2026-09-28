"""Fixed source commands execute native bytes and reuse them across independent parents."""

from __future__ import annotations

import hashlib
import json
import queue
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from weakref import WeakValueDictionary

import msgspec
import pytest
import tensorfs

from cozy_runtime import canonical_json
from cozy_runtime.internal import storage_admission
from cozy_runtime.internal.worker import source_steps, workspace_memo
from cozy_runtime.internal.worker.source_calls import SourceCalls
from cozy_runtime.internal.worker.stage_progress import SAMPLE_SECONDS
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
        messages: queue.Queue[pb.NativeSourceStatus] = queue.Queue()
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
            resolved = messages.get(timeout=30)
            assert resolved.state == pb.NATIVE_SOURCE_STATE_RESOLVED, resolved.safe_code
            command.phase = pb.NATIVE_SOURCE_PHASE_EXECUTE
            command.selection.CopyFrom(resolved.selection)
            calls.handle(command)
            result = messages.get(timeout=30)
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
        messages: queue.Queue[pb.NativeSourceStatus] = queue.Queue()
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
            resolution = messages.get(timeout=30)
            assert resolution.state == pb.NATIVE_SOURCE_STATE_RESOLVED, resolution.safe_code
            source.phase = pb.NATIVE_SOURCE_PHASE_EXECUTE
            source.selection.CopyFrom(resolution.selection)
            calls.handle(source)
            downloaded = messages.get(timeout=30)
            assert downloaded.state == pb.NATIVE_SOURCE_STATE_SUCCEEDED, downloaded.safe_code
            command = pb.NativeSourceCommand()
            command.CopyFrom(source)
            command.operation = pb.NATIVE_SOURCE_OPERATION_CONVERT
            command.ClearField("selection")
            command.parent_call.call_index = 1
            command.parent_call.export = "convert_cozytensors"
            args = {
                "source": canonical_json.decode(downloaded.result_canonical_bytes),
                "profiles": ["fixture/first/1"],
            }
            command.parent_call.request_canonical_bytes = canonical_json.encode(args)
            command.parent_call.intent_digest = hashlib.sha256(
                canonical_json.encode(
                    {
                        "module": source_interfaces.MODULE,
                        "export": "convert_cozytensors",
                        "request": args,
                    }
                )
            ).digest()
            command.service_id = workspace_sources.identity("owner", parent, 1)
            calls.handle(command)
            converted = messages.get(timeout=30)
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


def test_killed_source_process_adopts_durable_prefix_in_a_fresh_script(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A killed download's durable prefix is adopted: the fresh script fetches only the rest.

    TensorFS asks for a run with one ranged GET and makes it durable every 64 MiB mid-body.
    The origin stalls the first script right after that checkpoint; once its download
    progress reports the prefix durable, the process is killed.
    """
    # This SourceCalls has no Worker admission owner; a Worker leaked by another test would
    # otherwise hold local write admission for the whole download and block the resolve.
    monkeypatch.setattr(storage_admission, "_reclaimers", WeakValueDictionary())
    checkpoint = 64 << 20
    chunk = bytes(range(256)) * 4096
    length = checkpoint + len(chunk)
    digest = hashlib.sha256(chunk * (length // len(chunk))).hexdigest()
    stalled = threading.Event()
    release = threading.Event()
    refetched: list[tuple[int, int]] = []
    frames: list[dict[str, object]] = []

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
            resumed = release.is_set()
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{length}")
            self.send_header("Content-Length", str(end - start + 1))
            self.end_headers()
            offset = start
            try:
                while offset <= end:
                    if not resumed and offset == checkpoint:
                        stalled.set()
                        release.wait(60)
                        return
                    if not resumed and offset + len(chunk) == checkpoint:
                        # The durable report arrives after the progress lane's throttle.
                        time.sleep(1.5 * SAMPLE_SECONDS)
                    piece = chunk[offset % len(chunk) :][: end + 1 - offset]
                    self.wfile.write(piece)
                    offset += len(piece)
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                if resumed:
                    refetched.append((start, offset - start))

    origin = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=origin.serve_forever)
    thread.start()
    try:
        workspace = Workspace(tmp_path / "store")
        messages: queue.Queue[pb.NativeSourceStatus] = queue.Queue()
        calls = SourceCalls(
            workspace,
            lambda: "owner",
            messages.put,
            lambda _: True,
            endpoints={"huggingface": f"http://127.0.0.1:{origin.server_port}"},
            progress=lambda *row: frames.append(row[-1]),
        )
        first = accepted(workspace, "crashing-script")
        first.phase = pb.NATIVE_SOURCE_PHASE_RESOLVE
        calls.handle(first)
        resolved = messages.get(timeout=30)
        assert resolved.state == pb.NATIVE_SOURCE_STATE_RESOLVED
        first.phase = pb.NATIVE_SOURCE_PHASE_EXECUTE
        first.selection.CopyFrom(resolved.selection)
        calls.handle(first)
        assert stalled.wait(60), "download never reached its first checkpoint"
        # Another metadata control operation can run while the payload transfer waits.
        second = accepted(workspace, "corrected-script")
        second.phase = pb.NATIVE_SOURCE_PHASE_RESOLVE
        calls.handle(second)
        resolved = messages.get(timeout=30)
        assert resolved.state == pb.NATIVE_SOURCE_STATE_RESOLVED
        deadline = time.monotonic() + 60
        while not any(frame.get("position") == checkpoint for frame in frames):
            assert time.monotonic() < deadline, "the checkpoint was never reported durable"
            time.sleep(0.05)
        with calls.lock:
            process = calls.processes[first.service_id]
            assert process is not None
            process.kill()
        stopped = messages.get(timeout=30)
        assert stopped.state == pb.NATIVE_SOURCE_STATE_FAILED
        release.set()
        second.phase = pb.NATIVE_SOURCE_PHASE_EXECUTE
        second.selection.CopyFrom(resolved.selection)
        calls.handle(second)
        finished = messages.get(timeout=30)
        assert finished.state == pb.NATIVE_SOURCE_STATE_SUCCEEDED, finished.safe_code
        assert not finished.memo_hit
        # Exactly the bytes the killed script had not made durable are fetched again.
        assert refetched and all(start >= checkpoint for start, _ in refetched)
        assert sum(moved for _, moved in refetched) == length - checkpoint
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
    messages: queue.Queue[pb.NativeSourceStatus] = queue.Queue()
    calls = SourceCalls(workspace, lambda: "owner", messages.put, lambda _: True)
    calls.handle(stopped)
    ack = pb.AttemptOutcomeAck(
        request_id=command.parent_call.parent_request_id,
        attempt_ordinal=1,
        invocation_spec_digest=command.parent_call.parent_invocation_spec_digest,
        outcome_id="source-parent-outcome",
        outcome_digest=digest,
        retain_work=False,
    )
    workspace.acknowledge("owner", ack)
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
        url = f"http://127.0.0.1:{origin.server_port}/weights"
        step = source_steps.Download(
            native_owner=native_owner,
            service_id="source-parent-death",
            pin=source_steps.Pin(
                [("weights.safetensors", "sha256:" + digest, len(data), url)], ["127.0.0.1"], []
            ),
            access=source_steps.Access(
                credential="", huggingface=None, civitai=None, allow_local=True
            ),
        )
        envelope = {"store": store.root, "step": msgspec.to_builtins(step)}
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


def _second_carrier(body: bytes) -> bytes:
    """The fixture decoder: the first carrier's F32[1024] header over different values."""
    header_length = int.from_bytes(body[:8], "little")
    return body[: 8 + header_length] + bytes([7]) * (len(body) - 8 - header_length)


def test_several_reviewed_profiles_compose_one_model_and_memoize(tmp_path: Path) -> None:
    from cozy_runtime.internal import source_interfaces
    from cozy_runtime.internal.worker import workspace_sources

    fixture = Path(__file__).parent / "testdata/native_source"
    first = (fixture / "first.safetensors").read_bytes()
    files = {
        "provider/first.safetensors": first,
        "provider/second.safetensors": _second_carrier(first),
    }
    served: list[str] = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args: object) -> None:
            pass

        def do_GET(self) -> None:
            if self.path.startswith("/api/models/"):
                data = json.dumps(
                    [
                        {
                            "type": "file",
                            "path": path,
                            "size": len(body),
                            "lfs": {"oid": hashlib.sha256(body).hexdigest(), "size": len(body)},
                        }
                        for path, body in files.items()
                    ]
                ).encode()
                self.send_response(200)
            else:
                served.append(self.path)
                body = next(body for path, body in files.items() if self.path.endswith("/" + path))
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
        messages: queue.Queue[pb.NativeSourceStatus] = queue.Queue()
        calls = SourceCalls(
            workspace,
            lambda: "owner",
            messages.put,
            lambda _: True,
            endpoints={"huggingface": f"http://127.0.0.1:{origin.server_port}"},
            native_registry=(fixture / "registry.json").read_bytes(),
        )

        def convert(parent: str, arguments: dict[str, object]) -> pb.NativeSourceStatus:
            source = accepted(workspace, parent, {"carriers": sorted(files)})
            source.phase = pb.NATIVE_SOURCE_PHASE_RESOLVE
            calls.handle(source)
            resolution = messages.get(timeout=30)
            assert resolution.state == pb.NATIVE_SOURCE_STATE_RESOLVED, resolution.safe_code
            source.phase = pb.NATIVE_SOURCE_PHASE_EXECUTE
            source.selection.CopyFrom(resolution.selection)
            calls.handle(source)
            downloaded = messages.get(timeout=30)
            assert downloaded.state == pb.NATIVE_SOURCE_STATE_SUCCEEDED, downloaded.safe_code
            command = pb.NativeSourceCommand()
            command.CopyFrom(source)
            command.operation = pb.NATIVE_SOURCE_OPERATION_CONVERT
            command.ClearField("selection")
            command.parent_call.call_index = 1
            command.parent_call.export = "convert_cozytensors"
            args = {"source": canonical_json.decode(downloaded.result_canonical_bytes), **arguments}
            command.parent_call.request_canonical_bytes = canonical_json.encode(args)
            command.parent_call.intent_digest = hashlib.sha256(
                canonical_json.encode(
                    {
                        "module": source_interfaces.MODULE,
                        "export": "convert_cozytensors",
                        "request": args,
                    }
                )
            ).digest()
            command.service_id = workspace_sources.identity("owner", parent, 1)
            calls.handle(command)
            return messages.get(timeout=30)

        profiles = ["fixture/first/1", "fixture/second/1"]
        both: dict[str, object] = {"profiles": profiles}
        composed = convert("compose", both)
        assert composed.state == pb.NATIVE_SOURCE_STATE_SUCCEEDED, composed.safe_code
        assert not composed.memo_hit
        store = tensorfs.Store.open(str(workspace.store_root))
        manifest = canonical_json.decode(composed.result_canonical_bytes)["manifest"]
        header_bytes = store.manifest(manifest["digest"])["header"]
        assert header_bytes is not None
        header = tensorfs.parse_header(header_bytes)
        assert sorted(header["components"]) == ["decoder", "encoder"]
        assert sorted(served) == sorted(set(served)), "a carrier was downloaded twice"
        assert {path.rsplit("/", 1)[-1] for path in served} == {
            "first.safetensors",
            "second.safetensors",
        }

        again = convert("compose-again", both)
        assert again.state == pb.NATIVE_SOURCE_STATE_SUCCEEDED, again.safe_code
        assert again.memo_hit
        assert canonical_json.decode(again.result_canonical_bytes)["manifest"] == manifest
    finally:
        origin.shutdown()
        origin.server_close()
        thread.join()
