"""New recipients reuse real native bytes without rewriting their producer."""

import hashlib
import json
import queue
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import msgspec
import pytest
import tensorfs

from cozy_runtime import canonical_json
from cozy_runtime.author._artifacts import ModelArtifact, ObjectRef
from cozy_runtime.internal import source_interfaces
from cozy_runtime.internal.worker import workspace_sources
from cozy_runtime.internal.worker.machine_artifacts import model_retention
from cozy_runtime.internal.worker.source_calls import SourceCalls
from cozy_runtime.internal.worker.workspace import Workspace, WorkspaceRefusal
from cozy_runtime.protocol import worker_pb2 as pb
from test_workspace_custody import produced
from test_workspace_memo import completed, release_original
from test_workspace_native_memo import accepted


def test_independent_machine_hold_survives_producer_release(tmp_path: Path) -> None:
    store, workspace, receipt = produced(tmp_path)
    artifact = receipt.artifact
    source = model_retention(workspace, "owner", "parent", "result.model", artifact)
    assert source == model_retention(workspace, "owner", "parent", "result.model", artifact)
    first = workspace.retain("owner", source)
    release_original(workspace, completed(workspace, receipt), receipt)
    tensorfs.gc(str(store.root))
    successor = model_retention(workspace, "owner", "later-parent", "input", artifact)
    assert successor.retention_id != source.retention_id
    assert successor.weights_transaction_id == source.weights_transaction_id
    second = workspace.retain("owner", successor)
    assert first.manifest == second.manifest
    workspace.retain("owner", source, release=True)
    tensorfs.gc(str(store.root))
    with workspace.held_model("owner", successor, first.manifest):
        assert store.manifest(artifact.manifest.digest)["manifest"]
    assert artifact.producer_request_id == "producer"


def test_model_hold_rejects_forged_provenance_and_other_owners(tmp_path: Path) -> None:
    _, workspace, receipt = produced(tmp_path)
    artifact = receipt.artifact
    for changed in (
        msgspec.structs.replace(artifact, producer_request_id="not-the-producer"),
        msgspec.structs.replace(artifact, output_slot="other"),
        msgspec.structs.replace(artifact, tensorfs_receipt_digest="sha256:" + "ab" * 32),
    ):
        with pytest.raises(WorkspaceRefusal, match="provenance"):
            model_retention(workspace, "owner", "parent", "result", changed)
    with pytest.raises(WorkspaceRefusal, match="provenance"):
        model_retention(workspace, "different-owner", "parent", "result", artifact)
    with workspace.locked() as db:
        assert db.execute("SELECT count(*) FROM holds").fetchone()[0] == 0


def _converted(tmp_path: Path) -> tuple[Workspace, ModelArtifact, str]:
    """One real native download and cozytensors conversion from a local origin."""
    fixture = Path(__file__).parent / "testdata/native_source"
    body = (fixture / "first.safetensors").read_bytes()
    digest = hashlib.sha256(body).hexdigest()

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
        source = accepted(workspace, "converter")
        source.phase = pb.NATIVE_SOURCE_PHASE_RESOLVE
        calls.handle(source)
        resolution = messages.get(timeout=30)
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
        command.service_id = workspace_sources.identity("owner", "converter", 1)
        calls.handle(command)
        converted = messages.get(timeout=30)
        assert converted.state == pb.NATIVE_SOURCE_STATE_SUCCEEDED, converted.safe_code
    finally:
        origin.shutdown()
        origin.server_close()
        thread.join()
    artifact = msgspec.convert(
        canonical_json.decode(converted.result_canonical_bytes), type=ModelArtifact
    )
    return workspace, artifact, command.service_id


def test_native_conversion_hold_reads_another_versions_additive_members(tmp_path: Path) -> None:
    workspace, artifact, service = _converted(tmp_path)
    planned = model_retention(workspace, "owner", "parent", "input", artifact)

    # The conversion result and TensorFS receipt as another Runtime version retained them.
    with workspace.locked() as db:
        row = db.execute(
            "SELECT result,native_receipt FROM native_calls WHERE owner=? AND service_id=?",
            ("owner", service),
        ).fetchone()
        receipt = canonical_json.decode(row["native_receipt"])
        receipt["manifest"]["encoding"] = "cozytensors/2"
        receipt["writer"] = "tensorfs-next"
        retained = canonical_json.encode(receipt)
        result = canonical_json.decode(row["result"])
        result["tensorfs_receipt_digest"] = "sha256:" + hashlib.sha256(retained).hexdigest()
        result["manifest"]["encoding"] = "cozytensors/2"
        result["retention_hint"] = "warm"
        db.execute(
            "UPDATE native_calls SET result=?,native_receipt=? WHERE owner=? AND service_id=?",
            (canonical_json.encode(result), retained, "owner", service),
        )
    current = msgspec.convert(result, type=ModelArtifact)
    later = model_retention(workspace, "owner", "parent", "input", current)
    assert later.weights_transaction_id == planned.weights_transaction_id
    assert later.tensorfs_receipt_digest == hashlib.sha256(retained).digest()

    for changed in (
        msgspec.structs.replace(current, output_slot="other"),
        msgspec.structs.replace(
            current, manifest=ObjectRef(current.manifest.digest, current.manifest.length + 1)
        ),
        artifact,
    ):
        with pytest.raises(WorkspaceRefusal, match="provenance"):
            model_retention(workspace, "owner", "parent", "input", changed)
