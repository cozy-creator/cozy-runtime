"""Ordinary output custody is native and survives independent recipient ownership."""

from __future__ import annotations

import hashlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import grpc
import msgspec
import pytest
import tensorfs

from cozy_runtime import canonical_json
from cozy_runtime.internal.executor_replies import AttemptReply
from cozy_runtime.internal.worker import workspace_byte_outputs as outputs
from cozy_runtime.internal.worker import workspace_memo
from cozy_runtime.internal.worker.artifact_transfer import NativeArtifactTransfers
from cozy_runtime.internal.worker.attempts import AttemptEngine, AttemptRecord
from cozy_runtime.internal.worker.machine_model_resolve import Resolutions
from cozy_runtime.internal.worker.workspace import Workspace, WorkspaceRefusal
from cozy_runtime.internal.worker.workspace_rpc import Service, WorkspaceRPC
from cozy_runtime.protocol import documents
from cozy_runtime.protocol import worker_pb2 as pb
from cozy_runtime.protocol import worker_pb2_grpc as rpc


def producing(tmp_path: Path) -> tuple[Workspace, bytes, bytes, Path]:
    workspace = Workspace(tmp_path / "store")
    raw, digest = documents.identity(pb.InvocationSpec(job=pb.JobInvocationSpec()))
    workspace.accept(
        "owner",
        pb.AttemptOffer(
            request_id="producer",
            attempt_ordinal=1,
            invocation_spec_digest=digest,
            invocation_spec_canonical_bytes=raw,
        ),
    )
    workspace.mark_running(
        "owner",
        pb.AttemptAccepted(
            request_id="producer",
            attempt_ordinal=1,
            invocation_spec_digest=digest,
        ),
    )
    path = tmp_path / "report.json"
    data = canonical_json.encode({"report": "observed" * 12000})
    path.write_bytes(data)
    manifest = canonical_json.encode(
        {
            "entries": [
                {
                    "path": "payload",
                    "kind": "file",
                    "blob": {
                        "sha256": hashlib.sha256(data).hexdigest(),
                        "length": len(data),
                    },
                }
            ]
        }
    )
    return workspace, digest, manifest, path


def test_reserved_capture_grant_commits_native_tree_without_author_asset(tmp_path: Path) -> None:
    workspace, digest, _, _ = producing(tmp_path)
    spool = tmp_path / "spool"
    captured = spool / "runtime.capture"
    captured.mkdir(parents=True)
    metadata, sketches = b'{"capture":"observed"}', b"\0" * 16
    (captured / "capture.json").write_bytes(metadata)
    (captured / "sketches.f32").write_bytes(sketches)
    capture = {
        "root": str(captured),
        "output_id": "runtime.capture",
        "length": len(metadata) + len(sketches),
        "content_digest": "sha256:" + hashlib.sha256(metadata + sketches).hexdigest(),
    }
    attempt = SimpleNamespace(
        job=None,
        weights_receipts={},
        grant=SimpleNamespace(outputs={"runtime.capture": object()}, expires_at_unix=0),
        spec={"capture": {"components": ["model"], "steps": [0]}},
        spool=spool,
        request_id="producer",
        attempt=1,
        digest=digest,
    )
    engine = SimpleNamespace(
        tensorfs_root=workspace.store_root,
        calls=SimpleNamespace(),
        owner_scope=lambda: "owner",
        _encode_frames=lambda *args: None,
        _publication=lambda *args: SimpleNamespace(document=lambda: {}, digest=lambda: b"r" * 32),
        records=SimpleNamespace(append=lambda *args: None),
        products=None,
    )
    # Only the post-processing collaborators are needed; TensorFS custody is real.
    post = cast(AttemptEngine, engine)
    record = cast(AttemptRecord, attempt)
    result, error = AttemptEngine._tail(
        post, record, msgspec.convert({"ok": True, "capture": capture}, AttemptReply)
    )
    assert error is None and result is not None
    assert [entry.output_id for entry in result.outputs] == ["runtime.capture"]
    assert result.outputs[0].native_tree.content_bytes == len(metadata) + len(sketches)
    missing, error = AttemptEngine._tail(post, record, AttemptReply(ok=True))
    assert missing is None and error is not None and error.startswith("capture_output_absent:")


def test_real_tree_commit_replay_retention_gc_and_release(tmp_path: Path) -> None:
    workspace, spec, manifest, path = producing(tmp_path)
    data = path.read_bytes()
    source = outputs.commit(
        workspace, "owner", "producer", 1, spec, "report", manifest, [("payload", path)], 200000
    )
    path.unlink()
    reopened = Workspace(workspace.store_root)
    assert (
        outputs.commit(
            reopened, "owner", "producer", 1, spec, "report", manifest, [("payload", path)], 200000
        )
        == source
    )
    hold = pb.NativeByteRetentionRequest(source=source, retention_id="sha256:" + "aa" * 32)
    assert not outputs.change_hold(reopened, "owner", hold, release=False).released
    outcome_bytes, outcome_digest = documents.identity(
        pb.AttemptOutcomeBody(
            request_id="producer",
            attempt_ordinal=1,
            invocation_spec_digest=documents.spell(spec),
            status=pb.OUTCOME_STATUS_SUCCEEDED,
            execution_started=True,
            output_manifest=pb.OutputManifest(
                outputs=[
                    pb.OutputEntry(
                        output_id="report",
                        digest=hashlib.sha256(data).digest(),
                        length=len(data),
                        mime_type="application/json",
                        native_tree=source,
                    )
                ]
            ),
        )
    )
    reopened.outcome(
        "owner",
        pb.AttemptOutcome(
            request_id="producer",
            attempt_ordinal=1,
            invocation_spec_digest=spec,
            outcome_id="outcome",
            outcome_digest=outcome_digest,
            outcome_canonical_bytes=outcome_bytes,
        ),
    )
    ack = pb.AttemptOutcomeAck(
        request_id="producer",
        attempt_ordinal=1,
        invocation_spec_digest=spec,
        outcome_id="outcome",
        outcome_digest=outcome_digest,
    )
    computation = b"k" * 32
    recorded = workspace_memo.record(
        reopened,
        "owner",
        pb.RecordOperationResultCall(
            computation_digest=computation,
            request_id="producer",
            attempt_ordinal=1,
            invocation_spec_digest=spec,
            outcome_id="outcome",
            outcome_digest=outcome_digest,
        ),
    )
    assert recorded.recorded
    reopened.acknowledge("owner", ack)
    tensorfs.gc(str(workspace.store_root))
    with outputs.leased(reopened, "owner", hold) as (_, lease, members):
        assert len(members) == 1
        actual = bytearray(len(data))
        lease.read_into("sha256:" + members[0].blob.sha256, len(data), 0, len(data), actual)
        assert actual == data
    memo = workspace_memo.lookup(
        reopened,
        "owner",
        pb.LookupOperationCall(
            computation_digest=computation,
            consumer_request_id="fresh-script",
        ),
    )
    assert memo.found and len(memo.byte_retentions) == 1 and not memo.retentions
    assert memo.byte_retentions[0].source == source
    consumer = pb.NativeByteRetentionRequest(
        source=source, retention_id=memo.byte_retentions[0].retention_id
    )
    with outputs.leased(reopened, "owner", consumer):
        pass
    assert outputs.change_hold(reopened, "owner", hold, release=True).released
    with pytest.raises(WorkspaceRefusal):
        outputs.change_hold(reopened, "owner", hold, release=False)
    with pytest.raises(WorkspaceRefusal), outputs.leased(reopened, "owner", hold):
        pytest.fail("released bytes became readable")
    # A fresh script must receive a miss if a retained cache result lost bytes.
    # Do not disguise this with an implementation-key change or a second store.
    object_id = hashlib.sha256(data).hexdigest()
    blob = workspace.store_root / "blobs" / object_id[:2] / object_id[2:4] / object_id
    blob.unlink()
    missing = workspace_memo.lookup(
        Workspace(workspace.store_root),
        "owner",
        pb.LookupOperationCall(
            computation_digest=computation, consumer_request_id="after-storage-loss"
        ),
    )
    assert not missing.found and not missing.byte_retentions


def test_changed_output_and_unowned_retention_refuse(tmp_path: Path) -> None:
    workspace, spec, manifest, path = producing(tmp_path)
    source = outputs.commit(
        workspace, "owner", "producer", 1, spec, "report", manifest, [("payload", path)], 200000
    )
    with pytest.raises(WorkspaceRefusal, match="changed"):
        outputs.commit(
            workspace,
            "owner",
            "producer",
            1,
            spec,
            "report",
            manifest.replace(b"payload", b"another"),
            [("another", path)],
            200000,
        )
    hold = pb.NativeByteRetentionRequest(source=source, retention_id="sha256:" + "bb" * 32)
    with pytest.raises(WorkspaceRefusal, match="not retained"):
        outputs.change_hold(workspace, "another-owner", hold, release=False)
    hold.source.content_bytes += 1
    with pytest.raises(WorkspaceRefusal):
        outputs.change_hold(workspace, "owner", hold, release=False)


def test_release_before_acquire_cannot_resurrect_recipient(tmp_path: Path) -> None:
    workspace, spec, manifest, path = producing(tmp_path)
    source = outputs.commit(
        workspace, "owner", "producer", 1, spec, "report", manifest, [("payload", path)], 200000
    )
    hold = pb.NativeByteRetentionRequest(source=source, retention_id="sha256:" + "cc" * 32)
    assert outputs.change_hold(workspace, "owner", hold, release=True).released
    with pytest.raises(WorkspaceRefusal, match="released"):
        outputs.change_hold(workspace, "owner", hold, release=False)


def test_real_native_tree_rpc_stream_is_membership_checked(tmp_path: Path) -> None:
    workspace, spec, manifest, path = producing(tmp_path)
    data = path.read_bytes()
    source = outputs.commit(
        workspace, "owner", "producer", 1, spec, "report", manifest, [("payload", path)], 200000
    )

    class Handler(WorkspaceRPC, rpc.RuntimePreparationServicer):  # type: ignore[misc] # generated gRPC base
        pass

    def authorize(claim: pb.Claim) -> str:
        if claim.worker_id != "worker":
            raise WorkspaceRefusal("wrong claimed worker")
        return "owner"

    handler = Handler()
    handler.workspace_service = Service(
        workspace, authorize, Resolutions()
    )
    pool = ThreadPoolExecutor(max_workers=2)
    server = grpc.server(pool)
    rpc.add_RuntimePreparationServicer_to_server(handler, server)
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()
    try:
        with grpc.insecure_channel(f"127.0.0.1:{port}") as channel:
            client = rpc.RuntimePreparationStub(channel)
            hold = pb.NativeByteRetentionRequest(source=source, retention_id="sha256:" + "dd" * 32)
            claim = pb.Claim(worker_id="worker")
            held = client.WorkspaceRetainByteTree(
                pb.NativeByteRetentionCall(claim=claim, request=hold)
            )
            assert held.source == source and not held.released
            transfers = NativeArtifactTransfers(workspace, lambda: "owner")
            inventory = transfers.execute(
                pb.NativeArtifactTransfer(
                    effect_id="attachment",
                    command_id=1,
                    byte_source=hold,
                    manifest=source.manifest,
                    limit=32,
                )
            )
            assert not inventory.safe_code and inventory.byte_source == hold
            assert not inventory.HasField("source")
            assert {obj.object_id for obj in inventory.objects} == {
                documents.spell(source.manifest.digest),
                documents.spell(hashlib.sha256(data).digest()),
            }
            read = pb.NativeByteReadCall(
                claim=claim,
                source=hold,
                object=pb.Ref(digest=hashlib.sha256(data).digest(), length=len(data)),
            )
            chunks = list(client.WorkspaceReadByteTreeObject(read))
            assert b"".join(c.data for c in chunks) == data
            assert len(chunks) > 1 and all(0 < len(c.data) <= 32 << 10 for c in chunks)
            read.offset = 17
            assert b"".join(c.data for c in client.WorkspaceReadByteTreeObject(read)) == data[17:]
            read.object.digest = b"x" * 32
            with pytest.raises(grpc.RpcError):
                list(client.WorkspaceReadByteTreeObject(read))
            client.WorkspaceReleaseByteTree(pb.NativeByteRetentionCall(claim=claim, request=hold))
            read.object.digest = hashlib.sha256(data).digest()
            with pytest.raises(grpc.RpcError):
                list(client.WorkspaceReadByteTreeObject(read))
    finally:
        server.stop(0).wait()
        pool.shutdown()


@pytest.mark.parametrize("fault", ["corrupt", "directory"])
def test_native_integrity_fault_is_permanent_rpc_refusal(tmp_path: Path, fault: str) -> None:
    workspace, spec, manifest, path = producing(tmp_path)
    data = path.read_bytes()
    source = outputs.commit(
        workspace, "owner", "producer", 1, spec, "report", manifest, [("payload", path)], 200000
    )
    digest = hashlib.sha256(data).hexdigest()
    blob = workspace.store_root / "blobs" / digest[:2] / digest[2:4] / digest
    blob.unlink()
    if fault == "directory":
        blob.mkdir()
        expected = "NOT_REGULAR_FILE"
    else:
        blob.write_bytes(b"x" * len(data))
        expected = "OBJECT_CORRUPT"

    class Handler(WorkspaceRPC, rpc.RuntimePreparationServicer):  # type: ignore[misc] # generated gRPC base
        pass

    handler = Handler()
    handler.workspace_service = Service(
        workspace, lambda _: "owner", Resolutions()
    )
    pool = ThreadPoolExecutor(max_workers=2)
    server = grpc.server(pool)
    rpc.add_RuntimePreparationServicer_to_server(handler, server)
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()
    try:
        with grpc.insecure_channel(f"127.0.0.1:{port}") as channel:
            client = rpc.RuntimePreparationStub(channel)
            with pytest.raises(grpc.RpcError) as refused:
                client.WorkspaceRetainByteTree(
                    pb.NativeByteRetentionCall(
                        claim=pb.Claim(worker_id="worker"),
                        request=pb.NativeByteRetentionRequest(
                            source=source, retention_id="sha256:" + "ee" * 32
                        ),
                    ),
                    timeout=10,
                )
            assert refused.value.code() == grpc.StatusCode.FAILED_PRECONDITION
            assert refused.value.details() == f"workspace integrity refused: {expected}"
            assert str(tmp_path) not in refused.value.details()
    finally:
        server.stop(0).wait()
        pool.shutdown(wait=True)
