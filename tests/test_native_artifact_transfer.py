"""Native retained-artifact transfer, with actual TensorFS roots and HTTP object writes."""

from pathlib import Path

from cozy_runtime.internal.worker.artifact_transfer import NativeArtifactTransfers
from cozy_runtime.protocol import documents
from cozy_runtime.protocol import worker_pb2 as pb
from test_weights_upload import ObjectStore, serving
from test_workspace_custody import produced


def test_native_artifact_inventory_upload_replay_and_exact_grant(tmp_path: Path) -> None:
    _store, workspace, receipt = produced(tmp_path)
    source = pb.DerivedRetentionRequest(
        weights_transaction_id=receipt.weights_transaction_id,
        tensorfs_receipt_digest=documents.raw(receipt.tensorfs_receipt_digest),
        retention_id="sha256:" + "f" * 64,
    )
    held = workspace.retain("owner", source)
    transfers = NativeArtifactTransfers(
        workspace, lambda: "owner", lambda _: None, lambda _: True, allow_private_egress=True
    )
    command = pb.NativeArtifactTransfer(
        effect_id="effect-one", source=source, manifest=held.manifest, command_id=1, limit=128
    )
    inventory = transfers.execute(command)
    assert not inventory.safe_code and inventory.objects and not inventory.has_more
    manifest_id = documents.spell(held.manifest.digest)
    object_row = next(row for row in inventory.objects if row.object_id != manifest_id)
    sink = ObjectStore()
    with serving(sink) as url:
        command.command_id = 2
        command.grant_revision = 1
        command.server_time_unix = 100
        command.grant.CopyFrom(
            pb.WeightsUploadGrant(
                object_id=object_row.object_id,
                length=object_row.length,
                url=url,
                expires_at_unix=200,
            )
        )
        upload = transfers.execute(command)
        assert upload.outcome == pb.WEIGHTS_UPLOAD_OUTCOME_UPLOADED and not upload.safe_code
        assert upload.checksum_sha256 == object_row.object_id
        assert len(sink.body) == object_row.length
        # Lost-reply retry and another publication use the same immutable object;
        # no old per-producer upload ledger suppresses a new effect operation.
        sink.status = 412
        command.command_id = 3
        replay = transfers.execute(command)
        assert replay.outcome == pb.WEIGHTS_UPLOAD_OUTCOME_ALREADY_PRESENT
        command.effect_id = "effect-two"
        command.command_id = 4
        assert transfers.execute(command).outcome == pb.WEIGHTS_UPLOAD_OUTCOME_ALREADY_PRESENT
        command.grant.expires_at_unix = 100
        assert transfers.execute(command).safe_code == "weights_grant_expired"
        command.grant.expires_at_unix = 200
        command.grant.object_id = "sha256:" + "0" * 64
        assert transfers.execute(command).outcome == pb.WEIGHTS_UPLOAD_OUTCOME_REFUSED
        command.grant.object_id = object_row.object_id
        command.manifest.length += 1
        assert transfers.execute(command).outcome == pb.WEIGHTS_UPLOAD_OUTCOME_REFUSED
        command.manifest.CopyFrom(held.manifest)
        workspace.retain("owner", source, release=True)
        assert transfers.execute(command).outcome == pb.WEIGHTS_UPLOAD_OUTCOME_REFUSED
    transfers.pool.shutdown(wait=True)


def test_native_upload_lost_http_response_retries_same_bytes(tmp_path: Path) -> None:
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    _store, workspace, receipt = produced(tmp_path)
    source = pb.DerivedRetentionRequest(
        weights_transaction_id=receipt.weights_transaction_id,
        tensorfs_receipt_digest=documents.raw(receipt.tensorfs_receipt_digest),
        retention_id="sha256:" + "e" * 64,
    )
    held = workspace.retain("owner", source)
    transfers = NativeArtifactTransfers(
        workspace, lambda: "owner", lambda _: None, lambda _: True, allow_private_egress=True
    )
    command = pb.NativeArtifactTransfer(
        effect_id="lost-response", source=source, manifest=held.manifest, command_id=1, limit=128
    )
    inventory = transfers.execute(command)
    row = next(
        row for row in inventory.objects if row.object_id != documents.spell(held.manifest.digest)
    )
    bodies: list[bytes] = []

    class Handler(BaseHTTPRequestHandler):
        def do_PUT(self) -> None:
            bodies.append(self.rfile.read(int(self.headers["Content-Length"])))
            if len(bodies) == 1:
                self.connection.shutdown(2)
                self.connection.close()
                return
            self.send_response(412)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def log_message(self, *_: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        command.grant.CopyFrom(
            pb.WeightsUploadGrant(
                object_id=row.object_id,
                length=row.length,
                url=f"http://127.0.0.1:{server.server_port}/object",
                expires_at_unix=200,
            )
        )
        command.grant_revision = 1
        command.server_time_unix = 100
        result = transfers.execute(command)
        assert result.outcome == pb.WEIGHTS_UPLOAD_OUTCOME_ALREADY_PRESENT
        assert len(bodies) == 2 and bodies[0] == bodies[1] and len(bodies[0]) == row.length
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
        transfers.close()


def test_upload_refusal_keeps_object_identity_before_hold_validation(tmp_path: Path) -> None:
    _store, workspace, receipt = produced(tmp_path)
    source = pb.DerivedRetentionRequest(
        weights_transaction_id=receipt.weights_transaction_id,
        tensorfs_receipt_digest=documents.raw(receipt.tensorfs_receipt_digest),
        retention_id="sha256:" + "c" * 64,
    )
    held = workspace.retain("owner", source)
    transfers = NativeArtifactTransfers(workspace, lambda: "owner", lambda _: None, lambda _: True)
    try:
        command = pb.NativeArtifactTransfer(
            effect_id="refused-effect",
            source=source,
            manifest=held.manifest,
            command_id=7,
            grant_revision=1,
            grant=pb.WeightsUploadGrant(object_id="sha256:" + "3" * 64, length=4),
        )
        workspace.retain("owner", source, release=True)
        refused = transfers.execute(command)
        assert refused.outcome == pb.WEIGHTS_UPLOAD_OUTCOME_REFUSED
        assert refused.object_id == command.grant.object_id
        assert refused.source == command.source and refused.manifest == command.manifest
        assert refused.command_id == command.command_id
    finally:
        transfers.close()


def test_reclaimed_control_stream_does_not_drop_reused_command_number(tmp_path: Path) -> None:
    import queue
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    _store, workspace, receipt = produced(tmp_path)
    source = pb.DerivedRetentionRequest(
        weights_transaction_id=receipt.weights_transaction_id,
        tensorfs_receipt_digest=documents.raw(receipt.tensorfs_receipt_digest),
        retention_id="sha256:" + "b" * 64,
    )
    held = workspace.retain("owner", source)
    responses: queue.Queue[pb.WorkerFrame] = queue.Queue()
    active_epoch = [1]
    transfers = NativeArtifactTransfers(
        workspace,
        lambda: "owner",
        responses.put,
        lambda command: command.control_stream_epoch == active_epoch[0],
        allow_private_egress=True,
    )
    received = threading.Event()
    finish = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def do_PUT(self) -> None:
            self.rfile.read(int(self.headers["Content-Length"]))
            received.set()
            finish.wait(10)
            self.send_response(200)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def log_message(self, *_: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    serving_thread = threading.Thread(target=server.serve_forever, daemon=True)
    serving_thread.start()
    try:
        old = pb.NativeArtifactTransfer(
            effect_id="reclaimed-effect",
            source=source,
            manifest=held.manifest,
            record_owner_epoch=1,
            control_stream_epoch=1,
            worker_boot_id="boot",
            command_id=1,
            grant_revision=1,
            server_time_unix=100,
            grant=pb.WeightsUploadGrant(
                object_id=documents.spell(held.manifest.digest),
                length=held.manifest.length,
                url=f"http://127.0.0.1:{server.server_port}/object",
                expires_at_unix=200,
            ),
        )
        transfers.handle(old)
        assert received.wait(5)
        active_epoch[0] = 2
        new = pb.NativeArtifactTransfer(
            effect_id=old.effect_id,
            source=source,
            manifest=held.manifest,
            record_owner_epoch=1,
            control_stream_epoch=2,
            worker_boot_id="boot",
            command_id=1,
            limit=128,
        )
        transfers.handle(new)
        try:
            answer = responses.get(timeout=2).native_artifact_transfer_status
        except queue.Empty:
            raise AssertionError(
                "new control command was hidden by the old active command"
            ) from None
        assert answer.command_id == 1 and answer.objects and not answer.safe_code
    finally:
        finish.set()
        transfers.pool.shutdown(wait=True)
        server.shutdown()
        server.server_close()
        serving_thread.join()
