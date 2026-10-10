"""Native retained-artifact transfer, with actual TensorFS roots and HTTP object writes."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import cast

from cozy_runtime import canonical_json
from cozy_runtime.internal.worker import machine_model_defaults
from cozy_runtime.internal.worker import workspace_byte_outputs as outputs
from cozy_runtime.internal.worker.artifact_transfer import NativeArtifactTransfers
from cozy_runtime.internal.worker.machine_model_resolve import Resolutions
from cozy_runtime.internal.worker.machine_publication import PublicationAuthority
from cozy_runtime.internal.worker.session import Worker
from cozy_runtime.internal.worker.workspace import WorkspaceRefusal
from cozy_runtime.internal.worker.workspace_rpc import Service, WorkspaceRPC
from cozy_runtime.protocol import documents
from cozy_runtime.protocol import worker_pb2 as pb
from test_weights_upload import ObjectStore, serving
from test_workspace_byte_outputs import producing
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
        workspace, lambda: "owner", private_egress_allowed=lambda: True
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
        workspace, lambda: "owner", private_egress_allowed=lambda: True
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


def test_upload_refusal_keeps_object_identity_before_hold_validation(tmp_path: Path) -> None:
    _store, workspace, receipt = produced(tmp_path)
    source = pb.DerivedRetentionRequest(
        weights_transaction_id=receipt.weights_transaction_id,
        tensorfs_receipt_digest=documents.raw(receipt.tensorfs_receipt_digest),
        retention_id="sha256:" + "c" * 64,
    )
    held = workspace.retain("owner", source)
    transfers = NativeArtifactTransfers(workspace, lambda: "owner")
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


def test_byte_tree_upload_leases_only_the_object_it_sends(tmp_path: Path) -> None:
    """The tree's retention guards collection; a command opens just the member it sends.

    A member this command does not send is removed locally: a whole-tree lease would
    refuse on it. Releasing the retention still refuses the next command.
    """
    workspace, spec, _, _ = producing(tmp_path)
    bodies = {"a": b"alpha" * 4000, "b": b"bravo" * 4000}
    files = []
    for name, body in bodies.items():
        (tmp_path / name).write_bytes(body)
        files.append((name, tmp_path / name))
    manifest = canonical_json.encode(
        {
            "entries": [
                {
                    "path": name,
                    "kind": "file",
                    "blob": {"sha256": hashlib.sha256(body).hexdigest(), "length": len(body)},
                }
                for name, body in bodies.items()
            ]
        }
    )
    source = outputs.commit(
        workspace, "owner", "producer", 1, spec, "report", manifest, files, 1 << 20
    )
    hold = pb.NativeByteRetentionRequest(source=source, retention_id="sha256:" + "dd" * 32)
    outputs.change_hold(workspace, "owner", hold, release=False)
    unsent = hashlib.sha256(bodies["a"]).hexdigest()
    (workspace.store_root / "blobs" / unsent[:2] / unsent[2:4] / unsent).unlink()
    sent = hashlib.sha256(bodies["b"]).hexdigest()
    transfers = NativeArtifactTransfers(
        workspace, lambda: "owner", private_egress_allowed=lambda: True
    )
    sink = ObjectStore()
    with serving(sink) as url:
        command = pb.NativeArtifactTransfer(
            effect_id="attachment",
            command_id=1,
            byte_source=hold,
            manifest=source.manifest,
            grant_revision=1,
            server_time_unix=100,
            grant=pb.WeightsUploadGrant(
                object_id="sha256:" + sent,
                length=len(bodies["b"]),
                url=url,
                expires_at_unix=200,
            ),
        )
        upload = transfers.execute(command)
        assert upload.outcome == pb.WEIGHTS_UPLOAD_OUTCOME_UPLOADED and not upload.safe_code
        assert sink.body == bodies["b"]
        outputs.change_hold(workspace, "owner", hold, release=True)
        command.command_id = 2
        assert transfers.execute(command).outcome == pb.WEIGHTS_UPLOAD_OUTCOME_REFUSED


def test_the_machine_connection_pages_and_uploads_a_retained_output(tmp_path: Path) -> None:
    """`cozy run upload` over the machine connection: the claim authorizes the call, and the
    command pages the retained closure and uploads a granted object, with no control stream."""
    _store, workspace, receipt = produced(tmp_path)
    source = pb.DerivedRetentionRequest(
        weights_transaction_id=receipt.weights_transaction_id,
        tensorfs_receipt_digest=documents.raw(receipt.tensorfs_receipt_digest),
        retention_id="sha256:" + "e" * 64,
    )
    held = workspace.retain("owner", source)

    def authorize(claim: pb.Claim) -> str:
        if claim.record_owner_id != "owner":
            raise WorkspaceRefusal("claim names another owner")
        return "owner"

    class Machine(WorkspaceRPC):
        workspace_service = Service(
            workspace,
            authorize,
            Resolutions(),
            private_egress_allowed=lambda: True,
        )

    class Context:
        def is_active(self) -> bool:
            return True

        def abort(self, code: object, detail: str) -> None:
            raise AssertionError(f"{code}: {detail}")

    machine, claim = Machine(), pb.Claim(record_owner_id="owner")
    command = pb.NativeArtifactTransfer(
        effect_id="upload-one", source=source, manifest=held.manifest, command_id=1, limit=128
    )
    call = pb.NativeArtifactTransferCall(claim=claim, request=command)
    inventory = machine.WorkspaceNativeArtifactTransfer(call, Context())  # type: ignore[arg-type]
    assert not inventory.safe_code and inventory.objects and not inventory.has_more
    manifest_id = documents.spell(held.manifest.digest)
    row = next(row for row in inventory.objects if row.object_id != manifest_id)
    sink = ObjectStore()
    with serving(sink) as url:
        call.request.command_id, call.request.grant_revision = 2, 1
        call.request.server_time_unix = 100
        call.request.grant.CopyFrom(
            pb.WeightsUploadGrant(
                object_id=row.object_id, length=row.length, url=url, expires_at_unix=200
            )
        )
        upload = machine.WorkspaceNativeArtifactTransfer(call, Context())  # type: ignore[arg-type]
    assert upload.outcome == pb.WEIGHTS_UPLOAD_OUTCOME_UPLOADED and len(sink.body) == row.length

    # A machine whose own Hub is not local never uploads to a private address.
    class Production(WorkspaceRPC):
        workspace_service = Service(workspace, authorize, Resolutions())

    with serving(ObjectStore()) as url:
        call.request.command_id, call.request.grant.url = 3, url
        refused = Production().WorkspaceNativeArtifactTransfer(call, Context())  # type: ignore[arg-type]
    assert refused.outcome == pb.WEIGHTS_UPLOAD_OUTCOME_REFUSED, refused
    # Only a machine whose own Hub (its grant's origin) is on loopback is a development one.
    for grant, private in (
        (PublicationAuthority("http://127.0.0.1:8819", "w", "t"), True),
        (PublicationAuthority("https://tensorhub.example", "w", "t"), False),
        (None, False),
    ):
        worker = cast(
            Worker, SimpleNamespace(options=SimpleNamespace(publication_authority=grant, hubs=()))
        )
        assert machine_model_defaults.local_development(worker) is private, grant
    call.claim.record_owner_id = "stranger"
    try:
        machine.WorkspaceNativeArtifactTransfer(call, Context())  # type: ignore[arg-type]
    except AssertionError as refused:
        assert "claim names another owner" in str(refused)
    else:
        raise AssertionError("another owner's claim moved a retained output")


def test_cached_transfers_follow_current_hub_registration_and_revocation(tmp_path: Path) -> None:
    """An offline-started Service uses current policy for each actual native HTTP upload."""
    _store, workspace, receipt = produced(tmp_path)
    source = pb.DerivedRetentionRequest(
        weights_transaction_id=receipt.weights_transaction_id,
        tensorfs_receipt_digest=documents.raw(receipt.tensorfs_receipt_digest),
        retention_id="sha256:" + "c" * 64,
    )
    retained = workspace.retain("owner", source)
    access = tmp_path / "hub-access.json"
    worker = cast(
        Worker,
        SimpleNamespace(
            options=SimpleNamespace(
                publication_authority=None,
                hubs=(),
                hub_access_path=access,
            )
        ),
    )
    service = Service(
        workspace,
        lambda claim: "owner",
        Resolutions(),
        private_egress_allowed=lambda: machine_model_defaults.local_development(worker),
    )
    transfers = service.transfers("owner")
    command = pb.NativeArtifactTransfer(
        effect_id="late-hub",
        source=source,
        manifest=retained.manifest,
        command_id=1,
        limit=128,
    )
    inventory = transfers.execute(command)
    assert inventory.objects and not inventory.safe_code
    row = next(
        row
        for row in inventory.objects
        if row.object_id != documents.spell(retained.manifest.digest)
    )
    sink = ObjectStore()
    with serving(sink) as url:
        command.command_id = 2
        command.grant_revision = 1
        command.server_time_unix = 100
        command.grant.CopyFrom(
            pb.WeightsUploadGrant(
                object_id=row.object_id,
                length=row.length,
                url=url,
                expires_at_unix=200,
            )
        )
        assert transfers.execute(command).safe_code == "egress_scheme"
        assert not sink.received.is_set()

        def register(origin: str) -> None:
            access.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "hubs": [
                            {
                                "origin": origin,
                                "access_token": "owner-scoped-access",
                                "expires_at": 4102444800,
                            }
                        ],
                    }
                )
            )

        register(url)
        assert service.transfers("owner") is transfers  # Same cached object created offline.
        command.command_id = 3
        sent = transfers.execute(command)
        assert sent.outcome == pb.WEIGHTS_UPLOAD_OUTCOME_UPLOADED and not sent.safe_code
        assert sent.checksum_sha256 == row.object_id and len(sink.body) == row.length

        # A remaining public Hub does not retain the removed loopback Hub's permission.
        register("https://tensorhub.example")
        sink.received.clear()
        command.command_id = 4
        assert transfers.execute(command).safe_code == "egress_scheme"
        assert not sink.received.is_set()
        # HTTPS private endpoints are also denied, before attempting a TLS connection.
        command.grant.url = url.replace("http:", "https:")
        assert transfers.execute(command).safe_code == "egress_blocked_address"
        assert not sink.received.is_set()
        access.unlink()
        command.command_id = 5
        command.grant.url = url
        assert transfers.execute(command).safe_code == "egress_scheme"
        assert not sink.received.is_set()
