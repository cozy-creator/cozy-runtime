"""Claimed retained-artifact inventory and one-object upload using the existing mover."""

from __future__ import annotations

import hashlib
import io
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any, BinaryIO, cast
from urllib.parse import urlparse

from cozy_runtime import canonical_json
from cozy_runtime.internal import egress, fill
from cozy_runtime.internal.worker import workspace_byte_outputs
from cozy_runtime.internal.worker.weights import _LeaseReader
from cozy_runtime.internal.worker.workspace import Workspace, WorkspaceRefusal
from cozy_runtime.protocol import documents
from cozy_runtime.protocol import worker_pb2 as pb


class NativeArtifactTransfers:
    def __init__(
        self,
        workspace: Workspace,
        owner: Callable[[], str],
        send: Callable[[pb.WorkerFrame], None],
        valid: Callable[[pb.NativeArtifactTransfer], bool],
        *,
        allow_private_egress: bool = False,
    ) -> None:
        self.workspace, self.owner, self.send, self.valid = workspace, owner, send, valid
        self.allow_private_egress = allow_private_egress
        self.pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="native-artifact")
        self.lock = threading.Lock()
        self.active: dict[tuple[str, int, int, str, str, int], bytes] = {}

    def handle(self, command: pb.NativeArtifactTransfer) -> None:
        if not self.valid(command):
            raise WorkspaceRefusal("native artifact command has a stale claim")
        copied = pb.NativeArtifactTransfer.FromString(command.SerializeToString())
        key = (
            self.owner(),
            command.record_owner_epoch,
            command.control_stream_epoch,
            command.worker_boot_id,
            command.effect_id,
            command.command_id,
        )
        fingerprint = hashlib.sha256(copied.SerializeToString(deterministic=True)).digest()
        refusal = ""
        with self.lock:
            previous = self.active.get(key)
            if previous == fingerprint:
                return
            if previous is not None:
                refusal = "native_artifact_refused"
            elif len(self.active) >= 32:
                refusal = "native_artifact_busy"
            else:
                self.active[key] = fingerprint
        if refusal:
            result = self._status(copied)
            result.outcome = pb.WEIGHTS_UPLOAD_OUTCOME_REFUSED
            result.safe_code = refusal
            result.safe_detail = "native artifact command changed or has no available transfer slot"
            self.send(pb.WorkerFrame(native_artifact_transfer_status=result))
            return

        def run() -> None:
            try:
                result = self.execute(copied)
                self.send(pb.WorkerFrame(native_artifact_transfer_status=result))
            finally:
                with self.lock:
                    self.active.pop(key, None)

        self.pool.submit(run)

    def close(self) -> None:
        self.pool.shutdown(wait=False, cancel_futures=True)

    def _hold(self, command: pb.NativeArtifactTransfer) -> tuple[Any, str]:
        source = command.source
        manifest = documents.spell(command.manifest.digest)
        if (
            not self.valid(command)
            or not command.effect_id
            or len(command.effect_id) > 128
            or not command.command_id
            or len(command.manifest.digest) != 32
            or not command.manifest.length
        ):
            raise WorkspaceRefusal("native artifact command identity is incomplete")
        with self.workspace.locked() as db:
            held = db.execute(
                "SELECT * FROM holds WHERE id=? AND owner=?", (source.retention_id, self.owner())
            ).fetchone()
            if (
                held is None
                or held["state"] != "held"
                or held["transaction_id"] != source.weights_transaction_id
                or held["native_digest"] != source.tensorfs_receipt_digest
                or held["manifest"] != command.manifest.digest
                or held["manifest_length"] != command.manifest.length
            ):
                raise WorkspaceRefusal("native artifact has no matching held owner obligation")
            store = fill.store(self.workspace.store_root)
            lease = db.native(lambda: store.acquire_cozytensors(manifest))
            current = db.execute(
                "SELECT state FROM holds WHERE id=?", (source.retention_id,)
            ).fetchone()
            if current is None or current["state"] != "held":
                lease.release()
                raise WorkspaceRefusal("native artifact hold was released during acquisition")
            return lease, manifest

    @staticmethod
    def _status(command: pb.NativeArtifactTransfer) -> pb.NativeArtifactTransferStatus:
        # Correlate even refusals raised before a native source can be opened.
        result = pb.NativeArtifactTransferStatus(
            effect_id=command.effect_id,
            manifest=command.manifest,
            command_id=command.command_id,
            grant_revision=command.grant_revision,
            object_id=command.grant.object_id if command.HasField("grant") else "",
        )
        if command.HasField("byte_source"):
            result.byte_source.CopyFrom(command.byte_source)
        elif command.HasField("source"):
            result.source.CopyFrom(command.source)
        return result

    def execute(self, command: pb.NativeArtifactTransfer) -> pb.NativeArtifactTransferStatus:
        result = self._status(command)
        lease = None
        byte_context = None
        reader: Any = None
        try:
            if command.HasField("source") == command.HasField("byte_source"):
                raise WorkspaceRefusal("native artifact transfer requires exactly one source")
            if command.HasField("byte_source"):
                if (
                    not self.valid(command)
                    or not command.effect_id
                    or len(command.effect_id) > 128
                    or not command.command_id
                    or command.manifest != command.byte_source.source.manifest
                ):
                    raise WorkspaceRefusal("ordinary artifact transfer changed its exact subject")
                byte_context = workspace_byte_outputs.leased(
                    self.workspace, self.owner(), command.byte_source
                )
                store, lease, _ = byte_context.__enter__()
                manifest = documents.spell(command.manifest.digest)
            else:
                lease, manifest = self._hold(command)
                store = fill.store(self.workspace.store_root)
            held = store.manifest(manifest)
            manifest_bytes = bytes(held["manifest"])
            if (
                len(manifest_bytes) != command.manifest.length
                or hashlib.sha256(manifest_bytes).digest() != command.manifest.digest
            ):
                raise WorkspaceRefusal("native artifact manifest identity changed")
            objects = {manifest: command.manifest.length}
            for row in (
                store.walk(manifest)
                if byte_context is not None
                else store.walk_cozytensors(manifest)
            ):
                object_id, length = str(row["id"]), int(row["length"])
                if object_id in objects and objects[object_id] != length:
                    raise WorkspaceRefusal("native artifact closure changes an object length")
                objects[object_id] = length
            ordered = sorted(objects.items())
            result.closure_digest = hashlib.sha256(
                canonical_json.encode([[key, length] for key, length in ordered])
            ).digest()
            if not command.HasField("grant"):
                if not 0 < command.limit <= 128 or command.offset > len(ordered):
                    raise WorkspaceRefusal("native artifact inventory page is outside its bound")
                end = min(command.offset + command.limit, len(ordered))
                result.objects.extend(
                    pb.WeightsObjectRef(object_id=key, length=length)
                    for key, length in ordered[command.offset : end]
                )
                result.next_offset, result.has_more = end, end < len(ordered)
                return result
            grant = command.grant
            result.object_id = grant.object_id
            if not command.grant_revision or grant.length != objects.get(grant.object_id):
                raise WorkspaceRefusal("native artifact grant is outside its exact closure")
            if command.server_time_unix <= 0 or grant.expires_at_unix <= command.server_time_unix:
                result.safe_code = "weights_grant_expired"
                raise WorkspaceRefusal("native artifact grant expired before transfer")
            if len({row.name.lower() for row in grant.required_headers}) != len(
                grant.required_headers
            ):
                raise WorkspaceRefusal("native artifact grant repeats a header")
            if grant.object_id == manifest:
                reader = io.BytesIO(manifest_bytes)
            else:
                reader = _LeaseReader(
                    store.acquire(manifest, [(grant.object_id, grant.length)]),
                    grant.object_id,
                    grant.length,
                )
            host = urlparse(grant.url).hostname or ""
            receipt = egress.put_from(
                grant.url,
                egress.EgressPolicy(allowed_hosts=(host,), allow_private=self.allow_private_egress),
                cast(BinaryIO, reader),
                expected_length=grant.length,
                media_type="application/octet-stream",
                what="native artifact object",
                required_headers={row.name: row.value for row in grant.required_headers},
                accept_precondition_failed=True,
            )
            result.http_status = receipt.http_status
            result.transferred_bytes = receipt.length
            result.outcome = (
                pb.WEIGHTS_UPLOAD_OUTCOME_ALREADY_PRESENT
                if receipt.http_status == egress.HTTP_PRECONDITION_FAILED
                else pb.WEIGHTS_UPLOAD_OUTCOME_UPLOADED
            )
            if result.outcome == pb.WEIGHTS_UPLOAD_OUTCOME_UPLOADED:
                if receipt.sha256.hex() != grant.object_id.removeprefix("sha256:"):
                    raise WorkspaceRefusal("native artifact uploaded bytes changed identity")
                result.checksum_sha256 = grant.object_id
            return result
        except Exception as exc:
            result.outcome = pb.WEIGHTS_UPLOAD_OUTCOME_REFUSED
            result.safe_code = result.safe_code or getattr(exc, "code", "native_artifact_refused")
            # Never return grant URLs, credentials or arbitrary exception text.
            result.safe_detail = "native artifact inventory or transfer was refused"
            return result
        finally:
            if reader is not None:
                reader.close()
            if byte_context is not None:
                byte_context.__exit__(None, None, None)
            elif lease is not None:
                lease.release()
