"""Claimed retained-artifact inventory and one-object upload using the existing mover."""

from __future__ import annotations

import hashlib
import io
import threading
from collections import OrderedDict
from collections.abc import Callable
from typing import BinaryIO, cast
from urllib.parse import urlparse

from cozy_runtime import canonical_json
from cozy_runtime.internal import egress, fill
from cozy_runtime.internal.worker import workspace_byte_outputs
from cozy_runtime.internal.worker.weights import _LeaseReader
from cozy_runtime.internal.worker.workspace import Workspace, WorkspaceRefusal
from cozy_runtime.protocol import documents
from cozy_runtime.protocol import worker_pb2 as pb

# Recent content-addressed closures. One upload effect sends one command per object.
_CLOSURES = 8


class _Closure:
    __slots__ = ("digest", "manifest", "objects", "ordered")

    def __init__(self, manifest: bytes, objects: dict[str, int]) -> None:
        self.manifest, self.objects = manifest, objects
        self.ordered = sorted(objects.items())
        self.digest = hashlib.sha256(
            canonical_json.encode([[key, length] for key, length in self.ordered])
        ).digest()


class NativeArtifactTransfers:
    def __init__(
        self,
        workspace: Workspace,
        owner: Callable[[], str],
        *,
        private_egress_allowed: Callable[[], bool] = lambda: False,
    ) -> None:
        self.workspace, self.owner = workspace, owner
        self.private_egress_allowed = private_egress_allowed
        self.lock = threading.Lock()
        self.closures: OrderedDict[tuple[str, bool], _Closure] = OrderedDict()

    def _held(self, command: pb.NativeArtifactTransfer) -> Callable[[], bool]:
        """Check the retained owner obligation; the native retention keeps the closure.

        No closure-wide lease: that opened every object for every object sent.
        """
        source = command.source
        if (
            not command.effect_id
            or len(command.effect_id) > 128
            or not command.command_id
            or len(command.manifest.digest) != 32
            or not command.manifest.length
        ):
            raise WorkspaceRefusal("native artifact command identity is incomplete")

        def held() -> bool:
            return self.workspace.model_held(self.owner(), source, command.manifest)

        if not held():
            raise WorkspaceRefusal("native artifact has no matching held owner obligation")
        return held

    def _closure(self, store: fill.Store, manifest: str, length: int, trees: bool) -> _Closure:
        key = manifest, trees
        with self.lock:
            cached = self.closures.get(key)
            if cached is not None:
                self.closures.move_to_end(key)
                return cached
        body = store.manifest(manifest)["manifest"]
        if len(body) != length or "sha256:" + hashlib.sha256(body).hexdigest() != manifest:
            raise WorkspaceRefusal("native artifact manifest identity changed")
        objects = {manifest: length}
        for row in store.walk(manifest) if trees else store.walk_cozytensors(manifest):
            object_id, size = str(row["id"]), int(row["length"])
            if object_id in objects and objects[object_id] != size:
                raise WorkspaceRefusal("native artifact closure changes an object length")
            objects[object_id] = size
        closure = _Closure(body, objects)
        with self.lock:
            self.closures[key] = closure
            while len(self.closures) > _CLOSURES:
                self.closures.popitem(last=False)
        return closure

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
        reader: io.BytesIO | _LeaseReader | None = None
        try:
            if command.HasField("source") == command.HasField("byte_source"):
                raise WorkspaceRefusal("native artifact transfer requires exactly one source")
            trees = command.HasField("byte_source")
            if trees:
                if (
                    not command.effect_id
                    or len(command.effect_id) > 128
                    or not command.command_id
                    or command.manifest != command.byte_source.source.manifest
                ):
                    raise WorkspaceRefusal("ordinary artifact transfer changed its exact subject")
                store, held = workspace_byte_outputs.retained(
                    self.workspace, self.owner(), command.byte_source
                )
            else:
                held = self._held(command)
                store = fill.store(self.workspace.store_root)
            manifest = documents.spell(command.manifest.digest)
            closure = self._closure(store, manifest, command.manifest.length, trees)
            objects, ordered = closure.objects, closure.ordered
            result.closure_digest = closure.digest
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
                reader = io.BytesIO(closure.manifest)
            else:
                reader = _LeaseReader(
                    store.acquire(manifest, [(grant.object_id, grant.length)]),
                    grant.object_id,
                    grant.length,
                )
                if not held():
                    raise WorkspaceRefusal("native artifact hold was released during acquisition")
            host = urlparse(grant.url).hostname or ""
            receipt = egress.put_from(
                grant.url,
                egress.EgressPolicy(
                    allowed_hosts=(host,), allow_private=self.private_egress_allowed()
                ),
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
