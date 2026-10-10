"""Authenticated local/Host entrypoints into the same workspace implementation."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator

import grpc

from cozy_runtime.internal.worker import (
    artifact_transfer,
    machine_model_resolve,
    workspace_byte_outputs,
    workspace_input_trees,
    workspace_memo,
)
from cozy_runtime.internal.worker.servicer_context import ServicerContext
from cozy_runtime.internal.worker.workspace import Workspace, WorkspaceBusy, WorkspaceRefusal
from cozy_runtime.protocol import documents
from cozy_runtime.protocol import worker_pb2 as pb

_INTEGRITY_CODES = (
    "OBJECT_CORRUPT",
    "OBJECT_ID_MISMATCH",
    "LENGTH_MISMATCH",
    "NOT_REGULAR_FILE",
)


class Service:
    def __init__(
        self,
        workspace: Workspace,
        authorize: Callable[[pb.Claim], str],
        resolutions: machine_model_resolve.Resolutions,
        *,
        private_egress_allowed: Callable[[], bool] = lambda: False,
    ):
        self.workspace, self.authorize = workspace, authorize
        self.resolutions = resolutions
        #: Uploads may reach private addresses only for a machine whose own Hub is local.
        self.private_egress_allowed = private_egress_allowed
        self._transfers: dict[str, artifact_transfer.NativeArtifactTransfers] = {}

    def transfers(self, owner: str) -> artifact_transfer.NativeArtifactTransfers:
        """One owner's retained-artifact transfers on the machine connection. Its claim was
        checked by the call, so each command is valid as it stands."""
        held = self._transfers.get(owner)
        if held is None:
            held = self._transfers[owner] = artifact_transfer.NativeArtifactTransfers(
                self.workspace,
                lambda: owner,
                private_egress_allowed=self.private_egress_allowed,
            )
        return held


class WorkspaceRPC:
    workspace_service: Service | None

    def _workspace_call[T](
        self, claim: pb.Claim, context: ServicerContext, operation: Callable[[Service, str], T]
    ) -> T:
        service = self.workspace_service
        if service is None:
            context.abort(grpc.StatusCode.UNIMPLEMENTED, "workspace custody is unavailable")
        try:

            def guard() -> None:
                if not context.is_active():
                    raise WorkspaceRefusal("workspace call was canceled")
                service.authorize(claim)

            with service.workspace.authorized(guard):
                return operation(service, service.authorize(claim))
        except WorkspaceBusy as exc:
            context.abort(grpc.StatusCode.RESOURCE_EXHAUSTED, str(exc)[:512])
        except WorkspaceRefusal as exc:
            context.abort(grpc.StatusCode.FAILED_PRECONDITION, str(exc)[:512])
        except Exception as exc:
            # Retrying a corrupt object cannot establish custody. Keep native
            # paths/details private; the owner needs only the permanent code.
            code = getattr(exc, "code", None)
            if code in _INTEGRITY_CODES:
                context.abort(
                    grpc.StatusCode.FAILED_PRECONDITION, f"workspace integrity refused: {code}"
                )
            context.abort(grpc.StatusCode.UNAVAILABLE, "workspace custody operation failed")

    def RecordOperationResult(
        self, request: pb.RecordOperationResultCall, context: ServicerContext
    ) -> pb.RecordOperationResultResult:
        return self._workspace_call(
            request.claim,
            context,
            lambda s, owner: workspace_memo.record(s.workspace, owner, request),
        )

    def LookupOperation(
        self, request: pb.LookupOperationCall, context: ServicerContext
    ) -> pb.LookupOperationResult:
        return self._workspace_call(
            request.claim,
            context,
            lambda s, owner: workspace_memo.lookup(s.workspace, owner, request),
        )

    def PruneOperationCache(
        self, request: pb.PruneOperationCacheCall, context: ServicerContext
    ) -> pb.PruneOperationCacheResult:
        return self._workspace_call(
            request.claim, context, lambda s, owner: workspace_memo.prune(s.workspace, owner)
        )

    def WorkspaceRetainDerivedResult(
        self, request: pb.DerivedRetentionCall, context: ServicerContext
    ) -> pb.DerivedRetentionResult:
        return self._workspace_call(
            request.claim, context, lambda s, owner: s.workspace.retain(owner, request.request)
        )

    def WorkspaceReleaseDerivedRetention(
        self, request: pb.DerivedRetentionCall, context: ServicerContext
    ) -> pb.DerivedRetentionResult:
        return self._workspace_call(
            request.claim,
            context,
            lambda s, owner: s.workspace.retain(owner, request.request, release=True),
        )

    def WorkspaceReleaseDerivedResult(
        self, request: pb.DerivedResultReleaseCall, context: ServicerContext
    ) -> pb.DerivedResultReleaseResult:
        return self._workspace_call(
            request.claim,
            context,
            lambda s, owner: s.workspace.release_result(owner, request.request),
        )

    def WorkspaceRetainByteTree(
        self, request: pb.NativeByteRetentionCall, context: ServicerContext
    ) -> pb.NativeByteRetentionResult:
        return self._workspace_call(
            request.claim,
            context,
            lambda s, owner: workspace_byte_outputs.change_hold(
                s.workspace,
                owner,
                request.request,
                release=False,
            ),
        )

    def WorkspaceReleaseByteTree(
        self, request: pb.NativeByteRetentionCall, context: ServicerContext
    ) -> pb.NativeByteRetentionResult:
        return self._workspace_call(
            request.claim,
            context,
            lambda s, owner: workspace_byte_outputs.change_hold(
                s.workspace,
                owner,
                request.request,
                release=True,
            ),
        )

    def WorkspaceNativeArtifactTransfer(
        self, request: pb.NativeArtifactTransferCall, context: ServicerContext
    ) -> pb.NativeArtifactTransferStatus:
        return self._workspace_call(
            request.claim, context, lambda s, owner: s.transfers(owner).execute(request.request)
        )

    def WorkspaceForgetPackage(
        self, request: pb.ForgetPackageCall, context: ServicerContext
    ) -> pb.ForgetPackageResult:
        def forget(service: Service, _owner: str) -> pb.ForgetPackageResult:
            service.resolutions.forget(request.package)
            return pb.ForgetPackageResult()

        return self._workspace_call(request.claim, context, forget)

    def WorkspaceReadByteTreeObject(
        self, request: pb.NativeByteReadCall, context: ServicerContext
    ) -> Iterator[pb.NativeByteReadChunk]:
        service = self.workspace_service
        if service is None:
            context.abort(grpc.StatusCode.UNIMPLEMENTED, "workspace custody is unavailable")
        try:

            def guard() -> None:
                if not context.is_active():
                    raise WorkspaceRefusal("ordinary artifact read was canceled")

            # One proof per stream: the held closure cannot change under the lease.
            owner = service.authorize(request.claim)
            with (
                service.workspace.authorized(guard),
                workspace_byte_outputs.leased(service.workspace, owner, request.source) as (
                    store,
                    lease,
                    members,
                ),
            ):
                source = request.source.source
                manifest = documents.spell(source.manifest.digest)
                objects = {"sha256:" + m.blob.sha256: m.blob.length for m in members}
                objects[manifest] = source.manifest.length
                selected = documents.spell(request.object.digest)
                if (
                    objects.get(selected) != request.object.length
                    or request.offset > request.object.length
                ):
                    raise WorkspaceRefusal("ordinary artifact read is outside its held closure")
                manifest_body = (
                    bytes(store.manifest(manifest)["manifest"]) if selected == manifest else None
                )
                offset = request.offset
                while offset < request.object.length:
                    guard()
                    size = min(32 << 10, request.object.length - offset)
                    if manifest_body is not None:
                        data = manifest_body[offset : offset + size]
                    else:
                        buffer = bytearray(size)
                        lease.read_into(selected, request.object.length, offset, size, buffer)
                        data = bytes(buffer)
                    yield pb.NativeByteReadChunk(offset=offset, data=data)
                    offset += size
        except WorkspaceRefusal as exc:
            context.abort(grpc.StatusCode.FAILED_PRECONDITION, str(exc)[:512])
        except Exception as exc:
            code = getattr(exc, "code", None)
            if code in _INTEGRITY_CODES:
                context.abort(
                    grpc.StatusCode.FAILED_PRECONDITION, f"workspace integrity refused: {code}"
                )
            else:
                context.abort(grpc.StatusCode.UNAVAILABLE, "ordinary artifact read failed")

    def ImportInputTree(
        self, requests: Iterable[pb.InputTreeImportFrame], context: ServicerContext
    ) -> pb.NativeByteRetentionResult:
        iterator = iter(requests)
        first = next(iterator, None)
        if first is None or first.WhichOneof("body") != "header":
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT, "input tree needs its exact first header"
            )
        header = first.header
        return self._workspace_call(
            header.claim,
            context,
            lambda service, owner: workspace_input_trees.receive(
                service.workspace, owner, header, iterator
            ),
        )
