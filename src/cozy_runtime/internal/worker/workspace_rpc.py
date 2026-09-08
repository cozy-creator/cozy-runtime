"""Authenticated local/Host entrypoints into the same workspace implementation."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

from cozy_runtime.internal.worker import workspace_byte_outputs, workspace_memo
from cozy_runtime.internal.worker.workspace import Workspace, WorkspaceRefusal
from cozy_runtime.protocol import documents
from cozy_runtime.protocol import worker_pb2 as pb


class Service:
    def __init__(
        self,
        workspace: Workspace,
        authorize: Callable[[pb.Claim], str],
        ready: Callable[[pb.WeightsIntentReadyCall], pb.WeightsHostAck],
    ):
        self.workspace, self.authorize, self.ready = workspace, authorize, ready


class WorkspaceRPC:
    workspace_service: Service | None

    def _workspace_call(
        self, request: Any, context: Any, operation: Callable[[Service, str], Any]
    ) -> Any:
        import grpc

        service = self.workspace_service
        if service is None:
            return context.abort(grpc.StatusCode.UNIMPLEMENTED, "workspace custody is unavailable")
        try:

            def guard() -> None:
                if not context.is_active():
                    raise WorkspaceRefusal("workspace call was canceled")
                service.authorize(request.claim)

            with service.workspace.authorized(guard):
                return operation(service, service.authorize(request.claim))
        except WorkspaceRefusal as exc:
            return context.abort(grpc.StatusCode.FAILED_PRECONDITION, str(exc)[:512])
        except Exception:
            return context.abort(grpc.StatusCode.UNAVAILABLE, "workspace custody operation failed")

    def RecordOperationResult(self, request: pb.RecordOperationResultCall, context: Any) -> Any:
        return self._workspace_call(
            request, context, lambda s, owner: workspace_memo.record(s.workspace, owner, request)
        )

    def LookupOperation(self, request: pb.LookupOperationCall, context: Any) -> Any:
        return self._workspace_call(
            request, context, lambda s, owner: workspace_memo.lookup(s.workspace, owner, request)
        )

    def PruneOperationCache(self, request: pb.PruneOperationCacheCall, context: Any) -> Any:
        return self._workspace_call(
            request, context, lambda s, owner: workspace_memo.prune(s.workspace, owner)
        )

    def WorkspaceRetainDerivedResult(self, request: pb.DerivedRetentionCall, context: Any) -> Any:
        return self._workspace_call(
            request, context, lambda s, owner: s.workspace.retain(owner, request.request)
        )

    def WorkspaceReleaseDerivedRetention(
        self, request: pb.DerivedRetentionCall, context: Any
    ) -> Any:
        return self._workspace_call(
            request,
            context,
            lambda s, owner: s.workspace.retain(owner, request.request, release=True),
        )

    def WorkspaceReleaseDerivedResult(
        self, request: pb.DerivedResultReleaseCall, context: Any
    ) -> Any:
        return self._workspace_call(
            request, context, lambda s, owner: s.workspace.release_result(owner, request.request)
        )

    def WorkspaceRetainByteTree(self, request: pb.NativeByteRetentionCall, context: Any) -> Any:
        return self._workspace_call(
            request,
            context,
            lambda s, owner: workspace_byte_outputs.change_hold(
                s.workspace,
                owner,
                request.request,
                release=False,
            ),
        )

    def WorkspaceReleaseByteTree(self, request: pb.NativeByteRetentionCall, context: Any) -> Any:
        return self._workspace_call(
            request,
            context,
            lambda s, owner: workspace_byte_outputs.change_hold(
                s.workspace,
                owner,
                request.request,
                release=True,
            ),
        )

    def WorkspaceReadByteTreeObject(
        self, request: pb.NativeByteReadCall, context: Any
    ) -> Iterable[pb.NativeByteReadChunk]:
        import grpc

        service = self.workspace_service
        if service is None:
            context.abort(grpc.StatusCode.UNIMPLEMENTED, "workspace custody is unavailable")
            return
        try:

            def guard() -> None:
                if not context.is_active():
                    raise WorkspaceRefusal("ordinary artifact read was canceled")
                service.authorize(request.claim)

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
                objects = {"sha256:" + m["blob"]["sha256"]: m["blob"]["length"] for m in members}
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
        except Exception:
            context.abort(grpc.StatusCode.UNAVAILABLE, "ordinary artifact read failed")

    def WorkspaceWeightsIntentReady(self, request: pb.WeightsIntentReadyCall, context: Any) -> Any:
        return self._workspace_call(request, context, lambda s, _: s.ready(request))

    def ImportWorkspace(self, requests: Iterable[pb.WorkspaceImportFrame], context: Any) -> Any:
        import grpc

        service = self.workspace_service
        if service is None:
            return context.abort(grpc.StatusCode.UNIMPLEMENTED, "workspace custody is unavailable")
        from cozy_runtime.internal.worker import workspace_import

        try:
            return workspace_import.receive(service.workspace, requests, service.authorize)
        except WorkspaceRefusal as exc:
            return context.abort(grpc.StatusCode.FAILED_PRECONDITION, str(exc)[:512])
        except Exception:
            return context.abort(grpc.StatusCode.UNAVAILABLE, "workspace import failed")

    def ActivateWorkspaceImport(self, request: pb.WorkspaceActivationCall, context: Any) -> Any:
        from cozy_runtime.internal.worker import workspace_import

        return self._workspace_call(
            request,
            context,
            lambda s, _: workspace_import.activate(s.workspace, request, s.authorize),
        )
