"""Authenticated local/Host entrypoints into the same workspace implementation."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

from cozy_runtime.internal.worker import workspace_memo
from cozy_runtime.internal.worker.workspace import Workspace, WorkspaceRefusal
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
