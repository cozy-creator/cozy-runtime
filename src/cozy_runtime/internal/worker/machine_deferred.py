"""Install a published callee on its first selection (workspace deferred_installations)."""

from __future__ import annotations

import threading
from collections.abc import Mapping
from concurrent.futures import Future
from pathlib import Path
from typing import TYPE_CHECKING, Any

from cozy_runtime.internal import package_interface
from cozy_runtime.internal.canonical import Json
from cozy_runtime.internal.worker.machine_builtin import Preparing
from cozy_runtime.internal.worker.workspace import WorkspaceRefusal
from cozy_runtime.protocol import documents
from cozy_runtime.protocol import worker_pb2 as pb

if TYPE_CHECKING:
    from .session import Worker

#: A retained installation document: its placement row and installation id.
Installed = dict[str, Any]


def rows(capture: Mapping[str, object]) -> dict[str, pb.DeferredInstallation]:
    """The capture's deferred callees by key, each checked against its own release."""
    out: dict[str, pb.DeferredInstallation] = {}
    listed = capture.get("deferred_installations", [])
    if not isinstance(listed, list) or len(listed) > 128:
        raise WorkspaceRefusal("execution capture has unbounded deferred installations")
    for row in (documents.from_body(item, pb.DeferredInstallation) for item in listed):
        if not row.key or row.key in out or row.key != f"{row.package}@{row.release}":
            raise WorkspaceRefusal("execution capture repeats or misnames a deferred installation")
        out[row.key] = row
    return out


def interface(row: pb.DeferredInstallation | Mapping[str, object]) -> dict[str, Json]:
    """The release interface a deferred row carries, for binding before it is installed."""
    if not isinstance(row, pb.DeferredInstallation):  # machine_release_roots' facts document
        row = documents.from_body(row, pb.DeferredInstallation)
    if not row.preparation.package_interface:
        raise WorkspaceRefusal("deferred installation carries no release interface")
    return package_interface.read_bytes(row.preparation.package_interface, "deferred release")


class Deferred:
    """One installation per key: the first selection installs it, every concurrent selection
    waits for that one, and a failure fails exactly the calls that selected it."""

    def __init__(self, worker: Worker):
        self.worker = worker
        self.lock = threading.Lock()
        self.installs: dict[str, Future[Installed]] = {}

    def prepared(self, row: pb.DeferredInstallation, *, wait: bool = True) -> Installed:
        """The callee's retained installation. A call waits for it; a hint (`wait=False`)
        starts it and gets `Preparing` until it lands."""
        key = row.key
        with self.lock:
            future = self.installs.get(key)
            mine = future is None or (future.done() and future.exception() is not None)
            if mine:
                future = self.installs[key] = Future()
        assert future is not None
        if mine and wait:
            self._prepare(row, future)
        elif mine:
            threading.Thread(
                target=self._prepare, args=(row, future), name=f"deferred-{key}", daemon=True
            ).start()
        if not wait and not future.done():
            raise Preparing(key)
        return future.result()

    def _prepare(self, row: pb.DeferredInstallation, future: Future[Installed]) -> None:
        from .session import read_placement_set

        key = row.key
        try:
            request = pb.PreparePackageSetRequest()
            request.CopyFrom(row.preparation)
            request.install_root = str(Path(self.worker.options.install_root or ""))
            self.worker.note("preparation", f"installing {key} on its first selection")
            entries = read_placement_set(self.worker.prepare_package_set(request).placement_set)
            if len(entries) != 1 or entries[0].package.package != row.package:
                raise WorkspaceRefusal("deferred preparation returned another installation")
            future.set_result(
                {
                    "placement": documents.body(entries[0]),
                    "installation_id": entries[0].installation_id,
                }
            )
        except Exception as exc:  # the calls that selected it fail, nothing else
            future.set_exception(
                WorkspaceRefusal(f"deferred_installation_failed: {key}: {exc}"[:1024])
            )
