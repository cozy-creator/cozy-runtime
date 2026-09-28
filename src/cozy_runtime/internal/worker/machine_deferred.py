"""Install a published callee on its first selection (workspace deferred_installations)."""

from __future__ import annotations

import base64
import threading
from concurrent.futures import Future
from pathlib import Path
from typing import TYPE_CHECKING, Any

from google.protobuf import json_format

from cozy_runtime.internal import package_interface
from cozy_runtime.internal.worker.machine_builtin import Preparing
from cozy_runtime.internal.worker.workspace import WorkspaceRefusal
from cozy_runtime.protocol import worker_pb2 as pb

if TYPE_CHECKING:
    from .session import Worker

#: A retained installation document: its placement row and installation id.
Installed = dict[str, Any]


def rows(capture: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """The capture's deferred callees by key, each checked against its own release."""
    out: dict[str, dict[str, Any]] = {}
    listed = capture.get("deferred_installations", [])
    if not isinstance(listed, list) or len(listed) > 128:
        raise WorkspaceRefusal("execution capture has unbounded deferred installations")
    for row in listed:
        key = row.get("key", "") if isinstance(row, dict) else ""
        if not key or key in out or key != f"{row.get('package', '')}@{row.get('release', '')}":
            raise WorkspaceRefusal("execution capture repeats or misnames a deferred installation")
        out[key] = row
    return out


def interface(row: dict[str, Any]) -> dict[str, Any]:
    """The release interface a deferred row carries, for binding before it is installed."""
    raw = row.get("preparation", {}).get("package_interface", "")
    if not raw:
        raise WorkspaceRefusal("deferred installation carries no release interface")
    return package_interface.read_bytes(base64.b64decode(raw, validate=True), "deferred release")


class Deferred:
    """One installation per key: the first selection installs it, every concurrent selection
    waits for that one, and a failure fails exactly the calls that selected it."""

    def __init__(self, worker: Worker):
        self.worker = worker
        self.lock = threading.Lock()
        self.installs: dict[str, Future[Installed]] = {}

    def prepared(self, row: Installed, *, wait: bool = True) -> Installed:
        """The callee's retained installation. A call waits for it; a hint (`wait=False`)
        starts it and gets `Preparing` until it lands."""
        key = row["key"]
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

    def _prepare(self, row: Installed, future: Future[Installed]) -> None:
        from .session import read_placement_set

        key = row["key"]
        try:
            request = json_format.ParseDict(
                row["preparation"], pb.PreparePackageSetRequest(), ignore_unknown_fields=True
            )
            request.install_root = str(Path(self.worker.options.install_root or ""))
            self.worker.note("preparation", f"installing {key} on its first selection")
            entries = read_placement_set(self.worker.prepare_package_set(request).placement_set)
            if len(entries) != 1 or entries[0].get("package", {}).get("package") != row["package"]:
                raise WorkspaceRefusal("deferred preparation returned another installation")
            future.set_result(
                {"placement": entries[0], "installation_id": entries[0]["installation_id"]}
            )
        except Exception as exc:  # the calls that selected it fail, nothing else
            future.set_exception(
                WorkspaceRefusal(f"deferred_installation_failed: {key}: {exc}"[:1024])
            )
