"""Install a published callee on its first selection (workspace deferred_installations)."""

from __future__ import annotations

import base64
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any

from google.protobuf import json_format

from cozy_runtime.internal import package_interface
from cozy_runtime.internal.worker.machine_builtin import Preparing
from cozy_runtime.internal.worker.workspace import WorkspaceRefusal
from cozy_runtime.protocol import worker_pb2 as pb

if TYPE_CHECKING:
    from .session import Worker


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
    """One preparation per key; concurrent selections wait on it and a failure fails its call."""

    def __init__(self, worker: Worker):
        self.worker = worker
        self.lock = threading.Lock()
        self.ready: dict[str, dict[str, Any]] = {}
        self.running: set[str] = set()
        self.failed: dict[str, str] = {}

    def prepared(self, row: dict[str, Any]) -> dict[str, Any]:
        """The callee's retained installation, or Preparing while it installs."""
        key = row["key"]
        with self.lock:
            if key in self.ready:
                return self.ready[key]
            if key in self.failed:
                raise WorkspaceRefusal(self.failed.pop(key))
            if key not in self.running:
                self.running.add(key)
                threading.Thread(
                    target=self._prepare, args=(row,), name=f"deferred-{key}", daemon=True
                ).start()
        raise Preparing(key)

    def _prepare(self, row: dict[str, Any]) -> None:
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
            prepared = {"placement": entries[0], "installation_id": entries[0]["installation_id"]}
            with self.lock:
                self.ready[key] = prepared
        except Exception as exc:  # the one child call that selected it fails, nothing else
            with self.lock:
                self.failed[key] = f"deferred_installation_failed: {key}: {exc}"[:1024]
        finally:
            with self.lock:
                self.running.discard(key)
