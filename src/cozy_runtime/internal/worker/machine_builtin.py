"""Runtime library operations use its ordinary retained installation lifetime."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any

from cozy_runtime import canonical_json
from cozy_runtime.internal import (
    builtin_environment,
    builtin_operations,
    job_plan,
    package_installation,
    package_interface,
)
from cozy_runtime.internal.worker import package_prepare
from cozy_runtime.internal.worker.plan import JobBinding
from cozy_runtime.internal.worker.workspace import WorkspaceRefusal
from cozy_runtime.protocol import documents
from cozy_runtime.protocol import worker_pb2 as pb

if TYPE_CHECKING:
    from .attempts import AttemptRecord
    from .machine_child_target import Target
    from .session import Worker
    from .workspace_calls import Call


class Preparing(Exception):
    """Retained for the ordinary call scheduler's pending preparation branch."""


class Builtins:
    def __init__(self, worker: Worker):
        self.worker = worker
        self.captured: dict[str, Any] | None = None
        # Concurrent first submissions capture once; the rest take the same result.
        self.capturing = threading.Lock()

    def close(self) -> None:
        pass

    def capture(self) -> dict[str, Any]:
        with self.capturing:
            if self.captured is None:
                self.captured = self._capture()
            return self.captured

    def _capture(self) -> dict[str, Any]:
        worker = self.worker
        root = Path(worker.options.install_root or "")
        prepared = builtin_environment.prepare(root)
        installed = prepared.installed
        interface_raw = worker._describe_installed(
            installed,
            "cozy-runtime",
            application=builtin_operations.APPLICATION,
        )
        interface = package_interface.read_bytes(interface_raw)
        interface_path = (
            root / "installations" / installed.installation_id / "package-interface.json"
        )
        package_interface.publish(interface_path, interface_raw)
        package_prepare._stage_job_plans(
            interface,
            root=worker.config.cozy_home / "job-plans",
            installation_id=installed.installation_id,
            interface_path=interface_path,
            installed=installed,
        )
        placement = pb.Placement(
            placement_id="runtime-operations-" + installed.installation_id,
            package=pb.PackageSelection(package="runtime/operations", release=installed.release),
            installation_id=installed.installation_id,
            package_interface=interface_raw,
            bindings_digest=documents.digest_of(
                canonical_json.encode({"entrypoints": [], "models": []})
            ),
        )
        return {
            "placement": documents.body(placement),
            "installation_id": installed.installation_id,
            "runtime_version": installed.release,
        }

    def resolve(self, parent: AttemptRecord, call: Call, intent: dict[str, Any]) -> Target:
        from .machine_child_target import Target

        worker = self.worker
        if (
            intent["module"] != builtin_operations.MODULE
            or intent["export"] not in builtin_operations.EXPORTS
        ):
            raise WorkspaceRefusal("unknown Runtime operation")
        assert worker.executions is not None
        owner = worker.fence.record_owner_id
        root = worker.executions.capture_root(owner, parent.request_id)
        captured = worker.executions.preparation(owner, root).get("runtime_builtin")
        if not isinstance(captured, dict):
            raise WorkspaceRefusal("Runtime operation installation was not retained")
        installed = package_installation.open_installation(
            Path(worker.options.install_root or ""),
            captured["installation_id"],
        )
        placement = worker._placement_from_entry(captured["placement"], b"")
        placement.installed = installed
        worker.prepared_installations[placement.prepared_key] = placement
        interface = package_interface.read_bytes(placement.package_interface)
        export = intent["export"]
        declared = next(row for row in interface["jobs"] if row["name"] == export)
        descriptor = package_interface.job_descriptor_id(interface, export)
        binding = JobBinding.read(
            json.loads(
                job_plan.path(
                    worker.config.cozy_home / "job-plans",
                    installed.installation_id,
                    descriptor,
                ).read_bytes()
            )
        )
        operation_identity = str(declared.get("invocable", {}).get("operation_identity", ""))
        if declared.get("invocable", {}).get("memoize") and not operation_identity:
            worker.note(
                "memoization", f"Runtime {export} runs uncached: operation identity unavailable"
            )
        return Target(
            installed.installation_id,
            export,
            declared,
            {
                "placement": placement.document,
                "installation_id": installed.installation_id,
                "runtime_builtin": captured,
            },
            binding,
            operation_identity,
        )
