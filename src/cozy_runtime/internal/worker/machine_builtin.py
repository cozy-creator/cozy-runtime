"""Runtime library operations use its ordinary retained installation lifetime."""

from __future__ import annotations

import json
import threading
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

import msgspec

from cozy_runtime import canonical_json
from cozy_runtime.internal import (
    builtin_environment,
    builtin_operations,
    job_plan,
    package_installation,
    package_interface,
)
from cozy_runtime.internal.worker import machine_capture, package_prepare
from cozy_runtime.internal.worker.plan import JobBinding
from cozy_runtime.internal.worker.workspace import WorkspaceRefusal
from cozy_runtime.internal.worker.workspace_calls import CallIntent
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
        self.captured: machine_capture.Builtin | None = None
        # Concurrent first submissions capture once; the rest take the same result.
        self.capturing = threading.Lock()

    def close(self) -> None:
        pass

    def capture(self) -> machine_capture.Builtin:
        with self.capturing:
            if self.captured is None:
                self.captured = self._capture()
            return self.captured

    def _capture(self) -> machine_capture.Builtin:
        worker = self.worker
        root = Path(worker.options.install_root or "")
        installed = builtin_environment.prepare(root)
        interface_path = (
            root / "installations" / installed.installation_id / "package-interface.json"
        )
        try:
            # Described by an earlier process of this same installed Runtime.
            interface_raw = interface_path.read_bytes()
            package_interface.parse(interface_raw)
        except (OSError, package_interface.StalePackageInterface):
            interface_raw = worker._describe_installed(
                installed,
                "cozy-runtime",
                application=builtin_operations.APPLICATION,
            )
            package_interface.parse(interface_raw)
            package_interface.publish(interface_path, interface_raw)
        package_prepare._stage_job_plans(
            interface_raw,
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
        return machine_capture.Builtin(
            placement=documents.body(placement),
            installation_id=installed.installation_id,
            runtime_version=installed.release,
        )

    def resolve(self, parent: AttemptRecord, call: Call, intent: Mapping[str, object]) -> Target:
        # machine_child_target -> machine_deferred -> this module's Preparing: a cycle.
        from .machine_child_target import Target

        worker = self.worker
        try:
            named = msgspec.convert(intent, CallIntent)
        except msgspec.ValidationError as exc:
            raise WorkspaceRefusal(f"Runtime operation intent is malformed: {exc}") from exc
        if (
            named.module != builtin_operations.MODULE
            or named.export not in builtin_operations.EXPORTS
        ):
            raise WorkspaceRefusal("unknown Runtime operation")
        assert worker.executions is not None
        owner = worker.fence.record_owner_id
        root = worker.executions.capture_root(owner, parent.request_id)
        captured = machine_capture.preparation(
            worker.executions.preparation(owner, root)
        ).runtime_builtin
        if captured is None:
            raise WorkspaceRefusal("Runtime operation installation was not retained")
        installed = package_installation.open_installation(
            Path(worker.options.install_root or ""), captured.installation_id
        )
        placement = worker._placement_from_entry(
            documents.from_body(captured.placement, pb.Placement), b""
        )
        placement.installed = installed
        worker.prepared_installations[placement.prepared_key] = placement
        body = package_interface.read_bytes(placement.package_interface)
        export = named.export
        declared = next(row for row in body["jobs"] if row["name"] == export)
        typed = msgspec.convert(declared, package_interface.CallableDoc)
        descriptor = package_interface.job_descriptor_id(body, export)
        binding = JobBinding.read(
            json.loads(
                job_plan.path(
                    worker.config.cozy_home / "job-plans",
                    installed.installation_id,
                    descriptor,
                ).read_bytes()
            )
        )
        operation_identity = machine_capture.operation_identity(typed)
        if (
            typed.invocable is not msgspec.UNSET
            and typed.invocable.memoize
            and not operation_identity
        ):
            worker.note(
                "memoization", f"Runtime {export} runs uncached: operation identity unavailable"
            )
        return Target(
            installed.installation_id,
            export,
            declared,
            {
                "placement": captured.placement,
                "installation_id": installed.installation_id,
                "runtime_builtin": msgspec.to_builtins(captured),
            },
            binding,
            operation_identity,
        )
