"""Lazy numerical preparation for Runtime's fixed captured builtin surface."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any

from cozy_runtime import canonical_json
from cozy_runtime.internal import (
    builtin_environment,
    builtin_operations,
    job_plan,
    numerical_environment,
    package_environment,
    package_interface,
)
from cozy_runtime.internal.worker import package_prepare
from cozy_runtime.internal.worker.plan import JobBinding
from cozy_runtime.internal.worker.workspace import WorkspaceRefusal
from cozy_runtime.internal.worker.workspace_calls import Call
from cozy_runtime.protocol import documents
from cozy_runtime.protocol import worker_pb2 as pb

if TYPE_CHECKING:
    from .attempts import AttemptRecord
    from .machine_child_target import Target
    from .session import Worker


class Preparing(Exception):
    """The ordinary call remains pending while its immutable environment is prepared."""


def capture_sdk() -> dict[str, str] | None:
    try:
        return builtin_environment.sdk_identity()
    except (
        importlib.metadata.PackageNotFoundError,
        numerical_environment.NumericalEnvironmentRefusal,
    ):
        # Source-only development interpreters can run ordinary package jobs, but
        # cannot promise an immutable Runtime builtin they do not actually hold.
        return None


class Builtins:
    def __init__(self, worker: Worker):
        self.worker = worker
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="builtin-prepare")
        self.prepared: dict[str, Future[dict[str, Any]]] = {}
        self.captured: dict[str, Any] | None = None

    def close(self) -> None:
        self.pool.shutdown(wait=True, cancel_futures=True)

    def capture(self) -> dict[str, Any] | None:
        """Retain SDK bytes before acceptance, without resolving numerical extras."""
        if self.captured is not None:
            return self.captured
        sdk = capture_sdk()
        if sdk is None:
            return None
        root = Path(self.worker.options.install_root or "") / "runtime-builtins"
        filename, raw = builtin_operations.carrier(sdk["runtime_version"])
        directory = root / "carriers"
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        carrier = directory / filename
        if carrier.exists() and (carrier.is_symlink() or carrier.read_bytes() != raw):
            raise WorkspaceRefusal("captured Runtime carrier changed")
        if not carrier.exists():
            carrier.write_bytes(raw)
            carrier.chmod(0o400)
        captured = builtin_environment.prepare(
            carrier,
            root / "capture",
            capture=True,
            dependency_cache=self.worker.config.dependency_cache,
        )
        self.captured = {
            "sdk": sdk,
            "environment_digest": captured.environment_digest,
            "receipt_digest": captured.installed.receipt_digest,
            "carrier": {
                "filename": filename,
                "digest": documents.spell(hashlib.sha256(raw).digest()),
                "length": len(raw),
            },
            "package_interface": canonical_json.decode(builtin_operations.document()),
        }
        return self.captured

    def resolve(self, parent: AttemptRecord, call: Call, intent: dict[str, Any]) -> Target:
        from .machine_child_target import Target

        worker = self.worker
        assert worker.executions is not None
        owner = worker.fence.record_owner_id
        root = worker.executions.capture_root(owner, parent.request_id)
        frozen = worker.executions.preparation(owner, root).get("runtime_builtin")
        if not isinstance(frozen, dict) or set(frozen) != {
            "sdk",
            "environment_digest",
            "receipt_digest",
            "carrier",
            "package_interface",
        }:
            raise WorkspaceRefusal("builtin has no exact accepted SDK snapshot")
        sdk = frozen["sdk"]
        export = intent["export"]
        if (intent["interface_digest"], intent["module"]) != (
            sdk["interface_digest"],
            builtin_operations.MODULE,
        ) or export not in builtin_operations.EXPORTS:
            raise WorkspaceRefusal("builtin call changed its captured interface")
        prior = canonical_json.decode(call.prepared) if call.prepared else {}
        entry = prior.get("builtin_preparation")
        if entry is None:
            key = canonical_json.digest(frozen)
            future = self.prepared.get(key)
            if future is None:
                self.prepared[key] = self.pool.submit(self._prepare, frozen)
                raise Preparing
            if not future.done():
                raise Preparing
            entry = future.result()
        placement = worker._placement_from_entry(entry["placement"], b"")
        revision = placement.development["local_revision_digest"]
        installed = package_environment.open_generation(
            Path(worker.options.install_root or ""),
            receipt_digest=entry["receipt_digest"],
            environment_digest=documents.spell(placement.environment_digest),
            python=Path(worker.options.python or sys.executable),
        )
        placement.installed = installed
        worker.prepared_revisions[(placement.placement_id, revision)] = placement
        interface = package_interface.read_bytes(
            canonical_json.encode(frozen["package_interface"]), "Runtime builtin"
        )
        declared = next(row for row in interface["jobs"] if row["name"] == export)
        descriptor = package_interface.job_descriptor_id(interface, export)
        binding = JobBinding.read(
            json.loads(
                job_plan.path(
                    worker.config.cozy_home / "job-plans", revision, descriptor
                ).read_bytes()
            )
        )
        identity = canonical_json.digest(
            {
                "local_revision_digest": revision,
                "interface_digest": sdk["interface_digest"],
                "module": builtin_operations.MODULE,
                "export": export,
                "entrypoint": export,
                "environment_content_digest": installed.content_digest,
            }
        )
        return Target(revision, export, sdk["interface_digest"], declared, entry, binding, identity)

    def _prepare(self, frozen: dict[str, Any]) -> dict[str, Any]:
        worker = self.worker
        sdk = frozen["sdk"]
        builtin_root = Path(worker.options.install_root or "") / "runtime-builtins"
        captured = package_environment.open_generation(
            builtin_root / "capture",
            receipt_digest=frozen["receipt_digest"],
            environment_digest=frozen["environment_digest"],
            python=Path(worker.options.python or sys.executable),
        )
        filename = frozen["carrier"]["filename"]
        carrier = builtin_root / "carriers" / filename
        raw = carrier.read_bytes()
        if (
            len(raw) != frozen["carrier"]["length"]
            or documents.spell(hashlib.sha256(raw).digest()) != frozen["carrier"]["digest"]
        ):
            raise WorkspaceRefusal("captured Runtime carrier is absent or changed")
        wheel_digest = hashlib.sha256(raw).digest()
        source_digest = documents.raw(
            canonical_json.digest(
                {
                    "carrier_digest": documents.spell(wheel_digest),
                    "interface_digest": sdk["interface_digest"],
                    "implementation_digest": sdk["implementation_digest"],
                }
            )
        )
        interface = canonical_json.encode(frozen["package_interface"])
        revision = pb.LocalPackageRevision(
            package="local/" + builtin_operations.NAME,
            release=sdk["runtime_version"],
            source_digest=source_digest,
            package_interface=pb.Ref(
                digest=documents.raw(sdk["interface_digest"]), length=len(interface)
            ),
            files=[pb.LocalPackageFileRef(digest=wheel_digest, length=len(raw), filename=filename)],
        )
        _, revision_digest = documents.identity(revision)
        operation = "builtin-" + revision_digest.hex()
        install_root = Path(worker.options.install_root or "")
        stage = install_root / ".stage" / operation / "wheels"
        stage.mkdir(parents=True, exist_ok=True, mode=0o700)
        wheel = stage / filename
        if wheel.exists():
            if wheel.is_symlink() or wheel.read_bytes() != raw:
                raise WorkspaceRefusal("Runtime builtin carrier changed")
        else:
            wheel.write_bytes(raw)
            wheel.chmod(0o400)
        environment = dict(worker.config.child_base_env)
        environment["PATH"] = str(captured.python.parent) + os.pathsep + environment.get("PATH", "")
        environment["COZY_HOME"] = str(worker.config.cozy_home)
        if worker.config.dependency_cache is not None:
            environment["COZY_DEPENDENCY_CACHE"] = str(worker.config.dependency_cache)
        command = subprocess.run(
            [
                str(captured.python.with_name("cozy-runtime")),
                "--json",
                "builtin-prepare",
                "operations",
                "--wheel",
                str(carrier),
                "--out",
                str(builtin_root / "numerical"),
            ],
            env=environment,
            capture_output=True,
            check=False,
        )
        if command.returncode or len(command.stdout) > 1 << 20:
            raise WorkspaceRefusal("captured Runtime numerical preparation failed")
        numerical = json.loads(command.stdout)
        if (
            numerical.get("runtime_version") != sdk["runtime_version"]
            or numerical.get("implementation_digest") != sdk["implementation_digest"]
            or numerical.get("package_interface") != frozen["package_interface"]
        ):
            raise WorkspaceRefusal("prepared builtin changed its accepted SDK identity")
        numerical_environment = package_environment.open_generation(
            builtin_root / "numerical",
            receipt_digest=numerical["receipt_digest"],
            environment_digest=numerical["environment_digest"],
            python=captured.python,
        )
        materialized: list[package_environment.InstalledEnvironment] = []

        def verified(digest: str, length: int, path: Path) -> None:
            worker.verified_artifacts.record(digest, length, path)

        result = package_prepare.prepare_local_package(
            pb.PrepareLocalPackageRequest(
                operation_id=operation,
                install_root=str(install_root),
                package=pb.DevelopmentPackage(
                    package=revision.package,
                    release=revision.release,
                    source_digest=source_digest,
                    local_revision_digest=revision_digest,
                ),
                wheels=[
                    pb.LocalPackageWheel(
                        digest=wheel_digest, length=len(raw), filename=filename, path=str(wheel)
                    )
                ],
            ),
            artifact_cache=Path(worker.options.artifact_cache or ""),
            install_root=install_root,
            python=Path(worker.options.python or sys.executable),
            describe=lambda _installed, _distribution: interface,
            verified=verified,
            job_plan_root=worker.config.cozy_home / "job-plans",
            base=worker.base,
            dependency_python=numerical_environment.python,
            dependency_cache=worker.config.dependency_cache,
            materialized=materialized.append,
        )
        placements = documents.read(
            result.placement_set.placement_set_canonical_bytes, pb.PlacementSet
        )["placements"]
        if len(placements) != 1 or len(materialized) != 1:
            raise WorkspaceRefusal("builtin preparation omitted its immutable environment")
        return {
            "placement": placements[0],
            "receipt_digest": materialized[0].receipt_digest,
            "runtime_builtin": frozen,
        }
