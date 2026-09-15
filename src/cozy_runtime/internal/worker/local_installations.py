"""Reopen installed local revisions without wheel transfers or package imports."""

from __future__ import annotations

import dataclasses
import hashlib
import os
import stat
import uuid
from pathlib import Path

from cozy_runtime.internal import (
    base_observation,
    canonical,
    package_environment,
    package_interface,
    placement_materialization,
)
from cozy_runtime.internal.refusal import LaunchRefusal
from cozy_runtime.protocol import documents
from cozy_runtime.protocol import worker_pb2 as pb

from . import package_prepare


def _path(root: Path, request: pb.PrepareLocalPackageRequest) -> Path:
    return root / "local-installations" / (request.package.local_revision_digest.hex() + ".json")


def record(
    root: Path,
    request: pb.PrepareLocalPackageRequest,
    result: pb.PreparePackageSetResult,
    installed: package_environment.InstalledEnvironment,
    base: base_observation.Observation,
) -> None:
    """Publish the installed entrypoint inventory beside its immutable environment receipt."""
    path = _path(root, request)
    path.parent.mkdir(mode=0o700, exist_ok=True)
    identity = canonical.parse_canonical(
        package_prepare._read_file(
            installed.generation / package_environment.CONTENT_IDENTITY_NAME,
            canonical.DOC_MAX_BYTES,
        )
    )
    raw = canonical.write(
        {
            "format": "cozy.runtime.LocalInstallation/1",
            "base_digest": canonical.digest(dataclasses.asdict(base)),
            "placement_set": documents.read(
                result.placement_set.placement_set_canonical_bytes, pb.PlacementSet
            ),
            # A local SDK snapshot is frozen at installation, independently of its
            # mutable source venv. The receipt itself is derived from these facts.
            **(
                {"dependency_snapshot": identity["dependency_snapshot"]}
                if "dependency_snapshot" in identity
                else {}
            ),
        }
    )
    temporary = path.parent / (".install-" + uuid.uuid4().hex)
    try:
        with temporary.open("xb") as stream:
            stream.write(raw)
            stream.flush()
            os.fchmod(stream.fileno(), 0o400)
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    finally:
        temporary.unlink(missing_ok=True)


def reopen(
    root: Path,
    request: pb.PrepareLocalPackageRequest,
    *,
    artifact_cache: Path,
    python: Path,
    base: base_observation.Observation,
    job_plan_root: Path,
) -> tuple[pb.PreparePackageSetResult, package_environment.InstalledEnvironment] | None:
    """Return live installed code, or a miss; never execute a package to check a hit.

    Immutable generation/receipt validation is the same bounded check used by
    execution admission. We read only the installation and interface documents,
    not wheel archives or the installed dependency tree.
    """
    if (
        len(request.package.local_revision_digest) != 32
        or Path(request.install_root) != root
        or not root.is_absolute()
        or package_prepare._OPERATION_ID.fullmatch(request.operation_id) is None
    ):
        return None
    path = _path(root, request)
    try:
        for entry in (root, path.parent, path):
            info = entry.lstat()
            if info.st_uid != os.geteuid() or info.st_mode & 0o022 or stat.S_ISLNK(info.st_mode):
                return None
        if path.stat().st_mode & 0o222:
            return None
        raw = package_prepare._read_file(path, canonical.DOC_MAX_BYTES)
        value = canonical.parse_canonical(raw)
        if (
            not isinstance(value, dict)
            or value.get("format") != "cozy.runtime.LocalInstallation/1"
            or value.get("base_digest") != canonical.digest(dataclasses.asdict(base))
        ):
            return None
        placement_set = value["placement_set"]
        placements = placement_set["placements"]
        if len(placements) != 1:
            return None
        placement = placements[0]
        interface = placement["package_interface"]
        package_prepare._verify_local_revision(
            request.package, (interface["digest"], interface["length"]), request.wheels
        )
        development = dict(placement["development"])
        development.pop("project_wheel")
        if development != documents.body(request.package):
            return None
        wheels = [
            placement_materialization._wheel(
                placement["development"]["project_wheel"],
                "project",
                import_roots=True,
                nested_ref=True,
            ),
            *placement_materialization._wheels(
                placement["environment"]["local_wheels"],
                "environment",
                import_roots=True,
                nested_ref=True,
            ),
        ]
        project = (
            package_prepare._DISTRIBUTION.sub(
                "-", request.package.package.rsplit("/", 1)[-1]
            ).lower(),
            request.package.release,
        )
        expected = set()
        for row in request.wheels:
            named = package_environment.wheel_name_version(row.filename)
            if named is None:
                raise package_prepare.PreparationRefusal(
                    "local_package_revision_mismatch", "revision wheel filename is invalid"
                )
            # Cold preparation verifies import ownership before omitting these
            # protected image distributions. Every other supplied wheel must remain.
            if named == project or not package_environment.base_owned_distribution(named[0], base):
                expected.add((documents.spell(row.digest), row.length, row.filename))
        supplied = {(wheel.digest, wheel.length, wheel.filename) for wheel in wheels}
        if (
            supplied != expected
            or len(supplied) != len(wheels)
            or (wheels[0].distribution, wheels[0].version) != project
            or any(
                package_environment.wheel_name_version(wheel.filename)
                != (wheel.distribution, wheel.version)
                for wheel in wheels
            )
        ):
            raise package_prepare.PreparationRefusal(
                "local_package_revision_mismatch", "installed wheels differ from the revision"
            )
        environment_digest = documents.spell(
            documents.embedded_document_digest(
                placement["environment"], "cozy.worker.v1.Environment"
            )
        )
        if environment_digest != placement["environment_digest"]:
            return None
        receipt_digest = package_environment.wheel_receipt_digest(
            base,
            wheels=wheels,
            environment_digest=environment_digest,
            dependency_snapshot=value.get("dependency_snapshot"),
        )
        installed = package_environment.open_generation(
            root,
            receipt_digest=receipt_digest,
            environment_digest=environment_digest,
            python=python,
        )
        interface_path = artifact_cache / documents.raw(interface["digest"]).hex()
        interface_raw = package_prepare._read_file(interface_path, canonical.DOC_MAX_BYTES)
        if (
            len(interface_raw) != interface["length"]
            or documents.spell(documents.digest_of(interface_raw)) != interface["digest"]
        ):
            return None
        body = package_interface.read_bytes(interface_raw, "installed local package")
        package_prepare._stage_job_plans(
            body,
            root=job_plan_root,
            build_id=documents.spell(request.package.local_revision_digest),
            interface_path=interface_path,
            installed=installed,
        )
        placement["placement_id"] = (
            "package-"
            + hashlib.sha256(
                request.operation_id.encode("ascii") + request.package.local_revision_digest
            ).hexdigest()[:24]
        )
        raw = canonical.write(placement_set)
        return pb.PreparePackageSetResult(
            placement_set=pb.DesiredPlacementSet(
                placement_set_digest=documents.digest_of(raw),
                placement_set_canonical_bytes=raw,
            )
        ), installed
    except package_prepare.PreparationRefusal as exc:
        if exc.code == "package_prepare_file_unreadable":
            return None
        raise
    except (
        OSError,
        KeyError,
        TypeError,
        ValueError,
        canonical.CanonicalError,
        documents.DocumentError,
        package_environment.EnvironmentRefusal,
        package_interface.StalePackageInterface,
        LaunchRefusal,
    ):
        return None
