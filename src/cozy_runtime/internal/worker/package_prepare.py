"""Prepare one downloaded package set without giving package code worker authority."""

from __future__ import annotations

import hashlib
import os
import re
import stat
import uuid
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from cozy_runtime import qualification
from cozy_runtime.internal import canonical, descriptor, overlay, placement_materialization
from cozy_runtime.internal.refusal import LaunchRefusal
from cozy_runtime.protocol import documents
from cozy_runtime.protocol import worker_pb2 as pb

MAX_FILES = 4096
MAX_FILE_BYTES = 8 << 30
_DISTRIBUTION = re.compile(r"[-_.]+")


class PreparationRefusal(Exception):
    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def prepare_package_set(
    request: pb.PreparePackageSetRequest,
    *,
    artifact_cache: Path,
    environment_root: Path,
    python: Path,
    describe: Callable[[overlay.InstalledOverlay, str], bytes],
    verified: Callable[[str, int, Path], None],
    base_manifest: Path = overlay.DEFAULT_BASE_MANIFEST,
) -> pb.PreparePackageSetResult:
    """Materialize and describe one weightless package into an existing PlacementSet."""

    if Path(request.environment_root) != environment_root or not environment_root.is_absolute():
        raise PreparationRefusal(
            "package_prepare_environment_root_invalid", request.environment_root
        )
    try:
        delegation = documents.read(request.download_delegation, pb.DownloadDelegation)
    except documents.DocumentError as exc:
        raise PreparationRefusal("package_prepare_delegation_invalid", exc.detail) from exc
    packages = delegation.get("packages")
    models = delegation.get("models")
    if not isinstance(packages, list) or len(packages) != 1 or models != []:
        raise PreparationRefusal(
            "package_prepare_weightless_only", "exactly one package and zero models are supported"
        )
    selected = packages[0]
    if not isinstance(selected, dict):
        raise PreparationRefusal("package_prepare_delegation_invalid", "package row")
    package_name = str(selected.get("package", ""))
    release = str(selected.get("release", ""))
    release_digest = str(selected.get("release_digest", ""))
    try:
        documents.raw(release_digest)
    except (documents.DocumentError, ValueError) as exc:
        raise PreparationRefusal("package_prepare_release_digest_invalid", release_digest) from exc

    base_raw = _read_file(base_manifest, 64 << 20)
    base_digest = "sha256:" + hashlib.sha256(base_raw).hexdigest()
    try:
        base = overlay.parse_base_manifest(base_raw, base_digest)
    except overlay.OverlayRefusal as exc:
        raise PreparationRefusal(exc.code, exc.detail) from exc

    rows = list(request.files)
    if not 1 <= len(rows) <= MAX_FILES:
        raise PreparationRefusal("package_prepare_file_count", str(len(rows)))
    facts: list[tuple[overlay.Wheel, Path, overlay.VerifiedFile]] = []
    prior = b""
    parents: set[Path] = set()
    for row in rows:
        if len(row.digest) != 32 or row.digest <= prior or not 0 < row.length <= MAX_FILE_BYTES:
            raise PreparationRefusal("package_prepare_file_invalid", row.filename)
        prior = bytes(row.digest)
        if row.kind not in {
            pb.LOCAL_DOWNLOAD_KIND_PROJECT_WHEEL,
            pb.LOCAL_DOWNLOAD_KIND_DEPENDENCY_WHEEL,
        }:
            raise PreparationRefusal("package_prepare_weightless_only", row.filename)
        path = Path(row.path)
        if (
            not path.is_absolute()
            or path.name != row.filename
            or not path.is_relative_to(artifact_cache / "sets")
            or path.resolve(strict=True) != path
        ):
            raise PreparationRefusal("package_prepare_file_path_invalid", row.filename)
        parents.add(path.parent)
        owner = "project" if row.kind == pb.LOCAL_DOWNLOAD_KIND_PROJECT_WHEEL else "custom"
        try:
            wheel, receipt = overlay.measure_wheel(path, owner, base)
        except overlay.OverlayRefusal as exc:
            raise PreparationRefusal(exc.code, exc.detail) from exc
        if wheel.digest != documents.spell(bytes(row.digest)) or wheel.length != row.length:
            raise PreparationRefusal("package_prepare_file_identity_mismatch", row.filename)
        facts.append((wheel, path, receipt))
    if len(parents) != 1:
        raise PreparationRefusal("package_prepare_file_path_invalid", "files span package sets")
    projects = [item for item in facts if item[0].owner == "project"]
    if len(projects) != 1:
        raise PreparationRefusal("package_prepare_project_wheel_count", str(len(projects)))
    project = projects[0][0]
    expected_distribution = _DISTRIBUTION.sub("-", package_name.rsplit("/", 1)[-1]).lower()
    if project.distribution != expected_distribution or project.version != release:
        raise PreparationRefusal("package_prepare_project_wheel_mismatch", project.filename)

    dependency_facts = sorted(
        (wheel for wheel, _path, _receipt in facts if wheel.owner == "custom"),
        key=lambda wheel: wheel.distribution,
    )
    environment = pb.Environment(
        wheelhouse_manifest=pb.Ref(digest=documents.raw(base_digest), length=len(base_raw)),
        wheels=[_wheel_fact(wheel) for wheel in dependency_facts],
    )
    environment_body = documents.body(environment)
    environment_body.setdefault("wheels", [])
    environment_digest = documents.spell(
        documents.embedded_document_digest(environment_body, "cozy.worker.v1.Environment")
    )
    package = pb.PackageSelection(
        package=package_name,
        release=release,
        release_digest=documents.raw(release_digest),
        project_wheel=_wheel_fact(project),
    )
    wheel_paths = {wheel.digest: path for wheel, path, _receipt in facts}
    receipts = {wheel.digest: receipt for wheel, _path, receipt in facts}
    try:
        source = placement_materialization.package_overlay_input(
            package=documents.body(package),
            environment=environment_body,
            environment_digest=environment_digest,
            wheelhouse_manifest_bytes=base_raw,
            wheel_paths=wheel_paths,
            verified_wheels=receipts,
        )
        installed = overlay.materialize(
            source,
            environment_root,
            python=python,
            base_manifest=base_manifest,
        )
    except (overlay.OverlayRefusal, LaunchRefusal) as exc:
        code = getattr(exc, "code", "package_prepare_overlay_failed")
        detail = getattr(exc, "detail", type(exc).__name__)
        raise PreparationRefusal(str(code), str(detail)) from exc
    for wheel, path, _receipt in facts:
        verified(wheel.digest, wheel.length, path)

    descriptor_raw = describe(installed, project.distribution)
    try:
        descriptor_body = descriptor.read_bytes(descriptor_raw, "prepared package descriptor")
    except descriptor.StalePackageDescriptor as exc:
        raise PreparationRefusal("package_prepare_descriptor_invalid", str(exc)) from exc
    descriptor_ref = _stage(artifact_cache, descriptor_raw)
    _stage(artifact_cache, base_raw)
    entrypoints = _entrypoints(descriptor_body)
    entrypoint_bodies = [documents.body(row) for row in entrypoints]
    for entrypoint_body in entrypoint_bodies:
        entrypoint_body.setdefault("slots", [])
    bindings_digest = documents.digest_of(
        canonical.write({"entrypoints": entrypoint_bodies, "models": []})
    )
    receipt_ref = {"digest": installed.receipt_digest, "length": len(installed.receipt_bytes)}
    descriptor_value = {"digest": descriptor_ref[0], "length": descriptor_ref[1]}
    selection = qualification.selection_digest(
        bindings_digest=documents.spell(bindings_digest),
        environment_digest=environment_digest,
        environment_receipt=receipt_ref,
        package_descriptor=descriptor_value,
        package=documents.body(package),
    )
    qualification_raw = canonical.write(qualification.document(selection))
    qualification_ref = _stage(artifact_cache, qualification_raw)
    placement = pb.Placement(
        placement_id="private-" + hashlib.sha256(request.download_delegation).hexdigest()[:24],
        package=package,
        environment_digest=documents.raw(environment_digest),
        environment_receipt_digest=documents.raw(installed.receipt_digest),
        package_descriptor=pb.Ref(
            digest=documents.raw(descriptor_ref[0]), length=descriptor_ref[1]
        ),
        bindings_digest=bindings_digest,
        models=[],
        entrypoints=entrypoints,
        qualification=pb.Ref(
            digest=documents.raw(qualification_ref[0]), length=qualification_ref[1]
        ),
        environment=environment,
    )
    placement_body = documents.body(placement)
    placement_body.setdefault("models", [])
    placement_body["environment"].setdefault("wheels", [])
    for entrypoint_body in placement_body["entrypoints"]:
        entrypoint_body.setdefault("slots", [])
    placement_bytes = canonical.write(
        {"format": "cozy.worker.v1.PlacementSet/1", "placements": [placement_body]}
    )
    return pb.PreparePackageSetResult(
        placement_set=pb.DesiredPlacementSet(
            placement_set_digest=documents.digest_of(placement_bytes),
            placement_set_canonical_bytes=placement_bytes,
        )
    )


def _wheel_fact(wheel: overlay.Wheel) -> pb.WheelFact:
    return pb.WheelFact(
        ref=pb.Ref(digest=documents.raw(wheel.digest), length=wheel.length),
        distribution=wheel.distribution,
        version=wheel.version,
        filename=wheel.filename,
        import_roots=list(wheel.import_roots),
        tags=list(wheel.tags),
    )


def _entrypoints(body: Mapping[str, Any]) -> list[pb.Entrypoint]:
    rows = body.get("entrypoints")
    if not isinstance(rows, list) or not rows:
        raise PreparationRefusal("package_prepare_entrypoints_absent", "package is not serving")
    out: list[pb.Entrypoint] = []
    for value in rows:
        if not isinstance(value, dict) or not isinstance(value.get("name"), str):
            raise PreparationRefusal("package_prepare_descriptor_invalid", "entrypoint")
        if value.get("models") not in (None, []):
            raise PreparationRefusal("package_prepare_weightless_only", value["name"])
        entrypoint = pb.Entrypoint(name=value["name"], slots=[])
        entrypoint.entrypoint_binding_digest = documents.digest_of(
            canonical.write({"name": entrypoint.name, "slots": []})
        )
        out.append(entrypoint)
    out.sort(key=lambda row: row.name)
    if len({row.name for row in out}) != len(out):
        raise PreparationRefusal("package_prepare_descriptor_invalid", "duplicate entrypoint")
    return out


def _read_file(path: Path, maximum: int) -> bytes:
    try:
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode) or not 0 < info.st_size <= maximum:
            raise OSError("invalid file")
        return path.read_bytes()
    except OSError as exc:
        raise PreparationRefusal("package_prepare_file_unreadable", str(path)) from exc


def _stage(root: Path, raw: bytes) -> tuple[str, int]:
    digest = "sha256:" + hashlib.sha256(raw).hexdigest()
    target = root / digest.removeprefix("sha256:")
    if target.exists():
        if _read_file(target, max(len(raw), 1)) != raw:
            raise PreparationRefusal("package_prepare_cache_conflict", digest)
        return digest, len(raw)
    temporary = root / f".prepare-{uuid.uuid4().hex}"
    descriptor_fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    try:
        with os.fdopen(descriptor_fd, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, target)
    except FileExistsError:
        if _read_file(target, max(len(raw), 1)) != raw:
            raise PreparationRefusal("package_prepare_cache_conflict", digest) from None
    finally:
        temporary.unlink(missing_ok=True)
    return digest, len(raw)
