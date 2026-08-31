"""Prepare one downloaded package set without giving package code worker authority."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import stat
import uuid
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from cozy_runtime.internal import (
    canonical,
    descriptor,
    fill,
    model_config,
    package_environment,
    placement_materialization,
    pull,
)
from cozy_runtime.internal.refusal import LaunchRefusal
from cozy_runtime.internal.worker import acquire
from cozy_runtime.protocol import documents
from cozy_runtime.protocol import worker_pb2 as pb

MAX_FILES = 4096
MAX_FILE_BYTES = 8 << 30
_DISTRIBUTION = re.compile(r"[-_.]+")
_OPERATION_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,255}")


class PreparationRefusal(Exception):
    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def prepare_package_set(
    request: pb.PreparePackageSetRequest,
    *,
    artifact_cache: Path,
    artifact_store: Path,
    environment_root: Path,
    python: Path,
    describe: Callable[[package_environment.InstalledEnvironment, str], bytes],
    verified: Callable[[str, int, Path], None],
    job_plan_root: Path | None = None,
    base_manifest: Path = package_environment.DEFAULT_BASE_MANIFEST,
) -> pb.PreparePackageSetResult:
    """Parse one signed delegation, then use the shared package preparation path."""

    if (
        Path(request.environment_root) != environment_root
        or not environment_root.is_absolute()
        or not artifact_store.is_absolute()
        or (job_plan_root is not None and not job_plan_root.is_absolute())
    ):
        raise PreparationRefusal(
            "package_prepare_environment_root_invalid", request.environment_root
        )
    try:
        delegation = documents.read(request.download_delegation, pb.DownloadDelegation)
    except documents.DocumentError as exc:
        raise PreparationRefusal("package_prepare_delegation_invalid", exc.detail) from exc
    packages = delegation.get("packages")
    models = delegation.get("models")
    if not isinstance(packages, list) or len(packages) != 1 or not isinstance(models, list):
        raise PreparationRefusal(
            "package_prepare_selection_invalid",
            "exactly one package and a model array are required",
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

    return prepare_package(
        package_name=package_name,
        release=release,
        release_digest=release_digest,
        files=request.files,
        placement_seed=request.download_delegation,
        artifact_cache=artifact_cache,
        artifact_store=artifact_store,
        models=models,
        environment_root=environment_root,
        python=python,
        describe=describe,
        verified=verified,
        job_plan_root=job_plan_root,
        base_manifest=base_manifest,
    )


def prepare_private_package(
    request: pb.PreparePrivatePackageRequest,
    *,
    artifact_cache: Path,
    environment_root: Path,
    python: Path,
    describe: Callable[[package_environment.InstalledEnvironment, str], bytes],
    verified: Callable[[str, int, Path], None],
    job_plan_root: Path | None = None,
    base_manifest: Path = package_environment.DEFAULT_BASE_MANIFEST,
) -> pb.PreparePackageSetResult:
    """Prepare one exact private wheel revision without inventing Hub provenance."""

    if (
        Path(request.environment_root) != environment_root
        or not environment_root.is_absolute()
        or (job_plan_root is not None and not job_plan_root.is_absolute())
    ):
        raise PreparationRefusal(
            "private_package_environment_root_invalid", request.environment_root
        )
    package = request.package
    if (
        _OPERATION_ID.fullmatch(request.operation_id) is None
        or package.package.count("/") != 1
        or not package.release
        or len(package.source_digest) != 32
        or len(package.private_revision_digest) != 32
        or package.HasField("project_wheel")
    ):
        raise PreparationRefusal("private_package_identity_invalid", request.operation_id)
    if any(
        row.kind
        not in {
            pb.LOCAL_DOWNLOAD_KIND_PROJECT_WHEEL,
            pb.LOCAL_DOWNLOAD_KIND_DEPENDENCY_WHEEL,
        }
        for row in request.files
    ):
        raise PreparationRefusal("private_package_file_kind_invalid", request.operation_id)
    return prepare_package(
        package_name=package.package,
        release=package.release,
        release_digest=documents.spell(package.private_revision_digest),
        files=request.files,
        placement_seed=(request.operation_id.encode("ascii") + package.private_revision_digest),
        artifact_cache=artifact_cache,
        environment_root=environment_root,
        python=python,
        describe=describe,
        verified=verified,
        job_plan_root=job_plan_root,
        base_manifest=base_manifest,
        development=package,
    )


def prepare_private_placement(
    request: pb.PreparePrivatePlacementRequest,
    *,
    prepared: Mapping[str, Any],
    artifact_cache: Path,
    artifact_store: Path | None,
    verified: Callable[[str, int, Path], None],
) -> pb.PreparePackageSetResult:
    """Bind verified models to one exact already-prepared private revision core."""

    development = prepared.get("development")
    descriptor_ref = prepared.get("package_descriptor")
    if (
        _OPERATION_ID.fullmatch(request.operation_id) is None
        or len(request.private_revision_digest) != 32
        or not isinstance(development, dict)
        or development.get("private_revision_digest")
        != documents.spell(request.private_revision_digest)
        or not isinstance(descriptor_ref, dict)
        or not isinstance(descriptor_ref.get("digest"), str)
        or not isinstance(descriptor_ref.get("length"), int)
    ):
        raise PreparationRefusal("private_placement_identity_invalid", request.operation_id)
    try:
        delegation = documents.read(request.download_delegation, pb.DownloadDelegation)
    except documents.DocumentError as exc:
        raise PreparationRefusal("private_placement_delegation_invalid", exc.detail) from exc
    packages = delegation.get("packages", [])
    models = delegation.get("models", [])
    if packages != [] or not isinstance(models, list) or len(models) > 32:
        raise PreparationRefusal(
            "private_placement_delegation_invalid",
            "private model delegation carries no package refs and at most 32 models",
        )

    rows = list(request.files)
    model_files: dict[str, tuple[int, int, Path]] = {}
    prior = b""
    parents: set[Path] = set()
    for row in rows:
        path = Path(row.path)
        if (
            len(row.digest) != 32
            or row.digest <= prior
            or not 0 < row.length <= MAX_FILE_BYTES
            or row.kind
            not in {
                pb.LOCAL_DOWNLOAD_KIND_MODEL_MANIFEST,
                pb.LOCAL_DOWNLOAD_KIND_MODEL_OBJECT,
            }
            or not path.is_absolute()
            or path.name != row.filename
            or not path.is_relative_to(artifact_cache / "sets")
            or path.resolve(strict=True) != path
        ):
            raise PreparationRefusal("private_placement_file_invalid", row.filename)
        prior = bytes(row.digest)
        parents.add(path.parent)
        model_files[documents.spell(bytes(row.digest))] = (row.kind, row.length, path)
    if rows and len(parents) != 1:
        raise PreparationRefusal("private_placement_file_invalid", "files span package sets")

    digest = str(descriptor_ref["digest"])
    length = int(descriptor_ref["length"])
    try:
        documents.raw(digest)
    except (documents.DocumentError, ValueError) as exc:
        raise PreparationRefusal("private_placement_descriptor_invalid", digest) from exc
    descriptor_raw = _read_file(artifact_cache / digest.removeprefix("sha256:"), length)
    if (
        len(descriptor_raw) != length
        or "sha256:" + hashlib.sha256(descriptor_raw).hexdigest() != digest
    ):
        raise PreparationRefusal("private_placement_descriptor_invalid", digest)
    try:
        descriptor_body = descriptor.read_bytes(descriptor_raw, "prepared private descriptor")
    except descriptor.StalePackageDescriptor as exc:
        raise PreparationRefusal("private_placement_descriptor_invalid", str(exc)) from exc
    selected_models, entrypoints = _entrypoints(
        descriptor_body,
        package_name=str(development.get("package", "")),
        selections=models,
        files=model_files,
        artifact_cache=artifact_cache,
        artifact_store=artifact_store,
        verified=verified,
    )
    entrypoint_bodies = [documents.body(row) for row in entrypoints]
    for entrypoint in entrypoint_bodies:
        entrypoint.setdefault("slots", [])
    model_bodies = [documents.body(row) for row in selected_models]
    joined = copy.deepcopy(dict(prepared))
    joined["models"] = model_bodies
    joined["entrypoints"] = entrypoint_bodies
    joined["bindings_digest"] = documents.spell(
        documents.digest_of(
            canonical.write({"entrypoints": entrypoint_bodies, "models": model_bodies})
        )
    )
    placement_bytes = canonical.write(
        {"format": "cozy.worker.v1.PlacementSet/1", "placements": [joined]}
    )
    return pb.PreparePackageSetResult(
        placement_set=pb.DesiredPlacementSet(
            placement_set_digest=documents.digest_of(placement_bytes),
            placement_set_canonical_bytes=placement_bytes,
        )
    )


def prepare_package(
    *,
    package_name: str,
    release: str,
    release_digest: str,
    files: Sequence[pb.LocalDownloadFile],
    placement_seed: bytes,
    artifact_cache: Path,
    environment_root: Path,
    python: Path,
    describe: Callable[[package_environment.InstalledEnvironment, str], bytes],
    verified: Callable[[str, int, Path], None],
    base_manifest: Path,
    models: Sequence[Any] = (),
    artifact_store: Path | None = None,
    job_plan_root: Path | None = None,
    allow_native_custom: bool = False,
    installed_environment: package_environment.InstalledEnvironment | None = None,
    development: pb.DevelopmentPackage | None = None,
) -> pb.PreparePackageSetResult:
    """Inspect, install, import, and describe one exact package.

    The pod session uses the closed default. Creator's private same-user command alone opts
    into native dependency wheels after the same digest, tag, RECORD, and collision checks.
    """

    try:
        documents.raw(release_digest)
    except (documents.DocumentError, ValueError) as exc:
        raise PreparationRefusal("package_prepare_release_digest_invalid", release_digest) from exc

    base_raw = _read_file(base_manifest, 64 << 20)
    base_digest = "sha256:" + hashlib.sha256(base_raw).hexdigest()
    try:
        base = package_environment.parse_base_manifest(base_raw, base_digest)
    except package_environment.EnvironmentRefusal as exc:
        raise PreparationRefusal(exc.code, exc.detail) from exc

    rows = list(files)
    if not 1 <= len(rows) <= MAX_FILES:
        raise PreparationRefusal("package_prepare_file_count", str(len(rows)))
    facts: list[tuple[package_environment.Wheel, Path, package_environment.VerifiedFile]] = []
    model_files: dict[str, tuple[int, int, Path]] = {}
    prior = b""
    parents: set[Path] = set()
    for row in rows:
        if len(row.digest) != 32 or row.digest <= prior or not 0 < row.length <= MAX_FILE_BYTES:
            raise PreparationRefusal("package_prepare_file_invalid", row.filename)
        prior = bytes(row.digest)
        if row.kind not in {
            pb.LOCAL_DOWNLOAD_KIND_PROJECT_WHEEL,
            pb.LOCAL_DOWNLOAD_KIND_DEPENDENCY_WHEEL,
            pb.LOCAL_DOWNLOAD_KIND_MODEL_MANIFEST,
            pb.LOCAL_DOWNLOAD_KIND_MODEL_OBJECT,
        }:
            raise PreparationRefusal("package_prepare_file_invalid", row.filename)
        path = Path(row.path)
        if (
            not path.is_absolute()
            or path.name != row.filename
            or not path.is_relative_to(artifact_cache / "sets")
            or path.resolve(strict=True) != path
        ):
            raise PreparationRefusal("package_prepare_file_path_invalid", row.filename)
        parents.add(path.parent)
        digest = documents.spell(bytes(row.digest))
        if row.kind in {
            pb.LOCAL_DOWNLOAD_KIND_MODEL_MANIFEST,
            pb.LOCAL_DOWNLOAD_KIND_MODEL_OBJECT,
        }:
            model_files[digest] = (row.kind, row.length, path)
            continue
        owner = "project" if row.kind == pb.LOCAL_DOWNLOAD_KIND_PROJECT_WHEEL else "custom"
        try:
            wheel, receipt = package_environment.measure_wheel(
                path,
                owner,
                base,
                allow_native_custom=allow_native_custom,
                allow_base_owned=installed_environment is not None or owner == "custom",
            )
        except package_environment.EnvironmentRefusal as exc:
            raise PreparationRefusal(exc.code, exc.detail) from exc
        if wheel.digest != digest or wheel.length != row.length:
            raise PreparationRefusal("package_prepare_file_identity_mismatch", row.filename)
        if (
            installed_environment is None
            and owner == "custom"
            and package_environment.base_owned_wheel(wheel, base)
        ):
            verified(digest, row.length, path)
            continue
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
    environment = pb.Environment(wheels=[_wheel_fact(wheel) for wheel in dependency_facts])
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
        source = placement_materialization.package_environment_input(
            package=documents.body(package),
            environment=environment_body,
            environment_digest=environment_digest,
            wheelhouse_manifest_bytes=base_raw,
            wheel_paths=wheel_paths,
            verified_wheels=receipts,
        )
        if installed_environment is None:
            installed = package_environment.materialize(
                source,
                environment_root,
                python=python,
                base_manifest=base_manifest,
                allow_native_custom=allow_native_custom,
            )
        else:
            package_environment.verify_existing_environment(
                source,
                base_manifest=base_manifest,
                allow_native_custom=allow_native_custom,
            )
            installed = installed_environment
    except (package_environment.EnvironmentRefusal, LaunchRefusal) as exc:
        code = getattr(exc, "code", "package_environment_install_failed")
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
    if development is not None:
        _verify_private_revision(development, descriptor_ref, rows)
    jobs = _stage_job_plans(
        descriptor_body,
        root=job_plan_root,
        build_id=release_digest,
        descriptor_path=artifact_cache / descriptor_ref[0].removeprefix("sha256:"),
        installed=installed,
        include_modeled=development is not None,
    )
    if development is None:
        selected_models, entrypoints = _entrypoints(
            descriptor_body,
            package_name=package_name,
            selections=models,
            files=model_files,
            artifact_cache=artifact_cache,
            artifact_store=artifact_store,
            verified=verified,
        )
    else:
        # Private transport freezes code, descriptor and Environment only. A later ordinary
        # PlacementSet supplies exact modeled serving bindings; JobDirective + InvocationSpec
        # supplies a job's exact model inputs. Weightless serving bindings need no later model
        # authority and are already exact consequences of this retained descriptor.
        selected_models, entrypoints = _entrypoints(
            descriptor_body,
            package_name=package_name,
            selections=(),
            files={},
            artifact_cache=artifact_cache,
            artifact_store=None,
            verified=verified,
        )
    if development is None and not entrypoints and not jobs:
        raise PreparationRefusal(
            "package_prepare_callable_absent",
            "package has no runnable serving or weightless job callable",
        )
    entrypoint_bodies = [documents.body(row) for row in entrypoints]
    for entrypoint_body in entrypoint_bodies:
        entrypoint_body.setdefault("slots", [])
    bindings_digest = documents.digest_of(
        canonical.write(
            {
                "entrypoints": entrypoint_bodies,
                "models": [documents.body(model) for model in selected_models],
            }
        )
    )
    placement = pb.Placement(
        placement_id="package-" + hashlib.sha256(placement_seed).hexdigest()[:24],
        environment_digest=documents.raw(environment_digest),
        package_descriptor=pb.Ref(
            digest=documents.raw(descriptor_ref[0]), length=descriptor_ref[1]
        ),
        bindings_digest=bindings_digest,
        models=selected_models,
        entrypoints=entrypoints,
        environment=environment,
    )
    if development is None:
        placement.package.CopyFrom(package)
    else:
        placement.development.CopyFrom(
            pb.DevelopmentPackage(
                package=development.package,
                release=development.release,
                source_digest=development.source_digest,
                private_revision_digest=development.private_revision_digest,
                project_wheel=_wheel_fact(project),
            )
        )
    placement_body = documents.body(placement)
    placement_body.setdefault("models", [])
    placement_body.setdefault("entrypoints", [])
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


def _wheel_fact(wheel: package_environment.Wheel) -> pb.WheelFact:
    return pb.WheelFact(
        ref=pb.Ref(digest=documents.raw(wheel.digest), length=wheel.length),
        distribution=wheel.distribution,
        version=wheel.version,
        filename=wheel.filename,
        import_roots=list(wheel.import_roots),
        tags=list(wheel.tags),
    )


def _verify_private_revision(
    development: pb.DevelopmentPackage,
    descriptor_ref: tuple[str, int],
    files: Sequence[pb.LocalDownloadFile],
) -> None:
    revision = pb.PrivatePackageRevision(
        package=development.package,
        release=development.release,
        source_digest=development.source_digest,
        package_descriptor=pb.Ref(
            digest=documents.raw(descriptor_ref[0]), length=descriptor_ref[1]
        ),
        files=[
            pb.PrivatePackageFileRef(
                digest=row.digest,
                filename=row.filename,
                kind=row.kind,
                length=row.length,
            )
            for row in files
        ],
    )
    _raw, measured = documents.identity(revision)
    if measured != development.private_revision_digest:
        raise PreparationRefusal(
            "private_package_revision_mismatch",
            f"measured {documents.spell(measured)}, expected "
            f"{documents.spell(development.private_revision_digest)}",
        )


def _entrypoints(
    body: Mapping[str, Any],
    *,
    package_name: str,
    selections: Sequence[Any],
    files: Mapping[str, tuple[int, int, Path]],
    artifact_cache: Path,
    artifact_store: Path | None,
    verified: Callable[[str, int, Path], None],
) -> tuple[list[pb.Model], list[pb.Entrypoint]]:
    rows = body.get("entrypoints")
    if not isinstance(rows, list):
        raise PreparationRefusal("package_prepare_descriptor_invalid", "entrypoints")
    selected = _selected_models(
        package_name, selections, files, artifact_cache, artifact_store, verified
    )
    model_ids = {slot: f"model-{index:04d}" for index, slot in enumerate(sorted(selected))}
    models = [
        pb.Model(
            id=model_ids[slot],
            repo=str(value["selection"]["model"]),
            version=str(value["selection"]["release"]),
            lane=str(value["selection"].get("lane", "")),
            manifest=pb.Ref(
                digest=documents.raw(str(value["selection"]["manifest"])),
                length=int(value["manifest_length"]),
            ),
        )
        for slot, value in sorted(selected.items())
    ]
    out: list[pb.Entrypoint] = []
    described_slots: set[str] = set()
    for value in rows:
        if not isinstance(value, dict) or not isinstance(value.get("name"), str):
            raise PreparationRefusal("package_prepare_descriptor_invalid", "entrypoint")
        described = value.get("models", [])
        if not isinstance(described, list):
            raise PreparationRefusal("package_prepare_descriptor_invalid", "model slots")
        described_paths: set[str] = set()
        for row in described:
            if not isinstance(row, dict) or not isinstance(row.get("path"), str):
                raise PreparationRefusal("package_prepare_descriptor_invalid", "model slot")
            described_paths.add(str(row["path"]))
        if len(described_paths) != len(described):
            raise PreparationRefusal(
                "package_prepare_model_selection_mismatch", f"{value['name']}: duplicate slot"
            )
        if described_paths and not described_paths.issubset(selected):
            # One desired package set names the entrypoints it can actually bind. A
            # different invocation may select another model later; it should not force
            # this worker to download unrelated weights now.
            continue
        slots: list[pb.Slot] = []
        for row in described:
            path = str(row["path"])
            prefix = f"{value['name']}.models."
            if (
                not path.startswith(prefix)
                or path not in selected
                or row.get("stamps") not in ({}, None)
            ):
                raise PreparationRefusal("package_prepare_model_selection_mismatch", path)
            described_slots.add(path)
            model = selected[path]
            model_id = model_ids[path]
            slots.append(
                pb.Slot(
                    slot=path.removeprefix(prefix),
                    reference_model_id=model_id,
                    config=pb.Config(
                        document=pb.Ref(
                            digest=documents.raw(str(model["config_digest"])),
                            length=len(model["config_bytes"]),
                        ),
                        assets=[],
                    ),
                    components=[
                        pb.Component(component=component, model_id=model_id)
                        for component in model["components"]
                    ],
                    stamps=[],
                )
            )
        slots.sort(key=lambda row: row.slot)
        entrypoint = pb.Entrypoint(name=value["name"], slots=slots)
        entrypoint.entrypoint_binding_digest = documents.digest_of(
            canonical.write(
                {"name": entrypoint.name, "slots": [documents.body(slot) for slot in slots]}
            )
        )
        out.append(entrypoint)
    out.sort(key=lambda row: row.name)
    if len({row.name for row in out}) != len(out):
        raise PreparationRefusal("package_prepare_descriptor_invalid", "duplicate entrypoint")
    if described_slots != set(selected):
        raise PreparationRefusal(
            "package_prepare_model_selection_mismatch",
            f"missing={sorted(set(selected) - described_slots)} "
            f"extra={sorted(described_slots - set(selected))}",
        )
    return models, out


def _stage_job_plans(
    body: Mapping[str, Any],
    *,
    root: Path | None,
    build_id: str,
    descriptor_path: Path,
    installed: package_environment.InstalledEnvironment,
    include_modeled: bool = False,
) -> set[str]:
    rows = body.get("jobs")
    if not isinstance(rows, list):
        raise PreparationRefusal("package_prepare_descriptor_invalid", "jobs")
    selected = [
        row for row in rows if isinstance(row, dict) and (include_modeled or not row.get("models"))
    ]
    if not selected or root is None:
        return set()
    root.mkdir(parents=True, exist_ok=True, mode=0o755)
    if root.is_symlink() or not root.is_dir():
        raise PreparationRefusal("package_prepare_job_plan_root_invalid", str(root))
    application = str(body.get("application", ""))
    staged: set[str] = set()
    for row in selected:
        name = str(row.get("name", ""))
        descriptor_id = descriptor.job_descriptor_id(body, name)
        resources = row.get("resources") or {}
        if not isinstance(resources, dict):
            raise PreparationRefusal("package_prepare_descriptor_invalid", f"{name}.resources")
        record = {
            "job_descriptor_id": descriptor_id,
            "build_id": build_id,
            "application": application,
            "package_descriptor": str(descriptor_path),
            "python": str(installed.python),
            "environment_content_digest": installed.content_digest,
            "job": name,
            "gpu_count": int(resources.get("gpu_count", 0)),
            "publishes": bool(row.get("publishes", False)),
            "emits_media": False,
            "gpu_rate_micro_usd_per_hour": 0,
            "cap_micro_usd": 0,
            "reclaim_on_terminal": True,
        }
        target = root / f"{descriptor_id.removeprefix('sha256:')}.json"
        temporary = root / f".stage-{uuid.uuid4().hex}"
        temporary.write_text(json.dumps(record, sort_keys=True, separators=(",", ":")))
        temporary.chmod(0o444)
        temporary.replace(target)
        staged.add(descriptor_id)
    return staged


def _selected_models(
    package_name: str,
    selections: Sequence[Any],
    files: Mapping[str, tuple[int, int, Path]],
    artifact_cache: Path,
    artifact_store: Path | None,
    verified: Callable[[str, int, Path], None],
) -> dict[str, dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    for row in selections:
        if not isinstance(row, dict) or row.get("package") != package_name:
            raise PreparationRefusal("package_prepare_model_selection_mismatch", "package")
        slot = str(row.get("slot", ""))
        if not slot or slot in selected:
            raise PreparationRefusal("package_prepare_model_selection_mismatch", slot)
        selected[slot] = {"selection": row}

    manifest_refs = {str(value["selection"]["manifest"]) for value in selected.values()}
    manifest_files = {
        digest: (length, path)
        for digest, (kind, length, path) in files.items()
        if kind == pb.LOCAL_DOWNLOAD_KIND_MODEL_MANIFEST
    }
    object_files = {
        digest: (length, path)
        for digest, (kind, length, path) in files.items()
        if kind == pb.LOCAL_DOWNLOAD_KIND_MODEL_OBJECT
    }
    carried_models = bool(manifest_files or object_files)
    if carried_models and set(manifest_files) != manifest_refs:
        raise PreparationRefusal(
            "package_prepare_model_manifest_set_mismatch",
            f"wanted={sorted(manifest_refs)} got={sorted(manifest_files)}",
        )
    if not selected:
        if object_files:
            raise PreparationRefusal(
                "package_prepare_model_object_set_mismatch", "unexpected objects"
            )
        return selected

    if artifact_store is None:
        raise PreparationRefusal(
            "package_prepare_model_store_absent", "modeled package requires a TensorFS store"
        )

    try:
        tensorfs = fill.tensorfs_module()
        store = pull.open_store(tensorfs, artifact_store)
        closures: dict[str, set[tuple[str, int]]] = {}
        reached: set[tuple[str, int]] = set()
        manifests: dict[str, tuple[int, bytes, Path | None]] = {}
        if carried_models:
            for digest, (length, path) in sorted(manifest_files.items()):
                raw = _read_file(path, int(tensorfs.manifest_max_bytes()))
                store.put_manifest(raw, digest, length)
                manifests[digest] = (length, raw, path)
        else:
            for value in selected.values():
                selection = value["selection"]
                digest = str(selection["manifest"])
                length = int(selection.get("manifest_length", 0))
                model = str(selection.get("model", ""))
                release = str(selection.get("release", ""))
                lane = str(selection.get("lane", ""))
                parts = model.split("/")
                if len(parts) != 2 or not all(parts) or not release or not lane:
                    raise PreparationRefusal("package_prepare_model_selection_mismatch", model)
                resolved = store.resolve_release(parts[0], parts[1], release, lane)
                if (
                    str(resolved["manifest_digest"]) != digest
                    or int(resolved["manifest_length"]) != length
                ):
                    raise PreparationRefusal(
                        "package_prepare_model_selection_mismatch",
                        f"{model}@{release}/{lane} does not resolve to {digest}",
                    )
                held = store.manifest(digest)
                raw = bytes(held["manifest"])
                if (
                    length <= 0
                    or len(raw) != length
                    or "sha256:" + hashlib.sha256(raw).hexdigest() != digest
                ):
                    raise PreparationRefusal("package_prepare_model_manifest_mismatch", digest)
                manifests[digest] = (length, raw, None)
        for digest, (_length, raw, _path) in sorted(manifests.items()):
            closure = acquire.manifest_entries(tensorfs, raw)
            closures[digest] = closure
            reached.update(closure)
        if (
            carried_models
            and {(digest, length) for digest, (length, _path) in object_files.items()} != reached
        ):
            raise PreparationRefusal(
                "package_prepare_model_object_set_mismatch",
                "downloaded model objects do not equal the selected Manifest closure",
            )
        for digest, (length, path) in sorted(object_files.items()):
            store.put_file(str(path), digest, length)
            verified(digest, length, path)
        for digest, (length, _raw, manifest_path) in sorted(manifests.items()):
            store.acquire_manifest(digest).release()
            held = store.manifest(digest)
            header = tensorfs.parse_header(bytes(held["header"]))
            configs = header.get("configs")
            components = header.get("components")
            if (
                not isinstance(components, dict)
                or not components
                or header.get("assets") not in ({}, None)
            ):
                raise PreparationRefusal(
                    "package_prepare_model_header_unsupported",
                    f"{digest}: expected configs, components, and no assets",
                )
            try:
                config_digest, config_bytes = model_config.construction_document(store, configs)
            except model_config.ModelConfigRefusal as exc:
                raise PreparationRefusal(
                    "package_prepare_model_config_mismatch", f"{digest}: {exc}"
                ) from exc
            _stage(artifact_cache, config_bytes)
            for value in selected.values():
                if value["selection"]["manifest"] == digest:
                    value.update(
                        manifest_length=length,
                        config_digest=config_digest,
                        config_bytes=config_bytes,
                        components=tuple(sorted(str(component) for component in components)),
                    )
            if manifest_path is not None:
                verified(digest, length, manifest_path)
    except PreparationRefusal:
        raise
    except (acquire.AcquisitionRefusal, pull.PullRefusal, fill.FillRefusal) as exc:
        raise PreparationRefusal(
            "package_prepare_model_refused", f"{getattr(exc, 'code', type(exc).__name__)}: {exc}"
        ) from exc
    except Exception as exc:
        raise PreparationRefusal(
            "package_prepare_model_refused", f"{type(exc).__name__}: {exc}"
        ) from exc
    return selected


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
