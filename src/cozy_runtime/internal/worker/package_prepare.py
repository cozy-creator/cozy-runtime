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
    base_observation,
    canonical,
    derive_child,
    fill,
    image_inventory,
    model_config,
    package_environment,
    package_interface,
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
    base: base_observation.Observation | None = None,
    native_environment: Mapping[str, str] | None = None,
    native_device_index: int = 0,
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
    published_interface, set_directory = _published_interface(
        request.files,
        artifact_cache=artifact_cache,
    )
    inventory_raw = _read_beside(
        set_directory / image_inventory.INVENTORY_FILENAME,
        "package_prepare_image_inventory_missing",
    )
    try:
        inventory = image_inventory.read_bytes(inventory_raw)
    except image_inventory.ImageInventoryRefusal as exc:
        raise PreparationRefusal("package_prepare_image_inventory_invalid", exc.detail) from exc

    return prepare_package(
        package_name=package_name,
        release=release,
        published_interface=published_interface,
        files=request.files,
        placement_seed=placement_identity(
            package_name, release, str(selected.get("release_digest", "")), models
        ),
        artifact_cache=artifact_cache,
        artifact_store=artifact_store,
        models=models,
        environment_root=environment_root,
        python=python,
        describe=describe,
        verified=verified,
        job_plan_root=job_plan_root,
        base=base,
        native_environment=native_environment,
        native_device_index=native_device_index,
        inventory=inventory,
    )


def placement_identity(
    package_name: str, release: str, release_digest: str, models: Sequence[canonical.Json]
) -> bytes:
    """The placement's identity is WHAT is prepared, never the credential that
    authorized it. The delegation carries a per-signing expiry and worker pins
    (other clocks — cr-004), so seeding identity with its raw bytes made every
    re-signed converge a brand-new placement and defeated ledger reuse
    (xs-014 finding 1). Model rows are re-sorted here so identity cannot
    depend on the author's ordering."""

    def model_key(row: canonical.Json) -> str:
        body = row if isinstance(row, Mapping) else {}
        return "\x00".join(
            str(body.get(field, ""))
            for field in ("package", "slot", "model", "release", "manifest")
        )

    return canonical.write(
        {
            "models": sorted((row for row in models), key=model_key),
            "package": package_name,
            "release": release,
            "release_digest": release_digest,
        }
    )


def _published_interface(
    files: Sequence[pb.LocalDownloadFile],
    *,
    artifact_cache: Path,
) -> tuple[bytes, Path]:
    """Read the exact PackageInterface staged beside ordinary verified package files."""
    rows = [row for row in files if row.kind == pb.LOCAL_DOWNLOAD_KIND_PACKAGE_INTERFACE]
    if len(rows) != 1:
        raise PreparationRefusal("package_prepare_interface_count", str(len(rows)))
    row = rows[0]
    path = Path(row.path)
    parent = path.parent.parent
    if (
        len(row.digest) != 32
        or not 0 < row.length <= canonical.DOC_MAX_BYTES
        or not path.is_absolute()
        or row.filename != package_interface.FILENAME
        or path != parent / package_interface.FILENAME
        or not parent.is_relative_to(artifact_cache / "sets")
        or path.resolve() != path
    ):
        raise PreparationRefusal("package_prepare_interface_ref_invalid", row.filename)
    raw = _read_beside(path, "package_prepare_published_interface_invalid")
    digest = documents.spell(bytes(row.digest))
    if len(raw) != row.length or documents.spell(hashlib.sha256(raw).digest()) != digest:
        raise PreparationRefusal(
            "package_prepare_published_interface_invalid",
            "staged package-interface bytes do not match its exact file row",
        )
    try:
        package_interface.read_bytes(raw, "published package interface")
    except package_interface.StalePackageInterface as exc:
        raise PreparationRefusal("package_prepare_published_interface_invalid", str(exc)) from exc
    return raw, parent


def _read_beside(path: Path, refusal: str) -> bytes:
    try:
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode) or not 0 < info.st_size <= canonical.DOC_MAX_BYTES:
            raise PreparationRefusal(refusal, str(path))
        return path.read_bytes()
    except OSError as exc:
        raise PreparationRefusal(refusal, f"{path}: {type(exc).__name__}") from exc


def prepare_local_package(
    request: pb.PrepareLocalPackageRequest,
    *,
    artifact_cache: Path,
    environment_root: Path,
    python: Path,
    describe: Callable[[package_environment.InstalledEnvironment, str], bytes],
    verified: Callable[[str, int, Path], None],
    job_plan_root: Path | None = None,
    base: base_observation.Observation | None = None,
    native_environment: Mapping[str, str] | None = None,
    native_device_index: int = 0,
    materialized: Callable[[package_environment.InstalledEnvironment], None] | None = None,
) -> pb.PreparePackageSetResult:
    """Prepare one exact local wheel revision without inventing Hub provenance."""

    if (
        Path(request.environment_root) != environment_root
        or not environment_root.is_absolute()
        or (job_plan_root is not None and not job_plan_root.is_absolute())
    ):
        raise PreparationRefusal("local_package_environment_root_invalid", request.environment_root)
    package = request.package
    if (
        _OPERATION_ID.fullmatch(request.operation_id) is None
        or package.package.count("/") != 1
        or not package.release
        or len(package.source_digest) != 32
        or len(package.local_revision_digest) != 32
        or package.HasField("project_wheel")
    ):
        raise PreparationRefusal("local_package_identity_invalid", request.operation_id)
    if any(
        row.kind
        not in {
            pb.LOCAL_DOWNLOAD_KIND_PROJECT_WHEEL,
            pb.LOCAL_DOWNLOAD_KIND_DEPENDENCY_WHEEL,
        }
        for row in request.files
    ):
        raise PreparationRefusal("local_package_file_kind_invalid", request.operation_id)
    return prepare_package(
        package_name=package.package,
        release=package.release,
        files=request.files,
        placement_seed=(request.operation_id.encode("ascii") + package.local_revision_digest),
        artifact_cache=artifact_cache,
        environment_root=environment_root,
        python=python,
        describe=describe,
        verified=verified,
        job_plan_root=job_plan_root,
        base=base,
        native_environment=native_environment,
        native_device_index=native_device_index,
        development=package,
        materialized=materialized,
    )


def prepare_private_placement(
    request: pb.PreparePrivatePlacementRequest,
    *,
    prepared: Mapping[str, Any],
    installed: package_environment.InstalledEnvironment | None,
    artifact_cache: Path,
    artifact_store: Path | None,
    verified: Callable[[str, int, Path], None],
) -> pb.PreparePackageSetResult:
    """Bind verified models to one exact already-prepared local revision core."""

    if installed is None:
        raise PreparationRefusal(
            "private_placement_environment_absent",
            f"{request.operation_id}: the local revision's environment is not held, so its "
            "slots cannot be derived",
        )
    development = prepared.get("development")
    interface_ref = prepared.get("package_interface")
    if (
        _OPERATION_ID.fullmatch(request.operation_id) is None
        or len(request.local_revision_digest) != 32
        or not isinstance(development, dict)
        or development.get("local_revision_digest")
        != documents.spell(request.local_revision_digest)
        or not isinstance(interface_ref, dict)
        or not isinstance(interface_ref.get("digest"), str)
        or not isinstance(interface_ref.get("length"), int)
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

    digest = str(interface_ref["digest"])
    length = int(interface_ref["length"])
    try:
        documents.raw(digest)
    except (documents.DocumentError, ValueError) as exc:
        raise PreparationRefusal("private_placement_interface_invalid", digest) from exc
    interface_raw = _read_file(artifact_cache / digest.removeprefix("sha256:"), length)
    if (
        len(interface_raw) != length
        or "sha256:" + hashlib.sha256(interface_raw).hexdigest() != digest
    ):
        raise PreparationRefusal("private_placement_interface_invalid", digest)
    try:
        interface_body = package_interface.read_bytes(
            interface_raw, "prepared local package interface"
        )
    except package_interface.StalePackageInterface as exc:
        raise PreparationRefusal("private_placement_interface_invalid", str(exc)) from exc
    selected_models, entrypoints = _entrypoints(
        interface_body,
        package_name=str(development.get("package", "")),
        selections=models,
        files=model_files,
        artifact_store=artifact_store,
        verified=verified,
        derive=_deriver(installed, str(interface_body.get("application", ""))),
    )
    entrypoint_bodies = [documents.body(row) for row in entrypoints]
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
    files: Sequence[pb.LocalDownloadFile],
    placement_seed: bytes,
    artifact_cache: Path,
    environment_root: Path,
    python: Path,
    describe: Callable[[package_environment.InstalledEnvironment, str], bytes],
    verified: Callable[[str, int, Path], None],
    base: base_observation.Observation | None = None,
    models: Sequence[Any] = (),
    artifact_store: Path | None = None,
    job_plan_root: Path | None = None,
    native_environment: Mapping[str, str] | None = None,
    native_device_index: int = 0,
    installed_environment: package_environment.InstalledEnvironment | None = None,
    development: pb.DevelopmentPackage | None = None,
    published_interface: bytes | None = None,
    inventory: image_inventory.Inventory | None = None,
    materialized: Callable[[package_environment.InstalledEnvironment], None] | None = None,
) -> pb.PreparePackageSetResult:
    """Inspect, install, import, and describe one exact package.

    `materialized` receives the environment this preparation installed. A local revision's
    wheels are the pod host's carriers, deleted once the PlacementSet is journaled, so the
    placement that follows cannot re-acquire them: the session retains the environment and
    serves the placement from it.

    A native dependency wheel is admitted on evidence, not on the caller's identity: the
    installed bytes are statically inspected, joined to this host's interpreter/libc/loader/GPU
    seat, and loaded and imported in the bounded operator child under `native_environment` —
    the caller's own sealed execution environment. Without one there is no qualification host,
    so the wheel refuses rather than installs.
    """

    expected_distribution = _DISTRIBUTION.sub("-", package_name.rsplit("/", 1)[-1]).lower()
    try:
        observed_base = base or package_environment.observe_base(python)
    except package_environment.EnvironmentRefusal as exc:
        raise PreparationRefusal(exc.code, exc.detail) from exc
    image_owned: dict[str, str] = {}
    if inventory is not None:
        # The delivered inventory is the authority on what the image provides; the live
        # observation is its cross-check (cr-070). A disagreement falsifies the inventory
        # rather than silently preparing against an image that is not what was registered.
        lie = image_inventory.disagreement(inventory, observed_base.distribution_versions)
        if lie is not None:
            raise PreparationRefusal("package_prepare_image_inventory_disagrees", lie)
        image_owned = inventory.versions
    rows = list(files)
    if not 1 <= len(rows) <= MAX_FILES:
        raise PreparationRefusal("package_prepare_file_count", str(len(rows)))
    facts: list[tuple[package_environment.Wheel, Path, package_environment.VerifiedFile]] = []
    model_files: dict[str, tuple[int, int, Path]] = {}
    image_served: dict[str, str] = {}
    prior = b""
    parents: set[Path] = set()
    for row in rows:
        if len(row.digest) != 32 or row.digest <= prior or not 0 < row.length <= MAX_FILE_BYTES:
            raise PreparationRefusal("package_prepare_file_invalid", row.filename)
        prior = bytes(row.digest)
        if row.kind not in {
            pb.LOCAL_DOWNLOAD_KIND_PROJECT_WHEEL,
            pb.LOCAL_DOWNLOAD_KIND_DEPENDENCY_WHEEL,
            pb.LOCAL_DOWNLOAD_KIND_PACKAGE_INTERFACE,
            pb.LOCAL_DOWNLOAD_KIND_MODEL_MANIFEST,
            pb.LOCAL_DOWNLOAD_KIND_MODEL_OBJECT,
        }:
            raise PreparationRefusal("package_prepare_file_invalid", row.filename)
        path = Path(row.path)
        interface_file = row.kind == pb.LOCAL_DOWNLOAD_KIND_PACKAGE_INTERFACE
        set_root = path.parent.parent if interface_file else path.parent
        if (
            not path.is_absolute()
            or (
                row.filename != package_interface.FILENAME
                if interface_file
                else path.name != row.filename
            )
            or path != set_root / row.filename
            or not path.is_relative_to(artifact_cache / "sets")
            or path.resolve(strict=True) != path
        ):
            raise PreparationRefusal("package_prepare_file_path_invalid", row.filename)
        parents.add(set_root)
        digest = documents.spell(bytes(row.digest))
        if interface_file:
            if (
                published_interface is None
                or row.length != len(published_interface)
                or digest != "sha256:" + hashlib.sha256(published_interface).hexdigest()
            ):
                raise PreparationRefusal(
                    "package_prepare_published_interface_invalid", row.filename
                )
            verified(digest, row.length, path)
            continue
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
                observed_base,
                allow_base_owned=installed_environment is not None or owner == "custom",
            )
        except package_environment.EnvironmentRefusal as exc:
            raise PreparationRefusal(exc.code, exc.detail) from exc
        if wheel.digest != digest or wheel.length != row.length:
            raise PreparationRefusal("package_prepare_file_identity_mismatch", row.filename)
        if owner == "project" and not wheel.import_roots:
            raise PreparationRefusal(
                "package_prepare_project_wheel_no_import_roots",
                f"{row.filename} installs no importable module",
            )
        if installed_environment is None and owner == "custom":
            if package_environment.base_owned_wheel(wheel, observed_base):
                # The supplied wheel proves which protected distribution the frozen package
                # closure selected; the worker image supplies the executable bytes. The
                # project and remaining dependency metadata constrain the image version.
                verified(digest, row.length, path)
                continue
            if wheel.distribution in image_owned:
                # Name-based export-minus-provided (cr-070): an inventory-owned NAME is
                # served by the image's site-packages copy, so its lock wheel never enters
                # the venv. Lock pins do not constrain image-owned names on pods; a declared
                # range the image version cannot satisfy boot-installs during materialize.
                verified(digest, row.length, path)
                image_served[wheel.distribution] = wheel.version
                continue
        facts.append((wheel, path, receipt))
    if len(parents) != 1:
        raise PreparationRefusal("package_prepare_file_path_invalid", "files span package sets")
    if image_served:
        served = ", ".join(
            f"{name} (image {image_owned[name]} over lock {version})"
            for name, version in sorted(image_served.items())
        )
        print(f"[package-prepare] image inventory serves {served}", flush=True)
    projects = [item for item in facts if item[0].owner == "project"]
    if len(projects) != 1:
        raise PreparationRefusal("package_prepare_project_wheel_count", str(len(projects)))
    project = projects[0][0]
    if project.distribution != expected_distribution or project.version != release:
        raise PreparationRefusal("package_prepare_project_wheel_mismatch", project.filename)

    dependency_facts = sorted(
        (wheel for wheel, _path, _receipt in facts if wheel.owner == "custom"),
        key=lambda wheel: wheel.distribution,
    )
    environment = pb.Environment(wheels=[_wheel_fact(wheel) for wheel in dependency_facts])
    environment_body = documents.body(environment)
    environment_digest = documents.spell(
        documents.embedded_document_digest(environment_body, "cozy.worker.v1.Environment")
    )
    package = pb.PackageSelection(
        package=package_name,
        release=release,
        project_wheel=_wheel_fact(project),
    )
    wheel_paths = {wheel.digest: path for wheel, path, _receipt in facts}
    receipts = {wheel.digest: receipt for wheel, _path, receipt in facts}
    try:
        source = placement_materialization.package_environment_input(
            package=documents.body(package),
            environment=environment_body,
            environment_digest=environment_digest,
            wheel_paths=wheel_paths,
            verified_wheels=receipts,
        )
        if installed_environment is None:
            installed = package_environment.materialize(
                source,
                environment_root,
                python=python,
                base=observed_base,
                native_environment=native_environment,
                native_device_index=native_device_index,
            )
        else:
            package_environment.verify_existing_environment(
                source,
                base=observed_base,
                installed=installed_environment,
                native_environment=native_environment,
                native_device_index=native_device_index,
            )
            installed = installed_environment
    except (package_environment.EnvironmentRefusal, LaunchRefusal) as exc:
        code = getattr(exc, "code", "package_environment_install_failed")
        detail = getattr(exc, "detail", type(exc).__name__)
        raise PreparationRefusal(str(code), str(detail)) from exc
    for wheel, path, _receipt in facts:
        verified(wheel.digest, wheel.length, path)
    if materialized is not None:
        materialized(installed)

    if published_interface is None:
        interface_raw = describe(installed, project.distribution)
    else:
        # The PUBLISHED interface is the authoritative wiring (cr-067). Import-time
        # enumeration still runs -- it is a free cross-check -- but a disagreement
        # falsifies the release rather than silently rewiring the pod: refuse, and the
        # record owner relays the defect (th-106).
        interface_raw = published_interface
        derived_raw = describe(installed, project.distribution)
        if derived_raw != published_interface:
            raise PreparationRefusal(
                "package_prepare_interface_disagrees",
                "published "
                + documents.spell(hashlib.sha256(published_interface).digest())
                + " != derived "
                + documents.spell(hashlib.sha256(derived_raw).digest()),
            )
    try:
        interface_body = package_interface.read_bytes(interface_raw, "prepared package interface")
    except package_interface.StalePackageInterface as exc:
        raise PreparationRefusal("package_prepare_interface_invalid", str(exc)) from exc
    interface_ref = _stage(artifact_cache, interface_raw)
    if development is not None:
        _verify_local_revision(development, interface_ref, rows)
    jobs = _stage_job_plans(
        interface_body,
        root=job_plan_root,
        build_id=project.digest,
        interface_path=artifact_cache / interface_ref[0].removeprefix("sha256:"),
        installed=installed,
    )
    if development is None:
        selected_models, entrypoints = _entrypoints(
            interface_body,
            package_name=package_name,
            selections=models,
            files=model_files,
            artifact_store=artifact_store,
            verified=verified,
            derive=_deriver(installed, str(interface_body.get("application", ""))),
        )
    else:
        # Local package transport freezes code, PackageInterface, and Environment only. A
        # later ordinary PlacementSet supplies exact modeled serving bindings; JobDirective
        # + InvocationSpec
        # supplies a job's exact model inputs. Weightless serving bindings need no later model
        # authority and are already exact consequences of this retained package interface.
        selected_models, entrypoints = _entrypoints(
            interface_body,
            package_name=package_name,
            selections=(),
            files={},
            artifact_store=None,
            verified=verified,
            derive=None,
        )
    if development is None and not entrypoints and not jobs:
        raise PreparationRefusal(
            "package_prepare_callable_absent",
            "package has no runnable serving or weightless job callable",
        )
    entrypoint_bodies = [documents.body(row) for row in entrypoints]
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
        package_interface=pb.Ref(digest=documents.raw(interface_ref[0]), length=interface_ref[1]),
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
                local_revision_digest=development.local_revision_digest,
                project_wheel=_wheel_fact(project),
            )
        )
    placement_body = documents.body(placement)
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


def _verify_local_revision(
    development: pb.DevelopmentPackage,
    interface_ref: tuple[str, int],
    files: Sequence[pb.LocalDownloadFile],
) -> None:
    revision = pb.LocalPackageRevision(
        package=development.package,
        release=development.release,
        source_digest=development.source_digest,
        package_interface=pb.Ref(digest=documents.raw(interface_ref[0]), length=interface_ref[1]),
        files=[
            pb.LocalPackageFileRef(
                digest=row.digest,
                filename=row.filename,
                kind=row.kind,
                length=row.length,
            )
            for row in files
        ],
    )
    _raw, measured = documents.identity(revision)
    if measured != development.local_revision_digest:
        raise PreparationRefusal(
            "local_package_revision_mismatch",
            f"measured {documents.spell(measured)}, expected "
            f"{documents.spell(development.local_revision_digest)}",
        )


#: `(slot path, config bytes, verified model assets) -> constructed components`.
Deriver = Callable[[str, bytes, Mapping[str, bytes]], tuple[str, ...]]


def _deriver(installed: package_environment.InstalledEnvironment, application: str) -> Deriver:
    """Private request-time construction in the package's own environment:
    `Slot.components` is what the factory constructs for this config document (D9), and
    this plane never reads it off the header. The child imports the derive machinery; this
    plane imports only the request/response seam (cr-067's guard, kept by construction)."""

    def derive(
        slot: str, config_bytes: bytes, assets: Mapping[str, bytes]
    ) -> tuple[str, ...]:
        if not application:
            raise PreparationRefusal("package_prepare_interface_invalid", "application")
        try:
            request = derive_child.DeriveRequest(
                (derive_child.SlotRequest(slot, config_bytes, assets),),
                application=application,
            )
            (held,) = derive_child.derive_in(installed.python, request)
            return held.components
        except derive_child.DeriveRefusal as exc:
            raise PreparationRefusal(
                "package_prepare_derive_refused", f"{slot}: {exc.code}: {exc.detail}"
            ) from exc

    return derive


def _entrypoints(
    body: Mapping[str, Any],
    *,
    package_name: str,
    selections: Sequence[Any],
    files: Mapping[str, tuple[int, int, Path]],
    artifact_store: Path | None,
    verified: Callable[[str, int, Path], None],
    derive: Deriver | None,
) -> tuple[list[pb.Model], list[pb.Entrypoint]]:
    rows = body.get("entrypoints")
    if not isinstance(rows, list):
        raise PreparationRefusal("package_prepare_interface_invalid", "entrypoints")
    selected = _selected_models(package_name, selections, files, artifact_store, verified)
    if selected and derive is None:
        raise PreparationRefusal(
            "package_prepare_derive_refused", "a modeled selection needs the package environment"
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
            raise PreparationRefusal("package_prepare_interface_invalid", "entrypoint")
        described = value.get("models", [])
        if not isinstance(described, list):
            raise PreparationRefusal("package_prepare_interface_invalid", "model slots")
        described_paths: set[str] = set()
        for row in described:
            if not isinstance(row, dict) or not isinstance(row.get("path"), str):
                raise PreparationRefusal("package_prepare_interface_invalid", "model slot")
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
            if not path.startswith(prefix) or path not in selected:
                raise PreparationRefusal("package_prepare_model_selection_mismatch", path)
            described_slots.add(path)
            model = selected[path]
            model_id = model_ids[path]
            assert derive is not None
            constructed = derive(path, model["config_bytes"], model["assets"])
            slots.append(
                pb.Slot(
                    slot=path.removeprefix(prefix),
                    reference_model_id=model_id,
                    # WHAT THE DERIVE CONSTRUCTED, in construction order (D9) — never the
                    # header's component set. Whether those components exist in the
                    # checkpoint is the fit's question, answered at construction.
                    components=[
                        pb.Component(component=component, model_id=model_id)
                        for component in constructed
                    ],
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
        raise PreparationRefusal("package_prepare_interface_invalid", "duplicate entrypoint")
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
    interface_path: Path,
    installed: package_environment.InstalledEnvironment,
) -> set[str]:
    rows = body.get("jobs")
    if not isinstance(rows, list):
        raise PreparationRefusal("package_prepare_interface_invalid", "jobs")
    selected = [row for row in rows if isinstance(row, dict)]
    if not selected or root is None:
        return set()
    root.mkdir(parents=True, exist_ok=True, mode=0o755)
    if root.is_symlink() or not root.is_dir():
        raise PreparationRefusal("package_prepare_job_plan_root_invalid", str(root))
    application = str(body.get("application", ""))
    staged: set[str] = set()
    for row in selected:
        name = str(row.get("name", ""))
        descriptor_id = package_interface.job_descriptor_id(body, name)
        record = {
            "job_descriptor_id": descriptor_id,
            "build_id": build_id,
            "application": application,
            "package_interface": str(interface_path),
            "python": str(installed.python),
            "environment_content_digest": installed.content_digest,
            "job": name,
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
        delivered = {(digest, length) for digest, (length, _path) in object_files.items()}
        placed: set[str] = set()
        for digest, (_length, raw, _path) in sorted(manifests.items()):
            if carried_models:
                # The header is the sole direct snapshot entry needed to discover the runtime
                # closure. README, sample, license, and other ordinary siblings stay outside
                # model preparation even when the snapshot retains them.
                header_digest, header_length = acquire.cozytensors_ref(tensorfs, raw)
                row = object_files.get(header_digest)
                if row is None or row[0] != header_length:
                    raise PreparationRefusal(
                        "package_prepare_model_object_set_mismatch",
                        f"{digest} names CozyTensors header {header_digest} "
                        f"({header_length} B); not delivered",
                    )
                if header_digest not in placed:
                    store.put_file(str(row[1]), header_digest, header_length)
                    placed.add(header_digest)
            closure = {
                (str(row["id"]), int(row["length"])) for row in store.walk_cozytensors(digest)
            }
            closures[digest] = closure
            reached.update(closure)
        if carried_models and delivered != reached:
            missing = sorted(digest for digest, _length in reached - delivered)
            extra = sorted(digest for digest, _length in delivered - reached)
            raise PreparationRefusal(
                "package_prepare_model_object_set_mismatch",
                f"delivered={len(delivered)} reached={len(reached)} "
                f"missing={missing[:4]} extra={extra[:4]}",
            )
        for digest, (length, path) in sorted(object_files.items()):
            if digest not in placed:
                store.put_file(str(path), digest, length)
            verified(digest, length, path)
        for digest, (length, _raw, manifest_path) in sorted(manifests.items()):
            store.acquire_cozytensors(digest).release()
            held = store.manifest(digest)
            header = tensorfs.parse_header(bytes(held["header"]))
            configs = header.get("configs")
            if not header.get("components"):
                raise PreparationRefusal(
                    "package_prepare_model_header_unsupported",
                    f"{digest}: expected configs and components",
                )
            try:
                config_bytes = model_config.construction_config(configs)
                assets = fill.Checkpoint(artifact_store, digest).model_assets()
            except model_config.ModelConfigRefusal as exc:
                raise PreparationRefusal(
                    "package_prepare_model_config_mismatch", f"{digest}: {exc}"
                ) from exc
            for value in selected.values():
                if value["selection"]["manifest"] == digest:
                    value.update(
                        manifest_length=length,
                        config_bytes=config_bytes,
                        assets=assets,
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
