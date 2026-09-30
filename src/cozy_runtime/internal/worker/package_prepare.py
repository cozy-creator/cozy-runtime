"""Prepare one downloaded package set without giving package code worker authority."""

from __future__ import annotations

import contextlib
import hashlib
import importlib.metadata
import os
import re
import stat
import uuid
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import msgspec
from packaging.requirements import Requirement
from packaging.version import InvalidVersion, Version

from cozy_runtime.internal import (
    base_observation,
    canonical,
    census_cache,
    fill,
    installed_interfaces,
    job_plan,
    model_config,
    package_environment,
    package_installation,
    package_interface,
    python_interpreters,
)
from cozy_runtime.internal.canonical import Json
from cozy_runtime.internal.worker import acquire
from cozy_runtime.protocol import documents, weights_limits
from cozy_runtime.protocol import worker_pb2 as pb

MAX_FILE_BYTES = 8 << 30
_DISTRIBUTION = re.compile(r"[-_.]+")
_OPERATION_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,255}")


class PreparationRefusal(Exception):
    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


class CheckpointSelection(msgspec.Struct, frozen=True, kw_only=True):
    model: str = ""
    manifest: str = ""
    release: str = ""
    lane: str = ""


class AdapterSelection(CheckpointSelection, frozen=True, kw_only=True):
    component: str
    source_component: str = "adapter"
    scale: str = "1"


class Selection(CheckpointSelection, frozen=True, kw_only=True):
    """Original slot selection plus a prepared view and its ordered source custody."""

    package: str = ""
    slot: str = ""
    adapters: tuple[AdapterSelection, ...] = ()
    composed: CheckpointSelection | None = None


def selections(rows: Iterable[Mapping[str, object] | pb.DownloadModelRef]) -> list[Selection]:
    """Selection rows a caller states as JSON or a download set signs, decoded once here."""
    try:
        return [
            msgspec.convert(
                documents.body(row) if isinstance(row, pb.DownloadModelRef) else row, Selection
            )
            for row in rows
        ]
    except msgspec.ValidationError as exc:
        raise PreparationRefusal("package_prepare_model_selection_mismatch", str(exc)) from exc


class _Segment(msgspec.Struct, frozen=True):
    sha256: str
    length: int


class _Asset(msgspec.Struct, frozen=True):
    """A header-declared asset's stored segments, as TensorFS's header projection states them."""

    segments: tuple[_Segment, ...] = ()


@dataclass(frozen=True)
class Construction:
    """What constructing a served slot reads of its checkpoint's header."""

    unnormalized: str
    config_bytes: bytes
    header: bytes
    assets: Callable[[], dict[str, bytes]]
    tensor_dtypes: dict[str, str]
    adapters: bytes = b""


@dataclass(frozen=True)
class Selected:
    """A selection admitted from the local store: its held manifest length and, for a served
    slot, its construction facts."""

    selection: Selection
    manifest_length: int
    construction: Construction | None = None
    adapter_lengths: tuple[int, ...] = ()
    composed_length: int = 0


def _select_python(
    python: Path,
    requires: str,
    version: str,
    base: base_observation.Observation | None = None,
    *,
    root: Path | None = None,
    cancel: Callable[[], bool] | None = None,
    progress: Callable[[str], None] | None = None,
) -> tuple[Path, base_observation.Observation | None]:
    try:
        if not requires and not version:
            actual = python_interpreters.probe(python)
            if (
                ".".join(actual.version.split(".")[:2])
                not in python_interpreters.supported_minors()
            ):
                raise python_interpreters.InterpreterRefusal(actual.version)
            return python, base
        selected = (
            python_interpreters.ensure(
                requires, version, default=python, root=root, cancel=cancel, progress=progress
            )
            if root is not None
            else python_interpreters.select(requires, version, default=python)
        )
        if selected.executable == python:
            return python, base
        return selected.executable, package_environment.observe_base(selected.executable)
    except (python_interpreters.InterpreterRefusal, package_environment.EnvironmentRefusal) as exc:
        raise PreparationRefusal(exc.code, exc.detail) from exc


def prepare_package_set(
    request: pb.PreparePackageSetRequest,
    *,
    artifact_cache: Path,
    tensorfs_root: Path,
    install_root: Path,
    python: Path,
    verified: Callable[[str, int, Path], None],
    job_plan_root: Path | None = None,
    base: base_observation.Observation | None = None,
    preinstalled_python: Path | None = None,
    dependency_cache: Path | None = None,
    materialized: Callable[[package_installation.InstalledEnvironment], None] | None = None,
    cancel: Callable[[], bool] | None = None,
    progress: Callable[[str], None] | None = None,
    python_root: Path | None = None,
    describe: Callable[[package_installation.InstalledEnvironment, str], bytes] | None = None,
) -> pb.PreparePackageSetResult:
    """Install the requested release with uv and discover its installed interface.

    The worker owns the installation and its interface. The caller supplies release
    selection, dependency requirements and model selections, never a code fingerprint.
    While the Host is still landing the models (`models_landing`) none is read: the result
    is the installation, and its PlacementSet binds no model.
    """

    if (
        Path(request.install_root) != install_root
        or not install_root.is_absolute()
        or not tensorfs_root.is_absolute()
        or (job_plan_root is not None and not job_plan_root.is_absolute())
    ):
        raise PreparationRefusal("package_prepare_install_root_invalid", request.install_root)
    try:
        download_set = documents.parse(request.download_delegation, pb.DownloadDelegation)
    except documents.DocumentError as exc:
        raise PreparationRefusal("package_prepare_download_set_invalid", exc.detail) from exc
    if len(download_set.packages) != 1:
        raise PreparationRefusal("package_prepare_selection_invalid", "exactly one package")
    package_name = download_set.packages[0].package
    release = download_set.packages[0].release
    try:
        locked = package_environment.read_locked_requirements(bytes(request.locked_requirements))
    except package_environment.EnvironmentRefusal as exc:
        raise PreparationRefusal("package_prepare_locked_requirements_invalid", exc.detail) from exc
    python, base = _select_python(
        python,
        request.python_requires,
        request.python_version,
        base,
        root=python_root or install_root / "python",
        cancel=cancel,
        progress=progress,
    )
    return _prepare_published(
        package_name=package_name,
        release=release,
        locked=locked,
        models=[] if request.models_landing else selections(download_set.models),
        artifact_cache=artifact_cache,
        tensorfs_root=tensorfs_root,
        install_root=install_root,
        python=python,
        interface=lambda installed, distribution: _installed_interface(
            installed, distribution, artifact_cache, describe
        ),
        verified=verified,
        job_plan_root=job_plan_root,
        base=base,
        dependency_cache=dependency_cache,
        materialized=materialized,
        hub=request.hub,
    )


def landing_models(request: pb.PreparePackageSetRequest, interface: bytes) -> list[Json]:
    """The model slots of every serving entrypoint the download set selects whole: what a
    root of this preparation constructs, and so the cards its executor starts on."""
    try:
        download_set = documents.parse(request.download_delegation, pb.DownloadDelegation)
    except documents.DocumentError as exc:
        raise PreparationRefusal("package_prepare_download_set_invalid", exc.detail) from exc
    selected = {row.slot for row in download_set.models}
    typed = package_interface.parse(interface, "landing package interface")
    body = canonical.parse(interface)
    return [
        slot
        for entry, document in zip(typed.entrypoints, body["entrypoints"], strict=True)
        if entry.models and {model.path for model in entry.models} <= selected
        for slot in document["models"]
    ]


def _installed_interface(
    installed: package_installation.InstalledEnvironment,
    distribution: str,
    artifact_cache: Path,
    describe: Callable[[package_installation.InstalledEnvironment, str], bytes] | None,
) -> bytes:
    """Read the interface from this installed environment, never a host SDK snapshot."""
    candidates = [
        item
        for item in importlib.metadata.distributions(path=[str(installed.site_packages)])
        if _DISTRIBUTION.sub("-", item.metadata["Name"]).lower() == distribution
    ]
    if len(candidates) != 1:
        raise PreparationRefusal(
            "package_prepare_interface_missing",
            f"installed distribution {distribution!r} is absent or ambiguous",
        )
    metadata = [
        Path(str(candidates[0].locate_file(row)))
        for row in candidates[0].files or ()
        if str(row).endswith(".dist-info/package-interface.json")
    ]
    if len(metadata) > 1:
        raise PreparationRefusal(
            "package_prepare_interface_invalid", f"{distribution!r} has multiple interfaces"
        )
    if metadata:
        raw = _read_installed_interface(metadata[0], installed.site_packages)
    elif describe is not None:
        raw = describe(installed, distribution)
    else:
        raise PreparationRefusal(
            "package_prepare_interface_missing",
            f"{distribution!r} has no installed interface and no sandbox describe is available",
        )
    try:
        interface = package_interface.parse(raw, "installed package interface")
    except package_interface.StalePackageInterface as exc:
        raise PreparationRefusal("package_prepare_interface_invalid", str(exc)) from exc
    if (
        metadata
        and describe is not None
        and any(
            entry.invocable is not msgspec.UNSET and entry.invocable.memoize
            for _, entry in interface.callables()
        )
    ):
        # Installed callee/helper definitions own memo invalidation. The wheel's
        # static schema is never a second code reading to compare or admit against.
        raw = describe(installed, distribution)
        package_interface.parse(raw, "installed operation interface")
    return raw


def _read_installed_interface(path: Path, site_packages: Path) -> bytes:
    """Follow uv's installed metadata links, including the immutable image seed.

    Only an installed distribution's direct interface member permits links. Cache
    artifacts and receipts continue to use the no-symlink `_read_file` boundary.
    """
    maximum = canonical.DOC_MAX_BYTES
    try:
        relative = path.relative_to(site_packages)
        if (
            len(relative.parts) != 2
            or not relative.parts[0].endswith(".dist-info")
            or relative.parts[1] != "package-interface.json"
        ):
            raise OSError("invalid installed interface path")
        resolved = path.resolve(strict=True)
        if not stat.S_ISREG(resolved.lstat().st_mode):
            raise OSError("invalid file")
        with resolved.open("rb") as stream:
            info = os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode) or not 0 < info.st_size <= maximum:
                raise OSError("invalid file")
            raw = stream.read(maximum + 1)
        if not 0 < len(raw) <= maximum:
            raise OSError("invalid file size")
        return raw
    except (OSError, RuntimeError, ValueError) as exc:
        raise PreparationRefusal("package_prepare_file_unreadable", str(path)) from exc


def prepare_published_local(
    *,
    package_name: str,
    release: str,
    locked: package_environment.LockedRequirements,
    models: Sequence[Mapping[str, object]],
    artifact_cache: Path,
    tensorfs_root: Path,
    install_root: Path,
    python: Path,
    describe: Callable[[package_installation.InstalledEnvironment, str], bytes],
    installed_environment: package_installation.InstalledEnvironment,
    job_plan_root: Path | None = None,
) -> pb.PreparePackageSetResult:
    """Creator's local published preparation over its already-materialized locked venv."""

    return _prepare_published(
        package_name=package_name,
        release=release,
        locked=locked,
        models=selections(models),
        artifact_cache=artifact_cache,
        tensorfs_root=tensorfs_root,
        install_root=install_root,
        python=python,
        interface=lambda installed, distribution: _installed_interface(
            installed, distribution, artifact_cache, describe
        ),
        verified=lambda _digest, _length, _path: None,
        job_plan_root=job_plan_root,
        base=None,
        installed_environment=installed_environment,
    )


def _same_version(locked: str, release: str) -> bool:
    """`1.0.0rc1` (a wheel's spelling) and `1.0.0-rc.1` (a release's) are one version."""
    try:
        return Version(locked) == Version(release)
    except InvalidVersion:
        return locked == release


def _filtered_rows(
    package_name: str,
    release: str,
    locked: package_environment.LockedRequirements,
) -> list[package_environment.LockedRow]:
    """Validate the selected project pin and retain the complete locked wheel closure."""

    expected = _DISTRIBUTION.sub("-", package_name.rsplit("/", 1)[-1]).lower()
    projects = [row for row in locked.rows if row.name == expected]
    if not projects or any(not _same_version(row.version, release) for row in projects):
        raise PreparationRefusal(
            "package_prepare_project_pin_missing", f"the export must pin {expected}=={release}"
        )
    return list(locked.rows)


def _prepare_published(
    *,
    package_name: str,
    release: str,
    locked: package_environment.LockedRequirements,
    models: Sequence[Selection],
    artifact_cache: Path,
    tensorfs_root: Path | None,
    install_root: Path,
    python: Path,
    interface: Callable[[package_installation.InstalledEnvironment, str], bytes],
    verified: Callable[[str, int, Path], None],
    job_plan_root: Path | None,
    base: base_observation.Observation | None,
    installed_environment: package_installation.InstalledEnvironment | None = None,
    dependency_cache: Path | None = None,
    materialized: Callable[[package_installation.InstalledEnvironment], None] | None = None,
    hub: str = "",
) -> pb.PreparePackageSetResult:
    """Install from the exact export, read the interface, and author the PlacementSet.

    The installed interface is the authority for application and model slots; caller
    copies of those facts are advisory. Both local and rented preparation materialize the
    complete locked closure; the controller SDK never supplies package dependencies.
    """

    _filtered_rows(package_name, release, locked)
    if installed_environment is not None:
        installed = installed_environment
    else:
        requirements = (
            "\n".join(
                [
                    *(
                        f"{('--index-url' if i == 0 else '--extra-index-url')} {url}"
                        for i, url in enumerate(locked.indexes)
                    ),
                    *(row.text for row in locked.rows),
                ]
            ).encode()
            + b"\n"
        )
        try:
            installed = package_installation.install(
                install_root,
                package=package_name,
                release=release,
                python=python,
                installation_id=installation_key(package_name, release, locked, python),
                requirements=requirements,
                cache=dependency_cache,
            )
        except package_environment.EnvironmentRefusal as exc:
            raise PreparationRefusal(exc.code, exc.detail) from exc

    expected_distribution = _DISTRIBUTION.sub("-", package_name.rsplit("/", 1)[-1]).lower()
    interface_raw = interface(installed, expected_distribution)
    try:
        interface_body = package_interface.parse(interface_raw, "release package interface")
    except package_interface.StalePackageInterface as exc:
        raise PreparationRefusal("package_prepare_interface_invalid", str(exc)) from exc
    interface_ref = _stage(artifact_cache, interface_raw)
    jobs = _stage_job_plans(
        interface_raw,
        root=job_plan_root,
        installation_id=installed.installation_id,
        interface_path=artifact_cache / interface_ref[0].removeprefix("sha256:"),
        installed=installed,
    )
    selected_models, entrypoints = _entrypoints(
        interface_body,
        package_name=package_name,
        selections=models,
        tensorfs_root=tensorfs_root,
        verified=verified,
        census=_census(installed, interface_body.application, artifact_cache),
    )
    # A captured serving dependency needs its exact code/interface before its
    # caller chooses models. Retain that sealed revision without advertising an
    # executable binding, just as unpublished preparation does.
    code_only = (
        not models
        and bool(interface_body.entrypoints)
        and all(row.models for row in interface_body.entrypoints)
    )
    if not entrypoints and not jobs and not code_only:
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
        placement_id=placement_id(package_name, release),
        package=pb.PackageSelection(package=package_name, release=release, hub=hub),
        installation_id=installed.installation_id,
        package_interface=interface_raw,
        bindings_digest=bindings_digest,
        models=selected_models,
        entrypoints=entrypoints,
    )
    placement_bytes = canonical.write(
        {"format": "cozy.worker.v1.PlacementSet/1", "placements": [documents.body(placement)]}
    )
    if materialized is not None:
        materialized(installed)
    return pb.PreparePackageSetResult(
        installed_package=pb.InstalledPackage(
            installation_id=installed.installation_id,
            package=package_name,
            release=release,
            package_interface=interface_raw,
        ),
        placement_set=pb.DesiredPlacementSet(
            placement_set_digest=documents.digest_of(placement_bytes),
            placement_set_canonical_bytes=placement_bytes,
        ),
    )


def installation_key(
    package_name: str,
    release: str,
    locked: package_environment.LockedRequirements,
    python: Path,
) -> str:
    """A published environment's identity: its release, locked closure, interpreter ABI and
    the machine SDK its Runtime was chosen against.

    Index URLs and wheel locations are where bytes came from, not what was installed, so
    a mirror change reuses the environment; a different lock, ABI or machine Runtime makes a
    new one, and the superseded one is collected.
    """
    try:
        abi = python_interpreters.probe(python).abi
    except python_interpreters.InterpreterRefusal as exc:
        raise PreparationRefusal(exc.code, exc.detail) from exc
    rows = sorted(
        f"{row.name}=={row.version};{Requirement(row.text.split(' --')[0]).marker or ''};"
        + ",".join(sorted(row.hashes))
        for row in locked.rows
    )
    document = {
        "package": package_name,
        "release": release,
        "abi": abi,
        "rows": rows,
        "sdk": package_installation.machine_sdk(),
    }
    digest = hashlib.sha256(canonical.write(document)).hexdigest()[:32]
    return package_installation.PUBLISHED_PREFIX + digest


def _printable(value: str) -> str:
    return "".join(character if " " <= character <= "~" else "?" for character in value)[:1024]


def placement_identity(package_name: str, release: str) -> bytes:
    """One placement per package release, stable as models are added (wire 61)."""

    return package_name.encode() + b"\x00" + release.encode()


def placement_id(package_name: str, release: str) -> str:
    return "package-" + hashlib.sha256(placement_identity(package_name, release)).hexdigest()[:24]


def model_id(slot: str) -> str:
    """A slot's Model id, stable as other slots are added (wire 61)."""

    return "model-" + hashlib.sha256(slot.encode()).hexdigest()[:16]


def prepare_local_package(
    request: pb.PrepareLocalPackageRequest,
    *,
    artifact_cache: Path,
    install_root: Path,
    python: Path,
    describe: Callable[[package_installation.InstalledEnvironment, str], bytes],
    verified: Callable[[str, int, Path], None],
    job_plan_root: Path | None = None,
    base: base_observation.Observation | None = None,
    dependency_python: Path | None = None,
    builtin_snapshot: package_installation.InstalledEnvironment | None = None,
    dependency_cache: Path | None = None,
    materialized: Callable[[package_installation.InstalledEnvironment], None] | None = None,
    cancel: Callable[[], bool] | None = None,
    progress: Callable[[str], None] | None = None,
    python_root: Path | None = None,
    hub: package_installation.HubIndex | None = None,
) -> pb.PreparePackageSetResult:
    """Install a private source sync without publishing it to a repository."""

    if (
        Path(request.install_root) != install_root
        or not install_root.is_absolute()
        or (job_plan_root is not None and not job_plan_root.is_absolute())
    ):
        raise PreparationRefusal("local_package_install_root_invalid", request.install_root)
    package = request.package
    if (
        _OPERATION_ID.fullmatch(request.operation_id) is None
        or package.package.count("/") != 1
        or not package.release
        or not package.installation_id
    ):
        raise PreparationRefusal("local_package_identity_invalid", request.operation_id)
    return prepare_package(
        python_root=python_root,
        cancel=cancel,
        progress=progress,
        package_name=package.package,
        release=package.release,
        files=request.files,
        source_archive=request.source_archive,
        # proto-031: editable wheels stage beside the install namespace that owns them.
        wheel_root=install_root / ".stage" / request.operation_id / "wheels",
        installation_id=package.installation_id,
        artifact_cache=artifact_cache,
        install_root=install_root,
        python=python,
        describe=describe,
        verified=verified,
        job_plan_root=job_plan_root,
        base=base,
        development=package,
        dependency_python=dependency_python,
        builtin_snapshot=builtin_snapshot,
        dependency_cache=dependency_cache,
        materialized=materialized,
        dependency_requirements=bytes(request.dependency_requirements),
        python_requires=request.python_requires,
        python_version=request.python_version,
        hub=hub,
    )


def prepare_unpublished_placement(
    request: pb.PreparePrivatePlacementRequest,
    *,
    prepared: pb.Placement,
    installed: package_installation.InstalledEnvironment | None,
    artifact_cache: Path,
    tensorfs_root: Path | None,
    verified: Callable[[str, int, Path], None],
    model_overrides: tuple[Sequence[Mapping[str, object]], Sequence[pb.NativeModelBinding]]
    | None = None,
) -> pb.PreparePackageSetResult:
    """Bind retained models to an already-installed private package."""

    if installed is None:
        raise PreparationRefusal(
            "private_placement_environment_absent",
            f"{request.operation_id}: the installation's environment is not held, so its "
            "slots cannot be derived",
        )
    if (
        _OPERATION_ID.fullmatch(request.operation_id) is None
        or request.installation_id != installed.installation_id
        or prepared.installation_id != installed.installation_id
    ):
        raise PreparationRefusal("private_placement_identity_invalid", request.operation_id)
    models: Sequence[pb.DownloadModelRef] = ()
    if request.download_delegation:
        try:
            download_set = documents.parse(request.download_delegation, pb.DownloadDelegation)
        except documents.DocumentError as exc:
            raise PreparationRefusal("private_placement_download_set_invalid", exc.detail) from exc
        if download_set.packages:
            raise PreparationRefusal(
                "private_placement_download_set_invalid",
                "an unpublished model download set carries no package refs",
            )
        models = download_set.models
    elif request.download_delegation_signature:
        raise PreparationRefusal(
            "private_placement_download_set_invalid", "download signature has no model set"
        )
    return prepare_model_placement(
        prepared=prepared,
        installed=installed,
        models=models if model_overrides is None else model_overrides[0],
        native_models=request.native_models if model_overrides is None else model_overrides[1],
        artifact_cache=artifact_cache,
        tensorfs_root=tensorfs_root,
        verified=verified,
    )


def prepare_model_placement(
    *,
    prepared: pb.Placement,
    installed: package_installation.InstalledEnvironment,
    native_models: Sequence[pb.NativeModelBinding],
    artifact_cache: Path,
    tensorfs_root: Path | None,
    verified: Callable[[str, int, Path], None],
    models: Sequence[Mapping[str, object] | pb.DownloadModelRef] = (),
) -> pb.PreparePackageSetResult:
    """Derive new Model bindings from either origin's exact immutable preparation."""
    package = acquire.package_selection(prepared).package
    slots = [row.slot for row in native_models]
    if slots != sorted(set(slots)) or not all(slots):
        raise PreparationRefusal(
            "prepared_placement_native_models_invalid",
            "native model slots must be unique and sorted",
        )
    chosen = [
        *selections(models),
        *(
            Selection(
                package=package,
                slot=row.slot,
                model=row.model,
                manifest=documents.spell(row.manifest.digest),
            )
            for row in native_models
        ),
    ]
    if not 0 < len(chosen) <= 32:
        raise PreparationRefusal(
            "prepared_placement_models_invalid", "a placement requires between 1 and 32 model slots"
        )
    try:
        interface_body = package_interface.parse(
            prepared.package_interface, "prepared package interface"
        )
    except package_interface.StalePackageInterface as exc:
        raise PreparationRefusal("prepared_placement_interface_invalid", str(exc)) from exc
    selected_models, entrypoints = _entrypoints(
        interface_body,
        package_name=package,
        selections=chosen,
        tensorfs_root=tensorfs_root,
        verified=verified,
        census=_census(installed, interface_body.application, artifact_cache),
    )
    joined = pb.Placement()
    joined.CopyFrom(prepared)
    joined.ClearField("models")
    joined.models.extend(selected_models)
    joined.ClearField("entrypoints")
    joined.entrypoints.extend(entrypoints)
    joined.bindings_digest = documents.digest_of(
        canonical.write(
            {
                "entrypoints": [documents.body(row) for row in entrypoints],
                "models": [documents.body(row) for row in selected_models],
            }
        )
    )
    placement_bytes, digest = documents.identity(pb.PlacementSet(placements=[joined]))
    return pb.PreparePackageSetResult(
        placement_set=pb.DesiredPlacementSet(
            placement_set_digest=digest, placement_set_canonical_bytes=placement_bytes
        )
    )


def prepare_package(
    *,
    package_name: str,
    release: str,
    files: Sequence[pb.LocalPackageFile],
    wheel_root: Path,
    installation_id: str,
    artifact_cache: Path,
    install_root: Path,
    python: Path,
    describe: Callable[[package_installation.InstalledEnvironment, str], bytes],
    verified: Callable[[str, int, Path], None],
    source_archive: str = "",
    base: base_observation.Observation | None = None,
    dependency_python: Path | None = None,
    builtin_snapshot: package_installation.InstalledEnvironment | None = None,
    dependency_cache: Path | None = None,
    job_plan_root: Path | None = None,
    development: pb.DevelopmentPackage | None = None,
    dependency_requirements: bytes = b"",
    python_requires: str = "",
    python_version: str = "",
    materialized: Callable[[package_installation.InstalledEnvironment], None] | None = None,
    cancel: Callable[[], bool] | None = None,
    progress: Callable[[str], None] | None = None,
    python_root: Path | None = None,
    hub: package_installation.HubIndex | None = None,
) -> pb.PreparePackageSetResult:
    """Install a synchronized source project or ordinary wheels; describe installed code."""
    if development is None:
        raise PreparationRefusal("package_prepare_development_absent", package_name)
    if not 1 <= len(files) <= weights_limits.MAX_LOCAL_PACKAGE_FILES:
        raise PreparationRefusal("package_prepare_file_count", str(len(files)))
    try:
        installed = package_installation.open_installation(install_root, installation_id)
    except package_environment.EnvironmentRefusal as exc:
        if exc.code != "package_installation_absent":
            raise PreparationRefusal(exc.code, exc.detail) from exc
        installed = None
    if installed is not None:
        if (installed.package, installed.release) != (package_name, release):
            raise PreparationRefusal(
                "package_installation_conflict",
                "installation ID belongs to another package release",
            )
        try:
            installed = package_installation.refresh(
                install_root, installation_id, cache=dependency_cache
            )
        except package_environment.EnvironmentRefusal as exc:
            raise PreparationRefusal(exc.code, exc.detail) from exc
    elif any(not row.path for row in files):
        # Host removes operation carriers after recording successful installation.
        # Its inventory probe/replay intentionally has names but no transfer paths.
        raise PreparationRefusal("local_package_reuse_unavailable", installation_id)
    else:
        installed = _install_private_files(
            package_name=package_name,
            release=release,
            files=files,
            wheel_root=wheel_root,
            installation_id=installation_id,
            install_root=install_root,
            python=python,
            verified=verified,
            source_archive=source_archive,
            base=base,
            dependency_cache=dependency_cache,
            dependency_requirements=dependency_requirements,
            python_requires=python_requires,
            python_version=python_version,
            cancel=cancel,
            progress=progress,
            python_root=python_root,
            hub=hub,
        )
    if materialized is not None:
        materialized(installed)
    distribution = _DISTRIBUTION.sub("-", package_name.rsplit("/", 1)[-1]).lower()
    interface_raw = (
        _installed_interface(installed, distribution, artifact_cache, describe)
        if installed.incarnation is not None
        else describe(installed, distribution)
    )
    interface_body = package_interface.parse(interface_raw, "installed package interface")
    interface_path = install_root / "installations" / installation_id / "package-interface.json"
    package_interface.publish(interface_path, interface_raw)
    _stage_job_plans(
        interface_raw,
        root=job_plan_root,
        installation_id=installation_id,
        interface_path=interface_path,
        installed=installed,
    )
    selected_models, entrypoints = _entrypoints(
        interface_body,
        package_name=package_name,
        selections=(),
        tensorfs_root=None,
        verified=verified,
        census=None,
    )
    bindings_digest = documents.digest_of(
        canonical.write(
            {
                "entrypoints": [documents.body(row) for row in entrypoints],
                "models": [],
            }
        )
    )
    placement = pb.Placement(
        placement_id="package-" + installation_id,
        development=pb.DevelopmentPackage(
            package=package_name,
            release=release,
            installation_id=installation_id,
        ),
        installation_id=installation_id,
        package_interface=interface_raw,
        bindings_digest=bindings_digest,
        models=selected_models,
        entrypoints=entrypoints,
    )
    placement_bytes = canonical.write(
        {
            "format": "cozy.worker.v1.PlacementSet/1",
            "placements": [documents.body(placement)],
        }
    )
    return pb.PreparePackageSetResult(
        installed_package=pb.InstalledPackage(
            installation_id=installation_id,
            package=package_name,
            release=release,
            package_interface=interface_raw,
        ),
        placement_set=pb.DesiredPlacementSet(
            placement_set_digest=documents.digest_of(placement_bytes),
            placement_set_canonical_bytes=placement_bytes,
        ),
    )


def _install_private_files(
    *,
    package_name: str,
    release: str,
    files: Sequence[pb.LocalPackageFile],
    wheel_root: Path,
    installation_id: str,
    install_root: Path,
    python: Path,
    verified: Callable[[str, int, Path], None],
    source_archive: str,
    base: base_observation.Observation | None,
    dependency_cache: Path | None,
    dependency_requirements: bytes,
    python_requires: str,
    python_version: str,
    cancel: Callable[[], bool] | None,
    progress: Callable[[str], None] | None,
    python_root: Path | None,
    hub: package_installation.HubIndex | None,
) -> package_installation.InstalledEnvironment:
    """Read transferred carriers only when this installation does not exist yet."""
    python, base = _select_python(
        python,
        python_requires,
        python_version,
        base,
        root=python_root or install_root / "python",
        cancel=cancel,
        progress=progress,
    )
    archive = None
    wheels = []
    names = [row.filename for row in files]
    if names != sorted(set(names)):
        raise PreparationRefusal("package_prepare_file_invalid", "filenames must be sorted unique")
    for row in files:
        path = Path(row.path)
        if (
            not path.is_absolute()
            or path.name != row.filename
            or path.parent != wheel_root
            or path.resolve(strict=True) != path
            or not 0 < row.length <= MAX_FILE_BYTES
            or path.stat().st_size != row.length
        ):
            raise PreparationRefusal("package_prepare_file_path_invalid", row.filename)
        if row.filename == source_archive:
            if row.digest:
                raise PreparationRefusal(
                    "package_prepare_source_invalid", "source is operation-scoped"
                )
            archive = path
        else:
            if (
                len(row.digest) != 32
                or package_environment.wheel_name_version(row.filename) is None
            ):
                raise PreparationRefusal("package_prepare_file_invalid", row.filename)
            with path.open("rb") as stream:
                measured = hashlib.file_digest(stream, "sha256").digest()
            if measured != row.digest:
                raise PreparationRefusal("package_prepare_wheel_integrity", row.filename)
            verified(documents.spell(row.digest), row.length, path)
            wheels.append(path)
    if source_archive and archive is None:
        raise PreparationRefusal("package_prepare_source_absent", source_archive)
    try:
        installed = package_installation.install(
            install_root,
            package=package_name,
            release=release,
            python=python,
            installation_id=installation_id,
            source_archive=archive,
            requirements=dependency_requirements,
            wheels=tuple(wheels),
            cache=dependency_cache,
            hub=hub,
        )
    except package_environment.EnvironmentRefusal as exc:
        raise PreparationRefusal(exc.code, exc.detail) from exc
    return installed


#: `{slot path: (selected model, interface slot)} -> {slot path: census}`.
CensusLookup = Callable[
    [Mapping[str, tuple[Selected, package_interface.ModelSlot]]], Mapping[str, census_cache.Census]
]


def _census(
    installed: package_installation.InstalledEnvironment, application: str, artifact_cache: Path
) -> CensusLookup:
    """`Slot.components` is what the factory constructs for this config (D9), read from the
    content-keyed census cache; misses construct once, in the package's own environment,
    through the `derive_child` seam (cr-067's guard, kept by construction)."""

    def lookup(
        wanted: Mapping[str, tuple[Selected, package_interface.ModelSlot]],
    ) -> Mapping[str, census_cache.Census]:
        if not wanted:
            return {}
        built = {path: (_constructed(model), slot) for path, (model, slot) in wanted.items()}
        try:
            return census_cache.ensure(
                artifact_cache / "census",
                python=installed.python,
                environment=installed.installation_id,
                application=application,
                slots=[
                    census_cache.SlotInput(
                        slot=path,
                        config=construction.config_bytes,
                        tensor_dtypes=construction.tensor_dtypes,
                        header=construction.header,
                        encoded_leaves=slot.encoded_leaves == "accept",
                        assets=construction.assets,
                        adapters=construction.adapters,
                    )
                    for path, (construction, slot) in sorted(built.items())
                ],
            )
        except census_cache.CensusRefusal as exc:
            # A construction that cannot read a checkpoint stored as-is says so, and why.
            remedy = next((c.unnormalized for c, _ in built.values() if c.unnormalized), "")
            if remedy:
                raise PreparationRefusal(
                    model_config.UNNORMALIZED, f"{remedy}; {exc.code}: {exc.detail}"
                ) from exc
            raise PreparationRefusal(exc.code, exc.detail) from exc

    return lookup


def _constructed(model: Selected) -> Construction:
    # Every entrypoint slot is a construction slot, so its selection read the header.
    assert model.construction is not None
    return model.construction


def _slot_paths(callables: Iterable[package_interface.CallableDoc]) -> set[str]:
    """Every model-slot path the callables declare.

    cozy.package.interface/1 declares `models[].path` in BOTH `entrypoints` and `jobs`.
    Reading only the first is how a release whose callables are all jobs came to describe
    no model slot at all, and how the one slot a job bound became undeclarable.
    """
    return {model.path for entry in callables for model in entry.models}


def _entrypoints(
    interface: package_interface.PackageInterface,
    *,
    package_name: str,
    selections: Sequence[Selection],
    tensorfs_root: Path | None,
    verified: Callable[[str, int, Path], None],
    census: CensusLookup | None,
) -> tuple[list[pb.Model], list[pb.Entrypoint]]:
    construction_slots = _slot_paths(interface.entrypoints)
    selected = _selected_models(
        package_name, selections, tensorfs_root, verified, construction_slots=construction_slots
    )
    if construction_slots.intersection(selected) and census is None:
        raise PreparationRefusal(
            "package_prepare_derive_refused", "a modeled selection needs the package environment"
        )
    model_ids = {slot: model_id(slot) for slot in selected}
    models = [
        pb.Model(
            id=model_ids[slot],
            repo=value.selection.model,
            version=value.selection.release,
            lane=value.selection.lane,
            manifest=pb.Ref(
                digest=documents.raw(value.selection.manifest), length=value.manifest_length
            ),
        )
        for slot, value in sorted(selected.items(), key=lambda item: model_ids[item[0]])
    ]
    adapter_ids: dict[str, list[str]] = {}
    composed_ids: dict[str, str] = {}
    for path, value in sorted(selected.items()):
        adapter_ids[path] = []
        extra: list[tuple[str, CheckpointSelection, int]] = [
            (model_id(f"{path}.adapter.{index}"), row, length)
            for index, (row, length) in enumerate(
                zip(value.selection.adapters, value.adapter_lengths, strict=True)
            )
        ]
        adapter_ids[path] = [identifier for identifier, _, _ in extra]
        if value.selection.composed is not None:
            composed_ids[path] = model_id(path + ".composed")
            extra.append((composed_ids[path], value.selection.composed, value.composed_length))
        for identifier, source, length in extra:
            models.append(
                pb.Model(
                    id=identifier,
                    repo=source.model,
                    version=source.release,
                    lane=source.lane,
                    manifest=pb.Ref(digest=documents.raw(source.manifest), length=length),
                )
            )
    models.sort(key=lambda row: row.id)
    bound: list[package_interface.CallableDoc] = []
    for entry in interface.entrypoints:
        paths = {model.path for model in entry.models}
        if len(paths) != len(entry.models):
            raise PreparationRefusal(
                "package_prepare_model_selection_mismatch", f"{entry.name}: duplicate slot"
            )
        if paths and not paths.issubset(selected):
            # One desired package set names the entrypoints it can actually bind. A
            # different invocation may select another model later; it should not force
            # this worker to download unrelated weights now.
            continue
        for path in paths:
            if not path.startswith(f"{entry.name}.models."):
                raise PreparationRefusal("package_prepare_model_selection_mismatch", path)
        bound.append(entry)
    wanted = {
        model.path: (selected[model.path], model) for entry in bound for model in entry.models
    }
    censuses = census(wanted) if census is not None else {}
    out: list[pb.Entrypoint] = []
    for entry in bound:
        prefix = f"{entry.name}.models."
        slots: list[pb.Slot] = []
        for row in entry.models:
            path = row.path
            chosen = selected[path].selection
            found = censuses[path]
            if not found.fit_ok:
                raise PreparationRefusal(
                    "checkpoint_incompatible",
                    _printable(
                        f"{path} does not fit {chosen.model}@{chosen.release}/{chosen.lane}: "
                        f"{found.fit_detail}"
                    ),
                )
            slots.append(
                pb.Slot(
                    slot=path.removeprefix(prefix),
                    reference_model_id=model_ids[path],
                    # WHAT THE DERIVE CONSTRUCTED, in construction order (D9) — never the
                    # header's component set; the census's fit proved they agree.
                    components=[
                        pb.Component(
                            component=component, model_id=composed_ids.get(path, model_ids[path])
                        )
                        for component in found.components
                    ],
                    adapters=[
                        pb.ModelAdapter(
                            component=adapter.component,
                            model_id=identifier,
                            source_component=adapter.source_component,
                            scale=adapter.scale,
                        )
                        for adapter, identifier in zip(
                            chosen.adapters, adapter_ids[path], strict=True
                        )
                    ],
                )
            )
        slots.sort(key=lambda row: row.slot)
        entrypoint = pb.Entrypoint(name=entry.name, slots=slots)
        entrypoint.entrypoint_binding_digest = documents.digest_of(
            canonical.write(
                {"name": entrypoint.name, "slots": [documents.body(slot) for slot in slots]}
            )
        )
        out.append(entrypoint)
    out.sort(key=lambda row: row.name)
    if len({row.name for row in out}) != len(out):
        raise PreparationRefusal("package_prepare_interface_invalid", "duplicate entrypoint")
    unbound = sorted(set(selected) - set(wanted))
    if unbound:
        # A JOB's model slot is legitimately unbound here. A job has no placement and no
        # residency: its Model parameter is an attempt-held, derive-only view of one
        # TensorFS Manifest, so the selection exists to admit those bytes to this pod,
        # not to construct an Entrypoint binding. Only a selection NO callable declares
        # -- or one that would leave an entrypoint half-bound -- is a mismatch.
        stranded = [path for path in unbound if path not in _slot_paths(interface.jobs)]
        if stranded:
            raise PreparationRefusal(
                "package_prepare_model_selection_mismatch",
                f"nothing binds {stranded}: no job declares them, and no entrypoint that "
                "declares them had its whole slot set selected",
            )
    return models, out


def _stage_job_plans(
    interface_raw: bytes,
    *,
    root: Path | None,
    installation_id: str,
    interface_path: Path,
    installed: package_installation.InstalledEnvironment,
) -> set[str]:
    interface = package_interface.parse(interface_raw, "staged package interface")
    if not interface.jobs or root is None:
        return set()
    root.mkdir(parents=True, exist_ok=True, mode=0o755)
    if root.is_symlink() or not root.is_dir():
        raise PreparationRefusal("package_prepare_job_plan_root_invalid", str(root))
    body = canonical.parse(interface_raw)
    call_interfaces = installed_interfaces.read(installed)
    staged: set[str] = set()
    for job in interface.jobs:
        descriptor_id = package_interface.job_descriptor_id(body, job.name)
        record: dict[str, Json] = {
            "job_descriptor_id": descriptor_id,
            "installation_id": installation_id,
            "application": interface.application,
            "package_interface": str(interface_path),
            "python": str(installed.python),
            "job": job.name,
            "publishes": job.publishes is True,
            "emits_media": False,
            "gpu_rate_micro_usd_per_hour": 0,
            "cap_micro_usd": 0,
        }
        if call_interfaces:
            record["call_interfaces"] = call_interfaces
        try:
            job_plan.write(root, record)
        except ValueError as exc:
            raise PreparationRefusal("local_package_job_plan_conflict", str(exc)) from exc
        staged.add(descriptor_id)
    return staged


def _selected_models(
    package_name: str,
    selections: Sequence[Selection],
    tensorfs_root: Path | None,
    verified: Callable[[str, int, Path], None],
    *,
    construction_slots: set[str],
) -> dict[str, Selected]:
    """Resolve each selected model from the local TensorFS Store.

    The privileged host and TensorFS admitted every model byte before preparation ran; this
    plane cross-checks release rows when selected, or reads an exact admitted checkpoint
    without a release alias. It takes no closure lease: the placement's acquisition is the
    one verified walk. Serving slots carry the header, config and dtype projection; their
    assets are read (asset segments only) when the census cache misses. Job Model arguments
    are source-only capabilities. Nothing here writes to the store.
    """

    chosen: dict[str, Selection] = {}
    for row in selections:
        if row.package != package_name:
            raise PreparationRefusal("package_prepare_model_selection_mismatch", "package")
        if not row.slot:
            raise PreparationRefusal("package_prepare_model_selection_mismatch", "slot")
        first = chosen.setdefault(row.slot, row)
        if first != row:
            raise PreparationRefusal(
                "package_prepare_model_selection_mismatch",
                f"{package_name} slot {row.slot} is selected twice with different models "
                f"({first.model} {first.manifest} and {row.model} {row.manifest}); "
                "select one model for this slot",
            )
    if not chosen:
        return {}

    if tensorfs_root is None:
        raise PreparationRefusal(
            "package_prepare_model_store_absent", "modeled package requires a TensorFS store"
        )

    try:
        facade = fill.tensorfs_module()
        store = fill.open_store(tensorfs_root)
        manifests: dict[str, tuple[int, bytes, bytes]] = {}
        closure = [
            source
            for selected in chosen.values()
            for source in (
                selected,
                *selected.adapters,
                *((selected.composed,) if selected.composed else ()),
            )
        ]
        for selection in closure:
            # The selected manifest digest is the model's identity; release and lane are
            # labels. Where a mutable lane points now does not change what was selected.
            digest = selection.manifest
            parts = selection.model.split("/")
            if len(parts) != 2 or not all(parts) or bool(selection.release) != bool(selection.lane):
                raise PreparationRefusal(
                    "package_prepare_model_selection_mismatch", selection.model
                )
            held = store.manifest(digest)
            raw, held_header = held["manifest"], held["header"]
            if (
                not raw
                or held_header is None
                or "sha256:" + hashlib.sha256(raw).hexdigest() != digest
            ):
                raise PreparationRefusal("package_prepare_model_manifest_mismatch", digest)
            manifests[digest] = (len(raw), raw, held_header)
        constructions: dict[str, Construction] = {}
        for digest, (_length, raw, header_raw) in sorted(manifests.items()):
            header_digest, header_length = acquire.cozytensors_ref(raw)
            if (
                len(header_raw) != header_length
                or "sha256:" + hashlib.sha256(header_raw).hexdigest() != header_digest
            ):
                raise PreparationRefusal(
                    "package_prepare_model_header_unavailable",
                    f"{digest}: held header does not match {header_digest}",
                )
            header = facade.parse_header(header_raw)
            if not header["components"]:
                raise PreparationRefusal(
                    "package_prepare_model_header_unsupported",
                    f"{digest}: expected tensor components",
                )
            if not any(
                slot in construction_slots and (selection.composed or selection).manifest == digest
                for slot, selection in chosen.items()
            ):
                continue
            try:
                constructions[digest] = Construction(
                    unnormalized=model_config.unnormalized(header["configs"]),
                    config_bytes=model_config.construction_config(header["configs"]),
                    header=header_raw,
                    assets=_asset_reader(
                        store,
                        digest,
                        (header_digest, header_length),
                        header_raw,
                        msgspec.convert(header["assets"], dict[str, _Asset]),
                    ),
                    tensor_dtypes=model_config.tensor_dtypes(header),
                    adapters=model_config.adapter_graph(header["configs"]),
                )
            except model_config.ModelConfigRefusal as exc:
                raise PreparationRefusal(
                    exc.code or "package_prepare_model_config_mismatch", f"{digest}: {exc}"
                ) from exc
    except PreparationRefusal:
        raise
    except (acquire.AcquisitionRefusal, fill.FillRefusal) as exc:
        raise PreparationRefusal(
            "package_prepare_model_refused", f"{getattr(exc, 'code', type(exc).__name__)}: {exc}"
        ) from exc
    except Exception as exc:
        raise PreparationRefusal(
            "package_prepare_model_refused", f"{type(exc).__name__}: {exc}"
        ) from exc
    return {
        slot: Selected(
            selection,
            manifests[selection.manifest][0],
            constructions[(selection.composed or selection).manifest]
            if slot in construction_slots
            else None,
            tuple(manifests[row.manifest][0] for row in selection.adapters),
            manifests[selection.composed.manifest][0] if selection.composed else 0,
        )
        for slot, selection in chosen.items()
    }


def _asset_reader(
    store: fill.Store,
    manifest: str,
    header_ref: tuple[str, int],
    header_raw: bytes,
    assets: Mapping[str, _Asset],
) -> Callable[[], dict[str, bytes]]:
    """Read header-declared assets through a lease on the header and asset segments only."""

    def read() -> dict[str, bytes]:
        names = sorted(assets)
        if not names:
            return {}
        segments = {header_ref}
        for name in names:
            segments.update(("sha256:" + row.sha256, row.length) for row in assets[name].segments)
        try:
            lease = store.acquire(manifest, sorted(segments))
            try:
                return {
                    name: bytes(lease.read_asset(header_raw, name, max_bytes=64 << 20))
                    for name in names
                }
            finally:
                lease.release()
        except Exception as exc:
            raise census_cache.CensusRefusal(
                "package_prepare_model_asset_unavailable",
                f"{manifest}: {type(exc).__name__}: {exc}",
            ) from exc

    return read


def _read_file(path: Path, maximum: int) -> bytes:
    try:
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode) or not 0 < info.st_size <= maximum:
            raise OSError("invalid file")
        return path.read_bytes()
    except OSError as exc:
        raise PreparationRefusal("package_prepare_file_unreadable", str(path)) from exc


def _stage(root: Path, raw: bytes) -> tuple[str, int]:
    """Stage one control document into the worker's content-addressed cache.

    THIS FUNCTION OWNS ITS ROOT, and the mode it leaves behind is a uid decision. Its
    former sibling `acquire._stage` did both -- `mkdir(parents=True)` and `chmod
    0444` -- which is why the serving half survived th-128 deleting
    `/var/lib/cozy/tmp/transfers` from the worker image and this half did not: run 188
    refused a rented pod with `[Errno 2] No such file or directory:
    '/var/lib/cozy/prepare-cache/.prepare-<uuid>'` because nothing had created the
    directory. The mode matters for the same reason the directory does: the document's
    PATH is recorded into job plans and handed to the EXECUTOR, which runs as its own uid
    and does `Path(interface_path).read_bytes()` -- so a root-owned 0400 file is a second
    refusal waiting one layer further in."""

    # THE MODE IS SET, NOT REQUESTED. `mkdir(mode=)` and `os.open(mode=)` are both masked
    # by the process umask, so the modes this function asks for are the modes it gets only
    # under umask 0022. Measured against this exact code shape: 0022 -> 755/444,
    # 0027 -> 750/440, 0077 -> 700/400. A root-owned 0400 document is precisely what this
    # module's own negative control proves the executor cannot read, and the executor reads
    # this path as its own uid.
    #
    # It is also PERMANENT on the pod that hits it: `exist_ok=True` never re-applies a
    # directory mode, and the write below short-circuits on `target.exists()`, so no retry
    # can repair either. A fresh pod with a different umask works and the same pod never
    # does — the shape that makes a fault look deterministic and hides its cause.
    #
    # The former `acquire._stage`, the sibling this docstring says it matches, chmod'd
    # explicitly. This half only asked.
    root.mkdir(parents=True, exist_ok=True, mode=0o755)
    root.chmod(0o755)
    digest = "sha256:" + hashlib.sha256(raw).hexdigest()
    target = root / digest.removeprefix("sha256:")
    if target.exists():
        # A document left by a prior preparation under a stricter umask is unreadable to
        # the executor that will be handed its path, and re-reading it here proves nothing
        # about that. Re-assert the mode on the path we are about to publish.
        _readable(target, 0o444)
        with contextlib.suppress(PreparationRefusal):
            if _read_file(target, max(len(raw), 1)) == raw:
                return digest, len(raw)
    # Absent, or a damaged cache entry under this digest: publish the known bytes atomically.
    temporary = root / f".prepare-{uuid.uuid4().hex}"
    descriptor_fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        with os.fdopen(descriptor_fd, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        # Before the rename, so the document is never visible under a masked mode.
        _readable(temporary, 0o444)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return digest, len(raw)


def _readable(path: Path, mode: int) -> None:
    """Set an exact mode, tolerating a path this process does not own.

    The mode is the point: every caller here publishes a path that a DIFFERENT uid will
    open. A chmod that fails because another uid already owns the file is not a reason to
    refuse a preparation — the file may well already carry the right mode — so the failure
    is left to the read that actually needs it, which reports what it could not open.
    """
    with contextlib.suppress(OSError):
        path.chmod(mode)
