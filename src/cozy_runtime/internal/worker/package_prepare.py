"""Prepare one downloaded package set without giving package code worker authority."""

from __future__ import annotations

import contextlib
import copy
import hashlib
import importlib.metadata
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
    tensorfs_root: Path,
    install_root: Path,
    python: Path,
    describe: Callable[[package_environment.InstalledEnvironment, str], bytes],
    verified: Callable[[str, int, Path], None],
    job_plan_root: Path | None = None,
    base: base_observation.Observation | None = None,
    preinstalled_python: Path | None = None,
) -> pb.PreparePackageSetResult:
    """Parse the desired download set plus the request's release facts (wire 30) and prepare.

    The request carries everything the interface used to be opened for — `application` and
    the interface-declared model-slot paths — plus the placed image's pinned inventory and
    the release's exact locked-requirements export. No file rides the wire: wheels come from
    the export's indexes under its own hashes, the interface is derived from the installed
    environment, and model manifests resolve from the local TensorFS Store.
    """

    if (
        Path(request.install_root) != install_root
        or not install_root.is_absolute()
        or not tensorfs_root.is_absolute()
        or (job_plan_root is not None and not job_plan_root.is_absolute())
    ):
        raise PreparationRefusal("package_prepare_install_root_invalid", request.install_root)
    try:
        download_set = documents.read(request.download_delegation, pb.DownloadDelegation)
    except documents.DocumentError as exc:
        raise PreparationRefusal("package_prepare_download_set_invalid", exc.detail) from exc
    packages = download_set.get("packages")
    models = download_set.get("models")
    if not isinstance(packages, list) or len(packages) != 1 or not isinstance(models, list):
        raise PreparationRefusal(
            "package_prepare_selection_invalid",
            "exactly one package and a model array are required",
        )
    selected = packages[0]
    if not isinstance(selected, dict):
        raise PreparationRefusal("package_prepare_download_set_invalid", "package row")
    package_name = str(selected.get("package", ""))
    release = str(selected.get("release", ""))
    application, slot_paths, inventory, locked = _request_facts(
        request, models, preinstalled=preinstalled_python is not None
    )
    installed = None
    if preinstalled_python is not None:
        # This path is selected only by the local worker's launch configuration.
        # RPC callers cannot choose an interpreter or claim an image inventory.
        installed = package_environment.open_existing_environment(
            preinstalled_python, canonical.digest({"locked_requirements": locked.digest})
        )
    return _prepare_published(
        package_name=package_name,
        release=release,
        application=application,
        model_slot_paths=slot_paths,
        inventory=inventory,
        locked=locked,
        models=models,
        placement_seed=placement_identity(
            package_name, release, str(selected.get("release_digest", "")), models
        ),
        artifact_cache=artifact_cache,
        tensorfs_root=tensorfs_root,
        install_root=install_root,
        python=python,
        describe=describe,
        verified=verified,
        job_plan_root=job_plan_root,
        base=base,
        installed_environment=installed,
    )


def prepare_published_local(
    *,
    package_name: str,
    release: str,
    locked: package_environment.LockedRequirements,
    models: Sequence[Any],
    placement_seed: bytes,
    artifact_cache: Path,
    tensorfs_root: Path,
    install_root: Path,
    python: Path,
    describe: Callable[[package_environment.InstalledEnvironment, str], bytes],
    installed_environment: package_environment.InstalledEnvironment,
    job_plan_root: Path | None = None,
) -> pb.PreparePackageSetResult:
    """Creator's local published preparation over its already-materialized locked venv.

    The two facts are adopted from the derived interface — Creator compares that derivation
    against the committed release itself — and no image inventory exists locally.
    """

    return _prepare_published(
        package_name=package_name,
        release=release,
        application=None,
        model_slot_paths=None,
        inventory=None,
        locked=locked,
        models=models,
        placement_seed=placement_seed,
        artifact_cache=artifact_cache,
        tensorfs_root=tensorfs_root,
        install_root=install_root,
        python=python,
        describe=describe,
        verified=lambda _digest, _length, _path: None,
        job_plan_root=job_plan_root,
        base=None,
        installed_environment=installed_environment,
    )


def _request_facts(
    request: pb.PreparePackageSetRequest, models: Sequence[Any], *, preinstalled: bool = False
) -> tuple[
    str, tuple[str, ...], image_inventory.Inventory | None, package_environment.LockedRequirements
]:
    """Admit the wire-30 release facts: the two facts, the inventory, and the lock."""

    application = request.application
    if not application or _printable(application) != application or len(application) > 256:
        raise PreparationRefusal("package_prepare_application_invalid", application[:64])
    slot_paths = tuple(request.model_slot_paths)
    if (
        len(slot_paths) > 256
        or list(slot_paths) != sorted(set(slot_paths))
        or any(not path or _printable(path) != path or len(path) > 512 for path in slot_paths)
    ):
        raise PreparationRefusal("package_prepare_model_slot_paths_invalid", str(len(slot_paths)))
    for row in models:
        slot = str(row.get("slot", "")) if isinstance(row, Mapping) else ""
        if slot not in slot_paths:
            raise PreparationRefusal("package_prepare_model_slot_unknown", slot)
    inventory = None
    if preinstalled:
        if request.HasField("image_inventory"):
            raise PreparationRefusal(
                "package_prepare_local_inventory_invalid",
                "local prepared environment has no managed image inventory",
            )
    else:
        if not request.HasField("image_inventory"):
            raise PreparationRefusal("package_prepare_image_inventory_missing", "image_inventory")
        try:
            inventory = image_inventory.from_wire(
                request.image_inventory.profile,
                request.image_inventory.python,
                request.image_inventory.distributions,
            )
        except image_inventory.ImageInventoryRefusal as exc:
            raise PreparationRefusal("package_prepare_image_inventory_invalid", exc.detail) from exc
    try:
        locked = package_environment.read_locked_requirements(bytes(request.locked_requirements))
    except package_environment.EnvironmentRefusal as exc:
        raise PreparationRefusal("package_prepare_locked_requirements_invalid", exc.detail) from exc
    return application, slot_paths, inventory, locked


def _filtered_rows(
    package_name: str,
    release: str,
    locked: package_environment.LockedRequirements,
    inventory: image_inventory.Inventory,
    platform_versions: Mapping[str, str],
) -> list[package_environment.LockedRow]:
    """The export minus immutable image and physically inherited framework providers.

    A development image can inventory only its frozen heavy base. Its actual inherited
    Runtime/TensorFS pair still owns those names; a package lock cannot shadow them.
    The package's own pin remains required and always installs.
    """

    expected = _DISTRIBUTION.sub("-", package_name.rsplit("/", 1)[-1]).lower()
    projects = [row for row in locked.rows if row.name == expected]
    if len(projects) != 1 or projects[0].version != release:
        raise PreparationRefusal(
            "package_prepare_project_pin_missing",
            f"the export must pin {expected}=={release} exactly once",
        )
    provided = inventory.versions
    for name in ("cozy-runtime", "tensorfs"):
        if name in platform_versions:
            provided.setdefault(name, platform_versions[name])
    install_rows: list[package_environment.LockedRow] = []
    served: dict[str, str] = {}
    for row in locked.rows:
        if row.name != expected and row.name in provided:
            served[row.name] = row.version
            continue
        install_rows.append(row)
    if served:
        listed = ", ".join(
            f"{name} (base {provided[name]} over lock {version})"
            for name, version in sorted(served.items())
        )
        print(f"[package-prepare] worker base serves {listed}", flush=True)
    return install_rows


def _refuse_platform_shadow(
    installed: package_environment.InstalledEnvironment,
    package_name: str,
    platform_versions: Mapping[str, str],
) -> None:
    """An older cached venv must not silently keep app-local framework copies."""
    project = _DISTRIBUTION.sub("-", package_name.rsplit("/", 1)[-1]).lower()
    for distribution in importlib.metadata.distributions(path=[str(installed.site_packages)]):
        name = _DISTRIBUTION.sub("-", distribution.metadata.get("Name", "")).lower()
        if name != project and name in {"cozy-runtime", "tensorfs"} and name in platform_versions:
            raise PreparationRefusal(
                "package_prepare_platform_shadowed",
                f"cached application environment contains its own {name}; rebuild that environment",
            )


def _prepare_published(
    *,
    package_name: str,
    release: str,
    application: str | None,
    model_slot_paths: tuple[str, ...] | None,
    inventory: image_inventory.Inventory | None,
    locked: package_environment.LockedRequirements,
    models: Sequence[Any],
    placement_seed: bytes,
    artifact_cache: Path,
    tensorfs_root: Path | None,
    install_root: Path,
    python: Path,
    describe: Callable[[package_environment.InstalledEnvironment, str], bytes],
    verified: Callable[[str, int, Path], None],
    job_plan_root: Path | None,
    base: base_observation.Observation | None,
    installed_environment: package_environment.InstalledEnvironment | None = None,
) -> pb.PreparePackageSetResult:
    """Install from the exact export, derive the interface, and author the PlacementSet.

    On the wire path the request's two facts cross-check the derived interface and the
    image inventory filters the export. On the local path (`installed_environment`) the
    venv already exists from the same export and Creator compares the derived interface
    against the committed release itself, so the facts are adopted from the derivation.
    """

    environment = pb.Environment(
        locked_requirements=pb.Ref(digest=documents.raw(locked.digest), length=locked.length)
    )
    environment_body = documents.body(environment)
    environment_digest = documents.spell(
        documents.embedded_document_digest(environment_body, "cozy.worker.v1.Environment")
    )
    if installed_environment is not None:
        installed = installed_environment
    else:
        if inventory is None:
            raise PreparationRefusal("package_prepare_image_inventory_missing", "image_inventory")
        try:
            observed_base = base or package_environment.observe_base(python)
        except package_environment.EnvironmentRefusal as exc:
            raise PreparationRefusal(exc.code, exc.detail) from exc

        lie = image_inventory.disagreement(inventory, observed_base.distribution_versions)
        if lie is not None:
            raise PreparationRefusal("package_prepare_image_inventory_disagrees", lie)
        platform = (
            observed_base.distribution_versions
            if package_environment.is_base_interpreter(python)
            else {}
        )
        install_rows = _filtered_rows(package_name, release, locked, inventory, platform)
        try:
            installed = package_environment.materialize_locked(
                locked,
                install_rows,
                install_root,
                environment_digest=environment_digest,
                python=python,
                base=observed_base,
            )
        except package_environment.EnvironmentRefusal as exc:
            raise PreparationRefusal(exc.code, exc.detail) from exc
        _refuse_platform_shadow(installed, package_name, platform)

    expected_distribution = _DISTRIBUTION.sub("-", package_name.rsplit("/", 1)[-1]).lower()
    interface_raw = describe(installed, expected_distribution)
    try:
        interface_body = package_interface.read_bytes(interface_raw, "derived package interface")
    except package_interface.StalePackageInterface as exc:
        raise PreparationRefusal("package_prepare_interface_invalid", str(exc)) from exc
    derived_application = str(interface_body.get("application", ""))
    if application is not None and derived_application != application:
        raise PreparationRefusal(
            "package_prepare_application_disagrees",
            f"request {application!r} != derived {derived_application!r}",
        )
    application = derived_application
    declared = _slot_paths(interface_body, "entrypoints", "jobs")
    if model_slot_paths is not None and declared != set(model_slot_paths):
        raise PreparationRefusal(
            "package_prepare_model_slots_disagree",
            f"request-only={sorted(set(model_slot_paths) - declared)} "
            f"derived-only={sorted(declared - set(model_slot_paths))}",
        )
    interface_ref = _stage(artifact_cache, interface_raw)
    jobs = _stage_job_plans(
        interface_body,
        root=job_plan_root,
        build_id=environment_digest,
        interface_path=artifact_cache / interface_ref[0].removeprefix("sha256:"),
        installed=installed,
    )
    selected_models, entrypoints = _entrypoints(
        interface_body,
        package_name=package_name,
        selections=models,
        tensorfs_root=tensorfs_root,
        verified=verified,
        derive=_deriver(installed, application),
    )
    if not entrypoints and not jobs:
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
        package=pb.PackageSelection(package=package_name, release=release),
        environment_digest=documents.raw(environment_digest),
        package_interface=pb.Ref(digest=documents.raw(interface_ref[0]), length=interface_ref[1]),
        bindings_digest=bindings_digest,
        models=selected_models,
        entrypoints=entrypoints,
        environment=environment,
    )
    placement_bytes = canonical.write(
        {"format": "cozy.worker.v1.PlacementSet/1", "placements": [documents.body(placement)]}
    )
    return pb.PreparePackageSetResult(
        placement_set=pb.DesiredPlacementSet(
            placement_set_digest=documents.digest_of(placement_bytes),
            placement_set_canonical_bytes=placement_bytes,
        )
    )


def check_package_set_compatibility(
    request: pb.PreparePackageSetRequest,
    *,
    tensorfs_root: Path,
    install_root: Path,
    python: Path,
    base: base_observation.Observation | None = None,
) -> pb.CheckPackageSetCompatibilityResult:
    """Fit package code against headers while tensor bodies are still downloading.

    Wire 30: no file rides the request. The environment materializes from the export's own
    indexes and hashes; the selected manifests, their CozyTensors headers, and the
    header-declared assets are read from the local TensorFS Store the host already admitted
    them to — a partial object-set lease, never the full tensor closure. It constructs on
    CPU/meta, calls TensorFS fit with transient rows, and retains no compatibility artifact.
    """

    if Path(request.install_root) != install_root or not install_root.is_absolute():
        raise PreparationRefusal("package_prepare_install_root_invalid", request.install_root)
    try:
        download_set = documents.read(request.download_delegation, pb.DownloadDelegation)
    except documents.DocumentError as exc:
        raise PreparationRefusal("package_prepare_download_set_invalid", exc.detail) from exc
    packages = download_set.get("packages")
    models = download_set.get("models")
    if not isinstance(packages, list) or len(packages) != 1 or not isinstance(models, list):
        raise PreparationRefusal(
            "package_prepare_selection_invalid",
            "exactly one package and a model array are required",
        )
    selected_package = packages[0]
    if not isinstance(selected_package, dict):
        raise PreparationRefusal("package_prepare_download_set_invalid", "package row")
    package_name = str(selected_package.get("package", ""))
    release = str(selected_package.get("release", ""))
    application, slot_paths, inventory, locked = _request_facts(request, models)
    if inventory is None:
        raise PreparationRefusal("package_prepare_image_inventory_missing", "image_inventory")
    installed = _compatibility_environment(
        package_name=package_name,
        release=release,
        install_root=install_root,
        python=python,
        inventory=inventory,
        locked=locked,
        base=base,
    )
    candidates = _compatibility_models(models, tensorfs_root=tensorfs_root)
    # SUBSET, not equality. `candidates` is keyed by the slots THIS INVOCATION binds;
    # `slot_paths` describes every slot THE RELEASE declares. A release with more than
    # one callable always declares more than any single invocation binds -- anima
    # declares generate.models.model and quantize.models.source while an invocation
    # binds one of them -- so equality refused every such release. What the guard is
    # actually for is the other direction: a download set naming a slot the package does
    # not have.
    undeclared = sorted(set(candidates) - set(slot_paths))
    if undeclared:
        raise PreparationRefusal(
            "package_prepare_model_selection_mismatch",
            f"the download set binds {undeclared}, which the release does not declare; "
            f"declared={sorted(slot_paths)}",
        )
    requests = tuple(
        derive_child.SlotRequest(path, row["config"], row["assets"])
        for path, row in sorted(candidates.items())
    )
    if not requests:
        # NOTHING TO FIT. A set that selects no model has no header to check code against,
        # and the environment half of this check has already run above. Handing the derive
        # child an empty slot set refuses `derive_request_invalid: slots is a non-empty
        # array` -- which is how EVERY modelless preparation died on a pod, naming a
        # request shape rather than the absent selection, while `prepare_package_set`
        # would have accepted the same set (`_entrypoints` derives nothing when nothing is
        # selected). The check must not be stricter than the preparation it guards.
        return pb.CheckPackageSetCompatibilityResult()
    try:
        answers = derive_child.derive_in(
            installed.python,
            derive_child.DeriveRequest(requests, application=application),
        )
    except derive_child.DeriveRefusal as exc:
        raise PreparationRefusal(exc.code, exc.detail) from exc
    tensorfs = fill.tensorfs_module()
    for answer in answers:
        if isinstance(answer, derive_child.SourceSlot):
            # Discovery classified this exact path as a job's source view. Its
            # header/slot checks above still apply; serving construction does not.
            continue
        candidate = candidates[answer.slot]
        requirements = [
            tensorfs.TensorRequirement(
                component=component,
                key=key,
                shape=list(shape),
                logical_dtype=fill.tensorfs_requirement_dtype(dtype),
            )
            for component, key, dtype, shape in answer.rows
        ]
        try:
            verdict = dict(
                tensorfs.fit(
                    requirements,
                    candidate["header"],
                    custody="canonical",
                    # The per-slot encoded_leaves declaration lives in the interface, which
                    # no longer rides the cold path. Lenient here is the one-sided error:
                    # a header the interface would refuse fails at real preparation, while
                    # a strict default would falsely cancel valid downloads.
                    encoded_leaves=True,
                )
            )
        except tensorfs.errors.Refusal as exc:
            raise PreparationRefusal(
                "package_prepare_model_header_invalid", f"{answer.slot}: {exc.code}: {exc}"
            ) from exc
        if not verdict.get("ok"):
            detail = _printable(
                f"{answer.slot} does not fit {candidate['model']}@{candidate['release']}"
                f"/{candidate['lane']}: {verdict.get('code')} - {verdict.get('detail')}"
            )
            raise PreparationRefusal("checkpoint_incompatible", detail)
    return pb.CheckPackageSetCompatibilityResult()


def _compatibility_environment(
    *,
    package_name: str,
    release: str,
    install_root: Path,
    python: Path,
    inventory: image_inventory.Inventory,
    locked: package_environment.LockedRequirements,
    base: base_observation.Observation | None,
) -> package_environment.InstalledEnvironment:
    try:
        observed = base or package_environment.observe_base(python)
    except package_environment.EnvironmentRefusal as exc:
        raise PreparationRefusal(exc.code, exc.detail) from exc
    lie = image_inventory.disagreement(inventory, observed.distribution_versions)
    if lie is not None:
        raise PreparationRefusal("package_prepare_image_inventory_disagrees", lie)
    platform = (
        observed.distribution_versions if package_environment.is_base_interpreter(python) else {}
    )
    install_rows = _filtered_rows(package_name, release, locked, inventory, platform)
    environment = pb.Environment(
        locked_requirements=pb.Ref(digest=documents.raw(locked.digest), length=locked.length)
    )
    environment_digest = documents.spell(
        documents.embedded_document_digest(
            documents.body(environment), "cozy.worker.v1.Environment"
        )
    )
    try:
        installed = package_environment.materialize_locked(
            locked,
            install_rows,
            install_root,
            environment_digest=environment_digest,
            python=python,
            base=observed,
        )
    except package_environment.EnvironmentRefusal as exc:
        raise PreparationRefusal(exc.code, exc.detail) from exc
    _refuse_platform_shadow(installed, package_name, platform)
    return installed


def _compatibility_models(
    models: Sequence[Any],
    *,
    tensorfs_root: Path,
) -> dict[str, dict[str, Any]]:
    """Read each selected manifest, its CozyTensors header, and the header-declared assets
    from the local TensorFS Store through one explicit object-set lease — the manifests and
    small objects the host admits first, never the tensor closure still in flight. An absent
    manifest or header is a resumable refusal, not a verdict."""

    tensorfs = fill.tensorfs_module()
    selections: dict[str, dict[str, Any]] = {}
    wanted_manifests: set[str] = set()
    for value in models:
        if not isinstance(value, dict):
            raise PreparationRefusal("package_prepare_model_selection_mismatch", "model row")
        slot = str(value.get("slot", ""))
        manifest = str(value.get("manifest", ""))
        model = str(value.get("model", ""))
        release = str(value.get("release", ""))
        lane = str(value.get("lane", ""))
        if (
            not slot
            or slot in selections
            or not manifest
            or bool(release) != bool(lane)
            or len(model.split("/")) != 2
            or not all(model.split("/"))
        ):
            raise PreparationRefusal("package_prepare_model_selection_mismatch", slot or model)
        wanted_manifests.add(manifest)
        selections[slot] = {
            "selection": value,
            "manifest": manifest,
            "model": model,
            "release": release,
            "lane": lane,
        }
    if not selections:
        # NOTHING SELECTED: no manifest to read, so no Store to open. Opening one anyway
        # made a modelless set depend on a Store it never touches. `_selected_models`
        # returns here for the same reason, and the preparation this check guards
        # therefore accepted sets the check refused.
        return selections
    by_manifest: dict[str, dict[str, Any]] = {}
    try:
        store = fill.open_store(tensorfs, tensorfs_root)
    except fill.FillRefusal as exc:
        raise PreparationRefusal("package_prepare_model_store_absent", str(exc)) from exc
    for digest in sorted(wanted_manifests):
        try:
            held = store.manifest(digest)
            manifest_raw = bytes(held["manifest"])
            header_digest, header_length = acquire.cozytensors_ref(tensorfs, manifest_raw)
            header_raw = bytes(held["header"])
            if (
                len(header_raw) != header_length
                or "sha256:" + hashlib.sha256(header_raw).hexdigest() != header_digest
            ):
                raise ValueError(f"held header does not match {header_digest}")
            header = tensorfs.parse_header(header_raw)
            config = model_config.construction_config(header.get("configs"))
        except PreparationRefusal:
            raise
        except Exception as exc:
            raise PreparationRefusal(
                "package_prepare_model_header_unavailable",
                f"{digest}: {type(exc).__name__}: {exc}",
            ) from exc
        assets: dict[str, bytes] = {}
        asset_names = sorted(str(name) for name in header.get("assets", {}))
        if asset_names:
            try:
                segments: list[tuple[str, int]] = [(header_digest, header_length)]
                for name in asset_names:
                    asset = header["assets"][name]
                    for segment in asset.get("segments", []):
                        segments.append(
                            (
                                "sha256:" + str(segment.get("sha256", "")),
                                int(segment.get("length", 0)),
                            )
                        )
                lease = store.acquire(digest, sorted(set(segments)))
                try:
                    for name in asset_names:
                        assets[name] = bytes(lease.read_asset(header_raw, name, max_bytes=64 << 20))
                finally:
                    lease.release()
            except Exception as exc:
                raise PreparationRefusal(
                    "package_prepare_model_asset_unavailable",
                    f"{digest}: {type(exc).__name__}: {exc}",
                ) from exc
        by_manifest[digest] = {"header": header_raw, "config": config, "assets": assets}
    for selected in selections.values():
        selected.update(by_manifest[selected["manifest"]])
    return selections


def _printable(value: str) -> str:
    return "".join(character if " " <= character <= "~" else "?" for character in value)[:1024]


def placement_identity(
    package_name: str, release: str, release_digest: str, models: Sequence[canonical.Json]
) -> bytes:
    """The placement's identity is WHAT is prepared, never a credential. The document
    this is derived from used to carry a per-signing expiry and worker pins (other
    clocks — cr-004), so seeding identity with its raw bytes made every re-signed
    converge a brand-new placement and defeated ledger reuse (xs-014 finding 1). Those
    fields are deleted with the whole delegation plane (owner ruling 2026-09-03), so a
    download set is now a pure function of its content; identity is still derived from
    the named facts rather than the bytes, because that is what it means. Model rows are
    re-sorted here so identity cannot depend on the author's ordering."""

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


def prepare_local_package(
    request: pb.PrepareLocalPackageRequest,
    *,
    artifact_cache: Path,
    install_root: Path,
    python: Path,
    describe: Callable[[package_environment.InstalledEnvironment, str], bytes],
    verified: Callable[[str, int, Path], None],
    job_plan_root: Path | None = None,
    base: base_observation.Observation | None = None,
    dependency_python: Path | None = None,
    materialized: Callable[[package_environment.InstalledEnvironment], None] | None = None,
) -> pb.PreparePackageSetResult:
    """Prepare one exact local wheel revision without inventing Hub provenance."""

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
        or len(package.source_digest) != 32
        or len(package.local_revision_digest) != 32
        or package.HasField("project_wheel")
    ):
        raise PreparationRefusal("local_package_identity_invalid", request.operation_id)
    return prepare_package(
        package_name=package.package,
        release=package.release,
        wheels=request.wheels,
        # proto-031: editable wheels stage beside the install namespace that owns them.
        wheel_root=install_root / ".stage" / request.operation_id / "wheels",
        placement_seed=(request.operation_id.encode("ascii") + package.local_revision_digest),
        artifact_cache=artifact_cache,
        install_root=install_root,
        python=python,
        describe=describe,
        verified=verified,
        job_plan_root=job_plan_root,
        base=base,
        development=package,
        dependency_python=dependency_python,
        materialized=materialized,
    )


def prepare_private_placement(
    request: pb.PreparePrivatePlacementRequest,
    *,
    prepared: Mapping[str, Any],
    installed: package_environment.InstalledEnvironment | None,
    artifact_cache: Path,
    tensorfs_root: Path | None,
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
        download_set = documents.read(request.download_delegation, pb.DownloadDelegation)
    except documents.DocumentError as exc:
        raise PreparationRefusal("private_placement_download_set_invalid", exc.detail) from exc
    packages = download_set.get("packages", [])
    models = download_set.get("models", [])
    if packages != [] or not isinstance(models, list) or len(models) > 32:
        raise PreparationRefusal(
            "private_placement_download_set_invalid",
            "a private model download set carries no package refs and at most 32 models",
        )
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
        tensorfs_root=tensorfs_root,
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
    wheels: Sequence[pb.LocalPackageWheel],
    wheel_root: Path,
    placement_seed: bytes,
    artifact_cache: Path,
    install_root: Path,
    python: Path,
    describe: Callable[[package_environment.InstalledEnvironment, str], bytes],
    verified: Callable[[str, int, Path], None],
    base: base_observation.Observation | None = None,
    dependency_python: Path | None = None,
    job_plan_root: Path | None = None,
    development: pb.DevelopmentPackage | None = None,
    materialized: Callable[[package_environment.InstalledEnvironment], None] | None = None,
) -> pb.PreparePackageSetResult:
    """Inspect, install, import, and describe one exact local package revision.

    This is the ONE preparation that still receives files: an editable revision's wheels
    exist on no index, so their exact bytes are the install input (xs-017). Every row
    sits directly in `wheel_root` — `<install_root>/.stage/<operation>/wheels` per
    proto-031 — beside the install namespace that owns it. The project wheel is the row
    that MEASURES as the package's own distribution and release — no kind row travels.

    `materialized` receives the environment this preparation installed. A local revision's
    wheels are the pod host's carriers, deleted once the PlacementSet is journaled, so the
    placement that follows cannot re-acquire them: the session retains the environment and
    serves the placement from it.

    A native dependency wheel installs like any other wheel (cr-093): whether its extension
    loads is answered by the executor that imports it, not by a prepare-time inspection.
    """

    if development is None:
        raise PreparationRefusal("package_prepare_development_absent", package_name)
    expected_distribution = _DISTRIBUTION.sub("-", package_name.rsplit("/", 1)[-1]).lower()
    try:
        observed_base = base or package_environment.observe_base(python)
    except package_environment.EnvironmentRefusal as exc:
        raise PreparationRefusal(exc.code, exc.detail) from exc
    rows = list(wheels)
    if not 1 <= len(rows) <= MAX_FILES:
        raise PreparationRefusal("package_prepare_file_count", str(len(rows)))
    if not wheel_root.is_absolute():
        raise PreparationRefusal("package_prepare_file_path_invalid", str(wheel_root))
    facts: list[tuple[package_environment.Wheel, Path, package_environment.VerifiedFile]] = []
    prior = b""
    for row in rows:
        if len(row.digest) != 32 or row.digest <= prior or not 0 < row.length <= MAX_FILE_BYTES:
            raise PreparationRefusal("package_prepare_file_invalid", row.filename)
        prior = bytes(row.digest)
        path = Path(row.path)
        if (
            not path.is_absolute()
            or path.name != row.filename
            or path.parent != wheel_root
            or path.resolve(strict=True) != path
        ):
            raise PreparationRefusal("package_prepare_file_path_invalid", row.filename)
        digest = documents.spell(bytes(row.digest))
        named = package_environment.wheel_name_version(row.filename)
        if named is None:
            raise PreparationRefusal("package_prepare_file_invalid", row.filename)
        owner = "project" if named == (expected_distribution, release) else "custom"
        try:
            wheel, receipt = package_environment.measure_wheel(
                path, owner, observed_base, allow_base_owned=owner == "custom"
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
        try:
            base_owned = owner == "custom" and package_environment.base_owned_wheel(
                wheel, observed_base
            )
        except package_environment.EnvironmentRefusal as exc:
            raise PreparationRefusal(exc.code, exc.detail) from exc
        if base_owned:
            # The supplied wheel proves which protected distribution the frozen local
            # closure selected; this host's base supplies the executable bytes.
            verified(digest, row.length, path)
            continue
        facts.append((wheel, path, receipt))
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
    environment = pb.Environment(local_wheels=[_wheel_fact(wheel) for wheel in dependency_facts])
    environment_body = documents.body(environment)
    environment_digest = documents.spell(
        documents.embedded_document_digest(environment_body, "cozy.worker.v1.Environment")
    )
    wheel_paths = {wheel.digest: path for wheel, path, _receipt in facts}
    receipts = {wheel.digest: receipt for wheel, _path, receipt in facts}
    try:
        source = placement_materialization.package_environment_input(
            package_name=package_name,
            release=release,
            project_wheel=documents.body(_wheel_fact(project)),
            environment=environment_body,
            environment_digest=environment_digest,
            wheel_paths=wheel_paths,
            verified_wheels=receipts,
        )
        installed = package_environment.materialize(
            source,
            install_root,
            python=python,
            base=observed_base,
            dependency_python=dependency_python,
        )
    except (package_environment.EnvironmentRefusal, LaunchRefusal) as exc:
        code = getattr(exc, "code", "package_environment_install_failed")
        detail = getattr(exc, "detail", type(exc).__name__)
        raise PreparationRefusal(str(code), str(detail)) from exc
    for wheel, path, _receipt in facts:
        verified(wheel.digest, wheel.length, path)
    if materialized is not None:
        materialized(installed)

    interface_raw = describe(installed, project.distribution)
    try:
        interface_body = package_interface.read_bytes(interface_raw, "prepared package interface")
    except package_interface.StalePackageInterface as exc:
        raise PreparationRefusal("package_prepare_interface_invalid", str(exc)) from exc
    interface_ref = _stage(artifact_cache, interface_raw)
    _verify_local_revision(development, interface_ref, rows)
    _stage_job_plans(
        interface_body,
        root=job_plan_root,
        build_id=documents.spell(development.local_revision_digest),
        interface_path=artifact_cache / interface_ref[0].removeprefix("sha256:"),
        installed=installed,
    )
    # Local package transport freezes code, PackageInterface, and Environment only. A
    # later ordinary PlacementSet supplies exact modeled serving bindings; JobDirective
    # + InvocationSpec supplies a job's exact model inputs. Weightless serving bindings
    # need no later model authority and are already exact consequences of this retained
    # package interface.
    selected_models, entrypoints = _entrypoints(
        interface_body,
        package_name=package_name,
        selections=(),
        tensorfs_root=None,
        verified=verified,
        derive=None,
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
    wheels: Sequence[pb.LocalPackageWheel],
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
                length=row.length,
            )
            for row in wheels
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

    def derive(slot: str, config_bytes: bytes, assets: Mapping[str, bytes]) -> tuple[str, ...]:
        if not application:
            raise PreparationRefusal("package_prepare_interface_invalid", "application")
        try:
            request = derive_child.DeriveRequest(
                (derive_child.SlotRequest(slot, config_bytes, assets),),
                application=application,
            )
            (held,) = derive_child.derive_in(installed.python, request)
            if isinstance(held, derive_child.SourceSlot):
                raise derive_child.DeriveRefusal(
                    "derive_slot_not_serving", f"{slot} is a job source, not a serving model"
                )
            return held.components
        except derive_child.DeriveRefusal as exc:
            raise PreparationRefusal(
                "package_prepare_derive_refused", f"{slot}: {exc.code}: {exc.detail}"
            ) from exc

    return derive


def _slot_paths(body: Mapping[str, Any], *collections: str) -> set[str]:
    """Every model-slot path the interface declares in the named callable collections.

    cozy.package.interface/1 declares `models[].path` in BOTH `entrypoints` and `jobs`.
    Reading only the first is how a release whose callables are all jobs came to describe
    no model slot at all, and how the one slot a job bound became undeclarable.
    """

    paths: set[str] = set()
    for collection in collections:
        rows = body.get(collection, [])
        if not isinstance(rows, list):
            raise PreparationRefusal("package_prepare_interface_invalid", collection)
        for row in rows:
            if not isinstance(row, dict):
                continue
            models = row.get("models", [])
            if not isinstance(models, list):
                raise PreparationRefusal("package_prepare_interface_invalid", "model slots")
            paths.update(str(model.get("path", "")) for model in models if isinstance(model, dict))
    return paths


def _entrypoints(
    body: Mapping[str, Any],
    *,
    package_name: str,
    selections: Sequence[Any],
    tensorfs_root: Path | None,
    verified: Callable[[str, int, Path], None],
    derive: Deriver | None,
) -> tuple[list[pb.Model], list[pb.Entrypoint]]:
    rows = body.get("entrypoints")
    if not isinstance(rows, list):
        raise PreparationRefusal("package_prepare_interface_invalid", "entrypoints")
    selected = _selected_models(package_name, selections, tensorfs_root, verified)
    if selected and derive is None:
        raise PreparationRefusal(
            "package_prepare_derive_refused", "a modeled selection needs the package environment"
        )
    model_ids = {slot: f"model-{index:04d}" for index, slot in enumerate(sorted(selected))}
    models = [
        pb.Model(
            id=model_ids[slot],
            repo=str(value["selection"]["model"]),
            version=str(value["selection"].get("release", "")),
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
    unbound = sorted(set(selected) - described_slots)
    if unbound:
        # A JOB's model slot is legitimately unbound here. A job has no placement and no
        # residency: its Model parameter is an attempt-held, derive-only view of one
        # TensorFS Manifest, so the selection exists to admit those bytes to this pod,
        # not to construct an Entrypoint binding. Only a selection NO callable declares
        # -- or one that would leave an entrypoint half-bound -- is a mismatch.
        stranded = [path for path in unbound if path not in _slot_paths(body, "jobs")]
        if stranded:
            raise PreparationRefusal(
                "package_prepare_model_selection_mismatch",
                f"nothing binds {stranded}: no job declares them, and no entrypoint that "
                "declares them had its whole slot set selected",
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
    from cozy_runtime.internal import installed_interfaces

    call_interfaces = installed_interfaces.read(installed)
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
        if call_interfaces:
            record["call_interfaces"] = call_interfaces
        from cozy_runtime.internal import job_plan

        job_plan.write(root, record)
        staged.add(descriptor_id)
    return staged


def _resolved_manifest_length(
    resolved: Mapping[str, Any], digest: str, declared: int, where: str
) -> int:
    """The length of the manifest a selection resolves to, cross-checked against it.

    THE SELECTION DECLARES A DIGEST; THE STORE DECLARES THE LENGTH. `cli/package_prepare`
    requires `manifest_length` on every selection row it accepts, but the pod's rows come
    from a DownloadDelegation whose model entries are lane/manifest/model/package/
    release/slot -- there is no length in them. Reading one and comparing it to the store's
    real length compared 0 against a real number, which no store can satisfy: every modeled
    preparation on a pod refused `package_prepare_model_selection_mismatch` while naming the
    digest that had in fact matched. `local.py` already gets this right -- the length comes
    from the resolved row, and a caller-supplied one is cross-checked only when it exists --
    and this is that rule, given a name so the two halves cannot drift apart again.
    """

    resolved_digest = str(resolved["manifest_digest"])
    length = int(resolved["manifest_length"])
    if resolved_digest != digest or (declared and declared != length):
        # Name BOTH sides. The old message quoted only the digest the caller expected, so a
        # length disagreement and a digest disagreement read identically -- which sent a
        # reader hunting for a lane mix-up that was never there.
        raise PreparationRefusal(
            "package_prepare_model_selection_mismatch",
            f"{where} resolves to {resolved_digest}/{length}, "
            f"not the selected {digest}" + (f"/{declared}" if declared else ""),
        )
    return length


def _selected_models(
    package_name: str,
    selections: Sequence[Any],
    tensorfs_root: Path | None,
    verified: Callable[[str, int, Path], None],
) -> dict[str, dict[str, Any]]:
    """Resolve each selected model from the local TensorFS Store (wire 30).

    The privileged host and TensorFS admitted every model byte before preparation ran; this
    plane cross-checks release rows when selected, or reads an exact admitted checkpoint
    without a release alias. It proves the runtime closure complete with the store's own
    verified walk and reads config plus assets through it. Hub/Host owns repository
    membership; no model file rides this request and nothing here writes to the store.
    """

    selected: dict[str, dict[str, Any]] = {}
    for row in selections:
        if not isinstance(row, dict) or row.get("package") != package_name:
            raise PreparationRefusal("package_prepare_model_selection_mismatch", "package")
        slot = str(row.get("slot", ""))
        if not slot or slot in selected:
            raise PreparationRefusal("package_prepare_model_selection_mismatch", slot)
        selected[slot] = {"selection": row}
    if not selected:
        return selected

    if tensorfs_root is None:
        raise PreparationRefusal(
            "package_prepare_model_store_absent", "modeled package requires a TensorFS store"
        )

    try:
        tensorfs = fill.tensorfs_module()
        store = fill.open_store(tensorfs, tensorfs_root)
        manifests: dict[str, tuple[int, bytes]] = {}
        for value in selected.values():
            selection = value["selection"]
            digest = str(selection["manifest"])
            # A DOWNLOAD SET NAMES A DIGEST AND NO LENGTH. `cli/package_prepare` requires
            # `manifest_length` in every selection row it accepts, but the pod's rows come
            # from a DownloadDelegation whose model entries are
            # lane/manifest/model/package/release/slot -- there is no length to read. So
            # this defaulted to 0 and compared 0 against the store's real length, which no
            # store can ever satisfy: every modeled preparation on a pod refused
            # `package_prepare_model_selection_mismatch` naming the digest that had in fact
            # matched. `local.py` already resolves this correctly -- the length comes from
            # the resolved row and a caller-supplied one is only cross-checked when it
            # exists -- and this is that, for the same reason.
            declared = int(selection.get("manifest_length", 0))
            model = str(selection.get("model", ""))
            release = str(selection.get("release", ""))
            lane = str(selection.get("lane", ""))
            parts = model.split("/")
            if len(parts) != 2 or not all(parts) or bool(release) != bool(lane):
                raise PreparationRefusal("package_prepare_model_selection_mismatch", model)
            length = declared
            if release:
                length = _resolved_manifest_length(
                    store.resolve_release(parts[0], parts[1], release, lane),
                    digest,
                    declared,
                    f"{model}@{release}/{lane}",
                )
            held = store.manifest(digest)
            raw = bytes(held["manifest"])
            if not release and not declared:
                length = len(raw)
            if (
                length <= 0
                or len(raw) != length
                or "sha256:" + hashlib.sha256(raw).hexdigest() != digest
            ):
                raise PreparationRefusal("package_prepare_model_manifest_mismatch", digest)
            manifests[digest] = (length, raw)
        for digest, (length, _raw) in sorted(manifests.items()):
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
                assets = fill.Checkpoint(tensorfs_root, digest).model_assets()
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
    """Stage one control document into the worker's content-addressed cache.

    THIS FUNCTION OWNS ITS ROOT, and the mode it leaves behind is a uid decision. Its
    sibling `acquire._stage` has always done both -- `mkdir(parents=True)` and `chmod
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
    # `acquire._stage`, the sibling this docstring says it matches, has always chmod'd
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
        if _read_file(target, max(len(raw), 1)) != raw:
            raise PreparationRefusal("package_prepare_cache_conflict", digest)
        return digest, len(raw)
    temporary = root / f".prepare-{uuid.uuid4().hex}"
    descriptor_fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        with os.fdopen(descriptor_fd, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        # Before the link, so the document is never visible under a masked mode.
        _readable(temporary, 0o444)
        os.link(temporary, target)
    except FileExistsError:
        _readable(target, 0o444)
        if _read_file(target, max(len(raw), 1)) != raw:
            raise PreparationRefusal("package_prepare_cache_conflict", digest) from None
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
