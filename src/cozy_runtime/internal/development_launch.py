"""The development launch: a source checkout staged as one exact PlacementSet selection.

It reads the checkout's source, never imports it; the executor imports the package in its
own process. Model slots resolve against the local TensorFS Store and package.toml.
"""

from __future__ import annotations

import hashlib
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import msgspec

from cozy_runtime.internal import bindings as binding_table
from cozy_runtime.internal import (
    canonical,
    derive_child,
    fill,
    model_config,
    package_interface,
)
from cozy_runtime.internal.canonical import Json
from cozy_runtime.internal.worker.plan import (
    DeclaredBinding,
    ModelBinding,
)
from cozy_runtime.protocol import documents
from cozy_runtime.protocol import worker_pb2 as pb

if TYPE_CHECKING:
    pass


#: The one local deployment's owner-minted routing key (#446). Routing, never identity:
#: the InvocationSpec digest deliberately excludes it.
LOCAL_PLACEMENT = "plc-local"


@dataclass(frozen=True, slots=True)
class StagedDevelopmentPlacement:
    """One exact PlacementSet selection for source-checkout development."""

    bindings: Mapping[str, str]
    placement: pb.Placement
    tensorfs_root: Path
    artifact_cache: Path
    declared_bindings: Mapping[str, DeclaredBinding] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class DevelopmentPlacementDocuments:
    """Exact bytes and Placement behind local source-checkout execution."""

    package_interface: bytes
    placement: pb.Placement
    bindings: Mapping[str, DeclaredBinding] = field(default_factory=dict)

    def subjects(self) -> tuple[dict[str, str], ...]:
        return tuple(
            {
                "entrypoint": entrypoint.name,
                "digest": documents.spell(entrypoint.entrypoint_binding_digest),
            }
            for entrypoint in self.placement.entrypoints
        )


@dataclass(frozen=True, slots=True)
class _DevelopmentModel:
    slot: _InterfaceSlot
    repo: str
    name: str
    version: str
    lane: str
    manifest_digest: str
    manifest_length: int
    components: tuple[str, ...]
    store: Path
    variant: str
    custody: str
    model_weight_bytes: int


def stage_development_placement(
    cozy_home: Path,
    interface: Mapping[str, Json],
    *,
    package: str,
    release: str,
    installation_id: str,
    artifact_cache: Path,
    tensorfs_root: Path,
    project: Path | None = None,
    binding_overrides: Mapping[str, str] | None = None,
) -> StagedDevelopmentPlacement:
    """Stage one exact PlacementSet/1 plus its launch-local interface/config bytes.

    The package interface and config bytes the executor reads land in `artifact_cache` — the
    launcher's install-owned, content-addressed document store (cozy-creator's
    `installs/<id>/artifact-cache/`, a run's workspace), never under `cozy_home`: the home
    root holds what has a reason to be there, and a launch's derived documents are the
    launcher's to place and reclaim. `tensorfs_root` is named, never created here.
    """

    home = cozy_home.resolve()
    cache = artifact_cache.resolve()
    selected_store = tensorfs_root.resolve()
    documents_ = development_placement_documents(
        interface,
        package=package,
        release=release,
        installation_id=installation_id,
        project=project,
        cozy_home=home,
        tensorfs_root=selected_store,
        artifact_cache=cache,
        binding_overrides=binding_overrides,
    )
    bindings = {
        entrypoint.name: documents.spell(entrypoint.entrypoint_binding_digest)
        for entrypoint in documents_.placement.entrypoints
    }
    return StagedDevelopmentPlacement(
        bindings,
        documents_.placement,
        selected_store,
        cache,
        documents_.bindings,
    )


def development_placement_documents(
    document: Mapping[str, Json],
    *,
    package: str,
    release: str,
    installation_id: str,
    project: Path | None = None,
    cozy_home: Path | None = None,
    tensorfs_root: Path | None = None,
    artifact_cache: Path | None = None,
    binding_overrides: Mapping[str, str] | None = None,
) -> DevelopmentPlacementDocuments:
    """Construct exact source and local-TensorFS selection facts through existing owners."""

    interface_bytes = package_interface.canonical_bytes(document)
    interface = package_interface.parse(interface_bytes, "development package interface")
    if not interface.entrypoints and not interface.jobs:
        raise RuntimeError("a local package requires at least one visible entrypoint or job")
    if package.count("/") != 1 or not all(package.split("/")) or not release:
        raise RuntimeError("development package identity or release is invalid")
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,191}", installation_id) is None:
        raise RuntimeError("development installation ID is invalid")

    interface_path = ""
    if artifact_cache is not None:
        staged_interface = (
            artifact_cache / "installed-interfaces" / installation_id / "package-interface.json"
        )
        package_interface.publish(staged_interface, interface_bytes)
        interface_path = str(staged_interface)
    modeled_paths = sorted(slot.path for entry in interface.entrypoints for slot in entry.models)
    resolved_models: Mapping[str, _DevelopmentModel] = {}
    if modeled_paths:
        if project is None or cozy_home is None or tensorfs_root is None:
            raise RuntimeError(
                "development_model_resolution_absent: modeled development requires the "
                "discovered package, COZY_HOME, and local TensorFS store"
            )
        resolved_models = _resolve_development_models(
            interface, project, tensorfs_root, binding_overrides
        )
        if sorted(resolved_models) != modeled_paths:
            raise RuntimeError("development_model_mismatch: resolved binding paths changed")

    model_ids = {path: f"model-{index:04d}" for index, path in enumerate(modeled_paths)}
    models = [
        pb.Model(
            id=model_ids[path],
            repo=f"{selected.repo}/{selected.name}",
            version=selected.version,
            lane=selected.lane,
            manifest=pb.Ref(
                digest=documents.raw(selected.manifest_digest),
                length=selected.manifest_length,
            ),
        )
        for path, selected in sorted(resolved_models.items())
    ]
    entrypoints: list[pb.Entrypoint] = []
    declared_bindings: dict[str, DeclaredBinding] = {}
    for entry in sorted(interface.entrypoints, key=lambda row: row.name):
        name = entry.name
        selected_paths = sorted(slot.path for slot in entry.models)
        slots: list[pb.Slot] = []
        members: list[ModelBinding] = []
        for selected_path in selected_paths:
            selected = resolved_models[selected_path]
            model_id = model_ids[selected.slot.path]
            slots.append(
                pb.Slot(
                    slot=selected.slot.param,
                    reference_model_id=model_id,
                    components=[
                        pb.Component(component=component, model_id=model_id)
                        for component in selected.components
                    ],
                )
            )
            members.append(
                ModelBinding(
                    model_class=selected.slot.class_key,
                    model_binding_path=selected.slot.path,
                    model_parameter_name=selected.slot.param,
                    store=str(selected.store),
                    variant=selected.variant,
                    reference_snapshot=selected.manifest_digest,
                    components=selected.components,
                    snapshots={
                        component: selected.manifest_digest for component in selected.components
                    },
                    model_weight_bytes=selected.model_weight_bytes,
                    custody=selected.custody,
                    sequence_parallel_degrees=tuple(selected.slot.sequence_parallel),
                    model=f"{selected.repo}/{selected.name}@{selected.version}/{selected.lane}",
                )
            )
        slots.sort(key=lambda row: row.slot)
        members.sort(key=lambda row: row.model_binding_path)
        entrypoint = pb.Entrypoint(name=name, slots=slots)
        entrypoint.entrypoint_binding_digest = hashlib.sha256(
            canonical.write({"name": name, "slots": [documents.body(slot) for slot in slots]})
        ).digest()
        entrypoints.append(entrypoint)

        digest = documents.spell(entrypoint.entrypoint_binding_digest)
        model_set = tuple(members)
        first = model_set[0] if model_set else None
        declared_bindings[digest] = DeclaredBinding(
            entrypoint_binding_digest=digest,
            entrypoint=name,
            model_class=first.model_class if first is not None else "",
            model_binding_path=first.model_binding_path if first is not None else "",
            model_parameter_name=first.model_parameter_name if first is not None else "",
            release=release,
            installation_id=installation_id,
            models=model_set,
            application=interface.application,
            interface_path=interface_path,
            internal=entry.internal,
            development_project=str(project) if project is not None else "",
            store=first.store if first is not None else "",
            variant=first.variant if first is not None else "",
            reference_snapshot=first.reference_snapshot if first is not None else "",
            components=first.components if first is not None else (),
            snapshots=first.snapshots if first is not None else {},
            development=True,
            model_weight_bytes=sum(model.model_weight_bytes for model in model_set),
            custody=first.custody if first is not None else "canonical",
            package=f"{package}@{release}",
        )
    entrypoint_bodies = [documents.body(entrypoint) for entrypoint in entrypoints]
    bindings_digest = hashlib.sha256(
        canonical.write(
            {
                "entrypoints": entrypoint_bodies,
                "models": [documents.body(model) for model in models],
            }
        )
    ).digest()
    placement = pb.Placement(
        placement_id=LOCAL_PLACEMENT,
        development=pb.DevelopmentPackage(
            package=package,
            release=release,
            installation_id=installation_id,
        ),
        installation_id=installation_id,
        package_interface=interface_bytes,
        bindings_digest=bindings_digest,
        models=models,
        entrypoints=entrypoints,
    )
    return DevelopmentPlacementDocuments(interface_bytes, placement, declared_bindings)


@dataclass(frozen=True, slots=True)
class _InterfaceSlot:
    """One declared model slot, read off the interface DOCUMENT rather than off an imported
    app (#713). `<callable>.models.<param>` is the binding point; the class is its key."""

    path: str
    class_key: str
    sequence_parallel: tuple[int, ...] = ()

    @property
    def param(self) -> str:
        return self.path.rpartition(".")[2]


def _interface_slots(docs: Sequence[package_interface.CallableDoc]) -> list[_InterfaceSlot]:
    """A hidden callable is already absent from the document, so nothing is filtered here."""
    return [
        _InterfaceSlot(
            slot.path,
            slot.class_name,
            ()
            if slot.sequence_parallel is msgspec.UNSET
            else tuple(slot.sequence_parallel.degrees),
        )
        for doc in docs
        for slot in doc.models
    ]


def _resolve_development_models(
    interface: package_interface.PackageInterface,
    project: Path,
    tensorfs_root: Path,
    binding_overrides: Mapping[str, str] | None = None,
) -> Mapping[str, _DevelopmentModel]:
    # The strict resolve runs over the WHOLE declared slot set — jobs' included, so a
    # package.toml default for a job source (`<job>.models.<param>`) is a declared slot,
    # not `unknown_binding`. Only entrypoint slots stage models here: a serving closure
    # holds serving models, and a job's source binds per invocation (JobDirective).
    slots = _interface_slots(interface.entrypoints)
    job_slots = _interface_slots(interface.jobs)
    job_paths = frozenset(slot.path for slot in job_slots)
    declared = slots + job_slots
    try:
        table = binding_table.read_table(project / "package.toml")
        chosen = binding_table.resolve(
            declared, table, overrides=binding_overrides, optional=job_paths
        )
    except Exception as exc:
        raise RuntimeError(f"development_model_binding_invalid: {exc}") from exc
    try:
        tensorfs = fill.tensorfs_module()
    except fill.FillRefusal as exc:
        raise RuntimeError(f"development_tensorfs_unavailable: {exc}") from exc

    resolved: dict[str, _DevelopmentModel] = {}
    store_root = tensorfs_root.resolve()
    try:
        if not store_root.is_dir():
            raise FileNotFoundError(store_root)
        store = fill.store(store_root)
    except Exception as exc:
        raise RuntimeError(
            f"development_model_store_absent: {store_root}: {type(exc).__name__}: {exc}"
        ) from exc
    if not hasattr(store, "resolve_release"):
        raise RuntimeError(
            "development_tensorfs_resolver_unavailable: install TensorFS with Store.resolve_release"
        )
    for slot in sorted(slots, key=lambda item: item.path):
        binding = chosen[slot.path]
        repo, name, version = _exact_model_ref(binding.ref)
        lane = binding.lane or ""
        if not lane:
            raise RuntimeError(
                f"development_model_lane_missing: {binding.ref}: package.toml must select a lane"
            )
        try:
            release = store.resolve_release(repo, name, version, lane)
        except Exception as exc:
            code = str(getattr(exc, "code", "MODEL_RELEASE_ABSENT")).lower()
            raise RuntimeError(
                f"development_model_missing: {binding.ref} lane={lane}: {code}"
            ) from exc
        manifest_digest = str(release["manifest_digest"])
        manifest_length = int(release["manifest_length"])
        try:
            manifest = store.manifest(manifest_digest)
            manifest_bytes, header_bytes = manifest["manifest"], manifest["header"]
            if header_bytes is None:
                raise ValueError("the manifest has no cozytensors header")
            header = tensorfs.parse_header(header_bytes)
        except Exception as exc:
            raise RuntimeError(
                f"development_model_manifest_invalid: {binding.ref}: {type(exc).__name__}: {exc}"
            ) from exc
        measured_manifest = "sha256:" + hashlib.sha256(manifest_bytes).hexdigest()
        if measured_manifest != manifest_digest or (
            manifest_length and manifest_length != len(manifest_bytes)
        ):
            raise RuntimeError(
                f"development_model_manifest_mismatch: {binding.ref}: expected "
                f"{manifest_digest}/{manifest_length}, measured "
                f"{measured_manifest}/{len(manifest_bytes)}"
            )
        manifest_length = len(manifest_bytes)
        try:
            config_bytes = model_config.construction_config(header.get("configs"))
        except model_config.ModelConfigRefusal as exc:
            code = exc.code or "development_model_config_invalid"
            raise RuntimeError(f"{code}: {binding.ref}: {exc}") from exc
        if not header.get("components"):
            raise RuntimeError(f"development_model_components_absent: {binding.ref}")
        assets = fill.Checkpoint(store_root, manifest_digest).model_assets()
        components = _derive_components(
            project, slot.path, config_bytes, assets, model_config.tensor_dtypes(header)
        )
        try:
            model_weight_bytes = sum(
                int(row["length"]) for row in store.walk_cozytensors(manifest_digest)
            )
        except Exception as exc:
            raise RuntimeError(
                f"development_model_inventory_invalid: {binding.ref}: {exc}"
            ) from exc
        resolved[slot.path] = _DevelopmentModel(
            slot=slot,
            repo=repo,
            name=name,
            version=version,
            lane=lane,
            manifest_digest=manifest_digest,
            manifest_length=manifest_length,
            components=components,
            store=store_root,
            # Repository lane and accelerator hardware are independent facts. Development
            # carries no caller-authored hardware variant; the executor derives smNN from
            # the measured device after it initializes the accelerator.
            variant="",
            custody="canonical",
            model_weight_bytes=model_weight_bytes,
        )
    return resolved


def _derive_components(
    project: Path,
    slot: str,
    config_bytes: bytes,
    assets: Mapping[str, bytes],
    dtypes: Mapping[str, str],
) -> tuple[str, ...]:
    """What the private bounded child constructs for this config in this package venv."""
    try:
        request = derive_child.DeriveRequest(
            (derive_child.SlotRequest(slot, config_bytes, dict(assets), dict(dtypes)),),
            project=Path(project).resolve(),
        )
        (held,) = derive_child.derive_in(Path(sys.executable), request)
        if isinstance(held, derive_child.SourceSlot):
            raise derive_child.DeriveRefusal(
                "derive_slot_not_serving", f"{slot} is a job source, not a serving model"
            )
        return held.components
    except derive_child.DeriveRefusal as exc:
        raise RuntimeError(
            f"development_model_derive_refused: {slot}: {exc.code}: {exc.detail}"
        ) from exc


def _exact_model_ref(value: str) -> tuple[str, str, str]:
    model, separator, version = value.partition("@")
    parts = model.split("/")
    if not separator or len(parts) != 2 or not all((*parts, version)):
        raise RuntimeError(
            f"development_model_ref_inexact: {value!r} must be exact org/name@release"
        )
    return parts[0], parts[1], version
