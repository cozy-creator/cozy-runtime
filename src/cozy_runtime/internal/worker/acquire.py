"""Acquire the exact bytes selected by one PlacementSet/1.

The PlacementSet is the complete statement of WHAT to run. Tensorhub's authenticated HTTPS
response contains only short-lived locations for the closure the Hub derives from that
accepted set. This module
never reads a file role from the download plan and never reconstructs the deleted bundle,
environment-specification, object-set, or binding documents.
"""

from __future__ import annotations

import hashlib
import os
import time
from collections.abc import Callable, Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from threading import Lock, local
from typing import Annotated, Literal

import msgspec

from cozy_runtime.internal import model_config
from cozy_runtime.internal import (
    base_observation,
    fill,
    package_environment,
    package_installation,
    package_interface,
)
from cozy_runtime.internal.refusal import LaunchRefusal
from cozy_runtime.internal.worker.plan import (
    DeclaredBinding,
    ModelBinding,
)
from cozy_runtime.protocol import documents
from cozy_runtime.protocol import worker_pb2 as pb

type FaultKind = pb.FaultKind
type LegName = Literal["package", "model"]
type ObservationCallback = Callable[[pb.PlacementAcquisitionObservation], object]


class _Blob(msgspec.Struct, frozen=True):
    sha256: Annotated[str, msgspec.Meta(pattern="^[0-9a-f]{64}$")]
    length: Annotated[int, msgspec.Meta(gt=0)]


class _Entry(msgspec.Struct, frozen=True):
    kind: str
    blob: _Blob | None = None


class _Manifest(msgspec.Struct, frozen=True):
    """The TensorFS Manifest entries Runtime reads: the sole CozyTensors header's blob."""

    entries: tuple[_Entry, ...] = ()


class AcquisitionRefusal(Exception):
    """A typed, fail-closed acquisition verdict carrying the fault kind it reports as."""

    def __init__(self, kind: FaultKind, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.kind: FaultKind = kind
        self.code = code
        self.detail = detail


@dataclass(slots=True)
class _Leg:
    started: int = 0
    ended: int = 0
    downloaded: int = 0
    reused: int = 0


class AcquisitionObservation:
    """Acquisition-relative monotonic clocks and byte counters.

    Both legs share this observation's origin. Absolute host nanoseconds exceed
    canonical JSON's integer bound after 104 days of uptime; they also carry no
    useful meaning for another machine reading this placement's progress.
    """

    def __init__(self, callback: ObservationCallback | None = None) -> None:
        self._lock = Lock()
        self._origin_ns = time.monotonic_ns()
        self._legs = {"package": _Leg(), "model": _Leg()}
        self._callback = callback

    def start(self, leg: LegName) -> None:
        with self._lock:
            row = self._legs[leg]
            row.started = max(1, time.monotonic_ns() - self._origin_ns)
            row.ended = row.downloaded = row.reused = 0
            snapshot = self._snapshot_locked()
        self._publish(snapshot)

    def add(self, leg: LegName, *, downloaded: int = 0, reused: int = 0) -> None:
        with self._lock:
            row = self._legs[leg]
            if row.started == 0:
                return
            row.downloaded += max(0, downloaded)
            row.reused += max(0, reused)
            snapshot = self._snapshot_locked()
        self._publish(snapshot)

    def finish(self, leg: LegName) -> None:
        with self._lock:
            row = self._legs[leg]
            if row.started == 0:
                return
            row.ended = max(row.started, time.monotonic_ns() - self._origin_ns)
            snapshot = self._snapshot_locked()
        self._publish(snapshot)

    def snapshot(self) -> pb.PlacementAcquisitionObservation:
        with self._lock:
            return self._snapshot_locked()

    def _snapshot_locked(self) -> pb.PlacementAcquisitionObservation:
        value = pb.PlacementAcquisitionObservation()
        for name, row in self._legs.items():
            if row.started == 0:
                continue
            target = getattr(value, name)
            target.started_monotonic_ns = row.started
            target.ended_monotonic_ns = row.ended
            target.downloaded_bytes = row.downloaded
            target.reused_bytes = row.reused
        return value

    def _publish(self, snapshot: pb.PlacementAcquisitionObservation) -> None:
        if self._callback is None:
            return
        with suppress(Exception):
            self._callback(snapshot)


class VerifiedArtifactCache:
    """One worker-boot's stable-file receipts for exact acquisition bytes."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._files: dict[str, package_environment.VerifiedFile] = {}

    def trusted(
        self, digest: str, length: int, path: Path
    ) -> package_environment.VerifiedFile | None:
        with self._lock:
            receipt = self._files.get(digest)
            return (
                receipt if receipt is not None and receipt.matches(path, digest, length) else None
            )

    def record(self, digest: str, length: int, path: Path) -> package_environment.VerifiedFile:
        receipt = package_environment.VerifiedFile.record(path, digest, length)
        with self._lock:
            self._files[digest] = receipt
        return receipt


class HeldManifests:
    """The worker's ONE index of the TensorFS manifests its verified store holds COMPLETE
    (cr-080, residency-aware-routing.md §2): `ObservedWorkerState.held_manifests` and the
    snapshot's copy read it. Membership only — a manifest is held when TensorFS's own
    verified walk (`acquire_cozytensors`, the walk the model leg already ends on) admits
    every object of its runtime closure, whether or not a current placement uses it.

    Two writers: `hold`, when a placement's model leg settles a manifest, and `rebuild`
    at boot, which re-walks the store without a network. A manifest leaves the set when a
    walk finds it incomplete (the bytes went); nothing here removes bytes.
    """

    def __init__(self, on_change: Callable[[], None] | None = None) -> None:
        self._lock = Lock()
        self._held: set[str] = set()
        self._on_change = on_change

    def digests(self) -> list[str]:
        with self._lock:
            return sorted(self._held)

    def hold(self, store: fill.Store, digest: str) -> tuple[tuple[str, int], ...]:
        """One verified walk of `digest`'s runtime closure: record the answer and return the
        closure's objects, or re-raise the store's refusal after recording the absence."""
        try:
            lease = store.acquire_cozytensors(digest)
        except Exception:
            self._record(digest, False)
            raise
        try:
            objects = tuple(lease.objects)
        finally:
            lease.release()
        self._record(digest, True)
        return objects

    def _record(self, digest: str, held: bool) -> None:
        with self._lock:
            before = digest in self._held
            (self._held.add if held else self._held.discard)(digest)
        if held != before and self._on_change is not None:
            self._on_change()

    def rebuild(self, store: fill.Store) -> int:
        """Replace the index from TensorFS's verified complete-CozyTensors census."""

        held = set(store.complete_cozytensors_manifests())
        with self._lock:
            changed = held != self._held
            self._held = held
        if changed and self._on_change is not None:
            self._on_change()
        return len(held)


@dataclass(frozen=True, slots=True)
class AcquiredPlacement:
    installed: package_installation.InstalledEnvironment | None
    bindings: Mapping[str, DeclaredBinding]
    manifests: int
    documents: int
    wheels: int
    fetched_bytes: int
    verified_bytes: int
    trusted_reuse_bytes: int


def package_selection(placement: pb.Placement) -> pb.PackageSelection | pb.DevelopmentPackage:
    """The placement's one package mode: a published selection or a development package."""
    return placement.development if placement.HasField("development") else placement.package


class Acquirer:
    """One placement's concurrent package/model acquisition."""

    def __init__(
        self,
        *,
        cache_root: Path,
        install_root: Path,
        selection_root: Path,
        tensorfs_root: Path,
        python: Path,
        base: base_observation.Observation | None = None,
        installed_environment: package_installation.InstalledEnvironment | None = None,
        verified_cache: VerifiedArtifactCache | None = None,
        held_manifests: HeldManifests | None = None,
        on_observation: ObservationCallback | None = None,
    ) -> None:
        self.cache_root = cache_root
        self.install_root = install_root
        self.selection_root = selection_root
        self.tensorfs_root = tensorfs_root
        self.python = python
        self.base = base or package_environment.observe_base(python)
        self.installed_environment = installed_environment
        self.verified_cache = verified_cache or VerifiedArtifactCache()
        self.held_manifests = held_manifests or HeldManifests()
        self.fetched_bytes = self.verified_bytes = self.trusted_reuse_bytes = self.wheels = 0
        self._fetch_lock = Lock()
        self._new_digests: set[str] = set()
        self._metric_digests: set[str] = set()
        self._observed_digests: dict[LegName, set[str]] = {"package": set(), "model": set()}
        self._thread = local()
        self.observation = AcquisitionObservation(on_observation)

    def fetch(self, digest: str, length: int, *, observe: bool = True) -> Path:
        """Read one reached exact ref from the local artifact cache; this worker downloads
        nothing (cr-090) — bytes reach the pod through TensorFS's one transport."""

        target = self.cache_root / digest.removeprefix("sha256:")
        already = target.exists()
        trusted = self.verified_cache.trusted(digest, length, target)
        trusted_before = trusted is not None
        try:
            if trusted is not None:
                path = target
            elif already:
                _verify_file(target, digest, length)
                path = target
                trusted = self.verified_cache.record(digest, length, target)
            else:
                raise AcquisitionRefusal(
                    pb.FaultKind.FAULT_KIND_ARTIFACT_FETCH_FAILED,
                    "artifact_absent",
                    f"{digest} is reached by PlacementSet/1 and absent from the local "
                    "artifact cache; this runtime holds no downloader",
                )
        except AcquisitionRefusal:
            raise
        except LaunchRefusal as exc:
            raise AcquisitionRefusal(
                pb.FaultKind.FAULT_KIND_ARTIFACT_DIGEST_MISMATCH
                if exc.code
                in {
                    "local_artifact_digest_mismatch",
                    "local_artifact_identity_mismatch",
                    "local_artifact_invalid",
                }
                else pb.FaultKind.FAULT_KIND_ARTIFACT_FETCH_FAILED,
                exc.code,
                exc.detail,
            ) from exc
        with self._fetch_lock:
            if not already and digest not in self._new_digests:
                self._new_digests.add(digest)
                self.fetched_bytes += length
            if digest not in self._metric_digests:
                self._metric_digests.add(digest)
                if trusted_before:
                    self.trusted_reuse_bytes += length
                else:
                    self.verified_bytes += length
            leg = getattr(self._thread, "leg", None)
            if observe and leg in self._observed_digests:
                seen = self._observed_digests[leg]
                if digest not in seen:
                    seen.add(digest)
                    self.observation.add(
                        leg, downloaded=0 if already else length, reused=length if already else 0
                    )
        return path

    def acquire(self, placement: pb.Placement) -> AcquiredPlacement:
        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="cozy-acquire") as pool:
            model_future = pool.submit(self._run_model, placement)
            package_future = pool.submit(self._run_package, placement, model_future)
            installed, bindings, documents_count = package_future.result()
            manifests = model_future.result()
        return AcquiredPlacement(
            installed=installed,
            bindings=bindings,
            manifests=manifests,
            documents=documents_count,
            wheels=self.wheels,
            fetched_bytes=self.fetched_bytes,
            verified_bytes=self.verified_bytes,
            trusted_reuse_bytes=self.trusted_reuse_bytes,
        )

    def _run_package(
        self, placement: pb.Placement, models: Future[int]
    ) -> tuple[package_installation.InstalledEnvironment, Mapping[str, DeclaredBinding], int]:
        self._thread.leg = "package"
        self.observation.start("package")
        try:
            installed = self.installed_environment or self._environment(placement)
            # The header is the sole config authority. Environment installation overlaps the
            # model transfer, then binding resolution waits for the verified header closure.
            models.result()
            bindings, count = self._bindings(placement)
            return installed, bindings, count
        finally:
            self.observation.finish("package")

    def _run_model(self, placement: pb.Placement) -> int:
        self._thread.leg = "model"
        self.observation.start("model")
        try:
            return self._models(placement)
        finally:
            self.observation.finish("model")

    def _environment(self, placement: pb.Placement) -> package_installation.InstalledEnvironment:
        """Reopen the generation this placement's prepare published; move no wheel bytes.

        Wire 30 deletes wheel members and grants from the download path outright: a
        published install's exact content is its locked-requirements ref, an editable one's
        is its supplied wheel facts, and either way the receipt digest is a pure function of
        those placement facts plus the re-observed base. A generation this boot never
        published is a typed resumable refusal — the owner re-prepares; nothing here can
        re-acquire package bytes.
        """

        try:
            return package_installation.open_installation(
                self.install_root, placement.installation_id
            )
        except AcquisitionRefusal:
            raise
        except (LaunchRefusal, package_environment.EnvironmentRefusal) as exc:
            raise AcquisitionRefusal(
                pb.FaultKind.FAULT_KIND_CONFIG_REFUSED,
                getattr(exc, "code", "environment_generation_absent"),
                getattr(exc, "detail", str(exc)),
            ) from exc

    def _models(self, placement: pb.Placement) -> int:
        """Hold every distinct selected manifest with ONE verified walk each; returns the
        number of manifests held."""
        if not placement.models:
            return 0
        try:
            store = fill.open_store(self.tensorfs_root)
            manifests = {
                (documents.spell(model.manifest.digest), model.manifest.length)
                for model in placement.models
            }
            for manifest_digest, manifest_length in sorted(manifests):
                # TensorFS owns model materialization. A cache miss asks the owner to
                # ensure this exact selection again, including manifest and header.
                manifest_raw = bytes(store.manifest(manifest_digest)["manifest"])
                _verify_bytes(manifest_raw, manifest_digest, manifest_length)
                # This worker moves no model bytes (cr-090): the store is filled by
                # TensorFS's one transport before the placement is acquired. The leg ends
                # on TensorFS's verified walk over the manifest's runtime closure; the
                # same walk is the residency fact `held_manifests` reports.
                self.held_manifests.hold(store, manifest_digest)
            return len(manifests)
        except AcquisitionRefusal:
            raise
        except fill.FillRefusal as exc:
            raise AcquisitionRefusal(
                pb.FaultKind.FAULT_KIND_ARTIFACT_FETCH_FAILED,
                "model_materialization_required"
                if exc.code == "missing_object"
                else "tensorfs_refused",
                str(exc),
            ) from exc
        except Exception as exc:
            raise AcquisitionRefusal(
                pb.FaultKind.FAULT_KIND_ARTIFACT_FETCH_FAILED,
                "model_materialization_required"
                if getattr(exc, "code", "") == "OBJECT_ABSENT"
                else "tensorfs_refused",
                f"{type(exc).__name__}: {exc}",
            ) from exc

    def _bindings(self, placement: pb.Placement) -> tuple[Mapping[str, DeclaredBinding], int]:
        selection = package_selection(placement)
        interface_path = self.selection_root / placement.installation_id / "package-interface.json"
        package_interface.publish(interface_path, placement.package_interface)
        interface = _interface(placement.package_interface, str(interface_path))
        visible = _entrypoints_by_name(interface)
        models = {row.id: row for row in placement.models}
        package_code_digest = placement.installation_id
        bindings: dict[str, DeclaredBinding] = {}
        count = 1
        for entrypoint in placement.entrypoints:
            name = entrypoint.name
            binding_digest = documents.spell(entrypoint.entrypoint_binding_digest)
            described = visible.get(name)
            if described is None:
                raise AcquisitionRefusal(
                    pb.FaultKind.FAULT_KIND_CONFIG_REFUSED,
                    "binding_package_interface_invalid",
                    f"interface has no entrypoint {name!r}",
                )
            slots = entrypoint.slots
            if len(slots) != len(described.models):
                raise AcquisitionRefusal(
                    pb.FaultKind.FAULT_KIND_CONFIG_REFUSED, "binding_slot_set_mismatch", name
                )
            application = interface.application
            if not slots:
                bindings[binding_digest] = DeclaredBinding(
                    entrypoint_binding_digest=binding_digest,
                    entrypoint=name,
                    release=package_code_digest,
                    installation_id=placement.installation_id,
                    model_class="",
                    model_binding_path="",
                    model_parameter_name="",
                    application=application,
                    interface_path=str(interface_path),
                    internal=described.internal,
                )
                continue
            described_by_path = {row.path: row for row in described.models}
            if len(described_by_path) != len(described.models):
                raise AcquisitionRefusal(
                    pb.FaultKind.FAULT_KIND_CONFIG_REFUSED,
                    "binding_package_interface_invalid",
                    f"{name}: duplicate or malformed interface model slot",
                )
            members: list[ModelBinding] = []
            for slot in slots:
                path = f"{name}.models.{slot.slot}"
                declared = described_by_path.get(path)
                if declared is None:
                    raise AcquisitionRefusal(
                        pb.FaultKind.FAULT_KIND_CONFIG_REFUSED,
                        "binding_package_interface_invalid",
                        path,
                    )
                reference = models[slot.reference_model_id]
                reference_manifest = documents.spell(reference.manifest.digest)
                components = tuple(row.component for row in slot.components)
                if not components or len(set(components)) != len(components):
                    raise AcquisitionRefusal(
                        pb.FaultKind.FAULT_KIND_CONFIG_REFUSED,
                        "selected_binding_mismatch",
                        f"{path}: selected model has no complete unique component set",
                    )
                snapshots = {
                    row.component: documents.spell(models[row.model_id].manifest.digest)
                    for row in slot.components
                }
                try:
                    logical = fill.logical_weight_bytes(self.tensorfs_root, snapshots)
                except Exception as exc:
                    raise AcquisitionRefusal(
                        pb.FaultKind.FAULT_KIND_CONFIG_REFUSED,
                        "selected_model_manifest_unavailable",
                        f"{type(exc).__name__}: {exc}",
                    ) from exc
                degrees = (
                    ()
                    if declared.sequence_parallel is msgspec.UNSET
                    else declared.sequence_parallel.degrees
                )
                members.append(
                    ModelBinding(
                        model_class=declared.class_name,
                        model_binding_path=path,
                        model_parameter_name=slot.slot,
                        store=str(self.tensorfs_root),
                        variant="",
                        reference_snapshot=reference_manifest,
                        components=components,
                        snapshots=snapshots,
                        logical_weight_bytes=sum(logical.values()),
                        sequence_parallel_degrees=degrees,
                        prepared_adapters=any(
                            model_config.adapter_graph(
                                fill.Checkpoint(self.tensorfs_root, snapshot).header.get("configs")
                            )
                            for snapshot in set(snapshots.values())
                        ),
                        # The manifest is identity; lane remains the human repo pointer the
                        # desired download set carried beside it.
                        model=reference.repo
                        + (f"@{reference.version}" if reference.version else "")
                        + (f"/{reference.lane}" if reference.lane else ""),
                        adapters=tuple(
                            {
                                "component": adapter.component,
                                "model_id": adapter.model_id,
                                "source_component": adapter.source_component,
                                "scale": adapter.scale,
                            }
                            for adapter in slot.adapters
                        ),
                    )
                )
            members.sort(key=lambda member: member.model_binding_path)
            model_set = tuple(members)
            first = model_set[0]
            bindings[binding_digest] = DeclaredBinding(
                entrypoint_binding_digest=binding_digest,
                entrypoint=name,
                release=package_code_digest,
                models=model_set,
                installation_id=placement.installation_id,
                model_class=first.model_class,
                model_binding_path=first.model_binding_path,
                model_parameter_name=first.model_parameter_name,
                application=application,
                interface_path=str(interface_path),
                internal=described.internal,
                store=first.store,
                variant=first.variant,
                reference_snapshot=first.reference_snapshot,
                components=first.components,
                snapshots=first.snapshots,
                logical_weight_bytes=sum(member.logical_weight_bytes for member in model_set),
                package=f"{selection.package}@{selection.release}",
            )
        selected_entrypoints = {row.name for row in placement.entrypoints}
        # Preparation selects the entrypoints whose model slots were supplied. Each
        # selected entrypoint and its complete slot set was checked above; the package
        # may also declare other entrypoints that this placement does not bind.
        # Captured code, published or unpublished, may precede model selection.
        # This intermediate state has no executable bindings; a weightless
        # entrypoint must still be selected and selected models cannot disappear.
        code_only_preparation = not models and all(
            entrypoint.models for entrypoint in visible.values()
        )
        if (
            not selected_entrypoints
            and (placement.HasField("development") or visible)
            and not code_only_preparation
        ):
            raise AcquisitionRefusal(
                pb.FaultKind.FAULT_KIND_CONFIG_REFUSED,
                "binding_entrypoint_set_mismatch",
                "PlacementSet/1 does not select the required interface entrypoints",
            )
        return bindings, count


def _verify_file(path: Path, digest: str, length: int | None) -> None:
    """Hash a cached artifact as a stream; its size is never bounded by host memory."""
    try:
        with path.open("rb") as stream:
            measured = "sha256:" + hashlib.file_digest(stream, "sha256").hexdigest()
            size = os.fstat(stream.fileno()).st_size
    except OSError as exc:
        raise AcquisitionRefusal(
            pb.FaultKind.FAULT_KIND_ARTIFACT_FETCH_FAILED,
            "local_artifact_unreadable",
            str(exc),
        ) from exc
    if (length is not None and size != length) or measured != digest:
        raise AcquisitionRefusal(
            pb.FaultKind.FAULT_KIND_ARTIFACT_DIGEST_MISMATCH,
            "local_artifact_identity_mismatch",
            f"measured {measured}/{size}, expected {digest}/{length or 'measured length'}",
        )


def _verify_bytes(raw: bytes, digest: str, length: int) -> None:
    measured = "sha256:" + hashlib.sha256(raw).hexdigest()
    if len(raw) != length or measured != digest:
        raise AcquisitionRefusal(
            pb.FaultKind.FAULT_KIND_ARTIFACT_DIGEST_MISMATCH,
            "local_artifact_identity_mismatch",
            f"measured {measured}/{len(raw)}, expected {digest}/{length}",
        )


def cozytensors_ref(raw: bytes) -> tuple[str, int]:
    """Select the sole typed CozyTensors header through TensorFS's Manifest parser."""

    try:
        manifest = msgspec.convert(fill.tensorfs_module().parse_manifest(raw), _Manifest)
    except Exception as exc:
        raise AcquisitionRefusal(
            pb.FaultKind.FAULT_KIND_CONFIG_REFUSED,
            "model_manifest_invalid",
            f"TensorFS refused Manifest bytes: {type(exc).__name__}: {exc}",
        ) from exc
    blobs = [entry.blob for entry in manifest.entries if entry.kind == "cozytensors"]
    if len(blobs) != 1 or blobs[0] is None:
        raise AcquisitionRefusal(
            pb.FaultKind.FAULT_KIND_CONFIG_REFUSED,
            "model_manifest_invalid",
            "TensorFS Manifest has no positive CozyTensors header reference",
        )
    return "sha256:" + blobs[0].sha256, blobs[0].length


def _interface(raw: bytes, source: str) -> package_interface.PackageInterface:
    try:
        return package_interface.parse(raw, source)
    except package_interface.StalePackageInterface as exc:
        raise AcquisitionRefusal(
            pb.FaultKind.FAULT_KIND_CONFIG_REFUSED, "package_interface_invalid", str(exc)
        ) from exc


def _entrypoints_by_name(
    interface: package_interface.PackageInterface,
) -> dict[str, package_interface.CallableDoc]:
    rows = interface.entrypoints
    visible = {row.name: row for row in rows}
    if len(visible) != len(rows):
        raise AcquisitionRefusal(
            pb.FaultKind.FAULT_KIND_CONFIG_REFUSED,
            "package_interface_invalid",
            "entrypoint names are not unique",
        )
    return visible
