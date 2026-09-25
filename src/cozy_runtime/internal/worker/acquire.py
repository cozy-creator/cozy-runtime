"""Acquire the exact bytes selected by one PlacementSet/1.

The PlacementSet is the complete statement of WHAT to run. Tensorhub's authenticated HTTPS
response contains only short-lived locations for the closure the Hub derives from that
accepted set. This module
never reads a file role from the download plan and never reconstructs the deleted bundle,
environment-specification, object-set, or binding documents.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable, Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from threading import Lock, local
from typing import Any, Literal

from cozy_runtime.internal import (
    base_observation,
    fill,
    package_environment,
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

    def __init__(self, callback: Any = None) -> None:
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

    def hold(self, store: Any, digest: str) -> tuple[tuple[str, int], ...]:
        """One verified walk of `digest`'s runtime closure: record the answer and return the
        closure's objects, or re-raise the store's refusal after recording the absence."""
        try:
            lease = store.acquire_cozytensors(digest)
        except Exception:
            self._record(digest, False)
            raise
        try:
            objects = tuple((str(ref), int(length)) for ref, length in lease.objects)
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

    def rebuild(self, store: Any) -> int:
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
    installed: package_environment.InstalledEnvironment | None
    bindings: Mapping[str, DeclaredBinding]
    manifests: int
    documents: int
    wheels: int
    fetched_bytes: int
    verified_bytes: int
    trusted_reuse_bytes: int


def preinstalled_development(
    placement: Any,
    cache_root: Path,
    expected_placement: Any,
    bindings: Mapping[str, DeclaredBinding],
) -> AcquiredPlacement:
    """Admit one independently re-derived source-checkout selection.

    The launch Runtime has re-read package.toml and the exact local TensorFS Repository,
    Manifest, and inline header config. This function compares that derivation with every accepted
    Placement fact before handing the normal attempt kernel its local resolution paths. No
    PackageRelease or Environment is invented for this same-user lane.
    """

    actual = placement.document
    expected = documents.body(expected_placement)
    # Placement ID is RecordOwner routing, not package/model selection identity. A local
    # owner may remint it while preserving the exact development closure.
    comparable_actual = {key: value for key, value in actual.items() if key != "placement_id"}
    comparable_expected = {key: value for key, value in expected.items() if key != "placement_id"}
    if actual.get("package_interface") != expected.get("package_interface"):
        raise AcquisitionRefusal(
            pb.FaultKind.FAULT_KIND_CONFIG_REFUSED,
            "development_source_changed",
            "the current checkout derives a different PackageInterface",
        )
    if actual.get("models", []) != expected.get("models", []):
        raise AcquisitionRefusal(
            pb.FaultKind.FAULT_KIND_ARTIFACT_DIGEST_MISMATCH,
            "development_model_manifest_changed",
            "the local TensorFS Repository now resolves different exact model Manifests",
        )
    actual_entries = {str(row.get("name", "")): row for row in actual.get("entrypoints", [])}
    expected_entries = {str(row.get("name", "")): row for row in expected.get("entrypoints", [])}
    if set(actual_entries) != set(expected_entries):
        raise AcquisitionRefusal(
            pb.FaultKind.FAULT_KIND_CONFIG_REFUSED,
            "development_entrypoint_set_changed",
            "the current checkout exposes a different entrypoint set",
        )
    for name, selected in expected_entries.items():
        accepted = actual_entries[name]
        selected_slots = selected.get("slots", [])
        accepted_slots = accepted.get("slots", [])
        if len(selected_slots) != len(accepted_slots):
            raise AcquisitionRefusal(
                pb.FaultKind.FAULT_KIND_CONFIG_REFUSED,
                "development_model_slot_changed",
                name,
            )
        for selected_slot, accepted_slot in zip(selected_slots, accepted_slots, strict=True):
            if selected_slot.get("components", []) != accepted_slot.get("components", []):
                raise AcquisitionRefusal(
                    pb.FaultKind.FAULT_KIND_CONFIG_REFUSED,
                    "development_model_components_changed",
                    name,
                )
            if selected_slot != accepted_slot:
                raise AcquisitionRefusal(
                    pb.FaultKind.FAULT_KIND_CONFIG_REFUSED,
                    "development_model_slot_changed",
                    name,
                )
    if comparable_actual != comparable_expected:
        raise AcquisitionRefusal(
            pb.FaultKind.FAULT_KIND_CONFIG_REFUSED,
            "development_placement_changed",
            "the accepted Placement does not equal the launch Runtime's local derivation",
        )

    interface_digest, interface_length = placement.package_interface
    interface_path = cache_root / interface_digest.removeprefix("sha256:")
    _verify_file(interface_path, interface_digest, interface_length)
    try:
        interface = package_interface.read_bytes(interface_path.read_bytes(), str(interface_path))
    except package_interface.StalePackageInterface as exc:
        raise AcquisitionRefusal(
            pb.FaultKind.FAULT_KIND_CONFIG_REFUSED, "package_interface_invalid", str(exc)
        ) from exc
    visible = _visible_entrypoints(interface)
    if set(visible) != set(expected_entries):
        raise AcquisitionRefusal(
            pb.FaultKind.FAULT_KIND_CONFIG_REFUSED,
            "binding_entrypoint_set_mismatch",
            "PlacementSet/1 does not cover every interface entrypoint",
        )
    selected_digests = {
        str(row["entrypoint_binding_digest"]): row for row in expected_entries.values()
    }
    if set(bindings) != set(selected_digests):
        raise AcquisitionRefusal(
            pb.FaultKind.FAULT_KIND_CONFIG_REFUSED,
            "development_binding_set_changed",
            "the locally derived bindings do not cover the exact selected entrypoints",
        )
    verified = interface_length
    for digest, binding in bindings.items():
        selected = selected_digests[digest]
        slots = selected.get("slots", [])
        if not slots:
            continue
        selected_paths = {f"{selected['name']}.models.{slot['slot']}" for slot in slots}
        members = binding.model_bindings()
        if selected_paths != {member.model_binding_path for member in members}:
            raise AcquisitionRefusal(
                pb.FaultKind.FAULT_KIND_CONFIG_REFUSED,
                "development_model_slot_changed",
                str(selected["name"]),
            )
    return AcquiredPlacement(
        None,
        bindings,
        len({str(model["manifest"]["digest"]) for model in placement.models}),
        1,
        0,
        0,
        verified,
        0,
    )


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
        installed_environment: package_environment.InstalledEnvironment | None = None,
        verified_cache: VerifiedArtifactCache | None = None,
        held_manifests: HeldManifests | None = None,
        on_observation: Any = None,
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
        self._manifest_bytes: dict[str, int] = {}
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

    def acquire(self, placement: Any) -> AcquiredPlacement:
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
        self, placement: Any, models: Future[int]
    ) -> tuple[package_environment.InstalledEnvironment, Mapping[str, DeclaredBinding], int]:
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

    def _run_model(self, placement: Any) -> int:
        self._thread.leg = "model"
        self.observation.start("model")
        try:
            return self._models(placement)
        finally:
            self.observation.finish("model")

    def _environment(self, placement: Any) -> package_environment.InstalledEnvironment:
        """Reopen the generation this placement's prepare published; move no wheel bytes.

        Wire 30 deletes wheel members and grants from the download path outright: a
        published install's exact content is its locked-requirements ref, an editable one's
        is its supplied wheel facts, and either way the receipt digest is a pure function of
        those placement facts plus the re-observed base. A generation this boot never
        published is a typed resumable refusal — the owner re-prepares; nothing here can
        re-acquire package bytes.
        """

        environment_digest = documents.spell(placement.environment_digest)
        try:
            if placement.package:
                lock = placement.environment.get("locked_requirements")
                if not isinstance(lock, dict):
                    raise AcquisitionRefusal(
                        pb.FaultKind.FAULT_KIND_CONFIG_REFUSED,
                        "environment_lock_absent",
                        "published placement names no locked_requirements",
                    )
                _content, receipt_digest = package_environment.locked_identity(
                    self.base,
                    locked_digest=str(lock.get("digest", "")),
                    locked_length=int(lock.get("length", 0)),
                    environment_digest=environment_digest,
                )
            else:
                package = placement.development
                wheel_rows = [
                    package["project_wheel"],
                    *placement.environment.get("local_wheels", []),
                ]
                wheels = []
                for row in wheel_rows:
                    digest, _length = _ref(row["ref"], "wheel.ref")
                    wheels.append(
                        package_environment.Wheel(
                            distribution=str(row.get("distribution", "")),
                            version=str(row.get("version", "")),
                            filename=str(row.get("filename", "")),
                            digest=digest,
                            length=int(row["ref"]["length"]),
                            tags=tuple(row.get("tags", [])),
                            owner="project" if row is wheel_rows[0] else "custom",
                            import_roots=tuple(row.get("import_roots", [])),
                        )
                    )
                receipt_digest = package_environment.wheel_receipt_digest(
                    self.base,
                    wheels=wheels,
                    environment_digest=environment_digest,
                )
            return package_environment.open_generation(
                self.install_root,
                receipt_digest=receipt_digest,
                environment_digest=environment_digest,
                python=self.python,
            )
        except AcquisitionRefusal:
            raise
        except (LaunchRefusal, package_environment.EnvironmentRefusal) as exc:
            raise AcquisitionRefusal(
                pb.FaultKind.FAULT_KIND_CONFIG_REFUSED,
                getattr(exc, "code", "environment_generation_absent"),
                getattr(exc, "detail", str(exc)),
            ) from exc

    def _models(self, placement: Any) -> int:
        """Hold every distinct selected manifest with ONE verified walk each; returns the
        number of manifests held. The walk's object list prices the bindings."""
        if not placement.models:
            return 0
        try:
            tensorfs = fill.tensorfs_module()
            store = fill.open_store(tensorfs, self.tensorfs_root)
            manifests = {_ref(model["manifest"], "model.manifest") for model in placement.models}
            for manifest_digest, manifest_length in sorted(manifests):
                # TensorFS owns model materialization. A cache miss asks the owner to
                # ensure this exact selection again, including manifest and header.
                manifest_raw = bytes(store.manifest(manifest_digest)["manifest"])
                _verify_bytes(manifest_raw, manifest_digest, manifest_length)
                # This worker moves no model bytes (cr-090): the store is filled by
                # TensorFS's one transport before the placement is acquired. The leg ends
                # on TensorFS's verified walk over the manifest's runtime closure; the
                # same walk is the residency fact `held_manifests` reports.
                objects = self.held_manifests.hold(store, manifest_digest)
                self._manifest_bytes[manifest_digest] = sum(length for _ref_, length in objects)
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

    def _closure_bytes(self, digest: str) -> int:
        """The manifest's runtime-closure bytes from the model leg's one verified walk;
        listed (not verified) only when that leg did not run."""
        if digest not in self._manifest_bytes:
            store = fill.open_store(fill.tensorfs_module(), self.tensorfs_root)
            self._manifest_bytes[digest] = sum(
                int(row["length"]) for row in store.walk_cozytensors(digest)
            )
        return self._manifest_bytes[digest]

    def _bindings(self, placement: Any) -> tuple[Mapping[str, DeclaredBinding], int]:
        interface_digest, interface_length = placement.package_interface
        interface_path = self._stage(
            interface_digest, self.fetch(interface_digest, interface_length).read_bytes()
        )
        try:
            interface = package_interface.read_bytes(
                interface_path.read_bytes(), str(interface_path)
            )
        except package_interface.StalePackageInterface as exc:
            raise AcquisitionRefusal(
                pb.FaultKind.FAULT_KIND_CONFIG_REFUSED, "package_interface_invalid", str(exc)
            ) from exc
        visible = _visible_entrypoints(interface)
        models = {str(row["id"]): row for row in placement.models}
        development = dict(getattr(placement, "development", {}))
        project_wheel = development.get("project_wheel", {})
        # Published placements carry no wheel fact at wire 30; the exact derived package
        # interface is the code surface, so its digest is the binding's code identity.
        package_code_digest = (
            interface_digest
            if placement.package
            else str(
                project_wheel.get("ref", {}).get("digest", "")
                if isinstance(project_wheel, dict)
                else ""
            )
            or str(development.get("source_digest", ""))
        )
        bindings: dict[str, DeclaredBinding] = {}
        count = 1
        for entrypoint in placement.entrypoints:
            name = str(entrypoint["name"])
            binding_digest = str(entrypoint["entrypoint_binding_digest"])
            described = visible.get(name)
            if described is None:
                raise AcquisitionRefusal(
                    pb.FaultKind.FAULT_KIND_CONFIG_REFUSED,
                    "binding_package_interface_invalid",
                    f"interface has no entrypoint {name!r}",
                )
            slots = entrypoint.get("slots", [])
            described_models = described.get("models", [])
            if not isinstance(described_models, list) or len(slots) != len(described_models):
                raise AcquisitionRefusal(
                    pb.FaultKind.FAULT_KIND_CONFIG_REFUSED, "binding_slot_set_mismatch", name
                )
            application = str(interface.get("application", ""))
            if not slots:
                bindings[binding_digest] = DeclaredBinding(
                    entrypoint_binding_digest=binding_digest,
                    entrypoint=name,
                    release=package_code_digest,
                    environment_digest=documents.spell(placement.environment_digest),
                    model_class="",
                    model_binding_path="",
                    model_parameter_name="",
                    application=application,
                    interface_path=str(interface_path),
                    internal=bool(described.get("internal", False)),
                )
                continue
            described_by_path = {
                str(row.get("path", "")): row for row in described_models if isinstance(row, dict)
            }
            if len(described_by_path) != len(described_models):
                raise AcquisitionRefusal(
                    pb.FaultKind.FAULT_KIND_CONFIG_REFUSED,
                    "binding_package_interface_invalid",
                    f"{name}: duplicate or malformed interface model slot",
                )
            members: list[ModelBinding] = []
            staged_weights: list[int] = []
            for slot in slots:
                path = f"{name}.models.{slot['slot']}"
                declared = described_by_path.get(path)
                if declared is None:
                    raise AcquisitionRefusal(
                        pb.FaultKind.FAULT_KIND_CONFIG_REFUSED,
                        "binding_package_interface_invalid",
                        path,
                    )
                reference = models[str(slot["reference_model_id"])]
                reference_manifest = str(reference["manifest"]["digest"])
                components = tuple(str(row["component"]) for row in slot.get("components", []))
                if not components or len(set(components)) != len(components):
                    raise AcquisitionRefusal(
                        pb.FaultKind.FAULT_KIND_CONFIG_REFUSED,
                        "selected_binding_mismatch",
                        f"{path}: selected model has no complete unique component set",
                    )
                snapshots = {
                    str(row["component"]): str(models[str(row["model_id"])]["manifest"]["digest"])
                    for row in slot.get("components", [])
                }
                try:
                    model_weight_bytes = self._closure_bytes(reference_manifest)
                    staged_weights.append(
                        _model_group_weight_bytes(
                            self.tensorfs_root,
                            snapshots,
                            declared.get("component_use", {}),
                            model_weight_bytes,
                        )
                    )
                except Exception as exc:
                    raise AcquisitionRefusal(
                        pb.FaultKind.FAULT_KIND_CONFIG_REFUSED,
                        "selected_model_manifest_unavailable",
                        f"{type(exc).__name__}: {exc}",
                    ) from exc
                sequence_parallel = declared.get("sequence_parallel")
                degrees = (
                    tuple(int(d) for d in sequence_parallel.get("degrees", []))
                    if isinstance(sequence_parallel, dict)
                    else ()
                )
                members.append(
                    ModelBinding(
                        model_class=str(declared.get("class", "")),
                        model_binding_path=path,
                        model_parameter_name=str(slot["slot"]),
                        store=str(self.tensorfs_root),
                        variant="",
                        reference_snapshot=reference_manifest,
                        components=components,
                        snapshots=snapshots,
                        model_weight_bytes=model_weight_bytes,
                        sequence_parallel_degrees=degrees,
                        # The manifest is identity; lane remains the human repo pointer the
                        # desired download set carried beside it.
                        model=str(reference["repo"])
                        + (f"@{reference['version']}" if reference.get("version") else "")
                        + (f"/{reference['lane']}" if reference.get("lane") else ""),
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
                environment_digest=documents.spell(placement.environment_digest),
                model_class=first.model_class,
                model_binding_path=first.model_binding_path,
                model_parameter_name=first.model_parameter_name,
                application=application,
                interface_path=str(interface_path),
                internal=bool(described.get("internal", False)),
                store=first.store,
                variant=first.variant,
                reference_snapshot=first.reference_snapshot,
                components=first.components,
                snapshots=first.snapshots,
                model_weight_bytes=sum(member.model_weight_bytes for member in model_set),
                staged_weight_bytes=sum(staged_weights),
                package=(
                    f"{development['package']}@{development['release']}"
                    if development
                    else f"{placement.package['package']}@{placement.package['release']}"
                ),
            )
        selected_entrypoints = {str(row["name"]) for row in placement.entrypoints}
        # Preparation selects the entrypoints whose model slots were supplied. Each
        # selected entrypoint and its complete slot set was checked above; the package
        # may also declare other entrypoints that this placement does not bind.
        # Captured code, published or unpublished, may precede model selection.
        # This intermediate state has no executable bindings; a weightless
        # entrypoint must still be selected and selected models cannot disappear.
        code_only_preparation = not models and all(
            entrypoint.get("models") for entrypoint in visible.values()
        )
        if not selected_entrypoints and (development or visible) and not code_only_preparation:
            raise AcquisitionRefusal(
                pb.FaultKind.FAULT_KIND_CONFIG_REFUSED,
                "binding_entrypoint_set_mismatch",
                "PlacementSet/1 does not select the required interface entrypoints",
            )
        return bindings, count

    def _stage(self, digest: str, raw: bytes) -> Path:
        self.selection_root.mkdir(parents=True, exist_ok=True, mode=0o755)
        self.selection_root.chmod(0o755)
        target = self.selection_root / digest.removeprefix("sha256:")
        if target.exists():
            _verify_file(target, digest, len(raw))
            return target
        temporary = self.selection_root / f".stage-{digest[7:23]}"
        temporary.write_bytes(raw)
        temporary.chmod(0o444)
        temporary.replace(target)
        return target


def _model_group_weight_bytes(
    root: Path, snapshots: Mapping[str, str], scopes: Mapping[str, list[str]], artifact_bytes: int
) -> int:
    """Price complete staging declarations; otherwise retain full artifact accounting."""
    covered = {name for names in scopes.values() for name in names}
    if not scopes or covered != set(snapshots):
        return artifact_bytes
    checkpoints = {
        snapshot: fill.Checkpoint(root, snapshot) for snapshot in set(snapshots.values())
    }
    checkpoint_sizes = {}
    for snapshot, checkpoint in checkpoints.items():
        sizes = {}
        for component in checkpoint.header["components"]:
            rows = checkpoint.rows(component)
            # Fill adds scratch/derived buffers to this header bound later.
            sizes[component] = max(
                sum(row.nbytes for row in rows), sum(row.stored_nbytes for row in rows)
            )
        checkpoint_sizes[snapshot] = sizes
    full = max(artifact_bytes, sum(sum(sizes.values()) for sizes in checkpoint_sizes.values()))
    if any(
        component not in checkpoint_sizes[snapshot] for component, snapshot in snapshots.items()
    ):
        # The derive's constructed components need not be in the checkpoint.
        # Only model-code-fit judges that mismatch; it cannot justify cheap staging.
        return full
    components = {name: checkpoint_sizes[snapshot][name] for name, snapshot in snapshots.items()}
    return max(
        (sum(components[name] for name in set(names)) for names in scopes.values()), default=full
    )


def _verify_file(path: Path, digest: str, length: int | None) -> bytes:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise AcquisitionRefusal(
            pb.FaultKind.FAULT_KIND_ARTIFACT_FETCH_FAILED,
            "local_artifact_unreadable",
            str(exc),
        ) from exc
    measured = "sha256:" + hashlib.sha256(raw).hexdigest()
    if (length is not None and len(raw) != length) or measured != digest:
        raise AcquisitionRefusal(
            pb.FaultKind.FAULT_KIND_ARTIFACT_DIGEST_MISMATCH,
            "local_artifact_identity_mismatch",
            f"measured {measured}/{len(raw)}, expected {digest}/{length or 'measured length'}",
        )
    return raw


def _verify_bytes(raw: bytes, digest: str, length: int) -> None:
    measured = "sha256:" + hashlib.sha256(raw).hexdigest()
    if len(raw) != length or measured != digest:
        raise AcquisitionRefusal(
            pb.FaultKind.FAULT_KIND_ARTIFACT_DIGEST_MISMATCH,
            "local_artifact_identity_mismatch",
            f"measured {measured}/{len(raw)}, expected {digest}/{length}",
        )


def _ref(value: object, name: str) -> tuple[str, int]:
    if (
        not isinstance(value, dict)
        or set(value) != {"digest", "length"}
        or not isinstance(value.get("digest"), str)
        or not str(value["digest"]).startswith("sha256:")
        or len(str(value["digest"])) != 71
        or not isinstance(value.get("length"), int)
        or isinstance(value["length"], bool)
        or int(value["length"]) <= 0
    ):
        raise AcquisitionRefusal(
            pb.FaultKind.FAULT_KIND_CONFIG_REFUSED,
            f"{name}_ref_invalid",
            "reference is exactly {digest,length}",
        )
    return str(value["digest"]), int(value["length"])


def cozytensors_ref(tensorfs: Any, raw: bytes) -> tuple[str, int]:
    """Select the sole typed CozyTensors header through TensorFS's Manifest parser."""

    try:
        manifest = tensorfs.parse_manifest(raw)
        rows = manifest["entries"]
        selected = [row for row in rows if row.get("kind") == "cozytensors"]
        if len(selected) != 1:
            raise ValueError("expected exactly one cozytensors entry")
        blob = selected[0]["blob"]
        ref = "sha256:" + str(blob["sha256"]), int(blob["length"])
    except Exception as exc:
        raise AcquisitionRefusal(
            pb.FaultKind.FAULT_KIND_CONFIG_REFUSED,
            "model_manifest_invalid",
            f"TensorFS refused Manifest bytes: {type(exc).__name__}: {exc}",
        ) from exc
    if not rows or not ref[0].startswith("sha256:") or len(ref[0]) != 71 or ref[1] <= 0:
        raise AcquisitionRefusal(
            pb.FaultKind.FAULT_KIND_CONFIG_REFUSED,
            "model_manifest_invalid",
            "TensorFS Manifest has no positive CozyTensors header reference",
        )
    return ref


def _visible_entrypoints(interface: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    rows = interface.get("entrypoints")
    if not isinstance(rows, list):
        raise AcquisitionRefusal(
            pb.FaultKind.FAULT_KIND_CONFIG_REFUSED,
            "package_interface_invalid",
            "entrypoints is not an array",
        )
    visible = {
        str(row["name"]): row
        for row in rows
        if isinstance(row, dict) and isinstance(row.get("name"), str)
    }
    if len(visible) != len(rows):
        raise AcquisitionRefusal(
            pb.FaultKind.FAULT_KIND_CONFIG_REFUSED,
            "package_interface_invalid",
            "entrypoint names are not unique",
        )
    return visible
