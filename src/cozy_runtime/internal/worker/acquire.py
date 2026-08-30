"""Acquire the exact bytes selected by one PlacementSet/1.

The PlacementSet is the complete statement of WHAT to run. Tensorhub's mTLS response contains
only short-lived locations for the closure the Hub derives from that accepted set. This module
never reads a file role from the download plan and never reconstructs the deleted bundle,
environment-specification, object-set, or binding documents.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from threading import Lock, local
from typing import Any, Literal, TypeAlias

from cozy_runtime import qualification
from cozy_runtime.internal import (
    canonical,
    contract,
    fill,
    overlay,
    placement_materialization,
    pull,
)
from cozy_runtime.internal import (
    descriptor as package_descriptor,
)
from cozy_runtime.internal.refusal import LaunchRefusal
from cozy_runtime.internal.worker.downloads import DownloadPlan
from cozy_runtime.internal.worker.plan import DeclaredBinding
from cozy_runtime.protocol import documents
from cozy_runtime.protocol import worker_pb2 as pb

FaultKind: TypeAlias = pb.FaultKind
LegName: TypeAlias = Literal["package", "model"]
MAX_DOCUMENT_BYTES = 64 << 20


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
    """Observation-only acquisition clocks and byte counters."""

    def __init__(self, callback: Any = None) -> None:
        self._lock = Lock()
        self._legs = {"package": _Leg(), "model": _Leg()}
        self._callback = callback

    def start(self, leg: LegName) -> None:
        with self._lock:
            row = self._legs[leg]
            row.started = max(1, time.monotonic_ns())
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
            row.ended = max(row.started, time.monotonic_ns())
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
        self._files: dict[str, overlay.VerifiedFile] = {}

    def trusted(self, digest: str, length: int, path: Path) -> overlay.VerifiedFile | None:
        with self._lock:
            receipt = self._files.get(digest)
            return (
                receipt if receipt is not None and receipt.matches(path, digest, length) else None
            )

    def record(self, digest: str, length: int, path: Path) -> overlay.VerifiedFile:
        receipt = overlay.VerifiedFile.record(path, digest, length)
        with self._lock:
            self._files[digest] = receipt
        return receipt


@dataclass(frozen=True, slots=True)
class AcquiredPlacement:
    installed: overlay.InstalledOverlay | None
    bindings: Mapping[str, DeclaredBinding]
    model_objects: int
    documents: int
    wheels: int
    fetched_bytes: int
    verified_bytes: int
    trusted_reuse_bytes: int


def preinstalled_development(placement: Any, cache_root: Path) -> AcquiredPlacement:
    """Resolve a source-checkout development placement without inventing an install overlay.

    Published execution uses :class:`Acquirer` and an exact qualified EnvironmentReceipt. This
    explicit same-user lane executes its launch-local source and trusts no invented release,
    project wheel, environment, or qualification document.
    """

    if placement.models or any(entrypoint.get("slots") for entrypoint in placement.entrypoints):
        raise AcquisitionRefusal(
            pb.FaultKind.FAULT_KIND_CONFIG_REFUSED,
            "development_placement_not_weightless",
            "source-checkout execution supports only model-free entrypoints",
        )
    descriptor_digest, descriptor_length = placement.package_descriptor
    descriptor_path = cache_root / descriptor_digest.removeprefix("sha256:")
    _verify_file(descriptor_path, descriptor_digest, descriptor_length)
    try:
        descriptor = package_descriptor.read_bytes(
            descriptor_path.read_bytes(), str(descriptor_path)
        )
    except package_descriptor.StalePackageDescriptor as exc:
        raise AcquisitionRefusal(
            pb.FaultKind.FAULT_KIND_CONFIG_REFUSED, "package_descriptor_invalid", str(exc)
        ) from exc
    visible = _visible_entrypoints(descriptor)
    selected = {str(row["name"]): row for row in placement.entrypoints}
    if set(visible) != set(selected):
        raise AcquisitionRefusal(
            pb.FaultKind.FAULT_KIND_CONFIG_REFUSED,
            "binding_entrypoint_set_mismatch",
            "PlacementSet/1 does not cover every descriptor entrypoint",
        )
    source_digest = str(placement.development["source_digest"])
    bindings = {
        str(row["entrypoint_binding_digest"]): DeclaredBinding(
            entrypoint_binding_digest=str(row["entrypoint_binding_digest"]),
            entrypoint=name,
            package_revision_digest=source_digest,
            release=source_digest,
            environment_digest="",
            development=True,
            model_class="",
            model_binding_path="",
            model_parameter_name="",
            application=str(descriptor.get("application", "")),
            descriptor_path=str(descriptor_path),
        )
        for name, row in selected.items()
    }
    return AcquiredPlacement(None, bindings, 0, 1, 0, 0, descriptor_length, 0)


class Acquirer:
    """One placement's concurrent package/model acquisition."""

    def __init__(
        self,
        plan: DownloadPlan,
        *,
        cache_root: Path,
        environment_root: Path,
        selection_root: Path,
        artifact_store: Path,
        python: Path,
        base_manifest: Path = overlay.DEFAULT_BASE_MANIFEST,
        verified_cache: VerifiedArtifactCache | None = None,
        on_observation: Any = None,
    ) -> None:
        self.plan = plan
        self.cache_root = cache_root
        self.environment_root = environment_root
        self.selection_root = selection_root
        self.artifact_store = artifact_store
        self.python = python
        self.base_manifest = base_manifest
        self.verified_cache = verified_cache or VerifiedArtifactCache()
        self.fetched_bytes = self.verified_bytes = self.trusted_reuse_bytes = self.wheels = 0
        self._fetch_lock = Lock()
        self._new_digests: set[str] = set()
        self._metric_digests: set[str] = set()
        self._observed_digests: dict[LegName, set[str]] = {"package": set(), "model": set()}
        self._thread = local()
        self.observation = AcquisitionObservation(on_observation)

    def fetch(self, digest: str, length: int, *, observe: bool = True) -> Path:
        """Read one reached exact ref locally, or fetch it through the current Hub plan."""

        target = self.cache_root / digest.removeprefix("sha256:")
        planned = self.plan.subjects.get(digest)
        if planned is not None and planned != length:
            raise AcquisitionRefusal(
                pb.FaultKind.FAULT_KIND_ARTIFACT_DIGEST_MISMATCH,
                "artifact_length_conflict",
                f"{digest} is named at {length} B and planned at {planned} B",
            )
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
                location = self.plan.locations.get(digest)
                deadline = self.plan.deadline(digest)
                if planned is None or not location:
                    raise AcquisitionRefusal(
                        pb.FaultKind.FAULT_KIND_ARTIFACT_FETCH_FAILED,
                        "download_plan_missing",
                        f"{digest} is reached by PlacementSet/1 and absent locally and "
                        "from the plan",
                    )
                if deadline <= int(time.time()):
                    raise AcquisitionRefusal(
                        pb.FaultKind.FAULT_KIND_ARTIFACT_FETCH_FAILED,
                        "download_url_expired",
                        f"the presigned read for {digest} expired at {deadline}",
                    )
                path = placement_materialization.fetch_exact(
                    {"digest": digest, "length": length, "url": location},
                    self.cache_root,
                    deadline=deadline,
                    allow_local=self.plan.allow_local,
                )
                trusted = self.verified_cache.record(digest, length, path)
        except AcquisitionRefusal:
            raise
        except LaunchRefusal as exc:
            raise AcquisitionRefusal(
                pb.FaultKind.FAULT_KIND_ARTIFACT_DIGEST_MISMATCH
                if exc.code
                in {
                    "grant_identity_mismatch",
                    "grant_length_mismatch",
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
            package_future = pool.submit(self._run_package, placement)
            model_future = pool.submit(self._run_model, placement)
            installed, bindings, documents_count = package_future.result()
            model_objects = model_future.result()
        self._verify_model_refs(bindings)
        return AcquiredPlacement(
            installed=installed,
            bindings=bindings,
            model_objects=model_objects,
            documents=documents_count,
            wheels=self.wheels,
            fetched_bytes=self.fetched_bytes,
            verified_bytes=self.verified_bytes,
            trusted_reuse_bytes=self.trusted_reuse_bytes,
        )

    def _run_package(
        self, placement: Any
    ) -> tuple[overlay.InstalledOverlay, Mapping[str, DeclaredBinding], int]:
        self._thread.leg = "package"
        self.observation.start("package")
        try:
            installed = self._environment(placement)
            bindings, count = self._bindings(placement, installed)
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

    def _environment(self, placement: Any) -> overlay.InstalledOverlay:
        environment_digest = documents.spell(placement.environment_digest)
        receipt_digest = documents.spell(placement.environment_receipt_digest)
        wheelhouse = _ref(placement.environment["wheelhouse_manifest"], "wheelhouse_manifest")
        wheelhouse_raw = self.fetch(*wheelhouse).read_bytes()
        wheel_rows = [placement.package["project_wheel"], *placement.environment.get("wheels", [])]
        wheel_paths: dict[str, Path] = {}
        verified: dict[str, overlay.VerifiedFile] = {}
        for row in wheel_rows:
            digest, length = _ref(row["ref"], "wheel.ref")
            path = self.fetch(digest, length)
            wheel_paths[digest] = path
            receipt = self.verified_cache.trusted(digest, length, path)
            if receipt is None:
                raise AcquisitionRefusal(
                    pb.FaultKind.FAULT_KIND_ARTIFACT_DIGEST_MISMATCH,
                    "wheel_verification_receipt_absent",
                    digest,
                )
            verified[digest] = receipt
        self.wheels = len(wheel_paths)
        try:
            source = placement_materialization.package_overlay_input(
                package=placement.package,
                environment=placement.environment,
                environment_digest=environment_digest,
                wheelhouse_manifest_bytes=wheelhouse_raw,
                wheel_paths=wheel_paths,
                verified_wheels=verified,
            )
            return overlay.materialize(
                source,
                self.environment_root,
                python=self.python,
                base_manifest=self.base_manifest,
                expected_receipt_digest=receipt_digest,
            )
        except (LaunchRefusal, overlay.OverlayRefusal) as exc:
            raise AcquisitionRefusal(
                pb.FaultKind.FAULT_KIND_ENVIRONMENT_RECEIPT_MISMATCH
                if "receipt" in exc.code
                else pb.FaultKind.FAULT_KIND_CONFIG_REFUSED,
                exc.code,
                exc.detail,
            ) from exc

    def _models(self, placement: Any) -> int:
        if not placement.models:
            return 0
        try:
            tensorfs = fill.tensorfs_module()
            store = pull.open_store(tensorfs, self.artifact_store)
            count = 0
            for model in placement.models:
                manifest_digest, manifest_length = _ref(model["manifest"], "model.manifest")
                try:
                    manifest_raw = bytes(store.manifest(manifest_digest)["manifest"])
                    _verify_bytes(manifest_raw, manifest_digest, manifest_length)
                except Exception as exc:
                    if getattr(exc, "code", "") != "OBJECT_ABSENT":
                        raise
                    manifest_path = self.fetch(manifest_digest, manifest_length)
                    manifest_raw = manifest_path.read_bytes()
                    store.put_manifest(manifest_raw, manifest_digest, manifest_length)
                    for digest, length in _manifest_entries(manifest_raw):
                        path = self.fetch(digest, length)
                        store.put_file(str(path), digest, length)
                reached = {
                    (str(row["id"]), int(row["length"])) for row in store.walk(manifest_digest)
                }
                closure = {(manifest_digest, manifest_length), *reached}
                if self.plan.authorized():
                    reads: list[pull.Read] = []
                    for digest, length in sorted(closure):
                        planned = self.plan.subjects.get(digest)
                        location = self.plan.locations.get(digest)
                        if planned != length or not location:
                            raise AcquisitionRefusal(
                                pb.FaultKind.FAULT_KIND_ARTIFACT_FETCH_FAILED,
                                "model_manifest_plan_incomplete",
                                f"{digest} is reached at {length} B and not exactly planned",
                            )
                        reads.append(pull.Read(digest, length, location))
                    deadline = min(self.plan.deadline(digest) for digest, _ in closure)
                    outcome = pull.pull_granted_manifest(
                        tensorfs,
                        store,
                        manifest_digest,
                        manifest_length,
                        reads,
                        timeout=max(0.001, float(deadline) - time.time()),
                        session="placement-" + manifest_digest[7:19],
                        allow_local=self.plan.allow_local,
                    )
                    self.observation.add(
                        "model",
                        downloaded=outcome.bytes_moved,
                        reused=max(0, outcome.bytes_total - outcome.bytes_moved),
                    )
                else:
                    store.acquire_manifest(manifest_digest).release()
                count += len(closure)
            return count
        except AcquisitionRefusal:
            raise
        except (pull.PullRefusal, fill.FillRefusal) as exc:
            code = exc.code if isinstance(exc, pull.PullRefusal) else "tensorfs_refused"
            detail = exc.message if isinstance(exc, pull.PullRefusal) else str(exc)
            raise AcquisitionRefusal(
                pb.FaultKind.FAULT_KIND_ARTIFACT_FETCH_FAILED, code, detail
            ) from exc
        except Exception as exc:
            raise AcquisitionRefusal(
                pb.FaultKind.FAULT_KIND_ARTIFACT_FETCH_FAILED,
                "tensorfs_refused",
                f"{type(exc).__name__}: {exc}",
            ) from exc

    def _bindings(
        self, placement: Any, installed: overlay.InstalledOverlay
    ) -> tuple[Mapping[str, DeclaredBinding], int]:
        descriptor_digest, descriptor_length = placement.package_descriptor
        descriptor_path = self._stage(
            descriptor_digest, self.fetch(descriptor_digest, descriptor_length).read_bytes()
        )
        try:
            descriptor = package_descriptor.read_bytes(
                descriptor_path.read_bytes(), str(descriptor_path)
            )
        except package_descriptor.StalePackageDescriptor as exc:
            raise AcquisitionRefusal(
                pb.FaultKind.FAULT_KIND_CONFIG_REFUSED, "package_descriptor_invalid", str(exc)
            ) from exc
        qualification_digest, qualification_length = placement.qualification
        qualification_raw = self.fetch(qualification_digest, qualification_length).read_bytes()
        try:
            selected = qualification.read(qualification_raw)
            expected = qualification.selection_digest(
                bindings_digest=documents.spell(placement.bindings_digest),
                environment_digest=documents.spell(placement.environment_digest),
                environment_receipt={
                    "digest": installed.receipt_digest,
                    "length": len(installed.receipt_bytes),
                },
                package_descriptor={
                    "digest": descriptor_digest,
                    "length": descriptor_length,
                },
                package=placement.package,
            )
        except qualification.QualificationError as exc:
            raise AcquisitionRefusal(
                pb.FaultKind.FAULT_KIND_CONFIG_REFUSED, "qualification_invalid", str(exc)
            ) from exc
        if selected != expected:
            raise AcquisitionRefusal(
                pb.FaultKind.FAULT_KIND_CONFIG_REFUSED,
                "qualification_selection_mismatch",
                "Qualification does not authorize this exact PlacementSet/1 selection",
            )
        self._stage(qualification_digest, qualification_raw)
        visible = _visible_entrypoints(descriptor)
        models = {str(row["id"]): row for row in placement.models}
        release_digest = str(placement.package["release_digest"])
        bindings: dict[str, DeclaredBinding] = {}
        count = 2
        for entrypoint in placement.entrypoints:
            name = str(entrypoint["name"])
            binding_digest = str(entrypoint["entrypoint_binding_digest"])
            described = visible.get(name)
            if described is None:
                raise AcquisitionRefusal(
                    pb.FaultKind.FAULT_KIND_CONFIG_REFUSED,
                    "binding_package_descriptor_invalid",
                    f"descriptor has no entrypoint {name!r}",
                )
            slots = entrypoint.get("slots", [])
            described_models = described.get("models", [])
            if not isinstance(described_models, list) or len(slots) != len(described_models):
                raise AcquisitionRefusal(
                    pb.FaultKind.FAULT_KIND_CONFIG_REFUSED, "binding_slot_set_mismatch", name
                )
            if len(slots) > 1:
                raise AcquisitionRefusal(
                    pb.FaultKind.FAULT_KIND_CONFIG_REFUSED,
                    "binding_slot_count_unsupported",
                    "this Runtime currently supports at most one selected model slot per "
                    "entrypoint",
                )
            application = str(descriptor.get("application", ""))
            if not slots:
                bindings[binding_digest] = DeclaredBinding(
                    entrypoint_binding_digest=binding_digest,
                    entrypoint=name,
                    package_revision_digest=release_digest,
                    release=release_digest,
                    environment_digest=documents.spell(placement.environment_digest),
                    model_class="",
                    model_binding_path="",
                    model_parameter_name="",
                    application=application,
                    descriptor_path=str(descriptor_path),
                )
                continue
            slot = slots[0]
            declared = described_models[0]
            if (
                not isinstance(declared, dict)
                or declared.get("path") != f"{name}.models.{slot['slot']}"
            ):
                raise AcquisitionRefusal(
                    pb.FaultKind.FAULT_KIND_CONFIG_REFUSED,
                    "binding_package_descriptor_invalid",
                    name,
                )
            config_ref = _ref(slot["config"]["document"], "slot.config.document")
            config_path = self._stage(config_ref[0], self.fetch(*config_ref).read_bytes())
            contract_ref = _ref(slot["model_construction_contract"], "slot.mcc")
            contract_raw = self.fetch(*contract_ref).read_bytes()
            self._stage(contract_ref[0], contract_raw)
            count += 2
            try:
                construction = contract.read_contract(contract_raw)
            except contract.ContractError as exc:
                raise AcquisitionRefusal(
                    pb.FaultKind.FAULT_KIND_CONFIG_REFUSED,
                    "model_construction_contract_invalid",
                    str(exc),
                ) from exc
            candidate = construction.candidate
            reference = models[str(slot["reference_model_id"])]
            reference_manifest = str(reference["manifest"]["digest"])
            model_class = str(declared.get("class", ""))
            if (
                candidate.release
                not in {
                    release_digest,
                    release_digest.removeprefix("sha256:"),
                }
                or candidate.application != application
                or not candidate.model_class.endswith(":" + model_class)
                or candidate.snapshot != reference_manifest
                or candidate.config_digest != config_ref[0]
            ):
                raise AcquisitionRefusal(
                    pb.FaultKind.FAULT_KIND_CONFIG_REFUSED,
                    "selected_binding_mismatch",
                    f"{name}: MCC candidate does not equal PlacementSet/1",
                )
            components = tuple(str(row["component"]) for row in slot.get("components", []))
            if components != tuple(row.component for row in construction.destination_sets):
                raise AcquisitionRefusal(
                    pb.FaultKind.FAULT_KIND_CONFIG_REFUSED,
                    "selected_binding_mismatch",
                    f"{name}: component order differs from the MCC",
                )
            snapshots = {
                str(row["component"]): str(models[str(row["model_id"])]["manifest"]["digest"])
                for row in slot.get("components", [])
            }
            stamps: dict[str, set[str]] = {}
            for row in slot.get("stamps", []):
                stamps.setdefault(str(row["key"]), set()).update(str(v) for v in row["values"])
            descriptor_stamps = declared.get("stamps", {})
            if not isinstance(descriptor_stamps, dict) or any(
                value not in stamps.get(str(key), set()) for key, value in descriptor_stamps.items()
            ):
                raise AcquisitionRefusal(
                    pb.FaultKind.FAULT_KIND_CONFIG_REFUSED,
                    "selected_binding_mismatch",
                    f"{name}: descriptor stamps are not selected",
                )
            try:
                model_weight_bytes = contract.destination_resident_bytes(
                    construction, contract.tensorfs_destination_widths(fill.tensorfs_module())
                )
            except (contract.ContractError, fill.FillRefusal) as exc:
                raise AcquisitionRefusal(
                    pb.FaultKind.FAULT_KIND_CONFIG_REFUSED,
                    "model_construction_contract_invalid",
                    str(exc),
                ) from exc
            bindings[binding_digest] = DeclaredBinding(
                entrypoint_binding_digest=binding_digest,
                entrypoint=name,
                package_revision_digest=release_digest,
                release=release_digest,
                environment_digest=documents.spell(placement.environment_digest),
                model_class=model_class,
                model_binding_path=str(declared["path"]),
                model_parameter_name=str(slot["slot"]),
                application=application,
                descriptor_path=str(descriptor_path),
                store=str(self.artifact_store),
                config=str(config_path),
                variant=candidate.hardware_variant,
                reference_snapshot=reference_manifest,
                components=components,
                snapshots=snapshots,
                stamps={key: tuple(sorted(values)) for key, values in sorted(stamps.items())},
                model_weight_bytes=model_weight_bytes,
                expected_logical_model_digest=contract.logical_model_digest(construction),
            )
            for asset in slot["config"].get("assets", []):
                ref = _ref(asset["ref"], "slot.config.asset")
                self._stage(ref[0], self.fetch(*ref).read_bytes())
                count += 1
        if set(visible) != {str(row["name"]) for row in placement.entrypoints}:
            raise AcquisitionRefusal(
                pb.FaultKind.FAULT_KIND_CONFIG_REFUSED,
                "binding_entrypoint_set_mismatch",
                "PlacementSet/1 does not cover every descriptor entrypoint",
            )
        return bindings, count

    def _verify_model_refs(self, bindings: Mapping[str, DeclaredBinding]) -> None:
        if not any(not binding.weightless for binding in bindings.values()):
            return
        try:
            store = pull.open_store(fill.tensorfs_module(), self.artifact_store)
            for binding in bindings.values():
                for digest in {binding.reference_snapshot, *binding.snapshots.values()}:
                    store.acquire_manifest(digest).release()
        except Exception as exc:
            raise AcquisitionRefusal(
                pb.FaultKind.FAULT_KIND_ARTIFACT_DIGEST_MISMATCH,
                "selected_model_manifest_unavailable",
                f"{type(exc).__name__}: {exc}",
            ) from exc

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


def _document(raw: bytes, name: str) -> dict[str, Any]:
    if len(raw) > MAX_DOCUMENT_BYTES:
        raise AcquisitionRefusal(
            pb.FaultKind.FAULT_KIND_CONFIG_REFUSED, f"{name}_too_large", str(len(raw))
        )
    try:
        value = canonical.parse_canonical(raw)
    except canonical.CanonicalError as exc:
        raise AcquisitionRefusal(
            pb.FaultKind.FAULT_KIND_CONFIG_REFUSED, f"{name}_noncanonical", str(exc)
        ) from exc
    if not isinstance(value, dict):
        raise AcquisitionRefusal(
            pb.FaultKind.FAULT_KIND_CONFIG_REFUSED, f"{name}_invalid", "not an object"
        )
    return value


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


def _manifest_entries(raw: bytes) -> set[tuple[str, int]]:
    manifest = _document(raw, "model_manifest")
    if set(manifest) != {"entries"} or not isinstance(manifest["entries"], list):
        raise AcquisitionRefusal(
            pb.FaultKind.FAULT_KIND_CONFIG_REFUSED,
            "model_manifest_invalid",
            "Manifest has one entries array",
        )
    refs: set[tuple[str, int]] = set()
    for row in manifest["entries"]:
        if not isinstance(row, dict) or set(row) != {"blob", "kind", "path"}:
            raise AcquisitionRefusal(
                pb.FaultKind.FAULT_KIND_CONFIG_REFUSED,
                "model_manifest_invalid",
                "entry fields are closed",
            )
        refs.add(_ref(row["blob"], "manifest.entry.blob"))
    return refs


def _visible_entrypoints(descriptor: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    rows = descriptor.get("entrypoints")
    if not isinstance(rows, list):
        raise AcquisitionRefusal(
            pb.FaultKind.FAULT_KIND_CONFIG_REFUSED,
            "package_descriptor_invalid",
            "entrypoints is not an array",
        )
    visible = {
        str(row["name"]): row
        for row in rows
        if isinstance(row, dict) and isinstance(row.get("name"), str)
    }
    if len(visible) != len(rows) or not visible:
        raise AcquisitionRefusal(
            pb.FaultKind.FAULT_KIND_CONFIG_REFUSED,
            "package_descriptor_invalid",
            "entrypoint names are not unique",
        )
    return visible
