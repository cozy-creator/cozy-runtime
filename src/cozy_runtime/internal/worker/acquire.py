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
from typing import Any, Literal

from cozy_runtime.internal import (
    base_observation,
    canonical,
    fill,
    package_environment,
    placement_materialization,
    pull,
)
from cozy_runtime.internal import (
    descriptor as package_descriptor,
)
from cozy_runtime.internal.refusal import LaunchRefusal
from cozy_runtime.internal.worker.downloads import DownloadPlan
from cozy_runtime.internal.worker.plan import (
    DeclaredBinding,
    ModelBinding,
    model_binding_set_digest,
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


@dataclass(frozen=True, slots=True)
class AcquiredPlacement:
    installed: package_environment.InstalledEnvironment | None
    bindings: Mapping[str, DeclaredBinding]
    model_objects: int
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
    Manifest, header and config. This function compares that derivation with every accepted
    Placement fact before handing the normal attempt kernel its local resolution paths. No
    PackageRelease or Environment is invented for this same-user lane.
    """

    actual = placement.document
    expected = documents.body(expected_placement)
    # Placement ID is RecordOwner routing, not package/model selection identity. A local
    # owner may remint it while preserving the exact development closure.
    comparable_actual = {key: value for key, value in actual.items() if key != "placement_id"}
    comparable_expected = {key: value for key, value in expected.items() if key != "placement_id"}
    if actual.get("package_descriptor") != expected.get("package_descriptor"):
        raise AcquisitionRefusal(
            pb.FaultKind.FAULT_KIND_CONFIG_REFUSED,
            "development_source_changed",
            "the current checkout derives a different PackageDescriptor",
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
            if selected_slot.get("config") != accepted_slot.get("config"):
                raise AcquisitionRefusal(
                    pb.FaultKind.FAULT_KIND_ARTIFACT_DIGEST_MISMATCH,
                    "development_model_config_changed",
                    name,
                )
            if selected_slot.get("components", []) != accepted_slot.get("components", []):
                raise AcquisitionRefusal(
                    pb.FaultKind.FAULT_KIND_CONFIG_REFUSED,
                    "development_model_components_changed",
                    name,
                )
            if selected_slot.get("stamps", []) != accepted_slot.get("stamps", []):
                raise AcquisitionRefusal(
                    pb.FaultKind.FAULT_KIND_CONFIG_REFUSED,
                    "development_model_stamps_changed",
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
    if set(visible) != set(expected_entries):
        raise AcquisitionRefusal(
            pb.FaultKind.FAULT_KIND_CONFIG_REFUSED,
            "binding_entrypoint_set_mismatch",
            "PlacementSet/1 does not cover every descriptor entrypoint",
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
    verified = descriptor_length
    documents_count = 1
    for digest, binding in bindings.items():
        selected = selected_digests[digest]
        slots = selected.get("slots", [])
        if not slots:
            continue
        selected_configs = {
            f"{selected['name']}.models.{slot['slot']}": slot["config"]["document"]
            for slot in slots
        }
        members = binding.model_bindings()
        if set(selected_configs) != {member.model_binding_path for member in members}:
            raise AcquisitionRefusal(
                pb.FaultKind.FAULT_KIND_CONFIG_REFUSED,
                "development_model_slot_changed",
                str(selected["name"]),
            )
        for member in members:
            config = selected_configs[member.model_binding_path]
            _verify_file(Path(member.config), str(config["digest"]), int(config["length"]))
            verified += int(config["length"])
            documents_count += 1
    return AcquiredPlacement(
        None,
        bindings,
        len(placement.models),
        documents_count,
        0,
        0,
        verified,
        0,
    )


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
        base: base_observation.Observation | None = None,
        installed_environment: package_environment.InstalledEnvironment | None = None,
        verified_cache: VerifiedArtifactCache | None = None,
        on_observation: Any = None,
        native_environment: Mapping[str, str] | None = None,
        native_device_index: int = 0,
    ) -> None:
        self.plan = plan
        self.cache_root = cache_root
        self.environment_root = environment_root
        self.selection_root = selection_root
        self.artifact_store = artifact_store
        self.python = python
        self.base = base or package_environment.observe_base(python)
        self.installed_environment = installed_environment
        self.verified_cache = verified_cache or VerifiedArtifactCache()
        self.native_environment = native_environment
        self.native_device_index = native_device_index
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
                "weights_length_conflict",
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
                if planned is None or not location:
                    raise AcquisitionRefusal(
                        pb.FaultKind.FAULT_KIND_ARTIFACT_FETCH_FAILED,
                        "download_plan_missing",
                        f"{digest} is reached by PlacementSet/1 and absent locally and "
                        "from the plan",
                    )
                # The presign's expiry is the ORIGIN's to enforce, and it does (403). It is
                # not a deadline on the bytes: a fetch that is landing bytes when the URL
                # lapses is refused by the store, reported as a fetch fault, and re-planned
                # with fresh access — never cut here by a clock read off the URL.
                path = placement_materialization.fetch_exact(
                    {"digest": digest, "length": length, "url": location},
                    self.cache_root,
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
    ) -> tuple[package_environment.InstalledEnvironment, Mapping[str, DeclaredBinding], int]:
        self._thread.leg = "package"
        self.observation.start("package")
        try:
            installed = self.installed_environment or self._environment(placement)
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
        environment_digest = documents.spell(placement.environment_digest)
        package = placement.package or placement.development
        wheel_rows = [package["project_wheel"], *placement.environment.get("wheels", [])]
        expected_paths: dict[str, Path] = {}
        for row in wheel_rows:
            digest, _length = _ref(row["ref"], "wheel.ref")
            expected_paths[digest] = self.cache_root / digest.removeprefix("sha256:")
        try:
            package_environment.verify_inventory(
                placement_materialization.package_environment_input(
                    package=package,
                    environment=placement.environment,
                    environment_digest=environment_digest,
                    wheel_paths=expected_paths,
                ),
                self.base,
            )
        except (LaunchRefusal, package_environment.EnvironmentRefusal) as exc:
            raise AcquisitionRefusal(
                pb.FaultKind.FAULT_KIND_CONFIG_REFUSED, exc.code, exc.detail
            ) from exc
        wheel_paths: dict[str, Path] = {}
        verified: dict[str, package_environment.VerifiedFile] = {}
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
            source = placement_materialization.package_environment_input(
                package=package,
                environment=placement.environment,
                environment_digest=environment_digest,
                wheel_paths=wheel_paths,
                verified_wheels=verified,
            )
            return package_environment.materialize(
                source,
                self.environment_root,
                python=self.python,
                base=self.base,
                native_environment=self.native_environment,
                native_device_index=self.native_device_index,
            )
        except (LaunchRefusal, package_environment.EnvironmentRefusal) as exc:
            raise AcquisitionRefusal(
                pb.FaultKind.FAULT_KIND_CONFIG_REFUSED, exc.code, exc.detail
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
                    for digest, length in manifest_entries(tensorfs, manifest_raw):
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
                    # NO DEADLINE (xs-007 row 16). The presigns' expiry used to be the pull's
                    # deadline, which cut a 100 GB materialization at the URL lifetime and
                    # called it a credential error. The pull is per-object resumable and
                    # its own ledger ends a stalled stream; an expired presign is refused
                    # by the origin, reported as a fetch fault, and the next plan carries
                    # fresh access for exactly the objects still missing.
                    outcome = pull.pull_granted_manifest(
                        tensorfs,
                        store,
                        manifest_digest,
                        manifest_length,
                        reads,
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

    def _bindings(self, placement: Any) -> tuple[Mapping[str, DeclaredBinding], int]:
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
        visible = _visible_entrypoints(descriptor)
        models = {str(row["id"]): row for row in placement.models}
        development = dict(getattr(placement, "development", {}))
        release_digest = str(
            placement.package.get("release_digest")
            or development.get("private_revision_digest")
            or development.get("source_digest")
            or ""
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
                    "binding_package_descriptor_invalid",
                    f"descriptor has no entrypoint {name!r}",
                )
            slots = entrypoint.get("slots", [])
            described_models = described.get("models", [])
            if not isinstance(described_models, list) or len(slots) != len(described_models):
                raise AcquisitionRefusal(
                    pb.FaultKind.FAULT_KIND_CONFIG_REFUSED, "binding_slot_set_mismatch", name
                )
            application = str(descriptor.get("application", ""))
            if not slots:
                bindings[binding_digest] = DeclaredBinding(
                    entrypoint_binding_digest=binding_digest,
                    entrypoint=name,
                    package_revision_digest=release_digest,
                    release=release_digest,
                    environment_digest=documents.spell(placement.environment_digest),
                    config_digest=canonical.digest({}),
                    model_class="",
                    model_binding_path="",
                    model_parameter_name="",
                    application=application,
                    descriptor_path=str(descriptor_path),
                    development=bool(development),
                )
                continue
            described_by_path = {
                str(row.get("path", "")): row for row in described_models if isinstance(row, dict)
            }
            if len(described_by_path) != len(described_models):
                raise AcquisitionRefusal(
                    pb.FaultKind.FAULT_KIND_CONFIG_REFUSED,
                    "binding_package_descriptor_invalid",
                    f"{name}: duplicate or malformed descriptor model slot",
                )
            members: list[ModelBinding] = []
            for slot in slots:
                path = f"{name}.models.{slot['slot']}"
                declared = described_by_path.get(path)
                if declared is None:
                    raise AcquisitionRefusal(
                        pb.FaultKind.FAULT_KIND_CONFIG_REFUSED,
                        "binding_package_descriptor_invalid",
                        path,
                    )
                config_ref = _ref(slot["config"]["document"], "slot.config.document")
                config_path = self._stage(config_ref[0], self.fetch(*config_ref).read_bytes())
                count += 1
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
                stamps: dict[str, set[str]] = {}
                for row in slot.get("stamps", []):
                    stamps.setdefault(str(row["key"]), set()).update(
                        str(value) for value in row["values"]
                    )
                descriptor_stamps = declared.get("stamps", {})
                if not isinstance(descriptor_stamps, dict) or any(
                    value not in stamps.get(str(key), set())
                    for key, value in descriptor_stamps.items()
                ):
                    raise AcquisitionRefusal(
                        pb.FaultKind.FAULT_KIND_CONFIG_REFUSED,
                        "selected_binding_mismatch",
                        f"{path}: descriptor stamps are not selected",
                    )
                try:
                    store = pull.open_store(fill.tensorfs_module(), self.artifact_store)
                    model_weight_bytes = sum(
                        int(row["length"]) for row in store.walk(reference_manifest)
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
                        config_digest=config_ref[0],
                        store=str(self.artifact_store),
                        config=str(config_path),
                        variant="",
                        reference_snapshot=reference_manifest,
                        components=components,
                        snapshots=snapshots,
                        stamps={
                            key: tuple(sorted(values)) for key, values in sorted(stamps.items())
                        },
                        model_weight_bytes=model_weight_bytes,
                        sequence_parallel_degrees=degrees,
                    )
                )
                for asset in slot["config"].get("assets", []):
                    ref = _ref(asset["ref"], "slot.config.asset")
                    self._stage(ref[0], self.fetch(*ref).read_bytes())
                    count += 1
            members.sort(key=lambda member: member.model_binding_path)
            model_set = tuple(members)
            first = model_set[0]
            bindings[binding_digest] = DeclaredBinding(
                entrypoint_binding_digest=binding_digest,
                entrypoint=name,
                package_revision_digest=release_digest,
                release=release_digest,
                models=model_set,
                environment_digest=documents.spell(placement.environment_digest),
                config_digest=model_binding_set_digest(model_set),
                model_class=first.model_class,
                model_binding_path=first.model_binding_path,
                model_parameter_name=first.model_parameter_name,
                application=application,
                descriptor_path=str(descriptor_path),
                store=first.store,
                config=first.config,
                variant=first.variant,
                reference_snapshot=first.reference_snapshot,
                components=first.components,
                snapshots=first.snapshots,
                stamps=first.stamps,
                model_weight_bytes=sum(member.model_weight_bytes for member in model_set),
                expected_logical_model_digest="",
                development=bool(development),
            )
        selected_entrypoints = {str(row["name"]) for row in placement.entrypoints}
        if (development and not selected_entrypoints) or (
            not development and set(visible) != selected_entrypoints
        ):
            raise AcquisitionRefusal(
                pb.FaultKind.FAULT_KIND_CONFIG_REFUSED,
                "binding_entrypoint_set_mismatch",
                "PlacementSet/1 does not select the required descriptor entrypoints",
            )
        return bindings, count

    def _verify_model_refs(self, bindings: Mapping[str, DeclaredBinding]) -> None:
        if not any(not binding.weightless for binding in bindings.values()):
            return
        try:
            store = pull.open_store(fill.tensorfs_module(), self.artifact_store)
            for binding in bindings.values():
                for model in binding.model_bindings():
                    for digest in {model.reference_snapshot, *model.snapshots.values()}:
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


def manifest_entries(tensorfs: Any, raw: bytes) -> set[tuple[str, int]]:
    """Read exact blob refs through TensorFS's canonical Manifest parser."""

    try:
        manifest = tensorfs.parse_manifest(raw)
        rows = manifest["entries"]
        refs = {
            ("sha256:" + str(row["blob"]["sha256"]), int(row["blob"]["length"])) for row in rows
        }
    except Exception as exc:
        raise AcquisitionRefusal(
            pb.FaultKind.FAULT_KIND_CONFIG_REFUSED,
            "model_manifest_invalid",
            f"TensorFS refused Manifest bytes: {type(exc).__name__}: {exc}",
        ) from exc
    if not rows or len(refs) != len(rows) or any(length <= 0 for _digest, length in refs):
        raise AcquisitionRefusal(
            pb.FaultKind.FAULT_KIND_CONFIG_REFUSED,
            "model_manifest_invalid",
            "TensorFS Manifest refs are not unique and positive",
        )
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
    if len(visible) != len(rows):
        raise AcquisitionRefusal(
            pb.FaultKind.FAULT_KIND_CONFIG_REFUSED,
            "package_descriptor_invalid",
            "entrypoint names are not unique",
        )
    return visible
