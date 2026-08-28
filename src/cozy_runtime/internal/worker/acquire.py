"""cr-049: the ACQUISITION walk — a desired spec plus a standing grant become local bytes.

`converge_placement` used to only VERIFY: it read a receipt-selected generation that something
else had already built and binding-plan records something else had already staged. Nothing
built or staged them, because cr-048 retired the boot closure that did. This module is the
missing door.

**The set is the authority on WHAT; the grant is the authority on ACCESS** (rev-2 §3). The
walk therefore never reads the grant's subject `kind` to decide what a digest MEANS — it
starts at the two digests the `PlacementSpec` names (`environment_spec_digest`,
`installed_environment_receipt_digest`) plus the spec's binding-plan subjects, and every
further digest is REACHED from a document already fetched and verified:

    environment_spec -> endpoint_bundle -> resolved_wheel_set + wheelhouse_manifest -> wheels
    binding_plan     -> resolved_model_binding -> model_construction_contract + model config
    binding_plan     -> descriptor

That reachability is th-044/th-046's canonical join, and it already exists as documents;
`materialize_launch.environment_input` is its reader. What never existed was a caller. This
module is that caller, and it is why `apply_artifact_grant` no longer has to say it cannot
"reinterpret opaque subjects as a wheel closure" — it never needs to interpret one.

Every byte enters through `materialize_launch.fetch_exact`: one bounded exact-digest HTTPS
GET, no proxy, no redirect, no content transform, sha256-and-length before trust. There is
no second fetcher.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock, local
from typing import Any, Literal, TypeAlias

from cozy_runtime.internal import canonical, materialize_launch, overlay, provisioned, pull
from cozy_runtime.protocol import documents
from cozy_runtime.protocol import worker_pb2 as pb

FaultKind: TypeAlias = pb.FaultKind
LegName: TypeAlias = Literal["endpoint", "model"]

#: Documents in the walk are small and bounded; a wheel is capped by `fetch_exact` itself.
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
    """Observation-only acquisition clocks and byte counters.

    No method returns a verdict and callback failures are swallowed deliberately: telemetry
    cannot gate fetch, materialization, acceptance, or convergence.
    """

    def __init__(self, callback: Any = None) -> None:
        self._lock = Lock()
        self._legs = {"endpoint": _Leg(), "model": _Leg()}
        self._callback = callback

    def start(self, leg: LegName) -> None:
        with self._lock:
            row = self._legs[leg]
            row.started = max(1, time.monotonic_ns())
            row.ended = 0
            row.downloaded = 0
            row.reused = 0
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
        try:
            self._callback(snapshot)
        except Exception:
            return


class VerifiedArtifactCache:
    """One worker-boot's stable-file receipts for exact acquisition bytes.

    The table is deliberately memory-only: restart loses every receipt and therefore
    re-hashes cached bytes once.  Within a boot, root-owned cache files may be trusted only
    while their complete stat identity remains unchanged.
    """

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


def observe_model_sources(
    observation: AcquisitionObservation, outcome: materialize_launch.MaterializedObjects
) -> None:
    """Project TensorFS source evidence into the model leg's counters."""

    observation.add(
        "model",
        downloaded=outcome.sources["cloud"]["bytes"],
        reused=(outcome.sources["local"]["bytes"] + outcome.sources["network_cache"]["bytes"]),
    )


@dataclass(frozen=True, slots=True)
class StandingGrant:
    """The recorded acquisition authority — WHICH digests, WHERE, and UNTIL WHEN.

    A COMPLETE replacement of the previous authority (rev-2 §3): a refresh bumps only
    `revision` and touches no placement identity, because the identities live in the set
    document and a grant cannot reach it.
    """

    grant_id: str = ""
    revision: int = 0
    subjects: Mapping[str, int] = field(default_factory=dict)
    locations: Mapping[str, str] = field(default_factory=dict)
    expires_at_unix: int = 0
    base_url: str = ""

    @classmethod
    def read(cls, update: pb.ArtifactGrantUpdate) -> StandingGrant:
        grant = update.grant
        subjects: dict[str, int] = {}
        for subject in grant.subjects:
            spelled = documents.spell(subject.digest)
            if subject.length <= 0:
                raise AcquisitionRefusal(
                    pb.FaultKind.FAULT_KIND_CONFIG_REFUSED,
                    "artifact_subject_invalid",
                    f"{spelled} is authorized with length {subject.length}",
                )
            previous = subjects.setdefault(spelled, subject.length)
            if previous != subject.length:
                raise AcquisitionRefusal(
                    pb.FaultKind.FAULT_KIND_CONFIG_REFUSED,
                    "artifact_subject_conflict",
                    f"{spelled} is authorized at two lengths",
                )
        locations = {documents.spell(location.digest): location.url for location in grant.locations}
        return cls(
            grant_id=grant.grant_id,
            revision=update.grant_revision,
            subjects=subjects,
            locations=locations,
            expires_at_unix=grant.expires_at_unix,
            base_url=grant.base_url,
        )

    def authorized(self) -> bool:
        return bool(self.subjects)

    def local(self) -> bool:
        """`pull.guard`'s dev carve-out, decided by the GRANT'S OWN stated base and nothing
        else: a production grant names a public base, so the private-address door stays
        shut no matter what a location claims."""

        return bool(self.base_url) and pull.local_hub(self.base_url)


@dataclass(frozen=True, slots=True)
class AcquiredPlacement:
    """What one converged acquisition produced, all of it re-measurable on disk."""

    installed: overlay.InstalledOverlay
    object_set: Path
    model_objects: int
    plans: tuple[str, ...]
    documents: tuple[str, ...]
    wheels: int
    fetched_bytes: int
    verified_bytes: int
    trusted_reuse_bytes: int


class Acquirer:
    """One placement's acquisition, against one standing grant.

    Storage is a constructor argument and never a document field: local paths are the
    provisioner's choice and no canonical document may name one.
    """

    def __init__(
        self,
        grant: StandingGrant,
        *,
        cache_root: Path,
        environment_root: Path,
        plans_root: Path,
        documents_root: Path,
        object_sets_root: Path,
        artifact_store: Path,
        python: Path,
        verified_cache: VerifiedArtifactCache | None = None,
        on_observation: Any = None,
    ) -> None:
        self.grant = grant
        self.cache_root = cache_root
        self.environment_root = environment_root
        self.plans_root = plans_root
        self.documents_root = documents_root
        self.object_sets_root = object_sets_root
        self.artifact_store = artifact_store
        self.python = python
        self.verified_cache = verified_cache or VerifiedArtifactCache()
        self.fetched_bytes = 0
        self.verified_bytes = 0
        self.trusted_reuse_bytes = 0
        self.wheels = 0
        self._fetch_lock = Lock()
        self._new_digests: set[str] = set()
        self._metric_digests: set[str] = set()
        self._observed_digests: dict[LegName, set[str]] = {"endpoint": set(), "model": set()}
        self._thread = local()
        self.observation = AcquisitionObservation(on_observation)

    # ---- the one door bytes enter through -------------------------------------------

    def _deadline(self) -> int:
        """The grant's own expiry IS the deadline. No timeout is invented here."""

        if self.grant.expires_at_unix <= 0:
            raise AcquisitionRefusal(
                pb.FaultKind.FAULT_KIND_GRANT_EXPIRED,
                "grant_expiry_absent",
                "an acquisition grant states when it expires or it authorizes nothing",
            )
        if time.time() >= self.grant.expires_at_unix:
            raise AcquisitionRefusal(
                pb.FaultKind.FAULT_KIND_GRANT_EXPIRED,
                "grant_expired",
                f"the standing grant expired at {self.grant.expires_at_unix}",
            )
        return self.grant.expires_at_unix

    def fetch(self, digest: str, length: int, *, observe: bool = True) -> Path:
        """Fetch one AUTHORIZED digest. Authority is checked before a socket is opened."""

        authorized = self.grant.subjects.get(digest)
        if authorized is None:
            raise AcquisitionRefusal(
                pb.FaultKind.FAULT_KIND_GRANT_EXPIRED,
                "artifact_unauthorized",
                f"{digest} is reached by the desired spec but no standing grant names it",
            )
        if authorized != length:
            raise AcquisitionRefusal(
                pb.FaultKind.FAULT_KIND_ARTIFACT_DIGEST_MISMATCH,
                "artifact_length_conflict",
                f"{digest} is named at {length} B and authorized at {authorized} B",
            )
        url = self.grant.locations.get(digest)
        if not url:
            raise AcquisitionRefusal(
                pb.FaultKind.FAULT_KIND_ARTIFACT_FETCH_FAILED,
                "artifact_location_absent",
                f"{digest} is authorized but the grant carries no location for it",
            )
        deadline = self._deadline()
        target = self.cache_root / digest.removeprefix("sha256:")
        already = target.exists()
        trusted = self.verified_cache.trusted(digest, length, target)
        trusted_before = trusted is not None
        try:
            if trusted is not None:
                path = target
            else:
                path = materialize_launch.fetch_exact(
                    {"digest": digest, "length": length, "url": url},
                    self.cache_root,
                    deadline=deadline,
                    allow_local=self.grant.local(),
                )
                trusted = self.verified_cache.record(digest, length, path)
        except materialize_launch.LaunchRefusal as exc:
            raise AcquisitionRefusal(
                pb.FaultKind.FAULT_KIND_ARTIFACT_DIGEST_MISMATCH
                if exc.code
                in {
                    "grant_identity_mismatch",
                    "grant_length_mismatch",
                    "local_artifact_digest_mismatch",
                    "local_artifact_invalid",
                }
                else pb.FaultKind.FAULT_KIND_ARTIFACT_FETCH_FAILED,
                exc.code,
                exc.detail,
            ) from exc
        with self._fetch_lock:
            # Content-addressed resume: a complete object is never re-fetched, so the
            # measured cost of a replay is the objects that were still missing (§1
            # Recovery). Only new bytes are counted.
            if not already and digest not in self._new_digests and target == path:
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
                observed = self._observed_digests[leg]
                if digest not in observed:
                    observed.add(digest)
                    self.observation.add(
                        leg,
                        downloaded=0 if already else length,
                        reused=length if already else 0,
                    )
        return path

    # ---- the walk ---------------------------------------------------------------------

    def acquire(
        self,
        placement: Any,
        *,
        on_endpoint_ready: Callable[[overlay.InstalledOverlay], None] | None = None,
    ) -> AcquiredPlacement:
        """Fetch, build, and stage everything one PlacementSpec reaches. Fail closed."""

        # The endpoint closure and model-object closure are the two observed legs. They
        # start together. Binding/model documents are small endpoint control bytes and ride
        # the endpoint leg; model-object-set document bytes are identity and excluded from
        # the model byte counter by protocol definition.
        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="cozy-acquire") as pool:
            endpoint_future = pool.submit(self._run_endpoint, placement)
            model_future = pool.submit(self._run_model, placement)
            installed, plans, staged = endpoint_future.result()
            if on_endpoint_ready is not None:
                on_endpoint_ready(installed)
            object_set, model_objects = model_future.result()
        return AcquiredPlacement(
            installed=installed,
            object_set=object_set,
            model_objects=model_objects,
            plans=plans,
            documents=staged,
            wheels=self.wheels,
            fetched_bytes=self.fetched_bytes,
            verified_bytes=self.verified_bytes,
            trusted_reuse_bytes=self.trusted_reuse_bytes,
        )

    def _run_endpoint(
        self, placement: Any
    ) -> tuple[overlay.InstalledOverlay, tuple[str, ...], tuple[str, ...]]:
        self._thread.leg = "endpoint"
        self.observation.start("endpoint")
        try:
            installed = self._environment(placement)
            plans, staged = self._bindings(placement)
            return installed, plans, staged
        finally:
            self.observation.finish("endpoint")

    def _run_model(self, placement: Any) -> tuple[Path, int]:
        self._thread.leg = "model"
        self.observation.start("model")
        try:
            return self._models(placement)
        finally:
            self.observation.finish("model")

    def _models(self, placement: Any) -> tuple[Path, int]:
        """Fetch the exact object-set document, then hydrate every model root into TensorFS."""

        digest, length = placement.model_object_set
        if not digest or length <= 0:
            raise AcquisitionRefusal(
                pb.FaultKind.FAULT_KIND_CONFIG_REFUSED,
                "model_object_set_absent",
                "PlacementSpec/2 names one exact model_object_set subject",
            )
        raw = self.fetch(digest, length, observe=False).read_bytes()
        self.object_sets_root.mkdir(parents=True, exist_ok=True, mode=0o755)
        self.object_sets_root.chmod(0o755)
        try:
            object_set = provisioned.read_artifact_object_set(raw)
        except Exception as exc:
            raise AcquisitionRefusal(
                pb.FaultKind.FAULT_KIND_CONFIG_REFUSED,
                "model_object_set_invalid",
                str(exc),
            ) from exc
        locations: dict[str, str] = {}
        for object_digest, object_length in object_set.refs.items():
            authorized = self.grant.subjects.get(object_digest)
            if authorized != object_length:
                raise AcquisitionRefusal(
                    pb.FaultKind.FAULT_KIND_GRANT_EXPIRED,
                    "model_object_unauthorized",
                    f"{object_digest} is required at {object_length} B and authorized at "
                    f"{authorized!r}",
                )
            location = self.grant.locations.get(object_digest)
            if not location:
                raise AcquisitionRefusal(
                    pb.FaultKind.FAULT_KIND_ARTIFACT_FETCH_FAILED,
                    "model_object_location_absent",
                    object_digest,
                )
            locations[object_digest] = location
        try:
            outcome = materialize_launch.materialize_objects(
                locations,
                object_set,
                cache=None,
                deadline=self._deadline(),
                store_root=self.artifact_store,
                allow_local=self.grant.local(),
            )
        except materialize_launch.LaunchRefusal as exc:
            raise AcquisitionRefusal(
                pb.FaultKind.FAULT_KIND_ARTIFACT_FETCH_FAILED,
                exc.code,
                exc.detail,
            ) from exc
        observe_model_sources(self.observation, outcome)
        self._stage(self.object_sets_root, digest, raw)
        return self.object_sets_root / _name(digest), len(object_set.refs)

    def _environment(self, placement: Any) -> overlay.InstalledOverlay:
        """Environment spec -> source/custom wheels -> one small installed-base overlay."""

        spec_digest = documents.spell(placement.environment_spec_digest)
        receipt_digest = documents.spell(placement.installed_environment_receipt_digest)
        if not spec_digest or not receipt_digest:
            raise AcquisitionRefusal(
                pb.FaultKind.FAULT_KIND_CONFIG_REFUSED,
                "environment_identity_absent",
                "a PlacementSpec names its environment and its expected receipt, or it is "
                "not one this worker can materialize",
            )
        spec_length = self._authorized_length(spec_digest, "environment_spec")
        spec_bytes = self.fetch(spec_digest, spec_length).read_bytes()
        spec = self._parse(spec_bytes, "environment_spec")
        bundle_ref = self._digest_ref(spec, "endpoint_bundle_digest", "environment_spec")
        wheelhouse_ref = self._digest_ref(spec, "wheelhouse_manifest_digest", "environment_spec")
        bundle_bytes = self.fetch(
            bundle_ref, self._authorized_length(bundle_ref, "endpoint_bundle")
        ).read_bytes()
        bundle = self._parse(bundle_bytes, "endpoint_bundle")
        # Archive and lock prove how the project wheel was published; they are
        # not executable placement bytes. Validate their exact references but
        # fetch only the project/custom wheels the worker can actually run.
        self._ref(bundle.get("source_archive"), "source_archive")
        self._ref(bundle.get("source_lock"), "source_lock")
        resolved_ref = self._ref(bundle.get("resolved_wheel_set"), "resolved_wheel_set")
        wheelhouse_bundle_ref = self._ref(bundle.get("wheelhouse_manifest"), "wheelhouse_manifest")
        if wheelhouse_bundle_ref[0] != wheelhouse_ref:
            raise AcquisitionRefusal(
                pb.FaultKind.FAULT_KIND_CONFIG_REFUSED,
                "wheelhouse_manifest_ref_mismatch",
                "the endpoint bundle and the environment spec name different wheelhouses",
            )
        resolved_bytes = self.fetch(*resolved_ref).read_bytes()
        wheelhouse_bytes = self.fetch(*wheelhouse_bundle_ref).read_bytes()
        resolved = self._parse(resolved_bytes, "resolved_wheel_set")
        self._parse(wheelhouse_bytes, "wheelhouse_manifest")

        wheel_paths: dict[str, Path] = {}
        verified_wheels: dict[str, overlay.VerifiedFile] = {}
        for row in (
            bundle.get("project_wheel"),
            *self._rows(resolved.get("wheels"), "resolved_wheel_set"),
        ):
            digest, length = self._wheel_ref(row)
            wheel_paths[digest] = self.fetch(digest, length)
            verified = self.verified_cache.trusted(digest, length, wheel_paths[digest])
            if verified is None:
                raise AcquisitionRefusal(
                    pb.FaultKind.FAULT_KIND_ARTIFACT_DIGEST_MISMATCH,
                    "wheel_verification_receipt_absent",
                    digest,
                )
            verified_wheels[digest] = verified
        self.wheels = len(wheel_paths)

        try:
            source = materialize_launch.endpoint_overlay_input(
                environment_spec_bytes=spec_bytes,
                endpoint_bundle_bytes=bundle_bytes,
                resolved_wheel_set_bytes=resolved_bytes,
                wheelhouse_manifest_bytes=wheelhouse_bytes,
                wheel_paths=wheel_paths,
                verified_wheels=verified_wheels,
            )
            return overlay.materialize(
                source,
                self.environment_root,
                python=self.python,
                expected_receipt_digest=receipt_digest,
            )
        except (materialize_launch.LaunchRefusal, overlay.OverlayRefusal) as exc:
            raise AcquisitionRefusal(
                pb.FaultKind.FAULT_KIND_ENVIRONMENT_RECEIPT_MISMATCH
                if "receipt" in exc.code
                else pb.FaultKind.FAULT_KIND_CONFIG_REFUSED,
                exc.code,
                exc.detail,
            ) from exc

    def _bindings(self, placement: Any) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """Stage the plan records and every document those plans REACH."""

        self.plans_root.mkdir(parents=True, exist_ok=True, mode=0o755)
        self.documents_root.mkdir(parents=True, exist_ok=True, mode=0o755)
        self.plans_root.chmod(0o755)
        self.documents_root.chmod(0o755)
        staged: list[str] = []
        descriptor_digest = documents.spell(placement.descriptor_digest)
        for plan_id in placement.plan_ids:
            plan_digest, plan_length = placement.plan_refs[plan_id]
            plan_path = self.fetch(plan_digest, plan_length)
            self._stage(self.plans_root, plan_digest, plan_path.read_bytes())
            plan = self._parse(plan_path.read_bytes(), "entrypoint_binding_plan")
            plan_descriptor = self._ref(plan.get("descriptor"), "plan.descriptor")
            if plan_descriptor[0] != descriptor_digest:
                raise AcquisitionRefusal(
                    pb.FaultKind.FAULT_KIND_CONFIG_REFUSED,
                    "plan_descriptor_mismatch",
                    f"{plan_id} names descriptor {plan_descriptor[0]}, the spec names "
                    f"{descriptor_digest}",
                )
            staged.append(self._document(*plan_descriptor, "descriptor"))
            for binding in self._rows(plan.get("bindings"), "entrypoint_binding_plan"):
                if not isinstance(binding, dict):
                    raise AcquisitionRefusal(
                        pb.FaultKind.FAULT_KIND_CONFIG_REFUSED,
                        "plan_binding_invalid",
                        "a plan binding is a JSON object",
                    )
                binding_ref = self._ref(binding.get("binding"), "plan.binding")
                staged.append(self._document(*binding_ref, "resolved_model_binding"))
                model = self._parse(
                    (self.documents_root / _name(binding_ref[0])).read_bytes(),
                    "resolved_model_binding",
                )
                config = model.get("config")
                if not isinstance(config, dict):
                    raise AcquisitionRefusal(
                        pb.FaultKind.FAULT_KIND_CONFIG_REFUSED,
                        "resolved_model_binding_invalid",
                        "config is a JSON object",
                    )
                contract = self._ref(
                    model.get("model_construction_contract"), "model_construction_contract"
                )
                staged.append(self._document(*contract, "model_construction_contract"))
                document = self._ref(config.get("document"), "model_config")
                staged.append(self._document(*document, "model_config"))
        return tuple(placement.plan_ids), tuple(sorted(set(staged)))

    # ---- staging and parsing ------------------------------------------------------------

    def _document(self, digest: str, length: int, name: str) -> str:
        self._stage(self.documents_root, digest, self.fetch(digest, length).read_bytes())
        return digest

    def _stage(self, root: Path, digest: str, raw: bytes) -> None:
        """Publish exact bytes once under their own digest; a replay converges."""

        target = root / _name(digest)
        if target.exists():
            if target.read_bytes() != raw:
                raise AcquisitionRefusal(
                    pb.FaultKind.FAULT_KIND_ARTIFACT_DIGEST_MISMATCH,
                    "staged_document_conflict",
                    f"{target} already holds different bytes",
                )
            return
        temporary = root / f".stage-{digest.removeprefix('sha256:')[:16]}"
        temporary.write_bytes(raw)
        temporary.chmod(0o444)
        temporary.replace(target)

    def _authorized_length(self, digest: str, name: str) -> int:
        length = self.grant.subjects.get(digest)
        if length is None:
            raise AcquisitionRefusal(
                pb.FaultKind.FAULT_KIND_GRANT_EXPIRED,
                "artifact_unauthorized",
                f"{name} {digest} is named by the desired spec and by no standing grant",
            )
        return length

    def _parse(self, raw: bytes, name: str) -> dict[str, Any]:
        try:
            value = canonical.parse_canonical(raw)
        except canonical.CanonicalError as exc:
            raise AcquisitionRefusal(
                pb.FaultKind.FAULT_KIND_CONFIG_REFUSED, f"{name}_noncanonical", str(exc)
            ) from exc
        if not isinstance(value, dict):
            raise AcquisitionRefusal(
                pb.FaultKind.FAULT_KIND_CONFIG_REFUSED,
                f"{name}_invalid",
                "a canonical document is a JSON object",
            )
        return value

    def _digest_ref(self, document: Mapping[str, Any], key: str, name: str) -> str:
        value = document.get(key)
        if not isinstance(value, str) or not value.startswith("sha256:"):
            raise AcquisitionRefusal(
                pb.FaultKind.FAULT_KIND_CONFIG_REFUSED,
                f"{name}_invalid",
                f"{key} is not `sha256:<hex>`",
            )
        return value

    def _ref(self, value: object, name: str) -> tuple[str, int]:
        if (
            not isinstance(value, dict)
            or set(value) != {"digest", "length"}
            or not isinstance(value["digest"], str)
            or not value["digest"].startswith("sha256:")
            or not isinstance(value["length"], int)
            or isinstance(value["length"], bool)
            or value["length"] <= 0
        ):
            raise AcquisitionRefusal(
                pb.FaultKind.FAULT_KIND_CONFIG_REFUSED,
                f"{name}_ref_invalid",
                "an artifact reference is exactly {digest, length}",
            )
        return str(value["digest"]), int(value["length"])

    def _rows(self, value: object, name: str) -> list[Any]:
        if not isinstance(value, list):
            raise AcquisitionRefusal(
                pb.FaultKind.FAULT_KIND_CONFIG_REFUSED,
                f"{name}_invalid",
                "the inventory is not a list",
            )
        return list(value)

    def _wheel_ref(self, row: object) -> tuple[str, int]:
        if (
            not isinstance(row, dict)
            or not isinstance(row.get("digest"), str)
            or not str(row["digest"]).startswith("sha256:")
            or not isinstance(row.get("length"), int)
            or isinstance(row["length"], bool)
            or row["length"] <= 0
        ):
            raise AcquisitionRefusal(
                pb.FaultKind.FAULT_KIND_CONFIG_REFUSED,
                "wheel_inventory_invalid",
                "a wheel inventory row names its exact digest and length",
            )
        return str(row["digest"]), int(row["length"])


def _name(digest: str) -> str:
    return f"{digest.removeprefix('sha256:')}.json"
