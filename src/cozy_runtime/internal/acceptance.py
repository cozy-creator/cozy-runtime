"""Bank one profile-admission sample without creating a second executor path.

Tensorhub dispatches the sample through the ordinary Worker protocol.  This module does not
launch, prepare, fill, invoke, encode, upload, or decide admission.  It derives the one
deterministic descriptor fixture and, after the ordinary AttemptAccepted/AttemptOutcome exchange,
joins the exact provisioned closure into the execution arm of one bounded ``Qualification``.

The RecordOwner remains responsible for aborting qualification outputs and for authoring the
``ModelAdmissionDecision``.  Runtime cannot observe either act and does not echo them as facts.
"""

from __future__ import annotations

import base64
import hashlib
import os
import re
import stat
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from google.protobuf.json_format import ParseDict, ParseError

from cozy_runtime.internal import canonical, descriptor, overlay, provisioned, qualification, warm
from cozy_runtime.internal.placement_materialization import package_overlay_input
from cozy_runtime.internal.refusal import LaunchRefusal
from cozy_runtime.internal.worker.plan import PlanRefusal
from cozy_runtime.internal.worker.session import read_placement_set
from cozy_runtime.protocol import documents
from cozy_runtime.protocol import worker_pb2 as pb

MAX_OBSERVATION_BYTES = 1 << 20
MAX_STRING = 256

_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]{0,127}$")
_MIME = re.compile(r"^[a-z0-9][a-z0-9.+-]{0,63}/[a-z0-9][a-z0-9.+-]{0,127}$")
_BINDING_KINDS = frozenset(
    {
        "entrypoint_binding_plan",
        "resolved_model_binding",
        "model_construction_contract",
        "model_config",
    }
)
_FIXTURE_FIELDS = frozenset({"package_descriptor", "entrypoint", "input_root"})
_OBSERVATION_FIELDS = frozenset(
    {
        "accelerator_qualification",
        "artifact_object_set",
        "artifact_store",
        "attempt_accepted",
        "attempt_outcome",
        "binding_documents",
        "control_runtime_digest",
        "package_binding_release",
        "package_bundle",
        "package_descriptor",
        "package_environment_spec",
        "package_execution",
        "entrypoint",
        "environment_root",
        "fixture_payload",
        "invocation_spec",
        "installed_environment_receipt",
        "input_root",
        "observed_placement_set_digest",
        "placement_set",
        "resolved_wheel_set",
        "wheelhouse_manifest",
        "worker_resources",
    }
)


class AcceptanceRefusal(Exception):
    """A stable structural refusal for the qualification controller."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail[:4096]


@dataclass(frozen=True, slots=True)
class LocalDocument:
    digest: str
    length: int
    path: Path
    raw: bytes

    def ref(self) -> provisioned.ArtifactRef:
        return provisioned.ArtifactRef(self.digest, self.length)

    def exact(self) -> provisioned.ExactDocument:
        return provisioned.ExactDocument(self.ref(), self.path, self.raw)


@dataclass(frozen=True, slots=True)
class _Environment:
    bundle: Mapping[str, Any]
    profile: str
    control_runtime_digest: str
    environment_spec_digest: str
    installed: overlay.InstalledOverlay


def fixture_request(raw: bytes) -> bytes:
    """Derive the exact neutral payload Tensorhub dispatches as an ordinary attempt."""

    request = _request(raw, _FIXTURE_FIELDS)
    input_root = _directory(request["input_root"], "input_root")
    descriptor_doc = _local_document(
        request["package_descriptor"], "package_descriptor", input_root
    )
    entrypoint = _identifier(request["entrypoint"], "entrypoint")
    return _fixture_payload(descriptor_doc, entrypoint)


def execution_observation(raw: bytes) -> bytes:
    """Validate one ordinary attempt and author its canonical Runtime observation."""

    request = _request(raw, _OBSERVATION_FIELDS)
    entrypoint = _identifier(request["entrypoint"], "entrypoint")
    input_root = _directory(request["input_root"], "input_root")
    documents_by_name = {
        name: _local_document(request[name], name, input_root)
        for name in (
            "artifact_object_set",
            "attempt_outcome",
            "package_binding_release",
            "package_bundle",
            "package_descriptor",
            "package_environment_spec",
            "package_execution",
            "fixture_payload",
            "invocation_spec",
            "installed_environment_receipt",
            "placement_set",
            "resolved_wheel_set",
            "wheelhouse_manifest",
        )
    }
    environment_root = _directory(request["environment_root"], "environment_root")
    artifact_store = _directory(request["artifact_store"], "artifact_store")
    environment = _environment(documents_by_name, environment_root)
    package_release_id = _package_execution(documents_by_name, environment, entrypoint)
    selected, model_contracts, binding_release = _binding_release(
        request,
        documents_by_name,
        package_release_id,
        artifact_store,
        entrypoint,
        input_root,
    )
    placement_id, placement_set_digest = _placement(
        documents_by_name,
        binding_release,
        package_release_id,
        selected.entrypoint_binding_plan_id,
    )
    observed_set = _digest(
        request["observed_placement_set_digest"], "observed_placement_set_digest"
    )
    if observed_set != placement_set_digest:
        raise AcceptanceRefusal(
            "acceptance_placement_observation_mismatch",
            f"worker observed {observed_set}, exact PlacementSet is {placement_set_digest}",
        )

    fixture = documents_by_name["fixture_payload"]
    expected_fixture = _fixture_payload(documents_by_name["package_descriptor"], entrypoint)
    if fixture.raw != expected_fixture:
        raise AcceptanceRefusal(
            "acceptance_fixture_mismatch", "dispatched payload is not the descriptor fixture"
        )
    invocation = _invocation(
        documents_by_name["invocation_spec"],
        environment,
        package_release_id,
        selected.entrypoint_binding_plan_id,
        fixture,
    )
    accepted = _attempt_accepted(
        request["attempt_accepted"],
        documents_by_name["invocation_spec"].digest,
        placement_id,
    )
    logical_model_digest = str(selected.expected_logical_model_digest)
    if accepted["model_construction_digest"] != logical_model_digest:
        raise AcceptanceRefusal(
            "acceptance_model_construction_mismatch",
            "AttemptAccepted names a logical model other than the exact binding plan",
        )
    outcome = _outcome(
        documents_by_name["attempt_outcome"],
        accepted,
        documents_by_name["invocation_spec"].digest,
    )
    metrics = _protocol_projection(outcome.get("metrics", {}), pb.AttemptMetrics, "attempt_metrics")
    # Tensorhub supplies these three projections from its authenticated Worker session.
    # This offline helper validates their existing protocol shape and exact joins; it does
    # not and cannot authenticate the transport they arrived on.
    resources = _protocol_projection(
        request["worker_resources"], pb.WorkerResources, "worker_resources"
    )
    accelerator = _protocol_projection(
        request["accelerator_qualification"],
        pb.AcceleratorQualification,
        "accelerator_qualification",
    )
    output_manifest = _output_manifest(outcome.get("output_manifest"))
    result_digest, result_schema_digest = _result(outcome.get("result"))
    if result_digest:
        expected_schema = _result_schema_digest(documents_by_name["package_descriptor"], entrypoint)
        if result_schema_digest != expected_schema:
            raise AcceptanceRefusal(
                "attempt_result_schema_mismatch",
                f"terminal={result_schema_digest} descriptor={expected_schema}",
            )
    observed_control_runtime = _digest(request["control_runtime_digest"], "control_runtime_digest")
    if observed_control_runtime != environment.control_runtime_digest:
        raise AcceptanceRefusal(
            "control_runtime_identity_mismatch",
            f"ClaimAck observed {observed_control_runtime}, WHM owns "
            f"{environment.control_runtime_digest}",
        )

    observation = {
        "accelerator_qualification": accelerator,
        "attempt_metrics": metrics,
        "attempt_outcome_digest": documents_by_name["attempt_outcome"].digest,
        "attempt_plan_digest": accepted["plan_digest"],
        "base_worker_profile": environment.profile,
        # This is a Runtime invariant, not an echoed caller field: ordinary AttemptOffer
        # reaches Executor.invoke with boot_warmup=False. The warm command is a distinct
        # pre-ready path and cannot produce an AttemptAccepted/AttemptOutcome pair.
        "boot_warmup": False,
        "binding_plan_digest": selected.entrypoint_binding_plan_id,
        "control_runtime_digest": environment.control_runtime_digest,
        "package_descriptor_digest": documents_by_name["package_descriptor"].digest,
        "package_binding_release_digest": documents_by_name["package_binding_release"].digest,
        "package_execution_digest": documents_by_name["package_execution"].digest,
        "package_release_id": package_release_id,
        "entrypoint": entrypoint,
        "environment_spec_digest": documents_by_name["package_environment_spec"].digest,
        "execution_started": bool(outcome.get("execution_started", False)),
        "installed_environment_receipt_digest": environment.installed.receipt_digest,
        "invocation_spec_digest": documents_by_name["invocation_spec"].digest,
        "logical_model_digest": logical_model_digest,
        "model_construction_contract_digests": model_contracts,
        "object_set_digest": documents_by_name["artifact_object_set"].digest,
        "output_manifest": output_manifest,
        "overlay_content_digest": environment.installed.content_digest,
        "payload_digest": str(invocation["payload_digest"]),
        "placement_set_digest": placement_set_digest,
        "result_digest": result_digest,
        "result_schema_digest": result_schema_digest,
        "status": int(outcome.get("status", 0)),
        "worker_resources": resources,
    }
    encoded = canonical.write(qualification.document("execution", observation))
    if len(encoded) > MAX_OBSERVATION_BYTES:
        raise AcceptanceRefusal(
            "execution_observation_oversized",
            f"{len(encoded)} bytes exceeds {MAX_OBSERVATION_BYTES}",
        )
    return encoded


def _request(raw: bytes, fields: frozenset[str]) -> dict[str, Any]:
    try:
        value = canonical.parse(raw)
    except canonical.CanonicalError as exc:
        raise AcceptanceRefusal("acceptance_request_invalid", str(exc)) from exc
    if not isinstance(value, dict) or set(value) != fields:
        got = sorted(value) if isinstance(value, dict) else type(value).__name__
        raise AcceptanceRefusal(
            "acceptance_request_invalid", f"fields={got} expected={sorted(fields)}"
        )
    return value


def _local_document(value: object, name: str, input_root: Path) -> LocalDocument:
    if not isinstance(value, dict) or set(value) != {"digest", "length", "path"}:
        raise AcceptanceRefusal("acceptance_request_invalid", f"{name} is not an exact local ref")
    digest = _digest(value.get("digest"), f"{name}.digest")
    length = value.get("length")
    if (
        not isinstance(length, int)
        or isinstance(length, bool)
        or not 0 < length <= canonical.DOC_MAX_BYTES
    ):
        raise AcceptanceRefusal("acceptance_request_invalid", f"{name}.length={length!r}")
    path = _regular_file(value.get("path"), f"{name}.path")
    if not path.is_relative_to(input_root):
        raise AcceptanceRefusal(
            "acceptance_path_outside_input_root", f"{name}.path={path} root={input_root}"
        )
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise AcceptanceRefusal("acceptance_document_unreadable", f"{name}: {exc}") from exc
    got = "sha256:" + hashlib.sha256(raw).hexdigest()
    if len(raw) != length or got != digest:
        raise AcceptanceRefusal(
            "acceptance_document_identity_mismatch",
            f"{name}: measured {got}/{len(raw)}, expected {digest}/{length}",
        )
    return LocalDocument(digest, length, path, raw)


def _environment(documents_by_name: Mapping[str, LocalDocument], root: Path) -> _Environment:
    receipt = _canonical_object(
        documents_by_name["package_binding_release"].raw, "package_binding_release"
    ).get("installed_environment_receipt")
    receipt_ref = _embedded_ref(receipt, "package_binding_release.installed_environment_receipt")
    receipt_doc = documents_by_name["installed_environment_receipt"]
    if receipt_doc.ref() != receipt_ref:
        raise AcceptanceRefusal(
            "installed_environment_receipt_mismatch", "binding release names another receipt"
        )
    try:
        receipt_value = overlay.parse_environment_receipt(receipt_doc.raw)
    except overlay.OverlayRefusal as exc:
        raise AcceptanceRefusal("installed_environment_receipt_invalid", exc.detail) from exc
    rows = receipt_value["overlay_wheels"]
    wheel_paths: dict[str, Path] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or not isinstance(row.get("digest"), str):
            raise AcceptanceRefusal(
                "installed_environment_receipt_invalid", f"overlay_wheels[{index}]"
            )
        wheel_paths[_digest(row["digest"], f"overlay_wheels[{index}].digest")] = Path("/unread")
    try:
        source = package_overlay_input(
            environment_spec_bytes=documents_by_name["package_environment_spec"].raw,
            package_bundle_bytes=documents_by_name["package_bundle"].raw,
            resolved_wheel_set_bytes=documents_by_name["resolved_wheel_set"].raw,
            wheelhouse_manifest_bytes=documents_by_name["wheelhouse_manifest"].raw,
            wheel_paths=wheel_paths,
        )
    except LaunchRefusal as exc:
        raise AcceptanceRefusal(exc.code, exc.detail) from exc
    if source.environment_spec_digest != documents_by_name["package_environment_spec"].digest:
        raise AcceptanceRefusal(
            "environment_spec_identity_mismatch", source.environment_spec_digest
        )
    if overlay.expected_receipt_bytes(source) != receipt_doc.raw:
        raise AcceptanceRefusal(
            "installed_environment_receipt_mismatch",
            "receipt bytes are not implied by EES/PackageBundle/RWS/WHM",
        )
    try:
        installed = overlay.open_generation(
            root,
            receipt_digest=receipt_ref.digest,
            environment_spec_digest=source.environment_spec_digest,
            base_manifest=documents_by_name["wheelhouse_manifest"].path,
            python=Path(sys.executable),
        )
    except overlay.OverlayRefusal as exc:
        raise AcceptanceRefusal(exc.code, exc.detail) from exc
    spec = _canonical_object(documents_by_name["package_environment_spec"].raw, "environment_spec")
    profile = spec.get("compatibility_profile")
    if not isinstance(profile, dict):
        raise AcceptanceRefusal("environment_spec_invalid", "compatibility_profile is absent")
    try:
        profile_name = "-".join(
            (
                "torch" + str(profile["torch_release"]),
                str(profile["accelerator_build"]),
                str(profile["python_abi"]),
                str(profile["os_cpu"]),
            )
        )
    except KeyError as exc:
        raise AcceptanceRefusal("environment_spec_invalid", "compatibility_profile fields") from exc
    manifest = _canonical_object(
        documents_by_name["wheelhouse_manifest"].raw, "wheelhouse_manifest"
    )
    base_wheels = manifest.get("wheels")
    if not isinstance(base_wheels, list):
        raise AcceptanceRefusal("control_runtime_identity_invalid", "WHM wheels is not an array")
    runtime_rows = [
        row
        for row in base_wheels
        if isinstance(row, dict) and row.get("distribution") == "cozy-runtime"
    ]
    if len(runtime_rows) != 1:
        raise AcceptanceRefusal(
            "control_runtime_identity_invalid", f"WHM names {len(runtime_rows)} cozy-runtime rows"
        )
    control_runtime_digest = _digest(
        runtime_rows[0].get("digest"), "wheelhouse_manifest.cozy-runtime.digest"
    )
    bundle = _canonical_object(documents_by_name["package_bundle"].raw, "package_bundle")
    return _Environment(
        bundle,
        profile_name,
        control_runtime_digest,
        source.environment_spec_digest,
        installed,
    )


def _package_execution(
    docs: Mapping[str, LocalDocument], environment: _Environment, entrypoint: str
) -> str:
    execution = _canonical_object(docs["package_execution"].raw, "package_execution")
    if (
        set(execution)
        != {
            "canonical_execution",
            "compatible_accelerator_models",
            "function",
            "kind",
            "model_roots",
        }
        or execution.get("kind") != "tensorhub.package_execution/2"
    ):
        raise AcceptanceRefusal("package_execution_invalid", "wrong fields or kind")
    if execution.get("function") != entrypoint:
        raise AcceptanceRefusal("package_execution_invalid", "function differs from entrypoint")
    model_roots = execution.get("model_roots")
    if not isinstance(model_roots, list):
        raise AcceptanceRefusal("package_execution_invalid", "model_roots is not an array")
    prior_root = ""
    for index, root in enumerate(model_roots):
        if not isinstance(root, dict) or set(root) != {"checkpoint_id", "name", "org"}:
            raise AcceptanceRefusal("package_execution_invalid", f"model_roots[{index}]")
        key = "\0".join(str(root.get(field, "")) for field in ("org", "name", "checkpoint_id"))
        if not all(isinstance(root.get(field), str) and root[field] for field in ("org", "name")):
            raise AcceptanceRefusal("package_execution_invalid", f"model_roots[{index}]")
        _digest(root.get("checkpoint_id"), f"model_roots[{index}].checkpoint_id")
        if key <= prior_root:
            raise AcceptanceRefusal("package_execution_invalid", "model_roots not sorted unique")
        prior_root = key
    object_set = provisioned.read_artifact_object_set(docs["artifact_object_set"].raw)
    if bool(model_roots) != bool(object_set.roots):
        raise AcceptanceRefusal(
            "package_execution_model_closure_mismatch",
            "model_roots and exact ArtifactObjectSet disagree about weightless execution",
        )
    canonical_execution = execution.get("canonical_execution")
    expected = {
        "build_record_digest",
        "package_descriptor_digest",
        "package_control_manifest_digest",
        "environment_spec_digest",
        "installed_environment_receipt_digest",
    }
    if not isinstance(canonical_execution, dict) or set(canonical_execution) != expected:
        raise AcceptanceRefusal("package_execution_invalid", "canonical_execution is not closed")
    joins = {
        "package_descriptor_digest": docs["package_descriptor"].digest,
        "environment_spec_digest": docs["package_environment_spec"].digest,
        "installed_environment_receipt_digest": environment.installed.receipt_digest,
    }
    for field, wanted in joins.items():
        if canonical_execution.get(field) != wanted:
            raise AcceptanceRefusal("package_execution_ref_mismatch", field)
    release = environment.bundle.get("package_release_id")
    if not isinstance(release, str) or not release or len(release) > MAX_STRING:
        raise AcceptanceRefusal("package_bundle_invalid", "package_release_id")
    descriptor_ref = _embedded_ref(
        environment.bundle.get("package_descriptor"), "bundle.package_descriptor"
    )
    if descriptor_ref != docs["package_descriptor"].ref():
        raise AcceptanceRefusal("package_bundle_ref_mismatch", "package_descriptor")
    return release


def _binding_release(
    request: Mapping[str, Any],
    docs: Mapping[str, LocalDocument],
    package_release_id: str,
    artifact_store: Path,
    entrypoint: str,
    input_root: Path,
) -> tuple[Any, list[str], Mapping[str, Any]]:
    release = _canonical_object(docs["package_binding_release"].raw, "package_binding_release")
    expected = {
        "package_descriptor",
        "documents",
        "package_execution_digest",
        "environment_spec",
        "format",
        "installed_environment_receipt",
        "object_set_digest",
    }
    if set(release) != expected or release.get("format") != "tensorhub.package_binding_release/1":
        raise AcceptanceRefusal("package_binding_release_invalid", "wrong fields or format")
    if release.get("package_execution_digest") != docs["package_execution"].digest:
        raise AcceptanceRefusal("package_binding_release_invalid", "execution digest")
    for field, wanted in (
        ("package_descriptor", docs["package_descriptor"].ref()),
        ("environment_spec", docs["package_environment_spec"].ref()),
    ):
        if _embedded_ref(release.get(field), f"binding_release.{field}") != wanted:
            raise AcceptanceRefusal("package_binding_release_invalid", field)
    if release.get("object_set_digest") != docs["artifact_object_set"].digest:
        raise AcceptanceRefusal("package_binding_release_invalid", "object_set_digest")

    supplied = request.get("binding_documents")
    if not isinstance(supplied, list):
        raise AcceptanceRefusal("acceptance_request_invalid", "binding_documents is not an array")
    local: dict[str, tuple[str, LocalDocument]] = {}
    for index, row in enumerate(supplied):
        if (
            not isinstance(row, dict)
            or set(row) != {"kind", "ref"}
            or row.get("kind") not in _BINDING_KINDS
        ):
            raise AcceptanceRefusal("acceptance_request_invalid", f"binding_documents[{index}]")
        document = _local_document(row["ref"], f"binding_documents[{index}].ref", input_root)
        if document.digest in local:
            raise AcceptanceRefusal("acceptance_request_invalid", "duplicate binding document")
        local[document.digest] = (str(row["kind"]), document)
    declared = release.get("documents")
    if not isinstance(declared, list) or len(declared) != len(local):
        raise AcceptanceRefusal("package_binding_release_invalid", "document set size")
    prior = ("", "")
    plans: dict[str, provisioned.ExactDocument] = {}
    model_contracts: list[str] = []
    for index, row in enumerate(declared):
        if not isinstance(row, dict) or set(row) != {"kind", "ref"}:
            raise AcceptanceRefusal("package_binding_release_invalid", f"documents[{index}]")
        kind = str(row.get("kind", ""))
        ref = _embedded_ref(row.get("ref"), f"binding_release.documents[{index}].ref")
        if (kind, ref.digest) <= prior or kind not in _BINDING_KINDS:
            raise AcceptanceRefusal(
                "package_binding_release_invalid", "documents not sorted unique"
            )
        held = local.get(ref.digest)
        if held is None or held[0] != kind or held[1].ref() != ref:
            raise AcceptanceRefusal("package_binding_release_invalid", f"document {ref.digest}")
        if kind == "entrypoint_binding_plan":
            plans[ref.digest] = held[1].exact()
        elif kind == "model_construction_contract":
            model_contracts.append(ref.digest)
        prior = (kind, ref.digest)

    object_set = provisioned.read_artifact_object_set(docs["artifact_object_set"].raw)
    widths: Any = {}
    if object_set.roots:
        try:
            import tensorfs
        except ImportError as exc:
            raise AcceptanceRefusal(
                "model_closure_unavailable",
                "model-bound acceptance requires the qualified TensorFS runtime",
            ) from exc
        widths = provisioned.tensorfs_destination_widths(tensorfs)
    all_documents = {digest: document for digest, (_kind, document) in local.items()}
    all_documents[docs["package_descriptor"].digest] = docs["package_descriptor"]

    def read_document(ref: provisioned.ArtifactRef, name: str) -> provisioned.ExactDocument:
        held = all_documents.get(ref.digest)
        if held is None or held.ref() != ref:
            raise PlanRefusal("binding_document_absent", f"{name}: {ref.digest}")
        return held.exact()

    try:
        resolved = provisioned.read_provisioned_release(
            plans,
            read_document=read_document,
            package_release=package_release_id,
            store_root=artifact_store,
            object_set=object_set,
            dtype_widths=widths,
        )
    except PlanRefusal as exc:
        raise AcceptanceRefusal(exc.code, exc.detail) from exc
    selected = next(
        (row for row in resolved.bindings.values() if row.entrypoint == entrypoint), None
    )
    if selected is None:
        raise AcceptanceRefusal("acceptance_entrypoint_unbound", entrypoint)
    return selected, sorted(model_contracts), release


def _placement(
    docs: Mapping[str, LocalDocument],
    release: Mapping[str, Any],
    package_release_id: str,
    binding_plan_digest: str,
) -> tuple[str, str]:
    placed = pb.DesiredPlacementSet(
        placement_set_digest=documents.raw(docs["placement_set"].digest),
        placement_set_canonical_bytes=docs["placement_set"].raw,
    )
    try:
        rows = read_placement_set(placed)
    except documents.DocumentError as exc:
        raise AcceptanceRefusal(exc.code, exc.detail) from exc
    if len(rows) != 1:
        raise AcceptanceRefusal("acceptance_placement_invalid", f"{len(rows)} placements")
    row = rows[0]
    placement_id = row.get("placement_id")
    if not isinstance(placement_id, str) or not placement_id:
        raise AcceptanceRefusal("acceptance_placement_invalid", "placement id")
    expected = {
        "package_release_id": package_release_id,
        "environment_spec_digest": docs["package_environment_spec"].digest,
        "installed_environment_receipt_digest": _embedded_ref(
            release.get("installed_environment_receipt"), "binding_release.receipt"
        ).digest,
        "package_descriptor_digest": docs["package_descriptor"].digest,
    }
    for field, wanted in expected.items():
        if row.get(field) != wanted:
            raise AcceptanceRefusal("acceptance_placement_invalid", field)
    object_set = row.get("model_object_set")
    if not isinstance(object_set, dict) or set(object_set) != {
        "digest",
        "kind",
        "length",
        "subject_id",
    }:
        raise AcceptanceRefusal("acceptance_placement_invalid", "model_object_set")
    if (
        object_set.get("kind") != "model_object_set"
        or object_set.get("subject_id") != docs["artifact_object_set"].digest
        or object_set.get("digest") != docs["artifact_object_set"].digest
        or object_set.get("length") != docs["artifact_object_set"].length
    ):
        raise AcceptanceRefusal("acceptance_placement_invalid", "model_object_set identity")
    plans = row.get("binding_plans")
    if not isinstance(plans, list):
        raise AcceptanceRefusal("acceptance_placement_invalid", "binding plans")
    release_documents = release.get("documents")
    if not isinstance(release_documents, list):
        raise AcceptanceRefusal("package_binding_release_invalid", "documents")
    release_plans = {
        ref.digest: ref.length
        for row in release_documents
        if isinstance(row, dict) and row.get("kind") == "entrypoint_binding_plan"
        for ref in [_embedded_ref(row.get("ref"), "binding_release.plan")]
    }
    placed_plans: dict[str, int] = {}
    for index, plan in enumerate(plans):
        if not isinstance(plan, dict) or set(plan) != {"digest", "kind", "length", "subject_id"}:
            raise AcceptanceRefusal("acceptance_placement_invalid", f"binding_plans[{index}]")
        digest = _digest(plan.get("digest"), f"binding_plans[{index}].digest")
        if plan.get("kind") != "plan" or plan.get("subject_id") != digest:
            raise AcceptanceRefusal("acceptance_placement_invalid", f"binding_plans[{index}]")
        length = plan.get("length")
        if not isinstance(length, int) or isinstance(length, bool) or length <= 0:
            raise AcceptanceRefusal("acceptance_placement_invalid", f"binding_plans[{index}]")
        if digest in placed_plans:
            raise AcceptanceRefusal("acceptance_placement_invalid", "duplicate binding plan")
        placed_plans[digest] = length
    if placed_plans != release_plans or binding_plan_digest not in placed_plans:
        raise AcceptanceRefusal("acceptance_placement_invalid", "binding plan set")
    return placement_id, docs["placement_set"].digest


def _invocation(
    document: LocalDocument,
    environment: _Environment,
    package_release_id: str,
    binding_plan_digest: str,
    fixture: LocalDocument,
) -> Mapping[str, Any]:
    try:
        value = documents.read(document.raw, pb.InvocationSpec)
    except documents.DocumentError as exc:
        raise AcceptanceRefusal(exc.code, exc.detail) from exc
    if value.get("package_release_id") != package_release_id:
        raise AcceptanceRefusal("acceptance_invocation_invalid", "package_release_id")
    if value.get("environment_spec_digest") != environment.environment_spec_digest:
        raise AcceptanceRefusal("acceptance_invocation_invalid", "environment_spec_digest")
    if value.get("payload_digest") != fixture.digest:
        raise AcceptanceRefusal("acceptance_invocation_invalid", "payload_digest")
    serving = value.get("serving")
    if (
        not isinstance(serving, dict)
        or serving.get("entrypoint_binding_plan_id") != binding_plan_digest
    ):
        raise AcceptanceRefusal("acceptance_invocation_invalid", "entrypoint binding plan")
    return value


def _attempt_accepted(value: object, invocation_digest: str, placement_id: str) -> dict[str, Any]:
    fields = {
        "executor_generation",
        "invocation_spec_digest",
        "model_construction_digest",
        "plan_digest",
        "placement_id",
        "request_id",
        "attempt_ordinal",
    }
    if not isinstance(value, dict) or set(value) != fields:
        raise AcceptanceRefusal("attempt_accepted_invalid", "wrong fields")
    if (
        value.get("invocation_spec_digest") != invocation_digest
        or value.get("placement_id") != placement_id
    ):
        raise AcceptanceRefusal("attempt_accepted_invalid", "invocation or placement")
    _digest(value.get("plan_digest"), "attempt_accepted.plan_digest")
    construction = value.get("model_construction_digest")
    if not isinstance(construction, str):
        raise AcceptanceRefusal("attempt_accepted_invalid", "model_construction_digest")
    if construction:
        _digest(construction, "attempt_accepted.model_construction_digest")
    if (
        not isinstance(value.get("request_id"), str)
        or not 0 < len(str(value["request_id"])) <= MAX_STRING
    ):
        raise AcceptanceRefusal("attempt_accepted_invalid", "request_id")
    for field in ("attempt_ordinal", "executor_generation"):
        if (
            not isinstance(value.get(field), int)
            or isinstance(value.get(field), bool)
            or value[field] <= 0
        ):
            raise AcceptanceRefusal("attempt_accepted_invalid", field)
    return cast(dict[str, Any], value)


def _outcome(
    document: LocalDocument, accepted: Mapping[str, Any], invocation_digest: str
) -> dict[str, Any]:
    try:
        value = documents.read(document.raw, pb.AttemptOutcomeBody)
    except documents.DocumentError as exc:
        raise AcceptanceRefusal(exc.code, exc.detail) from exc
    if (
        value.get("request_id") != accepted["request_id"]
        or value.get("attempt_ordinal") != accepted["attempt_ordinal"]
        or value.get("invocation_spec_digest") != invocation_digest
    ):
        raise AcceptanceRefusal("attempt_outcome_invalid", "attempt identity")
    status = value.get("status")
    if not isinstance(status, int) or isinstance(status, bool) or status <= 0:
        raise AcceptanceRefusal("attempt_outcome_invalid", "terminal status")
    return value


def _fixture_payload(descriptor_doc: LocalDocument, entrypoint: str) -> bytes:
    row = _descriptor_entrypoint(descriptor_doc, entrypoint)
    with tempfile.TemporaryDirectory(prefix="cozy-acceptance-fixture-") as temporary:
        payload, reason = warm.synthesize(row.get("request"), Path(temporary))
    if reason:
        raise AcceptanceRefusal("acceptance_fixture_unsynthesizable", reason)
    if _contains_file_url(payload):
        raise AcceptanceRefusal(
            "acceptance_fixture_asset_unsupported",
            "required asset fixtures need a separately granted exact input; none is fabricated",
        )
    return canonical.write(payload)


def _descriptor_entrypoint(descriptor_doc: LocalDocument, entrypoint: str) -> dict[str, Any]:
    try:
        body = descriptor.read_bytes(descriptor_doc.raw, str(descriptor_doc.path))
    except descriptor.StalePackageDescriptor as exc:
        raise AcceptanceRefusal(exc.default_code, str(exc)) from exc
    entries = body.get("entrypoints")
    if not isinstance(entries, list):
        raise AcceptanceRefusal(
            "acceptance_package_descriptor_invalid", "entrypoints is not an array"
        )
    row = next(
        (item for item in entries if isinstance(item, dict) and item.get("name") == entrypoint),
        None,
    )
    if not isinstance(row, dict):
        raise AcceptanceRefusal("acceptance_entrypoint_absent", entrypoint)
    return row


def _result_schema_digest(descriptor_doc: LocalDocument, entrypoint: str) -> str:
    row = _descriptor_entrypoint(descriptor_doc, entrypoint)
    result = row.get("result")
    if result is None:
        raise AcceptanceRefusal("acceptance_package_descriptor_invalid", "result schema is absent")
    return canonical.digest(result)


def _result(value: object) -> tuple[str, str]:
    if value is None:
        return "", ""
    if not isinstance(value, dict) or not set(value).issubset(
        {"adjustments", "checkpoint_ref", "inline_result", "result_blob", "result_schema_digest"}
    ):
        raise AcceptanceRefusal("attempt_result_invalid", "result envelope")
    schema_digest = _digest(value.get("result_schema_digest"), "result_schema_digest")
    inline = value.get("inline_result")
    blob = value.get("result_blob")
    if (inline is None) == (blob is None):
        raise AcceptanceRefusal("attempt_result_invalid", "result is not exactly inline xor blob")
    if inline is not None:
        if not isinstance(inline, str):
            raise AcceptanceRefusal("attempt_result_invalid", "inline_result is not base64")
        try:
            raw = base64.b64decode(inline, validate=True)
            canonical.parse_canonical(raw)
        except (ValueError, canonical.CanonicalError) as exc:
            raise AcceptanceRefusal("attempt_result_invalid", "inline result bytes") from exc
        return "sha256:" + hashlib.sha256(raw).hexdigest(), schema_digest
    if not isinstance(blob, dict):
        raise AcceptanceRefusal("attempt_result_invalid", "result_blob")
    _output_entry(blob, "result_blob")
    return str(blob["digest"]), schema_digest


def _output_manifest(value: object) -> dict[str, Any]:
    if value is None:
        return {"outputs": []}
    if not isinstance(value, dict) or set(value) not in (
        {"outputs"},
        {"outputs", "publication_receipt_digest"},
    ):
        raise AcceptanceRefusal("attempt_output_manifest_invalid", "manifest fields")
    publication = value.get("publication_receipt_digest")
    if publication is not None:
        _digest(publication, "output_manifest.publication_receipt_digest")
    rows = value.get("outputs", [])
    if not isinstance(rows, list):
        raise AcceptanceRefusal("attempt_output_manifest_invalid", "outputs is not an array")
    prior = ""
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise AcceptanceRefusal("attempt_output_manifest_invalid", f"outputs[{index}]")
        _output_entry(row, f"outputs[{index}]")
        output_id = row.get("output_id")
        if not isinstance(output_id, str) or output_id <= prior:
            raise AcceptanceRefusal("attempt_output_manifest_invalid", "outputs not sorted unique")
        prior = output_id
    return cast(dict[str, Any], value)


def _output_entry(value: Mapping[str, Any], name: str) -> None:
    if set(value) != {"digest", "length", "mime_type", "output_id"}:
        raise AcceptanceRefusal("attempt_output_manifest_invalid", f"{name} fields")
    _identifier(value.get("output_id"), f"{name}.output_id")
    _digest(value.get("digest"), f"{name}.digest")
    length = value.get("length")
    if not isinstance(length, int) or isinstance(length, bool) or not 0 <= length <= 1 << 40:
        raise AcceptanceRefusal("attempt_output_manifest_invalid", f"{name}.length")
    mime = value.get("mime_type")
    if not isinstance(mime, str) or _MIME.fullmatch(mime) is None:
        raise AcceptanceRefusal("attempt_output_manifest_invalid", f"{name}.mime_type")


def _protocol_projection(value: object, cls: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise AcceptanceRefusal(f"{name}_invalid", "not an object")
    try:
        message = cls()
        ParseDict(value, message, ignore_unknown_fields=False)
        rendered = documents.body(message)
    except (ParseError, documents.DocumentError, ValueError, TypeError) as exc:
        raise AcceptanceRefusal(f"{name}_invalid", str(exc)) from exc
    if rendered != value:
        raise AcceptanceRefusal(f"{name}_invalid", "not the canonical protocol projection")
    return rendered


def _canonical_object(raw: bytes, name: str) -> dict[str, Any]:
    try:
        value = canonical.parse_canonical(raw)
    except canonical.CanonicalError as exc:
        raise AcceptanceRefusal(f"{name}_invalid", str(exc)) from exc
    if not isinstance(value, dict):
        raise AcceptanceRefusal(f"{name}_invalid", "not an object")
    return value


def _embedded_ref(value: object, name: str) -> provisioned.ArtifactRef:
    if not isinstance(value, dict) or set(value) != {"digest", "length"}:
        raise AcceptanceRefusal("acceptance_ref_invalid", name)
    digest = _digest(value.get("digest"), f"{name}.digest")
    length = value.get("length")
    if (
        not isinstance(length, int)
        or isinstance(length, bool)
        or not 0 < length <= canonical.DOC_MAX_BYTES
    ):
        raise AcceptanceRefusal("acceptance_ref_invalid", f"{name}.length")
    return provisioned.ArtifactRef(digest, length)


def _digest(value: object, name: str) -> str:
    if not isinstance(value, str) or canonical.DIGEST_RE.fullmatch(value) is None:
        raise AcceptanceRefusal("acceptance_digest_invalid", f"{name}={value!r}")
    return value


def _identifier(value: object, name: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise AcceptanceRefusal("acceptance_identifier_invalid", f"{name}={value!r}")
    return value


def _absolute(value: object, name: str) -> Path:
    if not isinstance(value, str):
        raise AcceptanceRefusal("acceptance_path_invalid", f"{name} is not a string")
    path = Path(value)
    if not path.is_absolute() or path == Path("/") or path != Path(os.path.normpath(path)):
        raise AcceptanceRefusal("acceptance_path_invalid", f"{name}={value!r}")
    return path


def _regular_file(value: object, name: str) -> Path:
    path = _absolute(value, name)
    try:
        resolved = path.resolve(strict=True)
        info = path.lstat()
    except OSError as exc:
        raise AcceptanceRefusal("acceptance_path_invalid", f"{name}: {type(exc).__name__}") from exc
    if not stat.S_ISREG(info.st_mode) or path.is_symlink():
        raise AcceptanceRefusal("acceptance_path_invalid", f"{name} is not a regular file")
    return resolved


def _directory(value: object, name: str) -> Path:
    path = _absolute(value, name)
    try:
        resolved = path.resolve(strict=True)
        info = path.lstat()
    except OSError as exc:
        raise AcceptanceRefusal("acceptance_path_invalid", f"{name}: {type(exc).__name__}") from exc
    if not stat.S_ISDIR(info.st_mode) or path.is_symlink():
        raise AcceptanceRefusal("acceptance_path_invalid", f"{name} is not a directory")
    return resolved


def _contains_file_url(value: object) -> bool:
    if isinstance(value, str):
        return value.startswith("file://")
    if isinstance(value, Mapping):
        return any(_contains_file_url(item) for item in value.values())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return any(_contains_file_url(item) for item in value)
    return False
