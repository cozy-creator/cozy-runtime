"""One reader for the selected provisioned serving release.

Tensorhub authors one content-addressed plan for every visible entrypoint.  This module is
the sole Runtime authority that follows those plan bytes through the descriptor, optional
RMB, model config and MCC, while keeping TensorFS root membership intact.  Materialization
uses it against freshly fetched grants; the worker uses it again against the exact staged
files after the process boundary.
"""

from __future__ import annotations

import hashlib
import re
import stat
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cozy_runtime.internal import canonical, contract
from cozy_runtime.internal import descriptor as endpoint_descriptor
from cozy_runtime.internal.encoding import TORCH_DTYPES
from cozy_runtime.internal.worker.plan import DeclaredBinding, PlanRefusal

IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]{0,127}$")
MAX_PROVISIONED_ARTIFACT_BYTES = 8 << 30
ARTIFACT_OBJECT_SET_KIND = "tensorhub.resolved_object_set/1"


@dataclass(frozen=True, slots=True)
class ArtifactRef:
    digest: str
    length: int


@dataclass(frozen=True, slots=True)
class ExactDocument:
    ref: ArtifactRef
    path: Path
    raw: bytes


@dataclass(frozen=True, slots=True)
class ModelRoot:
    root: ArtifactRef
    objects: tuple[ArtifactRef, ...]


@dataclass(frozen=True, slots=True)
class ArtifactObjectSet:
    """The complete delivered model-root closure; empty for a weightless endpoint."""

    roots: tuple[ModelRoot, ...]

    @property
    def refs(self) -> Mapping[str, int]:
        refs: dict[str, int] = {}
        for group in self.roots:
            for ref in (group.root, *group.objects):
                previous = refs.setdefault(ref.digest, ref.length)
                if previous != ref.length:
                    raise PlanRefusal(
                        "artifact_object_set_invalid",
                        f"{ref.digest} has conflicting lengths {previous} and {ref.length}",
                    )
        return refs


@dataclass(frozen=True, slots=True)
class ProvisionedRelease:
    bindings: Mapping[str, DeclaredBinding]
    documents: Mapping[str, int]
    config_assets: Mapping[str, int]


DocumentReader = Callable[[ArtifactRef, str], ExactDocument]
DTypeWidths = Mapping[str, int] | Callable[[], Mapping[str, int]]


def tensorfs_destination_widths(tensorfs: Any) -> Mapping[str, int]:
    """Project TensorFS's one dtype-width table into contract/torch spellings."""

    source = getattr(tensorfs, "DTYPES", {})
    return {
        torch_name: int(source[storage_name])
        for storage_name, torch_name in TORCH_DTYPES.items()
        if storage_name in source
    }


def read_artifact_object_set(raw: bytes) -> ArtifactObjectSet:
    document = _canonical_object(raw, "artifact_object_set")
    _closed_keys(document, {"kind", "roots"}, "artifact_object_set")
    if document["kind"] != ARTIFACT_OBJECT_SET_KIND:
        raise PlanRefusal("artifact_object_set_format", str(document["kind"]))
    rows = document["roots"]
    if not isinstance(rows, list):
        raise PlanRefusal("artifact_object_set_invalid", "roots is not an array")
    roots: list[ModelRoot] = []
    prior_root = ""
    for index, row in enumerate(rows):
        _closed_keys(row, {"objects", "root"}, f"artifact_object_set.roots[{index}]")
        root = _artifact_ref(row["root"], f"artifact_object_set.roots[{index}].root")
        if root.digest <= prior_root:
            raise PlanRefusal("artifact_object_set_invalid", "roots are not sorted unique")
        object_rows = row["objects"]
        if not isinstance(object_rows, list) or not object_rows:
            raise PlanRefusal("artifact_object_set_invalid", "root objects are absent")
        objects: list[ArtifactRef] = []
        prior_object = ""
        for object_index, value in enumerate(object_rows):
            ref = _artifact_ref(
                value,
                f"artifact_object_set.roots[{index}].objects[{object_index}]",
            )
            if ref.digest <= prior_object or ref.digest == root.digest:
                raise PlanRefusal(
                    "artifact_object_set_invalid",
                    "objects are not sorted unique or include their root",
                )
            prior_object = ref.digest
            objects.append(ref)
        roots.append(ModelRoot(root, tuple(objects)))
        prior_root = root.digest
    result = ArtifactObjectSet(tuple(roots))
    _ = result.refs  # prove cross-root length consistency once
    return result


def read_provisioned_release(
    plans: Mapping[str, ExactDocument],
    *,
    read_document: DocumentReader,
    endpoint_release: str,
    store_root: Path,
    object_set: ArtifactObjectSet,
    dtype_widths: DTypeWidths,
) -> ProvisionedRelease:
    """Resolve the complete selected plan set through one closed typed reader."""

    if not plans:
        raise PlanRefusal("binding_plan_set_invalid", "selected plan set is empty")
    documents: dict[str, int] = {}
    config_assets: dict[str, int] = {}
    bindings: dict[str, DeclaredBinding] = {}
    used_model_roots: set[str] = set()
    descriptor_ref: ArtifactRef | None = None
    descriptor_doc: ExactDocument | None = None
    descriptor: dict[str, Any] | None = None
    visible: dict[str, dict[str, Any]] = {}

    for plan_id, plan_doc in sorted(plans.items()):
        _verify_exact(plan_doc, "entrypoint_binding_plan")
        if plan_id != plan_doc.ref.digest:
            raise PlanRefusal(
                "binding_plan_subject_invalid", f"subject {plan_id} != digest {plan_doc.ref.digest}"
            )
        plan = _canonical_object(plan_doc.raw, "entrypoint_binding_plan")
        _closed_keys(plan, {"bindings", "descriptor", "entrypoint", "format"}, "plan")
        if plan["format"] != "cozy.endpoint.EntrypointBindingPlan/1":
            raise PlanRefusal("binding_plan_format", str(plan["format"]))
        entrypoint = _identifier(plan["entrypoint"], "plan.entrypoint")
        current_descriptor_ref = _document_ref(plan["descriptor"], "plan.descriptor")
        if descriptor_ref is None:
            descriptor_ref = current_descriptor_ref
            descriptor_doc = read_document(descriptor_ref, "descriptor")
            _verify_exact(descriptor_doc, "descriptor")
            try:
                descriptor = endpoint_descriptor.read_bytes(
                    descriptor_doc.raw, str(descriptor_doc.path)
                )
            except endpoint_descriptor.StaleDescriptor as exc:
                raise PlanRefusal("descriptor_invalid", str(exc)) from exc
            documents[descriptor_ref.digest] = descriptor_ref.length
            visible = _visible_entrypoints(descriptor)
        elif current_descriptor_ref != descriptor_ref:
            raise PlanRefusal("binding_plan_set_invalid", "plans name different descriptors")
        if entrypoint in bindings:
            raise PlanRefusal("binding_plan_set_invalid", f"duplicate entrypoint {entrypoint!r}")
        described = visible.get(entrypoint)
        if described is None:
            raise PlanRefusal("binding_descriptor_invalid", f"visible entrypoint {entrypoint!r}")
        assert descriptor_doc is not None and descriptor is not None
        bindings[entrypoint] = _read_binding(
            plan_id,
            plan,
            described,
            descriptor,
            descriptor_doc.path,
            read_document,
            endpoint_release,
            store_root,
            object_set,
            dtype_widths,
            documents,
            config_assets,
            used_model_roots,
        )

    if set(bindings) != set(visible):
        raise PlanRefusal(
            "binding_plan_set_invalid",
            f"missing={sorted(set(visible) - set(bindings))} "
            f"extra={sorted(set(bindings) - set(visible))}",
        )
    delivered_model_roots = {root.root.digest for root in object_set.roots}
    if used_model_roots != delivered_model_roots:
        raise PlanRefusal(
            "artifact_object_set_mismatch",
            f"used model roots {sorted(used_model_roots)} differ from delivered model roots "
            f"{sorted(delivered_model_roots)}",
        )
    return ProvisionedRelease(
        bindings={binding.entrypoint_binding_plan_id: binding for binding in bindings.values()},
        documents=documents,
        config_assets=config_assets,
    )


def staged_document_reader(root: Path) -> DocumentReader:
    def read(ref: ArtifactRef, name: str) -> ExactDocument:
        path = root / f"{ref.digest.removeprefix('sha256:')}.json"
        try:
            raw = path.read_bytes()
        except OSError as exc:
            raise PlanRefusal("binding_document_absent", f"{name}: {ref.digest}") from exc
        document = ExactDocument(ref, path, raw)
        _verify_exact(document, name)
        return document

    return read


def _visible_entrypoints(descriptor: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    _nonempty(descriptor.get("application"), "descriptor.application")
    entries = descriptor.get("entrypoints")
    if not isinstance(entries, list):
        raise PlanRefusal("binding_descriptor_invalid", "entrypoints is not an array")
    visible: dict[str, dict[str, Any]] = {}
    for row in entries:
        if not isinstance(row, dict) or row.get("hidden") is not False:
            continue
        name = _identifier(row.get("name"), "descriptor.entrypoint.name")
        if name in visible:
            raise PlanRefusal("binding_descriptor_invalid", f"duplicate entrypoint {name!r}")
        visible[name] = row
    if not visible:
        raise PlanRefusal("binding_descriptor_invalid", "descriptor has no visible entrypoint")
    return visible


def _read_binding(
    plan_id: str,
    plan: Mapping[str, Any],
    described: Mapping[str, Any],
    descriptor: Mapping[str, Any],
    descriptor_path: Path,
    read_document: DocumentReader,
    endpoint_release: str,
    store_root: Path,
    object_set: ArtifactObjectSet,
    dtype_widths: DTypeWidths,
    documents: dict[str, int],
    config_assets: dict[str, int],
    used_model_roots: set[str],
) -> DeclaredBinding:
    entrypoint = str(plan["entrypoint"])
    models = described.get("models", [])
    selected = plan["bindings"]
    if not isinstance(models, list) or not isinstance(selected, list):
        raise PlanRefusal("binding_plan_unsupported", "models and bindings must be arrays")
    if len(models) > 1 or len(selected) != len(models):
        raise PlanRefusal(
            "binding_plan_unsupported",
            "current Runtime requires zero or one selected model slot per entrypoint",
        )
    application = _nonempty(descriptor.get("application"), "descriptor.application")
    if not models:
        return DeclaredBinding(
            entrypoint_binding_plan_id=plan_id,
            entrypoint=entrypoint,
            endpoint_release=endpoint_release,
            model_class="",
            model_binding_path="",
            model_parameter_name="",
            release=endpoint_release,
            application=application,
            descriptor_path=str(descriptor_path),
        )

    model = models[0]
    if not isinstance(model, dict):
        raise PlanRefusal("binding_descriptor_invalid", f"entrypoint {entrypoint!r} model")
    slot_row = selected[0]
    _closed_keys(slot_row, {"binding", "slot"}, "plan.bindings[0]")
    slot = _identifier(slot_row["slot"], "plan.bindings[0].slot")
    if model.get("param") != slot:
        raise PlanRefusal("binding_descriptor_invalid", f"slot {slot!r}")
    binding_ref = _document_ref(slot_row["binding"], "plan.bindings[0].binding")
    binding_doc = read_document(binding_ref, "resolved_model_binding")
    _verify_exact(binding_doc, "resolved_model_binding")
    documents[binding_ref.digest] = binding_ref.length
    binding = _canonical_object(binding_doc.raw, "resolved_model_binding")
    _closed_keys(
        binding,
        {
            "checkpoint",
            "config",
            "execution_layout",
            "format",
            "model_construction_contract",
            "stamps",
        },
        "resolved_model_binding",
    )
    if binding["format"] != "cozy.endpoint.ResolvedModelBinding/1":
        raise PlanRefusal("resolved_model_binding_format", str(binding["format"]))

    checkpoint = binding["checkpoint"]
    _closed_keys(checkpoint, {"snapshot_root", "tensor_layout"}, "binding.checkpoint")
    snapshot = _model_root(checkpoint, object_set, "binding.checkpoint", used_model_roots)

    config_row = binding["config"]
    _closed_keys(config_row, {"assets", "document"}, "binding.config")
    assets = config_row["assets"]
    if not isinstance(assets, list):
        raise PlanRefusal("binding_config_invalid", "assets is not an array")
    reachable = object_set.refs
    prior_asset = ""
    for position, asset in enumerate(assets):
        _closed_keys(asset, {"name", "ref"}, f"binding.config.assets[{position}]")
        name = _identifier(asset["name"], f"binding.config.assets[{position}].name")
        if name <= prior_asset:
            raise PlanRefusal("binding_config_invalid", "asset names are not sorted unique")
        asset_ref = _artifact_ref(asset["ref"], f"binding.config.assets[{position}].ref")
        if reachable.get(asset_ref.digest) != asset_ref.length:
            raise PlanRefusal(
                "binding_config_invalid", f"asset {name!r} is outside the TensorFS closure"
            )
        previous = config_assets.setdefault(asset_ref.digest, asset_ref.length)
        if previous != asset_ref.length:
            raise PlanRefusal("binding_config_invalid", f"asset {name!r} length conflicts")
        prior_asset = name

    config_ref = _document_ref(config_row["document"], "binding.config.document")
    config_doc = read_document(config_ref, "binding_config")
    _verify_exact(config_doc, "binding_config")
    _canonical_object(config_doc.raw, "binding_config")
    documents[config_ref.digest] = config_ref.length
    contract_ref = _document_ref(
        binding["model_construction_contract"], "binding.model_construction_contract"
    )
    contract_doc = read_document(contract_ref, "model_construction_contract")
    _verify_exact(contract_doc, "model_construction_contract")
    documents[contract_ref.digest] = contract_ref.length
    try:
        construction = contract.read_contract(contract_doc.raw)
    except contract.ContractError as exc:
        raise PlanRefusal("model_construction_contract_invalid", str(exc)) from exc
    candidate = construction.candidate
    model_class = _identifier(model.get("class"), "descriptor.model.class")
    model_path = _nonempty(model.get("path"), "descriptor.model.path")
    if (
        candidate.release != endpoint_release
        or candidate.application != application
        or not candidate.model_class.endswith(":" + model_class)
        or candidate.snapshot != snapshot.digest
        or candidate.config_digest != config_ref.digest
    ):
        raise PlanRefusal("resolved_model_binding_mismatch", "MCC candidate join")

    layout = binding["execution_layout"]
    if not isinstance(layout, list) or not layout:
        raise PlanRefusal("resolved_model_binding_invalid", "execution_layout is empty")
    components: list[str] = []
    snapshots: dict[str, str] = {}
    for position, row in enumerate(layout):
        _closed_keys(
            row,
            {"component", "snapshot_root", "tensor_layout"},
            f"binding.execution_layout[{position}]",
        )
        component = _identifier(row["component"], f"binding.execution_layout[{position}].component")
        if component in snapshots:
            raise PlanRefusal("resolved_model_binding_invalid", "duplicate component")
        component_snapshot = _model_root(
            row, object_set, f"binding.execution_layout[{position}]", used_model_roots
        )
        components.append(component)
        snapshots[component] = component_snapshot.digest
    if components != [row.component for row in construction.destination_sets]:
        raise PlanRefusal("resolved_model_binding_mismatch", "MCC destination order")
    component_use = model.get("component_use")
    if not isinstance(component_use, dict):
        raise PlanRefusal("binding_descriptor_invalid", f"slot {slot!r} component_use")
    declared_components: set[str] = set()
    for method, values in component_use.items():
        if not isinstance(method, str) or not isinstance(values, list):
            raise PlanRefusal("binding_descriptor_invalid", f"slot {slot!r} component_use")
        declared_components.update(
            _identifier(value, f"descriptor.model.component_use.{method}") for value in values
        )
    if declared_components != set(components):
        raise PlanRefusal("resolved_model_binding_mismatch", "descriptor component set")

    stamps = _stamps(binding["stamps"], snapshots)
    descriptor_stamps = model.get("stamps")
    if not isinstance(descriptor_stamps, dict) or any(
        not isinstance(key, str) or not isinstance(value, str)
        for key, value in descriptor_stamps.items()
    ):
        raise PlanRefusal("binding_descriptor_invalid", f"slot {slot!r} stamps")
    for key, value in descriptor_stamps.items():
        if value not in stamps.get(key, set()):
            raise PlanRefusal("resolved_model_binding_mismatch", f"missing stamp {key}={value}")

    try:
        widths = dtype_widths() if callable(dtype_widths) else dtype_widths
        model_weight_bytes = contract.destination_resident_bytes(construction, widths)
    except contract.ContractError as exc:
        raise PlanRefusal(exc.code, str(exc)) from exc
    return DeclaredBinding(
        entrypoint_binding_plan_id=plan_id,
        entrypoint=entrypoint,
        endpoint_release=endpoint_release,
        model_class=model_class,
        model_binding_path=model_path,
        model_parameter_name=slot,
        release=endpoint_release,
        application=application,
        descriptor_path=str(descriptor_path),
        store=str(store_root),
        config=str(config_doc.path),
        variant=candidate.hardware_variant,
        reference_snapshot=snapshot.digest,
        components=tuple(components),
        snapshots=snapshots,
        stamps={key: tuple(sorted(values)) for key, values in sorted(stamps.items())},
        model_weight_bytes=model_weight_bytes,
        vram_floor_bytes=0,
        expected_logical_model_digest=contract.logical_model_digest(construction),
    )


def _model_root(
    value: Mapping[str, Any],
    object_set: ArtifactObjectSet,
    name: str,
    used_model_roots: set[str],
) -> ArtifactRef:
    root = _artifact_ref(value["snapshot_root"], f"{name}.snapshot_root")
    layout = _artifact_ref(value["tensor_layout"], f"{name}.tensor_layout")
    matches = [row for row in object_set.roots if row.root.digest == root.digest]
    if len(matches) != 1 or matches[0].root != root:
        raise PlanRefusal("artifact_object_set_mismatch", f"{name} does not name a model root")
    objects = {row.digest: row.length for row in matches[0].objects}
    if objects.get(layout.digest) != layout.length:
        raise PlanRefusal("artifact_object_set_mismatch", f"{name} layout is not reachable")
    used_model_roots.add(root.digest)
    return root


def _stamps(value: object, snapshots: Mapping[str, str]) -> dict[str, set[str]]:
    if not isinstance(value, list):
        raise PlanRefusal("resolved_model_binding_invalid", "stamps is not an array")
    stamps: dict[str, set[str]] = {}
    prior_stamp = ("", "")
    for position, row in enumerate(value):
        _closed_keys(row, {"component", "key", "values"}, f"binding.stamps[{position}]")
        component = _identifier(row["component"], f"binding.stamps[{position}].component")
        key = _identifier(row["key"], f"binding.stamps[{position}].key")
        values = row["values"]
        stamp_identity = (component, key)
        if (
            component not in snapshots
            or stamp_identity <= prior_stamp
            or not isinstance(values, list)
            or not values
        ):
            raise PlanRefusal("resolved_model_binding_invalid", f"stamps[{position}]")
        carried = stamps.setdefault(key, set())
        prior_value = ""
        for value in values:
            item = _nonempty(value, f"binding.stamps[{position}].values")
            if len(item) > 256 or item <= prior_value:
                raise PlanRefusal("resolved_model_binding_invalid", f"stamps[{position}].values")
            carried.add(item)
            prior_value = item
        prior_stamp = stamp_identity
    return stamps


def _verify_exact(document: ExactDocument, name: str) -> None:
    ref = document.ref
    try:
        info = document.path.lstat()
    except OSError as exc:
        raise PlanRefusal("binding_document_absent", f"{name}: {ref.digest}") from exc
    digest = "sha256:" + hashlib.sha256(document.raw).hexdigest()
    if (
        not document.path.is_absolute()
        or not stat.S_ISREG(info.st_mode)
        or info.st_size != len(document.raw)
        or digest != ref.digest
        or len(document.raw) != ref.length
    ):
        raise PlanRefusal("binding_document_identity_mismatch", name)


def _canonical_object(raw: bytes, name: str) -> dict[str, Any]:
    try:
        value = canonical.parse_canonical(raw)
    except canonical.CanonicalError as exc:
        raise PlanRefusal(f"{name}_invalid", str(exc)) from exc
    if not isinstance(value, dict):
        raise PlanRefusal(f"{name}_invalid", "document is not an object")
    return value


def _closed_keys(value: object, keys: set[str], name: str) -> None:
    if not isinstance(value, dict) or set(value) != keys:
        raise PlanRefusal(f"{name}_invalid", "document has a non-closed field set")


def _nonempty(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise PlanRefusal("binding_document_invalid", f"{name} is not a non-empty string")
    return value


def _identifier(value: object, name: str) -> str:
    text = _nonempty(value, name)
    if IDENTIFIER_RE.fullmatch(text) is None:
        raise PlanRefusal("binding_document_invalid", f"{name} is not an identifier")
    return text


def _artifact_ref(value: object, name: str) -> ArtifactRef:
    if not isinstance(value, dict) or set(value) != {"digest", "length"}:
        raise PlanRefusal("binding_document_invalid", f"{name} is not a closed reference")
    digest = value.get("digest")
    length = value.get("length")
    if (
        not isinstance(digest, str)
        or canonical.DIGEST_RE.fullmatch(digest) is None
        or not isinstance(length, int)
        or isinstance(length, bool)
        or length <= 0
        or length > MAX_PROVISIONED_ARTIFACT_BYTES
    ):
        raise PlanRefusal("binding_document_invalid", f"{name} reference is invalid")
    return ArtifactRef(digest, length)


def _document_ref(value: object, name: str) -> ArtifactRef:
    ref = _artifact_ref(value, name)
    if ref.length > canonical.DOC_MAX_BYTES:
        raise PlanRefusal("binding_document_invalid", f"{name} exceeds the document cap")
    return ref
