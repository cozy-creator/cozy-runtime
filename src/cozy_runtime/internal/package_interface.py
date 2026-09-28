"""The package interface generated from one frozen application.

Its one identity is SHA-256 over canonical JSON. Creator obtains those exact bytes from
`cozy-runtime --json describe`; nothing is written into the package repository.
"""

from __future__ import annotations

import hashlib
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal, NoReturn, cast

import msgspec

from cozy_runtime.author import ConformanceError, Surface, WeightsOutput
from cozy_runtime.author._markers import IMAGE_PREPARATION_PROFILE
from cozy_runtime.author._media import KIND_MEDIA
from cozy_runtime.author._model import ENCODED_LEAVES, FUSION
from cozy_runtime.author._model_defaults import DefaultLadder, default_ladder
from cozy_runtime.author._signature import AssetInputValue, AssetsBinding
from cozy_runtime.internal import canonical, invocable_interface, schema
from cozy_runtime.internal.canonical import Json

if TYPE_CHECKING:
    from cozy_runtime.internal.discovery import Discovered

Kind = Literal["entrypoint", "job"]

SCHEMA = "cozy.package.interface/1"
DIGEST_KEY = "package_interface_digest"
FILENAME = "metadata/package-interface.json"

#: The top-level fields every interface carries. Readers ignore additional fields.
TOP_LEVEL_KEYS = frozenset(
    {
        "format",
        "application",
        "entrypoints",
        "jobs",
    }
)


class StalePackageInterface(ConformanceError):
    """Supplied package-interface bytes are malformed or differ from the installed release."""

    default_code = "stale_package_interface"


# ------------------------------------------------------------------ the typed reading
#
# `parse` decodes the document once into these structs. Additive fields from a newer writer
# are ignored; the canonical bytes stay the identity, so nothing re-encodes a parsed
# interface to name it. Request/result schemas are an open recursive grammar (`Json`),
# checked by `_schema`; msgspec cannot analyze a recursive alias, so it sees `object`.

if TYPE_CHECKING:
    JsonValue = Json
else:
    JsonValue = object


class LadderRung(msgspec.Struct, frozen=True, kw_only=True, omit_defaults=True):
    gpu: str
    gpus: int = 0
    lane: str


class SequenceParallel(msgspec.Struct, frozen=True):
    degrees: tuple[Annotated[int, msgspec.Meta(ge=2)], ...]


class ModelSlot(msgspec.Struct, frozen=True, kw_only=True, omit_defaults=True):
    path: str
    class_name: str = msgspec.field(name="class")
    component_use: dict[str, tuple[str, ...]]
    encoded_leaves: str = "refuse"
    fusion: str = "refuse"
    default_ladder: tuple[LadderRung, ...] | msgspec.UnsetType = msgspec.UNSET
    sequence_parallel: SequenceParallel | msgspec.UnsetType = msgspec.UNSET

    @property
    def parameter(self) -> str:
        return self.path.rpartition(".")[2]


class ImagePreparation(msgspec.Struct, frozen=True, kw_only=True, omit_defaults=True):
    profile: str
    max_edge: Annotated[int, msgspec.Meta(gt=0)] | msgspec.UnsetType = msgspec.UNSET
    max_pixels: Annotated[int, msgspec.Meta(gt=0)] | msgspec.UnsetType = msgspec.UNSET


class AssetKind(msgspec.Struct, frozen=True, kw_only=True, omit_defaults=True):
    kind: Literal["image", "video", "audio", "file"]
    media_types: tuple[str, ...]
    max_count: Annotated[int, msgspec.Meta(ge=0)] | msgspec.UnsetType = msgspec.UNSET
    max_bytes: Annotated[int, msgspec.Meta(gt=0)] | msgspec.UnsetType = msgspec.UNSET
    max_decoded_bytes: Annotated[int, msgspec.Meta(gt=0)] | msgspec.UnsetType = msgspec.UNSET
    prepare: ImagePreparation | msgspec.UnsetType = msgspec.UNSET


class AssetsSlot(msgspec.Struct, frozen=True, kw_only=True, omit_defaults=True):
    parameter: str
    kinds: Annotated[tuple[AssetKind, ...], msgspec.Meta(min_length=1, max_length=4)]
    view: Literal["decoded"] | msgspec.UnsetType = msgspec.UNSET


class WeightsOutputDeclaration(msgspec.Struct, frozen=True, kw_only=True):
    output_id: Annotated[str, msgspec.Meta(pattern=r"^[a-z][a-z0-9_-]{0,63}$")]
    mime_type: Literal["application/vnd.cozy.model-manifest"]
    max_bytes: Annotated[int, msgspec.Meta(ge=0, le=(1 << 53) - 1)]


class CallableDoc(msgspec.Struct, frozen=True, kw_only=True, omit_defaults=True):
    name: str
    request: JsonValue
    result: JsonValue
    internal: bool = False
    models: Annotated[tuple[ModelSlot, ...], msgspec.Meta(max_length=16)] = ()
    assets: AssetsSlot | msgspec.UnsetType = msgspec.UNSET
    invocable: invocable_interface.Invocable | msgspec.UnsetType = msgspec.UNSET
    publishes: bool | msgspec.UnsetType = msgspec.UNSET
    accelerator: bool | msgspec.UnsetType = msgspec.UNSET
    weights_outputs: (
        Annotated[tuple[WeightsOutputDeclaration, ...], msgspec.Meta(min_length=1, max_length=16)]
        | msgspec.UnsetType
    ) = msgspec.UNSET


class PackageInterface(msgspec.Struct, frozen=True, kw_only=True):
    format: str
    application: Annotated[str, msgspec.Meta(min_length=1)]
    entrypoints: tuple[CallableDoc, ...]
    jobs: tuple[CallableDoc, ...]

    def callables(self) -> tuple[tuple[Kind, CallableDoc], ...]:
        return (
            *(("entrypoint", entry) for entry in self.entrypoints),
            *(("job", entry) for entry in self.jobs),
        )

    def job(self, name: str) -> CallableDoc | None:
        return next((entry for entry in self.jobs if entry.name == name), None)


# ------------------------------------------------------------------------ the document


def build(found: Discovered) -> dict[str, Json]:
    """Derive one package interface from the frozen (imported) application."""
    docs = [(surface.kind, _callable(surface)) for surface in found.surfaces if not surface.hidden]
    return assemble(found.application, docs, found.bindings)


def assemble(
    application: str,
    docs: Sequence[tuple[Kind, dict[str, Json]]],
    bindings: Mapping[str, Mapping[str, object]],
) -> dict[str, Json]:
    """One package interface from its callable documents, whichever reader produced them."""
    entrypoints: list[Json] = [doc for kind, doc in docs if kind == "entrypoint"]
    jobs: list[Json] = [doc for kind, doc in docs if kind == "job"]
    body: dict[str, Json] = {
        "format": SCHEMA,
        "application": application,
        "entrypoints": sorted(entrypoints, key=_by_name),
        "jobs": sorted(jobs, key=_by_name),
    }
    assert set(body) == TOP_LEVEL_KEYS, sorted(set(body) ^ TOP_LEVEL_KEYS)
    _fixed_point(body, _binding_values(bindings))
    return body


def _by_name(doc: Json) -> str:
    assert isinstance(doc, dict)
    name = doc["name"]
    assert isinstance(name, str)
    return name


def _callable(surface: Surface) -> dict[str, Json]:
    doc = callable_doc(
        name=surface.name,
        kind=surface.kind,
        payload_type=surface.payload_type,
        result_type=surface.result_type,
        models=[
            model_slot(
                binding.path,
                binding.class_key,
                binding.encoded_leaves,
                binding.fusion,
                binding.components,
                binding.sequence_parallel,
                defaults=surface.model_defaults.get(binding.param, ()),
            )
            for binding in surface.model_bindings
        ],
        assets=surface.assets_binding,
        publishes=surface.publishes,
        weights_outputs=surface.weights_outputs,
        internal=surface.internal,
        accelerator=surface.accelerator,
    )
    if surface.invocable or surface.kind == "entrypoint":
        doc["invocable"] = invocable_interface.metadata(surface)
        if surface.invocable and surface.kind == "entrypoint":
            invocable_interface.project_serving_models(doc)
    return doc


def callable_doc(
    *,
    name: str,
    kind: Kind,
    payload_type: object,
    result_type: object,
    models: Sequence[dict[str, Json]],
    assets: AssetsBinding | None,
    publishes: bool,
    weights_outputs: Sequence[WeightsOutput],
    internal: bool = False,
    accelerator: bool | None = None,
) -> dict[str, Json]:
    """One callable's document from plain facts — the import and static readers meet here."""
    doc: dict[str, Json] = {
        "name": name,
        "request": schema.request(payload_type),
        "result": schema.render(result_type, decoded_bounds=True),
    }
    if internal:
        doc["internal"] = True
    if models:
        degrees = [set(model.get("sequence_parallel", {}).get("degrees", [])) for model in models]
        if any(degrees) and not set.intersection(*degrees):
            raise ConformanceError(
                f"{name}: its Model slots declare @sequence_parallel degrees {degrees} with "
                "none in common; a group shards every slot, so each must share a degree",
                code="sequence_parallel_declaration_conflict",
            )
        doc["models"] = list(models)
    if assets:
        kinds: list[Json] = []
        for cls, bound in assets.kinds:
            row: dict[str, Json] = {
                "kind": cls.kind,
                "media_types": list(
                    bound.media_types if bound and bound.media_types else KIND_MEDIA[cls.kind]
                ),
            }
            counts = dict(assets.counts)
            if cls.kind in counts:
                row["max_count"] = counts[cls.kind]
            if bound and bound.max_bytes is not None:
                row["max_bytes"] = bound.max_bytes
            if bound and bound.max_decoded_bytes is not None:
                row["max_decoded_bytes"] = bound.max_decoded_bytes
            if cls.kind == "image" and assets.image_preparation is not None:
                row["prepare"] = dict(assets.image_preparation.descriptor())
            kinds.append(row)
        slot: dict[str, Json] = {"parameter": assets.parameter, "kinds": kinds}
        if assets.decoded:
            slot["view"] = "decoded"
        doc["assets"] = slot
    if kind == "job":
        doc["publishes"] = publishes
        if accelerator is not None:
            doc["accelerator"] = accelerator
        if weights_outputs:
            outputs: list[Json] = []
            for output in sorted(weights_outputs, key=lambda row: row.name):
                declared: dict[str, Json] = {
                    "output_id": output.name,
                    "mime_type": "application/vnd.cozy.model-manifest",
                    "max_bytes": output.max_new_bytes,
                }
                outputs.append(declared)
            doc["weights_outputs"] = outputs
    return doc


def model_slot(
    path: str,
    class_name: str,
    encoded_leaves: str,
    fusion: str,
    components: Mapping[str, Sequence[str]],
    sequence_parallel: Sequence[int],
    *,
    defaults: DefaultLadder = (),
) -> dict[str, Json]:
    """One model slot. The PATH is the binding point, the CLASS carries `encoded_leaves`,
    and the per-public-method component sets are the ComponentUseContracts. Optional authored
    defaults are unresolved source metadata; exact artifact selection remains dispatch-owned."""
    doc: dict[str, Json] = {
        "path": path,
        "class": class_name,
        "encoded_leaves": encoded_leaves,
        # Declaration order encodes no control flow and no residency choice (§1.1), so the
        # contract sorts — a reordered method body is not a surface change.
        "component_use": {method: sorted(names) for method, names in sorted(components.items())},
    }
    if defaults:
        for _, gpus, _ in defaults:
            if gpus > 1 and gpus not in sequence_parallel:
                raise ConformanceError(
                    "default GPU count is not a declared sequence-parallel degree",
                    code="model_defaults",
                )
        doc["default_ladder"] = [
            {"gpu": gpu, **({"gpus": gpus} if gpus else {}), "lane": lane}
            for gpu, gpus, lane in defaults
        ]
    if sequence_parallel:
        # The author's opt-in to a group lane (cr-068): the degrees the class's attention
        # shards at. Closed and optional; a slot without it is never sharded.
        doc["sequence_parallel"] = {"degrees": list(sequence_parallel)}
    if fusion != "refuse":
        # The fused-glue consent (h3a-015); the conservative default is the absent key.
        doc["fusion"] = fusion
    return doc


# -------------------------------------------------------------------- fixed-point scan

_MACHINE_PATH = re.compile(r"(^/|^[A-Za-z]:[\\/]|/home/|\\Users\\|site-packages|\.venv/)")
_TIMESTAMP = re.compile(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}")
_DIGEST = re.compile(r"^(sha256:)?[0-9a-f]{64}$")
_SHA256 = re.compile(r"sha256:[0-9a-f]{64}")

#: Keys that can only carry a fact from a LATER clock: a binding, a candidate, a build.
_FORBIDDEN_KEYS = frozenset(
    {
        "release",
        "built_at",
        "installation_id",
        "timestamp",
        "created_at",
        "generated_at",
        "model",
        "repo",
        "lane",
        "checkpoint",
        "checkpoint_ref",
        "artifact",
        "snapshot",
        "candidate",
        "destinations",
    }
)


def _binding_values(bindings: Mapping[str, Mapping[str, object]]) -> frozenset[str]:
    """The binding SELECTION spellings a package.toml states — model, release, and the
    composed ref, because `model` and `release` are stored apart but named together
    everywhere else. The `lane` member is deliberately NOT scanned (cr-077): lane names
    are format vocabulary ("bf16", "fp8") that legitimately coincides with author surface
    names — a quantize job's weights outputs ARE the lanes it derives — and an interface
    containing the word gains no pointer to the mutable selection."""
    values: set[str] = set()
    for entry in bindings.values():
        model, release = entry.get("model"), entry.get("release")
        if isinstance(model, str):
            values.add(model)
            if isinstance(release, str):
                values.add(release)
                values.add(f"{model}@{release}")
    return frozenset(values)


def _fixed_point(value: Json, bindings: frozenset[str], path: str = "") -> None:
    """The package interface's fixed point, proven over the built document (§1.0)."""
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key == "operation_identity" and re.fullmatch(
                r"\.(?:entrypoints|jobs)\[\d+\]\.invocable", path
            ):
                # This belongs to the deterministic operation cache, not package
                # installation admission or source snapshot identity.
                if not isinstance(item, str) or re.fullmatch(r"sha256:[0-9a-f]{64}", item) is None:
                    raise ConformanceError("invalid operation memo identity", code="fixed_point")
                continue
            if key == "default_ladder" and re.fullmatch(
                r"\.(?:entrypoints|jobs)\[\d+\]\.models\[\d+\]", path
            ):
                default_ladder(item)
                continue
            if key in _FORBIDDEN_KEYS:
                raise ConformanceError(
                    f"{path}.{key}: the package interface is the SOURCE-STABLE surface — release "
                    "digests, timestamps, machine paths, mutable bindings and "
                    "candidate-selected facts live on other clocks (cr-004)",
                    code="fixed_point",
                    fields=[f"{path}.{key}"],
                )
            _fixed_point(item, bindings, f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, item in enumerate(value):
            _fixed_point(item, bindings, f"{path}[{index}]")
    elif isinstance(value, str):
        if value in bindings:
            raise ConformanceError(
                f"{path}: {value!r} is a package.toml binding value — bindings are MUTABLE "
                "and post-deploy authority is the hub's, so no binding can key a release",
                code="fixed_point",
                fields=[path],
            )
        for pattern, why in (
            (_MACHINE_PATH, "a machine path"),
            (_TIMESTAMP, "a build timestamp"),
            (_DIGEST, "a digest of something on another clock"),
        ):
            if pattern.search(value):
                raise ConformanceError(
                    f"{path}: {value!r} is {why}; the interface must reproduce byte-identically "
                    "in every environment of one release",
                    code="fixed_point",
                    fields=[path],
                )


# ------------------------------------------------------------------------ identity + file


def canonical_bytes(doc: Mapping[str, Json]) -> bytes:
    """The semantic interface bytes used for identity and control-plane transport."""
    return canonical.write(dict(doc))


def package_interface_digest(doc: Mapping[str, Json]) -> str:
    """The package interface's formatting-invariant semantic identity."""
    return canonical.digest(dict(doc))


def job_descriptor_id(body: Mapping[str, Json], name: str) -> str:
    """`job_descriptor_id` — sha256 of ONE job's exact callable, schemas, effects, resource
    caps and publication contract (README §4).

    DERIVED, never stored in the document: a digest of a subject that lives on this same
    clock is still a digest, and `_fixed_point` refuses one anywhere in the body — for the
    good reason that a reader who finds a `sha256:` string in a source-stable surface has no
    way to know which clock it came from. The runner computes it when it needs it, from the
    document, and every environment of one release computes the same one.
    """
    jobs = body.get("jobs", [])
    assert isinstance(jobs, list)
    for entry in jobs:
        assert isinstance(entry, dict)
        if entry["name"] == name:
            encoded = canonical.write(entry)
            return (
                "sha256:" + hashlib.sha256(b"cozy.runtime.job-descriptor\0" + encoded).hexdigest()
            )
    raise ConformanceError(
        f"this release registers no job named {name!r}: it has "
        f"{', '.join(str(j['name']) for j in jobs if isinstance(j, dict)) or 'no jobs'}",
        code="unknown_job",
    )


#: Facts only the installed environment can state: a memoized callable's operation identity
#: hashes its installed helpers, so a source reading never carries it.
INSTALLED_FACTS = frozenset({"operation_identity", "operation_identity_unavailable"})


def source_facts(doc: Mapping[str, Json]) -> dict[str, Json]:
    """The document as a source reading knows it: installed-environment facts removed."""

    def callable_facts(entry: Json) -> Json:
        if not isinstance(entry, dict) or not isinstance(entry.get("invocable"), dict):
            return entry
        invocable = {k: v for k, v in entry["invocable"].items() if k not in INSTALLED_FACTS}
        return {**entry, "invocable": invocable}

    return {
        key: [callable_facts(entry) for entry in value]
        if key in ("entrypoints", "jobs") and isinstance(value, list)
        else value
        for key, value in doc.items()
    }


def compare(left: Mapping[str, Json], right: Mapping[str, Json], names: tuple[str, str]) -> None:
    """Refuse unless two package interfaces have the same content."""
    if canonical_bytes(left) == canonical_bytes(right):
        return
    detail = "; ".join(_differences(left, right)) or "the documents differ"
    raise StalePackageInterface(
        f"{names[0]} and {names[1]} are different documents: {detail}",
        code="stale_package_interface",
    )


def _differences(left: Json, right: Json, path: str = "") -> list[str]:
    """The first few disagreements, deepest key named — a reviewer needs the FIELD."""
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        out: list[str] = []
        for key in sorted(set(left) | set(right)):
            if key not in left:
                out.append(f"{path}.{key} only in the second")
            elif key not in right:
                out.append(f"{path}.{key} only in the first")
            else:
                out += _differences(left[key], right[key], f"{path}.{key}")
            if len(out) >= 4:
                break
        return out[:4]
    if (
        isinstance(left, Sequence)
        and isinstance(right, Sequence)
        and not isinstance(left, (str, bytes))
        and not isinstance(right, (str, bytes))
    ):
        if len(left) != len(right):
            return [f"{path or 'root'}: {len(left)} vs {len(right)} entries"]
        out = []
        for index, (a, b) in enumerate(zip(left, right, strict=True)):
            out += _differences(a, b, f"{path}[{index}]")
            if len(out) >= 4:
                break
        return out[:4]
    if left != right:
        return [f"{path or 'root'}: {left!r} vs {right!r}"]
    return []


def parse(raw: bytes, source: str = "package interface") -> PackageInterface:
    """Decode one interface once, refusing a malformed document by the field it names."""
    try:
        parsed = canonical.parse(raw)
    except canonical.CanonicalError as exc:
        raise StalePackageInterface(f"{source}: {exc}", code="malformed_package_interface") from exc
    if isinstance(parsed, dict) and "format" in parsed and parsed["format"] != SCHEMA:
        raise StalePackageInterface(
            f"{source}: format {parsed['format']!r} is not {SCHEMA!r}", code="unknown_format"
        )
    try:
        interface = msgspec.convert(parsed, PackageInterface, strict=True)
    except msgspec.ValidationError as exc:
        raise StalePackageInterface(f"{source}: {exc}", code="malformed_package_interface") from exc
    _validate(interface, source)
    return interface


def read_bytes(raw: bytes, source: str = "package interface") -> dict[str, Any]:
    """The validated document as untyped JSON, for readers not yet moved to `parse`."""
    parse(raw, source)
    parsed: dict[str, Any] = canonical.parse(raw)
    return parsed


def publish(path: Path, raw: bytes) -> None:
    """Publish interface bytes at `path` for an executor to read, never partially.

    Sibling replicas of one installation activate concurrently (#725) and publish the same
    path while executors read it, so the bytes land by rename. The mode is set, not masked:
    the executor reads as its own uid."""

    try:
        if path.stat().st_mode & 0o777 == 0o644 and path.read_bytes() == raw:
            return
    except OSError:
        pass
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(dir=path.parent, prefix=".package-interface-")
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(name, 0o644)
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def _validate(interface: PackageInterface, source: str) -> None:
    """The cross-field rules the struct types cannot state. One current grammar, no legacy."""
    for kind, entries in (("entrypoint", interface.entrypoints), ("job", interface.jobs)):
        collection = kind + "s"
        names: set[str] = set()
        for index, item in enumerate(entries):
            path = f"{collection}[{index}]"
            if not item.name or item.name in names:
                _invalid(source, f"{path}.name", "must be non-empty and unique")
            names.add(item.name)
            if (kind == "job") != (item.publishes is not msgspec.UNSET):
                _invalid(source, f"{path}.publishes", "is required exactly on jobs")
            if kind != "job" and (
                item.accelerator is not msgspec.UNSET or item.weights_outputs is not msgspec.UNSET
            ):
                _invalid(source, path, "only jobs declare accelerator or weights outputs")
            _schema(item.request, source, f"{path}.request")
            _schema(item.result, source, f"{path}.result")
            if item.assets is not msgspec.UNSET:
                _assets_slot(item.assets, item.request, source, f"{path}.assets")
            if item.invocable is not msgspec.UNSET:
                invocable_interface.validate(item.invocable, item.request, item.result, kind=kind)
            for model_index, slot in enumerate(item.models):
                _model_slot(slot, item.name, source, f"{path}.models[{model_index}]")
            outputs = () if item.weights_outputs is msgspec.UNSET else item.weights_outputs
            output_ids = [output.output_id for output in outputs]
            if len(set(output_ids)) != len(output_ids):
                _invalid(source, f"{path}.weights_outputs", "output ids must be unique")


def _model_slot(slot: ModelSlot, name: str, source: str, path: str) -> None:
    for consent, value, vocabulary in (
        ("encoded_leaves", slot.encoded_leaves, ENCODED_LEAVES),
        ("fusion", slot.fusion, FUSION),
    ):
        if value not in vocabulary:
            _invalid(source, f"{path}.{consent}", "must be " + " or ".join(vocabulary))
    degrees = () if slot.sequence_parallel is msgspec.UNSET else slot.sequence_parallel.degrees
    if slot.sequence_parallel is not msgspec.UNSET and (
        not degrees or list(degrees) != sorted(set(degrees))
    ):
        _invalid(source, f"{path}.sequence_parallel.degrees", "must be non-empty, sorted, unique")
    if slot.default_ladder is not msgspec.UNSET and any(
        gpus > 1 and gpus not in degrees
        for _, gpus, _ in default_ladder(msgspec.to_builtins(slot.default_ladder))
    ):
        _invalid(source, path, "default GPU count is not a declared sequence-parallel degree")
    prefix = f"{name}.models."
    if not slot.path.startswith(prefix) or not slot.path[len(prefix) :]:
        _invalid(source, f"{path}.path", f"must start with {prefix!r}")
    if not slot.class_name:
        _invalid(source, f"{path}.class", "must be a non-empty string")


def _assets_slot(slot: AssetsSlot, request: Json, source: str, path: str) -> None:
    if not slot.parameter.isidentifier():
        _invalid(source, path, "assets parameter must be an identifier")
    fields = request.get("fields", []) if isinstance(request, dict) else []
    selected = (
        [f for f in fields if isinstance(f, dict) and f.get("name") == slot.parameter]
        if isinstance(fields, list)
        else []
    )
    if len(selected) != 1 or selected[0].get("type") != schema.render(list[AssetInputValue]):
        _invalid(source, path, "assets parameter must name the declared occurrence records")
    kinds = [row.kind for row in slot.kinds]
    if len(set(kinds)) != len(kinds):
        _invalid(source, path, "assets kinds must be unique")
    media: set[str] = set()
    for row in slot.kinds:
        if slot.view == "decoded" and row.kind == "file":
            _invalid(source, path, "decoded Assets does not include generic file handles")
        if not row.media_types and row.kind != "file":
            _invalid(source, path, "a media kind needs explicit MIME types")
        for mime in row.media_types:
            if mime != mime.strip().lower() or "/" not in mime or mime in media:
                _invalid(source, path, "assets MIME types must be distinct and canonical")
            media.add(mime)
        if row.prepare is not msgspec.UNSET and (
            row.kind != "image"
            or slot.view != "decoded"
            or row.prepare.profile != IMAGE_PREPARATION_PROFILE
            or (row.prepare.max_edge is msgspec.UNSET and row.prepare.max_pixels is msgspec.UNSET)
        ):
            _invalid(source, path, "image preparation needs decoded image Assets and a cap")


def _schema(value: Json, source: str, path: str, union_tag_field: str | None = None) -> None:
    if isinstance(value, str):
        if value not in _SCALARS:
            _invalid(source, path, f"unknown scalar {value!r}")
        return
    if not isinstance(value, dict):
        _invalid(source, path, "must be a scalar or schema object")
    forms = set(value) & _SCHEMA_FORMS
    if len(forms) != 1:
        _invalid(source, path, "must carry exactly one schema form")
    form = next(iter(forms))
    if form == "fields":
        struct = _fields(value, {"fields"}, source, path)
        tag = struct.get("tag")
        tag_field = struct.get("tag_field")
        active_tag_field: Json
        if union_tag_field is not None:
            if tag is None or tag_field is not None:
                _invalid(source, path, "a tagged-union member carries only its tag")
            active_tag_field = union_tag_field
        elif (tag is None) != (tag_field is None):
            _invalid(source, path, "a standalone tag and tag_field must appear together")
        else:
            active_tag_field = tag_field
        if active_tag_field is not None and (
            not isinstance(active_tag_field, str) or not active_tag_field
        ):
            _invalid(source, f"{path}.tag_field", "must be a non-empty string")
        fields = struct["fields"]
        if not isinstance(fields, list):
            _invalid(source, f"{path}.fields", "must be an array")
        names: set[str] = set()
        for index, value_field in enumerate(fields):
            field_path = f"{path}.fields[{index}]"
            field = _fields(value_field, {"name", "type"}, source, field_path)
            name = field["name"]
            if not isinstance(name, str) or not name or name in names or name == active_tag_field:
                _invalid(
                    source,
                    f"{field_path}.name",
                    "must be non-empty, unique, and not repeat the tag field",
                )
            names.add(name)
            wire = field.get("wire")
            if wire is not None and wire not in {"optional", "omissible"}:
                _invalid(
                    source,
                    f"{field_path}.wire",
                    "may only be optional or omissible; required is the default",
                )
            _schema(field["type"], source, f"{field_path}.type")
            if "constraints" in field:
                # msgspec constraints ride through VERBATIM: a consumer that does not
                # understand one ignores it, and a closed list here would refuse a
                # package over a constraint Creator has simply not learned to read.
                constraints = field["constraints"]
                if not isinstance(constraints, dict) or not constraints:
                    _invalid(source, f"{field_path}.constraints", "must be a non-empty object")
            if "asset_bound" in field and not isinstance(field["asset_bound"], dict):
                _invalid(source, f"{field_path}.asset_bound", "must be an object")
        return
    if form == "union":
        union = _fields(value, {"union"}, source, path)
        branches = union["union"]
        if not isinstance(branches, list) or not branches:
            _invalid(source, f"{path}.union", "must be a non-empty array")
        tag_field = union.get("tag_field")
        if tag_field is not None and (not isinstance(tag_field, str) or not tag_field):
            _invalid(source, f"{path}.tag_field", "must be a non-empty string")
        for index, branch in enumerate(branches):
            _schema(branch, source, f"{path}.union[{index}]", tag_field)
        return
    exact = _fields(value, {form}, source, path)
    child = exact[form]
    if form == "list":
        _schema(child, source, f"{path}.list")
    elif form == "map":
        entry = _fields(child, {"key", "value"}, source, f"{path}.map")
        _schema(entry["key"], source, f"{path}.map.key")
        _schema(entry["value"], source, f"{path}.map.value")
    elif form == "tuple":
        if not isinstance(child, list) or not child:
            _invalid(source, f"{path}.tuple", "must be a non-empty array")
        for index, member in enumerate(child):
            _schema(member, source, f"{path}.tuple[{index}]")
    elif form == "literal":
        if not isinstance(child, list):
            _invalid(source, f"{path}.literal", "must be an array")
    elif not isinstance(child, str) or not child:
        _invalid(source, f"{path}.{form}", "must be a non-empty string")


def _fields(value: Json, required: set[str], source: str, path: str) -> dict[str, Json]:
    """An object carrying `required`; additive fields from newer writers are ignored."""
    if not isinstance(value, dict):
        _invalid(source, path, "must be an object")
    missing = sorted(required - set(value))
    if missing:
        _invalid(source, path, ", ".join(f"missing {name}" for name in missing))
    return cast(dict[str, Json], value)


def _invalid(source: str, path: str, detail: str) -> NoReturn:
    raise StalePackageInterface(f"{source}: {path} {detail}", code="malformed_package_interface")


_SCALARS = frozenset({"bool", "float", "int", "null", "str"})
_SCHEMA_FORMS = frozenset(
    {"asset", "fields", "input", "list", "literal", "map", "opaque", "tuple", "union"}
)
#: The asset-bound facts this Runtime consumes; a newer writer's additional bounds are ignored.
ASSET_BOUND_KEYS = frozenset({"max_bytes", "max_decoded_bytes", "media_types"})


def counts(body: Mapping[str, Json]) -> dict[str, int]:
    """Aggregates ride the listing, so no follow-up call is needed."""
    entrypoints = body.get("entrypoints", [])
    jobs = body.get("jobs", [])
    assert isinstance(entrypoints, list) and isinstance(jobs, list)
    slots = 0
    for doc in [*entrypoints, *jobs]:
        assert isinstance(doc, dict)
        models = doc.get("models", [])
        assert isinstance(models, list)
        slots += len(models)
    return {
        "entrypoints": len(entrypoints),
        "jobs": len(jobs),
        "model_bindings": slots,
    }
