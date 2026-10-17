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
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal, NoReturn, cast

import msgspec

from cozy_runtime.author import ConformanceError, Surface, WeightsOutput
from cozy_runtime.author._markers import (
    AUDIO_PREPARATION_PROFILE,
    IMAGE_PREPARATION_PROFILE,
    VIDEO_PREPARATION_PROFILE,
    AudioPreparation,
    ImagePreparation,
    VideoPreparation,
)
from cozy_runtime.author._media import KIND_MEDIA
from cozy_runtime.author._model import ENCODED_LEAVES, Model, component_use
from cozy_runtime.author._model_defaults import model_default
from cozy_runtime.author._signature import AssetInputValue, AssetsBinding
from cozy_runtime.internal import canonical, invocable_interface, schema
from cozy_runtime.internal.canonical import Json

if TYPE_CHECKING:
    from cozy_runtime.internal.discovery import Discovered

SCHEMA = "cozy.package.interface/1"
DIGEST_KEY = "package_interface_digest"
FILENAME = "metadata/package-interface.json"

#: The top-level fields every interface carries. Readers ignore additional fields.
TOP_LEVEL_KEYS = frozenset({"format", "application", "entrypoints"})


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


class SequenceParallel(msgspec.Struct, frozen=True):
    degrees: tuple[Annotated[int, msgspec.Meta(ge=2)], ...]


class ModelSlot(msgspec.Struct, frozen=True, kw_only=True, omit_defaults=True):
    path: str
    class_name: str = msgspec.field(name="class")
    component_use: dict[str, tuple[str, ...]]
    encoded_leaves: str = "refuse"
    #: `[org/]model@release`: the release a run takes when it names none for this slot
    default: str | msgspec.UnsetType = msgspec.UNSET
    min_bits: Literal[16, 8, 4] = 8
    phase: Literal["compute", "decode"] = "compute"
    sequence_parallel: SequenceParallel | msgspec.UnsetType = msgspec.UNSET
    #: the components this callable runs; unset means all of them
    components: tuple[str, ...] | msgspec.UnsetType = msgspec.UNSET
    #: a `Weights[M]` argument: raw checkpoint data, never constructed
    weights: bool = False
    #: built, and its weights fetched, when the function first reads it (`on_demand=`)
    on_demand: bool = False

    @property
    def parameter(self) -> str:
        return self.path.rpartition(".")[2]


class ImageFit(msgspec.Struct, frozen=True, kw_only=True, forbid_unknown_fields=True):
    profile: Literal["image-fit/1"]
    max_edge: Annotated[int, msgspec.Meta(gt=0)] | None = None
    max_pixels: Annotated[int, msgspec.Meta(gt=0)] | None = None


_Seconds = Annotated[int, msgspec.Meta(gt=0)] | Annotated[float, msgspec.Meta(gt=0)]


class Trim(msgspec.Struct, frozen=True, kw_only=True, forbid_unknown_fields=True):
    profile: Literal["video-trim/1", "audio-trim/1"]
    excess: Literal["refuse", "truncate"]
    min_seconds: _Seconds | None = None
    max_seconds: _Seconds | None = None


#: Profile -> its grammar and the marker that enforces it. A profile not listed here is a
#: newer writer's, and readers ignore it.
PREPARATION_PROFILES: dict[
    str, tuple[type[msgspec.Struct], type[ImagePreparation | VideoPreparation | AudioPreparation]]
] = {
    IMAGE_PREPARATION_PROFILE: (ImageFit, ImagePreparation),
    VIDEO_PREPARATION_PROFILE: (Trim, VideoPreparation),
    AUDIO_PREPARATION_PROFILE: (Trim, AudioPreparation),
}


class AssetKind(msgspec.Struct, frozen=True, kw_only=True, omit_defaults=True):
    kind: Literal["image", "video", "audio", "file"]
    media_types: tuple[str, ...]
    max_count: Annotated[int, msgspec.Meta(ge=0)] | msgspec.UnsetType = msgspec.UNSET
    max_bytes: Annotated[int, msgspec.Meta(gt=0)] | msgspec.UnsetType = msgspec.UNSET
    max_decoded_bytes: Annotated[int, msgspec.Meta(gt=0)] | msgspec.UnsetType = msgspec.UNSET
    prepare: dict[str, JsonValue] | msgspec.UnsetType = msgspec.UNSET


class AssetsSlot(msgspec.Struct, frozen=True, kw_only=True, omit_defaults=True):
    parameter: str
    kinds: Annotated[tuple[AssetKind, ...], msgspec.Meta(min_length=1, max_length=4)]


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
    publishes: bool = False
    #: written only as false: CPU only. Absent: the machine's GPU when it has one.
    accelerator: bool | msgspec.UnsetType = msgspec.UNSET
    weights_outputs: (
        Annotated[tuple[WeightsOutputDeclaration, ...], msgspec.Meta(min_length=1, max_length=16)]
        | msgspec.UnsetType
    ) = msgspec.UNSET


class PackageInterface(msgspec.Struct, frozen=True, kw_only=True):
    format: str
    application: Annotated[str, msgspec.Meta(min_length=1)]
    entrypoints: tuple[CallableDoc, ...]

    def function(self, name: str) -> CallableDoc | None:
        return next((entry for entry in self.entrypoints if entry.name == name), None)


# ------------------------------------------------------------------------ the document


def build(found: Discovered, *, installed: bool = False) -> dict[str, Json]:
    """Derive one package interface from the frozen (imported) application."""
    docs = [_callable(surface) for surface in found.surfaces if not surface.hidden]
    return assemble(found.application, docs, installed=installed)


def assemble(
    application: str, docs: Sequence[dict[str, Json]], *, installed: bool = False
) -> dict[str, Json]:
    """One package interface from its callable documents, whichever reader produced them.

    `installed` describes a published wheel, so it holds that release to the rules it was
    published under."""
    for doc in docs:
        declared = doc.get("invocable")
        if isinstance(declared, dict) and "memo_requires" in declared:
            invocable_interface.check_memo_requires(
                msgspec.convert(declared, invocable_interface.Invocable, strict=True),
                doc["request"],
                authoring=not installed,
            )
    body: dict[str, Json] = {
        "format": SCHEMA,
        "application": application,
        "entrypoints": sorted(docs, key=_by_name),
    }
    assert set(body) == TOP_LEVEL_KEYS, sorted(set(body) ^ TOP_LEVEL_KEYS)
    _fixed_point(body)
    return body


def _by_name(doc: Json) -> str:
    assert isinstance(doc, dict)
    name = doc["name"]
    assert isinstance(name, str)
    return name


def _callable(surface: Surface) -> dict[str, Json]:
    doc = callable_doc(
        name=surface.name,
        payload_type=surface.payload_type,
        result_type=surface.result_type,
        models=[
            model_slot(
                binding.path,
                ModelSpec.of(binding.model_class),
                default=surface.model_defaults.get(binding.param),
                uses=surface.model_components.get(binding.param, ()),
                weights=binding.weights,
                on_demand=binding.param in surface.model_on_demand,
            )
            for binding in surface.model_bindings
        ],
        assets=surface.assets_binding,
        publishes=surface.publishes,
        weights_outputs=surface.weights_outputs,
        internal=surface.internal,
        accelerator=surface.accelerator,
    )
    doc["invocable"] = invocable_interface.metadata(surface)
    if surface.invocable:
        invocable_interface.project_serving_models(doc)
    return doc


def callable_doc(
    *,
    name: str,
    payload_type: object,
    result_type: object,
    models: Sequence[dict[str, Json]],
    assets: AssetsBinding | None,
    publishes: bool,
    weights_outputs: Sequence[WeightsOutput],
    internal: bool = False,
    accelerator: bool = True,
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
            if (preparation := assets.preparation(cls.kind)) is not None:
                row["prepare"] = dict(preparation.descriptor())
            kinds.append(row)
        doc["assets"] = {"parameter": assets.parameter, "kinds": kinds}
    if publishes:
        doc["publishes"] = True
    if not accelerator:
        doc["accelerator"] = False
    if weights_outputs:
        doc["weights_outputs"] = [
            {
                "output_id": output.name,
                "mime_type": "application/vnd.cozy.model-manifest",
                "max_bytes": output.max_new_bytes,
            }
            for output in sorted(weights_outputs, key=lambda row: row.name)
        ]
    return doc


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """A Model class's interface facts, read by import (`of`) or from source alone."""

    name: str
    encoded_leaves: str = "refuse"
    min_bits: int = 8
    phase: str = "compute"
    sequence_parallel: tuple[int, ...] = ()
    components: Mapping[str, tuple[str, ...]] = field(default_factory=dict)

    @classmethod
    def of(cls, model: type[Model[Any]]) -> ModelSpec:
        return cls(
            model.__name__,
            model.__encoded_leaves__,
            model.__min_bits__,
            model.__phase__,
            tuple(model.__sequence_parallel__),
            component_use(model),
        )


def model_slot(
    path: str,
    spec: ModelSpec,
    *,
    default: str | None = None,
    uses: Sequence[str] = (),
    weights: bool = False,
    on_demand: bool = False,
) -> dict[str, Json]:
    """One model slot. The PATH is the binding point, the CLASS carries `encoded_leaves` and
    the precision tolerances, and the per-public-method component sets are the
    ComponentUseContracts. `default` is an unresolved release; the machine resolves it and
    picks its lane. `uses` names the components the callable runs (`components`); empty
    means all. A `weights` slot is raw checkpoint data, so `uses` names its checkpoint
    components."""
    doc: dict[str, Json] = {
        "path": path,
        "class": spec.name,
        "encoded_leaves": spec.encoded_leaves,
        # Declaration order encodes no control flow and no residency choice (§1.1), so the
        # contract sorts — a reordered method body is not a surface change.
        "component_use": {
            method: sorted(names) for method, names in sorted(spec.components.items())
        },
    }
    if default:
        doc["default"] = default
    if spec.min_bits != 8:
        doc["min_bits"] = spec.min_bits
    if spec.phase != "compute":
        doc["phase"] = spec.phase
    if uses:
        declared = {name for names in spec.components.values() for name in names}
        unknown = sorted(set(uses) - declared)
        if unknown and not weights:
            raise ConformanceError(
                f"{path}: components names {', '.join(unknown)}, which no "
                f"@uses_components method of {spec.name} declares",
                code="model_components",
                fields=unknown,
            )
        doc["components"] = sorted(set(uses))
    if spec.sequence_parallel:
        # The author's opt-in to a group lane (cr-068): the degrees the class's attention
        # shards at. Closed and optional; a slot without it is never sharded.
        doc["sequence_parallel"] = {"degrees": list(spec.sequence_parallel)}
    if weights:
        doc["weights"] = True
    if on_demand:
        doc["on_demand"] = True
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


def _fixed_point(value: Json, path: str = "") -> None:
    """The package interface's fixed point, proven over the built document (§1.0)."""
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key == "operation_identity" and re.fullmatch(
                r"\.entrypoints\[\d+\]\.invocable", path
            ):
                # This belongs to the deterministic operation cache, not package
                # installation admission or source snapshot identity.
                if not isinstance(item, str) or re.fullmatch(r"sha256:[0-9a-f]{64}", item) is None:
                    raise ConformanceError("invalid operation memo identity", code="fixed_point")
                continue
            if key == "default" and re.fullmatch(r"\.entrypoints\[\d+\]\.models\[\d+\]", path):
                model_default(item)
                continue
            if key in _FORBIDDEN_KEYS:
                raise ConformanceError(
                    f"{path}.{key}: the package interface is the SOURCE-STABLE surface — release "
                    "digests, timestamps, machine paths, mutable bindings and "
                    "candidate-selected facts live on other clocks (cr-004)",
                    code="fixed_point",
                    fields=[f"{path}.{key}"],
                )
            _fixed_point(item, f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, item in enumerate(value):
            _fixed_point(item, f"{path}[{index}]")
    elif isinstance(value, str):
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


def descriptor_id(body: Mapping[str, Json], name: str) -> str:
    """sha256 of ONE function's exact interface entry: its operation identity.

    Derived, never stored: `_fixed_point` refuses a digest inside the source-stable body.
    """
    entries = body.get("entrypoints", [])
    assert isinstance(entries, list)
    for entry in entries:
        assert isinstance(entry, dict)
        if entry["name"] == name:
            encoded = canonical.write(entry)
            return "sha256:" + hashlib.sha256(b"cozy.runtime.descriptor\0" + encoded).hexdigest()
    raise ConformanceError(
        f"this release registers no function named {name!r}: it has "
        f"{', '.join(str(e['name']) for e in entries if isinstance(e, dict)) or 'none'}",
        code="unknown_function",
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
        if key == "entrypoints" and isinstance(value, list)
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
    names: set[str] = set()
    for index, item in enumerate(interface.entrypoints):
        path = f"entrypoints[{index}]"
        if not item.name or item.name in names:
            _invalid(source, f"{path}.name", "must be non-empty and unique")
        names.add(item.name)
        if item.accelerator is True:
            _invalid(source, f"{path}.accelerator", "is written only as false")
        _schema(item.request, source, f"{path}.request")
        _schema(item.result, source, f"{path}.result")
        if item.assets is not msgspec.UNSET:
            _assets_slot(item.assets, item.request, source, f"{path}.assets")
        outputs = () if item.weights_outputs is msgspec.UNSET else item.weights_outputs
        if item.invocable is not msgspec.UNSET:
            invocable_interface.validate(item.invocable, item.request, item.result)
            if item.invocable.memoize and (outputs or item.publishes):
                _invalid(
                    source, path, "a function that writes weights or publishes is not memoized"
                )
        for model_index, slot in enumerate(item.models):
            _model_slot(slot, item.name, source, f"{path}.models[{model_index}]")
        output_ids = [output.output_id for output in outputs]
        if len(set(output_ids)) != len(output_ids):
            _invalid(source, f"{path}.weights_outputs", "output ids must be unique")


def _model_slot(slot: ModelSlot, name: str, source: str, path: str) -> None:
    if slot.encoded_leaves not in ENCODED_LEAVES:
        _invalid(source, f"{path}.encoded_leaves", "must be " + " or ".join(ENCODED_LEAVES))
    degrees = () if slot.sequence_parallel is msgspec.UNSET else slot.sequence_parallel.degrees
    if slot.sequence_parallel is not msgspec.UNSET and (
        not degrees or list(degrees) != sorted(set(degrees))
    ):
        _invalid(source, f"{path}.sequence_parallel.degrees", "must be non-empty, sorted, unique")
    if slot.default is not msgspec.UNSET:
        model_default(slot.default)
    if slot.components is not msgspec.UNSET and (
        not slot.components
        or len(set(slot.components)) != len(slot.components)
        or not (
            slot.weights
            or set(slot.components) <= {c for names in slot.component_use.values() for c in names}
        )
    ):
        _invalid(source, f"{path}.components", "must be non-empty, unique and within component_use")
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
        if not row.media_types and row.kind != "file":
            _invalid(source, path, "a media kind needs explicit MIME types")
        for mime in row.media_types:
            if mime != mime.strip().lower() or "/" not in mime or mime in media:
                _invalid(source, path, "assets MIME types must be distinct and canonical")
            media.add(mime)
        if row.prepare is not msgspec.UNSET:
            _preparation(row.prepare, source, f"{path}.prepare", frozenset({row.kind}))


def preparation_members(
    value: Json,
) -> tuple[type[ImagePreparation | VideoPreparation | AudioPreparation], dict[str, Json]] | None:
    """A known profile's marker and its grammar members, or None for a newer writer's profile.

    A known profile admits exactly its grammar: an extra member refuses.
    """
    if not isinstance(value, dict) or not isinstance(value.get("profile"), str):
        raise ConformanceError("must be an object naming its profile", code="preparation")
    known = PREPARATION_PROFILES.get(value["profile"])
    if known is None:
        return None
    grammar, marker = known
    try:
        members = msgspec.structs.asdict(msgspec.convert(value, grammar, strict=True))
    except msgspec.ValidationError as exc:
        raise ConformanceError(str(exc), code="preparation") from exc
    members = {key: item for key, item in members.items() if key != "profile" and item is not None}
    marker(**members)
    return marker, members


def _preparation(value: Json, source: str, path: str, kinds: frozenset[str]) -> None:
    """A known profile's members, and that it prepares a kind the declaration carries."""
    try:
        known = preparation_members(value)
    except (ConformanceError, TypeError) as exc:
        _invalid(source, path, f"is not a valid preparation: {exc}")
    if known is not None and known[0].kind not in kinds:
        _invalid(source, path, f"{value['profile']} does not prepare {sorted(kinds)} inputs")


def _schema_asset_kinds(value: Json) -> frozenset[str]:
    """The asset kinds a rendered type reaches through containers, stopping at records."""
    if not isinstance(value, dict):
        return frozenset()
    if isinstance(value.get("asset"), str):
        return frozenset({value["asset"]})
    if isinstance(value.get("union"), list):
        members: list[Json] = value["union"]
    elif isinstance(value.get("tuple"), list):
        members = value["tuple"]
    elif "list" in value:
        members = [value["list"]]
    elif isinstance(value.get("map"), dict):
        members = [value["map"].get("value")]
    else:
        return frozenset()
    return frozenset(kind for member in members for kind in _schema_asset_kinds(member))


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
            if "asset_bound" in field:
                asset_bound = field["asset_bound"]
                if not isinstance(asset_bound, dict):
                    _invalid(source, f"{field_path}.asset_bound", "must be an object")
                if "prepare" in asset_bound:
                    _preparation(
                        asset_bound["prepare"],
                        source,
                        f"{field_path}.asset_bound.prepare",
                        _schema_asset_kinds(field["type"]),
                    )
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
    assert isinstance(entrypoints, list)
    slots = 0
    for doc in entrypoints:
        assert isinstance(doc, dict)
        models = doc.get("models", [])
        assert isinstance(models, list)
        slots += len(models)
    return {"functions": len(entrypoints), "model_bindings": slots}
