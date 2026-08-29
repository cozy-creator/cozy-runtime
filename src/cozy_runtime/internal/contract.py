"""The ModelConstructionContract and its Attestation — the binding-clock documents.

The PackageDescriptor (cr-003) is the SOURCE-STABLE surface; this is the other clock.
One contract per EXACT candidate — (package release, model factory, artifact snapshot,
artifact configs, hardware variant) — derived by EXECUTING the author's `load()` under the
cr-004 harness, never by trusting a declaration. It is stored apart from
the PackageDescriptor and never merges into it.

Two properties are structural rather than checked:

**Fusion, permutation, re-keying, quantization and subclass reconstruction are
UNREPRESENTABLE.** A `LogicalTensor` carries ONE name. There is no second name, no source
key, no part list, no transform, no scale reference and no converter — so a contract cannot
even spell "this destination is built from those two checkpoint keys, transposed". Those are
AOT artifact work (§1.1). `_prove_unrepresentable()` runs at IMPORT and asserts the exact
field set of every row type against a frozen table, so the property cannot drift in by an
innocent-looking field.

**Component-use declarations are NOT construction identity.** The author's
`@uses_components` map is descriptor content (cr-003). Its validation against the derived
destinations rides the ATTESTATION, outside `contract_digest` — so widening a method's
declared set can never re-key a construction contract.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from typing import Any, Literal, cast

from cozy_runtime.internal import canonical
from cozy_runtime.internal.canonical import Json

CONTRACT_SCHEMA = "cozy.runtime.model_construction_contract.v1"
ATTESTATION_SCHEMA = "cozy.runtime.model_construction_attestation.v1"

#: The dtype-diet rule, shared with the checkpoint header's plain/1-absence rule (§1.1):
#: dtype is serialized ONLY where it deviates from bf16. Absence means bf16.
DEFAULT_DTYPE = "bfloat16"

TensorKind = Literal["parameter", "buffer"]


class ContractError(Exception):
    """A contract refusal: the derived facts do not form an admissible contract."""

    def __init__(self, message: str, *, code: str, fields: Sequence[str] = ()) -> None:
        super().__init__(message)
        self.code = code
        self.fields = tuple(fields)


# ------------------------------------------------------------------------------ rows


@dataclass(frozen=True, slots=True)
class Candidate:
    """The EXACT candidate a contract is derived for. Every term is a construction fact;
    a mutable binding or a method-use declaration is deliberately absent."""

    release: str
    application: str
    model_class: str
    factory: str
    snapshot: str
    config_digest: str
    hardware_variant: str


@dataclass(frozen=True, slots=True)
class LogicalTensor:
    """One logical tensor the construction demands, and the ONE destination it lands in.

    `key` is both. There is no source key, so no re-key is expressible; no part list, so no
    fusion; no transform, so no permutation; no encoding, so no quantization. A checkpoint
    either supplies this exact key at this exact shape or the match refuses (§1.1).
    """

    key: str
    component: str
    shape: tuple[int, ...]
    kind: TensorKind
    module_class: str
    alias_group: int | None
    dtype: str | None
    """Present only where it deviates from bf16 (the dtype-diet rule). Where present it is
    the destination's COMPUTE/interface contract and the decode target."""


@dataclass(frozen=True, slots=True)
class DerivedTensor:
    """A tensor the construction BUILDS and no checkpoint fills — a non-persistent buffer.

    Recorded because it is real residency the ledger must know about, and because a
    GB-scale one is the "named saved component when GB-scale" authoring rule violated.

    It carries NO device, and neither does `LogicalTensor`: placement is platform policy
    derived from measurement (§1), so a device is never a construction fact. A tensor that
    lands anywhere but meta refuses during the walk rather than being recorded — the
    contract has no vocabulary for where a tensor lives, which is exactly the point.
    """

    key: str
    component: str
    shape: tuple[int, ...]
    dtype: str | None


@dataclass(frozen=True, slots=True)
class DestinationSet:
    """The COMPLETE destination set of one named component. Complete is the point: a
    partial set would let a narrow declaration silently omit a heavyweight tensor."""

    component: str
    module_class: str
    training: bool
    compiled: bool
    keys: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DeclaredCache:
    """A `loader.cache(...)` handle: memory-only, declared-bounded (§1.3)."""

    name: str
    max_entries: int
    max_bytes: int


@dataclass(frozen=True, slots=True)
class ModelConstructionContract:
    """The code topology plus load bookkeeping, for one exact candidate."""

    candidate: Candidate
    logical_tensors: tuple[LogicalTensor, ...]
    destination_sets: tuple[DestinationSet, ...]
    derived: tuple[DerivedTensor, ...]
    caches: tuple[DeclaredCache, ...]

    def render(self) -> dict[str, Json]:
        """The canonical body. `resources` is a CLOSED expression over these rows —
        recomputed here, never carried in from outside, so it cannot reference a request
        shape, a device, or a measured coefficient."""
        body: dict[str, Json] = {
            "schema": CONTRACT_SCHEMA,
            "candidate": _row(self.candidate),
            "logical_tensors": [_row(t) for t in self.logical_tensors],
            "destination_sets": [_row(d) for d in self.destination_sets],
            "derived": [_row(d) for d in self.derived],
            "caches": [_row(c) for c in self.caches],
            "resources": closed_resources(self),
        }
        return body

    @property
    def digest(self) -> str:
        return canonical.digest(self.render())

    @property
    def components(self) -> tuple[str, ...]:
        return tuple(d.component for d in self.destination_sets)


@dataclass(frozen=True, slots=True)
class ModelConstructionAttestation:
    """What the harness observed while deriving, and what it validated afterwards.

    Everything here is OUTSIDE `contract_digest` on purpose: harness identity, the
    double-derive result, the component-use validation, and H3's finite-schedule map are
    facts ABOUT a construction, never terms OF its identity.
    """

    contract_digest: str
    substrate: str
    torch_version: str
    fences: tuple[str, ...]
    double_derived: bool
    component_use: Mapping[str, tuple[str, ...]]
    timestep_plans: Mapping[str, str]

    def render(self) -> dict[str, Json]:
        return {
            "schema": ATTESTATION_SCHEMA,
            "contract_digest": self.contract_digest,
            "harness": {
                "substrate": self.substrate,
                "torch": self.torch_version,
                "fences": sorted(self.fences),
            },
            "double_derived": self.double_derived,
            "component_use": {m: list(c) for m, c in self.component_use.items()},
            "timestep_plans": dict(self.timestep_plans),
        }

    @property
    def digest(self) -> str:
        return canonical.digest(self.render())


# ------------------------------------------------------------------------ projection


def _row(value: object) -> dict[str, Json]:
    """One dataclass row as canonical JSON. A `None` field is OMITTED, which is what makes
    the dtype-diet rule (absence = bf16) a property of the bytes and not of a reader."""
    out: dict[str, Json] = {}
    for f in fields(value):  # type: ignore[arg-type]
        item = getattr(value, f.name)
        if item is None:
            continue
        out[f.name] = list(item) if isinstance(item, tuple) else item
    return out


def closed_resources(contract: ModelConstructionContract) -> dict[str, Json]:
    """The contract's resource expression: a pure function of its own rows.

    CLOSED means exactly that — every term is derived from `logical_tensors`, with no free
    variable. Nothing here can be set by hand: `render()` calls this, so a resource number
    that disagrees with the rows cannot be serialized at all.
    """
    per: dict[str, dict[str, Json]] = {}
    for tensor in contract.logical_tensors:
        slot = per.setdefault(tensor.component, {"tensors": 0, "elements": 0})
        slot["tensors"] = int(slot["tensors"]) + 1
        elements = 1
        for dim in tensor.shape:
            elements *= dim
        slot["elements"] = int(slot["elements"]) + elements
    return {
        "per_component": per,
        "tensors": len(contract.logical_tensors),
        "elements": sum(int(v["elements"]) for v in per.values()),
    }


def destination_resident_bytes(
    construction: ModelConstructionContract, dtype_widths: Mapping[str, int]
) -> int:
    """Price the exact destination census using the substrate's dtype-width authority.

    This mirrors ``StreamingFillBackend.reserve_components``: every named logical or
    derived tensor returned by the deep walk is a destination charged by the allocator.
    Alias groups are deliberately not collapsed because that allocator walks with
    ``remove_duplicate=False`` and charges the same named census before ``to_empty``.
    """

    def tensor_bytes(tensor: LogicalTensor | DerivedTensor) -> int:
        dtype = tensor.dtype or DEFAULT_DTYPE
        width = dtype_widths.get(dtype)
        if not isinstance(width, int) or isinstance(width, bool) or width <= 0:
            raise ContractError(
                f"destination dtype {dtype!r} has no positive width in this TensorFS build",
                code="contract_destination_dtype_invalid",
                fields=[tensor.key],
            )
        elements = 1
        for extent in tensor.shape:
            elements *= extent
        return elements * width

    return sum(tensor_bytes(tensor) for tensor in construction.logical_tensors) + sum(
        tensor_bytes(tensor) for tensor in construction.derived
    )


def read_contract(raw: bytes) -> ModelConstructionContract:
    """Read one exact canonical ModelConstructionContract, closed at every row.

    Publication owns the bytes; Runtime independently reconstructs the typed value and
    requires its own renderer to reproduce them byte-for-byte.  This is deliberately not a
    permissive projection: an added field changes construction meaning and must first gain
    one reader.
    """

    try:
        document = canonical.parse_canonical(raw)
        root = _object(
            document,
            {
                "caches",
                "candidate",
                "derived",
                "destination_sets",
                "logical_tensors",
                "resources",
                "schema",
            },
            "contract",
        )
        candidate_row = _object(
            root["candidate"],
            {
                "application",
                "config_digest",
                "factory",
                "hardware_variant",
                "model_class",
                "release",
                "snapshot",
            },
            "candidate",
        )
        candidate = Candidate(
            release=_string(candidate_row["release"], "candidate.release"),
            application=_string(candidate_row["application"], "candidate.application"),
            model_class=_string(candidate_row["model_class"], "candidate.model_class"),
            factory=_string(candidate_row["factory"], "candidate.factory"),
            snapshot=_digest(candidate_row["snapshot"], "candidate.snapshot"),
            config_digest=_digest(candidate_row["config_digest"], "candidate.config_digest"),
            hardware_variant=_string(
                candidate_row["hardware_variant"], "candidate.hardware_variant"
            ),
        )
        logical = tuple(
            _logical(row, index)
            for index, row in enumerate(_array(root["logical_tensors"], "logical_tensors"))
        )
        destinations = tuple(
            _destination_set(row, index)
            for index, row in enumerate(_array(root["destination_sets"], "destination_sets"))
        )
        derived = tuple(
            _derived(row, index) for index, row in enumerate(_array(root["derived"], "derived"))
        )
        caches = tuple(
            _cache(row, index) for index, row in enumerate(_array(root["caches"], "caches"))
        )
        contract = ModelConstructionContract(candidate, logical, destinations, derived, caches)
        if root["schema"] != CONTRACT_SCHEMA or contract.render() != root:
            raise ValueError("document does not equal the closed Runtime rendering")
        return contract
    except (canonical.CanonicalError, KeyError, TypeError, ValueError) as exc:
        raise ContractError(
            f"ModelConstructionContract bytes refused: {type(exc).__name__}: {exc}",
            code="contract_document_invalid",
        ) from exc


def logical_model_digest(contract: ModelConstructionContract) -> str:
    """The serve/derive fence projected only from an exact published contract.

    The executor independently derives each field after construction.  Sorting topology and
    component rows makes traversal order non-authoritative and lets the same formula run on
    publication bytes and the serving record without carrying a second asserted digest.
    """

    topology_rows = sorted(
        f"{tensor.key}:{list(tensor.shape)}:{tensor.dtype or DEFAULT_DTYPE}"
        for tensor in contract.logical_tensors
    )
    topology = (
        "blake2b:" + hashlib.blake2b("\n".join(topology_rows).encode(), digest_size=16).hexdigest()
    )
    candidate = contract.candidate
    return canonical.digest(
        {
            "format": "cozy.runtime.ModelConstructionContract/1",
            "release": candidate.release,
            "model_class": candidate.model_class,
            "artifact": candidate.snapshot,
            "config_digest": candidate.config_digest,
            "factory_identity": candidate.factory,
            "topology_digest": topology,
            "components": sorted(row.component for row in contract.destination_sets),
            "variant": candidate.hardware_variant,
        }
    )


def _object(value: object, keys: set[str], name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError(f"{name} has a non-closed field set")
    return value


def _array(value: object, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{name} is not an array")
    return value


def _string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} is not a non-empty string")
    return value


def _digest(value: object, name: str) -> str:
    text = _string(value, name)
    if canonical.DIGEST_RE.fullmatch(text) is None:
        raise ValueError(f"{name} is not a sha256 digest")
    return text


def _shape(value: object, name: str) -> tuple[int, ...]:
    rows = _array(value, name)
    if any(not isinstance(n, int) or isinstance(n, bool) or n < 0 for n in rows):
        raise ValueError(f"{name} is not a non-negative integer shape")
    return tuple(rows)


def _logical(value: object, index: int) -> LogicalTensor:
    name = f"logical_tensors[{index}]"
    row = _object(
        value,
        {"alias_group", "component", "dtype", "key", "kind", "module_class", "shape"}
        if isinstance(value, dict) and "dtype" in value and "alias_group" in value
        else {"component", "key", "kind", "module_class", "shape"}
        | ({"dtype"} if isinstance(value, dict) and "dtype" in value else set())
        | ({"alias_group"} if isinstance(value, dict) and "alias_group" in value else set()),
        name,
    )
    kind = row["kind"]
    if kind not in ("parameter", "buffer"):
        raise ValueError(f"{name}.kind is invalid")
    alias = row.get("alias_group")
    if alias is not None and (not isinstance(alias, int) or isinstance(alias, bool) or alias < 0):
        raise ValueError(f"{name}.alias_group is invalid")
    dtype = row.get("dtype")
    if dtype is not None and (not isinstance(dtype, str) or not dtype):
        raise ValueError(f"{name}.dtype is invalid")
    return LogicalTensor(
        key=_string(row["key"], f"{name}.key"),
        component=_string(row["component"], f"{name}.component"),
        shape=_shape(row["shape"], f"{name}.shape"),
        kind=cast(TensorKind, kind),
        module_class=_string(row["module_class"], f"{name}.module_class"),
        alias_group=alias,
        dtype=dtype,
    )


def _destination_set(value: object, index: int) -> DestinationSet:
    name = f"destination_sets[{index}]"
    row = _object(value, {"compiled", "component", "keys", "module_class", "training"}, name)
    keys = tuple(_string(item, f"{name}.keys") for item in _array(row["keys"], f"{name}.keys"))
    if not isinstance(row["training"], bool) or not isinstance(row["compiled"], bool):
        raise ValueError(f"{name} flags are not boolean")
    return DestinationSet(
        component=_string(row["component"], f"{name}.component"),
        module_class=_string(row["module_class"], f"{name}.module_class"),
        training=row["training"],
        compiled=row["compiled"],
        keys=keys,
    )


def _derived(value: object, index: int) -> DerivedTensor:
    name = f"derived[{index}]"
    keys = {"component", "key", "shape"}
    if isinstance(value, dict) and "dtype" in value:
        keys.add("dtype")
    row = _object(value, keys, name)
    dtype = row.get("dtype")
    if dtype is not None and (not isinstance(dtype, str) or not dtype):
        raise ValueError(f"{name}.dtype is invalid")
    return DerivedTensor(
        key=_string(row["key"], f"{name}.key"),
        component=_string(row["component"], f"{name}.component"),
        shape=_shape(row["shape"], f"{name}.shape"),
        dtype=dtype,
    )


def _cache(value: object, index: int) -> DeclaredCache:
    name = f"caches[{index}]"
    row = _object(value, {"max_bytes", "max_entries", "name"}, name)
    for field in ("max_entries", "max_bytes"):
        if not isinstance(row[field], int) or isinstance(row[field], bool) or row[field] <= 0:
            raise ValueError(f"{name}.{field} is not positive")
    return DeclaredCache(_string(row["name"], f"{name}.name"), row["max_entries"], row["max_bytes"])


# ------------------------------------------------------------------------- agreement


def check_agreement(contract: ModelConstructionContract, destinations: Sequence[str]) -> None:
    """EXACT ordered one-to-one agreement, at both plumbing levels.

    `destinations` is the fill plane's own key list (cr-002b's `Loader` census, which
    cr-005 turns into real byte fills). The contract's `logical_tensors` come from the
    harness's independent deep walk. Membership proves topology; sequence proves the MCC
    records the same construction traversal the Loader will hand to TensorFS.
    """
    actual = list(destinations)
    keys = [t.key for t in contract.logical_tensors]
    duplicates = sorted({k for k in keys if keys.count(k) > 1})
    if duplicates:
        raise ContractError(
            f"{len(duplicates)} logical tensor key(s) appear twice: {', '.join(duplicates[:6])} "
            "— a destination is ONE tensor; a repeated key is a re-key by another name",
            code="duplicate_destination",
            fields=duplicates[:6],
        )
    covered: list[str] = []
    for dset in contract.destination_sets:
        covered.extend(dset.keys)
    if sorted(covered) != sorted(keys):
        only_set = sorted(set(covered) - set(keys))
        only_log = sorted(set(keys) - set(covered))
        raise ContractError(
            "the DestinationSets do not partition the logical tensors exactly "
            f"(+{only_set[:4]} / -{only_log[:4]}) — one COMPLETE DestinationSet per "
            "component is what makes a narrow component-use declaration authoritative",
            code="incomplete_destination_set",
            fields=(only_set + only_log)[:6],
        )
    missing = sorted(set(keys) - set(actual))
    surplus = sorted(set(actual) - set(keys))
    if missing or surplus:
        raise ContractError(
            "the harness census and the fill plane disagree: "
            f"{len(missing)} contract-only {missing[:4]}, {len(surplus)} fill-only "
            f"{surplus[:4]} — the contract the serve path replays must be the derived one",
            code="census_disagreement",
            fields=(missing + surplus)[:6],
        )
    if covered != actual:
        first = next(
            (
                index
                for index, (declared, filled) in enumerate(zip(covered, actual, strict=False))
                if declared != filled
            ),
            min(len(covered), len(actual)),
        )
        declared = covered[first] if first < len(covered) else "<end>"
        filled = actual[first] if first < len(actual) else "<end>"
        raise ContractError(
            "the ModelConstructionContract and Loader census contain the same destinations "
            f"in different order at index {first}: contract {declared!r}, Loader "
            f"{filled!r} — header/load order is construction identity",
            code="census_order_disagreement",
            fields=[declared, filled],
        )


# ------------------------------------------------------- unrepresentability, at import

#: The EXACT field set of every contract row. Not a minimum — an equality. Adding a field
#: to any row type below fails at import until this table is updated, which is the review
#: gate: someone must justify the new fact in a diff that touches this constant.
_ROW_FIELDS: Mapping[type, frozenset[str]] = {
    Candidate: frozenset(
        {
            "release",
            "application",
            "model_class",
            "factory",
            "snapshot",
            "config_digest",
            "hardware_variant",
        }
    ),
    LogicalTensor: frozenset(
        {"key", "component", "shape", "kind", "module_class", "alias_group", "dtype"}
    ),
    DerivedTensor: frozenset({"key", "component", "shape", "dtype"}),
    DestinationSet: frozenset({"component", "module_class", "training", "compiled", "keys"}),
    DeclaredCache: frozenset({"name", "max_entries", "max_bytes"}),
    ModelConstructionContract: frozenset(
        {"candidate", "logical_tensors", "destination_sets", "derived", "caches"}
    ),
}


def _prove_unrepresentable() -> None:
    """Run at import: the row types can express a destination and nothing else."""
    for row, expected in _ROW_FIELDS.items():
        actual = {f.name for f in fields(row)}
        if actual != expected:
            raise ContractError(
                f"{row.__name__} fields are {sorted(actual)}, the reviewed set is "
                f"{sorted(expected)} — destination-side fusion, permutation, re-key, "
                "quantization and subclass reconstruction stay UNREPRESENTABLE by keeping "
                "this an equality (§1.1); those are AOT artifact work",
                code="contract_shape",
                fields=sorted(actual ^ expected),
            )


_prove_unrepresentable()
