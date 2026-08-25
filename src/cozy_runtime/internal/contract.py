"""The ModelConstructionContract and its Attestation — the binding-clock documents.

The EndpointDescriptor (cr-003) is the SOURCE-STABLE surface; this is the other clock.
One contract per EXACT candidate — (endpoint release, model factory, artifact snapshot,
artifact configs, hardware variant) — derived by EXECUTING the author's `load()` under the
cr-004 harness, never by trusting a declaration. It is stored apart from
`endpoint.descriptor.json` and never merges into it.

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

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from typing import Literal

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


# ------------------------------------------------------------------------- agreement


def check_agreement(
    contract: ModelConstructionContract, destinations: Sequence[str]
) -> None:
    """EXACT one-to-one agreement, in both directions and at both plumbing levels.

    `destinations` is the fill plane's own key list (cr-002b's `Loader` census, which
    cr-005 turns into real byte fills). The contract's `logical_tensors` come from the
    harness's independent deep walk. Two independently derived sets that must be equal is
    a far stronger claim than one set checked against itself.
    """
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
    missing = sorted(set(keys) - set(destinations))
    surplus = sorted(set(destinations) - set(keys))
    if missing or surplus:
        raise ContractError(
            "the harness census and the fill plane disagree: "
            f"{len(missing)} contract-only {missing[:4]}, {len(surplus)} fill-only "
            f"{surplus[:4]} — the contract the serve path replays must be the derived one",
            code="census_disagreement",
            fields=(missing + surplus)[:6],
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

#: Vocabulary that would make a destination-side transform expressible. A field whose name
#: contains one of these refuses even if someone remembers to update `_ROW_FIELDS`.
FORBIDDEN_VOCABULARY: frozenset[str] = frozenset(
    {
        "source",
        "source_key",
        "from_key",
        "from_keys",
        "maps_from",
        "rekey",
        "rename",
        "transform",
        "transpose",
        "permute",
        "permutation",
        "reorder",
        "fuse",
        "fused",
        "unfuse",
        "split",
        "chunk",
        "concat",
        "cat",
        "slice",
        "part",
        "parts",
        "shard",
        "scale_of",
        "zero_point",
        "quant",
        "quantize",
        "dequantize",
        "encoding",
        "codec",
        "decoder",
        "converter",
        "convert",
        "adapter",
        "reconstruct",
        "subclass",
        "wrapper",
        "view_of",
        "reshape",
        "expr",
        "formula",
    }
)


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
        for name in actual:
            for word in name.split("_"):
                if word in FORBIDDEN_VOCABULARY:
                    raise ContractError(
                        f"{row.__name__}.{name} contains {word!r}: a contract row may name a "
                        "destination, never a way to build one from something else",
                        code="contract_vocabulary",
                        fields=[f"{row.__name__}.{name}"],
                    )


_prove_unrepresentable()
