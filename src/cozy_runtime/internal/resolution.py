"""#549.1 — the ResolvedModelPlan: ONE object, ONE decision, and a label that derives.

`encoding.Selector` offers each tensor the routes this card admits (`selection.admit`, a rule
of the card's tensor cores); this takes one route per tensor by the package's consent: the
encoded GEMM where the package accepts encoded leaves and the card serves the leaf, else the
decode floor, else verbatim. Nothing measured ranks anything (xs-065, owner 2026-10-22): a
card no one has benchmarked resolves exactly as a measured one of its row.

The plan is resolved once, before a destination is reserved, and holds:

* the EXACT snapshot map (component -> snapshot digest), not a variant NAME;
* one `TensorResolution` per destination: encoding, implementation digest and revision,
  route, validated geometry, output dtype, and the REASON;
* device and runtime identity (#549.9), placement and construction facts.

`digest()` is what enters generation identity. `label()` DERIVES from the route census, so a
mixed plan says `mixed(encoded_gemm x743, decoded_float x120, verbatim x937)` and cannot say
anything else. `delivery.py`'s rung table describes what a route costs and when.
"""

from __future__ import annotations

import dataclasses
import hashlib
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol

import msgspec

from cozy_runtime.author._prepared_use import PreparedUse
from cozy_runtime.internal import arch, canonical
from cozy_runtime.internal.delivery import DELIVERY_RUNGS, ROUTE_KERNEL
from cozy_runtime.internal.encoding.adapters import _nvfp4_kernel
from cozy_runtime.internal.encoding.formats import Encoded, EncodingRefusal, LeafProvider, Provider
from cozy_runtime.internal.encoding.selection import (
    DeviceFacts,
    Refusal,
    RuntimeIdentity,
    Selection,
    Selector,
    TensorCandidates,
    device_line,
    implementation_digest,
)
from cozy_runtime.internal.nvfp4_linear import DECODE_ROWS

#: The route a tensor PREFERS under each policy (the package's consent decides the policy);
#: the first route in the list that this tensor actually has wins.
_PREFERENCE: dict[str, tuple[str, ...]] = {
    "encoded_gemm": ("encoded_gemm", "decoded_float", "verbatim"),
    "decode_floor": ("decoded_float", "verbatim", "encoded_gemm"),
}


def route_label(census: Mapping[str, int]) -> str:
    """A route census -> the plan's human-facing name. A census carrying more than one route
    names all of them with counts."""
    present = {route: n for route, n in census.items() if n}
    if not present:
        return "empty"
    if len(present) == 1:
        route = next(iter(present))
        return {
            "verbatim": "verbatim",
            "decoded_float": "aot_decode",
            "encoded_gemm": "encoded_gemm",
        }[route]
    return "mixed(" + ", ".join(f"{r} x{present[r]}" for r in sorted(present)) + ")"


class Step(msgspec.Struct, omit_defaults=True):
    """The resolution's record: the variant, its verdict (`chosen`, or the refusal code)
    and why."""

    variant: str
    verdict: str
    why: str = ""
    policy: str = ""
    plan: str = ""
    label: str = ""
    routes: dict[str, int] | None = None


class PlanRefusal(Exception):
    """A typed pre-construction refusal on the plan axis."""

    def __init__(self, code: str, detail: str, steps: Sequence[Step] = ()) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail
        self.steps = tuple(steps)


# ------------------------------------------------------------------------ the variant


@dataclass(frozen=True, slots=True)
class Variant:
    """The artifact a plan reads: its store and the snapshot of each component."""

    name: str
    store: str
    snapshot: str
    snapshots: Mapping[str, str] = field(default_factory=dict)
    """component -> snapshot digest. A REAL map: it is what the plan digest binds (#549.2)."""

    def snapshot_map(self) -> dict[str, str]:
        return dict(self.snapshots)


@dataclass(frozen=True, slots=True)
class ConstructionFacts:
    """The half of a generation's identity that is NOT about bytes or hardware.

    Hashed into the plan beside the selected snapshot and per-tensor resolution. The snapshot
    already commits its inline construction config and tensor schema.
    """

    release: str
    model_class: str
    components: tuple[str, ...]
    custody: str = "canonical"

    def document(self) -> dict[str, str | list[str]]:
        return {
            "release": self.release,
            "model_class": self.model_class,
            "components": list(self.components),
            "custody": self.custody,
        }


# ------------------------------------------------------------------- the per-tensor row


@dataclass(frozen=True, slots=True)
class TensorResolution:
    """ONE destination's whole answer: the encoding, the exact implementation, the route, the
    geometry that was VALIDATED, and why."""

    key: str
    component: str
    encoding: str
    """The stored scheme (`plain` for a plain tensor)."""
    implementation: str
    implementation_digest: str
    implementation_revision: str
    route: str
    geometry: tuple[int, ...]
    output_dtype: str
    numel: int
    stored_bytes: int
    reason: str
    """One sentence. #549.5's ragged-K case reads: the leaf refused this shape, here is the
    rule, and the decode floor serves it — IN the plan, not in a log nobody keeps."""
    refused: Refusal | None = None
    """Set when this row is served decoded because this card does not serve its encoded
    leaf; None when its route is by design (`reason`)."""

    def identity(self) -> dict[str, str | list[int]]:
        """What the plan DIGEST takes from this row."""
        return {
            "key": self.key,
            "encoding": self.encoding,
            "implementation_digest": self.implementation_digest,
            "implementation_revision": self.implementation_revision,
            "route": self.route,
            "geometry": list(self.geometry),
            "output_dtype": self.output_dtype,
        }

    def document(self) -> dict[str, str | int | list[int]]:
        return {
            **self.identity(),
            "component": self.component,
            "implementation": self.implementation,
            "numel": self.numel,
            "stored_bytes": self.stored_bytes,
            "reason": self.reason,
            "refused": f"{self.refused.leaf}: {self.refused.reason}" if self.refused else "",
        }


@dataclass(frozen=True, slots=True)
class RefusedLeaves:
    """One component's tensors of one lane that serve decoded because this card does not
    serve their leaf, for one reason. A run served by such a plan says so in its record."""

    component: str
    lane: str
    leaf: str
    reason: str
    dtype: str
    """What they serve as instead: the destination's float dtype."""
    tensors: int


# ------------------------------------------------------------------------- THE PLAN


@dataclass(frozen=True, slots=True)
class ResolvedModelPlan:
    """The one immutable object a generation is. Resolved once, before anything is reserved.
    Its `digest()` enters generation identity and its `label()` derives from what it
    actually resolved."""

    variant: str
    store: str
    snapshots: Mapping[str, str]
    policy: str
    tensors: tuple[TensorResolution, ...]
    device: DeviceFacts
    runtime: RuntimeIdentity
    placement: str
    construction: ConstructionFacts
    encoded_leaves: str
    walk: tuple[Step, ...] = ()
    prepared_use: PreparedUse | None = None
    prepared_use_reason: str = ""
    prepared_footprint_bytes: int = 0

    # ------------------------------------------------------------------ derived facts

    def by_key(self) -> dict[str, TensorResolution]:
        """contract key -> its resolution. The fill plane EXECUTES this map (cr-025)."""
        return {row.key: row for row in self.tensors}

    def refused_leaves(self) -> tuple[RefusedLeaves, ...]:
        groups = Counter(
            (row.component, row.encoding, row.refused.leaf, row.refused.reason, row.output_dtype)
            for row in self.tensors
            if row.refused is not None
        )
        return tuple(RefusedLeaves(*group, n) for group, n in sorted(groups.items()))

    def route_census(self) -> dict[str, int]:
        census: dict[str, int] = {}
        for row in self.tensors:
            census[row.route] = census.get(row.route, 0) + 1
        return census

    def label(self) -> str:
        """The human-facing name, DERIVED from the census — a mixed plan says so (#549.1)."""
        return route_label(self.route_census())

    def wire_delivery(self) -> str:
        """The wire's closed `delivery` vocabulary (`native | float`): one decoded tensor makes
        the generation a float one."""
        return "float" if self.route_census().get("decoded_float") else "native"

    def wire_materialization(self) -> str:
        """`aot_decode | staged_decode`, or empty when nothing decodes: the same decode paid
        once at fill, or again on every stage-in (`component_staged`)."""
        if not self.route_census().get("decoded_float"):
            return ""
        return "staged_decode" if self.placement == "component_staged" else "aot_decode"

    def wire_rung(self) -> str:
        """The rung NAME whose (delivery, materialization) pair equals this plan's own derived
        wire fields: a derivation, not a chooser's answer."""
        delivery, materialization = self.wire_delivery(), self.wire_materialization()
        for rung in DELIVERY_RUNGS:
            if (
                rung.status == "proven"
                and rung.delivery == delivery
                and rung.materialization == materialization
            ):
                return rung.name
        return "verbatim"

    def wire_document(self) -> dict[str, object]:
        """The `delivery` document a prepare reply carries; every value derives from the plan."""
        return {
            "rung": self.wire_rung(),
            "variant": self.variant,
            "delivery": self.wire_delivery(),
            "materialization": self.wire_materialization(),
            "label": self.label(),
            "plan_digest": self.digest(),
            "routes": self.route_census(),
            "kernels": self.kernels(),
        }

    def kernels(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for route, n in self.route_census().items():
            kernel = ROUTE_KERNEL[route]
            out[kernel] = out.get(kernel, 0) + n
        return out

    def replaces_leaves(self) -> bool:
        return bool(self.route_census().get("encoded_gemm"))

    def stored_bytes(self) -> int:
        return sum(row.stored_bytes for row in self.tensors)

    # --------------------------------------------------------------------- identity

    def identity(self) -> dict[str, object]:
        return {
            "store": self.store,
            "snapshots": dict(sorted(self.snapshots.items())),
            "tensors": [row.identity() for row in sorted(self.tensors, key=lambda r: r.key)],
            "device": self.device.document(),
            "runtime": self.runtime.document(),
            "placement": self.placement,
            "construction": self.construction.document(),
            **(
                {"prepared_use": msgspec.to_builtins(self.prepared_use)}
                if self.prepared_use
                else {}
            ),
        }

    def digest(self) -> str:
        """THE plan's canonical digest — what enters generation identity. It binds the snapshot
        BYTES, the runtime build, the per-tensor provider map, the placement and the
        construction contract. The variant NAME, the policy and consent are absent: two
        resolutions that reach the same assignment over the same bytes ARE the same plan."""
        encoded = canonical.write(self.identity())
        return (
            "sha256:" + hashlib.sha256(b"cozy.runtime.resolved-model-plan\0" + encoded).hexdigest()
        )

    def seam_document(self) -> dict[str, object]:
        """The plan as a load reply carries it: everything but the per-destination rows, which
        `plan_digest` covers."""
        whole = self.document()
        del whole["tensors"], whole["walk"]
        whole["tensors_count"] = len(self.tensors)
        return whole

    def document(self) -> dict[str, object]:
        return {
            "plan_digest": self.digest(),
            "label": self.label(),
            "variant": self.variant,
            "policy": self.policy,
            "store": self.store,
            "snapshots": dict(sorted(self.snapshots.items())),
            "delivery": self.wire_delivery(),
            "materialization": self.wire_materialization(),
            "routes": self.route_census(),
            "kernels": self.kernels(),
            "replaces_leaves": self.replaces_leaves(),
            "encoded_leaves": self.encoded_leaves,
            "placement": self.placement,
            "device": self.device.document(),
            "runtime": self.runtime.document(),
            "construction": self.construction.document(),
            "stored_bytes": self.stored_bytes(),
            "tensors": [row.document() for row in self.tensors],
            "walk": msgspec.to_builtins(self.walk),
            **(
                {
                    "prepared_use": {
                        "request": msgspec.to_builtins(self.prepared_use),
                        "reason": self.prepared_use_reason,
                        "required_bytes": self.prepared_footprint_bytes,
                    }
                }
                if self.prepared_use
                else {}
            ),
        }

    def summary(self) -> str:
        census = ", ".join(f"{r} x{n}" for r, n in sorted(self.route_census().items()))
        return f"{self.label()} [{self.digest()[:19]}…] over {self.variant!r}: {census}"


# ------------------------------------------------------------------------ resolution


class PlanRow(Protocol):
    """The header facts one destination contributes: what `fill.PlanRow` is, named here so
    `resolution.py` need not import the fill plane to resolve a plan from a header."""

    @property
    def key(self) -> str: ...
    @property
    def component(self) -> str: ...
    @property
    def dtype(self) -> str: ...
    @property
    def shape(self) -> tuple[int, ...]: ...
    @property
    def encoded(self) -> Encoded: ...
    @property
    def nbytes(self) -> int: ...


def resolve(
    *,
    variant: Variant,
    rows: Sequence[PlanRow],
    providers: Mapping[str, tuple[Provider, ...]],
    device: DeviceFacts,
    runtime: RuntimeIdentity,
    construction: ConstructionFacts,
    placement: str = "all_resident",
    encoded_leaves: str = "refuse",
    dtype_name: Callable[[str], str],
    prepared_use: PreparedUse | None = None,
    prepared_free_bytes: int = 0,
    prepared_granularity: int = 0,
    prepared_ineligible: str = "",
) -> ResolvedModelPlan:
    """The variant's header rows -> ONE plan, by rule.

    `dtype_name` maps a TensorFS dtype name to the torch dtype NAME a provider writes — handed
    in rather than imported, so this module stays free of the fill plane. The rows come from
    the HEADER (cr-006 measured the real fp8 UNet's at 400,592 B read, zero tensor bytes).
    """
    if not rows:
        raise PlanRefusal(
            "no_qualified_plan",
            f"{variant.name} has no tensor rows: a generation reads bytes or it does not exist",
            [Step(variant.name, "no_rows", "no tensor rows")],
        )
    selector = Selector(providers, device)
    try:
        found = [
            (row, selector.candidates(row.key, row.encoded, dtype_name(row.dtype), row.shape))
            for row in rows
        ]
    except EncodingRefusal as exc:
        step = Step(variant.name, exc.code, str(exc)[:600])
        raise PlanRefusal("no_qualified_plan", _refusal(variant, device, step), [step]) from exc
    unserved = next((c for _, c in found if not c.by_route), None)
    if unserved is not None:
        step = Step(variant.name, "unqualified", unserved.unqualified[:600])
        raise PlanRefusal("no_qualified_plan", _refusal(variant, device, step), [step])
    # THE CONSENT GATE (#549.1): a route that REPLACES the module that computes needs the
    # package to say so, because code that reads `.weight` finds nothing there.
    policy = "encoded_gemm" if encoded_leaves == "accept" else "decode_floor"
    retained, why, footprint = _prepared_choices(
        found,
        device,
        prepared_use,
        policy,
        prepared_free_bytes,
        prepared_granularity,
        prepared_ineligible,
    )
    plan = _build(
        variant=variant,
        policy=policy,
        resolved=found,
        device=device,
        runtime=runtime,
        construction=construction,
        placement=placement,
        encoded_leaves=encoded_leaves,
        dtype_name=dtype_name,
        retained=retained,
    )
    plan = dataclasses.replace(
        plan,
        prepared_use=prepared_use,
        prepared_use_reason=why,
        prepared_footprint_bytes=footprint,
    )
    step = Step(
        variant.name,
        "chosen",
        policy=policy,
        plan=plan.digest(),
        label=plan.label(),
        routes=plan.route_census(),
    )
    return dataclasses.replace(plan, walk=(step,))


def _prepared_choices(
    found: Sequence[tuple[PlanRow, TensorCandidates]],
    device: DeviceFacts,
    use: PreparedUse | None,
    policy: str,
    free: int,
    granularity: int,
    ineligible: str,
) -> tuple[frozenset[str], str, int]:
    """Price a fixed retained-float use before any leaf or destination exists.

    There is no block census yet. Deliberately overprice: a separate native allocation for
    every raw row and every logical row, all packed inputs live beside all final weights,
    plus the program's workspace. Unknown transforms cannot use this estimate.
    """
    empty: frozenset[str] = frozenset()
    if use is None:
        return empty, "", 0
    if use.target == "stored":
        return empty, "the program keeps the existing stored route", 0
    if ineligible:
        return empty, ineligible, 0
    if device.kind != "cuda" or arch.row(device.sm).fp4:
        return empty, "the existing device/native-FP4 route is preserved", 0
    if use.activation_rows <= DECODE_ROWS or use.repetitions <= 1 or not use.tensors:
        return empty, "use is unknown, mixed, small-M or not repeated", 0
    if use.compute_dtype != "BF16" or policy != "encoded_gemm":
        return empty, "retention preserves the existing BF16 logical dtype and leaf consent", 0
    if not _nvfp4_kernel(device.sm, "BF16"):
        return empty, "the resident-input native BF16 decoder is unavailable", 0
    rows = {row.key: (row, candidates) for row, candidates in found}
    if any(key not in rows for key in use.tensors):
        raise PlanRefusal("prepared_use_invalid", "prepared use names an absent tensor")
    for key in use.tensors:
        row, candidates = rows[key]
        decoded = candidates.by_route.get("decoded_float")
        if (
            row.encoded.scheme != "nvfp4"
            or row.dtype != "bf16"
            or "encoded_gemm" not in candidates.by_route
            or decoded is None
            or decoded.provider.name != "cozy.nvfp4.cores/1"
        ):
            return empty, "a requested tensor lacks the unchanged BF16 native decode route", 0
    if not use.workspace_bytes or granularity <= 0:
        return empty, "the full workspace/alignment allowance is unknown", 0

    def align(size: int) -> int:
        return (size + granularity - 1) // granularity * granularity

    weights, scratch = 0, 0
    for row, candidates in found:
        chosen = (
            candidates.by_route["decoded_float"]
            if row.key in use.tensors
            else _prefer(candidates.by_route, policy)
        )
        provider = chosen.provider
        if isinstance(provider, LeafProvider):
            parts = {part.role: part for part in row.encoded.parts}
            # Existing leaf declarations price padded roles and their fill scratch. Keep
            # the logical tensor as an additional conservative bound on dtype conversion.
            weights += align(max(row.nbytes, provider.resident_bytes(parts)))
            scratch += align(provider.fill_scratch_bytes(parts))
        elif provider.route == "verbatim" or row.key in use.tensors:
            weights += align(row.nbytes)
        else:
            return empty, "another decoder's complete preparation footprint is unknown", 0
    raw = sum(
        align(sum((part.nbytes + 255) // 256 * 256 for part in row.encoded.parts))
        for row, _ in found
    )
    required = weights + raw + scratch + use.workspace_bytes
    if weights > use.max_resident_bytes or required > free:
        return (
            empty,
            "complete retained weights, packed fill and workspace exceed the grant or free room",
            required,
        )
    return (
        frozenset(use.tensors),
        "fixed repeated large-M use retains its BF16 final buffers",
        required,
    )


def _refusal(variant: Variant, device: DeviceFacts, step: Step) -> str:
    return (
        f"the {variant.name!r} variant resolves to no executable plan on {device_line(device)}. "
        f"The tensor schema is known; what is missing is a route from these stored bytes to a "
        f"kernel this device has. {step.verdict} — {step.why[:200]}"
    )


def _build(
    *,
    variant: Variant,
    policy: str,
    resolved: Sequence[tuple[PlanRow, TensorCandidates]],
    device: DeviceFacts,
    runtime: RuntimeIdentity,
    construction: ConstructionFacts,
    placement: str,
    encoded_leaves: str,
    dtype_name: Callable[[str], str],
    retained: frozenset[str] = frozenset(),
) -> ResolvedModelPlan:
    """One policy -> its plan. Pure: no I/O, no device, no allocation."""
    tensors: list[TensorResolution] = []
    for row, found in resolved:
        selection = (
            found.by_route["decoded_float"]
            if row.key in retained
            else _prefer(found.by_route, policy)
        )
        numel = 1
        for extent in row.shape:
            numel *= extent
        tensors.append(
            TensorResolution(
                key=row.key,
                component=row.component,
                encoding=row.encoded.scheme,
                implementation=selection.provider.name,
                implementation_digest=implementation_digest(selection.provider),
                implementation_revision=selection.provider.code_revision,
                route=selection.provider.route,
                geometry=row.shape,
                output_dtype=dtype_name(row.dtype),
                numel=numel,
                stored_bytes=row.encoded.stored_nbytes,
                reason=("prepared repeated large-M BF16 use; " if row.key in retained else "")
                + _reason(selection, found, policy),
                refused=found.refused[0]
                if found.refused
                and encoded_leaves == "accept"
                and selection.provider.route == "decoded_float"
                else None,
            )
        )
    return ResolvedModelPlan(
        variant=variant.name,
        store=variant.store,
        snapshots=dict(
            variant.snapshot_map() or {c: variant.snapshot for c in construction.components}
        ),
        policy=policy,
        tensors=tuple(tensors),
        device=device,
        runtime=runtime,
        placement=placement,
        construction=construction,
        encoded_leaves=encoded_leaves,
    )


def _prefer(by_route: Mapping[str, Selection], policy: str) -> Selection:
    for route in _PREFERENCE[policy]:
        found = by_route.get(route)
        if found is not None:
            return found
    # `by_route` is non-empty by construction (`resolve` refuses an unserved tensor first).
    return next(iter(by_route.values()))


def _reason(selection: Selection, found: TensorCandidates, policy: str) -> str:
    """WHY this tensor got this route, in one sentence, recorded IN the plan (#549.5)."""
    parts = [selection.reason]
    if found.geometry_refused:
        parts.append(
            "GEOMETRY: "
            + " | ".join(f"{name} refused — {why}" for name, why in found.geometry_refused)
        )
    if found.refused:
        parts.append(" | ".join(f"{one.leaf}: {one.reason}" for one in found.refused))
    if policy == "decode_floor" and "encoded_gemm" in found.by_route:
        parts.append("the package declines encoded leaves, so the decode floor serves")
    return "; ".join(p for p in parts if p)
