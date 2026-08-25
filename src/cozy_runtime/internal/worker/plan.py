"""The PlanChooser, the AttemptPlan, and the local resolution of a binding plan.

Two things live together because they are one decision made in one place, before durable
acceptance (§3.2). The ledger they are decided AGAINST is `worker/ledger.py`.

* **PlanChooser** — pure arithmetic over the request shape, the bindings and the live
  ledger, returning ONE AttemptPlan. At cr-007 it is deliberately a SINGLE-plan chooser
  behind the real offer/prepare/accept seam: the closed proven set has one member, so the
  first-fit walk has one row. cr-008b replaces the body of `choose` and nothing else — no
  protocol change, no new state, no forward dependency. A probe or a benchmark observed
  inside `choose` is a defect by construction: this function takes no I/O of any kind.
* **ExecutorPreparer** — applies the chosen plan and DECIDES NOTHING. It is the reason
  "accepted" never includes seconds of unjournaled preparation: the supervisor prepares
  first, journals second.

The AttemptPlan is a canonical DOCUMENT (tfs-013's writer) so `plan_digest` on
AttemptAccepted is an identity a coordinator can hold, and the summary beside it is the
CLOSED observable projection — the coordinator sees what was chosen, not only that
something was.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from cozy_runtime.internal import canonical
from cozy_runtime.internal.worker.ledger import Ledger


class PlanRefusal(Exception):
    """A typed pre-acceptance refusal, carrying the structured shortfall when it has one."""

    def __init__(self, code: str, detail: str, shortfall: dict[str, Any] | None = None) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail
        self.shortfall = shortfall


@dataclass(frozen=True, slots=True)
class AttemptPlan:
    """One exact choice. `document()` is its identity; `summary()` is its projection."""

    delivery: str
    materialization: str
    compute_dtype: str
    placement: str
    reserved_vram_bytes: int
    reserved_host_bytes: int
    reserved_pinned_bytes: int
    headroom_bytes: int
    entrypoint_binding_plan_id: str
    model_construction_digest: str
    quantified_choice: str

    def document(self) -> dict[str, Any]:
        return {
            "format": "cozy.runtime.AttemptPlan/1",
            "delivery": self.delivery,
            "materialization": self.materialization,
            "compute_dtype": self.compute_dtype,
            "placement": self.placement,
            "reserved_vram_bytes": self.reserved_vram_bytes,
            "reserved_host_bytes": self.reserved_host_bytes,
            "reserved_pinned_bytes": self.reserved_pinned_bytes,
            "headroom_bytes": self.headroom_bytes,
            "entrypoint_binding_plan_id": self.entrypoint_binding_plan_id,
            "model_construction_digest": self.model_construction_digest,
        }

    def digest(self) -> str:
        return canonical.digest(self.document())


@dataclass(frozen=True, slots=True)
class BindingPlan:
    """The local resolution of one `entrypoint_binding_plan_id`.

    The coordinator names the plan by digest and never ships a path; the worker resolves
    that digest against its own records. At launch the record is a small local file staged
    under `<cozy_home>/binding-plans/` — th-004 owns the real EntrypointBindingPlan document
    and tfs-003 the artifact fetch, so this is the SEAM, named, not a pretend hub.
    """

    plan_id: str
    project: str
    model_class: str
    binding_path: str
    param: str
    component: str
    store: str
    config: str
    snapshot: str
    release: str
    variant: str
    vram_bytes: int
    host_bytes: int
    pinned_bytes: int
    entrypoint: str
    model_construction_digest: str = ""
    snapshots: str = ""
    """Per-component snapshots, as `name=sha256:…` comma-joined. Empty means every
    component reads the one `snapshot`.

    A read plan is COMPLETE over its header (tfs-007), so a component cannot be read out of
    a whole-pipeline snapshot without reading the pipeline. Per-component snapshots make a
    partial fill the store's ordinary complete read; the objects dedup, so it costs nothing.
    """
    components: str = ""
    """The ORDERED component set, comma-joined. Empty means `component` alone.

    Additive on purpose: cozy-creator writes this record (decisions #269, a two-sided closed
    contract with no fence to hold it), so a field it does not yet write must default to
    exactly what the single-component record meant.
    """
    vram_floor_bytes: int = 0
    """The PHYSICAL FLOOR the deployment declares: activations + scratch + one component
    chunk. Below it there is no rung, and a local capacity refusal is the honest answer."""
    resident_budget_bytes: int = 0
    """The DECLARED construction ceiling: components past it are parked and staged in on
    first declared use. 0 means every component stays resident."""
    tenancy: str = "local"
    """`local` (single-tenant: capacity never refuses above the floor, it deepens the ladder
    and confesses) or `shared` (a shortfall is a RESCHEDULE for the coordinator, never a
    structural-incompatibility claim)."""
    custody: str = "canonical"
    """`canonical` (the border classified every key: a serve-time extra is drift and
    refuses) or `local` (an un-bordered local bind: markers transcribe or refuse, unknown
    extras warn, are ignored and are confessed — §1.1's custody tier)."""
    strict_keys: bool = False
    """`--strict-keys`: a local bind's unknown extras refuse instead of confessing."""

    def snapshot_map(self) -> dict[str, str]:
        """component -> snapshot digest. Absent names fall back to the binding's own."""
        out: dict[str, str] = {}
        for pair in self.snapshots.split(","):
            name, _, digest = pair.partition("=")
            if name and digest:
                out[name] = digest
        return out

    def component_set(self) -> tuple[str, ...]:
        """The components this binding constructs, in the order it constructs them."""
        if not self.components:
            return (self.component,)
        return tuple(name for name in self.components.split(",") if name)

    def construction_key(self) -> str:
        """Two binding plans over the SAME construction share one executor generation."""
        return canonical.digest(
            {
                "store": self.store,
                "config": self.config,
                "snapshot": self.snapshot,
                "release": self.release,
                "variant": self.variant,
                "model_class": self.model_class,
                "component": self.component,
                "components": self.components,
            }
        )

    @classmethod
    def read(cls, record: dict[str, Any]) -> BindingPlan:
        known = {f for f in cls.__dataclass_fields__}
        unknown = sorted(set(record) - known)
        if unknown:
            raise PlanRefusal(
                "binding_plan_unknown_field",
                f"binding plan record carries {', '.join(unknown)}: the record's key set is "
                "closed, exactly like the documents it stands in for",
            )
        return cls(**record)


#: The CLOSED, ORDERED placement ladder (§3.2). Delivery is `native` at launch — `aot_decode`
#: is the materialization, and the encoded-delivery rungs are cr-008c's. Each rung says what
#: it needs RESIDENT AT ONCE and what it costs; a rung is eligible only once it has been
#: proven, which for launch means the two the corpus actually runs.
#:
#: Order is the point: first-fit walks it top to bottom, so the ladder DEEPENS under
#: pressure and never shallows. A rung's price is measured, not asserted, and the two
#: costs are distinguished (pgw#1505): `per_call_ms` is paid once per attempt, `per_step_ms`
#: multiplies with the request's own step count.
RUNGS: tuple[tuple[str, str, str], ...] = (
    (
        "all_resident",
        "every declared component resident for the whole attempt: no staging, no eviction, "
        "and the deepest component set the card can hold",
        "proven",
    ),
    (
        "component_staged",
        "the component-use set resident at a time: the runtime stages a declared set in "
        "before method entry and frees the LRU refcount-0 component to make room",
        "proven",
    ),
    (
        "host_offload",
        "weights parked in host memory between uses",
        "unproven",
    ),
)

#: How a coefficient was established. Enforcement arms ONLY at `measured` (§3.2): the v1
#: incident is an UNCALIBRATED 25% headroom charge refusing a lane on the very card that was
#: serving it. `bound` is the author's labelled hatch and is never rendered as a measurement.
PROVENANCE = ("uncalibrated", "bound", "measured", "ledger")


@dataclass(frozen=True, slots=True)
class Coefficient:
    """One number and how it was established. Claiming `measured` without a source refuses."""

    name: str
    value: int
    provenance: str
    source: str = ""

    def __post_init__(self) -> None:
        if self.provenance not in PROVENANCE:
            raise PlanRefusal(
                "coefficient_provenance",
                f"{self.name}: {self.provenance!r} is not one of {PROVENANCE}",
            )
        if self.provenance in ("measured", "ledger") and not self.source:
            raise PlanRefusal(
                "coefficient_unsourced",
                f"{self.name} claims {self.provenance} with no source: a number rendered as "
                "a measurement without naming what measured it is the thing this refuses",
            )

    @property
    def enforceable(self) -> bool:
        """Only a MEASURED-basis number may refuse or select a worse rung."""
        return self.provenance in ("measured", "ledger")

    def document(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "bytes": self.value,
            "provenance": self.provenance,
            "source": self.source,
            "enforceable": self.enforceable,
        }


class PlanChooser:
    """Pure arithmetic: request shape + bindings + LIVE LEDGER -> ONE AttemptPlan.

    cr-008a measured the defect this replaces: the chooser priced against a DECLARED
    `vram_bytes` budget and never read actual free VRAM, so a card with a neighbour on it
    was priced as an empty one and the refusal arrived from the allocator instead of from
    the plan. `Ledger.device_allocatable` is the measured number it reads now — the
    ALLOCATOR's view (driver-free plus the allocator's own reserved-but-unused segments),
    never arithmetic free bytes.

    It still takes NO I/O of any kind. A probe, a micro-benchmark or a decode observed
    inside `choose` is a defect by construction: every input is a number somebody else
    already measured, and the whole walk is demand arithmetic plus a first-fit over a
    PRECOMPUTED ordered set.
    """

    def __init__(self, ledger: Ledger) -> None:
        self.ledger = ledger
        self.choices = 0
        #: every choice's ladder walk, for the envelope and the triage bundle
        self.walks: list[dict[str, Any]] = []

    # ------------------------------------------------------------------ demand

    def demand(self, binding: BindingPlan, features: dict[str, Any]) -> list[Coefficient]:
        """The attempt's byte demand, each term declaring how it was established."""
        resident = self.ledger.resident or {}
        weights = sum(resident.values())
        terms = [
            Coefficient(
                "weights",
                weights or binding.vram_bytes,
                "ledger" if resident else "bound",
                "the fill plane's committed destination bytes per component" if resident else "",
            )
        ]
        if self.ledger.activations_peak > 0:
            terms.append(
                Coefficient(
                    "activations",
                    self.ledger.activations_peak,
                    "measured",
                    "max_memory_allocated() minus the post-fill baseline, banked per serve",
                )
            )
        else:
            # No serve has been banked on this generation, so there is no measurement. The
            # authored bound is LABELLED and cannot arm enforcement: an uncalibrated number
            # may not refuse and may not select a worse rung.
            terms.append(Coefficient("activations", 1 << 30, "bound", ""))
        if self.ledger.out_of_allocator > 0:
            terms.append(
                Coefficient(
                    "out_of_allocator",
                    self.ledger.out_of_allocator,
                    "measured",
                    "NVML per-process bytes minus the allocator's reservation",
                )
            )
        return terms

    # ------------------------------------------------------------------ the walk

    def choose(self, binding: BindingPlan, features: dict[str, Any]) -> AttemptPlan:
        self.choices += 1
        free_host = self.ledger.host_budget - self.ledger.host_charged
        if binding.host_bytes > free_host:
            raise PlanRefusal(
                "constraint_infeasible",
                f"no proven plan fits: host_ram needs {binding.host_bytes} B and "
                f"{free_host} B are free",
                {
                    "resource": "host_ram",
                    "scope": "plan",
                    "needed_bytes": binding.host_bytes,
                    "available_bytes": free_host,
                    "evidence_class": "measured",
                },
            )
        terms = {c.name: c for c in self.demand(binding, features)}
        resident = self.ledger.resident or {}
        # `device_allocatable` is what a NEW reservation can actually get. It already
        # accounts for whatever this generation holds, so the ladder's arithmetic is over
        # HEADROOM: what the attempt needs ON TOP of the weights that are already there.
        allocatable = self.ledger.allocatable
        largest = max(resident.values(), default=binding.vram_bytes)
        activations = terms["activations"]
        rows: list[dict[str, Any]] = []
        chosen = ""
        for placement, why, status in RUNGS:
            if status != "proven":
                rows.append({"rung": placement, "verdict": "unproven", "why": why})
                continue
            # Both rungs need the SAME headroom above the weights — activations do not
            # shrink because a component moved. What differs is the CAPACITY each rung can
            # raise: all_resident has only what is free right now, while component_staged
            # may free the LRU component first, so its capacity is `allocatable + largest`.
            headroom = activations.value
            capacity = allocatable if placement == "all_resident" else allocatable + largest
            fits = allocatable < 0 or capacity >= headroom
            row = {
                "rung": placement,
                "verdict": "fits" if fits else "short",
                "needs_bytes": headroom,
                "capacity_bytes": capacity,
                "allocatable_bytes": allocatable,
                "why": why,
            }
            rows.append(row)
            if fits and not chosen:
                chosen = placement
                row["verdict"] = "chosen"

        walk = {
            "coefficients": [c.document() for c in terms.values()],
            "allocatable_bytes": allocatable,
            "resident": dict(resident),
            "rungs": rows,
            "tenancy": binding.tenancy,
        }
        self.walks.append(walk)
        del self.walks[:-16]
        if not chosen:
            chosen, confession = self._deepen(binding, activations, allocatable, walk)
        else:
            confession = ""
        return AttemptPlan(
            delivery="native",
            materialization="aot_decode",
            compute_dtype="float16",
            placement=chosen,
            reserved_vram_bytes=sum(resident.values()) or binding.vram_bytes,
            reserved_host_bytes=binding.host_bytes,
            reserved_pinned_bytes=binding.pinned_bytes,
            headroom_bytes=activations.value,
            entrypoint_binding_plan_id=binding.plan_id,
            model_construction_digest=binding.model_construction_digest,
            quantified_choice=(
                f"{chosen}: {len(resident) or 1} component(s) totalling "
                f"{sum(resident.values()) or binding.vram_bytes} B with "
                f"{allocatable} B allocatable and {activations.value} B of activations "
                f"({activations.provenance}); request {features.get('entrypoint', '?')} "
                f"{features.get('scalars', {})}{confession}"
            ),
        )

    def _deepen(
        self,
        binding: BindingPlan,
        activations: Coefficient,
        allocatable: int,
        walk: dict[str, Any],
    ) -> tuple[str, str]:
        """Nothing on the ladder fits. What happens next depends on TENANCY (§3.2).

        LOCAL single-tenant: capacity NEVER refuses above the physical floor. The ladder
        deepens to the deepest authorized rung with a quantified confession, because the
        person in front of it wants their image, not a lecture about their card.

        SHARED cloud: a shortfall is a RESCHEDULE-shaped fact for the coordinator, never a
        structural-incompatibility claim — a different worker may fit it exactly.
        """
        floor = binding.vram_floor_bytes
        if not activations.enforceable:
            # An UNCALIBRATED coefficient MAY NOT REFUSE. It may deepen the ladder, because
            # a placement rung changes weight placement only — never the kernel, never the
            # request, never the bits (proven: the staged and resident plans produce the
            # same image digest). The v1 incident this protects against is a refusal on an
            # uncalibrated headroom charge, on the very card that was serving the lane.
            deepest = next(p for p, _, status in reversed(RUNGS) if status == "proven")
            walk["deepened_to"] = deepest
            walk["deepened_on"] = "uncalibrated"
            return deepest, (
                f" | UNCALIBRATED: no serve has been banked on this generation, so the "
                f"{activations.value} B activation envelope is an authored bound. It may "
                f"deepen placement (which changes no bits) and it may NOT refuse, so the "
                f"ladder goes to {deepest} and the first serve banks the measurement"
            )
        if binding.tenancy == "shared":
            raise PlanRefusal(
                "capacity_shortfall",
                f"no proven rung fits on THIS worker: {activations.value} B of headroom "
                f"needed and {allocatable} B allocatable. This is a capacity fact about one "
                "worker, not a structural incompatibility - a retry must select a different "
                "declared plan, and another worker may fit this one exactly",
                {
                    "resource": "vram",
                    "scope": "worker",
                    "needed_bytes": activations.value,
                    "available_bytes": max(allocatable, 0),
                    "evidence_class": "measured",
                },
            )
        if floor and allocatable >= 0 and allocatable < floor:
            raise PlanRefusal(
                "capacity_floor",
                f"below the PHYSICAL FLOOR: this deployment declares {floor} B for "
                f"activations plus scratch plus one component chunk, and {allocatable} B "
                f"are allocatable (short by {floor - allocatable} B). There is no rung "
                "below the floor, so this exits with a quantified capacity code rather "
                "than a plan that cannot run",
                {
                    "resource": "vram",
                    "scope": "physical_floor",
                    "needed_bytes": floor,
                    "available_bytes": max(allocatable, 0),
                    "evidence_class": "measured",
                },
            )
        deepest = next(p for p, _, status in reversed(RUNGS) if status == "proven")
        walk["deepened_to"] = deepest
        return deepest, (
            f" | DEGRADED: needed {activations.value} B of headroom, had "
            f"{max(allocatable, 0)} B, short by {max(activations.value - allocatable, 0)} B; "
            f"the ladder deepened to {deepest} rather than refusing a single-tenant request "
            "above its physical floor"
        )
