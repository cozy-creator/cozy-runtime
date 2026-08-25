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

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from cozy_runtime.internal import canonical
from cozy_runtime.internal.delivery import ROUTE_KERNEL, RUNGS_BY_NAME, Variant
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
    delivery_variant: str = ""
    """WHICH stored bytes served (cr-008c). Inside the document because `plan_digest` is an
    identity a coordinator holds, and two plans that read different artifacts of one model
    are not the same plan however identical their placement."""

    def document(self) -> dict[str, Any]:
        return {
            "format": "cozy.runtime.AttemptPlan/1",
            "delivery": self.delivery,
            "materialization": self.materialization,
            "delivery_variant": self.delivery_variant,
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

    @classmethod
    def for_job(cls, binding: JobBinding, caps: Mapping[str, int]) -> AttemptPlan:
        """The JOB lane's plan (cr-009). One attempt, one bounded envelope, no ladder.

        A job has no placement ladder to walk because it has no serving residency to trade
        against: it is one run-to-completion attempt that owns its devices for its whole
        life. The document is still a real AttemptPlan so `plan_digest` on AttemptAccepted
        stays an identity a coordinator can hold — the same fence, one lane over.
        """
        return cls(
            delivery="job",
            materialization="run_to_completion",
            compute_dtype="",
            placement="job_attempt",
            reserved_vram_bytes=int(caps.get("vram", 0)),
            reserved_host_bytes=int(caps.get("rss", 0)),
            reserved_pinned_bytes=0,
            headroom_bytes=0,
            entrypoint_binding_plan_id=binding.job_descriptor_id,
            model_construction_digest="",
            quantified_choice=(
                f"job_attempt: {binding.job!r} on {max(binding.gpu_count, 0)} declared device "
                f"floor, {caps.get('rss', 0)} B host and {caps.get('disk', 0)} B disk capped, "
                f"publishes={binding.publishes} emits_media={binding.emits_media}"
            ),
        )


@dataclass(frozen=True, slots=True)
class JobBinding:
    """The local resolution of one `job_descriptor_id` — the JOB lane's `BindingPlan`.

    Same seam, one lane over: the coordinator names the job by DIGEST and never ships a
    path; the worker resolves that digest against its own records, a small local file under
    `<cozy_home>/job-plans/`. On the network path th-004 ships the build and tfs-003 the
    materialized input trees; nothing below this record can tell the difference.

    There is deliberately no residency, no component set and no construction digest here. A
    job has no model residency: no warm pool, no serving pointer, no LRU, no idle dwell and
    no second network door (§2). A job that declares read-only Model params constructs them
    for THAT attempt through the ordinary Loader path and never retains them.
    """

    job_descriptor_id: str
    build_id: str
    project: str
    job: str
    """The registered `@app.job` name this descriptor id resolves to."""
    gpu_count: int = 0
    publishes: bool = False
    emits_media: bool = False
    gpu_rate_micro_usd_per_hour: int = 0
    """The GPU rate as an INTEGER of micro-USD: canonical documents are integer-only, and a
    cost fact that cannot be canonicalized cannot be journaled. It arrives on the typed
    config channel and never from environment (§2)."""
    cap_micro_usd: int = 0
    reclaim_on_terminal: bool = True

    @classmethod
    def read(cls, record: dict[str, Any]) -> JobBinding:
        known = {f for f in cls.__dataclass_fields__}
        unknown = sorted(set(record) - known)
        if unknown:
            raise PlanRefusal(
                "job_plan_unknown_field",
                f"job plan record carries {', '.join(unknown)}: the record's key set is "
                "closed, exactly like the documents it stands in for",
            )
        return cls(**record)


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
    variants: tuple[Variant, ...] = ()
    """cr-008c: the artifact variants this deployment OFFERS for one logical model —
    encodings of the same topology. Empty means the record offers exactly what `store` and
    `snapshot` name, which is what every single-variant record already meant."""
    objective: str = "latency"
    """The MEASURED axis the delivery ladder is ordered on (`latency | footprint |
    fidelity`). A deployment property, not a request knob: delivery decides which bytes are
    resident, so it is chosen once for the generation."""
    steps_basis: int = 20
    """The declared step count the `latency` objective prices a per-step cost against. It
    is a LABEL on the arithmetic, not a claim about any one request."""
    delivery_rung: str = ""
    """RESOLVED AT PREPARE, never written by the deployment: which `DELIVERY_RUNGS` entry
    the executor's `DeliveryChooser` took. Empty means `verbatim`."""
    delivery_variant: str = ""
    """RESOLVED AT PREPARE: which offered variant the generation actually read."""
    max_input_bytes: int = 64 << 20
    """cr-012: the DEPLOYMENT's per-input-asset ceiling, checked pre-accept against the
    grant's declared length. A field's own `AssetBound` narrows it and never widens it, so
    the two bounds live where their knowledge does: the deployment's here, the endpoint's
    on the field."""
    max_inputs_total_bytes: int = 256 << 20
    """The whole attempt's input budget. Input retention is its own bounded story: inputs
    take no CAS lease and end with the attempt spool."""
    egress_allowed_hosts: str = ""
    """Comma-joined hosts this deployment may fetch a remote grant from. EMPTY MEANS NO
    EGRESS: a `file://` grant needs none, and a deployment that has not declared a host
    cannot be talked into one at request time."""

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
        """Two binding plans over the SAME construction share one executor generation.

        The OFFERED variant set and the objective are inside the key because they decide
        which bytes get filled: two bindings that would make different delivery choices are
        not one construction, however identical the rest of the record reads.
        """
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
                "variants": [v.name for v in self.variants],
                "objective": self.objective,
            }
        )

    def offered(self) -> tuple[Variant, ...]:
        """The candidate set, with the single-variant record's implicit one made explicit.

        A record that names no variants offers exactly one — the bytes it already names —
        and that one IS the reference, because there is nothing to compare it against.
        """
        if self.variants:
            return self.variants
        return (
            Variant(
                name=self.variant or "default",
                store=self.store,
                snapshot=self.snapshot,
                snapshots=self.snapshots,
                reference=True,
            ),
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
        data = dict(record)
        offers = data.get("variants")
        if offers:
            data["variants"] = tuple(
                v if isinstance(v, Variant) else Variant.read(v) for v in offers
            )
        return cls(**data)


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
        absent = self.ledger.evicted or {}
        # The GENERATION's weights, resident or parked. A component a rung evicted is still
        # this deployment's weight — it costs its full size to bring back — so a demand term
        # that counted only what happened to be resident described a smaller model than the
        # one being served.
        weights = sum(resident.values()) + sum(absent.values())
        terms = [
            Coefficient(
                "weights",
                weights or binding.vram_bytes,
                "ledger" if resident or absent else "bound",
                "the fill plane's committed destination bytes per component, resident and "
                "parked alike"
                if resident or absent
                else "",
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
        if self.ledger.unreconciled:
            # A CHOOSER MAY NOT PRICE AGAINST A LEDGER THAT IS KNOWN TO BE WRONG. This is the
            # loud half of cl-003's defect: the drift existed, this repository's own
            # reconciliation had already NAMED it, and the next request was priced against it
            # anyway — so a capacity fact arrived as an OOM in the middle of a VAE decode and
            # was reported as the endpoint's exception. An unreconciled ledger reconciles
            # (a fresh generation re-establishes it) or it refuses; it never guesses.
            raise PlanRefusal(
                "ledger_unreconciled",
                f"this worker's ledger is not a truth right now and no rung may be priced "
                f"against it: {self.ledger.unreconciled}. The generation is invalidated and "
                "rebuilt, which re-establishes every number from a real construction; until "
                "then a plan chosen here would be arithmetic over a residency the worker "
                "already knows it cannot vouch for",
                {
                    "resource": "vram",
                    "scope": "ledger",
                    "needed_bytes": 0,
                    "available_bytes": max(self.ledger.allocatable, 0),
                    "evidence_class": "measured",
                },
            )
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
        # THE PHYSICAL FLOOR, before the ladder rather than after it. The floor is the
        # minimum device room ANY rung needs — activations plus scratch plus one component
        # chunk — so a device that cannot provide it admits no rung, and walking a ladder to
        # discover that is walking it for nothing. Checking it here is also what makes the
        # refusal reachable: `_deepen` only runs when nothing fits, and an UNCALIBRATED
        # coefficient never gets that far (#293, and correctly — it may not refuse). The
        # floor is not a coefficient: it is a DECLARED deployment fact compared against a
        # MEASURED allocatable, so nothing uncalibrated participates in it.
        floor = binding.vram_floor_bytes
        allocatable_now = self.ledger.allocatable
        # Against the DEEPEST rung's capacity, not against bare allocatable. Once the
        # weights are resident, `allocatable` is only what is left beside them — but the
        # staged rung can free the largest component, so "no rung exists" means "not even
        # after evicting everything evictable". Comparing against the bare number refused a
        # run that the ladder serves perfectly well, which is what running it showed.
        deepest_capacity = allocatable_now + sum((self.ledger.resident or {}).values())
        if floor and allocatable_now >= 0 and deepest_capacity < floor:
            raise PlanRefusal(
                "capacity_floor",
                f"below the PHYSICAL FLOOR: this deployment declares {floor} B for "
                f"activations plus scratch plus one component chunk, and the deepest rung "
                f"can raise {deepest_capacity} B ({allocatable_now} B allocatable plus every "
                f"resident component it may free; short by {floor - deepest_capacity} B). "
                f"There is no rung "
                "below the floor, so this exits with a quantified capacity code rather "
                "than a plan that cannot run",
                {
                    "resource": "vram",
                    "scope": "physical_floor",
                    "needed_bytes": floor,
                    "available_bytes": max(deepest_capacity, 0),
                    "evidence_class": "measured",
                },
            )
        terms = {c.name: c for c in self.demand(binding, features)}
        resident = self.ledger.resident or {}
        # THE COMPONENTS THIS GENERATION DECLARES BUT IS NOT HOLDING. They are not free: a
        # rung that wants them resident has to bring every one of them BACK, and the walk
        # below charges exactly that. Reading only the resident map is how a 1024px request
        # came to be planned `all_resident` on a generation holding one 167 MB VAE, with the
        # 6.3 GiB it would have to re-stage charged to nobody (cl-003).
        absent = self.ledger.evicted or {}
        absent_bytes = sum(absent.values())
        declared_bytes = sum(resident.values()) + absent_bytes
        # `device_allocatable` is what a NEW reservation can actually get. It already
        # accounts for whatever this generation holds, so the ladder's arithmetic is over
        # HEADROOM: what the attempt needs ON TOP of the weights that are already there.
        allocatable = self.ledger.allocatable
        largest = max((*resident.values(), *absent.values()), default=binding.vram_bytes)
        activations = terms["activations"]
        rows: list[dict[str, Any]] = []
        chosen = ""
        # THE CONSTRUCTION FLOOR. A generation whose deployment ceiling PARKED components
        # is staged as a FACT — the components are on `meta` and `stage()` moves them in on
        # first declared use. Headroom arithmetic can still say `all_resident` fits, and
        # cr-008c's cross caught exactly that: a plan reporting `all_resident` for a
        # generation with a parked UNet. The ladder may deepen below the floor; it may
        # never claim to be above it.
        parked = tuple(self.ledger.parked)
        for placement, why, status in RUNGS:
            if status != "proven":
                rows.append({"rung": placement, "verdict": "unproven", "why": why})
                continue
            if parked and placement == "all_resident":
                rows.append(
                    {
                        "rung": placement,
                        "verdict": "construction_parked",
                        "why": f"construction parked {list(parked)} under this deployment's "
                        "resident ceiling: the components are not held, so no plan over "
                        "this generation is all-resident",
                    }
                )
                continue
            # Both rungs need the same ACTIVATION headroom — activations do not shrink
            # because a component moved — and each additionally needs the WEIGHTS IT MUST
            # BRING BACK, which is where the two differ:
            #
            #   all_resident      needs every absent component back, and can free nothing;
            #                     its capacity is what is allocatable right now.
            #   component_staged  holds the declared set at a time, so the most it must
            #                     raise at once is the LARGEST declared component — and it
            #                     may free every resident one first, so its capacity is
            #                     `allocatable + resident`.
            #
            # On a generation with everything resident, `absent_bytes` is 0 and this is the
            # arithmetic that was already there. On one that staged, it is the difference
            # between a rung that fits and a kernel OOM.
            if placement == "all_resident":
                headroom = activations.value + absent_bytes
                capacity = allocatable
            else:
                headroom = activations.value + largest
                capacity = allocatable + sum(resident.values())
            fits = allocatable < 0 or capacity >= headroom
            row = {
                "rung": placement,
                "verdict": "fits" if fits else "short",
                "needs_bytes": headroom,
                "capacity_bytes": capacity,
                "allocatable_bytes": allocatable,
                "absent_bytes": absent_bytes,
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
            "absent": dict(absent),
            "declared_bytes": declared_bytes,
            "parked": list(parked),
            "rungs": rows,
            "tenancy": binding.tenancy,
        }
        self.walks.append(walk)
        del self.walks[:-16]
        if not chosen:
            chosen, confession = self._deepen(binding, activations, allocatable, walk)
        else:
            confession = ""
        rung = self._delivery(binding, chosen, walk)
        return AttemptPlan(
            delivery=rung.delivery,
            materialization=rung.materialization,
            compute_dtype="float16",
            placement=chosen,
            # THE GENERATION'S WEIGHTS, resident or not. A reservation that shrank to
            # whatever happened to be resident made the plan summary read `vram=167307726B`
            # for a 6.46 GiB pipeline — the VAE alone — which is the number cl-003 caught the
            # defect by. What this deployment holds does not change because a rung parked it.
            reserved_vram_bytes=declared_bytes or binding.vram_bytes,
            reserved_host_bytes=binding.host_bytes,
            reserved_pinned_bytes=binding.pinned_bytes,
            headroom_bytes=activations.value,
            entrypoint_binding_plan_id=binding.plan_id,
            model_construction_digest=binding.model_construction_digest,
            delivery_variant=binding.delivery_variant,
            quantified_choice=(
                f"{rung.delivery}/{rung.materialization or 'verbatim'} on variant "
                f"{binding.delivery_variant or binding.variant or 'default'!r} "
                f"({rung.cost_class} decode, {ROUTE_KERNEL[rung.route]} kernel) x "
                f"{chosen}: {len(resident) + len(absent) or 1} component(s) totalling "
                f"{declared_bytes or binding.vram_bytes} B of which {absent_bytes} B is not "
                f"resident, with "
                f"{allocatable} B allocatable and {activations.value} B of activations "
                f"({activations.provenance}); request {features.get('entrypoint', '?')} "
                f"{features.get('scalars', {})}{confession}"
            ),
        )

    def _delivery(self, binding: BindingPlan, placement: str, walk: dict[str, Any]) -> Any:
        """THE CROSS (cr-008c): the resolved delivery rung, crossed with the placement one.

        Delivery was chosen ONCE, at prepare, against measured envelopes — this function
        does not re-choose it and structurally cannot: it has no table, no device and no
        candidate set. What it does is read the placement the ladder just walked to and
        DERIVE the honest cost-class label, because a float delivery under a staged
        placement is not the same fact as one under all_resident: the decode is paid again
        on every stage-in (per_transition), not once for the generation (per_generation).
        That is pgw#1505's distinction, and collapsing it would price a staged encoded
        component at a cost it only pays the first time.

        Then it FENCES the axis: the rung it hands back must compute with the same kernel
        the prepare-time choice did. §3.2's claim is that a placement rung changes weight
        placement only and never the compute kernel; this is where that claim is checked
        rather than restated, so a delivery switch arriving as a placement side effect is a
        refusal instead of a silent reinterpretation of the request.
        """
        name = binding.delivery_rung or "verbatim"
        rung = RUNGS_BY_NAME.get(name)
        if rung is None:
            raise PlanRefusal(
                "unknown_delivery_rung",
                f"the binding resolved delivery rung {name!r} and the closed ladder holds "
                f"{sorted(RUNGS_BY_NAME)}; a rung nothing implements cannot be planned "
                "against",
            )
        crossed = rung
        if rung.name == "aot_decode" and placement == "component_staged":
            crossed = RUNGS_BY_NAME["staged_decode"]
        if ROUTE_KERNEL[crossed.route] != ROUTE_KERNEL[rung.route]:
            raise PlanRefusal(
                "delivery_side_effect",
                f"the placement walk to {placement!r} moved delivery from {rung.name!r} "
                f"({ROUTE_KERNEL[rung.route]}) to {crossed.name!r} "
                f"({ROUTE_KERNEL[crossed.route]}). A placement rung changes WEIGHT "
                "PLACEMENT only and never the compute kernel (§3.2); a delivery switch is "
                "an explicit, separately authorized choice made once at prepare, never "
                "something a capacity ladder does on the way past",
                {
                    "resource": "delivery",
                    "scope": "plan",
                    "needed_bytes": 0,
                    "available_bytes": 0,
                    "evidence_class": "structural",
                },
            )
        walk["delivery"] = {
            "variant": binding.delivery_variant,
            "rung": crossed.name,
            "delivery": crossed.delivery,
            "materialization": crossed.materialization,
            "cost_class": crossed.cost_class,
            "kernel": ROUTE_KERNEL[crossed.route],
            "chosen_at": "prepare",
        }
        return crossed

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
        # The floor was already checked at the top of `choose`: reaching here means the
        # device HAS the floor and the ladder still does not fit, which is the degrade case.
        deepest = next(p for p, _, status in reversed(RUNGS) if status == "proven")
        walk["deepened_to"] = deepest
        return deepest, (
            f" | DEGRADED: needed {activations.value} B of headroom, had "
            f"{max(allocatable, 0)} B, short by {max(activations.value - allocatable, 0)} B; "
            f"the ladder deepened to {deepest} rather than refusing a single-tenant request "
            "above its physical floor"
        )
