"""The PlanChooser, the AttemptPlan, and the local resolution of a binding.

Two things live together because they are one decision made in one place, before durable
acceptance (§3.2). The ledger they are decided AGAINST is `worker/ledger.py`.

* **PlanChooser** — pure arithmetic over the request shape, the bindings and the live
  ledger, returning ONE AttemptPlan. At cr-007 it is deliberately a SINGLE-plan chooser
  behind the real offer/prepare/accept seam: the closed proven set has one member, so the
  first-fit walk has one row. cr-008b replaces the body of `choose` and nothing else — no
  protocol change, no new state, no forward dependency. A probe or a benchmark observed
  inside `choose` is a defect by construction: this function takes no I/O of any kind.
* **ExecutorPreparer** — applies the chosen plan and DECIDES NOTHING. It is the reason
  "accepted" never includes seconds of unrecorded preparation: the worker prepares
  first, journals second.

The AttemptPlan is an in-process typed value. ``plan_digest`` is a domain-separated hash of its
closed projection, while the typed protocol summary tells the RecordOwner what was chosen.  It is
not a separately stored or version-negotiated document.
"""

from __future__ import annotations

import dataclasses
import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from cozy_runtime.internal import canonical
from cozy_runtime.internal.delivery import RUNGS_BY_NAME
from cozy_runtime.internal.resolution import ROUTE_KERNEL
from cozy_runtime.internal.worker.ledger import Ledger

UNCALIBRATED_ACTIVATION_BYTES = 1 << 30


def shape_cell(features: Mapping[str, int]) -> str:
    return ",".join(f"{axis}={value}" for axis, value in sorted(features.items())) or "-"


class PlanRefusal(Exception):
    """A typed pre-acceptance refusal, carrying the structured shortfall when it has one."""

    def __init__(self, code: str, detail: str, shortfall: dict[str, Any] | None = None) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail
        self.shortfall = shortfall


@dataclass(frozen=True, slots=True)
class PreparedRequest:
    """ONE request, resolved ONCE by the executor's kernel, as the worker holds it.

    The resolved payload is an author-typed object and cannot cross a process boundary, so
    it stays in the executor under `request_id` and this carries only what a plan can be
    priced against and process records can hold: the normalized shape axes and three digests.
    `invoke` runs the record these digests describe, so the plan and the handler agree by
    construction rather than by two derivations happening to match (codex audit, adopted
    2026-08-25 — they did not: the planner read an ad hoc scalar projection of the wire
    document and the real features were computed later, inside invocation).
    """

    request_id: str
    entrypoint: str
    features: Mapping[str, int]
    features_digest: str
    overlay_digest: str
    facts_digest: str
    adjustments: tuple[Mapping[str, str], ...] = ()

    def cell(self) -> str:
        """The normalized SHAPE CELL demand is banked and read per — never the whole
        request. Two 512px requests are one cell however their prompts differ."""
        return shape_cell(self.features)

    @classmethod
    def unresolved(cls, entrypoint: str, features: Mapping[str, int]) -> PreparedRequest:
        """A request NOBODY RESOLVED — what `fit` prices a hypothetical against.

        There is no package process to prepare it, so the three digests are EMPTY and say
        so. An admission path that reached for this would be pricing against an identity it
        cannot bind, which is the thing the record exists to stop.
        """
        return cls(
            request_id="",
            entrypoint=entrypoint,
            features=dict(features),
            features_digest="",
            overlay_digest="",
            facts_digest="",
        )


@dataclass(frozen=True, slots=True)
class AttemptPlan:
    """One exact choice. `document()` is its identity; `summary()` is its projection."""

    delivery: str
    materialization: str
    compute_dtype: str
    placement: str
    reserved_vram_bytes: int
    """The wire's word for the generation's WEIGHT bytes (th-024 owns the field name). It is
    not the resident set and it is not this attempt's activation envelope — `resident_bytes`
    and `headroom_bytes` are those, named separately because they are different facts."""
    headroom_bytes: int
    entrypoint_binding_digest: str
    decision_explanation: str
    """WHY this plan, in numbers, for a person reading a terminal. It rides the wire as
    `AttemptPlanSummary.quantified_choice`, which is th-024's name for the field."""
    constructed_model_digest: str = ""
    """WHAT WAS ACTUALLY BUILT: the selected bytes, the route that decodes them, the
    construction and the exact selected snapshot. The one
    construction identity — the MCC's "logical model" digest beside it is deleted (D12)."""
    request_features_digest: str = ""
    """The exact PreparedRequest this plan was priced against, bound into the plan document
    so `plan_digest` covers it. cr-002a claimed this end to end; it is true now."""
    preflight_facts_digest: str = ""
    overlay_digest: str = ""
    shape_cell: str = ""
    """The normalized shape cell — the key demand is banked and read per."""
    model_weight_bytes: int = 0
    """The generation's weights, resident or parked. What it COSTS to hold this model."""
    resident_bytes: int = 0
    """What is on the device right now. A staged rung moves this and never the weights."""
    delivery_variant: str = ""
    """WHICH stored bytes served (cr-008c). Inside the document because `plan_digest` is an
    identity a RecordOwner holds, and two plans that read different artifacts of one model
    are not the same plan however identical their placement."""
    scope_headroom_bytes: tuple[tuple[str, int], ...] = ()
    """Measured activation scratch by component-use method. The scalar headroom is its max;
    staged admission uses each method's own value so unrelated weight/scratch maxima never add."""

    def document(self) -> dict[str, Any]:
        return {
            "delivery": self.delivery,
            "materialization": self.materialization,
            "delivery_variant": self.delivery_variant,
            "compute_dtype": self.compute_dtype,
            "placement": self.placement,
            "reserved_vram_bytes": self.reserved_vram_bytes,
            "headroom_bytes": self.headroom_bytes,
            "scope_headroom_bytes": dict(self.scope_headroom_bytes),
            "entrypoint_binding_digest": self.entrypoint_binding_digest,
            "constructed_model_digest": self.constructed_model_digest,
            "request_features_digest": self.request_features_digest,
            "preflight_facts_digest": self.preflight_facts_digest,
            "overlay_digest": self.overlay_digest,
            "shape_cell": self.shape_cell,
            "model_weight_bytes": self.model_weight_bytes,
            "resident_bytes": self.resident_bytes,
        }

    def digest(self) -> str:
        encoded = canonical.write(self.document())
        return "sha256:" + hashlib.sha256(b"cozy.runtime.attempt-plan\0" + encoded).hexdigest()

    @classmethod
    def for_job(
        cls, binding: JobBinding, caps: Mapping[str, int], *, device_count: int
    ) -> AttemptPlan:
        """The JOB lane's plan (cr-009). One attempt, one bounded envelope, no ladder.

        A job has no placement ladder to walk because it has no serving residency to trade
        against: it is one run-to-completion attempt that owns its devices for its whole
        life. The plan still has a stable domain-separated digest on AttemptAccepted — the same
        fence, one lane over — without pretending the private planner value is a file format.
        """
        return cls(
            delivery="job",
            materialization="run_to_completion",
            compute_dtype="",
            placement="job_attempt",
            reserved_vram_bytes=int(caps.get("vram", 0)),
            headroom_bytes=0,
            entrypoint_binding_digest=binding.job_descriptor_id,
            decision_explanation=(
                f"job_attempt: {binding.job!r} on {max(device_count, 0)} assigned device(s), "
                f"{caps.get('rss', 0)} B host and {caps.get('disk', 0)} B disk capped, "
                f"publishes={binding.publishes} emits_media={binding.emits_media}"
            ),
        )


@dataclass(frozen=True, slots=True)
class JobBinding:
    """The local resolution of one `job_descriptor_id` — the JOB lane's `BindingPlan`.

    Same seam, one lane over: the RecordOwner names the job by DIGEST and never ships a
    path; the worker resolves that digest against its own records, a small local file under
    `<cozy_home>/job-plans/`. On the network path th-004 ships the build and tfs-003 the
    materialized input trees; nothing below this record can tell the difference.

    There is deliberately no residency, no component set and no construction digest here. A
    job has no model residency: no warm pool, no serving pointer, no LRU, no idle dwell and
    no second network door (§2). A job Model parameter is a derive-only typed view of one
    attempt-held TensorFS Manifest; package load/component access is unavailable.
    """

    job_descriptor_id: str
    build_id: str
    application: str
    package_interface: str
    python: str
    environment_content_digest: str
    job: str
    """The registered `@app.job` name this job descriptor id resolves to."""
    publishes: bool = False
    emits_media: bool = False
    gpu_rate_micro_usd_per_hour: int = 0
    """The GPU rate as an INTEGER of micro-USD: canonical documents are integer-only, and a
    cost fact that cannot be canonicalized cannot be recorded. It arrives on the typed
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
class PreparedModel:
    """What ONE `prepare` RESOLVED. Output only: a deployment never writes any of it.

    It lived on the binding record beside the deployment's own declarations, so one type
    carried both what was asked for and what came back, and a record could arrive claiming
    a delivery rung nothing had chosen (codex audit, adopted 2026-08-25). Two types, and
    the record's closed key set no longer admits a resolved field at all.
    """

    constructed_model_digest: str = ""
    """WHAT WAS BUILT: selected bytes, route, materialization, and tensor schema."""
    delivery_rung: str = ""
    """Which `DELIVERY_RUNGS` entry the executor's `DeliveryChooser` took. Empty means
    `verbatim`."""
    delivery_variant: str = ""
    """Which offered variant this generation actually read."""
    weight_bytes: int = 0
    """What the fill COMMITTED, measured. Distinct from the deployment's declared
    `model_weight_bytes` and from the ledger's live resident map — three different facts
    that were all called some form of "vram bytes"."""


@dataclass(frozen=True, slots=True)
class ModelBinding:
    """One exact model slot inside an entrypoint's complete binding set."""

    model_class: str
    model_binding_path: str
    model_parameter_name: str
    store: str
    variant: str
    reference_snapshot: str
    components: tuple[str, ...]
    snapshots: Mapping[str, str]
    objective: str = "latency"
    steps_basis: int = 20
    model_weight_bytes: int = 0
    custody: str = "canonical"
    sequence_parallel_degrees: tuple[int, ...] = ()
    """The interface slot's `sequence_parallel.degrees` (cr-068): what the author said the
    class shards at. A placement pinned to K devices needs K here."""
    model: str = ""
    """`org/name@release/lane` — the checkpoint's catalog name, for the one line a person
    reads when the code does not fit it (model-code-fit §3). Resolution, never identity."""

    def snapshot_for(self, component: str) -> str:
        return self.snapshots.get(component, self.reference_snapshot)

    def key_document(self) -> dict[str, Any]:
        return {
            "store": self.store,
            "snapshot": self.reference_snapshot,
            "variant": self.variant,
            "model_class": self.model_class,
            "model_binding_path": self.model_binding_path,
            "model_parameter_name": self.model_parameter_name,
            "components": list(self.components),
            "snapshots": dict(self.snapshots),
            "objective": self.objective,
            "steps_basis": self.steps_basis,
            "model_weight_bytes": self.model_weight_bytes,
            "custody": self.custody,
        }

@dataclass(frozen=True, slots=True)
class DeclaredBinding:
    """Runtime's private resolved value for one canonical entrypoint binding.

    This is not a document or wire schema. Only the shared provisioned-release reader
    constructs it after following stored-byte references; paths below are machine resolution.
    """

    entrypoint_binding_digest: str
    entrypoint: str
    model_class: str
    model_binding_path: str
    model_parameter_name: str
    release: str
    models: tuple[ModelBinding, ...] = ()
    environment_digest: str = ""
    """The exact inline Environment selected beside this binding."""
    application: str = ""
    """The exact PackageInterface application imported from the installed venv."""
    interface_path: str = ""
    """Local path to the exact granted package-interface bytes; resolution, never identity."""
    development_project: str = ""
    """Launch-local source path for DevelopmentPackage discovery; never protocol identity."""
    store: str = ""
    variant: str = ""
    reference_snapshot: str = ""
    """The RESOLVED reference snapshot: the artifact the `variant` names, and the fallback
    for any component the snapshot map omits. Resolution, not identity."""
    components: tuple[str, ...] = ()
    """The components this binding constructs, in construction order."""
    snapshots: Mapping[str, str] = dataclasses.field(default_factory=dict)
    """component -> snapshot digest. A read plan is COMPLETE over its header (tfs-007), so a
    component cannot be read out of a whole-pipeline snapshot without reading the pipeline;
    per-component snapshots make a partial fill the store's ordinary complete read, and the
    objects dedup so it costs nothing."""
    development: bool = False
    """True only for explicit same-user source-checkout execution with no Environment."""
    objective: str = "latency"
    """The MEASURED axis the delivery ladder is ordered on (`latency | footprint |
    fidelity`). A deployment property, not a request knob: delivery decides which bytes are
    resident, so it is chosen once for the generation."""
    steps_basis: int = 20
    """The declared step count the `latency` objective prices a per-step cost against. It
    is a LABEL on the arithmetic, not a claim about any one request."""
    model_weight_bytes: int = 0
    """The DECLARED total weight of this model. Not what is resident, and not what an
    attempt reserves. RESOLUTION (#569b): it is the artifact's own byte total, read from the
    index row on the machine that holds it. A launcher constant here was the ninth field of
    the same mistake — `vramBudget = 7 * gib` priced every model ever bound, and the fill
    of a 92.42 GiB artifact refused `budget_overrun` against it."""
    custody: str = "canonical"
    """`canonical` (the border classified every key: a serve-time extra is drift and
    refuses) or `local` (an un-bordered local bind: extras are listed, ignored and
    confessed — model-code-fit D5). RESOLUTION: custody is how THIS machine came to hold
    these bytes, which is exactly what an index row records and what a record-writer on
    another machine cannot know."""
    package: str = ""
    """`org/name@release` — the package this binding belongs to, for the fit refusal's
    one human line. Resolution, never identity."""
    max_input_bytes: int = 64 << 20
    """cr-012: the DEPLOYMENT's per-input-asset ceiling, checked pre-accept against the
    grant's declared length. A field's own `AssetBound` narrows it and never widens it."""
    max_inputs_total_bytes: int = 256 << 20
    """The whole attempt's input budget. Input retention is its own bounded story: inputs
    take no CAS lease and end with the attempt spool."""

    @property
    def weightless(self) -> bool:
        """This deployment declares NO MODEL: no class, no store, no artifact.

        A weightless package is the only thing a cardless runner can ever serve, so it has
        to be servable — the boot path prepared every staged binding by FILLING it, and a
        binding with nothing to fill had no way through (cl-013). Its executor imports no
        torch and touches no device.
        """
        return not self.model_bindings()

    def model_bindings(self) -> tuple[ModelBinding, ...]:
        if self.models:
            return self.models
        if not self.model_class:
            return ()
        return (
            ModelBinding(
                model_class=self.model_class,
                model_binding_path=self.model_binding_path,
                model_parameter_name=self.model_parameter_name,
                store=self.store,
                variant=self.variant,
                reference_snapshot=self.reference_snapshot,
                components=self.components,
                snapshots=self.snapshots,
                objective=self.objective,
                steps_basis=self.steps_basis,
                model_weight_bytes=self.model_weight_bytes,
                custody=self.custody,
            ),
        )

    def sequence_parallel_degrees(self) -> tuple[int, ...]:
        """The degrees EVERY model slot of this binding declares: a group shards the whole
        construction, so a slot that declares nothing makes the binding unshardable."""
        degrees: set[int] | None = None
        for model in self.model_bindings():
            declared = set(model.sequence_parallel_degrees)
            degrees = declared if degrees is None else degrees & declared
        return tuple(sorted(degrees or ()))

    def snapshot_for(self, component: str) -> str:
        """This component's snapshot, or the binding's own when the map omits it."""
        return self.snapshots.get(component, self.reference_snapshot)

    def construction_key(self) -> str:
        """Two bindings over the SAME construction share one executor generation.

        The OFFERED variant set is inside the key BY ITS BYTES — store, snapshot and the
        per-component snapshot map, not the label somebody typed. A name is a deployment's
        word for a candidate, and two records offering different artifacts under the same
        words are two constructions; keying on names alone coalesced them into one
        generation, so the second binding was served by whichever bytes the first one
        filled (codex audit, adopted 2026-08-25). The objective belongs here for the same
        reason: it decides which of the candidates gets filled.
        """
        if self.weightless:
            # Nothing is constructed, so the RELEASES are the construction: every weightless
            # binding of one package release shares one executor. It keyed on `project` —
            # this machine's staging directory — which made the key machine-local for no
            # gain, since the package release names the same thing without a path (#506a).
            return canonical.digest(
                {
                    "weightless": True,
                    "release": self.release,
                }
            )
        return canonical.digest(
            {
                "release": self.release,
                "models": [model.key_document() for model in self.model_bindings()],
            }
        )


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

    def __init__(self, ledger: Ledger, *, degree: int = 1) -> None:
        self.ledger = ledger
        #: the lane's sequence-parallel degree (cr-068). Above 1 the ladder is REFUSED, not
        #: deepened: K ranks run ONE plan priced against the tightest card (the ledger's
        #: `device_free` is the MIN over the lane's devices), and a rung one rank would
        #: need to deepen is a rung the group cannot run.
        self.degree = int(degree)
        self.choices = 0
        #: every choice's ladder walk, for the envelope and the triage bundle
        self.walks: list[dict[str, Any]] = []

    # ------------------------------------------------------------------ demand

    def demand(self, declared: DeclaredBinding, prepared: PreparedRequest) -> list[Coefficient]:
        """The attempt's byte demand for THIS request's shape, each term declaring how it
        was established. It read a generation-wide activation maximum before and ignored
        the request entirely, which is the same number for every shape a deployment serves
        (codex audit, adopted 2026-08-25)."""
        return self._demand(declared, prepared.cell())

    def _demand(self, declared: DeclaredBinding, cell: str) -> list[Coefficient]:
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
                weights or declared.model_weight_bytes,
                "ledger" if resident or absent else "bound",
                "the fill plane's committed destination bytes per component, resident and "
                "parked alike"
                if resident or absent
                else "",
            )
        ]
        activations, source = self.ledger.activations_for(cell)
        if activations > 0:
            terms.append(Coefficient("activations", activations, "measured", source))
        else:
            # No serve has been banked on this generation, so there is no measurement. The
            # authored bound is LABELLED and cannot arm enforcement: an uncalibrated number
            # may not refuse and may not select a worse rung.
            terms.append(Coefficient("activations", UNCALIBRATED_ACTIVATION_BYTES, "bound", ""))
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

    def _walk(
        self, declared: DeclaredBinding, cell: str
    ) -> tuple[dict[str, Coefficient], list[dict[str, Any]], str, dict[str, Any]]:
        """The ladder walk over the LIVE ledger: demand terms, one row per rung, the first
        rung that fits (or ""), and the walk document (the rows ride it as `rungs`).
        `choose` turns the result into a plan; `fits` asks only whether a rung exists
        (cr-022's lane arbitration)."""
        terms = {c.name: c for c in self._demand(declared, cell)}
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
        # THE MOST A STAGED RUNG MUST RAISE AT ONCE. A component scope holds its WHOLE
        # declared set resident for the length of the body, so the charge is the biggest
        # deduplicated declared set — not the biggest single component, which is what this
        # priced before and which under-charges every method that declares several
        # (codex audit, adopted 2026-08-25). Never below the single-component figure: a
        # generation that reported no scopes is priced exactly as it was.
        sizes = {**absent, **resident}
        scoped = [
            sum(sizes.get(name, 0) for name in dict.fromkeys(names))
            for names in self.ledger.declared_scopes.values()
        ]
        largest = max([max(sizes.values(), default=declared.model_weight_bytes), *scoped])
        activations = terms["activations"]
        measured_scopes = self.ledger.activation_scopes_for(cell)
        scope_headrooms = {
            method: measured_scopes.get(method, activations.value)
            for method in self.ledger.declared_scopes
        }
        scope_demands = {
            method: sum(sizes.get(name, 0) for name in dict.fromkeys(names))
            + scope_headrooms[method]
            for method, names in self.ledger.declared_scopes.items()
        }
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
            unsuitable = self.ledger.placement_refusal(placement)
            if unsuitable:
                rows.append(
                    {
                        "rung": placement,
                        "verdict": "observed_unsuitable",
                        "why": unsuitable,
                    }
                )
                continue
            # Both rungs charge the same ACTIVATION headroom and differ in the WEIGHTS
            # they must bring back: `all_resident` needs every absent component and can
            # free nothing, so its capacity is bare allocatable; `component_staged` holds
            # one declared set at a time and may free every resident component first
            # (cl-003).
            if placement == "all_resident":
                headroom = activations.value + absent_bytes
                capacity = allocatable
            else:
                headroom = max(scope_demands.values(), default=activations.value + largest)
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
            "declared_scopes": {m: list(n) for m, n in self.ledger.declared_scopes.items()},
            "largest_declared_set_bytes": largest,
            "scope_headroom_bytes": dict(scope_headrooms),
            "scope_demand_bytes": dict(scope_demands),
            "rungs": rows,
            "shape_cell": cell,
        }
        return terms, rows, chosen, walk

    def fits(self, declared: DeclaredBinding, cell: str = "") -> str:
        """The first proven rung that fits this generation on the ledger AS IT STANDS, or
        "" when none does. No I/O and no side effect: the lane reads it before pricing an
        attempt and again after re-measuring the device, to decide whether a co-tenant must
        give the card up (cr-022). An unreconciled ledger fits nothing — `choose` refuses it
        typed and no arbitration should move bytes on its account."""
        if self.ledger.unreconciled:
            return ""
        _terms, _rows, chosen, _walk = self._walk(declared, cell)
        return chosen

    def choose(
        self,
        declared: DeclaredBinding,
        prepared_model: PreparedModel,
        prepared: PreparedRequest,
    ) -> AttemptPlan:
        self.choices += 1
        if self.ledger.unreconciled:
            # A CHOOSER MAY NOT PRICE AGAINST A LEDGER THAT IS KNOWN TO BE WRONG. This is the
            # loud half of cl-003's defect: the drift existed, this repository's own
            # reconciliation had already NAMED it, and the next request was priced against it
            # anyway — so a capacity fact arrived as an OOM in the middle of a VAE decode and
            # was reported as the package's exception. An unreconciled ledger reconciles
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
        terms, _rows, chosen, walk = self._walk(declared, prepared.cell())
        self.walks.append(walk)
        del self.walks[:-16]
        activations = terms["activations"]
        allocatable = self.ledger.allocatable
        resident = walk["resident"]
        absent = walk["absent"]
        absent_bytes = sum(absent.values())
        weights = walk["declared_bytes"] or declared.model_weight_bytes
        scope_headrooms = walk["scope_headroom_bytes"]
        if not chosen:
            proven = tuple(placement for placement, _why, status in RUNGS if status == "proven")
            if proven and all(self.ledger.placement_refusal(placement) for placement in proven):
                reasons = "; ".join(
                    f"{placement}: {self.ledger.placement_refusal(placement)}"
                    for placement in proven
                )
                raise PlanRefusal(
                    "placement_capacity_falsified",
                    f"every proven placement rung was falsified on this executor generation: "
                    f"{reasons}",
                    {
                        "resource": "vram",
                        "scope": "placement",
                        "needed_bytes": activations.value,
                        "available_bytes": max(allocatable, 0),
                        "evidence_class": "measured",
                    },
                )
            chosen, confession = self._deepen(declared, activations, allocatable, walk)
        else:
            confession = ""
        rung = self._delivery(prepared_model, chosen, walk)
        return AttemptPlan(
            delivery=rung.delivery,
            materialization=rung.materialization,
            compute_dtype="float16",
            placement=chosen,
            # THREE DIFFERENT FACTS, three names (codex audit, adopted 2026-08-25). The
            # generation's WEIGHTS resident or not; what is resident RIGHT NOW; and what
            # this attempt is admitted against. `reserved_vram_bytes` is the wire's word
            # (th-024's) and keeps its meaning — the weights, because a reservation that
            # shrank to whatever happened to be resident made the summary read
            # `vram=167307726B` for a 6.46 GiB pipeline, which is how cl-003 caught the
            # defect. The other two are named beside it instead of being the same number
            # read three ways.
            reserved_vram_bytes=weights,
            model_weight_bytes=weights,
            resident_bytes=sum(resident.values()),
            headroom_bytes=activations.value,
            scope_headroom_bytes=tuple(sorted(scope_headrooms.items())),
            entrypoint_binding_digest=declared.entrypoint_binding_digest,
            constructed_model_digest=prepared_model.constructed_model_digest,
            request_features_digest=prepared.features_digest,
            preflight_facts_digest=prepared.facts_digest,
            overlay_digest=prepared.overlay_digest,
            shape_cell=prepared.cell(),
            delivery_variant=prepared_model.delivery_variant,
            decision_explanation=(
                f"{rung.delivery}/{rung.materialization or 'verbatim'} on variant "
                f"{prepared_model.delivery_variant or declared.variant or 'default'!r} "
                f"({rung.cost_class} decode, {ROUTE_KERNEL[rung.route]} kernel) x "
                f"{chosen}: {len(resident) + len(absent) or 1} component(s) totalling "
                f"{weights} B of which {absent_bytes} B is not resident, with "
                f"{allocatable} B allocatable and {activations.value} B of activations "
                f"({activations.provenance}); request {prepared.entrypoint} "
                f"on shape cell {prepared.cell()}{confession}"
            ),
        )

    def _delivery(self, prepared_model: PreparedModel, placement: str, walk: dict[str, Any]) -> Any:
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
        name = prepared_model.delivery_rung or "verbatim"
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
            "variant": prepared_model.delivery_variant,
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
        declared: DeclaredBinding,
        activations: Coefficient,
        allocatable: int,
        walk: dict[str, Any],
    ) -> tuple[str, str]:
        """Nothing on the ladder fits. What happens next depends on TENANCY (§3.2).

        LOCAL single-tenant: capacity NEVER refuses above the physical floor. The ladder
        deepens to the deepest authorized rung with a quantified confession, because the
        person in front of it wants their image, not a lecture about their card.

        SHARED cloud: a shortfall is a RESCHEDULE-shaped fact for the RecordOwner, never a
        structural-incompatibility claim — a different worker may fit it exactly.

        A GROUP (cr-068, cr-018): never deepened. Every rank runs the same plan, so a
        per-rank downgrade is unrepresentable; the refusal is `CONSTRAINT_INFEASIBLE`,
        priced against the MIN over the K devices' headroom, and the RecordOwner reads it
        as a placement that does not fit this group.
        """
        if self.degree > 1:
            walk["refused_under_group"] = self.degree
            raise PlanRefusal(
                "group_constraint_infeasible",
                f"no proven placement rung fits this {self.degree}-device group: needed "
                f"{activations.value} B of headroom ({activations.provenance}) against "
                f"{max(allocatable, 0)} B allocatable on the tightest device; the ladder is "
                "not deepened under a group because every rank runs one plan",
                {
                    "resource": "vram",
                    "scope": "group",
                    "needed_bytes": activations.value,
                    "available_bytes": max(allocatable, 0),
                    "evidence_class": "measured" if activations.enforceable else "bound",
                },
            )
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
        deepest = next(p for p, _, status in reversed(RUNGS) if status == "proven")
        walk["deepened_to"] = deepest
        return deepest, (
            f" | DEGRADED: needed {activations.value} B of headroom, had "
            f"{max(allocatable, 0)} B, short by {max(activations.value - allocatable, 0)} B; "
            f"the ladder deepened to {deepest} rather than refusing a single-tenant request "
            "above its physical floor"
        )
