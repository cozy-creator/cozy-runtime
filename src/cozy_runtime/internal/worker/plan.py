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
from cozy_runtime.internal.delivery import RUNGS_BY_NAME, DeliveryRefusal, cost_class
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
    model_deliveries: tuple[tuple[str, dict[str, Any]], ...] = ()
    """Each model's resolved delivery and its cost under this attempt's placement."""
    scope_headroom_bytes: tuple[tuple[str, int], ...] = ()
    """Measured activation scratch by component-use method. The scalar headroom is its max;
    staged admission uses each method's own value so unrelated weight/scratch maxima never add."""
    activation_envelope_bytes: int = 0
    """What this shape has been OBSERVED to allocate, whether or not that observation may
    be enforced here (cr-100). `headroom_bytes` above is what the attempt is admitted
    against and can be smaller — deliberately, when the observation was made on a roomier
    card. This one is the DESCRIPTION, and it is what the demand falsifier banks: an
    envelope that overstates demand shows no miss, one that understates confesses. It is
    outside `document()` because it decides nothing, and a plan's identity is its
    decisions."""

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
            **(
                {
                    "model_deliveries": {
                        name: {
                            key: row[key]
                            for key in (
                                "plan_digest",
                                "routes",
                                "kernels",
                                "delivery",
                                "materialization",
                                "cost_class",
                                "route_costs",
                            )
                            if key in row
                        }
                        for name, row in self.model_deliveries
                    }
                }
                if self.model_deliveries
                else {}
            ),
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
    installation_id: str
    application: str
    package_interface: str
    python: str
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
    call_interfaces: tuple[dict[str, Any], ...] = ()

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
        from cozy_runtime.internal import installed_interfaces

        observed = installed_interfaces.for_job(
            str(record.get("python", "")), str(record.get("installation_id", ""))
        )
        captured = dict(record)
        captured.pop("call_interfaces", None)
        return cls(**captured, call_interfaces=tuple(observed))


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
    model_deliveries: tuple[tuple[str, dict[str, Any]], ...] = ()
    """Named child delivery documents when this prepare constructed several models."""
    weight_bytes: int = 0
    """What the fill COMMITTED, measured. Distinct from the deployment's declared
    `model_weight_bytes` and from the ledger's live resident map — three different facts
    that were all called some form of "vram bytes"."""
    attention: dict[str, Any] = dataclasses.field(default_factory=dict)
    """WHAT THE RUNTIME PINNED on this generation's attention sites (cr-124): kernel per
    component, and what this device and image offered. A construction fact, read once at
    boot — the choice cannot vary between attempts of one generation."""

    @classmethod
    def from_reply(cls, reply: Mapping[str, Any]) -> PreparedModel:
        """Read exactly the executor's prepared generation, including each model's routes."""
        delivery, facts = reply.get("delivery") or {}, reply["facts"]
        return cls(
            constructed_model_digest=str(reply["constructed_model_digest"]),
            delivery_rung=str(delivery.get("rung", "")),
            delivery_variant=str(delivery.get("variant", "")),
            model_deliveries=tuple(
                (name, dict(row)) for name, row in sorted((delivery.get("models") or {}).items())
            ),
            weight_bytes=int(facts["filled_bytes"]),
            attention=dict(facts.get("attention") or {}),
        )


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
    adapters: tuple[Mapping[str, Any], ...] = ()

    def snapshot_for(self, component: str) -> str:
        return self.snapshots.get(component, self.reference_snapshot)

    def key_document(self) -> dict[str, Any]:
        return {
            "store": self.store,
            "snapshot": self.reference_snapshot,
            "variant": self.variant,
            "model_class": self.model_class,
            "components": list(self.components),
            "snapshots": dict(self.snapshots),
            "objective": self.objective,
            "steps_basis": self.steps_basis,
            "model_weight_bytes": self.model_weight_bytes,
            "custody": self.custody,
            "adapters": [dict(row) for row in self.adapters],
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
    internal: bool = False
    installation_id: str = ""
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
    staged_weight_bytes: int | None = None
    """Header-derived weight requirement of the largest declared component-use scope,
    summed across independent model slots. A preflight bound, excluding derived buffers,
    fill scratch and activations; actual executor preparation still measures those."""
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

    def group_weight_bytes(self) -> int:
        """Declared per-rank weight requirement before measured executor preparation."""
        return (
            self.model_weight_bytes
            if self.staged_weight_bytes is None
            else self.staged_weight_bytes
        )

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

        The binding PATH and PARAMETER NAME are outside it (h3a-018): they say where an
        entrypoint receives the constructed model, not what is constructed. Two entrypoints
        of one release over the same class and bytes are one generation, and the executor
        binds it under every parameter name the group spells.
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
#: serving it. `bound` is the author's labelled hatch — and, since cr-097, a real
#: measurement read outside the conditions it was measured in, which is the same kind of
#: number: an upper bound. `uncalibrated` is the third state, and it is NOT the same as
#: `bound`: a bound knows the shape has run and what it took with more room, an
#: uncalibrated coefficient knows nothing at all. `_walk` treats them differently for
#: exactly that reason (cr-100).
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
            # A MEASUREMENT IS ONLY ENFORCEABLE UNDER THE CONDITIONS IT WAS MEASURED IN
            # (cr-097). Activation demand is not a constant: kernels expand into the
            # headroom they are given, so the same model at the same shape measured
            # 3.91 GiB alone on a 15.6 GiB card and 1.92 GiB with a co-tenant beside it.
            # A peak observed with MORE room than there is now is an upper bound the
            # kernels chose, not a requirement they have -- so it may deepen the ladder
            # (which changes no bits) and it may NOT refuse, exactly like an authored
            # bound. Enforcing it anyway would evict a co-tenant that did not need
            # evicting, or refuse an attempt that would have succeeded.
            #
            # THREE GRADES, THREE DIFFERENT STATES OF KNOWLEDGE, and the walk has to tell
            # them apart (cr-100). `measured` is a requirement and may deny a rung.
            # `bound` is a real observation of THIS shape read outside its conditions —
            # it knows the serve completed and knows what the kernels took when they had
            # more room, and it may not cost a tenant residency it is already holding.
            # `uncalibrated` below knows nothing at all, which is why it still deepens.
            applies = self.ledger.carried_conditions_apply(cell)
            basis = self.ledger.activation_basis_free(cell)
            observed = source or "measured on this generation"
            terms.append(
                Coefficient("activations", activations, "measured", source)
                if applies
                else Coefficient(
                    "activations",
                    activations,
                    "bound",
                    f"an upper bound, not a requirement: {observed}, observed with "
                    f"{basis} B free against {self.ledger.device_free} B now",
                )
            )
        else:
            # NOTHING HAS EVER BEEN MEASURED for this shape — a different state from an
            # observation read out of its conditions, and it gets its own provenance. An
            # uncalibrated number may not refuse; it MAY deepen, because with no serve
            # behind it the deeper rung is the only conservative answer available, and
            # cr-063 measured what happens without one (an all-resident first attempt
            # that OOMs on an 8 GB card where the staged rung serves).
            terms.append(
                Coefficient("activations", UNCALIBRATED_ACTIVATION_BYTES, "uncalibrated", "")
            )
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
        nominal = {**absent, **resident}
        nominal.update({name: layout["total_bytes"] for name, layout in self.ledger.paging.items()})
        absent_bytes = sum(max(size - resident.get(name, 0), 0) for name, size in nominal.items())
        declared_bytes = sum(nominal.values())
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
        sizes = {
            name: self.ledger.paging.get(name, {}).get("working_bytes", size)
            for name, size in nominal.items()
        }
        scoped = [
            sum(sizes.get(name, 0) for name in dict.fromkeys(names))
            for names in self.ledger.declared_scopes.values()
        ]
        largest = max([max(sizes.values(), default=declared.model_weight_bytes), *scoped])
        activations = terms["activations"]
        free = self.ledger.device_free
        # WHAT MAY DENY A RUNG, AND WHAT MAY ONLY DESCRIBE ONE (cr-100).
        #
        # `Coefficient.enforceable` already says a number may refuse OR SELECT A WORSE RUNG
        # only on a measured basis. The walk was the consumer that honoured the first half
        # and not the second: an envelope the ledger had just labelled `bound` — an upper
        # bound the kernels chose on a roomier card — was still the number that pushed a
        # FULLY RESIDENT generation onto the staged rung. Measured on an rtx-a4000: sdxl
        # 12.6 s alone, ~25 s on every later run once a co-tenant had touched the pod,
        # with `all_resident` scored short by 1.18 GiB of activations it never had to have
        # and the staged rung "fitting" only because it is allowed to free the 6.46 GiB of
        # weights a restore had just put back.
        #
        # So an inapplicable envelope is charged NOTHING where it would only be spending
        # bytes the tenant already holds, and everything where a rung would have to ACQUIRE
        # bytes: declining to move gigabytes on a guess is conservative, while spending
        # residency that is already paid for on the same guess is not.
        planning_headroom = 0 if activations.provenance == "bound" else activations.value
        measured_scopes = self.ledger.activation_scopes_for(cell)
        planning_scope_headrooms = {
            method: measured_scopes.get(method, planning_headroom)
            for method in self.ledger.declared_scopes
        }
        scope_demands = {
            method: sum(sizes.get(name, 0) for name in dict.fromkeys(names))
            + planning_scope_headrooms[method]
            for method, names in self.ledger.declared_scopes.items()
        }
        # THE SHALLOW RUNG'S HEADROOM, and the two direct observations that may overrule
        # the envelope in either direction. Both are in driver-free bytes, both compare
        # with the card as it stands, and neither is an inference from a peak:
        #   * a SHORTFALL — this cell already FAILED on this device with at least this much
        #     room, so a picture no better than that one cannot get a different answer.
        #   * a SUFFICIENCY — this cell already COMPLETED, with its resident set standing
        #     still, on a card with no more room than this one. That is the outcome the
        #     rung is trying to predict, already observed, so it outranks an envelope.
        sufficiency = self.ledger.resident_sufficiency(cell)
        # Restoring weights does not prove that the request can run beside them. A
        # successful staged request measured scratch after freeing other components;
        # even a tiny driver-free change can classify that measurement as a bound.
        # Only prior nonmoving execution can justify waiving that bound for residency.
        resident_headroom = (
            planning_headroom
            if absent_bytes == 0 and (activations.provenance != "bound" or sufficiency)
            else activations.value
        )
        headroom_basis = activations.provenance if resident_headroom else "already_resident"
        if absent_bytes == 0 and sufficiency and free >= 0 and sufficiency <= free:
            resident_headroom = 0
            headroom_basis = "completed_resident"
        shortfall = self.ledger.insufficient_for(cell, free) if free >= 0 else 0
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
            # The rungs differ in the WEIGHTS they must bring back and in what they are
            # allowed to free: `all_resident` needs every absent component and can free
            # nothing, so its capacity is bare allocatable; `component_staged` holds one
            # declared set at a time and may free every resident component first (cl-003).
            # They differ in HEADROOM too, and that is cr-100's half: the staged rung is
            # priced on what it must raise, while the shallow rung — when it has nothing
            # to bring back — is priced on the activations alone, and only on a figure
            # that may be enforced under the conditions that hold right now.
            denied = 0
            if placement == "all_resident":
                headroom = resident_headroom + absent_bytes
                capacity = allocatable
                denied = shortfall
            else:
                headroom = max(scope_demands.values(), default=planning_headroom + largest)
                capacity = allocatable + sum(resident.values())
            fits = (allocatable < 0 or capacity >= headroom) and not denied
            row = {
                "rung": placement,
                "verdict": "fits" if fits else "short",
                "needs_bytes": headroom,
                "capacity_bytes": capacity,
                "allocatable_bytes": allocatable,
                "absent_bytes": absent_bytes,
                "why": why,
            }
            if placement == "all_resident":
                row["headroom_basis"] = headroom_basis
                if denied:
                    row["verdict"] = "refused_on_measured_shortfall"
                    row["denied_at_free_bytes"] = denied
            rows.append(row)
            if fits and not chosen:
                chosen = placement
                row["verdict"] = "chosen"

        # Estimates can select a conservative strategy without becoming a required
        # allocation. In particular, the uncalibrated 1 GiB reserve must not refuse
        # a small staged model. Actual component/block checks still run in the executor.
        admitted_headroom = activations.value if activations.enforceable else 0
        scope_headrooms = {
            method: measured_scopes.get(method, admitted_headroom)
            for method in self.ledger.declared_scopes
        }
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
            # WHAT THIS ATTEMPT IS ADMITTED AGAINST, beside the envelope it was described
            # with. They are the same number whenever the envelope may be enforced, and
            # they part company exactly when it may not — which is the fact a reader needs
            # in front of them rather than one they have to reconstruct.
            "enforceable_headroom_bytes": admitted_headroom,
            "resident_headroom_bytes": resident_headroom,
            "headroom_basis": headroom_basis,
            "device_free_bytes": free,
            "resident_sufficiency_free_bytes": sufficiency,
            "measured_shortfall_free_bytes": shortfall,
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

    def should_restore(self, declared: DeclaredBinding, prepared: PreparedRequest) -> bool:
        """Do not refill weights that this request will immediately have to evict again.

        Unknown shapes retain the existing measured-capacity restore path. Prior
        nonmoving success retains the co-tenant recovery path. A staged-only history
        must fit its observed activation allowance before a full restore is useful.
        """
        cell = prepared.cell()
        if self.ledger.resident_sufficiency(cell):
            return True
        if not self.ledger.carried_activations_by_cell.get(cell, 0):
            return True
        return self.fits(declared, cell) == "all_resident"

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
            chosen, confession = self._deepen(activations, allocatable, walk)
        else:
            confession = ""
        rung = self._delivery(prepared_model, chosen, walk)
        return AttemptPlan(
            delivery=rung["delivery"],
            materialization=rung["materialization"],
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
            # WHAT THE EXECUTOR ADMITS AGAINST IS WHAT THE PLAN DECIDED WITH (cr-100).
            # `headroom_bytes` and the per-scope map ride `open_attempt` into `admit`,
            # which refuses `device_shortfall` on `size + headroom` INSIDE the executor,
            # past every upstream check. Sending the envelope after the walk has ruled it
            # inapplicable would re-litigate the conditions rule one layer down, in the
            # one place no gate above can see — the exact reach cr-097 named for the
            # carried scopes and closed there.
            headroom_bytes=int(walk["enforceable_headroom_bytes"]),
            activation_envelope_bytes=activations.value,
            scope_headroom_bytes=tuple(sorted(scope_headrooms.items())),
            entrypoint_binding_digest=declared.entrypoint_binding_digest,
            constructed_model_digest=prepared_model.constructed_model_digest,
            request_features_digest=prepared.features_digest,
            preflight_facts_digest=prepared.facts_digest,
            overlay_digest=prepared.overlay_digest,
            shape_cell=prepared.cell(),
            delivery_variant=prepared_model.delivery_variant,
            model_deliveries=tuple(sorted((rung.get("models") or {}).items())),
            decision_explanation=(
                f"{rung['delivery']}/{rung['materialization'] or 'verbatim'} on variant "
                f"{prepared_model.delivery_variant or declared.variant or 'default'!r} "
                f"({rung['cost_class']} decode, {rung['kernel']} kernel) x "
                f"{chosen}: {len(resident) + len(absent) or 1} component(s) totalling "
                f"{weights} B of which {absent_bytes} B is not resident, with "
                f"{allocatable} B allocatable and {activations.value} B of activations "
                f"({activations.provenance}), admitted against "
                f"{walk['enforceable_headroom_bytes']} B "
                f"({walk['headroom_basis']}); request {prepared.entrypoint} "
                f"on shape cell {prepared.cell()}{confession}"
            ),
        )

    def _delivery(
        self, prepared_model: PreparedModel, placement: str, walk: dict[str, Any]
    ) -> dict[str, Any]:
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
        if prepared_model.model_deliveries:
            models = {}
            for parameter, delivery in prepared_model.model_deliveries:
                child = PreparedModel(
                    delivery_rung=str(delivery.get("rung", "")),
                    delivery_variant=str(delivery.get("variant", "")),
                )
                child_crossed = self._delivery(child, placement, {})
                try:
                    route_costs = {
                        route: cost_class(route, placement)
                        for route, count in (delivery.get("routes") or {}).items()
                        if count
                    }
                except DeliveryRefusal as exc:
                    raise PlanRefusal(exc.code, f"model {parameter!r}: {exc.detail}") from exc
                # Keep the actual route/kernel census beside its placement-dependent cost.
                models[parameter] = {
                    **delivery,
                    **child_crossed,
                    "route_costs": route_costs,
                    "kernel": "+".join(
                        sorted(delivery.get("kernels") or {child_crossed["kernel"]: 1})
                    ),
                }
            materializations = {row["materialization"] for row in models.values()}
            summary = {
                "variant": prepared_model.delivery_variant,
                "delivery": "float"
                if any(row["delivery"] == "float" for row in models.values())
                else "native",
                "materialization": (
                    "staged_decode"
                    if "staged_decode" in materializations
                    else "aot_decode"
                    if "aot_decode" in materializations
                    else ""
                ),
                "cost_class": "+".join(sorted({row["cost_class"] for row in models.values()})),
                "kernel": "+".join(
                    sorted(
                        {
                            kernel
                            for row in models.values()
                            for kernel in (row.get("kernels") or {row["kernel"]: 1})
                        }
                    )
                ),
                "chosen_at": "prepare",
                "models": models,
            }
            walk["delivery"] = summary
            return summary
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
        result = {
            "variant": prepared_model.delivery_variant,
            "rung": crossed.name,
            "delivery": crossed.delivery,
            "materialization": crossed.materialization,
            "cost_class": crossed.cost_class,
            "kernel": ROUTE_KERNEL[crossed.route],
            "chosen_at": "prepare",
        }
        walk["delivery"] = result
        return result

    def _deepen(
        self,
        activations: Coefficient,
        allocatable: int,
        walk: dict[str, Any],
    ) -> tuple[str, str]:
        """Use the deepest supported residency plan before judging an active working set.

        A group already has one ledger priced against its tightest rank. Its leader
        broadcasts this same plan and scope headroom to every follower; rank-local
        fallback is neither required nor allowed. The executor still enforces the
        actual component/block working set before allocation. A plan that was already
        falsified on this generation is rejected by `choose`, not retried here.
        """
        if not activations.enforceable:
            # A NON-MEASURED coefficient MAY NOT REFUSE. It may deepen the ladder, because
            # a placement rung changes weight placement only — never the kernel, never the
            # request, never the bits (proven: the staged and resident plans produce the
            # same image digest). The v1 incident this protects against is a refusal on an
            # uncalibrated headroom charge, on the very card that was serving the lane.
            #
            # READ THIS BEFORE REACHING FOR THIS BRANCH TO FIX AN OVER-ESTIMATE (cr-100).
            # Arriving here means NO rung fit, so "deepest" is the only thing left and
            # marking a measurement inapplicable HERE can only ever select the SLOWEST
            # path. That is the trap cr-097's first cut fell into. An inapplicable
            # measurement is answered in `_walk`, where it stops denying a rung the tenant
            # can already afford; by the time the walk has run out of rungs, the shortfall
            # is in bytes that have to MOVE, and no relabelling of the envelope changes it.
            deepest = next(p for p, _, status in reversed(RUNGS) if status == "proven")
            walk["deepened_to"] = deepest
            walk["deepened_on"] = activations.provenance
            grade = (
                "no serve has ever been banked for this shape"
                if activations.provenance == "uncalibrated"
                else "the banked serve for this shape was measured with more room than "
                "there is now, so it is an upper bound the kernels chose"
            )
            return deepest, (
                f" | {activations.provenance.upper()}: {grade}, so the "
                f"{activations.value} B activation envelope may not refuse. It may deepen "
                f"placement (which changes no bits), so the ladder goes to {deepest} and "
                f"the next serve banks what this card actually needed"
            )
        deepest = next(p for p, _, status in reversed(RUNGS) if status == "proven")
        walk["deepened_to"] = deepest
        return deepest, (
            f" | DEGRADED: needed {activations.value} B of headroom, had "
            f"{max(allocatable, 0)} B, short by {max(activations.value - allocatable, 0)} B; "
            f"the ladder deepened to {deepest}; every executing rank retains its "
            "measured component and block admission checks"
        )
