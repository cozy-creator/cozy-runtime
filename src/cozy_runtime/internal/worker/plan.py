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

import msgspec

from cozy_runtime.internal import canonical, installed_interfaces, tolerant
from cozy_runtime.internal.canonical import Json
from cozy_runtime.internal.delivery import RUNGS_BY_NAME, DeliveryRefusal, cost_class
from cozy_runtime.internal.execution_evidence import Boot
from cozy_runtime.internal.resolution import ROUTE_KERNEL
from cozy_runtime.internal.worker.ledger import Ledger
from cozy_runtime.protocol import worker_pb2 as pb


def shape_cell(features: Mapping[str, int]) -> str:
    return ",".join(f"{axis}={value}" for axis, value in sorted(features.items())) or "-"


class PlanRefusal(Exception):
    """A typed pre-acceptance refusal, carrying the structured shortfall when it has one."""

    def __init__(
        self, code: str, detail: str, shortfall: pb.ResourceShortfall | None = None
    ) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail
        self.shortfall = shortfall


class ModelDelivery(msgspec.Struct, frozen=True, kw_only=True):
    """One model's delivery as its prepare resolved it; the executor's row carries more."""

    rung: str = ""
    variant: str = ""
    plan_digest: str = ""
    routes: dict[str, int] = {}
    kernels: dict[str, int] = {}


class CrossedDelivery(msgspec.Struct, frozen=True, kw_only=True, omit_defaults=True):
    """One model's delivery under this attempt's placement, as `plan_digest` covers it."""

    plan_digest: str = ""
    routes: dict[str, int] = {}
    kernels: dict[str, int] = {}
    delivery: str
    materialization: str
    cost_class: str
    route_costs: dict[str, str]


class Delivery(msgspec.Struct, frozen=True, kw_only=True, omit_defaults=True):
    """The prepare-time delivery crossed with a placement; several models carry `models`."""

    variant: str
    rung: str = ""
    delivery: str
    materialization: str
    cost_class: str
    kernel: str
    models: dict[str, CrossedDelivery] = {}


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
    model_deliveries: tuple[tuple[str, CrossedDelivery], ...] = ()
    """Each model's resolved delivery and its cost under this attempt's placement."""
    scope_headroom_bytes: tuple[tuple[str, int], ...] = ()
    """Measured activation scratch by component-use method. The scalar headroom is its max;
    staged admission uses each method's own value so unrelated weight/scratch maxima never add."""
    measured_scopes: tuple[str, ...] = ()
    """The methods whose scratch was measured at exactly this shape cell by a completed serve.
    Staged admission keeps other components beside these and frees everything for the rest."""

    def document(self) -> dict[str, Json]:
        document: dict[str, Json] = {
            "delivery": self.delivery,
            "materialization": self.materialization,
            "delivery_variant": self.delivery_variant,
            "compute_dtype": self.compute_dtype,
            "placement": self.placement,
            "reserved_vram_bytes": self.reserved_vram_bytes,
            "headroom_bytes": self.headroom_bytes,
            "scope_headroom_bytes": dict(self.scope_headroom_bytes),
            "measured_scopes": list(self.measured_scopes),
            "entrypoint_binding_digest": self.entrypoint_binding_digest,
            "constructed_model_digest": self.constructed_model_digest,
            "request_features_digest": self.request_features_digest,
            "preflight_facts_digest": self.preflight_facts_digest,
            "overlay_digest": self.overlay_digest,
            "shape_cell": self.shape_cell,
            "model_weight_bytes": self.model_weight_bytes,
            "resident_bytes": self.resident_bytes,
        }
        if self.model_deliveries:
            document["model_deliveries"] = {
                name: msgspec.to_builtins(row) for name, row in self.model_deliveries
            }
        return document

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


class JobBinding(msgspec.Struct, frozen=True, kw_only=True):
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
    call_interfaces: tuple[Mapping[str, object], ...] = ()
    """Observed from the installed environment on every read, never taken from the record."""

    @classmethod
    def read(cls, record: object) -> JobBinding:
        """Unknown members are ignored and absent optional ones take their defaults."""
        try:
            plan = msgspec.convert(record, cls)
        except msgspec.ValidationError as exc:
            raise PlanRefusal(
                "job_plan_incomplete", f"job plan: {exc}; re-prepare the package"
            ) from exc
        observed = installed_interfaces.for_job(plan.python, plan.installation_id)
        return msgspec.structs.replace(plan, call_interfaces=tuple(observed))


class _PreparedFacts(msgspec.Struct, frozen=True):
    filled_bytes: int
    attention: Boot | None = None


class _PreparedDelivery(msgspec.Struct, frozen=True):
    rung: str = ""
    variant: str = ""
    models: dict[str, ModelDelivery] = {}


class _PrepareReply(msgspec.Struct, frozen=True):
    """The prepare reply as a PreparedModel reads it; the executor sends more."""

    constructed_model_digest: str
    facts: _PreparedFacts
    delivery: _PreparedDelivery = msgspec.field(default_factory=_PreparedDelivery)


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
    model_deliveries: tuple[tuple[str, ModelDelivery], ...] = ()
    """Named child deliveries when this prepare constructed several models."""
    attention: Boot | None = None
    """WHAT THE RUNTIME PINNED on this generation's attention sites (cr-124): kernel per
    component, and what this device and image offered. A construction fact, read once at
    boot — the choice cannot vary between attempts of one generation."""

    @classmethod
    def from_reply(cls, reply: Mapping[str, object]) -> PreparedModel:
        """Read exactly the executor's prepared generation, including each model's routes."""
        # What was built and how it is delivered are exact; which kernels won is narration.
        read, _ = tolerant.read(reply, _PrepareReply, ("facts.attention",))
        models = read.delivery.models
        return cls(
            constructed_model_digest=read.constructed_model_digest,
            delivery_rung=read.delivery.rung,
            delivery_variant=read.delivery.variant,
            model_deliveries=tuple((name, models[name]) for name in sorted(models)),
            attention=read.facts.attention,
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
    logical_weight_bytes: int = 0
    custody: str = "canonical"
    sequence_parallel_degrees: tuple[int, ...] = ()
    """The interface slot's `sequence_parallel.degrees` (cr-068): what the author said the
    class shards at. A placement pinned to K devices needs K here."""
    model: str = ""
    """`org/name@release/lane` — the checkpoint's catalog name, for the one line a person
    reads when the code does not fit it (model-code-fit §3). Resolution, never identity."""
    adapters: tuple[Mapping[str, str], ...] = ()
    prepared_adapters: bool = False

    def snapshot_for(self, component: str) -> str:
        return self.snapshots.get(component, self.reference_snapshot)

    def key_document(self) -> dict[str, Json]:
        return {
            "store": self.store,
            "snapshot": self.reference_snapshot,
            "variant": self.variant,
            "model_class": self.model_class,
            "components": list(self.components),
            "snapshots": dict(self.snapshots),
            "objective": self.objective,
            "steps_basis": self.steps_basis,
            "logical_weight_bytes": self.logical_weight_bytes,
            "custody": self.custody,
            "adapters": [dict(row) for row in self.adapters],
            "prepared_adapters": self.prepared_adapters,
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
    logical_weight_bytes: int = 0
    """The LOGICAL weight bytes of this model's constructed components, from their checkpoint
    headers: what their destinations hold, never the smaller stored closure of an encoded
    checkpoint. Not what is resident, and not what an attempt reserves."""
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
                logical_weight_bytes=self.logical_weight_bytes,
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


class PlanChooser:
    """ONE AttemptPlan from the request shape, the binding and the tenant's residency.

    Nothing is priced here: the memory manager already made room for this call on every
    device it needs (`memory.py`), and says whether the call's measured need fits. The rung
    follows: `all_resident` when it fits and the construction holds everything it built,
    else `component_staged`: one declared scope resident at a time, blocks paged, inside
    whatever every idle tenant's eviction left. The executor's admission of each scope is
    the check that bytes exist. No I/O.
    """

    def __init__(self, ledger: Ledger) -> None:
        self.ledger = ledger

    def choose(
        self,
        declared: DeclaredBinding,
        prepared_model: PreparedModel,
        prepared: PreparedRequest,
        *,
        fits: bool = True,
    ) -> AttemptPlan:
        ledger = self.ledger
        if ledger.unreconciled:
            raise PlanRefusal(
                "ledger_unreconciled",
                f"this generation's residency numbers are known to be wrong "
                f"({ledger.unreconciled}); it is rebuilt before it serves again",
                pb.ResourceShortfall(resource="vram", scope="ledger", evidence_class="measured"),
            )
        cell = prepared.cell()
        resident = dict(ledger.resident)
        nominal = {**ledger.evicted, **resident}
        nominal.update({name: row["total_bytes"] for name, row in ledger.paging.items()})
        absent = sum(max(size - resident.get(name, 0), 0) for name, size in nominal.items())
        weights = sum(nominal.values()) or declared.logical_weight_bytes
        parked = [name for name in ledger.parked if name not in resident]
        headroom = ledger.activations_for(cell)
        scopes = ledger.activation_scopes_for(cell)
        measured = ledger.measured_scopes(cell) & set(ledger.declared_scopes)
        placement = "all_resident" if fits and not absent and not parked else "component_staged"
        rung = self._delivery(prepared_model, placement)
        return AttemptPlan(
            delivery=rung.delivery,
            materialization=rung.materialization,
            compute_dtype="float16",
            placement=placement,
            reserved_vram_bytes=weights,
            model_weight_bytes=weights,
            resident_bytes=sum(resident.values()),
            headroom_bytes=headroom,
            scope_headroom_bytes=tuple(sorted(scopes.items())),
            measured_scopes=tuple(sorted(measured)),
            entrypoint_binding_digest=declared.entrypoint_binding_digest,
            constructed_model_digest=prepared_model.constructed_model_digest,
            request_features_digest=prepared.features_digest,
            preflight_facts_digest=prepared.facts_digest,
            overlay_digest=prepared.overlay_digest,
            shape_cell=cell,
            delivery_variant=prepared_model.delivery_variant,
            model_deliveries=tuple((name, rung.models[name]) for name in sorted(rung.models)),
            decision_explanation=(
                f"{rung.delivery}/{rung.materialization or 'verbatim'} on variant "
                f"{prepared_model.delivery_variant or declared.variant or 'default'!r} "
                f"({rung.cost_class} decode, {rung.kernel} kernel) x {placement}: "
                f"{len(nominal) or 1} component(s) totalling {weights} B, {absent} B not "
                f"resident{f', {parked} parked at construction' if parked else ''}; the "
                f"call's measured need {'fits' if fits else 'does not fit'} on its devices "
                f"with {headroom} B of measured activations; request {prepared.entrypoint} "
                f"on shape cell {cell}"
            ),
        )

    def _delivery(self, prepared_model: PreparedModel, placement: str) -> Delivery:
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
            models: dict[str, CrossedDelivery] = {}
            kernels: set[str] = set()
            for parameter, delivery in prepared_model.model_deliveries:
                child = self._delivery(
                    PreparedModel(delivery_rung=delivery.rung, delivery_variant=delivery.variant),
                    placement,
                )
                try:
                    route_costs = {
                        route: cost_class(route, placement)
                        for route, count in delivery.routes.items()
                        if count
                    }
                except DeliveryRefusal as exc:
                    raise PlanRefusal(exc.code, f"model {parameter!r}: {exc.detail}") from exc
                # Keep the actual route/kernel census beside its placement-dependent cost.
                models[parameter] = CrossedDelivery(
                    plan_digest=delivery.plan_digest,
                    routes=delivery.routes,
                    kernels=delivery.kernels,
                    delivery=child.delivery,
                    materialization=child.materialization,
                    cost_class=child.cost_class,
                    route_costs=route_costs,
                )
                kernels.update(delivery.kernels or (child.kernel,))
            materializations = {row.materialization for row in models.values()}
            return Delivery(
                variant=prepared_model.delivery_variant,
                delivery="float"
                if any(row.delivery == "float" for row in models.values())
                else "native",
                materialization=(
                    "staged_decode"
                    if "staged_decode" in materializations
                    else "aot_decode"
                    if "aot_decode" in materializations
                    else ""
                ),
                cost_class="+".join(sorted({row.cost_class for row in models.values()})),
                kernel="+".join(sorted(kernels)),
                models=models,
            )
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
                pb.ResourceShortfall(
                    resource="delivery", scope="plan", evidence_class="structural"
                ),
            )
        return Delivery(
            variant=prepared_model.delivery_variant,
            rung=crossed.name,
            delivery=crossed.delivery,
            materialization=crossed.materialization,
            cost_class=crossed.cost_class,
            kernel=ROUTE_KERNEL[crossed.route],
        )
