"""#549.1 — the ResolvedExecutionPlan: ONE object, ONE decision, and a label that derives.

**The defect this closes.** Two choosers ran in parallel and neither could see the other.
`encoding.Selector` picked a provider PER TENSOR — correctly, from that tensor's own spec
digest — while `delivery.DeliveryChooser` picked ONE artifact-wide RUNG and reported it as
what the generation was. On a mixed artifact those disagree by construction: since the
`quantize-mixed` job kind (#543g) one component legitimately holds a native leaf beside a
decoded float weight beside a verbatim copy, and the plan reported `native_encoded` for all
of it. The rung was a LABEL WITH NO EXECUTION AUTHORITY — nothing downstream read it to fill
anything — which made it exactly the kind of field that is wrong without being noticed.

Worse, the rung was hashed into the generation's identity, so two materially different fills
could share a constructed digest and one materially identical fill could differ.

**What replaces it.** One immutable `ResolvedExecutionPlan`, resolved once, before a
destination is reserved:

* the EXACT snapshot map (component -> snapshot digest), not a variant NAME;
* one `TensorResolution` per destination: encoding digest, implementation digest and
  revision, route, validated geometry, output dtype, and the REASON that provider won;
* device identity and runtime identity (#549.9);
* placement, objective, construction facts;
* the whole-plan measurements, looked up by THIS PLAN'S DIGEST (#549.2).

`digest()` is what enters generation identity. `label()` DERIVES from the route census, so a
mixed plan says `mixed(encoded_gemm x743, decoded_float x120, verbatim x937)` and cannot say
anything else. `delivery.py`'s rung table and `encoding`'s admission are now INPUTS to this
resolution rather than parallel authorities: rungs describe what a route COSTS and when, and
admission says which implementations were measured here.

**The objective dead end dissolves here** (#549.3). `Capabilities.admit` used to refuse
`objective_unranked` whenever the objective was `latency` and two providers qualified —
which is the DEFAULT objective, so the default path refused as soon as a second
implementation existed, one layer above the store that holds latency measurements. Now the
objective ranks WHOLE PLANS, and every axis has somewhere real to stand:

* `latency` — from the plan fact's own `fill_ms + step_ms * steps`. Measured per plan.
* `footprint` — from the plan fact if banked, else DERIVED from the capability records'
  measured `resident_bits_per_element` weighted by each tensor's element count.
* `kernel_numeric_error` — the WORST per-tensor route deviation the records measured. Worst
  and not mean, because a plan is as wrong as its wrongest tensor.

An objective nothing can score does not refuse: the resolver takes the reference variant's
DECODE FLOOR and confesses UNCALIBRATED, which is what cr-008b does one axis over and what
the two-chooser split could not do without an artifact-wide rung to fall back to.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from cozy_runtime.internal import canonical
from cozy_runtime.internal.encoding import (
    DeviceFacts,
    EncodingRefusal,
    RuntimeIdentity,
    Selection,
    Selector,
    device_line,
    geometry_class,
)
from cozy_runtime.internal.planfacts import PlanFact, PlanFacts

#: WHICH KERNEL each route computes with. The two float routes compute in the destination's
#: own dtype through the ordinary library GEMM — that is what "kernel-neutral" means, and it
#: is data so the placement walk can be checked not to have moved it.
ROUTE_KERNEL: dict[str, str] = {
    "verbatim": "library_float",
    "decoded_float": "library_float",
    "encoded_gemm": "encoded_gemm",
}

#: The measured axis a deployment optimizes. A property of the DEPLOYMENT, not a request
#: knob, because the plan is resolved once for the generation.
#:
#: `fidelity` is GONE and `kernel_numeric_error` stands where it did (#549.11). The old name
#: promised model quality and delivered a Frobenius norm over one GEMM; quality is a paired
#: video+audio eval on real outputs and belongs to the eval plane.
OBJECTIVES = ("latency", "footprint", "kernel_numeric_error")

#: The ROUTE POLICIES a plan may be resolved under — the closed, ordered replacement for the
#: artifact-wide rung ladder. A policy says which route a tensor PREFERS when it has more
#: than one; it never says which route a tensor GETS, because a tensor with no encoded-GEMM
#: candidate takes the floor under either policy and the plan records that per tensor.
POLICIES = ("encoded_gemm", "decode_floor")

#: Preference order per policy. First route in the list that this tensor actually has wins.
_PREFERENCE: dict[str, tuple[str, ...]] = {
    "encoded_gemm": ("encoded_gemm", "decoded_float", "verbatim"),
    "decode_floor": ("decoded_float", "verbatim", "encoded_gemm"),
}


def route_label(census: Mapping[str, int]) -> str:
    """A route census -> the plan's human-facing name. THE derivation, in one place.

    The old rung name was chosen by a chooser that had no per-tensor knowledge, so an
    artifact with 743 encoded leaves, 120 decoded weights and 937 verbatim copies reported
    `native_encoded` and every reader downstream believed it. This function cannot: a census
    carrying more than one route is UNNAMEABLE as a single rung, so it names all of them with
    counts and the caller gets a label it can print without lying.
    """
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


class PlanRefusal(Exception):
    """A typed pre-construction refusal on the plan axis."""

    def __init__(self, code: str, detail: str, walk: Sequence[Mapping[str, Any]] = ()) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail
        self.walk = [dict(row) for row in walk]


# ------------------------------------------------------------------------ the variants


@dataclass(frozen=True, slots=True)
class Variant:
    """One artifact of ONE logical model — the same topology, different stored bytes.

    `reference` marks the variant every other one is measured AGAINST. It is the deviation
    origin, so there is exactly one and a candidate set without one has no scale to put
    deviations on — a refusal, not a default.
    """

    name: str
    store: str
    snapshot: str
    snapshots: Mapping[str, str] = field(default_factory=dict)
    """component -> snapshot digest. A REAL map: a comma-packed one is a second grammar
    every reader has to agree about. It is the thing the plan digest binds, which is what
    stops a re-pointed variant NAME from inheriting the old bytes' measurements (#549.2)."""
    reference: bool = False

    def snapshot_map(self) -> dict[str, str]:
        return dict(self.snapshots)

    @classmethod
    def read(cls, record: Mapping[str, Any]) -> Variant:
        known = {f for f in cls.__dataclass_fields__}
        unknown = sorted(set(record) - known)
        if unknown:
            raise PlanRefusal(
                "variant_unknown_field",
                f"a variant record carries {', '.join(unknown)}: the key set is closed, "
                "exactly like the documents it stands in for",
            )
        return cls(**record)


@dataclass(frozen=True, slots=True)
class ConstructionFacts:
    """The half of a generation's identity that is NOT about bytes or hardware.

    Hashed into the plan because #549.2's key has to bind the construction contract and the
    config: a different topology under one variant name is a different resident footprint,
    and reusing its fill wall was one of the four unbound halves of the old `(variant, rung)`
    key.
    """

    release: str
    model_class: str
    config_digest: str
    topology_digest: str
    components: tuple[str, ...]
    custody: str = "canonical"
    strict_keys: bool = False

    def document(self) -> dict[str, Any]:
        return {
            "release": self.release,
            "model_class": self.model_class,
            "config_digest": self.config_digest,
            "topology_digest": self.topology_digest,
            "components": list(self.components),
            "custody": self.custody,
            "strict_keys": self.strict_keys,
        }


# ------------------------------------------------------------------- the per-tensor row


@dataclass(frozen=True, slots=True)
class TensorResolution:
    """ONE destination's whole answer. The unit the plan is made of.

    The old plan had no such unit: it had a per-tensor `Selection` that lived only inside the
    fill backend's dict, and an artifact-wide rung that was reported. This is what a reader
    and a digest both need — the encoding, the exact implementation, the route, the geometry
    that was VALIDATED, and why.
    """

    key: str
    component: str
    encoding: str
    """The spec object digest. Identity."""
    alias: str
    """Display only."""
    implementation: str
    implementation_digest: str
    implementation_revision: str
    route: str
    geometry: tuple[int, ...]
    geometry_class: str
    output_dtype: str
    numel: int
    stored_bytes: int
    resident_bits_per_element: float
    kernel_numeric_error: float
    reason: str
    """One sentence. #549.5's ragged-K case reads: the leaf refused this shape, here is the
    rule, and the decode floor serves it — IN the plan, not in a log nobody keeps."""

    def identity(self) -> dict[str, Any]:
        """What the plan DIGEST takes from this row. Measurements are excluded on purpose:
        the fact store is keyed BY the digest, so hashing a measurement into it would make
        every re-measure a different plan."""
        return {
            "key": self.key,
            "encoding": self.encoding,
            "implementation_digest": self.implementation_digest,
            "implementation_revision": self.implementation_revision,
            "route": self.route,
            "geometry": list(self.geometry),
            "output_dtype": self.output_dtype,
        }

    def document(self) -> dict[str, Any]:
        return {
            **self.identity(),
            "component": self.component,
            "alias": self.alias,
            "implementation": self.implementation,
            "geometry_class": self.geometry_class,
            "numel": self.numel,
            "stored_bytes": self.stored_bytes,
            "resident_bits_per_element": self.resident_bits_per_element,
            "kernel_numeric_error": self.kernel_numeric_error,
            "reason": self.reason,
        }


# ------------------------------------------------------------------------- THE PLAN


@dataclass(frozen=True, slots=True)
class ResolvedExecutionPlan:
    """The one immutable object a generation is. Resolved once, before anything is reserved.

    Its `digest()` enters generation identity and its `label()` derives from what it actually
    resolved — which is the inversion #549.1 asked for. Before, a LABEL was hashed into
    identity and the per-tensor truth was not.
    """

    variant: str
    store: str
    snapshots: Mapping[str, str]
    reference_variant: str
    policy: str
    tensors: tuple[TensorResolution, ...]
    device: DeviceFacts
    runtime: RuntimeIdentity
    placement: str
    objective: str
    construction: ConstructionFacts
    encoded_leaves: str
    steps_basis: int = 20
    measurements: PlanFact | None = None
    confession: str = ""
    walk: tuple[dict[str, Any], ...] = ()

    # ------------------------------------------------------------------ derived facts

    def by_key(self) -> dict[str, TensorResolution]:
        """contract key -> its resolution. The fill plane EXECUTES this map (cr-025)."""
        return {row.key: row for row in self.tensors}

    def route_census(self) -> dict[str, int]:
        census: dict[str, int] = {}
        for row in self.tensors:
            census[row.route] = census.get(row.route, 0) + 1
        return census

    def label(self) -> str:
        """The human-facing name, DERIVED from the census — a mixed plan says so (#549.1)."""
        return route_label(self.route_census())

    def wire_delivery(self) -> str:
        """The wire's closed `delivery` vocabulary (`native | float`), derived HONESTLY.

        `native` means nothing decoded. One decoded tensor makes the generation a float one,
        because a float destination exists and is resident — which is what the word is for.
        """
        census = self.route_census()
        return "float" if census.get("decoded_float") else "native"

    def wire_materialization(self) -> str:
        """`aot_decode | staged_decode | jit_decode`, or empty when nothing decodes.

        Derived from the PLACEMENT, which is where the difference actually lives: the same
        decode paid once at fill is `aot_decode` and paid again on every stage-in is
        `staged_decode`. `jit_decode` stays unbuilt — see `delivery.py`.
        """
        if not self.route_census().get("decoded_float"):
            return ""
        return "staged_decode" if self.placement == "component_staged" else "aot_decode"

    def wire_rung(self) -> str:
        """The rung NAME the worker's placement fence prices this plan against.

        The worker seam is closed and it needs one: `worker/plan.py` looks the name up in
        the rung table to decide whether a walk to `component_staged` would change the
        compute KERNEL, which is §3.2's fence and is exactly the check that must keep
        working. So this is a DERIVATION, not a chooser's answer: it names the rung whose
        (delivery, materialization) pair equals this plan's own derived wire fields.

        A mixed plan therefore prices as what it COSTS — one decoded tensor makes the
        generation pay a decode, which is `aot_decode` or `staged_decode` — and it still
        REPORTS `label()`, which is the honest name. That is the split #549.1 asked for:
        the wire's closed vocabulary is satisfied by derivation, and nothing claims to be a
        single-route generation that is not.
        """
        # Imported HERE, not at module scope: `delivery.py` imports `ROUTE_KERNEL` from this
        # module, so a top-level import back would close the cycle. The dependency is one
        # way by design — resolution decides, the rung table describes.
        from cozy_runtime.internal.delivery import DELIVERY_RUNGS

        delivery, materialization = self.wire_delivery(), self.wire_materialization()
        for rung in DELIVERY_RUNGS:
            if (
                rung.status == "proven"
                and rung.delivery == delivery
                and rung.materialization == materialization
            ):
                return rung.name
        return "verbatim"

    def wire_document(self) -> dict[str, Any]:
        """The `delivery` document the worker reads off a prepare reply.

        Its KEY SET is the worker's, not this module's: `session.py` reads `rung` and
        `variant` and prints `delivery`/`materialization`/`confession`, and that seam is
        closed. Every value is derived from the resolution, and `label` and `plan_digest`
        ride along so a reader of the reply sees the honest name beside the wire's one.
        """
        return {
            "rung": self.wire_rung(),
            "variant": self.variant,
            "delivery": self.wire_delivery(),
            "materialization": self.wire_materialization(),
            "label": self.label(),
            "plan_digest": self.digest(),
            "routes": self.route_census(),
            "kernels": self.kernels(),
            "objective": self.objective,
            "calibrated": self.measurements is not None,
            "confession": self.confession,
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

    def derived_resident_bytes(self) -> int:
        """Resident bytes the CAPABILITY RECORDS predict for this plan.

        Every term is a measured number (`resident_bits_per_element`, minted by a probe that
        actually filled a tensor) times a header fact (`numel`). It is a DERIVATION, and the
        resolver prefers a banked whole-plan measurement whenever one exists — but it is
        derivable, which is why `footprint` never has to fall back to uncalibrated.
        """
        total = 0.0
        for row in self.tensors:
            bits = row.resident_bits_per_element
            total += (bits if bits >= 0.0 else 0.0) * row.numel
        return int(total / 8.0)

    def derived_kernel_numeric_error(self) -> float:
        """The WORST per-tensor route deviation. -1.0 when nothing measured any of them.

        Worst rather than mean: a plan is as numerically wrong as its wrongest tensor, and
        averaging 743 encoded leaves against 937 verbatim copies would report a plan that
        replaced every linear op as almost exact.
        """
        seen = [row.kernel_numeric_error for row in self.tensors if row.kernel_numeric_error >= 0.0]
        return max(seen) if seen else -1.0

    # --------------------------------------------------------------------- identity

    def identity(self) -> dict[str, Any]:
        return {
            "format": "cozy.runtime.ResolvedExecutionPlan/1",
            "store": self.store,
            "snapshots": dict(sorted(self.snapshots.items())),
            "tensors": [row.identity() for row in sorted(self.tensors, key=lambda r: r.key)],
            "device": self.device.document(),
            "runtime": self.runtime.document(),
            "placement": self.placement,
            "construction": self.construction.document(),
        }

    def digest(self) -> str:
        """THE plan's canonical digest — what enters generation identity.

        It binds the snapshot BYTES, the runtime build, the per-tensor provider map, the
        placement, the construction contract and the config, which are the things the old
        `(variant, rung)` fact key bound none of (#549.2).

        FOUR fields are deliberately ABSENT, and each absence is load-bearing:

        * the VARIANT NAME — two bindings that name the same bytes differently are one plan,
          and a name that re-points at different bytes is a different one;
        * the OBJECTIVE — it is the question that was asked, not the answer. Two deployments
          that state different axes and land on the same tensors, bytes and device are
          running the SAME execution, and a digest that separated them would file their
          measurements apart forever — so the resolver could never compare a candidate plan
          against a fact a differently-stated deployment banked. Found by an arm: with the
          objective inside, a plan measured under `footprint` was invisible to the `latency`
          resolution that had just built the identical plan;
        * the POLICY — a label for how the route assignment was reached. Two policies that
          reach the same assignment ARE the same plan, which is exactly why resolution dedups
          candidates by digest rather than by policy name;
        * CONSENT — it decides which plans get BUILT, never what a built plan does.
        """
        return canonical.digest(self.identity())

    def seam_document(self) -> dict[str, Any]:
        """The plan as ONE CONTROL FRAME — everything but the per-destination census.

        `document()` carries a row per destination because a plan's identity is per-tensor
        and a reader on disk wants all of it. The executor/worker seam is a 64 KiB CONTROL
        frame, and 2,651 destinations is about a megabyte: on H3 the prepare reply could not
        be sent at all, and what the worker got instead was a mutilated success (#574b).

        A control frame carries the DIGEST and the COUNTS; the rows stay where they are
        written whole. The digest is over the full document either way, so nothing about
        identity is weakened by not shipping the body of it down a control seam.
        """
        whole = self.document()
        rows = whole.pop("tensors", [])
        walk = whole.pop("walk", [])
        whole["tensors_count"] = len(rows)
        whole["walk_count"] = len(walk)
        whole["census_elided"] = (
            "the per-destination rows and the resolution walk are not on this seam: a "
            "control frame is 64 KiB and this plan resolves "
            f"{len(rows)} destinations. `plan_digest` is over the WHOLE document"
        )
        return whole

    def document(self) -> dict[str, Any]:
        return {
            "plan_digest": self.digest(),
            "label": self.label(),
            "variant": self.variant,
            "reference_variant": self.reference_variant,
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
            "objective": self.objective,
            "steps_basis": self.steps_basis,
            "device": self.device.document(),
            "runtime": self.runtime.document(),
            "construction": self.construction.document(),
            "stored_bytes": self.stored_bytes(),
            "derived_resident_bytes": self.derived_resident_bytes(),
            "derived_kernel_numeric_error": self.derived_kernel_numeric_error(),
            "calibrated": self.measurements is not None and self.measurements.samples > 0,
            "measured": self.measurements.document() if self.measurements else None,
            "confession": self.confession,
            "tensors": [row.document() for row in self.tensors],
            "walk": list(self.walk),
        }

    def summary(self) -> str:
        census = ", ".join(f"{r} x{n}" for r, n in sorted(self.route_census().items()))
        samples = self.measurements.samples if self.measurements else 0
        return (
            f"{self.label()} [{self.digest()[:19]}…] over {self.variant!r}: {census}; "
            f"objective {self.objective}, policy {self.policy}, {samples} banked sample(s)"
        )


# --------------------------------------------------------------------------- scoring


def score(plan: ResolvedExecutionPlan, objective: str) -> tuple[float, str]:
    """Lower is better. Returns `(score, provenance)`; `(-1.0, why)` means UNSCORABLE.

    Provenance is returned WITH the number because the two are not separable: a footprint
    score derived from capability records and one folded from four banked prepares are both
    honest and they are not the same claim, and a walk that printed only the number would let
    a reader mistake one for the other.
    """
    fact = plan.measurements
    if objective == "latency":
        # THE AXIS THAT USED TO REFUSE. It is measurable here and nowhere else: a fill wall
        # and a per-step wall are properties of a whole plan on a whole device, not of one
        # provider, which is exactly why `Capabilities.admit` had nothing to rank on and was
        # wrong to refuse instead of deferring (#549.3).
        if fact is None or not fact.timed or fact.fill_ms < 0:
            return -1.0, (
                "no banked fill+step wall for this plan digest; latency is a WHOLE-PLAN "
                "measurement and this plan has not run"
            )
        return (
            float(fact.fill_ms) + fact.step_ms * plan.steps_basis,
            f"{fact.fill_ms} ms cold fill + {fact.step_ms:.3f} ms/step x {plan.steps_basis} "
            f"declared steps, median of {fact.samples} run(s)",
        )
    if objective == "footprint":
        if fact is not None and fact.resident_bytes >= 0:
            scratch = max(fact.scratch_bytes, 0)
            return (
                float(fact.resident_bytes + scratch),
                f"{fact.resident_bytes} B resident + {scratch} B scratch, MEASURED, median "
                f"of {fact.samples} run(s)",
            )
        derived = plan.derived_resident_bytes()
        if derived <= 0:
            return -1.0, "no measured resident_bits_per_element on any of this plan's tensors"
        return (
            float(derived),
            f"{derived} B resident, DERIVED from each tensor's measured "
            "resident_bits_per_element x its element count",
        )
    if objective == "kernel_numeric_error":
        worst = plan.derived_kernel_numeric_error()
        if worst < 0.0:
            return -1.0, "no measured kernel_numeric_error on any of this plan's tensors"
        return worst, (
            f"{worst:.6f} relative, the WORST of this plan's {len(plan.tensors)} tensors, "
            "each measured against an f64 reference over its own exactly-dequantized weights"
        )
    return -1.0, f"{objective!r} names no axis this build can score"


# ------------------------------------------------------------------------ resolution


@dataclass(frozen=True, slots=True)
class PlanRow:
    """The header facts one destination contributes. Structurally what `fill.PlanRow` is;
    named here as a Protocol-shaped duck so `resolution.py` need not import the fill plane
    (which imports torch machinery) to resolve a plan from a header."""

    key: str
    component: str
    dtype_name: str
    shape: tuple[int, ...]
    encoded: Any


def resolve(
    *,
    variants: Sequence[Variant],
    rows_for: Mapping[str, Sequence[Any]],
    providers: Mapping[str, Any],
    capabilities: Any,
    device: DeviceFacts,
    runtime: RuntimeIdentity,
    construction: ConstructionFacts,
    facts: PlanFacts,
    objective: str = "latency",
    placement: str = "all_resident",
    encoded_leaves: str = "refuse",
    steps_basis: int = 20,
    dtype_name: Any,
) -> ResolvedExecutionPlan:
    """Offered variants + header rows + capability records + banked facts -> ONE plan.

    `rows_for` is variant name -> that variant's header rows (`fill.PlanRow`-shaped:
    `.key`, `.component`, `.dtype`, `.shape`, `.encoded`). `dtype_name` maps a TensorFS dtype
    name to the torch dtype NAME a capability record is keyed by — handed in rather than
    imported, so this module stays free of the fill plane.

    Every candidate is surveyed from its HEADER (cr-006 measured the whole per-tensor
    encoding plan of the real fp8 UNet at 400,592 B read and 6.7 MiB peak RSS, zero tensor
    bytes), so offering four variants costs four header reads and no load.
    """
    if objective not in OBJECTIVES:
        raise PlanRefusal(
            "unknown_objective",
            f"{objective!r} is not one of {OBJECTIVES}: the objective names a MEASURED axis, "
            "and one nothing measures cannot order anything",
        )
    if not variants:
        raise PlanRefusal(
            "no_plan_candidates",
            "this binding offers no artifact variant at all; a generation reads bytes or it "
            "does not exist",
        )
    reference = [v for v in variants if v.reference]
    if len(reference) != 1:
        raise PlanRefusal(
            "reference_variant_unresolved",
            f"{len(reference)} of {len(variants)} offered variants claim to be the reference "
            "and exactly one must: deviation is a number ABOUT a reference, so a candidate "
            "set without one puts every encoded route on no scale at all",
        )

    walk: list[dict[str, Any]] = []
    candidates: list[tuple[ResolvedExecutionPlan, float, str]] = []
    floors: dict[str, ResolvedExecutionPlan] = {}

    for variant in variants:
        rows = list(rows_for.get(variant.name, ()))
        if not rows:
            walk.append({"variant": variant.name, "verdict": "no_rows", "why": "no tensor rows"})
            continue
        selector = Selector(providers, capabilities, device, objective)
        try:
            resolved_candidates = [
                (
                    row,
                    selector.candidates(
                        row.key, row.encoded, dtype_name(row.dtype), tuple(row.shape)
                    ),
                )
                for row in rows
            ]
        except EncodingRefusal as exc:
            walk.append(
                {
                    "variant": variant.name,
                    "verdict": exc.code,
                    "why": str(exc)[:600],
                }
            )
            continue
        unqualified = next((c for _, c in resolved_candidates if not c.by_route), None)
        if unqualified is not None:
            walk.append(
                {
                    "variant": variant.name,
                    "verdict": "unqualified",
                    "why": unqualified.unqualified[:600],
                }
            )
            continue
        seen: set[str] = set()
        for policy in POLICIES:
            if policy == "encoded_gemm" and encoded_leaves != "accept":
                # THE CONSENT GATE, MOVED INTO RESOLUTION. It used to fire at `materialize`,
                # with the plan already chosen and the package already imported; a plan the
                # package cannot consent to is not a candidate, so it is never built.
                walk.append(
                    {
                        "variant": variant.name,
                        "policy": policy,
                        "verdict": "unconsented",
                        "why": f"the package declares encoded_leaves={encoded_leaves!r}; a "
                        "route that REPLACES the module that computes needs the package to "
                        "say so, because code that reads `.weight` finds nothing there",
                    }
                )
                continue
            plan = _build(
                variant=variant,
                reference=reference[0],
                policy=policy,
                resolved=resolved_candidates,
                device=device,
                runtime=runtime,
                construction=construction,
                objective=objective,
                placement=placement,
                encoded_leaves=encoded_leaves,
                steps_basis=steps_basis,
                dtype_name=dtype_name,
            )
            digest = plan.digest()
            if digest in seen:
                # The two policies COINCIDE whenever no tensor has a real choice, which is
                # every single-route artifact. Deduping by digest rather than by policy name
                # keeps the walk honest about how many DISTINCT plans there were.
                continue
            seen.add(digest)
            plan = _with_facts(plan, facts)
            if policy == "decode_floor" or not plan.route_census().get("encoded_gemm"):
                floors.setdefault(variant.name, plan)
            value, provenance = score(plan, objective)
            row: dict[str, Any] = {
                "variant": variant.name,
                "policy": policy,
                "plan": digest,
                "label": plan.label(),
                "routes": plan.route_census(),
                "samples": plan.measurements.samples if plan.measurements else 0,
            }
            if value < 0.0:
                row["verdict"] = "unscored"
                row["why"] = provenance
                walk.append(row)
                continue
            row["verdict"] = "eligible"
            row["score"] = value
            row["provenance"] = provenance
            walk.append(row)
            candidates.append((plan, value, provenance))

    if candidates:
        best, value, provenance = min(candidates, key=lambda e: (e[1], e[0].digest()))
        for row in walk:
            if row.get("plan") == best.digest():
                row["verdict"] = "chosen"
        confession = ""
        if best.variant != reference[0].name:
            deviation = best.measurements.reference_deviation_rel_l2 if best.measurements else -1.0
            confession = (
                f"PLAN: served from the {best.variant!r} variant rather than the "
                f"{reference[0].name!r} reference, chosen on the {objective} axis "
                f"({provenance}); measured deviation from the reference is "
                + (
                    f"{deviation:.5f} relative L2 on one denoise step"
                    if deviation >= 0.0
                    else "NOT MEASURED"
                )
            )
        return _replace(best, confession=confession, walk=tuple(walk))

    # NOTHING SCORED. This is the path the old design REFUSED on (`objective_unranked`) and
    # it is the ordinary state of a rig that has never run this plan: the deployment still
    # has to serve, and the one plan that needs no comparison to justify is the reference
    # variant's decode floor — the plan every deviation is measured against.
    floor = floors.get(reference[0].name)
    if floor is not None:
        for row in walk:
            if row.get("plan") == floor.digest():
                row["verdict"] = "chosen_uncalibrated"
        return _replace(
            floor,
            confession=(
                f"UNCALIBRATED: no offered plan has a banked {objective} measurement on "
                f"{device.tuple_key()}, so resolution takes the {reference[0].name!r} "
                f"REFERENCE variant's DECODE FLOOR ({floor.label()}). An unmeasured plan may "
                "not be SELECTED over a measured one, and this prepare banks the fact the "
                "next resolution reads. This is a CONFESSION, not a refusal: the objective "
                "having no number yet is the ordinary first-run state, and refusing here is "
                "what #549.3 removed"
            ),
            walk=tuple(walk),
        )
    raise PlanRefusal(
        "no_qualified_plan",
        f"no offered artifact variant resolves to an executable plan on "
        f"{device_line(device)}. The topology is known and understood; what is missing is a "
        "route from these stored bytes to a kernel this device has. "
        + " | ".join(
            f"{r.get('variant', '-')}/{r.get('policy', '-')}: {r.get('verdict')} — "
            f"{str(r.get('why', ''))[:200]}"
            for r in walk
        ),
        walk,
    )


def script_plan(
    *,
    rows: Sequence[Any],
    components: Sequence[str],
    release: str,
    store: str,
    snapshot: str,
    providers: Mapping[str, Any],
    capabilities: Any,
    device: DeviceFacts,
    runtime: RuntimeIdentity,
    facts: PlanFacts,
    dtype_name: Any,
    snapshots: Mapping[str, str] | None = None,
    objective: str = "latency",
    placement: str = "all_resident",
    encoded_leaves: str = "accept",
) -> ResolvedExecutionPlan:
    """MODEL-LESS resolution for a driver script: one reference variant, these header rows.

    cr-025's one-selection-authority law says the fill plane executes a resolved plan and
    never selects below it. The executor resolves through `resolve()` with the model
    class's own construction facts; a GPU driver script fills a raw component with no
    model class at all, so this builds the construction identity from the rows alone
    (`script:<components>`, the empty config) and runs THE SAME `resolve()`. Nothing here
    is a second chooser: every candidate, score and confession is `resolve()`'s.
    """
    construction = ConstructionFacts(
        release=release,
        model_class="script:" + "+".join(components),
        config_digest=canonical.digest({}),
        topology_digest=canonical.digest(
            {
                row.key: [list(row.shape), dtype_name(row.dtype), row.encoded.alias]
                for row in sorted(rows, key=lambda r: str(r.key))
            }
        ),
        components=tuple(components),
    )
    variant = Variant(
        name="script",
        store=store,
        snapshot=snapshot,
        snapshots=dict(snapshots or {}),
        reference=True,
    )
    return resolve(
        variants=[variant],
        rows_for={"script": list(rows)},
        providers=providers,
        capabilities=capabilities,
        device=device,
        runtime=runtime,
        construction=construction,
        facts=facts,
        objective=objective,
        placement=placement,
        encoded_leaves=encoded_leaves,
        dtype_name=dtype_name,
    )


def _build(
    *,
    variant: Variant,
    reference: Variant,
    policy: str,
    resolved: Sequence[tuple[Any, Any]],
    device: DeviceFacts,
    runtime: RuntimeIdentity,
    construction: ConstructionFacts,
    objective: str,
    placement: str,
    encoded_leaves: str,
    steps_basis: int,
    dtype_name: Any,
) -> ResolvedExecutionPlan:
    """One (variant, policy) -> its plan. Pure: no I/O, no device, no allocation."""
    tensors: list[TensorResolution] = []
    for row, found in resolved:
        selection = _prefer(found.by_route, policy)
        shape = tuple(int(n) for n in row.shape)
        numel = 1
        for extent in shape:
            numel *= extent
        tensors.append(
            TensorResolution(
                key=row.key,
                component=row.component,
                encoding=row.encoded.encoding,
                alias=row.encoded.alias,
                implementation=str(selection.provider.name),
                implementation_digest=selection.record.implementation_digest,
                implementation_revision=str(selection.provider.code_revision),
                route=str(selection.provider.route),
                geometry=shape,
                geometry_class=geometry_class(shape),
                output_dtype=dtype_name(row.dtype),
                numel=numel,
                stored_bytes=row.encoded.stored_nbytes,
                resident_bits_per_element=selection.record.resident_bits_per_element,
                kernel_numeric_error=selection.record.kernel_numeric_error,
                reason=_reason(selection, found, policy),
            )
        )
    return ResolvedExecutionPlan(
        variant=variant.name,
        store=variant.store,
        snapshots=dict(
            variant.snapshot_map() or {c: variant.snapshot for c in construction.components}
        ),
        reference_variant=reference.name,
        policy=policy,
        tensors=tuple(tensors),
        device=device,
        runtime=runtime,
        placement=placement,
        objective=objective,
        construction=construction,
        encoded_leaves=encoded_leaves,
        steps_basis=steps_basis,
    )


def _prefer(by_route: Mapping[str, Selection], policy: str) -> Selection:
    for route in _PREFERENCE[policy]:
        found = by_route.get(route)
        if found is not None:
            return found
    # `by_route` is non-empty by construction (the caller refuses an unqualified tensor
    # before it gets here) and every route in it is one of the three.
    return next(iter(by_route.values()))


def _reason(selection: Selection, found: Any, policy: str) -> str:
    """WHY this tensor got this route, in one sentence, recorded IN the plan.

    #549.5's ragged-K case is the one that made this mandatory: a plan that quietly resolves
    a `mxfp8/1` tensor to the decode floor while its 743 neighbours take the leaf is CORRECT
    and completely opaque, and an operator looking at a footprint number that did not move
    has nowhere to look.
    """
    parts = [selection.reason]
    if found.geometry_refused:
        parts.append(
            "GEOMETRY: "
            + " | ".join(f"{name} refused — {why}" for name, why in found.geometry_refused)
        )
    if policy == "decode_floor" and "encoded_gemm" in found.by_route:
        parts.append(
            "an encoded-GEMM implementation qualified for this tensor and the decode-floor "
            "policy did not take it"
        )
    return "; ".join(p for p in parts if p)


def _with_facts(plan: ResolvedExecutionPlan, facts: PlanFacts) -> ResolvedExecutionPlan:
    fact = facts.get(plan.digest())
    return _replace(plan, measurements=fact if fact.samples else None)


def _replace(plan: ResolvedExecutionPlan, **changes: Any) -> ResolvedExecutionPlan:
    import dataclasses

    return dataclasses.replace(plan, **changes)
