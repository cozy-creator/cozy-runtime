"""cr-008c — the DELIVERY axis of the closed plan set: which stored bytes serve.

cr-008b's ladder is about WHERE weights sit (`all_resident | component_staged | ...`).
This is the other axis of the same cross: which ENCODING of the same logical model the
generation reads, and how its bytes become the thing that computes. cr-006 built the
per-tensor providers and the fail-closed capability table; what was missing is the choice
AMONG routes, which is this module.

Four rules hold it, and each is a refusal rather than a convention:

**A lane selects stored BYTES, never a kernel** (§3.2). A delivery rung says which
artifact the fill plane reads and what happens to its roles on the way to the
destination. It never picks a GEMM: on the decoded-float route the leaf that computes is
the author's ordinary module and the matmul is cuBLAS's, exactly as on the plain route.
`ROUTE_KERNEL` records that as data, and `PlanChooser` refuses a plan whose delivery
moved as a side effect of a placement rung.

**Arithmetic fit admits nothing.** A rung is ELIGIBLE only when (a) every encoding its
artifact cites has a qualified provider on THIS device — cr-006's capability records, fail
closed — and (b) a CONFORMANCE OBSERVATION exists proving the variant is semantically the
model it claims to be. A variant that merely fits refuses.

**A number nobody measured may not select a rung.** The table below is minted by running:
a variant's fill wall, resident bytes, decode scratch and step time are banked from real
prepares, its deviation from a real single-step comparison. With no measurement the
chooser takes the REFERENCE variant and confesses UNCALIBRATED — the same discipline
cr-008b's `Coefficient.enforceable` applies one axis over, and for the same v1 reason.

**The choice is made ONCE per generation, not per request.** Delivery decides which bytes
are resident, so it belongs to construction. `prepare` chooses; every attempt served by
that generation reports what it chose. A per-request delivery switch would mean a re-fill
per request, which is not a plan, it is a reload.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from cozy_runtime.internal import canonical
from cozy_runtime.internal.encoding import DeviceFacts, device_line


#: The CLOSED, ORDERED delivery ladder. `delivery` and `materialization` are the wire's
#: own vocabulary (worker-protocol/01 `AttemptPlanSummary`): delivery is `native | float`,
#: materialization is `aot_decode | staged_decode | jit_decode` and is float-only.
#:
#: `route` is the cr-006 provider route a rung DEMANDS, which is what ties a rung to the
#: capability table: a rung whose route no provider serves on this device is not eligible,
#: and saying so is the whole of "named, not built".
#:
#: `status` is `proven` or `unbuilt`. An unbuilt rung is DATA — there is no stub provider,
#: no dormant enum case and no empty module — and its only consumer is the refusal that
#: names it as the route that WOULD serve an artifact this build cannot.
@dataclass(frozen=True, slots=True)
class DeliveryRung:
    name: str
    delivery: str
    materialization: str
    route: str
    cost_class: str
    """WHEN the decode is paid, which is the whole of pgw#1505: `none` (nothing decodes),
    `per_generation` (once, at fill), `per_transition` (again on every stage-in) or
    `per_step` (inside the forward pass)."""
    why: str
    status: str
    owner: str = ""


DELIVERY_RUNGS: tuple[DeliveryRung, ...] = (
    DeliveryRung(
        name="verbatim",
        delivery="native",
        materialization="",
        route="verbatim",
        cost_class="none",
        why="the stored bytes ARE the logical bytes: the fill plane copies them into the "
        "destination and nothing decodes",
        status="proven",
    ),
    DeliveryRung(
        name="aot_decode",
        delivery="float",
        materialization="aot_decode",
        route="decoded_float",
        cost_class="per_generation",
        why="the stored roles decode into the destination's float dtype at FILL, once for "
        "the life of the generation; resident weights are float, so this buys bytes read "
        "and cold start and not VRAM",
        status="proven",
    ),
    DeliveryRung(
        name="staged_decode",
        delivery="float",
        materialization="staged_decode",
        route="decoded_float",
        cost_class="per_transition",
        why="the same decode, paid again on every stage-in: under a component_staged "
        "placement a parked component re-reads and re-decodes when it is next declared, so "
        "the decode is per-transition rather than once",
        status="proven",
    ),
    DeliveryRung(
        name="jit_decode",
        delivery="float",
        materialization="jit_decode",
        route="decoded_float",
        cost_class="per_step",
        why="the stored roles stay resident and each weight decodes inside the forward "
        "pass, which is the only rung that lowers steady-state residency below float",
        status="unbuilt",
        owner="the leaf-replacement machinery EXISTS now (encoding.py's `LeafProvider`, "
        "fill.py's `_install_leaf`) and this rung still cannot use it: a JIT rung would "
        "hold the roles resident and decode INSIDE the forward, where the native rung holds "
        "the roles resident and never decodes at all. On the decoded_float route there is "
        "no encoded kernel to compute with, so a jit leaf would have to dequantize a full "
        "weight per step — a real thing to build, priced per_step, and NOT what "
        "`native_encoded` does. This rung refuses rather than aliasing itself onto either "
        "neighbour",
    ),
    DeliveryRung(
        name="native_encoded",
        delivery="native",
        materialization="",
        route="native_encoded",
        cost_class="none",
        why="the tensor cores consume the stored elements directly: no decode at all, the "
        "stored roles ARE the resident weight, and the only rung on which an encoded lane "
        "is also a different kernel. It is therefore the only rung that lowers steady-state "
        "VRAM — MEASURED at 0.5003x of the decode floor on a real H3 curve-DiT block "
        "(367.8 vs 735.0 MiB, 8.005 resident bits per GEMM element, on the fill plane's own "
        "ledger; results/leaf-native-report.json) — and the only one an endpoint has to "
        "CONSENT to, because the module that computes is replaced. It costs 3.203 dB "
        "against the decode floor on identical stored bytes, which is the activation "
        "quantization a w8a8 route necessarily pays: this rung is a TRADE, and which side "
        "of it a deployment wants is what OBJECTIVES is for",
        status="proven",
    ),
)

RUNGS_BY_NAME: dict[str, DeliveryRung] = {r.name: r for r in DELIVERY_RUNGS}

#: WHICH KERNEL each route computes with. The launch tier's two proven rungs both compute
#: in the destination's float dtype through the ordinary library GEMM — that is what
#: "kernel-neutral" means and it is checked, not asserted: `PlanChooser` compares the
#: kernel of the chosen delivery before and after the placement walk.
ROUTE_KERNEL: dict[str, str] = {
    "verbatim": "library_float",
    "decoded_float": "library_float",
    "native_encoded": "native_encoded",
}

#: The measured axis a deployment optimizes. Not a request knob: it is a property of the
#: deployment, because the delivery choice is made once for the generation.
OBJECTIVES = ("latency", "footprint", "fidelity")


class DeliveryRefusal(Exception):
    """A typed pre-construction refusal on the delivery axis."""

    def __init__(self, code: str, detail: str, walk: Sequence[Mapping[str, Any]] = ()) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail
        self.walk = [dict(row) for row in walk]


@dataclass(frozen=True, slots=True)
class Variant:
    """One artifact of ONE logical model — the same topology, different stored bytes.

    `reference` marks the variant every other one is measured AGAINST. It is the fidelity
    origin, so there is exactly one and a candidate set without one has no scale to put
    deviations on — which is a refusal, not a default.
    """

    name: str
    store: str
    snapshot: str
    snapshots: Mapping[str, str] = field(default_factory=dict)
    """component -> snapshot digest. A REAL map, like the record it rides in: a comma-packed
    one is a second grammar every reader has to agree about."""
    reference: bool = False

    def snapshot_map(self) -> dict[str, str]:
        return dict(self.snapshots)

    @classmethod
    def read(cls, record: Mapping[str, Any]) -> Variant:
        known = {f for f in cls.__dataclass_fields__}
        unknown = sorted(set(record) - known)
        if unknown:
            raise DeliveryRefusal(
                "variant_unknown_field",
                f"a delivery variant record carries {', '.join(unknown)}: the key set is "
                "closed, exactly like the documents it stands in for",
            )
        return cls(**record)


@dataclass(frozen=True, slots=True)
class DeliveryFact:
    """One MEASURED envelope for one (variant, rung) on one device tuple.

    Every number here came off a real run and says which one. `samples` exists so a
    single-sample order can be LABELLED as such rather than presented as a distribution —
    cr-008c's checklist item, and the reason the chooser prints it in the walk.
    """

    variant: str
    rung: str
    stored_bytes: int
    resident_bytes: int
    scratch_bytes: int
    fill_ms: int
    decode_ms: int
    step_ms: float
    """Measured per-step wall of one forward pass. -1.0 means NOT MEASURED — a prepare can
    bank a fill wall but not a step time, and a zero would read as "instant" and win every
    latency comparison it was never entitled to enter."""
    deviation_rel_l2: float
    """Single-step relative L2 against the REFERENCE variant on identical inputs. The
    reference's own is 0.0 by construction. -1.0 means NOT MEASURED, which makes the
    variant ineligible: a conformance observation is what admits an encoded rung."""
    samples: int
    evidence: str
    encodings: tuple[str, ...] = ()

    @property
    def conformed(self) -> bool:
        return self.deviation_rel_l2 >= 0.0

    @property
    def timed(self) -> bool:
        return self.step_ms >= 0.0

    @property
    def single_sample(self) -> bool:
        return self.samples <= 1

    def document(self) -> dict[str, Any]:
        return {
            "variant": self.variant,
            "rung": self.rung,
            "stored_bytes": self.stored_bytes,
            "resident_bytes": self.resident_bytes,
            "scratch_bytes": self.scratch_bytes,
            "fill_ms": self.fill_ms,
            "decode_ms": self.decode_ms,
            "step_ms": self.step_ms,
            "deviation_rel_l2": self.deviation_rel_l2,
            "samples": self.samples,
            "evidence": self.evidence,
            "encodings": list(self.encodings),
        }

    @classmethod
    def read(cls, record: Mapping[str, Any]) -> DeliveryFact:
        data = dict(record)
        data["encodings"] = tuple(data.get("encodings") or ())
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})


class DeliveryTable:
    """The per-device plan table: measured envelopes, keyed by (variant, rung).

    Persisted beside the probe fact and for the same reason — it is a DEVICE property, not
    an artifact one — and written first-writer-wins so a fleet converges on one bank
    instead of racing. A row is added by whoever ran the thing that measured it.
    """

    FORMAT = "cozy.runtime.delivery-table/1"

    def __init__(self, tuple_key: str = "", rows: Sequence[DeliveryFact] = ()) -> None:
        self.tuple_key = tuple_key
        self.rows: dict[tuple[str, str], DeliveryFact] = {(r.variant, r.rung): r for r in rows}

    def get(self, variant: str, rung: str) -> DeliveryFact | None:
        return self.rows.get((variant, rung))

    def put(self, fact: DeliveryFact) -> None:
        self.rows[(fact.variant, fact.rung)] = fact

    def document(self) -> dict[str, Any]:
        return {
            "format": self.FORMAT,
            "tuple": self.tuple_key,
            "rows": [self.rows[key].document() for key in sorted(self.rows)],
        }

    @classmethod
    def path_for(cls, root: Path | str, device: DeviceFacts) -> Path:
        key = canonical.digest({"tuple": device.tuple_key()}).removeprefix("sha256:")[:32]
        return Path(root) / f"delivery-{key}.json"

    @classmethod
    def load(cls, path: Path | str | None, device: DeviceFacts) -> DeliveryTable:
        if path is None:
            return cls(device.tuple_key())
        file = Path(path)
        if not file.is_file():
            return cls(device.tuple_key())
        try:
            data = json.loads(file.read_text())
        except (OSError, ValueError):
            return cls(device.tuple_key())
        if not isinstance(data, dict) or data.get("format") != cls.FORMAT:
            return cls(device.tuple_key())
        return cls(str(data.get("tuple", "")), [DeliveryFact.read(r) for r in data.get("rows", ())])

    def save(self, path: Path | str) -> None:
        """MERGE, then write. Two builders measuring two variants must both survive: the
        probe cache can be O_EXCL because its content is one whole suite, and this file is
        an accumulating bank whose rows are minted at different times."""
        file = Path(path)
        file.parent.mkdir(parents=True, exist_ok=True)
        merged = DeliveryTable.load(file, DeviceFacts("", "", 0, "", ""))
        merged.tuple_key = self.tuple_key or merged.tuple_key
        for key in sorted(self.rows):
            merged.rows[key] = self.rows[key]
        body = json.dumps(merged.document(), sort_keys=True, indent=1).encode()
        temporary = file.with_suffix(".partial")
        with open(temporary, "wb") as handle:
            handle.write(body)
        os.replace(temporary, file)


@dataclass(frozen=True, slots=True)
class DeliveryChoice:
    """What the generation will read, and the whole walk that says why."""

    variant: Variant
    rung: DeliveryRung
    fact: DeliveryFact | None
    objective: str
    confession: str
    walk: tuple[dict[str, Any], ...] = field(default=())

    @property
    def calibrated(self) -> bool:
        return self.fact is not None

    def document(self) -> dict[str, Any]:
        return {
            "variant": self.variant.name,
            "rung": self.rung.name,
            "delivery": self.rung.delivery,
            "materialization": self.rung.materialization,
            "route": self.rung.route,
            "kernel": ROUTE_KERNEL[self.rung.route],
            "cost_class": self.rung.cost_class,
            "objective": self.objective,
            "calibrated": self.calibrated,
            "confession": self.confession,
            "measured": self.fact.document() if self.fact else None,
            "walk": list(self.walk),
        }


def eligible_rungs(routes: Mapping[str, int]) -> list[DeliveryRung]:
    """The rungs an artifact's own encodings admit, in ladder order.

    `routes` is route name -> how many of this artifact's tensors need it. An artifact
    whose tensors are all `verbatim` admits the verbatim rung; one carrying decoded_float
    tensors admits the decode rungs. A MIXED artifact — 937 plain and 743 encoded, which
    is exactly what the real fp8 SDXL UNet is — admits the decode rungs, because the plain
    tensors ride the same fill unchanged and the encoded ones decide the rung.

    **A rung is a LABEL over a per-tensor fill, and the two must not be confused.** Since
    the `quantize-mixed` job kind (#543g) an artifact may carry several ENCODINGS in one
    component — nvfp4 beside fp8-rowwise beside plain, allocation-map driven — and the fill
    plane resolves a provider per TENSOR from that tensor's own spec digest, so one module
    tree legitimately holds a native leaf beside a decoded float weight beside a verbatim
    copy. This function then reports every route such an artifact reaches, and the chooser
    ranks the rungs on their MEASURED envelopes; what it must never do is let one rung's
    label decide how a tensor of a different encoding is filled. It does not: nothing
    downstream reads the rung to pick a provider — `Selector.select` is called per
    destination, keyed on that destination's digest, and `_stream`/`_decode` branch per row.
    """
    wanted = {name for name, count in routes.items() if count > 0}
    if not wanted:
        return []
    if wanted == {"verbatim"}:
        return [r for r in DELIVERY_RUNGS if r.route == "verbatim"]
    encoded = sorted(wanted - {"verbatim"})
    return [r for r in DELIVERY_RUNGS if r.route in encoded]


class DeliveryChooser:
    """Candidates + capability records + the measured table -> ONE DeliveryChoice.

    Pure arithmetic, like `PlanChooser`: every input is a number or a record somebody else
    established by running something. It reads no header, opens no device and runs no
    probe — the caller does that once and hands the results in.
    """

    def __init__(self, table: DeliveryTable, device: DeviceFacts) -> None:
        self.table = table
        self.device = device

    def choose(
        self,
        candidates: Sequence[tuple[Variant, Mapping[str, int], str]],
        *,
        objective: str = "latency",
        steps_basis: int = 20,
    ) -> DeliveryChoice:
        """`candidates` is (variant, route -> tensor count, refusal-or-empty) per offer.

        The third element is the QUALIFICATION verdict the caller got from cr-006's
        `Selector` for that variant's encodings on this device: empty means every encoding
        it cites has a capability record, and a sentence means it does not — carried
        through rather than re-derived, so the refusal a reader sees is the one the
        capability table actually produced.
        """
        if objective not in OBJECTIVES:
            raise DeliveryRefusal(
                "unknown_objective",
                f"{objective!r} is not one of {OBJECTIVES}: the delivery objective names a "
                "MEASURED axis, and one nothing measures cannot order the ladder",
            )
        if not candidates:
            raise DeliveryRefusal(
                "no_delivery_candidates",
                "this binding offers no artifact variant at all; a generation reads bytes "
                "or it does not exist",
            )
        reference = [variant for variant, _, _ in candidates if variant.reference]
        if len(reference) != 1:
            raise DeliveryRefusal(
                "reference_variant_unresolved",
                f"{len(reference)} of {len(candidates)} offered variants claim to be the "
                "fidelity reference and exactly one must: deviation is a number ABOUT a "
                "reference, so a candidate set without one puts every encoded rung on no "
                "scale at all",
            )

        walk: list[dict[str, Any]] = []
        admitted: list[tuple[Variant, DeliveryRung, DeliveryFact]] = []
        for variant, routes, unqualified in candidates:
            rungs = eligible_rungs(routes)
            if not rungs:
                walk.append(
                    {"variant": variant.name, "verdict": "no_rung", "why": "no tensor rows"}
                )
                continue
            for rung in rungs:
                row: dict[str, Any] = {
                    "variant": variant.name,
                    "rung": rung.name,
                    "delivery": rung.delivery,
                    "materialization": rung.materialization,
                    "cost_class": rung.cost_class,
                }
                if rung.status != "proven":
                    row["verdict"] = "unbuilt"
                    row["why"] = f"{rung.why} — {rung.owner}"
                    walk.append(row)
                    continue
                if unqualified:
                    row["verdict"] = "unqualified"
                    row["why"] = unqualified
                    walk.append(row)
                    continue
                fact = self.table.get(variant.name, rung.name)
                if fact is None:
                    row["verdict"] = "unmeasured"
                    row["why"] = (
                        "no measured envelope for this (variant, rung) on "
                        f"{self.device.tuple_key()}: a rung with no measurement may not be "
                        "SELECTED, and the first run of this variant banks one"
                    )
                    walk.append(row)
                    continue
                if not fact.conformed and not variant.reference:
                    row["verdict"] = "unconformed"
                    row["why"] = (
                        "no conformance observation: an encoded rung is admitted by "
                        "evidence that it is still this model, never by fitting"
                    )
                    walk.append(row)
                    continue
                if not fact.timed:
                    row["verdict"] = "untimed"
                    row["why"] = (
                        "the fill wall is banked and the per-step wall is not. A prepare "
                        "measures a load; it cannot measure a forward pass, and a zero "
                        "would read as instant and win every latency comparison it was "
                        "never entitled to enter"
                    )
                    walk.append(row)
                    continue
                row["verdict"] = "eligible"
                row["score"] = self._score(fact, objective, steps_basis)
                row["measured"] = fact.document()
                row["single_sample"] = fact.single_sample
                walk.append(row)
                admitted.append((variant, rung, fact))

        if not admitted:
            return self._uncalibrated(reference[0], candidates, objective, walk)

        variant, rung, fact = min(
            admitted,
            key=lambda entry: (self._score(entry[2], objective, steps_basis), entry[0].name),
        )
        for row in walk:
            if row.get("variant") == variant.name and row.get("rung") == rung.name:
                row["verdict"] = "chosen"
        confession = ""
        if not variant.reference:
            confession = (
                f"DELIVERY: served from the {variant.name!r} variant rather than the "
                f"{reference[0].name!r} reference, chosen on the {objective} axis "
                f"({self._axis(fact, objective, steps_basis)}); measured deviation from the "
                f"reference is {fact.deviation_rel_l2:.5f} relative L2 on one denoise step"
                + (" (SINGLE SAMPLE)" if fact.single_sample else "")
            )
        return DeliveryChoice(variant, rung, fact, objective, confession, tuple(walk))

    def _uncalibrated(
        self,
        reference: Variant,
        candidates: Sequence[tuple[Variant, Mapping[str, int], str]],
        objective: str,
        walk: list[dict[str, Any]],
    ) -> DeliveryChoice:
        """Nothing measured qualifies. Take the REFERENCE if it is itself servable.

        The asymmetry is deliberate and is the same one cr-008b applies to coefficients: an
        unmeasured number may not SELECT a rung, but the deployment still has to serve. The
        reference variant is the one choice that needs no comparison to justify — it is
        what every deviation is measured against — so it is what an uncalibrated rig takes
        while the first prepare banks the measurement.
        """
        for variant, routes, unqualified in candidates:
            if variant is not reference:
                continue
            rungs = [r for r in eligible_rungs(routes) if r.status == "proven"]
            if unqualified or not rungs:
                break
            rung = rungs[0]
            for row in walk:
                if row.get("variant") == variant.name and row.get("rung") == rung.name:
                    row["verdict"] = "chosen_uncalibrated"
            return DeliveryChoice(
                variant,
                rung,
                None,
                objective,
                f"UNCALIBRATED: no measured envelope exists for any offered variant on "
                f"{self.device.tuple_key()}, so the ladder takes the {reference.name!r} "
                f"REFERENCE — an unmeasured rung may not be selected, and this prepare "
                f"banks the measurement the next one reads",
                tuple(walk),
            )
        raise DeliveryRefusal(
            "no_qualified_delivery",
            f"no offered artifact variant has a proven delivery rung on "
            f"{device_line(self.device)}. There is no valid execution plan for this "
            "binding: the topology is known and understood, and what is missing is a route "
            "from these stored bytes to a kernel this device has. "
            + " | ".join(
                f"{row['variant']}/{row.get('rung', '-')}: {row['verdict']} — {row.get('why', '')}"
                for row in walk
            ),
            walk,
        )

    @staticmethod
    def _score(fact: DeliveryFact, objective: str, steps_basis: int) -> float:
        """Lower is better, on the axis the objective names and on NOTHING else."""
        if objective == "latency":
            return float(fact.fill_ms) + fact.step_ms * steps_basis
        if objective == "footprint":
            return float(fact.resident_bytes + fact.scratch_bytes)
        return fact.deviation_rel_l2

    @staticmethod
    def _axis(fact: DeliveryFact, objective: str, steps_basis: int) -> str:
        if objective == "latency":
            return (
                f"{fact.fill_ms} ms cold fill + {fact.step_ms:.1f} ms/step x {steps_basis} "
                f"declared steps"
            )
        if objective == "footprint":
            return f"{fact.resident_bytes} B resident + {fact.scratch_bytes} B scratch"
        return f"{fact.deviation_rel_l2:.5f} relative L2"
