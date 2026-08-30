"""cr-008c — the DELIVERY axis, as DATA: what each route costs and when it is paid.

**What this module stopped being** (#549.1). It used to hold a second chooser. `DeliveryChooser`
picked ONE artifact-wide rung over a `(variant, rung)`-keyed table while `encoding.Selector`
picked a provider PER TENSOR, and the two could not see each other: a mixed artifact reported
`native_encoded` while containing native, decoded and verbatim tensors, and the rung it
reported was hashed into generation identity despite having no execution authority — nothing
downstream ever read it to fill anything.

The chooser is gone, the `(variant, rung)` table is gone (see `planfacts.py` for what
replaced it and why the old rows are INVALID rather than migrated), and `resolution.py` makes
one joint decision over whole plans.

**What it still is, and why it is worth keeping.** A route's COST CLASS is a real fact that
belongs somewhere, and nothing else knows it: *when* is a decode paid — never, once at fill,
again on every stage-in, or inside the forward pass? That question has an answer per route
and per placement, it is the whole of pgw#1505, and `ResolvedModelPlan.wire_materialization`
reads it to derive the wire's `materialization` field honestly. The `jit_decode` entry is the
other reason: an UNBUILT rung is data, and its only consumer is the sentence a refusal prints
when someone asks for it.

So this file is now a table and two lookups. It decides nothing.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

#: WHICH KERNEL each route computes with. Re-exported from `resolution.py` so there is one
#: definition: the launch tier's two float routes both compute through the ordinary library
#: GEMM — that is what "kernel-neutral" means and it is checked, not asserted.
from cozy_runtime.internal.resolution import ROUTE_KERNEL

__all__ = [
    "DELIVERY_RUNGS",
    "ROUTE_KERNEL",
    "RUNGS_BY_NAME",
    "DeliveryRefusal",
    "DeliveryRung",
    "cost_class",
    "rungs_for_route",
]


@dataclass(frozen=True, slots=True)
class DeliveryRung:
    """One (route, when-it-is-paid) pair, with the sentence that explains it.

    `delivery` and `materialization` are the WIRE's own vocabulary (worker-protocol/01
    `AttemptPlanSummary`): delivery is `native | float`, materialization is
    `aot_decode | staged_decode | jit_decode` and is float-only. A resolved plan DERIVES
    both from its own route census and placement rather than carrying a rung; this table is
    what those derivations mean.

    `status` is `proven` or `unbuilt`. An unbuilt rung is DATA — there is no stub provider,
    no dormant enum case and no empty module.
    """

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
        owner="the leaf-replacement machinery EXISTS now (encoding's `LeafProvider`, "
        "fill.py's `_install_leaf`) and this rung still cannot use it: a JIT rung would "
        "hold the roles resident and decode INSIDE the forward, where the encoded-GEMM route "
        "holds the roles resident and never decodes at all. On the decoded_float route there "
        "is no encoded kernel to compute with, so a jit leaf would have to dequantize a full "
        "weight per step — a real thing to build, priced per_step, and NOT what "
        "`encoded_gemm` does. This rung refuses rather than aliasing itself onto either "
        "neighbour",
    ),
    DeliveryRung(
        name="encoded_gemm",
        delivery="native",
        materialization="",
        route="encoded_gemm",
        cost_class="none",
        why="the tensor cores consume the stored elements directly: no decode at all, the "
        "stored roles ARE the resident weight, and the only route on which an encoded lane "
        "is also a different kernel. It is therefore the only route that lowers steady-state "
        "VRAM — MEASURED at 0.5003x of the decode floor on a real H3 curve-DiT block "
        "(367.8 vs 735.0 MiB, 8.005 resident bits per GEMM element, on the fill plane's own "
        "ledger; results/leaf-native-report.json) — and the only one a package has to "
        "CONSENT to, because the module that computes is replaced. It costs 3.203 dB "
        "against the decode floor on identical stored bytes, which is the activation "
        "quantization a w8a8 route necessarily pays: this route is a TRADE, and which side "
        "of it a deployment wants is what the plan's OBJECTIVE is for. RENAMED from "
        "`native_encoded` (#549.11): 'native' named a device generation, and what is "
        "actually different here is that the GEMM consumes the stored elements",
        status="proven",
    ),
)

RUNGS_BY_NAME: dict[str, DeliveryRung] = {r.name: r for r in DELIVERY_RUNGS}


class DeliveryRefusal(Exception):
    """A typed refusal about the delivery axis. Kept because the wire and the worker's
    refusal matrix both name it; plan-level refusals are `resolution.PlanRefusal`."""

    def __init__(self, code: str, detail: str, walk: Sequence[Mapping[str, Any]] = ()) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail
        self.walk = [dict(row) for row in walk]


def rungs_for_route(route: str) -> tuple[DeliveryRung, ...]:
    """Every rung that describes one route, in ladder order."""
    return tuple(r for r in DELIVERY_RUNGS if r.route == route)


def cost_class(route: str, placement: str) -> str:
    """WHEN this route's decode is paid, given where the weights sit.

    The only place the placement axis and the delivery axis meet, and it is a lookup rather
    than a decision: a `decoded_float` tensor under `component_staged` re-decodes on every
    stage-in, which is `per_transition`, and the same tensor under `all_resident` decodes
    once. `ResolvedModelPlan` reads this to derive its wire fields.
    """
    if route == "verbatim" or route == "encoded_gemm":
        return "none"
    if route == "decoded_float":
        return "per_transition" if placement == "component_staged" else "per_generation"
    raise DeliveryRefusal(
        "unknown_route",
        f"{route!r} is not a delivery route this build knows; the route set is closed and "
        f"is {sorted(ROUTE_KERNEL)}",
    )
