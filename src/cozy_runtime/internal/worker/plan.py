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
    plan: str
    config: str
    snapshot: str
    release: str
    variant: str
    vram_bytes: int
    host_bytes: int
    pinned_bytes: int
    entrypoint: str
    model_construction_digest: str = ""

    def construction_key(self) -> str:
        """Two binding plans over the SAME construction share one executor generation."""
        return canonical.digest(
            {
                "store": self.store,
                "plan": self.plan,
                "config": self.config,
                "snapshot": self.snapshot,
                "release": self.release,
                "variant": self.variant,
                "model_class": self.model_class,
                "component": self.component,
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


class PlanChooser:
    """Pure arithmetic: request shape + bindings + live ledger -> ONE AttemptPlan.

    Deliberately a stub single-plan chooser (cr-008b owns the real ladder). It is a stub in
    its TABLE, not in its shape: the offer/prepare/accept seam around it, the reservation,
    the document identity and the summary are the real ones, so cr-008b lands by replacing
    the closed set this walks.
    """

    def __init__(self, ledger: Ledger) -> None:
        self.ledger = ledger
        self.choices = 0

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
        return AttemptPlan(
            delivery="native",
            materialization="aot_decode",
            compute_dtype="float16",
            placement="all_resident",
            reserved_vram_bytes=binding.vram_bytes,
            reserved_host_bytes=binding.host_bytes,
            reserved_pinned_bytes=binding.pinned_bytes,
            entrypoint_binding_plan_id=binding.plan_id,
            model_construction_digest=binding.model_construction_digest,
            quantified_choice=(
                f"one proven plan in the authorized set: fp16 weights resident on one "
                f"device, {binding.vram_bytes >> 20} MiB reserved; request shape "
                f"{features.get('shape', 'unknown')} needs no ladder step"
            ),
        )
