"""cr-100: what a rung may be denied by, when the weights are already on the card.

The measured defect, on an rtx-a4000 with two placements: sdxl at 12.6 s alone and ~25 s on
every later run once a co-tenant had touched the pod, permanently, in any order. Queue time
never moved, the fill leg is 1.3 s, and the restore log says the weights came back — so the
loss was not scheduling and not data movement. It was the ladder walk. `all_resident` was
scored short by 1.18 GiB of ACTIVATIONS measured when sdxl had the card to itself, against a
card where the same model at the same shape had been observed to run in half that, and the
staged rung "fitted" only because it is allowed to free the 6.46 GiB of weights the restore
had just put back. The attempt then peaked BELOW its own weight set: admission threw them
away again inside the attempt and streamed them back per scope.

Two facts had to change together and neither is sufficient alone. The NUMBER: a peak
observed with more room than there is now is an upper bound the kernels chose (decisions
698), and `Coefficient.enforceable` already said such a number may not "refuse or select a
worse rung" — the walk was the consumer that ignored the second half. The RUNG: residency
that is already paid for is not re-priced against an estimate of what it would cost to
acquire, so a rung with nothing to bring back is denied only by evidence, never by an
estimate. Correcting only the number leaves the structure to lose on a tighter card;
correcting only the structure prices the fast rung against an inflated figure and buys the
fast path by evicting a neighbour that did not need evicting, which is the alternating
thrash the owner has named a defect.

The arithmetic arms below use the pod's own numbers and no device. The end-to-end arm is
`tests/test_resident_pricing_lane.py`, which needs a card.
"""

from __future__ import annotations

from cozy_runtime.internal.worker.ledger import Ledger
from cozy_runtime.internal.worker.plan import (
    UNCALIBRATED_ACTIVATION_BYTES,
    DeclaredBinding,
    PlanChooser,
    PreparedModel,
    PreparedRequest,
)

GiB = 1 << 30
CELL = "height=1024,width=1024"

#: sdxl's four components as the pod reported them, 6.461 GiB in total.
WEIGHTS = {
    "text_encoder": 246_000_000,
    "text_encoder_2": 1_400_000_000,
    "unet": 5_135_000_000,
    "vae": 167_307_726,
}
#: the activation peak a SUCCEEDING serve banked with the card to itself, and the free bytes
#: it was observed under. The same model at the same shape has run in 1.92 GiB beside a
#: co-tenant, which is what makes the larger figure a bound rather than a requirement.
SOLO_ACTIVATIONS = int(3.91 * GiB)
SOLO_FREE = int(9.1 * GiB)
#: the card after the co-tenant prepared and the restore put sdxl's weights back
TIGHT_FREE = int(2.81 * GiB)


def _binding() -> DeclaredBinding:
    return DeclaredBinding(
        entrypoint_binding_digest="sha256:binding",
        entrypoint="generate",
        model_class="Pipeline",
        model_binding_path="pkg:Pipeline",
        model_parameter_name="model",
        release="pkg@1",
        model_weight_bytes=sum(WEIGHTS.values()),
    )


def _resident_ledger(free: int = TIGHT_FREE) -> Ledger:
    """The pod after the restore: every declared component back on the card, a co-tenant
    beside it, and `free` bytes left for the work."""
    ledger = Ledger(worker_pid=1)
    ledger.begin_generation(357)
    ledger.observe_construction(
        {
            "filled_bytes": sum(WEIGHTS.values()),
            "filled": 2641,
            "allocator_bytes": sum(WEIGHTS.values()),
            "reserved_bytes": sum(WEIGHTS.values()),
            "resident": dict(WEIGHTS),
            "device_free_bytes": free,
            "device_total_bytes": int(15.6 * GiB),
            "declared_scopes": {
                "encode_prompt": ["text_encoder", "text_encoder_2"],
                "denoise": ["unet"],
                "decode": ["vae"],
            },
        }
    )
    ledger.device_process = sum(WEIGHTS.values())
    return ledger


def _bank_solo_serve(
    ledger: Ledger,
    peak: int = SOLO_ACTIVATIONS,
    free: int = SOLO_FREE,
    moved: bool = False,
) -> None:
    """One SUCCEEDING serve on a roomier card, with its conditions, exactly as a real one
    banks it: the peak, the per-scope peaks, and the free bytes it was observed under.

    `moved` is the difference between a serve that ran on the set the fill left and one
    that staged and evicted its way through — the second banks an envelope and NO
    sufficiency, because what it proves is about a card holding one component at a time.
    """
    was, ledger.device_free = ledger.device_free, free
    ledger.observe_attempt(
        {
            "activation_peaks": {
                "encode_prompt": peak // 8,
                "denoise": peak,
                "decode": peak // 3,
            }
        },
        resident_moved=moved,
        cell=CELL,
        succeeded=True,
    )
    ledger.device_free = was


def _walk(ledger: Ledger) -> tuple[str, dict]:
    chooser = PlanChooser(ledger)
    _terms, _rows, chosen, walk = chooser._walk(_binding(), CELL)
    return chosen, walk


def _row(walk: dict, rung: str) -> dict:
    return next(row for row in walk["rungs"] if row["rung"] == rung)


# --------------------------------------------------------------- the defect, in arithmetic


def test_a_generation_holding_its_weights_is_not_demoted_by_an_envelope_it_outgrew() -> None:
    """THE RED ARM. Every declared component is resident, so `all_resident` has nothing to
    bring back and nothing to move: its marginal cost is zero. The only banked envelope was
    measured with 9.1 GiB free and there are 2.81 GiB free now, so it is an upper bound the
    kernels chose — and an upper bound may not cost a tenant bytes it is already holding.

    Before this change the walk scored `all_resident` short by 1.18 GiB against that
    envelope and took `component_staged`, whose capacity counts freeing all 6.46 GiB of
    resident weights. That is the 25 s steady state: the plan chose to throw away the
    residency a restore had just paid for.
    """
    ledger = _resident_ledger()
    _bank_solo_serve(ledger)
    chosen, walk = _walk(ledger)
    assert chosen == "all_resident", walk["rungs"]
    resident = _row(walk, "all_resident")
    assert resident["needs_bytes"] == 0, resident
    assert resident["absent_bytes"] == 0 and resident["headroom_basis"] == "already_resident"
    # the envelope is still REPORTED, and still says what it is
    envelope = next(c for c in walk["coefficients"] if c["name"] == "activations")
    assert envelope["bytes"] == SOLO_ACTIVATIONS and envelope["provenance"] == "bound"
    assert not envelope["enforceable"] and "upper bound" in envelope["source"]


def test_what_the_executor_admits_against_is_what_the_walk_decided_with() -> None:
    """The plan's headroom rides `open_attempt` into `admit`, which refuses `device_shortfall`
    on `size + headroom` INSIDE the executor, past every upstream check. Handing it a figure
    the walk has just ruled inapplicable would re-litigate the conditions rule one layer
    down, where no gate above can see it — cr-097 closed exactly that reach for the carried
    scope measurements and this is the same closure for the scalar."""
    ledger = _resident_ledger()
    _bank_solo_serve(ledger)
    chooser = PlanChooser(ledger)
    plan = chooser.choose(
        _binding(),
        PreparedModel(delivery_rung="verbatim"),
        PreparedRequest.unresolved("generate", {"width": 1024, "height": 1024}),
    )
    assert plan.placement == "all_resident"
    assert plan.headroom_bytes == 0, plan.headroom_bytes
    assert dict(plan.scope_headroom_bytes) == dict.fromkeys(
        ("encode_prompt", "denoise", "decode"), 0
    ), plan.scope_headroom_bytes
    # and the person reading a terminal is told both numbers, not one of them
    assert f"{SOLO_ACTIVATIONS} B of activations (bound)" in plan.decision_explanation
    assert "admitted against 0 B (already_resident)" in plan.decision_explanation


def test_a_scope_measurement_that_does_not_apply_is_absent_whoever_measured_it() -> None:
    """cr-097 gated the CARRIED scope maps and left the generation's own live, so the same
    over-estimate still reached `admit` inside the generation that measured it. One
    predicate, at the top, for both."""
    ledger = _resident_ledger()
    _bank_solo_serve(ledger)
    assert ledger.activation_scopes_by_cell[CELL], "the generation banked its own scopes"
    assert ledger.activation_scopes_for(CELL) == {}, "and they do not apply on this card"
    ledger.device_free = SOLO_FREE
    assert ledger.activation_scopes_for(CELL)["denoise"] == SOLO_ACTIVATIONS


# ------------------------------------------------------------------------- the brakes


def test_a_measured_failure_at_no_more_room_than_this_denies_the_resident_rung() -> None:
    """THE BRAKE, and the reason the optimism above is bounded. An attempt for this shape
    already died on this device with 2.81 GiB free; a picture no better than that one
    cannot get a different answer, so the shallow rung is refused on evidence — not on an
    estimate — and the ladder deepens. One fault teaches the fact; nothing repeats it."""
    ledger = _resident_ledger()
    _bank_solo_serve(ledger)
    ledger.observe_device_shortfall(CELL, TIGHT_FREE)
    chosen, walk = _walk(ledger)
    assert chosen == "component_staged", walk["rungs"]
    resident = _row(walk, "all_resident")
    assert resident["verdict"] == "refused_on_measured_shortfall"
    assert resident["denied_at_free_bytes"] == TIGHT_FREE
    # and a picture with MORE room than the one that failed is not answered here
    ledger.device_free = TIGHT_FREE + GiB
    assert _walk(ledger)[0] == "all_resident"


def test_an_envelope_measured_on_a_card_this_tight_still_denies_the_rung() -> None:
    """The conditions rule cuts both ways. The same 3.91 GiB, measured with no more room
    than there is now, is a requirement the kernels actually had — it is enforceable, it
    denies the rung, and the ladder deepens exactly as it did before."""
    ledger = _resident_ledger()
    # banked by a STAGED serve on a card exactly this tight: it says what the kernels
    # allocated, and nothing about whether the whole set can be resident beside them.
    _bank_solo_serve(ledger, free=TIGHT_FREE, moved=True)
    chosen, walk = _walk(ledger)
    assert chosen == "component_staged", walk["rungs"]
    resident = _row(walk, "all_resident")
    assert resident["verdict"] == "short" and resident["needs_bytes"] == SOLO_ACTIVATIONS
    envelope = next(c for c in walk["coefficients"] if c["name"] == "activations")
    assert envelope["provenance"] == "measured" and envelope["enforceable"]


def test_nothing_measured_at_all_still_deepens() -> None:
    """A bound and an uncalibrated placeholder are NOT the same state, and only one of them
    knows anything. A bound has a completed serve behind it and a peak the kernels chose
    with more room; the placeholder has never seen this shape run. So it keeps the
    conservative answer it has always had — cr-063 measured what an over-eager resident
    rung costs on a card that cannot hold one (an all-resident warm pass that OOMs where
    the staged rung serves)."""
    ledger = _resident_ledger(free=UNCALIBRATED_ACTIVATION_BYTES // 2)
    chosen, walk = _walk(ledger)
    assert chosen == "component_staged", walk["rungs"]
    envelope = next(c for c in walk["coefficients"] if c["name"] == "activations")
    assert envelope["provenance"] == "uncalibrated"
    assert _row(walk, "all_resident")["needs_bytes"] == UNCALIBRATED_ACTIVATION_BYTES


def test_an_upper_bound_may_still_decline_to_acquire_weights() -> None:
    """The rule is about bytes ALREADY HELD, not about bytes in general. With components
    absent, the rung has real work to do — a multi-gigabyte re-stage and then the kernels —
    and declining to start it on a number nobody can enforce is the conservative answer.
    Spending residency that is already paid for on the same number is not."""
    ledger = _resident_ledger()
    _bank_solo_serve(ledger)
    ledger.evicted = {"unet": WEIGHTS["unet"]}
    ledger.resident = {k: v for k, v in WEIGHTS.items() if k != "unet"}
    chosen, walk = _walk(ledger)
    resident = _row(walk, "all_resident")
    assert resident["needs_bytes"] == SOLO_ACTIVATIONS + WEIGHTS["unet"], resident
    assert chosen != "all_resident", walk["rungs"]
    plan = PlanChooser(ledger).choose(
        _binding(),
        PreparedModel(delivery_rung="verbatim"),
        PreparedRequest.unresolved("generate", {"width": 1024, "height": 1024}),
    )
    assert plan.placement == "component_staged", plan.decision_explanation


# ------------------------------------------------------------------ the positive evidence


def test_a_serve_that_finished_without_moving_a_byte_outranks_the_envelope() -> None:
    """A peak is an inference about what the kernels would take; a completed serve at F free
    bytes, with the whole resident set standing still, is the outcome the rung is trying to
    predict, already observed. So it PERMITS the rung even against an envelope measured
    under conditions that do apply — and the steady state after this fix is EVIDENCED
    rather than merely un-denied."""
    ledger = _resident_ledger()
    # an envelope from a STAGED serve on a card exactly this tight: enforceable, and the
    # whole set plus that much scratch does not fit
    _bank_solo_serve(ledger, free=TIGHT_FREE, moved=True)
    assert _walk(ledger)[0] == "component_staged"
    # ... and then the same shape completes on this card with nothing moving
    ledger.observe_attempt(
        {"activation_peaks": {"denoise": int(1.92 * GiB)}},
        resident_moved=False,
        cell=CELL,
        succeeded=True,
    )
    assert ledger.resident_sufficiency(CELL) == TIGHT_FREE
    chosen, walk = _walk(ledger)
    assert chosen == "all_resident", walk["rungs"]
    assert _row(walk, "all_resident")["headroom_basis"] == "completed_resident"


def test_an_attempt_that_moved_components_proves_nothing_about_full_residency() -> None:
    """The fact is "this shape completed with its resident set STANDING STILL". A staged
    attempt freed and re-staged components while it ran, so it says nothing about whether
    the same work fits beside all of them at once — banking it would be the same category
    error one level down."""
    ledger = _resident_ledger()
    ledger.observe_attempt(
        {"activation_peaks": {"denoise": int(1.92 * GiB)}},
        resident_moved=True,
        cell=CELL,
        succeeded=True,
    )
    assert ledger.resident_sufficiency(CELL) == 0
