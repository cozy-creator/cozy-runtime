"""cr-100 end to end: a tenant that holds its weights keeps them when the card tightens.

The REAL worker, real executor processes, real weights through the real fill plane, two
model-bearing placements on one card, and the real Claim/Snapshot/DesiredState/Offer
conversation — `tests/test_shared_lane.py`'s harness, unmodified in what it drives. Nothing
is patched and no number is handed to the planner.

WHAT IS REAL AND WHAT IS SIMULATED. Everything the worker does is real. What is simulated is
the CARD SHRINKING BETWEEN ATTEMPTS: on the pod it was a co-tenant's prepare taking the
slack, and here it is a neighbour process (a harness actor, decision #249 — never a plant in
the runtime) allocating between turns, which the driver reports and the lane measures the
same way. The distinction the arm turns on is not who took the room; it is that the
activation envelope was banked when the room was there and is read after it is gone.

The fixture is `corpus/scoped`, the same synthetic checkpoint `corpus/tenant` holds but with
ONE SCOPE PER COMPONENT. That matters: the tenant's `touch` declares every component in a
single scope, so it ends fully resident under either rung and cannot observe a demotion at
all. Its workspace HALVES until it fits, which is what the real kernels do (cuBLAS/cuDNN
workspace selection) and is the reason an envelope measured on a roomier card is an upper
bound rather than a requirement.

THE RED STATE: turn 2 planned `component_staged`, its peak device memory below its own
weight set, and its row left holding one component with two evicted — the pod's 25 s steady
state in miniature. THE GREEN STATE: `all_resident` throughout, the co-tenant's weights never
touched, and no vacate on the lane.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from cozy_runtime.internal.local import LocalRequest
from cozy_runtime.internal.worker.plan import UNCALIBRATED_ACTIVATION_BYTES
from test_shared_lane import (
    Neighbour,
    TenantStore,
    Turn,
    _drive,
    _free,
    _vacates,
    _workspace,
    needs_tenant,
    write_tenant_store,
)

ROOT = Path(__file__).resolve().parent.parent
PACKAGE = ROOT / "corpus" / "scoped"
#: what the fixture's kernels ask for. They take less when the answer is no.
WORKSPACE_MIB = 4096


@pytest.fixture(scope="module")
def store(tmp_path_factory: pytest.TempPathFactory) -> TenantStore:
    """One small tenant release: two of them and their scratch fit on any card that can
    hold the executors at all, which is what leaves room for the envelope to be banked
    ROOMY and then read TIGHT."""
    base = tmp_path_factory.mktemp("cr100")
    return write_tenant_store(base / "small", blocks=2, scale=4)


def _turn(placement_id: str, request_id: str, seed: int) -> Turn:
    return Turn(
        placement_id,
        LocalRequest(
            entrypoint="sweep",
            payload={"seed": seed, "workspace_mib": WORKSPACE_MIB},
            outputs=(),
            request_id=request_id,
        ),
    )


def _rung(turn: Turn, slot: str) -> str:
    """Which rung the walk that priced this attempt chose."""
    rungs = turn.rows[slot]["walk"].get("rungs", [])
    return next((row["rung"] for row in rungs if row["verdict"] == "chosen"), "")


def _peak(turn: Turn) -> int:
    return int((turn.outcome.get("metrics") or {}).get("peak_device_memory_bytes", 0))


@needs_tenant
def test_a_tenant_that_holds_its_weights_is_not_demoted_when_the_card_tightens(
    store: TenantStore,
) -> None:
    """Four turns. A runs with the card roomy and banks its envelope under those
    conditions; a neighbour then takes the room that envelope was measured in; A runs
    again, twice, with B running in between.

    Every assertion below is about the second and later A turns — the ones the pod spent
    at +100% forever.
    """
    weights = store.weight_bytes
    held: list[Neighbour] = []
    first = _turn("plc-a", "a-1", 1)

    def tighten() -> None:
        """Take the room the envelope was banked in, and no more. The target is derived
        from what A ACTUALLY MEASURED, so the arm cannot be fooled by a fixture that
        happened to allocate less than expected: it is short of the banked envelope by
        construction, which is exactly the pod's picture."""
        banked = max(first.rows["plc-a"]["activations_by_cell"].values(), default=0)
        assert banked > 0, first.rows["plc-a"]
        # Short of the envelope by construction, and never so short that an UNCALIBRATED
        # attempt could not fit: the co-tenant's own first serve is priced at the 1 GiB
        # placeholder, and starving it would make this arm about a different defect.
        target = max(banked // 2, UNCALIBRATED_ACTIVATION_BYTES + (256 << 20))
        assert target < banked, (target, banked)
        held.append(Neighbour(target))

    first.after = tighten
    turns = [first, _turn("plc-a", "a-2", 2), _turn("plc-b", "b-1", 3), _turn("plc-a", "a-3", 4)]
    try:
        with _workspace() as root:
            owner, _worker, notes = _drive(store, turns, root, package=PACKAGE)
    finally:
        for neighbour in held:
            neighbour.close()
    assert owner.ended == "terminal", (owner.ended, owner.error, owner.refusals)
    assert [turn.outcome["status"] for turn in owner.turns] == [1, 1, 1, 1], [
        (turn.request.request_id, turn.outcome.get("cause"), turn.outcome.get("safe_message"))
        for turn in owner.turns
    ]
    a1, a2, b1, a3 = owner.turns
    assert held and held[0].free_after < max(a1.rows["plc-a"]["activations_by_cell"].values())

    # THE POINT. The card is now tighter than the envelope A banked, and A keeps the rung
    # it was already paying for. Before this change every turn from here on was staged.
    for turn in (a1, a2, a3):
        assert _rung(turn, "plc-a") == "all_resident", (
            turn.request.request_id,
            turn.rows["plc-a"]["walk"]["rungs"],
        )
    assert a2.rows["plc-a"]["walk"]["headroom_basis"] == "already_resident"
    # ... and once a serve has COMPLETED on the tight card with nothing moving, the rung is
    # positively evidenced rather than merely un-denied.
    assert a3.rows["plc-a"]["walk"]["headroom_basis"] == "completed_resident"

    # THE RESIDENCY ITSELF, from the worker's own row: nothing was thrown away inside the
    # attempt. The pod's degraded runs peaked BELOW their own weight set, which is what a
    # rung that evicts on entry looks like from the outside.
    for turn in (a1, a2, a3):
        row = turn.rows["plc-a"]
        assert row["evicted"] == {}, (turn.request.request_id, row)
        assert sum(row["resident"].values()) >= weights * 0.99, (turn.request.request_id, row)
        assert _peak(turn) >= weights * 0.99, (turn.request.request_id, _peak(turn), weights)

    # NO THRASH. The co-tenant is idle and its bytes are never taken: the fast rung is won
    # by not re-pricing residency that is already paid for, never by evicting a neighbour.
    assert _vacates(notes) == [], _vacates(notes)
    for turn in owner.turns:
        other = turn.rows["plc-b"]
        assert other["evicted"] == {}, (turn.request.request_id, other)
        assert sum(other["resident"].values()) >= weights * 0.99, (turn.request.request_id, other)
    assert b1.outcome["status"] == 1, b1.outcome

    # THE KERNELS ADAPTED, which is the fact the whole conditions rule rests on: the same
    # entrypoint at the same shape took a smaller workspace on the tighter card and
    # produced its result either way.
    roomy = _peak(a1) - weights
    tight = _peak(a2) - weights
    assert 0 < tight < roomy, (roomy, tight)


@needs_tenant
def test_the_swap_itself_is_timed(store: TenantStore) -> None:
    """A vacate and a restore report their DURATION, and a restore splits the copy leg out.

    Nothing decides on these numbers — a duration that starts governing behaviour is the
    next measurement to be enforced outside its conditions. They exist because the cost of
    alternating two models on one card has only ever been INFERRED from a construction's
    fill leg, which is a different act under different conditions.
    """
    turns = [_turn("plc-a", "a-1", 1), _turn("plc-b", "b-1", 2), _turn("plc-a", "a-2", 3)]
    context = _free()
    neighbour = Neighbour(max(context // 2, store.weight_bytes + (512 << 20)))
    try:
        with _workspace() as root:
            owner, _worker, notes = _drive(store, turns, root, package=PACKAGE)
    finally:
        neighbour.close()
    assert owner.ended == "terminal", (owner.ended, owner.error, owner.refusals)
    residency = [step for kind, step in notes if kind == "residency"]
    moves: list[str] = [step for step in residency if " vacated " in step or " restored " in step]
    if not moves:
        pytest.skip(f"the card never had to swap on this host: {residency}")
    for step in moves:
        assert " ms" in step, step
        if " restored " in step:
            assert "fill leg" in step and "outside it" in step, step
            assert "page cache" in step or "cold" in step or "unreadable" in step, step
