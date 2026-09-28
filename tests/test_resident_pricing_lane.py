"""cr-100 end to end: a tenant that holds its weights keeps them when the card tightens.

The REAL worker, real executor processes, real weights through the real fill plane, two
model-bearing placements on one card, and the real Claim/Snapshot/DesiredState/Offer
conversation — `tests/test_shared_lane.py`'s harness, unmodified in what it drives. Nothing
is patched and no number is handed to the planner.

WHAT IS REAL AND WHAT IS SIMULATED. Everything the worker does is real. What is simulated is
WHO TOOK THE ROOM: on the pod a co-tenant's prepare took it, and here a neighbour process
allocates between turns (a harness actor, decision #249 — never a plant in the runtime),
which the driver reports and the lane measures the same way. The arm does not turn on who
took it; it turns on the envelope having been banked when the room was there and read after
it is gone.

THE EXECUTOR IS KILLED between the first attempt and the rest, and that is load-bearing
rather than incidental. A caching allocator that has already cached the workspace it
measured keeps those segments, and `device_allocatable` counts them — so a generation that
banked a roomy envelope is still holding the room to honour it, and NOTHING is demoted while
it lives. The pod's degraded state is the state AFTER those segments went back: a vacate's
`release_cached`, or a process replacement. That is exactly what bake 3 measured (the OOM
poisoned the executor, the rebuild started a fresh generation, and cr-097's CARRIED envelope
priced it), and killing the process is the shortest honest way to reach it. It also means
this arm exercises the carried path end to end, while the arithmetic arms cover the
same-generation one.

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

import os
import signal
import subprocess
from pathlib import Path

import pytest

from local_owner import LocalRequest
from test_shared_lane import (
    DEVICE,
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
#: What the fixture's kernels ask for. They take less when the answer is no — but not for
#: free: torch's expandable segments keep the pages a FAILED allocation had already mapped,
#: so an appetite near the whole card leaves the process holding it and the next tenant
#: measuring 48 MB free. This is sized to succeed outright on a card this arm will run on,
#: which is what a real kernel's workspace selection does too.
WORKSPACE_MIB = 1536
#: room for one replacement executor's CUDA context, so the tightened card can still hold
#: the generation it is about to rebuild. Measured on this box at ~300 MiB; the arm only
#: needs it to be an over-estimate.
CONTEXT_PAD = 768 << 20
#: two executors, their weights, and a roomy workspace for the first attempt
NEEDS_FREE = 7 << 30


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


def _contended() -> str:
    """Why this card cannot answer the question right now, or "".

    The arm SIZES A NEIGHBOUR against the driver's free bytes and then reads what the lane
    measures. Another process arriving or leaving mid-run moves that number under it, and a
    run whose premise moved is not evidence either way — so it says so instead of failing
    or, worse, passing. Skipping is the honest answer for a shared box; the proof this arm
    stands in for is a rental with the card to itself.
    """
    try:
        rows = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.split("\n")
    except (OSError, subprocess.CalledProcessError) as exc:
        return f"the driver could not be asked who holds the card: {exc}"
    busy = [row for row in rows if row.strip()]
    if busy:
        return f"another process holds device {DEVICE}: {'; '.join(busy)}"
    if _free() < NEEDS_FREE:
        return f"the card has {_free()} B free and this arm needs {NEEDS_FREE} B"
    return ""


@needs_tenant
def test_a_tenant_that_holds_its_weights_is_not_demoted_when_the_card_tightens(
    store: TenantStore,
) -> None:
    """Four turns. A runs with the card roomy and banks its envelope under those
    conditions; its executor is replaced, so the generation that priced the next attempts
    holds no cached room; a neighbour takes what the envelope was measured in; A runs
    twice more.

    Every assertion is about those later A turns — the ones the pod spent at +100% forever.
    """
    contended = _contended()
    if contended:
        pytest.skip(contended)
    held: list[Neighbour] = []
    moved: list[str] = []
    first = _turn("plc-a", "a-1", 1)
    second = _turn("plc-b", "b-1", 2)

    def replace_executor() -> None:
        """The process replacement the pod reached through an OOM. Its allocator's cached
        segments go back to the driver with it, which is the state the envelope is then
        read in — and the carried measurement outlives it (cr-097)."""
        pid = int(first.rows["plc-a"]["executor_pid"])
        assert pid > 0, first.rows["plc-a"]
        os.kill(pid, signal.SIGKILL)

    def tighten() -> None:
        """Take the room the envelope was banked in, leaving enough for the replacement
        generation to refill and still work. The target is derived from what A ACTUALLY
        MEASURED, so the arm cannot be fooled by a fixture that allocated less than
        expected: it is short of the banked envelope by construction."""
        banked = max(first.rows["plc-a"]["activations_by_cell"].values(), default=0)
        target = store.weight_bytes + CONTEXT_PAD + (banked * 3) // 4
        if banked <= 0 or target >= banked or _free() <= target:
            # THE PREMISE DID NOT HOLD. A workspace that had to shrink because someone else
            # was on the card leaves nothing to tighten below, and a card already tighter
            # than the target is not this arm's card. Recorded and skipped at the end —
            # never asserted from inside the owner's callback, where it would surface as a
            # RecordOwner error and read like a defect in the worker.
            moved.append(
                f"banked {banked} B against a target of {target} B on a card with {_free()} B free"
            )
            return
        held.append(Neighbour(target))
        # The premise, re-read from the driver: if it did not land, the card is not ours
        # and the run is not evidence.
        if abs(held[-1].free_after - target) > (256 << 20):
            moved.append(
                f"the neighbour aimed at {target} B and the driver reports "
                f"{held[-1].free_after} B free"
            )

    first.after = replace_executor
    second.after = tighten
    turns = [first, second, _turn("plc-a", "a-2", 3), _turn("plc-a", "a-3", 4)]
    try:
        with _workspace() as root:
            owner, _worker, notes = _drive(store, turns, root, package=PACKAGE)
    finally:
        for neighbour in held:
            neighbour.close()
    if moved or _contended():
        pytest.skip(f"the card moved under this arm: {'; '.join(moved) or _contended()}")
    assert owner.ended == "terminal", (owner.ended, owner.error, owner.refusals)
    assert [turn.outcome["status"] for turn in owner.turns] == [1, 1, 1, 1], [
        (turn.request.request_id, turn.outcome.get("cause"), turn.outcome.get("safe_message"))
        for turn in owner.turns
    ]
    a1, b1, a2, a3 = owner.turns
    banked = max(a1.rows["plc-a"]["activations_by_cell"].values())
    tight = a2.rows["plc-a"]["walk"]["allocatable_bytes"]
    assert 0 < tight < banked, (tight, banked)

    # THE POINT. The card now has less room than the envelope A banked, on a generation
    # holding no cached segments to hide it — and A keeps the rung it is already paying
    # for. Before this change every turn from here on was staged, forever.
    for turn in (a1, a2, a3):
        assert _rung(turn, "plc-a") == "all_resident", (
            turn.request.request_id,
            turn.rows["plc-a"]["walk"]["rungs"],
        )
    assert a2.rows["plc-a"]["walk"]["headroom_basis"] in ("already_resident", "completed_resident")
    envelope = next(
        c for c in a2.rows["plc-a"]["walk"]["coefficients"] if c["name"] == "activations"
    )
    assert envelope["bytes"] == banked and not envelope["enforceable"], envelope
    assert a2.rows["plc-a"]["walk"]["enforceable_headroom_bytes"] == 0, a2.rows["plc-a"]["walk"]

    # THE RESIDENCY ITSELF, from the worker's own row: nothing was thrown away inside the
    # attempt. The pod's degraded runs peaked BELOW their own weight set, which is what a
    # rung that evicts on scope entry looks like from outside.
    for turn in (a2, a3):
        row = turn.rows["plc-a"]
        assert row["evicted"] == {}, (turn.request.request_id, row)
        assert row["resident"], (turn.request.request_id, row)

    # NO THRASH. The co-tenant is idle and its bytes are never taken: the fast rung is won
    # by not re-pricing residency that is already paid for, never by evicting a neighbour.
    assert _vacates(notes) == [], _vacates(notes)
    for turn in (a2, a3):
        other = turn.rows["plc-b"]
        assert other["evicted"] == {}, (turn.request.request_id, other)
        assert other["resident"], (turn.request.request_id, other)
    assert b1.outcome["status"] == 1, b1.outcome

    # THE KERNELS ADAPTED, which is the fact the whole conditions rule rests on: the same
    # entrypoint at the same shape took a smaller workspace on the tighter card, and
    # produced its result either way.
    assert 0 < _peak(a2) < _peak(a1), (_peak(a1), _peak(a2))
