"""One memory owner per GPU: the ladder, its termination and the one refusal, on a real
`Weights` (runtime-memory-v2.md §2-3).

Torch and the plane are real (CPU); the device's byte counts are the fixture
(`accel.allocation`), and what frees them is what would on a card: the owner's idle tenants
(`DeviceRoom`) and the caches kept for speed (`RELEASERS`)."""

from __future__ import annotations

import gc
import weakref
from typing import Any, cast

import pytest

from cozy_runtime.author import _split
from cozy_runtime.author._errors import DEVICE_OOM, MEMORY_EXHAUSTED, classify
from cozy_runtime.internal import accel, plane
from cozy_runtime.internal import memory as memory_module
from cozy_runtime.internal.executor import CAPACITY_CODES, Executor
from cozy_runtime.internal.memory import MARGIN, Ledger, MemoryExhausted, Rung
from cozy_runtime.internal.parallel import mirror
from cozy_runtime.internal.weights import Weights
from test_block_replay_contract import StatefulBlock, wrapped

torch = pytest.importorskip("torch")
pytest_plugins = ("test_request_isolation",)
MIB = 1 << 20


@pytest.fixture
def card(monkeypatch: pytest.MonkeyPatch) -> tuple[Weights, dict[str, int]]:
    weights = Weights(torch, torch.device("cpu"), "cpu")
    weights.host_only, weights.total = False, 8 << 30  # the device half, on a CPU plane
    monkeypatch.setattr(weights, "limit_torch", lambda: None)
    device = {"free": 0, "reserved": 0}
    monkeypatch.setattr(
        accel,
        "allocation",
        lambda *_, **__: {
            "allocated_bytes": 0,
            "reserved_bytes": device["reserved"],
            "driver_free_bytes": device["free"],
            "driver_total_bytes": 8 << 30,
        },
    )
    return weights, device


def gives(device: dict[str, int], amounts: list[int], calls: list[str], name: str) -> Any:
    """An owner or a cache that frees `amounts`, one per call, then nothing."""

    def give(*_: Any) -> bool:
        calls.append(name)
        device["free"] += amounts.pop(0) if amounts else 0
        return True

    return give


def test_each_rung_runs_while_it_gives_and_the_ladder_stops_once_enough_is_free(
    card: tuple[Weights, dict[str, int]], monkeypatch: pytest.MonkeyPatch
) -> None:
    weights, device = card
    calls: list[str] = []
    owner, asked = gives(device, [100 * MIB, 100 * MIB], calls, "owner"), []
    weights.room = lambda nbytes: (asked.append(nbytes), owner())
    monkeypatch.setattr(memory_module, "RELEASERS", [gives(device, [1 << 30], calls, "cache")])
    assert not weights.memory.free_for(500 * MIB, "aten.mm.default", upto=Rung.IDLE)
    assert calls == ["owner", "owner", "owner"], "asked while it gave, then left"
    assert asked == [500 * MIB + MARGIN] * 3, "each ask is all it wants free, never the rest"
    calls.clear()
    assert weights.memory.free_for(500 * MIB, "aten.mm.default")
    assert calls == ["owner", "cache"], "an idle owner first, then the running one's caches"
    assert weights.memory.free_for(500 * MIB, "aten.mm.default") and len(calls) == 2
    assert weights.modes["memory"] == f"idle 2x {200 * MIB} B, caches 1x {1 << 30} B"


def test_a_ladder_nothing_gives_ends_with_each_action_tried_once(
    card: tuple[Weights, dict[str, int]], monkeypatch: pytest.MonkeyPatch
) -> None:
    weights, device = card
    calls: list[str] = []
    weights.room = gives(device, [], calls, "owner")
    monkeypatch.setattr(memory_module, "RELEASERS", [gives(device, [], calls, "cache")])
    assert not weights.memory.free_for(1 << 30, "aten.mm.default")
    assert calls == ["owner", "cache"] and weights.memory.freed == 0
    assert weights.memory.again(1 << 30, "aten.mm.default") == -1


def test_an_attempt_runs_again_only_after_more_came_free(
    card: tuple[Weights, dict[str, int]],
) -> None:
    weights, device = card
    device["free"] = 1 << 30
    seen = weights.memory.again(MIB, "aten.mm.default")
    assert seen == 1 << 30
    assert weights.memory.again(MIB, "aten.mm.default", seen=seen) == -1, "the same room again"
    device["free"] += MIB
    assert weights.memory.again(MIB, "aten.mm.default", seen=seen) == seen + MIB


def test_the_one_refusal_carries_the_ledger_the_op_and_each_rung(
    card: tuple[Weights, dict[str, int]], monkeypatch: pytest.MonkeyPatch
) -> None:
    weights, device = card
    weights.room = gives(device, [64 * MIB], [], "owner")
    monkeypatch.setattr(memory_module, "RELEASERS", [])
    assert not weights.memory.free_for(1 << 30, "aten.mm.default")
    with pytest.raises(MemoryExhausted) as refused:
        weights.memory.exhausted(1 << 30, "aten.mm.default")
    failure = refused.value
    assert failure.code == MEMORY_EXHAUSTED and failure.nbytes == 1 << 30
    assert failure.tried == f"idle 1x {64 * MIB} B"
    assert "aten.mm.default needs 1073741824 B" in failure.detail
    assert "should have fit" in failure.detail and "unattributed 0" in failure.detail
    assert failure.shortfall == weights.memory.refusal
    assert failure.shortfall["needed_bytes"] == 1 << 30
    assert weights.modes["memory.exhausted"].startswith("aten.mm.default needs")
    # capacity everywhere it is read: the attempt fails, the process and its group stay
    outcome = classify(failure)
    assert (outcome.terminal, outcome.code) == ("failed", MEMORY_EXHAUSTED)
    assert MEMORY_EXHAUSTED in CAPACITY_CODES and DEVICE_OOM in CAPACITY_CODES
    assert mirror.out_of_memory(failure) == 1 << 30
    assert _split._short(failure)


def test_the_ledger_names_what_no_class_accounts_for() -> None:
    gib = 1 << 30
    ledger = Ledger(
        total=8 * gib,
        free=gib,
        context=gib // 2,
        foreign=2 * gib,
        reserved=3 * gib,
        allocated=2 * gib,
        mapped=gib,
        leased=gib // 4,
        budget=gib,
        shared=0,
        hosted=0,
        floor=0,
        cap=-1,
    )
    assert ledger.residue == gib // 2
    assert f"unattributed {gib // 2}" in ledger.text()
    unknown = Ledger(**{**{f: getattr(ledger, f) for f in Ledger.__slots__}, "foreign": -1})
    assert unknown.residue == 0 and "other processes unknown" in unknown.text()


class Ring:
    """A cursor a view into its ring keeps open: it closes once the view is gone."""

    def __init__(self, view: object) -> None:
        self.view = weakref.ref(view)

    def close(self) -> None:
        if self.view() is not None:
            raise plane.error("LeaseViolation")("1 live lease covers an open cursor")


def test_garbage_holding_a_ring_is_collected_before_anything_else_moves(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run 10562: a failed block's frames, in a reference cycle, held a view into its stage's
    ring, and the cursor could not close. Rung 1 collects the garbage, once, and closes the
    ring before it moves any weight; a ring a live view holds stays, and its weight set never
    moves to another grain under it (0.27.3)."""
    weights, _, _ = wrapped(StatefulBlock("counter"), monkeypatch)
    component, shrink, open_then = weights.components["model/1"], weights.shrink, []

    def unmap(*args: Any, **kwargs: Any) -> int:
        open_then.append(len(component.orphans))
        return shrink(*args, **kwargs)

    monkeypatch.setattr(weights, "shrink", unmap)

    class Frames:
        pass

    frames, live = Frames(), Frames()
    frames.cycle = frames  # type: ignore[attr-defined]
    component.orphans += [cast(Any, Ring(frames)), cast(Any, Ring(live))]
    del frames
    gc.disable()
    try:
        weights.memory.free_for(MIB, "stream", upto=Rung.IDLE)
    finally:
        gc.enable()
    assert open_then == [1], "weights moved only once the garbage was collected"
    assert not memory_module.reap(component)
    del live
    assert memory_module.reap(component) and not component.orphans


def test_no_room_ask_is_ever_for_nothing_or_less(
    card: tuple[Weights, dict[str, int]],
) -> None:
    """Runs 4267 and 10078 died on a negative ask from two sites (a stage's entry, a page-out
    of the machine's regions). Every ask goes through `Memory.ask`, which asks only for a
    positive number of bytes; with more free than needed nothing asks at all."""
    weights, device = card
    asked: list[int] = []
    weights.room = asked.append
    device["free"] = 4 << 30
    for nbytes in (-(22 * MIB), 0):
        weights.memory.ask(nbytes)
    assert weights.memory.free_for(MIB, "aten.mm.default")
    weights.cap = 2 << 30  # under an owner's cap, with room beside it
    weights.activation["denoise"] = 64 * MIB
    weights.room_before("denoise")
    device["reserved"] = 512 * MIB  # torch's idle cache covers the stage's activations
    weights.room_before("denoise")
    assert asked == []
    device["free"], device["reserved"] = 0, 0
    weights.room_before("denoise")
    assert asked and all(n > 0 for n in asked)


def test_a_capacity_failure_outside_the_handler_keeps_the_process(
    host: tuple[Executor, dict[str, Any]],
) -> None:
    executor, _ = host
    reply = executor.unsettled(executor, torch.OutOfMemoryError("CUDA out of memory."))
    assert reply["outcome"]["code"] == DEVICE_OOM and not reply["poisoned"]
    assert reply["quiescent"] and not executor.poisoned
    fault = executor.unsettled(executor, RuntimeError("an illegal memory access"))
    assert fault["poisoned"] and not fault["quiescent"] and executor.poisoned


MADE: list[weakref.ref[Any]] = []


@torch.library.custom_op("cozy_memory_test::grab", mutates_args=())  # type: ignore[untyped-decorator]
def grab(x: torch.Tensor) -> torch.Tensor:  # type: ignore[name-defined]
    """An op whose own frame holds a tensor when it runs out of memory, the first time."""
    held = torch.empty(4 * MIB, dtype=torch.uint8)
    MADE.append(weakref.ref(held))
    if len(MADE) == 1:
        raise torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 4.00 MiB")
    return x.clone()


def test_an_op_runs_again_only_once_its_failed_frames_let_go(
    card: tuple[Weights, dict[str, int]], monkeypatch: pytest.MonkeyPatch
) -> None:
    """pgw#1499: the failed op's frames held its tensors (6.92 GiB) while room was measured
    for it, through the exception's traceback. The op runner clears the traceback and leaves
    `except` before the ladder runs: a tensor the failed op made is gone by then."""
    weights, device = card
    seen: list[bool] = []

    def owner(_: int) -> None:
        seen.append(MADE[0]() is None)
        device["free"] += 1 << 30

    weights.room = owner
    monkeypatch.setattr(memory_module, "RELEASERS", [])
    MADE.clear()
    x = torch.ones(4)
    with weights.recovering():
        out = torch.ops.cozy_memory_test.grab(x)
    assert torch.equal(out, x) and len(MADE) == 2
    assert seen == [True], "the failed op's tensor was alive when room was measured"
    assert weights.oom_retries == 1


def test_an_allocation_refused_with_room_on_the_ledger_still_climbs_the_ladder(
    card: tuple[Weights, dict[str, int]],
) -> None:
    """Runs 10829 and 10837 (SDXL's encode on a 12 GB card beside an idle Klein): 25,493,504 B
    refused with 184,221,696 B free above the floor. The ledger said it fit, so the owner was
    never asked, and the run failed `memory_exhausted`. A second refusal with the same room is
    evidence the ledger is short for this op: the ask doubles and the ladder walks on."""
    weights, device = card
    asked: list[int] = []

    def owner(nbytes: int) -> None:  # the idle tenant's 7.56 GB, once asked for
        asked.append(nbytes)
        device["free"] += 7_560_000_000

    weights.room, weights.floor_free = owner, 268_435_456
    device["free"] = 184_221_696 + weights.floor_free
    need = 25_493_504
    seen = weights.memory.again(need, "aten.convolution.default")
    assert seen == 184_221_696 and not asked, "the first refusal runs again in what is free"
    now = weights.memory.again(need, "aten.convolution.default", seen=seen)
    assert now > seen and asked and asked[0] >= 2 * seen, "the second asks the owner"


@torch.library.custom_op("cozy_memory_test::stubborn", mutates_args=())  # type: ignore[untyped-decorator]
def stubborn(x: torch.Tensor) -> torch.Tensor:  # type: ignore[name-defined]
    """An op the allocator refuses while the ledger shows room, until an owner made more."""
    if not READY[0]:
        raise torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 24.31 MiB")
    return x.clone()


READY = [False]


def test_an_op_refused_with_room_on_the_ledger_runs_once_the_owner_made_more(
    card: tuple[Weights, dict[str, int]], monkeypatch: pytest.MonkeyPatch
) -> None:
    weights, device = card
    monkeypatch.setattr(memory_module, "RELEASERS", [])
    device["free"] = 184_221_696

    def owner(_: int) -> None:
        device["free"] += 7_560_000_000
        READY[0] = True

    weights.room = owner
    READY[0] = False
    x = torch.ones(4)
    with weights.recovering():
        assert torch.equal(torch.ops.cozy_memory_test.stubborn(x), x)
    assert READY[0] and weights.oom_retries == 2
