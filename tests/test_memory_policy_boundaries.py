"""Centralized policy keeps refusal facts and mechanism callback ordering unchanged."""

from collections.abc import Sequence
from contextlib import nullcontext
from types import SimpleNamespace
from typing import cast

import pytest

from cozy_runtime.internal import memory, tiers
from cozy_runtime.internal.executor_commands import Activate, Command, Restore
from cozy_runtime.internal.parallel.group import OpReady
from cozy_runtime.internal.worker.attempts import AttemptRecord, AttemptSlot
from cozy_runtime.internal.worker.session import Worker
from test_machine_memory_policy import Resident


@pytest.mark.parametrize("allowed", [False, True])
def test_startup_trim_requotes_then_requests_then_caps(
    monkeypatch: pytest.MonkeyPatch, allowed: bool
) -> None:
    torch = pytest.importorskip("torch")
    planner = memory.Planner(torch, torch.device("cpu"))
    planner.overhead, planner.grant = 40, 20
    readings = iter((6, 2))
    events: list[object] = []
    monkeypatch.setattr(planner, "_reserved", lambda: next(readings))
    monkeypatch.setattr(tiers, "trim", lambda _: events.append("trim"))
    monkeypatch.setattr(planner, "_cap", lambda: events.append(("cap", planner.grant)))

    def request(free: int, grant: int, measured: bool) -> int | None:
        events.append(("room", free, grant, measured))
        if allowed:
            planner.grant = grant
            return grant
        return None

    if allowed:
        planner.startup_budget(request)
    else:
        with pytest.raises(memory.MemoryRefusal) as failed:
            planner.startup_budget(request)
        assert (
            failed.value.code,
            failed.value.message,
            failed.value.needed,
            failed.value.available,
        ) == ("device_shortfall", "device startup exceeds its authorized footprint", 42, 20)
    assert events == ["trim", ("room", 0, 42, True), ("cap", 42 if allowed else 20)]


def test_machine_admission_reclaims_before_fresh_authorization() -> None:
    me = Resident("me", {0: 10}, {0: 10})
    peer = Resident("peer", {0: 70}, {0: 10})
    tenants: Sequence[memory.Tenant] = [me, peer]
    policy = memory.Machine(
        {-1: 100}, lambda _: (200 - sum(t.held[0] for t in tenants), 200), lambda _: None
    )

    def authorize() -> tuple[int, ...]:
        assert peer.actions == [("vacate", (0,))]
        assert peer.held == {0: 10}
        return policy.grants(me, tenants, "lane", ("0",), refuse=memory.MemoryRefusal)

    assert policy.attempt_grants(me, tenants, None, authorize) == (90,)
    assert peer.actions == [("vacate", (0,))]


def test_legacy_restore_gets_fresh_authorization_before_staging() -> None:
    events: list[str] = []
    quotes = iter(((17,), (23,)))

    def grant(*_args: object) -> tuple[int, ...]:
        events.append("quote")
        return next(quotes)

    def call(command: Command, **_kwargs: object) -> dict[str, object]:
        if isinstance(command, Activate):
            assert command.construction == "construction"
            assert command.authorized_device_limit_bytes == 17
            assert command.memory_grants == (17,)
            events.append("activate")
        else:
            assert isinstance(command, Restore) and command.names == ("weights",)
            events.append("restore")
        return {"ok": True}

    executor = SimpleNamespace(
        hello={"memory": ()},
        active="construction",
        call=call,
        watched=lambda *_: nullcontext(),
    )
    row = SimpleNamespace(
        peak={},
        executor=lambda: executor,
        ledger=SimpleNamespace(
            evicted={"weights": 8},
            observe_restore=lambda *_: None,
            observe_devices=lambda *_: None,
        ),
    )
    lane = SimpleNamespace(
        ordinals=(0,),
        row=lambda _: row,
        touch=lambda _: None,
        measure=lambda _: {},
        device_facts=lambda _: {},
    )
    worker = SimpleNamespace(
        accepted=SimpleNamespace(mode="serving"),
        lanes=SimpleNamespace(by_id={"lane": lane}),
        memory=SimpleNamespace(
            executor_placement=memory.Machine.executor_placement,
            attempt_grants=lambda _me, _tenants, _peak, authorize: authorize(),
        ),
        _tenants=lambda _: [],
        _grants=grant,
        _room=lambda *_: None,
        _observe_memory=lambda *_: None,
    )
    attempt = SimpleNamespace(
        spec={"serving": {"entrypoint_binding_digest": "binding"}},
        placement_id="p",
        prepared=None,
    )
    slot = SimpleNamespace(
        bindings={"binding": SimpleNamespace(weightless=False)},
        lane_id="lane",
    )
    Worker._arbitrate_residency(
        cast(Worker, worker), cast(AttemptRecord, attempt), cast(AttemptSlot, slot)
    )
    assert events == ["quote", "activate", "restore", "quote"]
    assert attempt.memory_grants == (23,) and attempt.memory_grant == 23


@pytest.mark.parametrize("kind", ["unknown", "negative", "zero"])
def test_machine_grant_refusal_messages_and_zero_boundary(kind: str) -> None:
    me = Resident("me", {0: 10}, {0: 10})
    peer = Resident("peer", {0: 150 if kind == "negative" else 100}, {0: 10}, status="active")
    policy = memory.Machine(
        {-1: 100}, lambda _: None if kind == "unknown" else (100, 300), lambda _: None
    )
    if kind == "zero":
        assert policy.grants(me, [me, peer], "lane", ("0",), refuse=memory.MemoryRefusal) == (0,)
    else:
        with pytest.raises(memory.MemoryRefusal) as failed:
            policy.grants(me, [me, peer], "lane", ("0",), refuse=memory.MemoryRefusal)
        if kind == "unknown":
            assert failed.value.code == "device_capacity_unreadable"
            assert (
                failed.value.message
                == "lane: device capacity or a live tenant's rank footprint is unknown for ('0',)"
            )
        else:
            assert failed.value.code == "device_shortfall"
            assert (
                failed.value.message == "lane: other tenants exceed the device budget: [(0, -50)]"
            )
    assert not peer.actions


@pytest.mark.parametrize(
    ("code", "reusable", "tried", "expected"),
    [
        ("host_shortfall", True, False, (True, True, False)),
        ("device_shortfall", True, False, (True, True, True)),
        ("device_out_of_memory", True, True, (True, False, True)),
        ("device_out_of_memory", False, False, (False, False, False)),
        ("canceled", True, False, (True, False, False)),
    ],
)
def test_group_retry_keeps_reusability_and_host_first_eligibility(
    code: str, reusable: bool, tried: bool, expected: tuple[bool, bool, bool]
) -> None:
    statuses = [
        OpReady(operation=1, rank=0, ok=False, code=code, reusable=reusable, host_needed=20)
    ]
    assert (
        memory.Planner.group_retry(
            statuses, host_tried=tried, host_available=True, device_available=True
        )
        == expected
    )
