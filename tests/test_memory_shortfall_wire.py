"""Signed planner debt must not destroy its terminal at the unsigned public wire.

CPU models exercise scope admission with controlled allocator readings; this is
not a qualification of physical GPU reclamation.
"""

from types import SimpleNamespace

import msgspec
import pytest

from cozy.worker.v1 import worker_pb2 as pb
from cozy_runtime.internal import executor_replies, memory, tiers
from cozy_runtime.internal.executor_replies import AttemptReply
from cozy_runtime.internal.worker.attempts import AttemptEngine, AttemptRecord
from cozy_runtime.internal.worker.ledger import Ledger
from cozy_runtime.protocol import documents


def terminal(monkeypatch: pytest.MonkeyPatch, needed: int, available: int) -> dict[str, object]:
    refusal = memory.MemoryRefusal(
        memory.SHORTFALL,
        f"scope needs {needed} B; {available} B are allocatable",
        needed,
        available,
    )
    reply = executor_replies.decode({"ok": False, "shortfall": refusal.shortfall}, AttemptReply)
    assert reply.shortfall is not None
    assert reply.shortfall.needed_bytes == needed and reply.shortfall.available_bytes == available
    engine = object.__new__(AttemptEngine)
    engine.boot_id, engine.instance_id = "boot", "instance"
    recorded: list[object] = []
    monkeypatch.setattr(
        engine,
        "records",
        SimpleNamespace(append=lambda *args: recorded.append(args)),
        raising=False,
    )
    monkeypatch.setattr(engine, "commit_bundle", lambda *args: None)
    attempt = AttemptRecord("request", 1, b"a" * 32, {}, executions=1)
    outcome = engine.outcome(
        attempt,
        pb.OUTCOME_STATUS_FAILED,
        pb.CAUSE_CODE_CONSTRAINT_INFEASIBLE,
        pb.CAUSE_ORIGIN_RUNTIME,
        str(refusal),
        shortfall=reply.shortfall,
    )
    body = documents.read(outcome.outcome_canonical_bytes, pb.AttemptOutcomeBody)
    assert recorded and attempt.state == "outcome"
    assert str(available) in body["safe_message"]
    assert refusal.available == available and refusal.needed == needed
    ledger = Ledger()
    ledger.observe_probe({"memory": {"reserve_room_bytes": available}})
    counter = next(row for row in ledger.counters() if row.name == "device_allocatable")
    assert counter.bytes == available and counter.kind == "measured"
    assert msgspec.to_builtins(ledger.measurement())["classes"]
    return body


@pytest.mark.parametrize("needed,available", [(0, -8650752), (-4, -9), (12, 7)])
def test_signed_shortfall_reaches_real_terminal_wire(
    monkeypatch: pytest.MonkeyPatch, needed: int, available: int
) -> None:
    body = terminal(monkeypatch, needed, available)
    shortfall = body["cause"]["shortfall"]
    assert shortfall.get("needed_bytes", 0) == max(needed, 0)
    assert shortfall.get("available_bytes", 0) == max(available, 0)
    assert body["cause"]["code"] == pb.CAUSE_CODE_CONSTRAINT_INFEASIBLE
    assert body["execution_started"] is True


@pytest.mark.parametrize("reclaimed", [True, False])
def test_scope_reclaims_cache_debt_before_zero_byte_admission(
    monkeypatch: pytest.MonkeyPatch, reclaimed: bool
) -> None:
    torch = pytest.importorskip("torch")
    root = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 2))
    root._no_split_modules = ["Linear"]
    owner = memory.Planner(torch, torch.device("cpu"))
    held = owner.adopt("generation", "unet", root, None, tiers.DEVICE)
    assert held.part.blocks and held.part.common.nbytes > 0
    for block in held.part.blocks:
        block.tier = tiers.HOST
    owner.cuda = True
    owner.grant = 100
    reserved = [108]
    monkeypatch.setattr(owner, "_driver", lambda: (1000, 2000))
    monkeypatch.setattr(owner, "_reserved", lambda: reserved[0])
    monkeypatch.setattr(owner, "_allocated", lambda: 100)
    monkeypatch.setattr(owner, "_cap", lambda: None)
    trims: list[int] = []

    def trim(_: object) -> None:
        trims.append(reserved[0])
        if reclaimed:
            reserved[0] = 100

    monkeypatch.setattr(tiers, "trim", trim)
    assert owner.allocatable() == -8
    if reclaimed:
        owner.enter("generation", "forward", ("unet",))
        assert owner.allocatable() == 0
        owner.exit()
    else:
        with pytest.raises(memory.MemoryRefusal) as raised:
            owner.enter("generation", "forward", ("unet",))
        assert raised.value.needed == 0 and raised.value.available == -8
        assert owner.allocatable() == -8 and not owner.scopes
        terminal(monkeypatch, raised.value.needed, raised.value.available)
    assert trims == [108] and owner.grant == 100
