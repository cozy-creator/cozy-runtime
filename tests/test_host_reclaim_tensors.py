"""Real CPU tensor storage and fill-plane eviction; no CUDA calls or process retirement."""

from dataclasses import dataclass

import pytest

from cozy_runtime.internal import fill, memory, tiers

torch = pytest.importorskip("torch")


def resident(name, elements=32):
    owner = memory.Planner(torch, torch.device("cpu"))
    root = torch.nn.Module()
    root.weight = torch.nn.Parameter(torch.ones(elements))
    backend = fill.StreamingFillBackend.__new__(fill.StreamingFillBackend)
    backend.torch = torch
    backend.components = {name: root}
    backend.parked = {}
    backend.replaced = {}
    backend.stage_log = []
    backend.component_memory = {name: fill.ComponentMemory(destinations=elements * 4)}
    held = owner.adopt(name, name, root, None, tiers.HOST)
    held.part.store = backend
    return owner, held


@dataclass
class Tenant:
    slot: str
    planner: memory.Planner
    last_used: int = 0
    calls: int = 0

    def state(self):
        return "idle"

    def reclaim_host(self, nbytes):
        self.calls += 1
        released = self.planner.reclaim_host(nbytes)
        return released, self.planner.host_memory()["managed_host_bytes"]


@pytest.mark.parametrize(
    ("elements", "budget", "small", "needed"),
    [(32, 300, 40, 100), (12 * 1024 * 1024, 100 * 1024 * 1024, 1024, 39354880)],
)
def test_two_idle_tenants_reclaim_only_pressure_victim_and_remeasure_backing(
    elements, budget, small, needed
):
    first, a = resident("next", elements)
    second, b = resident("later", elements)
    tenants = [Tenant("next", first), Tenant("later", second)]
    policy = memory.Machine(
        {},
        lambda _: None,
        lambda _: None,
        host_available=lambda: (
            budget - sum(t.planner.host_memory()["managed_host_bytes"] for t in tenants)
        ),
    )
    policy.update_queue([memory.Demand("next", "a", "fp32", (0,), 0, 10)])
    assert policy.reserve_host("fits", small, requester="active", tenants=tenants)
    assert not a.part.root.weight.is_meta and not b.part.root.weight.is_meta
    policy.release_host("fits")
    assert policy.reserve_host("copy", needed, requester="active", tenants=tenants)
    assert not a.part.root.weight.is_meta and b.part.root.weight.is_meta
    assert tenants[0].calls == 0 and tenants[1].calls == 1
    assert first.host_memory()["managed_host_bytes"] == elements * 4
    assert second.host_memory()["managed_host_bytes"] == 0


@pytest.mark.parametrize("protected", ["mutable", "alias", "compiled", "active"])
def test_reclaim_preserves_retry_alias_compiled_and_active_state(protected):
    owner, held = resident("model")
    root = held.part.root
    original = root.weight
    if protected == "mutable":
        root._cozy_retry_tensors = lambda: (root.weight,)
    elif protected == "alias":
        root.register_buffer("view", root.weight.detach()[1:])
    elif protected == "compiled":
        root._compiled_call_impl = lambda *args: None
    else:
        owner.scopes.append(("model", frozenset()))
    assert owner.reclaim_host(128) == 0
    assert root.weight is original and not original.is_meta


def test_unique_host_accounting_and_clean_device_cache_release_preserve_device_weight():
    owner, held = resident("model")
    unit = held.part.common
    unit.tier = tiers.DEVICE  # CPU device stand-in; reclaim invokes no accelerator API.
    original = held.part.root.weight
    backing = torch.ones(64)
    unit.host[(1, "a")] = backing[:32]
    unit.host[(2, "b")] = backing[32:]
    assert owner.host_memory() == {
        "managed_host_bytes": 256,
        "resident_host_bytes": 0,
        "cached_host_bytes": 256,
    }
    assert owner.reclaim_host(256) == 256
    assert held.part.root.weight is original and not original.is_meta
    assert not unit.host
    # The caller still owns backing: logical release makes no claim of RSS relief.
    assert backing.numel() == 64


def test_executor_reclaim_reply_reports_residency_and_actual_rss_without_device_changes(tmp_path):
    import socket

    from cozy_runtime.internal.executor import Executor
    from cozy_runtime.internal.executor_commands import ReclaimHost, decode, encode
    from cozy_runtime.internal.seam import Channel

    owner, held = resident("model")
    ours, theirs = socket.socketpair()
    executor = Executor(Channel(ours), tmp_path)
    executor.torch, executor.planner, executor.device_kind = torch, owner, "cpu"
    try:
        reply = executor.handle(decode(encode(ReclaimHost(nbytes=128))))
        assert reply["ok"] and reply["managed_released_bytes"] == 128
        assert reply["host_memory"]["managed_host_bytes"] == 0
        assert reply["residency"]["disk_epoch"][held.key] == 1
        assert reply["rss_before_bytes"] > 0 and reply["rss_after_bytes"] > 0
        assert reply["rss_released_bytes"] == max(
            0, reply["rss_before_bytes"] - reply["rss_after_bytes"]
        )
    finally:
        executor.channel.close()
        theirs.close()
