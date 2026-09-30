"""Real Worker placement/queue projection; its hardware sampler is the scheduler fixture's."""

import pytest

from cozy_runtime.protocol import documents
from test_gpu_scheduler import Machine, _warm_shot
from test_gpu_scheduler import machine as machine


def test_queue_hints_keep_exact_templates_and_do_not_change_gpu_grants(
    machine: Machine, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, by_lane = _warm_shot(machine, monkeypatch)
    worker = machine.worker
    first, second = (worker.hosted[by_lane[name]].placement for name in ("lane-1", "lane-2"))
    before = worker.gpu.view()
    worker._memory_waiting = (
        (first.installation_id, documents.spell(first.bindings_digest)),
        (second.installation_id, documents.spell(second.bindings_digest)),
    )
    worker._refresh_memory_queue()
    # Another retained replica of the same immutable template is also a valid reuse
    # hint; the scheduler still chooses its device independently.
    expected = {
        h.placement.placement_id
        for h in worker.hosted.values()
        if (h.placement.installation_id, documents.spell(h.placement.bindings_digest))
        in worker._memory_waiting
    }
    assert {d.tenant for d in worker.memory.queue} == expected
    assert {(1,), (2,)} <= {d.devices for d in worker.memory.prefetch_frontier()}
    assert worker.gpu.view() == before
    # Identical installation alone is insufficient: a different checkpoint binding has
    # no known prepared representation and remains with ordinary artifact preparation.
    worker._memory_waiting = ((first.installation_id, "sha256:" + "a" * 64),)
    worker._refresh_memory_queue()
    assert not worker.memory.queue and not worker._page_warm_frontier
    assert worker.gpu.view() == before


def test_canceled_queue_interest_does_not_withdraw_execution(
    machine: Machine, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, by_lane = _warm_shot(machine, monkeypatch)
    worker = machine.worker
    placement = worker.hosted[by_lane["lane-0"]].placement
    worker._memory_waiting = (
        (placement.installation_id, documents.spell(placement.bindings_digest)),
    )
    worker._refresh_memory_queue()
    frontier = worker._page_warm_frontier
    before = worker.gpu.view()
    worker._memory_waiting = ()
    worker._refresh_memory_queue()
    assert frontier and not worker._page_warm_frontier
    assert worker.gpu.view() == before and placement.placement_id in worker.hosted
