from types import SimpleNamespace

from cozy_runtime.internal.worker import machine_lanes


def test_counted_child_reserves_exact_group_and_old_capture_keeps_prior_behavior():
    worker = SimpleNamespace(
        lanes=SimpleNamespace(wide=True, entries=[0, 1, 2, 3], by_id={}),
        executions=SimpleNamespace(pending=lambda owner: []),
        engine=SimpleNamespace(history={}),
    )
    placement = {"installation_id": "test-install", "bindings_digest": "sha256:" + "a" * 64}
    h3 = [{"sequence_parallel": {"degrees": [2, 4]}}]
    _, pins = machine_lanes.replica(worker, "owner", placement, h3, [{"gpus": 2}])
    assert list(pins[0].device_ordinals) == [0, 1]
    _, pins = machine_lanes.replica(worker, "owner", placement, [{}], [{"gpus": 1}])
    assert list(pins[0].device_ordinals) == [0]
    _, pins = machine_lanes.replica(worker, "owner", placement, h3)
    assert list(pins[0].device_ordinals) == [0, 1, 2, 3]


def test_counted_group_cannot_exceed_granted_pool_even_on_a_larger_host():
    import pytest

    from cozy_runtime.internal.worker.workspace import WorkspaceRefusal

    worker = SimpleNamespace(lanes=SimpleNamespace(wide=False, entries=[0]))
    with pytest.raises(WorkspaceRefusal, match="granted device pool"):
        machine_lanes.replica(
            worker, "owner", {}, [{"sequence_parallel": {"degrees": [2, 4]}}], [{"gpus": 2}]
        )
