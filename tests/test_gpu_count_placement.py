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


def test_counted_capture_needs_feature63_but_old_capture_accepts62():
    import copy

    import pytest

    from cozy_runtime.internal.worker import machine_model_defaults
    from cozy_runtime.internal.worker.workspace import WorkspaceRefusal
    from cozy_runtime.protocol import worker_pb2 as pb

    row = {
        "callee_installation_id": "installed",
        "entrypoint": "run",
        "parameter": "model",
        "public_origin": "https://hub.example",
        "rungs": [
            {
                "gpu": "H100",
                "repository": "proof/h3",
                "manifest": {"digest": "sha256:" + "a" * 64, "length": 1},
            }
        ],
    }
    capture = {
        "bindings": [{"callee_installation_id": "installed", "entrypoint": "run"}],
        "model_defaults": [row],
    }
    interfaces = {
        "installed": {"entrypoints": [{"name": "run", "models": [{"path": "run.models.model"}]}]}
    }
    worker = SimpleNamespace(config=SimpleNamespace(bootstrap_credential=""))
    request = pb.MachineExecutionSubmit(claim=pb.Claim(wire_minor=62))
    machine_model_defaults.verify(worker, request, capture, interfaces)
    row["rungs"][0]["gpus"] = 2
    second = copy.deepcopy(row["rungs"][0])
    second["gpus"] = 4
    row["rungs"].append(second)
    with pytest.raises(WorkspaceRefusal, match="require wire63"):
        machine_model_defaults.verify(worker, request, capture, interfaces)
    request.claim.wire_minor = 63
    machine_model_defaults.verify(worker, request, capture, interfaces)
