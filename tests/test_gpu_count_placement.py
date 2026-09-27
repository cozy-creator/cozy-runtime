import copy
from types import SimpleNamespace
from typing import Any, cast

import pytest

from cozy_runtime.internal import hostfacts
from cozy_runtime.internal.worker import machine_model_defaults
from cozy_runtime.internal.worker.workspace import WorkspaceRefusal
from cozy_runtime.protocol import worker_pb2 as pb


def test_counted_capture_needs_feature63_but_old_capture_accepts62() -> None:
    rung: dict[str, Any] = {
        "gpu": "H100",
        "repository": "proof/h3",
        "manifest": {"digest": "sha256:" + "a" * 64, "length": 1},
    }
    row: dict[str, Any] = {
        "callee_installation_id": "installed",
        "entrypoint": "run",
        "parameter": "model",
        "public_origin": "https://hub.example",
        "rungs": [rung],
    }
    capture = {
        "bindings": [{"callee_installation_id": "installed", "entrypoint": "run"}],
        "model_defaults": [row],
    }
    interfaces = {
        "installed": {"entrypoints": [{"name": "run", "models": [{"path": "run.models.model"}]}]}
    }
    worker = cast(Any, SimpleNamespace(config=SimpleNamespace(bootstrap_credential="")))
    request = pb.MachineExecutionSubmit(claim=pb.Claim(wire_minor=62))
    machine_model_defaults.verify(worker, request, capture, interfaces)
    rung["gpus"] = 2
    second = copy.deepcopy(rung)
    second["gpus"] = 4
    row["rungs"].append(second)
    with pytest.raises(WorkspaceRefusal, match="require wire63"):
        machine_model_defaults.verify(worker, request, capture, interfaces)
    request.claim.wire_minor = 63
    machine_model_defaults.verify(worker, request, capture, interfaces)


def test_selection_uses_granted_pool_and_old_rows_do_not_need_count_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    row: dict[str, Any] = {
        "callee_installation_id": "installed",
        "entrypoint": "run",
        "parameter": "model",
        "public_origin": "https://hub.example",
        "rungs": [{"gpu": "H100"}],
    }
    worker = SimpleNamespace(
        options=SimpleNamespace(accelerator_backend="cuda"),
        host_facts=lambda: hostfacts.measure("cuda"),
        executions=SimpleNamespace(
            capture_root=lambda *_: "root", capture=lambda *_: {"model_defaults": [row]}
        ),
    )
    target = cast(
        Any,
        SimpleNamespace(
            installation_id="installed",
            callee="",
            prepared_installation={},
            entrypoint="run",
            declaration={"models": [{"path": "run.models.model"}]},
        ),
    )
    call = cast(Any, SimpleNamespace(parent_request="parent"))
    monkeypatch.setattr(
        hostfacts,
        "measure",
        lambda _: hostfacts.HostFacts(
            gpu_name="NVIDIA H100", gpu_count=4, unreadable=("gpu_count",)
        ),
    )
    select = machine_model_defaults.select
    assert "gpus" not in select(cast(Any, worker), "owner", call, target, {})[0]
    row["rungs"] = [{"gpu": "H100", "gpus": 4}, {"gpu": "H100", "gpus": 2}]
    with pytest.raises(WorkspaceRefusal, match="readable granted GPU count"):
        select(cast(Any, worker), "owner", call, target, {})
    worker.lanes = SimpleNamespace(entries=[0, 1])
    assert select(cast(Any, worker), "owner", call, target, {})[0]["gpus"] == 2
