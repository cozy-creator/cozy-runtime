from types import SimpleNamespace
from typing import Any, cast

import pytest

from cozy_runtime.internal import hostfacts, package_interface
from cozy_runtime.internal.worker import machine_model_defaults
from cozy_runtime.internal.worker.workspace import WorkspaceRefusal
from cozy_runtime.protocol import worker_pb2 as pb


def test_counted_capture_verifies() -> None:
    slot = package_interface.ModelSlot(path="run.models.model", class_name="M", component_use={})
    run = package_interface.CallableDoc(name="run", request={}, result={}, models=(slot,))
    interfaces = {
        "installed": package_interface.PackageInterface(
            format=package_interface.SCHEMA, application="app", entrypoints=(run,), jobs=()
        )
    }
    rung = pb.MachineModelDefaultRung(
        gpu="H100", repository="proof/h3", manifest=pb.Ref(digest=b"\xaa" * 32, length=1)
    )
    capture = pb.MachineExecutionCapture(
        bindings=[pb.MachineCallableBinding(callee_installation_id="installed", entrypoint="run")],
        model_defaults=[
            pb.MachineModelDefault(
                callee_installation_id="installed",
                entrypoint="run",
                parameter="model",
                public_origin="https://hub.example",
                rungs=[rung],
            )
        ],
    )
    worker = cast(
        Any, SimpleNamespace(options=SimpleNamespace(publication_authority=None, hubs=()))
    )
    machine_model_defaults.verify(worker, capture, interfaces)
    rungs = capture.model_defaults[0].rungs
    rungs[0].gpus = 2
    rungs.add().CopyFrom(rungs[0])
    rungs[1].gpus = 4
    machine_model_defaults.verify(worker, capture, interfaces)


def test_selection_counts_the_granted_pool_and_uncounted_rows_need_no_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rung = {
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
    worker = SimpleNamespace(
        options=SimpleNamespace(accelerator_backend="cuda"),
        host_facts=lambda: hostfacts.measure("cuda"),
        executions=SimpleNamespace(
            capture_root=lambda *_: "root", capture=lambda *_: {"model_defaults": [row]}
        ),
        lanes=SimpleNamespace(entries=()),
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
    row["rungs"] = [{**rung, "gpus": 4}, {**rung, "gpus": 2}]
    with pytest.raises(WorkspaceRefusal, match="no rung for this accelerator"):
        select(cast(Any, worker), "owner", call, target, {})
    worker.lanes = SimpleNamespace(entries=[0, 1])
    assert select(cast(Any, worker), "owner", call, target, {})[0]["gpus"] == 2
