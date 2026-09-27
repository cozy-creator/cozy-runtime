"""An upload on a TensorFS that predates it refuses when admitted, before any byte moves."""

from __future__ import annotations

import queue
import types
from pathlib import Path
from weakref import WeakValueDictionary

import pytest
import tensorfs

from cozy_runtime.internal import fill, storage_admission
from cozy_runtime.internal.worker import upload_plan
from cozy_runtime.internal.worker.source_calls import SourceCalls
from cozy_runtime.internal.worker.workspace import Workspace
from cozy_runtime.protocol import worker_pb2 as pb
from test_bounded_upload import upload
from upload_standins import Hub


def older_tensorfs() -> types.ModuleType:
    """The installed TensorFS with only the upload bindings absent, as 0.3.59 shipped it."""
    module = types.ModuleType("tensorfs")
    for name in dir(tensorfs):
        if name not in upload_plan.BINDINGS:
            setattr(module, name, getattr(tensorfs, name))
    return module


def test_upload_refuses_early_on_a_tensorfs_without_its_bindings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(storage_admission, "_reclaimers", WeakValueDictionary())
    monkeypatch.setattr(fill, "tensorfs_module", older_tensorfs)
    workspace = Workspace(tmp_path / "store")
    messages: queue.Queue[pb.WorkerFrame] = queue.Queue()
    calls = SourceCalls(
        workspace,
        lambda: "owner",
        messages.put,
        lambda _: True,
        endpoints={"huggingface": "http://127.0.0.1:9"},
        publication=lambda *_: Hub().client(),
    )
    command = upload(workspace, "older-tensorfs")
    command.phase = pb.NATIVE_SOURCE_PHASE_RESOLVE
    calls.handle(command)
    status = messages.get(timeout=60).native_source_status
    assert status.state == pb.NATIVE_SOURCE_STATE_FAILED
    assert status.safe_code == "tensorfs_capability_missing"
    assert status.safe_detail == "upload_huggingface needs tensorfs >= 0.3.60"
    assert calls.failure_detail(command.service_id) == status.safe_detail
    assert calls.processes.get(command.service_id) is None, "a native child was launched"
