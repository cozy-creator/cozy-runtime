"""Failure observations survive the real warm, control and durable history boundaries."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from cozy_runtime.internal import prepare_diagnostics as diagnostics


def test_readers_fail_independently_without_inventing_zero() -> None:
    def unavailable() -> dict[str, int]:
        raise RuntimeError("private driver details")

    result = diagnostics.failure_memory(unavailable, lambda: {"grant_bytes": 123})
    assert result == {
        "sample": "after_failure_before_retirement",
        "allocation_unavailable": "RuntimeError",
        "residency": {"grant_bytes": 123},
    }
    inverse = diagnostics.failure_memory(lambda: {"allocated_bytes": 0}, unavailable)
    assert inverse["allocation"] == {"allocated_bytes": 0}
    assert "residency" not in inverse


def test_real_warm_refusal_reaches_control_socket_and_durable_phase(tmp_path: Path) -> None:
    script = r"""
import json, socket, sys
from pathlib import Path
from types import SimpleNamespace
import torch
from cozy_runtime import canonical_json
from cozy_runtime.author import Context, Model
from cozy_runtime.author._context import Device
from cozy_runtime.internal import memory, tiers, prepare_diagnostics
from cozy_runtime.internal.executor import Executor, _send_reply
from cozy_runtime.internal.seam import Channel
from cozy_runtime.internal.warm import WarmFailed, warm_generation
from cozy_runtime.internal.worker.attempts import AttemptRefusal
from cozy_runtime.internal.worker.session import Worker, _progress_document
from cozy_runtime.internal.worker.workspace import Workspace
from cozy_runtime.internal.worker.workspace_executions import Executions
from cozy_runtime.protocol import worker_pb2 as pb
from test_machine_execution import offer

planner = memory.Planner(torch, torch.device("cpu"))
root = torch.nn.Linear(2, 2)
planner.adopt("generation", "network", root, None, tiers.HOST)
before = root.weight.detach().clone()
class FailingWarm(Model[None]):
    def warm(self, ctx: Context) -> None:
        root(torch.ones(2))
        raise memory.MemoryRefusal("device_out_of_memory", "test warm capacity", 17, 0)

try:
    warm_generation(FailingWarm(), device=Device("cpu"), cancel=lambda: False)
except WarmFailed as exc:
    assert isinstance(exc.__cause__, memory.MemoryRefusal)
    observer = SimpleNamespace(device_kind="cpu", _device_ready=True, torch=torch,
        host=SimpleNamespace(planner=planner), construction="generation")
    evidence = Executor._prepare_memory_evidence(observer)
    assert evidence["allocation"] == {}  # a CPU failure cannot initialize CUDA to measure it
    observer.device_kind, observer._device_ready = "cuda", False
    assert Executor._prepare_memory_evidence(observer)["allocation"] == {}
    reply = {"reply": "load", "ok": False, "code": exc.__cause__.code,
             "detail": str(exc), "preparation_memory": evidence}
else:
    raise AssertionError("warm must refuse")
assert torch.equal(before, root.weight)
assert not torch.cuda.is_initialized()
left, right = socket.socketpair()
with left, right:
    _send_reply(Channel(left), reply)
    received = Channel(right).recv(timeout=2)
assert received["preparation_memory"] == evidence

workspace = Workspace(Path(sys.argv[1]) / "store")
executions = Executions(workspace)
executions.submit("owner", "root", b"c" * 32, offer("root"),
                  expected_execution_workspace_id=executions.workspace_id)
def journal(request, ordinal, frame):
    executions.progress("owner", request, ordinal, _progress_document(frame))
worker = SimpleNamespace(machine_calls=None, _journal_row=journal)
refused = AttemptRefusal(received["code"], received["detail"],
    cause=pb.CAUSE_CODE_CONSTRAINT_INFEASIBLE,
    preparation_memory=received["preparation_memory"])
try:
    with Worker._call_phase(worker, "root", 1, "Loading model weights"):
        raise refused
except AttemptRefusal as exc:
    assert exc is refused
history = [canonical_json.decode(row.body) for row in executions.events("owner", "root").events]
phase = next(row["payload"] for row in history
             if row.get("payload", {}).get("name") == "Loading model weights")
assert phase["fields"]["completed"] is False
assert "preparation_memory" not in phase["fields"]  # keep the scalar observation contract
assert phase["preparation_memory"] == evidence
assert evidence["residency"]["evicted"]["generation/network"] == 24
assert not torch.cuda.is_initialized()
print(json.dumps({"code": refused.code, "phase": phase}))
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        capture_output=True,
        text=True,
        check=True,
    )
    evidence = json.loads(result.stdout)
    assert evidence["code"] == "device_out_of_memory"
    assert evidence["phase"]["preparation_memory"]["allocation"] == {}


def test_allocator_snapshot_uses_planner_device() -> None:
    script = r"""
from types import SimpleNamespace
from cozy_runtime.internal.executor import Executor
selected = object()
seen = []
def record(device):
    assert device is selected
    seen.append(device)
    return 0
cuda = SimpleNamespace(is_initialized=lambda: True,
    mem_get_info=lambda device: (record(device), 100),
    memory_allocated=record, memory_reserved=record)
planner = SimpleNamespace(device=selected, document=lambda key: {"grant_bytes": 100})
owner = SimpleNamespace(device_kind="cuda", _device_ready=True,
    torch=SimpleNamespace(cuda=cuda), host=SimpleNamespace(planner=planner),
    construction="generation")
evidence = Executor._prepare_memory_evidence(owner)
assert evidence["allocation"] == {"allocated_bytes": 0, "reserved_bytes": 0,
    "driver_free_bytes": 0, "driver_total_bytes": 100}
assert len(seen) == 3
"""
    subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True)


def test_oversize_optional_evidence_keeps_original_refusal() -> None:
    script = r"""
import socket
from cozy_runtime.internal.executor import _send_reply
from cozy_runtime.internal.seam import Channel
left, right = socket.socketpair()
with left, right:
    _send_reply(Channel(left), {"reply": "load", "ok": False,
        "code": "device_out_of_memory", "detail": "original refusal",
        "preparation_memory": {"residency": {"log": ["x" * 100000]}}})
    reply = Channel(right).recv(timeout=2)
assert reply["code"] == "device_out_of_memory"
assert reply["detail"] == "original refusal"
assert "preparation_memory" not in reply
assert reply["observation_caps"]["dropped_preparation_memory_for_frame"] == 1
"""
    subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True)
