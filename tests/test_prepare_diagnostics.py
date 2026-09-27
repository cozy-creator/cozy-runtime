"""Preparation diagnostics retain causal locations without inventing measurements."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from cozy_runtime.internal import prepare_diagnostics as diagnostics


def test_real_chained_warm_retains_causal_frames_without_values() -> None:
    if importlib.util.find_spec("torch") is None:
        pytest.skip("the real warm boundary requires Torch")
    # The warm/derive boundary belongs to an executor, not the test runner that
    # also proves worker metadata preparation never imports derive.
    script = """
import json, socket
from cozy_runtime.author import Context, Model
from cozy_runtime.author._context import Device
from cozy_runtime.internal import prepare_diagnostics as diagnostics
from cozy_runtime.internal.executor import _send_reply
from cozy_runtime.internal.seam import Channel
from cozy_runtime.internal.warm import WarmFailed, warm_generation

class ChainedWarm(Model[None]):
    def warm(self, ctx: Context) -> None:
        ctx.raise_if_cancelled()
        try:
            self.collective()
        except OSError as exc:
            raise RuntimeError("collective failed") from exc

    @staticmethod
    def collective() -> None:
        secret = "not-for-diagnostics"
        exc = OSError(secret)
        exc.add_note("private diagnostic note")
        raise exc

try:
    warm_generation(ChainedWarm(), device=Device("cpu"), cancel=lambda: False)
except WarmFailed as exc:
    trace = diagnostics.exception_trace(exc)
    detail = str(exc)
else:
    raise AssertionError("the warm must fail")
left, right = socket.socketpair()
with left, right:
    _send_reply(Channel(left), {
        "reply": "load", "ok": False, "code": "warm_failed", "detail": detail,
        "traceback": trace, "envelope": "x" * 100000,
    })
    reply = Channel(right).recv(timeout=2)
assert reply is not None and reply["code"] == "warm_failed"
print(json.dumps(reply))
"""
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True
    )
    reply = json.loads(result.stdout)
    retained = "\n".join(diagnostics.activity_lines(reply["traceback"]))
    assert retained.index("OSError") < retained.index("RuntimeError") < retained.index("WarmFailed")
    assert "in collective" in retained and "in warm_generation" in retained
    assert "<string>:" in retained
    assert "not-for-diagnostics" not in retained
    assert "private diagnostic note" not in retained
    assert "secret =" not in retained and str(__file__) not in retained


def test_deep_and_cyclic_exceptions_stay_bounded() -> None:
    error = RuntimeError("message" * 100000)
    for _ in range(100):
        parent = RuntimeError("parent")
        parent.__cause__ = error
        error = parent
    trace = diagnostics.exception_trace(error)
    assert "earlier exceptions omitted" in trace
    assert len(trace.splitlines()) <= diagnostics.MAX_TRACE_LINES
    assert len(trace) <= diagnostics.MAX_TRACE_LINES * (diagnostics.MAX_LINE_CHARS + 1)
    error.__cause__ = error
    assert diagnostics.exception_trace(error).count("RuntimeError") == 1
    assert len(diagnostics.activity_lines("\n".join(["x" * 1000] * 1000))) <= 40
    assert all(len(line) <= 160 for line in diagnostics.activity_lines("x" * 100000))


def test_false_valued_cause_is_not_skipped() -> None:
    class FalseCause(ValueError):
        def __bool__(self) -> bool:
            return False

    error = RuntimeError("outer")
    error.__cause__ = FalseCause("inner")
    assert "FalseCause" in diagnostics.exception_trace(error)


def test_missing_memory_is_unknown_and_explicit_zero_is_preserved() -> None:
    for reply in (
        {},
        {"peak_allocator_bytes": None, "allocator_bytes": -1, "device_free_bytes": True},
    ):
        summary = diagnostics.memory_summary(reply)
        assert summary.count("unknown") == 3
        assert "0 B" not in summary and "-1 B" not in summary
    summary = diagnostics.memory_summary({"peak_allocator_bytes": 0, "device_free_bytes": 1024})
    assert "peak allocator: 0 B" in summary and "device free: 1024 B" in summary
    assert "allocated: unknown" in summary


def test_startup_observation_does_not_import_torch_or_initialize_cuda() -> None:
    script = """
import json, sys
from cozy_runtime.internal.prepare_diagnostics import startup_facts
facts = startup_facts(rank=2, world=4, device_kind='cuda')
assert 'torch' not in sys.modules
assert facts['cuda_initialized'] is False and 'selected_device' not in facts
print(json.dumps(facts))
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        env={
            **os.environ,
            "CUDA_VISIBLE_DEVICES": "0,1,2,3",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "NCCL_NVLS_ENABLE": "0",
            "AUTH_TOKEN": "never-print-this",
        },
        capture_output=True,
        text=True,
        check=True,
    )
    facts = json.loads(result.stdout)
    assert facts["rank"] == 2 and facts["world"] == 4
    assert facts["environment"]["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True"
    assert facts["environment"]["NCCL_NVLS_ENABLE"] == "0"
    assert set(facts["environment"]) == set(diagnostics.STARTUP_ENV)
    assert "AUTH_TOKEN" not in result.stdout and "never-print-this" not in result.stdout


@pytest.mark.skipif(sys.platform != "linux", reason="Linux protection observation")
def test_startup_reports_actual_protection_without_changing_it() -> None:
    script = """
import json
from cozy_runtime.internal import proctree
from cozy_runtime.internal.prepare_diagnostics import startup_facts
proctree.seal_no_new_privs()
proctree.deny_process_inspection()
facts = startup_facts(rank=0, world=1, device_kind='cpu')
assert facts['no_new_privs'] == 1 and facts['dumpable'] == 0
assert proctree.own_protection()['dumpable'] == 0
print(json.dumps(facts))
"""
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True
    )
    facts = json.loads(result.stdout)
    assert facts["uid"] == os.getuid() and facts["gid"] == os.getgid()


def test_executor_boundary_emits_startup_facts_before_initialization(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    script = """
import socket, sys
from pathlib import Path
import torch
from cozy_runtime.internal.executor import Executor
from cozy_runtime.internal.seam import Channel
left, right = socket.socketpair()
with left, right:
    executor = Executor(Channel(left), Path(sys.argv[1]), rank=2, world=4)
    executor.device_kind = 'cpu'
    assert not torch.cuda.is_initialized()
    assert executor._initialize_device(torch) is None
    assert executor._initialize_device(torch) is None
    assert not torch.cuda.is_initialized()
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)], capture_output=True, text=True, check=True
    )
    lines = [line for line in result.stderr.splitlines() if "[executor] prepare startup:" in line]
    assert len(lines) == 1
    facts = json.loads(lines[0].split(": ", 1)[1])
    assert facts["rank"] == 2 and facts["world"] == 4
    assert facts["cuda_initialized"] is False and "selected_device" not in facts
