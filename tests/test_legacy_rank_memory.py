"""Legacy process-birth reports index sealed entries by local rank."""

import os
from types import SimpleNamespace
from typing import Any, cast

import pytest

from cozy_runtime.internal import accel, proctree
from cozy_runtime.internal.worker.lanes import DeviceLane
from cozy_runtime.internal.worker.session import Worker


def test_legacy_memory_report_on_nonzero_envelope_ordinal(monkeypatch: pytest.MonkeyPatch) -> None:
    lane = DeviceLane("lane-3", (3,), "GPU-sealed-entry", worker_pid=os.getpid())
    row = lane.row("tenant")
    row.supervision = cast(
        Any,
        SimpleNamespace(
            current=SimpleNamespace(
                pid=321,
                epoch=1,
                hello={"memory": []},
                poisoned=False,
                alive=lambda: True,
                scope=SimpleNamespace(),
            )
        ),
    )
    monkeypatch.setattr(proctree, "process_scope_contains", lambda *_: True)
    monkeypatch.setattr(proctree, "process_state", lambda _: "R")
    monkeypatch.setattr(accel, "host_backend_family", lambda: "cuda")
    seen = []

    def measured(
        pids: set[int], kind: str, devices: tuple[str, ...], *, allow_subprocess: bool
    ) -> dict[str, dict[int, accel.ProcessMemory]]:
        seen.append((pids, kind, devices, allow_subprocess))
        return {"GPU-sealed-entry": {321: accel.ProcessMemory("present", 1234)}}

    monkeypatch.setattr(accel, "process_memories_by_device", measured)
    Worker._observe_memory(lane, row, {"processes": [{"rank": 0, "pid": 321, "started_ticks": 1}]})
    assert seen == [({321}, "cuda", ("GPU-sealed-entry",), False)]
    assert row.held == {3: 1234}
