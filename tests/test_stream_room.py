"""A stream made on a full card asks other executors' idle residents for room (cr-209).

Real GPU, real driver, varena's real ledger. Another process holds an idle model (as the smoke's
five packages did beside character-describer, run 14188); this one holds every other byte. Made
as written, the stream fails as 14188's did (`cuStreamCreateWithPriority`). Made through
`streams`, the memory asks the resident for room and the stream works.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("tensorfs")

from tensorfs.varena import Ledger, cuts  # noqa: E402
from tensorfs.varena.torch import LedgerGrant, Memory  # noqa: E402

from cozy_runtime.internal import streams  # noqa: E402

pytestmark = pytest.mark.real_gpu

RESIDENT = Path(__file__).parent / "testdata/stream_room_resident.py"


def _fill() -> list[object]:
    """Live tensors over every byte the allocator gets from the driver, halving down to its
    smallest segment."""
    held: list[object] = []
    size = torch.cuda.mem_get_info()[0]
    while size >= 1:
        try:
            held.append(torch.empty(size, dtype=torch.uint8, device="cuda"))
        except torch.OutOfMemoryError:
            size //= 2
    return held


def _resident(table: Ledger, pid: int) -> int:
    return sum(row.resident for row in table.snapshot() if row.pid == pid and row.state == "live")


def test_a_stream_on_a_full_card_takes_room_from_an_idle_resident(tmp_path: Path) -> None:
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")
    torch.cuda.init()
    free, total = torch.cuda.mem_get_info()
    table = Ledger.create(str(torch.cuda.get_device_properties(0).uuid), total)
    events = table.events_fd
    assert events is not None
    child = subprocess.Popen(
        [sys.executable, str(RESIDENT), str(table.fd), str(events), str(free // 4), str(tmp_path)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
        pass_fds=(table.fd, events),
    )
    held: list[object] = []
    try:
        assert child.stdout is not None and child.stdout.readline().startswith("resident ")
        memory = Memory.open(torch.device("cuda", 0), LedgerGrant(0, table.fd, events))
        before = _resident(table, child.pid)
        assert before >= free // 4
        held = _fill()
        filled = torch.cuda.mem_get_info()[0]
        try:
            torch.cuda.Stream()
        except torch.AcceleratorError as exc:
            assert "out of memory" in str(exc)
        else:
            pytest.skip(f"this driver made a stream in {torch.cuda.mem_get_info()[0]} B free")
        streams.install(cuts.launch)
        try:
            side = streams.stream(torch)
        finally:
            streams.install(None)
        with torch.cuda.stream(side):
            torch.zeros(1, device="cuda").add_(1)
        side.synchronize()
        after, gave = torch.cuda.mem_get_info()[0], before - _resident(table, child.pid)
        print(f"free {filled} B filled, {after} B after; the resident gave {gave} B")
        assert after > filled and gave > 0, "the idle resident gave the room"
        assert memory.cuda
    finally:
        held.clear()
        torch.cuda.empty_cache()
        if child.stdin is not None:
            child.stdin.close()
        child.wait()
