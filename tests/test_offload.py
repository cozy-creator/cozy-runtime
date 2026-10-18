"""Storages off the device and back (memory rungs 4-5, `internal/spill.py`), on real torch
through the executor's own dispatch mode. Releasing a storage and bringing it back before an op
reads it is the same on every device, so these legs run it on host tensors; the card legs are
in `test_weight_plane.py`."""

from __future__ import annotations

import gc
from collections.abc import Iterator
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from cozy_runtime.author import kernels  # noqa: E402
from cozy_runtime.internal import producers, spill  # noqa: E402
from cozy_runtime.internal.weights import _noting  # noqa: E402

CPU = torch.device("cpu")
FLOATS = 1 << 20  # 4 MiB of float32


@pytest.fixture(autouse=True)
def _pressed() -> Iterator[None]:
    """Nothing is off the device, and ops are noted from the test's start (a stage under
    pressure)."""
    spill.OFF.clear()
    spill.USED.clear()
    spill.PRESSED[0] = True
    yield
    spill.OFF.clear()
    spill.OUT[0] = 0
    spill.USED.clear()
    spill.NATIVE.clear()
    spill.PRESSED[0] = False


@pytest.mark.parametrize("disk", [False, True], ids=["host", "disk"])
def test_what_no_op_touched_this_stage_leaves_whole_and_comes_back_for_its_next_op(
    disk: bool,
) -> None:
    """An earlier stage's tensor leaves (to host memory, or to disk once the host can give no
    more), its storage released; one this stage touched, the op's own operand and one code
    outside the mode reads (`native`: a graph's input) stay. The next op that reads it brings
    it back first, and it reads the same values."""
    with _noting(torch, None, None):
        earlier = torch.arange(FLOATS, dtype=torch.float32)
        operand = torch.ones(FLOATS)
        live = torch.ones(FLOATS)
        graphed = torch.ones(FLOATS)
        spill.native(torch, graphed)
        spill.began()
        live.mul_(2)
        # no more than it may move: no other host tensor of this process is looked at
        need = 4 * FLOATS
        moved = spill.spill(torch, CPU, need, (operand,), 0 if disk else need, False, disk)
        assert moved == 4 * FLOATS
        assert earlier.untyped_storage().nbytes() == 0, "its bytes left with it"
        kept = (operand, live, graphed)
        assert all(t.untyped_storage().nbytes() == 4 * FLOATS for t in kept)
        off = spill.OFF[earlier.untyped_storage()._cdata]
        assert (off.file is not None) == disk and (off.host is None) == disk
        assert spill.TOTALS[5 if disk else 4] >= 4 * FLOATS
        got = earlier + 1  # an op touches it: back first
    assert torch.equal(got, torch.arange(FLOATS, dtype=torch.float32) + 1)
    assert earlier.untyped_storage().nbytes() == 4 * FLOATS and not spill.OFF


def test_a_copy_goes_with_its_last_tensor_and_the_rest_come_back_at_a_request() -> None:
    """A storage that left and that nothing holds any more takes its copy (its file) with it
    when the stage closes; one still held comes back as the next request begins."""
    with _noting(torch, None, None):
        dropped = torch.zeros(FLOATS)
        kept = torch.full((FLOATS,), 7.0)
        spill.began()
        spill.spill(torch, CPU, 2 * 4 * FLOATS, (), 0, mapped=False, disk=True)
    files = [off.file for off in spill.OFF.values()]
    assert len(files) == 2 and all(f is not None and not f.closed for f in files)
    del dropped
    gc.collect()
    spill.release(torch)
    assert len(spill.OFF) == 1 and sum(f is not None and f.closed for f in files) == 1
    assert spill.restore(torch, CPU, 1 << 30) == 4 * FLOATS
    assert not spill.OFF and torch.equal(kept, torch.full((FLOATS,), 7.0))
    spill.release(torch)
    assert not spill.PRESSED[0], "nothing off the device: ops are no longer noted"


def _left(disk: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """A tensor only kernels read (no op touched it this stage) taken out whole, and a view of
    it at an offset: once its storage is emptied, that view's pointer is not even null."""
    with _noting(torch, None, None):
        statistics = torch.arange(FLOATS, dtype=torch.float32)
        view = statistics[1:]
        spill.began()
        spill.spill(torch, CPU, 4 * FLOATS, (), 0 if disk else 4 * FLOATS, False, disk)
    assert statistics.untyped_storage().nbytes() == 0 and spill.OUT[0] == 1
    return statistics, view


def _read(tensor: torch.Tensor) -> torch.Tensor:
    """A native kernel: it reads the bytes behind the pointer, never through torch's dispatch."""
    assert tensor.untyped_storage().nbytes() >= tensor.numel() * tensor.element_size()
    return tensor.clone()


@pytest.mark.parametrize("disk", [False, True], ids=["host", "disk"])
def test_a_kernel_launch_brings_back_what_the_ladder_took_out(disk: bool) -> None:
    """Run 11343 (H3 on a 12 GB RTX 5070): the ladder took a tensor only kernels read out of the
    device whole, and a Triton launch was handed a pointer with nothing behind it ('Pointer
    argument cannot be accessed from Triton'). Every launch outside torch's dispatch reads its
    operands through `readable`, which brings them back first."""
    _, view = _left(disk)
    with kernels.readable(view) as (operand,):
        got = _read(operand)
    assert torch.equal(got, torch.arange(1, FLOATS, dtype=torch.float32))
    assert spill.OUT[0] == 0 and not spill.OFF


def test_a_producers_kernel_call_brings_back_what_the_ladder_took_out() -> None:
    """A package's attention producer calls its kernel module (Kitchen's `sol_attn_chunked`)
    with state it keeps across steps: the module Runtime hands it brings each call's tensors
    back first, nested in tuples and keywords too."""
    _, view = _left(False)
    kitchen = producers._native(SimpleNamespace(attend=lambda scales, kmean: _read(kmean)))
    got = kitchen.attend((torch.ones(4),), kmean=view)  # type: ignore[attr-defined]
    assert torch.equal(got, torch.arange(1, FLOATS, dtype=torch.float32)) and spill.OUT[0] == 0


def test_a_tensor_kernels_read_counts_as_read() -> None:
    """Under pressure a kernel's read counts as an op's does: a tensor only kernels read is not
    the first to leave."""
    with _noting(torch, None, None):
        kept = torch.arange(FLOATS, dtype=torch.float32)
        spill.began()
    with kernels.readable(kept) as (operand,):
        _read(operand)
    with _noting(torch, None, None):
        assert spill.spill(torch, CPU, 4 * FLOATS, (), 4 * FLOATS, False, False) == 0
    assert kept.untyped_storage().nbytes() == 4 * FLOATS and not spill.OFF
