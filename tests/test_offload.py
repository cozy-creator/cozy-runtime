"""Storages off the device and back (memory rungs 4-5, `internal/spill.py`), on real torch
through the executor's own dispatch mode. Releasing a storage and bringing it back before an op
reads it is the same on every device, so these legs run it on host tensors; the card legs are
in `test_weight_plane.py`."""

from __future__ import annotations

import gc
from collections.abc import Iterator

import pytest

torch = pytest.importorskip("torch")

from cozy_runtime.internal import spill  # noqa: E402
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
    spill.USED.clear()
    spill.PRESSED[0] = False


@pytest.mark.parametrize("disk", [False, True], ids=["host", "disk"])
def test_what_no_op_touched_this_stage_leaves_whole_and_comes_back_for_its_next_op(
    disk: bool,
) -> None:
    """An earlier stage's tensor leaves (to host memory, or to disk once the host can give no
    more), its storage released; one this stage touched, and the op's own operand, stay. The
    next op that reads it brings it back first, and it reads the same values."""
    with _noting(torch, None, None):
        earlier = torch.arange(FLOATS, dtype=torch.float32)
        operand = torch.ones(FLOATS)
        live = torch.ones(FLOATS)
        spill.began()
        live.mul_(2)
        # as much as the two it may move: no other host tensor of this process is looked at
        need = 2 * 4 * FLOATS
        moved = spill.spill(torch, CPU, need, (operand,), 0 if disk else need, False, disk)
        assert moved == 4 * FLOATS
        assert earlier.untyped_storage().nbytes() == 0, "its bytes left with it"
        assert operand.untyped_storage().nbytes() == live.untyped_storage().nbytes() == 4 * FLOATS
        off = spill.OFF[earlier.untyped_storage()._cdata]
        assert (off.file is not None) == disk and (off.host is None) == disk
        assert spill.TIERS[2 if disk else 1] >= 4 * FLOATS
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
