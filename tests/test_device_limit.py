"""Unit checks of machine-authority propagation, not GPU allocation qualification."""

from types import SimpleNamespace

import pytest
from msgspec import UNSET

from cozy_runtime.internal.executor import Executor
from cozy_runtime.internal.seam import SeamError


def executor(rank=0, world=1):
    process = object.__new__(Executor)
    process.host = process
    process.device_limit_bytes = None
    process.memory = SimpleNamespace(limit_bytes=None)
    process.device_kind = "cuda"
    process.rank, process.world = rank, world
    return process


def test_authorization_is_numeric_only_and_cannot_raise_an_existing_machine_limit():
    process = executor()
    process._authorize_device_limit(UNSET)
    assert process.device_limit_bytes is None
    process._authorize_device_limit(8 << 30)
    process._authorize_device_limit(6 << 30)
    process._authorize_device_limit(24 << 30)
    assert process.device_limit_bytes == process.memory.limit_bytes == 6 << 30
    for invalid in (0, -1, True):
        with pytest.raises(SeamError, match="positive"):
            process._authorize_device_limit(invalid)


def test_each_rank_clamps_the_fraction_to_its_real_card_and_keeps_that_bound():
    process = executor(rank=1, world=2)
    process._authorize_device_limit(24 << 30)
    applied = []

    def properties(index):
        return SimpleNamespace(total_memory=12 << 30)

    torch = SimpleNamespace(
        cuda=SimpleNamespace(
            get_device_properties=properties,
            set_per_process_memory_fraction=lambda fraction, index: applied.append(
                (fraction, index)
            ),
        )
    )
    process._apply_device_limit(torch)
    assert applied == [(1.0, 1)]
    assert process.device_limit_bytes == process.memory.limit_bytes == 12 << 30
    process._authorize_device_limit(6 << 30)
    process._apply_device_limit(torch)
    assert applied[-1] == (0.5, 1)
