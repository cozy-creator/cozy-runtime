"""The local GEMM tuner (xs-065 change 2): what it keeps, where, and what a call reads.

On the CPU: the verdict over real timing samples, the machine's table on a real filesystem (an
identity that moved reads nothing), what a call reads and counts, and the executor's idle
hand-off. On a GPU (`real_gpu`): one real shape measured, every config it keeps bit-identical to
the rule's and faster than it.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

from cozy_runtime.internal import int8_linear, kernel_cache, prebuilt, tune
from cozy_runtime.internal.tune import Identity, Table, Unit

CARD = Identity("sha256:kernel", "mma.sync", 89, "NVIDIA L4", 58, 50331648, "580.95.05")
KEY = ("int8", False, 154, 1280, 2048)


@pytest.fixture(autouse=True)
def fresh(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    monkeypatch.setattr(kernel_cache, "_MACHINE_STORE", [kernel_cache.Store(tmp_path / "u0")])
    for state in ("_OPS", "_TABLES", "_STREAMS", "_PENDING", "_SKIPPED"):
        monkeypatch.setattr(tune, state, {})
    monkeypatch.setattr(tune, "_BUSY", [0])
    monkeypatch.setattr(tune, "_SPENT", [0.0])
    monkeypatch.setattr(tune, "_STOPPED", [False])
    monkeypatch.setattr(tune, "_NEIGHBOURS", [])
    monkeypatch.setattr(tune, "_THREAD", [])
    yield tmp_path / "u0"
    tune.stop()
    for thread in tune._THREAD:
        thread.join(timeout=10)


def test_a_config_serves_only_beyond_both_spreads_and_never_faster_than_the_card() -> None:
    rule = [100.0, 101.0, 99.0, 100.5, 100.2]
    clear = tune.verdict({-1: rule, 6: [80.0, 81.0, 79.5, 80.2, 80.4]}, 10.0, {}, 4)
    assert clear.config == 6 and clear.gain() == pytest.approx(100.2 / 80.2)
    noise = tune.verdict({-1: rule, 6: [99.8, 99.0, 100.1, 99.6, 99.7]}, 10.0, {}, 4)
    assert noise.config is None and noise.gain() == 1.0
    impossible = tune.verdict({-1: rule, 2: [1.0, 1.1, 0.9, 1.0, 1.0]}, 10.0, {}, 4)
    assert impossible.config is None and impossible.implausible == [2]


def test_the_table_is_this_machines_and_a_moved_identity_reads_the_rule(fresh: Path) -> None:
    table = Table(CARD, tune._path(CARD))
    assert table.path is not None and table.path.is_relative_to(fresh / "tuned")
    table.merge({tune.name(KEY): Unit(config=6, rule_us=40.0, configs={"6": (30.0, 0.5)})})
    tune._keep(table)
    assert tune._read(table.path, CARD) == table.units
    moved = Identity("sha256:kernel", "mma.sync", 89, "NVIDIA L4", 58, 50331648, "590.10.01")
    assert tune._path(moved) != table.path and tune._read(table.path, moved) == {}
    table.path.write_text("{torn")
    assert tune._read(table.path, CARD) == {}


def test_a_call_reads_its_config_and_counts_a_shape_still_unmeasured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    table = Table(CARD, None)
    tune._TABLES[0] = table
    assert tune.config(0, KEY) is None and tune.config(0, KEY) is None
    assert tune._PENDING[0] == {KEY: 2}
    slow = ("int8", False, 4096, 1280, 1280)
    table.merge({tune.name(KEY): Unit(config=6), tune.name(slow): Unit()})
    tune._PENDING[0].clear()
    assert tune.config(0, KEY) == 6 and tune.config(0, slow) is None
    assert tune._PENDING[0] == {}  # measured: the rule serves it and nothing waits
    assert tune.config(1, KEY) is None and 1 not in tune._PENDING  # no library on device 1
    # LM shapes (M varies by prompt) stop being noted at the cap; counted ones keep counting.
    monkeypatch.setattr(tune, "PENDING_CAP", 2)
    for m in range(1, 6):
        tune.config(0, ("int8", False, m, 1280, 1280))
    assert len(tune._PENDING[0]) == 2


def test_an_idle_executor_tunes_what_its_calls_ran_and_keeps_no_failure() -> None:
    """The hand-off itself: a command's end starts the tuner, which waits while a neighbour on
    the GPU runs a request, then tries the shape. Here it has no library to measure with: the
    shape keeps the rule in this process, and the machine keeps nothing."""
    table = Table(CARD, tune._path(CARD))
    tune._TABLES[0] = table
    neighbour = threading.Event()
    neighbour.set()
    tune.watch(neighbour.is_set)
    with tune.command():
        assert tune.config(0, KEY) is None
        assert not tune._THREAD
    assert tune._THREAD
    time.sleep(0.5)
    assert not tune._SKIPPED and tune._PENDING[0] == {KEY: 1}  # a neighbour runs: it waits
    neighbour.clear()
    for _ in range(100):
        if tune._SKIPPED:
            break
        time.sleep(0.1)
    assert tune._SKIPPED[KEY].startswith("KeyError") and not tune._PENDING[0]
    assert not table.units and table.path is not None and not table.path.exists()
    assert tune.config(0, KEY) is None and not tune._PENDING[0]  # never noted again here


def test_a_command_waits_only_for_the_round_in_flight() -> None:
    entered, release = threading.Event(), threading.Event()

    def round_() -> None:
        with tune._DEVICE:
            entered.set()
            release.wait(5)

    held = threading.Thread(target=round_)
    held.start()
    entered.wait(5)
    started = threading.Event()

    def run() -> None:
        with tune.command():
            started.set()

    command = threading.Thread(target=run)
    command.start()
    assert not started.wait(0.2) and tune._busy()  # the round finishes first
    release.set()
    assert started.wait(5)
    command.join(5)
    held.join(5)
    assert not tune._busy()


@pytest.mark.real_gpu
def test_on_this_gpu_a_kept_config_is_bit_identical_and_faster() -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("a CUDA card")
    if int8_linear.native(torch, 0) is None:
        pytest.skip("this wheel serves no prebuilt 8-bit op here")
    ops = int8_linear.library(0)
    tune.attach(ops, prebuilt.artifact())
    unit = tune._unit(ops, tune._STREAMS[0], ("int8", False, 154, 1280, 2048))
    assert isinstance(unit, Unit) and unit.rounds == tune.ROUNDS and unit.configs, unit
    if unit.config is not None:
        assert str(unit.config) in unit.configs and unit.gain() > 1.0
    for choice, why in unit.refused.items():
        assert int(choice) not in map(int, unit.configs), why
