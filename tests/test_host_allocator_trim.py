"""Physical relief follows logical eviction without reclaiming live tensor storage."""

import ctypes
import os
import sys
import weakref
from types import SimpleNamespace
from typing import Any, cast

import pytest

from cozy_runtime.internal import executor, fill, memory, proctree, tiers


def test_non_linux_trim_does_not_load_libc(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "win32")

    def library(*args: object) -> object:
        pytest.fail("unsupported platform must not load libc")

    monkeypatch.setattr(ctypes, "CDLL", library)
    assert not proctree.trim_host_allocator()


@pytest.mark.parametrize("missing", [False, True])
def test_allocator_trim_is_optional(missing: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    def library(*args: object) -> object:
        if missing:
            raise OSError("no process library")
        return SimpleNamespace()

    monkeypatch.setattr(ctypes, "CDLL", library)
    assert not proctree.trim_host_allocator()


def test_allocator_trim_uses_size_t_and_int(monkeypatch: pytest.MonkeyPatch) -> None:
    class Trim:
        argtypes: list[type[ctypes.c_size_t]] = []
        restype: object = None

        def __call__(self, pad: int) -> int:
            assert pad == 0
            assert self.argtypes == [ctypes.c_size_t]
            assert self.restype is ctypes.c_int
            return 1

    trim = Trim()
    monkeypatch.setattr(ctypes, "CDLL", lambda _: SimpleNamespace(malloc_trim=trim))
    assert proctree.trim_host_allocator()


@pytest.mark.parametrize("released", [0, 123])
def test_executor_trims_after_reclaim_and_measures_actual_relief(
    released: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[str] = []
    readings = iter([1000, 700])

    def reclaim(nbytes: int) -> int:
        assert nbytes == 500
        events.append("reclaim")
        return released

    def trim() -> bool:
        events.append("trim")
        return True

    def rss(pid: int) -> int:
        events.append("rss")
        return next(readings)

    monkeypatch.setattr(proctree, "rss_bytes", rss)
    monkeypatch.setattr(proctree, "trim_host_allocator", trim)
    owner = SimpleNamespace(
        poisoned=False, world=1,
        _synchronize=lambda: events.append("sync"),
        planner=SimpleNamespace(
            reclaim_host=reclaim, host_memory=lambda: {}, host_reclaim_status=lambda: {}
        ),
        probe=lambda: {"ok": True},
    )
    reply = executor.Executor.reclaim_host(cast(Any, owner), 500)
    assert events == ["rss", "sync", "reclaim", "trim", "rss"]
    assert reply["managed_released_bytes"] == released
    assert reply["rss_released_bytes"] == 300


@pytest.mark.skipif(sys.platform != "linux", reason="Linux RSS and optional glibc trim")
def test_real_64_mib_reclaim_trims_free_pages_and_preserves_live_aliases() -> None:
    torch = pytest.importorskip("torch")
    if not hasattr(ctypes.CDLL(None), "malloc_trim"):
        pytest.skip("allocator has no malloc_trim")
    torch.set_num_threads(2)
    root = torch.nn.Module()
    # Keep another owned allocation and its view alive across logical and physical release.
    retained = torch.ones(16 * 1024 * 1024, dtype=torch.uint8)
    alias = retained[1::2]
    pointer = alias.untyped_storage().data_ptr()
    proctree.trim_host_allocator()
    before = proctree.rss_bytes(os.getpid())
    for index in range(128):
        root.register_parameter(str(index), torch.nn.Parameter(torch.ones(128 * 1024)))
    refs = [weakref.ref(weight) for weight in root.parameters()]
    backend = fill.StreamingFillBackend.__new__(fill.StreamingFillBackend)
    backend.torch = torch
    backend.components = {"model": root}
    backend.parked = {}
    backend.replaced = {}
    backend.stage_log = []
    size = 64 * 1024 * 1024
    backend.component_memory = {"model": fill.ComponentMemory(destinations=size)}
    planner = memory.Planner(torch, torch.device("cpu"))
    held = planner.adopt("model", "model", root, None, tiers.HOST)
    held.part.store = backend
    allocated = proctree.rss_bytes(os.getpid())
    assert planner.reclaim_host(size) == size
    assert all(ref() is None for ref in refs)
    assert planner.host_memory()["managed_host_bytes"] == 0
    assert all(weight.is_meta for weight in held.part.root.parameters())
    logical = proctree.rss_bytes(os.getpid())
    proctree.trim_host_allocator()
    trimmed = proctree.rss_bytes(os.getpid())
    # Some allocators return pages during free; either route must return real residency.
    assert allocated - trimmed >= size // 2
    if logical - before >= size // 2:
        assert logical - trimmed >= size // 2
    assert alias.untyped_storage().data_ptr() == pointer
    assert torch.equal(alias, torch.ones_like(alias))
    assert int(retained.sum()) == retained.numel()


def test_denied_local_copy_without_released_backing_does_not_trim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    planner = memory.Planner(torch, torch.device("cpu"), host_room=lambda *args: False)
    monkeypatch.setattr(proctree, "trim_host_allocator", lambda: pytest.fail("no backing released"))
    with planner._host_copy(1) as admitted:
        assert not admitted


@pytest.mark.skipif(sys.platform != "linux", reason="Linux RSS and optional glibc trim")
def test_local_pressure_retries_observed_room_after_trimming_64_mib(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    if not hasattr(ctypes.CDLL(None), "malloc_trim"):
        pytest.skip("allocator has no malloc_trim")
    torch.set_num_threads(2)
    planner = memory.Planner(torch, torch.device("cpu"))
    held = planner.adopt("idle", "weights", torch.nn.Linear(1, 1), None, tiers.DEVICE)
    proctree.trim_host_allocator()
    baseline = proctree.rss_bytes(os.getpid())
    held.part.common.host = {(index, "weight"): torch.ones(128 * 1024) for index in range(128)}
    # The live trailing allocation prevents automatic contraction of the heap tail.
    barrier = torch.full((128 * 1024,), 7.0)
    alias = barrier[1::2]
    pointer = alias.untyped_storage().data_ptr()
    refs = [weakref.ref(tensor) for tensor in held.part.common.host.values()]
    # An explicit test envelope uses actual RSS at both admission checks, never the
    # logical bytes dropped or malloc_trim's return value as available memory.
    limit = baseline + 16 * 1024 * 1024
    checks: list[tuple[int, bool]] = []
    trimmed: list[int] = []
    original_trim = proctree.trim_host_allocator

    def trim() -> bool:
        trimmed.append(proctree.rss_bytes(os.getpid()))
        return original_trim()

    def room(token: str, nbytes: int, release: bool) -> bool:
        if release:
            return True
        rss = proctree.rss_bytes(os.getpid())
        admitted = nbytes <= limit - rss
        checks.append((rss, admitted))
        return admitted

    planner.host_room = room
    monkeypatch.setattr(proctree, "trim_host_allocator", trim)
    with planner._host_copy(8 * 1024 * 1024) as admitted:
        assert admitted
    assert len(checks) == 2 and not checks[0][1] and checks[1][1]
    assert len(trimmed) == 1
    assert checks[0][0] - checks[1][0] >= 32 * 1024 * 1024
    assert all(ref() is None for ref in refs)
    assert planner.host_memory()["managed_host_bytes"] == 0
    assert alias.untyped_storage().data_ptr() == pointer
    assert bool((alias == 7).all()) and bool((barrier == 7).all())
