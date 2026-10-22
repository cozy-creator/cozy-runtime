"""CPU control-flow checks only: no CUDA idle or retained allocation is simulated as proof."""

from __future__ import annotations

import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

import pytest

torch = pytest.importorskip("torch")


from cozy_runtime.internal.executor import _InFlight  # noqa: E402


def test_last_attempt_fences_admission_until_retention_finishes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plane, waiting, admitted = _InFlight(), threading.Event(), threading.Event()
    wait_for = plane._changed.wait_for

    def observed_wait(predicate: Callable[[], bool], timeout: float | None = None) -> bool:
        if not predicate():
            waiting.set()
        return wait_for(predicate, timeout)

    monkeypatch.setattr(plane._changed, "wait_for", observed_wait)

    def next_attempt() -> None:
        with plane.attempt(False):
            admitted.set()

    with ThreadPoolExecutor(max_workers=1) as pool:
        with plane.attempt(False) as slot:
            plane.ready()
            assert plane.last(slot)
            future = pool.submit(next_attempt)
            assert waiting.wait(5), "the next attempt reached the admission fence"
            assert not admitted.is_set(), "retention still owns the drained device"
        future.result(timeout=5)
    assert admitted.is_set() and plane.count == 0
