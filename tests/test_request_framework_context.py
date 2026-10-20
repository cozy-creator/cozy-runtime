"""CPU control-flow units for thread-local CUDA setup; these do not qualify CUDA."""

from __future__ import annotations

import threading
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from cozy_runtime.author._errors import InvalidRequest
from cozy_runtime.internal.executor import Executor


class Boundary:
    def __init__(self) -> None:
        self.local = threading.local()
        self.events: dict[int, list[str]] = {}
        self.setup_error: BaseException | None = None
        self.measurement_error: BaseException | None = None
        self.cuda = True
        self.device = "selected-memory-device"

    def record(self, event: str) -> None:
        self.events.setdefault(threading.get_ident(), []).append(event)

    def _measure(self) -> None:
        self.record("measure")
        if getattr(self.local, "setup_failed", False) and self.measurement_error is not None:
            raise self.measurement_error
        self.local.budget_ready = True

    @contextmanager
    def selected(self, device: str) -> Iterator[None]:
        assert device == self.device
        previous = getattr(self.local, "device", None)
        self.local.device = device
        self.record("select")
        try:
            yield
        finally:
            self.local.device = previous
            self.record("restore")

    def blas(self) -> None:
        assert self.local.budget_ready and self.local.device == self.device
        self.record("blas")
        if self.setup_error is not None:
            self.local.setup_failed = True
            raise self.setup_error

    @contextmanager
    def request(self, request_id: str, **_: Any) -> Iterator[Any]:
        # The real Memory.request owns its measurement, native begin and end.
        self.record("begin")
        try:
            yield SimpleNamespace(id=request_id)
        finally:
            self.record("end")


def executor(boundary: Boundary) -> Executor:
    process = object.__new__(Executor)
    process.host = process
    process.construction = "construction"
    process.memory_requests = {}
    memory: Any = boundary
    process.memory = memory
    process.torch = SimpleNamespace(
        cuda=SimpleNamespace(device=boundary.selected, current_blas_handle=boundary.blas)
    )
    return process


def test_every_actual_calling_thread_initializes_before_request_admission() -> None:
    boundary = Boundary()
    process = executor(boundary)
    rendezvous = threading.Barrier(2)

    def call(request_id: str) -> int:
        with process._request(request_id) as request:
            assert request is process.memory_requests[request_id]
            assert boundary.local.device is None
            boundary.record("body")
            rendezvous.wait(timeout=10)
            assert len(process.memory_requests) == 2
            rendezvous.wait(timeout=10)
        return threading.get_ident()

    with ThreadPoolExecutor(max_workers=2) as pool:
        calls = [pool.submit(call, name) for name in ("first", "second")]
        assert len({call.result(timeout=15) for call in calls}) == 2
    assert not process.memory_requests
    assert (
        list(boundary.events.values())
        == [["measure", "select", "blas", "restore", "begin", "body", "end"]] * 2
    )


@pytest.mark.parametrize("measurement_fails", [False, True])
def test_partial_setup_failure_publishes_without_admitting_or_masking_the_error(
    measurement_fails: bool,
) -> None:
    boundary = Boundary()
    process = executor(boundary)
    failure = RuntimeError("framework setup exhausted its budget")
    boundary.setup_error = failure
    if measurement_fails:
        boundary.measurement_error = ValueError("measurement unavailable")
    with pytest.raises(RuntimeError) as caught, process._request("failed"):
        pytest.fail("failed setup must not admit a request")
    assert caught.value is failure and not process.memory_requests
    assert list(boundary.events.values()) == [["measure", "select", "blas", "restore", "measure"]]
    assert getattr(failure, "__notes__", []) == (
        ["framework context measurement failed: measurement unavailable"]
        if measurement_fails
        else []
    )


def test_duplicate_request_is_rejected_before_framework_setup() -> None:
    boundary = Boundary()
    process = executor(boundary)
    active: Any = object()
    process.memory_requests["active"] = active
    with pytest.raises(InvalidRequest, match="already active"), process._request("active"):
        pytest.fail("a duplicate request must not run")
    assert not boundary.events and process.memory_requests == {"active": active}


@pytest.mark.parametrize("guard", ["cpu", "no-torch", "no-memory"])
def test_non_cuda_or_absent_planes_do_not_initialize_cuda(guard: str) -> None:
    boundary = Boundary()
    process = executor(boundary)
    if guard == "cpu":
        boundary.cuda = False
    elif guard == "no-torch":
        process.torch = None
    else:
        process.memory = None
    with process._request("host") as request:
        assert (request is None) == (guard == "no-memory")
    assert list(boundary.events.values()) == ([] if guard == "no-memory" else [["begin", "end"]])
    assert not process.memory_requests


def test_real_cpu_request_still_closes_after_body_failure(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    from tensorfs.varena.torch import Memory

    boundary = Boundary()
    process = executor(boundary)
    process.memory = Memory.open(torch.device("cpu"))
    process.torch = torch
    failure = ValueError("body failed")
    with pytest.raises(ValueError) as caught, process._request("body-error") as request:
        assert request is process.memory_requests["body-error"]
        raise failure
    assert caught.value is failure
    assert not process.memory.active and not process.memory_requests
    assert request is not None and request.stats is not None and request.stats.leftover == ()
