"""Library and asynchronous device allocation failures are device OOM, not author faults."""

from __future__ import annotations

import pytest

from cozy_runtime.author._errors import DEVICE_OOM, classify, is_device_oom

ALLOCATION_FAILURES = (
    "CUDA error: CUBLAS_STATUS_ALLOC_FAILED when calling `cublasCreate(handle)`",
    "cuDNN error: CUDNN_STATUS_ALLOC_FAILED",
    "CUDA error: out of memory\nCUDA kernel errors might be asynchronously reported",
)


@pytest.mark.parametrize("message", ALLOCATION_FAILURES)
def test_library_allocation_failure_is_a_runtime_device_oom(message: str) -> None:
    outcome = classify(RuntimeError(message))
    assert (outcome.terminal, outcome.origin, outcome.code) == ("failed", "runtime", DEVICE_OOM)
    assert outcome.message == message.splitlines()[0]


@pytest.mark.parametrize(
    "exc",
    [
        RuntimeError("CUDA error: an illegal memory access was encountered"),
        MemoryError("host memory exhausted"),
        ValueError("CUBLAS_STATUS_ALLOC_FAILED"),
    ],
)
def test_other_failures_stay_the_authors(exc: BaseException) -> None:
    assert not is_device_oom(exc)
    assert classify(exc).code == "unhandled_exception"


def test_torch_device_oom_is_still_recognized() -> None:
    torch = pytest.importorskip("torch")
    exc = torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 GiB")
    assert classify(exc).code == DEVICE_OOM
