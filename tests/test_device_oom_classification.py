"""Library and asynchronous device allocation failures are device OOM, not author faults."""

from __future__ import annotations

import pytest

from cozy_runtime.author._errors import (
    DEVICE_OOM,
    MEMORY_EXHAUSTED,
    RuntimeFailure,
    classify,
    is_device_oom,
)
from cozy_runtime.internal.memory import shortage
from cozy_runtime.internal.parallel.group import ENDED
from cozy_runtime.internal.parallel.mirror import ending

ALLOCATION_FAILURES = (
    "CUDA error: CUBLAS_STATUS_ALLOC_FAILED when calling `cublasCreate(handle)`",
    "cuDNN error: CUDNN_STATUS_ALLOC_FAILED",
    "CUDA error: out of memory\nCUDA kernel errors might be asynchronously reported",
    "cuFFT error: CUFFT_ALLOC_FAILED",
    "cusolver error: CUSOLVER_STATUS_ALLOC_FAILED, when calling `cusolverDnCreate(handle)`",
    "CUDA error: CUSPARSE_STATUS_ALLOC_FAILED when calling `cusparseCreate(handle)`",
    "NCCL error in: ProcessGroupNCCL.cpp:3356, unhandled cuda error, NCCL version 2.27.5\n"
    "ncclUnhandledCudaError: Call to CUDA function failed.\nLast error:\n"
    "Cuda failure 2 'out of memory'",
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
    assert classify(torch.OutOfMemoryError()).code == DEVICE_OOM, (
        "an empty message still classifies"
    )


def test_a_driver_call_out_of_device_memory_is_a_device_oom() -> None:
    """Run 4297: the driver could not allocate a stream on a full card."""
    torch = pytest.importorskip("torch")
    exc = torch.AcceleratorError(
        "CUDA error: out of memory\nSearch for `cudaErrorMemoryAllocation' in the CUDA runtime "
        "API for more information.\nThe CUDA driver logged these messages:\n"
        "Returning 2 (CUDA_ERROR_OUT_OF_MEMORY) from cuStreamCreateWithPriority"
    )
    assert is_device_oom(exc)
    assert classify(exc).code == DEVICE_OOM


def test_a_collective_out_of_memory_is_the_groups_to_run_again() -> None:
    """No op runs a collective again alone: its peers wait in it. The group does, and every
    other way a GPU's part ends either keeps the group (capacity, an ended collective) or
    breaks it (a fault)."""
    torch = pytest.importorskip("torch")
    nccl = torch.distributed.DistBackendError(ALLOCATION_FAILURES[-1])
    assert shortage(nccl) is None and ending(nccl) == DEVICE_OOM
    assert ending(torch.OutOfMemoryError("CUDA out of memory.")) == MEMORY_EXHAUSTED
    assert ending(RuntimeFailure("the ladder ran out", code=MEMORY_EXHAUSTED)) == MEMORY_EXHAUSTED
    assert ending(torch.distributed.DistBackendError("NCCL communicator was aborted")) == ENDED
    assert ending(ValueError("a fault")) is None
