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
    "Triton Error [CUDA]: out of memory",
    "CUDA driver error: out of memory",
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


def test_a_plane_driver_call_out_of_device_memory_is_a_device_oom() -> None:
    """Run 10996: the weight plane's own driver calls run out of device memory too."""
    errors = pytest.importorskip("tensorfs.varena")
    exc = errors.CudaError("cuda driver: cuEventCreate failed with CUDA_ERROR_OUT_OF_MEMORY (2)")
    assert classify(exc).code == DEVICE_OOM
    other = errors.CudaError("cuda driver: cuMemMap failed with CUDA_ERROR_INVALID_VALUE (1)")
    assert not is_device_oom(other)


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


def test_a_gpu_short_of_memory_ends_the_group_call_whole() -> None:
    """Any GPU of a group short of memory, in a collective or not, ends the attempt with the
    group whole (memory-varena §5.5); an ended collective keeps it too, and a fault breaks it."""
    torch = pytest.importorskip("torch")
    nccl = torch.distributed.DistBackendError(ALLOCATION_FAILURES[-1])
    assert ending(nccl) == MEMORY_EXHAUSTED
    assert ending(torch.OutOfMemoryError("CUDA out of memory.")) == MEMORY_EXHAUSTED
    assert ending(RuntimeFailure("it did not fit", code=MEMORY_EXHAUSTED)) == MEMORY_EXHAUSTED
    assert ending(torch.distributed.DistBackendError("NCCL communicator was aborted")) == ENDED
    assert ending(ValueError("a fault")) is None


def test_varena_exhausted_is_capacity() -> None:
    """`tensorfs.varena.torch.Exhausted`, known by its module and name, fails the attempt as
    capacity; its `Unbound` is a weight bound to nothing, a fault."""
    varena = pytest.importorskip("tensorfs.varena.torch")
    peaks = (varena.Peak("unet", "1024x1024", 1 << 30, ()),)
    exhausted = varena.Exhausted(0, 1 << 20, "bind unet", {"unet": 5 << 30}, peaks)
    assert classify(exhausted).code == MEMORY_EXHAUSTED
    assert classify(varena.Unbound("a weight binds to nothing")).code == "weight_unbound"
