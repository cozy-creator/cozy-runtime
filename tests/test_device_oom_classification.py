"""One classifier (`author._errors.fault`): an allocation failure is OOM by torch's type, its CUDA
code, a CUDA library's alloc status or Triton's prefix; a sticky CUDA error ends the process;
any other CUDA or NCCL error ends the executor. varena's `Exhausted` is capacity
(`test_execution.py` raises a real one)."""

from __future__ import annotations

import pytest

from cozy_runtime.author._errors import (
    DEVICE_FAULT,
    DEVICE_OOM,
    MEMORY_EXHAUSTED,
    Fault,
    RuntimeFailure,
    classify,
    fault,
)
from cozy_runtime.internal.parallel.group import ENDED
from cozy_runtime.internal.parallel.mirror import ending


@pytest.mark.parametrize(
    "message",
    [
        "CUDA error: CUBLAS_STATUS_ALLOC_FAILED when calling `cublasCreate(handle)`",
        "cuDNN error: CUDNN_STATUS_ALLOC_FAILED",
        "Triton Error [CUDA]: out of memory",
    ],
)
def test_a_library_allocation_status_is_a_runtime_device_oom(message: str) -> None:
    assert fault(RuntimeError(message)) is Fault.OOM
    outcome = classify(RuntimeError(message))
    assert (outcome.terminal, outcome.origin, outcome.code) == ("failed", "runtime", DEVICE_OOM)


@pytest.mark.parametrize(
    "exc",
    [
        # no code: the text of an OOM says nothing the classifier reads
        RuntimeError("CUDA error: out of memory"),
        RuntimeError("an error mentioning Triton Error [CUDA]: out of memory later"),
        MemoryError("host memory exhausted"),
        ValueError("CUBLAS_STATUS_ALLOC_FAILED"),
    ],
)
def test_other_failures_stay_the_authors(exc: BaseException) -> None:
    assert fault(exc) is Fault.OTHER
    assert classify(exc).code == "unhandled_exception"


def test_torch_s_oom_type_and_cuda_codes() -> None:
    torch = pytest.importorskip("torch")
    assert fault(torch.OutOfMemoryError()) is Fault.OOM
    assert classify(torch.OutOfMemoryError("CUDA out of memory.")).code == DEVICE_OOM

    def accelerator(code: int) -> BaseException:
        exc = torch.AcceleratorError(f"CUDA error {code}")
        exc.error_code = code  # torch sets it from cudaError_t
        return exc

    assert fault(accelerator(2)) is Fault.OOM
    assert fault(accelerator(700)) is Fault.STICKY  # an illegal address: the context is gone
    assert fault(accelerator(999)) is Fault.DEVICE  # cudaErrorUnknown: the executor ends
    assert classify(accelerator(700)).code == classify(accelerator(999)).code == DEVICE_FAULT
    nccl = torch.distributed.DistBackendError("NCCL error: unhandled cuda error")
    assert fault(nccl) is Fault.DEVICE


def test_a_gpu_short_of_memory_ends_the_group_call_whole() -> None:
    """Any GPU of a group short of memory ends the attempt with the group whole; an ended
    collective keeps it too, and a fault breaks it."""
    torch = pytest.importorskip("torch")
    assert ending(torch.OutOfMemoryError("CUDA out of memory.")) == MEMORY_EXHAUSTED
    assert ending(RuntimeFailure("it did not fit", code=MEMORY_EXHAUSTED)) == MEMORY_EXHAUSTED
    assert ending(torch.distributed.DistBackendError("NCCL communicator was aborted")) == ENDED
    assert ending(ValueError("a fault")) is None
