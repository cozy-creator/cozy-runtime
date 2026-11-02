"""Status-preserving calls into the prepared dense provider; no retry or fallback."""
from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

_TORCH = "08187d9e0fba026dc8217405802ab5381dc88d90"
_LT_SETTING = os.environ.get("DISABLE_ADDMM_CUDA_LT")


@dataclass(frozen=True)
class Policy:
    workspace_limit: int
    half_accumulation: bool
    reduction: int
    sm_target: int
    preferred_blas: str
    lt_environment: str | None


def policy(device: int) -> Policy | None:
    """The reviewed effective math/workspace policy, read before query and before writes.

    As with Torch's cached flag, the process contract fixes DISABLE_ADDMM_CUDA_LT
    before its first Torch BLAS use. A current environment read cannot reconstruct
    earlier changes; they are unsupported, not silently treated as an effective getter.
    """
    if (torch.version.git_version != _TORCH or torch.version.hip is not None
            or torch.is_grad_enabled() or not torch.is_autocast_enabled("cuda")
            or torch.get_autocast_dtype("cuda") is not torch.float16
            or _LT_SETTING not in (None, "0")
            or os.environ.get("DISABLE_ADDMM_CUDA_LT") != _LT_SETTING
            or torch.cuda.tunable.is_enabled()):
        return None
    cuda = torch.backends.cuda
    backend = cuda.preferred_blas_library()
    if backend not in (torch._C._BlasBackend.Cublas, torch._C._BlasBackend.Cublaslt):
        return None
    reduced = cuda.matmul.allow_fp16_reduced_precision_reduction
    split_k = cuda.matmul.allow_fp16_reduced_precision_reduction_split_k
    if reduced and not split_k:
        return None
    reduction = 0 if reduced else (1 if split_k else 2)
    carveout = torch._C._get_sm_carveout_experimental()
    target = 0
    if carveout is not None:
        count = torch.cuda.get_device_properties(device).multi_processor_count
        if not 0 <= carveout < count:
            return None
        target = count - carveout
    return Policy(int(cuda.cublaslt_workspace_size()), cuda.matmul.allow_fp16_accumulation,
                  reduction, target, str(backend), _LT_SETTING)


def checked(call: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    # Import is lazy: source-only/pure wheels and unrelated consumers do not need Lt.
    from cozy_runtime._kernels import _dense

    try:
        return call(*args, **kwargs)
    except _dense.Error as error:
        failure: BaseException
        if error.domain == "cublas" and error.status == 3:  # CUBLAS_STATUS_ALLOC_FAILED
            failure = torch.OutOfMemoryError(str(error))
        elif error.domain in ("cuda", "driver"):
            failure = torch.AcceleratorError(str(error))
            failure.error_code = error.status  # type: ignore[attr-defined]
        else:
            failure = RuntimeError(str(error))
        # Preparation has no numerical side effect to replay, and an accepted launch
        # may already have written output. Neither path reenters application byte policy.
        failure.varena_no_retry = True  # type: ignore[attr-defined]
        raise failure from error
