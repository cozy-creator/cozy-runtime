"""Exact Torch CUDA RMSNorm allocations, admitted before the existing kernel executes."""

from __future__ import annotations

import hashlib
import math

import tensorfs.varena.torch as vt
import torch
from torch import Tensor, nn

from cozy_runtime.author._errors import Fault, fault
from cozy_runtime.author.kernels import read
from cozy_runtime.internal.execution import catch, phase
from cozy_runtime.internal.execution.native_split import _layout
from cozy_runtime.internal.residency import _REQUEST

# This is an allocation ABI, not an estimate of a generic torch normalization. At this
# reviewed build, matching contiguous CUDA operands allocate Y and FP32 rstd only.
# Other implementations keep their ordinary path until their actual contract is reviewed.
_TORCH = "08187d9e0fba026dc8217405802ab5381dc88d90"


def rms_norm(site: str, module: nn.Module, x: Tensor) -> Tensor | None:
    """Matched API-only control: decline before native admission or execution."""
    return None
