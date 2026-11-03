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
    request = _REQUEST.get()
    if (
        torch.version.git_version != _TORCH or type(module) is not nn.RMSNorm
        or request is None or request.idle or not request.memory.cuda
        or type(x) is not Tensor or not x.is_cuda or x.layout is not torch.strided
        or not x.is_contiguous() or x.device != request.memory.device or x.data_ptr() % 16
        or x.dtype not in (torch.float16, torch.bfloat16, torch.float32)
        or torch.is_grad_enabled() or torch.is_autocast_enabled("cuda")
        or catch.uncommitted_scope() or torch.cuda.is_current_stream_capturing()
        or (torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1)
    ):
        return None
    shape = tuple(int(n) for n in module.normalized_shape)
    if (len(shape) != 1 or x.dim() <= 1 or any(n <= 0 for n in shape)
            or tuple(x.shape[-len(shape):]) != shape):
        return None
    eps = module.eps
    if eps is not None and (not math.isfinite(eps) or eps < 0):
        return None
    width = math.prod(shape)
    rows = x.numel() // width
    if rows < 2 or rows > 2**31 - 1:
        return None
    device = x.device.index or 0
    if not vt.split_context(device)["eligible"]:
        return None
    # The actual source binding precedes admission. CUDA's expect_contiguous must not
    # conceal a copy, and mismatched weight dtype selects a different composite recipe.
    weight = read(module.weight)
    if weight is not None and (
        type(weight) not in (Tensor, nn.Parameter) or weight.device != x.device
        or weight.dtype != x.dtype or tuple(weight.shape) != shape
        or not weight.is_contiguous() or weight.data_ptr() % 16
    ):
        return None
    flat = x.view(rows, *shape)
    total = x.numel() * x.element_size()
    # H3 already runs this operator whole. Splitting retains a full join output beside
    # each window and generally increases demand; there is no useful synthetic row menu.
    choices = [("whole", (total, rows * 4))]
    with request.memory.lock:
        models = tuple(model for model in request.memory.models.values()
                       if model.name in request.current)
    identities = tuple((model.content, model.wanted_identity()) for model in models)
    if any(identity is None for _, identity in identities):
        return None
    native = torch.ops.aten._fused_rms_norm.default
    semantic = phase.configuration((
        "torch-fused-rms/1", _TORCH, str(native._schema), site,
        tuple(sorted(identities)), _layout(x), _layout(weight), shape, eps, tuple(choices),
    ))
    if semantic is None:
        return None
    key = "runtime.split.rms_norm/1:" + hashlib.sha256(semantic.encode()).hexdigest()
    request.memory.remember_working_phase(key, models or (request,))
    stream = torch.cuda.current_stream(device)
    stream_id = int(stream.cuda_stream)
    selection = vt.split_select(device, key, choices, preferred=choices[0][0], stream=stream_id)
    selected = str(selection["choice_id"])
    record = request.working_splits.setdefault(
        key, {"site": site, "key": key, "selections": 0},
    )
    record.update(selection=dict(selection), row_chunk=rows, rows=rows, width=width,
                  dtype=str(x.dtype), implementation=_TORCH,
                  allocations=dict(choices)[selected], native_calls=1)
    record["selections"] += 1
    phase.choice(("native-split", key, selected))
    token = phase.begin_split(device, key, selected, selection, stream=stream_id)
    submitted = False
    try:
        with torch.cuda.device(device), torch.cuda.stream(stream):
            result, rstd = native(flat, shape, weight, eps)
            del rstd
            submitted = True
            return result.view(x.shape)
    except BaseException as error:
        if fault(error) is Fault.OOM:
            error.varena_no_retry = True  # type: ignore[attr-defined]
        raise
    finally:
        record["submission"] = vt.split_end(device, token, submitted=submitted)
