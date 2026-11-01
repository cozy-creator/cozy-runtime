"""The declared single-use int8 projection: exact kernel plans and one native row choice."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import tensorfs.varena.torch as vt
import torch
from torch import Tensor

from cozy_runtime import _kernels, lora
from cozy_runtime.author._errors import Fault, fault
from cozy_runtime.author._split import Cut
from cozy_runtime.author.kernels import (
    _LINEAR_SPLIT,
    _RECENT,
    _SHARED,
    INT8_CONVROT,
    SingleUseProjection,
    read,
)
from cozy_runtime.internal import int8_linear, prebuilt
from cozy_runtime.internal.execution import catch, phase
from cozy_runtime.internal.residency import _REQUEST


def _layout(tensor: Tensor | None) -> object:
    if tensor is None:
        return None
    return tuple(tensor.shape), tuple(tensor.stride()), str(tensor.dtype), tensor.data_ptr() % 16


def _chunks(rows: int) -> tuple[int, ...]:
    """Whole then balanced half-size windows, down to the legal one-row minimum."""
    chunks = [rows]
    while chunks[-1] > 1:
        chunks.append((chunks[-1] + 1) // 2)
    return tuple(chunks)


@dataclass(frozen=True)
class _Dispatch:
    rows: int
    kind: int
    config: int
    workspace: int


@dataclass(frozen=True)
class _Geometry:
    rows: int
    full: _Dispatch
    tail: _Dispatch
    quantized_rows: int
    workspace: int

    @property
    def identity(self) -> str:
        return (f"rows:{self.rows}/full:{self.full.kind}:{self.full.config}"
                f"/tail:{self.tail.rows}:{self.tail.kind}:{self.tail.config}")


def launch_projection(
    library: prebuilt.Prebuilt, data: Tensor, scale: Tensor, bias: Tensor | None,
    part: Tensor, target: Tensor, dispatch: _Dispatch,
    q: Tensor | None, scales: Tensor | None, scratch: Any, stream_id: int,
    *, quantized: bool = False,
) -> None:
    """Execute one already-selected projection into supplied storage; no allocation/policy."""
    device = part.device.index or 0
    half = part.dtype == torch.float16
    width, channels = part.shape[-1], data.shape[0]
    count = part.numel() // width
    output_stride = target.stride(-2)
    native = _kernels._gemm
    bias_ptr = 0 if bias is None else bias.data_ptr()
    if dispatch.kind == 1:
        if output_stride != channels:
            raise RuntimeError("a committed GEMV requires contiguous output rows")
        code = native.gemv(
            device, part.data_ptr(), half, data.data_ptr(), scale.data_ptr(),
            bias_ptr, target.data_ptr(), half, count, channels, width, stream_id,
        )
    else:
        assert q is not None and scales is not None
        if not quantized:
            code = native.quantize(
                device, False, part.data_ptr(), half, q.data_ptr(), scales.data_ptr(),
                count, width, 0, 0, 0, 0, 1, 0, 0.0, 0, 0, stream_id,
            )
            library._check(code, "the committed quantization")
        code = native.gemm(
            device, 0, False, q.data_ptr(), scales.data_ptr(), data.data_ptr(),
            scale.data_ptr(), bias_ptr, target.data_ptr(), output_stride, half,
            count, channels, width, dispatch.config, scratch, stream_id,
        )
    # A refusal ends this operation. It cannot choose another kernel or replay
    # already-written output rows through the legacy catch policy.
    library._check(code, "the committed projection")


@dataclass
class _Plan:
    module: Any
    x: Tensor
    data: Tensor
    scale: Tensor
    bias: Tensor | None
    library: prebuilt.Prebuilt
    request: Any
    stream: Any
    key: str
    geometry: _Geometry
    selection: dict[str, Any]
    ownership: str = "learned_or_inference"

    def run(self, x: Tensor, operand: object, out: Tensor | None) -> Tensor:
        """Entered by the exact module's ordinary forward, with its hooks still installed."""
        if x is not self.x or operand is not None or out is not None:
            raise RuntimeError("the committed int8 row plan received different operands")
        device = x.device.index or 0
        stream_id = int(self.stream.cuda_stream)
        if int(torch.cuda.current_stream(device).cuda_stream) != stream_id:
            raise RuntimeError("the committed int8 row plan changed streams")
        geometry = self.geometry
        record = self.request.working_splits.setdefault(
            self.key, {"site": "int8-linear-rows", "key": self.key, "selections": 0},
        )
        record.update(selection=self.selection, row_chunk=geometry.rows,
                      input_ownership=self.ownership,
                      dispatch={"full": vars(geometry.full), "tail": vars(geometry.tail)},
                      workspace_bytes=geometry.workspace)
        record["selections"] += 1
        phase.choice(("native-split", self.key, geometry.identity))
        observation_id = phase.begin_split(
            device, self.key, geometry.identity, self.selection, stream=stream_id,
        )
        submitted = False
        try:
            result = self._execute()
            submitted = True
            return result
        except BaseException as error:
            if fault(error) is Fault.OOM:
                error.varena_no_retry = True  # type: ignore[attr-defined]
            raise
        finally:
            record["submission"] = vt.split_end(device, observation_id, submitted=submitted)

    def _execute(self) -> Tensor:
        x, geometry = self.x, self.geometry
        device = x.device.index or 0
        rows, width, channels = x.numel() // x.shape[-1], x.shape[-1], self.data.shape[0]
        stream_id = int(self.stream.cuda_stream)
        with torch.cuda.device(device), torch.cuda.stream(self.stream):
            result = torch.empty((*x.shape[:-1], channels), dtype=x.dtype, device=x.device)
            flat, output = x.view(rows, width), result.view(rows, channels)
            q = scales = workspace = None
            if geometry.quantized_rows:
                q = torch.empty((geometry.quantized_rows, width), dtype=torch.int8, device=x.device)
                scales = torch.empty(geometry.quantized_rows, dtype=torch.float32, device=x.device)
            if geometry.workspace:
                workspace = torch.empty(geometry.workspace, dtype=torch.uint8, device=x.device)

            def scratch(size: int) -> Tensor:
                if workspace is None or size > workspace.numel():
                    raise RuntimeError("the committed GEMM requested undeclared workspace")
                return workspace

            for at in range(0, rows, geometry.rows):
                catch.checkpoint()
                count = min(geometry.rows, rows - at)
                dispatch = geometry.full if count == geometry.rows else geometry.tail
                part, target = flat.narrow(0, at, count), output.narrow(0, at, count)
                launch_projection(
                    self.library, self.data, self.scale, self.bias, part, target,
                    dispatch, q, scales, scratch, stream_id,
                )
            return result


def split_declared(module: Any, value: object, cuts: tuple[Cut, ...]) -> Tensor | None:
    """Generic calls retain actual learned/inference ownership eligibility."""
    return _split_declared(module, value, cuts, declared_single_use=False)


def split_single_use_projection(
    intent: SingleUseProjection, out: Any, cuts: tuple[Cut, ...],
) -> Tensor | None:
    """Matched API-only control: decline before native admission or execution."""
    return None


def _split_declared(
    module: Any, value: object, cuts: tuple[Cut, ...], *, declared_single_use: bool,
) -> Tensor | None:
    """None is capability exclusion before selection, never recovery from real failure."""
    if type(value) is not Tensor:
        return None
    x: Any = value
    request = _REQUEST.get()
    if (
        request is None or request.idle or not request.memory.cuda
        or not x.is_cuda or x.layout is not torch.strided
        or not x.is_contiguous() or x.dim() not in (2, 3) or not x.numel()
        or x.data_ptr() % 16
        or x.device != request.memory.device or (x.dim() == 3 and x.shape[0] != 1)
        or x.dtype not in (torch.bfloat16, torch.float16)
        or cuts != (Cut(x.dim() - 2),)
        or torch.is_grad_enabled() or torch.is_autocast_enabled("cuda")
        or catch.uncommitted_scope() or _LINEAR_SPLIT.get() is not None
        or torch.cuda.is_current_stream_capturing()
        or (torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1)
    ):
        return None
    from cozy_runtime.internal.encoding.leaves import _native_leaf_class

    if type(module) is not _native_leaf_class(torch):
        return None
    leaf: Any = module
    if (
        leaf.provider != INT8_CONVROT or leaf.out_dtype != x.dtype
        or "forward" in vars(leaf) or "linear" in vars(leaf)
        or leaf._compiled_call_impl is not None or leaf._forward_pre_hooks
        or torch.nn.modules.module._global_forward_pre_hooks
        or torch.nn.modules.module._global_forward_hooks
        or any(type(hook) is not lora.Hook or not hook._cozy_baked
               for hook in leaf._forward_hooks.values())
    ):
        return None
    # A declaration covers only a fresh producer-owned value. Live shared ownership and
    # an observed multi-consumer group still refuse; generic calls must actually learn.
    held = getattr(_RECENT, "held", None)
    last = getattr(_RECENT, "last", None)
    group = getattr(leaf, "_cozy_siblings", None)
    if (
        any(shared.tensor is x for shared in _SHARED.get())
        or (held is not None and held.tensor is not None and held.tensor() is x)
        or (declared_single_use and last is not None and last[0]() is x)
        or (declared_single_use and group not in (None, 1))
        or (not x.is_inference() and group != 1
            and not (declared_single_use and group is None))
    ):
        return None
    device = x.device.index or 0
    if not vt.split_context(device)["eligible"]:
        return None
    if int8_linear.serving(device) != int8_linear.OP:
        return None
    library = int8_linear.library(device)
    if library.family != "mma.sync" or not hasattr(_kernels._gemm, "int8_plan"):
        return None
    data, scale, bias = read(leaf.data), read(leaf.scale), read(leaf.bias)
    if scale.data_ptr() % 16 or (
        bias is not None
        and (bias.dtype != x.dtype or not bias.is_contiguous() or bias.data_ptr() % 16)
    ):
        return None  # No hidden cast/copy or weaker alignment than the queried operands.
    width = int(x.shape[-1])
    library._weight(prebuilt.INT8, data, scale, bias, width, x.dtype)
    rows, channels = x.numel() // width, int(data.shape[0])
    if rows < 2 or rows > 2**31 - 1:
        return None
    geometry: dict[str, _Geometry] = {}
    choices: list[tuple[str, tuple[int, ...]]] = []
    queried: dict[int, _Dispatch] = {}

    def query(count: int) -> _Dispatch | None:
        if count not in queried:
            info = _kernels._gemm.int8_plan(device, x.dtype == torch.float16,
                                            count, channels, width, channels)
            if info["status"] == _kernels._gemm.NOT_SUPPORTED:
                return None
            library._check(info["status"], "the projection plan")
            queried[count] = _Dispatch(count, info["kind"], info["config"], info["workspace_bytes"])
        return queried[count]

    output_bytes = rows * channels * x.element_size()
    for chunk in _chunks(rows):
        full, tail = query(chunk), query(rows % chunk or chunk)
        if full is None or tail is None:
            return None  # Unsupported dispatch is excluded before selecting or executing.
        quantized = max((part.rows for part in (full, tail) if part.kind == 0), default=0)
        scratch = max(full.workspace, tail.workspace)
        shape = _Geometry(chunk, full, tail, quantized, scratch)
        geometry[shape.identity] = shape
        allocations = [output_bytes]
        if quantized:
            allocations.extend((quantized * width, quantized * 4))
        if scratch:
            allocations.append(scratch)
        choices.append((shape.identity, tuple(allocations)))
    with request.memory.lock:
        models = tuple(model for model in request.memory.models.values()
                       if model.name in request.current)
    identities = tuple((model.content, model.wanted_identity()) for model in models)
    if any(identity is None for _, identity in identities):
        return None
    semantic = phase.configuration((
        "int8-linear-rows/1", tuple(sorted(identities)),
        _layout(x), _layout(data), _layout(scale), _layout(bias),
        ("contiguous-output", (*x.shape[:-1], channels), str(x.dtype)),
        library.identity, library.family, leaf.provider, leaf.encoding,
        tuple(choices), tuple(geometry.values()),
    ))
    if semantic is None:
        return None
    key = "runtime.split.int8_linear/1:" + hashlib.sha256(semantic.encode()).hexdigest()
    request.memory.remember_working_phase(key, models or (request,))
    stream = torch.cuda.current_stream(device)
    selection = vt.split_select(device, key, choices, preferred=choices[0][0],
                                stream=int(stream.cuda_stream))
    plan = _Plan(leaf, x, data, scale, bias, library, request, stream, key,
                 geometry[str(selection["choice_id"])], dict(selection),
                 "declared_single_use" if declared_single_use else "learned_or_inference")
    owner = _LINEAR_SPLIT.set(plan)
    try:
        return leaf(x)
    finally:
        _LINEAR_SPLIT.reset(owner)
