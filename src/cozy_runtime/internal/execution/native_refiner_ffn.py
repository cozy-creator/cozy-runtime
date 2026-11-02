"""Complete Diffusers SwiGLU FFN: actual int8 plans, supplied storage, one native choice.

H3's refiner calls this exact FeedForward after its norm. Norm, attention and residual
addition remain outside this operation. The residual/AdaLN FFN is a different contract.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import tensorfs.varena.torch as vt
import torch
from torch import Tensor, nn

from cozy_runtime import _kernels
from cozy_runtime.author._errors import Fault, fault
from cozy_runtime.author.kernels import _LINEAR_SPLIT, _RECENT, _SHARED, read
from cozy_runtime.internal import int8_linear, prebuilt
from cozy_runtime.internal.execution import catch, phase
from cozy_runtime.internal.execution.native_linear_split import _chunks, _Dispatch, _layout
from cozy_runtime.internal.execution.native_residual_ffn import _Invocation, _leaf, _Projection
from cozy_runtime.internal.residency import _REQUEST


def _plain_call(module: Any) -> bool:
    """Only an original method or the exact Runtime-installed pure-cut wrapper."""
    forward = vars(module).get("forward")
    if forward is not None and forward is vars(module).get("_cozy_execution_forward"):
        forward = getattr(forward, "__wrapped__", forward)
    return bool(
        not module.training
        and (forward is None or forward is vars(module).get("_cozy_ffn_cut_forward"))
        and module._compiled_call_impl is None
        and not module._forward_pre_hooks
        and not module._forward_hooks
    )


@dataclass(frozen=True)
class _Contract:
    module: Any
    net: Any
    activation: Any
    silu: Any
    dropout: Any
    up: Any
    down: Any
    forwards: tuple[Any, ...]

    def unchanged(self) -> bool:
        parts = (self.module, self.net, self.activation, self.silu, self.dropout)
        return bool(
            self.module.net is self.net
            and tuple(self.net) == (self.activation, self.dropout, self.down)
            and self.activation.proj is self.up
            and self.activation.activation is self.silu
            and self.dropout.p == 0.0
            and not self.dropout.inplace
            and not self.silu.inplace
            and tuple(type(part).forward for part in parts) == self.forwards
            and all(_plain_call(part) for part in parts)
        )


def declare(module: Any) -> _Contract | None:
    """Recognize the complete stock operation, never widths or class names alone.

    Called before Runtime installs this FFN's wrapper. No Diffusers import is required
    for unrelated modules. The nested activation wrapper is recognized by its identity.
    """
    if (type(module).__name__, type(module).__module__) != (
        "FeedForward", "diffusers.models.attention"
    ):
        return None
    from diffusers.models.activations import SwiGLU
    from diffusers.models.attention import FeedForward

    if type(module) is not FeedForward or type(module.net) is not nn.ModuleList:
        return None
    net = module.net
    if len(net) != 3 or type(net[0]) is not SwiGLU or type(net[1]) is not nn.Dropout:
        return None
    if type(net[0].activation) is not nn.SiLU:
        return None
    parts = (module, net, net[0], net[0].activation, net[1])
    contract = _Contract(
        *parts, net[0].proj, net[2], tuple(type(part).forward for part in parts)
    )
    return contract if contract.unchanged() else None


@dataclass(frozen=True)
class _Geometry:
    rows: int
    up: _Dispatch
    up_tail: _Dispatch
    down: _Dispatch
    down_tail: _Dispatch
    quantized_bytes: int
    scale_rows: int
    workspace: int

    @property
    def identity(self) -> str:
        return f"rows:{self.rows}/" + "/".join(
            f"{part.rows}:{part.kind}:{part.config}"
            for part in (self.up, self.up_tail, self.down, self.down_tail)
        )

    def allocations(self, rows: int, width: int, inner: int, itemsize: int) -> tuple[int, ...]:
        # All buffers coexist: full output, packed up, SiLU output, multiply output.
        sizes = [rows * width * itemsize, self.rows * 2 * inner * itemsize,
                 self.rows * inner * itemsize, self.rows * inner * itemsize]
        if self.quantized_bytes:
            sizes.extend((self.quantized_bytes, self.scale_rows * 4))
        if self.workspace:
            sizes.append(self.workspace)
        return tuple(sizes)


def _weight(module: Any, x: Tensor, width: int, channels: int) -> _Projection | None:
    """No cast, contiguous copy, bias epilogue or provider substitution is admitted."""
    if (not _leaf(module, x.dtype) or module.training or module.bias is not None
        or module.in_features != width or module.out_features != channels
        or getattr(module, "_cozy_siblings", None) not in (None, 1)):
        return None
    data, scale = read(module.data), read(module.scale)
    if (
        type(data) not in (Tensor, nn.Parameter) or type(scale) not in (Tensor, nn.Parameter)
        or data.device != x.device or scale.device != x.device
        or data.dtype != torch.int8 or scale.dtype != torch.float32
        or tuple(data.shape) != (channels, width) or tuple(scale.shape) != (channels,)
        or not data.is_contiguous() or not scale.is_contiguous()
        or data.data_ptr() % 16 or scale.data_ptr() % 16
        or channels % 8 or width % 256
    ):
        return None
    return _Projection(module, data, scale, None)


def split_ffn(contract: _Contract, x: Tensor) -> Tensor | None:
    """Decline only before selection. An accepted call bypasses both old FFN choosers."""
    request = _REQUEST.get()
    if (
        request is None or request.idle or not request.memory.cuda
        or type(x) is not Tensor or not x.is_cuda or x.device != request.memory.device
        or x.dtype not in (torch.bfloat16, torch.float16)
        or x.layout is not torch.strided or x.dim() != 3 or x.shape[0] != 1
        or not x.numel() or not x.is_contiguous() or x.data_ptr() % 16
        or torch.is_grad_enabled() or torch.is_autocast_enabled("cuda")
        or catch.uncommitted_scope() or _LINEAR_SPLIT.get() is not None
        or torch.cuda.is_current_stream_capturing()
        or (torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1)
        or nn.modules.module._global_forward_pre_hooks or nn.modules.module._global_forward_hooks
        or not contract.unchanged()
    ):
        return None
    rows, width = int(x.shape[1]), int(x.shape[2])
    inner = getattr(contract.down, "in_features", 0)
    if not 0 < rows <= 2**31 - 1 or type(inner) is not int or inner <= 0:
        return None
    held = getattr(_RECENT, "held", None)
    last = getattr(_RECENT, "last", None)
    if (any(shared.tensor is x for shared in _SHARED.get())
        or (held is not None and held.tensor is not None and held.tensor() is x)
        or (last is not None and last[0]() is x)):
        return None
    device = x.device.index or 0
    if not vt.split_context(device)["eligible"] or int8_linear.serving(device) != int8_linear.OP:
        return None
    library = int8_linear.library(device)
    if library.family != "mma.sync" or not hasattr(_kernels._gemm, "int8_plan"):
        return None
    up = _weight(contract.up, x, width, 2 * inner)
    down = _weight(contract.down, x, inner, width)
    if up is None or down is None:
        return None
    library._weight(prebuilt.INT8, up.data, up.scale, None, width, x.dtype)
    library._weight(prebuilt.INT8, down.data, down.scale, None, inner, x.dtype)
    queried: dict[tuple[int, int, int], _Dispatch] = {}

    def query(count: int, channels: int, columns: int) -> _Dispatch | None:
        key = count, channels, columns
        if key not in queried:
            info = _kernels._gemm.int8_plan(
                device, x.dtype == torch.float16, count, channels, columns, channels
            )
            if info["status"] == _kernels._gemm.NOT_SUPPORTED:
                return None
            library._check(info["status"], "the refiner FFN plan")
            if info["kind"] not in (0, 1):
                return None
            queried[key] = _Dispatch(count, info["kind"], info["config"], info["workspace_bytes"])
        return queried[key]

    geometries, choices = {}, []
    for chunk in _chunks(rows):
        tail = rows % chunk or chunk
        selected = (query(chunk, 2 * inner, width), query(tail, 2 * inner, width),
                    query(chunk, width, inner), query(tail, width, inner))
        if any(part is None for part in selected):
            return None
        u, ut, d, dt = selected
        assert u is not None and ut is not None and d is not None and dt is not None
        q_bytes = max((part.rows * columns
                       for part, columns in ((u, width), (ut, width), (d, inner), (dt, inner))
                       if part.kind == 0), default=0)
        scale_rows = max((part.rows for part in (u, ut, d, dt) if part.kind == 0), default=0)
        geometry = _Geometry(chunk, u, ut, d, dt, q_bytes, scale_rows,
                             max(u.workspace, ut.workspace, d.workspace, dt.workspace))
        geometries[geometry.identity] = geometry
        choices.append((geometry.identity,
                        geometry.allocations(rows, width, inner, x.element_size())))
    with request.memory.lock:
        models = tuple(model for model in request.memory.models.values()
                       if model.name in request.current)
    identities = tuple((model.content, model.wanted_identity()) for model in models)
    if any(identity is None for _, identity in identities):
        return None
    semantic = phase.configuration((
        "refiner-swiglu-ffn/1", tuple(sorted(identities)),
        tuple(_layout(value) for value in (x, up.data, up.scale, down.data, down.scale)),
        ("fresh-contiguous-output", tuple(x.shape), str(x.dtype)),
        ("aten.silu.out", "aten.mul.out", str(torch.__version__), torch.version.cuda),
        library.identity, library.family, up.module.provider, up.module.encoding,
        down.module.provider, down.module.encoding, tuple(choices), tuple(geometries.values()),
    ))
    if semantic is None:
        return None
    key = "runtime.split.refiner_ffn/1:" + hashlib.sha256(semantic.encode()).hexdigest()
    request.memory.remember_working_phase(key, models or (request,))
    stream = torch.cuda.current_stream(device)
    selection = vt.split_select(device, key, choices, preferred=choices[0][0],
                                stream=int(stream.cuda_stream))
    geometry = geometries[str(selection["choice_id"])]
    record = request.working_splits.setdefault(
        key, {"site": "refiner-swiglu-ffn", "key": key, "selections": 0}
    )
    record.update(selection=dict(selection), row_chunk=geometry.rows,
                  dispatch={name: vars(getattr(geometry, name))
                            for name in ("up", "up_tail", "down", "down_tail")},
                  allocation_bytes=geometry.allocations(rows, width, inner, x.element_size()),
                  workspace_bytes=geometry.workspace, output_ownership="fresh-result")
    record["selections"] += 1
    phase.choice(("native-split", key, geometry.identity))
    observation_id = phase.begin_split(device, key, geometry.identity, selection,
                                       stream=int(stream.cuda_stream))
    submitted = False
    try:
        result = _execute(x, up, down, geometry, library, stream)
        submitted = True
        return result
    except BaseException as error:
        if fault(error) is Fault.OOM:
            error.varena_no_retry = True  # type: ignore[attr-defined]
        raise
    finally:
        record["submission"] = vt.split_end(device, observation_id, submitted=submitted)


def _activate(packed: Tensor, silu: Tensor, activated: Tensor) -> None:
    """The stock two ATen pointwise operations, with exact dtype/shape outputs supplied.

    TensorIterator needs no device workspace for these pointwise out kernels. Two
    separate buffers retain the stock SiLU and product rounding/lifetime semantics.
    """
    value, gate = packed.chunk(2, dim=-1)
    torch.ops.aten.silu.out(gate, out=silu)
    torch.ops.aten.mul.out(value, silu, out=activated)


def _execute(x: Tensor, up: _Projection, down: _Projection, geometry: _Geometry,
             library: prebuilt.Prebuilt, stream: Any) -> Tensor:
    """Allocate the complete menu before the first launch; reuse every chunk buffer."""
    rows, inner = int(x.shape[1]), int(down.data.shape[1])
    chunk, stream_id = geometry.rows, int(stream.cuda_stream)
    with torch.cuda.device(x.device), torch.cuda.stream(stream):
        output = torch.empty_like(x)
        packed = torch.empty((1, chunk, 2 * inner), dtype=x.dtype, device=x.device)
        silu = torch.empty((1, chunk, inner), dtype=x.dtype, device=x.device)
        activated = torch.empty_like(silu)
        q = (torch.empty(geometry.quantized_bytes, dtype=torch.int8, device=x.device)
             if geometry.quantized_bytes else None)
        scales = (torch.empty(geometry.scale_rows, dtype=torch.float32, device=x.device)
                  if geometry.scale_rows else None)
        workspace = (torch.empty(geometry.workspace, dtype=torch.uint8, device=x.device)
                     if geometry.workspace else None)
        for at in range(0, rows, chunk):
            catch.checkpoint()
            count = min(chunk, rows - at)
            both, gate, activation = (buffer[:, :count] for buffer in (packed, silu, activated))
            u = geometry.up if count == chunk else geometry.up_tail
            d = geometry.down if count == chunk else geometry.down_tail
            _Invocation(up, x[:, at:at + count], both, u, q, scales, workspace,
                        library, stream_id).call()
            _activate(both, gate, activation)
            _Invocation(down, activation, output[:, at:at + count], d, q, scales, workspace,
                        library, stream_id).call()
        return output
