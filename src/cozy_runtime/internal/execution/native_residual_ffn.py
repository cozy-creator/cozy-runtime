"""One declared residual/RMSNorm/SwiGLU tail: native choice, supplied output, no retries."""

from __future__ import annotations

import hashlib
import math
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
    ResidualFFN,
    TritonEntry,
    read,
)
from cozy_runtime.internal import int8_linear, prebuilt
from cozy_runtime.internal.execution import catch, phase
from cozy_runtime.internal.execution.native_linear_split import (
    _Dispatch,
    _layout,
    launch_projection,
)
from cozy_runtime.internal.residency import _REQUEST


def _tail_chunks(rows: int) -> tuple[int, ...]:
    """Whole then approximately half-sized even windows; odd final lengths stay legal.

    Existing GLUE kernels require 16-byte Int64 index-pointer alignment. Whole calls
    start at zero; every later window must start on an even row, without an index copy.
    """
    choices = [rows]
    while choices[-1] > 2:
        choices.append(max(2, ((choices[-1] + 1) // 2) // 2 * 2))
    return tuple(choices)


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
        choices = (self.up, self.up_tail, self.down, self.down_tail)
        return f"rows:{self.rows}/" + "/".join(
            f"{part.rows}:{part.kind}:{part.config}" for part in choices
        )


@dataclass(frozen=True)
class _Projection:
    module: Any
    data: Any
    scale: Any
    bias: Any


@dataclass
class _Invocation:
    """One normal NativeLinear invocation; its complete storage is owned by the tail."""

    projection: _Projection
    x: Any
    target: Any
    dispatch: _Dispatch
    q: Any
    scales: Any
    workspace: Any
    library: prebuilt.Prebuilt
    stream_id: int

    @property
    def module(self) -> Any:
        return self.projection.module

    def run(self, x: Any, operand: object, out: Any) -> Any:
        if x is not self.x or operand is not None or out is not self.target:
            raise RuntimeError("the committed FFN projection received different operands")
        if int(torch.cuda.current_stream(x.device).cuda_stream) != self.stream_id:
            raise RuntimeError("the committed FFN projection changed streams")

        def scratch(size: int) -> Any:
            if self.workspace is None or size > self.workspace.numel():
                raise RuntimeError("the FFN GEMM requested undeclared workspace")
            return self.workspace

        p = self.projection
        launch_projection(
            self.library,
            p.data,
            p.scale,
            p.bias,
            x,
            out,
            self.dispatch,
            self.q,
            self.scales,
            scratch,
            self.stream_id,
        )
        return out

    def call(self) -> None:
        token = _LINEAR_SPLIT.set(self)
        try:
            result = self.module(self.x, out=self.target)
            if result is not self.target:
                raise RuntimeError("the FFN projection replaced its declared output")
        finally:
            _LINEAR_SPLIT.reset(token)


def _entry_identity(entry: TritonEntry) -> tuple[Any, ...]:
    return (
        str(entry.kernel.cache_key),
        tuple(sorted(entry.signature.items())),
        tuple(sorted(entry.constexprs.items())),
        entry.aligned,
        entry.num_warps,
        entry.num_stages,
        entry.fp_fusion,
    )


def _entries(intent: ResidualFFN, width: int, device: int) -> tuple[Any, ...] | None:
    from cozy_runtime.author.kernels import _SETS
    from cozy_runtime.internal.triton_kernels import _LOADED, _Launcher

    launch = intent.launch
    if type(launch) is not _Launcher or not launch.key:
        return None
    sets = tuple(
        _SETS[name]
        for (name, index), loaded in _LOADED.items()
        if loaded is launch and index == device and name in _SETS
    )
    if not sets:
        return None
    block = 1 << (width - 1).bit_length()
    expected = (
        (
            intent.norm_mod,
            {
                "x_ptr": "*bf16",
                "branch_ptr": "*bf16",
                "gate_ptr": "*bf16",
                "hidden_ptr": "*bf16",
                "out_ptr": "*bf16",
                "scale_out_ptr": "*bf16",
                "weight_ptr": "*bf16",
                "scale_ptr": "*bf16",
                "shift_ptr": "*bf16",
                "index_ptr": "*i64",
                "n_cols": "i32",
                "n_index": "i32",
                "stride_x": "i32",
                "stride_table": "i32",
                "eps": "fp32",
                "RESIDUAL": "constexpr",
                "FP8": "constexpr",
                "BLOCK": "constexpr",
            },
            {"RESIDUAL": 1, "FP8": 0, "BLOCK": block},
        ),
        (
            intent.swiglu,
            {
                "x_ptr": "*bf16",
                "out_ptr": "*bf16",
                "scale_out_ptr": "*bf16",
                "half": "i32",
                "stride_x": "i32",
                "stride_out": "i32",
                "FP8": "constexpr",
                "BLOCK": "constexpr",
            },
            {"FP8": 0, "BLOCK": 1024},
        ),
        (
            intent.gate_add,
            {
                "x_ptr": "*bf16",
                "branch_ptr": "*bf16",
                "gate_ptr": "*bf16",
                "out_ptr": "*bf16",
                "index_ptr": "*i64",
                "n_cols": "i32",
                "n_index": "i32",
                "stride_x": "i32",
                "stride_table": "i32",
                "BLOCK": "constexpr",
            },
            {"BLOCK": block},
        ),
    )
    identities = []
    for entry, signature, constants in expected:
        if (
            type(entry) is not TritonEntry
            or dict(entry.signature) != signature
            or dict(entry.constexprs) != constants
            or entry.key not in launch.compiled
            or tuple(entry.kernel.arg_names) != tuple(signature)
            or not any(
                actual.key == entry.key and _entry_identity(actual) == _entry_identity(entry)
                for kernels in sets
                for actual in kernels.entries
            )
        ):
            return None
        # Only a compiled entry of the exact supplied package kernel is used. Kernel body,
        # round points and compile options are part of the immutable execution identity.
        identities.append(_entry_identity(entry))
    return (launch.key, *identities)


def _leaf(module: Any, dtype: Any, *, materialized: bool = False) -> bool:
    from cozy_runtime.internal.encoding.leaves import _native_leaf_class

    if type(module) is not _native_leaf_class(torch):
        return False
    leaf: Any = module
    return bool(
        leaf.provider == INT8_CONVROT
        and leaf.out_dtype == dtype
        and "forward" not in vars(leaf)
        and "linear" not in vars(leaf)
        and leaf._compiled_call_impl is None
        and not leaf._forward_pre_hooks
        and (not materialized or leaf._forward_hooks)
        and all(
            type(hook) is lora.Hook and hook._cozy_baked for hook in leaf._forward_hooks.values()
        )
    )


def _storage(tensor: Any) -> tuple[int, int]:
    storage = tensor.untyped_storage()
    begin = int(storage.data_ptr())
    return begin, begin + storage.nbytes()


def _overlaps(left: Any, right: Any) -> bool:
    a, b = _storage(left)
    c, d = _storage(right)
    return a < d and c < b


def _plain(tensor: Any, device: Any, dtype: Any) -> bool:
    return bool(
        type(tensor) in (Tensor, torch.nn.Parameter)
        and tensor.device == device
        and tensor.dtype == dtype
        and tensor.layout is torch.strided
        and tensor.numel()
        and tensor.data_ptr() % 16 == 0
    )


def split_residual_ffn(intent: ResidualFFN, output: Any, cuts: tuple[Cut, ...]) -> Tensor | None:
    """Return None only for an unsupported contract before native selection/execution."""
    request = _REQUEST.get()
    if (
        type(intent) is not ResidualFFN
        or request is None
        or request.idle
        or not request.memory.cuda
    ):
        return None
    x = intent.residual
    if (
        intent.exclusive_out is not True
        or type(x) is not Tensor
        or type(output) is not Tensor
        or not x.is_cuda
        or x.dtype != torch.bfloat16
        or x.device != request.memory.device
        or x.dim() != 3
        or x.shape[0] != 1
        or not x.numel()
        or not x.is_contiguous()
        or not _plain(x, x.device, torch.bfloat16)
        or not _plain(output, x.device, torch.bfloat16)
        or not output.is_contiguous()
        or output.shape != x.shape
        or cuts != (Cut(1),)
        or torch.is_grad_enabled()
        or torch.is_autocast_enabled("cuda")
        or catch.uncommitted_scope()
        or _LINEAR_SPLIT.get() is not None
        or torch.cuda.is_current_stream_capturing()
        or (torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1)
        or torch.nn.modules.module._global_forward_pre_hooks
        or torch.nn.modules.module._global_forward_hooks
        or not _leaf(intent.up, x.dtype)
        or not _leaf(intent.down, x.dtype, materialized=True)
        or type(intent.eps) not in (int, float)
        or not math.isfinite(intent.eps)
        or intent.eps <= 0
    ):
        return None
    device = x.device.index or 0
    if not vt.split_context(device)["eligible"] or int8_linear.serving(device) != int8_linear.OP:
        return None
    library = int8_linear.library(device)
    if library.family != "mma.sync" or not hasattr(_kernels._gemm, "int8_plan"):
        return None
    width, rows = int(x.shape[-1]), int(x.shape[1])
    if not 0 < rows <= 2**31 - 1:
        return None
    kernel_identity = _entries(intent, width, device)
    if kernel_identity is None:
        return None
    held = getattr(_RECENT, "held", None)
    if any(shared.tensor is x for shared in _SHARED.get()) or (
        held is not None and held.tensor is not None and held.tensor() is x
    ):
        return None
    # Materialize every readonly source before native selection. No hidden reads or dtype
    # conversion can enter after the first in-place output write.
    norm, gate_msa, scale, shift, gate, index = (
        read(value)
        for value in (
            intent.norm_weight,
            intent.gate_msa,
            intent.scale_mlp,
            intent.shift_mlp,
            intent.gate_mlp,
            intent.index,
        )
    )
    if (
        not _plain(norm, x.device, x.dtype)
        or tuple(norm.shape) != (width,)
        or not norm.is_contiguous()
        or not _plain(index, x.device, torch.int64)
        or tuple(index.shape) != (rows,)
        or not index.is_contiguous()
    ):
        return None
    tables = (gate_msa, scale, shift, gate)
    if any(
        not _plain(table, x.device, x.dtype)
        or table.dim() != 2
        or table.shape[1] != width
        or table.stride(1) != 1
        or table.stride(0) % 16
        for table in tables
    ):
        return None
    if (
        any(table.shape != scale.shape for table in tables)
        or gate_msa.stride(0) != scale.stride(0)
        or shift.stride(0) != scale.stride(0)
    ):
        return None
    projections = []
    for module in (intent.up, intent.down):
        data, scales, bias = read(module.data), read(module.scale), read(module.bias)
        if scales.data_ptr() % 16 or (
            bias is not None
            and (bias.dtype != x.dtype or not bias.is_contiguous() or bias.data_ptr() % 16)
        ):
            return None
        projections.append(_Projection(module, data, scales, bias))
    up, down = projections
    inner = int(up.data.shape[0]) // 2
    if (
        up.data.shape[0] != 2 * inner
        or up.data.shape[1] != width
        or tuple(down.data.shape) != (width, inner)
    ):
        return None
    library._weight(prebuilt.INT8, up.data, up.scale, up.bias, width, x.dtype)
    library._weight(prebuilt.INT8, down.data, down.scale, down.bias, inner, x.dtype)
    readonly = (
        x,
        norm,
        gate_msa,
        scale,
        shift,
        gate,
        index,
        up.data,
        up.scale,
        down.data,
        down.scale,
        *((up.bias,) if up.bias is not None else ()),
        *((down.bias,) if down.bias is not None else ()),
    )
    if any(_overlaps(output, value) for value in readonly):
        return None

    queried: dict[tuple[int, int, int], _Dispatch] = {}

    def query(count: int, channels: int, columns: int) -> _Dispatch | None:
        key = count, channels, columns
        if key not in queried:
            info = _kernels._gemm.int8_plan(device, False, count, channels, columns, channels)
            if info["status"] == _kernels._gemm.NOT_SUPPORTED:
                return None
            library._check(info["status"], "the residual FFN plan")
            queried[key] = _Dispatch(count, info["kind"], info["config"], info["workspace_bytes"])
        return queried[key]

    geometries = {}
    choices = []
    for chunk in _tail_chunks(rows):
        tail = rows % chunk or chunk
        selected = (
            query(chunk, 2 * inner, width),
            query(tail, 2 * inner, width),
            query(chunk, width, inner),
            query(tail, width, inner),
        )
        if any(part is None for part in selected):
            return None
        u, ut, d, dt = selected
        assert u is not None and ut is not None and d is not None and dt is not None
        q_bytes = max(
            (
                part.rows * columns
                for part, columns in ((u, width), (ut, width), (d, inner), (dt, inner))
                if part.kind == 0
            ),
            default=0,
        )
        scale_rows = max((part.rows for part in (u, ut, d, dt) if part.kind == 0), default=0)
        workspace = max(u.workspace, ut.workspace, d.workspace, dt.workspace)
        geometry = _Geometry(chunk, u, ut, d, dt, q_bytes, scale_rows, workspace)
        geometries[geometry.identity] = geometry
        allocations = [2 * chunk * width, 4 * chunk * inner, 2 * chunk * inner, 2 * chunk * width]
        if q_bytes:
            allocations.extend((q_bytes, 4 * scale_rows))
        if workspace:
            allocations.append(workspace)
        choices.append((geometry.identity, tuple(allocations)))
    with request.memory.lock:
        models = tuple(
            model for model in request.memory.models.values() if model.name in request.current
        )
    identities = tuple((model.content, model.wanted_identity()) for model in models)
    if any(identity is None for _, identity in identities):
        return None
    semantic = phase.configuration(
        (
            "residual-ffn-tail/1",
            tuple(sorted(identities)),
            tuple(_layout(t) for t in readonly),
            _layout(output),
            intent.eps,
            kernel_identity,
            library.identity,
            library.family,
            up.module.encoding,
            down.module.encoding,
            tuple(choices),
            tuple(geometries.values()),
        )
    )
    if semantic is None:
        return None
    key = "runtime.split.residual_ffn/1:" + hashlib.sha256(semantic.encode()).hexdigest()
    request.memory.remember_working_phase(key, models or (request,))
    stream = torch.cuda.current_stream(device)
    selection = vt.split_select(
        device, key, choices, preferred=choices[0][0], stream=int(stream.cuda_stream)
    )
    geometry = geometries[str(selection["choice_id"])]
    record = request.working_splits.setdefault(
        key, {"site": "residual-ffn-tail", "key": key, "selections": 0}
    )
    record.update(
        selection=dict(selection),
        row_chunk=geometry.rows,
        dispatch={
            name: vars(getattr(geometry, name)) for name in ("up", "up_tail", "down", "down_tail")
        },
        workspace_bytes=geometry.workspace,
        output_is_outside_live=True,
    )
    record["selections"] += 1
    phase.choice(("native-split", key, geometry.identity))
    observation_id = phase.begin_split(
        device,
        key,
        geometry.identity,
        selection,
        stream=int(stream.cuda_stream),
    )
    submitted = False
    try:
        _execute(intent, x, output, norm, tables, index, up, down, geometry, library, stream)
        submitted = True
        return output
    except BaseException as error:
        if fault(error) is Fault.OOM:
            error.varena_no_retry = True  # type: ignore[attr-defined]
        raise
    finally:
        record["submission"] = vt.split_end(device, observation_id, submitted=submitted)


def _execute(
    intent: ResidualFFN,
    x: Any,
    output: Any,
    norm: Any,
    tables: tuple[Any, ...],
    index: Any,
    up: _Projection,
    down: _Projection,
    geometry: _Geometry,
    library: prebuilt.Prebuilt,
    stream: Any,
) -> None:
    """All allocation precedes the first output mutation; every chunk reuses this storage."""
    device, width, rows = x.device.index or 0, int(x.shape[-1]), int(x.shape[1])
    inner, chunk = int(down.data.shape[1]), geometry.rows
    stream_id = int(stream.cuda_stream)
    gate_msa, scale, shift, gate = tables
    with torch.cuda.device(device), torch.cuda.stream(stream):
        normed = torch.empty((chunk, width), dtype=x.dtype, device=x.device)
        packed = torch.empty((chunk, 2 * inner), dtype=x.dtype, device=x.device)
        activated = torch.empty((chunk, inner), dtype=x.dtype, device=x.device)
        ffn = torch.empty((chunk, width), dtype=x.dtype, device=x.device)
        q = (
            torch.empty(geometry.quantized_bytes, dtype=torch.int8, device=x.device)
            if geometry.quantized_bytes
            else None
        )
        scales = (
            torch.empty(geometry.scale_rows, dtype=torch.float32, device=x.device)
            if geometry.scale_rows
            else None
        )
        workspace = (
            torch.empty(geometry.workspace, dtype=torch.uint8, device=x.device)
            if geometry.workspace
            else None
        )
        for at in range(0, rows, chunk):
            catch.checkpoint()
            count = min(chunk, rows - at)
            residual = x[0].narrow(0, at, count)
            target = output[0].narrow(0, at, count)
            positions = index.narrow(0, at, count)
            n, both, activation, result = (
                buffer.narrow(0, 0, count) for buffer in (normed, packed, activated, ffn)
            )
            # Package-declared ABI and exact prebuilt kernel/constexprs; FP8 stays off.
            intent.launch(
                intent.norm_mod.kernel,
                (count, 1, 1),
                residual,
                target,
                gate_msa,
                target,
                n,
                n,
                norm,
                scale,
                shift,
                positions,
                width,
                count,
                width,
                int(scale.stride(0)),
                float(intent.eps),
                1,
                0,
                intent.norm_mod.constexprs["BLOCK"],
            )
            up_dispatch = geometry.up if count == chunk else geometry.up_tail
            down_dispatch = geometry.down if count == chunk else geometry.down_tail
            _Invocation(
                up,
                n.unsqueeze(0),
                both.unsqueeze(0),
                up_dispatch,
                q,
                scales,
                workspace,
                library,
                stream_id,
            ).call()
            intent.launch(
                intent.swiglu.kernel,
                (count, 1, 1),
                both,
                activation,
                activation,
                inner,
                2 * inner,
                inner,
                0,
                intent.swiglu.constexprs["BLOCK"],
            )
            _Invocation(
                down,
                activation.unsqueeze(0),
                result.unsqueeze(0),
                down_dispatch,
                q,
                scales,
                workspace,
                library,
                stream_id,
            ).call()
            intent.launch(
                intent.gate_add.kernel,
                (count, 1, 1),
                target,
                result,
                gate,
                target,
                positions,
                width,
                count,
                width,
                int(gate.stride(0)),
                intent.gate_add.constexprs["BLOCK"],
            )
