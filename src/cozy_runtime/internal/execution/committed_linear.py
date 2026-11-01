"""An explicit whole phase containing one raw-source native INT8 projection.

Inventory reads metadata only. Calling the inventory enters ordinary source hooks once,
then commits and executes precisely one numerical operation. Opaque loops never opt in.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import tensorfs.varena.torch as vt
import torch
from torch import Tensor

from cozy_runtime import _kernels, lora
from cozy_runtime.author._errors import Fault, fault
from cozy_runtime.author.kernels import _LINEAR_SPLIT, _RECENT, _SHARED, INT8_CONVROT, read
from cozy_runtime.internal import int8_linear, prebuilt
from cozy_runtime.internal.encoding.leaves import _native_leaf_class
from cozy_runtime.internal.execution import catch, phase
from cozy_runtime.internal.execution.native_linear_split import _chunks, _Dispatch, _Geometry, _Plan
from cozy_runtime.internal.residency import _REQUEST, Model, _Block


class Unsupported(ValueError):
    """The explicit phase is not closed under the supported producer contract."""


def _metadata(tensor: Any) -> object:
    if tensor is None:
        return None
    # In particular, do not ask a Placeholder for data_ptr, read or an actual storage.
    return tuple(tensor.shape), tuple(tensor.stride()), str(tensor.dtype), str(tensor.device)


def _identity(model: Model) -> tuple[Any, ...]:
    if (model.closed or model.retiring or model.fill is not None or model.bind is not None
            or model.finals is not None or model.wanted_identity() is None):
        raise Unsupported("closed linear phases require an established raw stored representation")
    return model.content, model.wanted_identity(), model.layout_identity


def _hooks(model: Model, module: Any) -> tuple[Any, ...]:
    global_hooks = torch.nn.modules.module
    if (global_hooks._global_forward_pre_hooks or global_hooks._global_forward_hooks
            or module._compiled_call_impl is not None
            or "forward" in vars(module) or "linear" in vars(module)
            or module._backward_hooks or module._backward_pre_hooks):
        raise Unsupported("closed linear phases cannot contain opaque module hooks")
    pre, post = tuple(module._forward_pre_hooks.values()), tuple(module._forward_hooks.values())
    if (any(hook not in model.source_hooks for hook in pre)
            or any(hook not in model.source_hooks
                   and not (type(hook) is lora.Hook and hook._cozy_baked) for hook in post)):
        raise Unsupported("closed linear phases accept only their source and baked identity hooks")
    return pre, post


def _slots(model: Model, module: Any) -> tuple[tuple[str, str, object], ...]:
    result: list[tuple[str, str, object]] = []
    for role in ("data", "scale", "bias"):
        value = getattr(module, role)
        if value is None:
            result.append((role, "", None))
            continue
        names = sorted(
            name for name, sites in model.sites.items()
            if any(site.table is module._buffers and site.name == role for site in sites)
        )
        if not names or not any(
            (id(module._buffers), role) in block.raw_views for block in model.groups()
        ):
            raise Unsupported("a projection operand has no declared raw source layout")
        result.append((role, names[0], _metadata(value)))
    return tuple(result)


def _bound(model: Model, module: Any, role: str, tensor: Any) -> bool:
    if tensor is None:
        return True
    # Validate source membership as well as shape: an unrelated same-shaped buffer is not
    # the declared binding. Raw-view offsets are established without materializing storage.
    for block in model.groups():
        view = block.raw_views.get((id(module._buffers), role))
        if view is not None and block.region is not None:
            offset, _ = view
            return int(tensor.data_ptr()) == int(block.region.ptr) + offset
    return False


def _environment(model: Model, module: Any, x: Tensor) -> Any:
    request = _REQUEST.get()
    if (request is None or request.idle or request.memory is not model.memory
            or not request.memory.cuda or x.device != model.memory.device
            or not x.is_cuda or type(x) is not Tensor or not x.is_contiguous()
            or x.layout is not torch.strided or x.dim() not in (2, 3)
            or (x.dim() == 3 and x.shape[0] != 1) or not x.numel() or x.data_ptr() % 16
            or x.dtype not in (torch.float16, torch.bfloat16)
            or torch.is_grad_enabled() or torch.is_autocast_enabled("cuda")
            or _LINEAR_SPLIT.get() is not None or phase._ACTIVE.get() is not None
            or catch.uncommitted_scope() or torch.cuda.is_current_stream_capturing()
            or (torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1)):
        raise Unsupported("closed linear execution requires a standalone eager CUDA request")
    if type(module) is not _native_leaf_class(torch):
        raise Unsupported("closed linear execution requires the declared native INT8 leaf")
    leaf: Any = module
    if leaf.provider != INT8_CONVROT or leaf.out_dtype != x.dtype:
        raise Unsupported("closed linear execution requires the declared native INT8 leaf")
    held = getattr(_RECENT, "held", None)
    if (any(shared.tensor is x for shared in _SHARED.get())
            or (held is not None and held.tensor is not None and held.tensor() is x)):
        raise Unsupported("an explicit linear phase cannot consume a shared quantization owner")
    return request


@dataclass(frozen=True)
class LinearPhase:
    model: Model
    module: Any
    x: Tensor
    source: tuple[Any, ...]
    slots: tuple[tuple[str, str, object], ...]
    hooks: tuple[Any, ...]
    input_layout: object
    library: prebuilt.Prebuilt
    library_identity: str
    geometries: tuple[_Geometry, ...]
    choices: tuple[tuple[str, tuple[int, ...]], ...]
    configuration: str
    site: str
    next_block: _Block | None
    stream: Any

    def __call__(self) -> Tensor:
        """Execute this explicit whole phase; no warm-up forward or inner fallback."""
        request = _environment(self.model, self.module, self.x)
        if _identity(self.model) != self.source or _hooks(self.model, self.module) != self.hooks:
            raise RuntimeError("the current source or hooks differ from the declared linear phase")
        invocation = _Invocation(self, request)
        # Source/lifetime preparation is outside the committed compute phase. An outer
        # owner also covers leaves nested below their block's normal forward entry point.
        with self.model.passing(self.site, executing=True):
            owner = _LINEAR_SPLIT.set(invocation)
            try:
                result = self.module(self.x)
            finally:
                _LINEAR_SPLIT.reset(owner)
        if invocation.calls != 1 or result is not invocation.result:
            raise RuntimeError("the explicit linear phase did not execute its one declared call")
        return result


@dataclass
class _Invocation:
    inventory: LinearPhase
    request: Any
    calls: int = 0
    result: Tensor | None = None

    @property
    def module(self) -> Any:
        return self.inventory.module

    def run(self, x: Tensor, operand: object, out: Tensor | None) -> Tensor:
        inv = self.inventory
        self.calls += 1
        if self.calls != 1 or x is not inv.x or operand is not None or out is not None:
            raise RuntimeError("the explicit linear phase received a different invocation")
        # These are the original leaf's real source reads. They may fail, but cannot train
        # the compute sample: no Execution has entered yet and no output has been allocated.
        data, scale, bias = read(inv.module.data), read(inv.module.scale), read(inv.module.bias)
        device, stream_id = x.device.index or 0, int(inv.stream.cuda_stream)
        if (_identity(inv.model) != inv.source or inv.model.identity != inv.model.wanted_identity()
                or _slots(inv.model, inv.module) != inv.slots
                or _hooks(inv.model, inv.module) != inv.hooks
                or _metadata(x) != inv.input_layout
                or int(torch.cuda.current_stream(device).cuda_stream) != stream_id
                or int8_linear.serving(device) != int8_linear.OP
                or int8_linear.library(device) is not inv.library
                or inv.library.identity != inv.library_identity):
            raise RuntimeError("the actual binding differs from the current linear inventory")
        for (role, _, layout), actual in zip(inv.slots, (data, scale, bias), strict=True):
            if (_metadata(actual) != layout or not _bound(inv.model, inv.module, role, actual)
                    or (actual is not None and actual.data_ptr() % 16)):
                raise RuntimeError(f"the actual {role} binding differs from its declared layout")
        # Metadata checks above prevent _weight's optional bias cast/copy. This only validates
        # the actual aligned raw pointers against the already-declared native requirements.
        inv.library._weight(prebuilt.INT8, data, scale, bias, x.shape[-1], x.dtype)
        execution = vt.commit_execution(
            device, inv.configuration, inv.site, inv.choices,
            preferred=inv.choices[0][0], stream=stream_id,
        )
        selection = execution.selection
        geometry = next(item for item in inv.geometries if item.identity == selection["choice_id"])
        plan = _Plan(inv.module, x, data, scale, bias, inv.library, self.request,
                     inv.stream, inv.site, geometry, selection)
        self.request.memory.remember_working_phase(execution.key, (inv.model,))
        self.request.memory.remember_working_phase(inv.site, (inv.model,))
        record = {"site": inv.site, "key": execution.key, "selection": selection,
                  "row_chunk": geometry.rows, "invocations": self.calls,
                  "source_scope": "prepared_before_compute", "completion": "whole_phase_stream"}
        self.request.working_splits[inv.site] = record
        try:
            execution.enter()
            # Only native measured admission may speculate. The explicit producer names
            # its next raw block; it does not learn future order from an earlier trace.
            if inv.next_block is not None:
                self.request.memory._prefetch_block(inv.next_block, self.request)
            execution.begin()
            self.result = plan._execute()
            record["submission"] = execution.end(submitted=True)
            inv.stream.synchronize()  # Completion of this explicit whole phase, once.
            observed = execution.finish(completed=True)
            self.request.working_phases.append(observed)
            record["phase"] = observed
        except BaseException as error:
            if fault(error) is Fault.OOM:
                error.varena_no_retry = True  # type: ignore[attr-defined]
            aborted = execution.abort()
            if aborted is not None:
                self.request.working_phases.append(aborted)
                record["phase"] = aborted
            raise
        # The optional source reader has its own checked completion, outside compute.
        # Finalize while this Model owner is still open; ordinary close otherwise cancels.
        self.request.memory._settle_prefetch()
        return self.result


def inventory(
    model: Model, module: Any, x: Tensor, *, next_block: str | None = None,
) -> LinearPhase:
    """Describe the current operation without reading, filling, mapping or pinning weights."""
    _environment(model, module, x)
    source, hooks, slots = _identity(model), _hooks(model, module), _slots(model, module)
    width, rows = int(x.shape[-1]), x.numel() // x.shape[-1]
    data, scale, bias = module.data, module.scale, module.bias
    if (rows < 2 or rows > 2**31 - 1 or width % 256
            or data.dtype != torch.int8 or data.dim() != 2 or data.shape[1] != width
            or not data.is_contiguous() or data.shape[0] % 8
            or scale.dtype != torch.float32 or scale.numel() != data.shape[0]
            or not scale.is_contiguous()
            or (bias is not None and (bias.dtype != x.dtype or not bias.is_contiguous()
                                     or bias.numel() != data.shape[0]))):
        raise Unsupported("the raw projection layout is not supported by the native provider")
    device, channels = x.device.index or 0, int(data.shape[0])
    if int8_linear.serving(device) != int8_linear.OP or not hasattr(_kernels._gemm, "int8_plan"):
        raise Unsupported("the actual native INT8 query provider is not installed")
    library = int8_linear.library(device)
    if library.family != "mma.sync" or not vt.split_context(device)["eligible"]:
        raise Unsupported("the current kernel or execution scope cannot be committed")
    queried: dict[int, _Dispatch] = {}

    def query(count: int) -> _Dispatch:
        if count not in queried:
            info = _kernels._gemm.int8_plan(device, x.dtype == torch.float16,
                                           count, channels, width, channels)
            library._check(info["status"], "the explicit linear inventory")
            queried[count] = _Dispatch(count, info["kind"], info["config"], info["workspace_bytes"])
        return queried[count]

    geometries, choices = [], []
    for chunk in _chunks(rows):
        full, tail = query(chunk), query(rows % chunk or chunk)
        quantized = max((part.rows for part in (full, tail) if part.kind == 0), default=0)
        workspace = max(full.workspace, tail.workspace)
        geometry = _Geometry(chunk, full, tail, quantized, workspace)
        allocations = [rows * channels * x.element_size()]
        if quantized:
            allocations.extend((quantized * width, quantized * 4))
        if workspace:
            allocations.append(workspace)
        geometries.append(geometry)
        choices.append((geometry.identity, tuple(allocations)))
    following = None if next_block is None else model.blocks.get(next_block)
    if next_block is not None and (following is None or following.aliases is None):
        raise Unsupported("the next raw block has no established source layout")
    configuration = phase.configuration((
        "runtime.closed-linear/1", source, slots, _metadata(x),
        module.provider, module.encoding, library.identity, library.family,
        tuple(choices), tuple(geometries), next_block,
    ))
    assert configuration is not None
    site = "runtime.closed-linear/1:" + hashlib.sha256(configuration.encode()).hexdigest()
    return LinearPhase(model, module, x, source, slots, hooks, _metadata(x), library,
                       library.identity, tuple(geometries), tuple(choices), configuration,
                       site, following, torch.cuda.current_stream(device))
