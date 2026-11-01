"""The declared fused layer-scale block: metadata first, one bound invocation, one choice.

The first closed producer is the actual2048-wide FP16-autocast VAE layout with FP32
residuals, two RMS norms, six biased dense projections and unmasked Kitchen INT8.
An enclosing legacy cut remains unsupported; its geometry is not a committed plan.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import tensorfs.varena.torch as vt
import torch
from torch import Tensor, nn

from cozy_runtime.author._errors import Fault, fault
from cozy_runtime.author._split import Cut
from cozy_runtime.author.kernels import _BLOCK_SPLIT, LayerScaleBlock, read
from cozy_runtime.internal import attention, dense_linear
from cozy_runtime.internal.execution import catch, phase
from cozy_runtime.internal.execution.native_heads import _Attention, _attention, _route
from cozy_runtime.internal.execution.native_linear_split import _chunks, _layout
from cozy_runtime.internal.residency import _REQUEST, Model, Placeholder

if TYPE_CHECKING:
    from cozy_runtime import fusion

class _Unsupported(ValueError):
    pass


def _metadata(value: Any) -> tuple[Any, ...]:
    return tuple(value.shape), tuple(value.stride()), str(value.dtype), str(value.device)


@dataclass(frozen=True)
class _Slot:
    module: Any
    name: str
    value: Any
    metadata: tuple[Any, ...]
    alignment: int
    model: Any = None
    block: Any = None
    offset: int = 0

    def bind(self) -> Tensor:
        current = self.module._parameters[self.name]
        value = read(current)
        if (
            type(value) not in (Tensor, nn.Parameter)
            or _metadata(value) != self.metadata
            or not value.is_contiguous()
            or value.data_ptr() == 0
            or value.data_ptr() % self.alignment
        ):
            raise RuntimeError("declared VAE source binding changed before execution")
        if self.model is None:
            if current is not self.value:
                raise RuntimeError("declared VAE parameter changed before execution")
        elif (
            self.block.region is None
            or value.data_ptr() != int(self.block.region.ptr) + self.offset
        ):
            raise RuntimeError("declared VAE parameter is not its current raw source binding")
        return value


def _slot(
    module: Any,
    name: str,
    shape: tuple[int, ...],
    dtypes: tuple[Any, ...],
    models: tuple[Model, ...],
    *,
    matrix: bool = False,
) -> _Slot:
    value = module._parameters.get(name)
    if (
        type(value) not in (Tensor, nn.Parameter, Placeholder)
        or tuple(value.shape) != shape
        or value.dtype not in dtypes
        or tuple(value.stride()) != tuple(math.prod(shape[i + 1 :]) for i in range(len(shape)))
    ):
        raise _Unsupported("VAE source slot does not match its declared dense layout")
    alignment = 256 if matrix and value.dtype is torch.float16 else 16
    for model in models:
        if not any(
            site.table is module._parameters and site.name == name
            for sites in model.sites.values()
            for site in sites
        ):
            continue
        if (
            model.closed
            or model.retiring
            or model.fill is not None
            or model.bind is not None
            or model.finals is not None
            or model.wanted_identity() is None
        ):
            raise _Unsupported(
                "VAE native source requires an established raw stored representation"
            )
        for block in model.groups():
            view = block.raw_views.get((id(module._parameters), name))
            if view is not None:
                offset, _ = view
                if offset % alignment:
                    raise _Unsupported("VAE raw source offset does not meet the kernel alignment")
                return _Slot(module, name, value, _metadata(value), alignment, model, block, offset)
        raise _Unsupported("VAE source slot has no raw address layout")
    if type(value) not in (Tensor, nn.Parameter) or hasattr(value, "_varena_read"):
        raise _Unsupported("VAE placeholder has no current declared owner")
    if value.data_ptr() % alignment:
        raise _Unsupported("VAE parameter does not meet the kernel alignment")
    return _Slot(module, name, value, _metadata(value), alignment)


def _source_identity(models: tuple[Model, ...]) -> tuple[Any, ...]:
    identities = tuple(
        (model.content, model.wanted_identity(), model.layout_identity) for model in models
    )
    if any(identity is None for _, identity, _ in identities):
        raise _Unsupported("VAE source has no declared current identity")
    return tuple(sorted(identities))


@dataclass(frozen=True)
class _Inventory:
    intent: LayerScaleBlock
    request: Any
    site: fusion._Site
    models: tuple[Model, ...]
    source: tuple[Any, ...]
    slots: tuple[_Slot, ...]
    policy: dense_linear.Policy
    attention: _Attention
    input_layout: object
    rotary_layout: tuple[object, ...]

    @property
    def module(self) -> Any:
        return self.intent.module


def _inventory(intent: LayerScaleBlock, out: Any, cuts: tuple[Cut, ...]) -> _Inventory:
    request = _REQUEST.get()
    x, module = intent.hidden, intent.module
    if (
        request is None
        or request.idle
        or not request.memory.cuda
        or out is not None
        or cuts != (Cut(0),)
        or type(x) is not Tensor
        or not x.is_cuda
        or x.device != request.memory.device
        or x.dim() != 3
        or x.shape[-1] != 2048
        or min(x.shape) < 1
        or x.shape[1] < 2
        or not x.is_contiguous()
        or x.data_ptr() % 16
        or x.dtype not in (torch.float16, torch.float32)
        or torch.is_grad_enabled()
        or catch.uncommitted_scope()
        or _BLOCK_SPLIT.get() is not None
        or torch.cuda.is_current_stream_capturing()
        or (torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1)
    ):
        raise _Unsupported("VAE requires an independent eager CUDA batch call")
    from cozy_runtime import fusion
    from cozy_runtime.author.kernels import _SETS
    from cozy_runtime.internal.triton_kernels import _LOADED, _Launcher

    device = x.device.index or 0
    policy = dense_linear.policy(device)
    if policy is None or not vt.split_context(device)["eligible"]:
        raise _Unsupported("VAE dense math/admission policy is not supported")
    site = getattr(module, "_cozy_dit", None)
    if (
        type(site) is not fusion._Site
        or type(module).forward is not fusion._block_forward
        or module.training
        or module._compiled_call_impl is not None
        or (
            "forward" in vars(module)
            and module.forward is not vars(module).get("_cozy_execution_forward")
        )
    ):
        raise _Unsupported("VAE requires its installed ordinary fused forward")
    layout = site.layout
    if (
        layout.width != 2048
        or layout.head_dim != 64
        or layout.rows != "fp16"
        or layout.norm != "rms"
        or layout.rms_rounding != "once"
        or layout.centered
        or layout.parallel
        or not layout.composed
        or layout.scale
        or layout.shift
        or layout.gate != "identity"
        or layout.qk_affine
        or layout.mlp.up is not None
        or not layout.mlp.value_first
        or layout.rope is None
        or layout.rope.interleaved
        or layout.rotary != 48
        or layout.rope.rounding != "product"
        or site.switch.launch is None
        or len(site.norms) != 2
        or len(site.inputs) != 1
    ):
        raise _Unsupported("VAE fused layout is outside the complete producer contract")
    if (
        type(site.switch.launch) is not _Launcher
        or _LOADED.get((layout.kernels.name, device)) is not site.switch.launch
        or getattr(_SETS.get(layout.kernels.name), "identity", None) != layout.kernels.identity
    ):
        raise _Unsupported("VAE requires its actual ready compiled fused kernel set")
    if any(
        type(norm) is not nn.RMSNorm
        or norm.weight is not None
        or tuple(norm.normalized_shape) != (64,)
        for norm in site.qk
    ):
        raise _Unsupported("VAE head norms differ from the declared weightless operation")
    if any(
        type(norm) is not nn.RMSNorm or tuple(norm.normalized_shape) != (2048,)
        for norm in site.norms
    ):
        raise _Unsupported("VAE row norms differ from the declared operation")
    if any(
        not math.isfinite(float(eps)) or eps < 0 for eps in (*site.eps, *(n.eps for n in site.qk))
    ):
        raise _Unsupported("VAE normalization epsilon is invalid")
    processor = site.attention.processor
    route = _route(processor._attention_backend, int(x.shape[1]))
    if (
        route is None
        or route.candidate is not attention.BY_NAME["kitchen-int8"]
        or getattr(processor, "_parallel_config", None) is not None
    ):
        raise _Unsupported("VAE requires the actually selected single-GPU Kitchen INT8 route")
    kernel = _attention(route, device)
    if kernel is None:
        raise _Unsupported("VAE selected Kitchen artifact is not ready")
    rotary = intent.rotary
    if (
        not isinstance(rotary, tuple)
        or len(rotary) != 2
        or any(
            type(t) is not Tensor
            or t.dtype is not torch.float32
            or t.device != x.device
            or not t.is_contiguous()
            or t.data_ptr() % 16
            or tuple(t.shape[-2:]) != (x.shape[1], 48)
            or t.numel() != x.shape[1] * 48
            for t in rotary
        )
    ):
        raise _Unsupported("VAE rotary must be the complete prepared FP32 split-half tables")
    with request.memory.lock:
        # A standalone declared block has not entered its source hook yet. Resolve
        # ownership from real source tables, without binding any model's weights.
        models = tuple(request.memory.models.values())
    linears = (*site.qkv, site.out, site.inputs[0], site.down)
    shapes = ((2048, 2048),) * 4 + ((16384, 2048), (2048, 8192))
    slots: list[_Slot] = []
    for leaf, (channels, width) in zip(linears, shapes, strict=True):
        if (
            type(leaf) is not nn.Linear
            or leaf.training
            or "forward" in vars(leaf)
            or leaf._compiled_call_impl is not None
            or leaf._forward_pre_hooks
            or leaf._forward_hooks
            or leaf._backward_hooks
            or leaf._backward_pre_hooks
        ):
            raise _Unsupported("VAE dense children must have their plain inference forward")
        slots.append(
            _slot(
                leaf,
                "weight",
                (channels, width),
                (torch.float16, torch.float32),
                models,
                matrix=True,
            )
        )
        slots.append(_slot(leaf, "bias", (channels,), (torch.float16, torch.float32), models))
    slots += [_slot(norm, "weight", (2048,), (torch.float32,), models) for norm in site.norms]
    slots += [
        _slot(module, name, (2048,), (torch.float32,), models)
        for name in (intent.gate_msa, intent.gate_mlp)
    ]
    actual_owners = {id(slot.model) for slot in slots if slot.model is not None}
    models = tuple(model for model in models if id(model) in actual_owners)
    source = _source_identity(models)
    global_hooks = nn.modules.module
    owned = {id(hook) for model in models for hook in model.source_hooks}
    if (
        global_hooks._global_forward_pre_hooks
        or global_hooks._global_forward_hooks
        or module._backward_hooks
        or module._backward_pre_hooks
        or any(
            id(hook) not in owned
            for hook in (*module._forward_pre_hooks.values(), *module._forward_hooks.values())
        )
    ):
        raise _Unsupported("VAE block has an opaque hook")
    if any(slot.metadata[3] != str(x.device) for slot in slots):
        raise _Unsupported("VAE source slots must declare the input's CUDA device")
    return _Inventory(
        intent,
        request,
        site,
        models,
        source,
        tuple(slots),
        policy,
        kernel,
        _layout(x),
        tuple(_layout(t) for t in rotary),
    )


def _attention_shapes(batch: int, tokens: int) -> tuple[tuple[tuple[int, ...], Any], ...]:
    return (
        ((batch, 32, tokens, 64), torch.int8),
        ((batch, 32, tokens, 64), torch.int8),
        ((batch * 32 * 64, ((tokens + 63) // 64) * 64), torch.int8),
        ((batch, 32, ((tokens + 127) // 128) * 32), torch.float32),
        ((batch, 32, ((tokens + 63) // 64) * 4), torch.float32),
        ((batch * 32 * 64,), torch.float32),
        ((batch, 32), torch.int32),
    )


@dataclass(frozen=True)
class _Variant:
    batch: int
    full: tuple[Any, ...]
    tail: tuple[Any, ...]
    tail_batch: int
    workspace: int
    allocations: tuple[int, ...]
    identity: str


@dataclass
class _Buffers:
    output: Tensor
    packed: Tensor
    expanded: Tensor
    weight: Tensor | None
    workspace: Tensor | None
    attention: tuple[Tensor, ...]
    batch: int
    tokens: int

    @classmethod
    def allocate(cls, x: Tensor, variant: _Variant, weight_bytes: int) -> _Buffers:
        batch, tokens, _ = x.shape
        at = variant.batch
        return cls(
            torch.empty((batch, tokens, 2048), dtype=torch.float32, device=x.device),
            torch.empty((at * tokens, 16384), dtype=torch.float16, device=x.device),
            torch.empty((at * tokens, 8192), dtype=torch.float16, device=x.device),
            torch.empty(weight_bytes // 2, dtype=torch.float16, device=x.device)
            if weight_bytes
            else None,
            torch.empty(variant.workspace, dtype=torch.uint8, device=x.device)
            if variant.workspace
            else None,
            tuple(
                torch.empty(shape, dtype=dtype, device=x.device)
                for shape, dtype in _attention_shapes(at, tokens)
            ),
            at,
            tokens,
        )

    def narrow(self, batch: int) -> tuple[tuple[Tensor, ...], Tensor, Tensor, tuple[Tensor, ...]]:
        size = self.batch * self.tokens * 2048
        flat = self.packed.view(-1)
        rows = tuple(
            flat.narrow(0, i * size, size).view(self.batch, self.tokens, 2048)[:batch]
            for i in range(5)
        )
        up = flat[: batch * self.tokens * 16384].view(batch * self.tokens, 16384)
        expanded = self.expanded.view(-1)[: batch * self.tokens * 8192].view(
            batch * self.tokens, 8192
        )
        qi, ki, vi, qs, ks, vs, anchors = self.attention
        scratch = (
            qi[:batch],
            ki[:batch],
            vi[: batch * 32 * 64],
            qs[:batch],
            ks[:batch],
            vs[: batch * 32 * 64],
            anchors[:batch],
        )
        return rows, up, expanded, scratch


def _arguments(
    plan: Any, x: Tensor, weight: Tensor, out: Tensor, workspace: Tensor | None, stream: int
) -> tuple[Any, ...]:
    return (
        x.data_ptr(),
        x.numel() * 2,
        weight.data_ptr(),
        weight.numel() * 2,
        out.data_ptr(),
        out.numel() * 2,
        0 if workspace is None else workspace.data_ptr(),
        0 if workspace is None else workspace.numel(),
        stream,
    )


def _linear(
    plan: Any,
    x: Tensor,
    weight: Tensor,
    bias: Tensor,
    out: Tensor,
    buffers: _Buffers,
    bias_buffer: Tensor,
    stream: int,
) -> None:
    if weight.dtype is torch.float32:
        assert buffers.weight is not None
        operand = buffers.weight[: weight.numel()].view(weight.shape)
        operand.copy_(weight)
    else:
        operand = weight
    bias_buffer[: bias.numel()].copy_(bias)
    dense_linear.checked(
        plan.execute, *_arguments(plan, x, operand, out, buffers.workspace, stream)
    )


def _attend(
    kernel: _Attention,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    output: Tensor,
    target: Tensor,
    scratch: tuple[Tensor, ...],
    stream: int,
) -> None:
    qi, ki, vi, qs, ks, vs, anchors = scratch

    def capsule(tensor: Tensor) -> Any:
        return tensor.__dlpack__(stream=-1)

    kernel.extension.sage_sdpa_quantize(
        capsule(q.transpose(1, 2)),
        capsule(k.transpose(1, 2)),
        capsule(v.transpose(1, 2)),
        capsule(qi),
        capsule(qs),
        capsule(ki),
        capsule(ks),
        capsule(vi),
        capsule(vs),
        64,
        1,
        stream,
        anchors.data_ptr(),
    )
    kernel.extension.sage_sdpa_prequantized(
        capsule(qi),
        capsule(ki),
        capsule(vi),
        capsule(output),
        capsule(qs),
        capsule(ks),
        capsule(vs),
        64,
        0.125,
        1,
        stream,
        None,
    )
    # Same-device transpose copy; the real consumer census must prove no hidden storage.
    target.copy_(output.transpose(1, 2))


def _variants(
    x: Tensor,
    slots: tuple[_Slot, ...],
    bias: Tensor,
    policy: dense_linear.Policy,
    context: Any,
    stream: int,
) -> tuple[tuple[_Variant, ...], int, tuple[Any, ...]]:
    batch, tokens, _ = (int(n) for n in x.shape)
    cast_bytes = max(
        (math.prod(s.metadata[0]) * 2 for s in slots if s.metadata[2] == "torch.float32"), default=0
    )
    queries: dict[tuple[int, int, int], tuple[Any, Any]] = {}

    def dense(at: int) -> tuple[Any, ...]:
        result = []
        for slot in slots:
            channels, width = slot.metadata[0]
            geometry = (at * tokens, int(width), int(channels))
            if geometry not in queries:
                plan = dense_linear.checked(
                    context.query,
                    *geometry,
                    bias.data_ptr(),
                    bias.numel() * 2,
                    policy.half_accumulation,
                    policy.reduction,
                    policy.sm_target,
                    policy.workspace_limit,
                    stream,
                )
                queries[geometry] = plan, plan.info()
            result.append(queries[geometry][0])
        return tuple(result)

    result = []
    for at in _chunks(batch):
        tail_batch = batch % at
        full = dense(at)
        tail = dense(tail_batch) if tail_batch else full
        workspace = max(plan.info()["workspace_bytes"] for plan in (*full, *tail))
        allocations = [batch * tokens * 2048 * 4, at * tokens * 16384 * 2, at * tokens * 8192 * 2]
        if cast_bytes:
            allocations.append(cast_bytes)
        if workspace:
            allocations.append(workspace)
        allocations.extend(
            math.prod(shape) * (1 if dtype is torch.int8 else 4)
            for shape, dtype in _attention_shapes(at, tokens)
        )
        result.append(
            _Variant(
                at,
                full,
                tail,
                tail_batch,
                workspace,
                tuple(allocations),
                f"batch:{at}/tail:{tail_batch}",
            )
        )
    # The current compute recipe includes actual algorithms/library/policy, never the
    # incidental bias or Region address. Those addresses are checked separately.
    facts = tuple(
        (
            geometry,
            info["algorithm"],
            info["workspace_bytes"],
            info["library_version"],
            info["header_version"],
        )
        for geometry, (_, info) in sorted(queries.items())
    )
    return tuple(result), cast_bytes, facts


def _validate_all(
    x: Tensor, variant: _Variant, weights: tuple[Tensor, ...], buffers: _Buffers, stream: int
) -> None:
    for batch, plans in ((variant.batch, variant.full), (variant.tail_batch, variant.tail)):
        if not batch:
            continue
        rows, up, expanded, _ = buffers.narrow(batch)
        normalized = expanded.view(-1)[: batch * buffers.tokens * 2048].view(
            batch * buffers.tokens, 2048
        )
        pairs = (
            (normalized, rows[0]),
            (normalized, rows[1]),
            (normalized, rows[2]),
            (rows[1], rows[2]),
            (normalized, up),
            (expanded, rows[0]),
        )
        for plan, weight, (source, target) in zip(plans, weights, pairs, strict=True):
            operand = weight
            if weight.dtype is torch.float32:
                assert buffers.weight is not None
                operand = buffers.weight[: weight.numel()].view(weight.shape)
            dense_linear.checked(
                plan.validate, *_arguments(plan, source, operand, target, buffers.workspace, stream)
            )
    # Kernel-only operands have the reviewed16-byte ABI. All are live disjoint working
    # allocations or the declared row-bank aliases; no normalization/copy has run yet.
    if any(
        t.data_ptr() % 16 or not t.is_contiguous()
        for t in (x, buffers.output, buffers.packed, buffers.expanded, *buffers.attention)
    ):
        raise RuntimeError("declared VAE caller storage changed before writes")


def _execute(
    inventory: _Inventory,
    values: tuple[Tensor, ...],
    variant: _Variant,
    buffers: _Buffers,
    bias: Tensor,
    index: Tensor,
    stream: int,
) -> Tensor:
    from cozy_runtime import fusion

    x, site = inventory.intent.hidden, inventory.site
    weights, biases = values[:12:2], values[1:12:2]
    norm1, norm2, gate1, gate2 = values[12:]
    launch, entries = site.switch.launch, site.layout.entries
    assert entries.second is not None and entries.rope is not None and launch is not None
    cos, sin = inventory.intent.rotary
    tables = (cos, sin, int(cos.stride(-2)), 1)
    for start in range(0, x.shape[0], variant.batch):
        batch = min(variant.batch, x.shape[0] - start)
        plans = variant.full if batch == variant.batch else variant.tail
        rows, up, expanded, scratch = buffers.narrow(batch)
        input_rows = x[start : start + batch]
        target = buffers.output[start : start + batch]
        normalized = expanded.view(-1)[: batch * buffers.tokens * 2048].view(
            batch * buffers.tokens, 2048
        )
        fusion._kernels.norm_modulate(
            launch,
            entries.first,
            input_rows,
            norm1,
            None,
            None,
            index,
            site.eps[0],
            norm_out=normalized,
        )
        for i in range(3):
            _linear(plans[i], normalized, weights[i], biases[i], rows[i], buffers, bias, stream)
        for i in range(2):
            fusion._kernels.head_norm_rope(
                launch,
                entries.rope,
                rows[i].view(batch, buffers.tokens, 32, 64),
                None,
                site.qk[i].eps,
                tables,
                torch.float16,
                out=rows[i + 3].view(batch * buffers.tokens, 2048),
            )
        _attend(
            inventory.attention,
            rows[3].view(batch, buffers.tokens, 32, 64),
            rows[4].view(batch, buffers.tokens, 32, 64),
            rows[2].view(batch, buffers.tokens, 32, 64),
            rows[0].view(batch, 32, buffers.tokens, 64),
            rows[1].view(batch, buffers.tokens, 32, 64),
            scratch,
            stream,
        )
        _linear(plans[3], rows[1], weights[3], biases[3], rows[2], buffers, bias, stream)
        fusion._kernels.norm_modulate(
            launch,
            entries.second,
            input_rows,
            norm2,
            None,
            None,
            index,
            site.eps[1],
            branch=rows[2],
            gate=gate1[None],
            hidden_out=target.view(batch * buffers.tokens, 2048),
            norm_out=normalized,
        )
        _linear(plans[4], normalized, weights[4], biases[4], up, buffers, bias, stream)
        value, gate = up.chunk(2, dim=-1)
        fusion._kernels.swiglu(launch, entries.swiglu, gate, value, out=expanded)
        _linear(plans[5], expanded, weights[5], biases[5], rows[0], buffers, bias, stream)
        fusion._kernels.norm_modulate(
            launch,
            entries.residual,
            target,
            None,
            None,
            None,
            index,
            0.0,
            branch=rows[0],
            gate=gate2[None],
            hidden_out=target.view(batch * buffers.tokens, 2048),
        )
    return buffers.output


@dataclass
class _Call:
    inventory: _Inventory
    bias: Tensor
    index: Tensor
    variants: tuple[_Variant, ...]
    cast_bytes: int
    algorithms: tuple[Any, ...]
    dense_identity: str
    stream: int
    calls: int = 0

    @property
    def module(self) -> Any:
        return self.inventory.module

    def run(self, *args: Any, **kwargs: Any) -> Tensor:
        inventory = self.inventory
        site = inventory.site
        assert site.layout.rope is not None
        self.calls += 1
        if self.calls != 1 or getattr(self.module, "_cozy_dit", None) is not inventory.site:
            raise RuntimeError("declared VAE forward was replaced or invoked more than once")
        bound = site.signature.bind(self.module, *args, **kwargs)
        bound.apply_defaults()
        if (
            bound.arguments[site.layout.hidden_states] is not inventory.intent.hidden
            or bound.arguments[site.layout.rope.argument] is not inventory.intent.rotary
            or _layout(inventory.intent.hidden) != inventory.input_layout
            or tuple(_layout(t) for t in inventory.intent.rotary) != inventory.rotary_layout
            or _source_identity(inventory.models) != inventory.source
        ):
            raise RuntimeError("declared VAE call or current source changed before binding")
        values = tuple(slot.bind() for slot in inventory.slots)
        if _source_identity(inventory.models) != inventory.source:
            raise RuntimeError("VAE source identity changed during binding")
        x, request = inventory.intent.hidden, inventory.request
        device = x.device.index or 0
        stream = int(torch.cuda.current_stream(device).cuda_stream)
        if stream != self.stream or dense_linear.policy(device) != inventory.policy:
            raise RuntimeError("VAE dense stream/policy changed before binding")
        variants, cast_bytes, algorithms = self.variants, self.cast_bytes, self.algorithms
        bias, index = self.bias, self.index
        choices = [(item.identity, item.allocations) for item in variants]
        semantic = phase.configuration(
            (
                "layer-scale-block/1",
                inventory.site.layout.kernels.identity,
                inventory.source,
                inventory.input_layout,
                inventory.rotary_layout,
                tuple(slot.metadata for slot in inventory.slots),
                inventory.site.eps,
                tuple(float(norm.eps) for norm in inventory.site.qk),
                inventory.attention.identity,
                asdict(inventory.policy),
                algorithms,
                choices,
                self.dense_identity,
            )
        )
        if semantic is None:
            raise RuntimeError("VAE current compute configuration cannot be serialized")
        key = "runtime.split.layer_scale_block/1:" + hashlib.sha256(semantic.encode()).hexdigest()
        request.memory.remember_working_phase(key, inventory.models or (request,))
        selection = vt.split_select(device, key, choices, preferred=choices[0][0], stream=stream)
        variant = next(item for item in variants if item.identity == selection["choice_id"])
        record = request.working_splits.setdefault(
            key, {"site": "fused-layer-scale-block", "key": key, "selections": 0}
        )
        record.update(
            batch_chunk=variant.batch,
            selection=dict(selection),
            choices=choices,
            algorithms=algorithms,
            attention_identity=inventory.attention.identity,
            preparation_allocations=(32768, 8),
            invocation_count=self.calls,
        )
        record["selections"] += 1
        phase.choice(("native-split", key, variant.identity))
        token = phase.begin_split(device, key, variant.identity, selection, stream=stream)
        submitted = False
        try:
            buffers = _Buffers.allocate(x, variant, cast_bytes)
            _validate_all(x, variant, values[:12:2], buffers, stream)
            if dense_linear.policy(device) != inventory.policy:
                raise RuntimeError("VAE dense policy changed before the first numerical write")
            result = _execute(inventory, values, variant, buffers, bias, index, stream)
            submitted = True
            return result
        finally:
            record["submission"] = vt.split_end(device, token, submitted=submitted)


def _prepare(inventory: _Inventory) -> _Call:
    from cozy_runtime._kernels import _dense
    from cozy_runtime.internal.execution.native_heads import _binary_identity

    x, memory = inventory.intent.hidden, inventory.request.memory
    device = x.device.index or 0
    stream = int(torch.cuda.current_stream(device).cuda_stream)
    context = getattr(memory, "_dense_context", None)
    if context is None:
        context = dense_linear.checked(_dense.Context, device, stream)
        memory._dense_context = context
    # These real execution holdings precede query and source invocation. No parameter
    # read or numerical model call is needed to enumerate the current complete menu.
    bias = torch.empty(16384, dtype=torch.float16, device=x.device)
    index = torch.zeros(1, dtype=torch.int64, device=x.device)
    variants, cast_bytes, algorithms = _variants(
        x, inventory.slots[:12:2], bias, inventory.policy, context, stream
    )
    binary = Path(_dense.__file__)
    stat = binary.stat()
    identity = _binary_identity(str(binary), stat.st_size, stat.st_mtime_ns)
    return _Call(inventory, bias, index, variants, cast_bytes, algorithms, identity, stream)


def split_layer_scale_block(
    intent: LayerScaleBlock, out: Any, cuts: tuple[Cut, ...]
) -> Tensor | None:
    try:
        inventory = _inventory(intent, out, cuts)
    except _Unsupported:
        return None
    from cozy_runtime._kernels import _dense

    if _dense.API != 1 or not hasattr(_dense, "NOT_SUPPORTED"):
        raise RuntimeError(
            "native VAE requires the prepared dense provider and typed query refusal")
    with torch.cuda.device(inventory.intent.hidden.device):
        try:
            call = _prepare(inventory)
        except RuntimeError as error:
            # A valid no-algorithm outcome declines before source hooks or numerical work.
            cause = error.__cause__
            if (
                isinstance(cause, _dense.Error)
                and cause.domain == "cublas"
                and cause.status == _dense.NOT_SUPPORTED
            ):
                return None
            if fault(error) in (Fault.OOM, Fault.EXHAUSTED):
                error.varena_no_retry = True  # type: ignore[attr-defined]
            raise
        token = _BLOCK_SPLIT.set(call)
        try:
            # Source/module hooks run once. The installed fused forward intercepts exactly
            # this numerical body after binding; preparation never invokes a warm forward.
            result = intent.module(intent.hidden, intent.rotary)
            if call.calls != 1:
                raise RuntimeError("declared VAE body was not reached exactly once")
            return result
        except BaseException as error:
            if fault(error) in (Fault.OOM, Fault.EXHAUSTED):
                error.varena_no_retry = True  # type: ignore[attr-defined]
            raise
        finally:
            _BLOCK_SPLIT.reset(token)
