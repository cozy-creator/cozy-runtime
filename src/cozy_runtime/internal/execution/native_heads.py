"""Complete projected H3 heads on the already-selected dense INT4QK kernel."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any

import tensorfs.varena.torch as vt
import torch
from torch import Tensor

from cozy_runtime import _kernels
from cozy_runtime.author._attention_scope import _ACTIVE_LAYOUT
from cozy_runtime.author._errors import Fault, fault
from cozy_runtime.author._split import Cut
from cozy_runtime.author.kernels import (
    _LINEAR_SPLIT,
    _RECENT,
    _SHARED,
    ProjectedHeads,
    TritonEntry,
    read,
)
from cozy_runtime.internal import attention, attention_sol, attention_ulysses, int8_linear, prebuilt
from cozy_runtime.internal.execution import catch, phase
from cozy_runtime.internal.execution.native_linear_split import (
    _Dispatch,
    _layout,
    launch_projection,
)
from cozy_runtime.internal.execution.native_residual_ffn import (
    _entry_identity,
    _leaf,
    _overlaps,
    _plain,
)
from cozy_runtime.internal.residency import _REQUEST
from cozy_runtime.rows import rows as projection_rows


@dataclass(frozen=True)
class _Route:
    backend: str
    implementation: str
    reason: str
    layout: tuple[Any, ...]
    candidate: Any


def _route(backend: object, tokens: int) -> _Route | None:
    """Resolve the existing selected policy before constructing or accepting a menu."""
    name = getattr(backend, "value", str(backend))
    candidate = next((c for c in attention.known().values() if c.backend == name), None)
    if candidate is None:
        return None
    if candidate is attention.BY_NAME["kitchen-int4qk"]:
        return _Route(name, "kitchen-int4qk", "", (), candidate)
    original = attention.BY_NAME["kitchen-sol-int4qk"]
    base = candidate.producer.base if candidate.producer is not None else candidate.name
    if (
        base != original.name
        or candidate.dense != original.dense
        or candidate.local.forward is not original.local.forward
        or candidate.local.impl is not original.local.impl
    ):
        return None
    layout, site = _ACTIVE_LAYOUT.get(), attention_sol._SITE.get()
    if layout is None or site is None or layout.live_tokens != tokens:
        return None
    dense_path, exact_path = attention_sol.policy(layout, site)
    if exact_path or (not dense_path and layout.step >= layout.dense_until_step):
        return None
    identity = (
        layout.live_tokens,
        layout.protected_prefix,
        layout.dense_until_step,
        layout.dense_paths,
        layout.exact_paths,
        layout.tau,
    )
    return _Route(
        name, "kitchen-int4qk", "dense_path" if dense_path else "dense_step", identity, candidate
    )


def _entries(intent: ProjectedHeads, device: int) -> tuple[Any, ...] | None:
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
    expected = (
        (
            intent.rope,
            {
                "x_ptr": "*bf16",
                "out_ptr": "*bf16",
                "weight_ptr": "*bf16",
                "cos_ptr": "*fp32",
                "sin_ptr": "*fp32",
                "seq": "i32",
                "heads": "i32",
                "eps": "fp32",
                "HEAD_DIM": "constexpr",
                "ROTARY": "constexpr",
                "HEADS": "constexpr",
            },
            {"HEAD_DIM": 128, "ROTARY": 96, "HEADS": 8},
        ),
        (
            intent.key_sum,
            {
                "key_ptr": "*bf16",
                "out_ptr": "*fp32",
                "tokens": "i32",
                "stride_t": "i32",
                "HEAD_DIM": "constexpr",
                "BLOCK_T": "constexpr",
            },
            {"HEAD_DIM": 128, "BLOCK_T": 64},
        ),
    )
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
    return (launch.key, _entry_identity(intent.rope), _entry_identity(intent.key_sum))


@dataclass(frozen=True)
class _Projection:
    module: Any
    data: Tensor
    scale: Tensor
    bias: Tensor | None
    call: Any


@dataclass(frozen=True)
class _Geometry:
    heads: int
    full: _Dispatch
    tail: _Dispatch
    quantized: bool
    workspace: int

    @property
    def identity(self) -> str:
        return (
            f"heads:{self.heads}/full:{self.full.kind}:{self.full.config}"
            f"/tail:{self.tail.kind}:{self.tail.config}"
        )


def _head_chunks(heads: int) -> tuple[int, ...]:
    found = [heads]
    while found[-1] > 1:
        found.append((found[-1] + 1) // 2)
    return tuple(found)


def _attention_shapes(tokens: int, heads: int) -> tuple[tuple[tuple[int, ...], Any], ...]:
    tile = 128 if tokens > 1024 else 64
    blocks = (tokens + tile - 1) // tile
    return (
        ((1, heads, tokens, 64), torch.int8),
        ((1, heads, tokens, 64), torch.int8),
        ((heads * 128, blocks * tile), torch.int8),
        ((1, heads, ((tokens + 127) // 128) * 32), torch.float32),
        ((1, heads, blocks * 4), torch.float32),
        ((heads, 128), torch.float32),
        ((heads, 128), torch.float32),
    )


@dataclass
class _Quantized:
    ready: bool = False


@dataclass
class _Window:
    module: Any
    invocation: _RowInvocation

    def run(self, x: Any, operand: Any, out: Any) -> Any:
        return self.invocation.run(x, operand, out)


@dataclass
class _RowInvocation:
    module: Any
    projection: _Projection
    x: Tensor
    target: Tensor
    start: int
    stop: int
    dispatch: _Dispatch
    q: Tensor | None
    scales: Tensor | None
    workspace: Tensor | None
    quantization: _Quantized
    library: Any
    stream: int

    def rows(self, window: Any, x: Any, start: int, stop: int, out: Any) -> Any:
        if x is not self.x or start != self.start or stop != self.stop or out is not self.target:
            raise RuntimeError("the committed head projection changed its row invocation")
        p = self.projection
        if (
            window.data.data_ptr() != p.data[start:stop].data_ptr()
            or window.scale.data_ptr() != p.scale[start:stop].data_ptr()
            or (None if window.bias is None else window.bias.data_ptr())
            != (None if p.bias is None else p.bias[start:stop].data_ptr())
        ):
            raise RuntimeError("the committed head projection changed its weight views")
        token = _LINEAR_SPLIT.set(_Window(window, self))
        try:
            return window.linear(x, None, out)
        finally:
            _LINEAR_SPLIT.reset(token)

    def run(self, x: Any, operand: Any, out: Any) -> Any:
        if x is not self.x or operand is not None or out is not self.target:
            raise RuntimeError("the committed head projection changed its operands")
        if int(torch.cuda.current_stream(x.device).cuda_stream) != self.stream:
            raise RuntimeError("the committed head projection changed streams")

        def scratch(size: int) -> Tensor:
            if self.workspace is None or size > self.workspace.numel():
                raise RuntimeError("the head projection requested undeclared workspace")
            return self.workspace

        p, start, stop = self.projection, self.start, self.stop
        launch_projection(
            self.library,
            p.data[start:stop],
            p.scale[start:stop],
            None if p.bias is None else p.bias[start:stop],
            x,
            out,
            self.dispatch,
            self.q,
            self.scales,
            scratch,
            self.stream,
            quantized=self.quantization.ready,
        )
        if self.dispatch.kind == 0:
            self.quantization.ready = True
        return out

    def call(self) -> None:
        token = _LINEAR_SPLIT.set(self)
        try:
            result = self.projection.call(self.x, self.start, self.stop, out=self.target)
            if result is not self.target:
                raise RuntimeError("the committed Rows call replaced its output")
        finally:
            _LINEAR_SPLIT.reset(token)


@dataclass
class _Plan:
    intent: ProjectedHeads
    route: _Route
    request: Any
    library: Any
    projections: tuple[_Projection, ...]
    norm_q: Tensor
    norm_k: Tensor
    cos: Tensor
    sin: Tensor
    seed: Tensor | None
    geometries: dict[str, _Geometry]
    choices: list[tuple[str, tuple[int, ...]]]
    key: str
    stream: Any

    def run(self) -> Tensor:
        device = self.intent.hidden.device.index or 0
        stream = int(self.stream.cuda_stream)
        selection = vt.split_select(
            device, self.key, self.choices, preferred=self.choices[0][0], stream=stream
        )
        geometry = self.geometries[str(selection["choice_id"])]
        record = self.request.working_splits.setdefault(
            self.key, {"site": "projected-heads", "key": self.key, "selections": 0}
        )
        record.update(
            selection=dict(selection),
            head_chunk=geometry.heads,
            backend=self.route.backend,
            implementation=self.route.implementation,
            choices=self.choices,
            workspace_bytes=geometry.workspace,
        )
        record["selections"] += 1
        phase.choice(("native-split", self.key, geometry.identity))
        observation = vt.split_begin(
            device,
            self.key,
            geometry.identity,
            stream=stream,
            completion_owner=phase.completion_owner(device, stream),
        )
        submitted = False
        try:
            result = self.execute(geometry)
            submitted = True
            return result
        except BaseException as error:
            if fault(error) is Fault.OOM:
                error.varena_no_retry = True  # type: ignore[attr-defined]
            raise
        finally:
            record["submission"] = vt.split_end(device, observation, submitted=submitted)

    def execute(self, geometry: _Geometry) -> Tensor:
        intent, x = self.intent, self.intent.hidden
        tokens, width, count = int(x.shape[1]), int(x.shape[2]), geometry.heads
        device, stream = x.device.index or 0, int(self.stream.cuda_stream)
        with torch.cuda.device(device), torch.cuda.stream(self.stream):
            output = torch.empty((1, tokens, intent.heads, 128), dtype=x.dtype, device=x.device)
            projected = [
                torch.empty((1, tokens, count * 128), dtype=x.dtype, device=x.device)
                for _ in range(3)
            ]
            quantized = (
                torch.empty((tokens, width), dtype=torch.int8, device=x.device)
                if geometry.quantized
                else None
            )
            scales = (
                torch.empty(tokens, dtype=torch.float32, device=x.device)
                if geometry.quantized
                else None
            )
            workspace = (
                torch.empty(geometry.workspace, dtype=torch.uint8, device=x.device)
                if geometry.workspace
                else None
            )
            scratch = [
                torch.empty(shape, dtype=dtype, device=x.device)
                for shape, dtype in _attention_shapes(tokens, count)
            ]
            seed_work = (
                torch.empty((count, 128), dtype=torch.float32, device=x.device)
                if self.seed is not None
                else None
            )
            operand = _Quantized()
            # Every allocation precedes projection, normalization, output, and seed writes.
            for first in range(0, intent.heads, count):
                catch.checkpoint()
                heads = min(count, intent.heads - first)
                dispatch = geometry.full if heads == count else geometry.tail
                views = [
                    buffer.view(-1)[: tokens * heads * 128].view(1, tokens, heads * 128)
                    for buffer in projected
                ]
                for projection, target in zip(self.projections, views, strict=True):
                    _RowInvocation(
                        projection.module,
                        projection,
                        x,
                        target,
                        first * 128,
                        (first + heads) * 128,
                        dispatch,
                        quantized,
                        scales,
                        workspace,
                        operand,
                        self.library,
                        stream,
                    ).call()
                q, k, v = (value.view(1, tokens, heads, 128) for value in views)
                for value, norm, eps in (
                    (q, self.norm_q, intent.eps_q),
                    (k, self.norm_k, intent.eps_k),
                ):
                    intent.launch(
                        intent.rope.kernel,
                        (tokens, (heads + 7) // 8, 1),
                        value,
                        value,
                        norm,
                        self.cos,
                        self.sin,
                        tokens,
                        heads,
                        float(eps),
                        128,
                        96,
                        8,
                    )
                if seed_work is not None:
                    assert self.seed is not None
                    means = seed_work[:heads]
                    intent.launch(
                        intent.key_sum.kernel, (heads, 1, 1), k, means, tokens, heads * 128, 128, 64
                    )
                    means.div_(tokens)
                    self.seed[first : first + heads].copy_(means)
                target = output[:, :, first : first + heads]
                q4, k4, v8, qs, ks, sums, vs = scratch

                def strides(value: Tensor) -> tuple[int, int, int]:
                    return value.stride(0), value.stride(2), value.stride(1)

                code = _kernels._attn4.attention(
                    device,
                    q.data_ptr(),
                    k.data_ptr(),
                    v.data_ptr(),
                    target.data_ptr(),
                    q4.data_ptr(),
                    qs.data_ptr(),
                    k4.data_ptr(),
                    ks.data_ptr(),
                    sums.data_ptr(),
                    v8.data_ptr(),
                    vs.data_ptr(),
                    1,
                    heads,
                    heads,
                    tokens,
                    tokens,
                    128 if tokens > 1024 else 64,
                    True,
                    128**-0.5,
                    strides(q),
                    strides(k),
                    strides(v),
                    strides(target),
                    stream,
                )
                if code:
                    raise RuntimeError(f"kitchen-int4qk: {_kernels._attn4.error_name(code)}")
                options = attention_ulysses.Options(
                    attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False
                )
                attention_ulysses.record(
                    self.route.backend, self.route.candidate.local.impl(q, k, v, options)
                )
                dense = attention.BY_NAME["kitchen-int4qk"]
                attention_ulysses.record(dense.backend, dense.local.impl(q, k, v, options))
                if self.route.reason:
                    attention_sol._count(self.route.reason)
        return output


def prepare_projected_heads(
    intent: ProjectedHeads, output: Any, cuts: tuple[Cut, ...]
) -> _Plan | None:
    """Prepare exact descriptors separately from native choice/execution; never guess a key."""
    request = _REQUEST.get()
    if (
        type(intent) is not ProjectedHeads
        or request is None
        or request.idle
        or not request.memory.cuda
    ):
        return None
    x = intent.hidden
    if (
        output is not None
        or type(x) is not Tensor
        or not x.is_cuda
        or x.dtype != torch.bfloat16
        or x.device != request.memory.device
        or x.dim() != 3
        or x.shape[0] != 1
        or not x.is_contiguous()
        or not x.numel()
        or x.data_ptr() % 16
        or cuts != (Cut(2),)
        or type(intent.heads) is not int
        or intent.heads < 1
        or torch.is_grad_enabled()
        or torch.is_autocast_enabled("cuda")
        or catch.uncommitted_scope()
        or _LINEAR_SPLIT.get() is not None
        or torch.cuda.is_current_stream_capturing()
        or (torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1)
        or torch.nn.modules.module._global_forward_pre_hooks
        or torch.nn.modules.module._global_forward_hooks
        or any(not _leaf(module, x.dtype) for module in (intent.query, intent.key, intent.value))
        or any(
            type(eps) not in (int, float) or not math.isfinite(eps) or eps <= 0
            for eps in (intent.eps_q, intent.eps_k)
        )
    ):
        return None
    tokens, width, device = int(x.shape[1]), int(x.shape[2]), x.device.index or 0
    if (
        tokens * intent.heads * 128 > 2**31 - 1
        or int8_linear.serving(device) != int8_linear.OP
        or not vt.split_context(device)["eligible"]
    ):
        return None
    held = getattr(_RECENT, "held", None)
    if any(shared.tensor is x for shared in _SHARED.get()) or (
        held is not None and held.tensor is not None and held.tensor() is x
    ):
        return None
    route = _route(intent.backend, tokens)
    kernel_identity = _entries(intent, device)
    attn_identity = prebuilt.artifact("_attn4")
    library = int8_linear.library(device)
    if (
        route is None
        or kernel_identity is None
        or not attn_identity
        or library.family != "mma.sync"
        or not callable(getattr(getattr(_kernels, "_attn4", None), "attention", None))
    ):
        return None
    norm_q, norm_k, cos, sin, seed = (
        read(value) for value in (intent.norm_q, intent.norm_k, intent.cos, intent.sin, intent.seed)
    )
    if any(
        not _plain(value, x.device, x.dtype)
        or tuple(value.shape) != (128,)
        or not value.is_contiguous()
        for value in (norm_q, norm_k)
    ):
        return None
    if any(
        not _plain(value, x.device, torch.float32)
        or tuple(value.shape) != (tokens, 96)
        or not value.is_contiguous()
        for value in (cos, sin)
    ):
        return None
    if seed is not None and (
        type(seed) is not Tensor
        or not _plain(seed, x.device, torch.float32)
        or tuple(seed.shape) != (intent.heads, 128)
        or not seed.is_contiguous()
    ):
        return None
    projections = []
    for module in (intent.query, intent.key, intent.value):
        data, scale, bias = read(module.data), read(module.scale), read(module.bias)
        if (
            tuple(data.shape) != (intent.heads * 128, width)
            or scale.data_ptr() % 16
            or (
                bias is not None
                and (bias.dtype != x.dtype or not bias.is_contiguous() or bias.data_ptr() % 16)
            )
        ):
            return None
        library._weight(prebuilt.INT8, data, scale, bias, width, x.dtype)
        call = projection_rows(module)
        if call is None:
            return None
        projections.append(_Projection(module, data, scale, bias, call))
    readonly = (
        x,
        norm_q,
        norm_k,
        cos,
        sin,
        *(value for p in projections for value in (p.data, p.scale, p.bias) if value is not None),
    )
    if seed is not None and any(_overlaps(seed, value) for value in readonly):
        return None
    queried = {}

    def query(heads: int) -> _Dispatch | None:
        if heads not in queried:
            info = _kernels._gemm.int8_plan(device, False, tokens, heads * 128, width, heads * 128)
            if info["status"] == _kernels._gemm.NOT_SUPPORTED:
                return None
            library._check(info["status"], "the projected-head plan")
            queried[heads] = _Dispatch(
                tokens, info["kind"], info["config"], info["workspace_bytes"]
            )
        return queried[heads]

    geometries, choices = {}, []
    for chunk in _head_chunks(intent.heads):
        full, tail = query(chunk), query(intent.heads % chunk or chunk)
        if full is None or tail is None:
            return None
        quantized = any(part.kind == 0 for part in (full, tail))
        geometry = _Geometry(chunk, full, tail, quantized, max(full.workspace, tail.workspace))
        allocations = [tokens * intent.heads * 128 * 2, *([tokens * chunk * 128 * 2] * 3)]
        if quantized:
            allocations.extend((tokens * width, tokens * 4))
        if geometry.workspace:
            allocations.append(geometry.workspace)
        allocations.extend(
            math.prod(shape) * dtype.itemsize for shape, dtype in _attention_shapes(tokens, chunk)
        )
        if seed is not None:
            allocations.append(chunk * 128 * 4)
        geometries[geometry.identity] = geometry
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
            "projected-heads-int4/1",
            tuple(sorted(identities)),
            tuple(_layout(value) for value in readonly),
            _layout(seed),
            intent.heads,
            intent.eps_q,
            intent.eps_k,
            route.backend,
            route.implementation,
            route.reason,
            route.layout,
            kernel_identity,
            attn_identity,
            library.identity,
            library.family,
            tuple(p.module.encoding for p in projections),
            tuple(choices),
            tuple(geometries.values()),
        )
    )
    if semantic is None:
        return None
    key = "runtime.split.projected_heads/1:" + hashlib.sha256(semantic.encode()).hexdigest()
    request.memory.remember_working_phase(key, models or (request,))
    return _Plan(
        intent,
        route,
        request,
        library,
        tuple(projections),
        norm_q,
        norm_k,
        cos,
        sin,
        seed,
        geometries,
        choices,
        key,
        torch.cuda.current_stream(device),
    )


def split_projected_heads(
    intent: ProjectedHeads, output: Any, cuts: tuple[Cut, ...]
) -> Tensor | None:
    plan = prepare_projected_heads(intent, output, cuts)
    return None if plan is None else plan.run()
