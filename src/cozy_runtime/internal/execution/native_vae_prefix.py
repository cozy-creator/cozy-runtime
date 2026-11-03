"""Restricted H3 proj_in, token packing and shared rotary tables.

Projection is an explicit fused-bias Lt ALTERNATIVE to stock strided matmul plus
separate FP16 bias. Grid/RoPE operations preserve the original eager round points.
The caller owns one complete admission, source hooks and explicit stream ordering.
"""
from __future__ import annotations

import math
import os
import threading
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor, nn

TORCH_COMMIT = "08187d9e0fba026dc8217405802ab5381dc88d90"
IMPLEMENTATION = "h3-prefix-lt-fused-bias/1"
GEOMETRY, PATCHES, TOKENS, CHANNELS, WIDTH = (7, 16, 16), 1792, 1797, 24, 2048
_UNPROVEN: list[Bound] = []


def _metadata(value: Tensor) -> tuple[Any, ...]:
    return tuple(value.shape), tuple(value.stride()), value.dtype, value.device


def _ordinary(value: Any) -> bool:
    return (type(value) in (Tensor, nn.Parameter) and value.layout == torch.strided
            and not value.is_conj() and not value.is_neg() and not value.requires_grad)


def _configuration(decoder: nn.Module) -> tuple[Any, ...]:
    modules = (decoder, decoder.proj_in, decoder.rope)
    hooks = nn.modules.module
    calls = tuple((type(m).forward, m.training, m._compiled_call_impl,
                   vars(m).get("forward"), vars(m).get("_cozy_execution_forward"),
                   *(tuple(id(h) for h in table.values()) for table in (
                       m._forward_pre_hooks, m._forward_hooks, m._backward_pre_hooks, m._backward_hooks)))
                  for m in modules)
    return (id(decoder.proj_in), id(decoder.rope), decoder.num_register_tokens, decoder.proj_in.in_features, decoder.proj_in.out_features,
            calls, tuple(id(h) for h in hooks._global_forward_pre_hooks.values()),
            tuple(id(h) for h in hooks._global_forward_hooks.values()))


def _capacity(value: Tensor | None) -> int:
    return 0 if value is None else value.numel() * value.element_size()


def _pointer(value: Tensor | None) -> int:
    return 0 if value is None else value.data_ptr()


def _overlap(left: Tensor, right: Tensor) -> bool:
    # Independently exported Storage objects can still name the same memory.
    # Check the actual contiguous spans as well as Torch's Storage identity.
    return (torch._C._is_alias_of(left, right)
            or max(left.data_ptr(), right.data_ptr())
            < min(left.data_ptr() + _capacity(left), right.data_ptr() + _capacity(right)))


def _scope() -> None:
    from cozy_runtime.internal.execution import catch
    if catch.uncommitted_scope():
        raise ValueError("H3 prefix cannot execute inside a retryable parent cut")


def _no_retry(error: BaseException) -> None:
    error.varena_no_retry = True  # type: ignore[attr-defined]


@dataclass(frozen=True)
class Description:
    decoder: Any
    input_metadata: tuple[Any, ...]
    parameter_metadata: tuple[tuple[Any, ...], ...]
    configuration: tuple[Any, ...]
    parameters: tuple[Any, ...]
    inv_freq: Tensor
    frequency_metadata: tuple[Any, ...]
    slots: tuple[Any, ...] = ()
    models: tuple[Any, ...] = ()
    source_identity: tuple[Any, ...] = ()

    @property
    def batch(self) -> int:
        return self.input_metadata[0][0]

    @property
    def rows(self) -> int:
        return self.batch * PATCHES

    @property
    def weight_cast(self) -> bool:
        return self.parameter_metadata[0][2] is torch.float32

    def buffer_specs(self, workspace_bytes: int) -> tuple[tuple[str, tuple[int, ...], Any], ...]:
        if type(workspace_bytes) is not int or workspace_bytes < 0:
            raise ValueError("workspace must be the actual nonnegative provider query")
        specs = [("input_rows", (self.rows, CHANNELS), torch.float16),
                 ("dense_result", (self.rows, WIDTH), torch.float16),
                 ("hidden", (self.batch, TOKENS, WIDTH), torch.float16)]
        if self.weight_cast:
            specs.append(("weight_cast", (WIDTH, CHANNELS), torch.float16))
        if workspace_bytes:
            specs.append(("workspace", (workspace_bytes,), torch.uint8))
        specs.extend((name, shape, torch.float32) for name, shape in (
            ("grid_t", (7,)), ("grid_h", (16,)), ("grid_w", (16,)),
            ("positions", (1, TOKENS, 3)), ("scaled_positions", (1, TOKENS, 3, 1)),
            ("angles", (1, TOKENS, 3, 8)), ("tiled_angles", (1, TOKENS, 1, 48)),
            ("cosine", (1, TOKENS, 1, 48)), ("sine", (1, TOKENS, 1, 48))))
        return tuple(specs)

    def allocations(self, workspace_bytes: int) -> tuple[tuple[str, int], ...]:
        sizes = {torch.float16: 2, torch.float32: 4, torch.uint8: 1}
        # The actual 4096-byte bias destination already lives in preparation.
        return tuple((name, math.prod(shape) * sizes[dtype])
                     for name, shape, dtype in self.buffer_specs(workspace_bytes))


def describe(decoder: Any, value: Any, *, geometry: tuple[int, ...]) -> Description | None:
    """Metadata-only capability; no source views, hooks, CUDA or numeric work."""
    if (torch.version.git_version != TORCH_COMMIT or tuple(geometry) != GEOMETRY
            or type(value) is not Tensor or not _ordinary(value) or value.dtype is not torch.float16
            or value.dim() != 5 or tuple(value.shape[1:]) != (CHANNELS, *GEOMETRY)
            or value.shape[0] < 1 or value.shape[0] * PATCHES > 2**31 - 1 or not value.is_contiguous()):
        return None
    from diffusers.models.autoencoders.autoencoder_kl_minimax_h3 import (
        MiniMaxH3VideoViTDecoder3d, MiniMaxH3VideoRotaryPosEmbed,
    )
    if (type(decoder) is not MiniMaxH3VideoViTDecoder3d or type(decoder.proj_in) is not nn.Linear
            or type(decoder.rope) is not MiniMaxH3VideoRotaryPosEmbed
            or decoder.num_register_tokens != 4 or decoder.proj_in.in_features != CHANNELS
            or decoder.proj_in.out_features != WIDTH):
        return None
    modules = (decoder, decoder.proj_in, decoder.rope)
    for module in modules:
        if (module.training or module._compiled_call_impl is not None
                or ("forward" in vars(module)
                    and module.forward is not vars(module).get("_cozy_execution_forward"))):
            return None
    sources = ((decoder.proj_in, "weight", (WIDTH, CHANNELS), (torch.float16, torch.float32), True),
               (decoder.proj_in, "bias", (WIDTH,), (torch.float32,), False),
               (decoder, "register_tokens", (1, 4, WIDTH), (torch.float32,), False))
    parameters = tuple(module._parameters.get(name) for module, name, *_ in sources)
    for tensor, (_, _, shape, dtypes, _) in zip(parameters, sources, strict=True):
        if (tensor is None or tuple(tensor.shape) != shape or tensor.dtype not in dtypes
                or tensor.device != value.device or not tensor.is_contiguous()
                or (type(tensor) in (Tensor, nn.Parameter) and not _ordinary(tensor))):
            return None
    frequency = decoder.rope._buffers.get("inv_freq")
    if (not _ordinary(frequency) or hasattr(frequency, "_varena_read")
            or tuple(frequency.shape) != (8,) or frequency.dtype is not torch.float32
            or frequency.device != value.device or not frequency.is_contiguous()):
        return None
    slots: tuple[Any, ...] = ()
    models: tuple[Any, ...] = ()
    identity: tuple[Any, ...] = ()
    if (not all(_ordinary(t) for t in parameters) or any(hasattr(t, "_varena_read") for t in parameters)
            or any(m._forward_pre_hooks or m._forward_hooks for m in modules)):
        if value.device.type != "cuda":
            return None
        from cozy_runtime.internal.execution.native_vae import _slot, _source_identity, _Unsupported
        from cozy_runtime.internal.residency import _REQUEST, Placeholder
        request = _REQUEST.get()
        if request is None or any(type(t) not in (Tensor, nn.Parameter, Placeholder) for t in parameters):
            return None
        with request.memory.lock:
            models = tuple(m for m in request.memory.models.values() if m.name in request.current)
        try:
            slots = tuple(_slot(m, n, shape, dtypes, models, matrix=matrix)
                          for m, n, shape, dtypes, matrix in sources)
            owners = {id(slot.model) for slot in slots if slot.model is not None}
            models = tuple(m for m in models if id(m) in owners)
            identity = _source_identity(models)
        except _Unsupported:
            return None
    owned_hooks = {id(hook) for model in models for hook in model.source_hooks}
    hooks = nn.modules.module
    if hooks._global_forward_pre_hooks or hooks._global_forward_hooks:
        return None
    for module in modules:
        if (module._backward_hooks or module._backward_pre_hooks
                or any(id(hook) not in owned_hooks for hook in (
                    *module._forward_pre_hooks.values(), *module._forward_hooks.values()))):
            return None
    return Description(decoder, _metadata(value), tuple(_metadata(t) for t in parameters),
                       _configuration(decoder), parameters, frequency, _metadata(frequency), slots, models, identity)


@dataclass
class Buffers:
    values: dict[str, Tensor]

    def __getitem__(self, name: str) -> Tensor:
        return self.values[name]

    @classmethod
    def allocate(cls, description: Description, workspace_bytes: int) -> Buffers:
        return cls({name: torch.empty(shape, dtype=dtype, device=description.input_metadata[3])
                    for name, shape, dtype in description.buffer_specs(workspace_bytes)})


def pack_input(value: Tensor, target: Tensor) -> None:
    target.view(value.shape[0], PATCHES, CHANNELS).copy_(value.flatten(2).transpose(1, 2))


def pack_tokens_and_rotary(registers: Tensor, inv_freq: Tensor, buffers: Buffers) -> tuple[Tensor, tuple[Tensor, Tensor]]:
    """Write declared destinations with the original FP32 arithmetic round points."""
    b = buffers
    hidden = b["hidden"]
    hidden[:, :PATCHES].copy_(b["dense_result"].view(hidden.shape[0], PATCHES, WIDTH))
    hidden[:, PATCHES:PATCHES + 4].copy_(registers)
    hidden[:, PATCHES + 4:].zero_()
    for name, size in zip(("grid_t", "grid_h", "grid_w"), GEOMETRY, strict=True):
        grid = b[name]
        torch.arange(0.5, size, out=grid)
        grid.div_(size).mul_(2.0).sub_(1.0)
    positions = b["positions"]
    cube = positions[:, :PATCHES].view(1, 7, 16, 16, 3)
    cube[..., 0].copy_(b["grid_t"].view(1, 7, 1, 1))
    cube[..., 1].copy_(b["grid_h"].view(1, 1, 16, 1))
    cube[..., 2].copy_(b["grid_w"].view(1, 1, 1, 16))
    positions[:, PATCHES:].zero_()
    # Keep two separate eager FP32 multiplies. Combining inv_freq with 2*pi
    # before multiplying positions would change the original rounding.
    torch.mul(positions.unsqueeze(-1), 2.0 * math.pi, out=b["scaled_positions"])
    torch.mul(b["scaled_positions"], inv_freq.view(1, 1, 1, 8), out=b["angles"])
    angles = b["angles"].view(1, TOKENS, 1, 24)
    b["tiled_angles"][..., :24].copy_(angles)
    b["tiled_angles"][..., 24:].copy_(angles)
    torch.cos(b["tiled_angles"], out=b["cosine"])
    torch.sin(b["tiled_angles"], out=b["sine"])
    return hidden, (b["cosine"], b["sine"])


def prepare(description: Description, *, context: Any, stream: Any) -> Prepared | None:
    if (torch.version.git_version != TORCH_COMMIT or description.input_metadata[3].type != "cuda"
            or torch.is_grad_enabled()):
        return None
    _scope()
    from cozy_runtime._kernels import _dense
    from cozy_runtime.internal import dense_linear
    device = description.input_metadata[3]
    if type(context) is not _dense.Context or stream.device != device or stream.cuda_stream <= 2:
        return None
    policy = dense_linear.policy(device.index or 0)
    if policy is None:
        return None
    with torch.cuda.device(device), torch.cuda.stream(stream):
        if torch.cuda.is_current_stream_capturing():
            return None
        bias = torch.empty(WIDTH, dtype=torch.float16, device=device)
        try:
            plan = dense_linear.checked(context.query, description.rows, CHANNELS, WIDTH,
                bias.data_ptr(), _capacity(bias), policy.half_accumulation, policy.reduction,
                policy.sm_target, policy.workspace_limit, stream.cuda_stream)
        except RuntimeError as error:
            cause = error.__cause__
            if isinstance(cause, _dense.Error) and cause.domain == "cublas" and cause.status == _dense.NOT_SUPPORTED:
                return None
            raise
    return Prepared(description, context, stream, policy, bias, plan, plan.info(), os.getpid())


@dataclass
class Prepared:
    description: Description
    context: Any
    stream: Any
    policy: Any
    bias: Tensor
    plan: Any
    info: dict[str, Any]
    pid: int
    bound: bool = False
    lock: Any = field(default_factory=threading.Lock, repr=False)

    @property
    def allocations(self) -> tuple[tuple[str, int], ...]:
        return self.description.allocations(self.info["workspace_bytes"])

    def facts(self) -> dict[str, Any]:
        d = self.description
        return {"implementation": IMPLEMENTATION, "torch_commit": TORCH_COMMIT,
                "projection": "declared fused-bias Lt alternative to stock strided matmul plus FP16 bias",
                "preserves_stock_projection_round_points": False,
                "packing_round_points": "original FP32 grids and two separate rotary multiplies",
                "input_shape": d.input_metadata[0], "input_dtype": str(d.input_metadata[2]),
                "parameter_dtypes": tuple(str(m[2]) for m in d.parameter_metadata),
                "inv_freq_shape": d.frequency_metadata[0], "inv_freq_dtype": str(d.frequency_metadata[2]),
                "geometry": GEOMETRY, "patch_rows_per_tile": PATCHES, "tokens_per_tile": TOKENS,
                "hidden_shape": (d.batch, TOKENS, WIDTH), "rotary_shape": (1, TOKENS, 1, 48),
                "preparation_allocations": (("bias", _capacity(self.bias)),),
                "execution_allocations": self.allocations, "provider": dict(self.info)}

    def bind(self, value: Tensor, buffers: Buffers) -> Bound:
        from cozy_runtime.internal import dense_linear
        if self.pid != os.getpid():
            raise RuntimeError("H3 prefix preparation cannot be inherited after fork")
        with self.lock:
            if self.bound:
                raise RuntimeError("H3 prefix preparation binds once")
            self.bound = True
        _scope()
        d = self.description
        if (torch.version.git_version != TORCH_COMMIT or not _ordinary(value)
                or _metadata(value) != d.input_metadata or _configuration(d.decoder) != d.configuration
                or dense_linear.policy(value.device.index or 0) != self.policy):
            raise ValueError("H3 prefix invocation changed after query")
        if d.slots:
            from cozy_runtime.internal.execution.native_vae import _source_identity
            if _source_identity(d.models) != d.source_identity:
                raise ValueError("H3 prefix Source identity changed before binding")
            parameters = tuple(slot.bind() for slot in d.slots)
            if _source_identity(d.models) != d.source_identity:
                raise ValueError("H3 prefix Source identity changed during binding")
        else:
            parameters = (d.decoder.proj_in.weight, d.decoder.proj_in.bias, d.decoder.register_tokens)
            if any(t is not prior for t, prior in zip(parameters, d.parameters, strict=True)):
                raise ValueError("H3 prefix parameter was replaced")
        if any(not _ordinary(t) or _metadata(t) != metadata
               for t, metadata in zip(parameters, d.parameter_metadata, strict=True)):
            raise ValueError("H3 prefix source metadata changed")
        frequency = d.decoder.rope._buffers.get("inv_freq")
        if frequency is not d.inv_freq or not _ordinary(frequency) or _metadata(frequency) != d.frequency_metadata:
            raise ValueError("H3 prefix frequency buffer changed")
        if any(t.data_ptr() == 0 or t.data_ptr() % 16 for t in (value, *parameters, frequency)):
            raise ValueError("H3 prefix sources require aligned native layouts")
        expected = d.buffer_specs(self.info["workspace_bytes"])
        if set(buffers.values) != {name for name, _, _ in expected}:
            raise ValueError("H3 prefix buffers differ from the declared inventory")
        destinations = []
        for name, shape, dtype in (*expected, ("prepared_bias", (WIDTH,), torch.float16)):
            tensor = self.bias if name == "prepared_bias" else buffers[name]
            if (not _ordinary(tensor) or tuple(tensor.shape) != shape or tensor.dtype != dtype
                    or tensor.device != value.device or not tensor.is_contiguous()
                    or tensor.data_ptr() == 0 or tensor.data_ptr() % 256):
                raise ValueError("H3 prefix buffer differs from its declared allocation")
            if any(_overlap(tensor, other) for other in (value, *parameters, frequency, *destinations)):
                raise ValueError("H3 prefix writes require independent owned storage")
            destinations.append(tensor)
        matrix = buffers["weight_cast"] if d.weight_cast else parameters[0]
        bound = Bound(self, value, parameters, frequency, matrix, buffers)
        dense_linear.checked(self.plan.validate, *bound.arguments())
        bound.binding_state = bound.state()
        return bound


@dataclass
class Bound:
    prepared: Prepared
    value: Tensor
    parameters: tuple[Tensor, ...]
    inv_freq: Tensor
    matrix: Tensor
    buffers: Buffers
    claimed: bool = False
    submitted: bool = False
    submission_closed: bool = False
    completed: bool = False
    retained: bool = False
    binding_state: tuple[Any, ...] = ()
    lock: Any = field(default_factory=threading.Lock, repr=False)

    def state(self) -> tuple[Any, ...]:
        values = (self.value, *self.parameters, self.inv_freq, self.matrix,
                  self.prepared.bias, *self.buffers.values.values())
        return (tuple(self.buffers.values),
                tuple((id(t), _metadata(t), t.data_ptr(), _ordinary(t)) for t in values))

    def arguments(self) -> tuple[int, ...]:
        b = self.buffers
        workspace = b.values.get("workspace")
        return (_pointer(b["input_rows"]), _capacity(b["input_rows"]), _pointer(self.matrix), _capacity(self.matrix),
                _pointer(b["dense_result"]), _capacity(b["dense_result"]), _pointer(workspace), _capacity(workspace),
                self.prepared.stream.cuda_stream)

    def _process(self) -> None:
        if self.prepared.pid != os.getpid():
            raise RuntimeError("H3 prefix binding cannot be inherited after fork")

    def validate(self) -> None:
        from cozy_runtime.internal import dense_linear
        self._process()
        _scope()
        p, d = self.prepared, self.prepared.description
        if (torch.version.git_version != TORCH_COMMIT or self.state() != self.binding_state
                or _configuration(d.decoder) != d.configuration
                or d.decoder.rope._buffers.get("inv_freq") is not self.inv_freq
                or dense_linear.policy(self.value.device.index or 0) != p.policy):
            raise ValueError("H3 prefix binding/configuration/policy changed before execution")
        if d.slots:
            from cozy_runtime.internal.execution.native_vae import _source_identity
            if _source_identity(d.models) != d.source_identity:
                raise ValueError("H3 prefix Source identity changed before execution")
            for slot, value in zip(d.slots, self.parameters, strict=True):
                valid = (slot.module._parameters[slot.name] is slot.value if slot.model is None
                         else slot.block.region is not None
                         and value.data_ptr() == int(slot.block.region.ptr) + slot.offset)
                if not valid:
                    raise ValueError("H3 prefix raw Source binding changed before execution")
        else:
            current = (d.decoder.proj_in.weight, d.decoder.proj_in.bias, d.decoder.register_tokens)
            if any(value is not prior for value, prior in zip(current, self.parameters, strict=True)):
                raise ValueError("H3 prefix parameter changed before execution")
        dense_linear.checked(p.plan.validate, *self.arguments())

    def execute(self) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        from cozy_runtime.internal import dense_linear
        self._process()
        _scope()
        with self.lock:
            if self.claimed:
                raise RuntimeError("H3 prefix binding is one-shot")
            self.claimed = True
        try:
            self.validate()
            self.submitted = True
            with torch.cuda.device(self.value.device), torch.cuda.stream(self.prepared.stream):
                weight, bias, registers = self.parameters
                pack_input(self.value, self.buffers["input_rows"])
                if self.prepared.description.weight_cast:
                    self.buffers["weight_cast"].copy_(weight)
                self.prepared.bias.copy_(bias)
                dense_linear.checked(self.prepared.plan.execute, *self.arguments())
                return pack_tokens_and_rotary(registers, self.inv_freq, self.buffers)
        except BaseException as error:
            _no_retry(error)
            if self.submitted:
                self._retain()
            raise
        finally:
            self.submission_closed = True

    def _retain(self) -> None:
        if not self.retained:
            _UNPROVEN.append(self)
            self.retained = True

    def _complete(self) -> None:
        self.completed = True
        if self.retained:
            _UNPROVEN[:] = [owner for owner in _UNPROVEN if owner is not self]
            self.retained = False

    def poll(self) -> bool:
        self._process()
        if not self.submission_closed:
            raise RuntimeError("H3 prefix host submission has not closed")
        if not self.completed:
            try:
                if self.prepared.stream.query():
                    self._complete()
            except BaseException as error:
                _no_retry(error)
                self._retain()
                raise
        return self.completed

    def wait(self) -> None:
        self._process()
        if not self.submission_closed:
            raise RuntimeError("H3 prefix host submission has not closed")
        try:
            self.prepared.stream.synchronize()
            self._complete()
        except BaseException as error:
            _no_retry(error)
            self._retain()
            raise

    def __del__(self) -> None:
        if self.submitted and not self.completed:
            self._retain()
