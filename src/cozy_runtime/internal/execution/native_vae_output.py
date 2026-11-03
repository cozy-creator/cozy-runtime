"""The actual H3 final affine LayerNorm, dense projection and pixel assembly.

The caller admits one complete operation. Functional Torch LayerNorm really creates
three tensors; they are declared and retained, never hidden behind an out= wrapper.
No chooser, nested split, source hook or application retry is installed here.
"""
from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor, nn

TORCH_COMMIT = "08187d9e0fba026dc8217405802ab5381dc88d90"
IMPLEMENTATION = "h3-output-functional-layernorm-lt/1"
GEOMETRY = (7, 16, 16)
PATCHES, TOKENS, WIDTH, PROJECTED = 1792, 1797, 2048, 3072
_UNPROVEN: list[Bound] = []


def _metadata(value: Tensor) -> tuple[Any, ...]:
    return tuple(value.shape), tuple(value.stride()), value.dtype, value.device


def _ordinary(value: Any) -> bool:
    return (type(value) in (Tensor, nn.Parameter) and value.layout == torch.strided
            and not value.is_conj() and not value.is_neg() and not value.requires_grad)


def _configuration(module: nn.Module) -> tuple[Any, ...]:
    hooks = nn.modules.module
    geometry = ((module.normalized_shape, module.eps, module.elementwise_affine)
                if type(module) is nn.LayerNorm else (module.in_features, module.out_features))
    return (geometry, module.training, type(module).forward,
            vars(module).get("forward"), vars(module).get("_cozy_execution_forward"),
            module._compiled_call_impl,
            *(tuple(id(h) for h in table.values()) for table in (
                module._forward_pre_hooks, module._forward_hooks, module._backward_pre_hooks,
                module._backward_hooks, hooks._global_forward_pre_hooks, hooks._global_forward_hooks)))


def _scope() -> None:
    from cozy_runtime.internal.execution import catch
    if catch.uncommitted_scope():
        raise ValueError("H3 output cannot execute inside a retryable parent cut")


def _capacity(value: Tensor | None) -> int:
    return 0 if value is None else value.numel() * value.element_size()


def _pointer(value: Tensor | None) -> int:
    return 0 if value is None else value.data_ptr()


def _no_retry(error: BaseException) -> None:
    error.varena_no_retry = True  # type: ignore[attr-defined]


@dataclass(frozen=True)
class Description:
    norm: nn.LayerNorm
    projection: nn.Linear
    input_metadata: tuple[Any, ...]
    parameter_metadata: tuple[tuple[Any, ...], ...]
    configurations: tuple[tuple[Any, ...], ...]
    parameters: tuple[Any, ...]
    slots: tuple[Any, ...] = ()
    models: tuple[Any, ...] = ()
    source_identity: tuple[Any, ...] = ()

    @property
    def batch(self) -> int:
        return self.input_metadata[0][0]

    @property
    def rows(self) -> int:
        return self.batch * TOKENS

    @property
    def weight_cast(self) -> bool:
        return self.parameter_metadata[2][2] is torch.float32

    @property
    def result_shape(self) -> tuple[int, ...]:
        return self.batch, 3, 28, 256, 256

    @property
    def functional_allocations(self) -> tuple[tuple[str, int], ...]:
        return (("norm_output", self.rows * WIDTH * 4),
                ("norm_mean", self.rows * 4), ("norm_rstd", self.rows * 4))

    def preallocations(self, workspace_bytes: int) -> tuple[tuple[str, int], ...]:
        if type(workspace_bytes) is not int or workspace_bytes < 0:
            raise ValueError("workspace must be the actual nonnegative provider query")
        result = [("input_rows", self.rows * WIDTH * 2),
                  ("dense_result", self.rows * PROJECTED * 2),
                  ("result", self.batch * PATCHES * PROJECTED * 2)]
        if self.weight_cast:
            result.append(("weight_cast", PROJECTED * WIDTH * 2))
        if workspace_bytes:
            result.append(("workspace", workspace_bytes))
        return tuple(result)

    def allocations(self, workspace_bytes: int) -> tuple[tuple[str, int], ...]:
        # The real bias is already live from query preparation and counted once there.
        return self.preallocations(workspace_bytes) + self.functional_allocations


def describe(norm: Any, projection: Any, value: Any, *, geometry: tuple[int, ...]) -> Description | None:
    """Read metadata only. Unreviewed Torch builds, including 2.15, decline."""
    if (torch.version.git_version != TORCH_COMMIT
            or type(norm) is not nn.LayerNorm or type(projection) is not nn.Linear
            or tuple(geometry) != GEOMETRY or tuple(norm.normalized_shape) != (WIDTH,)
            or not norm.elementwise_affine or norm.eps != 1e-5
            or projection.in_features != WIDTH or projection.out_features != PROJECTED
            or type(value) is not Tensor or not _ordinary(value)
            or value.dtype is not torch.float32 or value.dim() != 3
            or tuple(value.shape[1:]) != (TOKENS, WIDTH) or value.shape[0] < 1
            or value.shape[0] * TOKENS > 2**31 - 1 or not value.is_contiguous()):
        return None
    modules = (norm, projection)
    for module in modules:
        if (module.training or module._compiled_call_impl is not None
                or ("forward" in vars(module)
                    and module.forward is not vars(module).get("_cozy_execution_forward"))):
            return None
    sources = ((norm, "weight", (WIDTH,), (torch.float32,), False),
               (norm, "bias", (WIDTH,), (torch.float32,), False),
               (projection, "weight", (PROJECTED, WIDTH), (torch.float16, torch.float32), True),
               (projection, "bias", (PROJECTED,), (torch.float32,), False))
    parameters = tuple(module._parameters.get(name) for module, name, *_ in sources)
    for tensor, (_, _, shape, dtypes, _) in zip(parameters, sources, strict=True):
        if (tensor is None or tuple(tensor.shape) != shape or tensor.dtype not in dtypes
                or tensor.device != value.device or not tensor.is_contiguous()
                or (type(tensor) in (Tensor, nn.Parameter) and not _ordinary(tensor))):
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
    return Description(norm, projection, _metadata(value), tuple(_metadata(t) for t in parameters),
                       tuple(_configuration(m) for m in modules), parameters, slots, models, identity)


def assemble(rows: Tensor, target: Tensor) -> None:
    """Preserve all projected rows, then copy the patch prefix without another buffer."""
    batch = target.shape[0]
    prefix = rows.view(batch, TOKENS, PROJECTED)[:, :PATCHES, :]
    pixels = prefix.view(batch, 7, 16, 16, 3, 4, 16, 16).permute(0, 4, 1, 5, 2, 6, 3, 7)
    target.view(batch, 3, 7, 4, 16, 16, 16, 16).copy_(pixels)


@dataclass
class Buffers:
    input_rows: Tensor
    dense_result: Tensor
    result: Tensor
    weight_cast: Tensor | None
    workspace: Tensor | None

    @classmethod
    def allocate(cls, description: Description, workspace_bytes: int) -> Buffers:
        """Only fixed buffers: functional norm allocates its own three declared tensors."""
        description.preallocations(workspace_bytes)
        options = {"dtype": torch.float16, "device": description.input_metadata[3]}
        return cls(torch.empty((description.rows, WIDTH), **options),
                   torch.empty((description.rows, PROJECTED), **options),
                   torch.empty(description.result_shape, **options),
                   torch.empty((PROJECTED, WIDTH), **options) if description.weight_cast else None,
                   torch.empty(workspace_bytes, dtype=torch.uint8, device=options["device"])
                   if workspace_bytes else None)


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
        bias = torch.empty(PROJECTED, dtype=torch.float16, device=device)
        try:
            plan = dense_linear.checked(context.query, description.rows, WIDTH, PROJECTED,
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
                "input_shape": d.input_metadata[0], "input_dtype": str(d.input_metadata[2]),
                "parameter_dtypes": tuple(str(m[2]) for m in d.parameter_metadata),
                "geometry": GEOMETRY, "projected_rows_per_tile": TOKENS, "patch_rows_per_tile": PATCHES,
                "result_shape": d.result_shape, "preparation_allocations": (("bias", _capacity(self.bias)),),
                "preallocations": d.preallocations(self.info["workspace_bytes"]),
                "functional_allocations": d.functional_allocations, "execution_allocations": self.allocations,
                "norm_schema": str(torch.ops.aten.native_layer_norm.default._schema),
                "norm_results_release": "after checked stream completion", "provider": dict(self.info)}

    def bind(self, value: Tensor, buffers: Buffers) -> Bound:
        from cozy_runtime.internal import dense_linear
        if self.pid != os.getpid():
            raise RuntimeError("H3 output preparation cannot be inherited after fork")
        with self.lock:
            if self.bound:
                raise RuntimeError("H3 output preparation binds once")
            self.bound = True
        _scope()
        d = self.description
        if (torch.version.git_version != TORCH_COMMIT or not _ordinary(value)
                or _metadata(value) != d.input_metadata
                or tuple(_configuration(m) for m in (d.norm, d.projection)) != d.configurations
                or dense_linear.policy(value.device.index or 0) != self.policy):
            raise ValueError("H3 output invocation changed after query")
        if d.slots:
            from cozy_runtime.internal.execution.native_vae import _source_identity
            if _source_identity(d.models) != d.source_identity:
                raise ValueError("H3 output Source identity changed before binding")
            parameters = tuple(slot.bind() for slot in d.slots)
            if _source_identity(d.models) != d.source_identity:
                raise ValueError("H3 output Source identity changed during binding")
        else:
            parameters = (d.norm.weight, d.norm.bias, d.projection.weight, d.projection.bias)
            if any(t is not prior for t, prior in zip(parameters, d.parameters, strict=True)):
                raise ValueError("H3 output parameter was replaced")
        if any(not _ordinary(t) or _metadata(t) != metadata
               for t, metadata in zip(parameters, d.parameter_metadata, strict=True)):
            raise ValueError("H3 output source metadata changed")
        if any(t.data_ptr() % 16 for t in (value, *parameters)):
            raise ValueError("H3 output sources require aligned native layouts")
        expected = [(buffers.input_rows, (d.rows, WIDTH), torch.float16),
                    (buffers.dense_result, (d.rows, PROJECTED), torch.float16),
                    (buffers.result, d.result_shape, torch.float16),
                    (self.bias, (PROJECTED,), torch.float16)]
        if d.weight_cast:
            expected.append((buffers.weight_cast, (PROJECTED, WIDTH), torch.float16))
        elif buffers.weight_cast is not None:
            raise ValueError("undeclared H3 output weight cast")
        workspace = self.info["workspace_bytes"]
        if workspace:
            expected.append((buffers.workspace, (workspace,), torch.uint8))
        elif buffers.workspace is not None:
            raise ValueError("undeclared H3 output workspace")
        for tensor, shape, dtype in expected:
            if (tensor is None or not _ordinary(tensor) or tuple(tensor.shape) != shape
                    or tensor.dtype != dtype or tensor.device != value.device
                    or not tensor.is_contiguous() or tensor.data_ptr() % 256):
                raise ValueError("H3 output buffer differs from its declared allocation")
        destinations = [t for t, _, _ in expected]
        for index, tensor in enumerate(destinations):
            if any(torch._C._is_alias_of(tensor, other)
                   for other in (value, *parameters, *destinations[:index])):
                raise ValueError("H3 output writes require independent owned storage")
        matrix = buffers.weight_cast if d.weight_cast else parameters[2]
        bound = Bound(self, value, parameters, matrix, buffers)
        dense_linear.checked(self.plan.validate, *bound.arguments())
        bound.binding_state = bound.state()
        return bound


@dataclass
class Bound:
    prepared: Prepared
    value: Tensor
    parameters: tuple[Tensor, ...]
    matrix: Tensor
    buffers: Buffers
    norm_results: tuple[Tensor, Tensor, Tensor] | None = None
    claimed: bool = False
    submitted: bool = False
    submission_closed: bool = False
    completed: bool = False
    retained: bool = False
    binding_state: tuple[Any, ...] = ()
    lock: Any = field(default_factory=threading.Lock, repr=False)

    def state(self) -> tuple[Any, ...]:
        b = self.buffers
        values = (self.value, *self.parameters, self.matrix, self.prepared.bias,
                  b.input_rows, b.dense_result, b.result, b.weight_cast, b.workspace)
        return tuple(None if t is None else (id(t), _metadata(t), t.data_ptr(), _ordinary(t)) for t in values)

    def arguments(self) -> tuple[int, ...]:
        b = self.buffers
        return (_pointer(b.input_rows), _capacity(b.input_rows), _pointer(self.matrix), _capacity(self.matrix),
                _pointer(b.dense_result), _capacity(b.dense_result), _pointer(b.workspace), _capacity(b.workspace),
                self.prepared.stream.cuda_stream)

    def _process(self) -> None:
        if self.prepared.pid != os.getpid():
            raise RuntimeError("H3 output binding cannot be inherited after fork")

    def validate(self) -> None:
        from cozy_runtime.internal import dense_linear
        self._process()
        _scope()
        p, d = self.prepared, self.prepared.description
        if (torch.version.git_version != TORCH_COMMIT or self.state() != self.binding_state
                or tuple(_configuration(m) for m in (d.norm, d.projection)) != d.configurations
                or dense_linear.policy(self.value.device.index or 0) != p.policy):
            raise ValueError("H3 output binding/configuration/policy changed before execution")
        if d.slots:
            from cozy_runtime.internal.execution.native_vae import _source_identity
            if _source_identity(d.models) != d.source_identity:
                raise ValueError("H3 output Source identity changed before execution")
            for slot, value in zip(d.slots, self.parameters, strict=True):
                valid = (slot.module._parameters[slot.name] is slot.value if slot.model is None
                         else slot.block.region is not None
                         and value.data_ptr() == int(slot.block.region.ptr) + slot.offset)
                if not valid:
                    raise ValueError("H3 output raw Source binding changed before execution")
        else:
            current = (d.norm.weight, d.norm.bias, d.projection.weight, d.projection.bias)
            if any(value is not prior for value, prior in zip(current, self.parameters, strict=True)):
                raise ValueError("H3 output parameter changed before execution")
        dense_linear.checked(p.plan.validate, *self.arguments())

    def execute(self) -> Tensor:
        from cozy_runtime.internal import dense_linear
        self._process()
        _scope()
        with self.lock:
            if self.claimed:
                raise RuntimeError("H3 output binding is one-shot")
            self.claimed = True
        try:
            self.validate()
            self.submitted = True
            with torch.cuda.device(self.value.device), torch.cuda.stream(self.prepared.stream):
                norm_weight, norm_bias, weight, bias = self.parameters
                # This is the real functional provider. Its three allocations are
                # part of the one admitted inventory and are retained until completion.
                self.norm_results = torch.ops.aten.native_layer_norm.default(
                    self.value, [WIDTH], norm_weight, norm_bias, self.prepared.description.norm.eps)
                self.buffers.input_rows.copy_(self.norm_results[0].view(-1, WIDTH))
                if self.buffers.weight_cast is not None:
                    self.buffers.weight_cast.copy_(weight)
                self.prepared.bias.copy_(bias)
                dense_linear.checked(self.prepared.plan.execute, *self.arguments())
                assemble(self.buffers.dense_result, self.buffers.result)
            return self.buffers.result
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
        self.norm_results = None
        if self.retained:
            _UNPROVEN[:] = [owner for owner in _UNPROVEN if owner is not self]
            self.retained = False

    def poll(self) -> bool:
        self._process()
        if not self.submission_closed:
            raise RuntimeError("H3 output host submission has not closed")
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
            raise RuntimeError("H3 output host submission has not closed")
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
