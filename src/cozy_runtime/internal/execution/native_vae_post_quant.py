"""Prepared pointwise-GEMM alternative to H3's biased 24-channel Conv3d.

No chooser or module hook is installed here. The caller admits the returned actual
allocation inventory, invokes ordinary source hooks once, binds, and submits once.
This is an alternate implementation, not a claim of cuDNN numerical equivalence.
"""

from __future__ import annotations

import math
import os
import threading
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor, nn

from cozy_runtime.internal.execution.native_vae_metadata import TensorSpec

IMPLEMENTATION = "h3-post-quant-pointwise-lt/1"
CHANNELS = 24
_UNPROVEN: list[Bound] = []


def _metadata(value: Tensor) -> tuple[Any, ...]:
    return tuple(value.shape), tuple(value.stride()), value.dtype, value.device


def _ordinary(value: Any) -> bool:
    return (
        type(value) in (Tensor, nn.Parameter)
        and value.layout == torch.strided
        and not value.is_conj()
        and not value.is_neg()
        and not value.requires_grad
    )


def _configuration(module: nn.Conv3d) -> tuple[Any, ...]:
    hooks = nn.modules.module
    return (
        module.in_channels,
        module.out_channels,
        module.kernel_size,
        module.stride,
        module.padding,
        module.dilation,
        module.groups,
        module.padding_mode,
        module.training,
        type(module).forward,
        vars(module).get("forward"),
        vars(module).get("_conv_forward"),
        module._compiled_call_impl,
        tuple(id(hook) for hook in module._forward_pre_hooks.values()),
        tuple(id(hook) for hook in module._forward_hooks.values()),
        tuple(id(hook) for hook in module._backward_pre_hooks.values()),
        tuple(id(hook) for hook in module._backward_hooks.values()),
        tuple(id(hook) for hook in hooks._global_forward_pre_hooks.values()),
        tuple(id(hook) for hook in hooks._global_forward_hooks.values()),
    )


@dataclass(frozen=True)
class Description:
    """Metadata only; this retains no new physical weight view or native allocation."""

    module: nn.Conv3d
    input_metadata: tuple[Any, ...]
    weight_metadata: tuple[Any, ...]
    bias_metadata: tuple[Any, ...]
    configuration: tuple[Any, ...]
    weight: Any
    bias: Any
    slots: tuple[Any, ...] = ()
    models: tuple[Any, ...] = ()
    source_identity: tuple[Any, ...] = ()

    @property
    def shape(self) -> tuple[int, ...]:
        return self.input_metadata[0]

    @property
    def rows(self) -> int:
        batch, _, depth, height, width = self.shape
        return batch * depth * height * width

    @property
    def weight_cast(self) -> bool:
        return self.weight_metadata[2] is torch.float32

    def allocations(self, workspace_bytes: int) -> tuple[tuple[str, int], ...]:
        """Actual tensor payloads, not rounded capacities or a memory estimate.

        Bias is the separate preparation holding required by the provider query.
        The caller must count that already-live allocation once, not add it here.
        """
        if type(workspace_bytes) is not int or workspace_bytes < 0:
            raise ValueError("workspace must be the nonnegative provider query result")
        matrix = self.rows * CHANNELS * 2
        result = [("input_rows", matrix), ("dense_result", matrix), ("result", matrix)]
        if self.weight_cast:
            result.append(("weight_cast", CHANNELS * CHANNELS * 2))
        if workspace_bytes:
            result.append(("workspace", workspace_bytes))
        return tuple(result)

    def buffer_specs(self, workspace_bytes: int) -> tuple[tuple[str, tuple[int, ...], Any], ...]:
        self.allocations(workspace_bytes)
        specs = [("input_rows", (self.rows, CHANNELS), torch.float16),
                 ("dense_result", (self.rows, CHANNELS), torch.float16),
                 ("result", self.shape, torch.float16)]
        if self.weight_cast:
            specs.append(("weight_cast", (CHANNELS, CHANNELS), torch.float16))
        if workspace_bytes:
            specs.append(("workspace", (workspace_bytes,), torch.uint8))
        return tuple(specs)


def describe(module: Any, value: Any) -> Description | None:
    """Decline unsupported metadata without initializing CUDA or reading weights."""
    spec = TensorSpec.read(value)
    if spec is None:
        return None
    if (
        type(module) is not nn.Conv3d
        or module.training
        or module._compiled_call_impl is not None
        or (
            "forward" in vars(module)
            and module.forward is not vars(module).get("_cozy_execution_forward")
        )
        or ("_conv_forward" in vars(module) and not vars(module).get("_cozy_cut"))
        or module.in_channels != CHANNELS
        or module.out_channels != CHANNELS
        or module.kernel_size != (1, 1, 1)
        or module.stride != (1, 1, 1)
        or module.dilation != (1, 1, 1)
        or module.padding != (0, 0, 0)
        or module.groups != 1
        or module.padding_mode != "zeros"
        or len(spec.shape) != 5
        or spec.shape[1] != CHANNELS
        or min(spec.shape) < 1
        or not spec.is_contiguous()
        or spec.dtype not in (torch.float16, torch.float32)
    ):
        return None
    weight, bias = module._parameters.get("weight"), module._parameters.get("bias")
    expected = ((CHANNELS, CHANNELS, 1, 1, 1), (CHANNELS,))
    if any(
        tensor is None
        or tuple(tensor.shape) != shape
        or tensor.dtype not in (torch.float16, torch.float32)
        or tensor.device != spec.device
        or not tensor.is_contiguous()
        for tensor, shape in zip((weight, bias), expected, strict=True)
    ):
        return None
    if any(type(t) in (Tensor, nn.Parameter) and not _ordinary(t) for t in (weight, bias)):
        return None
    rows = math.prod(spec.shape) // CHANNELS
    if rows > 2**31 - 1:
        return None
    slots: tuple[Any, ...] = ()
    models: tuple[Any, ...] = ()
    source_identity: tuple[Any, ...] = ()
    if (
        not all(_ordinary(tensor) for tensor in (weight, bias))
        or any(hasattr(tensor, "_varena_read") for tensor in (weight, bias))
        or module._forward_pre_hooks
        or module._forward_hooks
    ):
        # Reuse the reviewed raw-source slot contract, not a parallel binding layer.
        if spec.device.type != "cuda":
            return None
        from cozy_runtime.internal.execution.native_vae import (
            _slot,
            _source_identity,
            _Unsupported,
        )
        from cozy_runtime.internal.residency import _REQUEST, Placeholder

        request = _REQUEST.get()
        if request is None or any(
            type(t) not in (Tensor, nn.Parameter, Placeholder) for t in (weight, bias)
        ):
            return None
        with request.memory.lock:
            models = tuple(m for m in request.memory.models.values() if m.name in request.current)
        try:
            slots = tuple(
                _slot(
                    module,
                    name,
                    shape,
                    (torch.float16, torch.float32),
                    models,
                    matrix=name == "weight",
                )
                for name, shape in zip(("weight", "bias"), expected, strict=True)
            )
            owners = {id(slot.model) for slot in slots if slot.model is not None}
            models = tuple(model for model in models if id(model) in owners)
            source_identity = _source_identity(models)
        except _Unsupported:
            return None
    owned_hooks = {id(hook) for model in models for hook in model.source_hooks}
    hooks = nn.modules.module
    if (
        hooks._global_forward_pre_hooks
        or hooks._global_forward_hooks
        or module._backward_hooks
        or module._backward_pre_hooks
        or any(
            id(hook) not in owned_hooks
            for hook in (*module._forward_pre_hooks.values(), *module._forward_hooks.values())
        )
    ):
        return None
    return Description(
        module,
        spec.metadata,
        _metadata(weight),
        _metadata(bias),
        _configuration(module),
        weight,
        bias,
        slots,
        models,
        source_identity,
    )


def pack_input(value: Tensor, target: Tensor) -> None:
    """Copy/cast into row-major voxels without a temporary permute/reshape copy."""
    batch, channels, depth, height, width = value.shape
    target.view(batch, depth * height * width, channels).copy_(value.flatten(2).transpose(1, 2))


def unpack_output(rows: Tensor, target: Tensor) -> None:
    batch, channels, depth, height, width = target.shape
    target.flatten(2).copy_(rows.view(batch, depth * height * width, channels).transpose(1, 2))


@dataclass
class Buffers:
    input_rows: Tensor
    dense_result: Tensor
    result: Tensor
    weight_cast: Tensor | None
    workspace: Tensor | None

    @classmethod
    def allocate(cls, description: Description, workspace_bytes: int) -> Buffers:
        """Caller invokes this only after admitting the complete allocation list."""
        description.allocations(workspace_bytes)
        options = {"device": description.input_metadata[3], "dtype": torch.float16}
        return cls(
            torch.empty((description.rows, CHANNELS), **options),
            torch.empty((description.rows, CHANNELS), **options),
            torch.empty(description.shape, **options),
            torch.empty((CHANNELS, CHANNELS), **options) if description.weight_cast else None,
            torch.empty(workspace_bytes, dtype=torch.uint8, device=options["device"])
            if workspace_bytes
            else None,
        )


def _capacity(tensor: Tensor | None) -> int:
    return 0 if tensor is None else tensor.numel() * tensor.element_size()


def _pointer(tensor: Tensor | None) -> int:
    return 0 if tensor is None else tensor.data_ptr()


def _overlap(left: Tensor, right: Tensor) -> bool:
    # Imported/exported tensors may have distinct Storage objects for overlapping
    # memory. Bounds here are valid only after contiguous same-device validation.
    if left.device != right.device or not left.is_contiguous() or not right.is_contiguous():
        raise ValueError("VAE span checks require contiguous tensors on one device")
    return (torch._C._is_alias_of(left, right)
            or max(left.data_ptr(), right.data_ptr())
            < min(left.data_ptr() + _capacity(left), right.data_ptr() + _capacity(right)))


def _no_retry(error: BaseException) -> None:
    error.varena_no_retry = True  # type: ignore[attr-defined]


def _scope() -> None:
    from cozy_runtime.internal.execution import catch

    if catch.uncommitted_scope():
        raise ValueError("post-quant cannot execute inside a retryable parent cut")


def prepare(description: Description, *, context: Any, stream: Any) -> Prepared | None:
    """Query the existing device-owned provider; no source hook or numerical write."""
    if description.input_metadata[3].type != "cuda" or torch.is_grad_enabled():
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
    # A real execution bias buffer is required by the provider's alignment query.
    with torch.cuda.device(device), torch.cuda.stream(stream):
        if torch.cuda.is_current_stream_capturing():
            return None
        bias = torch.empty(CHANNELS, dtype=torch.float16, device=device)
        try:
            plan = dense_linear.checked(
                context.query,
                description.rows,
                CHANNELS,
                CHANNELS,
                bias.data_ptr(),
                _capacity(bias),
                policy.half_accumulation,
                policy.reduction,
                policy.sm_target,
                policy.workspace_limit,
                stream.cuda_stream,
            )
        except RuntimeError as error:
            cause = error.__cause__
            if (
                isinstance(cause, _dense.Error)
                and cause.domain == "cublas"
                and cause.status == _dense.NOT_SUPPORTED
            ):
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
        return {
            "implementation": IMPLEMENTATION,
            "shape": self.description.shape,
            "input_dtype": str(self.description.input_metadata[2]),
            "weight_dtype": str(self.description.weight_metadata[2]),
            "bias_dtype": str(self.description.bias_metadata[2]),
            "preparation_allocations": (("bias", _capacity(self.bias)),),
            "execution_allocations": self.allocations,
            "provider": dict(self.info),
            "cudnn_equivalence": False,
        }

    def bind(self, value: Tensor, buffers: Buffers) -> Bound:
        """Retain actual source views after the caller's ordinary source hooks."""
        from cozy_runtime.internal import dense_linear

        if self.pid != os.getpid():
            raise RuntimeError("post-quant preparation cannot be inherited after fork")
        with self.lock:
            if self.bound:
                raise RuntimeError("post-quant preparation binds one invocation only")
            self.bound = True
        _scope()
        d = self.description
        if (
            _configuration(d.module) != d.configuration
            or _metadata(value) != d.input_metadata
            or not _ordinary(value)
        ):
            raise ValueError("post-quant input/module changed after query")
        if dense_linear.policy(value.device.index or 0) != self.policy:
            raise ValueError("post-quant math policy changed after query")
        if d.slots:
            from cozy_runtime.internal.execution.native_vae import _source_identity

            if _source_identity(d.models) != d.source_identity:
                raise ValueError("post-quant Source identity changed before binding")
            weight, bias = (slot.bind() for slot in d.slots)
            if _source_identity(d.models) != d.source_identity:
                raise ValueError("post-quant Source identity changed during binding")
        else:
            weight, bias = d.module.weight, d.module.bias
            if weight is not d.weight or bias is not d.bias:
                raise ValueError("post-quant source parameter was replaced")
        if (
            not _ordinary(weight)
            or not _ordinary(bias)
            or _metadata(weight) != d.weight_metadata
            or _metadata(bias) != d.bias_metadata
        ):
            raise ValueError("post-quant bound source layout changed")
        expected = [
            (buffers.input_rows, (d.rows, CHANNELS), torch.float16),
            (buffers.dense_result, (d.rows, CHANNELS), torch.float16),
            (buffers.result, d.shape, torch.float16),
            (self.bias, (CHANNELS,), torch.float16),
        ]
        if d.weight_cast:
            expected.append((buffers.weight_cast, (CHANNELS, CHANNELS), torch.float16))
        elif buffers.weight_cast is not None:
            raise ValueError("undeclared post-quant weight cast allocation")
        workspace = self.info["workspace_bytes"]
        if workspace:
            expected.append((buffers.workspace, (workspace,), torch.uint8))
        elif buffers.workspace is not None:
            raise ValueError("undeclared post-quant workspace allocation")
        for tensor, shape, dtype in expected:
            if (
                tensor is None
                or not _ordinary(tensor)
                or tuple(tensor.shape) != shape
                or tensor.dtype != dtype
                or tensor.device != value.device
                or not tensor.is_contiguous()
                or tensor.data_ptr() % 256
            ):
                raise ValueError("post-quant buffer differs from the declared allocation")
        outputs = [tensor for tensor, _, _ in expected]
        for index, tensor in enumerate(outputs):
            if any(
                _overlap(tensor, other)
                for other in [value, weight, bias, *outputs[:index]]
            ):
                raise ValueError("post-quant writes require independent owned storage")
        matrix = buffers.weight_cast if d.weight_cast else weight.view(CHANNELS, CHANNELS)
        bound = Bound(self, value, weight, bias, matrix, buffers)
        dense_linear.checked(self.plan.validate, *bound.arguments())
        bound.binding_state = bound.state()
        return bound


@dataclass
class Bound:
    """One submission and its exact owners; finish/poll before discarding it.

    Inputs/sources must already be ordered on the explicit stream and remain
    immutable through completion. Returned ordinary result tensors retain their
    own Working storage; later consumers retain/order/record their own streams.
    """

    prepared: Prepared
    value: Tensor
    weight: Tensor
    bias: Tensor
    matrix: Tensor
    buffers: Buffers
    submitted: bool = False
    claimed: bool = False
    submission_closed: bool = False
    completed: bool = False
    retained: bool = False
    binding_state: tuple[Any, ...] = ()
    lock: Any = field(default_factory=threading.Lock, repr=False)

    def state(self) -> tuple[Any, ...]:
        b = self.buffers
        tensors = (
            self.value,
            self.weight,
            self.bias,
            self.matrix,
            self.prepared.bias,
            b.input_rows,
            b.dense_result,
            b.result,
            b.weight_cast,
            b.workspace,
        )
        return tuple(
            None if t is None else (id(t), _metadata(t), t.data_ptr(), _ordinary(t))
            for t in tensors
        )

    def arguments(self) -> tuple[int, ...]:
        b = self.buffers
        return (
            _pointer(b.input_rows),
            _capacity(b.input_rows),
            _pointer(self.matrix),
            _capacity(self.matrix),
            _pointer(b.dense_result),
            _capacity(b.dense_result),
            _pointer(b.workspace),
            _capacity(b.workspace),
            self.prepared.stream.cuda_stream,
        )

    def _check_process(self) -> None:
        if self.prepared.pid != os.getpid():
            raise RuntimeError("post-quant binding cannot be inherited after fork")

    def _retain(self) -> None:
        if not self.retained:
            _UNPROVEN.append(self)
            self.retained = True

    def _complete(self) -> None:
        self.completed = True
        if self.retained:
            _UNPROVEN[:] = [owner for owner in _UNPROVEN if owner is not self]
            self.retained = False

    def validate(self) -> None:
        """Recheck selected bindings without another source read or preparation."""
        from cozy_runtime.internal import dense_linear

        self._check_process()
        p, d = self.prepared, self.prepared.description
        _scope()
        if self.state() != self.binding_state:
            raise ValueError("post-quant tensor ownership or layout changed after binding")
        if (
            _configuration(d.module) != d.configuration
            or _metadata(self.value) != d.input_metadata
            or _metadata(self.weight) != d.weight_metadata
            or _metadata(self.bias) != d.bias_metadata
            or dense_linear.policy(self.value.device.index or 0) != p.policy
        ):
            raise ValueError("post-quant invocation changed before execution")
        if d.slots:
            from cozy_runtime.internal.execution.native_vae import _source_identity

            if _source_identity(d.models) != d.source_identity:
                raise ValueError("post-quant Source identity changed before execution")
            for slot, value in zip(d.slots, (self.weight, self.bias), strict=True):
                if slot.model is None:
                    valid = slot.module._parameters[slot.name] is slot.value
                else:
                    valid = (
                        slot.block.region is not None
                        and value.data_ptr() == int(slot.block.region.ptr) + slot.offset
                    )
                if not valid:
                    raise ValueError("post-quant raw Source binding changed before execution")
        elif d.module.weight is not self.weight or d.module.bias is not self.bias:
            raise ValueError("post-quant source parameter changed before execution")
        dense_linear.checked(p.plan.validate, *self.arguments())

    def execute(self) -> Tensor:
        """No selection, query, tensor allocation or retry after the first copy."""
        from cozy_runtime.internal import dense_linear

        self._check_process()
        _scope()
        with self.lock:
            if self.claimed:
                raise RuntimeError("post-quant binding is one-shot")
            self.claimed = True
        try:
            # No source hooks/read, query or tensor allocation occurs on this check.
            # A claimed invocation never reenters an application chooser, even when
            # the final pre-write check detects that its binding became stale.
            self.validate()
            self.submitted = True  # A failed CUDA call can already have submitted work.
            with torch.cuda.device(self.value.device), torch.cuda.stream(self.prepared.stream):
                pack_input(self.value, self.buffers.input_rows)
                if self.buffers.weight_cast is not None:
                    self.buffers.weight_cast.copy_(self.weight.view(CHANNELS, CHANNELS))
                self.prepared.bias.copy_(self.bias)
                dense_linear.checked(self.prepared.plan.execute, *self.arguments())
                unpack_output(self.buffers.dense_result, self.buffers.result)
            return self.buffers.result
        except BaseException as error:
            _no_retry(error)
            if self.submitted:
                self._retain()
            raise
        finally:
            self.submission_closed = True

    def poll(self) -> bool:
        self._check_process()
        if not self.submission_closed:
            raise RuntimeError("post-quant host submission has not closed")
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
        self._check_process()
        if not self.submission_closed:
            raise RuntimeError("post-quant host submission has not closed")
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
