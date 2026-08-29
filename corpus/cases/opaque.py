"""Constructs the derivation substrate cannot see through: custom ops, C extensions,
tensor subclasses that answer for themselves.

This is where v1 bled. `is_virtual` exists because pgw#1661 asked "does this tensor have
storage?" with `isinstance(t, FakeTensor)` and 300 torchao `Float8Tensor` weights over fake
data answered FALSE, pricing a config-only tree at 23 GB on an 8 GiB card. The wrapper
cases below are that defect, reproduced as package code.

v1 had NOTHING for custom `torch.library` ops — no `register_fake`, no meta-kernel
handling anywhere in its derivation path — and it FENCED OUT the artifact classes whose
module graph is decided by the weight table rather than the config (GGUF, w4a4, svdq),
refusing them outright rather than deriving them wrong. That refusal was correct and v2
keeps it by a different route: the provider/hardware VARIANT is part of candidate identity,
so those graphs derive per variant instead of not at all (`lifecycle.py`).
"""

from __future__ import annotations

import torch
from torch import nn

from corpus.harness import Case, Config, Pipeline

# A custom op with NO meta kernel: opaque to any weightless substrate by construction.
_LIB = torch.library.Library("cozy_corpus", "DEF")
_LIB.define("opaque_table(Tensor x, int n) -> Tensor")


def _opaque_cpu(x: torch.Tensor, n: int) -> torch.Tensor:
    return x.new_zeros(n, n)


_LIB.impl("opaque_table", _opaque_cpu, "CPU")

_LIB.define("registered_table(Tensor x, int n) -> Tensor")
_LIB.impl("registered_table", _opaque_cpu, "CPU")
# ...and the same op WITH one. A meta kernel is a shape function, which is exactly the
# fact a derivation needs and the only thing it needs.
_LIB.impl("registered_table", lambda x, n: x.new_empty(n, n), "Meta")


def _custom_op_no_meta(config: Config) -> Pipeline:
    width = config.as_int("width", 16)

    class Opaque(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = nn.Linear(width, width, dtype=torch.bfloat16)
            table = torch.ops.cozy_corpus.opaque_table(self.proj.weight, width)
            self.register_buffer("table", table)

    return Pipeline({"transformer": Opaque()})


def _custom_op_with_meta(config: Config) -> Pipeline:
    width = config.as_int("width", 16)

    class Registered(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = nn.Linear(width, width, dtype=torch.bfloat16)
            table = torch.ops.cozy_corpus.registered_table(self.proj.weight, width)
            self.register_buffer("table", table)

    return Pipeline({"transformer": Registered()})


def _triton_kernel(config: Config) -> Pipeline:
    """A Triton kernel armed during construction. Triton compiles for a real device; with
    no GPU the arming itself must fail by name rather than silently no-op."""
    width = config.as_int("width", 16)
    import triton
    import triton.language as tl

    @triton.jit  # type: ignore[misc]
    def _fill(out_ptr, n: tl.constexpr) -> None:
        offsets = tl.arange(0, n)
        tl.store(out_ptr + offsets, offsets.to(tl.float32))

    class Armed(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = nn.Linear(width, width, dtype=torch.bfloat16)
            out = torch.zeros(width, device="cuda")
            _fill[(1,)](out, n=width)
            self.register_buffer("table", out)

    return Pipeline({"transformer": Armed()})


class _Quantized(torch.Tensor):
    """A traceable wrapper subclass over a plain tensor — the `Float8Tensor` shape.

    It reports its OUTER metadata (shape, dtype, device) and holds its real storage inside.
    An `isinstance(t, FakeTensor)` virtuality check reads it as a full real checkpoint; a
    `t.device` check reads whatever the outer says. Only recursion answers correctly.
    """

    @staticmethod
    def __new__(cls, inner: torch.Tensor) -> _Quantized:
        return torch.Tensor._make_wrapper_subclass(  # type: ignore[attr-defined,no-any-return]
            cls, inner.shape, dtype=inner.dtype, device=inner.device, requires_grad=False
        )

    def __init__(self, inner: torch.Tensor) -> None:
        self.inner = inner

    def __tensor_flatten__(self) -> tuple[list[str], None]:
        return ["inner"], None

    @staticmethod
    def __tensor_unflatten__(held: dict, meta: None, size: object, stride: object) -> _Quantized:
        return _Quantized(held["inner"])

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):  # type: ignore[no-untyped-def]
        unwrapped = tuple(a.inner if isinstance(a, _Quantized) else a for a in args)
        result = func(*unwrapped, **(kwargs or {}))
        # re-wrapping is required: torch refuses a Parameter whose detach() changes type
        return _Quantized(result) if isinstance(result, torch.Tensor) else result


def _subclass_param(config: Config) -> Pipeline:
    width = config.as_int("width", 16)

    class Packed(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = nn.Linear(width, width, bias=False, dtype=torch.bfloat16)
            self.proj.weight = nn.Parameter(
                _Quantized(self.proj.weight.detach()), requires_grad=False
            )

    return Pipeline({"transformer": Packed()})


class _Loud(torch.Tensor):
    """A `__torch_function__` subclass that intercepts everything the harness does."""

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):  # type: ignore[no-untyped-def]
        with torch._C.DisableTorchFunction():
            result = func(*args, **(kwargs or {}))
        # re-wrapping is required: torch refuses a Parameter whose detach() changes type
        return result.as_subclass(cls) if isinstance(result, torch.Tensor) else result


def _torch_function_param(config: Config) -> Pipeline:
    width = config.as_int("width", 16)

    class Loud(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = nn.Linear(width, width, bias=False, dtype=torch.bfloat16)
            self.proj.weight = nn.Parameter(
                self.proj.weight.detach().as_subclass(_Loud), requires_grad=False
            )

    return Pipeline({"transformer": Loud()})


def _c_extension(config: Config) -> Pipeline:
    """A third-party C extension consuming a parameter during construction. numpy is the
    one every package already has, and it reaches storage through the buffer protocol."""
    width = config.as_int("width", 16)
    import numpy as np

    class Extension(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = nn.Linear(width, width, dtype=torch.float32)
            eigen = np.linalg.norm(self.proj.weight.detach().numpy())
            self.register_buffer("norm", torch.tensor(float(eigen)))

    return Pipeline({"transformer": Extension()})


CASES = (
    Case(
        "custom_op_no_meta_kernel",
        "a torch.library op with a CPU impl and no Meta impl",
        _custom_op_no_meta,
        "typed-harness-gap",
        names="opaque_table",
        note="the fix lane is a meta kernel in the package's own op, or this refusal",
    ),
    Case(
        "custom_op_with_meta_kernel",
        "the same op WITH a registered Meta impl",
        _custom_op_with_meta,
        "derived-correct",
        note="a meta kernel is a shape function — the only fact a derivation needs",
    ),
    Case(
        "triton_kernel_arming",
        "a Triton kernel launched during construction",
        _triton_kernel,
        "typed-refusal",
        names="device pin",
        ground_truth=False,
    ),
    Case(
        "wrapper_subclass_param",
        "a traceable wrapper subclass as a Parameter (the pgw#1661 shape)",
        _subclass_param,
        "derived-correct",
        note="is_virtual must recurse; the outer metadata is not the answer",
    ),
    Case(
        "torch_function_param",
        "a __torch_function__ subclass intercepting harness ops",
        _torch_function_param,
        "derived-correct",
    ),
    Case(
        "c_extension_numpy",
        "numpy consuming a parameter through the buffer protocol",
        _c_extension,
        "typed-refusal",
        names="host egress",
    ),
)
