"""Hand-written kernels a package brings to its own models, and the Runtime's own.

Model code lives in the model's package and the Runtime stays generic (owner ruling,
2026-10-14); kernels are hand-written, never `torch.compile`, and compile on the GPU that runs
them while the executor starts. Compiling fixed shipped source is permitted; consumer
benchmarking, autotuning, torch.compile and competing implementation search are not.
A package declares its Triton kernels once, at import, as a `TritonKernels`. Its release ships
them compiled for every card row in `<package>/_triton` (`python -m
cozy_runtime.internal.triton_prebuild <package>/_triton --package <package>`, on a CPU). Each
executor that imports the package submits their build for its GPUs' architectures when it starts,
beside the weight download; the machine's compile manager copies what the package ships, else
builds each key once, niced, out of process. `Model.execution_plan` receives a `Plan` whose `load`
waits for that build and returns the launcher. A request never compiles, and a launch maps any
weight operand the plane has not mapped (`readable`).

Also public here: the operand an encoded leaf takes already quantized (`PreQuantized`), a scope
in which the leaves that read one activation quantize it once (`quantize_once`), a producer's
norm, modulation and activation folded into that quantization (`int8_rows`), and the
compiled CUDA kernels the native Runtime wheels carry (`rms_rope_split_half`).
"""

from __future__ import annotations

import hashlib
import re
import threading
import weakref
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from types import FunctionType
from typing import TYPE_CHECKING, Protocol, cast

from cozy_runtime import _kernels

if TYPE_CHECKING:
    import torch
    from torch import Tensor, nn

ROWWISE_FP8 = "cozy.fp8-rowwise.native-leaf/1"
INT8_CONVROT = "cozy.int8-convrot256.native-leaf/1"
"""The `provider`s of the encoded leaves whose forward takes a `PreQuantized` operand."""

_NAME = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")


class KernelsUnavailable(RuntimeError):
    """A package's kernels cannot serve on this GPU now. `code`: `kernels_absent` (no machine
    kernel store, no Triton, a CPU, or no entry compiled for these constexprs),
    `kernels_failed` (this machine run's build failed) or `kernels_corrupt` (built, but it does
    not load). The eager path serves any of them, and the run says why."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(detail)
        self.code = code


class JitKernel(Protocol):
    """What the build reads of a module-level `@triton.jit` kernel (Triton ships no types)."""

    __name__: str
    arg_names: list[str]

    @property
    def fn(self) -> FunctionType: ...

    @property
    def cache_key(self) -> str: ...


def jit[**P](fn: Callable[P, object]) -> JitKernel:
    """`triton.jit` for a kernel a set declares, typed (Triton ships no types)."""
    import triton

    kernel: JitKernel = triton.jit(fn)
    return kernel


@dataclass(frozen=True, slots=True)
class TritonEntry:
    """One compiled specialization of a module-level `@triton.jit` kernel: Triton's signature
    (`{"x_ptr": "*bf16", "n": "i32", "BLOCK": "constexpr"}`), the constexpr values, the pointer
    and integer arguments it may assume 16-aligned, its warps, its pipeline stages, and whether
    the compiler may fuse a multiply and an add (`fp_fusion`; False keeps each one rounded)."""

    kernel: JitKernel
    signature: Mapping[str, str]
    constexprs: Mapping[str, int] = field(default_factory=dict)
    aligned: tuple[str, ...] = ()
    num_warps: int = 4
    num_stages: int = 3
    fp_fusion: bool = True

    @property
    def options(self) -> dict[str, object]:
        """Its compile options, as Triton takes them."""
        return {
            "num_warps": self.num_warps,
            "num_stages": self.num_stages,
            "enable_fp_fusion": self.fp_fusion,
        }

    @property
    def key(self) -> str:
        return entry_key(self.kernel.__name__, self.constexprs)


def entry_key(kernel: str, constexprs: Mapping[str, object]) -> str:
    return f"{kernel}[" + ",".join(f"{k}={v}" for k, v in sorted(constexprs.items())) + "]"


#: Every set this process declared, by name: what an executor compiles when it starts.
_SETS: dict[str, TritonKernels] = {}


class TritonKernels:
    """A package's Triton kernels, compiled for each GPU architecture its executors hold.

    Declare it at module scope (`GLUE = TritonKernels("my-package-glue", [...])`); the kernels
    must be module-level functions, which the out-of-process build imports by name. The name
    keys the build in the machine's kernel store. `origin`: a module whose import declares
    the set, when that is not the kernels' own (a set Runtime builds from a declaration).
    """

    def __init__(self, name: str, entries: Iterable[TritonEntry], *, origin: str = "") -> None:
        if _NAME.fullmatch(name) is None:
            raise ValueError(f"kernel set name {name!r} must match {_NAME.pattern}")
        self.name = name
        self.origin = origin
        self.entries = tuple(entries)
        if len({entry.key for entry in self.entries}) != len(self.entries):
            raise ValueError(f"{name} declares one entry twice")
        _SETS[name] = self

    @property
    def modules(self) -> frozenset[str]:
        """What a process imports to hold this set: its kernels' modules and its origin."""
        found = {entry.kernel.fn.__module__ for entry in self.entries}
        return frozenset(found | {self.origin} if self.origin else found)

    @property
    def identity(self) -> str:
        """Triton's own hash of each kernel and every function it calls, with the entries."""
        rows = sorted(f"{entry.key}:{entry.kernel.cache_key}" for entry in self.entries)
        return "sha256:" + hashlib.sha256("\n".join(rows).encode()).hexdigest()


type KernelArgument = Tensor | int | float | None


class Launch(Protocol):
    """`launch(kernel, grid, *args)`: run the entry whose constexpr values `args` carries, every
    argument positional as the kernel declares it."""

    def __call__(
        self, kernel: JitKernel, grid: tuple[int, ...], /, *args: KernelArgument
    ) -> None: ...


def _unreported(name: str, document: Mapping[str, object]) -> None:
    """A plan run outside an executor (a test, a conformance script) reports nowhere."""


@dataclass(frozen=True, slots=True)
class Plan:
    """What Runtime hands `Model.execution_plan`: the GPU this construction serves on, the
    package's compiled kernels for it, and where its outcome is reported."""

    device: torch.device
    load: Callable[[TritonKernels], Launch]
    """Wait for this machine's build of the set for `device` (started when the executor did)
    and return its launcher. Raises `KernelsUnavailable`."""
    when_ready: Callable[[TritonKernels, Callable[[Launch], None]], bool] | None = None
    """`install(launch)` once the set is built for every GPU of the construction, on each at
    the same boundary: the end of its preparation, else the start of the first request after.
    A construction serves while its kernels compile, and a run never waits for one. True when
    this GPU's build is done now. Raises `KernelsUnavailable` as `load` does; an install that
    fails leaves the construction as it was. None: `load`, then install."""
    report: Callable[[str, Mapping[str, object]], None] = _unreported
    """What the plan installed under `name`: applied or not, with its code and detail. The
    construction's receipts carry it (`measurements.execution.plan_reports`), so a refusal shows
    in `run show`, not only in the executor's log."""


@dataclass(frozen=True, slots=True)
class PreQuantized:
    """An activation already in an encoded leaf's operand geometry (h3a-015): `provider`'s.

    A fused kernel that holds a row in registers quantizes it there instead of writing bf16
    for the leaf to read back; the leaf then feeds the payload and per-row scale straight to
    its GEMM. `shape` is the bf16 activation's, so the leaf's output reshape is unchanged.
    """

    payload: Tensor
    scale: Tensor
    shape: tuple[int, ...]
    provider: str = ROWWISE_FP8


class _Shared:
    __slots__ = ("operands", "tensor")

    def __init__(self, tensor: Tensor) -> None:
        self.tensor = tensor
        self.operands: dict[str, PreQuantized | None] = {}


_SHARED: ContextVar[tuple[_Shared, ...]] = ContextVar("cozy_quantize_once", default=())


@contextmanager
def quantize_once(*tensors: Tensor | None) -> Iterator[None]:
    """Inside, the encoded leaves that take one of `tensors` (that object) as input quantize it
    once between them, as a fused qkv GEMM would: q, k and v of one normed input, or an MLP's
    gate and up. None is skipped. A tensor must not change in place inside the scope."""
    shared = tuple(_Shared(t) for t in tensors if t is not None)
    token = _SHARED.set(_SHARED.get() + shared)
    try:
        yield
    finally:
        _SHARED.reset(token)


@dataclass(slots=True)
class _Recent:
    """The operand an encoded leaf last quantized on this thread, while sibling leaves may still
    read the same input: `tensor` unchanged (its version, storage and geometry) and `provider`'s.
    `left` is how many more its maker's group read (learned), None until it is known."""

    tensor: weakref.ref[Tensor] | None
    version: int
    key: tuple[object, ...]
    provider: str
    operand: PreQuantized | None
    maker: _Learns | None
    uses: int
    left: int | None


class _Learns(Protocol):
    """A leaf that reads shared operands: `_cozy_siblings`, once learned, is how many leaves read
    its input (an int torch's `Module.__setattr__` typing cannot admit)."""

    _cozy_siblings: int


class _Prologue(Protocol):
    """The native int8 leaf's producer prologue (`int8_rows`)."""

    def int8_prologue(
        self,
        x: Tensor,
        consumers: tuple[nn.Module, ...],
        norm: str | None,
        eps: float,
        scale: Tensor | None,
        shift: Tensor | None,
        activation: str | None,
        up: Tensor | None = None,
    ) -> PreQuantized | None: ...


_RECENT = threading.local()


def _geometry(x: Tensor) -> tuple[object, ...]:
    return (x.data_ptr(), tuple(x.shape), tuple(x.stride()), x.dtype)


def _retire(held: _Recent) -> None:
    """Drop the held operand at once (its device memory returns now, not at a cycle collection);
    its maker learns how many leaves read it, the most it has seen."""
    if getattr(_RECENT, "held", None) is held:
        _RECENT.held = None
    if held.maker is not None:
        held.maker._cozy_siblings = max(getattr(held.maker, "_cozy_siblings", 1), held.uses)
    held.operand = held.maker = held.tensor = None


#: The group whose last learned reader just retired its operand: (input, version, geometry, maker).
_RECENT.last = None


def shared_operand(
    x: Tensor,
    provider: str,
    make: Callable[[], PreQuantized | None],
    consumer: nn.Module | None = None,
) -> PreQuantized | None:
    """`provider`'s operand for `x`, `make()` by the first leaf to ask, the rest reading it: inside
    a `quantize_once` scope that holds `x`, and otherwise for the sibling leaves that read one
    unchanged input in a row (q, k and v of one normed input, an MLP's gate and up, any model).
    Outside a scope the operand is held only until the leaves that read it before (`consumer`'s
    learned group) have read it again, another input is quantized, or `x` is freed: one operand
    held at most, never a stale one. No `consumer` (a leaf recording a graph), or an inference
    tensor (no version counter to see an in-place change by), shares in scopes only. None when
    `make` declines (the leaf's own op serves it)."""
    for shared in _SHARED.get():
        if shared.tensor is x:
            if provider not in shared.operands:
                shared.operands[provider] = make()
            return shared.operands[provider]
    held: _Recent | None = getattr(_RECENT, "held", None)
    if consumer is None or x.is_inference():  # an inference tensor counts no in-place change
        if held is not None:
            _retire(held)
        return None
    if (
        held is not None
        and held.tensor is not None
        and held.provider == provider
        and held.tensor() is x
        and held.version == x._version
        and held.key == _geometry(x)
    ):
        held.uses += 1
        operand = held.operand
        if held.left is not None:
            held.left -= 1
            if held.left <= 0:
                # Its group has read it; one more reader of the same input grows the group.
                _RECENT.last = (held.tensor, held.version, held.key, held.maker)
                _retire(held)
        return operand
    if held is not None:
        _retire(held)
    last = getattr(_RECENT, "last", None)
    _RECENT.last = None
    if last is not None and last[0]() is x and last[1:3] == (x._version, _geometry(x)):
        last[3]._cozy_siblings += 1
    operand = make()
    if operand is None:
        return None
    group = getattr(consumer, "_cozy_siblings", None)
    if group == 1:  # this leaf reads its input alone: nothing to hold
        return operand
    entry = _Recent(
        weakref.ref(x, lambda _: _retire(entry)),
        x._version,
        _geometry(x),
        provider,
        operand,
        cast("_Learns", consumer),  # it learns its group on retiring (`_retire`)
        1,
        None if group is None else group - 1,
    )
    _RECENT.held = entry
    return operand


def int8_rows(
    x: Tensor,
    consumers: tuple[nn.Module, ...],
    *,
    norm: str | None = None,
    eps: float = 0.0,
    scale: Tensor | None = None,
    shift: Tensor | None = None,
    activation: str | None = None,
    up: Tensor | None = None,
) -> PreQuantized | None:
    """The rows a producer would hand `consumers`, `activation(norm(x) * (1 + scale) + shift)`,
    rotated and quantized for int8 in one pass of the Runtime's op: pass the result to each
    consumer in place of its input. Each part is optional: `norm` "layer" or "rms" (no affine),
    `activation` "gelu" or "gelu_tanh"; `scale` and `shift` are `[..., width]` tables, and x's
    row r reads table row `r // (x rows / table rows)`. `activation="situ"` is gated and stands
    alone: x is `[..., 2F]`, each row its [hidden | gate] halves, and the F-wide rows are
    `4 tanh(g / 4) sigmoid(g) * 25 tanh(h / 25)` (MOSS-Image's SiTU). `activation="swiglu"`
    stands alone too: x is the gate rows and `up` the up rows of its shape, and the rows are
    `silu(x) * up`, rounded as an eager SwiGLU rounds them; without `up`, x is `[..., 2F]`, each
    row the [up | gate] halves of one fused projection (Diffusers' `SwiGLU`).

    None unless every consumer is an encoded leaf taking int8.convrot's operand (an int8 or w4a8
    leaf) of x's width, or a LoRA'd or LoKr'd layer whose update is baked into one, with no hook
    that changes its forward, and the op serves this card: the caller then runs its own ops."""
    consumers = tuple(_baked_base(c) for c in consumers)
    if not consumers or getattr(consumers[0], "operand", None) != INT8_CONVROT:
        return None
    # Only the native int8 leaf's operand is int8.convrot's, and it has the prologue.
    leader = cast("_Prologue", consumers[0])
    return leader.int8_prologue(x, consumers, norm, eps, scale, shift, activation, up)


def _baked_base(module: nn.Module) -> nn.Module:
    """The leaf a consumer's forward hands its operand to: itself, or a composed layer's base
    (`lora_composition`'s `_cozy_baked`) once its update is in the base's bytes."""
    while getattr(module, "_cozy_baked", False) and not (
        module._forward_hooks or module._forward_pre_hooks
    ):
        base: nn.Module | None = getattr(module, "base_layer", None)
        if base is None:
            break
        module = base
    return module


def native() -> str:
    """Why this machine has no compiled CUDA kernels, or "" when it has them."""
    return _kernels.ABSENT


@contextmanager
def readable[*Ts](*tensors: *Ts) -> Iterator[tuple[*Ts]]:
    """The operands of a kernel launched outside torch's dispatch (Triton, a compiled
    extension), which reads them by pointer. A weight varena has not bound now (no forward of
    its block or model runs) is a placeholder whose pointer is 0: its read binds it, and the
    tensor read holds the pin until it dies, past the launch (`Placeholder._varena_read`).
    Every such launch goes through here, so none is handed a null pointer (cr-208)."""
    yield cast("tuple[*Ts]", tuple(read(t) for t in tensors))


def read[T](tensor: T) -> T:
    """`readable` for one operand, without its context: a hot launch's (an int8 decode GEMV)."""
    bind: Callable[[], T] | None = getattr(tensor, "_varena_read", None)
    return tensor if bind is None else bind()


def rms_rope_split_half(
    q: Tensor,
    k: Tensor,
    freqs: Tensor,
    q_scale: Tensor,
    k_scale: Tensor,
    epsilon: float = 1e-6,
    rot_dim: int = 0,
) -> tuple[Tensor, Tensor]:
    """Per-head RMSNorm of `q` and `k` (scaled by `q_scale`/`k_scale`), then split-half RoPE by
    the `[..., rot_dim // 2, 2, 2]` rotation matrix `freqs`, out of place, in one launch.
    `native()` says why this machine cannot (its Runtime wheel carries no kernels).
    """
    with readable(q, k, freqs, q_scale, k_scale) as (q, k, freqs, q_scale, k_scale):
        return _kernels.rms_rope_split_half(q, k, freqs, q_scale, k_scale, epsilon, rot_dim)


__all__ = [
    "INT8_CONVROT",
    "ROWWISE_FP8",
    "KernelsUnavailable",
    "Launch",
    "Plan",
    "PreQuantized",
    "TritonEntry",
    "TritonKernels",
    "int8_rows",
    "native",
    "quantize_once",
    "rms_rope_split_half",
]
