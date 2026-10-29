"""Hand-written kernels a package brings to its own models, and the Runtime's own.

Model code lives in the model's package and the Runtime stays generic (owner ruling,
2026-10-14); kernels are hand-written, never `torch.compile`, and compile on the GPU that runs
them while the executor starts. Compiling fixed shipped source is permitted; consumer
benchmarking, autotuning, torch.compile and competing implementation search are not.
A package declares its Triton kernels once, at import, as a
`TritonKernels`. Each executor that imports the package submits their build for its GPUs'
architectures when it starts, beside the weight download; the machine's compile manager builds
each key once, niced, out of process. `Model.execution_plan` receives a `Plan` whose `load`
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
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, cast

from cozy_runtime import _kernels

ROWWISE_FP8 = "cozy.fp8-rowwise.native-leaf/1"
INT8_CONVROT = "cozy.int8-convrot256.native-leaf/1"
"""The `provider`s of the encoded leaves whose forward takes a `PreQuantized` operand."""

_NAME = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")


class KernelsUnavailable(RuntimeError):
    """A package's kernels cannot serve on this GPU now. `code`: `kernels_absent` (no machine
    kernel store, no Triton, a CPU, or no entry compiled for these constexprs),
    `kernels_failed` (this machine run's build failed) or `kernels_corrupt` (built, but it does
    not load: breakage, never a reason to serve another kernel quietly)."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(detail)
        self.code = code


@dataclass(frozen=True, slots=True)
class TritonEntry:
    """One compiled specialization of a module-level `@triton.jit` kernel: Triton's signature
    (`{"x_ptr": "*bf16", "n": "i32", "BLOCK": "constexpr"}`), the constexpr values, the pointer
    and integer arguments it may assume 16-aligned, its warps, its pipeline stages, and whether
    the compiler may fuse a multiply and an add (`fp_fusion`; False keeps each one rounded)."""

    kernel: Any
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


Launch = Callable[..., None]
"""`launch(kernel, grid, *args)`: run the entry whose constexpr values `args` carries, every
argument positional as the kernel declares it."""


def _unreported(name: str, document: Mapping[str, object]) -> None:
    """A plan run outside an executor (a test, a conformance script) reports nowhere."""


@dataclass(frozen=True, slots=True)
class Plan:
    """What Runtime hands `Model.execution_plan`: the GPU this construction serves on, the
    package's compiled kernels for it, and where its outcome is reported."""

    device: Any
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

    payload: Any
    scale: Any
    shape: tuple[int, ...]
    provider: str = ROWWISE_FP8


class _Shared:
    __slots__ = ("operands", "tensor")

    def __init__(self, tensor: Any) -> None:
        self.tensor = tensor
        self.operands: dict[str, PreQuantized | None] = {}


_SHARED: ContextVar[tuple[_Shared, ...]] = ContextVar("cozy_quantize_once", default=())


@contextmanager
def quantize_once(*tensors: Any) -> Iterator[None]:
    """Inside, the encoded leaves that take one of `tensors` (that object) as input quantize it
    once between them, as a fused qkv GEMM would: q, k and v of one normed input, or an MLP's
    gate and up. None is skipped. A tensor must not change in place inside the scope."""
    shared = tuple(_Shared(t) for t in tensors if t is not None)
    token = _SHARED.set(_SHARED.get() + shared)
    try:
        yield
    finally:
        _SHARED.reset(token)


def shared_operand(
    x: Any, provider: str, make: Callable[[], PreQuantized | None]
) -> PreQuantized | None:
    """`provider`'s operand for `x` inside a `quantize_once` scope that holds it, `make()` by the
    first leaf to ask; None outside one, or when `make` declines (the leaf's own op then serves
    it)."""
    for shared in _SHARED.get():
        if shared.tensor is x:
            if provider not in shared.operands:
                shared.operands[provider] = make()
            return shared.operands[provider]
    return None


def int8_rows(
    x: Any,
    consumers: tuple[Any, ...],
    *,
    norm: str | None = None,
    eps: float = 0.0,
    scale: Any = None,
    shift: Any = None,
    activation: str | None = None,
) -> PreQuantized | None:
    """The rows a producer would hand `consumers`, `activation(norm(x) * (1 + scale) + shift)`,
    rotated and quantized for int8 in one pass of the Runtime's op: pass the result to each
    consumer in place of its input. Each part is optional: `norm` "layer" or "rms" (no affine),
    `activation` "gelu" or "gelu_tanh"; `scale` and `shift` are `[..., width]` tables, and x's
    row r reads table row `r // (x rows / table rows)`. `activation="situ"` is gated and stands
    alone: x is `[..., 2F]`, each row its [hidden | gate] halves, and the F-wide rows are
    `4 tanh(g / 4) sigmoid(g) * 25 tanh(h / 25)` (MOSS-Image's SiTU).

    None unless every consumer is an un-hooked encoded leaf taking int8.convrot's operand (an int8
    or w4a8 leaf) of x's width and the op serves this card: the caller then runs its own ops."""
    if not consumers or getattr(consumers[0], "operand", None) != INT8_CONVROT:
        return None
    rows: PreQuantized | None = consumers[0].int8_prologue(
        x, consumers, norm, eps, scale, shift, activation
    )
    return rows


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


def read(tensor: Any) -> Any:
    """`readable` for one operand, without its context: a hot launch's (an int8 decode GEMV)."""
    bind = getattr(tensor, "_varena_read", None)
    return tensor if bind is None else bind()


def rms_rope_split_half(
    q: Any,
    k: Any,
    freqs: Any,
    q_scale: Any,
    k_scale: Any,
    epsilon: float = 1e-6,
    rot_dim: int = 0,
) -> tuple[Any, Any]:
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
    "read",
    "readable",
    "rms_rope_split_half",
    "shared_operand",
]
