"""Hand-written kernels a package brings to its own models, and the Runtime's own.

Model code lives in the model's package and the Runtime stays generic (owner ruling,
2026-10-14); kernels are hand-written, never `torch.compile`, and compile on the GPU that runs
them while the executor starts. A package declares its Triton kernels once, at import, as a
`TritonKernels`. Each executor that imports the package submits their build for its GPUs'
architectures when it starts, beside the weight download; the machine's compile manager builds
each key once, niced, out of process. `Model.execution_plan` receives a `Plan` whose `load`
waits for that build and returns the launcher. A request never compiles, and a launch maps any
weight operand the plane has not mapped (`readable`).

Also public here: the operand an `fp8-rowwise` encoded leaf takes already quantized
(`PreQuantized`), and the Runtime's compiled CUDA kernels (`rms_rope_split_half`), which ride
its linux x86 cp312-abi3 wheel.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from importlib.util import find_spec
from typing import Any, cast

from cozy_runtime import _kernels

ROWWISE_FP8 = "cozy.fp8-rowwise.native-leaf/1"
"""The `provider` of an encoded leaf whose forward takes a `PreQuantized` operand."""

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
    and integer arguments it may assume 16-aligned, and its warps."""

    kernel: Any
    signature: Mapping[str, str]
    constexprs: Mapping[str, int] = field(default_factory=dict)
    aligned: tuple[str, ...] = ()
    num_warps: int = 4

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
    keys the build in the machine's kernel store.
    """

    def __init__(self, name: str, entries: Iterable[TritonEntry]) -> None:
        if _NAME.fullmatch(name) is None:
            raise ValueError(f"kernel set name {name!r} must match {_NAME.pattern}")
        self.name = name
        self.entries = tuple(entries)
        if len({entry.key for entry in self.entries}) != len(self.entries):
            raise ValueError(f"{name} declares one entry twice")
        _SETS[name] = self

    @property
    def identity(self) -> str:
        """Triton's own hash of each kernel and every function it calls, with the entries."""
        rows = sorted(f"{entry.key}:{entry.kernel.cache_key}" for entry in self.entries)
        return "sha256:" + hashlib.sha256("\n".join(rows).encode()).hexdigest()


Launch = Callable[..., None]
"""`launch(kernel, grid, *args)`: run the entry whose constexpr values `args` carries, every
argument positional as the kernel declares it."""


@dataclass(frozen=True, slots=True)
class Plan:
    """What Runtime hands `Model.execution_plan`: the GPU this construction serves on, and the
    package's compiled kernels for it."""

    device: Any
    load: Callable[[TritonKernels], Launch]
    """Wait for this machine's build of the set for `device` (started when the executor did)
    and return its launcher. Raises `KernelsUnavailable`."""


@dataclass(frozen=True, slots=True)
class PreQuantized:
    """An activation already in the `fp8-rowwise` leaf's operand geometry (h3a-015).

    A fused kernel that holds a row in registers quantizes it there instead of writing bf16
    for the leaf to read back; the leaf then feeds the payload and per-row scale straight to
    its GEMM. `shape` is the bf16 activation's, so the leaf's output reshape is unchanged.
    """

    payload: Any
    scale: Any
    shape: tuple[int, ...]


def native() -> str:
    """Why this Runtime has no compiled CUDA kernels here, or "" when it has them."""
    if find_spec("cozy_runtime._kernels._C") is None:
        return "this Runtime wheel carries no compiled kernels (cozy_runtime._kernels._C)"
    return ""


@contextmanager
def readable[*Ts](*tensors: *Ts) -> Iterator[tuple[*Ts]]:
    """The operands of a kernel launched outside torch's dispatch (Triton, a compiled
    extension), which reads them by pointer. A weight with no plane bytes bound is a stand-in
    with nothing behind its pointer: its plane maps and holds it for the launch, or refuses
    (`weights.Unbound.reading`). Every such launch goes through here, so none is handed one."""
    standins = [t for t in tensors if hasattr(type(t), "reading")]
    if not standins:
        yield tensors
        return
    reading = getattr(type(standins[0]), "reading")  # noqa: B009
    with reading(standins, "a kernel launch") as read:
        yield cast("tuple[*Ts]", tuple(read.operand(t) for t in tensors))


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
    `native()` says why this Runtime cannot (only its linux x86 wheel carries the kernel).
    """
    with readable(q, k, freqs, q_scale, k_scale) as (q, k, freqs, q_scale, k_scale):
        return _kernels.rms_rope_split_half(q, k, freqs, q_scale, k_scale, epsilon, rot_dim)


__all__ = [
    "ROWWISE_FP8",
    "KernelsUnavailable",
    "Launch",
    "Plan",
    "PreQuantized",
    "TritonEntry",
    "TritonKernels",
    "native",
    "readable",
    "rms_rope_split_half",
]
