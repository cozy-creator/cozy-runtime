"""How the engine's step launches a kernel of `_kernels` (`Run`): through its model's loaded set
(`kernels`) where the set declares the entry, else through Triton's JIT."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Protocol, cast

import torch

from cozy_runtime.accel.kernels import entry_key

if TYPE_CHECKING:
    from cozy_runtime.accel.kernels import JitKernel, KernelArgument, Launch, TritonEntry, TritonKernels

#: Triton's name of each dtype an engine kernel reads or writes.
TRITON = {
    torch.bfloat16: "bf16", torch.float16: "fp16", torch.float32: "fp32", torch.int32: "i32",
    torch.int64: "i64", torch.uint8: "u8", torch.bool: "i1",
}  # fmt: skip


def _off(entry: TritonEntry, names: list[str], args: tuple[object, ...]) -> str:
    """Why `args` don't fit `entry` as compiled, or "": a tensor of another dtype, or an operand it
    compiled as 16-aligned that isn't."""
    for name, value in zip(names, args, strict=True):
        declared = entry.signature[name]
        if isinstance(value, torch.Tensor):
            if declared != f"*{TRITON.get(value.dtype, '?')}":
                return f"{name} is {value.dtype}, compiled as {declared}"
            if name in entry.aligned and value.data_ptr() % 16:
                return f"{name} is not 16-byte aligned"
        elif isinstance(value, int) and declared == "i32" and name in entry.aligned and value % 16:
            return f"{name} = {value} is not a multiple of 16"
    return ""


class Run:
    """How the engine's step launches a kernel of `_kernels`: through `launch`, a loaded set's
    launcher, where `kernels` declares the entry the arguments' constexprs name and they fit it;
    else through Triton's JIT (`JIT`: no set), each such key said once."""

    def __init__(self, launch: Launch | None = None, kernels: TritonKernels | None = None) -> None:
        self.launch = launch
        self.entries = {entry.key: entry for entry in kernels.entries} if kernels is not None else {}
        self.said: set[str] = set()

    def __call__(self, kernel: JitKernel, grid: tuple[int, ...], *args: object, num_warps: int) -> None:
        if self.launch is not None:
            names = list(kernel.arg_names)
            key = entry_key(kernel.__name__, {n: args[i] for i, n in enumerate(names) if n.isupper()})
            entry = self.entries.get(key)
            why = "its set declares no such entry" if entry is None else _off(entry, names, args)
            if not why:
                self.launch(kernel, grid, *cast("tuple[KernelArgument, ...]", args))
                return
            if key not in self.said:
                self.said.add(key)
                print(f"[engine] {key} through Triton's JIT: {why}", file=sys.stderr)
        cast("_Jit", kernel)[grid](*args, num_warps=num_warps)


class _Jit(Protocol):
    def __getitem__(self, grid: tuple[int, ...]) -> _Launched: ...


class _Launched(Protocol):
    def __call__(self, *args: object, num_warps: int) -> object: ...


#: The step's kernels through Triton's JIT: a bench's own shapes, a decoder built without a set.
JIT = Run()
