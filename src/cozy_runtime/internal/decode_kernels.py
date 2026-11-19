"""The decode cores' GPU calls as one declared Runtime kernel set (xs-213). Every row of
`decode_schemes` is traced through the cores (`_Traced`, a launch that launches nothing) at each
listed value of its params, with each combination of optional parts its rules allow, into each
destination dtype, so the set holds the rows' own calls (`kernels`). It builds where the
Runtime's sets do, at machine start (`machine_kernels.triton_sets`) and into the wheel
(`triton_prebuild`), opens at device init (`prime`), and serves once every traced call decodes on
the card bit for bit as the reference does (`probe`). A decode never compiles: until the set
serves, and for a call it does not hold, the reference decodes on the host (`decode`).
"""

from __future__ import annotations

import itertools
import sys
from collections.abc import Iterator, Mapping
from functools import cache, partial
from types import ModuleType
from typing import TYPE_CHECKING

from cozy_runtime.internal import decode_schemes as ds
from cozy_runtime.internal.decode_cores import HAS_TRITON, Cores

if TYPE_CHECKING:
    from torch import Tensor
    from torch import device as Device

    from cozy_runtime.accel.kernels import JitKernel, Launch, TritonEntry, TritonKernels
    from cozy_runtime.internal import kernel_ready, verify

NAME = "cozy-decode-cores"
#: The traced tensor: rows and columns every listed group, block and scale tile divides.
SHAPE = (128, 256)
DESTINATIONS = ("float32", "bfloat16", "float16")
_POINTERS = {"float32": "*fp32", "int32": "*i32", "int16": "*i16", "int8": "*i8", "uint8": "*u8"}
#: (scheme, entry) pairs the set does not hold, said once each.
_ABSENT: set[str] = set()


class _Traced:
    """A launch that launches nothing: each call's entry, typed from its operands (every integer
    i64, so no entry is specialized on a size)."""

    def __init__(self) -> None:
        self.entries: dict[str, TritonEntry] = {}

    def __call__(self, kernel: JitKernel, grid: tuple[int, ...], /, *args: object) -> None:
        from cozy_runtime.accel.kernels import TritonEntry

        named = dict(zip(kernel.arg_names, args, strict=True))
        constexprs = {n: v for n, v in named.items() if n.isupper() and isinstance(v, int)}
        signature = {n: "constexpr" if n.isupper() else _typed(v) for n, v in named.items()}
        if len(constexprs) != sum(n.isupper() for n in named):
            raise TypeError(f"{kernel.__name__}: a constexpr that is no integer")
        entry = TritonEntry(kernel, signature, constexprs, fp_fusion=False)
        held = self.entries.setdefault(entry.key, entry)
        if dict(held.signature) != signature:
            raise TypeError(f"{entry.key}: operands typed {dict(held.signature)} and {signature}")


def _typed(value: object) -> str:
    if isinstance(value, int):
        return "i64"
    dtype = str(getattr(value, "dtype", "")).removeprefix("torch.")
    if dtype not in _POINTERS:
        raise TypeError(f"a core operand of {type(value).__name__} {dtype}")
    return _POINTERS[dtype]


def calls(torch: ModuleType) -> Iterator[tuple[ds.Adapter, dict[str, object], dict[str, Tensor]]]:
    """Each row at each listed value of its params (a predicate's at its probe params'), with
    each combination of optional parts its rules allow: its sample parts on the host."""
    generator = torch.Generator().manual_seed(213)
    for a in ds.ADAPTERS:
        listed = [(a.probe_params[k],) if callable(rule) else rule for k, rule in a.params.items()]
        optional = sorted(a.optional)
        for values in itertools.product(*listed):
            params = dict(zip(a.params, values, strict=True))
            for count in range(len(optional) + 1):
                for present in itertools.combinations(optional, count):
                    parts = ds.sample(torch, a, SHAPE, generator, params, present)
                    described = {r: (_named(p), tuple(p.shape)) for r, p in parts.items()}
                    if not ds.parts_rule(a, SHAPE, params, described):
                        yield a, params, parts


def _named(part: Tensor) -> str:
    dtype = str(part.dtype).removeprefix("torch.")
    return next(name for name, torch_name in ds.TORCH.items() if torch_name == dtype)


@cache
def kernels() -> TritonKernels:
    """The set: every traced call's entry. A sample the row refuses (a check on its values) adds
    none; such a call decodes by the reference (`decode`)."""
    from cozy_runtime.accel.kernels import TritonKernels

    if not HAS_TRITON:
        return TritonKernels(NAME, [])
    import torch

    traced = _Traced()
    for a, params, parts in calls(torch):
        for name in DESTINATIONS:
            out = torch.empty(SHAPE, dtype=getattr(torch, name))
            try:
                ds.run(a, Cores(torch, traced), parts, params, out)
            except ValueError:
                continue
    return TritonKernels(NAME, sorted(traced.entries.values(), key=lambda entry: entry.key))


def _opener(torch: ModuleType, device: Device) -> kernel_ready.Opener[Launch]:
    from cozy_runtime.internal import kernel_ready

    def opener(wait: bool) -> Launch | None:  # traces on the opening thread, not the asker's
        return kernel_ready.TRITON[0](kernels(), partial(probe, torch, device), wait, device=device)

    return opener


def prime(torch: ModuleType, index: int) -> None:
    """Opens the set on cuda:`index` at device init, beside the fill (`kernel_ready.prime`)."""
    from cozy_runtime.internal import kernel_ready

    if HAS_TRITON and kernel_ready.TRITON:
        kernel_ready.prime(f"{NAME} cuda:{index}", _opener(torch, torch.device("cuda", index)))


def launcher(torch: ModuleType, device: Device) -> Launch | None:
    """`device`'s launcher once the set serves there (`kernel_ready.serve`), else None."""
    from cozy_runtime.internal import kernel_ready

    if not HAS_TRITON or not kernel_ready.TRITON:
        return None
    capturing = bool(torch.cuda.is_current_stream_capturing())
    launch: Launch | None = kernel_ready.serve(
        f"{NAME} cuda:{device.index or 0}", _opener(torch, device), capturing
    )
    return launch


def decode(
    torch: ModuleType, a: ds.Adapter, parts: Mapping[str, Tensor], params: Mapping[str, object],
    out: Tensor,
) -> None:  # fmt: skip
    """`out` by the row's core calls: launched where the set serves on `out`'s GPU; else, and for
    a call it does not hold, by the reference on the host, copied in (on the GPU while a CUDA
    graph records, which holds no host copy)."""
    from cozy_runtime.accel.kernels import KernelsUnavailable

    if not out.is_cuda:
        ds.run(a, Cores(torch, None), parts, params, out)
        return
    launch = launcher(torch, out.device)
    if launch is not None:
        try:
            with torch.cuda.device(out.device):
                ds.run(a, Cores(torch, launch), parts, params, out)
            return
        except KernelsUnavailable as exc:
            said = f"{a.scheme}: {exc}"
            if said not in _ABSENT:
                _ABSENT.add(said)
                note = f"[kernels] {NAME} holds no call of {said[:300]}; the reference decodes it"
                print(note, file=sys.stderr)
    if torch.cuda.is_current_stream_capturing():
        ds.run(a, Cores(torch, None), parts, params, out)
        return
    host = torch.empty(out.shape, dtype=out.dtype)
    ds.run(a, Cores(torch, None), {role: part.cpu() for role, part in parts.items()}, params, host)
    out.copy_(host)


def _bits(torch: ModuleType, t: Tensor) -> Tensor:
    view: Tensor = t.reshape(-1).view(torch.int32 if t.element_size() == 4 else torch.int16)
    return view


def probe(torch: ModuleType, device: Device, launch: Launch) -> verify.Verdict:
    """Every entry on `device`, by the first traced call that launches it, against the reference
    on the host, bit for bit."""
    from cozy_runtime.internal import verify

    probed: set[str] = set()
    total, wrong = 0, []
    for a, params, parts in calls(torch):
        for name in DESTINATIONS:
            want = torch.empty(SHAPE, dtype=getattr(torch, name))
            traced = _Traced()
            try:
                ds.run(a, Cores(torch, traced), parts, params, want)
            except ValueError:
                continue
            if set(traced.entries) <= probed:
                continue
            probed |= set(traced.entries)
            ds.run(a, Cores(torch, None), parts, params, want)
            got = torch.empty(SHAPE, dtype=want.dtype, device=device)
            on = {role: part.to(device) for role, part in parts.items()}
            with torch.cuda.device(device):
                ds.run(a, Cores(torch, launch), on, params, got)
            total += 1
            if not torch.equal(_bits(torch, got.cpu()), _bits(torch, want)):
                wrong.append(f"{a.scheme} {params} into {name}")
    detail = f"{len(probed)} entries, {total - len(wrong)} of {total} decodes the reference's bits"
    off = f"; first off: {wrong[0]}" if wrong else ""
    return verify.Verdict(total > 0 and not wrong, detail + off)
