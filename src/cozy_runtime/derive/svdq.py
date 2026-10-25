"""Model-owned SVDQ calibration hooks and immutable completed statistics.

TensorFS Writer owns smoothing, factorization, alpha selection and encoding. Runtime never
falls back to the historical Torch fitter; the independent release gate retains that oracle.
"""

from __future__ import annotations

import random
from collections.abc import Iterable, Mapping
from types import ModuleType
from typing import Any, NamedTuple

from cozy_runtime import graphs
from cozy_runtime.author._context import device as placed_device
from cozy_runtime.internal import svdq_linear

from .microscale import NVFP4_BLOCK

#: The branch's rank (Nunchaku's and the xs-064 study's).
RANK = 32
#: Calibration rows kept per linear, sampled uniformly over every call.
ROWS = 512
#: Rows sampled from each call before the reservoir draw.
PER_CALL = 32
#: The column multiple the kernels need (`svdq_linear`'s K step).
K_ALIGN = svdq_linear.K_ALIGN


class Stats(NamedTuple):
    """One linear's calibration: per-channel `|x|` max `[K]` and sampled input rows `[n, K]`."""

    xmax: Any
    rows: Any


class Calibration:
    """While open, the forward pre-hooks on every `nn.Linear` of `components` whose
    `component.<module>.weight` is in `keys` record its inputs (`Stats`) without a device sync:
    the per-channel `|x|` max on the device, and `PER_CALL` sampled rows of each call into a
    uniform reservoir of `ROWS` in pinned host memory. Runtime's CUDA graphs are bypassed while
    it is open, so every call runs the hooks."""

    def __init__(
        self, torch: ModuleType, components: Mapping[str, Any], keys: Iterable[str]
    ) -> None:
        self.torch = torch
        wanted = set(keys)
        self.modules: dict[str, Any] = {}
        for component, root in components.items():
            for name, module in root.named_modules():
                key = f"{component}.{name}.weight"
                if key in wanted and isinstance(module, torch.nn.Linear):
                    self.modules[key] = module
        missing = sorted(wanted - set(self.modules))
        if missing:
            raise ValueError(f"calibration finds no linear for {missing[:3]} ({len(missing)})")
        self._xmax: dict[str, Any] = {}
        self._rows: dict[str, Any] = {}
        self._seen: dict[str, int] = {}
        self._handles: list[Any] = []
        self._bypass: Any = None
        self._slots = random.Random(0)
        self._picks: dict[Any, Any] = {}
        self.completed = False

    def __enter__(self) -> Calibration:
        self.completed = False
        self._bypass = graphs.BYPASS.set(True)
        for key, module in self.modules.items():
            self._handles.append(module.register_forward_pre_hook(self._hook(key)))
        return self

    def __exit__(self, kind: object, *exc: object) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        graphs.BYPASS.reset(self._bypass)
        if placed_device().type == "cuda":
            self.torch.cuda.synchronize()  # the reservoir's row copies have landed
        missing = sorted(set(self.modules) - set(self._seen))
        if kind is None and missing:
            raise ValueError(
                f"calibration saw no input for {missing[:3]} ({len(missing)}): the prompts never "
                "reached those linears"
            )
        self.completed = kind is None

    def _hook(self, key: str) -> Any:
        torch = self.torch

        def record(module: Any, args: tuple[Any, ...]) -> None:
            with torch.no_grad():
                flat = args[0].detach().reshape(-1, args[0].shape[-1])
                amax = flat.abs().amax(0).float()
                held = self._xmax.get(key)
                self._xmax[key] = amax if held is None else torch.maximum(held, amax)
                picks = self._picks.get(flat.device)
                if picks is None:
                    picks = self._picks[flat.device] = torch.Generator(flat.device).manual_seed(0)
                take = min(PER_CALL, flat.shape[0])
                pick = torch.randint(flat.shape[0], (take,), generator=picks, device=flat.device)
                rows = flat[pick].to(torch.bfloat16)
                if key not in self._rows:
                    self._rows[key] = torch.empty(
                        ROWS, flat.shape[1], dtype=torch.bfloat16, pin_memory=flat.is_cuda
                    )
                reservoir, seen = self._rows[key], self._seen.get(key, 0)
                for row in rows:
                    seen += 1
                    slot = seen - 1 if seen <= ROWS else self._slots.randrange(seen)
                    if slot < ROWS:
                        reservoir[slot].copy_(row, non_blocking=True)
                self._seen[key] = seen

        return record

    def stats(self, component: str, key: str) -> Stats:
        name = f"{component}.{key}"
        if name not in self._seen:
            raise ValueError(f"calibration saw no input for {name}: run the model first")
        return Stats(self._xmax[name].cpu(), self._rows[name][: min(self._seen[name], ROWS)])


def companions(rows: int, cols: int) -> tuple[tuple[str, str, tuple[int, ...]], ...]:
    """The parts after `weight`, `(role, dtype, shape)`, in their declared storage order."""
    return (
        ("scale", "F8_E4M3", (rows, cols // NVFP4_BLOCK)),
        ("scale_2", "F32", ()),
        ("pre_quant_scale", "F32", (cols,)),
        ("lowrank_down", "BF16", (RANK, cols)),
        ("lowrank_up", "BF16", (rows, RANK)),
        ("input_scale", "F32", ()),
    )
