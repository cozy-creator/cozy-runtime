"""A linear cut by its output features: `rows(linear)(x, start, stop)` is
`linear(x)[..., start:stop]` computed from weight rows [start, stop) alone; with `out=` (a view,
e.g. its range of a packed `[..., q | k | v]` buffer a kernel reads whole) it lands there, an
encoded leaf's GEMM writing it in place. A package cuts a
projection with it where its consumer reads output features apart (attention's heads), and
`author.split` sizes the windows, so the whole projection is never live at once.

A row reads only its own weights and scale under a plain `nn.Linear` and under the per-row-scaled
encoded leaves (int8.convrot, fp8.rowwise): a window is the whole's arithmetic in a narrower GEMM
(whose kernel may round differently). An overlay's LoRA (`lora.Hook`) adds its rows of `up`, and
a user LoRA's layer (`lora_composition`) its base's rows and each adapter's rows of `lora_B`;
baked, the base's rows alone. Anything else (a tensor-global scale, an unknown hook, another
PEFT layer) does not cut: `rows` is None and the caller computes the whole."""

from __future__ import annotations

import copy
import sys
from typing import Any, Protocol

from torch import Tensor, nn
from torch.nn import functional as F

from cozy_runtime import lora
from cozy_runtime.author.kernels import INT8_CONVROT, ROWWISE_FP8, PreQuantized, readable


class Rows(Protocol):
    def __call__(
        self, x: Tensor | PreQuantized, start: int, stop: int, out: Tensor | None = None
    ) -> Tensor: ...


#: Encoded leaves whose weight scale is one per output row.
_ROWWISE = frozenset({INT8_CONVROT, ROWWISE_FP8})


def rows(linear: nn.Module) -> Rows | None:
    """`linear`'s output windows, or None where its rows do not compute alone."""
    hooks = linear._forward_hooks.values()
    if linear._forward_pre_hooks or any(not isinstance(h, lora.Hook) for h in hooks):
        return None
    if _composed(linear):
        # A user LoRA's layer: its base's rows, and each adapter's rows of `lora_B` until the
        # weight plane bakes them in. Refusing it ran every adapted H3 call whole (run 24886).
        base = rows(linear.base_layer)
        if base is None:
            return None
        return lambda x, a, b, out=None: _adapted(linear, x, base(x, a, b, out), a, b)
    if type(linear) is nn.Linear:
        return lambda x, a, b, out=None: _hooked(linear, x, _plain(linear, x, a, b, out), a, b)
    if getattr(linear, "provider", None) in _ROWWISE and set(linear.roles()) == {"weight", "scale"}:
        return lambda x, a, b, out=None: _hooked(linear, x, _leaf(linear, x, a, b, out), a, b)
    return None


def _plain(
    linear: nn.Linear, x: Tensor | PreQuantized, start: int, stop: int, out: Tensor | None
) -> Tensor:
    if isinstance(x, PreQuantized):
        raise ValueError("an nn.Linear takes its float input, not a quantized operand")
    with readable(linear.weight, linear.bias) as (weight, bias):
        found = F.linear(x, weight[start:stop], None if bias is None else bias[start:stop])
    if out is None:
        return found
    out.copy_(found)
    return out


def _leaf(
    leaf: nn.Module, x: Tensor | PreQuantized, start: int, stop: int, out: Tensor | None
) -> Tensor:
    """The leaf's own forward over a copy holding its rows: one code path for its GEMM, its
    operand sharing (`quantize_once`) and its autocast rule."""
    with readable(leaf.data, leaf.scale, leaf.bias) as (data, scale, bias):
        window = copy.copy(leaf)
        vars(window)["_buffers"] = {
            **leaf._buffers,
            "data": data[start:stop],
            "scale": scale[start:stop],
            "bias": None if bias is None else bias[start:stop],
        }
        vars(window)["_forward_hooks"] = {}
        vars(window)["out_features"] = stop - start
        # `linear`, never `forward`: a residency wrapper on the leaf's forward is the whole's.
        if out is not None and window.WRITES_INTO:
            return window.linear(x, None, out)
        found = window.linear(x, None)
    if out is None:
        return found
    out.copy_(found)
    return out


def _composed(linear: nn.Module) -> bool:
    """A layer `lora_composition` built: PEFT's linear with the Runtime's forward (`_cozy_baked`),
    whose update is `lora.accumulate` per active adapter. Another PEFT layer's forward (dropout,
    DoRA) is not known to cut."""
    peft = getattr(sys.modules.get("peft.tuners.lora.layer"), "Linear", None)
    return peft is not None and isinstance(linear, peft) and hasattr(type(linear), "_cozy_baked")


def _adapted(layer: Any, x: Tensor | PreQuantized, out: Tensor, start: int, stop: int) -> Tensor:
    """A composed layer's update over rows [start, stop): each active adapter's rows of `lora_B`
    (`lora.accumulate`, as its forward adds the whole's); none once baked."""
    if layer._cozy_baked:
        return out
    if not isinstance(x, Tensor):
        raise ValueError("an unbaked LoRA update needs the float input, not its operand")
    for name in layer.active_adapters:
        up = layer.lora_B[name].weight
        lora.accumulate(x, out, layer.lora_A[name].weight, up[start:stop], layer.scaling[name])
    return out


def _hooked(
    linear: nn.Module, x: Tensor | PreQuantized, out: Tensor, start: int, stop: int
) -> Tensor:
    """An armed overlay's update, its rows of `up` (`lora.Hook`); none once baked."""
    for hook in linear._forward_hooks.values():
        armed = hook.lora_factors()
        if armed is None or hook._cozy_baked:
            continue
        if not isinstance(x, Tensor):
            raise ValueError("an unbaked LoRA update needs the float input, not its operand")
        _, down, up, scale = armed
        lora.accumulate(x, out, down, up[start:stop], scale)
    return out
