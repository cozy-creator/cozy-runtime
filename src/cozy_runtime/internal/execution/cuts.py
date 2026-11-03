"""Declared native operations and ordinary eager calls, with no Python memory policy.

Only complete native contracts may choose capacity-dependent geometry. An ordinary
operation executes once. Authored fixed windows retain their declared semantics;
there is no fitted byte model, room snapshot, retry or fallback cut ladder.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import torch
from torch import Tensor, nn

from cozy_runtime.internal.encoding.leaves import NVFP4
from cozy_runtime.internal.execution import catch, phase

type Attention = Callable[[Tensor, Tensor, Tensor, Tensor | None], Tensor]
_FFN = frozenset({"FeedForward", "GELU", "GEGLU", "SwiGLU", "ApproximateGELU", "LinearActivation"})


def install(module: nn.Module) -> int:
    """Install only supported native norm/FFN entrypoints; leave other operators eager."""
    count = 0
    for path, sub in module.named_modules():
        if type(sub) is nn.RMSNorm and "forward" not in vars(sub) and "_cozy_cut" not in vars(sub):
            sub.forward = catch.transparent(_norm(f"{path}:tokens", sub, _forward(sub)))
            vars(sub)["_cozy_cut"] = True
            count += 1
            continue
        if "forward" in vars(sub) or not type(sub).forward.__module__.startswith("diffusers."):
            continue
        kinds = [c.__name__ for c in type(sub).__mro__ if c.__module__.startswith("diffusers.")]
        if any(k in _FFN for k in kinds) and "_cozy_cut" not in vars(sub):
            sub.forward = catch.transparent(_ffn(sub, _forward(sub)))
            vars(sub)["_cozy_ffn_cut_forward"] = sub.forward
            vars(sub)["_cozy_cut"] = True
            count += 1
    return count


def _forward(module: nn.Module) -> Callable[..., object]:
    return vars(module).get("forward") or catch.current(module)


def _ffn(module: nn.Module, forward: Callable[..., object]) -> Callable[..., object]:
    from cozy_runtime.internal.execution.native_refiner_ffn import declare, split_ffn

    declared = declare(module)

    def execute(x: object, *args: object, **kwargs: object) -> object:
        if (
            not args
            and not kwargs
            and declared is not None
            and isinstance(x, Tensor)
            and x.is_cuda
            and x.dim() >= 2
            and not module.training
            and not torch.is_grad_enabled()
            and not any(getattr(child, "provider", None) == NVFP4 for child in module.modules())
        ):
            result = split_ffn(declared, x)
            if result is not None:
                return result
        return forward(x, *args, **kwargs)

    return execute


def _norm(site: str, module: nn.Module, forward: Callable[..., object]) -> Callable[..., object]:
    dims = len(cast("tuple[int, ...]", module.normalized_shape))

    def execute(x: object, *args: object, **kwargs: object) -> object:
        if (
            not args
            and not kwargs
            and isinstance(x, Tensor)
            and not torch.is_grad_enabled()
            and x.dim() > dims
        ):
            from cozy_runtime.internal.execution.native_rms_norm import rms_norm

            result = rms_norm(site, module, x)
            if result is not None:
                return result
        return forward(x, *args, **kwargs)

    return execute


def attend(
    call: Attention,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    mask: Tensor | None = None,
    *,
    rows: int = 0,
    heads: bool = True,
    scratch: str = "flash",
) -> Tensor:
    """Ordinary selected attention once; native Ulysses uses its explicit prepared provider."""
    catch.checkpoint()
    return call(q, k, v, mask)


def split[T](name: str, key: str, total: int, finest: int, run: Callable[[int], T]) -> T:
    """Run the author's actual fixed geometry once; no byte prediction or replay."""
    catch.checkpoint()
    phase.choice(("authored-window", name, key, total, finest))
    return run(total)
