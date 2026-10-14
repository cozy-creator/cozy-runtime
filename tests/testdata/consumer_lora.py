"""A model-owned LoRA read through consumer hooks on linears (`weights.LoRASource`): its factors
live in their own model slot beside the module they adapt and apply only while armed, as a
package's turbo overlay does. The weight plane may bake them into the linear's bytes."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


class Factors(nn.Module):  # type: ignore[misc]
    def __init__(self, in_features: int, out_features: int, rank: int, scale: float) -> None:
        super().__init__()
        self.lora_down = nn.Parameter(torch.empty(rank, in_features))
        self.lora_up = nn.Parameter(torch.empty(out_features, rank))
        self.scale = scale


class Overlay(nn.Module):  # type: ignore[misc]
    """One `Factors` per adapted linear path."""

    def __init__(self, sites: dict[str, tuple[int, int]], rank: int, scale: float) -> None:
        super().__init__()
        self.sites = nn.ModuleDict(
            {_key(path): Factors(i, o, rank, scale) for path, (i, o) in sites.items()}
        )

    def factors(self, path: str) -> Factors:
        return self.sites[_key(path)]


class Arming:
    """The overlay the adapted module's current forward reads, or None for a base forward."""

    def __init__(self) -> None:
        self.overlay: Overlay | None = None


class Hook:
    """`out += scale * up(down(x))` while armed, unless the plane baked the update."""

    __slots__ = ("_cozy_baked", "arming", "path")

    def __init__(self, arming: Arming, path: str) -> None:
        self.arming, self.path, self._cozy_baked = arming, path, False

    def __call__(self, module: nn.Module, args: tuple[Any, ...], out: torch.Tensor) -> None:
        overlay = self.arming.overlay
        if overlay is not None and not self._cozy_baked:
            factors = overlay.factors(self.path)
            low = F.linear(args[0], factors.lora_down.to(args[0].dtype))
            out.add_(F.linear(low, factors.lora_up.to(out.dtype)), alpha=factors.scale)

    def lora_factors(self) -> tuple[object, torch.Tensor, torch.Tensor, float] | None:
        overlay = self.arming.overlay
        if overlay is None:
            return None
        factors = overlay.factors(self.path)
        return factors, factors.lora_down, factors.lora_up, float(factors.scale)


#: A transformer block's attention and feed-forward linears, as Diffusers names them.
FAMILIES = ("to_q", "to_k", "to_v", "to_out.0", "ff.net.0.proj", "ff.net.2")


def block_linears(module: nn.Module) -> list[str]:
    """Every block's attention and feed-forward linear paths: what a turbo LoRA adapts."""
    return [
        path for path, _ in module.named_modules() if "blocks." in path and path.endswith(FAMILIES)
    ]


def install(module: nn.Module, paths: Iterable[str]) -> list[Hook]:
    """Hooks on `module`'s linears at `paths`, armed through `module.consumers`."""
    module.consumers = Arming()
    hooks = [Hook(module.consumers, path) for path in paths]
    for hook in hooks:
        module.get_submodule(hook.path).register_forward_hook(hook)
    return hooks


@contextmanager
def armed(module: nn.Module, overlay: nn.Module) -> Iterator[None]:
    module.consumers.overlay = overlay
    try:
        yield
    finally:
        module.consumers.overlay = None


def overlay_for(module: nn.Module, paths: Iterable[str], rank: int, scale: float) -> Overlay:
    sites = {}
    for path in paths:
        linear = module.get_submodule(path)
        sites[path] = (linear.in_features, linear.out_features)
    return Overlay(sites, rank, scale)


def _key(path: str) -> str:
    return path.replace(".", "__")
