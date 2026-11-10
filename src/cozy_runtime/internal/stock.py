"""Ordinary Diffusers and Transformers calls on the Runtime's models.

A component root answers its own `.to()`, `.cuda()`, `.cpu()`, `.half()`, `.float()`, `.bfloat16()`
and `.double()`, ahead of its class's: placement is the Runtime's, so a device is ignored, and a dtype
is served through `hold_dtype` (the stored bytes are cast at the next fill, counted). Diffusers'
offload helpers do nothing: varena already pages every component.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING

from cozy_runtime.author._model import PLACED, WARNINGS, WarningRow
from cozy_runtime.internal.accel import StaleBinding

if TYPE_CHECKING:
    import torch
    from torch import nn

#: The module methods a component root answers itself.
MOVES = ("to", "cuda", "cpu", "half", "float", "bfloat16", "double")

type Hold = Callable[[torch.dtype], None]


def dtype_of(args: tuple[object, ...], kwargs: Mapping[str, object]) -> torch.dtype | None:
    """The dtype a `.to(...)` asks for: `to(dtype)`, `to(device, dtype)`, `to(dtype=)`, `to(tensor)`."""
    import torch

    for value in (*args[:2], kwargs.get("dtype"), kwargs.get("tensor")):
        if isinstance(value, torch.dtype):
            return value
        if isinstance(value, torch.Tensor):
            return value.dtype
    return None


def answer_moves(root: nn.Module, hold: Hold) -> None:
    """`root`'s own device and dtype methods: a device is ignored, a dtype goes to `hold`."""
    import torch

    def to(*args: object, **kwargs: object) -> nn.Module:
        dtype = dtype_of(args, kwargs)
        if dtype is not None:
            hold(dtype)
        return root

    def stay(*_: object, **__: object) -> nn.Module:
        return root

    def cast(dtype: torch.dtype) -> Callable[[], nn.Module]:
        def held() -> nn.Module:
            hold(dtype)
            return root

        return held

    vars(root).update(
        to=to,
        cuda=stay,
        cpu=stay,
        half=cast(torch.float16),
        float=cast(torch.float32),
        bfloat16=cast(torch.bfloat16),
        double=cast(torch.float64),
    )


def forget_moves(root: nn.Module) -> None:
    for name in MOVES:
        vars(root).pop(name, None)


def _warn(row: WarningRow) -> None:
    if row not in WARNINGS:
        WARNINGS.append(row)


def _held(root: nn.Module, dtype: torch.dtype) -> None:
    """A placed root's dtype change: its placeholders change now, its bytes at the next fill. A
    dtype its lane cannot serve (encoded weights; not f16, bf16 or f32) is served as stored."""
    placed = PLACED.get(root)
    if placed is None:
        return
    residency, component = placed
    try:
        change = residency.represent(component, dtype)
    except StaleBinding as exc:
        _warn((
            f"{component}.to({str(dtype).removeprefix('torch.')}) is served as stored: {exc}",
            {"code": "dtype_held", "component": component},
        ))  # fmt: skip
        return
    change()


def place(roots: Mapping[str, nn.Module]) -> None:
    """Placed component roots answer `.to()` and its kin from here on."""
    for root in roots.values():
        answer_moves(root, lambda dtype, root=root: _held(root, dtype))


def _nothing(*_: object, **__: object) -> None:
    return None


#: Diffusers' memory helpers: varena pages every component, so each is a no-op.
_OFFLOAD = {
    "diffusers.pipelines.pipeline_utils:DiffusionPipeline": (
        "enable_model_cpu_offload",
        "enable_sequential_cpu_offload",
        "enable_group_offload",
    ),
    "diffusers.models.modeling_utils:ModelMixin": (
        "enable_group_offload",
        "enable_layerwise_casting",
    ),
}


def _installed(library: str) -> bool:
    return library in sys.modules or importlib.util.find_spec(library) is not None


def prepare() -> None:
    """Before a construction: Diffusers' offload helpers do nothing. Diffusers is imported here when
    the package's environment has it, so a helper called inside `load()` is already one."""
    if not _installed("diffusers"):
        return
    for where, names in _OFFLOAD.items():
        module, _, name = where.partition(":")
        owner: object = getattr(importlib.import_module(module), name)
        for method in names:
            if hasattr(owner, method):
                setattr(owner, method, _nothing)
