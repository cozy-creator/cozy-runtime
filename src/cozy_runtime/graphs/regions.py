"""Shipped purity contracts for graph regions, independent of memory placement.

A region reads its arguments and its complete module subtree, writes only fresh outputs,
uses the current stream, and has no observable host effect. Successful tracing does not
establish that contract. Unknown code remains eager around eligible descendants.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any


def safe(torch: Any, module: Any) -> bool:
    """Known library implementations, with no package hooks or overwritten forwards."""
    if (
        getattr(type(module), "_cozy_original_call", type(module).__call__)
        is not torch.nn.Module.__call__
    ):
        return False
    for sub in module.modules():
        for hooks in (sub._forward_pre_hooks, sub._forward_hooks):
            if any(
                not getattr(h, "__module__", "").startswith("tensorfs.varena.")
                for h in hooks.values()
            ):
                return False
        forward = vars(sub).get("forward")
        if (
            forward is not None
            and forward is not vars(sub).get("_cozy_execution_forward")
            and not getattr(forward, "__module__", "").startswith(
                "cozy_runtime.internal.execution."
            )
        ):
            return False
    simple = (
        torch.nn.Linear,
        torch.nn.Conv1d,
        torch.nn.Conv2d,
        torch.nn.Conv3d,
        torch.nn.LayerNorm,
        torch.nn.GroupNorm,
        torch.nn.RMSNorm,
        torch.nn.GELU,
        torch.nn.SiLU,
        torch.nn.ReLU,
    )
    if any(type(module).forward is cls.forward for cls in simple):
        return not getattr(module, "inplace", False)
    impl = type(module).forward
    if impl.__module__ == "cozy_runtime.internal.encoding.leaves" and impl.__qualname__.endswith(
        "EncodedLinear.forward"
    ):
        return True  # the installed provider's immutable role buffers and pure GEMM
    if type(module).forward is torch.nn.Sequential.forward:
        return all(safe(torch, child) for child in module.children())
    # These registered Diffusers implementations are token-wise pure in eval mode. A
    # subclass's custom forward is deliberately excluded even when its base has this name.
    return (
        not module.training
        and impl.__module__.startswith("diffusers.models.")
        and impl.__qualname__.split(".")[0]
        in {
            "FeedForward",
            "GELU",
            "GEGLU",
            "SwiGLU",
            "ApproximateGELU",
            "LinearActivation",
        }
        and all(_children_safe(torch, child) for child in module.children())
    )


def _children_safe(torch: Any, module: Any) -> bool:
    if type(module).forward is torch.nn.Dropout.forward:
        return not module.training or module.p == 0
    if type(module) in (torch.nn.ModuleList, torch.nn.ModuleDict):
        return all(_children_safe(torch, child) for child in module.children())
    return safe(torch, module)


def regions(torch: Any, root: Any, path: str = "") -> Iterator[tuple[str, Any]]:
    """Largest declared-safe subtrees; eager parents preserve Python phase/routing behavior."""
    if safe(torch, root):
        yield path, root
    else:
        for name, child in root.named_children():
            yield from regions(torch, child, f"{path}.{name}" if path else name)
