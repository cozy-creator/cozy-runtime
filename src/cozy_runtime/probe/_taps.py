"""Tap resolution: WHERE the probe measures, named by module path.

Tap identity is the module path in the constructed lane (state_dict key prefix, e.g.
``unet.down_blocks.1.attentions.0``) — stable across lanes because both lanes construct the
SAME class from the SAME config; lanes change weights, never tensor schema.

Block-level by default, never every module. Containers (ModuleList/Sequential/ModuleDict)
and pure GROUPINGS — modules whose weighted children all live in containers, like a UNet
down block holding its ``attentions`` and ``resnets`` lists — are structure and get
subdivided; every other weighted module is a block, tapped WHOLE at its output. ``zoom``
re-resolves one block subdivided when the map needs detail.

Hooks need nn.Modules; §1.1 guarantees only ``state_dict()``, so a non-Module component
degrades to ONE component-output tap — typed, never silent.
"""

from __future__ import annotations

from typing import Any, Literal

import msgspec

#: More taps than this is "every module", which the probe never does — zoom instead.
MAX_TAPS = 512

TapKind = Literal["module", "component_output"]


class ProbeRefusal(Exception):
    """A typed probe refusal: what could not be measured and why, never a wrong number."""

    def __init__(self, message: str, *, code: str) -> None:
        super().__init__(message)
        self.code = code


class Tap(msgspec.Struct, frozen=True):
    """One measurement point. ``path`` is the report identity; ``kind`` says how it is
    captured — ``module`` by forward hook, ``component_output`` from the call's return."""

    path: str
    kind: TapKind


class TapPlan(msgspec.Struct, frozen=True):
    """The resolved tap set for one probe, identical across its lanes."""

    component: str
    taps: tuple[Tap, ...]

    def module_path(self, tap: Tap) -> str:
        """``tap``'s path relative to the lane root (the component prefix stripped)."""
        if tap.path == self.component:
            return ""
        return tap.path.removeprefix(self.component + ".")


def resolve_taps(root: object, *, component: str, zoom: str | None = None) -> TapPlan:
    """Resolve block-level taps over one constructed component.

    ``component`` prefixes every tap path (the pipeline name of ``root``, e.g. ``unet``).
    ``zoom`` names one existing tap to re-resolve subdivided: the plan becomes that tap
    plus its children, for the rerun that turns a coarse map into a fine one.
    """
    import torch

    if not isinstance(root, torch.nn.Module):
        if zoom is not None:
            raise ProbeRefusal(
                f"cannot zoom {zoom!r}: component {component!r} is a {type(root).__name__}, "
                "not an nn.Module — hooks need Modules, so it degrades to ONE "
                "component-output tap with nothing to subdivide",
                code="zoom_needs_module",
            )
        return TapPlan(component=component, taps=(Tap(path=component, kind="component_output"),))

    nn = torch.nn
    paths: list[str] = []
    if zoom is None:
        for name, child in root.named_children():
            _place(child, f"{component}.{name}", paths, nn)
        if not paths and _weighted(root):
            paths.append(component)
    else:
        target = _zoom_target(root, component, zoom)
        paths.append(zoom)
        subdivided: list[str] = []
        for name, child in target.named_children():
            _place(child, f"{zoom}.{name}", subdivided, nn)
        if not subdivided:
            raise ProbeRefusal(
                f"cannot zoom {zoom!r}: it has no weighted children to subdivide",
                code="zoom_indivisible",
            )
        paths.extend(subdivided)
    if not paths:
        raise ProbeRefusal(
            f"component {component!r} holds no weighted modules to tap", code="no_taps"
        )
    if len(paths) > MAX_TAPS:
        raise ProbeRefusal(
            f"component {component!r} resolved {len(paths)} taps (cap {MAX_TAPS}) — block-level "
            "taps are dozens, never every module; zoom one block instead",
            code="too_many_taps",
        )
    return TapPlan(component=component, taps=tuple(Tap(path=path, kind="module") for path in paths))


def _zoom_target(root: Any, component: str, zoom: str) -> Any:
    if zoom == component:
        return root
    if not zoom.startswith(component + "."):
        raise ProbeRefusal(
            f"zoom {zoom!r} is outside component {component!r}", code="zoom_outside_component"
        )
    try:
        return root.get_submodule(zoom.removeprefix(component + "."))
    except AttributeError as exc:
        raise ProbeRefusal(
            f"zoom {zoom!r} names no module in this lane: {exc}", code="zoom_unknown_path"
        ) from exc


def _place(module: Any, path: str, paths: list[str], nn: Any) -> None:
    if _container(module, nn) or _grouping(module, nn):
        for name, child in module.named_children():
            _place(child, f"{path}.{name}", paths, nn)
    elif _weighted(module):
        paths.append(path)


def _container(module: Any, nn: Any) -> bool:
    return isinstance(module, (nn.ModuleList, nn.Sequential, nn.ModuleDict))


def _grouping(module: Any, nn: Any) -> bool:
    """Structure, not a block: no weight of its own outside its container children."""
    children = list(module.children())
    if not any(_container(child, nn) for child in children):
        return False
    if next(module.parameters(recurse=False), None) is not None:
        return False
    if next(module.buffers(recurse=False), None) is not None:
        return False
    return not any(_weighted(child) for child in children if not _container(child, nn))


def _weighted(module: Any) -> bool:
    return (
        next(module.parameters(recurse=True), None) is not None
        or next(module.buffers(recurse=True), None) is not None
    )
