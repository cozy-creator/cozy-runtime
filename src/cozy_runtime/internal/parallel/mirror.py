"""Mirror only sharded component forwards; the sampler and its Python state stay on rank 0."""

from __future__ import annotations

import functools
from collections.abc import Callable
from pathlib import Path
from typing import Any

from cozy_runtime.internal.parallel import cp, wire
from cozy_runtime.internal.parallel.group import RankGroup
from cozy_runtime.internal.parallel.plan import GroupRefusal


def mirror_component(
    component: Any,
    *,
    model: Any,
    name: str,
    group: RankGroup,
    spool: Callable[[], Path | None],
) -> None:
    """Wrap the module call outside all torch and diffusers hooks.

    Leave forward and its signature intact: diffusers filters layout kwargs through it.
    Scope admission has already happened on the leader. Followers receive that same
    complete scope, including any overlay weights used by component hooks.
    """
    if component._compiled_call_impl is not None:
        raise cp.ContextParallelUnavailable(
            "install context parallelism before compiling the component"
        )
    call = component._call_impl

    @functools.wraps(call)
    def mirrored(*args: Any, **kwargs: Any) -> Any:
        import torch

        if cp.in_gated_call():
            return call(*args, **kwargs)
        active = model._cozy_active
        directory = spool()
        if active is None or directory is None:
            raise cp.UngatedShardedForward(
                f"{name}: a sharded forward needs an active component scope and attempt spool"
            )
        group.check_alive()
        payloads = wire.TensorSpool(directory / "ranks")
        try:
            command = {
                **wire.run_command(args, kwargs, payloads),
                "component": name,
                "spool": str(directory),
                "scope_method": active[0],
                "scope_components": list(active[1]),
                "grad_enabled": torch.is_grad_enabled(),
                "inference_mode": torch.is_inference_mode_enabled(),
            }
            residency = model._cozy_residency
            if residency is not None:
                command.update(
                    placement=residency.placement,
                    headroom_bytes=residency.headroom,
                    scope_headroom_bytes=residency.scope_headrooms,
                )
            try:
                group.broadcast(command)
                with cp.gated_call():
                    result = call(*args, **kwargs)
                for reply in group.collect("run"):
                    if not reply.get("ok"):
                        raise GroupRefusal(
                            f"{name}: rank {reply.get('rank')} failed: {reply.get('code')}: "
                            f"{reply.get('detail', '')}"[:600],
                            code=str(reply.get("code") or "group_broken"),
                        )
                return result
            except BaseException:
                group._break(f"mirrored component {name!r} failed")
                raise
        finally:
            payloads.clear()

    component._call_impl = mirrored
