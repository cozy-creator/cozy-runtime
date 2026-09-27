"""Mirror only sharded component forwards; the sampler and its Python state stay on rank 0."""

from __future__ import annotations

import functools
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from cozy_runtime.author._errors import Cancelled
from cozy_runtime.internal import attention_sol, execution_evidence
from cozy_runtime.internal.parallel import cp, wire
from cozy_runtime.internal.parallel.group import RankGroup
from cozy_runtime.internal.parallel.plan import GroupRefusal


def mirror_component(
    component: Any,
    *,
    name: tuple[str, str],
    prepared: Mapping[tuple[str, str], tuple[Any, Any]],
    group: RankGroup,
    spool: Callable[[], Path | None],
    cancelled: Callable[[], bool] = lambda: False,
) -> None:
    """Wrap the module call outside all torch and diffusers hooks.

    Each call is a cancellation safe point: a cancelled attempt raises here, before any rank
    is commanded, so every rank stops at the same boundary and none is left mid-collective.

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
        models = {key[0]: model for key, (model, _root) in prepared.items()}
        active = {
            key: model._cozy_active
            for key, model in models.items()
            if model._cozy_active is not None
        }
        directory = spool()
        if name[0] not in active or directory is None:
            raise cp.UngatedShardedForward(
                f"{name}: a sharded forward needs an active component scope and attempt spool"
            )
        group.check_alive()
        if cancelled():
            raise Cancelled(f"the attempt was cancelled before a mirrored {name[1]!r} call")
        references = {
            key: root
            for key, (_model, root) in prepared.items()
            if key[0] in active and key[1] in active[key[0]][1]
        }
        scopes = []
        for key, (method, components) in active.items():
            scope = {"model": key, "method": method, "components": list(components)}
            residency = models[key]._cozy_residency
            if residency is not None:
                scope.update(
                    placement=residency.placement,
                    headroom_bytes=residency.headroom,
                    scope_headroom_bytes=residency.scope_headrooms,
                    measured_scopes=sorted(residency.measured),
                )
            scopes.append(scope)
        payloads = wire.TensorSpool(directory / "ranks", references)
        try:
            command = {
                **wire.run_command(args, kwargs, payloads),
                "component": list(name),
                "spool": str(directory),
                "scopes": scopes,
                "grad_enabled": torch.is_grad_enabled(),
                "inference_mode": torch.is_inference_mode_enabled(),
            }
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
                    attention_sol.record_rank(reply["rank"], reply["sol_calls"])
                    rank = int(reply["rank"])
                    group.working_peaks[rank] = max(
                        group.working_peaks.get(rank, 0), int(reply.get("working_peak_bytes", 0))
                    )
                    if reply.get("evidence"):
                        execution_evidence.widen(group.rank_records, reply["evidence"])
                return result
            except BaseException:
                group._break(f"mirrored component {name!r} failed")
                raise
        finally:
            payloads.clear()

    component._call_impl = mirrored
