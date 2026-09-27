"""Group component calls: sharded forwards run on every rank, hosted ones on their home rank.

The sampler and its Python state stay on rank 0."""

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
        if cp.in_gated_call():
            return call(*args, **kwargs)
        command, payloads = _command(name, prepared, group, spool, args, kwargs, cancelled)
        try:
            try:
                group.broadcast(command)
                with cp.gated_call():
                    result = call(*args, **kwargs)
                for reply in group.collect("run"):
                    _accept(name, group, reply)
                    attention_sol.record_rank(reply["rank"], reply["sol_calls"])
                return result
            except BaseException:
                group._break(f"mirrored component {name!r} failed")
                raise
        finally:
            payloads.clear()

    component._call_impl = mirrored


def host_component(
    component: Any,
    *,
    name: tuple[str, str],
    prepared: Mapping[tuple[str, str], tuple[Any, Any]],
    group: RankGroup,
    spool: Callable[[], Path | None],
    rank: int,
    device: Any,
    cancelled: Callable[[], bool] = lambda: False,
) -> None:
    """Run every call into `component`, its root or any submodule, on follower `rank`.

    Rank 0 holds none of its bytes. The home rank admits the leader's active scope, runs the
    same module call on its resident copy and returns the result through the attempt spool.
    """
    for path, module in component.named_modules():
        module._call_impl = _hosted(path, name, prepared, group, spool, rank, device, cancelled)


def _hosted(
    path: str,
    name: tuple[str, str],
    prepared: Mapping[tuple[str, str], tuple[Any, Any]],
    group: RankGroup,
    spool: Callable[[], Path | None],
    rank: int,
    device: Any,
    cancelled: Callable[[], bool],
) -> Callable[..., Any]:
    def hosted(*args: Any, **kwargs: Any) -> Any:
        command, payloads = _command(name, prepared, group, spool, args, kwargs, cancelled)
        results = wire.TensorSpool(Path(command["spool"]) / "hosted")
        try:
            try:
                reply = group.call_one(rank, {**command, "module": path})
                _accept(name, group, reply)
            except BaseException:
                group._break(f"hosted component {name!r} failed")
                raise
            model = prepared[name][0]
            residency, active = model._cozy_residency, model._cozy_active
            if residency is not None and active is not None:
                residency.observe_remote(active[0], int(reply.get("working_peak_bytes", 0)))
            return wire.unmarshal(reply["result"], results, device=device)
        finally:
            payloads.clear()
            results.clear()

    return hosted


def hosting_plan(
    *,
    placeable: tuple[str, ...],
    sizes: Mapping[str, int],
    sharded: set[str],
    scopes: Mapping[str, tuple[str, ...]],
    budget: int,
    world: int,
) -> dict[str, int]:
    """Placeable component -> the follower that keeps it resident, largest first.

    Only when rank 0 cannot hold every component: a follower hosts one if it fits beside
    that rank's largest sharded scope. Every rank derives the same answer from the same
    agreed sizes and ceiling, so no command carries it. H3 on an H100: the 51.5 GB text
    encoder beside a 21 GB DiT shard, where rank 0 re-staged it for every request.
    """
    if world < 2 or sum(sizes.values()) <= budget:
        return {}
    shard = max(
        (sum(sizes.get(n, 0) for n in names if n in sharded) for names in scopes.values()),
        default=0,
    )
    room = {follower: budget - shard for follower in range(1, world)}
    hosted: dict[str, int] = {}
    for name in sorted(placeable, key=lambda n: (-sizes.get(n, 0), n)):
        follower = max(room, key=lambda r: (room[r], -r))
        if name in sizes and name not in sharded and sizes[name] <= room[follower]:
            hosted[name] = follower
            room[follower] -= sizes[name]
    return hosted


def _command(
    name: tuple[str, str],
    prepared: Mapping[tuple[str, str], tuple[Any, Any]],
    group: RankGroup,
    spool: Callable[[], Path | None],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    cancelled: Callable[[], bool],
) -> tuple[dict[str, Any], wire.TensorSpool]:
    """The `run` command for one component call under the leader's active scopes.

    Scope admission has already happened on the leader. Followers receive that same
    complete scope, including any overlay weights used by component hooks. A cancelled
    attempt raises here, before any rank is commanded.
    """
    import torch

    models = {key[0]: model for key, (model, _root) in prepared.items()}
    active = {
        key: model._cozy_active for key, model in models.items() if model._cozy_active is not None
    }
    directory = spool()
    if name[0] not in active or directory is None:
        raise cp.UngatedShardedForward(
            f"{name}: a group component call needs an active component scope and attempt spool"
        )
    group.check_alive()
    if cancelled():
        raise Cancelled(f"the attempt was cancelled before a group {name[1]!r} call")
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
    command = {
        **wire.run_command(args, kwargs, payloads),
        "component": list(name),
        "spool": str(directory),
        "scopes": scopes,
        "grad_enabled": torch.is_grad_enabled(),
        "inference_mode": torch.is_inference_mode_enabled(),
    }
    return command, payloads


def _accept(name: tuple[str, str], group: RankGroup, reply: Mapping[str, Any]) -> None:
    if not reply.get("ok"):
        raise GroupRefusal(
            f"{name}: rank {reply.get('rank')} failed: {reply.get('code')}: "
            f"{reply.get('detail', '')}"[:600],
            code=str(reply.get("code") or "group_broken"),
        )
    rank = int(reply["rank"])
    group.working_peaks[rank] = max(
        group.working_peaks.get(rank, 0), int(reply.get("working_peak_bytes", 0))
    )
    if reply.get("evidence"):
        execution_evidence.widen(group.rank_records, reply["evidence"])
