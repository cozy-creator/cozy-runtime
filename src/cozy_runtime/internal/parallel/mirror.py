"""Group component calls: sharded forwards run on every rank, hosted ones on their home rank,
spread ones on whichever ranks are idle and have room.

The sampler and its Python state stay on rank 0."""

from __future__ import annotations

import contextlib
import functools
import weakref
from collections import deque
from collections.abc import Callable, Iterator, Mapping
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any

import msgspec

from cozy_runtime.author._attention_scope import _ACTIVE_LAYOUT
from cozy_runtime.author._errors import Cancelled, classify, is_device_oom
from cozy_runtime.author._model import remote_scope
from cozy_runtime.internal import attention_sol, execution_evidence, memory
from cozy_runtime.internal.parallel import cp, wire
from cozy_runtime.internal.parallel.group import (
    OpReady,
    PreparationRefusal,
    PreparedAbort,
    PreparedGroup,
    RankGroup,
    SpreadLog,
)
from cozy_runtime.internal.parallel.plan import GroupRefusal


def mirror_component(
    component: Any,
    *,
    name: tuple[str, str],
    prepared: Mapping[tuple[str, str], tuple[Any, Any]],
    group: RankGroup,
    spool: Callable[[], Path | None],
    cancelled: Callable[[], bool] = lambda: False,
    planner: memory.Planner | None = None,
    room: Callable[[list[OpReady], dict[int, tuple[int, int]]], bool] | None = None,
    host_room: Callable[[list[OpReady]], AbstractContextManager[tuple[int, ...]]] | None = None,
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
        key = f"{name[0]}/{name[1]}"
        owned = planner.held.get(key) if planner is not None else None
        try:
            # Every follower joins the collective: none may be answering an overlapped call.
            with group.hold(range(1, group.degree)), contextlib.ExitStack() as host_reservations:
                required = {"prepare_commit"}
                if host_room is not None:
                    required.add("group_host_credit")
                if any(
                    not required.issubset(group.memory_capabilities.get(rank, ()))
                    for rank in range(1, group.degree)
                ):
                    if planner is not None:
                        planner.preparation_failure = "group_preparation_failed"
                    raise PreparationRefusal(
                        "a follower does not support pre-collective preparation"
                    )
                progress: dict[int, tuple[int, int]] = {}
                host_grants: tuple[int, ...] = ()
                host_tried = False
                while True:
                    _check_cancelled(cancelled, name)
                    transaction = PreparedGroup(
                        group,
                        msgspec.to_builtins(
                            msgspec.structs.replace(
                                command, memory_grants=group.memory_grants, host_grants=host_grants
                            )
                        ),
                    )

                    def ready(current: PreparedGroup = transaction) -> None:
                        if not current.ready(
                            preparation_status(planner, current.operation, 0),
                            lambda: _check_cancelled(cancelled, name),
                        ):
                            raise PreparedAbort()

                    try:
                        credit = (
                            planner.host_credit(host_grants[0] if host_grants else None)
                            if planner is not None
                            else contextlib.nullcontext()
                        )
                        with cp.gated_call(), credit:
                            if planner is not None and owned is not None:
                                with planner.gating(key, ready):
                                    result = planner.run(owned, "forward", call, args, kwargs)
                            else:
                                ready()
                                result = call(*args, **kwargs)
                    except BaseException as exc:
                        if transaction.committed or group.broken:
                            group._break(f"mirrored component {name!r} failed after commit")
                            raise
                        if not transaction.statuses:
                            transaction.ready(
                                preparation_status(planner, transaction.operation, 0, exc)
                            )
                        exc.__traceback__ = None
                        transaction.abort()
                        failed = [status for status in transaction.statuses if not status.ok]
                        if host_grants:
                            remaining = tuple(
                                status.host_remaining
                                for status in sorted(transaction.statuses, key=lambda s: s.rank)
                            )
                            if len(remaining) != len(host_grants) or any(
                                left < 0 or left > original
                                for left, original in zip(remaining, host_grants, strict=True)
                            ):
                                group._break("invalid remaining group host credits")
                                raise GroupRefusal(group.broken) from None
                            host_grants = remaining
                        reusable, try_host, try_device = memory.Planner.group_retry(
                            transaction.statuses,
                            host_tried=host_tried,
                            host_available=host_room is not None,
                            device_available=room is not None,
                        )
                        if not reusable:
                            group._break(f"{name!r}: preparation left a rank unusable")
                            raise GroupRefusal(group.broken) from None
                        if try_host and host_room is not None:
                            host_tried = True
                            host_grants = host_reservations.enter_context(
                                host_room(transaction.statuses)
                            )
                            if host_grants:
                                continue
                        if try_device and room is not None and room(transaction.statuses, progress):
                            continue
                        if isinstance(exc, Cancelled):
                            raise
                        refused = (
                            failed[0]
                            if failed
                            else preparation_status(planner, transaction.operation, 0, exc)
                        )
                        detail = (
                            f"{name!r}: rank {refused.rank} refused before collective entry: "
                            f"{refused.detail}"
                        )
                        failure: PreparationRefusal | memory.MemoryRefusal
                        if refused.code in memory.CAPACITY_REFUSALS:
                            failure = memory.MemoryRefusal(
                                refused.code,
                                detail,
                                refused.needed,
                                refused.available,
                            )
                            if refused.code == "host_shortfall":
                                failure.shortfall["resource"] = "ram"
                            if planner is not None:
                                planner.refusal = failure
                        else:
                            failure = PreparationRefusal(
                                detail,
                                code=refused.code or "group_preparation_failed",
                            )
                        if planner is not None:
                            planner.preparation_failure = failure.code
                        raise failure from None
                    finally:
                        if not transaction.committed:
                            transaction.abort()
                    try:
                        for follower, frame in zip(
                            group.followers, group.collect("run"), strict=True
                        ):
                            reply = _accept(name, group, follower.rank, frame)
                            attention_sol.record_rank(follower.rank, reply.sol_calls)
                        return result
                    except BaseException:
                        group._break(f"mirrored component {name!r} failed after commit")
                        raise
        finally:
            payloads.clear()

    mirrored.__dict__["__cozy_group_router__"] = weakref.ref(mirrored)
    component._call_impl = mirrored


def _check_cancelled(cancelled: Callable[[], bool], name: tuple[str, str]) -> None:
    if cancelled():
        raise Cancelled(f"the attempt was cancelled before committing {name!r}")


def preparation_status(
    planner: memory.Planner | None,
    operation: int,
    rank: int,
    error: BaseException | None = None,
) -> OpReady:
    """A prepared rank's control-plane report; no CUDA allocation is needed to send it."""
    facts = planner.facts() if planner is not None else {}
    numbers = planner.preparation_bytes(error) if planner is not None else {}
    code = classify(error).code if error is not None else ""
    if code in ("group_broken", "group_unformed", "gpu_divergence"):
        # A scope/input refusal before commit did not break the communicator. Protocol
        # failures take the explicit group._break path instead of this reusable outcome.
        code = "group_preparation_failed"
    if error is not None and is_device_oom(error):
        code = "device_out_of_memory"
    return OpReady(
        operation=operation,
        rank=rank,
        ok=error is None,
        code=code,
        detail=str(error)[:900] if error is not None else "",
        reusable=not (planner is not None and planner.unusable)
        and bool(getattr(error, "reusable", True)),
        needed=numbers.get("needed", 0),
        available=numbers.get("available", 0),
        free_bytes=numbers.get("free_bytes", 0),
        grant_bytes=numbers.get("grant_bytes", 0),
        free_wanted=numbers.get("free_wanted", 0),
        grant_wanted=numbers.get("grant_wanted", 0),
        memory=facts,
        host_needed=planner.host_needed() if planner is not None else 0,
        host_remaining=planner.host_remaining() if planner is not None else 0,
    )


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
        command, payloads = _command(
            name, prepared, group, spool, args, kwargs, cancelled, f"ranks-{rank}", only=True
        )
        # Per-rank directories: calls to different ranks may be in flight at once.
        results = wire.TensorSpool(Path(command.spool) / f"hosted-{rank}")
        hosted = msgspec.structs.replace(command, module=path, results=f"hosted-{rank}")
        try:
            try:
                frame = group.call_one(rank, msgspec.to_builtins(hosted))
                reply = _accept(name, group, rank, frame)
            except memory.MemoryRefusal as exc:
                if not exc.reusable:
                    group._break(f"hosted component {name!r} failed")
                raise
            except BaseException:
                group._break(f"hosted component {name!r} failed")
                raise
            return wire.unmarshal(reply.result, results, device=device)
        finally:
            payloads.clear()
            results.clear()

    return hosted


def install_spread(
    component: Any,
    *,
    name: tuple[str, str],
    prepared: Mapping[tuple[str, str], tuple[Any, Any]],
    group: RankGroup,
    spool: Callable[[], Path | None],
    device: Any,
    cancelled: Callable[[], bool] = lambda: False,
) -> None:
    """Let `author.spread` run `component`'s pure method calls on the idle ranks.

    Each round offers one call to every follower that is idle (no call in flight from
    another thread, no open remote scope homed there) and has not declined this spread, and
    runs one here. A follower that cannot take its call without freeing something it may not
    free declines it; the call goes back to the queue and that follower sits out the rest.
    Results are handed out in argument order.
    """
    followers = tuple(range(1, group.degree))

    def spread(method: str, arguments: Iterator[tuple[Any, ...]], spare: bool) -> Iterator[Any]:
        op = getattr(component, "_cozy_op", None)
        local = op(method) if op is not None else getattr(component, method)
        if prepared[name][0]._cozy_active is None:
            # Outside a component scope no follower may stage it: run here.
            yield from (local(*args) for args in arguments)
            return
        # This call's decliners sit out its remaining rounds; the attempt's log sums calls.
        declined: dict[str, str] = {}
        queue: deque[tuple[int, tuple[Any, ...]]] = deque()
        pending = enumerate(arguments)
        finished: dict[int, Any] = {}
        emitted = 0
        exhausted = False
        while True:
            taken = group.take(r for r in followers if str(r) not in declined)
            try:
                batch: list[tuple[int, tuple[Any, ...]]] = []
                while len(batch) < 1 + len(taken):
                    if queue:
                        batch.append(queue.popleft())
                    elif not exhausted and (item := next(pending, None)) is not None:
                        batch.append(item)
                    else:
                        exhausted = True
                        break
                if not batch:
                    break
                # String keys: the log crosses the seam as JSON.
                log = group.spread_log.setdefault(
                    f"{name[1]}.{method}", {"calls": {}, "declined": {}, "spare": spare}
                )
                _round(batch, taken, finished, queue, declined, log, method, spare, local)
            finally:
                group.give(taken)
            while emitted in finished:
                yield finished.pop(emitted)
                emitted += 1

    def _round(
        batch: list[tuple[int, tuple[Any, ...]]],
        taken: tuple[int, ...],
        finished: dict[int, Any],
        queue: deque[tuple[int, tuple[Any, ...]]],
        declined: dict[str, str],
        log: SpreadLog,
        method: str,
        spare: bool,
        local: Callable[..., object],
    ) -> None:
        sent: list[tuple[int, int, tuple[Any, ...], wire.TensorSpool]] = []
        calls: dict[str, int] = log["calls"]
        try:
            for rank, (index, args) in zip(taken, batch[1:], strict=False):
                command, payloads = _command(
                    name, prepared, group, spool, args, {}, cancelled, f"ranks-{rank}", only=True
                )
                sent.append((rank, index, args, payloads))
                spread = msgspec.structs.replace(command, method=method, p2p=True, spare=spare)
                group.send_one(rank, msgspec.to_builtins(spread))
            index, args = batch[0]
            finished[index] = local(*args)
            calls["0"] = calls.get("0", 0) + 1
            for rank, index, args, _payloads in sent:
                result = wire.recv_tensor(rank, group.pg, device)
                reply = _accept(name, group, rank, group.reply_one(rank, "run"))
                if result is None:
                    declined[str(rank)] = (reply.declined or "declined")[:300]
                    log["declined"][str(rank)] = declined[str(rank)]
                    queue.append((index, args))
                else:
                    finished[index] = result
                    calls[str(rank)] = calls.get(str(rank), 0) + 1
        except BaseException:
            group._break(f"spread {name!r}.{method} failed")
            raise
        finally:
            for _rank, _index, _args, payloads in sent:
                payloads.clear()

    object.__setattr__(component, "_cozy_spread", spread)


class GroupPlacement:
    """Rank 0's `author.Placement`: which of one model's components live on which follower.

    A remote scope reserves its components' home ranks, so no spread round takes them while
    the scope's calls may still arrive there.
    """

    def __init__(self, hosted: Mapping[str, int], group: RankGroup, torch: Any) -> None:
        self.hosted = dict(hosted)
        self.group = group
        self.torch = torch

    def remote(self, components: tuple[str, ...]) -> bool:
        return bool(components) and all(name in self.hosted for name in components)

    def enter(self, components: tuple[str, ...]) -> None:
        self.group.reserve(self.hosted[name] for name in components)

    def leave(self, components: tuple[str, ...]) -> None:
        self.group.unreserve(self.hosted[name] for name in components)

    def carry[T](self, call: Callable[[], T], components: tuple[str, ...]) -> Callable[[], T]:
        """`call` under this thread's autograd, inference and autocast modes and device:
        torch keeps all four per thread, and a new thread starts with none of them. The
        home ranks are reserved from here, before the thread runs, so a spread the caller
        starts next never takes them."""
        torch = self.torch
        self.enter(components)
        grad = torch.is_grad_enabled()
        inference = torch.is_inference_mode_enabled()
        autocast = [
            (kind, torch.get_autocast_dtype(kind))
            for kind in ("cuda", "cpu")
            if torch.is_autocast_enabled(kind)
        ]
        cuda = torch.cuda.current_device() if torch.cuda.is_initialized() else None

        def carried() -> T:
            with contextlib.ExitStack() as stack:
                stack.callback(self.leave, components)
                if cuda is not None:
                    stack.enter_context(torch.cuda.device(cuda))
                stack.enter_context(torch.inference_mode(inference))
                stack.enter_context(torch.set_grad_enabled(grad))
                for kind, dtype in autocast:
                    stack.enter_context(torch.autocast(device_type=kind, dtype=dtype))
                return call()

        return carried


def _command(
    name: tuple[str, str],
    prepared: Mapping[tuple[str, str], tuple[Any, Any]],
    group: RankGroup,
    spool: Callable[[], Path | None],
    args: tuple[object, ...],
    kwargs: Mapping[str, object],
    cancelled: Callable[[], bool],
    payloads_dir: str = "ranks",
    *,
    only: bool = False,
) -> tuple[wire.RunCommand, wire.TensorSpool]:
    """The `run` command for one component call under the leader's active scopes.

    Scope admission has already happened on the leader. Followers receive that same
    complete scope, including any overlay weights used by component hooks, and the same
    autocast. A call made inside a remote scope carries that scope alone. `only`: the call
    touches its own component and nothing else (a hosted module call, a spread method), so
    the follower admits just that component under the scope's method. A cancelled attempt
    raises here, before any rank is commanded.
    """
    import torch

    models = {key[0]: model for key, (model, _root) in prepared.items()}
    active = {
        key: model._cozy_active for key, model in models.items() if model._cozy_active is not None
    }
    remote = remote_scope(models[name[0]])
    if remote is not None:
        active = {name[0]: remote}
    if only and name[0] in active:
        active = {name[0]: (active[name[0]][0], (name[1],))}
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
    scopes = [wire.Scope(key, method, components) for key, (method, components) in active.items()]
    payloads = wire.TensorSpool(directory / payloads_dir, references)
    command = wire.RunCommand(
        component=name,
        spool=str(directory),
        payloads=payloads_dir,
        scopes=tuple(scopes),
        memory_grants=group.memory_grants,
        grad_enabled=torch.is_grad_enabled(),
        inference_mode=torch.is_inference_mode_enabled(),
        autocast={
            kind: str(torch.get_autocast_dtype(kind)).removeprefix("torch.")
            for kind in ("cuda", "cpu")
            if torch.is_autocast_enabled(kind)
        },
        attention_layout=_ACTIVE_LAYOUT.get(),
        args=[wire.marshal(a, payloads, path=f"args[{i}]") for i, a in enumerate(args)],
        kwargs={k: wire.marshal(v, payloads, path=k) for k, v in kwargs.items()},
    )
    return command, payloads


def _accept(
    name: tuple[str, str], group: RankGroup, rank: int, frame: Mapping[str, object]
) -> wire.RunReply:
    """Rank `rank`'s answer to one call: its failure raised by name, its peak and evidence
    folded into the attempt's."""
    try:
        reply = msgspec.convert(frame, wire.RunReply)
    except msgspec.ValidationError as exc:
        raise GroupRefusal(f"{name}: {group.gpu(rank)} answered malformed: {exc}"[:600]) from exc
    if not reply.ok:
        if reply.reusable:
            failure = memory.MemoryRefusal(
                reply.code, reply.detail, reply.needed, reply.available, reusable=True
            )
            if reply.code == "host_shortfall":
                failure.shortfall["resource"] = "ram"
            raise failure
        raise GroupRefusal(
            f"{name}: {group.gpu(rank)} failed its part of the call: {reply.code}: {reply.detail}"[
                :600
            ],
            code=reply.code or "group_broken",
        )
    group.working_peaks[rank] = max(group.working_peaks.get(rank, 0), reply.working_peak_bytes)
    if reply.evidence is not None:
        execution_evidence.widen(group.rank_records, reply.evidence)
    return reply
