"""The torch side of Runtime's Ulysses attention: the process group a parallel configuration
names, and the exchange around a head-local kernel (`attention_ulysses`). Sequence shards cross
to whole sequences per head subset, `compute` runs, and its output crosses back.

Two rounds of metadata over the group's store come first. Every rank proposes its rows and the
level native selection takes for it; every rank takes the finest proposed, admits it and allocates
its first buffers; every rank reports ready or refused. No rank enters a collective alone, and a
refusal there selects the next finer level on every rank. Once a tensor crosses nothing runs again.
"""

from __future__ import annotations

import json
import threading
import weakref
from collections.abc import Callable, Sequence
from typing import Any, cast

import tensorfs.varena.torch as vt
import torch
import torch.distributed as dist
from torch import Tensor
from torch.distributed import ProcessGroup
from torch.distributed.distributed_c10d import _get_process_group_store

from cozy_runtime.author._errors import Fault, fault
from cozy_runtime.internal import attention_ulysses
from cozy_runtime.internal.execution import catch, sizing
from cozy_runtime.internal.parallel import copy_engine
from cozy_runtime.internal.parallel.group import abort_process_group

type Compute = Callable[[Tensor, Tensor, Tensor], Tensor]


def group(config: object) -> ProcessGroup:
    """The process group of Runtime's initialized Ulysses configuration; ring is refused."""
    cp = getattr(config, "context_parallel_config", None)
    if cp is None:
        raise ValueError("context parallelism requires an initialized Ulysses configuration")
    if cp.ring_degree != 1:
        raise ValueError("ring attention needs LSE merging; only Ulysses is served")
    mesh = cp._ulysses_mesh
    handle = None if mesh is None else mesh.get_group()
    if handle is None or not dist.is_initialized():
        raise ValueError("the Ulysses configuration is not initialized")
    if dist.get_world_size(handle) != cp.ulysses_degree:
        raise ValueError("the Ulysses group does not match its declared degree")
    return cast(ProcessGroup, handle)


class _Rendezvous:
    """Metadata rounds over the group's store; no tensor crosses here."""

    def __init__(self, handle: ProcessGroup) -> None:
        self.store = dist.PrefixStore(
            f"cozy.ulysses/{handle.group_name}/", _get_process_group_store(handle)
        )
        self.rank, self.world = dist.get_rank(handle), dist.get_world_size(handle)
        self.sequence = 0
        self.lock = threading.Lock()

    def gather(self, stage: str, value: dict[str, Any]) -> list[dict[str, Any]]:
        prefix = f"{self.sequence}/{stage}/"
        keys = [f"{prefix}{rank}" for rank in range(self.world)]
        self.store.set(keys[self.rank], json.dumps(value))
        rows = [json.loads(data) for data in self.store.multi_get(keys)]
        # Rank 0 deletes once every rank has read, so a long-lived group's store stays small.
        done = [f"{prefix}read/{rank}" for rank in range(self.world)]
        self.store.set(done[self.rank], b"1")
        if self.rank == 0:
            self.store.multi_get(done)
            for key in (*keys, *done):
                self.store.delete_key(key)
        return rows


_RENDEZVOUS: weakref.WeakKeyDictionary[ProcessGroup, _Rendezvous] = weakref.WeakKeyDictionary()
_RENDEZVOUS_LOCK = threading.Lock()


def _rendezvous(handle: ProcessGroup) -> _Rendezvous:
    with _RENDEZVOUS_LOCK:
        found = _RENDEZVOUS.get(handle)
        if found is None:
            found = _RENDEZVOUS[handle] = _Rendezvous(handle)
        return found


def _steps(heads: int, world: int) -> list[int]:
    """Head groups a rank can take whole heads of, whole first."""
    return sorted({heads, *(n for n in range(world, heads, world) if heads % n == 0)}, reverse=True)


def _need(
    step: int, batch: int, rows: int, heads: int, dim: int, b: int, world: int
) -> tuple[int, ...]:
    """A group of `step` heads: the output held throughout; its q, k and v sent and received; then
    received beside the local attention; then the attended rows and their return. Every rank's rows
    are taken as this one's: an estimate."""
    total, local = rows * world, step // world
    each = batch * rows * step * dim * b
    attention = sizing.Attention(batch, total, local, local, total, dim, b).need(local, total)
    return sizing.peak(
        batch * rows * heads * dim * b,
        (each, each, each, each, each, each),
        (each, each, each, *attention),
        (each, each),
    )


def _refusal(rows: Sequence[dict[str, Any]]) -> None:
    """Raise a refusal some rank reported, as what it was."""
    failed = [row for row in rows if row.get("error")]
    if failed:
        kind = torch.OutOfMemoryError if any(row["oom"] for row in failed) else RuntimeError
        raise kind(f"Ulysses refused before its first crossing: {[r['error'] for r in failed]}")


def _failure(exc: BaseException) -> dict[str, Any]:
    return {"error": f"{type(exc).__name__}: {exc}", "oom": fault(exc) is Fault.OOM}


def _cross(
    handle: ProcessGroup,
    engines: copy_engine.CopyEngineExchange | None,
    sends: list[Tensor],
    receives: list[Tensor],
    matrix: list[list[int]],
    rank: int,
) -> None:
    """One crossing of `matrix`'s rows, over the copy engines when the group formed over them."""
    if engines is not None:
        engines.all_to_all_into(sends, matrix, receives)
        return
    works = [
        dist.all_to_all_single(
            out,
            value,
            output_split_sizes=[row[rank] for row in matrix],
            input_split_sizes=matrix[rank],
            group=handle,
            async_op=True,
        )
        for value, out in zip(sends, receives, strict=True)
    ]
    for work in works:
        work.wait()


def _pack(source: Tensor, target: Tensor, start: int, world: int) -> None:
    """`source`'s heads from `start`, into send rows block per GPU; heads past its end are the
    padding every GPU agreed."""
    batch, rows, heads, dim = source.shape
    local = target.shape[2]
    grouped = target.view(world, rows, batch, local, dim)
    if start + world * local > heads:
        grouped.zero_()
    for peer in range(world):
        first = start + peer * local
        count = min(local, max(heads - first, 0))
        if count:
            grouped[peer, :, :, :count].permute(1, 0, 2, 3).copy_(
                source[:, :, first : first + count]
            )


def _buffers(query: Tensor, world: int, total: int, local: int) -> list[list[Tensor]]:
    """A head group's sends and receives, for q, k and v."""
    batch, rows, _, dim = (int(n) for n in query.shape)
    return [
        [query.new_empty((n, batch, local, dim)) for _ in range(3)] for n in (world * rows, total)
    ]


def exchange(
    query: Tensor, key: Tensor, value: Tensor, handle: ProcessGroup, compute: Compute
) -> Tensor:
    """`compute` over whole sequences per head subset, back to this GPU's rows of every head."""
    meet = _rendezvous(handle)
    held = catch.LEAF.acquire(blocking=False)
    try:
        # One model pass owns its group's collective order.
        with meet.lock:
            meet.sequence += 1
            return _exchange(meet, query, key, value, handle, compute, held)
    finally:
        if held:
            catch.LEAF.release()


def _exchange(
    meet: _Rendezvous,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    handle: ProcessGroup,
    compute: Compute,
    held: bool,
) -> Tensor:
    rank, world = meet.rank, meet.world
    batch, mine, heads, dim = (int(n) for n in query.shape)
    padded = heads + (-heads % world)
    steps = _steps(padded, world)
    menu = tuple(
        (str(at), _need(step, batch, mine, padded, dim, query.element_size(), world))
        for at, step in enumerate(steps)
    )
    site = f"ulysses:{hash(menu):x}"
    device = query.device.index or 0
    request = catch.CURRENT.get()
    native = (
        held
        and request is not None
        and request.memory.cuda
        and vt.split_context(device)["eligible"]
    )
    stream = 0
    if native:
        assert request is not None
        stream = int(torch.cuda.current_stream(device).cuda_stream)
        request.memory.remember_working_phase(site, catch.owners(request))
    floor = 0
    while True:
        proposal: dict[str, Any] = {"rows": mine, "at": max(floor, catch.LEVELS.get(site, 0))}
        selection = None
        try:
            if native:
                proposal["at"], selection = catch.select(device, site, menu, stream, floor)
        except Exception as exc:
            proposal |= _failure(exc)
        proposals = meet.gather(f"propose/{floor}", proposal)
        _refusal(proposals)
        sizes = [int(row["rows"]) for row in proposals]
        at = max(int(row["at"]) for row in proposals)
        step, total, local = steps[at], sum(sizes), steps[at] // world
        token, buffers, ready = None, None, {}
        try:
            if native:
                token = catch.begin(device, site, at, selection, stream)
            out = query.new_empty((batch, mine, padded, dim))
            buffers = _buffers(query, world, total, local)
        except Exception as exc:
            ready = _failure(exc)
        answers = meet.gather(f"ready/{floor}", ready)
        if not any(row.get("error") for row in answers):
            break
        if token is not None:
            vt.split_end(device, token, submitted=False)
        buffers = out = None  # type: ignore[assignment]
        if at + 1 == len(steps) or not all(row["oom"] for row in answers if row.get("error")):
            _refusal(answers)
        catch.went_finer("ulysses", at)
        floor = at + 1
    catch.LEVELS[site] = at
    if request is not None:
        catch.note(request, site, "ulysses", at)
    assert buffers is not None
    sends, receives = buffers
    del buffers
    sent = [[n] * world for n in sizes]
    back = [list(sizes) for _ in range(world)]
    engines = copy_engine.for_group(handle)
    crossed = submitted = False
    try:
        for start in range(0, padded, step):
            for source, target in zip((query, key, value), sends, strict=True):
                _pack(source, target, start, world)
            crossed = True
            _cross(handle, engines, sends, receives, sent, rank)
            del sends
            attended = compute(*(t.permute(1, 0, 2, 3).contiguous() for t in receives))
            del receives
            if attended.shape != (batch, total, local, dim) or attended.dtype != query.dtype:
                raise ValueError("Ulysses attention returned incompatible output")
            returned = query.new_empty((world * mine, batch, local, dim))
            _cross(
                handle, engines, [attended.permute(1, 0, 2, 3).contiguous()], [returned], back, rank
            )
            del attended
            part = out[:, :, start : start + step].view(batch, mine, world, local, dim)
            part.copy_(returned.view(world, mine, batch, local, dim).permute(2, 1, 0, 3, 4))
            del returned
            if start + step < padded:
                sends, receives = _buffers(query, world, total, local)
        submitted = True
    except BaseException:
        if crossed:
            abort_process_group()
        raise
    finally:
        if token is not None:
            vt.split_end(device, token, submitted=submitted)
    return out[:, :, :heads] if padded != heads else out


attention_ulysses.serve(group, exchange)
