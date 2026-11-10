"""The torch side of Runtime's Ulysses attention: the process group a parallel configuration
names, and the exchange around a head-local kernel (`attention_ulysses`). Sequence shards cross
to whole sequences per head subset, `compute` runs, and its output crosses back.

The heads cross in groups, pipelined: the next group's Q/K/V and the last group's output cross
while this group's attention runs (an exchange's own streams, or NCCL's), so the links and the
SMs work at once. The next group is packed and sent before this one is waited for, so a link
never idles between two groups. A level of the native split is a head group; an exchange large
enough to pipeline runs in at least `_ROUNDS` of them (`_pipelined`).

One round of metadata over the group's store comes first: every rank proposes its rows and the
level native selection takes for it, its room made, or its refusal; every rank takes the finest
proposed. A refusal there is every rank's, before anything crosses. Room taken after it (another
program, between selection and the split) fails that rank alone: the group's interrupt ends the
collectives its peers entered and the attempt ends short of memory (`mirror.mirror_component`).
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

from cozy_runtime.author._errors import DoesNotFit, Fault, fault
from cozy_runtime.internal import attention_ulysses
from cozy_runtime.internal.execution import catch, sizing
from cozy_runtime.internal.parallel import copy_engine
from cozy_runtime.internal.parallel.group import abort_process_group, gather

type Compute = Callable[[Tensor, Tensor, Tensor], Tensor]
type Wait = Callable[[], None]

#: Head groups an exchange that pipelines runs in, at least.
_ROUNDS = 4
#: Each group's Q/K/V to one peer, at least, for a group's crossing to outlast its launches.
_ROUND_BYTES = 16 << 20
#: Hadamard matrices by device, dtype and order (`_narrow`).
_TURNS: dict[tuple[Any, Any, int], Tensor] = {}


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
        #: the level an exchange's geometry pipelines from, as the last agreed rows made it: the
        #: same on every rank, so every rank's menu starts there
        self.floors: dict[tuple[int, ...], int] = {}

    def gather(self, value: dict[str, Any]) -> list[dict[str, Any]]:
        """Every rank's `value` for this sequence: four store calls a rank. Each rank has read
        the last sequence's values before it proposes this one, so its own of them goes. A rank
        that never proposes ends the wait (`group.gather`); ranks whose values name different
        sites made different calls, and each of them says so."""
        keys = [f"{self.sequence}/{rank}" for rank in range(self.world)]
        self.store.set(keys[self.rank], json.dumps(value))
        what = f"Ulysses call {self.sequence} at {value['site']}"
        rows = [json.loads(data) for data in gather(self.store, keys, what)]
        if self.sequence > 1:
            self.store.delete_key(f"{self.sequence - 1}/{self.rank}")
        sites = [row["site"] for row in rows]
        if len(set(sites)) > 1:
            raise RuntimeError(
                f"Ulysses call {self.sequence} is not one call: "
                + ", ".join(f"GPU {rank} at {site}" for rank, site in enumerate(sites))
            )
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
    """A group of `step` heads: the output held throughout. Whole: its q, k and v sent and
    received; then received beside the local attention; then the attended rows and their return.
    Pipelined: beside a group's received q, k, v and its attention, the next group's sent and
    received and the last group's return; while the next is packed, this group's sent too. Every
    rank's rows are taken as this one's: an estimate."""
    total, local = rows * world, step // world
    each = batch * rows * step * dim * b
    attention = sizing.Attention(batch, total, local, local, total, dim, b).need(local, total)
    out = batch * rows * heads * dim * b
    if step == heads:
        return sizing.peak(out, (each,) * 6, (each, each, each, *attention), (each, each))
    return sizing.peak(out, (each,) * 14, (*(each,) * 11, *attention))


def _pipelined(steps: Sequence[int], sizes: Sequence[int], batch: int, dim: int, b: int) -> int:
    """The coarsest level of at least `_ROUNDS` groups whose Q/K/V to a peer is `_ROUND_BYTES`,
    or the finest that still is; 0 where even the whole exchange is smaller (latency-bound)."""
    world, rows, heads = len(sizes), max(sizes), steps[0]
    found = 0
    for at, step in enumerate(steps):
        if 3 * batch * rows * (step // world) * dim * b < _ROUND_BYTES:
            break
        found = at
        if heads // step >= _ROUNDS:
            break
    return found


def _refusal(rows: Sequence[dict[str, Any]]) -> None:
    """Raise a refusal some rank reported, as what it was."""
    failed = [row for row in rows if row.get("error")]
    if failed:
        kinds = {row["kind"] for row in failed}
        kind: type[Exception] = RuntimeError
        if "fit" in kinds:
            kind = DoesNotFit
        elif "oom" in kinds:
            kind = torch.OutOfMemoryError
        raise kind(f"Ulysses refused before its first crossing: {[r['error'] for r in failed]}")


def _failure(exc: BaseException) -> dict[str, Any]:
    kind = "fit" if isinstance(exc, DoesNotFit) else "oom" if fault(exc) is Fault.OOM else ""
    return {"error": f"{type(exc).__name__}: {exc}", "kind": kind}


def _start(
    handle: ProcessGroup,
    engines: copy_engine.Exchange | None,
    sends: list[Tensor],
    receives: list[Tensor],
    matrix: list[list[int]],
    rank: int,
) -> Wait:
    """One crossing of `matrix`'s rows in flight after the current stream's work so far, over
    the group's exchange when it formed with one, else on NCCL's stream. The call returned
    orders the current stream after it; it holds the crossing's tensors, so no allocation on
    the current stream takes their memory before then."""
    if engines is not None:
        done = engines.start(sends, matrix, receives)

        def landed(held: tuple[list[Tensor], list[Tensor]] = (sends, receives)) -> None:
            torch.cuda.current_stream(held[0][0].device).wait_event(done)

        return landed
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

    def waited(held: tuple[list[Tensor], list[Tensor]] = (sends, receives)) -> None:
        for work in works:
            work.wait()

    return waited


def _turn(dim: int, like: Tensor) -> Tensor:
    """The orthonormal Hadamard matrix of order `dim` (a power of two): its own inverse."""
    key = (like.device, like.dtype, dim)
    found = _TURNS.get(key)
    if found is None:
        turn = torch.ones(1, 1)
        while turn.shape[0] < dim:
            turn = torch.cat([torch.cat([turn, turn], 1), torch.cat([turn, -turn], 1)])
        found = _TURNS[key] = (turn * dim**-0.5).to(like.device, like.dtype)
    return found


def _narrow(x: Tensor) -> tuple[Tensor, Tensor]:
    """`x`'s rows turned and scaled to their own absmax, as int8; and those scales."""
    turned = x @ _turn(x.shape[-1], x)
    scale = torch.linalg.vector_norm(turned, float("inf"), dim=-1, keepdim=True) / 127
    scale.clamp_(min=torch.finfo(x.dtype).tiny)
    return turned.div_(scale).round_().clamp_(-127, 127).to(torch.int8), scale


def _widen(values: Tensor, scale: Tensor) -> Tensor:
    """`_narrow`'s rows back in `scale`'s dtype."""
    return values.to(scale.dtype).mul_(scale) @ _turn(values.shape[-1], scale)


def _pack(
    source: Tensor, target: Tensor, start: int, world: int, scale: Tensor | None = None
) -> None:
    """`source`'s heads from `start`, into send rows block per GPU (narrowed where `scale`
    takes their scales); heads past its end are the padding every GPU agreed."""
    batch, rows, heads, dim = source.shape
    local = target.shape[2]
    grouped = target.view(world, rows, batch, local, dim)
    factors = None if scale is None else scale.view(world, rows, batch, local, 1)
    if start + world * local > heads:
        grouped.zero_()
        if factors is not None:
            factors.zero_()
    for peer in range(world):
        first = start + peer * local
        count = min(local, max(heads - first, 0))
        if count:
            part = source[:, :, first : first + count]
            if factors is not None:
                part, factor = _narrow(part)
                factors[peer, :, :, :count].permute(1, 0, 2, 3).copy_(factor)
            grouped[peer, :, :, :count].permute(1, 0, 2, 3).copy_(part)


def _send(
    handle: ProcessGroup,
    engines: copy_engine.Exchange | None,
    sources: tuple[Tensor, Tensor, Tensor],
    start: int,
    total: int,
    local: int,
    matrix: list[list[int]],
    rank: int,
    narrow: bool,
) -> tuple[Wait, list[Tensor]]:
    """A head group's q, k and v packed and crossing; its receivers, Q and V with their scales
    where `narrow`."""
    batch, rows, _, dim = (int(n) for n in sources[0].shape)
    world = len(matrix)
    sends: list[Tensor] = []
    receives: list[Tensor] = []
    scales: list[tuple[Tensor, Tensor]] = []
    for index, source in enumerate(sources):
        wire = narrow and index != 1
        dtype = torch.int8 if wire else source.dtype
        target = source.new_empty((world * rows, batch, local, dim), dtype=dtype)
        scale = source.new_empty((world * rows, batch, local, 1)) if wire else None
        _pack(source, target, start, world, scale)
        sends.append(target)
        receives.append(source.new_empty((total, batch, local, dim), dtype=dtype))
        if scale is not None:
            scales.append((scale, source.new_empty((total, batch, local, 1))))
    sends += [sent for sent, _ in scales]
    receives += [received for _, received in scales]
    return _start(handle, engines, sends, receives, matrix, rank), receives


def _operands(receives: list[Tensor]) -> list[Tensor]:
    """A group's received q, k and v as the kernel takes them: [batch, sequence, heads, dim]."""
    q, k, v, *scales = receives
    if scales:
        q, v = _widen(q, scales[0]), _widen(v, scales[1])
    return [t.permute(1, 0, 2, 3).contiguous() for t in (q, k, v)]


def _land(
    out: Tensor, wait: Wait, returned: list[Tensor], start: int, world: int, waits: list[Any]
) -> None:
    """A group's attended rows, back on this GPU, into its heads of `out`."""
    batch, mine, _, dim = out.shape
    local = returned[0].shape[2]
    _waited(out, wait, waits)
    rows = _widen(*returned) if len(returned) == 2 else returned[0]
    part = out[:, :, start : start + world * local].view(batch, mine, world, local, dim)
    part.copy_(rows.view(world, mine, batch, local, dim).permute(2, 1, 0, 3, 4))


def _waited(on: Tensor, wait: Wait, waits: list[Any]) -> None:
    """`wait`, between two marks: what the calling stream idled for a crossing."""
    before = _Clock.mark(on)
    wait()
    waits.append((before, _Clock.mark(on)))


class _Clock:
    """This GPU's exchanges on their calling stream, by CUDA events read once complete: each
    one's span, the attention inside and the waits for crossings (Q/K/V arriving, output
    landing), so the rest of what communication left exposed is its packing."""

    _SUMS = ("calls", "groups", "crossed", "exchange", "attention", "arriving", "landing")

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.pending: list[tuple[Any, Any, tuple[list[Any], list[Any], list[Any]], int]] = []
        self.totals = dict.fromkeys(self._SUMS, 0.0)

    @staticmethod
    def mark(on: Tensor) -> Any:
        if not on.is_cuda or torch.cuda.is_current_stream_capturing():
            return None
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        return event

    def add(
        self,
        began: Any,
        ended: Any,
        spans: tuple[list[Any], list[Any], list[Any]],
        crossed: int,
    ) -> None:
        """`spans`: the attention, the waits for Q/K/V and the waits for output, as mark pairs."""
        if began is None or ended is None:
            return
        with self.lock:
            self.pending.append((began, ended, spans, crossed))
            if len(self.pending) > 256:
                self._fold(wait=False)

    def _fold(self, *, wait: bool) -> None:
        while self.pending:
            began, ended, spans, crossed = self.pending[0]
            if wait:
                ended.synchronize()
            elif not ended.query():
                return
            self.pending.pop(0)
            totals = self.totals
            totals["calls"] += 1
            totals["groups"] += len(spans[0])
            totals["crossed"] += crossed
            totals["exchange"] += began.elapsed_time(ended)
            for name, pairs in zip(("attention", "arriving", "landing"), spans, strict=True):
                totals[name] += sum(
                    a.elapsed_time(z) for a, z in pairs if a is not None and z is not None
                )

    def take(self) -> dict[str, float]:
        with self.lock:
            self._fold(wait=True)
            totals, self.totals = self.totals, dict.fromkeys(self._SUMS, 0.0)
        exchange, attention = totals["exchange"], totals["attention"]
        return {
            "calls": int(totals["calls"]),
            "groups": int(totals["groups"]),
            "crossed_gb": round(totals["crossed"] / 1e9, 3),
            "exchange_ms": round(exchange, 1),
            "attention_ms": round(attention, 1),
            "exposed_ms": round(exchange - attention, 1),
            "arriving_ms": round(totals["arriving"], 1),
            "landing_ms": round(totals["landing"], 1),
        }


_CLOCK = _Clock()


def taken() -> dict[str, float]:
    """What this GPU's exchanges moved and took since the last take (`cozy run show`:
    `transport.exchange`): device ms on the calling stream, the attention inside them, and the
    rest, which communication left exposed: of it, the stream's waits for Q/K/V to arrive and
    for output to land; what remains is packing."""
    return _CLOCK.take()


def exchange(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    handle: ProcessGroup,
    compute: Compute,
    site: str = "",
    narrow: bool | Callable[[int], bool] = False,
) -> Tensor:
    """`compute` over whole sequences per head subset, back to this GPU's rows of every head.
    `site` names the caller's attention module where it knows it: every GPU must be there.
    `narrow`: the kernel quantizes Q and V to int8 or narrower itself (a callable: for a call of
    `total` tokens), so they and its output cross as int8, each head's row Hadamard-turned and
    scaled to its own absmax (about 5/8 of the bytes). K crosses as it is: its mean dominates
    its rows until the kernel's shift."""
    meet = _rendezvous(handle)
    held = catch.LEAF.acquire(blocking=False)
    try:
        # One model pass owns its group's collective order.
        with meet.lock:
            meet.sequence += 1
            return _exchange(meet, query, key, value, handle, compute, held, site, narrow)
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
    called: str,
    narrow: bool | Callable[[int], bool],
) -> Tensor:
    rank, world = meet.rank, meet.world
    batch, mine, heads, dim = (int(n) for n in query.shape)
    b = query.element_size()
    padded = heads + (-heads % world)
    steps = _steps(padded, world)
    geometry = (batch, padded, dim, b)
    floor = min(meet.floors.get(geometry, 0), len(steps) - 1)
    # What a rank's rows and heads make of a group fixes the menu; each rank's menu stays its own.
    menu = tuple(
        (str(at), _need(step, batch, mine, padded, dim, b, world)) for at, step in enumerate(steps)
    )
    levels = menu[floor:]
    site = f"ulysses:{hash(levels):x}"
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
    proposal: dict[str, Any] = {
        "rows": mine,
        "at": max(catch.LEVELS.get(site, floor), floor),
        "site": f"{called or 'attention'} [{batch} x {padded} heads x {dim}, {b} B]",
    }
    selection: dict[str, Any] = {}
    try:
        if native:
            selection = catch.select(device, site, "ulysses", levels, stream)
            proposal["at"] = int(selection["choice_id"])
    except Exception as exc:
        proposal |= _failure(exc)
    proposals = meet.gather(proposal)
    _refusal(proposals)
    sizes = [int(row["rows"]) for row in proposals]
    total = sum(sizes)
    # Every GPU's kernel and total agree; a Hadamard turn needs a power-of-two order.
    narrow = dim & (dim - 1) == 0 and (narrow(total) if callable(narrow) else narrow)
    pipelined = meet.floors[geometry] = _pipelined(steps, sizes, batch, dim, b)
    at = max(pipelined, *(int(row["at"]) for row in proposals))
    step, local = steps[at], steps[at] // world
    token = None
    if native:
        token = catch.begin(device, site, "ulysses", menu, at, selection, stream)
    catch.LEVELS[site] = at
    if request is not None:
        catch.note(request, site, "ulysses", at)
    sent = [[n] * world for n in sizes]
    back = [list(sizes) for _ in range(world)]
    engines = copy_engine.for_group(handle)
    starts = range(0, padded, step)
    crossed = submitted = False
    began = _Clock.mark(query)
    inner: list[Any] = []
    arrived: list[Any] = []
    landed: list[Any] = []
    try:
        out = query.new_empty((batch, mine, padded, dim))
        sources = (query, key, value)
        # In flight: the next group's q, k, v, and the last group's output (`_need`).
        arriving = [_send(handle, engines, sources, starts[0], total, local, sent, rank, narrow)]
        crossed = True
        returning: list[tuple[Wait, list[Tensor], int]] = []
        for index, start in enumerate(starts):
            wait, receives = arriving.pop()
            if index + 1 < len(starts):
                following = starts[index + 1]
                arriving.append(
                    _send(handle, engines, sources, following, total, local, sent, rank, narrow)
                )
            _waited(query, wait, arrived)
            del wait  # its sends go: the current stream is past their crossing
            operands = _operands(receives)
            del receives
            first = _Clock.mark(query)
            attended = compute(*operands)
            inner.append((first, _Clock.mark(query)))
            del operands
            if attended.shape != (batch, total, local, dim) or attended.dtype != query.dtype:
                raise ValueError("Ulysses attention returned incompatible output")
            if returning:
                _land(out, *returning.pop(), world, landed)
            flat = attended.permute(1, 0, 2, 3).contiguous()
            del attended
            sends = list(_narrow(flat)) if narrow else [flat]
            del flat
            returned = [t.new_empty((world * mine, *t.shape[1:])) for t in sends]
            returning.append(
                (_start(handle, engines, sends, returned, back, rank), returned, start)
            )
            del sends, returned
        _land(out, *returning.pop(), world, landed)
        submitted = True
    except BaseException:
        if crossed:
            abort_process_group()
        raise
    finally:
        if token is not None:
            vt.split_end(device, token, submitted=submitted)
    width = 1 + b / dim if narrow else b  # a narrowed element's bytes, its scale's share too
    row = len(starts) * batch * local * dim
    moved = row * (mine * (world - 1) * (2 * width + b) + (total - mine) * width)
    _CLOCK.add(began, _Clock.mark(query), (inner, arrived, landed), round(moved))
    return out[:, :, :heads] if padded != heads else out


attention_ulysses.serve(group, exchange)
