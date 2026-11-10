"""The torch side of Runtime's Ulysses attention: the process group a parallel configuration
names, and the exchange around a head-local kernel (`attention_ulysses`). Sequence shards cross
to whole sequences per head subset, `compute` runs, and its output crosses back.

The heads cross in groups, pipelined: the next group's Q/K/V and the last group's output cross
while this group's attention runs (an exchange's own streams, or NCCL's), so the links and the
SMs work at once. Packing runs on a stream of its own beside the attention: the next group is
packed and sent before this one is waited for, so a link never idles between two groups. A level
of the native split is a head group; an exchange large enough to pipeline runs in at least
`_ROUNDS` of them (`_pipelined`).

Where the kernel quantizes its operands itself, Q, K, V and the output cross as int8 (`turn`):
about half the bytes. K crosses less its mean over the whole sequence, which every GPU takes from
one all-reduce: attention is the same under a shift of every key, and the rows left are its own.

One round of metadata over the group's store comes first: every rank proposes its rows and the
level native selection takes for it, its room made, or its refusal; every rank takes the finest
proposed. A refusal there is every rank's, before anything crosses. Room taken after it (another
program, between selection and the split) fails that rank alone: the group's interrupt ends the
collectives its peers entered and the attempt ends short of memory (`spmd`).
"""

from __future__ import annotations

import json
import threading
import weakref
from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager, nullcontext
from typing import Any, cast

import tensorfs.varena.torch as vt
import torch
import torch.distributed as dist
from torch import Tensor
from torch.distributed import ProcessGroup
from torch.distributed.distributed_c10d import _get_process_group_store

from cozy_runtime.accel.kernels import Launch
from cozy_runtime.author._errors import DoesNotFit, Fault, fault
from cozy_runtime.graphs import streams
from cozy_runtime.internal import attention_ulysses
from cozy_runtime.internal.execution import catch, sizing
from cozy_runtime.internal.parallel import copy_engine, turn
from cozy_runtime.internal.parallel.group import abort_process_group, gather

type Compute = Callable[[Tensor, Tensor, Tensor], Tensor]
type Wait = Callable[[], None]

#: Head groups an exchange that pipelines runs in, at least.
_ROUNDS = 4
#: Each group's Q/K/V to one peer, at least, for a group's crossing to outlast its launches.
_ROUND_BYTES = 16 << 20
#: Each GPU's packing stream (`_Packer`)
_PACKING: dict[int, tuple[Any, ...]] = {}


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
    step: int, batch: int, rows: int, heads: int, dim: int, b: int, world: int, narrow: bool
) -> tuple[int, ...]:
    """A group of `step` heads: the output held throughout. Whole: its q, k and v sent and
    received, its operands too; then they beside the local attention; then the attended rows and
    their return. Pipelined: beside a group's operands and its attention, the next group's sent,
    received and operands (allocated with its send) and the last group's return. What crosses
    is int8 with a scale a row where `narrow`. Every rank's rows are taken as this one's: an
    estimate."""
    total, local = rows * world, step // world
    each = batch * rows * step * dim * b
    wire = batch * rows * step * (dim + b) if narrow else each
    attention = sizing.Attention(batch, total, local, local, total, dim, b).need(local, total)
    out = batch * rows * heads * dim * b
    if step == heads:
        sent = (*(wire,) * 6, *(each,) * 3)
        return sizing.peak(out, sent, (each, each, each, *attention), (wire, wire))
    return sizing.peak(out, (*(each,) * 6, *(wire,) * 8, *attention))


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


def _groups(padded: int, step: int, world: int) -> list[tuple[int, int]]:
    """The head groups of a call, as (first head, heads per GPU): `step` heads each; where the
    exchange pipelines, the first and last take one head per GPU, so a call waits only for a
    small crossing before its first attention and after its last."""
    if step in (padded, world):
        return [(first, step // world) for first in range(0, padded, step)]
    groups = [(0, 1)]
    first, end = world, padded - world
    while first < end:
        count = min(step, end - first)
        groups.append((first, count // world))
        first += count
    return [*groups, (end, 1)]


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


class _Packer:
    """Where a call's packing runs: the fused int8 kernels (`turn.launcher`) or `turn.law`; and,
    unless only the law serves, three streams of this GPU's own beside the attention: `out` packs
    a group's Q, K and V, `back` its attended rows, and `inbound` unpacks what crossed (the
    operands; the landing), so a group's packing never queues behind the last one's return.
    Every buffer is the calling stream's: a side stream writes one only after an event the
    calling stream recorded once it was allocated (`mark`). The calling stream waits for a
    group's unpacking before its attention and for the last landing before it returns; a
    crossing's buffers go only once it has waited for what read them (`held`)."""

    def __init__(self, like: Tensor, narrow: bool) -> None:
        self.launch: Launch | None = turn.launcher(like) if narrow else None
        self.out: Any = None
        self.back: Any = None
        self.inbound: Any = None
        if like.is_cuda and (self.launch is not None or not narrow):
            device = like.device.index or 0
            found = _PACKING.get(device)
            if found is None:
                made = tuple(streams.stream(torch, like.device, priority=-1) for _ in range(3))
                found = _PACKING[device] = made
            self.out, self.back, self.inbound = found
        #: what an unpacking or landing on `inbound` read, kept until the calling stream waits
        self.held: list[object] = []
        #: the attended rows `back` packs, each with the event after its packing (`read`)
        self.reading: list[tuple[Tensor, Any]] = []

    def mark(self) -> Any:
        """An event on the calling stream: what it allocated so far is free for a side stream."""
        if self.out is None:
            return None
        event = torch.cuda.Event()
        event.record()
        return event

    def on(self, stream: Any, after: Any = None) -> AbstractContextManager[object]:
        """`stream` current (the calling stream's own where there is none), after `after`."""
        if stream is None:
            return nullcontext()
        if after is not None:
            stream.wait_event(after)
        return torch.cuda.stream(stream)

    def mark_on(self, stream: Any) -> Any:
        """An event on `stream` after its work so far (None without one)."""
        if stream is None:
            return None
        event = torch.cuda.Event()
        event.record(stream)
        return event

    def caught_up(self) -> None:
        """The calling stream waits for every unpacking and landing so far; what they read goes."""
        if self.inbound is not None:
            torch.cuda.current_stream(self.inbound.device).wait_stream(self.inbound)
        self.held.clear()

    def settle(self) -> None:
        """The calling stream waits for every side stream, so nothing goes while one reads it."""
        for stream in (self.out, self.back, self.inbound):
            if stream is not None:
                torch.cuda.current_stream(stream.device).wait_stream(stream)
        self.held.clear()
        self.reading.clear()

    def read(self, keep: int) -> None:
        """The attended rows `back` packed, all but the last `keep`, go once the calling stream
        waits for their packing: long done by then, so the wait costs the calling stream nothing."""
        while len(self.reading) > keep:
            _, packed = self.reading.pop(0)
            if packed is not None:
                torch.cuda.current_stream(packed.device).wait_event(packed)


def _pack(
    packer: _Packer,
    source: Tensor,
    target: Tensor,
    start: int,
    world: int,
    scale: Tensor | None = None,
    shift: Tensor | None = None,
) -> None:
    """`source`'s heads from `start`, into send rows block per GPU (narrowed where `scale` takes
    their scales, less the turned `shift` of each head); heads past its end are the padding every
    GPU agreed."""
    batch, rows, heads, dim = source.shape
    local = target.shape[2]
    grouped = target.view(world, rows, batch, local, dim).transpose(1, 2)
    factors = None if scale is None else scale.view(world, rows, batch, local, 1).transpose(1, 2)
    whole = start + world * local <= heads
    if not whole:
        grouped.zero_()
        if factors is not None:
            factors.zero_()
    # Every GPU's block in one pass where all are whole heads, else block by block.
    for peer in [slice(None)] if whole else range(world):
        first = start if whole else start + cast(int, peer) * local
        count = local if whole else min(local, max(heads - first, 0))
        if not count:
            continue
        part = source[:, :, first : first + (world * local if whole else count)]
        offset = None if shift is None else shift[:, None, first : first + part.shape[2]]
        if whole:
            part = part.unflatten(2, (world, local)).permute(2, 0, 1, 3, 4)
            if offset is not None:
                offset = offset.unflatten(2, (world, local)).permute(2, 0, 1, 3, 4)
        into = grouped[peer, :, :, :count]
        if factors is None:
            into.copy_(part)
        else:
            turn.narrow(packer.launch, part, into, factors[peer, :, :, :count], offset)


class _Arriving:
    """A head group's Q, K, V crossing: its receivers (Q, K, V and their scales, each stacked in
    one tensor), and the operands its unpacking writes (stacked too: one launch unpacks all)."""

    def __init__(
        self, wait: Wait, values: Tensor, scales: Tensor | None, operands: Tensor, mark: Any
    ) -> None:
        self.wait, self.values, self.scales, self.operands = wait, values, scales, operands
        self.mark = mark


def _send(
    handle: ProcessGroup,
    engines: copy_engine.Exchange | None,
    packer: _Packer,
    sources: tuple[Tensor, Tensor, Tensor],
    start: int,
    total: int,
    local: int,
    matrix: list[list[int]],
    rank: int,
    narrow: bool,
    shift: _Shift | None,
) -> _Arriving:
    """A head group's q, k and v packed on `out` and crossing; its receivers, with their scales
    where `narrow` (K less the turned `shift`), and its operands, allocated now."""
    batch, rows, _, dim = (int(n) for n in sources[0].shape)
    world = len(matrix)
    like = sources[0]
    dtype = torch.int8 if narrow else like.dtype
    sends = [like.new_empty((world * rows, batch, local, dim), dtype=dtype) for _ in sources]
    values = like.new_empty((3, total, batch, local, dim), dtype=dtype)
    receives = list(values.unbind())
    scales: list[Tensor] = []
    received = None
    if narrow:
        scales = [like.new_empty((world * rows, batch, local, 1)) for _ in sources]
        received = like.new_empty((3, total, batch, local, 1))
        receives += received.unbind()
    operands = like.new_empty((3, batch, total, local, dim))
    mark = packer.mark()
    with packer.on(packer.out, mark):
        for index in (0, 2, 1):  # K last: its shift may still be on its way
            scale = scales[index] if narrow else None
            offset = shift() if shift is not None and index == 1 else None
            _pack(packer, sources[index], sends[index], start, world, scale, offset)
        wait = _start(handle, engines, sends + scales, receives, matrix, rank)
    return _Arriving(wait, values, received, operands, mark)


def _unpack(packer: _Packer, group: _Arriving) -> list[Tensor]:
    """A group's received q, k and v unpacked on `inbound` into its operands, as the kernel takes
    them ([batch, sequence, heads, dim]); the calling stream waits for them."""
    rows, scales, into = group.values.transpose(1, 2), group.scales, group.operands
    with packer.on(packer.inbound, group.mark):
        group.wait()
        if scales is None:
            into.copy_(rows)
        else:
            turn.widen(packer.launch, rows, scales.transpose(1, 2), into)
    packer.held.append((group.wait, group.values, scales))
    packer.caught_up()
    return list(into.unbind())


def _land(
    packer: _Packer,
    out: Tensor,
    wait: Wait,
    returned: list[Tensor],
    start: int,
    world: int,
) -> None:
    """A group's attended rows, back on this GPU, unpacked on `inbound` into its heads of `out`."""
    batch, mine, _, dim = out.shape
    local = returned[0].shape[2]
    with packer.on(packer.inbound):
        wait()
        part = out[:, :, start : start + world * local].view(batch, mine, world, local, dim)
        part = part.permute(2, 1, 0, 3, 4)
        rows = returned[0].view(world, mine, batch, local, dim)
        if len(returned) == 2:
            factors = returned[1].view(world, mine, batch, local, 1)
            turn.widen(packer.launch, rows, factors, part)
        else:
            part.copy_(rows)
    packer.held.append((wait, returned))


def _returned(
    handle: ProcessGroup,
    engines: copy_engine.Exchange | None,
    packer: _Packer,
    attended: Tensor,
    mine: int,
    back: list[list[int]],
    rank: int,
    narrow: bool,
) -> tuple[Wait, list[Tensor]]:
    """A group's attended rows, [batch, sequence, heads, dim], packed on `back` and crossing back:
    its receivers, with their scales where `narrow`."""
    batch, total, local, dim = attended.shape
    world = len(back)
    rows = attended.transpose(0, 1)
    dtype = torch.int8 if narrow else attended.dtype
    sends = [attended.new_empty((total, batch, local, dim), dtype=dtype)]
    returned = [attended.new_empty((world * mine, batch, local, dim), dtype=dtype)]
    if narrow:
        sends.append(attended.new_empty((total, batch, local, 1)))
        returned.append(attended.new_empty((world * mine, batch, local, 1)))
    with packer.on(packer.back, packer.mark()):
        if narrow:
            turn.narrow(packer.launch, rows, sends[0], sends[1])
        else:
            sends[0].copy_(rows)
        wait = _start(handle, engines, sends, returned, back, rank)
    # `attended` stays until the calling stream has waited for this packing (`_Packer.read`): a
    # block another stream still reads, freed on the calling stream, would be reused under it,
    # and `record_stream` instead holds every such block while the host runs ahead of the GPU.
    packer.reading.append((attended, packer.mark_on(packer.back)))
    return wait, returned


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


class _Shift:
    """K's mean over the whole sequence, per head, turned (`turn.turned`), so every GPU shifts
    every key alike: every GPU's sums, all-reduced on the stream current when it is made (`out`),
    which waits for them only at the first K packing."""

    def __init__(self, key: Tensor, total: int, handle: ProcessGroup) -> None:
        self.summed = key.sum(1, dtype=torch.float32)
        self.work = dist.all_reduce(self.summed, group=handle, async_op=True)
        self.total = total
        self.value: Tensor | None = None

    def __call__(self) -> Tensor:
        if self.value is None:
            if self.work is not None:
                self.work.wait()
            self.value = turn.turned(self.summed.div_(self.total))
        return self.value


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
    `narrow`: the kernel quantizes Q, K and V to int8 or narrower itself (a callable: for a call
    of `total` tokens), so they and its output cross as int8 (`turn`, about half the bytes), K
    less its mean over the whole sequence."""
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
    # The wire's width as the call's total will make it (`narrow` decides on the agreed total).
    wide = narrow(mine * world) if callable(narrow) else narrow
    crosses = dim & (dim - 1) == 0 and wide
    menu = tuple(
        (str(at), _need(step, batch, mine, padded, dim, b, world, crosses))
        for at, step in enumerate(steps)
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
    step = steps[at]
    token = None
    if native:
        token = catch.begin(device, site, "ulysses", menu, at, selection, stream)
    catch.LEVELS[site] = at
    if request is not None:
        catch.note(request, site, "ulysses", at)
    sent = [[n] * world for n in sizes]
    back = [list(sizes) for _ in range(world)]
    engines = copy_engine.for_group(handle)
    packer = _Packer(query, narrow)
    groups = _groups(padded, step, world)
    crossed = submitted = False
    began = _Clock.mark(query)
    inner: list[Any] = []
    arrived: list[Any] = []
    landed: list[Any] = []
    try:
        out = query.new_empty((batch, mine, padded, dim))
        sources = (query, key, value)
        crossed = True
        shift = None
        if narrow:  # on `out`, beside the first group's Q and V packing
            with packer.on(packer.out, packer.mark()):
                shift = _Shift(key, total, handle)

        def send(start: int, local: int) -> _Arriving:
            args = (sources, start, total, local, sent, rank, narrow, shift)
            return _send(handle, engines, packer, *args)

        # In flight: the next group's q, k, v, and the last group's output (`_need`).
        arriving = send(*groups[0])
        returning: tuple[Wait, list[Tensor], int] | None = None
        for index, (start, local) in enumerate(groups):
            current = arriving
            if index + 1 < len(groups):
                arriving = send(*groups[index + 1])
            before = _Clock.mark(query)
            operands = _unpack(packer, current)
            arrived.append((before, _Clock.mark(query)))
            del current
            packer.read(keep=1)
            first = _Clock.mark(query)
            attended = compute(*operands)
            inner.append((first, _Clock.mark(query)))
            del operands
            if attended.shape != (batch, total, local, dim) or attended.dtype != query.dtype:
                raise ValueError("Ulysses attention returned incompatible output")
            if returning is not None:
                _land(packer, out, *returning, world)
            args = (attended, mine, back, rank, narrow)
            returning = (*_returned(handle, engines, packer, *args), start)
            del attended
        assert returning is not None
        _land(packer, out, *returning, world)
        before = _Clock.mark(query)
        packer.caught_up()
        landed.append((before, _Clock.mark(query)))
        packer.read(keep=0)
        submitted = True
    except BaseException:
        if crossed:
            abort_process_group()
        packer.settle()
        raise
    finally:
        if token is not None:
            vt.split_end(device, token, submitted=submitted)
    width = 1 + b / dim if narrow else b  # a narrowed element's bytes, its scale's share too
    row = batch * sum(local for _, local in groups) * dim
    moved = row * width * (mine * (world - 1) * 3 + (total - mine))
    _CLOCK.add(began, _Clock.mark(query), (inner, arrived, landed), round(moved))
    return out[:, :, :heads] if padded != heads else out


attention_ulysses.serve(group, exchange)
