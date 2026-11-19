"""K/V and recurrent state for the engine's rows, in two varena Caches (cr-336; design
inference-engine.md §4.2 and §4.3).

K/V is paged: a block is `block_tokens` tokens of K and V for every attention layer
(`[block][layer][K|V][token][kv_head][head_dim]`), block `i` at `i * block_bytes` of one Cache
reservation. A page is mapped while a block on it is held and released when its last block is
free, so memory follows the rows, a page at a time, and nothing is sized for a window. A row's
blocks are listed in `block_table`; a block index means the same tokens in every layer.

Each running row also holds one state slot (a gated-delta layer's conv and recurrent state: fixed
per sequence), a whole number of pages in a second Cache, and while it drafts, room for the
verify span's candidate states (`state_scratch`).

`block_table`, `lengths`, `slot_mapping` and `state_slot` are device tensors at addresses that
never change for the manager's life, so a CUDA graph holds them. The step's graph writes
`lengths`, `slot_mapping` and the accepted state itself. The host writes only when a row joins,
leaves or gains a block, between steps, on the step's stream: every method here that writes the
device or releases pages does so on the calling thread's current stream.

Every size comes from the model's layout, the request and the card's page size. Pages of a row
that left are released as it leaves, so nothing here is idle memory for the core to ask back.

A host tier (`host`, V2's HostCache in pinned host memory the device addresses) holds the K/V the
card doesn't: block ids at and above `card_blocks` are its blocks, laid out layer-major
(`[layer][block][K|V][token][kv_head][head_dim]`, `host_view`), so one layer's off-card blocks are
one range. Which of a row's blocks are there is decided when it joins (`admit`'s `host_tokens`): its
oldest whole blocks, never its tail, so a step writes nothing on the host. A step reads them through
a window on the card each attention layer's off-card blocks are copied to one layer ahead, on a copy
stream while the layer before computes (`CopyAhead`; cr-336's decision from the kv-tiers
measurement); reading them in place (`host_view`) is the fallback. A prefill writes and reads them by
copies too. Where a host block is read from is the attention op's one `resolve`.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from math import prod
from typing import TYPE_CHECKING

import torch

from cozy_runtime.engine.attention import HostTier
from cozy_runtime.engine.rungs import KvNeed

if TYPE_CHECKING:
    from collections.abc import Sequence

    from tensorfs.varena import _runtime as varena
    from torch import Tensor

#: A table's and a state tensor's alignment inside its range.
_ALIGN = 256


@dataclass(frozen=True)
class StateSpec:
    """One tensor of a row's recurrent state: its shape for one row, and its dtype."""

    shape: tuple[int, ...]
    dtype: torch.dtype

    @property
    def nbytes(self) -> int:
        return prod(self.shape) * self.dtype.itemsize


@dataclass(frozen=True)
class KvLayout:
    """A model's K/V and state, from its config: fixed for the model."""

    attn_layers: int
    kv_heads: int
    head_dim: int
    dtype: torch.dtype
    states: tuple[StateSpec, ...] = ()
    block_tokens: int = 16
    depth: int = 2

    @property
    def span(self) -> int:
        """The queries of a drafted step: its token and `depth` drafts."""
        return self.depth + 1

    @property
    def token_bytes(self) -> int:
        return self.attn_layers * 2 * self.kv_heads * self.head_dim * self.dtype.itemsize

    @property
    def block_bytes(self) -> int:
        return self.block_tokens * self.token_bytes

    @property
    def state_bytes(self) -> int:
        """One row's state tensors, each aligned."""
        return sum(_up(spec.nbytes, _ALIGN) for spec in self.states)

    def written_blocks(self, ahead: int) -> int:
        """The most blocks `ahead` consecutive tokens touch."""
        return (ahead + self.block_tokens - 2) // self.block_tokens + 1


def _up(n: int, unit: int) -> int:
    return -(-n // unit) * unit


class _Pages:
    """Which pages of a Cache its ranges hold: a page is mapped at its first holder and released
    at its last. A range may share its first and last page with its neighbours."""

    def __init__(self, cache: varena.Cache, device: torch.device) -> None:
        self.cache, self.page, self.device = cache, cache.granularity, device
        self.held: dict[int, int] = {}

    def _stream(self) -> int:
        return int(torch.cuda.current_stream(self.device).cuda_stream)

    def _span(self, offset: int, nbytes: int) -> range:
        return range(offset // self.page, (offset + nbytes - 1) // self.page + 1)

    def take(self, ranges: Sequence[tuple[int, int]]) -> None:
        """Hold each (offset, bytes), mapping the pages none held: all of them or, refused, none
        (the refusal is raised with the pages mapped here released)."""
        fresh = sorted(
            {p for at, n in ranges for p in self._span(at, n) if p not in self.held}
        )
        mapped: list[tuple[int, int]] = []
        try:
            for first, count in _runs(fresh):
                self.cache.commit(first * self.page, count * self.page)
                mapped.append((first, count))
        except BaseException:
            for first, count in mapped:  # nothing has read or written them
                self.cache.release(first * self.page, count * self.page, self._stream())
            raise
        for at, n in ranges:
            for p in self._span(at, n):
                self.held[p] = self.held.get(p, 0) + 1

    def drop(self, ranges: Sequence[tuple[int, int]]) -> None:
        """Let each (offset, bytes) go; a page none holds any more is released behind the
        current stream, which every reader and writer of it is ordered on."""
        gone: list[int] = []
        for at, n in ranges:
            for p in self._span(at, n):
                self.held[p] -= 1
                if not self.held[p]:
                    del self.held[p]
                    gone.append(p)
        for first, count in _runs(sorted(gone)):
            self.cache.release(first * self.page, count * self.page, self._stream())

    def mapped(self, offset: int, nbytes: int) -> bool:
        return all(p in self.held for p in self._span(offset, nbytes))


class _HostPages:
    """`_Pages` for a HostCache: windows mapped at their first holder and released at their last.
    A commit the host has no room for raises, naming the bytes."""

    def __init__(self, cache: varena.HostCache) -> None:
        self.cache, self.page = cache, cache.granularity
        self.held: dict[int, int] = {}

    def _span(self, offset: int, nbytes: int) -> range:
        return range(offset // self.page, (offset + nbytes - 1) // self.page + 1)

    def take(self, ranges: Sequence[tuple[int, int]]) -> None:
        fresh = sorted({p for at, n in ranges for p in self._span(at, n) if p not in self.held})
        mapped: list[tuple[int, int]] = []
        for first, count in _runs(fresh):
            if not self.cache.commit(first * self.page, count * self.page):
                for at, n in mapped:
                    self.cache.release(at * self.page, n * self.page)
                raise RuntimeError(f"the host has no room for {count * self.page} more bytes of K/V now")
            mapped.append((first, count))
        for at, n in ranges:
            for p in self._span(at, n):
                self.held[p] = self.held.get(p, 0) + 1

    def drop(self, ranges: Sequence[tuple[int, int]]) -> None:
        """Release what none holds now; it waits for the device's queued work (HostCache.release)."""
        gone: list[int] = []
        for at, n in ranges:
            for p in self._span(at, n):
                self.held[p] -= 1
                if not self.held[p]:
                    del self.held[p]
                    gone.append(p)
        for first, count in _runs(sorted(gone)):
            self.cache.release(first * self.page, count * self.page)


def _runs(pages: Sequence[int]) -> list[tuple[int, int]]:
    """Sorted pages as (first, count) of each consecutive run."""
    runs: list[tuple[int, int]] = []
    for p in pages:
        if runs and runs[-1][0] + runs[-1][1] == p:
            runs[-1] = (runs[-1][0], runs[-1][1] + 1)
        else:
            runs.append((p, 1))
    return runs


class KvManager:
    """One model's K/V blocks and state slots on one GPU, for the model's life in the executor.

    `max_rows` is the most rows decoding at once and `max_blocks` a row's ceiling, from the
    model's context length. The K/V reservation is address space for what the card could ever
    hold of them, nothing mapped. The calling thread's current stream is an explicit one (the
    step's), never the default stream: the core orders releases on it."""

    def __init__(
        self,
        gpu: varena.Device,
        layout: KvLayout,
        max_rows: int,
        max_blocks: int,
        host: varena.Pages | None = None,
    ) -> None:
        self.layout, self.max_rows, self.max_blocks = layout, max_rows, max_blocks
        self.device = torch.device("cuda", gpu.device)
        block = layout.block_bytes
        card = int(torch.cuda.get_device_properties(self.device).total_memory)
        #: blocks below this index are on the card
        self.card_blocks = min(max_rows * max_blocks, card // block)
        self._kv = gpu.cache(self.card_blocks * block)
        self._kv_pages = _Pages(self._kv, self.device)
        # Block 0's page stays mapped for the manager's life: the card tier's base is every K/V
        # kernel's pointer, and a launcher asks the driver of it (an unmapped one is refused).
        self._kv_pages.take([(0, block)])
        page = self._kv.granularity
        span = layout.span
        tables = (
            ("block_table", torch.int32, (max_rows, max_blocks)),
            ("lengths", torch.int32, (max_rows,)),
            ("slot_mapping", torch.int64, (max_rows, span)),
            ("state_slot", torch.int32, (max_rows,)),
        )
        at, placed = 0, []
        for _, dtype, shape in tables:
            placed.append(at)
            at += _up(prod(shape) * dtype.itemsize, _ALIGN)
        self._tables = _up(at, page)
        self._slot = _up(layout.state_bytes, page)
        self._scratch = _up(span * layout.state_bytes, page)
        self._state = gpu.cache(self._tables + max_rows * (self._slot + self._scratch))
        self._state_pages = _Pages(self._state, self.device)
        views = [
            self._state.tensor(dtype, shape, byte_offset=offset)
            for (_, dtype, shape), offset in zip(tables, placed, strict=True)
        ]
        #: int32 [max_rows, max_blocks]: each row's blocks in order, -1 unassigned
        self.block_table: Tensor = views[0]
        #: int32 [max_rows]: tokens in the row's cache, -1 for a row that isn't decoding
        self.lengths: Tensor = views[1]
        #: int64 [max_rows, depth + 1]: the token slot each query of this step writes, -1 none
        self.slot_mapping: Tensor = views[2]
        #: int32 [max_rows]: the row's state slot, -1 none
        self.state_slot: Tensor = views[3]
        #: [card_blocks, attn_layers, 2, block_tokens, kv_heads, head_dim], the whole reservation
        self.blocks: Tensor = self._kv.tensor(
            layout.dtype,
            (self.card_blocks, layout.attn_layers, 2, layout.block_tokens, layout.kv_heads,
             layout.head_dim),
        )  # fmt: skip
        slots = self._state.tensor(torch.uint8, (max_rows, self._slot), byte_offset=self._tables)
        scratch = self._state.tensor(
            torch.uint8,
            (max_rows, self._scratch),
            byte_offset=self._tables + max_rows * self._slot,
        )
        self._states: list[Tensor] = []
        #: per state tensor, [max_rows, depth + 1, *shape]: the verify span's candidate states
        self.state_scratch: list[Tensor] = []
        at = 0
        for spec in layout.states:
            one = slots[:, at : at + spec.nbytes].view(spec.dtype)
            self._states.append(one.unflatten(1, spec.shape))
            many = scratch[:, span * at : span * (at + spec.nbytes)].view(spec.dtype)
            self.state_scratch.append(many.unflatten(1, (span, *spec.shape)))
            at += _up(spec.nbytes, _ALIGN)
        #: each admitted row's blocks, in order, and the length the host last knew it at
        self._held: dict[int, list[int]] = {}
        self._known: dict[int, int] = {}
        self._slots: dict[int, int] = {}
        self._drafting: set[int] = set()
        self._free: set[int] = set(range(1, self.card_blocks))
        #: free blocks on pages a held block keeps mapped: taken before any page is mapped
        self._spare: set[int] = set()
        #: the other free blocks, lowest first; an entry may be stale (`_take` checks it)
        self._fresh: list[int] = list(range(1, self.card_blocks))
        self._respare([0])
        # The host tier: address space for a whole row's blocks a row, its first window mapped for
        # the manager's life (the tier's base is a kernel's pointer: Triton asks the driver of it).
        self.host_blocks = 0
        self._host: varena.HostCache | None = None
        self._host_pages: _HostPages | None = None
        self._host_free: list[int] = []
        self._window: varena.Cache | None = None
        self._window_pages: _Pages | None = None
        self._window_extent = 0
        self._window_taken: list[tuple[int, int]] = []
        if host is not None:
            layer_bytes = layout.block_bytes // layout.attn_layers
            self.host_blocks = max_rows * max_blocks
            self._host = host.cache(layout.attn_layers * self.host_blocks * layer_bytes)
            self._host_pages = _HostPages(self._host)
            self._host_pages.take([(0, layer_bytes)])
            self._host_free = list(range(self.host_blocks))
            # Two slots of one layer's host blocks on the card, each indexed by host block id.
            self._window = gpu.cache(2 * self.host_blocks * layer_bytes)
            self._window_pages = _Pages(self._window, self.device)
            self._window_taken = [(0, layer_bytes), (self.host_blocks * layer_bytes, layer_bytes)]
            self._window_pages.take(self._window_taken)
        #: [attn_layers, host_blocks, 2, block_tokens, kv_heads, head_dim] as the device addresses it
        #: (a view over the whole host reservation; only mapped windows are touched), or a 1-block
        #: stand-in on the card where there is no host tier
        self.host_view: Tensor = (
            self._host.tensor(layout.dtype, (layout.attn_layers, self.host_blocks, 2, layout.block_tokens,
                                             layout.kv_heads, layout.head_dim))
            if self._host is not None
            else self.blocks[:1].transpose(0, 1)
        )  # fmt: skip
        #: the same host bytes as pinned host memory: what copies to and from the card read and write
        self.host_cpu: Tensor | None = (
            self._host.host(layout.dtype, tuple(self.host_view.shape)) if self._host is not None else None
        )
        #: [2, host_blocks, 2, block_tokens, kv_heads, head_dim]: the copy-ahead windows on the card
        self.windows: Tensor | None = (
            self._window.tensor(layout.dtype, (2, *self.host_view.shape[1:])) if self._window is not None else None
        )

    # ------------------------------------------------------------------ what a call needs

    @property
    def ceiling(self) -> int:
        """The most tokens a row holds: its ceiling less a drafted step's reach past its length."""
        return self.max_blocks * self.layout.block_tokens - self.layout.depth

    def declare(self, rows: Sequence[tuple[int, int]], *, drafted: bool, ahead: int) -> KvNeed:
        """What a call can need, exactly: for each row's (prompt tokens, most new tokens), its
        blocks, its state slot and, drafting, its scratch. And of that, what a step writes: the
        blocks `ahead` tokens past a row's known length touch (what the caller passes
        `reserve_ahead`), the slot and the scratch, which stay on the card."""
        layout = self.layout
        state = self._slot + (self._scratch if drafted else 0)
        reach = layout.depth if drafted else 0
        declared = written = 0
        for prompt, new in rows:
            if prompt + new > self.ceiling:
                raise ValueError(f"a row of {prompt + new} tokens: the model holds {self.ceiling}")
            blocks = -(-(prompt + new + reach) // layout.block_tokens)
            declared += blocks * layout.block_bytes + state
            written += min(blocks, layout.written_blocks(ahead)) * layout.block_bytes + state
        return KvNeed(declared, written)

    # ------------------------------------------------------------------ views

    def kv(self, layer: int) -> tuple[Tensor, Tensor]:
        """`layer`'s K and V, each [card_blocks, block_tokens, kv_heads, head_dim]: views over
        the whole reservation, one block every `block_bytes` (they are not contiguous)."""
        return self.blocks[:, layer, 0], self.blocks[:, layer, 1]

    def state(self, index: int) -> Tensor:
        """State tensor `index` of every slot, [max_rows, *shape]."""
        return self._states[index]

    # ------------------------------------------------------------------ rows

    def admit(self, row: int, prompt_tokens: int, *, drafted: bool = False, host_tokens: int = 0) -> Tensor:
        """`row` joins with a prompt of `prompt_tokens`: its blocks, a state slot and, drafting,
        its scratch are mapped, all or none. Its first `host_tokens // block_tokens` blocks (whole
        ones, never its tail) are on the host tier, the rest on the card. Returns the token slot of
        each prompt position (int64 [prompt_tokens]) for `write`. Its state is the prefill's to write."""
        if row in self._held or not 0 <= row < self.max_rows:
            raise ValueError(f"row {row}: already decoding, or not one of {self.max_rows}")
        layout = self.layout
        first = not self._held
        slot = min(set(range(self.max_rows)) - set(self._slots.values()))
        state = [(self._tables + slot * self._slot, self._slot)] if self._slot else []
        if drafted and self._scratch:
            state.append((self._tables + self.max_rows * self._slot + row * self._scratch,
                          self._scratch))  # fmt: skip
        if first:
            state.append((0, self._tables))
        self._state_pages.take(state)
        count = -(-prompt_tokens // layout.block_tokens)
        hosted = min(host_tokens // layout.block_tokens, max(count - 1, 0))
        if hosted and self._host is None:
            raise ValueError(f"{hosted} blocks asked of a host tier this manager has none of")
        try:
            blocks = self._take_host(hosted) + self._take(count - hosted)
        except BaseException:
            self._state_pages.drop(state)
            raise
        if first:
            for table in (self.block_table, self.lengths, self.slot_mapping, self.state_slot):
                table.fill_(-1)
        self._held[row], self._known[row], self._slots[row] = blocks, prompt_tokens, slot
        if drafted:
            self._drafting.add(row)
        if blocks:
            self.block_table[row, : len(blocks)] = self._ints(blocks, torch.int32)
        self.lengths[row] = prompt_tokens
        self.state_slot[row] = slot if self._slot else -1
        at = torch.arange(prompt_tokens, device=self.device)
        chosen = self._ints(blocks, torch.int64)
        mapping: Tensor = (
            chosen[at // layout.block_tokens] * layout.block_tokens + at % layout.block_tokens
        )
        return mapping

    def slot(self, row: int) -> int:
        """`row`'s state slot."""
        return self._slots[row]

    def committed(self, row: int) -> int:
        """The positions of `row` its mapped blocks hold: no step may write at or past it."""
        return len(self._held[row]) * self.layout.block_tokens

    def token_slots(self, row: int, start: int, count: int) -> Tensor:
        """The token slots (int64 `[count]`) of `row`'s positions `start + [0, count)`, all of them
        on mapped blocks (`reserve_ahead` first)."""
        tokens, held = self.layout.block_tokens, self._held[row]
        at = range(start, start + count)
        return self._ints([held[p // tokens] * tokens + p % tokens for p in at], torch.int64)

    def gather(self, layer: int, row: int, start: int, count: int) -> tuple[Tensor, Tensor]:
        """`row`'s K and V of `layer` at positions `start + [0, count)`, each contiguous
        `[count, kv_heads, head_dim]`: what a prefill's attention reads (its blocks mapped), from
        either tier."""
        tokens, held = self.layout.block_tokens, self._held[row]
        first, last = start // tokens, (start + count - 1) // tokens
        chosen = held[first : last + 1]
        at = start - first * tokens
        parts: list[tuple[Tensor, Tensor]] = []
        for on_card, run in _tier_runs(chosen, self.card_blocks):
            if on_card:
                keys, values = self.kv(layer)
                index = self._ints(run, torch.int64)
                parts.append((keys.index_select(0, index), values.index_select(0, index)))
            else:
                pieces = self._from_host(layer, [b - self.card_blocks for b in run])
                parts.append((pieces[:, 0], pieces[:, 1]))
        keys = torch.cat([k for k, _ in parts]) if len(parts) > 1 else parts[0][0]
        values = torch.cat([v for _, v in parts]) if len(parts) > 1 else parts[0][1]
        return keys.flatten(0, 1)[at : at + count], values.flatten(0, 1)[at : at + count]

    def settle(self, row: int, length: int) -> None:
        """The host read `row`'s length back: `reserve_ahead` counts from it."""
        self._known[row] = length

    def reserve_ahead(self, row: int, tokens: int) -> None:
        """Map blocks so `row` can be written `tokens` past the length last settled (a step
        writes at most `depth + 1`, so before queueing one: the steps queued since, and it,
        times that). No step then writes past a mapped page, and the host reads no length."""
        layout, held = self.layout, self._held[row]
        need = min(-(-(self._known[row] + tokens) // layout.block_tokens), self.max_blocks)
        if need > len(held):
            more = self._take(need - len(held))
            self.block_table[row, len(held) : need] = self._ints(more, torch.int32)
            held.extend(more)

    def write(self, layer: int, k: Tensor, v: Tensor, slots: Tensor) -> None:
        """A prefill's K and V for `layer` ([n, kv_heads, head_dim], contiguous in the prompt)
        into their token slots (`admit`'s), on whichever tier each block is."""
        tokens = self.layout.block_tokens
        entry, offset = slots // tokens, slots % tokens
        keys, values = self.kv(layer)
        if self._host is None:
            keys.index_put_((entry, offset), k.to(keys.dtype))
            values.index_put_((entry, offset), v.to(values.dtype))
            return
        card = entry < self.card_blocks
        keys.index_put_((entry[card], offset[card]), k[card].to(keys.dtype))
        values.index_put_((entry[card], offset[card]), v[card].to(values.dtype))
        hosted = sorted({int(e) - self.card_blocks for e in entry[~card].tolist()})
        if not hosted:
            return
        # The host blocks this write touches, read to the card, written there, and copied back.
        pieces = self._from_host(layer, hosted)
        where = torch.searchsorted(self._ints(hosted, torch.int64), entry[~card] - self.card_blocks)
        pieces[:, 0].index_put_((where, offset[~card]), k[~card].to(pieces.dtype))
        pieces[:, 1].index_put_((where, offset[~card]), v[~card].to(pieces.dtype))
        self._to_host(layer, hosted, pieces)

    def free(self, row: int) -> None:
        """`row` leaves: its table entries go, and every page only it held is released behind
        the current stream."""
        held, slot = self._held.pop(row), self._slots.pop(row)
        blocks = [b for b in held if b < self.card_blocks]
        self._give_host([b for b in held if b >= self.card_blocks])
        del self._known[row]
        self.block_table[row].fill_(-1)
        self.lengths[row] = -1
        self.slot_mapping[row].fill_(-1)
        self.state_slot[row] = -1
        self._give(blocks)
        state = [(self._tables + slot * self._slot, self._slot)] if self._slot else []
        if row in self._drafting:
            self._drafting.discard(row)
            if self._scratch:
                state.append((self._tables + self.max_rows * self._slot + row * self._scratch,
                              self._scratch))  # fmt: skip
        if not self._held:
            state.append((0, self._tables))
        self._state_pages.drop(state)

    # ------------------------------------------------------------------ blocks

    def _ints(self, values: Sequence[int], dtype: torch.dtype) -> Tensor:
        return torch.tensor(values, dtype=dtype, device=self.device)

    def _range(self, block: int) -> tuple[int, int]:
        return block * self.layout.block_bytes, self.layout.block_bytes

    def _take(self, count: int) -> list[int]:
        """`count` free blocks, those on mapped pages first, then the lowest; all or none."""
        if count > len(self._free):
            # The card's reservation is full: what the judge held the call's K/V to was exceeded.
            raise RuntimeError(f"{count} K/V blocks asked with {len(self._free)} free on the card")
        spare = heapq.nsmallest(count, self._spare)
        fresh: list[int] = []
        seen: set[int] = set()
        while len(spare) + len(fresh) < count:
            block = heapq.heappop(self._fresh)
            if block in self._free and block not in self._spare and block not in seen:
                fresh.append(block)
                seen.add(block)
        blocks = spare + fresh
        try:
            self._kv_pages.take([self._range(b) for b in blocks])
        except BaseException:
            for block in fresh:
                heapq.heappush(self._fresh, block)
            raise
        self._free.difference_update(blocks)
        self._spare.difference_update(blocks)
        self._respare(fresh)
        return blocks

    def _host_runs(self, layer: int, blocks: Sequence[int]) -> list[tuple[int, int, int]]:
        """Sorted host block ids of `layer` as (first id, count, at): runs of consecutive ids that lie
        in one host registration window each (a copy may not span two), `at` the run's place in
        `blocks`. A block's piece is a whole fraction of a window, so none straddles two."""
        piece = self.layout.block_bytes // self.layout.attn_layers
        assert self._host is not None
        window = self._host.granularity
        runs: list[tuple[int, int, int]] = []
        for i, b in enumerate(blocks):
            starts = ((layer * self.host_blocks + b) * piece) % window == 0
            if runs and runs[-1][0] + runs[-1][1] == b and not starts:
                runs[-1] = (runs[-1][0], runs[-1][1] + 1, runs[-1][2])
            else:
                runs.append((b, 1, i))
        return runs

    def _from_host(self, layer: int, blocks: Sequence[int]) -> Tensor:
        """Host blocks `blocks` (sorted ids) of `layer`, copied to the card: `[n, 2, T, kv_heads, head_dim]`."""
        assert self.host_cpu is not None
        out = torch.empty((len(blocks), *self.host_cpu.shape[2:]), dtype=self.layout.dtype, device=self.device)
        for first, count, at in self._host_runs(layer, blocks):
            out[at : at + count].copy_(self.host_cpu[layer, first : first + count], non_blocking=True)
        return out

    def _to_host(self, layer: int, blocks: Sequence[int], pieces: Tensor) -> None:
        """`pieces` (`_from_host`'s shape) copied back to host blocks `blocks` of `layer`."""
        assert self.host_cpu is not None
        for first, count, at in self._host_runs(layer, blocks):
            self.host_cpu[layer, first : first + count].copy_(pieces[at : at + count], non_blocking=True)

    def host_extent(self) -> int:
        """One past the highest host block id a row holds: what a step's copies move a layer."""
        hosted = [b - self.card_blocks for held in self._held.values() for b in held if b >= self.card_blocks]
        return max(hosted) + 1 if hosted else 0

    def map_windows(self, extent: int) -> None:
        """Both copy-ahead windows mapped for host block ids below `extent`."""
        if extent <= self._window_extent or self._window_pages is None:
            return
        piece = self.layout.block_bytes // self.layout.attn_layers
        ranges = [(slot * self.host_blocks * piece, extent * piece) for slot in (0, 1)]
        self._window_pages.take(ranges)
        self._window_taken += ranges
        self._window_extent = extent

    def _host_range(self, block: int, layer: int) -> tuple[int, int]:
        piece = self.layout.block_bytes // self.layout.attn_layers
        return (layer * self.host_blocks + block) * piece, piece

    def _take_host(self, count: int) -> list[int]:
        """`count` host blocks (ids from `card_blocks`), the lowest free; their windows mapped."""
        if not count:
            return []
        assert self._host_pages is not None
        if count > len(self._host_free):
            raise RuntimeError(f"{count} host K/V blocks asked with {len(self._host_free)} free")
        chosen = heapq.nsmallest(count, self._host_free)
        self._host_pages.take([self._host_range(b, layer) for b in chosen for layer in range(self.layout.attn_layers)])
        taken = set(chosen)
        self._host_free = [b for b in self._host_free if b not in taken]
        heapq.heapify(self._host_free)
        return [self.card_blocks + b for b in chosen]

    def _give_host(self, blocks: Sequence[int]) -> None:
        if not blocks:
            return
        assert self._host_pages is not None
        own = [b - self.card_blocks for b in blocks]
        self._host_pages.drop([self._host_range(b, layer) for b in own for layer in range(self.layout.attn_layers)])
        for b in own:
            heapq.heappush(self._host_free, b)

    def host_bytes(self) -> int:
        """The host tier's bytes held by rows now."""
        return sum(1 for held in self._held.values() for b in held if b >= self.card_blocks) * self.layout.block_bytes

    def _give(self, blocks: Sequence[int]) -> None:
        self._kv_pages.drop([self._range(b) for b in blocks])
        self._free.update(blocks)
        self._respare(blocks)
        for block in blocks:
            if block not in self._spare:
                heapq.heappush(self._fresh, block)

    def _respare(self, near: Sequence[int]) -> None:
        """Bring `_spare` up to date around `near`: a free block is spare while every page it
        lies on is still mapped by a held one."""
        reach = max(self._kv_pages.page // self.layout.block_bytes, 1) + 1
        for b in near:
            for other in range(max(b - reach, 0), min(b + reach + 1, self.card_blocks)):
                if other in self._free and self._kv_pages.mapped(*self._range(other)):
                    self._spare.add(other)
                elif other in self._spare:
                    self._spare.discard(other)
                    if other in self._free:
                        heapq.heappush(self._fresh, other)

    def close(self) -> None:
        """Return both reservations: with no row decoding and every released page passed."""
        if self._held:
            raise RuntimeError(f"{len(self._held)} rows are still decoding")
        del self.block_table, self.lengths, self.slot_mapping, self.state_slot, self.blocks
        self._states.clear()
        self.state_scratch.clear()
        self._kv_pages.drop([(0, self.layout.block_bytes)])
        self._kv.close()
        self._state.close()
        if self._host is not None and self._host_pages is not None:
            piece = self.layout.block_bytes // self.layout.attn_layers
            del self.host_view, self.host_cpu, self.windows
            self._host_pages.drop([(0, piece)])
            self._host.close()
            if self._window is not None and self._window_pages is not None:
                self._window_pages.drop(self._window_taken)
                self._window.close()


def _tier_runs(blocks: Sequence[int], card_blocks: int) -> list[tuple[bool, list[int]]]:
    """`blocks` in order as runs on one tier each: (on the card, the run)."""
    runs: list[tuple[bool, list[int]]] = []
    for b in blocks:
        on_card = b < card_blocks
        if runs and runs[-1][0] == on_card:
            runs[-1][1].append(b)
        else:
            runs.append((on_card, [b]))
    return runs


class CopyAhead:
    """A step's reads of host blocks (cr-336's decision): before each attention layer reads its
    off-card blocks, they were copied into one of two windows on the card, on a copy stream while the
    layer before computed; a window is refilled only once the attention that read it is done.
    `begin` takes the step's attention layers in the order it reads them; `tier(layer)` is what that
    layer's attention reads its off-card blocks from, `done()` follows each, and `end()` joins the copy
    stream back. Captured into a step's graph, every copy is a node of it: the graph is recorded for
    one `extent` (host block ids in use), so the decoder keys its graphs by it. `in_place` reads the
    host blocks where they are instead (the fallback where the device addresses them)."""

    def __init__(self, kv: KvManager, *, in_place: bool = False) -> None:
        self.kv, self.in_place = kv, in_place
        self.copier = torch.cuda.Stream(kv.device)
        self.order: list[int] = []
        self.at = 0
        self.copied: dict[int, torch.cuda.Event] = {}
        self.read: list[torch.cuda.Event | None] = [None, None]

    def begin(self, order: Sequence[int]) -> None:
        kv = self.kv
        extent = kv.host_extent() if not self.in_place else 0
        self.order = list(order) if extent else []
        self.extent, self.at, self.copied, self.read = extent, 0, {}, [None, None]
        if not self.order:
            return
        kv.map_windows(extent)
        self.copier.wait_stream(torch.cuda.current_stream(kv.device))
        self._issue(0)
        self._issue(1)

    def _issue(self, i: int) -> None:
        if i >= len(self.order):
            return
        kv, slot = self.kv, i % 2
        assert kv.windows is not None and kv.host_cpu is not None
        with torch.cuda.stream(self.copier):
            if (last := self.read[slot]) is not None:
                self.copier.wait_event(last)
            layer = self.order[i]
            for first, count, _ in kv._host_runs(layer, range(self.extent)):
                kv.windows[slot, first : first + count].copy_(kv.host_cpu[layer, first : first + count], non_blocking=True)
            event = torch.cuda.Event()
            event.record(self.copier)
            self.copied[i] = event

    def tier(self, layer: int) -> HostTier:
        if not self.order:
            return HostTier(self.kv.host_view, layer)
        assert self.order[self.at] == layer, f"attention read layer {layer}, the step declared {self.order[self.at]}"
        assert self.kv.windows is not None
        torch.cuda.current_stream(self.kv.device).wait_event(self.copied[self.at])
        slot = self.at % 2
        return HostTier(self.kv.windows[slot : slot + 1], 0)

    def done(self) -> None:
        if not self.order:
            return
        event = torch.cuda.Event()
        event.record(torch.cuda.current_stream(self.kv.device))
        self.read[self.at % 2] = event
        self.at += 1
        self._issue(self.at + 1)

    def end(self) -> None:
        if self.order:
            torch.cuda.current_stream(self.kv.device).wait_stream(self.copier)
        self.order = []
