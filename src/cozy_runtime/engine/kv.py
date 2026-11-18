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
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from math import prod
from typing import TYPE_CHECKING

import torch

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
        self, gpu: varena.Device, layout: KvLayout, max_rows: int, max_blocks: int
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

    def admit(self, row: int, prompt_tokens: int, *, drafted: bool = False) -> Tensor:
        """`row` joins with a prompt of `prompt_tokens`: its blocks, a state slot and, drafting,
        its scratch are mapped, all or none. Returns the token slot of each prompt position
        (int64 [prompt_tokens]) for `write`. Its state is the prefill's to write."""
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
        try:
            blocks = self._take(-(-prompt_tokens // layout.block_tokens))
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
        into their token slots (`admit`'s)."""
        tokens = self.layout.block_tokens
        where = (slots // tokens, slots % tokens)
        keys, values = self.kv(layer)
        keys.index_put_(where, k.to(keys.dtype))
        values.index_put_(where, v.to(values.dtype))

    def free(self, row: int) -> None:
        """`row` leaves: its table entries go, and every page only it held is released behind
        the current stream."""
        blocks, slot = self._held.pop(row), self._slots.pop(row)
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
