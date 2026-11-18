"""The engine's decode step for one bucket of rows (xs-213 phase 2): every input it reads and
everything it writes are device tensors at fixed addresses, so it is captured once as a graph and
run as the body of the launcher's loop.

A plain step decodes a row's token. A drafted step (depth 2) drafts two tokens with the model's
multi-token-prediction head, verifies the row's token and both drafts in one pass of the model
(one attention call a layer for the span), keeps each row's tokens up to its first disagreement
or its stop, writes the next step's token, the head's state and the kept recurrent state, and
reports what it made. Whether another step follows is the step's own answer (`go`): some row is
still decoding, and no row's next step would write past the K/V its host committed (`budget`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from cozy_runtime.engine.qwen35 import Tables

if TYPE_CHECKING:
    from torch import Tensor

    from cozy_runtime.engine.qwen35 import Qwen35
    from cozy_runtime.engine.sampling import Sampling


@dataclass(frozen=True)
class Rows:
    """Each row's decode state, `[max_rows, ...]` on the device; a bucket of `rows` reads the
    first `rows`. The host writes them only when a row joins or leaves, between launches."""

    #: the token each row feeds next (at its length), int64
    ids: Tensor
    #: tokens the row has made, int32
    made: Tensor
    #: the most tokens it may make, int32
    limit: Tensor
    #: it stopped (a stop token or its limit), bool
    done: Tensor
    #: the position below which its K/V pages are committed, int32: no step writes at or past it
    budget: Tensor
    #: Qwen-VL's text position offset (rope delta), int32
    delta: Tensor
    #: its stop tokens, int64 `[max_rows, stops]`, -1 padding
    stops: Tensor
    #: the tokens it has seen, for the presence penalty, f32 `[max_rows, vocab]`
    seen: Tensor
    #: the head's state for its next first draft, `[max_rows, hidden]`
    hidden: Tensor
    #: what each step made, int32 `[max_rows, 1 + span]`: the count (-1: not decoding), the tokens
    report: Tensor
    #: whether another step follows, int32 `[1]`
    go: Tensor


def rows_for(max_rows: int, span: int, hidden: int, vocab: int, stops: int, device: torch.device) -> Rows:
    def z(shape: tuple[int, ...], dtype: torch.dtype) -> Tensor:
        return torch.zeros(shape, dtype=dtype, device=device)

    return Rows(
        ids=z((max_rows,), torch.int64), made=z((max_rows,), torch.int32),
        limit=z((max_rows,), torch.int32), done=torch.ones(max_rows, dtype=torch.bool, device=device),
        budget=z((max_rows,), torch.int32), delta=z((max_rows,), torch.int32),
        stops=torch.full((max_rows, stops), -1, dtype=torch.int64, device=device),
        seen=z((max_rows, vocab), torch.float32), hidden=z((max_rows, hidden), torch.bfloat16),
        report=torch.full((max_rows, 1 + span), -1, dtype=torch.int32, device=device),
        go=z((1,), torch.int32),
    )  # fmt: skip


class Step:
    """The step of a bucket of `rows`, drafted or plain, over `model` and its `KvManager`."""

    def __init__(self, model: Qwen35, rows: Rows, sampling: Sampling, size: int, drafted: bool) -> None:
        self.model, self.rows, self.sampling, self.size = model, rows, sampling, size
        self.drafted = drafted
        kv = model.kv
        self.span = kv.layout.span if drafted else 1
        device = kv.device
        self.steps = torch.arange(self.span, device=device)
        #: the draft head's tables: a position ahead of the body's, seeing from 1
        self.mtp_lengths = torch.full((size,), -1, dtype=torch.int32, device=device)
        self.mtp_slots = torch.full((size, kv.layout.span), -1, dtype=torch.int64, device=device)
        self.live_lengths = torch.full((size,), -1, dtype=torch.int32, device=device)

    # ------------------------------------------------------------------ pieces

    def _slots(self, start: Tensor, count: int, live: Tensor) -> Tensor:
        """Each row's token slot at positions `start + [0, count)` (block entry * T + offset),
        -1 where the row isn't live."""
        kv = self.model.kv
        tokens = kv.layout.block_tokens
        at = start.long()[:, None] + self.steps[:count]
        index = (at // tokens).clamp(0, kv.max_blocks - 1)
        entry = kv.block_table[: self.size].gather(1, index).long()
        return torch.where(live[:, None] & (entry >= 0), entry * tokens + at % tokens, -1)

    def _scores(self, logits: Tensor, ahead: Tensor | None = None) -> Tensor:
        """Logits after `top_k` and the presence penalty (`ahead`: what a span's earlier
        positions saw, `[rows, span, vocab]`)."""
        r, s = self.rows, self.size
        scores = self.sampling.keep(logits, s)
        seen = r.seen[:s]
        if ahead is None:
            return scores - self.sampling.penalty[:s, None] * seen
        return scores - self.sampling.penalty[:s, None, None] * torch.maximum(seen[:, None], ahead)

    # ------------------------------------------------------------------ the step

    def __call__(self) -> None:
        if self.drafted:
            self._drafted()
        else:
            self._plain()

    def _plain(self) -> None:
        m, r, s, kv = self.model, self.rows, self.size, self.model.kv
        length = kv.lengths[:s]
        live = (length >= 0) & ~r.done[:s]
        self.live_lengths.copy_(torch.where(live, length, -1))
        kv.slot_mapping[:s, :1].copy_(self._slots(length, 1, live))
        positions = (length + r.delta[:s])[:, None]
        y = m.body(r.ids[:s, None], positions, Tables(self.live_lengths, kv.slot_mapping[:s]))
        scores = self._scores(m.logits(y[:, 0]))
        token = torch.where(live, self.sampling.pick(scores, s, r.made[:s].long()), r.ids[:s])
        made = r.made[:s]
        made.add_(live.int())
        stopped = (token[:, None] == r.stops[:s]).any(-1) | (made >= r.limit[:s])
        r.done[:s] |= stopped & live
        r.ids[:s].copy_(token)
        r.seen[:s].scatter_reduce_(1, token[:, None], live.float()[:, None], "amax")
        length.copy_(torch.where(live, length + 1, length))
        r.report[:s, 0].copy_(torch.where(live, 1, -1))
        r.report[:s, 1].copy_(token.int())
        self._go(length, 1)

    def _drafted(self) -> None:
        m, r, s, kv, span = self.model, self.rows, self.size, self.model.kv, self.span
        depth, steps = span - 1, self.steps
        length = kv.lengths[:s]
        live = (length >= 0) & ~r.done[:s]
        n = r.made[:s].long()
        delta = r.delta[:s]
        hidden = r.hidden[:s]
        self.live_lengths.copy_(torch.where(live, length, -1))
        self.mtp_lengths.copy_(torch.where(live, length + 1, -1))
        # Draft 0 from the head's state the last step left; draft 1 from the head at position L + 1.
        drafts = [self.sampling.pick(self._scores(m.logits(hidden)), s, n)]
        self.mtp_slots[:, :1].copy_(self._slots(length + 1, 1, live))
        state = hidden[:, None]
        for i in range(1, depth):
            at = (length + i + delta)[:, None]
            state = m.draft(drafts[-1][:, None], state, at, Tables(self.mtp_lengths, self.mtp_slots, 1))
            seen = r.seen[:s].scatter(1, torch.stack(drafts, 1), 1.0)
            scores = self.sampling.keep(m.logits(state[:, 0]), s) - self.sampling.penalty[:s, None] * seen
            drafts.append(self.sampling.pick(scores, s, n + i))
            if i + 1 < depth:
                self.mtp_lengths.copy_(torch.where(live, length + i + 1, -1))
                self.mtp_slots[:, :1].copy_(self._slots(length + i + 1, 1, live))
        drafted = torch.stack(drafts, 1)
        # Verify the row's token and its drafts in one pass.
        kv.slot_mapping[:s, :span].copy_(self._slots(length, span, live))
        ids = torch.cat([r.ids[:s, None], drafted], 1)
        y = m.body(ids, (length + delta)[:, None] + steps, Tables(self.live_lengths, kv.slot_mapping[:s]))
        logits = m.logits(y)
        ahead = torch.zeros_like(logits).scatter_(2, drafted[..., None], 1.0).cumsum(1)
        ahead = torch.cat([torch.zeros_like(ahead[:, :1]), ahead[:, :-1]], 1).clamp(max=1)
        best = self.sampling.pick(self._scores(logits, ahead), s, n[:, None] + steps)
        agreed = (best[:, :depth] == drafted).long().cumprod(-1).sum(-1)
        valid = (steps[None] <= agreed[:, None]) & live[:, None]
        limit = (n[:, None] + steps[None] + 1) >= r.limit[:s, None]
        stopped = ((best[..., None] == r.stops[:s, None]).any(-1) | limit) & valid
        after = (stopped.long().cumsum(-1) - stopped.long()) > 0
        keep = valid & ~after
        made = keep.long().sum(-1)
        last = (made - 1).clamp(min=0)
        token = best.gather(1, last[:, None])[:, 0]
        r.made[:s].add_(made.int())
        r.done[:s] |= stopped.any(-1)
        r.seen[:s].scatter_reduce_(1, best, keep.float(), "amax")
        r.ids[:s].copy_(torch.where(live, token, r.ids[:s]))
        # The head's K/V for the kept tokens, from the model's state before each, and its state
        # after them: the next step's first draft.
        self.mtp_lengths.copy_(torch.where(live, length + 1, -1))
        self.mtp_slots[:, :span].copy_(self._slots(length + 1, span, live))
        tokens = torch.where(keep, best, r.ids[:s, None])
        states = m.draft(tokens, y, (length + 1 + delta)[:, None] + steps, Tables(self.mtp_lengths, self.mtp_slots, 1))
        kept_state = states.gather(1, last[:, None, None].expand(-1, 1, states.shape[-1]))[:, 0]
        r.hidden[:s].copy_(torch.where(live[:, None], kept_state, hidden))
        m.keep(torch.where(live, last, -1).int())
        length.copy_(torch.where(live, length + made.int(), length))
        r.report[:s, 0].copy_(torch.where(live, made.int(), -1))
        r.report[:s, 1 : 1 + span].copy_(torch.where(keep, best, -1).int())
        self._go(length, span + 1)

    def _go(self, length: Tensor, reach: int) -> None:
        """Another step follows while a row decodes and none's next step would write at or past
        its committed `budget` (`reach`: the positions past its length a step writes)."""
        r, s = self.rows, self.size
        running = (length >= 0) & ~r.done[:s]
        room = (length + reach <= r.budget[:s]) | ~running
        r.go.copy_((running.any() & room.all()).int().view(1))
