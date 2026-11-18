"""The engine's decoder for one loaded Qwen3.5 model on its GPU (xs-213 phase 2): rows prefilled
by the model's own forward and moved into the engine's K/V blocks and state slots, then decoded
by one captured step a bucket of rows, replayed until every row stops.

Bring-up form: the host replays the step's graph `SEGMENT` times between reads of what it made
(the launcher's WHILE-conditional segment, `_kernels._engine`, takes this loop's place), and a
row's prompt is prefilled alone, text only. The step itself is the engine's whole decode.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Protocol, cast

import torch

from cozy_runtime.engine.attention import HostTier
from cozy_runtime.engine.kv import KvManager
from cozy_runtime.engine.qwen35 import Qwen35, layout_of, text_of
from cozy_runtime.engine.sampling import Sampling
from cozy_runtime.engine.step import Rows, Step, rows_for

class _TextConfig(Protocol):
    vocab_size: int


class _Config(Protocol):
    def get_text_config(self, decoder: bool = ...) -> _TextConfig: ...


class _Out(Protocol):
    last_hidden_state: torch.Tensor


class _Rotary(Protocol):
    def __call__(self, x: torch.Tensor, positions: torch.Tensor, /) -> tuple[torch.Tensor, torch.Tensor]: ...


class _Decoder(Protocol):
    rotary_emb: _Rotary

    def __call__(self, *, input_ids: torch.Tensor, past_key_values: object, use_cache: bool) -> _Out: ...


class _Call(Protocol):
    def __call__(self, x: torch.Tensor, /) -> torch.Tensor: ...


class _Weighted(Protocol):
    weight: torch.Tensor

    def __call__(self, x: torch.Tensor, /) -> torch.Tensor: ...


class _HeadLayer(Protocol):
    def __call__(
        self, x: torch.Tensor, *, position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None, past_key_values: object,
    ) -> torch.Tensor: ...  # fmt: skip


class _Head(Protocol):
    fc: _Call
    pre_fc_norm_embedding: _Weighted
    pre_fc_norm_hidden: _Weighted
    layers: Sequence[_HeadLayer]
    norm: _Weighted


class _Model(Protocol):
    """A loaded Qwen3.5 Transformers model, as the prefill reads it."""

    config: _Config
    mtp: _Head

    def get_decoder(self) -> _Decoder: ...


class _Binding(Protocol):
    """A placement's retained weights (`Residency.retain`)."""

    identity: object

    def valid(self) -> bool: ...

    def prepare(self) -> contextlib.AbstractContextManager[object]: ...


class _Keys(Protocol):
    keys: torch.Tensor
    values: torch.Tensor


class _States(Protocol):
    conv_states: dict[int, torch.Tensor]
    recurrent_states: dict[int, torch.Tensor]


#: Replays of a step between two reads of what it made (bring-up; the launcher runs until done).
SEGMENT = 16
#: Row buckets: a call's rows take the smallest that holds them.
BUCKETS = (1, 2, 4, 8, 16)
#: Stop tokens a row holds.
STOPS = 8


@dataclass(frozen=True)
class Request:
    """One call's rows and decoding, as token ids."""

    prompts: Sequence[Sequence[int]]
    max_new: int
    stops: Sequence[int]
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 0
    min_p: float = 0.0
    presence: float = 0.0
    frequency: float = 0.0
    repetition: float = 1.0
    seed: int = 0


@dataclass
class Made:
    """What a call made: each row's tokens and the times it took."""

    tokens: list[list[int]]
    #: each row's tokens kept a decode step (1 a plain step; 1 to depth + 1 a drafted one)
    kept: list[list[int]] = field(default_factory=list)
    prefill_s: float = 0.0
    decode_s: float = 0.0
    steps: int = 0
    captured: bool = False
    notes: list[str] = field(default_factory=list)


class Decoder:
    """`model`'s engine on its GPU, for at most `max_rows` rows of at most `max_tokens` tokens."""

    def __init__(self, model: torch.nn.Module, *, max_rows: int = 16, max_tokens: int = 32768) -> None:
        from tensorfs.varena import _runtime as varena

        self.model = model
        self.text = text_of(model)
        self.drafting = getattr(model, "mtp", None) is not None
        device = next(model.parameters()).device
        self.device = device
        self.stream = torch.cuda.Stream(device)
        layout = layout_of(self.text, depth=2, drafting=self.drafting)
        self.max_rows = max_rows
        with torch.cuda.stream(self.stream):
            self.kv = KvManager(varena.Device(device.index or 0), layout, max_rows, -(-max_tokens // layout.block_tokens))
        nothing = torch.zeros((1, 1, 2, layout.block_tokens, layout.kv_heads, layout.head_dim),
                              dtype=layout.dtype, device=device)  # fmt: skip
        self.qwen = Qwen35(model, self.kv, HostTier(nothing, 0))
        self.hf = cast("_Model", model)
        vocab = int(self.hf.config.get_text_config(decoder=True).vocab_size)
        self.rows: Rows = rows_for(max_rows, layout.span, self.text.hidden, vocab, STOPS, device)

        def full(dtype: torch.dtype) -> torch.Tensor:
            return torch.zeros(max_rows, dtype=dtype, device=device)

        #: the rows' decoding, at fixed addresses a graph holds; each call fills them
        self.decoding = tuple(full(t) for t in (torch.float32, torch.float32, torch.int64, torch.float32,
                                                    torch.float32, torch.float32, torch.float32, torch.int64))  # fmt: skip
        #: a bucket's step and its graph, by (rows, drafted, sampled, top_k kept)
        self.steps: dict[tuple[int, bool, bool, int], tuple[Step, torch.cuda.CUDAGraph | None]] = {}
        #: the step's weights, retained from the model's placement where it has one, and the
        #: identity the graphs above were recorded against
        self.binding: object | None = None
        self.recorded: object | None = None

    # ------------------------------------------------------------------ a call

    def __call__(self, request: Request, told: Callable[[int, list[int]], None] | None = None) -> Made:
        import time

        rows = len(request.prompts)
        if not 0 < rows <= self.max_rows:
            raise ValueError(f"{rows} rows: this decoder holds 1 to {self.max_rows}")
        size = next(b for b in BUCKETS if b >= rows) if rows <= BUCKETS[-1] else rows
        sampling = self._sampling(request, size)
        drafted = self.drafting
        made = Made(tokens=[[] for _ in range(rows)], kept=[[] for _ in range(rows)])
        with torch.cuda.stream(self.stream), torch.inference_mode():
            began = time.perf_counter()
            for row, prompt in enumerate(request.prompts):
                made.tokens[row].append(self._prefill(row, prompt, request, sampling, drafted))
            self.stream.synchronize()
            made.prefill_s = time.perf_counter() - began
            if told is not None:
                for row in range(rows):
                    told(row, made.tokens[row][-1:])
            began = time.perf_counter()
            try:
                self._decode(size, rows, drafted, sampling, request, made, told)
            finally:
                for row in range(rows):
                    self.kv.free(row)
                r = self.rows
                r.done.fill_(True)
                self.stream.synchronize()
            made.decode_s = time.perf_counter() - began
        return made

    def _sampling(self, request: Request, size: int) -> Sampling:
        sampled = request.temperature > 0
        # Every row of a call samples alike, so a top-k bounds every sampling row's draw.
        kept = request.top_k if sampled and request.top_k > 0 else 0
        penalized = bool(request.presence or request.frequency or request.repetition != 1.0)
        values = (request.temperature, request.top_p, request.top_k if sampled else 0, request.min_p,
                  request.presence, request.frequency, request.repetition, request.seed)  # fmt: skip
        for tensor, value in zip(self.decoding, values, strict=True):
            tensor.fill_(value)
        temperature, top_p, top_k, min_p, presence, frequency, repetition, seed = self.decoding
        return Sampling(temperature, top_p, top_k, min_p, presence, frequency, repetition, seed,
                        kept=kept, sampled=sampled, penalized=penalized)  # fmt: skip

    # ------------------------------------------------------------------ prefill

    def _prefill(self, row: int, prompt: Sequence[int], request: Request, sampling: Sampling, drafted: bool) -> int:
        """`row`'s prompt through the model's own forward: its K/V into its blocks, its states
        into its slot, the head's K/V for positions 1 to P and its state for the first draft;
        returns the first token."""
        from transformers import DynamicCache

        hf, kv, qwen, r = self.hf, self.kv, self.qwen, self.rows
        ids = torch.tensor([list(prompt)], dtype=torch.long, device=self.device)
        length = ids.shape[1]
        cache = DynamicCache(config=hf.config.get_text_config(decoder=True))  # type: ignore[arg-type]  # its own config type
        out = hf.get_decoder()(input_ids=ids, past_key_values=cache, use_cache=True)
        y = out.last_hidden_state[0]  # [P, hidden], final-normed
        r.counts[row].zero_()
        r.prompt[row].zero_()
        r.prompt[row, ids[0]] = True
        logits = qwen.logits(y[-1:])
        scores = sampling.penalties(logits, 1, r.counts[row : row + 1], r.prompt[row : row + 1])
        first = int(sampling.pick(scores, 1, torch.zeros(1, dtype=torch.long, device=self.device))[0])
        slots = kv.admit(row, length, drafted=drafted)
        for layer, index in qwen.slot_of.items():
            keys = cast("_Keys", cache.layers[layer])
            kv.write(index, keys.keys[0].transpose(0, 1), keys.values[0].transpose(0, 1), slots)
        slot = kv.slot(row)
        for layer, index in qwen.delta_of.items():
            states = cast("_States", cache.layers[layer])
            kv.state(0)[slot, index].copy_(states.conv_states[0][0, :, 1:])
            kv.state(1)[slot, index].copy_(states.recurrent_states[0][0])
        reach = kv.layout.span + 1
        kv.reserve_ahead(row, (SEGMENT + 1) * reach)
        if drafted:
            hidden = self._draft_prompt(row, ids[0], first, y)
            r.hidden[row].copy_(hidden)
        r.ids[row] = first
        r.made[row] = 1
        r.limit[row] = request.max_new
        stops = list(request.stops)[:STOPS]
        r.stops[row].fill_(-1)
        if stops:
            r.stops[row, : len(stops)] = torch.tensor(stops, dtype=torch.long, device=self.device)
        r.done[row] = first in stops or request.max_new <= 1
        r.delta[row] = 0
        r.counts[row, first] += 1.0
        r.budget[row] = kv.committed(row)
        return first

    def _draft_prompt(self, row: int, ids: torch.Tensor, first: int, y: torch.Tensor) -> torch.Tensor:
        """The head's K/V for positions 1 to P (token p from the model's state at p - 1) and its
        state at P, from which the first step drafts: `lm`'s `_drafted_prompt`, one row."""
        from transformers import DynamicCache

        hf, kv, qwen, text = self.hf, self.kv, self.qwen, self.text
        mtp = hf.mtp
        length = int(ids.shape[0])
        tokens = torch.cat([ids[1:], torch.tensor([first], dtype=torch.long, device=self.device)])
        eps = text.eps
        from cozy_runtime.engine.qwen35 import _norm

        e = _norm(qwen.embed(tokens[None]), mtp.pre_fc_norm_embedding.weight, eps)
        h = _norm(y[None], mtp.pre_fc_norm_hidden.weight, eps)
        x = mtp.fc(torch.cat([e, h], -1))
        positions = torch.arange(1, length + 1, device=self.device)
        rope = hf.get_decoder().rotary_emb(x, positions.view(1, 1, -1).expand(3, 1, -1))
        cache = DynamicCache()
        x = mtp.layers[0](x, position_embeddings=rope, attention_mask=None, past_key_values=cache)
        held = cast("_Keys", cache.layers[0])
        slots = kv.token_slots(row, 1, length)
        mine = kv.layout.attn_layers - 1
        kv.write(mine, held.keys[0].transpose(0, 1), held.values[0].transpose(0, 1), slots)
        return mtp.norm(x)[0, -1]

    # ------------------------------------------------------------------ weights

    @contextmanager
    def _leased(self) -> Iterator[None]:
        """The weights a step reads (the language model, its head and, drafting, the draft head)
        held where they are while a graph records or replays: varena moves no Region a graph
        holds the address of. A model with no placement (a bench's plain tensors) holds nothing.
        Where the binding's identity changed since the graphs were recorded, they are dropped."""
        from cozy_runtime.author._model import PLACED

        placed = PLACED.get(self.model)
        if placed is None:
            yield
            return
        residency, name = placed
        binding = cast("_Binding | None", self.binding)
        if binding is None or not binding.valid():
            model = self.model
            reads = (model.get_decoder(), model.get_output_embeddings(), getattr(model, "mtp", None))  # type: ignore[operator]  # Transformers'
            wanted = {id(m) for m in reads if m is not None}
            paths = tuple(n for n, m in model.named_modules() if n and id(m) in wanted) or ("",)
            binding = cast("_Binding", residency.retain(name, paths))
            self.binding = binding
        with binding.prepare():
            if self.recorded is not None and self.recorded != binding.identity:
                self.steps.clear()
            self.recorded = binding.identity
            yield

    # ------------------------------------------------------------------ decode

    def _decode(
        self,
        size: int,
        rows: int,
        drafted: bool,
        sampling: Sampling,
        request: Request,
        made: Made,
        told: Callable[[int, list[int]], None] | None,
    ) -> None:
        kv, r = self.kv, self.rows
        key = (size, drafted, sampling.sampled, sampling.kept)
        step, graph = self.steps.get(key) or (Step(self.qwen, r, sampling, size, drafted), None)
        reach = (step.span + 1) if drafted else 1
        ring = torch.empty((SEGMENT, *r.report[:size].shape), dtype=torch.int32, device=self.device)
        live = [True] * rows
        while any(live):
            for row in range(rows):  # room for a whole segment: no step writes past a mapped page
                if live[row]:
                    kv.settle(row, int(kv.lengths[row]))
                    kv.reserve_ahead(row, (SEGMENT + 1) * reach)
                    r.budget[row] = kv.committed(row)
            with self._leased():
                if key not in self.steps and graph is not None:
                    graph = None  # recorded against weights that have since moved
                for k in range(SEGMENT):
                    if graph is not None:
                        graph.replay()
                    elif made.steps and key not in self.steps:  # the second step is recorded, then run
                        graph = self._capture(step, made)
                        self.steps[key] = (step, graph)
                        graph.replay() if graph is not None else step()
                    else:
                        step()
                    ring[k].copy_(r.report[:size])
                    made.steps += 1
            self.stream.synchronize()
            for k, report in enumerate(ring.cpu().tolist()):
                for row in range(rows):
                    count, *tokens = report[row]
                    if count > 0 and live[row]:
                        made.tokens[row].extend(tokens[:count])
                        made.kept[row].append(count)
                        if told is not None:
                            told(row, tokens[:count])
                del k
            live = [not d for d in r.done[:rows].tolist()]

    def _capture(self, step: Step, made: Made) -> torch.cuda.CUDAGraph | None:
        """The step's graph, recorded once into a pool of the model's graph domain (the pool
        grows by measure where it runs short); None, with why in `made.notes`, where it can't be."""
        import tensorfs.varena.torch as vt

        from cozy_runtime.graphs import capture, capture_domain
        from cozy_runtime.graphs.domain import Pool

        domain = capture_domain(torch, self.model)
        try:
            domain.initialize()
            domain.prime_blas()
            graph, why = capture(torch, step, Pool(torch.cuda.graph_pool_handle()), domain, vt.pool_bytes(64 << 20), alone=True)
        except Exception as exc:  # reported, and the step runs eagerly
            graph, why = None, f"{type(exc).__name__}: {exc}"
        if graph is None:
            made.notes.append(f"step not captured: {why}")
        made.captured = graph is not None
        return graph
