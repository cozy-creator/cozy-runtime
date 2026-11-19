"""The engine's decoder for one loaded Qwen3.5 model on its GPU (xs-213 phase 2): rows prefilled
by the model's own forward and moved into the engine's K/V blocks and state slots, then decoded
by one captured step a bucket of rows, replayed until every row stops.

The step's graph runs as the body of the launcher's WHILE node (`_kernels._engine.Segment`): one
launch runs steps until the step itself says stop (every row done, or a row's next step would
write past the K/V its host committed), each step reporting what it made into a host ring the
host reads `POLLS` times a second. Bring-up: a row's prompt is prefilled alone, text only.

A call takes the step's weights (`_leased`) and every row's K/V at its start and holds them to its
end. Refused room there, before anything ran, it raises `Short`: its caller serves the call
another way.
"""

from __future__ import annotations

import contextlib
import functools
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, cast

import torch

from cozy_runtime.engine.kv import KvManager
from cozy_runtime.engine.launch import JIT, Run
from cozy_runtime.engine.qwen35 import Qwen35, layout_of, text_of
from cozy_runtime.engine.sampling import Sampling
from cozy_runtime.engine.step import Rows, Step, rows_for

if TYPE_CHECKING:
    from tensorfs.varena import _runtime as varena
    from transformers.cache_utils import CacheLayerMixin

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


#: Replays of a step between two reads of what it made, where the host replays it (`launcher=False`).
SEGMENT = 16
#: The most steps one launch runs: the K/V a row commits ahead covers them, and the ring holds their reports.
LAUNCH_STEPS = 1024
#: Reads of the ring a second while a launch runs and a caller is told tokens as they come.
POLLS = 10
#: Row buckets: a call's rows take the smallest that holds them.
BUCKETS = (1, 2, 4, 8, 16)
#: Stop tokens a row holds.
STOPS = 8
#: A prompt's tokens a forward of the prefill takes at a time: a multiple of the gated-delta rule's
#: 64-token chunks, so each piece computes as in a whole-prompt call; fixed, so a capped and an
#: uncapped call cut a prompt alike.
PREFILL_CHUNK = 2048


class Short(Exception):
    """A call refused room at its start, before anything ran: the step's weights or its rows' K/V
    don't fit beside what the card holds now. Nothing was told and nothing is held."""


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
    """`model`'s engine on its GPU, for at most `max_rows` rows of at most `max_tokens` tokens (by
    default the model's context length: the K/V reservations are address space, mapped as rows grow)."""

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        max_rows: int = 16,
        max_tokens: int | None = None,
        launcher: bool = True,
        fused: bool = True,
        prefill_chunk: int = PREFILL_CHUNK,
        kv_card_bytes: int | None = None,
        in_place: bool = False,
        run: Run = JIT,
    ) -> None:
        from tensorfs.varena import _runtime as varena

        from cozy_runtime._kernels import NO_ENGINE

        self.model = model
        self.text = text_of(model)
        self.drafting = getattr(model, "mtp", None) is not None
        device = next(model.parameters()).device
        self.device = device
        self.stream = torch.cuda.Stream(device)
        layout = layout_of(self.text, depth=2, drafting=self.drafting)
        self.max_rows = max_rows
        if max_tokens is None:
            text_config = cast("_Model", model).config.get_text_config(decoder=True)
            max_tokens = int(getattr(text_config, "max_position_embeddings", 0) or 32768)
        #: the card's share of a call's K/V, in bytes; the rest goes to the host tier (None: all on the card)
        self.kv_card_bytes = kv_card_bytes
        host = _host_tier(model, device.index or 0) if kv_card_bytes is not None else None
        with torch.cuda.stream(self.stream):
            self.kv = KvManager(varena.Device(device.index or 0), layout, max_rows,
                                -(-max_tokens // layout.block_tokens), host=host)  # fmt: skip
        self.qwen = Qwen35(model, self.kv, fused=fused, in_place=in_place, run=run)
        if prefill_chunk <= 0 or prefill_chunk % 64:
            raise ValueError(f"a prefill chunk of {prefill_chunk} tokens: a multiple of the gated-delta rule's 64")
        #: a prompt's tokens a prefill forward takes at a time
        self.prefill_chunk = prefill_chunk
        self.hf = cast("_Model", model)
        vocab = int(self.hf.config.get_text_config(decoder=True).vocab_size)
        self.rows: Rows = rows_for(max_rows, layout.span, self.text.hidden, vocab, STOPS, device)

        def full(dtype: torch.dtype) -> torch.Tensor:
            return torch.zeros(max_rows, dtype=dtype, device=device)

        #: the rows' decoding, at fixed addresses a graph holds; each call fills them
        self.decoding = tuple(full(t) for t in (torch.float32, torch.float32, torch.int64, torch.float32,
                                                    torch.float32, torch.float32, torch.float32, torch.int64))  # fmt: skip
        #: a bucket's step and its graph, by (rows, drafted, sampled, host extent)
        self.steps: dict[tuple[int, bool, bool, int], tuple[Step, torch.cuda.CUDAGraph | None]] = {}
        #: the step's weights, retained from the model's placement where it has one, and the
        #: identity the graphs above were recorded against
        self.binding: object | None = None
        self.recorded: object | None = None
        #: steps run by the launcher's WHILE node, else replayed by the host `SEGMENT` at a time
        self.launcher = launcher and not NO_ENGINE
        self.ring = _Ring(LAUNCH_STEPS + 2 * SEGMENT, self.rows.report.nbytes) if self.launcher else None
        #: a bucket's launcher over its step's graph, by the key of `steps`
        self.segments: dict[tuple[int, bool, bool, int], object] = {}

    # ------------------------------------------------------------------ a call

    def __call__(self, request: Request, told: Callable[[int, list[int]], None] | None = None) -> Made:
        rows = len(request.prompts)
        if not 0 < rows <= self.max_rows:
            raise ValueError(f"{rows} rows: this decoder holds 1 to {self.max_rows}")
        size = next(b for b in BUCKETS if b >= rows) if rows <= BUCKETS[-1] else rows
        if (longest := max(len(p) for p in request.prompts)) + request.max_new > self.kv.ceiling:
            raise ValueError(f"a row of {longest} + {request.max_new} tokens: it holds {self.kv.ceiling}")
        sampling = self._sampling(request, size)
        drafted = self.drafting
        made = Made(tokens=[[] for _ in range(rows)], kept=[[] for _ in range(rows)])
        admitted: list[int] = []
        with torch.cuda.stream(self.stream), torch.inference_mode(), ExitStack() as call:
            began = time.perf_counter()
            try:
                try:
                    call.enter_context(self._leased())
                    for row, prompt in enumerate(request.prompts):
                        self._admit(row, len(prompt), request, drafted)
                        admitted.append(row)
                except Exception as exc:
                    if not short_of_room(exc):
                        raise
                    answered(exc)
                    raise Short(f"its weights and K/V don't fit at the call's start ({str(exc)[:200]})") from exc
                for row, prompt in enumerate(request.prompts):
                    made.tokens[row].append(self._prefill(row, prompt, request, sampling, drafted))
                self.stream.synchronize()
                made.prefill_s = time.perf_counter() - began
                if told is not None:
                    for row in range(rows):
                        told(row, made.tokens[row][-1:])
                began = time.perf_counter()
                self._decode(size, rows, drafted, sampling, request, made, told)
            finally:
                for row in admitted:
                    self.kv.free(row)
                self.rows.done.fill_(True)
                self.stream.synchronize()
            made.decode_s = time.perf_counter() - began
        return made

    def _admit(self, row: int, length: int, request: Request, drafted: bool) -> None:
        """`row`'s blocks for its prompt and every token it may make (its most new tokens, the
        head's position P and a drafted step's reach past them, up to the row's ceiling), its state
        slot: the call's whole K/V, committed before anything runs, so a refusal for room is the
        call's `Short` at its start, never a failure after a token is told."""
        kv = self.kv
        kv.admit(row, length, drafted=drafted, host_tokens=self._host_tokens(length, request))
        kv.reserve_ahead(row, request.max_new + (SEGMENT + 1) * (kv.layout.span + 1))

    def _sampling(self, request: Request, size: int) -> Sampling:
        sampled = request.temperature > 0
        penalized = bool(request.presence or request.frequency or request.repetition != 1.0)
        values = (request.temperature, request.top_p, request.top_k if sampled else 0, request.min_p,
                  request.presence, request.frequency, request.repetition, request.seed)  # fmt: skip
        for tensor, value in zip(self.decoding, values, strict=True):
            tensor.fill_(value)
        temperature, top_p, top_k, min_p, presence, frequency, repetition, seed = self.decoding
        return Sampling(temperature, top_p, top_k, min_p, presence, frequency, repetition, seed,
                        sampled=sampled, penalized=penalized)  # fmt: skip

    # ------------------------------------------------------------------ prefill

    def _prefill(self, row: int, prompt: Sequence[int], request: Request, sampling: Sampling, drafted: bool) -> int:
        """`row`'s prompt (admitted) through the model's own forward, `prefill_chunk` tokens at a
        time: each attention layer's K/V into the row's blocks as it is made (`_PagedLayer`), the
        gated-delta layers' states carried by Transformers' cache from chunk to chunk and into the
        row's slot at the end; the head's K/V for positions 1 to P chunk by chunk, and its state for
        the first draft. Returns the first token."""
        from transformers import DynamicCache

        hf, kv, qwen, r = self.hf, self.kv, self.qwen, self.rows
        ids = torch.tensor([list(prompt)], dtype=torch.long, device=self.device)
        length = ids.shape[1]
        cache = DynamicCache(config=hf.config.get_text_config(decoder=True))  # type: ignore[arg-type]  # its own config type
        for layer, index in qwen.slot_of.items():
            cache.layers[layer] = cast("CacheLayerMixin", _paged_layer(kv, row, index))
        head = _paged_layer(kv, row, kv.layout.attn_layers - 1, start=1) if drafted else None
        r.counts[row].zero_()
        r.prompt[row].zero_()
        r.prompt[row, ids[0]] = True
        first = -1
        hidden: torch.Tensor | None = None
        for c0 in range(0, length, self.prefill_chunk):
            chunk = ids[:, c0 : c0 + self.prefill_chunk]
            y = hf.get_decoder()(input_ids=chunk, past_key_values=cache, use_cache=True).last_hidden_state[0]
            last = c0 + chunk.shape[1] >= length
            if last:  # [n, hidden], final-normed: the prompt's last position gives the first token
                logits = qwen.logits(y[-1:])
                scores = sampling.penalties(logits, 1, r.counts[row : row + 1], r.prompt[row : row + 1])
                first = int(sampling.pick(scores, 1, torch.zeros(1, dtype=torch.long, device=self.device))[0])
            if head is not None:  # position p's head input is token p with the model's state at p - 1
                after = ids[0, c0 + 1 : c0 + chunk.shape[1] + 1]
                if last:
                    after = torch.cat([after, torch.tensor([first], dtype=torch.long, device=self.device)])
                hidden = self._draft_prompt(after, y, c0 + 1, head)
            del y
        slot = kv.slot(row)
        for layer, index in qwen.delta_of.items():
            states = cast("_States", cache.layers[layer])
            kv.state(0)[slot, index].copy_(states.conv_states[0][0, :, 1:])
            kv.state(1)[slot, index].copy_(states.recurrent_states[0][0])
        del cache
        if hidden is not None:
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

    def _host_tokens(self, prompt: int, request: Request) -> int:
        """A row's tokens of K/V beyond its share of the card's (`kv_card_bytes` across the call's
        rows): its oldest whole blocks, on the host tier."""
        if self.kv_card_bytes is None:
            return 0
        layout = self.kv.layout
        need = prompt + request.max_new + layout.span + 1
        card = self.kv_card_bytes // max(len(request.prompts), 1) // layout.token_bytes
        return max(need - card, 0)

    def _draft_prompt(self, tokens: torch.Tensor, y: torch.Tensor, at: int, head: _Paged) -> torch.Tensor:
        """The head's K/V for positions `at` to `at + n - 1` (token p with the model's state at
        p - 1: `tokens` and `y`, `[n]` and `[n, hidden]`) into its layer's blocks, attending to its
        K/V so far; returns its state at the last of them: `lm`'s `_drafted_prompt`, one row."""
        from transformers import DynamicCache

        hf, qwen, text = self.hf, self.qwen, self.text
        mtp = hf.mtp
        n = int(tokens.shape[0])
        eps = text.eps
        from cozy_runtime.engine.qwen35 import _norm

        e = _norm(qwen.embed(tokens[None]), mtp.pre_fc_norm_embedding.weight, eps, qwen.fused, qwen.run)
        h = _norm(y[None], mtp.pre_fc_norm_hidden.weight, eps, qwen.fused, qwen.run)
        x = mtp.fc(torch.cat([e, h], -1))
        positions = torch.arange(at, at + n, device=self.device)
        rope = hf.get_decoder().rotary_emb(x, positions.view(1, 1, -1).expand(3, 1, -1))
        seen = head.get_seq_length()
        # Causal over the head's K/V so far and this chunk: query i sees keys up to seen + i.
        mask = None if not seen else (
            torch.arange(seen + n, device=self.device)[None, :] <= (seen + torch.arange(n, device=self.device))[:, None]
        )[None, None]
        layer = mtp.layers[0]
        # The head's layer updates the cache at its own index, whatever that is: every slot up to it is `head`.
        index = int(getattr(getattr(layer, "self_attn", None), "layer_idx", 0) or 0)
        cache = DynamicCache()
        cache.layers.extend([cast("CacheLayerMixin", head)] * (index + 1))
        x = layer(x, position_embeddings=rope, attention_mask=mask, past_key_values=cache)
        return mtp.norm(x)[0, -1]

    # ------------------------------------------------------------------ weights

    @contextmanager
    def _leased(self) -> Iterator[None]:
        """The weights a step reads (the language model, its head and, drafting, the draft head)
        held where they are for a call: varena moves no Region a graph holds the address of. A
        model with no placement (a bench's plain tensors) holds nothing. Where the binding's
        identity changed since the graphs were recorded, they are dropped."""
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
                self.segments.clear()
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
        if self.launcher:
            self._launched(size, rows, drafted, sampling, request, made, told)
            return
        kv, r = self.kv, self.rows
        # A graph's copies ahead move the host blocks in use when it was recorded: keyed by their extent.
        key = (size, drafted, sampling.sampled, self.kv.host_extent())
        step, graph = self.steps.get(key) or (Step(self.qwen, r, sampling, size, drafted, self.ring), None)
        made.captured = graph is not None  # a graph an earlier call recorded serves this one
        reach = (step.span + 1) if drafted else 1
        ring = torch.empty((SEGMENT, *r.report[:size].shape), dtype=torch.int32, device=self.device)
        live = [True] * rows
        while any(live):
            for row in range(rows):  # room for a whole segment: no step writes past a mapped page
                if live[row]:
                    kv.settle(row, int(kv.lengths[row]))
                    kv.reserve_ahead(row, (SEGMENT + 1) * reach)
                    r.budget[row] = kv.committed(row)
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

    def _launched(
        self,
        size: int,
        rows: int,
        drafted: bool,
        sampling: Sampling,
        request: Request,
        made: Made,
        told: Callable[[int, list[int]], None] | None,
    ) -> None:
        """The call's steps as launches of the bucket's segment: the first step of a new key runs
        eagerly (it loads its kernels and is what the capture is measured from), then each launch
        runs until the step's `go` is 0. Between launches the host commits each live row's K/V for
        `LAUNCH_STEPS` more steps; during one it reads the ring and tells what came."""
        from cozy_runtime._kernels import _engine

        kv, r, ring = self.kv, self.rows, cast("_Ring", self.ring)
        # A graph's copies ahead move the host blocks in use when it was recorded: keyed by their extent.
        key = (size, drafted, sampling.sampled, self.kv.host_extent())
        step, graph = self.steps.get(key) or (Step(self.qwen, r, sampling, size, drafted, ring), None)
        made.captured = graph is not None
        reach = (step.span + 1) if drafted else 1
        width = int(r.report.shape[1])
        live = [True] * rows

        def take() -> None:
            for record in ring.reader.poll():
                ints = memoryview(record).cast("i").tolist()
                made.steps += 1
                for row in range(rows):
                    count, *tokens = ints[row * width : (row + 1) * width]
                    if count > 0 and live[row]:
                        made.tokens[row].extend(tokens[:count])
                        made.kept[row].append(count)
                        if told is not None:
                            told(row, tokens[:count])

        while any(live):
            for row in range(rows):  # room for a whole launch: no step writes past a mapped page
                if live[row]:
                    length = int(kv.lengths[row])
                    kv.settle(row, length)
                    ahead = min(LAUNCH_STEPS, max(request.max_new - len(made.tokens[row]), 0)) + reach
                    kv.reserve_ahead(row, ahead)
                    r.budget[row] = min(kv.committed(row), length + ahead)
            if graph is None and made.steps and key not in self.steps:  # the second step is recorded
                graph = self._capture(step, made, keep_graph=True)
                self.steps[key] = (step, graph)
                if graph is not None:
                    self.segments[key] = _engine.Segment(graph.raw_cuda_graph(), r.go.data_ptr(), self.device.index or 0)
            if graph is None:  # a new key's first step, or a step that can't be captured: eagerly
                step()
                self.stream.synchronize()
            else:
                import tensorfs.varena.torch as vt

                segment = cast("_engine.Segment", self.segments[key])
                # The graph's pool mapped and retained through every launch on the stream,
                # the segment's parent graph holding its clone among them (var-032).
                with vt.launching(graph, stream=self.stream):
                    segment.launch(self.stream.cuda_stream)
                    done = torch.cuda.Event(blocking=True)
                    done.record(self.stream)
                    if told is not None:
                        while not done.query():
                            time.sleep(1 / POLLS)
                            take()
                    done.synchronize()
            take()
            live = [not d for d in r.done[:rows].tolist()]

    def _capture(self, step: Step, made: Made, *, keep_graph: bool = False) -> torch.cuda.CUDAGraph | None:
        """The step's graph, recorded once into a pool of the model's graph domain (the pool
        grows by measure where it runs short); None, with why in `made.notes`, where it can't be."""
        import tensorfs.varena.torch as vt

        from cozy_runtime.graphs import capture, capture_domain
        from cozy_runtime.graphs.domain import Pool

        domain = capture_domain(torch, self.model)
        try:
            domain.initialize()
            domain.prime_blas()
            graph, why = capture(torch, step, Pool(torch.cuda.graph_pool_handle()), domain, vt.pool_bytes(64 << 20),
                                 alone=True, keep_graph=keep_graph)
        except Exception as exc:  # reported, and the step runs eagerly
            graph, why = None, f"{type(exc).__name__}: {exc}"
        if graph is None:
            made.notes.append(f"step not captured: {why}")
        made.captured = graph is not None
        return graph


class _Ring:
    """Pinned host slots a step reports what it made into (`_kernels._engine.report`, a step's last
    op), the step's device counter of reports, and their reader. A few hundred KiB at most, so it
    stays in torch's pinned memory, outside varena's host ledger."""

    def __init__(self, slots: int, record: int) -> None:
        from cozy_runtime._kernels import _engine

        self.slots = slots
        self.slot_bytes = _engine.HEADER + -(-record // 16) * 16
        self.host = torch.zeros(slots * self.slot_bytes, dtype=torch.uint8).pin_memory()
        self.counter = torch.zeros(1, dtype=torch.int64, device="cuda")
        self.reader = _engine.Reader(self.host.data_ptr(), slots, self.slot_bytes)

    def report(self, src: torch.Tensor) -> None:
        from cozy_runtime._kernels import _engine

        _engine.report(
            self.host.data_ptr(), self.slots, self.slot_bytes, self.counter.data_ptr(), src.data_ptr(),
            src.nbytes, torch.cuda.current_stream().cuda_stream,
        )  # fmt: skip


def _host_tier(model: torch.nn.Module, index: int) -> varena.Pages:
    """The GPU's host tier the model's placement opened (`Memory.pages`), else one of this
    decoder's own (a model with no placement: a bench's)."""
    from tensorfs.varena import _runtime

    from cozy_runtime.author._model import PLACED

    placed = PLACED.get(model)
    pages = getattr(getattr(placed[0], "memory", None), "pages", None) if placed is not None else None
    if pages is not None:
        return cast("varena.Pages", pages)
    if index not in _OWN_PAGES:  # one tier a GPU, as a placement keeps one
        _OWN_PAGES[index] = _runtime.Pages(index)
    return _OWN_PAGES[index]


#: the host tier of a GPU no placement opened one for, made once
_OWN_PAGES: dict[int, varena.Pages] = {}


def short_of_room(exc: BaseException) -> bool:
    """Whether `exc` refuses a call room (lm's test): an allocation refused, varena's `Exhausted`
    (`is_capacity`), a native refusal before anything ran (`DoesNotFit`), or weights whose storage
    a held binding can't serve (`StaleBinding`)."""
    from cozy_runtime.author._errors import DoesNotFit, is_capacity
    from cozy_runtime.internal.accel import StaleBinding

    return is_capacity(exc) or isinstance(exc, (DoesNotFit, StaleBinding))


def answered(exc: BaseException) -> None:
    """A refusal varena counted (`vt.ooms`) that the Runtime answers by running another way, so the
    package's scope is not judged to have swallowed it (as lm's)."""
    from tensorfs.varena import Exhausted

    from cozy_runtime.author._errors import DoesNotFit
    from cozy_runtime.internal.accel import StaleBinding
    from cozy_runtime.internal.execution.catch import CURRENT

    counted = exc.__cause__ if isinstance(exc, DoesNotFit) else exc
    if counted is None or isinstance(counted, (Exhausted, StaleBinding)):
        return
    if (serving := CURRENT.get()) is not None:
        serving.retries += 1


class _Paged(Protocol):
    """An attention layer's prefill cache over the engine's blocks (`_paged_layer`)."""

    def get_seq_length(self) -> int: ...


def _paged_layer(kv: KvManager, row: int, index: int, start: int = 0) -> _Paged:
    made: _Paged = _paged_layer_class()(kv, row, index, start)
    return made


@functools.cache
def _paged_layer_class() -> type:
    from transformers.cache_utils import DynamicLayer

    class PagedLayer(DynamicLayer):  # type: ignore[no-untyped-call]  # Transformers' __init_subclass__
        """Transformers' cache of one attention layer for one row's prefill, over the engine's K/V
        blocks: each chunk's K and V go into the row's blocks at their token slots (from `start`), and
        the layer attends to the row's K/V so far, read back from them (`KvManager.gather`)."""

        def __init__(self, kv: KvManager, row: int, index: int, start: int = 0) -> None:
            super().__init__()  # type: ignore[no-untyped-call]  # Transformers is partly annotated
            self.kv, self.row, self.index, self.start, self.length = kv, row, index, start, 0

        def update(self, key_states: torch.Tensor, value_states: torch.Tensor, *args: object, **kwargs: object
                   ) -> tuple[torch.Tensor, torch.Tensor]:  # fmt: skip
            if not self.is_initialized:
                self.lazy_initialization(key_states, value_states)
            n = int(key_states.shape[-2])
            slots = self.kv.token_slots(self.row, self.start + self.length, n)
            self.kv.write(self.index, key_states[0].transpose(0, 1), value_states[0].transpose(0, 1), slots)
            self.length += n
            keys, values = self.kv.gather(self.index, self.row, self.start, self.length)
            return keys.transpose(0, 1)[None], values.transpose(0, 1)[None]

        def get_seq_length(self) -> int:
            return int(self.length)

    return PagedLayer
