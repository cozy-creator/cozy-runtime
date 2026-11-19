"""Fast decode for Hugging Face causal LMs: a token costs what streaming the weights costs, not
the host's Python.

    from cozy_runtime import lm

    new = lm.generate(model, inputs, max_new_tokens=160, stop=[eos])

The prompt prefills through the model's own forward into a cache of its own; its keys and values
then go, unpadded, into each row's blocks of the engine's K/V (`engine.kv`: varena Caches, pages
mapped as rows join and released as they leave, CozyEngine's), and a linear-attention layer's conv
and recurrent state into the row's state slot there. Every later token is one replay of a CUDA
graph of the one-token step, captured per batch: its attention is `engine.attention.paged_step`,
through Transformers' attention interface, which reads each row's K/V through its block table, so
one graph serves every length and a longer call never records again. The step picks its token on
the GPU, writes it where the next step reads it and advances its own positions, so replays queue
back to back and the host reads the stop flag a few steps behind. The engine decodes on CUDA; a
CPU's call goes through Transformers' loop.

A token is picked as vLLM 0.19.1's sampler picks it (`cozy_runtime.sampling`): bans (`allowed`,
`no_repeat_ngram`), then the penalties, then the argmax at temperature 0, else a draw after the
temperature, min-p, top-k and top-p from noise hashed from (seed, the token's index, the vocabulary
entry). A draw depends only on its own row's scores, seed and position, so it replays inside a
graph and does not depend on what else is in the batch; a drafted token is kept only where it is
the token the model draws there with the same noise, so a drafted call draws from the same
distribution as a plain one. Neither repeats bit for bit on every GPU: the verify pass rounds
otherwise than the one-token step, which can turn a near tie (greedy Qwen3.5-9B split at a margin of
0.125), and two plain sampled calls at one seed have split on a 4090 while greedy calls and eager
forwards repeated exactly (cr-346).

A step key's first step runs eagerly with nothing leased, its weights entering as any forward's
do, and is measured. Its graph is admitted (`_Judge`) only where the weights it reads, its pool,
what is live, one more eager step and its check's copies fit the card; otherwise the key stays
eager, and the record says why with the numbers. The lease belongs to an admitted graph alone
(`_lease`): taken for its capture and replays, let go before any eager step and any prefill. A
cached graph keeps only its weights' addresses; its lease brings their pages back. Offload retires
the graph before those addresses change; a later call can refill and capture anew. An eager
step is the model's forward, its weights streamed through the model's plan, which keeps what the
card holds of the loop and gives way from its end; an embedding table whose page left gathers its
rows on the host. A prefill runs as every forward does; its refusal is the call's. A call's rows
past the largest batch join as slots free, from its one prefill. A call's keys and state for its
prompt and `max_new_tokens` are committed as each row joins (`_Engine._join`), pages from varena's
one reclaim order; refused, the row waits for decoding rows to leave, and with none decoding the
call is refused once (`DoesNotFit`) with the bytes it needs. Errors after execution starts are
final; package code is never restarted.
After a safe capture refusal (a run warning), the same step runs eagerly. A model this
does not serve (a layer type other than full or linear attention, an attention other than SDPA or
eager, a generation config that adds logits processors) generates through Transformers, said once.

With `allowed` (a `Constraint`), each step's tokens are read back to the host, which asks the
constraint what may follow and writes that into a bit mask the next step's graph reads: one sync a
step, as the stop flag is, but caught up with each step rather than a few behind. A drafted
constrained step runs its drafts and its verify as separate graphs, each draft read back before
the mask after it is written.

With a `draft` (a model's multi-token-prediction head) a step drafts `DEPTH` tokens, the model
verifies them and the token after them in one pass, and the step keeps the prefix the model
agrees with and the model's own token after it: the tokens the model alone gives, greedy or drawn
(each draft is drawn with the noise the model's token at its index is drawn with), one to
`DEPTH + 1` of them a replay. Keys past the kept prefix are masked and overwritten, and every
linear-attention layer's state goes back to the kept prefix's end (the pass keeps its state
after each token).
"""

from __future__ import annotations

import contextvars
import inspect
import math
import sys
import threading
import time
import weakref
from collections import OrderedDict, deque
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import ExitStack, contextmanager, nullcontext, suppress
from functools import partial
from typing import TYPE_CHECKING, Any, Literal, NamedTuple, Protocol, cast

import tensorfs.varena.torch as vt
from tensorfs.varena import Exhausted

from cozy_runtime.accel.kernels import KernelsUnavailable, Launch, Plan, TritonKernels
from cozy_runtime.author import _request_path as request_path
from cozy_runtime.author._context import CURRENT, Context
from cozy_runtime.author._errors import Cancelled, DoesNotFit, Fault, fault, is_capacity
from cozy_runtime.author._model import LANDED, PLACED, WARNINGS, Building, Yielded, recovered
from cozy_runtime.author._services import counting, stage_tokens
from cozy_runtime import graphs as _graphs
from cozy_runtime import sampling as sampler
from cozy_runtime.graphs import (
    BYPASS,
    DECODERS,
    _measured,
    _OverCard,
    capture,
    capture_domain,
    close_pool,
    no_room,
    observe,
    safeguard,
    say,
)
from cozy_runtime.graphs.domain import Pool
from cozy_runtime.internal import cache_owners, contract
from cozy_runtime.lm import _kernels

if TYPE_CHECKING:
    from contextlib import AbstractContextManager
    from types import ModuleType

    from torch.utils.hooks import RemovableHandle
    from transformers import PreTrainedConfig, PreTrainedModel
    from transformers.cache_utils import LinearAttentionCacheLayerMixin

    from tensorfs.varena._facts import Room

    from cozy_runtime.author._model import Residency, WarningRow
    from cozy_runtime.engine.kv import KvManager
    from cozy_runtime.graphs.domain import Domain
    from cozy_runtime.internal.residency import Binding, Region

try:
    import torch
    from torch.overrides import TorchFunctionMode
    from torch.utils._python_dispatch import _disable_current_modes
    from transformers import LogitsProcessorList, StoppingCriteria, StoppingCriteriaList
    from transformers.cache_utils import (
        Cache,
        CacheLayerMixin,
        LinearAttentionLayer,
        StaticLayer,
        get_layer_types_and_kwargs,
    )
    from transformers.generation.logits_process import NoRepeatNGramLogitsProcessor
except ImportError as _absent:  # pragma: no cover - the base package has neither
    if TYPE_CHECKING:
        raise
    else:
        ABSENT = f"torch and transformers do not import ({_absent})"
        StaticLayer = LinearAttentionLayer = Cache = TorchFunctionMode = StoppingCriteria = object
        CacheLayerMixin = object
else:
    ABSENT = ""

#: Steps the host may queue ahead of the stop flag it last read.
LAG = 2
#: Tokens a draft head proposes a step. Qwen3.5-9B's head on FLUX.3's prompt upsampler keeps
#: ~0.7 of first drafts and few third ones: 2 is fastest on a 4090 (1 and 3 lose 5%, 4 more).
DEPTH = 2
#: The positions from a row's length that one step can write in a key buffer: the model's for its
#: token and `DEPTH` drafts, and the head's for them, which start one further.
STEP_SLOTS = DEPTH + 2
_DRAFT: contextvars.ContextVar[torch.nn.Module | None] = contextvars.ContextVar(
    "cozy_lm_draft", default=None
)


#: Each model's own multi-token-prediction head, where its checkpoint keeps one (`vlm.construct`).
HEADS: weakref.WeakKeyDictionary[torch.nn.Module, Draft] = weakref.WeakKeyDictionary()


@contextmanager
def drafting(draft: torch.nn.Module | None) -> Iterator[None]:
    """`generate` calls in the body draft with `draft` (`Draft`), so a package's model can name
    the head another model holds without passing it through its own (memoized) calls."""
    token = _DRAFT.set(draft)
    try:
        yield
    finally:
        _DRAFT.reset(token)


#: Each model's continuous batch (buffers and graphs), kept until the plane wants the memory.
_ENGINES: weakref.WeakKeyDictionary[torch.nn.Module, _Engine] = weakref.WeakKeyDictionary()
_ENGINES_LOCK = threading.Lock()
#: The weight plane's paging hooks and holds are not thread-safe: in a process one decode at a
#: time runs through them, an engine's driver for its loop or a call through Transformers' loop.
#: Concurrent calls on one model still decode together, in its engine.
_PLANE = threading.RLock()
#: Batch sizes a step's graphs are recorded at: a step runs the smallest holding every row, and
#: the largest bounds the rows an engine decodes at once (more wait for a slot).
BUCKETS = (1, 2, 4, 8, 16)
#: The most rows a step drafts for: a verify pass keeps every linear-attention layer's state
#: after each of its tokens (Qwen3.5-9B: 24 x 3 x 2 MiB a row), and with more rows the model's
#: own step is compute-bound enough that drafts save little; larger buckets decode undrafted.
DRAFT_ROWS = 4
#: The most the draft head's prompt pass holds of attention scores at once (`_drafted_prompt`).
DRAFT_SCORES = 256 << 20
#: The most prompt tokens a gated-delta-net layer's prefill computes at once (`_prefill_pieces`).
PREFILL_TOKENS = 4096


#: Generation-config fields that add a logits processor to greedy decoding, and their no-op values.
#: Runtime's gated-delta-net decode kernels, built on the executor's GPUs when it starts, for
#: every package that imports this module (`install`).
KERNELS = TritonKernels("cozy-lm-decode", _kernels.entries(), origin=__name__)
_PROCESSORS: dict[str, tuple[object, ...]] = {
    "repetition_penalty": (None, 1.0),
    "encoder_repetition_penalty": (None, 1.0),
    "no_repeat_ngram_size": (None, 0),
    "encoder_no_repeat_ngram_size": (None, 0),
    "bad_words_ids": (None,),
    "min_length": (None, 0),
    "min_new_tokens": (None, 0),
    "suppress_tokens": (None,),
    "begin_suppress_tokens": (None,),
    "forced_bos_token_id": (None,),
    "forced_eos_token_id": (None,),
    "sequence_bias": (None,),
    "exponential_decay_length_penalty": (None,),
}


class Constraint(Protocol):
    """What may follow a row's tokens (`generate`'s `allowed`), an outlines Guide's shape."""

    def allowed(self, tokens: Sequence[int]) -> Sequence[int] | torch.Tensor | None:
        """The ids that may follow `tokens`, one row's generated ids (its prompt excluded), or
        None for any token; once the grammar is complete, its stop token. Ids as a sequence or a
        1-d host tensor; one returned again for the same tokens is read once (`_Engine._permit`).
        Pure in `tokens`: a draft is checked as `allowed(prefix + draft[:k])` for each k, with
        nothing to undo."""
        ...


class After:
    """`then`, from just after a row's first `marker` (a token id, or ids in a row) on, asked about
    the tokens after it; any token before it (a thinking model's thought free, its answer held to
    a grammar: vLLM's structured output under a reasoning parser)."""

    def __init__(self, marker: int | Sequence[int], then: Constraint) -> None:
        self.marker = [marker] if isinstance(marker, int) else [int(t) for t in marker]
        self.then = then

    def allowed(self, tokens: Sequence[int]) -> Sequence[int] | torch.Tensor | None:
        held, first, n = list(tokens), self.marker[0], len(self.marker)
        at = 0
        while True:
            try:
                at = held.index(first, at)
            except ValueError:
                return None
            if held[at : at + n] == self.marker:
                return self.then.allowed(held[at + n :])
            at += 1


class Index(Protocol):
    """An outlines-core `Index`'s shape (it ships no types): a language compiled over a
    vocabulary, as states and the tokens each may take."""

    def get_initial_state(self) -> int: ...

    def get_next_state(self, state: int, token: int) -> int | None: ...

    def get_allowed_tokens(self, state: int) -> Sequence[int] | None: ...


class Grammar:
    """An outlines-core `Index` (a JSON schema or regex compiled over a tokenizer's vocabulary) as
    a `Constraint`: the tokens its language may take after `tokens`. It keeps the states of the
    prefixes it was asked about lately (`PREFIXES`: every row's that shares it, and their drafts'),
    so a row's next token costs one transition, and the allowed ids of the states it met lately
    (`STATES`) as one host tensor each, so a step builds none."""

    #: prefixes whose state it keeps, the latest asked
    PREFIXES = 256
    #: states whose allowed ids it keeps, the latest asked
    STATES = 32
    #: how many tokens back from a prefix it looks for a kept one before reading it from the start
    BACK = 16

    def __init__(self, index: Index) -> None:
        self.index = index
        self.states: OrderedDict[tuple[int, ...], int] = OrderedDict()
        self.ids: OrderedDict[int, torch.Tensor] = OrderedDict()

    def allowed(self, tokens: Sequence[int]) -> torch.Tensor:
        held = tuple(tokens)
        done, state = 0, self.index.get_initial_state()
        for cut in range(len(held), max(len(held) - self.BACK, 0), -1):
            kept = self.states.get(held[:cut])
            if kept is not None:
                done, state = cut, kept
                break
        for token in held[done:]:
            following = self.index.get_next_state(state, token)
            if following is None:
                raise ValueError(f"token {token} is outside the grammar")
            state = following
        _keep(self.states, held, state, self.PREFIXES)
        ids = self.ids.get(state)
        if ids is None:
            said = self.index.get_allowed_tokens(state)
            if said is None:
                raise ValueError(f"state {state} is outside the grammar")
            ids = torch.tensor(list(said), dtype=torch.long)
        _keep(self.ids, state, ids, self.STATES)
        return ids


def _keep[K, V](kept: OrderedDict[K, V], key: K, value: V, most: int) -> None:
    """`key` as `kept`'s latest, at most `most` of them."""
    kept[key] = value
    kept.move_to_end(key)
    while len(kept) > most:
        kept.popitem(last=False)


class _Decoder(Protocol):
    """A causal LM's decoder (`get_decoder`): its rotary embedding, which a draft head shares."""

    rotary_emb: torch.nn.Module

    def register_forward_hook(
        self,
        hook: Callable[[torch.nn.Module, tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]], None],
        /,
    ) -> RemovableHandle: ...


class _Head(Protocol):
    """A causal LM's output head (`get_output_embeddings`): an int8 one has no `.weight`."""

    out_features: int

    def __call__(self, hidden: torch.Tensor, /) -> torch.Tensor: ...


class _Generating(Protocol):
    """What this module calls of a causal LM that Transformers leaves untyped, or types with a
    `self` (`GenerativePreTrainedModel`) no model class meets."""

    def generate(self, **kwargs: object) -> torch.Tensor: ...

    def prepare_inputs_for_generation(
        self, input_ids: torch.Tensor, **kwargs: object
    ) -> dict[str, object]: ...

    def _prepare_position_ids_for_generation(
        self, inputs_tensor: torch.Tensor, model_kwargs: Mapping[str, object]
    ) -> torch.Tensor: ...

    def get_decoder(self) -> _Decoder: ...

    def get_output_embeddings(self) -> _Head: ...


def _lm(model: PreTrainedModel) -> _Generating:
    """`model`'s generation methods, typed."""
    return cast("_Generating", model)  # cast: `generate` serves causal LMs (GenerationMixin's)


def _decoding(model: torch.nn.Module, drafting: bool) -> tuple[str, ...]:
    """The module paths a decode step reads: the language model's decoder, its head and, while
    it drafts, its draft head. A vision tower only a prefill reads stays out, evictable once the
    prefill is done: held with the rest, Qwen3.6-27B's decode pinned 21.8 of a 4090's 22.9 GiB
    (cr-319)."""
    lm = _lm(model)  # type: ignore[arg-type]  # a model `generate` was given
    head = getattr(model, "mtp", None) if drafting else None
    reads = (lm.get_decoder(), lm.get_output_embeddings(), head)
    wanted = {id(m) for m in reads if m is not None}
    paths = tuple(name for name, module in model.named_modules() if name and id(module) in wanted)
    return paths or ("",)


def _set[T](value: T | None) -> T:
    """`value`, which the engine's state has set where it is read (its slots' cache and
    device, its head, its graphs' domain and pool, a row's tokens once it ended)."""
    return cast("T", value)  # cast: each caller reads it only in the state that sets it


def generate(
    model: PreTrainedModel,
    inputs: Mapping[str, torch.Tensor],
    *,
    max_new_tokens: int,
    stop: Sequence[int],
    pad: int | None = None,
    presence_penalty: float = 0.0,
    frequency_penalty: float = 0.0,
    repetition_penalty: float = 1.0,
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: int = 0,
    min_p: float = 0.0,
    seed: int = 0,
    draft: torch.nn.Module | None = None,
    allowed: Constraint | None = None,
    progress: Callable[[int], None] | None = None,
    no_repeat_ngram: int | None = None,
    streamed: Callable[[list[list[int]]], None] | None = None,
) -> torch.Tensor:
    """Tokens after the prompt, `[batch, n]` (`model.generate(...)`'s tail): `inputs` as the
    processor returned them, generation ends at a `stop` token or after `max_new_tokens`, and a
    row that stopped early continues as `pad` (default: the first stop).
    Each token is picked as vLLM 0.19.1 picks it, each setting as its `SamplingParams` defines
    it: bans first (`allowed`, `no_repeat_ngram`); then the penalties, greedy or sampled:
    `repetition_penalty` divides a positive logit and multiplies a negative one of every token in
    the prompt or generated so far (1 is none), `frequency_penalty` subtracts that much for each
    time a token was generated, `presence_penalty` that much from every token generated (both in
    [-2, 2]). `temperature` 0 is then the argmax; above it, the logits over `temperature`, less
    every token below `min_p` times the likeliest's probability (0 is none), outside the `top_k`
    largest (0 is all) and outside the `top_p` nucleus of what is left, and a draw from what
    remains from `seed`: the same logits and seed give the same token (on a GPU, sampled calls
    are not reproducible bit for bit run to run; greedy ones have been). A drafted step draws as an
    undrafted one.
    `allowed` (`Constraint`) constrains every row: each token outside
    `allowed.allowed(its tokens so far)` is out. The host asks it between steps, so a
    constrained batch's steps do not queue ahead of it.
    `no_repeat_ngram` n bans, in each row, every token that would complete an n-gram the row
    already holds, its prompt included (Transformers' `no_repeat_ngram_size`).
    `draft(embeds, hidden, position_ids, past_key_values, attention_mask, rotary) -> hidden` is
    the model's multi-token-prediction head (`Draft`): from a token's embedding and the state the
    model's head read before it, the state the head reads for the token after; `rotary` is the
    model's own rotary embedding. It speeds up a
    batch of one, each token computed as decode computes it, and only where its step is captured
    (else decode goes plain). Without one, the head `drafting` names, else the model's own
    (`HEADS`).
    `progress(tokens)` is told the tokens decoded so far (a call's longest row), at most every
    `PROGRESS_S`, for display only; one that raises is dropped, with a run warning (a cancel is
    the decode's own check's to act on). Without one, the running telemetry stage is told them
    as a count of tokens, `max_new_tokens` only their cap. `ctx.progress` called from it counts
    tokens too, never steps.
    `streamed(rows)` is told each row's tokens so far, at most every `STREAM_S`, for display only
    (`vlm` streams their words); one that raises is dropped, with a run warning."""
    for named in (draft, _DRAFT.get(), HEADS.get(model)):
        if named is not None:
            draft = named
            break
    sampling = _Sampling(
        float(temperature),
        float(top_p),
        int(seed),
        0 if top_k == -1 else int(top_k),  # vLLM's -1 is 0: all
        float(min_p),
        float(presence_penalty),
        float(frequency_penalty),
        float(repetition_penalty),
    )
    if why := sampling.out_of_range():
        raise ValueError(why)
    ngram = int(no_repeat_ngram or 0)
    if ngram < 0:
        raise ValueError(f"no_repeat_ngram {no_repeat_ngram}: none, or at least 1")
    if progress is None and (stage := stage_tokens()) is not None:
        progress = partial(_share, stage, max_new_tokens)
    tell = _Tell(progress, streamed)
    stops = [int(token) for token in stop]
    filler = stops[0] if pad is None else int(pad)
    why = ABSENT or _served(inputs["input_ids"].device) or _unsupported(model)
    if why:
        say(
            type(model).__name__, f"[lm] {type(model).__name__} decodes through Transformers: {why}"
        )
        row: WarningRow = recovered("decode ran through Transformers' loop", why=why)
        if row not in WARNINGS:
            WARNINGS.append(row)
        with contract.owned():  # the Runtime's own use of Transformers' loop and caches
            said = (model, inputs, max_new_tokens, stops, filler, sampling, tell)
            return _transformers(*said, allowed, ngram)
    with _ENGINES_LOCK:
        engine = _ENGINES.get(model)
        if engine is None:
            engine = _ENGINES[model] = _Engine(model)
    request = _Request(inputs, max_new_tokens, stops, filler, sampling, tell, allowed, ngram)
    token = BYPASS.set(True)  # the component's calls are this loop's, its graphs its own
    try:
        with torch.inference_mode(), contract.owned():
            return engine.generate(request, draft)
    finally:
        BYPASS.reset(token)


def _share(stage: Callable[[int, int], None], most: int, tokens: int) -> None:
    """`tokens` decoded of at most `most`, as the running stage's count."""
    stage(tokens, most)


def release(model: torch.nn.Module) -> None:
    """Retire an idle model's decode buffers, graphs and virtual address ownership."""
    with _PLANE, _ENGINES_LOCK:
        engine = _ENGINES.pop(model, None)
        if engine is None:
            return
        if engine.driver is not None or engine.pending or engine.joining or engine._occupied():
            _ENGINES[model] = engine
            raise RuntimeError("cannot unload a language model while decode is active")
        engine._drop()
        for binding in engine.addresses.values():
            binding.close()
        engine.addresses.clear()
        engine.where = ()


def _graph_snapshot(
    model: torch.nn.Module,
) -> tuple[dict[str, object], list[Pool], Domain | None]:
    """Final-reply facts and live pool owners; no device work or per-token aggregation."""
    with _ENGINES_LOCK:
        engine = _ENGINES.get(model)
    if engine is None:
        return {}, [], None
    return (
        {
            "active_graphs": len(engine.graphs),
            "capture_attempts": int(engine.captured[0]),
            "captures": engine.recorded,
            "regrown": [list(pair) for pair in engine.regrown],
            "retired_graphs": engine.retired_graphs,
            "replays": engine.replayed,
            "eager_steps": engine.eager,
            "refused": sorted(set(engine.unrecorded.values())),
            "partial_restore": engine.partial,
        },
        [engine.pool] if engine.pool is not None else [],
        engine.domain,
    )


if TYPE_CHECKING:
    _Module = torch.nn.Module

    class _Mtp(torch.nn.Module):
        """`Draft.mtp`'s submodules, as `Draft.__init__` sets them."""

        fc: torch.nn.Linear
        pre_fc_norm_embedding: torch.nn.Module
        pre_fc_norm_hidden: torch.nn.Module
        layers: torch.nn.ModuleList
        norm: torch.nn.Module
else:
    _Module = torch.nn.Module if not ABSENT else object


class Draft(_Module):
    """The multi-token-prediction head of a Qwen3.5 or Qwen3-Next checkpoint, its weights where
    the checkpoint keeps them (`mtp.*`): the `draft` `generate` takes. From a token's embedding
    and the hidden state the model's head read before it: the state the head reads for the token
    after, which the next draft takes as its hidden state (as vLLM's and SGLang's do). `space` is
    the model's modeling module."""

    mtp: _Mtp

    def __init__(self, config: PreTrainedConfig, space: ModuleType) -> None:
        super().__init__()
        text = config.get_text_config(decoder=True)
        prefix = next(n[: -len("DecoderLayer")] for n in vars(space) if n.endswith("DecoderLayer"))
        norm = getattr(space, f"{prefix}RMSNorm")
        layer = getattr(space, f"{prefix}DecoderLayer")(
            text, list(text.layer_types).index("full_attention")
        )
        layer.self_attn.layer_idx = 0  # the draft's own cache has the one layer
        size, eps = text.hidden_size, text.rms_norm_eps
        self.mtp = cast("_Mtp", torch.nn.Module())  # cast: its submodules are set just below
        self.mtp.fc = torch.nn.Linear(2 * size, size, bias=False)
        self.mtp.pre_fc_norm_embedding = norm(size, eps=eps)
        self.mtp.pre_fc_norm_hidden = norm(size, eps=eps)
        self.mtp.layers = torch.nn.ModuleList([layer])
        self.mtp.norm = norm(size, eps=eps)

    def forward(
        self,
        embeds: torch.Tensor,
        hidden: torch.Tensor,
        positions: torch.Tensor,
        cache: Cache,
        mask: torch.Tensor,
        rotary: torch.nn.Module,
    ) -> torch.Tensor:
        """`rotary` is the model's: its tables are where the model is, at the model's precision
        (a copy of its own would be wherever a package left it, as the head's dtype)."""
        mtp = self.mtp
        joined = torch.cat([mtp.pre_fc_norm_embedding(embeds), mtp.pre_fc_norm_hidden(hidden)], -1)
        x: torch.Tensor = mtp.fc(joined)
        # Qwen3.5 carries the text position first and the rotary (mrope) ones after it.
        text, rope = (positions[0], positions[1:]) if positions.shape[0] == 4 else (None, positions)
        x = mtp.layers[0](
            x,
            position_embeddings=rotary(x, rope),
            attention_mask=mask,
            position_ids=text,
            past_key_values=cache,
        )
        normed: torch.Tensor = mtp.norm(x)
        return normed


def install(plan: Plan, model: torch.nn.Module) -> str:
    """Runtime's one-launch kernels in the Transformers kernel slots `model`'s layers call
    (`causal_conv1d_update`, `torch_recurrent_gated_delta_rule`): a gated-delta-net layer's
    token then costs two launches where the reference path runs a few dozen ops. Call it from
    `Model.execution_plan`; it never waits for the compile (the kernels serve from the first
    request after it lands). A layer of another geometry, or a CPU, keeps Transformers' path.
    Returns the plan's identity, "" when nothing applies. Either way the outcome is in the run's
    receipts (`plan_reports`)."""
    if ABSENT:
        return _report(plan, ABSENT)
    spaces = {sys.modules.get(type(module).__module__) for module in model.modules()}
    for space in spaces:  # the prefill's pieces need no kernel: every device, from now on
        current = getattr(space, "torch_chunk_gated_delta_rule", None)
        if space is not None and callable(current) and not getattr(current, "_cozy_slot", False):
            vars(space)["torch_chunk_gated_delta_rule"] = _prefill_pieces(current)
    slots = [
        (space, name)
        for space in spaces
        if space is not None
        for name in _SLOTS
        if callable(getattr(space, name, None))
    ]
    if not slots:
        return ""
    if _kernels.TRITON_ABSENT:
        return _report(plan, _kernels.TRITON_ABSENT)
    if (device := torch.device(plan.device)).type != "cuda":
        return _report(plan, f"its kernels serve CUDA, not {device}")

    def activate(launch: Launch) -> None:
        for space, name in slots:
            current = getattr(space, name)
            if not getattr(current, "_cozy_slot", False):
                setattr(space, name, _SLOTS[name](current, launch))
        _STEPS[0] = launch
        for engine in list(_ENGINES.values()):  # their graphs recorded Transformers' path
            engine.stale = True
        _report(plan)

    try:
        if plan.when_ready is None:
            activate(plan.load(KERNELS))
        elif not plan.when_ready(KERNELS, activate):
            say(
                KERNELS.name,
                f"[lm] {KERNELS.name} compiling: its kernels serve from the next request",
            )
            _report(plan, "compiling: its kernels serve from the next request")
    except KernelsUnavailable as exc:
        say(KERNELS.name, f"[lm] Transformers' gated-delta-net path: {exc.code}: {exc}")
        return _report(plan, f"{exc.code}: {exc}")
    return f"{KERNELS.name}:{KERNELS.identity}"


def _report(plan: Plan, reason: str = "") -> str:
    """The plan's outcome under its kernel set's name, as `fusion` reports it: applied, or why
    Transformers' path serves. Returns the plan's identity when refused: ""."""
    plan.report(KERNELS.name, {"applied": not reason, **({"reason": reason} if reason else {})})
    return ""


class _ConvUpdate(Protocol):
    """Transformers' `causal_conv1d_update` slot: one token's causal conv."""

    def __call__(
        self,
        hidden_states: torch.Tensor,
        conv_state: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
        activation: str | None = None,
        *args: object,
        **kwargs: object,
    ) -> torch.Tensor: ...


class _ConvFn(Protocol):
    """Transformers' `causal_conv1d_fn` slot: a sequence's causal conv."""

    def __call__(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
        *args: object,
        **kwargs: object,
    ) -> torch.Tensor: ...


class _DeltaRule(Protocol):
    """Transformers' gated-delta-rule slots (`torch_recurrent_gated_delta_rule`,
    `torch_chunk_gated_delta_rule`): the output and, asked for, the final state."""

    def __call__(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        initial_state: torch.Tensor | None = None,
        output_final_state: bool = False,
        use_qk_l2norm_in_kernel: bool = False,
        **kwargs: object,
    ) -> tuple[torch.Tensor, torch.Tensor | None]: ...


def _conv_slot(original: _ConvUpdate, launch: Launch) -> _ConvUpdate:
    def slot(
        hidden_states: torch.Tensor,
        conv_state: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
        activation: str | None = None,
        *args: object,
        **kwargs: object,
    ) -> torch.Tensor:
        batch, channels, length = hidden_states.shape
        bf16 = torch.bfloat16
        if (
            length == 1
            and bias is None
            and activation in ("silu", "swish")
            and hidden_states.is_cuda
            and hidden_states.dtype == weight.dtype == conv_state.dtype == bf16
            and tuple(conv_state.shape) == (batch, channels, _kernels.CONV)
            and tuple(weight.shape) == (channels, _kernels.CONV)
            and conv_state[0].is_contiguous()
            and weight.is_contiguous()
            and hidden_states.stride(1) == 1
        ):
            out = torch.empty((batch, channels, 1), dtype=bf16, device=hidden_states.device)
            _LAUNCHED[0] += 1
            launch(
                _kernels.conv_step_kernel,
                (batch, -(-channels // _kernels.CONV_BLOCK), 1),
                hidden_states,
                conv_state,
                weight,
                out,
                channels,
                hidden_states.stride(0),
                conv_state.stride(0),
                _kernels.CONV,
                _kernels.CONV_BLOCK,
            )
            return out
        return original(hidden_states, conv_state, weight, bias, activation, *args, **kwargs)

    slot._cozy_slot = True  # type: ignore[attr-defined]  # a marker on the function object
    return slot


def _prefill_pieces(original: _DeltaRule) -> _DeltaRule:
    """A gated-delta-net layer's prompt in pieces of `PREFILL_TOKENS`, each starting from the
    recurrent state the one before ended on: the torch path holds several fp32 copies of a
    piece's queries, keys, values and decays, so a long prompt's held all at once (Qwen3.5's
    upsampler at 31,804 tokens, 6.3 GiB live under an 8 GiB cap, run 46625). Pieces end on its
    chunk boundaries, so each chunk is computed as in the whole call."""

    def pieces(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        chunk_size: int = 64,
        initial_state: torch.Tensor | None = None,
        output_final_state: bool = False,
        use_qk_l2norm_in_kernel: bool = False,
        **kwargs: object,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        length = int(query.shape[1])
        piece = max(chunk_size, PREFILL_TOKENS // chunk_size * chunk_size)
        if length <= piece or kwargs.get("cu_seqlens") is not None:
            return original(
                query, key, value, g, beta, chunk_size=chunk_size, initial_state=initial_state,
                output_final_state=output_final_state,
                use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel, **kwargs,
            )  # fmt: skip
        outs: list[torch.Tensor] = []
        state = initial_state
        for at in range(0, length, piece):
            part = slice(at, at + piece)
            out, state = original(
                query[:, part], key[:, part], value[:, part], g[:, part], beta[:, part],
                chunk_size=chunk_size, initial_state=state, output_final_state=True,
                use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel, **kwargs,
            )  # fmt: skip
            outs.append(out)
        return torch.cat(outs, 1), state if output_final_state else None

    vars(pieces)["_cozy_slot"] = True
    return cast("_DeltaRule", pieces)


def _recurrent_slot(original: _DeltaRule, launch: Launch) -> _DeltaRule:
    def slot(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        initial_state: torch.Tensor | None = None,
        output_final_state: bool = False,
        use_qk_l2norm_in_kernel: bool = False,
        **kwargs: object,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch, length, heads, width = query.shape
        bf16 = torch.bfloat16
        if (
            length == 1
            and use_qk_l2norm_in_kernel
            and initial_state is not None
            and query.is_cuda
            and width == _kernels.KEY_DIM
            and tuple(key.shape) == tuple(query.shape)
            and tuple(value.shape) == (batch, 1, heads, _kernels.VALUE_DIM)
            and query.dtype == key.dtype == value.dtype == beta.dtype == bf16
            and g.dtype == initial_state.dtype == torch.float32
            and tuple(initial_state.shape) == (batch, heads, width, _kernels.VALUE_DIM)
            and tuple(g.shape) == tuple(beta.shape) == (batch, 1, heads)
            and initial_state[0].is_contiguous()
            and g.is_contiguous()
            and beta.is_contiguous()
            and query.stride(-1) == key.stride(-1) == value.stride(-1) == 1
        ):
            out = torch.empty(
                (batch, 1, heads, _kernels.VALUE_DIM), dtype=bf16, device=query.device
            )
            _LAUNCHED[0] += 1
            # The verify pass's kernel for one token, the state written over itself: a token
            # then takes the same instructions in both, so a drafted one keeps decode's bits.
            launch(
                _kernels.recurrent_steps_kernel,
                (batch * heads, _kernels.VALUE_DIM // _kernels.VALUE_BLOCK, 1),
                query,
                key,
                value,
                g,
                beta,
                initial_state,
                initial_state,
                out,
                heads,
                1,
                *query.stride()[:3],
                *key.stride()[:3],
                *value.stride()[:3],
                initial_state.stride(0),
                0,
                initial_state.stride(0),
                float(width) ** 0.5,
                _kernels.KEY_DIM,
                _kernels.VALUE_DIM,
                _kernels.VALUE_BLOCK,
            )
            return out, initial_state if output_final_state else None
        return original(
            query,
            key,
            value,
            g=g,
            beta=beta,
            initial_state=initial_state,
            output_final_state=output_final_state,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            **kwargs,
        )

    slot._cozy_slot = True  # type: ignore[attr-defined]  # a marker on the function object
    return slot


_SLOTS: dict[
    str,
    Callable[[_ConvUpdate, Launch], _ConvUpdate] | Callable[[_DeltaRule, Launch], _DeltaRule],
] = {"causal_conv1d_update": _conv_slot, "torch_recurrent_gated_delta_rule": _recurrent_slot}
#: Launches the slots made in this process (Python runs once per capture, not per replay).
_LAUNCHED = [0]
#: The kernels' launcher once they serve: a verify pass's tokens then take one launch per layer.
_STEPS: list[Launch | None] = [None]


def _unsupported(model: PreTrainedModel) -> str:
    config = model.config
    if getattr(config, "is_encoder_decoder", False):
        return "an encoder-decoder"
    text = config.get_text_config(decoder=True)
    if _fused_experts(model, text):
        return "MoE routing has no bounded decode graph contract (experts not split per expert)"
    attention = getattr(text, "_attn_implementation", None) or getattr(
        config, "_attn_implementation", ""
    )
    if attention not in ("sdpa", "eager"):
        return f"attention {attention!r} (SDPA and eager are served)"
    types, _ = get_layer_types_and_kwargs(text)
    other = sorted(set(types) - {*_ATTENTION, "linear_attention"})
    if other:
        return f"layer types {other}"
    shapes, why = _shapes(model)
    if why:
        return why
    if why := _paged_layout(shapes, model.dtype):
        return why
    extras = [
        name
        for name, idle in _PROCESSORS.items()
        if getattr(model.generation_config, name, None) not in idle
    ]
    return f"its generation config adds {extras}" if extras else ""


def _served(device: torch.device) -> str:
    """Why the engine can't decode on `device`, "" where it can: its K/V is varena's and its
    attention a Triton kernel, both CUDA's."""
    from cozy_runtime.engine import _kernels as engine_kernels

    if device.type != "cuda":
        return "the engine decodes on CUDA"
    return engine_kernels.TRITON_ABSENT


def _paged_layout(shapes: list[_Shape], dtype: torch.dtype) -> str:
    """Why the engine's K/V (`engine.kv`: one block layout across every attention layer, 16-bit,
    a power-of-two head width up to 256) can't hold these layers, "" where it can."""
    keys = {(shape[1], shape[2], shape[3]) for shape in shapes if shape[0] == "keys"}
    if len(keys) > 1:
        return f"its attention layers' K/V differ ({sorted(keys)}): one block layout holds all"
    for heads, width, values in keys:
        if width != values or width > 256 or width & (width - 1):
            return f"K/V heads {width} and {values} wide: paged attention takes one power of 2 to 256"
    if keys and dtype not in (torch.bfloat16, torch.float16):
        return f"its K/V would be {dtype}: the paged attention reads 16-bit K/V"
    return ""


#: A gated delta net's sizes in its config (`linear_*`): conv kernel, key heads and width, value
#: heads and width.
_DELTA_NET = (
    "conv_kernel_dim",
    "num_key_heads",
    "key_head_dim",
    "num_value_heads",
    "value_head_dim",
)


#: Attention layer types whose keys the engine keeps (`_Keys`): a sliding layer keeps all of its
#: keys too, and its step's mask closes those more than its window behind (`_Engine._masks`).
_ATTENTION = ("full_attention", "sliding_attention")


def _shapes(model: torch.nn.Module) -> tuple[list[_Shape], str]:
    """Each cache layer's state for one row, before any forward: an attention layer's
    `("keys", heads, key width, value width)`, its own attention module's (Gemma 4's widths and
    heads differ by layer type), a gated delta net's `("state", (conv channels, kernel), (heads,
    key width, value width))` from the config. "" or why not."""
    text = cast("PreTrainedModel", model).config.get_text_config(decoder=True)
    types, kwargs = get_layer_types_and_kwargs(text)
    if isinstance(kwargs, dict):  # type: ignore[unreachable]  # before 5.19, a dict
        kwargs = [kwargs] * len(types)  # type: ignore[unreachable]  # before 5.19, a dict
    decoder = getattr(_lm(cast("PreTrainedModel", model)).get_decoder(), "layers", None)
    linear = [getattr(text, f"linear_{name}", None) for name in _DELTA_NET]
    layers: list[_Shape] = []
    for i, (kind, extra) in enumerate(zip(types, kwargs, strict=True)):
        if kind in _ATTENTION:
            attention = getattr(decoder[i], "self_attn", None) if decoder is not None else None
            width = getattr(attention, "head_dim", None)
            groups = getattr(attention, "num_key_value_groups", None)
            if not isinstance(width, int) or not isinstance(groups, int):
                return [], f"layer {i}'s attention states no head_dim and num_key_value_groups"
            heads = int(text.num_attention_heads) // groups
            layers.append(("keys", heads, width, width))
        elif None in linear or extra.get("number_of_states", 1) != 1:
            return [], "its linear-attention state is not a gated delta net's (Qwen3-Next, Qwen3.5)"
        else:
            kernel, k_heads, k_width, v_heads, v_width = (int(_set(x)) for x in linear)
            conv = (2 * k_heads * k_width + v_heads * v_width, kernel)
            layers.append(("state", conv, (v_heads, k_width, v_width)))
    return layers, ""


#: Models whose experts were found fused (Transformers' host loop) or split (`moe.split`).
_FUSED: weakref.WeakKeyDictionary[torch.nn.Module, bool] = weakref.WeakKeyDictionary()


def _fused_experts(model: torch.nn.Module, text: PreTrainedConfig) -> bool:
    """A mixture of experts whose experts are not all `accel.moe.Experts`: Transformers'
    routes each in a host loop. Split ones decode in two launches a layer (`moe`)."""
    found = _FUSED.get(model)
    if found is None:
        width = max(int(getattr(text, f, 0) or 0) for f in ("num_experts", "num_local_experts"))
        mixture = "moe" in type(model).__name__.lower() or width > 1
        split = any(
            type(m).__name__ == "Experts" and type(m).__module__ == "cozy_runtime.accel.moe"
            for m in model.modules()
        )
        fused = any(
            p.dim() == 3 and name.endswith(("experts.gate_up_proj", "experts.down_proj"))
            for name, p in model.named_parameters()
        )
        found = _FUSED[model] = mixture and (fused or not split)
    return found


def _capture_contract(model: torch.nn.Module, draft: torch.nn.Module | None = None) -> str:
    """Why `model`'s decode step is no graph, "" when it is one: Transformers' own model whose
    class declares its forward one compiled graph (`_can_compile_fullgraph`: no host sync and no
    Python branch on a tensor's value, what torch.compile's CUDA graphs need too), dense or with
    experts split (`moe`), and a drafted step's head (`Draft` around Transformers' decoder
    layer); persistent KV, positions and counters are device buffers. One contract for every
    such LM and VLM (Qwen3, Qwen3.5, Qwen3-VL, Mistral3, Llama, ...), not a list of names (cr-289).

    The real warm step initializes cache views and advances accepted history once. Capture
    only records the next step's device writes. Custom Python hooks and arbitrary model code
    are never speculatively executed by this graph builder.
    """
    kind = type(model)
    if not kind.forward.__module__.startswith("transformers.models."):
        return "not a Transformers model's own forward"
    if not getattr(kind, "_can_compile_fullgraph", False):
        return f"Transformers does not declare {kind.__name__}'s forward one graph"
    # A component root's class is Runtime's graphed subclass (`graphs.install`): forward is Draft's.
    if draft is not None and (
        type(draft).forward is not Draft.forward
        # cast: its forward is Draft's, so its `mtp` is Draft's
        or not type(cast(Draft, draft).mtp.layers[0]).__module__.startswith("transformers.models.")
    ):
        return "no shipped draft head capture contract"
    for module in (model,) if draft is None else (model, draft):
        if module.training:
            return "training state"
        for sub in module.modules():
            for hooks in (sub._forward_pre_hooks, sub._forward_hooks):
                if any(
                    not getattr(h, "__module__", "").startswith("cozy_runtime.internal.residency")
                    for h in hooks.values()
                ):
                    return "custom Python hooks"
            own = vars(sub).get("forward")
            if (
                own is not None
                and own is not vars(sub).get("_cozy_execution_forward")
                and not getattr(own, "__module__", "").startswith("cozy_runtime.")
            ):
                return "custom Python forward"
    return ""


#: A buffer as `_Engine._layout` gives it: shape, dtype, fill.
type _Spec = tuple[tuple[int, ...], torch.dtype, bool | float]
#: A cache layer's state for one row (`_shapes`): a full-attention layer's
#: `("keys", heads, key width, value width)`, a gated delta net's
#: `("state", (conv channels, kernel), (heads, key width, value width))`.
type _Shape = (
    tuple[Literal["keys"], int, int, int]
    | tuple[Literal["state"], tuple[int, int], tuple[int, int, int]]
)


class _Sampling(NamedTuple):
    """A call's sampling (`generate`), each setting as vLLM's `SamplingParams` defines it:
    greedy at temperature 0, which then ignores `top_p`, `top_k` and `min_p` (vLLM's too)."""

    temperature: float
    top_p: float
    seed: int
    top_k: int = 0
    min_p: float = 0.0
    presence: float = 0.0
    frequency: float = 0.0
    repetition: float = 1.0

    def out_of_range(self) -> str:
        """Why vLLM's `SamplingParams._verify_args` would refuse it, "" if it would not."""
        if (
            self.temperature >= 0
            and 0 < self.top_p <= 1
            and 0 <= self.seed < 1 << 63
            and self.top_k >= 0
            and 0 <= self.min_p <= 1
            and -2 <= self.presence <= 2
            and -2 <= self.frequency <= 2
            and self.repetition > 0
        ):
            return ""
        return (
            f"temperature {self.temperature}, top_p {self.top_p}, top_k {self.top_k}, min_p "
            f"{self.min_p}, presence_penalty {self.presence}, frequency_penalty {self.frequency}, "
            f"repetition_penalty {self.repetition}, seed {self.seed}: out of range"
        )

    def kept(self) -> tuple[float, int, float]:
        """`top_p`, `top_k` and `min_p` as a row samples them: none cut a greedy row."""
        return (self.top_p, self.top_k, self.min_p) if self.temperature > 0 else (1.0, 0, 0.0)


#: A step's graph: (rows, drafted, sampled, constrained, and a constrained drafted step's part:
#: each draft, then the verify; else -1). No length: the rows' lengths are device data.
_Key = tuple[int, bool, bool, bool, int]


def _ids(constraint: Constraint, tokens: Sequence[int], vocab: int) -> torch.Tensor | None:
    """`constraint.allowed(tokens)` as a host index tensor: some, each in the vocabulary; None
    where it allows any token."""
    said = constraint.allowed(tokens)
    return None if said is None else _checked(said, len(tokens), vocab)


def _checked(said: Sequence[int] | torch.Tensor, after: int, vocab: int) -> torch.Tensor:
    """A constraint's ids (`allowed` after `after` tokens) as a host index tensor, each checked to
    be in the vocabulary."""
    ids = torch.as_tensor(said).to("cpu", torch.long).reshape(-1)
    if not ids.numel() or int(ids.min()) < 0 or int(ids.max()) >= vocab:
        raise ValueError(
            f"allowed() after {after} tokens gave {ids.numel()} ids: it gives at least one, "
            f"each below the vocabulary's {vocab} (a complete grammar, its stop token)"
        )
    return ids


def _grams(
    tokens: Sequence[int], n: int, index: dict[tuple[int, ...], set[int]], start: int
) -> int:
    """`index` (each (n-1)-token prefix: the tokens that followed it) with `tokens`' n-grams
    from position `start` on; where the next call starts."""
    for i in range(start, len(tokens) - n + 1):
        index.setdefault(tuple(tokens[i : i + n - 1]), set()).add(tokens[i + n - 1])
    return max(start, len(tokens) - n + 1)


def _repeats(tokens: Sequence[int], n: int, index: dict[tuple[int, ...], set[int]]) -> set[int]:
    """The tokens that would complete an n-gram `tokens` holds, `index` being its n-grams
    (Transformers' `no_repeat_ngram_size`: none before the row is n-1 tokens long)."""
    if len(tokens) + 1 < n:
        return set()
    return set(index.get(tuple(tokens[len(tokens) - n + 1 :]), ()))


def _ban(ids: Iterable[int], out: torch.Tensor) -> None:
    """Clear `ids`' bits of `out` (`_pack`'s layout): token 8i + j is bit j of byte i."""
    for token in ids:
        out[token >> 3] &= ~(1 << (token & 7)) & 0xFF


def _pack(ids: torch.Tensor, out: torch.Tensor) -> None:
    """`ids` as bits of `out` (`[ceil(vocab / 8)]` uint8, on the host): token 8i + j is bit j of
    byte i."""
    bits = torch.zeros(out.numel() * 8, dtype=torch.uint8)
    bits[ids] = 1
    out.copy_((bits.view(-1, 8) << torch.arange(8, dtype=torch.uint8)).sum(-1, dtype=torch.uint8))


type _Picking = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


def _per_row(
    sampling: _Sampling, rows: int, device: torch.device
) -> tuple[_Picking, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """`sampling` as per-row tensors: `sampler.pick`'s temperature, top-p, top-k, min-p and seed,
    and `sampler.penalized`'s presence, frequency and repetition penalties."""
    f32, long = torch.float32, torch.long
    top_p, top_k, min_p = sampling.kept()

    def full(value: float, dtype: torch.dtype) -> torch.Tensor:
        return torch.full((rows,), value, dtype=dtype, device=device)

    picking: _Picking = (
        full(sampling.temperature, f32),
        full(top_p, f32),
        full(top_k, long),
        full(min_p, f32),
        full(sampling.seed, long),
    )
    penalties = (
        full(sampling.presence, f32),
        full(sampling.frequency, f32),
        full(sampling.repetition, f32),
    )
    return picking, penalties


class _Pick:
    """Penalties and sampling as one Transformers logits processor, for `_transformers`, after
    its bans: every score but the picked token's is -inf, so its greedy loop takes the pick."""

    def __init__(self, inputs: Mapping[str, torch.Tensor], sampling: _Sampling) -> None:
        ids = inputs["input_ids"]
        mask = inputs.get("attention_mask")
        self.prompt, self.sampling = int(ids.shape[1]), sampling
        self.ids, self.mask = ids, torch.ones_like(ids) if mask is None else mask
        self.prompted: torch.Tensor | None = None

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        rows, vocab, device = int(scores.shape[0]), int(scores.shape[-1]), scores.device
        if self.prompted is None:
            self.prompted = sampler.prompted(self.ids.to(device), self.mask.to(device), vocab)
        generated = input_ids[:, self.prompt :].to(device)
        counts = torch.zeros((rows, vocab + 1), dtype=torch.float32, device=device)
        counts.scatter_add_(1, generated.clamp(max=vocab), torch.ones_like(generated).float())
        picking, penalties = _per_row(self.sampling, rows, device)
        scores = sampler.penalized(scores.float(), counts[:, :vocab], self.prompted, *penalties)
        index = torch.full((rows,), int(generated.shape[1]), dtype=torch.long, device=device)
        token = sampler.pick(scores, *picking, index, self.sampling.temperature > 0)
        return torch.full_like(scores, float("-inf")).scatter_(1, token[:, None], 0.0)


class _Allowed:
    """`allowed` as a Transformers logits processor, for `_transformers`: each row's tokens
    after the prompt, until a stop."""

    def __init__(self, prompt: int, constraint: Constraint, stops: list[int]) -> None:
        self.prompt, self.constraint, self.stops = prompt, constraint, set(stops)

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        keep = torch.ones(scores.shape, dtype=torch.bool)
        for row, tokens in enumerate(input_ids[:, self.prompt :].tolist()):
            vocab = int(scores.shape[-1])
            ids = None if self.stops.intersection(tokens) else _ids(self.constraint, tokens, vocab)
            if ids is not None:  # a finished row continues as padding; None allows any
                keep[row] = False
                keep[row, ids] = True
        return scores.masked_fill(~keep.to(scores.device), float("-inf"))


def _transformers(
    model: PreTrainedModel,
    inputs: Mapping[str, torch.Tensor],
    max_new_tokens: int,
    stops: list[int],
    pad: int,
    picking: _Sampling,
    tell: _Tell,
    allowed: Constraint | None = None,
    ngram: int = 0,
    why: str = "",
) -> torch.Tensor:
    """`why`: what sent the call here, said with its tokens a second in the run record. A
    repetition penalty of the call's own replaces its generation config's, as a vLLM request's
    replaces the checkpoint's default."""
    prompt = int(inputs["input_ids"].shape[1])
    processors: list[_Allowed | _Pick | NoRepeatNGramLogitsProcessor] = (
        [] if allowed is None else [_Allowed(prompt, allowed, stops)]
    )
    if ngram:
        processors.append(NoRepeatNGramLogitsProcessor(ngram))
    processors.append(_Pick(inputs, picking))
    own: dict[str, float] = {} if picking.repetition == 1 else {"repetition_penalty": 1.0}
    context = CURRENT.get()
    with _PLANE, torch.inference_mode(), ExitStack() as held:
        cancel = _Cancel(context, tell, _Loop(model, held, why), prompt)
        out = _lm(model).generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=stops,
            pad_token_id=pad,
            logits_processor=LogitsProcessorList(processors),
            stopping_criteria=StoppingCriteriaList([cancel]),
            **own,
        )
        cancel.loop.report(prompt, cancel.steps)
    return out[:, prompt:]


def _filled() -> int:
    """The weight bytes this request filled so far (every origin, `Request.moved`), the fills
    TensorFS logged read first: a decode's difference over its tokens is what it read again a
    token. 0 outside a request."""
    from cozy_runtime.internal.residency import _REQUEST as FILLING

    request = FILLING.get()
    if request is None:
        return 0
    _ = request.trace  # reading it takes in what TensorFS logged
    return sum(moved[0] for moved in request.moved.values())


class _Loop:
    """Transformers' loop's decode: from its first decode step its embedding tables gather their
    rows on the host once their page has left (`gathering`, cr-332), and its tokens a second and
    the weight bytes it read again a token go in the run record. Its weights stream through the
    model's plan as every forward's do."""

    def __init__(self, model: torch.nn.Module, held: ExitStack[bool | None], why: str) -> None:
        self.placed, self.why, self.held = PLACED.get(model), why, held
        self.decoded = 0.0
        self.filled = 0

    def step(self, steps: int) -> None:
        if self.placed is not None and steps == 1:  # the prompt's token: the decode begins
            self.decoded, self.filled = time.perf_counter(), _filled()
            self.held.enter_context(self.placed[0].gathering(self.placed[1]))

    def report(self, prompt: int, steps: int) -> None:
        if self.placed is None or steps < 2:
            return
        seconds = time.perf_counter() - self.decoded
        again = (_filled() - self.filled) / (steps - 1)
        text = (f"{self.why}: " if self.why else "") + (
            f"{steps} tokens after {prompt} through Transformers' loop: "
            f"{(steps - 1) / max(seconds, 1e-9):.2f} tokens/s decoding, {_gib(again)} of weights "
            "read again a token"
        )
        print(f"[lm] {text}", file=sys.stderr)
        self.placed[0].note(f"{self.placed[1]}.lm", text)


class _Cancel(StoppingCriteria):
    """Transformers' loop raises the attempt's cancellation between tokens, tells its progress
    and its decode's beginning (`_Loop`)."""

    def __init__(self, context: Context | None, tell: _Tell, loop: _Loop, prompt: int) -> None:
        self.context, self.tell, self.loop, self.steps = context, tell, loop, 0
        self.prompt = prompt

    def __call__(
        self, input_ids: torch.Tensor, scores: tuple[torch.Tensor] | None, **kwargs: object
    ) -> torch.BoolTensor:
        if self.context is not None:
            self.context.raise_if_cancelled()
        self.steps += 1  # one token a step: this loop drafts none
        self.loop.step(self.steps)
        self.tell(lambda: self.steps, lambda: input_ids[:, self.prompt :].tolist())
        flags = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
        return cast("torch.BoolTensor", flags)  # cast: Transformers types its flags as BoolTensor


#: The least time between two progress reports of one call.
PROGRESS_S = 0.25
#: The least time between two reports of a call's tokens (`streamed`): words grow about ten times a
#: second.
STREAM_S = 0.1


class _Tell:
    """A call's `progress`, told its tokens so far at most every `PROGRESS_S`, and `streamed`, told
    each row's tokens at most every `STREAM_S`: `count` and `rows` are read only then."""

    def __init__(
        self,
        progress: Callable[[int], None] | None,
        streamed: Callable[[list[list[int]]], None] | None = None,
    ) -> None:
        self.progress, self.told = progress, 0.0
        self.streamed, self.shown = streamed, 0.0

    def __call__(
        self, count: Callable[[], int], rows: Callable[[], list[list[int]]] | None = None
    ) -> None:
        now = time.perf_counter()
        if self.streamed is not None and rows is not None and now - self.shown >= STREAM_S:
            self.shown = now
            try:
                self.streamed(rows())
            except Cancelled:  # the decode's own check ends the call
                pass
            except Exception as exc:  # the decode outlives its words' display
                self.streamed = None
                WARNINGS.append(recovered("decode's words stopped streaming", error=str(exc)[:200]))
        if self.progress is None:
            return
        if now - self.told < PROGRESS_S:
            return
        self.told = now
        # What it tells counts tokens, though it is a package's `ctx.progress(step, total)`.
        unit = counting.set("tokens")
        try:
            self.progress(count())
        except Cancelled:  # the decode's own check ends the call
            return
        except Exception as exc:  # the decode (perhaps other calls' too) outlives its report
            self.progress = None
            WARNINGS.append(
                recovered("decode progress stopped: its report failed", error=str(exc)[:200])
            )
        finally:
            counting.reset(unit)


def _batch(rows: int) -> int:
    """The smallest batch a step records that holds `rows` (at most the largest)."""
    return next((b for b in BUCKETS if b >= rows), BUCKETS[-1])


def _gib(n: float) -> str:
    return f"{n / (1 << 30):.1f} GiB"


class _Keys(StaticLayer):  # type: ignore[no-untyped-call]  # Transformers' __init_subclass__
    """A full-attention layer's keys and values in fixed buffers of the cache's capacity, of which
    attention reads the first `window` positions (the rest are masked)."""

    # Set by `lazy_initialization` or `_Engine._buffers` before this module reads them (a head
    # that writes none leaves `keys` None: read only where checked).
    keys: torch.Tensor
    values: torch.Tensor
    cumulative_length: torch.Tensor  # type: ignore[assignment]  # a static layer's is a tensor

    def __init__(self, size: list[int], capacity: int, rows: bool = False) -> None:
        super().__init__(max_cache_len=capacity)
        #: the window, shared with the cache (a holder, not the cache: no reference cycle keeps
        #: the buffers past the call)
        self.size = size
        #: each row writes at its own length (`cumulative_length` has a row per slot)
        self.rows = rows

    def lazy_initialization(self, key_states: torch.Tensor, value_states: torch.Tensor) -> None:
        # Transformers' own, less `mark_static_address`: nothing here goes through dynamo.
        self.dtype, self.device = key_states.dtype, key_states.device
        self.batch_size, self.num_heads = key_states.shape[:2]
        self.k_head_dim, self.v_head_dim = key_states.shape[-1], value_states.shape[-1]
        shape = (self.batch_size, self.num_heads, self.max_cache_len)
        self.keys = torch.zeros((*shape, self.k_head_dim), dtype=self.dtype, device=self.device)
        self.values = torch.zeros((*shape, self.v_head_dim), dtype=self.dtype, device=self.device)
        self.cumulative_length = self.cumulative_length.to(self.device)
        self.is_initialized = True

    def update(
        self, key_states: torch.Tensor, value_states: torch.Tensor, *args: object, **kwargs: object
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.rows:
            keys, values = super().update(key_states, value_states, *args, **kwargs)
        else:
            keys, values = self.keys, self.values
            count = key_states.shape[-2]
            at = self.cumulative_length[:, None] + torch.arange(count, device=keys.device)
            for buffer, new in ((keys, key_states), (values, value_states)):
                index = at[:, None, :, None].expand(-1, new.shape[1], -1, new.shape[-1])
                buffer.scatter_(2, index, new)
            self.cumulative_length.add_(count)
        size = self.size[0]
        return keys[:, :, :size], values[:, :, :size]

    def get_mask_sizes(self, query_length: int, *args: object, **kwargs: object) -> tuple[int, int]:
        return self.size[0], 0


class _State(LinearAttentionLayer):
    """A linear-attention layer's conv and recurrent state, updated in place."""

    # Transformers declares them by their `None` defaults.
    device: torch.device | None  # type: ignore[assignment]  # set at its first state
    dtype: torch.dtype | None  # type: ignore[assignment]  # set at its first state

    def lazy_initialization(
        self,
        conv_states: torch.Tensor | None = None,
        recurrent_states: torch.Tensor | None = None,
        state_idx: int = 0,
        conv_kernel_size: int | None = None,
    ) -> None:
        # Transformers' own, less `mark_static_address`.
        if conv_states is not None:
            if self.device is None:
                self.dtype, self.device = conv_states.dtype, conv_states.device
            size = conv_states.shape[-1] if conv_kernel_size is None else conv_kernel_size
            self.conv_kernel_size[state_idx] = size
            self.conv_states[state_idx] = torch.zeros(
                (*conv_states.shape[:-1], size), dtype=conv_states.dtype, device=conv_states.device
            )
            self.is_conv_states_initialized[state_idx] = True
        if recurrent_states is not None:
            self.recurrent_states[state_idx] = torch.zeros_like(recurrent_states)
            self.is_recurrent_states_initialized[state_idx] = True

    def update_conv_state(
        self,
        conv_states: torch.Tensor,
        state_idx: int = 0,
        conv_kernel_size: int | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        verify = _VERIFY.get()
        if verify is None or not self.has_previous_state[state_idx]:
            return super().update_conv_state(
                conv_states, state_idx, conv_kernel_size=conv_kernel_size, **kwargs
            )
        state = _set(self.conv_states[state_idx])
        full = torch.cat([state, conv_states], dim=-1)
        verify.conv.append((state, full))  # its state after each token is a slice of `full`
        state.copy_(full[..., -state.shape[-1] :])
        return full


@contextmanager
def _taking(model: PreTrainedModel) -> Iterator[list[torch.Tensor]]:
    """The decoder's final states in the body, one per call: what the output head reads, at
    every position of the call."""
    taken: list[torch.Tensor] = []
    decoder = _lm(model).get_decoder()
    handle = decoder.register_forward_hook(lambda _module, _args, out: taken.append(out[0]))
    try:
        yield taken
    finally:
        handle.remove()


class _RowByRow(TorchFunctionMode):
    """A verify pass's plain linears one row at a time, as decoding runs them: cuBLAS takes
    another kernel for a few rows, whose sums differ in the last bit (Qwen3.5's `in_proj_a` and
    `in_proj_b`), and a drafted token must keep decode's bits. It is faster for those, too."""

    # torch's mode hook sees every torch function, its arguments and its result.
    def __torch_function__(  # type: ignore[explicit-any]  # every torch function
        self,
        func: Callable[..., Any],
        types: tuple[type, ...],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        kwargs = kwargs or {}
        x = args[0] if args else None
        if (
            func is torch.nn.functional.linear
            and x is not None
            and x.dim() > 1
            and 1 < x.shape[-2] <= 16
        ):
            rows = [func(x[..., i : i + 1, :], *args[1:], **kwargs) for i in range(x.shape[-2])]
            return torch.cat(rows, dim=-2)
        return func(*args, **kwargs)


class _Verify:
    """A verify pass's record for putting states back: each linear-attention layer's conv input
    (its state, then the new tokens) and its recurrent state after each token."""

    def __init__(self) -> None:
        self.conv: list[tuple[torch.Tensor, torch.Tensor]] = []
        self.recurrent: list[tuple[torch.Tensor, torch.Tensor]] = []


#: Set while the model verifies drafted tokens: its linear-attention layers then take each token
#: with the one-token kernels' arithmetic (one launch for all once they serve) and record what
#: `_Verify` keeps.
_VERIFY: contextvars.ContextVar[_Verify | None] = contextvars.ContextVar(
    "cozy_lm_verify", default=None
)
#: The Transformers kernel slots a verify pass replaces, and the one-token slots it steps through.
_MANY = ("causal_conv1d_fn", "torch_chunk_gated_delta_rule")
_ONE = ("causal_conv1d_update", "torch_recurrent_gated_delta_rule")


def _speculable(model: torch.nn.Module) -> bool:
    """Whether `model`'s linear-attention layers can verify drafted tokens: their modules carry
    every slot it steps through; it wraps the multi-token ones once."""
    found = {sys.modules.get(type(m).__module__) for m in model.modules()}
    spaces = {space for space in found if space is not None and hasattr(space, _MANY[1])}
    if not all(callable(getattr(space, name, None)) for space in spaces for name in _MANY + _ONE):
        return False
    wraps: tuple[
        Callable[[ModuleType, _ConvFn], _ConvFn] | Callable[[ModuleType, _DeltaRule], _DeltaRule],
        ...,
    ] = (_stepped_conv, _stepped_recurrence)
    for space in spaces:
        for name, wrap in zip(_MANY, wraps, strict=True):
            current = getattr(space, name)
            if not getattr(current, "_cozy_verify", False):
                setattr(space, name, wrap(space, current))
    return True


def _stepped_conv(space: ModuleType, original: _ConvFn) -> _ConvFn:
    def conv(
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
        *args: object,
        **kwargs: object,
    ) -> torch.Tensor:
        if _VERIFY.get() is None:
            return original(x, weight, bias, *args, **kwargs)
        activation = kwargs.get("activation", args[0] if args else None)
        batch, channels, length = x.shape
        size = weight.shape[-1]
        # Channels innermost, as a decoded token's are, so the caller's views of the new tokens
        # reach the recurrent kernels; it keeps only those.
        out = x.new_empty(batch, length, channels).transpose(1, 2)
        launch = _STEPS[0]
        if (
            launch is not None
            and size == _kernels.CONV
            and bias is None
            and activation in ("silu", "swish")
            and x.is_cuda
            and x.dtype == weight.dtype == torch.bfloat16
            and x.stride(-1) == 1
            and tuple(weight.shape) == (channels, size)
            and weight.is_contiguous()
        ):
            _LAUNCHED[0] += 1
            launch(
                _kernels.conv_steps_kernel,
                (batch, -(-channels // _kernels.CONV_BLOCK), 1),
                x,
                weight,
                out[..., size:],
                channels,
                length - size,
                x.stride(0),
                x.stride(1),
                out.stride(0),
                out.stride(2),
                _kernels.CONV,
                _kernels.CONV_BLOCK,
            )
            return out
        state = x[..., :size].clone()
        for t in range(size, length):
            out[..., t] = space.causal_conv1d_update(
                x[..., t : t + 1].contiguous(), state, weight, bias, activation
            )[..., 0]
        return out

    conv._cozy_verify = True  # type: ignore[attr-defined]  # a marker on the function object
    return conv


def _stepped_recurrence(space: ModuleType, original: _DeltaRule) -> _DeltaRule:
    def recurrence(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        initial_state: torch.Tensor | None = None,
        output_final_state: bool = False,
        use_qk_l2norm_in_kernel: bool = False,
        **kwargs: object,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        verify = _VERIFY.get()
        if verify is None or initial_state is None:
            return original(
                query,
                key,
                value,
                g,
                beta,
                initial_state=initial_state,
                output_final_state=output_final_state,
                use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
                **kwargs,
            )
        batch, tokens, heads, width = query.shape
        states = initial_state.new_empty((tokens, *initial_state.shape))
        launch = _STEPS[0]
        if (
            launch is not None
            and use_qk_l2norm_in_kernel
            and query.is_cuda
            and width == _kernels.KEY_DIM
            and tuple(key.shape) == tuple(query.shape)
            and tuple(value.shape) == (batch, tokens, heads, _kernels.VALUE_DIM)
            and query.dtype == key.dtype == value.dtype == beta.dtype == torch.bfloat16
            and g.dtype == initial_state.dtype == torch.float32
            and tuple(initial_state.shape) == (batch, heads, width, _kernels.VALUE_DIM)
            and tuple(g.shape) == tuple(beta.shape) == (batch, tokens, heads)
            and initial_state[0].is_contiguous()
            and g.is_contiguous()
            and beta.is_contiguous()
            and query.stride(-1) == key.stride(-1) == value.stride(-1) == 1
        ):
            out = value.new_empty((batch, tokens, heads, _kernels.VALUE_DIM))
            _LAUNCHED[0] += 1
            launch(
                _kernels.recurrent_steps_kernel,
                (batch * heads, _kernels.VALUE_DIM // _kernels.VALUE_BLOCK, 1),
                query,
                key,
                value,
                g,
                beta,
                initial_state,
                states,
                out,
                heads,
                tokens,
                *query.stride()[:3],
                *key.stride()[:3],
                *value.stride()[:3],
                initial_state.stride(0),
                states.stride(0),
                states.stride(1),
                float(width) ** 0.5,
                _kernels.KEY_DIM,
                _kernels.VALUE_DIM,
                _kernels.VALUE_BLOCK,
            )
            verify.recurrent.append((initial_state, states))
            return out, states[-1]
        state: torch.Tensor = initial_state.clone()
        outs: list[torch.Tensor] = []
        for t in range(tokens):
            out, state = space.torch_recurrent_gated_delta_rule(
                query[:, t : t + 1],
                key[:, t : t + 1],
                value[:, t : t + 1],
                g=g[:, t : t + 1],
                beta=beta[:, t : t + 1],
                initial_state=state,
                output_final_state=True,
                use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            )
            outs.append(out)
            states[t].copy_(state)
        verify.recurrent.append((initial_state, states))
        return torch.cat(outs, dim=1), state

    recurrence._cozy_verify = True  # type: ignore[attr-defined]  # a marker on the function object
    return recurrence


class _StaticCache(Cache):
    #: a view's head cache (`_Engine._view`)
    draft: Cache

    def __init__(
        self, text_config: PreTrainedConfig, capacity: int, size: int, rows: bool = False
    ) -> None:
        self.size = [size]
        types, kwargs = get_layer_types_and_kwargs(text_config)
        if isinstance(kwargs, dict):  # type: ignore[unreachable]  # before 5.19, one set for all
            kwargs = [kwargs] * len(types)  # type: ignore[unreachable]  # before 5.19, a dict
        layers: list[CacheLayerMixin | LinearAttentionCacheLayerMixin] = [
            _Keys(self.size, capacity, rows) if kind in _ATTENTION else _State(**extra)
            for kind, extra in zip(types, kwargs, strict=True)
        ]
        super().__init__(layers=layers)

    @property
    def window(self) -> int:
        """The attention window every full-attention layer returns and is masked to."""
        return self.size[0]

    @window.setter
    def window(self, size: int) -> None:
        self.size[0] = size


class _Paged(CacheLayerMixin):  # type: ignore[misc]  # Transformers' layer base is untyped
    """A full-attention layer of the engine's step, over the rows' K/V blocks (`engine.kv`): it
    gives the step's own K and V back as they came, naming its K/V layer to the attention that
    follows (`_Attending`), which writes them into each row's blocks and reads the rest there."""

    is_compileable = False
    is_sliding = False

    def __init__(self, attending: _Attending, layer: int, head: bool = False) -> None:
        super().__init__()
        self.attending, self.layer, self.head = attending, layer, head
        self.is_initialized = True

    def lazy_initialization(self, key_states: torch.Tensor, value_states: torch.Tensor) -> None:
        return None

    def update(
        self, key_states: torch.Tensor, value_states: torch.Tensor, *args: object, **kwargs: object
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.attending.layer, self.attending.head = self.layer, self.head
        return key_states, value_states

    def get_mask_sizes(self, query_length: int, *args: object, **kwargs: object) -> tuple[int, int]:
        return query_length, 0

    def get_seq_length(self) -> int:
        return 1  # it decodes after a prefill: the cache holds tokens

    def get_max_length(self) -> int:
        return -1


class _PagedCache(Cache):
    """The engine's step's cache over its first rows: `_Paged` for each attention layer, `_State`
    over each gated-delta layer's state in the rows' slots (`KvManager.state`)."""

    #: a view's head cache (`_Engine._view`)
    draft: Cache


class _Attending:
    """The paged attention of the engine's step (`engine.attention.PAGED`): over the K/V layer the
    cache's last update named, the model's through the rows' lengths and slots, the draft head's a
    position ahead and from position 1 (`_Engine._tables`); a sliding layer's window, its
    module's."""

    def __init__(self, engine: _Engine) -> None:
        self.engine = engine
        self.layer, self.head = 0, False

    def __call__(
        self,
        module: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        scaling: float,
    ) -> torch.Tensor:
        from cozy_runtime.engine.attention import HostTier, paged_step
        from cozy_runtime.engine.qwen35 import PARTS

        engine = self.engine
        kv = _set(engine.kv)
        rows = int(query.shape[0])
        lengths, slots = (
            (engine.head_lengths, engine.head_slots) if self.head else (kv.lengths, kv.slot_mapping)
        )
        out: torch.Tensor = paged_step(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            kv.blocks,
            HostTier(kv.host_view, self.layer),
            kv.block_table[:rows],
            lengths[:rows],
            slots[:rows],
            kv.layout,
            self.layer,
            PARTS,
            scaling,
            first_seen=1 if self.head else 0,
            window=int(getattr(module, "sliding_window", 0) or 0),
        )
        return out


def _layers(cache: Cache) -> list[_Keys | _State]:
    """The layers of a cache this module built (`_StaticCache`, a head's)."""
    return cast("list[_Keys | _State]", cache.layers)  # cast: it builds only these


class _Prefill:
    """A call's prompt through the model: its own cache, mask and positions, each row's first
    token, its prompt's tokens (`sampler.prompted`) and, drafting, the head's keys for the prompt
    and its state after it."""

    def __init__(
        self,
        cache: _StaticCache,
        mask: torch.Tensor,
        position: torch.Tensor | None,
        token: torch.Tensor,
        prompted: torch.Tensor,
        drafted: tuple[_Keys, torch.Tensor] | None,
    ) -> None:
        self.cache, self.mask, self.position, self.token, self.prompted, self.drafted = (
            cache,
            mask,
            position,
            token,
            prompted,
            drafted,
        )


def _derived(models: Iterable[torch.nn.Module]) -> list[torch.Tensor]:
    """The non-persistent buffers of `models` on a device: built from config (a rotary table),
    never a checkpoint's, so the plane neither places nor pins them."""
    return [
        tensor
        for model in models
        for module in model.modules()
        for name, tensor in module._buffers.items()
        if name in module._non_persistent_buffers_set
        and tensor is not None  # as `type(None)` is not "Tensor"
        and type(tensor).__name__ == "Tensor"
        and tensor.device.type != "meta"
        and tensor.device.type != "cpu"
    ]


class _Request:
    """One `generate` call: its rows from their prefill to the last one's retirement."""

    def __init__(
        self,
        inputs: Mapping[str, torch.Tensor],
        max_new: int,
        stops: list[int],
        pad: int,
        sampling: _Sampling,
        tell: _Tell,
        allowed: Constraint | None = None,
        ngram: int = 0,
    ) -> None:
        self.allowed, self.ngram = allowed, ngram
        #: its rows' tokens the host decides on between steps (`allowed`, `ngram`)
        self.constrained = allowed is not None or ngram > 0
        #: each row's prompt, as the host bans repeats from it (padding included, as Transformers')
        self.prompts: list[list[int]] = inputs["input_ids"].tolist() if ngram else []
        #: each row's first token's bans, from its prompt alone
        self.banned: list[list[int]] = []
        self.inputs, self.max_new, self.stops, self.pad, self.sampling = (
            inputs,
            max_new,
            stops,
            pad,
            sampling,
        )
        self.tell = tell
        rows = int(inputs["input_ids"].shape[0])
        self.prompt = int(inputs["input_ids"].shape[1])
        self.tokens: list[torch.Tensor | None] = [None] * rows
        self.left = rows
        self.error: BaseException | None = None
        #: why it decodes through Transformers' loop instead (its caller runs it)
        self.fallback = ""
        #: the head it drafts with (None: any, drafting keeps a model's tokens)
        self.draft: torch.nn.Module | None = None
        self.prefill_s = self.began = self.ended = 0.0
        #: the engine's (captures, capture seconds, eager steps) at admission, then over its life
        self.at: tuple[float, ...] = (0, 0.0, 0)
        #: its steps, the drafted ones among them, and its rows summed over them
        self.steps = self.drafted = self.live = 0
        #: its steps by the rung each ran on: "graph", or why it ran eagerly
        self.rungs: dict[str, int] = {}
        #: the weight bytes the request had filled when its decode began and when it ended
        #: (`_filled`)
        self.filled = [0, 0]
        #: its prefill, while rows of it wait for a slot (`parked`, `_Engine._join`)
        self.fresh: _Prefill | None = None
        self.parked: list[int] = []
        self.warnings: list[WarningRow] = []
        #: the attempt it serves (cancellation), and whether its rows left for it
        self.context: Context | None = CURRENT.get()
        self.cut = False
        #: it waits for decoding rows to leave (their slots, or the memory a larger batch needs):
        #: it asks again at the next boundary a row's end makes, never one of its own
        self.waits = False

    @property
    def finished(self) -> bool:
        return self.left == 0 or self.error is not None or bool(self.fallback) or self.cut

    @property
    def cancelled(self) -> bool:
        return self.context is not None and self.context.cancelled

    def result(self) -> torch.Tensor:
        width = max(int(_set(t).shape[0]) for t in self.tokens)
        out = torch.full((len(self.tokens), width), self.pad, dtype=torch.long)
        for row, tokens in enumerate(self.tokens):
            out[row, : int(_set(tokens).shape[0])] = _set(tokens).cpu()
        return out.to(self.inputs["input_ids"].device)


class _Row:
    """A slot's sequence: its call, its row of that call, and where the host knows it to be."""

    def __init__(self, request: _Request, row: int, length: int) -> None:
        self.request, self.row, self.length = request, row, length
        #: a constrained row's tokens as the host knows them (`_took`), and this step's drafts
        self.generated: list[int] | None = None
        self.drafts: list[int] = []
        #: an `ngram` row's n-grams over its prompt and `generated` (`_grams`), and where they end
        self.grams: dict[tuple[int, ...], set[int]] = {}
        self.counted = 0

    def ended(self) -> bool:
        """A constrained row the host knows done: at a stop token, or its limit."""
        said, request = self.generated or [], self.request
        return bool(said) and (said[-1] in request.stops or len(said) >= request.max_new)


class _Engine:
    """A model's continuous batch: every concurrent `generate` call on it decodes in one loop.

    Each sequence holds a slot: a row of the engine's K/V (`engine.kv`: its blocks, its
    linear-attention state in the slot of its index) and one row of every token buffer: its token,
    positions and output. A step runs the smallest bucket (`BUCKETS`) holding every occupied slot,
    recorded once as a CUDA graph and replayed, whatever the rows' lengths. A call joins at the
    next step after its prefill, which splices its rows into free slots, unpadded, and each row
    leaves when the host sees it done, its slot filled from the top so the bucket shrinks. No
    thread of its own: the caller that finds the engine idle drives, and hands driving to a
    waiting caller when its own rows are done."""

    # The per-row buffers, assigned together from one `fields` mapping when the slots are sized.
    ids: torch.Tensor
    length: torch.Tensor
    out: torch.Tensor
    n: torch.Tensor
    done: torch.Tensor
    limit: torch.Tensor
    stops: torch.Tensor
    pad: torch.Tensor
    presence: torch.Tensor
    frequency: torch.Tensor
    repetition: torch.Tensor
    temperature: torch.Tensor
    top_p: torch.Tensor
    top_k: torch.Tensor
    min_p: torch.Tensor
    seed: torch.Tensor
    counts: torch.Tensor
    prompted: torch.Tensor
    position: torch.Tensor
    hidden: torch.Tensor
    drafts: torch.Tensor
    dhidden: torch.Tensor
    allow: torch.Tensor
    #: the draft head's lengths and token slots for its step (`_tables`), a position ahead
    head_lengths: torch.Tensor
    head_slots: torch.Tensor
    #: `[0, DEPTH]`, a step's positions past a row's length
    steps: torch.Tensor

    def __init__(self, model: PreTrainedModel) -> None:
        self.model = model
        self.text = model.config.get_text_config(decoder=True)
        parameters = set(inspect.signature(model.forward).parameters)
        self.keep, self.positional = "logits_to_keep" in parameters, "position_ids" in parameters
        types, kwargs = get_layer_types_and_kwargs(self.text)
        if isinstance(kwargs, dict):  # type: ignore[unreachable]  # before 5.19, one set for all
            kwargs = [kwargs] * len(types)  # type: ignore[unreachable]  # before 5.19, a dict
        #: each layer's state for one row (`_shapes`), and a linear-attention layer's own arguments
        self.shapes = _shapes(model)[0]
        self.layer_kwargs = list(kwargs)
        #: each attention layer's K/V layer, and each gated-delta layer's state index, by layer
        attention = [i for i, shape in enumerate(self.shapes) if shape[0] == "keys"]
        recurrent = [i for i, shape in enumerate(self.shapes) if shape[0] == "state"]
        self.kv_layer = {i: j for j, i in enumerate(attention)}
        self.delta = {i: j for j, i in enumerate(recurrent)}
        #: the rows' K/V and state (`engine.kv`), made at the first call and with each head
        self.kv: KvManager | None = None
        #: the stream the K/V's own work runs on (`_kv_stream`)
        self.kv_stream: torch.cuda.Stream | None = None
        #: why the last row the K/V had no room for was refused (`_refusal`)
        self.refused = ""
        #: the rows the K/V holds, at their indices
        self.kv_rows: set[int] = set()
        self.attend = _Attending(self)
        #: the configs the model's and its head's attention read their implementation from
        self.configs: list[object] = []
        self.cond = threading.Condition(threading.RLock())
        self.pending: deque[_Request] = deque()
        #: calls taken from `pending` that the driver is prefilling or splicing
        self.incoming: list[_Request] = []
        #: calls prefilled whose rows wait for slots (`_Request.parked`), first come first
        self.joining: list[_Request] = []
        #: another head: the bindings are made again with it (`_bind`)
        self.rebind = False
        #: the leases of the weights an admitted graph reads and the graphs' pool (`_lease`), and
        #: the modules they cover
        self.lease: ExitStack[bool | None] | None = None
        self.leased: frozenset[int] = frozenset()
        #: the rung the last step ran on (`_Request.rungs`)
        self.rung = ""
        #: a key judged eager -> the room it was judged in and what it lacked: judged again once
        #: the room covers that
        self.judged: dict[_Key, tuple[int, int]] = {}
        #: a drafted step could not be captured: drafting pays only replayed, so steps go plain
        self.undrafted = False
        #: captures recorded in this engine's life and their seconds, steps run eagerly
        self.captured = [0, 0.0]
        self.eager = 0
        self.recorded = 0
        #: of those, each that took more than its warm-up's guess: the guess, and the room it
        #: was recorded in (`graphs.capture`)
        self.regrown: list[tuple[int, int]] = []
        #: a dev machine's checks: what the eager step without the whole write set put back gave
        self.partial: list[str] = []
        self.replayed = 0
        self.retired_graphs = 0
        #: `LANDED` when a capture, or its graph domain's opening, met a kernel still building
        #: (none records until one lands), and when the graphs were last started over
        self.building, self.landed = -1, LANDED[0]
        self.driver: int | None = None
        self.draft: torch.nn.Module | None = None
        self.slots: list[_Row | None] = []
        #: a row's most output tokens (`out`), and its stops' width
        self.capacity = 0
        self.width = 0
        self.views: dict[int, _PagedCache] = {}
        self.graphs: dict[_Key, torch.cuda.CUDAGraph] = {}
        #: Full shape variants whose real accepted step preseeded their capture pool, and the
        #: working peak varena measured it at: the pool its capture prepares first.
        self.warmed: set[_Key] = set()
        self.peaks: dict[_Key, int] = {}
        self.layout: tuple[object, ...] | None = None
        #: its decode graphs' capture pool (`domain.Pool`): the model's graph region, or a
        #: torch pool of its own with no varena memory
        self.pool: Pool | None = None
        self.addresses: dict[int, Binding] = {}
        #: the module paths each binding leases (`_decoding`)
        self.reads: dict[int, tuple[str, ...]] = {}
        self.completions: dict[torch.cuda.Stream, torch.cuda.Event] = {}
        self.unfenced: set[torch.cuda.Stream] = set()
        self.domain: Domain | None = None
        #: a step's graph key (`_Key`) -> why that step runs eagerly
        self.unrecorded: dict[_Key, str] = {}
        #: the graphs recorded a path that changed since (a kernel landed in `install`)
        self.stale = False
        #: seconds the last admission took: decoding gets as long before the next one
        self.debt = 0.0
        self.retire = False
        self.cache_device: torch.device | None = None
        # an int8 head: no .weight
        self.vocab = int(_lm(model).get_output_embeddings().out_features)
        #: the bits of each host tensor of ids a constraint returned lately (a grammar's state), by its identity
        self.packed: OrderedDict[int, tuple[torch.Tensor, torch.Tensor]] = OrderedDict()
        #: whether the graphs' pool is held mapped for an admitted graph's replays (`_pool`)
        self.pooled = False
        #: how a prefill's state differed from the slots its config sized: later calls decode
        #: through Transformers
        self.misfit = ""
        #: `_capture_contract` with and without the head, until the graphs are next forgotten
        self.contracts: dict[bool, str] = {}
        #: the part of a constrained drafted step being run (`_one`), -1 for a whole step
        self.part = -1
        self.open: bool | float
        self.closed: bool | float
        if self.text._attn_implementation == "sdpa":
            self.open, self.closed = True, False
        else:  # eager attention adds its mask: 0 where it attends, the dtype's minimum elsewhere
            self.open, self.closed = 0.0, torch.finfo(model.dtype).min

    # ------------------------------------------------------------------ calls

    def generate(self, request: _Request, draft: torch.nn.Module | None) -> torch.Tensor:
        request.draft = draft if draft is not None and self._speculable() else None
        if self.misfit:
            request.fallback = self.misfit
        if not request.finished:
            with self.cond:
                self.pending.append(request)
                self.cond.notify_all()
                while not request.finished and self.driver is not None:
                    self.cond.wait()
                if not request.finished:
                    self.driver = threading.get_ident()
        if not request.finished:
            self._drive(request)
        for warning in request.warnings:
            WARNINGS.append(warning)
        if request.cut and request.context is not None:
            request.context.raise_if_cancelled()
        if request.error is not None:
            raise request.error
        if request.fallback:
            say(
                f"lm:{type(self.model).__name__}:{request.fallback[:30]}",
                f"[lm] {request.fallback}",
            )
            WARNINGS.append(
                recovered("decode ran through Transformers' loop", why=request.fallback)
            )
            return _transformers(
                self.model,
                request.inputs,
                request.max_new,
                request.stops,
                request.pad,
                request.sampling,
                request.tell,
                request.allowed,
                request.ngram,
                request.fallback,
            )
        self._report(request)
        return request.result()

    def _speculable(self) -> bool:
        return _speculable(self.model)

    def _report(self, request: _Request) -> None:
        tokens = sum(int(_set(t).shape[0]) for t in request.tokens)
        rows = len(request.tokens)
        looped = (tokens - rows) / rows  # a row's tokens after its first, the prefill's
        captures, capture_s, eager = request.at
        loop_s = max(request.ended - request.began - capture_s, 0.0)
        per = 1000 * loop_s / max(looped, 1)
        mode = "eager" if _set(self.cache_device).type != "cuda" else "graphs"
        if eager:
            mode = "partly eager" if mode == "graphs" else mode
        if request.drafted:  # a head held but never drafted with (a CPU's steps) says nothing
            some = "" if request.drafted == request.steps else f" in {request.drafted} of its steps"
            mode += f", drafting {DEPTH}{some}: {looped / max(request.steps, 1):.2f} tokens a step"
        mode += f", {request.live / max(request.steps, 1):.1f} rows a step"
        rungs = "; ".join(f"{rung}: {n}" for rung, n in request.rungs.items())
        again = (request.filled[1] - request.filled[0]) / max(request.steps, 1)
        facts = (
            f"prefill {request.prefill_s:.3f} s",
            f"then {per:.2f} ms/token over {looped:.0f} ({mode})",
            f"{captures} captures {capture_s:.3f} s",
            f"gated-delta kernels {'on' if _LAUNCHED[0] else 'off'}",
            f"steps by rung: {rungs or 'none'}",
            f"{_gib(again)} of weights read again a step",
        )
        text = f"{tokens} tokens after {request.prompt}: " + ", ".join(facts)
        print(f"[lm] {text}", file=sys.stderr)
        placed = PLACED.get(self.model)
        if placed is not None:
            placed[0].note(f"{placed[1]}.lm", text)

    # ------------------------------------------------------------------ the loop

    def _drive(self, mine: _Request) -> None:
        with _PLANE:
            try:
                self._bind()
                while True:
                    if self.rebind:  # another head: its blocks are bound instead
                        self._bind()
                        self.rebind = False
                    with self.cond:
                        self.incoming = list(self.pending)
                        self.pending.clear()
                    if self.incoming or self.retire or self.joining:
                        self._settle(self.incoming)
                        self.incoming = []
                    occupied = sum(row is not None for row in self.slots)
                    with self.cond:
                        if not occupied and not self.pending:
                            break
                        if mine.finished and (occupied or self.pending):
                            break  # a waiting caller drives on
                    if occupied:
                        self._steps()
            except BaseException as exc:  # post-start failures never restart package generation
                self._fail(exc)
                raise
            finally:
                self._unlease()  # no step runs: its graphs' weights and pool may go idle
                with self.cond:
                    self.driver = None
                    self.cond.notify_all()

    def _fail(self, exc: BaseException) -> None:
        """Every held and waiting call fails; no partially executed generation is restarted."""
        with self.cond:
            requests = {id(row.request): row.request for row in self.slots if row is not None}
            requests.update({id(request): request for request in self.pending})
            requests.update({id(request): request for request in self.incoming})
            requests.update({id(request): request for request in self.joining})
            self.pending.clear()
            self.incoming = []
            for request in self.joining:
                request.fresh, request.parked = None, []
            self.joining = []
            for request in requests.values():
                request.error = exc
            self._drop()
            self.cond.notify_all()

    def _targets(self) -> list[tuple[torch.nn.Module, tuple[Residency, str]]]:
        return [(m, PLACED[m]) for m in (self.model, self.draft) if m is not None and m in PLACED]

    def _bind(self) -> None:
        """The bindings a step's graph reads, its addresses retained and nothing leased: the
        model's decode set and, while it drafts, its head's. One whose storage changed goes, with
        the graphs that read it. Nothing is prepared: a step's weights enter as its forward reads
        them, and a graph's lease takes them (`_lease`). A key judged eager in less room than the
        card has now is judged again. The KV pin has a separate lifetime so changing these
        bindings cannot make live decode rows reclaimable."""
        self._unlease()
        derived = _derived(m for m in (self.model, self.draft) if m is not None)
        built = tuple(t.data_ptr() for t in derived)
        identities: list[object] = []
        for module, (residency, name) in self._targets():
            drafting = self.draft is not None
            reads = _decoding(module, drafting) if module is self.model else ("",)
            binding = self.addresses.get(id(module))
            moved = self.reads.get(id(module)) != reads
            if binding is not None and (not binding.valid() or moved):
                self._forget()
                binding.close()
                binding = None
            if binding is None:
                binding = self.addresses[id(module)] = residency.retain(name, reads)
                self.reads[id(module)] = reads
            identities.append(binding.identity)
        self.where = (*identities, built)
        if self.judged and self.domain is not None:
            room = self._room()
            grown = [key for key, (then, short) in self.judged.items() if room >= then + short]
            for key in grown:
                del self.judged[key]
                self.unrecorded.pop(key, None)
                self.undrafted = self.undrafted and not key[1]

    def _room(self) -> int:
        """What the card leaves a graph beside what is pinned and live now (`vt.room`)."""
        room = vt.room(_set(self.domain).device)
        return room["capacity"] - room["pinned"] - room["working_live"]

    def _lease(self, drafted: bool) -> str:
        """The weights a graph of the step reads (`_read`) and the graphs' pool, held for its
        capture and replays until an eager step or a prefill (`_unlease`): "" once held, else why
        not: a refusal of them (room, or a binding whose storage changed), before anything of
        the step ran. Anything else ends the lease and is raised."""
        reads = frozenset(id(m) for m in self._read(drafted))
        if self.lease is not None and reads <= self.leased:
            return ""
        self._unlease()
        lease: ExitStack[bool | None] = ExitStack()
        try:
            for module in self._read(drafted):
                if (binding := self.addresses.get(id(module))) is not None:
                    lease.enter_context(binding.prepare())
            self.lease, self.leased = lease, reads
            self._pool(True)
        except BaseException as exc:
            if self.lease is None:
                lease.close()
            self._unlease()
            if not (_short_of_room(exc) or getattr(exc, "code", None) == "STALE_BINDING"):
                raise
            _unwind(exc)
            return no_room(_set(self.domain).device, "its graph's weights and pool", 0, exc)
        return ""

    def _unlease(self) -> None:
        """An admitted graph's lease ends (`_lease`): its weights and pool may go idle, their
        pages leaving at varena's choice and their addresses kept."""
        if self.lease is None:
            return
        self._pool(False)
        lease, self.lease, self.leased = self.lease, None, frozenset()
        lease.close()

    #: The retained bindings' fill generations and derived tensor addresses.
    where: tuple[object, ...] = ()

    def _cancelled(self) -> bool:
        """Whether a held row's call was cancelled or failed (its constraint raised): the host
        stops queueing steps for it."""
        return any(
            row.request.cancelled or row.request.error is not None
            for row in self.slots
            if row is not None
        )

    def _occupied(self) -> int:
        return sum(row is not None for row in self.slots)

    def _bucket(self) -> int:
        return _batch(self._occupied())

    def _one(self, size: int, cuda: bool) -> None:
        """One step at `size` rows, and what the host knows of the rows after it. A constrained
        step's host work comes between its parts, outside the graph domain: what each row may
        take, before each part (`_fill`); a drafted step's each draft, read back after it."""
        rows = [row for row in self.slots if row is not None]
        drafting, constrained = self._drafting(size), self._constrained()
        per = DEPTH + 1 if drafting else 1
        stream: torch.cuda.Stream | None = None
        event: torch.cuda.Event | None = None
        if _set(self.cache_device).type == "cuda":
            if self.domain is None:
                placed = PLACED.get(self.model)
                self.domain = (
                    placed[0].graph_domain(placed[1])
                    if placed is not None
                    else capture_domain(torch, self.model)
                )
            stream = torch.cuda.current_stream()
            event = self.completions.get(stream)
            if event is None:
                event = torch.cuda.Event()
                self.completions[stream] = event
                event.record(stream)  # instantiate before any accepted step can run
            self.unfenced.add(stream)
        seen: list[str] = []
        try:
            for part in range(DEPTH + 1) if drafting and constrained else (-1,):
                self.part = part
                if constrained:
                    self._fill(size, max(part, 0))
                self.rung = "eager" if stream is None else ""
                with _set(self.domain).using() if stream is not None else nullcontext():
                    self._replay(size, cuda)
                seen.append(self.rung)
                if part < 0 or part == DEPTH or not self._drafting(size):
                    break  # the whole step, its verify, or a plain step for a part not recorded
                if stream is not None:
                    stream.synchronize()
                self._drafted(size, part)
        finally:
            self.part = -1
            if event is not None:
                event.record(stream)
                self.unfenced.discard(stream)
        rung = next((rung for rung in seen if rung != "graph"), "graph")
        for request in {id(row.request): row.request for row in rows}.values():
            request.rungs[rung] = request.rungs.get(rung, 0) + 1
            request.steps += 1
            request.drafted += drafting
            request.live += len(rows)

    def _settle(self, incoming: list[_Request]) -> None:
        """Between steps, with the device caught up: done and cancelled rows leave, the rows
        above fill their slots, and new calls prefill into the slots that are free."""
        if self.kv is not None:
            with _disable_current_modes():
                if _set(self.cache_device).type == "cuda":
                    torch.cuda.current_stream().synchronize()
                self._retire()
                self.flag.zero_()
        self.retire = False
        for request in list(self.joining):  # a prefilled call's rows join before new calls
            self._join(request)
        waiting: list[_Request] = []
        admitted = False
        started = time.perf_counter()
        for request in incoming:
            if request.cancelled:
                request.cut = True
                continue
            # One call joins at a time while rows decode: their tokens keep coming meanwhile.
            if waiting or (admitted and self._occupied()):
                waiting.append(request)
                continue
            try:
                joined = self._admit(request)
            except DoesNotFit as exc:  # refused at its entry: this call alone, before it ran
                request.error = exc
                continue
            if joined:
                admitted = True
            else:
                waiting.append(request)
        self.debt = time.perf_counter() - started if admitted else 0.0
        with self.cond:
            self.pending.extendleft(reversed(waiting))
            if waiting and not self._occupied() and not admitted and not self.rebind:
                for request in waiting:  # nothing decodes that could leave: never spin on it
                    request.error = RuntimeError("[lm] decode admission made no progress")
                self.pending.clear()
            self.cond.notify_all()

    def _retire(self) -> None:
        n = self.n.tolist()
        done = self.done.tolist()
        for slot, row in enumerate(self.slots):
            if row is None or not (
                done[slot] or row.request.cancelled or row.request.error is not None
            ):
                continue
            request = row.request
            request.cut = request.cut or request.cancelled
            request.tokens[row.row] = self.out[slot, : n[slot]].clone()
            request.left -= 1
            request.ended = time.perf_counter()
            if not request.left:
                request.filled[1] = _filled()
                request.at = tuple(
                    now - then
                    for now, then in zip(
                        (self.captured[0], self.captured[1], self.eager), request.at, strict=True
                    )
                )
            self.slots[slot] = None
        # The rows above the holes move down, so a step runs the smallest bucket: their tokens
        # and, in the K/V, their blocks and state (`KvManager.replace`).
        kv = _set(self.kv)
        top = len(self.slots) - 1
        for slot in range(len(self.slots)):
            while top > slot and self.slots[top] is None:
                top -= 1
            if self.slots[slot] is None and top > slot:
                for tensor in self._rows():
                    tensor[slot].copy_(tensor[top])
                with self._kv_stream():  # a hole below the top is a row that just left
                    kv.replace(slot, top)
                self.kv_rows.discard(top)
                self.kv_rows.add(slot)
                self.slots[slot], self.slots[top] = self.slots[top], None
        for slot, row in enumerate(self.slots):
            if row is None:
                if slot in self.kv_rows:
                    with self._kv_stream():
                        kv.free(slot)
                    self.kv_rows.discard(slot)
                self._empty(slot)
        with self.cond:
            self.cond.notify_all()

    def _rows(self) -> Iterator[torch.Tensor]:
        """Every token buffer with a row per slot (the K/V's rows move in `KvManager`)."""
        yield from (self.ids, self.length, self.out, self.n, self.done, self.limit)
        yield from (self.stops, self.pad, self.counts, self.prompted, self.position)
        yield from (self.presence, self.frequency, self.repetition)
        yield from (self.temperature, self.top_p, self.seed, self.top_k, self.min_p)
        if self.draft is not None:
            yield self.hidden

    def _empty(self, slot: int) -> None:
        """A free slot as a finished row of length 0; the K/V holds no row there (length -1)."""
        self.done[slot] = True
        self.length[slot] = 0
        self.n[slot] = 0
        self.temperature[slot] = 0.0

    # ------------------------------------------------------------------ admission

    def _admit(self, request: _Request) -> bool:
        """`request` decided (`_enter`), then run: its prompt through the model's own forward
        into a cache of its own, unpadded, and its rows into free slots, with their first token;
        rows past the free slots join as slots free (`_join`). False: it waits for decoding rows
        to leave. Its prefill runs as every forward does: its refusal is the call's."""
        free = [slot for slot, row in enumerate(self.slots) if row is None]
        if request.draft is not None and request.draft is not self.draft:
            if len(free) < len(self.slots):
                return False  # another head's rows decode: it waits for them
            self._drop()  # the engine takes this head (its keys' shapes) when idle
            self.draft, self.rebind = request.draft, True
            return False  # admitted once its blocks are bound
        try:
            first = None if request.allowed is None else _ids(request.allowed, [], self.vocab)
            if request.ngram and not request.banned:  # admission may ask again
                for prompt in request.prompts:
                    index: dict[tuple[int, ...], set[int]] = {}
                    _grams(prompt, request.ngram, index, 0)
                    request.banned.append(sorted(_repeats(prompt, request.ngram, index)))
        except Exception as exc:  # the call's own constraint, asked before anything of it ran
            request.error = exc
            return True
        kwargs, mask, position = self._inputs(request)
        if not self._enter(request, position):
            return False
        self._unlease()  # a prefill runs as every forward does, nothing of the decode leased
        fresh = self._prefill(request, kwargs, mask, position, first)
        if why := self._misfit(fresh):
            self.misfit = f"its decode state is not what its config gives ({why})"
            request.error = RuntimeError(f"[lm] {type(self.model).__name__}: {self.misfit}")
            return True
        request.fresh, request.parked = fresh, list(range(len(request.tokens)))
        self.joining.append(request)
        self._join(request)
        return True

    def _join(self, request: _Request) -> None:
        """`request`'s rows that wait for a slot (`parked`) into the slots free now, each with its
        K/V committed (`_splice`): its prefill holds their prompts' keys and state until its last
        row joins. A row the K/V has no room for waits for decoding rows to leave; with none
        decoding the call is refused, with what it needs. A cancelled or failed call's rows go."""
        if request.cancelled or request.error is not None:
            request.cut = request.cut or request.cancelled
            request.parked = []
        free = [slot for slot, row in enumerate(self.slots) if row is None]
        rows = request.parked[: len(free)]
        joined = self._splice(request, _set(request.fresh), rows, free) if rows else 0
        request.parked = request.parked[joined:]
        if request.parked and not self._occupied() and request.error is None:
            request.error = DoesNotFit(self.refused)
            request.parked = []
        if not request.parked:
            request.fresh = None
            with suppress(ValueError):
                self.joining.remove(request)

    def _enter(self, request: _Request, position: torch.Tensor | None) -> bool:
        """Whether `request`'s rows can join, decided before its prefill: a free slot, and token
        buffers that hold its stops (`_tokens`, made at the first call, again only with no row
        decoding). Its K/V is committed as each row joins (`_join`). False: it waits for the rows
        decoding now to leave."""
        free = sum(row is None for row in self.slots)
        occupied = len(self.slots) - free
        request.waits = False
        if self.kv is not None and not free:
            request.waits = True  # slots free up as rows leave
            return False
        width = max(8, len(request.stops))
        if self.kv is None or width > self.width:
            if occupied:
                request.waits = True  # its stops' buffer is made again once these rows leave
                return False
            self._tokens(request.inputs["input_ids"].device, position, width)
        return True


    def _inputs(
        self, request: _Request
    ) -> tuple[dict[str, torch.Tensor | int | None], torch.Tensor, torch.Tensor | None]:
        """The prefill's arguments, its attention mask and positions: no forward."""
        model, inputs = self.model, request.inputs
        ids = inputs["input_ids"]
        kwargs: dict[str, torch.Tensor | int | None] = {
            key: value for key, value in inputs.items() if key != "input_ids"
        }
        mask = inputs.get("attention_mask")  # kwargs' own: it holds every input but the ids
        mask = torch.ones_like(ids) if mask is None else mask
        kwargs["attention_mask"] = mask
        position: torch.Tensor | None = None
        if self.positional:
            position = kwargs["position_ids"] = _lm(model)._prepare_position_ids_for_generation(
                ids, kwargs
            )
        if bool(mask.all()):  # as `generate`: no padding, no 2D mask
            kwargs["attention_mask"] = None
        if self.keep:
            kwargs["logits_to_keep"] = 1
        return kwargs, mask, position

    def _prefill(
        self,
        request: _Request,
        kwargs: dict[str, torch.Tensor | int | None],
        mask: torch.Tensor,
        position: torch.Tensor | None,
        first: torch.Tensor | None,
    ) -> _Prefill:
        """The prompt through the model's own forward into a cache of its own, and each row's
        first token, from what `first` (the constraint's ids for no tokens) allows."""
        started = time.perf_counter()
        model, ids = self.model, request.inputs["input_ids"]
        fresh = _StaticCache(self.text, request.prompt, request.prompt)
        prepared = _lm(model).prepare_inputs_for_generation(
            ids, past_key_values=fresh, use_cache=True, is_first_iteration=True, **kwargs
        )
        # Native allocation recovery runs underneath this single prefill invocation.
        with _taking(model) as taken:
            logits: torch.Tensor = model(**prepared, return_dict=True).logits[:, -1]
        hidden = taken[-1]
        with _disable_current_modes():
            scores = logits.float()
            if first is not None:
                allowed = torch.zeros(scores.shape[-1], dtype=torch.bool, device=scores.device)
                allowed[first.to(scores.device)] = True
                scores = scores.masked_fill(~allowed, float("-inf"))
            for row, banned in enumerate(request.banned):
                scores[row, torch.tensor(banned, dtype=torch.long, device=scores.device)] = float(
                    "-inf"
                )
            # Nothing is generated yet: of the penalties only repetition's prompt part reads.
            rows, vocab = int(ids.shape[0]), int(scores.shape[-1])
            prompted = sampler.prompted(ids, mask.to(ids.device), vocab)
            picking, penalties = _per_row(request.sampling, rows, scores.device)
            nothing = torch.zeros((1, vocab), dtype=torch.float32, device=scores.device)
            scores = sampler.penalized(scores, nothing, prompted.to(scores.device), *penalties)
            first = torch.zeros((rows,), dtype=torch.long, device=scores.device)
            token = sampler.pick(scores, *picking, first, request.sampling.temperature > 0)
            token = token.to(ids.device)
            drafted: tuple[_Keys, torch.Tensor] | None = None
            if self.draft is not None:
                drafted = self._drafted_prompt(ids, token, hidden, position)
        request.prefill_s += time.perf_counter() - started
        return _Prefill(fresh, mask, position, token, prompted, drafted)

    def _drafted_prompt(
        self,
        ids: torch.Tensor,
        token: torch.Tensor,
        hidden: torch.Tensor,
        position: torch.Tensor | None,
    ) -> tuple[_Keys, torch.Tensor]:
        """The head's keys for the prompt and the first token, as it attends to them while
        drafting: slot j from token j and the model's state for the token before it, so drafts
        read the whole context the head was trained on; and its state for the token after."""
        prompt, rows = int(ids.shape[1]), int(ids.shape[0])
        tokens = torch.cat([ids[:, 1:], token[:, None]], dim=1)
        slots = torch.arange(prompt + 1, device=ids.device)
        if position is not None:
            following = position[..., -1:] + 1
            at = torch.cat([position[..., 1:], following], dim=-1)
        else:
            at = slots[None, 1:]
        keys = _Keys([prompt + 1], prompt + 1)
        cache = Cache(layers=[keys])
        keys.cumulative_length = torch.ones((), dtype=torch.long, device=ids.device)
        embed = self.model.get_input_embeddings()
        rotary = _lm(self.model).get_decoder().rotary_emb
        # Its mask is explicit, so attention may materialize each query's scores over every slot:
        # queries go in pieces whose scores stay within DRAFT_SCORES, each writing its keys before
        # the next attends to them (a 23,388-token prompt asked 17.5 GB whole: 16 heads x L x L+1).
        heads = int(getattr(self.text, "num_attention_heads", 1))
        piece = max(1, DRAFT_SCORES // (rows * heads * (prompt + 1) * 4))
        drafted: torch.Tensor | None = None
        for start in range(0, prompt, piece):
            end = min(start + piece, prompt)
            allowed = (slots[None, :] >= 1) & (slots[None, :] <= slots[start + 1 : end + 1, None])
            mask = (
                allowed
                if self.closed is False
                else torch.where(allowed, self.open, self.closed).to(self.model.dtype)
            )
            mask = mask[None, None].expand(rows, 1, end - start, prompt + 1)
            these = embed(tokens[:, start:end]), hidden[:, start:end], at[..., start:end]
            drafted = _set(self.draft)(*these, cache, mask, rotary)
        last = _set(drafted)[:, -1:]
        return keys, last

    def _splice(
        self, request: _Request, fresh: _Prefill, rows: list[int], free: list[int]
    ) -> int:
        """`fresh`'s `rows` into slots `free`, as many as the K/V takes, and how many: each its
        prompt's keys and values unpadded into its blocks, which it commits for its prompt and its
        most new tokens (`KvManager.admit`, `reserve_ahead`), its gated-delta state into its slot,
        and its positions and first token. The call's decode begins with its first."""
        prompt = request.prompt
        kv = _set(self.kv)
        drafted = self.draft is not None
        stops = torch.full((self.stops.shape[1],), -1, dtype=torch.long)
        stops[: len(request.stops)] = torch.tensor(request.stops, dtype=torch.long)
        firsts = fresh.token.tolist() if request.constrained else []
        counts = fresh.mask.bool().sum(-1).tolist()
        joined = 0
        with _disable_current_modes():
            for row, slot in zip(rows, free, strict=False):
                count = int(counts[row])
                try:
                    with self._kv_stream():
                        # no scratch: lm's verify keeps each token's states itself (`_Verify`)
                        at = kv.admit(slot, count, own_slot=True)
                        self.kv_rows.add(slot)
                        kv.reserve_ahead(slot, request.max_new + DEPTH + 2)
                except Exception as exc:
                    if not _short_of_room(exc):
                        raise
                    _unwind(exc)
                    if slot in self.kv_rows:
                        with self._kv_stream():
                            kv.free(slot)
                        self.kv_rows.discard(slot)
                    self.refused = self._refusal(request, count, exc)
                    break
                real = fresh.mask[row].bool()
                with self._kv_stream():
                    for i, layer in enumerate(_layers(fresh.cache)):
                        if isinstance(layer, _Keys):
                            keys = layer.keys[row][:, real].transpose(0, 1)
                            values = layer.values[row][:, real].transpose(0, 1)
                            kv.write(self.kv_layer[i], keys, values, at)
                        else:
                            d = self.delta[i]
                            kv.state(0)[slot, d].copy_(_set(layer.conv_states[0])[row])
                            kv.state(1)[slot, d].copy_(_set(layer.recurrent_states[0])[row])
                    if drafted:
                        keys_, last = _set(fresh.drafted)
                        if keys_.keys is not None:  # position p's from token p and state p - 1
                            lead = prompt - count + 1
                            head = kv.layout.attn_layers - 1
                            kv.write(
                                head,
                                keys_.keys[row][:, lead : prompt + 1].transpose(0, 1),
                                keys_.values[row][:, lead : prompt + 1].transpose(0, 1),
                                kv.token_slots(slot, 1, count),
                            )
                        self.hidden[slot].copy_(last[row])
                self.length[slot] = count
                if fresh.position is not None:
                    self.position[slot].copy_(fresh.position[..., row, -1:] + 1)
                token = fresh.token[row]
                self.ids[slot, 0] = token
                self.out[slot, 0] = token
                self.n[slot] = 1
                self.limit[slot] = request.max_new
                self.stops[slot].copy_(stops)
                self.pad[slot] = request.pad
                sampling = request.sampling
                self.presence[slot] = sampling.presence
                self.frequency[slot] = sampling.frequency
                self.repetition[slot] = sampling.repetition
                self.temperature[slot] = sampling.temperature
                self.top_p[slot], self.top_k[slot], self.min_p[slot] = sampling.kept()
                self.seed[slot] = sampling.seed
                self.counts[slot].zero_()
                self.counts[slot, token] = 1.0
                self.prompted[slot].copy_(fresh.prompted[row])
                self.done[slot] = bool((self.stops[slot] == token).any()) or request.max_new <= 1
                held = self.slots[slot] = _Row(request, row, count)
                if firsts:
                    held.generated = [firsts[row]]
                joined += 1
        if joined and not request.began:
            request.began = time.perf_counter()
            request.at = (self.captured[0], self.captured[1], self.eager)
            request.filled[0] = _filled()
        self.retire = self.retire or bool(self.done[free[:joined]].any())
        return joined

    def _refusal(self, request: _Request, count: int, exc: BaseException) -> str:
        """`DoesNotFit`'s line for a row whose K/V the card had no room for, with none decoding."""
        kv = _set(self.kv)
        tokens = count + request.max_new + DEPTH + 2
        need = -(-tokens // kv.layout.block_tokens) * kv.layout.block_bytes + kv.layout.state_bytes
        device = request.inputs["input_ids"].device.index or 0
        return (
            f"{type(self.model).__name__} decode: a row of {count} + {request.max_new} tokens: "
            f"{no_room(device, 'its keys, values and state', need, exc)}; run it smaller "
            "(fewer max_new_tokens or a shorter prompt), or rent a larger card: "
            "`cozy run … --on new`"
        )


    def _misfit(self, fresh: _Prefill) -> str:
        """How a row of the prefill's state differs from what the K/V holds (`engine.kv`'s
        layout, from the config), "" when it does not: a write into a block or slot of another
        shape or dtype would cast it silently."""
        kv = _set(self.kv)
        layout = kv.layout
        pairs: list[tuple[object, tuple[object, ...], tuple[object, ...]]] = []
        attention = (layout.kv_heads, layout.head_dim, layout.dtype)
        for i, layer in enumerate(_layers(fresh.cache)):
            if isinstance(layer, _Keys):
                for t in (layer.keys, layer.values):
                    pairs.append((i, (int(t.shape[1]), int(t.shape[3]), t.dtype), attention))
            else:
                for state, spec in ((layer.conv_states.get(0), layout.states[0]),
                                    (layer.recurrent_states.get(0), layout.states[1])):  # fmt: skip
                    found = ("nothing",) if state is None else (*state.shape[1:], state.dtype)
                    pairs.append((i, found, (*spec.shape[1:], spec.dtype)))
        if self.draft is not None:
            keys, last = _set(fresh.drafted)
            if keys.keys is not None:  # `Draft` writes keys; a head of none reads none
                t = keys.keys
                pairs.append(("head", (int(t.shape[1]), int(t.shape[3]), t.dtype), attention))
            held = (*self.hidden.shape[1:], self.hidden.dtype)
            pairs.append(("head", (*last.shape[1:], last.dtype), held))
        for where, found, held in pairs:
            if found != held:
                return f"layer {where}: {found} a row, held {held}"
        return ""

    def _tokens(self, device: torch.device, position: torch.Tensor | None, stops: int) -> None:
        """The engine's K/V (`engine.kv`: every attention layer's, the draft head's after them,
        each gated-delta layer's conv and recurrent state in a row's slot) and its token buffers
        for the most rows a step decodes (`BUCKETS`), made once a head: nothing here grows with a
        call's length, so no graph is recorded again for one. The K/V reserves address space only;
        a row's pages are mapped as it joins and released as it leaves."""
        from tensorfs.varena import _runtime as varena

        from cozy_runtime.engine.kv import KvLayout, KvManager, StateSpec

        self._drop()
        dtype, long, f32 = self.model.dtype, torch.long, torch.float32
        keys = [shape for shape in self.shapes if shape[0] == "keys"]
        deltas = [shape for shape in self.shapes if shape[0] == "state"]
        states: tuple[StateSpec, ...] = ()
        if deltas:
            _, conv, recurrent = deltas[0]
            states = (
                StateSpec((len(deltas), *conv), dtype),
                StateSpec((len(deltas), *recurrent), f32),
            )
        _, heads, width, _ = keys[0]
        layout = KvLayout(
            attn_layers=len(keys) + (self.draft is not None),
            kv_heads=heads,
            head_dim=width,
            dtype=dtype,
            states=states,
            depth=DEPTH,
        )
        context = int(getattr(self.text, "max_position_embeddings", 0) or 0) or 1 << 15
        rows = BUCKETS[-1]
        self.kv_stream = torch.cuda.Stream(device)
        with self._kv_stream():
            self.kv = KvManager(
                varena.Device(device.index or 0), layout, rows, -(-context // layout.block_tokens)
            )
        self.capacity = self.kv.ceiling
        lead = tuple(position.shape[:-2]) if position is not None else ()
        fields: dict[str, _Spec] = {
            "ids": ((rows, 1), long, 0),
            "length": ((rows,), long, 0),
            "out": ((rows, self.capacity), long, 0),
            "n": ((rows,), long, 0),
            "done": ((rows,), torch.bool, True),
            "limit": ((rows,), long, 0),
            "stops": ((rows, stops), long, -1),
            "pad": ((rows,), long, 0),
            "presence": ((rows,), f32, 0),
            "frequency": ((rows,), f32, 0),
            "repetition": ((rows,), f32, 1),
            "temperature": ((rows,), f32, 0),
            "top_p": ((rows,), f32, 1),
            "top_k": ((rows,), long, 0),
            "min_p": ((rows,), f32, 0),
            "seed": ((rows,), long, 0),
            # each token's count in the row's output, and its prompt's tokens (the penalties')
            "counts": ((rows, self.vocab), f32, 0),
            "prompted": ((rows, self.vocab), torch.bool, False),
            "position": ((rows, *lead, 1), long, 0),
            # each step position's allowed tokens, as bits the host writes (`_fill`)
            "allow": ((DEPTH + 1, rows, -(-self.vocab // 8)), torch.uint8, 255),
            "head_lengths": ((rows,), torch.int32, -1),
            "head_slots": ((rows, DEPTH + 1), long, -1),
        }
        if self.draft is not None:
            fields["hidden"] = ((rows, 1, int(self.text.hidden_size)), dtype, 0)
            fields["drafts"] = ((rows, DEPTH), long, 0)  # a step's drafts, between its parts
            fields["dhidden"] = fields["hidden"]
        for name, (shape, kind, fill) in fields.items():
            setattr(self, name, torch.full(shape, fill, dtype=kind, device=device))
        widest = max([DEPTH + 1, *(int(shape[1][1]) for shape in deltas)])
        self.steps = torch.arange(DEPTH + 1, device=device)
        self.rows_index = torch.arange(widest, device=device)
        self.bit = torch.arange(8, dtype=torch.uint8, device=device)
        pinned = device.type == "cuda"
        self.flag = torch.zeros((rows,), dtype=torch.bool, pin_memory=pinned)
        # What a constrained step's host reads back (`_took`, `_drafted`) and writes (`_fill`).
        self.took = torch.zeros(((DEPTH + 2) * rows,), dtype=torch.long, pin_memory=pinned)
        self.dtook = torch.zeros((DEPTH * rows,), dtype=torch.long, pin_memory=pinned)
        self.allow_host = torch.full(
            tuple(self.allow.shape), 255, dtype=torch.uint8, pin_memory=pinned
        )
        self.slots = [None] * rows
        for slot in range(rows):
            self._empty(slot)
        self.cache_device, self.width, self.views = device, stops, {}
        modules = [*_lm(self.model).get_decoder().modules()]
        if self.draft is not None:
            modules += [*self.draft.modules()]
        found = {id(c): c for m in modules if (c := getattr(m, "config", None)) is not None}
        self.configs = [c for c in found.values() if hasattr(c, "_attn_implementation_internal")]

    @contextmanager
    def _kv_stream(self) -> Iterator[None]:
        """The K/V's own work (rows joining, leaving, moving) on its stream, after the current
        stream's and before what it queues next: `KvManager` releases pages behind its caller's
        stream, which is never the default one."""
        stream = _set(self.kv_stream)
        current = torch.cuda.current_stream(stream.device)
        stream.wait_stream(current)
        with torch.cuda.stream(stream):
            yield
        current.wait_stream(stream)

    @contextmanager
    def _attending(self) -> Iterator[None]:
        """The model's and its head's attention, in the forward in the body, through the engine's
        K/V (`engine.attention`, `_Attending`): their configs name the paged implementation."""
        from cozy_runtime.engine import attention

        name = attention.register()
        before = [getattr(c, "_attn_implementation_internal") for c in self.configs]
        token = attention.PAGED.set(self.attend)
        for config in self.configs:
            setattr(config, "_attn_implementation_internal", name)
        try:
            yield
        finally:
            for config, was in zip(self.configs, before, strict=True):
                setattr(config, "_attn_implementation_internal", was)
            attention.PAGED.reset(token)

    def _tables(self, size: int, span: int, head: bool = False) -> None:
        """Where each of the first `size` rows' `span` queries writes its K/V, from its length and
        block table, on the device: the model's at its length (`slot_mapping`), the draft head's a
        position ahead (`head_lengths`, `head_slots`). A row not decoding (-1) writes nothing."""
        kv = _set(self.kv)
        tokens = kv.layout.block_tokens
        lengths = kv.lengths[:size]
        if head:
            ahead = self.head_lengths[:size]
            ahead.copy_(torch.where(lengths >= 0, lengths + 1, -1))
            lengths = ahead
        at = lengths.long()[:, None] + self.steps[:span]
        index = (at // tokens).clamp(0, kv.max_blocks - 1)
        entry = kv.block_table[:size].long().gather(1, index)
        live = (lengths[:, None] >= 0) & (entry >= 0)
        slots = torch.where(live, entry * tokens + at % tokens, -1)
        (self.head_slots if head else kv.slot_mapping)[:size, :span].copy_(slots)

    def _lengths(self, size: int) -> None:
        """The K/V's lengths of the first `size` rows after a step: each decoding row's own."""
        lengths = _set(self.kv).lengths[:size]
        lengths.copy_(torch.where(lengths >= 0, self.length[:size].to(lengths.dtype), -1))

    # ------------------------------------------------------------------ steps

    def _view(self, size: int) -> _PagedCache:
        """The step's cache over the first `size` rows, which graphs read: each attention layer's
        K/V through the paged attention (`_Paged`), each gated-delta layer's state in the rows'
        slots (`KvManager.state`, slot i the row at i), the head's K/V layer last."""
        view = self.views.get(size)
        if view is None:
            kv = _set(self.kv)
            layers: list[CacheLayerMixin | LinearAttentionCacheLayerMixin] = []
            for i, shape in enumerate(self.shapes):
                if shape[0] == "keys":
                    layers.append(_Paged(self.attend, self.kv_layer[i]))
                    continue
                state = _State(**self.layer_kwargs[i])
                conv = kv.state(0)[:size, self.delta[i]]
                state.dtype, state.device = conv.dtype, conv.device
                state.conv_states[0] = conv
                state.recurrent_states[0] = kv.state(1)[:size, self.delta[i]]
                state.conv_kernel_size[0] = int(conv.shape[-1])
                state.is_conv_states_initialized[0] = True
                state.has_previous_state[0] = True
                state.is_recurrent_states_initialized[0] = True
                layers.append(state)
            view = _PagedCache(layers=layers)
            if self.draft is not None:
                head = _Paged(self.attend, kv.layout.attn_layers - 1, head=True)
                view.draft = Cache(layers=[head])
            self.views[size] = view
        return view

    def _steps(self) -> None:
        """Steps until a row is done or a call waits to join, queued `LAG` ahead of the host; a
        constrained batch's each caught up with before the next, whose mask its tokens decide."""
        cuda = _set(self.cache_device).type == "cuda"
        if cuda and (self.where != self.layout or self.landed != LANDED[0] or self.stale):
            # The weights moved, or a kernel landed (the graphs recorded its stand-in).
            self._forget()
            self.layout, self.landed, self.stale = self.where, LANDED[0], False
        queued: deque[torch.cuda.Event] = deque()
        size = self._bucket()
        held = [slot for slot, row in enumerate(self.slots[:size]) if row is not None]
        constrained = self._constrained()
        lag = 0 if constrained else LAG
        began = time.perf_counter()

        def ended() -> bool:  # a held row is done (a free slot in the bucket is, always)
            flags = self.flag[:size].tolist()
            return any(flags[slot] for slot in held)

        def joining() -> bool:  # a waiting call joins once decoding had its prefill's time
            # (one waiting for rows to leave, and those behind it, ask at a row's end instead)
            return (
                bool(self.pending)
                and not self.pending[0].waits
                and time.perf_counter() - began >= self.debt
            )

        with _disable_current_modes():
            while True:
                self._one(size, cuda)
                if not cuda:
                    if constrained:
                        self._took(size)
                    if ended() or joining() or self._cancelled():
                        break
                    self._tell()
                    continue
                event = torch.cuda.Event()
                event.record()
                queued.append(event)
                if len(queued) > lag:
                    queued.popleft().synchronize()
                    if constrained:
                        self._took(size)
                    if ended() or joining() or self._cancelled():
                        break
                    self._tell()
            if cuda:
                torch.cuda.current_stream().synchronize()
        self.retire = True

    def _fill(self, size: int, at: int) -> None:
        """What each held row may take at position `at` of the step (0: its next token; k: the
        one after its first k drafts), packed on the host from its constraint and copied to the
        device ahead of the part that reads it. A row with none, done, or whose drafts stopped
        takes any; one whose constraint raises fails its call alone."""
        host = self.allow_host[at, :size]
        for slot, row in enumerate(self.slots[:size]):
            free = True
            if row is not None and row.generated is not None:
                if not at:
                    row.drafts = []
                tokens, stops = [*row.generated, *row.drafts[:at]], row.request.stops
                if not row.ended() and not any(t in stops for t in row.drafts[:at]):
                    try:  # a row with `generated` has its call's constraint or n-gram bans
                        self._permit(row, tokens, host[slot])
                        free = False
                    except Exception as exc:  # the call's own constraint: only it fails
                        row.request.error = row.request.error or exc
            if free:
                host[slot].fill_(255)
        self.allow[at, :size].copy_(host, non_blocking=True)

    def _permit(self, row: _Row, tokens: list[int], out: torch.Tensor) -> None:
        """`out` (`_pack`'s bits): what `row` may take after `tokens`, by its call's constraint
        (any without one) less the tokens that would repeat one of its n-grams. Ids a constraint
        returns again (a grammar's state met again) are packed once (`packed`)."""
        request = row.request
        said = None if request.allowed is None else request.allowed.allowed(tokens)
        if said is None:
            out.fill_(255)
        elif (kept := self.packed.get(id(said))) is not None and kept[0] is said:
            _keep(self.packed, id(said), kept, Grammar.STATES)
            out.copy_(kept[1])
        else:
            _pack(_checked(said, len(tokens), self.vocab), out)
            if isinstance(said, torch.Tensor):  # one kept and returned again; a sequence is new each time
                _keep(self.packed, id(said), (said, out.clone()), Grammar.STATES)
        if request.ngram:
            n, held = request.ngram, [*request.prompts[row.row], *_set(row.generated)]
            row.counted = _grams(held, n, row.grams, row.counted)
            sequence = [*request.prompts[row.row], *tokens]
            drafted: dict[tuple[int, ...], set[int]] = {}  # n-grams ending in this step's drafts
            _grams(sequence, n, drafted, max(0, len(held) - n + 1))
            _ban(_repeats(sequence, n, row.grams) | _repeats(sequence, n, drafted), out)

    def _drafted(self, size: int, part: int) -> None:
        """Each constrained row's draft from part `part`, which the host has caught up with."""
        drafted = self.dtook[part * size : (part + 1) * size].tolist()
        for slot, row in enumerate(self.slots[:size]):
            if row is not None and row.generated is not None:
                row.drafts.append(drafted[slot])

    def _took(self, size: int) -> None:
        """Each constrained row's tokens from the step the host has just caught up with (`took`:
        how many it made, then each one)."""
        took = self.took[: (DEPTH + 2) * size].tolist()
        for slot, row in enumerate(self.slots[:size]):
            if row is not None and row.generated is not None:
                row.generated += [took[(1 + j) * size + slot] for j in range(took[slot])]

    def _tell(self) -> None:
        """Each held call's progress, its longest row's tokens (drafting decodes several a step),
        and each row's tokens so far where its words stream. The flag check has just caught the
        host up with the device, so reading `n` and `out` waits on nothing."""
        counts: list[int] = []

        def tokens(request: _Request) -> int:
            if not counts:
                counts.extend(self.n.tolist())
            held = (
                n
                for n, row in zip(counts, self.slots, strict=False)
                if row and row.request is request
            )
            return max(held, default=0)

        def rows(request: _Request) -> list[list[int]]:
            if not counts:
                counts.extend(self.n.tolist())
            held = {
                row.row: slot
                for slot, row in enumerate(self.slots)
                if row is not None and row.request is request
            }
            return [
                self.out[held[i], : counts[held[i]]].tolist()
                if i in held
                else ([] if done is None else done.tolist())
                for i, done in enumerate(request.tokens)
            ]

        for request in {id(row.request): row.request for row in self.slots if row}.values():
            request.tell(partial(tokens, request), partial(rows, request))

    def _read(self, drafted: bool) -> tuple[torch.nn.Module, ...]:
        """The modules a step reads: the model, and its head when the step drafts."""
        return (self.model, _set(self.draft)) if drafted else (self.model,)

    @contextmanager
    def _on_domain(self) -> Iterator[None]:
        """The domain's side stream, where a step is captured and its first step measured, joined
        back before the step's weights can go idle."""
        current = torch.cuda.current_stream()
        side = None
        domain = _set(self.domain)
        try:
            with ExitStack() as stack:
                domain.initialize()
                side = cast("torch.cuda.Stream", domain.stream)  # initialize() sets it
                side.wait_stream(current)
                domain.prime_blas()
                stack.enter_context(torch.cuda.stream(side))
                try:
                    yield
                finally:
                    current.wait_stream(side)  # before the discovery lease records last use
        finally:
            if side is not None:
                current.wait_stream(side)

    def _opens(self) -> bool:
        """Whether a step may warm up or record now. Its graph domain opens first, which a kernel
        still building refuses (`Building`) as it refuses a capture: the step decodes eagerly until
        a kernel lands, and its call never fails for it (cr-342)."""
        if self.building == LANDED[0]:
            return False
        try:
            _set(self.domain).initialize()
        except Building:
            self.building = LANDED[0]
            return False
        return True

    def _replay(self, size: int, cuda: bool) -> None:
        drafted = self._drafting(size)
        constrained = self._constrained()
        part = self.part if drafted else -1
        key = (size, drafted, self._sampled(), constrained, part)
        placed = PLACED.get(self.model)
        unaccounted = (
            cuda
            and key not in self.graphs
            and key not in self.unrecorded
            and any(m in PLACED and not PLACED[m][0].graph_accounted() for m in self._read(drafted))
        )
        fixed = self.contracts.get(drafted)
        if fixed is None:  # its hooks and forwards, walked once per recorded layout
            fixed = self.contracts[drafted] = _capture_contract(
                self.model, self.draft if drafted else None
            )
        reason = (
            ("driver/context memory is unmeasured" if unaccounted else "")
            or (self.domain.refusal() if cuda and self.domain is not None else "")
            or fixed
        )
        if drafted and (reason or key in self.unrecorded):
            if reason and key not in self.unrecorded:
                code = None if reason == fixed else "capture_refused"
                self._unrecorded(key, f"{reason}: it decodes plainly", code)
            self.undrafted = True
            return self._replay(size, cuda)
        if cuda and reason:
            if key not in self.unrecorded:
                self._unrecorded(key, reason, None if reason == fixed else "capture_refused")
            return self._eager(size, reason)
        graph = self.graphs.get(key)
        if graph is not None:
            why = self._lease(drafted)
            if not why:  # its weights and pool are held: the replay maps nothing
                vt.replay(graph, stream=torch.cuda.current_stream(_set(self.domain).device))
                self.replayed += 1
                self.rung = "graph"
                return None
            # varena refused them before anything ran: the key is eager, judged again later
            self._discard(key, graph, torch.cuda.current_stream(_set(self.domain).device))
            self._eager_key(key, why)
            return self._eager(size, why)
        if not cuda:
            return self._eager(size)
        if key in self.unrecorded:
            return self._eager(size, self.unrecorded[key])
        if not self._opens():
            return self._eager(size, "a kernel is building")
        domain = _set(self.domain)  # a CUDA step's (`_one`)
        if key not in self.warmed:
            # The caller's actual next token, run as written with nothing leased (its weights
            # enter as any forward's do): its working peak, as varena measured it, is what the
            # key's graph is judged by and its capture pool's first guess.
            self._unlease()
            with self._on_domain(), _measured(domain.device) as peak, self._gathering():
                self._advance(size)
            self.warmed.add(key)
            if peak:
                self.peaks[key] = peak[0]
            self.eager += 1
            self.rung = "first step, measured"
            return None
        if key not in self.peaks:  # a step never measured is never recorded
            why = "its first step's working memory was not measured"
            self._unrecorded(key, why)
            return self._eager(size, why)
        return self._record(key, size, drafted, placed)

    def _eager(self, size: int, why: str = "") -> None:
        """One step as written, nothing of it leased: its weights enter as any forward's do,
        through the model's plan, which keeps what the card holds of a loop (`_gathering`)."""
        self._unlease()
        with self._gathering():
            self._advance(size)
        self.eager += 1
        self.rung = f"eager ({why})" if why else "eager"

    def _gathering(self) -> AbstractContextManager[None]:
        """An eager step's: an embedding table of the model whose page left gathers the rows it
        reads on the host, never filling whole again (`Model.gathering`, cr-332)."""
        placed = PLACED.get(self.model)
        return placed[0].gathering(placed[1]) if placed is not None else nullcontext()

    def _eager_key(self, key: _Key, why: str, short: int = 1) -> None:
        """`key`'s graph does not fit (`_Judge`: by `short` bytes), or varena refused its lease:
        it decodes eagerly, said in the record with the numbers, and is judged again once the card
        has that much more room, measured with nothing of the decode leased."""
        self._unlease()
        self.judged[key] = (self._room(), max(short, 1))
        self._unrecorded(key, why, None)

    def _record(
        self, key: _Key, size: int, drafted: bool, placed: tuple[Residency, str] | None
    ) -> None:
        """`key`'s graph, judged before it is captured (`_Judge`): its weights leased, then judged
        with what the lease pinned, and its step captured, then checked against this token's
        eager step, which serves. A graph that does not fit leaves the key eager. A capture that
        fails otherwise is the recording's, never the call's (as a component's graph): the key
        decodes eagerly, and the record says why."""
        from cozy_runtime.internal.execution import rungs  # with Torch

        domain = _set(self.domain)
        self._unlease()
        before = vt.room(domain.device)
        if why := self._lease(drafted):
            self._eager_key(key, why)
            return self._eager(size, why)
        pool_key = self._pool_key(key)
        reserve = max(vt.pool_bytes(self.peaks[key]), rungs.pool(pool_key) if pool_key else 0)
        check = self._check_bytes(size, drafted, None)
        judge = _Judge(self, before, self.peaks[key], check, pool_key)
        if why := judge(reserve):
            self._eager_key(key, why, judge.short)
            return self._eager(size, why)
        capturing = time.perf_counter()
        # Its ops and kernels, set beside the step as written's when its check differs, and
        # what it writes in place, which its check puts back (cr-283).
        writes: list[torch.Tensor] = []
        traced, ops = observe.trace(torch, writes)
        launch = _STEPS[0]

        def step() -> None:
            with cast("AbstractContextManager[object]", traced):
                self._advance(size)

        graph: torch.cuda.CUDAGraph | None = None
        why, judged = "", False
        try:
            with self._on_domain(), request_path.known("cozy_runtime.lm._Engine._replay"):
                if self.pool is None:
                    self.pool = (
                        placed[0].graph_pool(placed[1])
                        if placed is not None
                        else Pool(torch.cuda.graph_pool_handle())
                    )
                pool = self.pool
                try:
                    # Refused, an allocation comes back here (`capture` regrows a short pool),
                    # never from varena's last resort.
                    with _fallback():
                        graph, why = capture(
                            torch, step, pool, domain, reserve, alone=not self.graphs, fits=judge
                        )
                except (_OverCard, Exhausted) as no:  # its pool, judged or prepared, has no room
                    why, judged = str(no), True
                except Exception as exc:  # the recording's, never the call's
                    if isinstance(exc, (Cancelled, Yielded)) or fault(exc) in (
                        Fault.STICKY,
                        Fault.DEVICE,
                    ):
                        raise
                    _unwind(exc)
                    if is_capacity(exc) or isinstance(exc, DoesNotFit):
                        _answered(exc)
                    why = f"its capture failed: {type(exc).__name__}: {exc}".splitlines()[0][:300]
                    self._unrecorded(key, why)  # a safeguard: said once, counted in the record
                    why = ""
                if graph is not None and pool.prepared > reserve:
                    self.regrown.append((reserve, pool.prepared))
        except Building:  # eager until a kernel lands
            graph, why, self.building = None, "a kernel is building", LANDED[0]
        finally:
            if graph is None:
                writes.clear()  # no graph to check: none of the step's state stays held
                if not self.graphs and self.pool is not None:
                    close_pool(self.pool, domain.device)
                    self.pool = None
        self.captured[0] += 1
        self.captured[1] += time.perf_counter() - capturing
        if graph is not None:
            self.graphs[key] = graph
            # A capture's end ends the pool's use: held again for its replays while its pages
            # are still mapped, so no replay maps.
            self.pooled = False
            self._pool(True)
            self.recorded += 1
            if pool_key:  # the measured pool: the next process judges by it
                rungs.put(pool_key, _set(self.pool).mapped)
            try:
                vt.pool_scratch(_set(self.pool).handle, (), domain.device)
            except (RuntimeError, ValueError) as exc:  # A live allocation: it stays mapped.
                safeguard("scratch_declined", str(exc))
            # What its check holds beside the pool, now that its writes are known.
            judge.check = self._check_bytes(size, drafted, writes)
            if why := judge(_set(self.pool).mapped):
                self._discard(key, graph, torch.cuda.current_stream(domain.device))
                writes.clear()
                self._eager_key(key, why, judge.short)
                return self._eager(size, why)
            try:
                self._verify(key, graph, size, ops, launch, writes)
            finally:
                writes.clear()  # its check is over, however it ended
            if key in self.graphs:
                self.rung = "graph, captured and checked"
            elif key in self.unrecorded:
                self.rung = f"eager ({self.unrecorded[key]})"
            return None
        if judged:
            self._eager_key(key, why, judge.short)
        elif why and self.building != LANDED[0]:
            self._unrecorded(key, why)
        return self._eager(size, why or self.unrecorded.get(key, ""))

    def _pool_key(self, key: _Key) -> str:
        """Where this machine keeps the pool `key`'s graph measured (`execution.rungs`): the
        model, the slots' layout and the key, and the request's models ("" outside a request)."""
        from cozy_runtime.internal.execution import rungs  # with Torch
        from cozy_runtime.internal.execution.catch import CURRENT as REQUEST

        request = REQUEST.get()
        if request is None:
            return ""
        site = f"lm:{type(self.model).__name__}:{self.capacity}:{self.width}:{key}"
        return rungs.key(site, sorted(request.current), _set(self.domain).device)

    def _check_bytes(self, size: int, drafted: bool, writes: list[torch.Tensor] | None) -> int:
        """What a new graph's check holds beside its pool (`_verify`): the state its step writes,
        saved and replayed (`_Before`), every K/V layer's token slots a step reaches, and each
        model layer's K/V at a row's length (`_written`). Before its capture (`writes` None), no
        tensor beyond that state."""
        kv = _set(self.kv)
        state = self._token_state(drafted, size)
        others = [] if writes is None else _beyond(writes, state, [kv.blocks])[1]
        whole = sum(t.nbytes for t in state)
        token = kv.layout.block_bytes // kv.layout.block_tokens  # every layer's K and V
        model = token * len(self.kv_layer) // kv.layout.attn_layers
        return 2 * whole + sum(t.nbytes for t in others) + size * (STEP_SLOTS * token + model)

    def _token_state(self, drafted: bool, size: int) -> list[torch.Tensor]:
        """What a step of `size` rows writes but its K/V, whose positions it writes before reading:
        its tokens, lengths and slots, its rows' gated-delta state; a drafted step's head state,
        drafts and the head's tables too."""
        kv = _set(self.kv)
        state = [self.ids, self.out, self.n, self.done, self.counts, self.length, self.position]
        state += [kv.lengths, kv.slot_mapping]
        if drafted:
            state += [self.hidden, self.drafts, self.dhidden, self.head_lengths, self.head_slots]
        state += [kv.state(i)[:size] for i in range(len(kv.layout.states))]
        return state

    def _written(self, size: int, at: torch.Tensor) -> list[torch.Tensor]:
        """Each row's K and V of every model layer at its position `at`, from its blocks."""
        kv = _set(self.kv)
        tokens = kv.layout.block_tokens
        index = (at // tokens).clamp(0, kv.max_blocks - 1)[:, None]
        entry = kv.block_table[:size].long().gather(1, index)[:, 0].clamp(min=0)
        layers = len(self.kv_layer)
        return [kv.blocks[entry, :layers, :, at % tokens]]


    def _before(self, state: list[torch.Tensor], writes: list[torch.Tensor], size: int) -> _Before:
        """A check's snapshot: `state` and any other device tensor its capture wrote, whole, and
        of the K/V blocks the token slots a step reaches in each row, `STEP_SLOTS` from its length
        (the model's at its length, the head's a position ahead)."""
        kv = _set(self.kv)
        _, others = _beyond(writes, state, [kv.blocks])
        return _Before([*state, *others], kv, size)

    def _verify(
        self,
        key: _Key,
        graph: torch.cuda.CUDAGraph,
        size: int,
        ops: list[tuple[object, ...]],
        launch: Launch | None,
        writes: list[torch.Tensor],
    ) -> None:
        """A new graph's one check: its replay, then this token's eager step from the same
        state, last so that what serves the call is eager's. The same state is all the step writes
        on the device (`writes`, its capture's), not only `_token_state`: the draft head's keys and
        values, which a replay writes ahead of the eager step, made Klein's check fail a sound
        graph (cr-283). Of a key buffer that is the slots a step reaches, never the buffer
        (`_Before`), and all of it goes as the check ends. What it holds beside the pool was judged
        before its capture (`_Judge`, `_check_bytes`). As a component graph's check (graphs'
        rule, cr-333), it is refused rather than given varena's last resort: refused, the graph
        goes and the token decodes eagerly from the state before it, the key judged again once the
        card has the room the refused allocation was short of. A difference discards the graph and explains itself (`_explain`)."""
        domain = _set(self.domain)
        state = self._token_state(key[1], size)
        stream = torch.cuda.current_stream(domain.device)
        start: _Before | None = None
        replayed: list[torch.Tensor] = []
        try:
            try:
                with _fallback():
                    at = self.length[:size].clone()
                    start = self._before(state, writes, size)
                    vt.replay(graph, stream=stream)  # its pool is held since its capture
                    replayed = [*(value.clone() for value in state), *self._written(size, at)]
                    start.restore()
                    self._advance(size)
            except Exception as exc:
                if not _short_of_room(exc):
                    raise
                short = getattr(exc, "requested_bytes", 0) - self._room()
                if start is not None:
                    start.restore()
                    start.release()
                replayed.clear()
                self._discard(key, graph, stream)
                why = no_room(domain.device, "its graph's check", 0, exc)
                _unwind(exc)
                _answered(exc)
                self._eager_key(key, f"{why}: it decodes eagerly", short)
                self._eager(size, why)
                return
            self.eager += 1
            why = vt.replay_differs([*state, *self._written(size, at)], replayed)
            if why:
                said = self._explain(key, size, state, start, at, replayed, ops, launch, graph)
                why += "; " + said
                self._discard(key, graph, stream)
                self._unrecorded(key, why, "replay_mismatch")
            elif _graphs.AUDIT and start.beyond(len(state))[0]:
                try:
                    self.partial.append(self._partial(size, state, start, at, graph, stream))
                except Exception as exc:  # a dev machine's diagnosis never fails its call
                    # A device fault, a yield or a cancel is the call's, never the diagnosis's.
                    if isinstance(exc, (Cancelled, Yielded)) or fault(exc) in (
                        Fault.STICKY,
                        Fault.DEVICE,
                    ):
                        raise
                    said = "no room to check without them" if is_capacity(exc) else "failed"
                    self.partial.append(f"{said}: {type(exc).__name__}: {exc}"[:200])
                    _unwind(exc)
                    start.restore()
                    self._advance(size)
        finally:
            if start is not None:
                start.release()
            replayed.clear()

    def _partial(
        self,
        size: int,
        state: list[torch.Tensor],
        start: _Before,
        at: torch.Tensor,
        graph: torch.cuda.CUDAGraph,
        stream: torch.cuda.Stream,
    ) -> str:
        """A dev machine's record of what the check before cr-283 would have found: the replay,
        then the eager step with only `_token_state` put back. The last run serves as before."""
        here = [*(value.clone() for value in state), *self._written(size, at)]
        start.restore()
        vt.replay(graph, stream=stream)
        start.restore(len(state))
        self._advance(size)
        old = [*(value.clone() for value in state), *self._written(size, at)]
        start.restore()
        self._advance(size)
        more, extra = start.beyond(len(state))
        said = vt.replay_differs(here, old) or "the same"
        return f"{more} more tensors put back ({extra >> 10} KiB); without them: {said}"

    def _explain(
        self,
        key: _Key,
        size: int,
        state: list[torch.Tensor],
        start: _Before,
        at: torch.Tensor,
        replayed: list[torch.Tensor],
        ops: list[tuple[object, ...]],
        launch: Launch | None,
        graph: torch.cuda.CUDAGraph,
    ) -> str:
        """Why a step's replay differed from it as written (tracker design
        cuda-graph-correctness.md §5.3, cr-283): the step as written again from the same state,
        then on its capture's stream and leases, its ops and kernels beside its capture's, and its
        graph replayed again (a replay that repeats froze something at capture; one that doesn't
        reads memory that changed). The last run is the check's eager step again, which serves.
        It runs only where the card holds its four copies of the step's state and one more step
        beside what is live; else the check's eager step serves, unexplained."""

        def run(on_capture: bool, mode: object = None) -> list[torch.Tensor]:
            start.restore()
            with (
                self._on_domain() if on_capture else nullcontext(),
                cast("AbstractContextManager[object]", mode or nullcontext()),
            ):
                self._advance(size)
            return [*(value.clone() for value in state), *self._written(size, at)]

        copy = sum(t.nbytes for t in replayed)
        need = 4 * copy + self.peaks.get(key, 0)
        if (room := self._room()) < need:
            return f"no room to run the step again: {_gib(need)} beside {_gib(room)} left"
        here = [*(value.clone() for value in state), *self._written(size, at)]
        mode, written = observe.trace(torch)
        served = False
        try:
            with _fallback():  # refused, the explanation stops short
                start.restore()
                vt.replay(graph, stream=torch.cuda.current_stream())
                twice = [*(value.clone() for value in state), *self._written(size, at)]
                again = run(False, mode)
                on_stream = run(True)
                run(False)
                served = True
        except Exception as exc:
            if not _short_of_room(exc):
                raise
            _unwind(exc)
            _answered(exc)
            return "no room to run the step again"
        finally:
            if not served:
                start.restore()
                self._advance(size)
        if noise := vt.replay_differs(here, again):
            return f"the step as written differs run to run ({noise}): a nondeterministic kernel"
        replays = vt.replay_differs(replayed, twice)
        said = [f"its replays differ ({replays})" if replays else "its replays repeat"]
        if _STEPS[0] is not launch:
            said.append("its kernels landed between its capture and its check")
        if op := observe.diverged(written, ops):
            said.append(op)
        if not vt.replay_differs(on_stream, replayed):
            said.append(
                "as written on its capture's stream it equals the replay: a library chose by stream"
            )
        if len(said) == 1:
            return (
                f"the same {len(written)} ops, and on its capture's stream it differs too: a "
                f"kernel outside Torch's dispatcher; {said[0]}"
            )
        return "the capture ran another program: " + "; ".join(said)

    def _discard(self, key: _Key, graph: torch.cuda.CUDAGraph, stream: torch.cuda.Stream) -> None:
        """A recorded graph goes, and its pool with the last of them."""
        stream.synchronize()
        if len(self.graphs) == 1:
            self._unlease()  # the pool's last graph goes: its use ends while it owns it
        del self.graphs[key]
        graph.reset()
        if not self.graphs:
            close_pool(_set(self.pool), _set(self.domain).device)
            self.pool = None

    def _forget(self) -> None:
        """Release recordings only at the driver's already-drained step boundary."""
        self.contracts = {}
        if self.graphs or self.pool is not None:
            for stream in self.unfenced:
                stream.synchronize()
            self.unfenced.clear()
            for event in self.completions.values():
                event.synchronize()
            self.completions.clear()
            self._unlease()
            self.retired_graphs += len(self.graphs)
            for graph in self.graphs.values():
                graph.reset()
            if self.pool is not None:
                close_pool(self.pool, _set(self.domain).device)
            self.graphs, self.pool = {}, None
        self._unlease()
        # What each key measured and was judged by goes with them: the next step measures again.
        self.warmed, self.unrecorded, self.peaks, self.judged = set(), {}, {}, {}
        self.undrafted = False

    def _pool(self, hold: bool) -> None:
        """Hold the graphs' pool mapped for an admitted graph's replays (`vt.prepare_pool(pool,
        0)`), or end that use so it may go idle (`vt.record_pool_stream`): taken with its lease
        (`_lease`) and after a capture, let go with the lease before an eager step or a prefill.
        Only the lease's take can need room: it refuses there (`Exhausted`), never at a replay."""
        if hold == self.pooled or self.pool is None:
            return
        if not self.graphs:  # nothing recorded owns it: no use to take or end
            self.pooled = False
            return
        device = _set(self.domain).device
        if hold:
            vt.prepare_pool(self.pool.handle, 0, device)
        else:
            vt.record_pool_stream(self.pool.handle, torch.cuda.current_stream(device), device)
        self.pooled = hold

    def _unrecorded(self, key: _Key, why: str, code: str | None = "capture_refused") -> None:
        """That step runs eagerly from now on, counted in the record (`safeguard`: said once,
        quietly) unless `code` is None: a property of the code, in the record alone
        (`decode.refused`)."""
        self.unrecorded[key] = why
        if code is None:
            return
        safeguard(code, f"{type(self.model).__name__}: {why}")

    def _drafting(self, size: int) -> bool:
        return (
            self.draft is not None
            and size <= DRAFT_ROWS
            and not self.undrafted
            and _set(self.cache_device).type == "cuda"
        )

    def _sampled(self) -> bool:
        """Whether a held row samples: such a step draws, an all-greedy one only takes argmaxes."""
        return any(row is not None and row.request.sampling.temperature > 0 for row in self.slots)

    def _constrained(self) -> bool:
        """Whether a held row has a constraint: its step reads `allow` and reports its tokens."""
        return any(row is not None and row.request.constrained for row in self.slots)

    def _allowed(
        self, scores: torch.Tensor, size: int, at: int | slice[int | None, int | None, int | None]
    ) -> torch.Tensor:
        """`scores` (`[size, ..., vocab]`) with what a constrained row may not take at -inf: the
        host's bits (`allow`) for position `at` of the step (a slice: `[size, positions, vocab]`);
        as they are where no held row is constrained."""
        if not self._constrained():
            return scores
        allowed = ((self.allow[at, :size, :, None] >> self.bit) & 1).flatten(-2)
        allowed = allowed[..., : self.vocab].bool()
        if isinstance(at, slice):
            allowed = allowed.movedim(0, 1)
        return scores.masked_fill(~allowed, float("-inf"))

    def _penalized(
        self, scores: torch.Tensor, size: int, ahead: torch.Tensor | None = None
    ) -> torch.Tensor:
        """`scores` (`[size, vocab]`, or a verify pass's `[size, span, vocab]`) after each row's
        penalties (`sampler.penalized`), counting its output so far and `ahead`: the drafts before
        each position, shaped as `scores`."""
        counts, prompted = self.counts[:size], self.prompted[:size]
        if scores.dim() == 3:
            counts, prompted = counts[:, None], prompted[:, None]
        if ahead is not None:
            counts = counts + ahead
        penalties = (self.presence[:size], self.frequency[:size], self.repetition[:size])
        return sampler.penalized(scores, counts, prompted, *penalties)

    def _pick(
        self, scores: torch.Tensor, size: int, index: torch.Tensor, sampled: bool
    ) -> torch.Tensor:
        """Each row's token from its penalized `scores` (`[size, ..., vocab]`) for the token at
        `index` (`sampler.pick`)."""
        temperature, top_p, top_k = self.temperature[:size], self.top_p[:size], self.top_k[:size]
        min_p, seed = self.min_p[:size], self.seed[:size]
        return sampler.pick(scores, temperature, top_p, top_k, min_p, seed, index, sampled)

    def _advance(self, size: int) -> None:
        if self._drafting(size):
            self._speculate(size)
        else:
            self._step(size)

    def _select(self, logits: torch.Tensor, size: int) -> torch.Tensor:
        """Each live row's next token, on the device, in `ids` and `out`; a row that stops (a
        stop token, or its limit) is done. Returns the rows live before it."""
        scores = self._penalized(self._allowed(logits.float(), size, 0), size)
        done = self.done[:size]
        live = ~done
        n = self.n[:size]
        token = torch.where(done, self.pad[:size], self._pick(scores, size, n, self._sampled()))
        if self._constrained():  # what the host reads back: how many, then the token
            made = torch.stack([live.long(), token])
            self.took[: 2 * size].view(2, size).copy_(made, non_blocking=True)
        self.out[:size].scatter_(1, n[:, None], token[:, None])
        n.add_(live.long())
        stopped = (token[:, None] == self.stops[:size]).any(-1) | (n >= self.limit[:size])
        done |= stopped
        self.ids[:size].copy_(token[:, None])
        self.counts[:size].scatter_add_(1, token[:, None], live.float()[:, None])
        self.flag[:size].copy_(done, non_blocking=True)
        return live


    def _step(self, size: int) -> None:
        """One token a row: its K and V written at its length in its blocks, attending to its
        row's K/V through the paged attention (`_tables`, `_attending`); select."""
        view, length = self._view(size), self.length[:size]
        self._tables(size, 1)
        kwargs: dict[str, torch.Tensor | _PagedCache | bool | int | None] = {
            "input_ids": self.ids[:size],
            "attention_mask": None,
            "past_key_values": view,
            "use_cache": True,
            "return_dict": True,
        }
        if self.positional:
            kwargs["position_ids"] = self.position[:size].movedim(0, -2)
        if self.keep:
            kwargs["logits_to_keep"] = 1
        with self._attending():
            logits = self.model(**kwargs).logits[:, -1]
        live = self._select(logits, size)
        length.add_(live.long())
        self._lengths(size)
        self.position[:size].add_(live.long().view(size, *[1] * (self.position.dim() - 1)))

    def _speculate(self, size: int) -> None:
        """`DEPTH` drafted tokens a row, verified with the row's token before them in one pass:
        each row keeps its tokens up to and including its first disagreement with the model,
        and its states go back to the kept tokens' end. A constrained step runs one part at a
        time (`part`: each draft, then the verify), so the host says what may follow each draft
        before the part after it reads that."""
        for part in range(DEPTH + 1) if self.part < 0 else (self.part,):
            if part < DEPTH:
                self._draft(size, part)
            else:
                self._commit(size)

    def _draft(self, size: int, i: int) -> None:
        """Each row's `i`th draft, into `drafts`: the head's token from its state for it (the
        first's, `hidden`, the step before left; a later one's from the draft before it, its K/V
        a position ahead of the row's)."""
        model, view, length = self.model, self._view(size), self.length[:size]
        if not i:
            hidden = self.hidden[:size]
        else:
            self._tables(size, 1, head=True)
            position = self.position[:size]
            at = (position + i).movedim(0, -2) if self.positional else (length + i)[:, None]
            embed, rotary = model.get_input_embeddings(), _lm(model).get_decoder().rotary_emb
            before = self.hidden[:size] if i == 1 else self.dhidden[:size]
            token = self.drafts[:size, i - 1 : i]
            with self._attending():
                hidden = _set(self.draft)(embed(token), before, at, view.draft, None, rotary)
            if i + 1 < DEPTH:
                self.dhidden[:size].copy_(hidden)
        logits = _lm(model).get_output_embeddings()(hidden)[:, -1].float()
        drafted: torch.Tensor | None = None
        if i:  # the drafts before it count as generated
            drafts, ones = self.drafts[:size, :i], torch.ones_like(logits[:, :i])
            drafted = torch.zeros_like(logits).scatter_add_(1, drafts, ones)
        scores = self._penalized(self._allowed(logits, size, i), size, drafted)
        token = self._pick(scores, size, self.n[:size] + i, self._sampled())
        self.drafts[:size, i].copy_(token)
        if self._constrained():  # the host's, for what may follow it
            self.dtook[i * size : (i + 1) * size].copy_(token, non_blocking=True)

    def _commit(self, size: int) -> None:
        """The drafts verified with the row's token before them in one pass (its K/V written at the
        row's length on, each query seeing the row's K/V and the pass's up to its own), what it
        keeps written, and the head's K/V and state and the model's states put at its end."""
        model, depth = self.model, DEPTH
        view = self._view(size)
        span = depth + 1
        embed, rotary = model.get_input_embeddings(), _lm(model).get_decoder().rotary_emb
        steps = self.steps[:span]
        length = self.length[:size]
        done = self.done[:size]
        live = ~done
        position = self.position[:size]
        sampled, n = self._sampled(), self.n[:size]
        drafted = self.drafts[:size]
        self._tables(size, span)
        kwargs: dict[str, torch.Tensor | _PagedCache | bool | int | None] = {
            "input_ids": torch.cat([self.ids[:size], drafted], dim=1),
            "attention_mask": None,
            "past_key_values": view,
            "use_cache": True,
            "return_dict": True,
        }
        if self.positional:
            kwargs["position_ids"] = (position + steps).movedim(0, -2)
        if self.keep:
            kwargs["logits_to_keep"] = span
        verify = _Verify()
        reset = _VERIFY.set(verify)
        try:
            with _taking(model) as taken, _RowByRow(), self._attending():
                scores: torch.Tensor = model(**kwargs).logits[:, -span:].float()
            scores = self._allowed(scores, size, slice(0, span))
        finally:
            _VERIFY.reset(reset)
        # position j counts the drafts before it as generated
        ahead = torch.zeros_like(scores).scatter_(2, drafted[..., None], 1.0).cumsum(1)
        ahead = torch.cat([torch.zeros_like(ahead[:, :1]), ahead[:, :-1]], dim=1)
        scores = self._penalized(scores, size, ahead)
        best = self._pick(scores, size, n[:, None] + steps[None], sampled)
        kept = (best[:, :depth] == drafted).long().cumprod(-1).sum(-1)
        valid = (steps[None] <= kept[:, None]) & live[:, None]
        limit = (n[:, None] + steps[None] + 1) >= self.limit[:size, None]
        stopped = ((best[..., None] == self.stops[:size, None]).any(-1) | limit) & valid
        after = (stopped.long().cumsum(-1) - stopped.long()) > 0
        keep = valid & ~after
        tokens = torch.where(keep, best, self.pad[:size, None])
        made = keep.long().sum(-1)
        at = n[:, None] + steps[None]
        width = self.out.shape[1]
        at = at.clamp(max=width - 1)
        self.out[:size].scatter_(1, at, torch.where(keep, tokens, self.out[:size].gather(1, at)))
        self.counts[:size].scatter_add_(1, best, keep.float())
        last = (made - 1).clamp(min=0)
        self.ids[:size].copy_(
            torch.where(live[:, None], best.gather(1, last[:, None]), self.pad[:size, None])
        )
        # The head's K/V for the committed tokens, a position ahead, each from the model's own
        # state for the token before it, and its state for the token after them: the next step's
        # first draft.
        self._tables(size, span, head=True)
        at = (
            (position + 1 + steps).movedim(0, -2)
            if self.positional
            else (length[:, None] + 1 + steps[None])
        )
        head = _set(self.draft)
        with self._attending():
            states: torch.Tensor = head(
                embed(tokens), taken[-1][:, -span:], at, view.draft, None, rotary
            )
        self.hidden[:size].copy_(
            states.gather(1, last[:, None, None].expand(-1, 1, states.shape[-1]))
        )
        # Back to the kept tokens' end: K/V past it is beyond the length and written over later.
        ended = length + made
        rows = torch.arange(size, device=length.device)
        for state, full in verify.conv:
            width_ = state.shape[-1]
            index = (self.rows_index[:width_][None] + made[:, None]).clamp(max=full.shape[-1] - 1)
            state.copy_(full.gather(-1, index[:, None, :].expand(-1, full.shape[1], -1)))
        for state, states_ in verify.recurrent:
            state.copy_(states_[(made - 1).clamp(min=0), rows])
        self.length[:size].copy_(ended)
        self._lengths(size)
        if self._constrained():  # what the host reads back: how many, then each token
            told = torch.cat([made[None], tokens.T])
            self.took[: (span + 1) * size].view(span + 1, size).copy_(told, non_blocking=True)
        n.add_(made)
        self.position[:size].add_(made.view(size, *[1] * (self.position.dim() - 1)))
        done |= stopped.any(-1)
        self.flag[:size].copy_(done, non_blocking=True)

    # ------------------------------------------------------------------ memory


    def _drop(self) -> None:
        """The K/V and the token buffers go (none decodes), and the graphs that read them."""
        if self.cache_device is not None and self.cache_device.type == "cuda":
            self._forget()
        if self.kv is not None:
            with self._kv_stream():
                for row in sorted(self.kv_rows):
                    self.kv.free(row)
                self.kv.close()
        self.kv, self.kv_rows, self.views = None, set(), {}
        self.slots, self.capacity, self.width = [], 0, 0
        self.layout = None
        # In particular, the pinned host flag must free while the domain stream is alive:
        # Torch's host allocator records that stream's completion when freeing the tensor.
        pinned = ("flag", "allow_host", "took", "dtook")
        for name in (*type(self).__annotations__, *pinned, "rows_index", "bit"):
            vars(self).pop(name, None)


def _beyond(
    writes: list[torch.Tensor], state: list[torch.Tensor], keyed: list[torch.Tensor]
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """What a step writes in place on the device beyond `_token_state`'s storages, from its
    capture's write set: the buffers among `keyed` (the K/V blocks, which `_Before` saves by token
    slot), and any other tensor. What it writes on the host (the flags and tokens the host reads
    back) is its output, never put back."""

    def storage(tensor: torch.Tensor) -> int:
        return int(tensor.untyped_storage().data_ptr())

    held = {storage(value) for value in state}
    named = {storage(t) for t in writes if t.is_cuda} - held
    buffers = [buffer for buffer in keyed if storage(buffer) in named]
    known = held | {storage(buffer) for buffer in keyed}
    return buffers, [t for t in writes if t.is_cuda and storage(t) not in known]


class _Before:
    """What a step writes in place, as it was before the step, for its graph's check to put back
    (`restore`) and let go (`release`): `whole` tensors entire, and of the K/V blocks
    (`[blocks, layers, 2, tokens, kv_heads, head_dim]`) only every layer's token slots a step
    reaches in each row, `STEP_SLOTS` from its length (the model's at its length, the head's a
    position ahead). A row not decoding reaches none: block 0, which no row holds, stands in.
    Key buffers saved whole were the decode cache again, 10 GiB at 16 rows of Qwen3.8-27B,
    allocated between a capture and its first replay (cr-333)."""

    def __init__(self, whole: list[torch.Tensor], kv: KvManager, size: int) -> None:
        self.whole, self.blocks = whole, kv.blocks
        tokens = kv.layout.block_tokens
        lengths = kv.lengths[:size].long()
        at = lengths[:, None] + torch.arange(STEP_SLOTS, device=lengths.device)
        index = (at // tokens).clamp(0, kv.max_blocks - 1)
        entry = kv.block_table[:size].long().gather(1, index)
        live = (lengths[:, None] >= 0) & (entry >= 0)
        self.entry = torch.where(live, entry, 0)
        self.offset = torch.where(live, at % tokens, 0)
        self.saved = [value.clone() for value in whole]
        self.kept: list[torch.Tensor] = [self.blocks[self.entry, :, :, self.offset]]

    def restore(self, first: int | None = None) -> None:
        """Everything as it was; with `first`, only that many of `whole`."""
        for tensor, value in zip(self.whole[:first], self.saved[:first], strict=True):
            tensor.copy_(value)
        if first is None:
            self.blocks[self.entry, :, :, self.offset] = self.kept[0]

    def beyond(self, first: int) -> tuple[int, int]:
        """The tensors it put back past the first `first` of `whole`, and the bytes it kept."""
        kept = [*self.saved[first:], *self.kept]
        return len(kept), sum(value.nbytes for value in kept)

    def release(self) -> None:
        self.whole, self.saved, self.kept = [], [], []


class _Judge:
    """A key's graph rung (cr-305's rule, the one diffusion's graphs take through `capture`'s
    `fits`): the weights its step reads, as its lease pinned them, its pool, what is live, one
    more eager step (its check's, beside the pool: the key's first step, measured) and the copies
    its check holds (`_check_bytes`), against the card less what else was pinned before the lease,
    what is live read then too (a pool the lease maps is the pool's). "" when they fit, else why,
    with the numbers; a pool found too large is remembered for the machine (`execution.rungs`), so
    the next process judges by it."""

    def __init__(self, engine: _Engine, before: Room, step: int, check: int, pool_key: str) -> None:
        pinned = vt.room(_set(engine.domain).device)["pinned"]
        #: what the step's lease pinned: the weights its graph reads, as the card holds them
        self.weights = max(pinned - before["pinned"], 0)
        self.card = before["capacity"] - before["pinned"]
        self.live, self.step, self.check, self.key = before["working_live"], step, check, pool_key
        #: by how much the last pool judged didn't fit
        self.short = 0

    def __call__(self, pool: int) -> str:
        from cozy_runtime.internal.execution import rungs  # with Torch

        need = self.weights + pool + self.live + self.step + self.check
        if need <= self.card:
            return ""
        self.short = need - self.card
        if self.key and pool > rungs.pool(self.key):
            rungs.put(self.key, pool)
        return (
            f"its graph's {_gib(self.weights)} of weights + {_gib(pool)} pool + {_gib(self.live)} "
            f"live + {_gib(self.step)} step + {_gib(self.check)} check exceed {_gib(self.card)}"
        )


def _unwind(exc: BaseException) -> None:
    """The refused attempt's frames and tensors go before anything counts room again."""
    from cozy_runtime.internal.execution import catch

    catch._unwind(exc)


def _short_of_room(exc: BaseException) -> bool:
    """Whether `exc` says an admitted call is short of room: an allocation refused (`is_capacity`)
    or a native refusal before anything ran (`DoesNotFit`: a split level, a pass's floor). Both are
    judged beside what this engine holds (its leased weights, slots, graphs' pool), which its slower
    path gives up, so each is that path's cue, never the call's end (cr-331: an edit's vision tower
    refused beside 21.4 GiB leased on a 4090 failed the call)."""
    return is_capacity(exc) or isinstance(exc, DoesNotFit)


def _fallback() -> AbstractContextManager[object]:
    """What the engine runs with a slower path behind a refusal (eager, undrafted, Transformers'
    loop): its allocations are refused, never given varena's last resort."""
    from cozy_runtime.internal.execution import catch

    return catch.fallback(True)


def _answered(exc: BaseException) -> None:
    """A refusal varena counted (`vt.ooms`) that the Runtime answered by running another way, so
    the package's scope is not judged to have swallowed it. `Exhausted` (a preparation refused
    before anything ran) is not one it counted, nor a split level refused before it ran
    (`DoesNotFit`) unless a refused allocation caused it (`catch.retry`)."""
    counted = exc.__cause__ if isinstance(exc, DoesNotFit) else exc
    if counted is None or isinstance(counted, Exhausted):
        return
    from cozy_runtime.internal.execution.catch import CURRENT as SERVING

    if (serving := SERVING.get()) is not None:
        serving.retries += 1


__all__ = [
    "KERNELS",
    "Draft",
    "drafting",
    "generate",
    "install",
]


DECODERS.append((_graph_snapshot, release))
