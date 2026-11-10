"""Fast decode for Hugging Face causal LMs: a token costs what streaming the weights costs, not
the host's Python.

    from cozy_runtime import lm

    new = lm.generate(model, inputs, max_new_tokens=160, stop=[eos])

The prompt prefills through the model's own forward into a static cache: keys and values in fixed
buffers, and a linear-attention layer's conv and recurrent state, which are fixed already. Every
later token is one replay of a CUDA graph of the one-token step, captured per (batch, window):
the window is the attention length, a power of two that grows with the sequence. The step picks
its token on the GPU (greedy, or sampled at a temperature and top-p from a seed, with an optional
presence penalty), writes it where the next step reads it and advances its own positions, so
replays queue back to back and the host reads the stop flag a few steps behind.

A sampled token is a Gumbel-max draw: the argmax, over the top-p nucleus, of the scores over the
temperature plus Gumbel noise hashed from (seed, the token's index, the vocabulary entry). A
draw depends only on its own row's scores, seed and position, so it replays inside a graph, does
not depend on what else is in the batch, and a drafted token is verified exactly as a greedy
one. Temperature 0 is the plain argmax.

A cached graph keeps only its weights' addresses; a driver's lease (`_hold`) brings their pages
back. Offload retires the graph before those addresses change; a later call can refill and capture
anew. If the full set cannot be prepared before execution, the call uses Transformers' loop. A
call's keys and state for its prompt and `max_new_tokens`, sized from the config, and the graphs'
pool are decided before its prefill (`_Engine._enter`): allocated or mapped then, else it waits for
decoding rows to leave, else `DoesNotFit` names what would fit; nothing after catches running out.
Errors after execution starts are final; package code is never restarted. On a CPU, or after a safe
capture refusal (a run warning), the same step runs eagerly. A model this does not
serve (a layer type other than full or linear attention, an attention other than SDPA or eager, a
generation config that adds logits processors) generates through Transformers, said once.

With `allowed` (a `Constraint`), each step's tokens are read back to the host, which asks the
constraint what may follow and writes that into a bit mask the next step's graph reads: one sync a
step, as the stop flag is, but caught up with each step rather than a few behind. A drafted
constrained step runs its drafts and its verify as separate graphs, each draft read back before
the mask after it is written.

With a `draft` (a model's multi-token-prediction head) a step drafts `DEPTH` tokens, the model
verifies them and the token after them in one pass, and the step keeps the prefix the model
agrees with and the model's own token after it: the greedy tokens the model alone gives, one to
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
from collections import deque
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import ExitStack, contextmanager, nullcontext
from functools import partial
from typing import TYPE_CHECKING, Any, Literal, NamedTuple, Protocol, cast

import tensorfs.varena.torch as vt
from tensorfs.varena import Exhausted

from cozy_runtime.accel.kernels import KernelsUnavailable, Launch, Plan, TritonKernels
from cozy_runtime.author import _request_path as request_path
from cozy_runtime.author._context import CURRENT, Context
from cozy_runtime.author._errors import Cancelled, DoesNotFit, is_capacity
from cozy_runtime.author._model import LANDED, PLACED, WARNINGS, Building
from cozy_runtime.author._services import stage_progress
from cozy_runtime.graphs import (
    BYPASS,
    DECODERS,
    _measured,
    capture,
    capture_domain,
    close_pool,
    no_room,
    safeguard,
    say,
)
from cozy_runtime.graphs.domain import Pool
from cozy_runtime.internal import cache_owners, contract
from cozy_runtime.lm import _kernels

if TYPE_CHECKING:
    from types import ModuleType

    from torch.utils.hooks import RemovableHandle
    from transformers import PreTrainedConfig, PreTrainedModel
    from transformers.cache_utils import CacheLayerMixin, LinearAttentionCacheLayerMixin

    from cozy_runtime.author._model import Residency, WarningRow
    from cozy_runtime.graphs.domain import Domain
    from cozy_runtime.internal.residency import Binding, Region

try:
    import torch
    from torch.overrides import TorchFunctionMode
    from torch.utils._python_dispatch import _disable_current_modes
    from transformers import LogitsProcessorList, StoppingCriteria, StoppingCriteriaList
    from transformers.cache_utils import (
        Cache,
        LinearAttentionLayer,
        StaticLayer,
        get_layer_types_and_kwargs,
    )
except ImportError as _absent:  # pragma: no cover - the base package has neither
    if TYPE_CHECKING:
        raise
    else:
        ABSENT = f"torch and transformers do not import ({_absent})"
        StaticLayer = LinearAttentionLayer = Cache = TorchFunctionMode = StoppingCriteria = object
else:
    ABSENT = ""

#: The smallest attention window; windows double from it up to the cache's capacity.
MIN_WINDOW = 256
#: Steps the host may queue ahead of the stop flag it last read.
LAG = 2
#: Tokens a draft head proposes a step. Qwen3.5-9B's head on FLUX.3's prompt upsampler keeps
#: ~0.7 of first drafts and few third ones: 2 is fastest on a 4090 (1 and 3 lose 5%, 4 more).
DEPTH = 2
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

    def allowed(self, tokens: Sequence[int]) -> Sequence[int]:
        """The ids that may follow `tokens`, one row's generated ids (its prompt excluded); once
        the grammar is complete, its stop token. Pure in `tokens`: a draft is checked as
        `allowed(prefix + draft[:k])` for each k, with nothing to undo."""
        ...


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
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: int = 0,
    seed: int = 0,
    draft: torch.nn.Module | None = None,
    allowed: Constraint | None = None,
    progress: Callable[[int], None] | None = None,
) -> torch.Tensor:
    """Tokens after the prompt, `[batch, n]` (`model.generate(...)`'s tail): `inputs` as the
    processor returned them, generation ends at a `stop` token or after `max_new_tokens`, and a
    row that stopped early continues as `pad` (default: the first stop). `presence_penalty`
    subtracts that much from every token already generated (vLLM's). `temperature` 0 is greedy;
    above it, each token is drawn from the `top_p` nucleus at that temperature, from `seed`: the
    same prompt and seed give the same tokens. `top_k` above 0 first keeps a sampling row's `top_k`
    largest logits, before the presence penalty; greedy decoding ignores it.
    `allowed` (`Constraint`) constrains every row: each token outside
    `allowed.allowed(its tokens so far)` is out before `top_k`, the presence penalty, temperature
    and top-p, greedy or sampled, drafted or not. The host asks it between steps, so a
    constrained batch's steps do not queue ahead of it.
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
    as its fraction of `max_new_tokens`."""
    for named in (draft, _DRAFT.get(), HEADS.get(model)):
        if named is not None:
            draft = named
            break
    if not (temperature >= 0 and 0 < top_p <= 1 and 0 <= seed < 1 << 63 and top_k >= 0):
        raise ValueError(
            f"temperature {temperature}, top_p {top_p}, top_k {top_k}, seed {seed}: out of range"
        )
    if progress is None and (stage := stage_progress()) is not None:
        progress = partial(_share, stage, max_new_tokens)
    sampling = _Sampling(float(temperature), float(top_p), int(seed), int(top_k))
    tell = _Tell(progress)
    stops = [int(token) for token in stop]
    filler = stops[0] if pad is None else int(pad)
    why = ABSENT or _unsupported(model)
    if why:
        say(
            type(model).__name__, f"[lm] {type(model).__name__} decodes through Transformers: {why}"
        )
        row: WarningRow = ("decode ran through Transformers' loop", {"why": why})
        if row not in WARNINGS:
            WARNINGS.append(row)
        with contract.owned():  # the Runtime's own use of Transformers' loop and caches
            said = (model, inputs, max_new_tokens, stops, filler, presence_penalty, sampling, tell)
            return _transformers(*said, allowed)
    with _ENGINES_LOCK:
        engine = _ENGINES.get(model)
        if engine is None:
            engine = _ENGINES[model] = _Engine(model)
    request = _Request(
        inputs, max_new_tokens, stops, filler, presence_penalty, sampling, tell, allowed
    )
    token = BYPASS.set(True)  # the component's calls are this loop's, its graphs its own
    try:
        with torch.inference_mode(), contract.owned():
            return engine.generate(request, draft)
    finally:
        BYPASS.reset(token)


def _share(stage: Callable[[float], None], most: int, tokens: int) -> None:
    """`tokens` decoded of at most `most`, as the running stage's fraction."""
    stage(min(tokens / max(most, 1), 1.0))


def release(model: torch.nn.Module) -> None:
    """Retire an idle model's decode buffers, graphs and virtual address ownership."""
    with _PLANE, _ENGINES_LOCK:
        engine = _ENGINES.pop(model, None)
        if engine is None:
            return
        if engine.driver is not None or engine.pending or engine._occupied():
            _ENGINES[model] = engine
            raise RuntimeError("cannot unload a language model while decode is active")
        engine._drop()
        if engine.region is not None:
            engine.region.close()
            engine.region = None
        for binding in engine.addresses.values():
            binding.close()
        engine.addresses.clear()
        engine.where = ()


def _graph_snapshot(
    model: torch.nn.Module,
) -> tuple[dict[str, int | list[str]], list[Pool], Domain | None]:
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
            "retired_graphs": engine.retired_graphs,
            "replays": engine.replayed,
            "eager_steps": engine.eager,
            "refused": sorted(set(engine.unrecorded.values())),
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
            and conv_state.is_contiguous()
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
                _kernels.CONV,
                _kernels.CONV_BLOCK,
            )
            return out
        return original(hidden_states, conv_state, weight, bias, activation, *args, **kwargs)

    slot._cozy_slot = True  # type: ignore[attr-defined]  # a marker on the function object
    return slot


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
            and initial_state.is_contiguous()
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
    other = sorted(set(types) - {"full_attention", "linear_attention"})
    if other:
        return f"layer types {other}"
    if why := _shapes(text)[1]:
        return why
    extras = [
        name
        for name, idle in _PROCESSORS.items()
        if getattr(model.generation_config, name, None) not in idle
    ]
    return f"its generation config adds {extras}" if extras else ""


#: A gated delta net's sizes in its config (`linear_*`): conv kernel, key heads and width, value
#: heads and width.
_DELTA_NET = (
    "conv_kernel_dim",
    "num_key_heads",
    "key_head_dim",
    "num_value_heads",
    "value_head_dim",
)


def _shapes(text: PreTrainedConfig) -> tuple[list[_Shape], str]:
    """Each cache layer's state for one row, from the config alone, before any forward: a
    full-attention layer's `("keys", heads, key width, value width)`, a gated delta net's
    `("state", (conv channels, kernel), (heads, key width, value width))`. "" or why not."""
    types, kwargs = get_layer_types_and_kwargs(text)
    if isinstance(kwargs, dict):  # type: ignore[unreachable]  # before 5.19, a dict
        kwargs = [kwargs] * len(types)  # type: ignore[unreachable]  # before 5.19, a dict
    heads = getattr(text, "num_key_value_heads", None) or text.num_attention_heads
    width = getattr(text, "head_dim", None) or text.hidden_size // text.num_attention_heads
    linear = [getattr(text, f"linear_{name}", None) for name in _DELTA_NET]
    layers: list[_Shape] = []
    for kind, extra in zip(types, kwargs, strict=True):
        if kind == "full_attention":
            layers.append(("keys", int(heads), int(width), int(width)))
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
    """Shipped decode contracts, dense and with experts split (`moe`), and a drafted step's
    head (`Draft` around Transformers' decoder layer); persistent KV, positions and counters
    are device buffers.

    The real warm step initializes cache views and advances accepted history once. Capture
    only records the next step's device writes. Custom Python hooks and arbitrary model code
    are never speculatively executed by this graph builder.
    """
    forward = type(model).forward
    if not forward.__module__.startswith("transformers.models.") or type(model).__name__ not in {
        "Qwen3ForCausalLM",
        "Qwen3_5ForCausalLM",
        "Qwen3_5ForConditionalGeneration",
        "Qwen3_5MoeForCausalLM",
        "Qwen3_5MoeForConditionalGeneration",
    }:
        return "no shipped decode capture contract"
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
    """A call's sampling: greedy at temperature 0."""

    temperature: float
    top_p: float
    seed: int
    top_k: int = 0


#: A step's graph: (rows, window, drafted, sampled, the most logits a sampling row keeps,
#: constrained, and a constrained drafted step's part: each draft, then the verify; else -1).
_Key = tuple[int, int, bool, bool, int, bool, int]


def _top_k(scores: torch.Tensor, k: int) -> torch.Tensor:
    """`scores` (`[..., vocab]`) with all but each row's `k` largest at -inf (ties kept)."""
    kth = scores.topk(min(k, scores.shape[-1]), dim=-1).values[..., -1:]
    return scores.masked_fill(scores < kth, float("-inf"))


def _ids(constraint: Constraint, tokens: Sequence[int], vocab: int) -> torch.Tensor:
    """`constraint.allowed(tokens)` as a host index tensor: some, each in the vocabulary."""
    ids = torch.as_tensor(constraint.allowed(tokens)).to("cpu", torch.long).reshape(-1)
    if not ids.numel() or int(ids.min()) < 0 or int(ids.max()) >= vocab:
        raise ValueError(
            f"allowed() after {len(tokens)} tokens gave {ids.numel()} ids: it gives at least one, "
            f"each below the vocabulary's {vocab} (a complete grammar, its stop token)"
        )
    return ids


def _pack(ids: torch.Tensor, out: torch.Tensor) -> None:
    """`ids` as bits of `out` (`[ceil(vocab / 8)]` uint8, on the host): token 8i + j is bit j of
    byte i."""
    bits = torch.zeros(out.numel() * 8, dtype=torch.uint8)
    bits[ids] = 1
    out.copy_((bits.view(-1, 8) << torch.arange(8, dtype=torch.uint8)).sum(-1, dtype=torch.uint8))


_LOW32 = 0xFFFFFFFF


def _times(x: torch.Tensor, constant: int) -> torch.Tensor:
    """`x * constant` mod 2**32 for 32-bit values held in int64, with no int64 overflow."""
    return (x * (constant & 0xFFFF) + (((x * (constant >> 16)) & 0xFFFF) << 16)) & _LOW32


def _mix(x: torch.Tensor) -> torch.Tensor:
    """A 32-bit integer hash (Wellons' lowbias32): the same bits on every device."""
    x = x ^ (x >> 16)
    x = _times(x, 0x7FEB352D)
    x = x ^ (x >> 15)
    x = _times(x, 0x846CA68B)
    return x ^ (x >> 16)


def _draw(
    scores: torch.Tensor,
    temperature: torch.Tensor,
    top_p: torch.Tensor,
    seed: torch.Tensor,
    index: torch.Tensor,
) -> torch.Tensor:
    """Each row's sampled token from `scores` (`[rows, ..., vocab]`, float32): the argmax over the
    `top_p` nucleus of scores / `temperature` plus Gumbel noise hashed from (`seed`, `index`, the
    vocabulary entry). `temperature`, `top_p` and `seed` are per row; `index` is the token's
    position in the row's output, shaped as `scores` without its vocabulary."""
    lead = (-1, *[1] * (scores.dim() - 1))
    z = scores / temperature.clamp(min=1e-6).view(lead)
    ordered, order = z.sort(-1, descending=True)
    probabilities = ordered.softmax(-1)
    cut = (probabilities.cumsum(-1) - probabilities >= top_p.view(lead)) & (top_p.view(lead) < 1)
    outside = torch.zeros_like(cut).scatter_(-1, order, cut)
    base = _mix(_mix(seed & _LOW32) ^ ((seed >> 32) & _LOW32)).view(lead)
    key = _mix(base ^ _mix(index[..., None] & _LOW32))
    entry = torch.arange(scores.shape[-1], device=scores.device, dtype=torch.long)
    bits = _mix(key ^ _mix(entry))
    # Below 1: float32 rounds the top hash value to 1, whose noise is +inf (that entry would win
    # at any score, or give NaN, which argmax takes, on a score at -inf).
    uniform = (((bits >> 8).float() + 0.5) * (1.0 / (1 << 24))).clamp_(max=1 - 2**-24)
    noise = -torch.log(-torch.log(uniform))
    return (z + noise).masked_fill_(outside, float("-inf")).argmax(-1)


class _Draw:
    """Sampling as a Transformers logits processor, for `_transformers`: every score but the
    drawn token's is -inf, so its greedy loop takes the draw."""

    def __init__(self, prompt: int, sampling: _Sampling) -> None:
        self.prompt, self.sampling = prompt, sampling

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        rows, device = int(scores.shape[0]), scores.device
        at = int(input_ids.shape[1]) - self.prompt
        index = torch.full((rows,), at, dtype=torch.long, device=device)
        token = _draw(scores.float(), *_per_row(self.sampling, rows, device), index)
        return torch.full_like(scores, float("-inf")).scatter_(1, token[:, None], 0.0)


def _per_row(
    sampling: _Sampling, rows: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """`sampling` as `_draw`'s per-row temperature, top-p and seed."""
    return (
        torch.full((rows,), sampling.temperature, dtype=torch.float32, device=device),
        torch.full((rows,), sampling.top_p, dtype=torch.float32, device=device),
        torch.full((rows,), sampling.seed, dtype=torch.long, device=device),
    )


class _Allowed:
    """`allowed` as a Transformers logits processor, for `_transformers`: each row's tokens
    after the prompt, until a stop."""

    def __init__(self, prompt: int, constraint: Constraint, stops: list[int]) -> None:
        self.prompt, self.constraint, self.stops = prompt, constraint, set(stops)

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        keep = torch.ones(scores.shape, dtype=torch.bool)
        for row, tokens in enumerate(input_ids[:, self.prompt :].tolist()):
            if not self.stops.intersection(tokens):  # a finished row continues as padding
                keep[row] = False
                keep[row, _ids(self.constraint, tokens, int(scores.shape[-1]))] = True
        return scores.masked_fill(~keep.to(scores.device), float("-inf"))


class _TopK:
    """`top_k` as a Transformers logits processor, for `_transformers`."""

    def __init__(self, k: int) -> None:
        self.k = k

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        return _top_k(scores, self.k)


class _Presence:
    """`presence_penalty` as a Transformers logits processor, for `_transformers`."""

    def __init__(self, prompt: int, penalty: float) -> None:
        self.prompt, self.penalty = prompt, penalty

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        generated = input_ids[:, self.prompt :]
        if not generated.shape[1]:
            return scores
        seen = torch.zeros_like(scores).scatter_(1, generated, 1.0)
        return scores - self.penalty * seen


def _transformers(
    model: PreTrainedModel,
    inputs: Mapping[str, torch.Tensor],
    max_new_tokens: int,
    stops: list[int],
    pad: int,
    penalty: float,
    sampling: _Sampling,
    tell: _Tell,
    allowed: Constraint | None = None,
) -> torch.Tensor:
    prompt = int(inputs["input_ids"].shape[1])
    processors: list[_Allowed | _TopK | _Presence | _Draw] = (
        [] if allowed is None else [_Allowed(prompt, allowed, stops)]
    )
    if sampling.temperature > 0 and sampling.top_k:
        processors.append(_TopK(sampling.top_k))
    if penalty:
        processors.append(_Presence(prompt, penalty))
    if sampling.temperature > 0:
        processors.append(_Draw(prompt, sampling))
    context = CURRENT.get()
    with _PLANE, torch.inference_mode():
        out = _lm(model).generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=stops,
            pad_token_id=pad,
            logits_processor=LogitsProcessorList(processors),
            stopping_criteria=StoppingCriteriaList([_Cancel(context, tell)]),
        )
    return out[:, prompt:]


class _Cancel(StoppingCriteria):
    """Transformers' loop raises the attempt's cancellation between tokens and tells its
    progress."""

    def __init__(self, context: Context | None, tell: _Tell) -> None:
        self.context, self.tell, self.steps = context, tell, 0

    def __call__(
        self, input_ids: torch.Tensor, scores: tuple[torch.Tensor] | None, **kwargs: object
    ) -> torch.BoolTensor:
        if self.context is not None:
            self.context.raise_if_cancelled()
        self.steps += 1  # one token a step: this loop drafts none
        self.tell(lambda: self.steps)
        flags = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
        return cast("torch.BoolTensor", flags)  # cast: Transformers types its flags as BoolTensor


#: The least time between two progress reports of one call.
PROGRESS_S = 0.25


class _Tell:
    """A call's `progress`, told its tokens so far at most every `PROGRESS_S`: `count` is read only
    then."""

    def __init__(self, progress: Callable[[int], None] | None) -> None:
        self.progress, self.told = progress, 0.0

    def __call__(self, count: Callable[[], int]) -> None:
        if self.progress is None:
            return
        now = time.perf_counter()
        if now - self.told < PROGRESS_S:
            return
        self.told = now
        try:
            self.progress(count())
        except Cancelled:  # the decode's own check ends the call
            return
        except Exception as exc:  # the decode (perhaps other calls' too) outlives its report
            self.progress = None
            WARNINGS.append(
                ("decode progress stopped: its report failed", {"error": str(exc)[:200]})
            )


def _batch(rows: int) -> int:
    """The smallest batch a step records that holds `rows` (at most the largest)."""
    return next((b for b in BUCKETS if b >= rows), BUCKETS[-1])


def _windows(length: int) -> int:
    """`length` positions, whole windows of `MIN_WINDOW`."""
    return -(-length // MIN_WINDOW) * MIN_WINDOW


def _gib(n: float) -> str:
    return f"{n / (1 << 30):.1f} GiB"


def window(length: int, capacity: int) -> int:
    """The attention window for a sequence of `length`: the next power of two from `MIN_WINDOW`,
    at most the cache's `capacity`."""
    size = MIN_WINDOW
    while size < length:
        size *= 2
    return min(size, capacity)


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

    def __torch_function__(
        self,
        func: Callable[..., Any],  # Any: torch's mode hook sees every torch function
        types: tuple[type, ...],
        args: tuple[Any, ...] = (),  # Any: that function's arguments
        kwargs: dict[str, Any] | None = None,  # Any: that function's arguments
    ) -> Any:  # Any: that function's result
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
            and initial_state.is_contiguous()
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
            _Keys(self.size, capacity, rows) if kind == "full_attention" else _State(**extra)
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


def _layers(cache: Cache) -> list[_Keys | _State]:
    """The layers of a cache this module built (`_StaticCache`, a head's)."""
    return cast("list[_Keys | _State]", cache.layers)  # cast: it builds only these


def _head_keys(cache: Cache) -> _Keys:
    """A head's cache's one layer (`Draft` attends with one full-attention layer)."""
    return cast(_Keys, cache.layers[0])  # cast: a head's cache is built of its one `_Keys`


def _pairs(mine: Cache, theirs: Cache) -> Iterator[tuple[_Keys | _State, _Keys | _State]]:
    """Two caches' layers side by side. Built from one config, each pair is of one type."""
    return zip(_layers(mine), _layers(theirs), strict=True)


def _like[L: (_Keys, _State)](mine: L, theirs: _Keys | _State) -> L:
    """`theirs`, `mine`'s pair from `_pairs`: of `mine`'s type."""
    return cast(L, theirs)  # cast: both caches were built from one config


class _Prefill:
    """A call's prompt through the model: its own cache, mask and positions, each row's first
    token and, drafting, the head's keys for the prompt and its state after it."""

    def __init__(
        self,
        cache: _StaticCache,
        mask: torch.Tensor,
        position: torch.Tensor | None,
        token: torch.Tensor,
        drafted: tuple[_Keys, torch.Tensor] | None,
    ) -> None:
        self.cache, self.mask, self.position, self.token, self.drafted = (
            cache,
            mask,
            position,
            token,
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
        penalty: float,
        sampling: _Sampling,
        tell: _Tell,
        allowed: Constraint | None = None,
    ) -> None:
        self.allowed = allowed
        self.inputs, self.max_new, self.stops, self.pad, self.penalty, self.sampling = (
            inputs,
            max_new,
            stops,
            pad,
            penalty,
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
        self.steps = self.live = 0
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
        #: the most tokens it can hold after the steps queued so far (the window's bound)
        self.reach = length + 1
        #: a constrained row's tokens as the host knows them (`_took`), and this step's drafts
        self.generated: list[int] | None = None
        self.drafts: list[int] = []

    def ended(self) -> bool:
        """A constrained row the host knows done: at a stop token, or its limit."""
        said, request = self.generated or [], self.request
        return bool(said) and (said[-1] in request.stops or len(said) >= request.max_new)


class _Engine:
    """A model's continuous batch: every concurrent `generate` call on it decodes in one loop.

    Each sequence holds a slot, one row of every buffer: its keys and values (`_Keys`, written
    at the row's own length), its linear-attention state, its token, positions, mask and output.
    A step runs the smallest bucket (`BUCKETS`) holding every occupied slot at the window of the
    longest row, recorded once as a CUDA graph and replayed. A call joins at the next step after
    its unpadded prefill, which splices its rows into free slots, and each row leaves when the
    host sees it done, its slot filled from the top so the bucket shrinks. No thread of its own:
    the caller that finds the engine idle drives, and hands driving to a waiting caller when its
    own rows are done."""

    # The per-row buffers, assigned together from one `fields` mapping when the slots are sized.
    ids: torch.Tensor
    length: torch.Tensor
    mask: torch.Tensor
    out: torch.Tensor
    n: torch.Tensor
    done: torch.Tensor
    limit: torch.Tensor
    stops: torch.Tensor
    pad: torch.Tensor
    penalty: torch.Tensor
    temperature: torch.Tensor
    top_p: torch.Tensor
    top_k: torch.Tensor
    seed: torch.Tensor
    seen: torch.Tensor
    position: torch.Tensor
    dmask: torch.Tensor
    hidden: torch.Tensor
    drafts: torch.Tensor
    dhidden: torch.Tensor
    allow: torch.Tensor

    def __init__(self, model: PreTrainedModel) -> None:
        self.model = model
        self.text = model.config.get_text_config(decoder=True)
        parameters = set(inspect.signature(model.forward).parameters)
        self.keep, self.positional = "logits_to_keep" in parameters, "position_ids" in parameters
        self.cond = threading.Condition(threading.RLock())
        self.pending: deque[_Request] = deque()
        #: calls taken from `pending` that the driver is prefilling or splicing
        self.incoming: list[_Request] = []
        self.rehold = False
        #: heads whose blocks did not fit at home beside the model's
        self.unfit: set[int] = set()
        #: a drafted step could not be captured: drafting pays only replayed, so steps go plain
        self.undrafted = False
        #: captures recorded in this engine's life and their seconds, steps run eagerly
        self.captured = [0, 0.0]
        self.eager = 0
        self.recorded = 0
        self.replayed = 0
        self.retired_graphs = 0
        #: `LANDED` when a capture met a kernel still building (none records until one lands),
        #: and when the graphs were last started over
        self.building, self.landed = -1, LANDED[0]
        self.driver: int | None = None
        self.draft: torch.nn.Module | None = None
        self.slots: list[_Row | None] = []
        self.capacity = 0
        self.width = 0
        self.cache: _StaticCache | None = None
        self.views: dict[int, _StaticCache] = {}
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
        self.unprepared: BaseException | None = None
        self.completions: dict[torch.cuda.Stream, torch.cuda.Event] = {}
        self.unfenced: set[torch.cuda.Stream] = set()
        self.domain: Domain | None = None
        #: a step's graph key (`_Key`) -> why that step runs eagerly
        self.unrecorded: dict[_Key, str] = {}
        #: the graphs recorded a path that changed since (a kernel landed in `install`)
        self.stale = False
        #: seconds the last admission took: decoding gets as long before the next one
        self.debt = 0.0
        #: KV storage stays pinned across driver handoffs while any live row remains.
        self.region: Region | None = None
        self.region_use: ExitStack[bool | None] | None = None
        self.retire = False
        self.cache_device: torch.device | None = None
        # an int8 head: no .weight
        self.vocab = int(_lm(model).get_output_embeddings().out_features)
        #: the device bytes of the slot buffers (`_need`), and whether the graphs' pool is held
        #: mapped for the rows decoding (`_pool`)
        self.held = 0
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
        rows = len(request.tokens)
        if rows > BUCKETS[-1]:
            request.fallback = f"{rows} rows: the engine decodes at most {BUCKETS[-1]} at once"
        elif self.misfit:
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
            WARNINGS.append(("decode ran through Transformers' loop", {"why": request.fallback}))
            return _transformers(
                self.model,
                request.inputs,
                request.max_new,
                request.stops,
                request.pad,
                request.penalty,
                request.sampling,
                request.tell,
                request.allowed,
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
        if self.draft is not None:
            mode += f", drafting {DEPTH}: {looped / max(request.steps, 1):.2f} tokens a step"
        mode += f", {request.live / max(request.steps, 1):.1f} rows a step"
        facts = (
            f"prefill {request.prefill_s:.3f} s",
            f"then {per:.2f} ms/token over {looped:.0f} ({mode})",
            f"{captures} captures {capture_s:.3f} s",
            f"gated-delta kernels {'on' if _LAUNCHED[0] else 'off'}",
        )
        text = f"{tokens} tokens after {request.prompt}: " + ", ".join(facts)
        print(f"[lm] {text}", file=sys.stderr)
        placed = PLACED.get(self.model)
        if placed is not None:
            placed[0].note(f"{placed[1]}.lm", text)

    # ------------------------------------------------------------------ the loop

    def _drive(self, mine: _Request) -> None:
        with _PLANE, self.region_use or ExitStack() as lifetime:
            held_region = self.region_use is not None
            self.region_use = None
            try:
                # The KV pin lasts through reholds and failure unwinding. Only weight
                # bindings below may be replaced while this driver owns live rows.
                if self.region is None and (placed := PLACED.get(self.model)) is not None:
                    self.region = placed[0].region(
                        f"lm:{type(self.model).__name__}",
                        "kv",
                        self.model,
                        release=self._drop,
                        shrink=self._try_drop,
                    )
                if (
                    not held_region
                    and self.region is not None
                    and lifetime.enter_context(self.region.pin())
                ):
                    self._drop()
                with ExitStack() as stack:
                    held = self._hold(stack)
                    while True:
                        if self.rehold:  # another head: its blocks are held instead
                            stack.close()
                            held, self.rehold = self._hold(stack), False
                        if held is False and self.draft is not None and not self._occupied():
                            # The head's weights and keys cost room the model's own decode set may
                            # need: the graphed decode without drafts beats Transformers' loop.
                            say(
                                f"lm:{type(self.model).__name__}:undrafted",
                                "[lm] drafting does not fit at home",
                            )
                            self.unfit.add(id(self.draft))
                            self._drop()
                            self.draft = None
                            stack.close()
                            held = self._hold(stack)
                        with self.cond:
                            self.incoming = list(self.pending)
                            self.pending.clear()
                        if held is False:
                            if self._occupied():
                                raise self.unprepared or RuntimeError("decode residency was lost")
                            for request in self.incoming:
                                request.fallback = "its decode set does not fit at home"
                            self.incoming = []
                        if self.incoming or self.retire:
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
                if not self._occupied():
                    self._pool(False)  # no row decodes: its graphs' pool may go idle
                with self.cond:
                    if self.region is not None and (self._occupied() or self.pending):
                        # A waiting caller takes the driver next. These rows are still live;
                        # only an idle decode cache is eligible for owner-driven discard.
                        self.region_use = lifetime.pop_all()
                    self.driver = None
                    self.cond.notify_all()

    def _fail(self, exc: BaseException) -> None:
        """Every held and waiting call fails; no partially executed generation is restarted."""
        with self.cond:
            requests = {id(row.request): row.request for row in self.slots if row is not None}
            requests.update({id(request): request for request in self.pending})
            requests.update({id(request): request for request in self.incoming})
            self.pending.clear()
            self.incoming = []
            for request in requests.values():
                request.error = exc
            self._drop()
            self.cond.notify_all()

    def _targets(self) -> list[tuple[torch.nn.Module, tuple[Residency, str]]]:
        return [(m, PLACED[m]) for m in (self.model, self.draft) if m is not None and m in PLACED]

    def _hold(self, stack: ExitStack) -> bool:
        """Prepare the retained weight tensors while this driver runs.

        A full decode that cannot prepare its dependencies refuses before model execution;
        Transformers can then stream the blocks eagerly. The KV pin has a separate lifetime
        so changing these bindings cannot make live decode rows reclaimable.
        """
        self.unprepared = None
        derived = _derived(m for m in (self.model, self.draft) if m is not None)
        built = tuple(t.data_ptr() for t in derived)
        identities: list[object] = []
        with ExitStack() as leases:
            for module, (residency, name) in self._targets():
                binding = self.addresses.get(id(module))
                if binding is not None and not binding.valid():
                    self._forget()
                    binding.close()
                    binding = None
                if binding is None:
                    binding = self.addresses[id(module)] = residency.retain(name, ("",))
                try:
                    leases.enter_context(binding.prepare())
                except Exception as exc:
                    if not is_capacity(exc) and getattr(exc, "code", None) != "STALE_BINDING":
                        raise
                    self.unprepared = exc
                    return False
                identities.append(binding.identity)
            self.where = (*identities, built)
            stack.enter_context(leases.pop_all())
        return True

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
        _set(self.cache).window = window(max(row.reach for row in rows) + per, self.capacity)
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
                event = torch.cuda.Event()  # type: ignore[no-untyped-call]  # unannotated in Torch
                self.completions[stream] = event
                event.record(stream)  # instantiate before any accepted step can run
            self.unfenced.add(stream)
        try:
            for part in range(DEPTH + 1) if drafting and constrained else (-1,):
                self.part = part
                if constrained:
                    self._fill(size, max(part, 0))
                with _set(self.domain).using() if stream is not None else nullcontext():
                    self._replay(size, cuda)
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
        for row in rows:
            row.reach = min(row.reach + per, self.capacity)
        for request in {id(row.request): row.request for row in rows}.values():
            request.steps += 1
            request.live += len(rows)

    def _settle(self, incoming: list[_Request]) -> None:
        """Between steps, with the device caught up: done and cancelled rows leave, the rows
        above fill their slots, and new calls prefill into the slots that are free."""
        if self.cache is not None:
            with _disable_current_modes():
                if _set(self.cache_device).type == "cuda":
                    torch.cuda.current_stream().synchronize()
                self._retire()
                self.flag.zero_()
        self.retire = False
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
            if waiting and not self._occupied() and not admitted and not self.rehold:
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
                request.at = tuple(
                    now - then
                    for now, then in zip(
                        (self.captured[0], self.captured[1], self.eager), request.at, strict=True
                    )
                )
            self.slots[slot] = None
        # The rows above the holes move down, so a step runs the smallest bucket.
        top = len(self.slots) - 1
        for slot in range(len(self.slots)):
            while top > slot and self.slots[top] is None:
                top -= 1
            if self.slots[slot] is None and top > slot:
                for tensor in self._rows():
                    tensor[slot].copy_(tensor[top])
                self.slots[slot], self.slots[top] = self.slots[top], None
        for slot, row in enumerate(self.slots):
            if row is None:
                self._empty(slot)
        with self.cond:
            self.cond.notify_all()

    def _rows(self) -> Iterator[torch.Tensor]:
        """Every buffer with a row per slot."""
        for layer in _layers(_set(self.cache)):
            if isinstance(layer, _Keys):
                yield from (layer.keys, layer.values, layer.cumulative_length)
            else:
                yield from (s for s in layer.conv_states.values() if s is not None)
                yield from (s for s in layer.recurrent_states.values() if s is not None)
        yield from (self.ids, self.length, self.mask, self.out, self.n, self.done, self.limit)
        yield from (self.stops, self.pad, self.penalty, self.seen, self.position)
        yield from (self.temperature, self.top_p, self.seed, self.top_k)
        if self.draft is not None:
            yield from (self.dkeys.keys, self.dkeys.values, self.dmask, self.hidden)

    def _empty(self, slot: int) -> None:
        """A free slot as a finished row of length 0 that attends to its first position."""
        self.done[slot] = True
        self.length[slot] = 0
        self.n[slot] = 0
        self.temperature[slot] = 0.0
        self.mask[slot].fill_(self.closed)
        self.mask[slot, ..., 0] = self.open
        if self.draft is not None:
            self.dmask[slot].fill_(self.closed)
            self.dmask[slot, ..., 0] = self.open

    # ------------------------------------------------------------------ admission

    def _admit(self, request: _Request) -> bool:
        """`request` decided (`_enter`), then run: its prompt through the model's own forward
        into a cache of its own, unpadded, and its rows into free slots, with their first token.
        False: it waits for decoding rows to leave."""
        free = [slot for slot, row in enumerate(self.slots) if row is None]
        if request.draft is not None and id(request.draft) in self.unfit:
            request.draft = None
            request.warnings.append(
                ("decoded without drafting", {"why": "the draft head does not fit"})
            )
        if request.draft is not None and request.draft is not self.draft:
            if len(free) < len(self.slots):
                return False  # another head's rows decode: it waits for them
            self._drop()  # the engine takes this head (its keys' shapes) when idle
            self.draft, self.rehold = request.draft, True
            return False  # admitted once its blocks are held
        try:
            first = None if request.allowed is None else _ids(request.allowed, [], self.vocab)
        except Exception as exc:  # the call's own constraint, asked before anything of it ran
            request.error = exc
            return True
        kwargs, mask, position = self._inputs(request)
        if not self._enter(request, position):
            return False
        fresh = self._prefill(request, kwargs, mask, position, first)
        if why := self._misfit(fresh):
            self.misfit = f"its decode state is not what its config gives ({why})"
            request.error = RuntimeError(f"[lm] {type(self.model).__name__}: {self.misfit}")
            return True
        self._splice(request, fresh, [slot for slot, row in enumerate(self.slots) if row is None])
        return True

    def _enter(self, request: _Request, position: torch.Tensor | None) -> bool:
        """Whether `request`'s rows have their decode memory, decided before its prefill, the one
        place it is: the slots held now where they can take it (the decode graphs' pool mapped
        back for them), else slots for every row's prompt and `max_new_tokens` allocated now
        (`_buffers`; `_need` is what they take). False: it waits for the rows decoding now to
        leave. `DoesNotFit` before anything of it ran: up front only where the slots exceed the
        whole card (`_never`), else where their allocation, which asks idle executors for their
        memory, evicts and waits as every allocation does, was refused after all of that."""
        rows = len(request.tokens)
        reach = request.prompt + request.max_new + DEPTH * (LAG + 2)
        free = sum(row is None for row in self.slots)
        occupied = len(self.slots) - free
        request.waits = False
        if (
            self.cache is not None
            and free >= rows
            and reach <= self.capacity
            and len(request.stops) <= self.width
        ):
            try:
                self._pool(True)
            except Exhausted as exc:  # no room for the pool to come back: nothing of it ran
                raise DoesNotFit(
                    f"{type(self.model).__name__} decode: its graphs' pool cannot come back "
                    f"beside {_gib(self.held)} of keys and state, and nothing more is free on "
                    f"this card ({exc}): run it smaller, or rent a larger card: "
                    "`cozy run … --on new`"
                ) from exc
            return True
        if occupied + rows > BUCKETS[-1]:
            request.waits = True  # slots free up as rows leave
            return False
        alone = (_batch(rows), _windows(reach), max(8, len(request.stops)))
        size, capacity, stops = alone
        if occupied:
            size, capacity = _batch(occupied + rows), max(self.capacity, capacity)
            stops = max(stops, self.width)
        device = request.inputs["input_ids"].device
        placed = device.type == "cuda" and self.model in PLACED
        if placed:
            card = self._card(device) + self.held
            if _never(self._need(*alone, position), card):
                raise DoesNotFit(self._refused(request, alone, position, card))
        if not occupied:
            self._drop()  # the idle slots go before the new ones come
        try:
            self._grow(size, capacity, stops, position, device)
        except Exception as exc:
            if not (placed and is_capacity(exc)):
                raise
            if occupied:
                request.waits = True  # it is tried again once these rows leave
                return False
            need = self._need(size, capacity, stops, position)
            raise DoesNotFit(
                f"{type(self.model).__name__} decode: "
                f"{no_room(device.index or 0, 'its decode state', need, exc)}: run it smaller, "
                "or rent a larger card: `cozy run … --on new`"
            ) from exc
        return True

    def _card(self, device: torch.device) -> int:
        """The most this process could ever have for slots: the whole card, less its own live
        working memory and weights in use. What other processes hold is not counted against it:
        they give memory back when an allocation asks."""
        room = vt.room(device.index or 0)
        total = int(torch.cuda.get_device_properties(device).total_memory)
        return max(total - int(room["working_live"]) - int(room["pinned"]), 0)

    def _refused(
        self,
        request: _Request,
        alone: tuple[int, int, int],
        position: torch.Tensor | None,
        room: int,
    ) -> str:
        """`DoesNotFit`'s line for slots the card cannot hold, with the longest that it could."""
        rows, prompt = len(request.tokens), request.prompt
        size, _, stops = alone
        # The bytes grow by the same amount each window of positions.
        one = self._need(size, MIN_WINDOW, stops, position)
        per = (self._need(size, 2 * MIN_WINDOW, stops, position) - one) / MIN_WINDOW
        fits = int(MIN_WINDOW + (room - one) / per) // MIN_WINDOW * MIN_WINDOW
        most = fits - prompt - DEPTH * (LAG + 2)
        smaller = f"max_new_tokens ≤ {most}" if most > 0 else "a shorter prompt"
        if rows > 1:
            smaller += " or fewer rows"
        return (
            f"{type(self.model).__name__} decode: {rows} row{'s' * (rows > 1)} of {prompt} + "
            f"{request.max_new} tokens need {_gib(self._need(*alone, position))} of keys and "
            f"state, and the whole card has {_gib(room)} beside the weights in use and live "
            "memory: "
            f"run it smaller ({smaller}), or rent a larger card: `cozy run … --on new`"
        )

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
            token = scores.argmax(-1)
            if request.sampling.temperature > 0:
                if request.sampling.top_k:
                    scores = _top_k(scores, request.sampling.top_k)
                rows = int(ids.shape[0])
                first = torch.zeros((rows,), dtype=torch.long, device=ids.device)
                token = _draw(scores, *_per_row(request.sampling, rows, ids.device), first)
            drafted: tuple[_Keys, torch.Tensor] | None = None
            if self.draft is not None:
                drafted = self._drafted_prompt(ids, token, hidden, position)
        request.prefill_s += time.perf_counter() - started
        return _Prefill(fresh, mask, position, token, drafted)

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
        prompt = int(ids.shape[1])
        tokens = torch.cat([ids[:, 1:], token[:, None]], dim=1)
        slots = torch.arange(prompt + 1, device=ids.device)
        allowed = (slots[None, :] >= 1) & (slots[None, :] <= slots[1:, None])
        mask = (
            allowed
            if self.closed is False
            else torch.where(allowed, self.open, self.closed).to(self.model.dtype)
        )
        mask = mask[None, None].expand(int(ids.shape[0]), 1, prompt, prompt + 1)
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
        drafted: torch.Tensor = _set(self.draft)(embed(tokens), hidden, at, cache, mask, rotary)
        last = drafted[:, -1:]
        return keys, last

    def _splice(self, request: _Request, fresh: _Prefill, free: list[int]) -> None:
        """`fresh`'s rows into slots `free`, each with its prompt's keys, state, mask and
        positions and its first token."""
        prompt = request.prompt
        stops = torch.full((self.stops.shape[1],), -1, dtype=torch.long)
        stops[: len(request.stops)] = torch.tensor(request.stops, dtype=torch.long)
        firsts = fresh.token.tolist() if request.allowed is not None else []
        with _disable_current_modes():
            for row in range(int(fresh.token.shape[0])):
                slot = free[row]
                for mine, theirs in _pairs(_set(self.cache), fresh.cache):
                    if isinstance(mine, _Keys):
                        mine.keys[slot, :, :prompt].copy_(_like(mine, theirs).keys[row])
                        mine.values[slot, :, :prompt].copy_(_like(mine, theirs).values[row])
                        mine.cumulative_length[slot] = prompt
                    else:
                        given = _like(mine, theirs)
                        for i, state in mine.conv_states.items():
                            if state is not None:
                                state[slot].copy_(_set(given.conv_states[i])[row])
                        for i, state in mine.recurrent_states.items():
                            if state is not None:
                                state[slot].copy_(_set(given.recurrent_states[i])[row])
                self.length[slot] = prompt
                self.mask[slot].fill_(self.closed)
                self.mask[slot, 0, 0, :prompt].masked_fill_(fresh.mask[row].bool(), self.open)
                if fresh.position is not None:
                    self.position[slot].copy_(fresh.position[..., row, -1:] + 1)
                token = fresh.token[row]
                self.ids[slot, 0] = token
                self.out[slot, 0] = token
                self.n[slot] = 1
                self.limit[slot] = request.max_new
                self.stops[slot].copy_(stops)
                self.pad[slot] = request.pad
                self.penalty[slot] = request.penalty
                self.temperature[slot] = request.sampling.temperature
                self.top_p[slot] = request.sampling.top_p
                self.top_k[slot] = request.sampling.top_k if request.sampling.temperature > 0 else 0
                self.seed[slot] = request.sampling.seed
                self.seen[slot].zero_()
                self.seen[slot, token] = 1.0
                self.done[slot] = bool((self.stops[slot] == token).any()) or request.max_new <= 1
                if self.draft is not None:
                    keys, last = _set(fresh.drafted)
                    if keys.keys is not None:
                        self.dkeys.keys[slot, :, : prompt + 1].copy_(keys.keys[row])
                        self.dkeys.values[slot, :, : prompt + 1].copy_(keys.values[row])
                    self.dmask[slot].fill_(self.closed)
                    self.dmask[slot, ..., 1 : prompt + 1] = self.open
                    self.hidden[slot].copy_(last[row])
                held = self.slots[slot] = _Row(request, row, prompt)
                if firsts:
                    held.generated = [firsts[row]]
        request.began = time.perf_counter()
        request.at = (self.captured[0], self.captured[1], self.eager)
        self.retire = self.retire or bool(self.done[free[: int(fresh.token.shape[0])]].any())

    def _grow(
        self,
        size: int,
        capacity: int,
        stops: int,
        position: torch.Tensor | None,
        device: torch.device,
    ) -> None:
        """Allocate the slots before the call's prefill, in the KV region."""
        with self.region.pin() if self.region is not None else nullcontext():
            self._buffers(size, capacity, stops, position, device)

    def _layout(
        self, size: int, capacity: int, stops: int, position: torch.Tensor | None
    ) -> tuple[list[list[_Spec]], list[_Spec], dict[str, _Spec]]:
        """What `size` slots of `capacity` positions hold, from the config alone (`_shapes`),
        each tensor as (shape, dtype, fill): every cache layer's, the head's keys, the rest."""
        dtype, long, f32 = self.model.dtype, torch.long, torch.float32
        shapes = _shapes(self.text)[0]
        layers: list[list[_Spec]] = []
        for shape in shapes:
            if shape[0] == "keys":
                _, heads, k_width, v_width = shape
                layers.append(
                    [
                        ((size, heads, capacity, k_width), dtype, 0),
                        ((size, heads, capacity, v_width), dtype, 0),
                        ((size,), long, 0),
                    ]
                )
            else:
                _, conv, recurrent = shape
                layers.append([((size, *conv), dtype, 0), ((size, *recurrent), f32, 0)])
        kind = torch.bool if self.closed is False else dtype
        mask: _Spec = ((size, 1, 1, capacity), kind, self.closed)
        lead = tuple(position.shape[:-2]) if position is not None else ()
        fields: dict[str, _Spec] = {
            "ids": ((size, 1), long, 0),
            "length": ((size,), long, 0),
            "mask": mask,
            "out": ((size, capacity), long, 0),
            "n": ((size,), long, 0),
            "done": ((size,), torch.bool, True),
            "limit": ((size,), long, 0),
            "stops": ((size, stops), long, -1),
            "pad": ((size,), long, 0),
            "penalty": ((size,), f32, 0),
            "temperature": ((size,), f32, 0),
            "top_p": ((size,), f32, 1),
            "top_k": ((size,), long, 0),
            "seed": ((size,), long, 0),
            "seen": ((size, self.vocab), f32, 0),
            "position": ((size, *lead, 1), long, 0),
            # each step position's allowed tokens, as bits the host writes (`_fill`)
            "allow": ((DEPTH + 1, size, -(-self.vocab // 8)), torch.uint8, 255),
        }
        draft: list[_Spec] = []
        if self.draft is not None:
            # `Draft`'s one layer is the model's own full attention: its keys are as the model's.
            heads, width = next((s[1], s[2]) for s in shapes if s[0] == "keys")
            keys: _Spec = ((size, heads, capacity, width), dtype, 0)
            draft = [keys, keys, ((size,), long, 0)]
            fields["dmask"] = mask
            fields["hidden"] = ((size, 1, int(self.text.hidden_size)), dtype, 0)
            fields["drafts"] = ((size, DEPTH), long, 0)  # a step's drafts, between its parts
            fields["dhidden"] = fields["hidden"]
        return layers, draft, fields

    def _need(self, size: int, capacity: int, stops: int, position: torch.Tensor | None) -> int:
        """The device bytes `_buffers` allocates for these slots."""
        layers, draft, fields = self._layout(size, capacity, stops, position)
        specs = [*(spec for layer in layers for spec in layer), *draft, *fields.values()]
        each = sum(math.prod(shape) * dtype.itemsize for shape, dtype, _ in specs)
        return each + capacity * torch.long.itemsize + 8  # and `rows_index`, `bit`

    def _misfit(self, fresh: _Prefill) -> str:
        """How a row of the prefill's state differs from the slots sized from the config, ""
        when it does not: a copy into a slot of another dtype would cast it silently."""
        pairs: list[tuple[object, torch.Tensor, torch.Tensor | None, int]] = []
        for j, (layer, other) in enumerate(_pairs(_set(self.cache), fresh.cache)):
            if isinstance(layer, _Keys):
                keys_ = _like(layer, other)
                pairs += [(j, layer.keys, keys_.keys, 2), (j, layer.values, keys_.values, 2)]
            else:
                state = _like(layer, other)
                pairs += [
                    (j, _set(layer.conv_states[0]), state.conv_states.get(0), -1),
                    (j, _set(layer.recurrent_states[0]), state.recurrent_states.get(0), -1),
                ]
        if self.draft is not None:
            keys, last = _set(fresh.drafted)
            if keys.keys is not None:  # `Draft` writes keys; a head of none reads none
                pairs += [("head", self.dkeys.keys, keys.keys, 2)]
                pairs += [("head", self.dkeys.values, keys.values, 2)]
            pairs.append(("head", self.hidden, last, -1))

        def row(t: torch.Tensor, positions: int) -> tuple[object, ...]:
            return (*(n for d, n in enumerate(t.shape) if d not in (0, positions)), t.dtype)

        for where, mine, theirs, positions in pairs:
            if theirs is None or row(mine, positions) != row(theirs, positions):
                found = "nothing" if theirs is None else row(theirs, positions)
                return f"layer {where}: {found} a row, sized {row(mine, positions)}"
        return ""

    def _buffers(
        self,
        size: int,
        capacity: int,
        stops: int,
        position: torch.Tensor | None,
        device: torch.device,
    ) -> None:
        """Every slot buffer for `size` rows of `capacity` positions (`_layout`), all allocated
        before any replaces the engine's, then the rows held now moved over."""

        def full(spec: _Spec) -> torch.Tensor:
            shape, dtype, fill = spec
            return torch.full(shape, fill, dtype=dtype, device=device)

        layers, draft_specs, specs = self._layout(size, capacity, stops, position)
        old, held = self.cache, [row for row in self.slots if row is not None]
        cache = _StaticCache(self.text, capacity, MIN_WINDOW, rows=True)
        for mine, layer in zip(_layers(cache), layers, strict=True):
            tensors = [full(spec) for spec in layer]
            mine.dtype, mine.device = tensors[0].dtype, device
            if isinstance(mine, _Keys):
                mine.keys, mine.values, mine.cumulative_length = tensors
                mine.is_initialized = True
            else:
                mine.conv_states[0], mine.recurrent_states[0] = tensors
                mine.conv_kernel_size[0] = int(tensors[0].shape[-1])
                mine.is_conv_states_initialized[0] = True
                mine.has_previous_state[0] = True
                mine.is_recurrent_states_initialized[0] = True
        fields = {name: full(spec) for name, spec in specs.items()}
        draft: _Keys | None = None
        if self.draft is not None:
            draft = _Keys([MIN_WINDOW], capacity, rows=True)
            draft.keys, draft.values, draft.cumulative_length = (full(s) for s in draft_specs)
            draft.dtype, draft.device = draft.keys.dtype, device
            draft.is_initialized = True
        rows_index = torch.arange(capacity, device=device)
        bit = torch.arange(8, dtype=torch.uint8, device=device)
        moving = list(self._rows()) if old is not None else []
        self.cache, self.capacity, self.cache_device = cache, capacity, device
        for name, tensor in fields.items():
            setattr(self, name, tensor)
        if self.draft is not None:
            self.dkeys = _set(draft)
            self.dcache = Cache(layers=[self.dkeys])
        # The rows held now keep their slots.
        if moving:
            with _disable_current_modes():
                for before, after in zip(moving, self._rows(), strict=True):
                    span = [
                        slice(0, min(a, b)) for a, b in zip(before.shape, after.shape, strict=True)
                    ]
                    after[tuple(span)].copy_(before[tuple(span)])
        self.slots = held + [None] * (size - len(held))
        for slot in range(len(held), size):
            self._empty(slot)
        self.rows_index, self.bit = rows_index, bit
        pinned = device.type == "cuda"
        self.flag = torch.zeros((size,), dtype=torch.bool, pin_memory=pinned)
        # What a constrained step's host reads back (`_took`, `_drafted`) and writes (`_fill`).
        self.took = torch.zeros(((DEPTH + 2) * size,), dtype=torch.long, pin_memory=pinned)
        self.dtook = torch.zeros((DEPTH * size,), dtype=torch.long, pin_memory=pinned)
        self.allow_host = torch.full(
            tuple(self.allow.shape), 255, dtype=torch.uint8, pin_memory=pinned
        )
        self.views = {}
        self._forget()
        self.width = stops
        self.held = self._need(size, capacity, stops, position)

    # ------------------------------------------------------------------ steps

    def _view(self, size: int) -> _StaticCache:
        """The cache over the first `size` slots: views of the buffers, which graphs read."""
        view = self.views.get(size)
        if view is None:
            cache = _set(self.cache)
            view = _StaticCache(self.text, self.capacity, cache.window, rows=True)
            view.size = cache.size
            for mine, theirs in _pairs(view, cache):
                if isinstance(mine, _Keys):
                    full = _like(mine, theirs)
                    mine.size = cache.size
                    mine.dtype, mine.device = full.dtype, full.device
                    mine.keys, mine.values = full.keys[:size], full.values[:size]
                    mine.cumulative_length = full.cumulative_length[:size]
                    mine.is_initialized = True
                else:
                    state = _like(mine, theirs)
                    mine.dtype, mine.device = state.dtype, state.device
                    for i, state_ in state.conv_states.items():
                        if state_ is not None:
                            mine.conv_states[i] = state_[:size]
                            mine.conv_kernel_size[i] = state.conv_kernel_size[i]
                            mine.is_conv_states_initialized[i] = True
                            mine.has_previous_state[i] = True
                    for i, state_ in state.recurrent_states.items():
                        if state_ is not None:
                            mine.recurrent_states[i] = state_[:size]
                            mine.is_recurrent_states_initialized[i] = True
            if self.draft is not None:
                keys = _Keys(cache.size, self.capacity, rows=True)
                keys.dtype, keys.device = self.dkeys.dtype, self.dkeys.device
                keys.keys, keys.values = self.dkeys.keys[:size], self.dkeys.values[:size]
                keys.cumulative_length = self.dkeys.cumulative_length[:size]
                keys.is_initialized = True
                view.draft = Cache(layers=[keys])
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
                event = torch.cuda.Event()  # type: ignore[no-untyped-call]  # unannotated in Torch
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
                    try:  # a row with `generated` has its call's constraint
                        allowed = _set(row.request.allowed)
                        _pack(_ids(allowed, tokens, self.vocab), host[slot])
                        free = False
                    except Exception as exc:  # the call's own constraint: only it fails
                        row.request.error = row.request.error or exc
            if free:
                host[slot].fill_(255)
        self.allow[at, :size].copy_(host, non_blocking=True)

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
        """Each held call's progress, its longest row's tokens: drafting decodes several a step. The
        flag check has just caught the host up with the device, so reading `n` waits on nothing."""
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

        for request in {id(row.request): row.request for row in self.slots if row}.values():
            request.tell(partial(tokens, request))

    def _read(self, drafted: bool) -> tuple[torch.nn.Module, ...]:
        """The modules a step reads: the model, and its head when the step drafts."""
        return (self.model, _set(self.draft)) if drafted else (self.model,)

    @contextmanager
    def _graph_memory(self, drafted: bool) -> Iterator[None]:
        """The leases of the weights the step reads and the domain's side stream, joined back
        before a lease can end."""
        current = torch.cuda.current_stream()
        side = None
        domain = _set(self.domain)
        try:
            with ExitStack() as stack:
                for module in self._read(drafted):
                    if (binding := self.addresses.get(id(module))) is not None:
                        stack.enter_context(binding.prepare(current))
                domain.initialize()
                side = domain.stream
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

    def _replay(self, size: int, cuda: bool) -> None:
        drafted = self._drafting(size)
        constrained = self._constrained()
        part = self.part if drafted else -1
        window_ = _set(self.cache).window
        key = (size, window_, drafted, self._sampled(), self._kept(), constrained, part)
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
            self._advance(size)
            self.eager += 1
            return
        graph = self.graphs.get(key)
        if graph is None and cuda and key not in self.unrecorded and self.building != LANDED[0]:
            domain = _set(self.domain)  # a CUDA step's (`_one`)
            if key not in self.warmed:
                # This is the caller's actual next token, never an extra speculative warmup; its
                # working peak, as varena measured it, is what its capture's pool takes.
                with self._graph_memory(drafted), _measured(domain.device) as peak:
                    self._advance(size)
                self.warmed.add(key)
                if peak:
                    self.peaks[key] = peak[0]
                self.eager += 1
                return
            if key not in self.peaks:  # a step never measured is never recorded
                self._unrecorded(key, "its warm-up's working memory was not measured")
                self._advance(size)
                self.eager += 1
                return
            capturing = time.perf_counter()
            reserve, why = vt.pool_bytes(self.peaks[key]), ""
            try:
                with (
                    self._graph_memory(drafted),
                    request_path.known("cozy_runtime.lm._Engine._replay"),
                ):
                    if self.pool is None:
                        self.pool = (
                            placed[0].graph_pool(placed[1])
                            if placed is not None
                            else Pool(torch.cuda.graph_pool_handle())
                        )
                    try:
                        graph, why = capture(
                            torch, lambda: self._advance(size), _set(self.pool), domain, reserve
                        )
                    except Exhausted as exc:  # its pool's room, before anything was captured
                        why = no_room(domain.device, "its pool", reserve, exc)
            except Building:  # eager until a kernel lands
                graph, why, self.building = None, "", LANDED[0]
            finally:
                if graph is None and not self.graphs and self.pool is not None:
                    close_pool(self.pool, domain.device)
                    self.pool = None
            if graph is not None:
                self.graphs[key] = graph
            # A capture's end, recorded or not, ends the pool's use: held again for the rows
            # decoding while its pages are still mapped, so no replay maps.
            self.pooled = False
            self._pool(True)
            self.captured[0] += 1
            self.captured[1] += time.perf_counter() - capturing
            if graph is not None:
                self.recorded += 1
                try:
                    vt.pool_scratch(_set(self.pool).handle, (), domain.device)
                except (RuntimeError, ValueError) as exc:  # A live allocation: it stays mapped.
                    safeguard("scratch_declined", str(exc))
                self._verify(key, graph, size)
                return
            elif why:
                self._unrecorded(key, why)
        if graph is not None:  # its pool is held (`_pool`): the replay maps nothing
            vt.replay(graph, stream=torch.cuda.current_stream(_set(self.domain).device))
            self.replayed += 1
        else:
            self._advance(size)
            self.eager += 1

    def _token_state(self, drafted: bool) -> list[torch.Tensor]:
        """What a step writes but its keys, each slot of which it writes before reading; a drafted
        step's head state, mask and key count too."""
        state = [self.ids, self.mask, self.out, self.n, self.done, self.seen, self.length]
        state.append(self.position)
        if drafted:
            state += [self.hidden, self.dmask, self.dkeys.cumulative_length]
            state += [self.drafts, self.dhidden]
        for layer in _layers(_set(self.cache)):
            if isinstance(layer, _Keys):
                state.append(layer.cumulative_length)
            else:
                held = (*layer.conv_states.values(), *layer.recurrent_states.values())
                state += [value for value in held if value is not None]
        return state

    def _written(self, size: int, at: torch.Tensor) -> list[torch.Tensor]:
        """Each row's keys and values at its slot `at`."""
        rows = torch.arange(size, device=at.device)
        layers = [layer for layer in _layers(_set(self.cache)) if isinstance(layer, _Keys)]
        return [buffer[rows, :, at] for layer in layers for buffer in (layer.keys, layer.values)]

    def _verify(self, key: _Key, graph: torch.cuda.CUDAGraph, size: int) -> None:
        """A new graph's one check: its replay, then this token's eager step from the same
        state, last so that what serves the call is eager's. A difference discards the graph."""
        domain = _set(self.domain)
        state, stream = self._token_state(key[2]), torch.cuda.current_stream(domain.device)
        start, at = [value.clone() for value in state], self.length[:size].clone()
        vt.replay(graph, stream=stream)  # its pool is held since its capture
        replayed = [*(value.clone() for value in state), *self._written(size, at)]
        for tensor, value in zip(state, start, strict=True):
            tensor.copy_(value)
        self._advance(size)
        self.eager += 1
        why = vt.replay_differs([*state, *self._written(size, at)], replayed)
        if why:
            stream.synchronize()
            if len(self.graphs) == 1:
                self._pool(False)  # the pool's last graph goes: its use ends while it owns it
            del self.graphs[key]
            graph.reset()
            if not self.graphs:
                close_pool(_set(self.pool), domain.device)
                self.pool = None
            self._unrecorded(key, why, "replay_mismatch")

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
            self._pool(False)
            self.retired_graphs += len(self.graphs)
            for graph in self.graphs.values():
                graph.reset()
            if self.pool is not None:
                close_pool(self.pool, _set(self.domain).device)
            self.graphs, self.warmed, self.unrecorded, self.peaks = {}, set(), {}, {}
            self.undrafted = False
            self.pool = None

    def _pool(self, hold: bool) -> None:
        """Hold the graphs' pool mapped for the rows decoding (`vt.prepare_pool(pool, 0)`), or
        end that use so it may go idle (`vt.record_pool_stream`). Taken at a call's admission,
        before its prefill, and after a capture; let go when no row decodes. Only an admission's
        take can need room: it refuses there (`Exhausted`), never at a replay."""
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
        """That step runs eagerly from now on, with a warning on the calls it serves, unless
        `code` is None: a property of the code, in the record alone (`decode.refused`)."""
        self.unrecorded[key] = why
        if code is None:
            return
        safeguard(code, f"{type(self.model).__name__}: {why}")
        warning: WarningRow = (
            "a CUDA graph was not used",
            {"module": type(self.model).__name__, "why": why},
        )
        for request in {id(r.request): r.request for r in self.slots if r is not None}.values():
            if warning not in request.warnings:
                request.warnings.append(warning)
        for row in self.slots:
            if row is not None:
                row.request.warnings.append(
                    ("a CUDA graph was not used", {"module": type(self.model).__name__, "why": why})
                )

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
        return any(row is not None and row.request.allowed is not None for row in self.slots)

    def _allowed(self, scores: torch.Tensor, size: int, at: int | slice) -> torch.Tensor:
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

    def _kept(self) -> int:
        """The most logits a held sampling row keeps (`top_k`), 0 when none cuts: a step's
        constant, so its graph takes `topk` of that many."""
        rows = [row.request.sampling for row in self.slots if row is not None]
        return min(max((s.top_k for s in rows if s.temperature > 0), default=0), self.vocab)

    def _keep(self, scores: torch.Tensor, size: int) -> torch.Tensor:
        """`scores` (`[size, ..., vocab]`, before the presence penalty) with each sampling row's
        logits below its `top_k`th largest at -inf; a greedy row's (`top_k` 0) as they are."""
        if not (most := self._kept()):
            return scores
        lead = (-1, *[1] * (scores.dim() - 1))
        rows = self.top_k[:size]
        at = (rows - 1).clamp(0, most - 1).view(lead).expand(*scores.shape[:-1], 1)
        kth = scores.topk(most, dim=-1).values.gather(-1, at)
        return scores.masked_fill((rows > 0).view(lead) & (scores < kth), float("-inf"))

    def _pick(
        self, scores: torch.Tensor, size: int, index: torch.Tensor, sampled: bool
    ) -> torch.Tensor:
        """Each row's token from `scores` (`[size, ..., vocab]`): its argmax, or a sampling row's
        draw (`_draw`) for the token at `index`."""
        best = scores.argmax(-1)
        if not sampled:
            return best
        temperature = self.temperature[:size]
        drawn = _draw(scores, temperature, self.top_p[:size], self.seed[:size], index)
        return torch.where((temperature > 0).view(-1, *[1] * (best.dim() - 1)), drawn, best)

    def _advance(self, size: int) -> None:
        if self._drafting(size):
            self._speculate(size)
        else:
            self._step(size)

    def _select(self, logits: torch.Tensor, size: int) -> torch.Tensor:
        """Each live row's next token, on the device, in `ids` and `out`; a row that stops (a
        stop token, or its limit) is done. Returns the rows live before it."""
        scores = self._keep(self._allowed(logits.float(), size, 0), size)
        scores = scores - self.penalty[:size, None] * self.seen[:size]
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
        self.seen[:size].scatter_reduce_(1, token[:, None], live.float()[:, None], "amax")
        self.flag[:size].copy_(done, non_blocking=True)
        return live

    def _step(self, size: int) -> None:
        """One token a row: attend to the window, write the row's keys at its length, select."""
        view, window_ = self._view(size), _set(self.cache).window
        length = self.length[:size]
        self.mask[:size].scatter_(3, length.view(size, 1, 1, 1), self.open)
        kwargs: dict[str, torch.Tensor | _StaticCache | bool | int] = {
            "input_ids": self.ids[:size],
            "attention_mask": self.mask[:size, ..., :window_],
            "past_key_values": view,
            "use_cache": True,
            "return_dict": True,
        }
        if self.positional:
            kwargs["position_ids"] = self.position[:size].movedim(0, -2)
        if self.keep:
            kwargs["logits_to_keep"] = 1
        live = self._select(self.model(**kwargs).logits[:, -1], size)
        length.add_(live.long())
        for layer in _layers(view):
            if isinstance(layer, _Keys):
                layer.cumulative_length.copy_(length)
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
        first's, `hidden`, the step before left; a later one's from the draft before it)."""
        model, w = self.model, _set(self.cache).window
        view, length = self._view(size), self.length[:size]
        if not i:
            _head_keys(view.draft).cumulative_length.copy_(length + 1)
            hidden = self.hidden[:size]
        else:
            L, keys = length[:, None], self.rows_index[:w]
            open_ = (keys[None] > L) & (keys[None] <= L + i)
            mask = torch.where(open_[:, None, None], self.open, self.dmask[:size, ..., :w])
            position = self.position[:size]
            at = (position + i).movedim(0, -2) if self.positional else (length + i)[:, None]
            embed, rotary = model.get_input_embeddings(), _lm(model).get_decoder().rotary_emb
            before = self.hidden[:size] if i == 1 else self.dhidden[:size]
            token = self.drafts[:size, i - 1 : i]
            hidden = _set(self.draft)(embed(token), before, at, view.draft, mask, rotary)
            if i + 1 < DEPTH:
                self.dhidden[:size].copy_(hidden)
        seen = self.seen[:size]
        if i:  # and the drafts before it
            seen = seen.scatter(1, self.drafts[:size, :i], 1.0)
        logits = _lm(model).get_output_embeddings()(hidden)[:, -1].float()
        scores = self._keep(self._allowed(logits, size, i), size)
        scores = scores - self.penalty[:size, None] * seen
        token = self._pick(scores, size, self.n[:size] + i, self._sampled())
        self.drafts[:size, i].copy_(token)
        if self._constrained():  # the host's, for what may follow it
            self.dtook[i * size : (i + 1) * size].copy_(token, non_blocking=True)

    def _commit(self, size: int) -> None:
        """The drafts verified with the row's token before them in one pass, what it keeps
        written, and the head's keys and state and the model's states put at its end."""
        model, depth, w = self.model, DEPTH, _set(self.cache).window
        view = self._view(size)
        span = depth + 1
        embed, rotary = model.get_input_embeddings(), _lm(model).get_decoder().rotary_emb
        keys, steps = self.rows_index[:w], self.rows_index[:span]
        length = self.length[:size]
        L = length[:, None]
        done = self.done[:size]
        live = ~done
        dkeys = _head_keys(view.draft)
        dmask = self.dmask[:size, ..., :w]
        position = self.position[:size]
        sampled, n = self._sampled(), self.n[:size]
        drafted = self.drafts[:size]
        # Verify: each row attends to its committed keys and this pass's up to its own.
        open_ = (keys[None, None] >= L[..., None]) & (
            keys[None, None] <= (L + steps[None])[..., None]
        )
        mask = torch.where(open_[:, None], self.open, self.mask[:size, ..., :w])
        kwargs: dict[str, torch.Tensor | _StaticCache | bool | int] = {
            "input_ids": torch.cat([self.ids[:size], drafted], dim=1),
            "attention_mask": mask,
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
            with _taking(model) as taken, _RowByRow():
                scores: torch.Tensor = model(**kwargs).logits[:, -span:].float()
            scores = self._keep(self._allowed(scores, size, slice(0, span)), size)
        finally:
            _VERIFY.reset(reset)
        # row j has seen the drafts before it
        ahead = torch.zeros_like(scores).scatter_(2, drafted[..., None], 1.0).cumsum(1)
        ahead = torch.cat([torch.zeros_like(ahead[:, :1]), ahead[:, :-1]], dim=1)
        scores = scores - self.penalty[:size, None, None] * torch.maximum(
            self.seen[:size, None], ahead.clamp(max=1)
        )
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
        self.seen[:size].scatter_reduce_(1, best, keep.float(), "amax")
        last = (made - 1).clamp(min=0)
        self.ids[:size].copy_(
            torch.where(live[:, None], best.gather(1, last[:, None]), self.pad[:size, None])
        )
        # The head's keys for the committed tokens, each from the model's own state for the
        # token before it, and its state for the token after them: the next step's first draft.
        dkeys.cumulative_length.copy_(length + 1)
        open_ = (keys[None, None] > L[..., None]) & (
            keys[None, None] <= (L + 1 + steps[None])[..., None]
        )
        mask = torch.where(open_[:, None], self.open, dmask)
        at = (position + 1 + steps).movedim(0, -2) if self.positional else (L + 1 + steps[None])
        head = _set(self.draft)
        states: torch.Tensor = head(
            embed(tokens), taken[-1][:, -span:], at, view.draft, mask, rotary
        )
        self.hidden[:size].copy_(
            states.gather(1, last[:, None, None].expand(-1, 1, states.shape[-1]))
        )
        # Back to the kept tokens' end: keys past it are masked and overwritten later.
        ended = length + made
        for layer in _layers(view):
            if isinstance(layer, _Keys):
                layer.cumulative_length.copy_(ended)
        rows = torch.arange(size, device=length.device)
        for state, full in verify.conv:
            width_ = state.shape[-1]
            index = (self.rows_index[:width_][None] + made[:, None]).clamp(max=full.shape[-1] - 1)
            state.copy_(full.gather(-1, index[:, None, :].expand(-1, full.shape[1], -1)))
        for state, states_ in verify.recurrent:
            state.copy_(states_[(made - 1).clamp(min=0), rows])
        every = self.rows_index[: self.capacity]
        self.mask[:size, 0, 0].masked_fill_(
            (every[None] >= L) & (every[None] < ended[:, None]), self.open
        )
        self.dmask[:size, 0, 0].masked_fill_(
            (every[None] > L) & (every[None] <= ended[:, None]), self.open
        )
        self.length[:size].copy_(ended)
        if self._constrained():  # what the host reads back: how many, then each token
            told = torch.cat([made[None], tokens.T])
            self.took[: (span + 1) * size].view(span + 1, size).copy_(told, non_blocking=True)
        n.add_(made)
        self.position[:size].add_(made.view(size, *[1] * (self.position.dim() - 1)))
        done |= stopped.any(-1)
        self.flag[:size].copy_(done, non_blocking=True)

    # ------------------------------------------------------------------ memory

    def _try_drop(self) -> bool:
        with cache_owners.try_locks([_PLANE, self.cond]) as claimed:
            if not claimed or self.driver is not None or self.pending or self._occupied():
                return False
            # _PLANE fixes the driver's binding inventory before its locks are claimed.
            bindings = tuple(self.addresses.values())
            locks = [self.domain.lock] if self.domain is not None else []
            locks.extend(binding.lock for binding in bindings)
            locks.extend(binding.model.memory.lock for binding in bindings)
            with cache_owners.try_locks(locks) as claimed:
                if not claimed:
                    return False
                if self.unfenced or any(
                    not event.query()  # type: ignore[no-untyped-call]  # unannotated in Torch
                    for event in self.completions.values()
                ):
                    return False
                if any(
                    binding.completion_lost
                    or (binding.last is not None and not binding.last.query())
                    for binding in bindings
                ):
                    return False
                self._drop()
                for binding in bindings:  # Retired graphs no longer read their Regions.
                    binding.close()
                self.addresses.clear()
                self.where = ()
                return True

    def _drop(self) -> None:
        if self.cache_device is not None and self.cache_device.type == "cuda":
            self._forget()
        self.cache, self.views = None, {}
        self.slots, self.capacity, self.held = [], 0, 0
        self.layout = None
        # In particular, the pinned host flag must free while the domain stream is alive:
        # Torch's host allocator records that stream's completion when freeing the tensor.
        pinned = ("flag", "allow_host", "took", "dtook")
        for name in (*type(self).__annotations__, *pinned, "rows_index", "bit", "dkeys", "dcache"):
            vars(self).pop(name, None)


def _never(need: int, card: int) -> bool:
    """Whether slots of `need` bytes can never fit: more than the card with every other byte
    gone. A need under that is allocated, and the allocation makes its room or is refused."""
    return need > card


__all__ = [
    "BUCKETS",
    "DEPTH",
    "KERNELS",
    "MIN_WINDOW",
    "Constraint",
    "Draft",
    "drafting",
    "generate",
    "install",
    "release",
    "window",
]


DECODERS.append((_graph_snapshot, release))
