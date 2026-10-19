"""Fast decode for Hugging Face causal LMs: a token costs what streaming the weights costs, not
the host's Python.

    from cozy_runtime import lm

    new = lm.generate(model, inputs, max_new_tokens=160, stop=[eos])

The prompt prefills through the model's own forward into a static cache: keys and values in fixed
buffers, and a linear-attention layer's conv and recurrent state, which are fixed already. Every
later token is one replay of a CUDA graph of the one-token step, captured per (batch, window):
the window is the attention length, a power of two that grows with the sequence. The step picks
its token on the GPU (greedy, with an optional presence penalty), writes it where the next step
reads it and advances its own positions, so replays queue back to back and the host reads the
stop flag a few steps behind.

A graph reads the weights at their plane homes: a weight set keeps one device address range for
its life, and a refill maps a region back at the same address. For the loop Runtime leases every
block the step reads at home and turns their paging hooks off (`Residency.held`). Where they do
not all fit, or the device has no room for a graph's pool beside them, the call decodes
through Transformers' loop, and so does one that runs out of device memory; on a CPU, or after a
capture fails (a run warning), the same step runs eagerly. A model this does not
serve (a layer type other than full or linear attention, an attention other than SDPA or eager, a
generation config that adds logits processors) generates through Transformers, said once.

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
import sys
import threading
import time
import weakref
from collections import deque
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import ExitStack, contextmanager
from typing import Any

from cozy_runtime.author._context import CURRENT, Context
from cozy_runtime.author._errors import is_capacity
from cozy_runtime.author._model import LANDED, PLACED, WARNINGS, Building
from cozy_runtime.author.kernels import KernelsUnavailable, Launch, Plan, TritonKernels
from cozy_runtime.graphs import BYPASS, capture, say
from cozy_runtime.lm import _kernels

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
_DRAFT: contextvars.ContextVar[Any] = contextvars.ContextVar("cozy_lm_draft", default=None)


@contextmanager
def drafting(draft: Any) -> Iterator[None]:
    """`generate` calls in the body draft with `draft` (`Draft`), so a package's model can name
    the head another model holds without passing it through its own (memoized) calls."""
    token = _DRAFT.set(draft)
    try:
        yield
    finally:
        _DRAFT.reset(token)


#: Each model's continuous batch (buffers and graphs), kept until the plane wants the memory.
_ENGINES: weakref.WeakKeyDictionary[Any, _Engine] = weakref.WeakKeyDictionary()
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


def generate(
    model: Any,
    inputs: Mapping[str, Any],
    *,
    max_new_tokens: int,
    stop: Sequence[int],
    pad: int | None = None,
    presence_penalty: float = 0.0,
    draft: Any = None,
) -> Any:
    """Greedy tokens after the prompt, `[batch, n]` (`model.generate(..., do_sample=False)`'s
    tail): `inputs` as the processor returned them, generation ends at a `stop` token or after
    `max_new_tokens`, and a row that stopped early continues as `pad` (default: the first stop).
    `presence_penalty` subtracts that much from every token already generated (vLLM's).
    `draft(embeds, hidden, position_ids, past_key_values, attention_mask, rotary) -> hidden` is
    the model's multi-token-prediction head (`Draft`): from a token's embedding and the state the
    model's head read before it, the state the head reads for the token after; `rotary` is the
    model's own rotary embedding. It speeds up a
    batch of one, each token computed as decode computes it. Without one, the head `drafting`
    names is used."""
    draft = draft if draft is not None else _DRAFT.get()
    stops = [int(token) for token in stop]
    filler = stops[0] if pad is None else int(pad)
    why = ABSENT or _unsupported(model)
    if why:
        say(
            type(model).__name__, f"[lm] {type(model).__name__} decodes through Transformers: {why}"
        )
        return _transformers(model, inputs, max_new_tokens, stops, filler, presence_penalty)
    with _ENGINES_LOCK:
        engine = _ENGINES.get(model)
        if engine is None:
            engine = _ENGINES[model] = _Engine(model)
    request = _Request(inputs, max_new_tokens, stops, filler, presence_penalty)
    token = BYPASS.set(True)  # the component's calls are this loop's, its graphs its own
    try:
        with torch.inference_mode():
            return engine.generate(request, draft)
    finally:
        BYPASS.reset(token)


class Draft(torch.nn.Module if not ABSENT else object):  # type: ignore[misc]
    """The multi-token-prediction head of a Qwen3.5 or Qwen3-Next checkpoint, its weights where
    the checkpoint keeps them (`mtp.*`): the `draft` `generate` takes. From a token's embedding
    and the hidden state the model's head read before it: the state the head reads for the token
    after, which the next draft takes as its hidden state (as vLLM's and SGLang's do). `space` is
    the model's modeling module."""

    def __init__(self, config: Any, space: Any) -> None:
        super().__init__()
        text = config.get_text_config(decoder=True)
        prefix = next(n[: -len("DecoderLayer")] for n in vars(space) if n.endswith("DecoderLayer"))
        norm = getattr(space, f"{prefix}RMSNorm")
        layer = getattr(space, f"{prefix}DecoderLayer")(
            text, list(text.layer_types).index("full_attention")
        )
        layer.self_attn.layer_idx = 0  # the draft's own cache has the one layer
        size, eps = text.hidden_size, text.rms_norm_eps
        self.mtp = torch.nn.Module()
        self.mtp.fc = torch.nn.Linear(2 * size, size, bias=False)
        self.mtp.pre_fc_norm_embedding = norm(size, eps=eps)
        self.mtp.pre_fc_norm_hidden = norm(size, eps=eps)
        self.mtp.layers = torch.nn.ModuleList([layer])
        self.mtp.norm = norm(size, eps=eps)

    def forward(
        self, embeds: Any, hidden: Any, positions: Any, cache: Any, mask: Any, rotary: Any
    ) -> Any:
        """`rotary` is the model's: its tables are where the model is, at the model's precision
        (a copy of its own would be wherever a package left it, as the head's dtype)."""
        mtp = self.mtp
        joined = torch.cat([mtp.pre_fc_norm_embedding(embeds), mtp.pre_fc_norm_hidden(hidden)], -1)
        x = mtp.fc(joined)
        # Qwen3.5 carries the text position first and the rotary (mrope) ones after it.
        text, rope = (positions[0], positions[1:]) if positions.shape[0] == 4 else (None, positions)
        x = mtp.layers[0](
            x,
            position_embeddings=rotary(x, rope),
            attention_mask=mask,
            position_ids=text,
            past_key_values=cache,
        )
        return mtp.norm(x)


def install(plan: Plan, model: Any) -> str:
    """Runtime's one-launch kernels in the Transformers kernel slots `model`'s layers call
    (`causal_conv1d_update`, `torch_recurrent_gated_delta_rule`): a gated-delta-net layer's
    token then costs two launches where the reference path runs a few dozen ops. Call it from
    `Model.execution_plan`; it never waits for the compile (the kernels serve from the first
    request after it lands). A layer of another geometry, or a CPU, keeps Transformers' path.
    Returns the plan's identity, "" when nothing applies."""
    if ABSENT or _kernels.TRITON_ABSENT or torch.device(plan.device).type != "cuda":
        return ""
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

    def activate(launch: Launch) -> None:
        for space, name in slots:
            current = getattr(space, name)
            if not getattr(current, "_cozy_slot", False):
                setattr(space, name, _SLOTS[name](current, launch))
        _STEPS[0] = launch
        for engine in list(_ENGINES.values()):  # their graphs recorded Transformers' path
            engine.stale = True

    try:
        if plan.when_ready is None:
            activate(plan.load(KERNELS))
        elif not plan.when_ready(KERNELS, activate):
            say(
                KERNELS.name,
                f"[lm] {KERNELS.name} compiling: its kernels serve from the next request",
            )
    except KernelsUnavailable as exc:
        if exc.code == "kernels_corrupt":
            raise
        say(KERNELS.name, f"[lm] Transformers' gated-delta-net path: {exc.code}: {exc}")
        return ""
    return f"{KERNELS.name}:{KERNELS.identity}"


def _conv_slot(original: Any, launch: Launch) -> Any:
    def slot(
        hidden_states: Any,
        conv_state: Any,
        weight: Any,
        bias: Any = None,
        activation: Any = None,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
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

    slot._cozy_slot = True  # type: ignore[attr-defined]
    return slot


def _recurrent_slot(original: Any, launch: Launch) -> Any:
    def slot(
        query: Any,
        key: Any,
        value: Any,
        g: Any,
        beta: Any,
        initial_state: Any = None,
        output_final_state: bool = False,
        use_qk_l2norm_in_kernel: bool = False,
        **kwargs: Any,
    ) -> Any:
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

    slot._cozy_slot = True  # type: ignore[attr-defined]
    return slot


_SLOTS = {"causal_conv1d_update": _conv_slot, "torch_recurrent_gated_delta_rule": _recurrent_slot}
#: Launches the slots made in this process (Python runs once per capture, not per replay).
_LAUNCHED = [0]
#: The kernels' launcher once they serve: a verify pass's tokens then take one launch per layer.
_STEPS: list[Launch | None] = [None]


def _unsupported(model: Any) -> str:
    config = model.config
    if getattr(config, "is_encoder_decoder", False):
        return "an encoder-decoder"
    text = config.get_text_config(decoder=True)
    attention = getattr(text, "_attn_implementation", None) or getattr(
        config, "_attn_implementation", ""
    )
    if attention not in ("sdpa", "eager"):
        return f"attention {attention!r} (SDPA and eager are served)"
    types, _ = get_layer_types_and_kwargs(text)
    other = sorted(set(types) - {"full_attention", "linear_attention"})
    if other:
        return f"layer types {other}"
    extras = [
        name
        for name, idle in _PROCESSORS.items()
        if getattr(model.generation_config, name, None) not in idle
    ]
    return f"its generation config adds {extras}" if extras else ""


class _Presence:
    """`presence_penalty` as a Transformers logits processor, for `_transformers`."""

    def __init__(self, prompt: int, penalty: float) -> None:
        self.prompt, self.penalty = prompt, penalty

    def __call__(self, input_ids: Any, scores: Any) -> Any:
        generated = input_ids[:, self.prompt :]
        if not generated.shape[1]:
            return scores
        seen = torch.zeros_like(scores).scatter_(1, generated, 1.0)
        return scores - self.penalty * seen


def _transformers(
    model: Any,
    inputs: Mapping[str, Any],
    max_new_tokens: int,
    stops: list[int],
    pad: int,
    penalty: float,
) -> Any:
    prompt = int(inputs["input_ids"].shape[1])
    processors = [_Presence(prompt, penalty)] if penalty else []
    context = CURRENT.get()
    with _PLANE, torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=stops,
            pad_token_id=pad,
            logits_processor=LogitsProcessorList(processors),
            stopping_criteria=StoppingCriteriaList([_Cancel(context)] if context else []),
        )
    return out[:, prompt:]


class _Cancel(StoppingCriteria):  # type: ignore[misc]
    """Transformers' loop raises the attempt's cancellation between tokens."""

    def __init__(self, context: Context) -> None:
        self.context = context

    def __call__(self, input_ids: Any, scores: Any, **kwargs: Any) -> Any:
        self.context.raise_if_cancelled()
        return torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)


def window(length: int, capacity: int) -> int:
    """The attention window for a sequence of `length`: the next power of two from `MIN_WINDOW`,
    at most the cache's `capacity`."""
    size = MIN_WINDOW
    while size < length:
        size *= 2
    return min(size, capacity)


class _Keys(StaticLayer):  # type: ignore[misc]
    """A full-attention layer's keys and values in fixed buffers of the cache's capacity, of which
    attention reads the first `window` positions (the rest are masked)."""

    cumulative_length: Any

    def __init__(self, size: list[int], capacity: int, rows: bool = False) -> None:
        super().__init__(max_cache_len=capacity)
        #: the window, shared with the cache (a holder, not the cache: no reference cycle keeps
        #: the buffers past the call)
        self.size = size
        #: each row writes at its own length (`cumulative_length` has a row per slot)
        self.rows = rows

    def lazy_initialization(self, key_states: Any, value_states: Any) -> None:
        # Transformers' own, less `mark_static_address`: nothing here goes through dynamo.
        self.dtype, self.device = key_states.dtype, key_states.device
        self.batch_size, self.num_heads = key_states.shape[:2]
        self.k_head_dim, self.v_head_dim = key_states.shape[-1], value_states.shape[-1]
        shape = (self.batch_size, self.num_heads, self.max_cache_len)
        self.keys = torch.zeros((*shape, self.k_head_dim), dtype=self.dtype, device=self.device)
        self.values = torch.zeros((*shape, self.v_head_dim), dtype=self.dtype, device=self.device)
        self.cumulative_length = self.cumulative_length.to(self.device)
        self.is_initialized = True

    def update(self, key_states: Any, value_states: Any, *args: Any, **kwargs: Any) -> Any:
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

    def get_mask_sizes(self, query_length: int, *args: Any, **kwargs: Any) -> tuple[int, int]:
        return self.size[0], 0


class _State(LinearAttentionLayer):  # type: ignore[misc]
    """A linear-attention layer's conv and recurrent state, updated in place."""

    device: Any

    def lazy_initialization(
        self,
        conv_states: Any = None,
        recurrent_states: Any = None,
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
        conv_states: Any,
        state_idx: int = 0,
        conv_kernel_size: int | None = None,
        **kwargs: Any,
    ) -> Any:
        verify = _VERIFY.get()
        if verify is None or not self.has_previous_state[state_idx]:
            return super().update_conv_state(
                conv_states, state_idx, conv_kernel_size=conv_kernel_size, **kwargs
            )
        state = self.conv_states[state_idx]
        full = torch.cat([state, conv_states], dim=-1)
        verify.conv.append((state, full))  # its state after each token is a slice of `full`
        state.copy_(full[..., -state.shape[-1] :])
        return full


@contextmanager
def _taking(model: Any) -> Iterator[list[Any]]:
    """The decoder's final states in the body, one per call: what the output head reads, at
    every position of the call."""
    taken: list[Any] = []
    handle = model.get_decoder().register_forward_hook(
        lambda _module, _args, out: taken.append(out[0])
    )
    try:
        yield taken
    finally:
        handle.remove()


class _RowByRow(TorchFunctionMode):  # type: ignore[misc]
    """A verify pass's plain linears one row at a time, as decoding runs them: cuBLAS takes
    another kernel for a few rows, whose sums differ in the last bit (Qwen3.5's `in_proj_a` and
    `in_proj_b`), and a drafted token must keep decode's bits. It is faster for those, too."""

    def __torch_function__(
        self, func: Any, types: Any, args: tuple[Any, ...] = (), kwargs: Any = None
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
        self.conv: list[tuple[Any, Any]] = []
        self.recurrent: list[tuple[Any, Any]] = []


#: Set while the model verifies drafted tokens: its linear-attention layers then take each token
#: with the one-token kernels' arithmetic (one launch for all once they serve) and record what
#: `_Verify` keeps.
_VERIFY: contextvars.ContextVar[_Verify | None] = contextvars.ContextVar(
    "cozy_lm_verify", default=None
)
#: The Transformers kernel slots a verify pass replaces, and the one-token slots it steps through.
_MANY = ("causal_conv1d_fn", "torch_chunk_gated_delta_rule")
_ONE = ("causal_conv1d_update", "torch_recurrent_gated_delta_rule")


def _speculable(model: Any) -> bool:
    """Whether `model`'s linear-attention layers can verify drafted tokens: their modules carry
    every slot it steps through; it wraps the multi-token ones once."""
    spaces = {sys.modules.get(type(m).__module__) for m in model.modules()}
    spaces = {space for space in spaces if space is not None and hasattr(space, _MANY[1])}
    if not all(callable(getattr(space, name, None)) for space in spaces for name in _MANY + _ONE):
        return False
    for space in spaces:
        for name, wrap in zip(_MANY, (_stepped_conv, _stepped_recurrence), strict=True):
            current = getattr(space, name)
            if not getattr(current, "_cozy_verify", False):
                setattr(space, name, wrap(space, current))
    return True


def _stepped_conv(space: Any, original: Any) -> Any:
    def conv(x: Any, weight: Any, bias: Any = None, *args: Any, **kwargs: Any) -> Any:
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

    conv._cozy_verify = True  # type: ignore[attr-defined]
    return conv


def _stepped_recurrence(space: Any, original: Any) -> Any:
    def recurrence(
        query: Any,
        key: Any,
        value: Any,
        g: Any,
        beta: Any,
        initial_state: Any = None,
        output_final_state: bool = False,
        use_qk_l2norm_in_kernel: bool = False,
        **kwargs: Any,
    ) -> Any:
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
        state, outs = initial_state.clone(), []
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

    recurrence._cozy_verify = True  # type: ignore[attr-defined]
    return recurrence


class _StaticCache(Cache):  # type: ignore[misc]
    def __init__(self, text_config: Any, capacity: int, size: int, rows: bool = False) -> None:
        self.size = [size]
        types, kwargs = get_layer_types_and_kwargs(text_config)
        if isinstance(kwargs, dict):  # Transformers before 5.19: one set for every layer
            kwargs = [kwargs] * len(types)
        layers = [
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


class _Prefill:
    """A call's prompt through the model: its own cache, mask and positions, each row's first
    token and, drafting, the head's keys for the prompt and its state after it."""

    def __init__(self, cache: Any, mask: Any, position: Any, token: Any, drafted: Any) -> None:
        self.cache, self.mask, self.position, self.token, self.drafted = (
            cache,
            mask,
            position,
            token,
            drafted,
        )


def _slot_bytes(fresh: _Prefill, capacity: int, drafted: bool, text: Any) -> int:
    """The device bytes a slot of `capacity` positions takes."""
    total = 0
    rows = int(fresh.token.shape[0])
    for layer in fresh.cache.layers:
        if isinstance(layer, _Keys):
            for buffer in (layer.keys, layer.values):
                total += buffer[0, :, :1].numel() * buffer.element_size() * capacity
        else:
            for state in (*layer.conv_states.values(), *layer.recurrent_states.values()):
                if state is not None:
                    total += state.numel() * state.element_size() // rows
    keys = fresh.drafted[0] if drafted else None
    if keys is not None and keys.keys is not None:
        total += 2 * keys.keys[0, :, :1].numel() * keys.keys.element_size() * capacity
    return total + capacity * 18 + 4 * int(getattr(text, "vocab_size", 0) or 0)


def _derived(models: Iterable[Any]) -> list[Any]:
    """The non-persistent buffers of `models` on a device: built from config (a rotary table),
    never a checkpoint's, so the plane neither places nor pins them."""
    return [
        tensor
        for model in models
        for module in model.modules()
        for name, tensor in module._buffers.items()
        if name in module._non_persistent_buffers_set
        and type(tensor).__name__ == "Tensor"
        and tensor.device.type != "meta"
        and tensor.device.type != "cpu"
    ]


def _reservation(model: Any, shed: Callable[[int], int]) -> Any:
    """The memory's count of an engine's buffers, shed on ask (`Residency.cache`); None for a
    model Runtime does not place (its tensors are where its caller put them)."""
    placed = PLACED.get(model)
    return None if placed is None else placed[0].cache(f"lm:{type(model).__name__}", shed)


class _Request:
    """One `generate` call: its rows from their prefill to the last one's retirement."""

    def __init__(
        self,
        inputs: Mapping[str, Any],
        max_new: int,
        stops: list[int],
        pad: int,
        penalty: float,
    ) -> None:
        self.inputs, self.max_new, self.stops, self.pad, self.penalty = (
            inputs,
            max_new,
            stops,
            pad,
            penalty,
        )
        rows = int(inputs["input_ids"].shape[0])
        self.prompt = int(inputs["input_ids"].shape[1])
        self.tokens: list[Any] = [None] * rows
        self.left = rows
        self.error: BaseException | None = None
        #: why it decodes through Transformers' loop instead (its caller runs it)
        self.fallback = ""
        #: the head it drafts with (None: any, drafting keeps a model's tokens)
        self.draft: Any = None
        self.prefill_s = self.began = self.ended = 0.0
        #: the engine's (captures, capture seconds, eager steps) at admission, then over its life
        self.at: tuple[float, ...] = (0, 0.0, 0)
        self.steps = self.live = 0
        self.warnings: list[tuple[str, dict[str, Any]]] = []
        #: the attempt it serves (cancellation), and whether its rows left for it
        self.context: Context | None = CURRENT.get()
        self.cut = False

    @property
    def finished(self) -> bool:
        return self.left == 0 or self.error is not None or bool(self.fallback) or self.cut

    @property
    def cancelled(self) -> bool:
        return self.context is not None and self.context.cancelled

    def result(self) -> Any:
        width = max(int(t.shape[0]) for t in self.tokens)
        out = torch.full((len(self.tokens), width), self.pad, dtype=torch.long)
        for row, tokens in enumerate(self.tokens):
            out[row, : int(tokens.shape[0])] = tokens.cpu()
        return out.to(self.inputs["input_ids"].device)


class _Row:
    """A slot's sequence: its call, its row of that call, and where the host knows it to be."""

    def __init__(self, request: _Request, row: int, length: int) -> None:
        self.request, self.row, self.length = request, row, length
        #: the most tokens it can hold after the steps queued so far (the window's bound)
        self.reach = length + 1


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
    seen: torch.Tensor
    position: torch.Tensor
    dmask: torch.Tensor
    hidden: torch.Tensor

    def __init__(self, model: Any) -> None:
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
        #: captures recorded in this engine's life and their seconds, steps run eagerly
        self.captured = [0, 0.0]
        self.eager = 0
        #: `LANDED` when a capture met a kernel still building (none records until one lands),
        #: and when the graphs were last started over
        self.building, self.landed = -1, LANDED[0]
        self.driver: int | None = None
        self.draft: Any = None
        self.slots: list[_Row | None] = []
        self.capacity = 0
        self.width = 0
        self.cache: Any = None
        self.views: dict[int, Any] = {}
        self.graphs: dict[tuple[int, int, bool], Any] = {}
        #: (rows, drafted) that ran one eager step: the next is recorded
        self.warmed: set[tuple[int, bool]] = set()
        self.layout: tuple[object, ...] | None = None
        self.pool: Any = None
        self.side: Any = None
        #: (bucket, window, drafted) -> why that step runs eagerly
        self.unrecorded: dict[tuple[int, int, bool], str] = {}
        #: the graphs recorded a path that changed since (a kernel landed in `install`)
        self.stale = False
        #: seconds the last admission took: decoding gets as long before the next one
        self.debt = 0.0
        self.reservation: Any = None
        self.retire = False
        self.cache_device: Any = None
        self.open: bool | float
        self.closed: bool | float
        if self.text._attn_implementation == "sdpa":
            self.open, self.closed = True, False
        else:  # eager attention adds its mask: 0 where it attends, the dtype's minimum elsewhere
            self.open, self.closed = 0.0, torch.finfo(model.dtype).min

    # ------------------------------------------------------------------ calls

    def generate(self, request: _Request, draft: Any) -> Any:
        request.draft = draft if draft is not None and self._speculable() else None
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
            )
        self._report(request)
        return request.result()

    def _speculable(self) -> bool:
        return _speculable(self.model)

    def _report(self, request: _Request) -> None:
        tokens = sum(int(t.shape[0]) for t in request.tokens)
        rows = len(request.tokens)
        looped = (tokens - rows) / rows  # a row's tokens after its first, the prefill's
        captures, capture_s, eager = request.at
        loop_s = max(request.ended - request.began - capture_s, 0.0)
        per = 1000 * loop_s / max(looped, 1)
        mode = "eager" if self.cache_device.type != "cuda" else "graphs"
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
        try:
            with _PLANE, ExitStack() as stack:
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
        except torch.OutOfMemoryError as exc:
            self._fail(None, f"OutOfMemoryError: {exc}".splitlines()[0][:300])
        except BaseException as exc:  # a failed step fails every row in it
            self._fail(exc, "")
            raise
        finally:
            with self.cond:
                self.driver = None
                self.cond.notify_all()

    def _fail(self, exc: BaseException | None, why: str) -> None:
        """Every held and waiting call ends: with `exc`, or through Transformers' loop (`why`)."""
        with self.cond:
            requests = {id(row.request): row.request for row in self.slots if row is not None}
            requests.update({id(request): request for request in self.pending})
            requests.update({id(request): request for request in self.incoming})
            self.pending.clear()
            self.incoming = []
            for request in requests.values():
                if exc is not None:
                    request.error = exc
                else:
                    request.fallback = why
            self._drop()
            self.cond.notify_all()

    def _targets(self) -> list[tuple[Any, Any]]:
        return [(m, PLACED[m]) for m in (self.model, self.draft) if m is not None and m in PLACED]

    def _hold(self, stack: ExitStack) -> bool:
        """The model (and its head) pinned at home for the decode graphs (`Residency.pin`);
        False when part of one streams (the calls then decode through Transformers' loop)."""
        targets = self._targets()
        # The steps also read buffers the model builds (its rotary tables): a rebuilt one (a
        # component placed again) records anew.
        derived = _derived(m for m in (self.model, self.draft) if m is not None)
        built = tuple(t.data_ptr() for t in derived)
        if not targets:
            self.where = (built,)
            return True
        pins = [r.pin(name, f"lm:{name}", self._unpinned) for _, (r, name) in targets]
        if any(pin is None for pin in pins):
            return False
        self.where = (*pins, built)
        return True

    #: the pins the graphs read weights through: they live while this engine refers to them
    where: tuple[object, ...] = ()

    def _unpinned(self) -> int:
        """Memory asks for the pinned weights: an idle engine lets its graphs and pins go now,
        a busy one at its next loop."""
        if self.idle():
            self.where = ()
        else:
            self.rehold = self.stale = True
        return 0

    def _cancelled(self) -> bool:
        """Whether a held row's call was cancelled: the host stops queueing steps for it."""
        return any(row.request.cancelled for row in self.slots if row is not None)

    def _occupied(self) -> int:
        return sum(row is not None for row in self.slots)

    def _bucket(self) -> int:
        return next(b for b in BUCKETS if b >= self._occupied())

    def _one(self, size: int, run: Callable[[int], None]) -> None:
        """One step at `size` rows, and what the host knows of the rows after it."""
        rows = [row for row in self.slots if row is not None]
        per = DEPTH + 1 if self._drafting(size) else 1
        self.cache.window = window(max(row.reach for row in rows) + per, self.capacity)
        run(size)
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
                if self.cache_device.type == "cuda":
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
            except torch.OutOfMemoryError as exc:  # this call alone takes Transformers' loop
                request.fallback = f"OutOfMemoryError: {exc}".splitlines()[0][:300]
                torch.cuda.empty_cache()
                continue
            if joined:
                admitted = True
            else:
                waiting.append(request)
        self.debt = time.perf_counter() - started if admitted else 0.0
        with self.cond:
            self.pending.extendleft(reversed(waiting))
            if waiting and not self._occupied() and not admitted and not self.rehold:
                for request in waiting:  # nothing will retire to make room
                    request.fallback = "its decode set does not fit beside the weights"
                self.pending.clear()
            self.cond.notify_all()

    def _retire(self) -> None:
        n = self.n.tolist()
        done = self.done.tolist()
        for slot, row in enumerate(self.slots):
            if row is None or not (done[slot] or row.request.cancelled):
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

    def _rows(self) -> Iterator[Any]:
        """Every buffer with a row per slot."""
        for layer in self.cache.layers:
            if isinstance(layer, _Keys):
                yield from (layer.keys, layer.values, layer.cumulative_length)
            else:
                yield from (s for s in layer.conv_states.values() if s is not None)
                yield from (s for s in layer.recurrent_states.values() if s is not None)
        yield from (self.ids, self.length, self.mask, self.out, self.n, self.done, self.limit)
        yield from (self.stops, self.pad, self.penalty, self.seen, self.position)
        if self.draft is not None:
            yield from (self.dkeys.keys, self.dkeys.values, self.dmask, self.hidden)

    def _empty(self, slot: int) -> None:
        """A free slot as a finished row of length 0 that attends to its first position."""
        self.done[slot] = True
        self.length[slot] = 0
        self.n[slot] = 0
        self.mask[slot].fill_(self.closed)
        self.mask[slot, ..., 0] = self.open
        if self.draft is not None:
            self.dmask[slot].fill_(self.closed)
            self.dmask[slot, ..., 0] = self.open

    # ------------------------------------------------------------------ admission

    def _admit(self, request: _Request) -> bool:
        """`request`'s prompt through the model's own forward into a cache of its own, unpadded,
        then its rows into free slots, with their first token. False when there is no room."""
        rows = int(request.inputs["input_ids"].shape[0])
        reach = request.prompt + request.max_new + DEPTH * (LAG + 2)
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
        if self.cache is not None and len(free) >= rows and reach <= self.capacity:
            fresh = self._prefill(request)
            self._splice(request, fresh, free)
            return True
        occupied = len(self.slots) - len(free)
        size = next((b for b in BUCKETS if b >= occupied + rows), 0)
        if not size:
            return False
        capacity = max(self.capacity, -(-reach // MIN_WINDOW) * MIN_WINDOW)
        fresh = self._prefill(request)
        if not self._grow(size, capacity, fresh):
            return False
        self._splice(request, fresh, [slot for slot, row in enumerate(self.slots) if row is None])
        return True

    def _prefill(self, request: _Request) -> Any:
        started = time.perf_counter()
        model, inputs = self.model, request.inputs
        ids = inputs["input_ids"]
        kwargs = {key: value for key, value in inputs.items() if key != "input_ids"}
        mask = kwargs.get("attention_mask")
        mask = torch.ones_like(ids) if mask is None else mask
        kwargs["attention_mask"] = mask
        position = None
        if self.positional:
            position = kwargs["position_ids"] = model._prepare_position_ids_for_generation(
                ids, kwargs
            )
        if bool(mask.all()):  # as `generate`: no padding, no 2D mask
            kwargs["attention_mask"] = None
        if self.keep:
            kwargs["logits_to_keep"] = 1
        fresh = _StaticCache(self.text, request.prompt, request.prompt)
        prepared = model.prepare_inputs_for_generation(
            ids, past_key_values=fresh, use_cache=True, is_first_iteration=True, **kwargs
        )
        # Under the caller's recovery, which makes room op by op, into this call's own cache.
        with _taking(model) as taken:
            logits = model(**prepared, return_dict=True).logits[:, -1]
        hidden = taken[-1]
        with _disable_current_modes():
            token = logits.float().argmax(-1)
            drafted = None
            if self.draft is not None:
                drafted = self._drafted_prompt(ids, token, hidden, position)
        request.prefill_s += time.perf_counter() - started
        return _Prefill(fresh, mask, position, token, drafted)

    def _drafted_prompt(self, ids: Any, token: Any, hidden: Any, position: Any) -> Any:
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
        cache = Cache(layers=[_Keys([prompt + 1], prompt + 1)])
        cache.layers[0].cumulative_length = torch.ones((), dtype=torch.long, device=ids.device)
        embed = self.model.get_input_embeddings()
        rotary = self.model.get_decoder().rotary_emb
        last = self.draft(embed(tokens), hidden, at, cache, mask, rotary)[:, -1:]
        return cache.layers[0], last

    def _splice(self, request: _Request, fresh: _Prefill, free: list[int]) -> None:
        """`fresh`'s rows into slots `free`, each with its prompt's keys, state, mask and
        positions and its first token."""
        prompt = request.prompt
        stops = torch.full((self.stops.shape[1],), -1, dtype=torch.long)
        stops[: len(request.stops)] = torch.tensor(request.stops, dtype=torch.long)
        with _disable_current_modes():
            for row in range(int(fresh.token.shape[0])):
                slot = free[row]
                for mine, theirs in zip(self.cache.layers, fresh.cache.layers, strict=True):
                    if isinstance(mine, _Keys):
                        mine.keys[slot, :, :prompt].copy_(theirs.keys[row])
                        mine.values[slot, :, :prompt].copy_(theirs.values[row])
                        mine.cumulative_length[slot] = prompt
                    else:
                        for i, state in mine.conv_states.items():
                            if state is not None:
                                state[slot].copy_(theirs.conv_states[i][row])
                        for i, state in mine.recurrent_states.items():
                            if state is not None:
                                state[slot].copy_(theirs.recurrent_states[i][row])
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
                self.seen[slot].zero_()
                self.seen[slot, token] = 1.0
                self.done[slot] = bool((self.stops[slot] == token).any()) or request.max_new <= 1
                if self.draft is not None:
                    keys, last = fresh.drafted
                    if keys.keys is not None:
                        self.dkeys.keys[slot, :, : prompt + 1].copy_(keys.keys[row])
                        self.dkeys.values[slot, :, : prompt + 1].copy_(keys.values[row])
                    self.dmask[slot].fill_(self.closed)
                    self.dmask[slot, ..., 1 : prompt + 1] = self.open
                    self.hidden[slot].copy_(last[row])
                self.slots[slot] = _Row(request, row, prompt)
        request.began = time.perf_counter()
        request.at = (self.captured[0], self.captured[1], self.eager)
        self.retire = self.retire or bool(self.done[free[: int(fresh.token.shape[0])]].any())

    def _grow(self, size: int, capacity: int, fresh: _Prefill) -> bool:
        """Buffers for `size` slots of `capacity` positions, counted as the memory's cache; the
        rows held now move over, and the graphs record again. False when they do not fit."""
        if self.reservation is None:
            self.reservation = _reservation(self.model, self._shed)
        try:
            self._buffers(size, capacity, fresh)
        except Exception as exc:
            if not is_capacity(exc):
                raise
            return False
        if self.reservation is not None:
            self.reservation.nbytes = (
                _slot_bytes(fresh, capacity, self.draft is not None, self.text) * size
            )
        return True

    def _buffers(self, size: int, capacity: int, fresh: _Prefill) -> None:
        device = fresh.token.device
        old, held = self.cache, [row for row in self.slots if row is not None]
        cache = _StaticCache(self.text, capacity, MIN_WINDOW, rows=True)
        for mine, theirs in zip(cache.layers, fresh.cache.layers, strict=True):
            if isinstance(mine, _Keys):
                k, v = theirs.keys, theirs.values
                mine.dtype, mine.device = k.dtype, k.device
                mine.keys = k.new_zeros((size, k.shape[1], capacity, k.shape[3]))
                mine.values = v.new_zeros((size, v.shape[1], capacity, v.shape[3]))
                mine.cumulative_length = torch.zeros(size, dtype=torch.long, device=device)
                mine.is_initialized = True
            else:
                mine.dtype, mine.device = theirs.dtype, theirs.device
                for i, state in theirs.conv_states.items():
                    if state is not None:
                        mine.conv_states[i] = state.new_zeros((size, *state.shape[1:]))
                        mine.conv_kernel_size[i] = theirs.conv_kernel_size[i]
                        mine.is_conv_states_initialized[i] = True
                        mine.has_previous_state[i] = True
                for i, state in theirs.recurrent_states.items():
                    if state is not None:
                        mine.recurrent_states[i] = state.new_zeros((size, *state.shape[1:]))
                        mine.is_recurrent_states_initialized[i] = True
        kind = torch.bool if self.closed is False else self.model.dtype
        mask = torch.full((size, 1, 1, capacity), self.closed, dtype=kind, device=device)
        stops = max(8, max((len(r.request.stops) for r in held), default=0))
        vocab = int(self.model.get_output_embeddings().weight.shape[0])
        long = {"dtype": torch.long, "device": device}
        fields = {
            "ids": torch.zeros((size, 1), **long),
            "length": torch.zeros((size,), **long),
            "mask": mask,
            "out": torch.zeros((size, capacity), **long),
            "n": torch.zeros((size,), **long),
            "done": torch.ones((size,), dtype=torch.bool, device=device),
            "limit": torch.zeros((size,), **long),
            "stops": torch.full((size, stops), -1, **long),
            "pad": torch.zeros((size,), **long),
            "penalty": torch.zeros((size,), dtype=torch.float32, device=device),
            "seen": torch.zeros((size, vocab), dtype=torch.float32, device=device),
            "position": torch.zeros(
                (size, *(fresh.position.shape[:-2] if fresh.position is not None else ()), 1),
                **long,
            ),
        }
        if self.draft is not None:
            keys, last = fresh.drafted
            # A head of its own keys (`Draft`) writes them; another reads none (shape 1, 1).
            like = keys.keys if keys.keys is not None else last.new_zeros((1, 1, 1, 1))
            draft = _Keys([MIN_WINDOW], capacity, rows=True)
            draft.dtype, draft.device = like.dtype, device
            draft.keys = like.new_zeros((size, like.shape[1], capacity, like.shape[3]))
            draft.values = like.new_zeros((size, like.shape[1], capacity, like.shape[3]))
            draft.cumulative_length = torch.zeros(size, dtype=torch.long, device=device)
            draft.is_initialized = True
            fields["dmask"] = torch.full_like(mask, self.closed)
            fields["hidden"] = last.new_zeros((size, *last.shape[1:]))
        moving = list(self._rows()) if old is not None else []
        self.cache, self.capacity, self.cache_device = cache, capacity, device
        for name, tensor in fields.items():
            setattr(self, name, tensor)
        if self.draft is not None:
            self.dkeys = draft
            self.dcache = Cache(layers=[draft])
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
        self.rows_index = torch.arange(capacity, device=device)
        self.flag = torch.zeros((size,), dtype=torch.bool, pin_memory=device.type == "cuda")
        self.views = {}
        self._forget()
        self.width = stops

    # ------------------------------------------------------------------ steps

    def _view(self, size: int) -> Any:
        """The cache over the first `size` slots: views of the buffers, which graphs read."""
        view = self.views.get(size)
        if view is None:
            view = _StaticCache(self.text, self.capacity, self.cache.window, rows=True)
            view.size = self.cache.size
            for mine, full in zip(view.layers, self.cache.layers, strict=True):
                if isinstance(mine, _Keys):
                    mine.size = self.cache.size
                    mine.dtype, mine.device = full.dtype, full.device
                    mine.keys, mine.values = full.keys[:size], full.values[:size]
                    mine.cumulative_length = full.cumulative_length[:size]
                    mine.is_initialized = True
                else:
                    mine.dtype, mine.device = full.dtype, full.device
                    for i, state in full.conv_states.items():
                        if state is not None:
                            mine.conv_states[i] = state[:size]
                            mine.conv_kernel_size[i] = full.conv_kernel_size[i]
                            mine.is_conv_states_initialized[i] = True
                            mine.has_previous_state[i] = True
                    for i, state in full.recurrent_states.items():
                        if state is not None:
                            mine.recurrent_states[i] = state[:size]
                            mine.is_recurrent_states_initialized[i] = True
            if self.draft is not None:
                keys = _Keys(self.cache.size, self.capacity, rows=True)
                keys.dtype, keys.device = self.dkeys.dtype, self.dkeys.device
                keys.keys, keys.values = self.dkeys.keys[:size], self.dkeys.values[:size]
                keys.cumulative_length = self.dkeys.cumulative_length[:size]
                keys.is_initialized = True
                view.draft = Cache(layers=[keys])
            self.views[size] = view
        return view

    def _steps(self) -> None:
        """Steps until a row is done or a call waits to join, queued `LAG` ahead of the host."""
        cuda = self.cache_device.type == "cuda"
        if cuda and (self.where != self.layout or self.landed != LANDED[0] or self.stale):
            # The weights moved, or a kernel landed (the graphs recorded its stand-in).
            self._forget()
            self.layout, self.landed, self.stale = self.where, LANDED[0], False
        queued: deque[Any] = deque()
        size = self._bucket()
        began = time.perf_counter()

        def joining() -> bool:  # a waiting call joins once decoding had its prefill's time
            return bool(self.pending) and time.perf_counter() - began >= self.debt

        with _disable_current_modes():
            while True:
                self._one(size, lambda size: self._replay(size, cuda))
                if not cuda:
                    if bool(self.flag[:size].any()) or joining() or self._cancelled():
                        break
                    continue
                event = torch.cuda.Event()
                event.record()
                queued.append(event)
                if len(queued) > LAG:
                    queued.popleft().synchronize()
                    if bool(self.flag[:size].any()) or joining() or self._cancelled():
                        break
            if cuda:
                torch.cuda.current_stream().synchronize()
        self.retire = True

    def _replay(self, size: int, cuda: bool) -> None:
        drafted = self._drafting(size)
        key = (size, self.cache.window, drafted)
        graph = self.graphs.get(key)
        if graph is None and cuda and key not in self.unrecorded and self.building != LANDED[0]:
            if (size, drafted) not in self.warmed:
                self.warmed.add((size, drafted))
                self._advance(size)
                return
            if self.pool is None:
                self.pool, self.side = torch.cuda.graph_pool_handle(), torch.cuda.Stream()
            capturing = time.perf_counter()
            try:
                graph, why = capture(torch, lambda: self._advance(size), self.pool, self.side)
            except Building:  # eager until a kernel lands
                graph, why, self.building = None, "", LANDED[0]
            self.captured[0] += 1
            self.captured[1] += time.perf_counter() - capturing
            if graph is not None:
                self.graphs[key] = graph
            else:
                self.pool = None  # an aborted capture spent its pool
                if why:
                    self._unrecorded(key, why)
        if graph is not None:
            graph.replay()
        else:
            self._advance(size)
            self.eager += 1

    def _forget(self) -> None:
        """Every recorded step goes, and its pool: a fresh pool records the next ones (a pool
        whose graphs are gone takes no new recording)."""
        if self.graphs or self.pool is not None:
            self.graphs, self.warmed, self.unrecorded, self.pool = {}, set(), {}, None
            torch.cuda.empty_cache()

    def _unrecorded(self, key: tuple[int, int, bool], why: str) -> None:
        """That step runs eagerly from now on, with a warning on the calls it serves."""
        self.unrecorded[key] = why
        say(f"capture:{type(self.model).__name__}", f"[lm] capture failed: {why}")
        warning = ("a CUDA graph was not used", {"module": type(self.model).__name__, "why": why})
        for request in {id(r.request): r.request for r in self.slots if r is not None}.values():
            if warning not in request.warnings:
                request.warnings.append(warning)
        for row in self.slots:
            if row is not None:
                row.request.warnings.append(
                    ("a CUDA graph was not used", {"module": type(self.model).__name__, "why": why})
                )

    def _drafting(self, size: int) -> bool:
        return self.draft is not None and size <= DRAFT_ROWS

    def _advance(self, size: int) -> None:
        if self._drafting(size):
            self._speculate(size)
        else:
            self._step(size)

    def _select(self, logits: Any, size: int) -> Any:
        """Each live row's next token, on the device, in `ids` and `out`; a row that stops (a
        stop token, or its limit) is done. Returns the rows live before it."""
        scores = logits.float() - self.penalty[:size, None] * self.seen[:size]
        done = self.done[:size]
        live = ~done
        token = torch.where(done, self.pad[:size], scores.argmax(-1))
        n = self.n[:size]
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
        view, window_ = self._view(size), self.cache.window
        length = self.length[:size]
        self.mask[:size].scatter_(3, length.view(size, 1, 1, 1), self.open)
        kwargs: dict[str, Any] = {
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
        for layer in view.layers:
            if isinstance(layer, _Keys):
                layer.cumulative_length.copy_(length)
        self.position[:size].add_(live.long().view(size, *[1] * (self.position.dim() - 1)))

    def _speculate(self, size: int) -> None:
        """`DEPTH` drafted tokens a row, verified with the row's token before them in one pass:
        each row keeps its tokens up to and including its first disagreement with the model,
        and its states go back to the kept tokens' end."""
        model, depth, w = self.model, DEPTH, self.cache.window
        view = self._view(size)
        span = depth + 1
        embed, head = model.get_input_embeddings(), model.get_output_embeddings()
        rotary = model.get_decoder().rotary_emb
        keys, steps = self.rows_index[:w], self.rows_index[:span]
        length = self.length[:size]
        L = length[:, None]
        done = self.done[:size]
        live = ~done
        dkeys = view.draft.layers[0]
        dkeys.cumulative_length.copy_(length + 1)
        dmask = self.dmask[:size, ..., :w]
        position = self.position[:size]
        token, hidden, drafts = self.ids[:size], self.hidden[:size], []
        seen = self.seen[:size]
        for i in range(depth):
            if i:
                open_ = (keys[None] > L) & (keys[None] <= L + i)
                mask = torch.where(open_[:, None, None], self.open, dmask)
                at = (position + i).movedim(0, -2) if self.positional else (length + i)[:, None]
                hidden = self.draft(embed(token), hidden, at, view.draft, mask, rotary)
            scores = head(hidden)[:, -1].float() - self.penalty[:size, None] * seen
            token = scores.argmax(-1, keepdim=True)
            drafts.append(token)
            seen = seen.scatter(1, token, 1.0)
        drafted = torch.cat(drafts, dim=1)
        # Verify: each row attends to its committed keys and this pass's up to its own.
        open_ = (keys[None, None] >= L[..., None]) & (
            keys[None, None] <= (L + steps[None])[..., None]
        )
        mask = torch.where(open_[:, None], self.open, self.mask[:size, ..., :w])
        kwargs: dict[str, Any] = {
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
                scores = model(**kwargs).logits[:, -span:].float()
        finally:
            _VERIFY.reset(reset)
        # row j has seen the drafts before it
        ahead = torch.zeros_like(scores).scatter_(2, drafted[..., None], 1.0).cumsum(1)
        ahead = torch.cat([torch.zeros_like(ahead[:, :1]), ahead[:, :-1]], dim=1)
        scores = scores - self.penalty[:size, None, None] * torch.maximum(
            self.seen[:size, None], ahead.clamp(max=1)
        )
        best = scores.argmax(-1)
        kept = (best[:, :depth] == drafted).long().cumprod(-1).sum(-1)
        n = self.n[:size]
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
        states = self.draft(embed(tokens), taken[-1][:, -span:], at, view.draft, mask, rotary)
        self.hidden[:size].copy_(
            states.gather(1, last[:, None, None].expand(-1, 1, states.shape[-1]))
        )
        # Back to the kept tokens' end: keys past it are masked and overwritten later.
        ended = length + made
        for layer in view.layers:
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
        n.add_(made)
        self.position[:size].add_(made.view(size, *[1] * (self.position.dim() - 1)))
        done |= stopped.any(-1)
        self.flag[:size].copy_(done, non_blocking=True)

    # ------------------------------------------------------------------ memory

    def _drop(self) -> None:
        if self.cache_device is not None and self.cache_device.type == "cuda":
            self._forget()
        self.cache, self.views = None, {}
        self.slots, self.capacity = [], 0
        self.layout = None
        if self.reservation is not None:
            self.reservation.close()
            self.reservation = None

    def idle(self) -> bool:
        """Its buffers and graphs go back when no call holds it: what the plane takes before
        it unmaps weights. Whether it had any."""
        if not self.cond.acquire(blocking=False):
            return False
        try:
            if self.driver is not None or self.pending or self.cache is None:
                return False
            self._drop()
            return True
        finally:
            self.cond.release()

    def _shed(self, nbytes: int) -> int:
        """Room the memory manager asks for under pressure: an idle engine drops its buffers;
        a busy one gives nothing (its rows are mid-flight)."""
        before = self.reservation.nbytes if self.reservation is not None else 0
        return before if self.idle() else 0


__all__ = [
    "BUCKETS",
    "DEPTH",
    "KERNELS",
    "MIN_WINDOW",
    "Draft",
    "drafting",
    "generate",
    "install",
    "release",
    "window",
]
