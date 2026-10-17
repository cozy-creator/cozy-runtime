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
import time
import weakref
from collections import deque
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import ExitStack, contextmanager
from typing import Any

from cozy_runtime.author._model import LANDED, PLACED, RELEASERS, WARNINGS, Building
from cozy_runtime.author.kernels import KernelsUnavailable, Launch, Plan, TritonKernels
from cozy_runtime.graphs import BYPASS, ROOM, capture, say
from cozy_runtime.lm import _kernels

try:
    import torch
    from torch.overrides import TorchFunctionMode
    from torch.utils._python_dispatch import _disable_current_modes
    from transformers import LogitsProcessorList
    from transformers.cache_utils import (
        Cache,
        LinearAttentionLayer,
        StaticLayer,
        get_layer_types_and_kwargs,
    )
except ImportError as _absent:  # pragma: no cover - the base package has neither
    ABSENT = f"torch and transformers do not import ({_absent})"
    StaticLayer = LinearAttentionLayer = Cache = TorchFunctionMode = object
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


#: A placed model's decode blocks, as its first call's first step ran them: later calls hold
#: them before their prefill.
_READS: weakref.WeakKeyDictionary[Any, frozenset[Any]] = weakref.WeakKeyDictionary()
#: Each model's decode session (buffers and graphs), kept until its geometry changes or the
#: plane wants the memory (`RELEASERS`).
_SESSIONS: weakref.WeakKeyDictionary[Any, _Decode] = weakref.WeakKeyDictionary()


def release() -> bool:
    """Drop every decode session: what the plane calls before it unmaps weights for room."""
    had = len(_SESSIONS) > 0
    _SESSIONS.clear()
    return had


RELEASERS.append(release)
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
    token = BYPASS.set(True)  # the component's calls are this loop's, its graphs its own
    try:
        with torch.inference_mode():
            return _generate(model, inputs, max_new_tokens, stops, filler, presence_penalty, draft)
    finally:
        BYPASS.reset(token)


def _generate(
    model: Any,
    inputs: Mapping[str, Any],
    max_new_tokens: int,
    stops: list[int],
    filler: int,
    presence_penalty: float,
    draft: Any,
) -> Any:
    try:
        tokens = _static(model, inputs, max_new_tokens, stops, filler, presence_penalty, draft)
        if tokens is None and draft is not None:
            # The head's weights and keys cost room the model's own decode set may need: the
            # graphed decode without drafts beats Transformers' loop by far.
            _SESSIONS.pop(model, None)
            say(f"lm:{type(model).__name__}:undrafted", "[lm] drafting does not fit at home")
            WARNINGS.append(("decoded without drafting", {"why": "the draft head does not fit"}))
            tokens = _static(model, inputs, max_new_tokens, stops, filler, presence_penalty, None)
        if tokens is not None:
            return tokens
        why = "its decode set does not fit at home"
    except torch.OutOfMemoryError as exc:
        why = f"OutOfMemoryError: {exc}".splitlines()[0][:300]
    _SESSIONS.pop(model, None)  # its buffers go back before Transformers' loop needs the room
    torch.cuda.empty_cache()
    say(f"lm:{type(model).__name__}:{why[:30]}", f"[lm] {type(model).__name__}: {why}")
    WARNINGS.append(("decode ran through Transformers' loop", {"why": why}))
    # Transformers' own loop under Runtime's recovery, which makes room op by op.
    return _transformers(model, inputs, max_new_tokens, stops, filler, presence_penalty)


def _static(
    model: Any,
    inputs: Mapping[str, Any],
    max_new_tokens: int,
    stops: list[int],
    filler: int,
    presence_penalty: float,
    draft: Any,
) -> Any:
    """The tokens from the static cache, replayed where its blocks are held at home; None when
    they do not all fit there."""
    ids = inputs["input_ids"]
    prompt, cuda = int(ids.shape[1]), ids.is_cuda
    if draft is not None and (ids.shape[0] != 1 or not _speculable(model)):
        draft = None  # a batch's rows would keep prefixes of their own
    per = DEPTH + 1 if draft is not None else 1
    # Prompts within one window step share a session (its buffers and graphs); a drafted step
    # can make `per` tokens, and the host learns their count `LAG` steps late.
    reach = prompt + max_new_tokens + (per - 1) * (LAG + 2)
    capacity = -(-reach // MIN_WINDOW) * MIN_WINDOW
    key = (
        int(ids.shape[0]),
        capacity,
        max_new_tokens,
        len(stops),
        presence_penalty,
        ids.device,
        draft,
    )
    # Only the prefill's forward runs under Runtime's recovery, into a cache of its own. The
    # recovery would journal, spill or band what an op writes, and the session's buffers are
    # read by address by its graphs and kept for later calls: everything that writes them runs
    # outside it, and room that a step cannot get ends the static path, not the call.
    decode = _SESSIONS.get(model)
    if decode is None or decode.key != key:
        _SESSIONS.pop(model, None)
        decode = _SESSIONS[model] = _Decode(model, key, presence_penalty, draft)
    with _disable_current_modes():
        decode.begin(inputs, stops, filler)
    targets = [(m, PLACED[m]) for m in (model, draft) if m is not None and m in PLACED]
    if not targets:  # modules Runtime does not place: their tensors stay where they are
        decode.prefill()
        with _disable_current_modes():
            return decode.run(graphs=cuda, layout=(), room=None)
    if any(m not in _READS for m, _ in targets):
        # The first call: its first step runs through the paging hooks, which record the
        # blocks a step reads; later calls hold them before their prefill.
        decode.prefill()
        with ExitStack() as stack:
            reads = {}
            for _, (residency, _name) in targets:
                if id(residency) not in reads:
                    reads[id(residency)] = stack.enter_context(residency.recording())
            stack.enter_context(_disable_current_modes())
            decode.first()
        for m, (residency, name) in targets:
            _READS[m] = frozenset(reads[id(residency)].get(name, ()))
    with ExitStack() as stack:
        holds = [stack.enter_context(r.held(name, _READS[m])) for m, (r, name) in targets]
        if not all(holds):
            return None
        if not decode.made:
            decode.prefill()
        layout = tuple(r.layout(name, _READS[m]) for m, (r, name) in targets)
        with _disable_current_modes():
            return decode.run(graphs=True, layout=layout, room=targets[0][1][0].allocatable)


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
        _SESSIONS.clear()  # their graphs recorded Transformers' path: the next call records anew

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
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=stops,
            pad_token_id=pad,
            logits_processor=LogitsProcessorList(processors),
        )
    return out[:, prompt:]


def _into(mine: Any, value: Any) -> Any:
    """`value` (a cache layer's field) copied into `mine` in place where they match, so a graph
    that reads `mine` reads it; else `value` itself."""
    if isinstance(value, torch.Tensor):
        same = isinstance(mine, torch.Tensor) and (mine.shape, mine.dtype, mine.device) == (
            value.shape,
            value.dtype,
            value.device,
        )
        return mine.copy_(value) if same else value
    if isinstance(value, list):
        if isinstance(mine, list) and len(mine) == len(value):
            return [_into(a, b) for a, b in zip(mine, value, strict=True)]
        return value
    if isinstance(value, dict):
        held = mine if isinstance(mine, dict) else {}
        return {key: _into(held.get(key), item) for key, item in value.items()}
    return value


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

    def __init__(self, size: list[int], capacity: int) -> None:
        super().__init__(max_cache_len=capacity)
        #: the window, shared with the cache (a holder, not the cache: no reference cycle keeps
        #: the buffers past the call)
        self.size = size

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
        keys, values = super().update(key_states, value_states, *args, **kwargs)
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
    def __init__(self, text_config: Any, capacity: int, size: int) -> None:
        self.size = [size]
        types, kwargs = get_layer_types_and_kwargs(text_config)
        if isinstance(kwargs, dict):  # Transformers before 5.19: one set for every layer
            kwargs = [kwargs] * len(types)
        layers = [
            _Keys(self.size, capacity) if kind == "full_attention" else _State(**extra)
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


class _Decode:
    """A model's decode state, kept between calls with the same geometry: the static cache, the
    step's buffers, and the graphs recorded over them per window, with where the weights they
    read were bound (`Residency.layout`): moved, they record again. Each call refills the
    buffers in place, so the graphs' addresses hold."""

    def __init__(self, model: Any, key: tuple[Any, ...], penalty: float, draft: Any) -> None:
        self.model, self.key, self.penalty, self.draft = model, key, penalty, draft
        batch, self.capacity, self.max_new_tokens = key[0], key[1], key[2]
        #: tokens a step can make: a drafted one, its drafts and the model's token after them
        self.per = DEPTH + 1 if draft is not None else 1
        self.text = model.config.get_text_config(decoder=True)
        self.cache = _StaticCache(self.text, self.capacity, self.capacity)
        self.keep = "logits_to_keep" in set(inspect.signature(model.forward).parameters)
        device = key[5]
        self.ids = torch.zeros((batch, 1), dtype=torch.long, device=device)
        self.length = torch.zeros((1,), dtype=torch.long, device=device)
        valid = torch.zeros((batch, 1, 1, self.capacity), dtype=torch.bool, device=device)
        self.open: bool | float
        self.closed: bool | float
        if self.text._attn_implementation == "sdpa":
            self.mask, self.open, self.closed = valid, True, False
        else:  # eager attention adds its mask: 0 where it attends, the dtype's minimum elsewhere
            self.mask = torch.zeros(valid.shape, dtype=model.dtype, device=device)
            self.open, self.closed = 0.0, torch.finfo(model.dtype).min
        self.out = torch.zeros(
            (batch, self.max_new_tokens + (self.per - 1) * (LAG + 2)),
            dtype=torch.long,
            device=device,
        )
        self.n = torch.zeros((1,), dtype=torch.long, device=device)
        #: `n` where the host reads it, a few steps late
        self.count = torch.zeros((), dtype=torch.long, pin_memory=device.type == "cuda")
        self.done = torch.zeros((batch,), dtype=torch.bool, device=device)
        self.stops = torch.zeros((key[3],), dtype=torch.long, device=device)
        self.pad = torch.zeros((), dtype=torch.long, device=device)
        self.flag = torch.zeros((), dtype=torch.bool, pin_memory=device.type == "cuda")
        self.position: Any = None
        self.seen: Any = None
        #: window -> its graph, and the weights' layout they were recorded over
        self.graphs: dict[int, Any] = {}
        #: `LANDED` when a capture met a kernel still building: none records until one lands
        self.building = -1
        self.layout: tuple[object, ...] | None = None
        self.pool: Any = None
        self.side: Any = None
        #: bytes an eager step allocates (`_measured`): what a capture's pool takes, about
        self.need: int | None = None
        if draft is not None:
            #: the draft head's own keys (its one attention layer), masked to the committed tokens
            self.dcache = Cache(layers=[_Keys(self.cache.size, self.capacity)])
            # on the device from the start: a step sets it before the head first writes a key
            self.dcache.layers[0].cumulative_length = self.length[0].clone()
            self.dmask = torch.full_like(self.mask, self.closed)
            self.rows = torch.arange(self.capacity, device=device)
            #: the state the model's head read for the token `ids` holds: the draft's input
            self.hidden: Any = None
            self.rotary = model.get_decoder().rotary_emb

    def begin(self, inputs: Mapping[str, Any], stops: list[int], pad: int) -> None:
        self.inputs = inputs
        self.prompt = int(inputs["input_ids"].shape[1])
        self.stops.copy_(torch.tensor(stops, dtype=torch.long))
        self.pad.fill_(pad)
        self.made, self.failed, self.facts = 0, "", ""
        self.prefill_s = self.first_s = 0.0

    def prefill(self) -> None:
        """The prompt through the model's own forward into the static cache, and its token."""
        started = time.perf_counter()
        model, inputs = self.model, self.inputs
        ids = inputs["input_ids"]
        # The prompt attends to exactly its own keys: unmasked and causal, the kernels
        # `generate` runs, so the prefill is the same arithmetic.
        self.cache.window = self.prompt
        kwargs = {key: value for key, value in inputs.items() if key != "input_ids"}
        mask = kwargs.get("attention_mask")
        mask = torch.ones_like(ids) if mask is None else mask
        kwargs["attention_mask"] = mask
        position = None
        if "position_ids" in set(inspect.signature(model.forward).parameters):
            position = kwargs["position_ids"] = model._prepare_position_ids_for_generation(
                ids, kwargs
            )
        if bool(mask.all()):  # as `generate`: no padding, no 2D mask
            kwargs["attention_mask"] = None
        if self.keep:
            kwargs["logits_to_keep"] = 1
        fresh = _StaticCache(self.text, self.capacity, self.prompt)
        prepared = model.prepare_inputs_for_generation(
            ids, past_key_values=fresh, use_cache=True, is_first_iteration=True, **kwargs
        )
        # Under the caller's recovery, which makes room op by op, into this call's own cache.
        with _taking(model) as taken:
            logits = model(**prepared, return_dict=True).logits[:, -1]
        with _disable_current_modes():
            self._settle(fresh, logits, mask, position)
            if self.draft is not None:
                self._drafted_prompt(taken[-1], ids, position)
        if ids.is_cuda:
            torch.cuda.current_stream().synchronize()
        self.prefill_s = time.perf_counter() - started

    def _drafted_prompt(self, hidden: Any, ids: Any, position: Any) -> None:
        """The head's keys for the prompt and the first token, as it attends to them while
        drafting: slot j from token j and the model's state for the token before it, so drafts
        read the whole context the head was trained on; and its state for the token after."""
        tokens = torch.cat([ids[:, 1:], self.ids], dim=1)
        slots = self.rows[: self.prompt + 1]
        mask = self.dmask[..., : self.prompt + 1].expand(-1, -1, self.prompt, -1).clone()
        mask.fill_(self.closed).masked_fill_((slots >= 1) & (slots <= slots[1:, None]), self.open)
        if position is not None:
            at = torch.cat([position[..., 1:], self.position], dim=-1)
        else:
            at = slots[None, 1:]
        self.cache.window = self.prompt + 1
        self.dcache.layers[0].cumulative_length.fill_(1)
        embed = self.model.get_input_embeddings()
        last = self.draft(embed(tokens), hidden, at, self.dcache, mask, self.rotary)[:, -1:]
        if self.hidden is None or self.hidden.shape != last.shape:
            self.hidden, self.graphs = torch.empty_like(last), {}
        self.hidden.copy_(last)
        self.dmask.fill_(self.closed)
        self.dmask[..., 1 : self.prompt + 1] = self.open

    def _settle(self, fresh: Any, logits: Any, mask: Any, position: Any) -> None:
        """The prefill's cache into the session's buffers, in place (its graphs read them by
        address), and the decode's state at the prompt's end."""
        for mine, theirs in zip(self.cache.layers, fresh.layers, strict=True):
            for field, value in vars(theirs).items():
                if field != "size":  # the window holder each cache shares with its layers
                    setattr(mine, field, _into(getattr(mine, field, None), value))
        ids = self.inputs["input_ids"]
        if position is not None:
            following = position[..., -1:] + 1
            if self.position is None or self.position.shape != following.shape:
                self.position, self.graphs = following.contiguous(), {}
            else:
                self.position.copy_(following)
        self.length.fill_(self.prompt)
        self.mask.fill_(self.closed)
        self.mask[:, 0, 0, : self.prompt].masked_fill_(mask.bool(), self.open)
        self.out.fill_(int(self.pad))
        self.n.zero_()
        self.done.zero_()
        if self.penalty:
            if self.seen is None:
                self.seen = torch.zeros(
                    (ids.shape[0], int(logits.shape[-1])), dtype=torch.float32, device=ids.device
                )
            self.seen.zero_()
        self.select(logits)
        self.made = 1

    def first(self) -> None:
        """The first step eagerly, through the paging hooks (they record what it reads)."""
        started = time.perf_counter()
        if self.made < self.max_new_tokens and not bool(self.flag):
            self.cache.window = window(self.prompt + self.made + self.per, self.capacity)
            self._measured()
            self.made = int(self.n)
            if self.ids.is_cuda:
                torch.cuda.current_stream().synchronize()
        self.first_s = time.perf_counter() - started

    def select(self, logits: Any) -> None:
        """The next token of each row, on the device: where the step reads it, in `out`, and in
        the stop flag the host reads."""
        scores = logits.float()
        if self.seen is not None:
            scores = scores - self.penalty * self.seen
        token = torch.where(self.done, self.pad, scores.argmax(-1))
        self.done |= (token[:, None] == self.stops).any(-1)
        self.ids.copy_(token[:, None])
        self.out.index_copy_(1, self.n, token[:, None])
        self.n.add_(1)
        if self.seen is not None:
            self.seen.scatter_(1, token[:, None], 1.0)
        self.flag.copy_(self.done.all(), non_blocking=True)
        self.count.copy_(self.n[0], non_blocking=True)

    def step(self) -> None:
        """One token: attend to the window, write this position's keys, select the next."""
        size = self.cache.window
        self.mask.index_fill_(3, self.length, self.open)
        kwargs: dict[str, Any] = {
            "input_ids": self.ids,
            "attention_mask": self.mask[..., :size],
            "past_key_values": self.cache,
            "use_cache": True,
            "return_dict": True,
        }
        if self.position is not None:
            kwargs["position_ids"] = self.position
        if self.keep:
            kwargs["logits_to_keep"] = 1
        self.select(self.model(**kwargs).logits[:, -1])
        self.length.add_(1)
        if self.position is not None:
            self.position.add_(1)

    def _advance(self) -> None:
        if self.draft is None:
            self.step()
        else:
            self._speculate()

    def _speculate(self) -> None:
        """`DEPTH` drafted tokens, verified with the drafted one before them in one pass: the
        model's tokens up to and including its first disagreement are kept, and every state
        goes back to the kept tokens' end."""
        model, depth, size = self.model, DEPTH, self.cache.window
        span = depth + 1
        embed, head = model.get_input_embeddings(), model.get_output_embeddings()
        keys, rows = self.rows[:size], self.rows[:span]
        # Drafts: the head's state for the token after `ids` is in hand; each later draft is the
        # head on the one before, its keys after the committed ones.
        self.dcache.layers[0].cumulative_length.copy_(self.length[0] + 1)
        token, hidden, seen, drafts = self.ids, self.hidden, self.seen, []
        for i in range(depth):
            if i:
                open_ = (keys > self.length) & (keys <= self.length + i)
                mask = torch.where(open_, self.open, self.dmask[..., :size])
                at = self.position + i if self.position is not None else (self.length + i)[:, None]
                hidden = self.draft(embed(token), hidden, at, self.dcache, mask, self.rotary)
            scores = head(hidden)[:, -1].float()
            if seen is not None:
                scores = scores - self.penalty * seen
            token = scores.argmax(-1, keepdim=True)
            drafts.append(token)
            if seen is not None:
                seen = seen.scatter(1, token, 1.0)
        drafted = torch.cat(drafts, dim=1)
        # Verify: each row attends to the committed keys and this pass's up to its own.
        open_ = (keys >= self.length) & (keys <= self.length + rows[:, None])
        kwargs: dict[str, Any] = {
            "input_ids": torch.cat([self.ids, drafted], dim=1),
            "attention_mask": torch.where(open_, self.open, self.mask[..., :size]),
            "past_key_values": self.cache,
            "use_cache": True,
            "return_dict": True,
        }
        if self.position is not None:
            kwargs["position_ids"] = self.position + rows
        if self.keep:
            kwargs["logits_to_keep"] = span
        verify = _Verify()
        reset = _VERIFY.set(verify)
        try:
            with _taking(model) as taken, _RowByRow():
                scores = model(**kwargs).logits[:, -span:].float()
        finally:
            _VERIFY.reset(reset)
        if self.seen is not None:  # row j has seen the drafts before it
            ahead = torch.zeros_like(scores).scatter_(2, drafted[..., None], 1.0).cumsum(1)
            ahead = torch.cat([torch.zeros_like(ahead[:, :1]), ahead[:, :-1]], dim=1)
            scores = scores - self.penalty * torch.maximum(self.seen[:, None], ahead.clamp(max=1))
        best = scores.argmax(-1)
        kept = (best[:, :depth] == drafted).long().cumprod(-1).sum(-1)  # [1]
        valid = rows <= kept
        stopped = (best[..., None] == self.stops).any(-1) & valid
        after = (stopped.long().cumsum(-1) - stopped.long()) > 0
        tokens = torch.where(self.done[:, None] | after, self.pad, best)
        self.done |= stopped.any(-1)
        at = (self.n + rows)[None]
        self.out.scatter_(1, at, torch.where(valid, tokens, self.out.gather(1, at)))
        if self.seen is not None:
            self.seen.scatter_reduce_(1, tokens, valid.float()[None].expand_as(tokens), "amax")
        self.ids.copy_(tokens.gather(1, kept[:, None]))
        # The head's keys for the committed tokens, each from the model's own state for the
        # token before it (as vLLM's and SGLang's drafters keep them), and its state for the
        # token after them: the next step's first draft.
        self.dcache.layers[0].cumulative_length.copy_(self.length[0] + 1)
        open_ = (keys > self.length) & (keys <= self.length + 1 + rows[:, None])
        mask = torch.where(open_, self.open, self.dmask[..., :size])
        at = (
            self.position + 1 + rows
            if self.position is not None
            else (self.length + 1 + rows)[None]
        )
        states = self.draft(embed(tokens), taken[-1][:, -span:], at, self.dcache, mask, self.rotary)
        self.hidden.copy_(states.index_select(1, kept))
        # Back to the kept tokens' end: keys past it are masked and overwritten later.
        ended = self.length + kept + 1
        for layer in self.cache.layers:
            if isinstance(layer, _Keys):
                layer.cumulative_length.copy_(ended[0])
        for state, full in verify.conv:
            state.copy_(full.index_select(-1, self.rows[: state.shape[-1]] + kept + 1))
        for state, states in verify.recurrent:
            state.copy_(states.index_select(0, kept)[0])
        every = self.rows[: self.capacity]
        self.mask.masked_fill_((every >= self.length) & (every < ended), self.open)
        self.dmask.masked_fill_((every > self.length) & (every <= ended), self.open)
        self.length.copy_(ended)
        self.n.add_(kept + 1)
        if self.position is not None:
            self.position.add_(kept + 1)
        self.flag.copy_(self.done.all(), non_blocking=True)
        self.count.copy_(self.n[0], non_blocking=True)

    def run(self, graphs: bool, layout: tuple[object, ...], room: Callable[[], int] | None) -> Any:
        """The steps after the prefill (and the first step, if it ran): each window's recorded
        once and replayed when `graphs` (over weights bound at `layout`), else eager. A window
        records only while `room()` (device bytes torch can take without unmapping a weight)
        is `ROOM` times what an eager step takes at its peak; after a capture fails, the call's
        steps run eagerly. Returns the new tokens."""
        cuda = self.ids.is_cuda
        captures, capture_s, looped = 0, 0.0, self.made
        began = time.perf_counter()
        if graphs and (self.pool is None or layout != self.layout):
            # New, or the weights moved: the graphs record again into a fresh pool, and the old
            # one's memory goes back (a spent pool's free blocks serve no other allocation).
            self.graphs, self.layout = {}, layout
            if self.pool is not None:
                torch.cuda.empty_cache()
            self.pool, self.side = torch.cuda.graph_pool_handle(), torch.cuda.Stream()
        recorded = self.graphs
        queued: deque[Any] = deque()
        # `made` bounds the tokens made so far from above (a drafted step makes 1 to `per`);
        # `counted` is what the host last read, a few steps late.
        counted, steps = self.made, 0
        while (
            counted < self.max_new_tokens
            and (self.per > 1 or self.made < self.max_new_tokens)
            and not (self.made == 1 and bool(self.flag))
        ):
            size = window(self.prompt + self.made + self.per, self.capacity)
            self.cache.window = size
            graph = recorded.get(size)
            if (
                graph is None
                and graphs
                and self.need is not None
                and not self.failed
                and self.building != LANDED[0]
            ):
                free = room() if room is not None else ROOM * self.need
                if free < ROOM * self.need:
                    self.failed = (
                        f"no room: {free >> 20} MiB free, a step takes {self.need >> 20} MiB"
                    )
                else:
                    capturing = time.perf_counter()
                    graph = self._capture()
                    if graph is not None:
                        recorded[size] = graph
                    capture_s += time.perf_counter() - capturing
                    captures += 1
            if graph is not None:
                graph.replay()
            elif graphs and self.need is None:
                self._measured()
            else:
                self._advance()
            self.made += self.per
            steps += 1
            if not cuda:
                counted = int(self.count)
                if bool(self.flag):
                    break
                continue
            event = torch.cuda.Event()
            event.record()
            queued.append(event)
            if len(queued) > LAG:
                queued.popleft().synchronize()
                counted = int(self.count)
                if bool(self.flag):
                    break
        if cuda:
            torch.cuda.current_stream().synchronize()
        self.made = min(int(self.n), self.max_new_tokens)
        loop_s, looped = time.perf_counter() - began, self.made - looped
        mode = (
            "graphs" if graphs and not self.failed else "eager" if not captures else "partly eager"
        )
        if self.draft is not None:
            mode += f", drafting {DEPTH}: {looped / max(steps, 1):.2f} tokens a step"
        tokens = self.out[:, : self.made]
        hit = (tokens[..., None] == self.stops).any(-1)
        if bool(hit.any(-1).all()):
            tokens = tokens[:, : int(hit.int().argmax(-1).max()) + 1]
        per = 1000 * (loop_s - capture_s) / max(looped, 1)
        facts = (
            f"prefill {self.prefill_s:.3f} s",
            *((f"first step {self.first_s:.3f} s",) if self.first_s else ()),
            f"then {per:.2f} ms/token over {looped} ({mode})",
            f"{captures} captures {capture_s:.3f} s",
            f"gated-delta kernels {'on' if _LAUNCHED[0] else 'off'}",
            *((f"capture failed: {self.failed}",) if self.failed else ()),
        )
        self.facts = f"{int(tokens.shape[1])} tokens after {self.prompt}: " + ", ".join(facts)
        print(f"[lm] {self.facts}", file=sys.stderr)
        placed = PLACED.get(self.model)
        if placed is not None:
            placed[0].note(f"{placed[1]}.lm", self.facts)
        return tokens

    def _measured(self) -> None:
        """An eager step, and the device bytes it takes at its peak: the segments it reserves
        beyond what torch holds with its idle cache given back, as a graph's pool would."""
        torch.cuda.empty_cache()
        before = torch.cuda.memory_reserved(self.ids.device)
        self._advance()
        self.need = max(self.need or 0, torch.cuda.memory_reserved(self.ids.device) - before)

    def _capture(self) -> Any:
        """The step at the current window as a CUDA graph, or None, with a run warning: the
        call's steps then run eagerly, and the next call records into a fresh pool. A capture
        executes nothing, so the buffers are as they were."""
        try:
            graph, why = capture(self._advance, self.pool, self.side)
        except Building:  # a kernel still builds (a fresh machine): eager until one lands
            self.building, self.pool = LANDED[0], None
            torch.cuda.empty_cache()
            return None
        if graph is None:
            self.failed, self.pool = why, None
            torch.cuda.empty_cache()
            say(f"capture:{type(self.model).__name__}", f"[lm] capture failed: {why}")
            WARNINGS.append(
                ("a CUDA graph was not used", {"module": type(self.model).__name__, "why": why})
            )
        return graph


__all__ = [
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
