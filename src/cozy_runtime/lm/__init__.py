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
not all fit, on a CPU, or when a capture fails, the same step runs eagerly. A model this does not
serve (a layer type other than full or linear attention, an attention other than SDPA or eager, a
generation config that adds logits processors) generates through Transformers, said once.
"""

from __future__ import annotations

import inspect
import sys
import time
import weakref
from collections import deque
from collections.abc import Mapping, Sequence
from typing import Any

from cozy_runtime.author._model import PLACED, RELEASERS
from cozy_runtime.author.kernels import KernelsUnavailable, Launch, Plan, TritonKernels
from cozy_runtime.graphs import capture, say
from cozy_runtime.lm import _kernels

try:
    import torch
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
    StaticLayer = LinearAttentionLayer = Cache = object
else:
    ABSENT = ""

#: The smallest attention window; windows double from it up to the cache's capacity.
MIN_WINDOW = 256
#: Steps the host may queue ahead of the stop flag it last read.
LAG = 2
#: Captures tried per window; an eager step runs between them.
CAPTURES = 2
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
) -> Any:
    """Greedy tokens after the prompt, `[batch, n]` (`model.generate(..., do_sample=False)`'s
    tail): `inputs` as the processor returned them, generation ends at a `stop` token or after
    `max_new_tokens`, and a row that stopped early continues as `pad` (default: the first stop).
    `presence_penalty` subtracts that much from every token already generated (vLLM's)."""
    stops = [int(token) for token in stop]
    filler = stops[0] if pad is None else int(pad)
    why = ABSENT or _unsupported(model)
    if why:
        say(
            type(model).__name__, f"[lm] {type(model).__name__} decodes through Transformers: {why}"
        )
        return _transformers(model, inputs, max_new_tokens, stops, filler, presence_penalty)
    with torch.inference_mode():
        ids = inputs["input_ids"]
        prompt, cuda = int(ids.shape[1]), ids.is_cuda
        # Prompts within one window step share a session (its buffers and graphs).
        capacity = -(-(prompt + max_new_tokens) // MIN_WINDOW) * MIN_WINDOW
        key = (
            int(ids.shape[0]),
            capacity,
            max_new_tokens,
            len(stops),
            presence_penalty,
            ids.device,
        )
        decode = _SESSIONS.get(model)
        if decode is None or decode.key != key:
            decode = _SESSIONS[model] = _Decode(model, key, presence_penalty)
        decode.begin(inputs, stops, filler)
        placed = PLACED.get(model)
        if placed is None:  # a module Runtime does not place: its tensors stay where they are
            decode.prefill()
            return decode.run(graphs=cuda, layout=())
        residency, name = placed
        blocks = _READS.get(model)
        if blocks is None:
            # The first call: its first step runs through the paging hooks, which record the
            # blocks a step reads; later calls hold them before their prefill.
            decode.prefill()
            with residency.recording() as read:
                decode.first()
            blocks = _READS[model] = frozenset(read.get(name, ()))
            with residency.held(name, blocks) as held:
                if held:
                    return decode.run(graphs=True, layout=residency.layout(name, blocks))
        else:
            with residency.held(name, blocks) as held:
                if held:
                    decode.prefill()
                    return decode.run(graphs=True, layout=residency.layout(name, blocks))
        # They do not all fit at home beside what this stage holds: Transformers' own loop
        # streams them, which a static cache's recovery copies would only slow down.
        say(
            f"held:{type(model).__name__}",
            f"[lm] {type(model).__name__}: no graphs, decode set does not fit",
        )
        return _transformers(model, inputs, max_new_tokens, stops, filler, presence_penalty)


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
            launch(
                _kernels.recurrent_step_kernel,
                (batch * heads, _kernels.VALUE_DIM // _kernels.VALUE_BLOCK, 1),
                query,
                key,
                value,
                g,
                beta,
                initial_state,
                out,
                heads,
                query.stride(0),
                query.stride(2),
                key.stride(0),
                key.stride(2),
                value.stride(0),
                value.stride(2),
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


class _StaticCache(Cache):  # type: ignore[misc]
    def __init__(self, text_config: Any, capacity: int, size: int) -> None:
        self.size = [size]
        types, kwargs = get_layer_types_and_kwargs(text_config)
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

    def __init__(self, model: Any, key: tuple[Any, ...], penalty: float) -> None:
        self.model, self.key, self.penalty = model, key, penalty
        batch, self.capacity, self.max_new_tokens = key[0], key[1], key[2]
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
        self.out = torch.zeros((batch, self.max_new_tokens), dtype=torch.long, device=device)
        self.n = torch.zeros((1,), dtype=torch.long, device=device)
        self.done = torch.zeros((batch,), dtype=torch.bool, device=device)
        self.stops = torch.zeros((key[3],), dtype=torch.long, device=device)
        self.pad = torch.zeros((), dtype=torch.long, device=device)
        self.flag = torch.zeros((), dtype=torch.bool, pin_memory=device.type == "cuda")
        self.position: Any = None
        self.seen: Any = None
        #: window -> its graph, and the weights' layout they were recorded over
        self.graphs: dict[int, Any] = {}
        self.layout: tuple[object, ...] | None = None
        self.pool: Any = None
        self.side: Any = None

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
        prepared = model.prepare_inputs_for_generation(
            ids, past_key_values=self.cache, use_cache=True, is_first_iteration=True, **kwargs
        )

        def forward() -> Any:
            self.cache.reset()
            return model(**prepared, return_dict=True).logits[:, -1]

        try:
            # Outside Runtime's per-op dispatch, as a paging hook runs a block that writes
            # nothing it did not make: held blocks have no hook (2.2k image tokens: 1.09 s
            # under it, 0.49 s without, on a 4090).
            with _disable_current_modes():
                logits = forward()
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            logits = forward()  # again under the recovery, which makes room op by op
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
        if ids.is_cuda:
            torch.cuda.current_stream().synchronize()
        self.prefill_s = time.perf_counter() - started

    def first(self) -> None:
        """The first step eagerly, through the paging hooks (they record what it reads)."""
        started = time.perf_counter()
        if self.made < self.max_new_tokens and not bool(self.flag):
            self.cache.window = window(self.prompt + self.made, self.capacity)
            self.step()
            self.made += 1
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

    def run(self, graphs: bool, layout: tuple[object, ...]) -> Any:
        """The steps after the prefill (and the first step, if it ran): each window's recorded
        once and replayed when `graphs` (over weights bound at `layout`), else eager. Returns
        the new tokens."""
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
        recorded, pool, side = self.graphs, self.pool, self.side
        tries: dict[int, int] = {}
        queued: deque[Any] = deque()
        while self.made < self.max_new_tokens and not (self.made == 1 and bool(self.flag)):
            size = window(self.prompt + self.made, self.capacity)
            self.cache.window = size
            if graphs and recorded.get(size) is None and tries.get(size, 0) < CAPTURES:
                # One more try after an eager step: a capture that ran out of memory leaves its
                # pool spent, and the eager step's recovery makes room.
                capturing = time.perf_counter()
                recorded[size] = self._capture(pool, side)
                if recorded[size] is None:
                    torch.cuda.empty_cache()
                    pool = self.pool = torch.cuda.graph_pool_handle()
                tries[size] = tries.get(size, 0) + 1
                capture_s += time.perf_counter() - capturing
                captures += 1
            graph = recorded.get(size)
            if graph is not None:
                graph.replay()
            else:
                self.step()
            self.made += 1
            if not cuda:
                if bool(self.flag):
                    break
                continue
            event = torch.cuda.Event()
            event.record()
            queued.append(event)
            if len(queued) > LAG:
                queued.popleft().synchronize()
                if bool(self.flag):
                    break
        if cuda:
            torch.cuda.current_stream().synchronize()
        loop_s, looped = time.perf_counter() - began, self.made - looped
        mode = "graphs" if graphs else "eager"
        if graphs and None in recorded.values():
            mode = "partly eager: a capture failed"
            self.graphs = {size: g for size, g in recorded.items() if g is not None}
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

    def _capture(self, pool: Any, side: Any) -> Any:
        """The step at the current window as a CUDA graph, or None (said once): it then runs
        eagerly."""
        graph, why = capture(self.step, pool, side)
        if graph is None:
            self.failed = self.failed or why
            say(f"capture:{type(self.model).__name__}", f"[lm] capture failed: {why}")
        return graph


__all__ = ["KERNELS", "MIN_WINDOW", "generate", "install", "release", "window"]
