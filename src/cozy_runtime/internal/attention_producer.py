"""Explicit one-GPU H3 Kitchen producer experiments; never a default.

Sparse blocks avoid full BF16 Q/K/V. Kernel norm/RoPE rounding, V margin1.1 and
INT8 dense prefix blocks are explicit quality changes. Reuse of previous-step
statistics is a separately named approximation, owned by one invocation only.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from importlib import import_module
from typing import Any

from cozy_runtime.author._attention_scope import _ACTIVE_LAYOUT, AttentionLayout
from cozy_runtime.internal import attention_sol, attention_ulysses, attention_upstream
from cozy_runtime.internal.attention_ulysses import Local, Options
from cozy_runtime.internal.encoding.leaves import PreQuantized

NAMES = frozenset(
    {"kitchen-sol-producer", "kitchen-sol-producer-reuse", "kitchen-sol-producer-fast"}
)
REUSE = frozenset({"kitchen-sol-producer-reuse", "kitchen-sol-producer-fast"})
BACKENDS = {"_cozy_" + name.replace("-", "_"): name for name in NAMES}
CHUNK = 4096


@dataclass
class RequestState:
    statistics: dict[tuple[str, str], tuple[int, tuple[int, ...], Any, Any]] = field(
        default_factory=dict
    )
    # One rotary presentation is shared by all blocks of the current step.
    rope: tuple[Any, Any, Any] | None = None


_STATE: ContextVar[RequestState | None] = ContextVar("cozy_h3_producer", default=None)


@contextmanager
def request_scope() -> Iterator[RequestState]:
    state = RequestState()
    token = _STATE.set(state)
    try:
        yield state
    finally:
        state.statistics.clear()
        # Upstream holds its packed rotary presentation in a one-entry weak-key
        # cache. Remove only our own entry, so no invocation GPU buffer survives
        # into Runtime's quiescence accounting or a later request.
        backend = sys.modules.get("comfy_kitchen.backends.cuda")
        cached = getattr(backend, "_ROPE_FAB_CACHE", {})
        if state.rope is not None:
            for key, (reference, _packed) in list(cached.items()):
                if reference() is state.rope[2]:
                    cached.pop(key, None)
        state.rope = None
        _STATE.reset(token)


def dense(*, query: Any, key: Any, value: Any, scale: float | None) -> Any:
    # Record nested Sage through the same observer as the BF16 Sol reference.
    return attention_ulysses._call(
        "_cozy_sage_int8_fp8",
        attention_upstream.SAGE,
        query,
        key,
        value,
        Options(attn_mask=None, dropout_p=0.0, is_causal=False, scale=scale, enable_gqa=False),
    )


def _missing_processor(*args: Any) -> Any:
    raise ValueError("Kitchen producer requires its H3 instance processor before QKV projection")


def materialized(query: Any, key: Any, value: Any, **options: Any) -> Any:
    """Dense policy branches of the original processor; sparse must use the producer."""
    return attention_sol._attention(
        query, key, value, **options, sparse=_missing_processor, dense=dense
    )


LOCAL = Local(
    materialized, lambda *_: "dense=sageattention; sparse=instance-producer", head_local=False
)


def materialized_kitchen(query: Any, key: Any, value: Any, **options: Any) -> Any:
    return attention_sol._attention(
        query,
        key,
        value,
        **options,
        sparse=_missing_processor,
        dense=attention_upstream.kitchen_int8,
    )


FAST_LOCAL = Local(
    materialized_kitchen,
    lambda *_: "dense=comfy_kitchen.int8; sparse=instance-producer",
    head_local=False,
)


def rows(value: Any, start: int, stop: int) -> Any:
    """View-only rows: preserve encoded payload/scales, never requantize an activation."""
    if isinstance(value, PreQuantized):
        if len(value.shape) != 3 or value.shape[0] != 1:
            raise ValueError("H3 producer requires batch1 activations")
        return PreQuantized(
            value.payload[start:stop], value.scale[start:stop], (1, stop - start, value.shape[-1])
        )
    return value[:, start:stop]


def qkv_chunks(attn: Any, hidden: Any, tokens: int) -> Iterator[Any]:
    import torch

    for start in range(0, tokens, CHUNK):
        chunk = rows(hidden, start, min(tokens, start + CHUNK))
        # Original module calls retain encoded leaves, resident weights and Turbo hooks.
        yield torch.cat(
            [projection(chunk) for projection in (attn.to_q, attn.to_k, attn.to_v)], dim=-1
        ).squeeze(0)


def rope_presentation(rotary: tuple[Any, Any], tokens: int, state: RequestState) -> Any:
    import torch

    cos, sin = rotary
    if state.rope is not None and state.rope[0] is cos and state.rope[1] is sin:
        return state.rope[2]
    if cos.shape != sin.shape or cos.shape[-1] % 8:
        raise ValueError("H3 producer requires matching split-half rotary tables")
    width = cos.shape[-1]
    c, s = cos.reshape(-1, width)[:tokens], sin.reshape(-1, width)[:tokens]
    if c.shape[0] != tokens:
        raise ValueError("H3 producer rotary tables do not cover the live document")
    half = width // 2
    freqs = torch.stack((c[:, :half], -s[:, :half], s[:, half:], c[:, half:]), dim=-1)
    freqs = freqs.reshape(tokens, 1, half, 2, 2).float()
    state.rope = cos, sin, freqs
    return freqs


def previous_statistics(
    state: RequestState, key: tuple[str, str], step: int, geometry: tuple[int, ...], reuse: bool
) -> tuple[Any, Any]:
    prior = state.statistics.pop(key, None)
    if reuse and prior is not None and prior[:2] == (step - 1, geometry):
        return prior[2], prior[3]
    return None, None


def sparse_step(layout: AttentionLayout, site: attention_sol.Site) -> bool:
    return layout.step >= layout.dense_until_step and not any(
        site.path == p or site.path.startswith(p + ".") for p in layout.dense_paths
    )


class Processor:
    """A wrapper installed on one H3 instance; restoring its backend restores original execution."""

    def __init__(self, original: Any) -> None:
        self.original = original

    @property
    def _attention_backend(self) -> Any:
        return self.original._attention_backend

    @_attention_backend.setter
    def _attention_backend(self, value: Any) -> None:
        self.original._attention_backend = value

    @property
    def _parallel_config(self) -> Any:
        return self.original._parallel_config

    @_parallel_config.setter
    def _parallel_config(self, value: Any) -> None:
        self.original._parallel_config = value

    def __call__(
        self, attn: Any, hidden_states: Any, rotary_emb: Any = None, attention_mask: Any = None
    ) -> Any:
        backend = getattr(self._attention_backend, "value", str(self._attention_backend))
        name = BACKENDS.get(backend)
        if name is None:
            return self.original(attn, hidden_states, rotary_emb, attention_mask)
        layout, site, state = _ACTIVE_LAYOUT.get(), attention_sol._SITE.get(), _STATE.get()
        if layout is None or site is None or state is None:
            raise ValueError("H3 producer requires model layout and invocation-scoped state")
        if getattr(self._parallel_config, "context_parallel_config", None) is not None:
            raise ValueError("H3 chunked producer currently requires one GPU")
        if attention_mask is not None:
            raise ValueError("H3 producer serves unmasked noncausal attention only")
        key = site.component, site.path
        if not sparse_step(layout, site):
            state.statistics.pop(key, None)
            return self.original(attn, hidden_states, rotary_emb, attention_mask)
        if rotary_emb is None or hidden_states.shape[0] != 1:
            raise ValueError("H3 producer requires batch1 and rotary tables")
        import torch

        operand = (
            hidden_states.payload if isinstance(hidden_states, PreQuantized) else hidden_states
        )
        if operand.device.type != "cuda" or torch.cuda.get_device_capability(operand.device) != (
            12,
            0,
        ):
            raise ValueError("H3 producer is an explicit SM120 experiment")
        tokens = layout.live_tokens
        if hidden_states.shape[1] != tokens:
            raise ValueError("single-GPU producer expects an unpadded live H3 document")
        if attn.norm_q.eps != attn.norm_k.eps:
            raise ValueError("Kitchen producer requires matching Q/K RMSNorm epsilon")
        geometry = (tokens, attn.heads, layout.protected_prefix)
        prior_mean, prior_scale = previous_statistics(
            state, key, layout.step, geometry, name in REUSE
        )
        reuse = prior_mean is not None
        freqs = rope_presentation(rotary_emb, tokens, state)
        kitchen = import_module("comfy_kitchen")
        output, mean, scale = kitchen.sol_attn_chunked(
            lambda: qkv_chunks(attn, hidden_states, tokens),
            tokens,
            attn.heads,
            freqs,
            (attn.norm_q.weight, attn.norm_k.weight),
            kmean=prior_mean,
            vscale=prior_scale,
            tau=1.0,
            scale=128**-0.5,
            rope_eps=attn.norm_q.eps,
            sink_blocks=[0, -(-layout.protected_prefix // 64)],
            sink_q=[0, -(-layout.protected_prefix // 64)],
            token_aug=0,
        )
        if name in REUSE:
            state.statistics[key] = (layout.step, geometry, mean, scale)
        statistics = "previous-step" if reuse else "current-call"
        attention_ulysses.record(
            backend,
            f"comfy_kitchen.sol_attn_chunked(int8, stats={statistics}, vmargin=1.1, prefix=int8)",
        )
        attention_sol._count("sparse")
        if layout.protected_prefix:
            attention_sol._count("dense_prefix")
        return attn.to_out[1](attn.to_out[0](output.flatten(2, 3)))


def validate(module: Any) -> None:
    from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3Attention

    if not isinstance(module, MiniMaxH3Attention) or module.fused_projections:
        raise ValueError("Kitchen producer requires unfused MiniMaxH3Attention projection modules")
    if module.head_dim != 128 or module.to_q.out_features != module.heads * 128:
        raise ValueError("Kitchen producer requires H3 head dimension128")
    original = (
        module.processor.original if isinstance(module.processor, Processor) else module.processor
    )
    if type(original).__name__ not in {"MiniMaxH3AttnProcessor", "FusedMiniMaxH3AttnProcessor"}:
        raise ValueError("Kitchen producer does not replace an unknown H3 processor")
    for projection in (module.to_q, module.to_k, module.to_v):
        # Weight-plane hooks are pre-existing Runtime custody; author forward hooks must
        # explicitly support replay/chunks. Turbo's hook does and keeps fixed4096 rows.
        for hook in projection._forward_hooks.values():
            if type(hook).__name__ != "_LoRAHook":
                raise ValueError("Kitchen producer cannot replay an unknown projection hook")


def install(module: Any) -> None:
    validate(module)
    if not isinstance(module.processor, Processor):
        module.set_processor(Processor(module.processor))
