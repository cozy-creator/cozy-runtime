"""H3 shared-QKV Kitchen producer with invocation-local sparse statistics.

The approved SM120 recipe (also supported by upstream on SM121) uses Sage3
global-mean NVFP4 for dense calls and
Kitchen INT8 Sol for sparse calls. Other admitted NVIDIA devices use Kitchen
INT8 for both. The model owns dense steps and protected paths; explicit pins
remain available. Sparse calls retain the tested4096-row projection chunks.
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
from cozy_runtime.internal import (
    attention_kitchen_sol,
    attention_sol,
    attention_ulysses,
    attention_upstream,
)
from cozy_runtime.internal.attention_ulysses import Local
from cozy_runtime.internal.encoding import leaves
from cozy_runtime.internal.encoding.leaves import PreQuantized

NAMES = frozenset(
    {"kitchen-sol-producer-fast-shared-qkv", "kitchen-sol-producer-fp4-lowmem-shared-qkv"}
)
REUSE = SHARED_QKV = NAMES
FP4_LOWMEM = frozenset({"kitchen-sol-producer-fp4-lowmem-shared-qkv"})
BACKENDS = {"_cozy_" + name.replace("-", "_"): name for name in NAMES}
CHUNK = 4096


@dataclass
class RequestState:
    # Current outputs plus the original input statistics: a block can replay this step
    # after a later allocation fails, outside the producer's own call.
    statistics: dict[tuple[str, str], tuple[int, tuple[int, ...], Any, Any, Any, Any]] = field(
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


def _missing_processor(*args: Any) -> Any:
    raise ValueError("Kitchen producer requires its H3 instance processor before QKV projection")


def rows(value: Any, start: int, stop: int) -> Any:
    """View-only rows: preserve encoded payload/scales, never requantize an activation."""
    if isinstance(value, PreQuantized):
        if len(value.shape) != 3 or value.shape[0] != 1:
            raise ValueError("H3 producer requires batch1 activations")
        return PreQuantized(
            value.payload[start:stop], value.scale[start:stop], (1, stop - start, value.shape[-1])
        )
    return value[:, start:stop]


def shared_qkv_refusal(attn: Any) -> str:
    """Only known tensor-preserving hooks may share the positional operand's codes.

    WeightResidency wraps forward rather than registering a pre-hook. Calling the
    original modules below keeps its acquire/bind/bake/release boundary intact.
    """
    from torch.nn.modules import module as torch_module

    from cozy_runtime.models.minimax_h3.turbo import _LoRAHook

    if torch_module._global_forward_pre_hooks or torch_module._global_forward_hooks:
        return "global-projection-hook"
    for projection in (attn.to_q, attn.to_k, attn.to_v):
        if getattr(projection, "provider", None) != "cozy.fp8-rowwise.native-leaf/1":
            return "non-rowwise-leaf"
        if projection._forward_pre_hooks:
            return "projection-pre-hook"
        if any(not isinstance(hook, _LoRAHook) for hook in projection._forward_hooks.values()):
            return "unknown-projection-hook"
    return ""


def qkv_chunks(
    attn: Any,
    hidden: Any,
    tokens: int,
    *,
    shared: bool = False,
    counts: dict[str, int] | None = None,
) -> Iterator[Any]:
    import torch

    refusal = shared_qkv_refusal(attn) if shared else "disabled"
    for start in range(0, tokens, CHUNK):
        chunk = rows(hidden, start, min(tokens, start + CHUNK))
        operand = None
        outcome = refusal
        if shared and isinstance(chunk, PreQuantized):
            outcome = "already-quantized"
        elif shared and not refusal:
            payload, scale = leaves.quantize_activation_rowwise(torch, chunk)
            operand = PreQuantized(payload, scale, tuple(chunk.shape))
            outcome = "shared"
        if shared and counts is not None:
            counts[outcome] = counts.get(outcome, 0) + 1
        # Original module calls retain encoded leaves, resident weights and Turbo hooks.
        yield torch.cat(
            [
                projection(chunk) if operand is None else projection(chunk, prequantized=operand)
                for projection in (attn.to_q, attn.to_k, attn.to_v)
            ],
            dim=-1,
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
    # H3 is guidance-distilled: one forward per site/evaluation, with no CFG branch.
    # Re-entering that same evaluation must retain its original quantization inputs.
    if not reuse:
        state.statistics.pop(key, None)
        return None, None
    prior = state.statistics.get(key)
    if prior is None or prior[1] != geometry:
        return None, None
    if prior[0] == step:
        return prior[4], prior[5]
    if prior[0] == step - 1:
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
        capability = (
            torch.cuda.get_device_capability(operand.device)
            if operand.device.type == "cuda"
            else None
        )
        admitted = ((12, 0), (12, 1)) if name in FP4_LOWMEM else attention_upstream.KITCHEN_SMS
        if capability not in admitted:
            raise ValueError(f"{name} does not support device capability {capability}")
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
        shared_counts: dict[str, int] = {}
        output, mean, scale = kitchen.sol_attn_chunked(
            lambda: qkv_chunks(
                attn, hidden_states, tokens, shared=name in SHARED_QKV, counts=shared_counts
            ),
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
            state.statistics[key] = (layout.step, geometry, mean, scale, prior_mean, prior_scale)
        statistics = "previous-step" if reuse else "current-call"
        shared_detail = ""
        if name in SHARED_QKV:
            shared_detail = ", qkv=" + ",".join(
                f"{kind}:{count}" for kind, count in sorted(shared_counts.items())
            )
        attention_ulysses.record(
            backend,
            f"comfy_kitchen.sol_attn_chunked(int8, stats={statistics}, vmargin=1.1, "
            f"prefix=int8{shared_detail})",
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


def materialized_kitchen(query: Any, key: Any, value: Any, **options: Any) -> Any:
    return attention_sol._attention(
        query,
        key,
        value,
        **options,
        sparse=_missing_processor,
        dense=attention_kitchen_sol.dense_kitchen,
        validate_device=attention_kitchen_sol.validate_device,
    )


def materialized_fp4_lowmem(query: Any, key: Any, value: Any, **options: Any) -> Any:
    return attention_sol._attention(
        query,
        key,
        value,
        **options,
        sparse=_missing_processor,
        dense=attention_kitchen_sol.dense_fp4_lowmem,
        validate_device=attention_kitchen_sol.validate_device,
    )


FAST_LOCAL = Local(
    materialized_kitchen,
    lambda *_: "dense=comfy_kitchen.int8; sparse=instance-producer",
    head_local=False,
)
FP4_LOWMEM_LOCAL = Local(
    materialized_fp4_lowmem,
    lambda *_: (
        "dense=sageattn3.nvfp4_global_head_mean(short_owned_lifetimes); sparse=instance-producer"
    ),
    head_local=False,
)
LOCALS = {
    "kitchen-sol-producer-fast-shared-qkv": FAST_LOCAL,
    "kitchen-sol-producer-fp4-lowmem-shared-qkv": FP4_LOWMEM_LOCAL,
}
