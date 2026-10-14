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
from typing import TYPE_CHECKING, Never, Unpack

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
from cozy_runtime.internal.weights import LoRASource

if TYPE_CHECKING:
    from cozy_runtime.internal.attention_ulysses import PartialOptions

    from collections.abc import Callable

    from diffusers.models.attention_dispatch import AttentionBackendName
    from diffusers.models._modeling_parallel import ParallelConfig
    from diffusers.models.transformers.transformer_minimax_h3 import (
        MiniMaxH3Attention,
        MiniMaxH3AttnProcessor,
        FusedMiniMaxH3AttnProcessor,
    )
    from torch import Tensor

NAMES = frozenset(
    {
        "kitchen-sol-producer-fast-shared-qkv",
        "kitchen-sol-producer-fp4-lowmem-shared-qkv",
        "kitchen-sol-producer-fp4-fixed-means-shared-qkv",
    }
)
REUSE = SHARED_QKV = NAMES
FP4_LOWMEM = frozenset(
    {
        "kitchen-sol-producer-fp4-lowmem-shared-qkv",
        "kitchen-sol-producer-fp4-fixed-means-shared-qkv",
    }
)
FIXED_MEANS = frozenset({"kitchen-sol-producer-fp4-fixed-means-shared-qkv"})
CP_NAMES = FIXED_MEANS
BACKENDS = {"_cozy_" + name.replace("-", "_"): name for name in NAMES}
CHUNK = 4096


@dataclass
class RequestState:
    statistics: dict[
        tuple[object, ...], tuple[int, tuple[int, ...], Tensor | None, Tensor | None]
    ] = field(default_factory=dict)
    # One rotary presentation is shared by all blocks of the current step.
    rope: tuple[Tensor, Tensor, Tensor] | None = None
    parallel_rotary: dict[object, tuple[Tensor, Tensor]] = field(default_factory=dict)
    replay_inputs: dict[
        tuple[object, ...], tuple[int, tuple[int, ...], Tensor | None, Tensor | None]
    ] = field(default_factory=dict)

    def clear(self) -> None:
        self.statistics.clear()
        self.parallel_rotary.clear()
        self.replay_inputs.clear()
        backend = sys.modules.get("comfy_kitchen.backends.cuda")
        cached = getattr(backend, "_ROPE_FAB_CACHE", {})
        if self.rope is not None:
            for key, (reference, _packed) in list(cached.items()):
                if reference() is self.rope[2]:
                    cached.pop(key, None)
        self.rope = None


_STATE: ContextVar[RequestState | None] = ContextVar("cozy_h3_producer", default=None)


@contextmanager
def request_scope(state: RequestState | None = None) -> Iterator[RequestState]:
    """Own one invocation, or borrow the follower's existing attempt state for one call."""
    owned = state is None
    state = RequestState() if state is None else state
    token = _STATE.set(state)
    try:
        yield state
    finally:
        if owned:
            state.clear()
        _STATE.reset(token)


def _missing_processor(*args: object) -> Never:
    raise ValueError("Kitchen producer requires its H3 instance processor before QKV projection")


def rows(value: Tensor | PreQuantized, start: int, stop: int) -> Tensor | PreQuantized:
    """View-only rows: preserve encoded payload/scales, never requantize an activation."""
    if isinstance(value, PreQuantized):
        if len(value.shape) != 3 or value.shape[0] != 1:
            raise ValueError("H3 producer requires batch1 activations")
        return PreQuantized(
            value.payload[start:stop], value.scale[start:stop], (1, stop - start, value.shape[-1])
        )
    return value[:, start:stop]


def shared_qkv_refusal(attn: MiniMaxH3Attention) -> str:
    """Only known tensor-preserving hooks may share the positional operand's codes: a
    model's LoRA hook reads the original tensor operand and adds into the output.

    WeightResidency wraps forward rather than registering a pre-hook. Calling the
    original modules below keeps its acquire/bind/bake/release boundary intact.
    """
    from torch.nn.modules import module as torch_module

    if torch_module._global_forward_pre_hooks or torch_module._global_forward_hooks:
        return "global-projection-hook"
    for projection in (attn.to_q, attn.to_k, attn.to_v):
        if getattr(projection, "provider", None) != "cozy.fp8-rowwise.native-leaf/1":
            return "non-rowwise-leaf"
        if projection._forward_pre_hooks:
            return "projection-pre-hook"
        if any(not isinstance(hook, LoRASource) for hook in projection._forward_hooks.values()):
            return "unknown-projection-hook"
    return ""


def qkv_chunks(
    attn: MiniMaxH3Attention,
    hidden: Tensor | PreQuantized,
    tokens: int,
    *,
    shared: bool = False,
    counts: dict[str, int] | None = None,
) -> Iterator[Tensor]:
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


def rope_presentation(rotary: tuple[Tensor, Tensor], tokens: int, state: RequestState) -> Tensor:
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
    state: RequestState,
    key: tuple[object, ...],
    step: int,
    geometry: tuple[int, ...],
    reuse: bool,
    *,
    replay_safe: bool = False,
) -> tuple[Tensor | None, Tensor | None]:
    # Only the opt-in CP recipe changes replay semantics. A later projection or
    # another rank may fail after this site succeeded; retry its original inputs.
    held = state.replay_inputs.get(key) if replay_safe else None
    if held is not None and held[:2] == (step, geometry):
        return held[2], held[3]
    prior = state.statistics.pop(key, None)
    inputs = (
        prior[2:]
        if reuse and prior is not None and prior[:2] == (step - 1, geometry)
        else (None, None)
    )
    if replay_safe:
        state.replay_inputs[key] = (step, geometry, *inputs)
    return inputs


def sparse_step(layout: AttentionLayout, site: attention_sol.Site) -> bool:
    return layout.step >= layout.dense_until_step and not any(
        site.path == p or site.path.startswith(p + ".") for p in layout.dense_paths
    )


class Processor:
    """A wrapper installed on one H3 instance; restoring its backend restores original execution."""

    def __init__(self, original: MiniMaxH3AttnProcessor | FusedMiniMaxH3AttnProcessor) -> None:
        self.original = original

    @property
    def _attention_backend(self) -> AttentionBackendName | str | None:
        return self.original._attention_backend

    @_attention_backend.setter
    def _attention_backend(self, value: AttentionBackendName | str | None) -> None:
        self.original._attention_backend = value

    @property
    def _parallel_config(self) -> ParallelConfig | None:
        return self.original._parallel_config

    @_parallel_config.setter
    def _parallel_config(self, value: ParallelConfig | None) -> None:
        self.original._parallel_config = value

    def __call__(
        self,
        attn: MiniMaxH3Attention,
        hidden_states: Tensor | PreQuantized,
        rotary_emb: tuple[Tensor, Tensor] | None = None,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        backend = getattr(self._attention_backend, "value", str(self._attention_backend))
        name = BACKENDS.get(backend)
        if name is None:
            return self.original(attn, hidden_states, rotary_emb, attention_mask)
        layout, site, state = _ACTIVE_LAYOUT.get(), attention_sol._SITE.get(), _STATE.get()
        if layout is None or site is None or state is None:
            raise ValueError("H3 producer requires model layout and invocation-scoped state")
        parallel = getattr(self._parallel_config, "context_parallel_config", None) is not None
        if parallel and name not in CP_NAMES:
            raise ValueError("This H3 chunked producer requires one GPU")
        if attention_mask is not None:
            raise ValueError("H3 producer serves unmasked noncausal attention only")
        key = (
            (site.component, site.path, id(attn))
            if name in CP_NAMES
            else (site.component, site.path)
        )
        if not sparse_step(layout, site):
            state.statistics.pop(key, None)
            state.replay_inputs.pop(key, None)
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
        admitted = (
            ((12, 0),)
            if name in FIXED_MEANS
            else ((12, 0), (12, 1))
            if name in FP4_LOWMEM
            else attention_upstream.KITCHEN_SMS
        )
        if capability not in admitted:
            raise ValueError(f"{name} does not support device capability {capability}")
        tokens = layout.live_tokens
        if not parallel and hidden_states.shape[1] != tokens:
            raise ValueError("single-GPU producer expects an unpadded live H3 document")
        if attn.norm_q.eps != attn.norm_k.eps:
            raise ValueError("Kitchen producer requires matching Q/K RMSNorm epsilon")
        shared_counts: dict[str, int] = {}
        if parallel:
            output, reuse = _parallel_produce(
                attn, hidden_states, self._parallel_config, name, layout, state, key, shared_counts
            )
        else:
            freqs = rope_presentation(rotary_emb, tokens, state)
            output, reuse = _produce(
                attn,
                layout,
                state,
                key,
                name,
                attn.heads,
                (tokens, attn.heads, layout.protected_prefix),
                freqs,
                lambda: qkv_chunks(
                    attn, hidden_states, tokens, shared=name in SHARED_QKV, counts=shared_counts
                ),
            )
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


def _produce(
    attn: MiniMaxH3Attention,
    layout: AttentionLayout,
    state: RequestState,
    key: tuple[object, ...],
    name: str,
    heads: int,
    geometry: tuple[int, ...],
    freqs: Tensor,
    chunks: Callable[[], Iterator[Tensor]],
) -> tuple[Tensor, bool]:
    prior_mean, prior_scale = previous_statistics(
        state, key, layout.step, geometry, name in REUSE, replay_safe=name in CP_NAMES
    )
    output, mean, scale = import_module("comfy_kitchen").sol_attn_chunked(
        chunks,
        layout.live_tokens,
        heads,
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
    return output, prior_mean is not None


def _project_qkv(
    attn: MiniMaxH3Attention,
    hidden: Tensor | PreQuantized,
    tokens: int,
    counts: dict[str, int] | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Project a local shard with one shared quantization and no QKV packing copy.

    Original module calls retain residency and Turbo hooks. The LoRA hook already
    owns its fixed-row arithmetic. Other projection semantics keep the prior chunk path.
    """
    if len(hidden.shape) != 3 or hidden.shape[0] != 1 or hidden.shape[1] != tokens or tokens < 1:
        raise ValueError("H3 CP projection requires the complete nonempty local token shard")
    if not isinstance(hidden, PreQuantized) and not shared_qkv_refusal(attn):
        import torch

        payload, scale = leaves.quantize_activation_rowwise(torch, hidden)
        operand = PreQuantized(payload, scale, tuple(hidden.shape))
        if counts is not None:
            counts["shared"] = counts.get("shared", 0) + 1
        query, key, value = (
            projection(hidden, prequantized=operand).reshape(1, tokens, attn.heads, 128)
            for projection in (attn.to_q, attn.to_k, attn.to_v)
        )
        return query, key, value
    packed = None
    offset = 0
    for chunk in qkv_chunks(attn, hidden, tokens, shared=True, counts=counts):
        if packed is None:
            packed = chunk.new_empty((tokens, chunk.shape[-1]))
        packed[offset : offset + chunk.shape[0]].copy_(chunk)
        offset += chunk.shape[0]
    if packed is None or offset != tokens:
        raise ValueError("H3 CP projection did not cover the local token shard")
    query, key, value = packed.reshape(1, tokens, 3, attn.heads, 128).unbind(2)
    return query, key, value


def _parallel_produce(
    attn: MiniMaxH3Attention,
    hidden: Tensor | PreQuantized,
    config: object,
    name: str,
    layout: AttentionLayout,
    state: RequestState,
    key: tuple[object, ...],
    counts: dict[str, int],
) -> tuple[Tensor, bool]:
    import torch
    import torch.distributed as dist

    handle = attention_ulysses.group(config)
    degree, rank = dist.get_world_size(handle), dist.get_rank(handle)
    tokens = layout.live_tokens
    expected = tokens // degree + int(rank < tokens % degree)
    if hidden.shape[1] != expected or attn.heads % degree:
        raise ValueError("H3 producer requires the actual unequal token shard and divisible heads")
    rotary_key = getattr(attn, "_cozy_producer_rotary_key", None)
    rotary = state.parallel_rotary.get(rotary_key)
    if rotary is None or rotary[0].shape[0] != tokens:
        raise ValueError("H3 CP producer requires this component's full pre-split rotary tables")
    freqs = rope_presentation(rotary, tokens, state)
    query, key_tensor, value = _project_qkv(attn, hidden, expected, counts)
    heads = attn.heads // degree
    reuse = False

    def compute(q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        nonlocal reuse

        def chunks() -> Iterator[Tensor]:
            for start in range(0, tokens, CHUNK):
                yield torch.cat(
                    [x[:, start : start + CHUNK].flatten(2, 3) for x in (q, k, v)], dim=-1
                ).squeeze(0)

        output, reuse = _produce(
            attn,
            layout,
            state,
            key,
            name,
            heads,
            (tokens, heads, layout.protected_prefix, degree, rank),
            freqs,
            chunks,
        )
        return output

    return attention_ulysses.exchange(query, key_tensor, value, handle, compute), reuse


def install_parallel_rotary(component: object) -> None:
    """Retain H3's replicated RoPE before Diffusers' subsequently installed split hook.

    The hook returns its output unchanged. Only an active sparse call using the
    opt-in CP recipe retains tensors, under the current request and component key.
    """
    try:
        from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3Transformer3DModel
    except ModuleNotFoundError as exc:
        if exc.name != "diffusers.models.transformers.transformer_minimax_h3":
            raise
        return  # H3 is optional; older Diffusers CP models keep their existing path.
    if not isinstance(component, MiniMaxH3Transformer3DModel):
        return
    from diffusers.hooks.hooks import HookRegistry, ModelHook

    rope = component.rope
    registry = HookRegistry.check_if_exists_or_initialize(rope)
    if registry.get_hook("cozy_cp_producer_rotary") is not None:
        return
    key = object()
    modules = tuple(block.attn for block in component.transformer_blocks)
    for module in modules:
        module._cozy_producer_rotary_key = key

    class RetainRotary(ModelHook):  # type: ignore[misc]  # optional untyped Diffusers hook API
        def post_forward(self, module: object, output: object) -> object:
            state, layout = _STATE.get(), _ACTIVE_LAYOUT.get()
            if state is None or layout is None or layout.step < layout.dense_until_step:
                return output
            backends = (getattr(m.processor, "_attention_backend", None) for m in modules)
            names = (BACKENDS.get(str(getattr(backend, "value", backend))) for backend in backends)
            active = any(name is not None and name in CP_NAMES for name in names)
            if active:
                if (
                    not isinstance(output, tuple)
                    or len(output) != 2
                    or output[0].shape[0] != layout.live_tokens
                ):
                    raise ValueError("H3 full rotary presentation changed before context splitting")
                state.parallel_rotary[key] = output
            return output

    registry.register_hook(RetainRotary(), "cozy_cp_producer_rotary")


def validate(module: object) -> None:
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
        # explicitly support replay/chunks. A model's LoRA hook adds a row-local update.
        for hook in projection._forward_hooks.values():
            if not isinstance(hook, LoRASource):
                raise ValueError("Kitchen producer cannot replay an unknown projection hook")


def install(module: MiniMaxH3Attention) -> None:
    validate(module)
    if not isinstance(module.processor, Processor):
        module.set_processor(Processor(module.processor))


def materialized_kitchen(
    query: Tensor, key: Tensor, value: Tensor, **options: Unpack[PartialOptions]
) -> Tensor:
    return attention_sol._attention(
        query,
        key,
        value,
        **options,
        sparse=_missing_processor,
        dense=attention_kitchen_sol.dense_kitchen,
        validate_device=attention_kitchen_sol.validate_device,
    )


def materialized_fp4_lowmem(
    query: Tensor, key: Tensor, value: Tensor, **options: Unpack[PartialOptions]
) -> Tensor:
    return attention_sol._attention(
        query,
        key,
        value,
        **options,
        sparse=_missing_processor,
        dense=attention_kitchen_sol.dense_fp4_lowmem,
        validate_device=attention_kitchen_sol.validate_device,
    )


def materialized_fp4_fixed_means(
    query: Tensor, key: Tensor, value: Tensor, **options: Unpack[PartialOptions]
) -> Tensor:
    return attention_sol._attention(
        query,
        key,
        value,
        **options,
        sparse=_missing_processor,
        dense=attention_kitchen_sol.dense_fp4_fixed_means,
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
FP4_FIXED_MEANS_LOCAL = Local(
    materialized_fp4_fixed_means,
    lambda *_: (
        "dense=sageattn3.nvfp4_fixed_means(upstream_bf16_correction); sparse=instance-producer"
    ),
    head_local=True,
)
LOCALS = {
    "kitchen-sol-producer-fp4-fixed-means-shared-qkv": FP4_FIXED_MEANS_LOCAL,
    "kitchen-sol-producer-fast-shared-qkv": FAST_LOCAL,
    "kitchen-sol-producer-fp4-lowmem-shared-qkv": FP4_LOWMEM_LOCAL,
}
