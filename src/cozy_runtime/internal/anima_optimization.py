"""Exact post-fill execution plan for Diffusers 0.40 Anima.

The package constructs and TensorFS fills the ordinary Diffusers module tree first.  Only
then does this module replace Python callables: parameters, buffers, state-dict names and
storage addresses are invariants.  This is intentionally one closed plan, not a generic
monkey-patching framework.
"""

from __future__ import annotations

import hashlib
import inspect
import json
from collections.abc import Callable
from dataclasses import dataclass
from importlib import import_module, metadata
from types import MethodType
from typing import Any

DIFFUSERS_VERSION = "0.40.0"
COMFY_KITCHEN_VERSION = "0.2.31"

_APPLICATION = "anima:app"
_MODEL_CLASS = "anima:AnimaModel"
_SOURCE_DIGESTS = {
    "RMSNorm.forward": "fa91ca74b4badaaf916ec4892e2b6ad90b4b904c7b570dba8ed99fd0ec69439b",
    "CosmosAttnProcessor2_0.__call__": (
        "5ec61c5c754278570e927dc2bfd0f8edb71afebfdd7b90e58e542142290b56d2"
    ),
    "CosmosRotaryPosEmbed.forward": (
        "ee752d88e85f73ccde5dac50c186bd8dd9382f2d1aea8c57db2571e86b8be0e1"
    ),
    "CosmosTransformerBlock.forward": (
        "c208749aa823185422d31daa08a40f374a86c13772a65179f5409f63a4703b0d"
    ),
    "AnimaPrepareLatentsStep.__call__": (
        "5f758c69ad2f82a37d707e3ca736b88011b41f04a958dbc657ec397b88c2bef5"
    ),
    "comfy_kitchen.rms_rope_split_half": (
        "6c1aca28177cd3ef9750e3130e741ab33d5d9d4eecbdd5fc7a6396028725603d"
    ),
}
_IDENTITY_DOCUMENT = {
    "diffusers": DIFFUSERS_VERSION,
    "comfy-kitchen": COMFY_KITCHEN_VERSION,
    "sources": _SOURCE_DIGESTS,
}
OPTIMIZATION_ID = (
    "sha256:"
    + hashlib.sha256(
        b"cozy.runtime.anima-cosmos-optimization\0"
        + json.dumps(_IDENTITY_DOCUMENT, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
)
_TRANSFORMER_CONFIG = {
    "_class_name": "CosmosTransformer3DModel",
    "_diffusers_version": "0.39.0.dev0",
    "adaln_lora_dim": 256,
    "attention_head_dim": 128,
    "concat_padding_mask": True,
    "controlnet_block_every_n": None,
    "crossattn_proj_in_channels": 1024,
    "encoder_hidden_states_channels": 1024,
    "extra_pos_embed_type": None,
    "img_context_dim_in": None,
    "img_context_dim_out": 2048,
    "img_context_num_tokens": 256,
    "in_channels": 16,
    "max_size": [128, 240, 240],
    "mlp_ratio": 4.0,
    "num_attention_heads": 16,
    "num_layers": 28,
    "out_channels": 16,
    "patch_size": [1, 2, 2],
    "rope_scale": [1.0, 4.0, 4.0],
    "text_embed_dim": 1024,
    "use_crossattn_projection": False,
}


class AnimaOptimizationRefusal(RuntimeError):
    """The named Anima plan was selected but its exact implementation did not match."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(detail)
        self.code = code


@dataclass(frozen=True, slots=True)
class AppliedAnimaOptimization:
    identity: str
    source: str
    operations: tuple[str, ...]

    def document(self) -> dict[str, object]:
        return {
            "identity": self.identity,
            "source": self.source,
            "operations": list(self.operations),
        }


def _refuse(code: str, detail: str) -> AnimaOptimizationRefusal:
    return AnimaOptimizationRefusal(code, detail)


def _source_digest(value: Callable[..., object]) -> str:
    return hashlib.sha256(inspect.getsource(value).encode()).hexdigest()


def _require_source(name: str, value: Callable[..., object]) -> None:
    actual = _source_digest(value)
    expected = _SOURCE_DIGESTS[name]
    if actual != expected:
        raise _refuse(
            "anima_optimization_source_mismatch",
            f"{name} has source sha256:{actual}, expected sha256:{expected}; refusing the "
            "Diffusers 0.40 execution plan rather than guessing across patched source",
        )


def _version(distribution: str, expected: str) -> None:
    try:
        actual = metadata.version(distribution)
    except metadata.PackageNotFoundError as exc:
        raise _refuse(
            "anima_optimization_dependency_absent",
            f"{distribution}=={expected} is required by the Anima execution plan",
        ) from exc
    if actual != expected:
        raise _refuse(
            "anima_optimization_dependency_mismatch",
            f"the Anima execution plan requires {distribution}=={expected}, found {actual}",
        )


def _type_name(value: object) -> str:
    cls = type(value)
    return f"{cls.__module__}:{cls.__qualname__}"


def _tensor_census(module: Any) -> tuple[tuple[object, ...], tuple[object, ...]]:
    topology = tuple(
        (name, tuple(value.shape), str(value.dtype), _type_name(value))
        for name, value in module.state_dict(keep_vars=True).items()
    )
    storage = tuple(
        (name, id(value), int(value.data_ptr()))
        for name, value in (
            *tuple(module.named_parameters(remove_duplicate=False)),
            *tuple(module.named_buffers(remove_duplicate=False)),
        )
    )
    return topology, storage


class _RopeMatrix:
    """One call-scoped conversion shared by all 28 self-attention blocks."""

    def __init__(self, torch: Any) -> None:
        self.torch = torch
        self.value: Any | None = None

    def clear(self) -> None:
        self.value = None

    def get(self, rotary: object) -> Any:
        if self.value is not None:
            return self.value
        if not isinstance(rotary, tuple) or len(rotary) != 2:
            raise _refuse(
                "anima_optimization_rope_layout",
                "Diffusers Cosmos did not provide its exact (cos, sin) rotary pair",
            )
        cos, sin = rotary
        if cos.ndim != 2 or sin.shape != cos.shape or cos.shape[-1] != 128:
            raise _refuse(
                "anima_optimization_rope_layout",
                f"Diffusers Cosmos rotary tensors have unsupported shapes {tuple(cos.shape)} "
                f"and {tuple(sin.shape)}",
            )
        half = cos.shape[-1] // 2
        # CosmosRotaryPosEmbed repeats the same half in source.  The source digest above is
        # the proof; doing a device->host equality read in every denoising call would erase
        # the kernel win this exact-source fence exists to make safe.
        cos, sin = cos[..., :half], sin[..., :half]
        self.value = self.torch.stack((cos, -sin, sin, cos), dim=-1).reshape(
            1, 1, cos.shape[0], half, 2, 2
        )
        return self.value


class _FusedSelfAttention:
    def __init__(self, torch: Any, kitchen: Any, rope: _RopeMatrix) -> None:
        self.torch = torch
        self.kitchen = kitchen
        self.rope = rope

    def __call__(
        self,
        attn: Any,
        hidden_states: Any,
        encoder_hidden_states: Any | None = None,
        attention_mask: Any | None = None,
        image_rotary_emb: Any | None = None,
    ) -> Any:
        from diffusers.models.attention_dispatch import dispatch_attention_fn

        if encoder_hidden_states is not None or attention_mask is not None:
            raise _refuse(
                "anima_optimization_attention_shape",
                "the fused Cosmos self-attention plan received cross-attention inputs",
            )
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)
        query = query.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        key = key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        value = value.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        if image_rotary_emb is None:
            raise _refuse(
                "anima_optimization_rope_absent",
                "the exact Anima Cosmos self-attention call omitted rotary embeddings",
            )
        query, key = self.kitchen.rms_rope_split_half(
            query,
            key,
            self.rope.get(image_rotary_emb),
            attn.norm_q.weight,
            attn.norm_k.weight,
            attn.norm_q.eps,
        )
        query_width, key_width, value_width = query.shape[3], key.shape[3], value.shape[3]
        key = key.repeat_interleave(query_width // key_width, dim=3)
        value = value.repeat_interleave(query_width // value_width, dim=3)
        output = dispatch_attention_fn(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
        )
        output = output.flatten(2, 3).type_as(query)
        return attn.to_out[1](attn.to_out[0](output))


def _native_rms_forward(module: Any, hidden_states: Any) -> Any:
    import torch.nn.functional as functional

    output = functional.rms_norm(hidden_states, module.dim, module.weight, module.eps)
    return output if module.bias is None else output + module.bias


def _block_forward(
    block: Any,
    hidden_states: Any,
    encoder_hidden_states: Any,
    embedded_timestep: Any,
    temb: Any | None = None,
    image_rotary_emb: Any | None = None,
    extra_pos_emb: Any | None = None,
    attention_mask: Any | None = None,
    controlnet_residual: Any | None = None,
    latents: Any | None = None,
    block_idx: int | None = None,
) -> Any:
    torch = hidden_states.__class__.__module__.split(".", 1)[0]
    if torch != "torch":
        raise _refuse("anima_optimization_tensor_type", "Cosmos received a non-torch tensor")
    import torch as torch_module

    if block.before_proj is not None:
        hidden_states = block.before_proj(hidden_states) + latents
    if extra_pos_emb is not None:
        hidden_states = hidden_states + extra_pos_emb
    normalized, gate = block.norm1(hidden_states, embedded_timestep, temb)
    output = block.attn1(normalized, image_rotary_emb=image_rotary_emb)
    hidden_states = torch_module.addcmul(hidden_states, gate, output)
    normalized, gate = block.norm2(hidden_states, embedded_timestep, temb)
    output = block.attn2(
        normalized,
        encoder_hidden_states=encoder_hidden_states,
        attention_mask=attention_mask,
    )
    hidden_states = torch_module.addcmul(hidden_states, gate, output)
    normalized, gate = block.norm3(hidden_states, embedded_timestep, temb)
    hidden_states = torch_module.addcmul(hidden_states, gate, block.ff(normalized))
    if controlnet_residual is not None:
        if block.after_proj is not None:
            raise _refuse(
                "anima_optimization_controlnet_shape",
                "Cosmos supplied a control residual together with after_proj",
            )
        hidden_states += controlnet_residual
    if block.after_proj is not None:
        return hidden_states, block.after_proj(hidden_states)
    return hidden_states


def _latent_prepare_call(step: Any, components: Any, state: Any) -> Any:
    import torch

    with torch.no_grad():
        block_state = step.get_block_state(state)
        block_state.height = block_state.height or components.default_height
        block_state.width = block_state.width or components.default_width
        step.check_inputs(components, block_state)
        block_state.latents = step.prepare_latents(
            batch_size=block_state.batch_size * block_state.num_images_per_prompt,
            num_channels_latents=components.num_channels_latents,
            height=block_state.height,
            width=block_state.width,
            vae_scale_factor=components.vae_scale_factor,
            dtype=torch.float32,
            device=components._execution_device,
            generator=block_state.generator,
            latents=block_state.latents,
        )
        block_state.padding_mask = block_state.latents.new_zeros(
            1,
            1,
            block_state.latents.shape[-2],
            block_state.latents.shape[-1],
            dtype=block_state.dtype,
        )
        step.set_block_state(state, block_state)
        return components, state


def _install_block_plan(block: Any, *, kitchen: Any, torch: Any, rope: _RopeMatrix) -> None:
    from diffusers.models.transformers.transformer_cosmos import CosmosAttnProcessor2_0

    if (
        block.before_proj is not None
        or block.after_proj is not None
        or type(block.attn1.processor) is not CosmosAttnProcessor2_0
        or type(block.attn2.processor) is not CosmosAttnProcessor2_0
    ):
        raise _refuse(
            "anima_optimization_block_shape",
            "Anima Cosmos block projections or attention processors do not match the plan",
        )
    block.attn1.set_processor(_FusedSelfAttention(torch, kitchen, rope))
    block.forward = MethodType(_block_forward, block)


def _install_transformer_plan(transformer: Any, *, kitchen: Any, torch: Any) -> None:
    from diffusers.models.normalization import RMSNorm
    from diffusers.models.transformers.transformer_cosmos import (
        CosmosTransformerBlock,
    )

    rope = _RopeMatrix(torch)
    blocks = tuple(transformer.transformer_blocks)
    if len(blocks) != 28 or any(type(block) is not CosmosTransformerBlock for block in blocks):
        raise _refuse(
            "anima_optimization_block_shape",
            "the exact Anima plan requires 28 unmodified Diffusers CosmosTransformerBlock rows",
        )
    norms = tuple(module for module in transformer.modules() if type(module) is RMSNorm)
    norm_shapes = sorted((tuple(norm.dim), norm.eps) for norm in norms)
    expected_norm_shapes = [((128,), 1e-5)] * 112 + [((2048,), 1e-6)]
    if norm_shapes != sorted(expected_norm_shapes):
        raise _refuse(
            "anima_optimization_norm_shape",
            "the exact Anima plan requires 112 head-width RMSNorms at eps=1e-5 and one "
            "model-width RMSNorm at eps=1e-6",
        )
    for norm in norms:
        if norm.weight is None or norm.bias is not None:
            raise _refuse(
                "anima_optimization_norm_shape",
                "Anima RMSNorm affine/bias/epsilon does not match the fused plan",
            )
        norm.forward = MethodType(_native_rms_forward, norm)
    for block in blocks:
        _install_block_plan(block, kitchen=kitchen, torch=torch, rope=rope)

    original_forward = transformer.forward

    def forward(_self: Any, *args: object, **kwargs: object) -> Any:
        rope.clear()
        try:
            return original_forward(*args, **kwargs)
        finally:
            rope.clear()

    transformer.forward = MethodType(forward, transformer)


def _require_transformer_config(transformer: Any) -> None:
    actual = json.loads(json.dumps(dict(transformer.config)))
    if actual != _TRANSFORMER_CONFIG:
        raise _refuse(
            "anima_optimization_config",
            "the filled Cosmos transformer config is not the exact Anima Base 1.0 config",
        )


def apply_anima_execution_plan(
    model: object, *, application: str, model_class: str
) -> AppliedAnimaOptimization | None:
    """Install the one exact plan after fill; return ``None`` for every other package."""
    if application != _APPLICATION or model_class != _MODEL_CLASS:
        return None

    _version("diffusers", DIFFUSERS_VERSION)
    _version("comfy-kitchen", COMFY_KITCHEN_VERSION)
    import torch
    from diffusers.models.normalization import RMSNorm
    from diffusers.models.transformers.transformer_cosmos import (
        CosmosAttnProcessor2_0,
        CosmosRotaryPosEmbed,
        CosmosTransformer3DModel,
        CosmosTransformerBlock,
    )
    from diffusers.modular_pipelines.anima.before_denoise import AnimaPrepareLatentsStep

    comfy_kitchen = import_module("comfy_kitchen")
    sources: dict[str, Callable[..., object]] = {
        "RMSNorm.forward": RMSNorm.forward,
        "CosmosAttnProcessor2_0.__call__": CosmosAttnProcessor2_0.__call__,
        "CosmosRotaryPosEmbed.forward": CosmosRotaryPosEmbed.forward,
        "CosmosTransformerBlock.forward": CosmosTransformerBlock.forward,
        "AnimaPrepareLatentsStep.__call__": AnimaPrepareLatentsStep.__call__,
        "comfy_kitchen.rms_rope_split_half": comfy_kitchen.rms_rope_split_half,
    }
    for name, source in sources.items():
        _require_source(name, source)

    if _type_name(model) != _MODEL_CLASS:
        raise _refuse(
            "anima_optimization_model_class",
            f"the binding names {_MODEL_CLASS}, but construction returned {_type_name(model)}",
        )
    pipe = getattr(model, "pipe", None)
    if _type_name(pipe) != "anima:AnimaPipeline":
        raise _refuse(
            "anima_optimization_pipeline_class",
            f"the exact Anima pipeline is anima:AnimaPipeline, found {_type_name(pipe)}",
        )
    components = getattr(pipe, "components", None)
    if not isinstance(components, dict) or set(components) != {
        "transformer",
        "text_encoder",
        "text_conditioner",
        "vae",
    }:
        raise _refuse(
            "anima_optimization_components",
            "Anima pipeline components do not match transformer/text_encoder/text_conditioner/vae",
        )
    transformer = components["transformer"]
    if type(transformer) is not CosmosTransformer3DModel:
        raise _refuse(
            "anima_optimization_transformer_class",
            "the exact transformer is Diffusers CosmosTransformer3DModel, found "
            f"{_type_name(transformer)}",
        )
    _require_transformer_config(transformer)

    before = _tensor_census(transformer)
    _install_transformer_plan(transformer, kitchen=comfy_kitchen, torch=torch)
    after = _tensor_census(transformer)
    if after != before:
        raise _refuse(
            "anima_optimization_topology_changed",
            "the post-fill plan changed state-dict topology or parameter/buffer storage identity",
        )
    AnimaPrepareLatentsStep.__call__ = _latent_prepare_call
    return AppliedAnimaOptimization(
        OPTIMIZATION_ID,
        f"diffusers=={DIFFUSERS_VERSION};comfy-kitchen=={COMFY_KITCHEN_VERSION}",
        (
            "fused_rmsnorm_split_half_rope",
            "addcmul_residuals",
            "latent_sized_zero_padding_mask",
            "native_rms_norm",
        ),
    )
