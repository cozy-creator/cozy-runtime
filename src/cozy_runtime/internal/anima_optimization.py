"""Reviewed post-fill execution plan for Diffusers Anima.

The package constructs and TensorFS fills the ordinary Diffusers module tree first.  Only
then does this module replace Python callables: parameters, buffers, state-dict names and
storage addresses are invariants.  This is intentionally one closed plan, not a generic
monkey-patching framework.

``torch.compile`` is a qualification-time discovery oracle, never an invocation-time path.
Reusable fusion belongs in Runtime's own in-package CUDA kernels; eager stays available.
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

from packaging.version import InvalidVersion, Version

from cozy_runtime.internal.accel import readable

DIFFUSERS_VERSION = "0.40.0"
# The kernels ride cozy-runtime's own cp312-abi3 wheel (cr-094): the module is probed,
# never version-pinned. The revision names the native sources' review state and is bumped
# whenever `native/` changes, so optimization identity tracks the kernels, not releases.
FUSED_ANIMA_CUDA_KERNEL_MODULE = "cozy_runtime._kernels"
FUSED_ANIMA_CUDA_KERNEL_REVISION = "0.1.0"
# This extension consumes DLPack arrays and a CUDA stream through nanobind; it
# neither links libtorch nor depends on Torch's C++ ABI. Keep an explicit Python
# API/numerical cohort rather than admitting every future Torch release.
FUSED_ANIMA_TORCH_RELEASES = ("2.13.0", "2.14.0")
FUSED_ANIMA_CUDA_RELEASE = "13.0"

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
    "cozy_runtime._kernels.rms_rope_split_half": (
        "084c60d95676ae003e285fb57ec6b9c3416ffba6e0a0f62d2097f26b8c8aa999"
    ),
}
_IDENTITY_DOCUMENT = {
    "diffusers": DIFFUSERS_VERSION,
    FUSED_ANIMA_CUDA_KERNEL_MODULE: FUSED_ANIMA_CUDA_KERNEL_REVISION,
    "plan": "direct-split-half-rope-matrix",
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


def _source_refusal(name: str, value: Callable[..., object]) -> str:
    actual = _source_digest(value)
    expected = _SOURCE_DIGESTS[name]
    if actual == expected:
        return ""
    return f"{name} has source sha256:{actual}, not the reviewed sha256:{expected}"


def _installed_version(distribution: str) -> str | None:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return None


def _type_name(value: object) -> str:
    cls = type(value)
    return f"{cls.__module__}:{cls.__qualname__}"


def _tensor_census(module: Any) -> tuple[tuple[object, ...], tuple[object, ...]]:
    tensor_schema = tuple(
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
    return tensor_schema, storage


class _RopeMatrix:
    """One call-scoped conversion shared by all 28 self-attention blocks."""

    def __init__(self, torch: Any) -> None:
        self.torch = torch
        self.value: Any | None = None

    def clear(self) -> None:
        self.value = None

    def get(self, rotary: Any) -> Any:
        if self.value is not None:
            return self.value
        if hasattr(rotary, "shape"):
            if rotary.ndim != 6 or tuple(rotary.shape[-3:]) != (64, 2, 2):
                raise _refuse(
                    "anima_optimization_rope_layout",
                    f"direct Cosmos rotary matrix has unsupported shape {tuple(rotary.shape)}",
                )
            return rotary
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


def _direct_rope_forward(rope: Any, hidden_states: Any, fps: int | None = None) -> Any:
    """Produce the fused Anima CUDA kernel matrix without duplicate cosine/sine tensors."""
    import torch

    _, _, frames, height, width = hidden_states.shape
    sizes = [
        frames // rope.patch_size[0],
        height // rope.patch_size[1],
        width // rope.patch_size[2],
    ]
    device = hidden_states.device
    seq = torch.arange(max(rope.max_size), device=device, dtype=torch.float32)
    dim_h = (
        torch.arange(0, rope.dim_h, 2, device=device, dtype=torch.float32)[: rope.dim_h // 2]
        / rope.dim_h
    )
    dim_w = (
        torch.arange(0, rope.dim_w, 2, device=device, dtype=torch.float32)[: rope.dim_w // 2]
        / rope.dim_w
    )
    dim_t = (
        torch.arange(0, rope.dim_t, 2, device=device, dtype=torch.float32)[: rope.dim_t // 2]
        / rope.dim_t
    )
    freq_h = 1.0 / ((10000.0 * rope.h_ntk_factor) ** dim_h)
    freq_w = 1.0 / ((10000.0 * rope.w_ntk_factor) ** dim_w)
    freq_t = 1.0 / ((10000.0 * rope.t_ntk_factor) ** dim_t)
    emb_h = torch.outer(seq[: sizes[1]], freq_h)[None, :, None, :].repeat(sizes[0], 1, sizes[2], 1)
    emb_w = torch.outer(seq[: sizes[2]], freq_w)[None, None, :, :].repeat(sizes[0], sizes[1], 1, 1)
    time = seq[: sizes[0]] if fps is None else seq[: sizes[0]] / fps * rope.base_fps
    emb_t = torch.outer(time, freq_t)[:, None, None, :].repeat(1, sizes[1], sizes[2], 1)
    phase = torch.cat((emb_t, emb_h, emb_w), dim=-1).flatten(0, 2).float()
    cos, sin = torch.cos(phase), torch.sin(phase)
    return torch.stack((cos, -sin, sin, cos), dim=-1).reshape(
        1, 1, phase.shape[0], phase.shape[1], 2, 2
    )


class _FusedSelfAttention:
    def __init__(self, torch: Any, anima_kernels: Any, rope: _RopeMatrix) -> None:
        self.torch = torch
        self.anima_kernels = anima_kernels
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
        # The kernel reads the scales by pointer, without calling the norms that own them.
        with readable(attn.norm_q.weight, attn.norm_k.weight) as (q_scale, k_scale):
            query, key = self.anima_kernels.rms_rope_split_half(
                query,
                key,
                self.rope.get(image_rotary_emb),
                q_scale,
                k_scale,
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


def _install_block_plan(block: Any, *, anima_kernels: Any, torch: Any, rope: _RopeMatrix) -> None:
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
    block.attn1.set_processor(_FusedSelfAttention(torch, anima_kernels, rope))
    block.forward = MethodType(_block_forward, block)


def _install_transformer_plan(transformer: Any, *, anima_kernels: Any, torch: Any) -> None:
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
        _install_block_plan(block, anima_kernels=anima_kernels, torch=torch, rope=rope)
    transformer.rope.forward = MethodType(_direct_rope_forward, transformer.rope)

    original_forward = transformer.forward

    def forward(_self: Any, *args: object, **kwargs: object) -> Any:
        rope.clear()
        try:
            return original_forward(*args, **kwargs)
        finally:
            rope.clear()

    transformer.forward = MethodType(forward, transformer)


def _capability() -> tuple[Any, Any] | str:
    """The supported Torch/native pair, or why this generation runs eager."""
    torch_version = _installed_version("torch")
    if torch_version is None:
        return "torch is not installed"
    try:
        installed = Version(torch_version)
    except InvalidVersion:
        return f"torch version {torch_version!r} does not parse"
    if installed.public not in FUSED_ANIMA_TORCH_RELEASES:
        return (
            f"torch {installed.public} is outside the fused lane's supported releases "
            f"{', '.join(FUSED_ANIMA_TORCH_RELEASES)}"
        )
    try:
        torch = import_module("torch")
    except ImportError:
        return "torch does not import"
    if getattr(getattr(torch, "version", None), "cuda", None) != FUSED_ANIMA_CUDA_RELEASE:
        return (
            f"torch.version.cuda {getattr(getattr(torch, 'version', None), 'cuda', None)!r} "
            f"!= the fused lane's {FUSED_ANIMA_CUDA_RELEASE!r}"
        )
    cuda = getattr(torch, "cuda", None)
    try:
        if cuda is None or not cuda.is_available():
            return "torch reports no available CUDA device"
    except (AttributeError, RuntimeError) as exc:
        return f"torch.cuda.is_available() raised {type(exc).__name__}"
    try:
        import_module(FUSED_ANIMA_CUDA_KERNEL_MODULE + "._C")
        anima_kernels = import_module(FUSED_ANIMA_CUDA_KERNEL_MODULE)
    except ImportError as exc:
        return (
            f"{FUSED_ANIMA_CUDA_KERNEL_MODULE}._C does not import ({exc}) — this wheel "
            "carries no compiled kernels, so this generation runs the eager path"
        )
    return torch, anima_kernels


def _plan_refusal(model: object, anima_kernels: Any) -> str:
    """Why this construction is not the reviewed Anima graph, or "" when it is."""
    try:
        from diffusers.models.normalization import RMSNorm
        from diffusers.models.transformers.transformer_cosmos import (
            CosmosAttnProcessor2_0,
            CosmosRotaryPosEmbed,
            CosmosTransformer3DModel,
            CosmosTransformerBlock,
        )
        from diffusers.modular_pipelines.anima.before_denoise import AnimaPrepareLatentsStep
    except ImportError as exc:
        return f"diffusers does not expose the reviewed Anima modules: {exc}"
    sources: dict[str, Callable[..., object]] = {
        "RMSNorm.forward": RMSNorm.forward,
        "CosmosAttnProcessor2_0.__call__": CosmosAttnProcessor2_0.__call__,
        "CosmosRotaryPosEmbed.forward": CosmosRotaryPosEmbed.forward,
        "CosmosTransformerBlock.forward": CosmosTransformerBlock.forward,
        "AnimaPrepareLatentsStep.__call__": AnimaPrepareLatentsStep.__call__,
        "cozy_runtime._kernels.rms_rope_split_half": anima_kernels.rms_rope_split_half,
    }
    for name, source in sources.items():
        refusal = _source_refusal(name, source)
        if refusal:
            return refusal
    if _type_name(model) != _MODEL_CLASS:
        return f"construction returned {_type_name(model)}, not {_MODEL_CLASS}"
    pipe = getattr(model, "pipe", None)
    if _type_name(pipe) != "anima:AnimaPipeline":
        return f"the pipeline is {_type_name(pipe)}, not anima:AnimaPipeline"
    components = getattr(pipe, "components", None)
    if not isinstance(components, dict) or set(components) != {
        "transformer",
        "text_encoder",
        "text_conditioner",
        "vae",
    }:
        return "Anima pipeline components are not transformer/text_encoder/text_conditioner/vae"
    transformer = components["transformer"]
    if type(transformer) is not CosmosTransformer3DModel:
        return f"the transformer is {_type_name(transformer)}, not CosmosTransformer3DModel"
    if json.loads(json.dumps(dict(transformer.config))) != _TRANSFORMER_CONFIG:
        return "the filled Cosmos transformer config is not the Anima Base 1.0 config"
    return ""


def _apply_fused_anima_cuda_plan(
    model: object, *, torch: Any, anima_kernels: Any
) -> AppliedAnimaOptimization:
    """Install the plan on a construction `_plan_refusal` admitted; storage never moves."""
    from diffusers.modular_pipelines.anima.before_denoise import AnimaPrepareLatentsStep

    transformer = model.pipe.components["transformer"]  # type: ignore[attr-defined]
    before = _tensor_census(transformer)
    _install_transformer_plan(transformer, anima_kernels=anima_kernels, torch=torch)
    after = _tensor_census(transformer)
    if after != before:
        raise _refuse(
            "anima_optimization_tensor_schema_changed",
            "the post-fill plan changed the tensor schema or parameter/buffer storage identity",
        )
    AnimaPrepareLatentsStep.__call__ = _latent_prepare_call
    return AppliedAnimaOptimization(
        OPTIMIZATION_ID,
        f"diffusers=={_installed_version('diffusers')};"
        f"{FUSED_ANIMA_CUDA_KERNEL_MODULE}=={FUSED_ANIMA_CUDA_KERNEL_REVISION}",
        (
            "fused_rmsnorm_split_half_rope",
            "addcmul_residuals",
            "latent_sized_zero_padding_mask",
            "native_rms_norm",
            "direct_split_half_rope_matrix",
        ),
    )


def apply_anima_execution_plan(
    model: object, *, application: str, model_class: str
) -> tuple[AppliedAnimaOptimization | None, str]:
    """Apply fused Anima CUDA kernels when supported; otherwise run eager and say why."""
    if application != _APPLICATION or model_class != _MODEL_CLASS:
        return None, ""
    capability = _capability()
    if isinstance(capability, str):
        return None, capability
    torch, anima_kernels = capability
    refusal = _plan_refusal(model, anima_kernels)
    if refusal:
        return None, refusal
    return _apply_fused_anima_cuda_plan(model, torch=torch, anima_kernels=anima_kernels), ""
