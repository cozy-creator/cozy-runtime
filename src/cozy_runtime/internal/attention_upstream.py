"""Runtime's local attention kernels; the kernels and their numerics are upstream.

Every adapter is forward-only over Diffusers' [batch, sequence, heads, dimension] layout,
validates the operands its kernel serves and calls ONE named upstream entry, so the mode that
ran is the mode recorded. `attention_ulysses` makes each of them context-parallel. Upstream
libraries are imported on use: they live in the image's kernel site, not in Runtime's
dependencies, and the executor imports torch only after its environment seal.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Collection
from importlib import import_module
from types import ModuleType
from typing import Any

from cozy_runtime.internal import attention_fp8
from cozy_runtime.internal.attention_ulysses import Local, Options

# Tensors and torch's SDPA backend enum are `Any`: torch is not installed in the check venv.

#: The loaded FlashAttention-3 snapshot (`kernels` loads it under a generated module name)
#: and the build variant it came from.
_FA3: ModuleType | None = None
_FA3_VARIANT = ""


def bind_fa3(module: ModuleType, variant: str) -> None:
    global _FA3, _FA3_VARIANT
    _FA3, _FA3_VARIANT = module, variant


def _options(
    name: str, attn_mask: Any, dropout_p: float, scale: float | None, *, masks: bool = False
) -> None:
    if dropout_p != 0.0:
        raise ValueError(f"{name} is an inference kernel and does not support dropout")
    if attn_mask is not None and not masks:
        raise ValueError(f"{name} does not support an attention mask")
    if scale is not None and not math.isfinite(scale):
        raise ValueError("attention scale must be finite")


def _heads(query: Any, key: Any, enable_gqa: bool) -> None:
    if query.shape[2] != key.shape[2] and not enable_gqa:
        raise ValueError("unequal Q/KV head counts require enable_gqa=True")


def _output(output: Any, query: Any) -> Any:
    import torch

    if (
        not isinstance(output, torch.Tensor)
        or output.shape != query.shape
        or output.dtype != query.dtype
        or output.device != query.device
    ):
        raise ValueError("upstream attention must return the query shape, dtype and device")
    return output


def _first(output: Any) -> Any:
    return output[0] if isinstance(output, (tuple, list)) else output


# ------------------------------------------------------------------ torch's own dispatch


def sdpa(
    query: Any,
    key: Any,
    value: Any,
    *,
    attn_mask: Any = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: float | None = None,
    enable_gqa: bool = False,
) -> Any:
    """`scaled_dot_product_attention` with Diffusers' native semantics, 2D key masks included."""
    import torch

    if (
        attn_mask is not None
        and attn_mask.ndim == 2
        and attn_mask.shape[0] == query.shape[0]
        and attn_mask.shape[1] == key.shape[1]
    ):
        attn_mask = attn_mask[:, None, None, :]
    q, k, v = (tensor.transpose(1, 2) for tensor in (query, key, value))
    return torch.nn.functional.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale,
        enable_gqa=enable_gqa,
    ).transpose(1, 2)


def sdpa_impl(query: Any, key: Any, value: Any, options: Options) -> str:
    """The backend torch's SDPA dispatcher picks for exactly these operands."""
    import torch
    from torch.nn.attention import SDPBackend

    q, k, v = (tensor.transpose(1, 2) for tensor in (query, key, value))
    mask = options["attn_mask"]
    if mask is not None and mask.ndim == 2:
        mask = mask[:, None, None, :]
    choice = torch._fused_sdp_choice(
        q,
        k,
        v,
        attn_mask=mask,
        dropout_p=options["dropout_p"],
        is_causal=options["is_causal"],
        scale=options["scale"],
        enable_gqa=options["enable_gqa"],
    )
    backend = SDPBackend(choice)
    names: dict[Any, str] = {
        SDPBackend.CUDNN_ATTENTION: "cudnn",
        SDPBackend.FLASH_ATTENTION: "flash",
        SDPBackend.EFFICIENT_ATTENTION: "efficient",
        SDPBackend.MATH: "math",
    }
    return names.get(backend, str(backend.name).lower())


def cudnn(
    query: Any,
    key: Any,
    value: Any,
    *,
    attn_mask: Any = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: float | None = None,
    enable_gqa: bool = False,
) -> Any:
    """SDPA restricted to torch's cuDNN backend: an identity check of `sdpa` on Hopper."""
    from torch.nn.attention import SDPBackend, sdpa_kernel

    with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
        return sdpa(
            query.contiguous(),
            key.contiguous(),
            value.contiguous(),
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
            scale=scale,
            enable_gqa=enable_gqa,
        )


# ------------------------------------------------------------------- FlashAttention 3 and 4


def _fa3() -> ModuleType:
    if _FA3 is None:
        raise RuntimeError("flash-attn3 was dispatched before its snapshot was loaded")
    return _FA3


def fa3(
    query: Any,
    key: Any,
    value: Any,
    *,
    attn_mask: Any = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: float | None = None,
    enable_gqa: bool = False,
) -> Any:
    """FlashAttention-3 BF16/FP16 forward, one KV split, as Diffusers' FA3 entry calls it."""
    _options("flash-attn3", attn_mask, dropout_p, scale)
    _heads(query, key, enable_gqa)
    return _first(
        _fa3().flash_attn_func(
            query, key, value, softmax_scale=scale, causal=is_causal, num_splits=1
        )
    )


def fa3_fp8(
    query: Any,
    key: Any,
    value: Any,
    *,
    attn_mask: Any = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: float | None = None,
    enable_gqa: bool = False,
) -> Any:
    """FA3's e4m3 forward from the same snapshot: per-(batch, head) amax quantisation per
    call, the descales handed to the kernel, BF16/FP16 output from the kernel itself. Every
    scale reduces within one head, so it is head-local under Ulysses."""
    _options("flash-attn3-fp8", attn_mask, dropout_p, scale)
    q8, k8, v8, descales = _e4m3(query, key, value)
    return _first(
        _fa3().flash_attn_func(
            q8, k8, v8, softmax_scale=scale, causal=is_causal, num_splits=1, **descales
        )
    )


def _e4m3(query: Any, key: Any, value: Any) -> tuple[Any, Any, Any, dict[str, Any]]:
    """Per-(batch, head) e4m3 Q/K/V and the `*_descale` keywords FA3 and FA4 take. Their
    descales are per KV head, so a grouped query (one scale per Q head) is refused."""
    if query.shape[2] != key.shape[2]:
        raise ValueError("FP8 FlashAttention needs equal Q/KV head counts")
    (q8, q), (k8, k), (v8, v) = (attention_fp8.quantise(t) for t in (query, key, value))
    return q8, k8, v8, {"q_descale": q, "k_descale": k, "v_descale": v}


def fa4(
    query: Any,
    key: Any,
    value: Any,
    *,
    attn_mask: Any = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: float | None = None,
    enable_gqa: bool = False,
) -> Any:
    """FlashAttention-4's CuTe-DSL forward (`flash_attn.cute`); it compiles for the card on
    first use, which preparation's probe absorbs."""
    _options("flash-attn4", attn_mask, dropout_p, scale)
    _heads(query, key, enable_gqa)
    cute = import_module("flash_attn.cute")
    return _first(cute.flash_attn_func(query, key, value, softmax_scale=scale, causal=is_causal))


def fa4_fp8(
    query: Any,
    key: Any,
    value: Any,
    *,
    attn_mask: Any = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: float | None = None,
    enable_gqa: bool = False,
) -> Any:
    """FlashAttention-4's SM100 e4m3 forward with FA3's per-(batch, head) descales; the kernel
    returns BF16. The public `flash_attn_func` takes no descales, so this calls the forward it
    wraps, as upstream's own FP8 benchmark does. Every scale is per head: head-local."""
    import torch

    _options("flash-attn4-fp8", attn_mask, dropout_p, scale)
    if query.dtype != torch.bfloat16:
        raise ValueError("flash-attn4-fp8 returns BF16 and requires BF16 inputs")
    q8, k8, v8, descales = _e4m3(query, key, value)
    interface = import_module("flash_attn.cute.interface")
    return _first(
        interface._flash_attn_fwd(q8, k8, v8, softmax_scale=scale, causal=is_causal, **descales)
    )


# --------------------------------------------------------------------- quantized kernels


def _sage(modes: dict[int, tuple[str, str, str]]) -> Local:
    """SageAttention2 through one named entry per SM: `sm -> (entry, qk_quant_gran,
    pv_accum_dtype)`, upstream's own per-architecture choice, selected by the query's card.
    INT8 Q/K with smoothed K; `entry` fixes the P/V precision. Its scales reduce within one
    head."""

    def mode(query: Any) -> tuple[str, str, str]:
        import torch

        if query.device.type != "cuda":
            raise ValueError("sageattention serves CUDA devices only")
        major, minor = torch.cuda.get_device_capability(query.device)
        found = modes.get(major * 10 + minor)
        if found is None:
            served = sorted(modes)
            raise ValueError(f"sageattention has no mode for sm{major}{minor}; it serves {served}")
        return found

    def forward(
        query: Any,
        key: Any,
        value: Any,
        *,
        attn_mask: Any = None,
        dropout_p: float = 0.0,
        is_causal: bool = False,
        scale: float | None = None,
        enable_gqa: bool = False,
    ) -> Any:
        _options("sageattention", attn_mask, dropout_p, scale)
        _heads(query, key, enable_gqa)
        entry, granularity, accumulation = mode(query)
        return getattr(import_module("sageattention"), entry)(
            query,
            key,
            value,
            tensor_layout="NHD",
            is_causal=is_causal,
            qk_quant_gran=granularity,
            sm_scale=scale,
            pv_accum_dtype=accumulation,
            smooth_k=True,
        )

    def impl(query: Any, *_: Any) -> str:
        entry, granularity, accumulation = mode(query)
        return f"{entry}({granularity}, {accumulation}, smooth_k)"

    return Local(forward, impl)


#: Kitchen's wheel carries device code for sm_75 up; these are the cards Runtime admits.
KITCHEN_SMS = ((9, 0), (10, 0), (10, 3), (12, 0), (12, 1))


def _cuda_inputs(
    query: Any,
    key: Any,
    value: Any,
    enable_gqa: bool,
    *,
    name: str,
    admitted: tuple[tuple[int, int], ...],
    dims: Collection[int] = range(1, 257),
    bf16: bool = False,
) -> None:
    import torch

    if any(not isinstance(x, torch.Tensor) for x in (query, key, value)):
        raise ValueError("attention Q/K/V must be tensors")
    if any(x.ndim != 4 for x in (query, key, value)):
        raise ValueError("attention Q/K/V must have shape [batch, sequence, heads, dimension]")
    if query.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError("attention Q/K/V must be BF16 or FP16")
    if key.dtype != query.dtype or value.dtype != query.dtype:
        raise ValueError("attention Q/K/V must have identical input dtypes")
    if bf16 and query.dtype != torch.bfloat16:
        raise ValueError(f"{name} preserves BF16 Q/K and requires BF16 inputs")
    batch, _sequence, heads, dim = query.shape
    if min(query.shape) <= 0 or min(key.shape) <= 0 or min(value.shape) <= 0:
        raise ValueError("attention tensor dimensions must be positive")
    if (
        key.shape != value.shape
        or key.shape[0] != batch
        or key.shape[3] != dim
        or heads % key.shape[2]
    ):
        raise ValueError("attention needs matching K/V and compatible batch/head dimensions")
    _heads(query, key, enable_gqa)
    if dim not in dims:
        raise ValueError(f"{name} does not serve head dimension {dim}")
    if query.device.type != "cuda" or key.device != query.device or value.device != query.device:
        raise ValueError("attention Q/K/V must be on the same CUDA device")
    if getattr(torch.version, "hip", None):
        raise ValueError("these attention adapters currently admit NVIDIA CUDA devices only")
    capability = torch.cuda.get_device_capability(query.device)
    if capability not in admitted:
        raise ValueError(f"{name} admits {admitted}, not {capability}")
    if torch.is_grad_enabled() and any(x.requires_grad for x in (query, key, value)):
        raise ValueError("this attention adapter is inference-only")


def kitchen_int8(
    query: Any,
    key: Any,
    value: Any,
    *,
    attn_mask: Any = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: float | None = None,
    enable_gqa: bool = False,
) -> Any:
    """Comfy Kitchen INT8 Q/K/V with unsigned INT8 P and FP32 softmax; masks preserved.
    Rotation, key shift and scales are per batch/head, so it is head-local under Ulysses."""
    if is_causal:
        raise ValueError("kitchen-int8 serves noncausal attention only")
    _options("kitchen-int8", attn_mask, dropout_p, scale, masks=True)
    _cuda_inputs(query, key, value, enable_gqa, name="kitchen-int8", admitted=KITCHEN_SMS)
    kitchen = import_module("comfy_kitchen")
    tensors = [x if x.stride(-1) == 1 else x.contiguous() for x in (query, key, value)]
    output = kitchen.int8_attention(
        *(x.transpose(1, 2) for x in tensors), scale=scale, attn_mask=attn_mask
    )
    return _output(output, query.transpose(1, 2)).transpose(1, 2)


def flashinfer_mixed(
    query: Any,
    key: Any,
    value: Any,
    *,
    attn_mask: Any = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: float | None = None,
    enable_gqa: bool = False,
) -> Any:
    """Blackwell CuTe FMHA: BF16 Q/K, E4M3 V/P, BF16 output. Its V scale is one scalar over
    every head, so V is quantised per (batch, head), the kernel's scale stays 1 and each head's
    descale multiplies that head's output (attention is linear in V): head-local, and no host
    synchronisation. The kernel is compiled from the installed source (CuTe JIT) and handed
    in, so the library never fetches its prebuilt artifacts at first use."""
    if is_causal:
        raise ValueError("flashinfer-bf16-fp8 serves noncausal attention only")
    _options("flashinfer-bf16-fp8", attn_mask, dropout_p, scale)
    _cuda_inputs(
        query,
        key,
        value,
        enable_gqa,
        name="flashinfer-bf16-fp8",
        admitted=((10, 0), (10, 3)),
        dims=(128,),
        bf16=True,
    )
    import torch

    batch, q_length, q_heads, dim = query.shape
    kv_length, kv_heads = key.shape[1:3]
    if batch * max(q_length, kv_length) >= 2**31:
        raise ValueError("FlashInfer ragged sequence offsets exceed int32")
    compile_fmha = import_module("flashinfer.cute_dsl.attention.fmha.compile")
    kernel = compile_fmha.compile_cute_dsl_fmha_kernel(
        qk_dtype=torch.bfloat16,
        v_dtype=torch.float8_e4m3fn,
        out_dtype=torch.bfloat16,
        num_qo_heads=q_heads,
        num_kv_heads=kv_heads,
        head_dim=dim,
        head_dim_v=dim,
        is_causal=False,
        with_lse=False,
        enable_sink=False,
        enable_skip_softmax=False,
        use_pdl=False,
        device=query.device,
    )
    v8, descale = attention_fp8.quantise(value.contiguous())
    q = query.reshape(batch * q_length, q_heads, dim).contiguous()
    output = torch.empty_like(q)
    offsets = torch.arange(batch + 1, device=query.device, dtype=torch.int32)
    import_module("flashinfer.attention.cute_dsl.fmha").cute_dsl_fmha_ragged_prefill(
        q,
        key.reshape(batch * kv_length, kv_heads, dim).contiguous(),
        v8.reshape(batch * kv_length, kv_heads, dim),
        output,
        offsets * q_length,
        offsets * kv_length,
        is_causal=False,
        sm_scale=scale,
        max_qo_len=q_length,
        max_kv_len=kv_length,
        kernel_fn=kernel,
        enable_tvm_ffi=True,
    )
    per_head = descale.repeat_interleave(q_heads // kv_heads, dim=1)[:, None, :, None]
    return _output(output.reshape(batch, q_length, q_heads, dim).mul_(per_head), query)


def sage3_fp4(
    query: Any,
    key: Any,
    value: Any,
    *,
    attn_mask: Any = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: float | None = None,
    enable_gqa: bool = False,
) -> Any:
    """SageAttention3's consumer-Blackwell FP4 kernel: NVFP4 Q/K/V with per-16 E4M3
    microscales, K mean and per-block Q mean smoothing. Every mean and scale is per head, so
    it is head-local. Upstream subtracts K's mean IN PLACE, so it gets a copy, and its softmax
    scale is fixed at D^-1/2."""
    import torch

    _options("sageattention3-fp4", attn_mask, dropout_p, scale)
    admitted = ((12, 0),)
    _cuda_inputs(
        query, key, value, enable_gqa, name="sageattention3-fp4", admitted=admitted, dims=(64, 128)
    )
    if query.shape[2] != key.shape[2]:
        raise ValueError("sageattention3-fp4 needs equal Q/KV head counts")
    if scale is not None and not math.isclose(scale, query.shape[-1] ** -0.5, rel_tol=1e-6):
        raise ValueError("sageattention3-fp4 fixes the softmax scale at D^-1/2")
    q, k, v = (tensor.transpose(1, 2) for tensor in (query, key, value))
    output = import_module("sageattn3").sageattn3_blackwell(
        q, k.clone(memory_format=torch.contiguous_format), v, is_causal=is_causal
    )
    return _output(output, q).transpose(1, 2)


def _named(name: str) -> Callable[..., str]:
    return lambda *_: name


SDPA = Local(sdpa, sdpa_impl)
CUDNN = Local(cudnn, _named("cudnn"))
FA3 = Local(fa3, lambda *_: f"flash_attn3 {_FA3_VARIANT}")
FA3_FP8 = Local(fa3_fp8, lambda *_: f"flash_attn3 {_FA3_VARIANT} e4m3 per-head")
FA4 = Local(fa4, _named("flash_attn.cute.flash_attn_func"))
FA4_FP8 = Local(fa4_fp8, _named("flash_attn.cute.interface._flash_attn_fwd e4m3 per-head"))
#: FP8 P/V, as upstream's `sageattn` dispatches it: the Hopper kernel (two-level FP32
#: accumulation only) and the SM89 kernel, per warp on consumer Blackwell.
SAGE = _sage(
    {
        89: ("sageattn_qk_int8_pv_fp8_cuda", "per_thread", "fp32+fp16"),
        90: ("sageattn_qk_int8_pv_fp8_cuda_sm90", "per_thread", "fp32+fp32"),
        120: ("sageattn_qk_int8_pv_fp8_cuda", "per_warp", "fp32+fp16"),
    }
)
#: FP16 P/V at upstream's default (and most accurate) FP32 accumulation; BF16 V is cast to FP16.
SAGE_FP16PV = _sage({90: ("sageattn_qk_int8_pv_fp16_cuda", "per_thread", "fp32")})
KITCHEN = Local(kitchen_int8, _named("comfy_kitchen.int8_attention"))
FLASHINFER = Local(flashinfer_mixed, _named("cute_dsl_fmha_ragged_prefill jit, e4m3 V per-head"))
SAGE3 = Local(sage3_fp4, _named("sageattn3.sageattn3_blackwell(nvfp4, per_block_mean)"))
