"""Head-local Kitchen Sol recipes under the model's dense/sparse policy."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from torch import Tensor

from cozy_runtime.internal import attention_sol, attention_ulysses, attention_upstream
from cozy_runtime.internal.attention_ulysses import Local, Options

NAMES = frozenset({"kitchen-sol", "kitchen-sol-fp4-lowmem"})
FP4 = frozenset(
    {
        "kitchen-sol-fp4-lowmem",
        "kitchen-sol-producer-fp4-lowmem-shared-qkv",
        "kitchen-sol-producer-fp4-fixed-means-shared-qkv",
    }
)


def validate_device(device: Any) -> None:
    torch = import_module("torch")
    capability = torch.cuda.get_device_capability(device) if device.type == "cuda" else None
    if capability not in attention_upstream.KITCHEN_SMS:
        raise ValueError(f"Kitchen Sol does not support device capability {capability}")


def sparse(query: Any, key: Any, value: Any, scale: float | None, sink_tokens: int) -> Any:
    kitchen = import_module("comfy_kitchen")
    tensors = (x if x.stride(-1) == 1 else x.contiguous() for x in (query, key, value))
    return kitchen.sol_attn(
        *tensors,
        tau=1.0,
        scale=scale,
        sink_blocks=[0, -(-sink_tokens // 64)],
        sink_q=[0, 0],
        token_aug=0,
    )


def dense_kitchen(*, query: Any, key: Any, value: Any, scale: float | None) -> Any:
    return attention_ulysses._call(
        "_cozy_kitchen_int8",
        attention_upstream.KITCHEN,
        query,
        key,
        value,
        Options(attn_mask=None, dropout_p=0.0, is_causal=False, scale=scale, enable_gqa=False),
    )


def dense_fp4_lowmem(*, query: Any, key: Any, value: Any, scale: float | None) -> Any:
    return attention_ulysses._call(
        "_cozy_sage3_fp4_global_lowmem",
        attention_upstream.SAGE3_GLOBAL_LOWMEM,
        query,
        key,
        value,
        Options(attn_mask=None, dropout_p=0.0, is_causal=False, scale=scale, enable_gqa=False),
    )


def dense_fp4_fixed_means(
    *, query: Tensor, key: Tensor, value: Tensor, scale: float | None
) -> Tensor:
    return attention_ulysses._call(
        "_cozy_sage3_fp4_fixed_means",
        attention_upstream.SAGE3_FIXED_MEANS,
        query,
        key,
        value,
        Options(attn_mask=None, dropout_p=0.0, is_causal=False, scale=scale, enable_gqa=False),
    )


def forward(query: Any, key: Any, value: Any, **options: Any) -> Any:
    return attention_sol._attention(
        query,
        key,
        value,
        **options,
        sparse=sparse,
        dense=dense_kitchen,
        validate_device=validate_device,
    )


def forward_fp4_lowmem(query: Any, key: Any, value: Any, **options: Any) -> Any:
    return attention_sol._attention(
        query,
        key,
        value,
        **options,
        sparse=sparse,
        dense=dense_fp4_lowmem,
        prefix_dense=dense_kitchen,
        validate_device=validate_device,
    )


KITCHEN_SOL = Local(forward, lambda *_: "comfy_kitchen.sol_attn(int8, tau=1, current-call stats)")
KITCHEN_SOL_FP4_LOWMEM = Local(
    forward_fp4_lowmem,
    lambda *_: (
        "dense=sageattn3.nvfp4_global_head_mean(short_owned_lifetimes); "
        "sparse=comfy_kitchen.sol_attn(int8)"
    ),
)
