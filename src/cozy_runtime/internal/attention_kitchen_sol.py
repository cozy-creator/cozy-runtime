"""Head-local Kitchen Sol recipes under the model's dense/sparse policy: Kitchen's sparse rows,
and its row's declared dense kernels for the dense ones (`attention.Candidate.dense`)."""

from __future__ import annotations

from collections.abc import Callable
from importlib import import_module
from typing import TYPE_CHECKING, Unpack

if TYPE_CHECKING:
    from cozy_runtime.internal.attention_ulysses import PartialOptions

    from torch import Tensor

from cozy_runtime.internal import attention_sol
from cozy_runtime.internal.attention_ulysses import Local


def sparse(
    query: Tensor, key: Tensor, value: Tensor, scale: float | None, sink_tokens: int
) -> Tensor:
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


def forward(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    *,
    dense: Callable[..., Tensor],
    prefix_dense: Callable[..., Tensor],
    **options: Unpack[PartialOptions],
) -> Tensor:
    return attention_sol._attention(
        query, key, value, **options, sparse=sparse, dense=dense, prefix_dense=prefix_dense
    )


def _named(implementation: str) -> Callable[..., str]:
    return lambda *_: implementation


KITCHEN_SOL = Local(forward, _named("comfy_kitchen.sol_attn(int8, tau=1, current-call stats)"))
KITCHEN_SOL_FP4_LOWMEM = Local(
    forward,
    _named(
        "dense=sageattn3.nvfp4_global_head_mean(short_owned_lifetimes); "
        "sparse=comfy_kitchen.sol_attn(int8)"
    ),
)
KITCHEN_SOL_FP4_FIXED_MEANS = Local(
    forward,
    _named(
        "dense=sageattn3.nvfp4_fixed_means(upstream_bf16_correction); "
        "sparse=comfy_kitchen.sol_attn(int8)"
    ),
)
