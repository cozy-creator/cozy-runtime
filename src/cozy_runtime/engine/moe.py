"""A Qwen3.5 mixture-of-experts MLP in the engine's step (xs-213): the router, the shared expert, the
routed experts and their combine, as Transformers' `Qwen3_5MoeSparseMoeBlock` computes them, rounding
where it rounds, every shape fixed and the routing on the device, so the step's graph holds it.

- The router's logits are the module's own `F.linear`; `route_kernel` then takes the softmax in f32,
  the top k (the lowest index on a tie), renormalizes them where the model does, and rounds the
  weights to the model's dtype, in one launch.
- The shared expert is the step's dense MLP (`Qwen35._mlp`); its gate is the module's linear.
- The routed experts are `accel.moe.Experts.pairs`: the grouped kernels of the experts' kind, through
  the model's own kernel set (bf16, int8.convrot), or llama.cpp's MUL_MAT_ID (ggml's types).
- `combine_kernel` weighs each pair, sums a token's, adds the gated shared expert, and writes the
  result in the residual's product dtype, so the residual add reads it as it reads attention's.

Unfused, the block is its own module's forward: the reference the fused ops are held against.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Protocol, cast

import torch
import torch.nn.functional as F

from cozy_runtime.accel import moe as grouped
from cozy_runtime.accel.kernels import read
from cozy_runtime.engine import _moe

if TYPE_CHECKING:
    from collections.abc import Callable

    from torch import Tensor

    from cozy_runtime.engine.launch import Run

#: A combine program's columns of a token's row.
COMBINE_BLOCK = 1024


@dataclass(frozen=True)
class Routed:
    """What a step's routed MLPs launch with: the experts a layer holds, the top k a token takes,
    whether their weights are renormalized to sum 1, and the grouped kernels' kind of the experts."""

    experts: int
    top_k: int
    renorm: int
    kind: str

    @property
    def index(self) -> torch.dtype:
        """The routing's dtype: llama.cpp's MUL_MAT_ID reads int32 ids, the Triton kernels int64."""
        return torch.int32 if self.kind == "ggml" else torch.int64

    @property
    def pairs(self) -> str:
        """The dtype `Experts.pairs` writes for this kind (Triton's name)."""
        return "fp32" if self.kind == "ggml" else "bf16"


class _Router(Protocol):
    weight: Tensor
    top_k: int


class _Linear(Protocol):
    def __call__(self, x: Tensor, /) -> Tensor: ...


class _Block(Protocol):
    """`Qwen3_5MoeSparseMoeBlock`, as the step reads it."""

    gate: _Router
    experts: grouped.Experts
    shared_expert: object
    shared_expert_gate: _Linear

    def __call__(self, h: Tensor, /) -> Tensor: ...


def is_routed(mlp: object) -> bool:
    """Whether `mlp` is a mixture of experts (it has a router and experts), not a dense MLP."""
    return hasattr(mlp, "experts") and hasattr(mlp, "gate")


def unserved(mlp: object) -> str:
    """Why the step can't run `mlp`, or "": a dense SwiGLU MLP, or a routed one as Qwen3.5's (a
    softmax top-k router over experts split per expert, of a kind the grouped kernels run, beside one
    shared expert behind a sigmoid gate)."""
    dense = ("gate_proj", "up_proj", "down_proj")
    if not is_routed(mlp):
        return "" if all(hasattr(mlp, p) for p in dense) else "its MLPs are neither SwiGLU nor routed experts"
    block = cast("_Block", mlp)
    if not isinstance(getattr(mlp, "experts", None), grouped.Experts):
        return "its experts are not split per expert (accel.moe.Experts)"
    weight = getattr(block.gate, "weight", None)
    top_k = getattr(block.gate, "top_k", None)
    if not isinstance(weight, torch.Tensor) or weight.dim() != 2 or int(weight.shape[0]) != len(block.experts):
        return "its router holds no [experts, hidden] weight"
    if not isinstance(top_k, int) or not 0 < top_k <= len(block.experts):
        return "its router states no top_k"
    shared, gate = getattr(block, "shared_expert", None), getattr(block, "shared_expert_gate", None)
    if not all(hasattr(shared, p) for p in dense):
        return "its routed MLPs have no SwiGLU shared expert"
    gated = getattr(gate, "weight", None)
    if not isinstance(gated, torch.Tensor) or tuple(gated.shape) != (1, int(weight.shape[1])):
        return "its shared expert has no [1, hidden] gate"
    return grouped.unserved(block.experts)


def shared(mlp: object) -> object:
    """A routed MLP's shared expert (`unserved` passed it)."""
    return cast("_Block", mlp).shared_expert


def shared_gate(mlp: object) -> object:
    """A routed MLP's shared expert's gate (`unserved` passed it)."""
    return cast("_Block", mlp).shared_expert_gate


def routed_of(mlp: object, renorm: bool) -> Routed:
    """A routed MLP's `Routed` (`unserved` passed it)."""
    block = cast("_Block", mlp)
    kind = grouped.kind(block.experts)
    assert kind is not None  # `unserved` passed it
    return Routed(len(block.experts), int(block.gate.top_k), int(renorm), kind)


class Block:
    """One routed MLP of the step, called as `_mlp` is: `h` `[rows, S, hidden]` to the MLP's
    addend, in the residual's product dtype. `shared` is the step's dense MLP over the shared
    expert; `product` a linear's product as the fused ops read it (`qwen35._product`)."""

    def __init__(
        self,
        module: object,
        routed: Routed,
        *,
        shared: Callable[[Tensor], Tensor],
        product: Callable[[_Linear, Tensor], Tensor],
        run: Run,
        fused: bool,
    ) -> None:
        self.module = cast("_Block", module)
        self.routed, self.shared, self.product, self.run, self.fused = routed, shared, product, run, fused
        #: the grouped Triton kernels through the model's set, as the step's own launch
        self.launch = partial(run, num_warps=4)

    def __call__(self, h: Tensor) -> Tensor:
        if not self.fused or _moe.TRITON_ABSENT or not h.is_cuda:
            return self.module(h)
        block, routed = self.module, self.routed
        rows, span, width = h.shape
        x = h.reshape(-1, width).contiguous()
        tokens, k = int(x.shape[0]), routed.top_k
        logits = F.linear(x, read(block.gate.weight)).contiguous()
        index = torch.empty((tokens, k), dtype=routed.index, device=x.device)
        weights = torch.empty((tokens, k), dtype=x.dtype, device=x.device)
        self.run(_moe.route_kernel, (tokens,), logits, index, weights, routed.experts,
                 1 << max(routed.experts - 1, 0).bit_length(), k, routed.renorm, num_warps=4)  # fmt: skip
        pairs = block.experts.pairs(x, index, self.launch)
        # As the set declares them: each hit expert's own linears (ggml's kernels not verified yet) write bf16.
        pairs = pairs.to(torch.float32 if routed.pairs == "fp32" else torch.bfloat16).contiguous()
        shared = self.shared(x).contiguous()
        gate = self.product(block.shared_expert_gate, x).contiguous()
        out = torch.empty((tokens, width), dtype=shared.dtype, device=x.device)
        self.run(_moe.combine_kernel, (tokens, -(-width // COMBINE_BLOCK)),
                 pairs, weights, shared, gate, out, width, k, COMBINE_BLOCK, num_warps=4)  # fmt: skip
        return out.view(rows, span, width)
