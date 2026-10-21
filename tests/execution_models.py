"""Real models for the execution layer's tests: weights in a TensorFS store, loaded through
varena's host pages with the catch installed, and a card that runs out of memory.

The card is a dispatch mode under the catch: an op whose result and largest operand pass its
`limit` raises torch's `OutOfMemoryError` after telling the allocator's OOM observer the bytes,
as torch does; with a seed, an op runs out at seeded random points (each such op once).
"""

from __future__ import annotations

import os
import random
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

import tensorfs
import torch
from diffusers import AutoencoderKL
from diffusers.models.attention import FeedForward
from diffusers.models.attention_dispatch import AttentionBackendName, dispatch_attention_fn
from tensorfs.varena import GpuId
from tensorfs.varena.torch import Declared, Memory, Weights
from torch import nn
from torch.utils._python_dispatch import TorchDispatchMode

from cozy_runtime.internal import attention_ulysses, execution
from cozy_runtime.internal.execution import catch

_DTYPES = {torch.float32: "F32", torch.float64: "F64", torch.bfloat16: "BF16"}


def memory(capacity: int = 1 << 30) -> Memory:
    """A memory over host pages on a table of its own: no other test or process shares it."""
    table = f"cozy-execution-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    return Memory.open(GpuId(table, "cpu", capacity))


def checkpoint(store: Path, name: str, module: nn.Module) -> tensorfs.Checkpoint:
    """`module`'s weights as component `name` of a checkpoint in `store`."""
    writer = tensorfs.Writer(cache_dir=store)
    component = writer.component(name, cls=("cozy-test", "0", type(module).__name__), config={})
    for key, tensor in module.state_dict().items():
        held = tensor.detach().contiguous()
        data = held.view(torch.uint8).numpy().tobytes()
        component.put(key, data, dtype=_DTYPES[held.dtype], shape=list(held.shape))
    return writer.commit()


def placed[M: nn.Module](
    mem: Memory, ck: tensorfs.Checkpoint, name: str, build: Callable[[], M]
) -> M:
    """`build()` on meta, loaded from `ck` through `mem`, under the catch."""
    with torch.device("meta"):
        module = build()
    vars(module)["_varena"] = Declared(Weights(ck, name))
    mem.load(module, "gpu")
    execution.install(module)
    return module


class Card(TorchDispatchMode):  # type: ignore[misc]
    """A device that runs out of memory (module docstring)."""

    def __init__(self, limit: int = 0, seed: int | None = None, rate: float = 0.0) -> None:
        super().__init__()
        self.limit, self.rate = limit, rate
        self.random = random.Random(seed) if seed is not None else None
        self.refused = 0

    def __torch_dispatch__(self, func: Any, types: Any, args: Any = (), kwargs: Any = None) -> Any:
        out = func(*args, **(kwargs or {}))
        if any(r.alias_info is not None for r in func._schema.returns) or func._schema.is_mutable:
            return out
        made = sum(t.numel() * t.element_size() for t in catch.tensors(out))
        read = max((t.numel() * t.element_size() for t in catch.tensors(args)), default=0)
        short = self.limit and made and made + read > self.limit
        if short or (self.random is not None and made and self.random.random() < self.rate):
            self.refused += 1
            del out
            catch._refused(0, made, 0, 0)  # the allocator's observer, as torch calls it
            raise torch.OutOfMemoryError(f"out of memory. Tried to allocate {made} B")
        return out


# --------------------------------------------------------------------- a tiny transformer


def scores(query: Any, key: Any, value: Any, **options: Any) -> Any:
    """[batch, sequence, heads, dim] attention whose scores are one tensor (what a card runs
    out on), each query row computed alone (`rows` 8)."""
    q, k, v = (t.transpose(1, 2) for t in (query, key, value))
    weights = ((q @ k.transpose(-1, -2)) * q.shape[-1] ** -0.5).softmax(-1)
    return (weights @ v).transpose(1, 2)


def attention_backend() -> AttentionBackendName:
    local = attention_ulysses.Local(scores, lambda *_: "scores", head_local=True, rows=8)
    return attention_ulysses.member("_cozy_test_execution", local)


class Block(nn.Module):  # type: ignore[misc]
    """Self-attention through the Runtime's attention dispatch, then Diffusers' FeedForward."""

    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        self.heads = heads
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.qkv, self.out = nn.Linear(dim, 3 * dim), nn.Linear(dim, dim)
        self.ff = FeedForward(dim, mult=4, activation_fn="gelu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, s, d = x.shape
        qkv = self.qkv(self.norm1(x)).view(b, s, 3, self.heads, d // self.heads)
        q, k, v = qkv.unbind(2)
        attended = dispatch_attention_fn(q, k, v, backend=attention_backend())
        x = x + self.out(attended.reshape(b, s, d))
        return x + torch.nn.functional.silu(self.ff(self.norm2(x)))


class Transformer(nn.Module):  # type: ignore[misc]
    _no_split_modules = ("Block",)

    def __init__(self, dim: int = 64, heads: int = 4, depth: int = 3) -> None:
        super().__init__()
        self.proj_in, self.proj_out = nn.Linear(16, dim), nn.Linear(dim, 16)
        self.blocks = nn.ModuleList(Block(dim, heads) for _ in range(depth))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.proj_in(x)
        for block in self.blocks:
            h = block(h)
        return self.proj_out(h)


def transformer() -> Transformer:
    torch.manual_seed(0)
    model = Transformer()
    model.to(torch.bfloat16).eval()
    return model


# --------------------------------------------------------------------------- a tiny VAE


VAE = dict(
    in_channels=3,
    out_channels=3,
    down_block_types=("DownEncoderBlock2D", "DownEncoderBlock2D"),
    up_block_types=("UpDecoderBlock2D", "UpDecoderBlock2D"),
    block_out_channels=(8, 16),
    layers_per_block=1,
    latent_channels=4,
    norm_num_groups=4,
    sample_size=32,
)


def vae() -> nn.Module:
    torch.manual_seed(1)
    return AutoencoderKL(**VAE).to(torch.bfloat16).eval()
