"""Real torch models stored in TensorFS and loaded through Varena for execution proofs."""

from __future__ import annotations

import os
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

import tensorfs
import torch
from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers.models.attention import FeedForward
from diffusers.models.attention_dispatch import AttentionBackendName, dispatch_attention_fn
from tensorfs.varena import GpuId
from tensorfs.varena.torch import Declared, Memory, Weights
from torch import nn

from cozy_runtime.internal import attention_ulysses, execution

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


# ------------------------------------------------------------- SDXL's UNet, narrow


#: SDXL's UNet topology (its blocks, its 70 transformer layers, its op count) at narrow widths
SDXL = dict(
    sample_size=16,
    layers_per_block=2,
    block_out_channels=(64, 128, 256),
    down_block_types=("DownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D"),
    up_block_types=("CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "UpBlock2D"),
    transformer_layers_per_block=(1, 2, 10),
    attention_head_dim=(2, 4, 8),
    cross_attention_dim=64,
    use_linear_projection=True,
    addition_embed_type="text_time",
    addition_time_embed_dim=8,
    projection_class_embeddings_input_dim=32 + 6 * 8,
)


def unet() -> nn.Module:
    torch.manual_seed(5)
    return UNet2DConditionModel(**SDXL).eval()


def unet_inputs() -> dict[str, Any]:
    gen = torch.Generator().manual_seed(6)
    return dict(
        sample=torch.randn(1, 4, 16, 16, generator=gen),
        timestep=torch.tensor(500),
        encoder_hidden_states=torch.randn(1, 77, 64, generator=gen),
        added_cond_kwargs={
            "text_embeds": torch.randn(1, 32, generator=gen),
            "time_ids": torch.randn(1, 6, generator=gen),
        },
    )
