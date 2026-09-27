"""Tiny real H3 plus an independent native projection for group preparation tests."""

from __future__ import annotations

import json
import os
import sys
from types import SimpleNamespace
from typing import Any, cast

import msgspec
import torch
from diffusers import MiniMaxH3Transformer3DModel

from cozy_runtime.author import App, Config, Loader, Model, sequence_parallel, uses_components
from cozy_runtime.author._attention_scope import AttentionLayout, attention_scope
from cozy_runtime.models.minimax_h3.official import OfficialH3Pipeline, _apply_transformer_dtype

app = App()

CONFIG = dict(
    num_attention_heads=56,
    attention_head_dim=128,
    hidden_size=16,
    num_layers=2,
    num_refiner_layers=2,
    ffn_dim=32,
    in_channels=2,
    audio_in_channels=2,
    patch_size=[1, 1, 1],
    text_dim=4,
    freq_dim=8,
    time_embed_hidden_dim=16,
    time_embed_dim=8,
    rope_freq_dim=2,
)


def batch(tokens: int = 1029) -> dict[str, Any]:
    generator = torch.Generator().manual_seed(91)
    tags = torch.tensor([1] * 16 + [2] * 12 + [0] * (tokens - 28))
    return {
        "hidden_states": torch.randn(1, tokens - 28, 2, generator=generator, dtype=torch.bfloat16),
        "audio_hidden_states": torch.randn(1, 12, 2, generator=generator, dtype=torch.bfloat16),
        "encoder_hidden_states": torch.randn(1, 16, 4, generator=generator, dtype=torch.bfloat16),
        "timestep": torch.tensor([0.0, 0.25, 0.75]),
        "timestep_indices": torch.arange(tokens) % 3,
        "token_tags": tags,
        "position_ids": torch.arange(tokens * 3).reshape(tokens, 3) % 5,
        "video_indices": torch.where(tags == 0)[0],
        "text_indices": torch.where(tags == 1)[0],
        "audio_indices": torch.where(tags == 2)[0],
        "return_dict": False,
    }


def execution_facts(module: Any, _args: Any, _kwargs: Any) -> None:
    # This is after CUDA initialization and supplements Runtime's pre-CUDA facts.
    # Registered on the real component, so followers report their own execution.
    print(
        json.dumps(
            {
                "event": "h3a093.component_enter",
                "pid": os.getpid(),
                "uid": os.getuid(),
                "tokens": int(_kwargs["token_tags"].numel()),
                "text_tokens": int(_kwargs["encoder_hidden_states"].shape[1]),
                "rank": torch.distributed.get_rank() if torch.distributed.is_initialized() else 0,
                "device": str(next(module.parameters()).device),
                "current_device": torch.cuda.current_device()
                if torch.cuda.is_initialized()
                else None,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "allocator": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
                "nvls": os.environ.get("NCCL_NVLS_ENABLE"),
                "segments": [
                    {
                        key: segment.get(key)
                        for key in ("device", "segment_type", "is_expandable", "total_size")
                    }
                    for segment in torch.cuda.memory_snapshot()
                ]
                if torch.cuda.is_initialized()
                else [],
            }
        ),
        file=sys.stderr,
        flush=True,
    )


class BasePipe:
    def __init__(self, config: Config) -> None:
        mapping = cast(dict[str, Any], config.mapping())
        constructor: Any = MiniMaxH3Transformer3DModel
        dit: Any = constructor(**mapping["dit"])
        dit = _apply_transformer_dtype(dit)
        dit.set_attention_backend("native")
        dit.register_forward_pre_hook(execution_facts, with_kwargs=True)
        self.components = {
            "fl2va_dit": dit,
            "spare": torch.nn.Linear(64, 64, bias=False, dtype=torch.bfloat16),
        }
        self._plans = {
            "fl2va": SimpleNamespace(
                schedules=[SimpleNamespace(video_timesteps=(1.0,), audio_timesteps=(1.0,))]
            )
        }

    def warm_dit(self) -> None:
        # Run the unchanged official warm input construction with the fixture schedule.
        OfficialH3Pipeline.warm_dit(cast(Any, self), "fl2va")


class OverlayPipe:
    def __init__(self, config: Config) -> None:
        del config
        self.components = {"projection": torch.nn.Linear(2, 2, bias=False, dtype=torch.bfloat16)}


@sequence_parallel(degrees=(2, 4, 7, 8))
class Base(Model[BasePipe], encoded_leaves="accept"):
    pipe: BasePipe

    def load(self, loader: Loader) -> None:
        self.pipe = loader.construct(BasePipe, factory=BasePipe)

        def scope_ready(module: Any, args: Any) -> None:
            del module, args
            active = self._cozy_active
            assert active is not None
            spare_device = self.pipe.components["spare"].weight.device.type
            assert spare_device == "meta", "warm kept unrelated filled weights resident"
            print(
                json.dumps(
                    {
                        "event": "h3a093.warm_scope",
                        "pid": os.getpid(),
                        "rank": torch.distributed.get_rank()
                        if torch.distributed.is_initialized()
                        else 0,
                        "spare_device": spare_device,
                        "active_components": list(active[1]),
                    }
                ),
                file=sys.stderr,
                flush=True,
            )

        self.pipe.components["fl2va_dit"].register_forward_pre_hook(scope_ready)

    @uses_components("spare")
    def spare_checksum(self) -> float:
        return float(self.pipe.components["spare"].weight.detach().sum())

    def warm(self, ctx: Any) -> None:
        ctx.raise_if_cancelled()
        device = next(self.pipe.components["fl2va_dit"].parameters()).device
        self.official_warm()
        for tokens in (1029, 1032):
            self.plain(
                {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch(tokens).items()
                }
            )

    @uses_components("fl2va_dit")
    def official_warm(self) -> None:
        self.pipe.warm_dit()

    @uses_components("fl2va_dit")
    def plain(self, values: dict[str, Any]) -> Any:
        with (
            torch.no_grad(),
            attention_scope(
                AttentionLayout(
                    values["token_tags"].numel(),
                    28,
                    0,
                    8,
                    ("token_refiner", "transformer_blocks.0", "transformer_blocks.1"),
                )
            ),
        ):
            return self.pipe.components["fl2va_dit"](**values)


@sequence_parallel(degrees=(2, 4, 7, 8))
class Overlay(Model[OverlayPipe]):
    pipe: OverlayPipe

    def load(self, loader: Loader) -> None:
        self.pipe = loader.construct(OverlayPipe, factory=OverlayPipe)

    @uses_components("projection")
    def checksum(self) -> float:
        return float(self.pipe.components["projection"].weight.detach().float().sum())


class Input(msgspec.Struct):
    value: int = 1


class Output(msgspec.Struct):
    value: int
    checksum: float


@app.entrypoint
def generate(payload: Input, base: Base, overlay: Overlay) -> Output:
    device = next(base.pipe.components["fl2va_dit"].parameters()).device
    values = {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch().items()
    }
    result = base.plain(values)
    return Output(payload.value, sum(float(value.sum()) for value in result) + overlay.checksum())
