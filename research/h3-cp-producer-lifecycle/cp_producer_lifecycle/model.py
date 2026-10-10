"""Real bias-free H3 attention with six bound tensors; no transformer/quality claim."""

import hashlib
import importlib.metadata
import json
import sys
from typing import Any, ClassVar

import torch
import torch.distributed as dist
from diffusers.models._modeling_parallel import ContextParallelInput, ContextParallelOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.transformers.transformer_minimax_h3 import (
    MiniMaxH3Attention,
    MiniMaxH3RotaryPosEmbed,
    MiniMaxH3Transformer3DModel,
)

from cozy_runtime.author import Config, Loader, Model, sequence_parallel, uses_components
from cozy_runtime.author._attention_scope import AttentionLayout, attention_scope

NAME = "kitchen-sol-producer-fp4-fixed-means-shared-qkv"
TOKENS = 4225  # Uneven local shards and a second global producer chunk.
PREFIX = 257


def tensor_hash(value: torch.Tensor) -> str:
    return hashlib.sha256(
        value.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
    ).hexdigest()


def stats_hash(pair: tuple[Any, Any]) -> list[str]:
    if any(value.dtype != torch.float32 or value.shape != (1, 128) for value in pair):
        raise RuntimeError("producer statistics must be one head of FP32 channels")
    if not torch.isfinite(torch.stack(pair)).all():
        raise RuntimeError("producer statistics must be finite")
    return [tensor_hash(value) for value in pair]


def report_tensor(report: dict[str, Any]) -> torch.Tensor:
    encoded = json.dumps(report, allow_nan=False).encode()
    if len(encoded) > 1 << 20:
        raise ValueError("lifecycle report exceeds its declared output bound")
    return torch.frombuffer(bytearray(encoded), dtype=torch.uint8).clone()


def read_report(value: torch.Tensor) -> dict[str, Any]:
    if value.device.type != "cpu" or value.dtype != torch.uint8 or value.ndim != 1:
        raise ValueError("component report must be a CPU byte tensor")
    result = json.loads(value.numpy().tobytes())
    if not isinstance(result, dict):
        raise ValueError("component report must be an object")
    return result


class Block(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attn = MiniMaxH3Attention(hidden_size=128, heads=2, dim_head=128).to(
            dtype=torch.bfloat16
        )

    def forward(
        self, hidden_states: torch.Tensor, rotary_emb: tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        return self.attn(hidden_states, rotary_emb=rotary_emb)


class Component(MiniMaxH3Transformer3DModel):
    """H3 type and hooks with one genuine attention site; only this site's weights exist."""

    _cp_plan: ClassVar[dict] = {
        "rope": {
            0: ContextParallelInput(split_dim=0, expected_dims=2, split_output=True),
            1: ContextParallelInput(split_dim=0, expected_dims=2, split_output=True),
        },
        "transformer_blocks.0": {
            "hidden_states": ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
        },
        "proj_out": ContextParallelOutput(gather_dim=1, expected_dims=3),
        "": {},  # Exercise the real root output hook with the encoded CPU report.
    }

    def __init__(self) -> None:
        ModelMixin.__init__(self)
        self.register_to_config(num_attention_heads=2, attention_head_dim=128, hidden_size=128)
        self.transformer_blocks = torch.nn.ModuleList([Block()])
        self.rope = MiniMaxH3RotaryPosEmbed(rope_freq_dim=16)
        self.proj_out = torch.nn.Identity()

    def forward(self, step: int, seed: int, first: bool) -> torch.Tensor:
        from cozy_runtime.internal import attention_producer as producer
        from cozy_runtime.internal.attention_ulysses import group
        from cozy_runtime.internal.parallel.cp import in_gated_call

        if not in_gated_call():
            raise RuntimeError("lifecycle proof requires actual Runtime mirrored calls")
        handle = group(self._parallel_config)
        rank, degree = dist.get_rank(handle), dist.get_world_size(handle)
        if degree != 2:
            raise RuntimeError("this lifecycle fixture requires exactly two ranks")
        state = producer._STATE.get()
        if state is None:
            raise RuntimeError("the invocation has no producer state")
        before = [
            (value[0], value[1], stats_hash(value[2:])) for value in state.statistics.values()
        ]
        if first and (before or state.replay_inputs or state.parallel_rotary):
            raise RuntimeError("previous invocation leaked producer state into its successor")
        attn = self.transformer_blocks[0].attn
        if not isinstance(attn.processor, producer.Processor):
            raise RuntimeError("the actual instance producer processor was not installed")
        if (
            producer.BACKENDS.get(
                str(
                    getattr(
                        attn.processor._attention_backend,
                        "value",
                        attn.processor._attention_backend,
                    )
                )
            )
            != NAME
        ):
            raise RuntimeError("the exact opt-in CP producer selector was not applied")
        device = attn.norm_q.weight.device
        properties = torch.cuda.get_device_properties(device)
        # Every rank creates the same full hidden tensor before actual Diffusers splitting.
        hidden = torch.randn(
            (1, TOKENS, 128),
            generator=torch.Generator(device=device).manual_seed(seed),
            device=device,
            dtype=torch.bfloat16,
        )
        positions = torch.arange(TOKENS, device=device).repeat(3, 1).t()
        rotary = self.rope(positions)
        output = self.proj_out(self.transformer_blocks[0](hidden_states=hidden, rotary_emb=rotary))
        if output.shape != (1, TOKENS, 128) or not torch.isfinite(output).all():
            raise RuntimeError(
                "the CP split/inverse/gather did not return finite full-token output"
            )
        if len(state.statistics) != 1 or len(state.replay_inputs) != 1:
            raise RuntimeError("the fixture must observe exactly one actual producer site")
        current = next(iter(state.statistics.values()))
        original = next(iter(state.replay_inputs.values()))
        retained = state.parallel_rotary[attn._cozy_producer_rotary_key]
        row = {
            "rank": rank,
            "pid": __import__("os").getpid(),
            "python_prefix": sys.prefix,
            "step": step,
            "first": first,
            "before": before,
            "input_stats": None if original[2] is None else stats_hash(original[2:]),
            "output_stats": stats_hash(current[2:]),
            "stats_shapes": [list(value.shape) for value in current[2:]],
            "full_rope_shape": list(retained[0].shape),
            "local_rope_shape": list(rotary[0].shape),
            "output_sha256": tensor_hash(output),
            "output_shape": list(output.shape),
            "backend": NAME,
            "runtime": importlib.metadata.version("cozy-runtime"),
            "tensorfs": importlib.metadata.version("tensorfs"),
            "torch": torch.__version__,
            "gpu": str(properties.name),
            "uuid": str(properties.uuid),
            "capability": [properties.major, properties.minor],
        }
        ranks: list[Any] = [None] * degree
        dist.all_gather_object(ranks, row, group=handle)
        return report_tensor({"ranks": ranks})


class Pipeline:
    def __init__(self) -> None:
        self.dit = Component()
        self.components = {"dit": self.dit}


def build(config: Config) -> Pipeline:
    return Pipeline()


@sequence_parallel(degrees=(2,))
class ProbeModel(Model[Pipeline]):
    pipe: Pipeline

    def load(self, loader: Loader) -> None:
        self.pipe = loader.construct(Pipeline, factory=build)

    @uses_components("dit")
    def verify(self, start_step: int, seed: int) -> dict[str, Any]:
        if start_step not in (4, 6):
            raise ValueError("only the two declared request boundaries are supported")
        calls = []
        for step, first in ((start_step, True), (start_step + 1, False), (start_step + 1, False)):
            with attention_scope(AttentionLayout(TOKENS, PREFIX, step, 4)):
                calls.append(read_report(self.pipe.dit(step, seed + step, first)))
        for rank in range(2):
            first, second, replay = [call["ranks"][rank] for call in calls]
            if first["input_stats"] is not None or first["before"]:
                raise RuntimeError("first call inherited previous-request statistics")
            if (
                second["input_stats"] != first["output_stats"]
                or replay["input_stats"] != second["input_stats"]
            ):
                raise RuntimeError(
                    "separate component calls/replay lost the original prior statistics"
                )
            if (
                second["output_sha256"] != replay["output_sha256"]
                or second["output_stats"] != replay["output_stats"]
            ):
                raise RuntimeError("same-input same-step replay changed producer output/statistics")
            if first["pid"] != second["pid"] or first["pid"] != replay["pid"]:
                raise RuntimeError("component calls changed executor identity")
        return {
            "passed": True,
            "start_step": start_step,
            "tokens": TOKENS,
            "prefix": PREFIX,
            "calls": calls,
            "scope": (
                "real H3 attention and producer lifecycle; BF16 tiny projections, "
                "no whole H3/video or FP8/LoRA proof"
            ),
        }
