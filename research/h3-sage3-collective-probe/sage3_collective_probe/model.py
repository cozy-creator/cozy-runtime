"""Tiny checkpoint admits a real Runtime group; its weights are not probe operands."""

from typing import Any

import torch
from diffusers.models.modeling_utils import ModelMixin

from cozy_runtime.author import Config, Loader, Model, sequence_parallel, uses_components


class ProbeComponent(ModelMixin):
    # Runtime installs the real ParallelConfig and mirrors this root's forward.
    # No input split hook is needed: forward creates its own full deterministic data
    # and explicitly takes this rank's unequal token shard before Ulysses exchange.
    _cp_plan = {"": {}}
    num_attention_heads = 56

    def __init__(self):
        super().__init__()
        # Exact declared tensors of the existing 133KB diagnostic checkpoint.
        self.to_q = torch.nn.Linear(128, 128, dtype=torch.bfloat16)
        self.to_k = torch.nn.Linear(128, 128, dtype=torch.bfloat16)
        self.to_v = torch.nn.Linear(128, 128, dtype=torch.bfloat16)
        self.to_out = torch.nn.ModuleList([torch.nn.Linear(128, 128, dtype=torch.bfloat16)])
        self.norm_q = torch.nn.RMSNorm(128, dtype=torch.bfloat16)
        self.norm_k = torch.nn.RMSNorm(128, dtype=torch.bfloat16)

    def forward(self, seed: int):
        from cozy_runtime.internal.attention_ulysses import group
        from cozy_runtime.internal.parallel.cp import in_gated_call

        from .probe import run

        if not in_gated_call():
            raise RuntimeError("This diagnostic requires the Runtime mirrored call gate")
        handle = group(self._parallel_config)
        return run(seed, handle)


class Pipeline:
    def __init__(self):
        self.dit = ProbeComponent()
        self.components = {"dit": self.dit}


def build(config: Config) -> Pipeline:
    return Pipeline()


@sequence_parallel(degrees=(4,))
class ProbeModel(Model[Pipeline]):
    pipe: Pipeline

    def load(self, loader: Loader) -> None:
        self.pipe = loader.construct(Pipeline, factory=build)

    @uses_components("dit")
    def verify(self, seed: int) -> Any:
        return self.pipe.dit(seed)
