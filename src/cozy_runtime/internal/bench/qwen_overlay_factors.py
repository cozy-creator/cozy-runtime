"""One LoRA site's factors (`qwen_overlay.LoRAGraph`), apart because it is a torch Module: imported once a LoRA loads."""

from __future__ import annotations

import torch
from torch import nn

from cozy_runtime.accel import lora


class Factors(nn.Module):
    alpha: torch.Tensor

    def __init__(self, inputs: int, outputs: int, rank: int, weights: torch.dtype, alpha: torch.dtype) -> None:
        super().__init__()
        self.lora_A = nn.Linear(inputs, rank, bias=False, dtype=weights)
        self.lora_B = nn.Linear(rank, outputs, bias=False, dtype=weights)
        self.register_buffer("alpha", torch.zeros((), dtype=alpha))
        self._scale: float | None = None

    @property
    def scale(self) -> float:
        if self._scale is None:
            self._scale = float(self.alpha) / self.lora_A.weight.shape[0]
        return self._scale

    def forward(self, x: torch.Tensor, out: torch.Tensor) -> None:
        lora.accumulate(x, out, self.lora_A.weight, self.lora_B.weight, self.scale)

    def factors(self) -> tuple[torch.Tensor, torch.Tensor, float]:
        return self.lora_A.weight, self.lora_B.weight, self.scale
