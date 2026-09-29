"""A real diffusers VAE on a card too small to decode whole-frame beside a resident DiT."""

from typing import Any

import torch
from diffusers import AutoencoderKL


class Card(AutoencoderKL):
    card: Any = None

    def _decode(self, z: Any, return_dict: bool = True) -> Any:
        if not self.use_tiling or "ref2va_dit" in self.card.components:
            raise torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 128.00 MiB.")
        return super()._decode(z, return_dict=return_dict)
