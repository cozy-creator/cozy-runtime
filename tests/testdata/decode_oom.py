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


class StreamingVAE(torch.nn.Module):
    """A video VAE that tiles but streams: its decode hands chunks out as it goes."""

    use_tiling = False

    def enable_tiling(self) -> None:
        self.use_tiling = True

    def disable_tiling(self) -> None:
        self.use_tiling = False

    def decode(self, z: Any, on_chunk: Any) -> None:
        on_chunk(z[:, :, :1])
        raise torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 128.00 MiB.")
