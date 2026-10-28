"""Decode a latent canvas in overlapping tiles: the decoder's working set is a tile's, not the
canvas's.

The arithmetic is Diffusers' `tiled_decode` (AutoencoderKL, Wan, Qwen-Image): tiles every `stride`
px, each blended linearly into its upper and left neighbours over their overlap and cropped to
`stride`. Diffusers blends one overlapped row or column at a time, four kernels each: 31,000
launches at 2048² in 512 px tiles, 10 s of host time on a slow host. Here a blend is a few kernels
whatever its width, with the same bits.
"""

from __future__ import annotations

from collections.abc import Callable

import torch


def tiled(
    decode: Callable[[torch.Tensor], torch.Tensor],
    latents: torch.Tensor,
    *,
    scale: int,
    side: int,
    stride: int,
) -> torch.Tensor:
    """`decode` (latents to pixels, `scale` px a latent) of `latents` in `side` px tiles every
    `stride` px. Latents and pixels are `[..., height, width]`."""
    height, width = latents.shape[-2:]
    tile, step, overlap = side // scale, stride // scale, side - stride
    rows: list[list[torch.Tensor]] = []
    for i in range(0, height, step):
        row: list[torch.Tensor] = []
        for j in range(0, width, step):
            pixels = decode(latents[..., i : i + tile, j : j + tile])
            if rows:
                _blend(rows[-1][len(row)], pixels, overlap, -2)
            if row:
                _blend(row[-1], pixels, overlap, -1)
            row.append(pixels)
        rows.append(row)
    canvas = torch.cat([torch.cat([p[..., :stride, :stride] for p in row], -1) for row in rows], -2)
    return canvas[..., : height * scale, : width * scale]


def _blend(before: torch.Tensor, pixels: torch.Tensor, overlap: int, dim: int) -> None:
    """Diffusers' `blend_v` (dim -2) or `blend_h` (-1), in place on `pixels`: line `k` of the
    overlap becomes `before * (1 - k/n) + pixels * (k/n)`, each product rounded to the pixels'
    dtype and their sum too, as its per-line scalar arithmetic rounds."""
    n = min(before.shape[dim], pixels.shape[dim], overlap)
    weight = torch.arange(n, dtype=torch.float64, device=pixels.device) / n
    if dim == -2:
        weight = weight[:, None]
    near = pixels.narrow(dim, 0, n)
    far = before.narrow(dim, before.shape[dim] - n, n)
    kept = (far * (1 - weight).float()).to(pixels.dtype)
    near.copy_(kept + (near * weight.float()).to(pixels.dtype))
