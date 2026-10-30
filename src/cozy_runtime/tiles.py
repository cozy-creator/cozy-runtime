"""Decode a latent canvas without its whole working set: in exact row strips (`rows`), or in
overlapping tiles (`tiled`).

`rows` runs a row-local block over the whole map a strip of rows at a time: each strip reads its
rows and a halo either side and keeps only its own output rows, so the result is the block's on the
whole map, and the block's intermediates are a strip's. ComfyUI decodes Wan-family VAEs this way
(`strip_apply`, v0.43.0).

`tiled` is Diffusers' `tiled_decode` (AutoencoderKL, Wan, Qwen-Image): tiles every `stride` px,
each blended linearly into its upper and left neighbours over their overlap and cropped to
`stride`. Diffusers blends one overlapped row or column at a time, four kernels each: 31,000
launches at 2048² in 512 px tiles, 10 s of host time on a slow host. Here a blend is a few kernels
whatever its width, with the same bits.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

#: Elements of a strip's input at its output's resolution (64 MiB in bf16): a block's intermediates
#: are a strip's, whatever the canvas. Smaller strips launch more kernels for the same work.
STRIP = 1 << 25


def rows(
    fn: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    *,
    halo: int,
    scale: int = 1,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """`fn(x)` over `x`'s rows (dim -2) in strips, for a row-local `fn`: output row `r` reads
    input rows `r // scale ± halo` at most, and `fn` maps `n` rows to `scale * n`. A strip reads
    its rows and `halo` more on each side inside the map, so any zero padding `fn` applies at a
    strip's inner edge falls in rows it discards: the result is `fn` of the whole map. With `out`
    (`fn(x)`'s shape) the result is added into it, and `out` returned."""
    height = int(x.shape[-2])
    count = min(height, -(-x.numel() * scale * scale // STRIP))
    if count <= 1:
        return fn(x) if out is None else out.add_(fn(x))
    adding = out is not None
    step = -(-height // count)
    for a in range(0, height, step):
        b = min(height, a + step)
        lo, hi = max(0, a - halo), min(height, b + halo)
        strip = fn(x.narrow(-2, lo, hi - lo)).narrow(-2, (a - lo) * scale, (b - a) * scale)
        if out is None:
            out = strip.new_empty((*strip.shape[:-2], height * scale, strip.shape[-1]))
        part = out.narrow(-2, a * scale, (b - a) * scale)
        if adding:
            part.add_(strip)
        else:
            part.copy_(strip)
        del strip
    assert out is not None
    return out


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
