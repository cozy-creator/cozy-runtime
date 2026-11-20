"""A Qwen3.5 row's pictures and videos for the engine's prefill (xs-213): the prompt's embeddings,
each picture's and video's tokens holding the vision tower's features, and its M-RoPE positions,
made by `Qwen3_5Model`'s own functions as its forward makes them. `CozyEngine` encodes a row before
the decoder leases the step's weights; the prefill reads its embeddings in place of the token
embeddings and its positions in place of the text positions; decode needs only the rope delta
(after the media every M-RoPE axis is equal, so the step's one-axis RoPE at `length + delta` is
Qwen's).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol, cast

import torch
from torch import Tensor


@dataclass(frozen=True)
class Media:
    """One row's pictures and videos as the processor read them (`TransformersEngine.inputs`)."""

    #: each token's kind (0 text, 1 picture, 2 video), `[1, L]`
    kinds: Tensor
    pixels: Tensor | None = None
    grid: Tensor | None = None
    video_pixels: Tensor | None = None
    video_grid: Tensor | None = None


@dataclass(frozen=True)
class Prompted:
    """A row's prompt as the language model reads it: `embeds` `[L, hidden]`, its M-RoPE
    `positions` `[3, L]`, and its rope delta: the position of the token after it, less `L`."""

    embeds: Tensor
    positions: Tensor
    delta: int


class _Features(Protocol):
    pooler_output: tuple[Tensor, ...]


class _Encoder(Protocol):
    def __call__(
        self, pixels: Tensor, grid: Tensor | None, /, *, return_dict: bool
    ) -> _Features: ...


class _Base(Protocol):
    """`Qwen3_5Model`, as the encode calls it."""

    get_image_features: _Encoder
    get_video_features: _Encoder

    def get_placeholder_mask(
        self,
        input_ids: Tensor,
        inputs_embeds: Tensor,
        image_features: Tensor | None = ...,
        video_features: Tensor | None = ...,
    ) -> tuple[Tensor, Tensor]: ...

    def get_rope_index(
        self,
        input_ids: Tensor,
        mm_token_type_ids: Tensor,
        image_grid_thw: Tensor | None = ...,
        video_grid_thw: Tensor | None = ...,
    ) -> tuple[Tensor, Tensor]: ...


class _Embed(Protocol):
    def __call__(self, ids: Tensor, /) -> Tensor: ...


def media_of(inputs: Mapping[str, Tensor]) -> Media:
    """A row's `Media` from the processor's inputs for it."""
    return Media(
        kinds=inputs["mm_token_type_ids"],
        pixels=inputs.get("pixel_values"),
        grid=inputs.get("image_grid_thw"),
        video_pixels=inputs.get("pixel_values_videos"),
        video_grid=inputs.get("video_grid_thw"),
    )


def encode(model: torch.nn.Module, ids: Tensor, media: Media) -> Prompted:
    """`ids` (`[1, L]`) and its `media` as `Qwen3_5Model.forward` reads them: the token embeddings,
    the vision tower's features of the pictures, then the videos, at their tokens; the positions
    of `get_rope_index`."""
    base = cast("_Base", model.model)
    embeds = cast("_Embed", model.get_input_embeddings())(ids)  # type: ignore[operator]  # Transformers'
    for pixels, grid, kind in (
        (media.pixels, media.grid, 0),
        (media.video_pixels, media.video_grid, 1),
    ):
        if pixels is None:
            continue
        encoder = (base.get_image_features, base.get_video_features)[kind]
        found = encoder(pixels, grid, return_dict=True).pooler_output
        features = torch.cat(found, dim=0).to(embeds.device, embeds.dtype)
        named = {("image_features", "video_features")[kind]: features}
        masks = base.get_placeholder_mask(ids, inputs_embeds=embeds, **named)
        embeds = embeds.masked_scatter(masks[kind], features)
    positions, deltas = base.get_rope_index(
        ids,
        mm_token_type_ids=media.kinds,
        image_grid_thw=media.grid,
        video_grid_thw=media.video_grid,
    )
    return Prompted(embeds=embeds[0], positions=positions[:, 0], delta=int(deltas.reshape(-1)[0]))
