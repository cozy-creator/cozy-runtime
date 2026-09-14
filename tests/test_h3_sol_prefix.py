"""Use upstream H3 packing to check Sol protects every non-target-video row."""

from types import SimpleNamespace
from typing import Any

import pytest


def _facts(packed: tuple[Any, ...]) -> Any:
    _, tags, video, audio, text, condition_video, condition_audio = packed
    return SimpleNamespace(
        denoiser_input_fields={
            "token_tags": tags,
            "video_indices": video,
            "audio_indices": audio,
            "text_indices": text,
        },
        num_condition_video_rows=condition_video,
        num_condition_audio_rows=condition_audio,
    )


def _check_prefix(packed: tuple[Any, ...]) -> None:
    import torch

    from cozy_runtime.models.minimax_h3.official import _attention_layout

    state = _facts(packed)
    layout = _attention_layout(state, 12)
    _, tags, video, audio, text, condition_video, _ = packed
    # Independent CPU oracle: upstream's first generated video position is the
    # boundary NVIDIA's H100 policy calls target_video_start/prefix_tokens.
    target_video_start = int(video[condition_video])
    assert layout.protected_prefix == target_video_start
    assert layout.live_tokens == len(tags)
    protected = torch.cat((text, video[:condition_video], audio)).sort().values
    assert torch.equal(protected, torch.arange(target_video_start))
    assert int(audio.max()) < layout.protected_prefix
    state.denoiser_input_fields = {
        name: tensor.to("meta") for name, tensor in state.denoiser_input_fields.items()
    }
    # Meta tensors have no values to read: serving can use shapes alone.
    assert _attention_layout(state, 12) == layout


@pytest.mark.parametrize("anchors", [(), ("first",), ("first", "last")])
@pytest.mark.parametrize("audio_latents", [1, 4])
def test_fl2va_protects_target_audio(anchors: tuple[str, ...], audio_latents: int) -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("diffusers")
    from diffusers.modular_pipelines.minimax_h3.before_denoise import MiniMaxH3PrepareLayoutStep

    _check_prefix(
        MiniMaxH3PrepareLayoutStep.build_packed_sequence(
            text_token_tags=torch.ones(3, dtype=torch.long),
            num_latent_frames=2,
            latent_height=4,
            latent_width=6,
            num_audio_latents=audio_latents,
            patch_size=(1, 2, 2),
            audio_channels=2,
            audio_tag=2,
            video_tag=0,
            keyframe_anchors=anchors,
        )
    )


def test_ref2va_protects_reference_and_target_audio() -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("diffusers")
    from diffusers.modular_pipelines.minimax_h3 import (
        MiniMaxH3AudioReference,
        MiniMaxH3ImageReference,
        MiniMaxH3VideoReference,
    )
    from diffusers.modular_pipelines.minimax_h3.before_denoise import (
        MiniMaxH3Ref2VAPrepareLayoutStep,
    )
    from PIL import Image

    references = [
        MiniMaxH3ImageReference(image=Image.new("RGB", (16, 16))),
        MiniMaxH3AudioReference(audio=torch.zeros(2, 8), sample_rate=32000),
        MiniMaxH3VideoReference(
            frames=torch.zeros(1, 3, 16, 16), fps=24, audio=torch.zeros(2, 8), sample_rate=32000
        ),
    ]
    _check_prefix(
        MiniMaxH3Ref2VAPrepareLayoutStep.build_ref2va_packed_sequence(
            text_token_tags=torch.ones(3, dtype=torch.long),
            references=references,
            condition_latents=[torch.zeros(1, 24, 1, 4, 6), torch.zeros(1, 24, 2, 4, 6)],
            audio_condition_latents=[torch.zeros(6, 32), torch.zeros(4, 32)],
            num_latent_frames=2,
            latent_height=4,
            latent_width=6,
            num_audio_latents=4,
            patch_size=(1, 2, 2),
            audio_channels=2,
            audio_tag=2,
            video_tag=0,
        )
    )
