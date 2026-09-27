"""The official H3 ref2va workflow's own blocks over tiny real VAEs, the bundled tokenizer and
processor, and the conditioner's shape: everything the intake attention length depends on.

Run as a script, it is a short-lived process that starts `count` requests and returns while
their intake statements may still be inside torch, as a CPU proof script does.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import Any

import torch
from diffusers import (
    AutoencoderKLMiniMaxH3,
    AutoencoderKLMiniMaxH3Audio,
    MiniMaxH3Blocks,
    MiniMaxH3ModularPipeline,
)
from PIL import Image

from cozy_runtime.models.minimax_h3 import official


def pipeline() -> official.OfficialH3Pipeline:
    vae = AutoencoderKLMiniMaxH3(
        block_out_channels=(8,) * 6,
        layers_per_block=1,
        norm_num_groups=4,
        decoder_num_layers=1,
        decoder_num_attention_heads=1,
        decoder_attention_head_dim=8,
        decoder_ffn_mult=1,
    ).eval()
    audio = AutoencoderKLMiniMaxH3Audio(
        encoder_dim=4,
        encoder_rates=(2,),
        latent_dim=8,
        latent_channels=4,
        num_attention_heads=1,
        decoder_dim=8,
        decoder_rates=(2,),
        decoder_kernel_sizes=(4,),
        latents_mean=[0.0] * 4,
        latents_std=[1.0] * 4,
    ).eval()
    # Only the conditioner's shape is read; its rows are the presentation's tokens.
    conditioner = SimpleNamespace(
        config=SimpleNamespace(text_config=SimpleNamespace(num_hidden_layers=64)),
        dtype=torch.bfloat16,
        device=torch.device("cpu"),
    )
    tokenizer, processor = official._processor()
    blocks = MiniMaxH3Blocks().get_workflow("ref2va")
    upstream = MiniMaxH3ModularPipeline(blocks=blocks)
    upstream.register_components(tokenizer=tokenizer, processor=processor, vae=vae, audio_vae=audio)
    pipe = official.OfficialH3Pipeline.__new__(official.OfficialH3Pipeline)
    pipe._blocks = {"ref2va": blocks}
    pipe._pipes = {"ref2va": upstream}
    pipe._plans = {"ref2va": official.canonical_timestep_plan("ref2va")}
    pipe._intake = official._IntakeLengths()
    pipe.components = {
        "text_encoder": conditioner,
        "video_vae": vae,
        "audio_vae": audio,
        "ref2va_dit": torch.nn.Linear(1, 1),
    }
    return pipe


def start(pipe: official.OfficialH3Pipeline, references: list[Any], edges: tuple[int, ...]) -> Any:
    return pipe.start_ref2va(
        prompt="<Picture 1> and <Picture 2> cross swords in the rain",
        references=references,
        generator=torch.Generator().manual_seed(1),
        steps=pipe._plans["ref2va"].steps[0],
        frames=official.frames_for(5),
        reference_image_short_edges=edges,
    )


def main(count: int) -> None:
    torch.set_num_threads(1)
    pipe = pipeline()
    references = [pipe.image_reference(Image.new("RGB", (1500, 1000), (10, 20, 30)))]
    for _ in range(count):
        start(pipe, references, (2048,))
    print(f"started {count}", flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]))
