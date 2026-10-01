"""MiniMax H3's real ref2va conditioning at a few million parameters, for group tests.

The official blocks, the bundled tokenizer and processor, a 64-layer Qwen3-VL conditioner cut
to its 50 kept layers (`build_text_conditioner`), tiny real video and audio VAEs, and the
tiny shardable DiT of `parallel_calls`. Every rank builds it from the same seeds, so a
follower's copy of a component equals rank 0's. Run as a script, it is a follower rank.
"""

from __future__ import annotations

import argparse
import socket
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import torch
from diffusers import (
    AutoencoderKLMiniMaxH3Audio,
    MiniMaxH3Blocks,
    MiniMaxH3ModularPipeline,
    MiniMaxH3Scheduler,
)
from diffusers.modular_pipelines.minimax_h3 import MiniMaxH3VideoReference
from PIL import Image
from transformers import Qwen3VLConfig

sys.path[:0] = [str(Path(__file__).parent), str(Path(__file__).parents[1])]

from parallel_calls import h3_model, prepared  # type: ignore[import-not-found]  # noqa: E402

from cozy_runtime.author import DerivedCache  # noqa: E402
from cozy_runtime.author._decode import DecodedAudio  # noqa: E402
from cozy_runtime.internal.executor import Executor, _capture_seal  # noqa: E402
from cozy_runtime.internal.executor_commands import decode  # noqa: E402
from cozy_runtime.internal.seam import Channel  # noqa: E402
from cozy_runtime.models.minimax_h3 import official  # noqa: E402
from cozy_runtime.models.minimax_h3.conditioner import (  # noqa: E402
    build_text_conditioner,
    text_conditioner_config,
)
from cozy_runtime.models.minimax_h3.model import H3Model  # noqa: E402
from cozy_runtime.models.minimax_h3.vae_tiles import TileBatchedVideoVAE  # noqa: E402

TASK: official.Task = "ref2va"
#: settled component bytes for the hosting plan: rank 0's capacity holds the DiT and the VAEs, a
#: follower holds the conditioner beside its DiT shard
SIZES = {"ref2va_dit": 100, "text_encoder": 60, "video_vae": 30, "audio_vae": 5}
HOSTED_CAPACITY = 170


def conditioner() -> Any:
    config = Qwen3VLConfig().to_dict()
    config["text_config"].update(
        num_hidden_layers=64,
        hidden_size=32,
        intermediate_size=64,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        rope_parameters={
            "rope_theta": 5_000_000.0,
            "rope_type": "default",
            "mrope_section": [4, 2, 2],
            "mrope_interleaved": True,
        },
    )
    config["vision_config"].update(
        depth=2,
        hidden_size=32,
        intermediate_size=64,
        num_heads=2,
        out_hidden_size=32,
        deepstack_visual_indexes=[0, 1],
        num_position_embeddings=64,
    )
    config["architectures"] = ["Qwen3VLForConditionalGeneration"]
    config["cozy_h3"] = text_conditioner_config()
    return build_text_conditioner(config)


def pipeline() -> Any:
    torch.manual_seed(11)
    text_encoder = conditioner()
    video_vae = TileBatchedVideoVAE(
        block_out_channels=(8,) * 6,
        layers_per_block=1,
        norm_num_groups=4,
        decoder_num_layers=1,
        decoder_num_attention_heads=1,
        decoder_attention_head_dim=8,
        decoder_ffn_mult=1,
    ).eval()
    audio_vae = AutoencoderKLMiniMaxH3Audio(
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
    dit = h3_model().dit
    tokenizer, processor = official._processor()
    blocks = MiniMaxH3Blocks().get_workflow(TASK)
    upstream = MiniMaxH3ModularPipeline(blocks=blocks)
    upstream.register_components(
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        processor=processor,
        vae=video_vae,
        audio_vae=audio_vae,
        scheduler=MiniMaxH3Scheduler(shift=12.0),
        audio_scheduler=MiniMaxH3Scheduler(shift=3.0),
        transformer_ref=dit,
    )
    pipe = official.OfficialH3Pipeline.__new__(official.OfficialH3Pipeline)
    pipe._blocks = {TASK: blocks}
    pipe._pipes = {TASK: upstream}
    pipe._plans = {TASK: official.canonical_timestep_plan(TASK)}
    pipe._intake = official._IntakeLengths()
    pipe.components = {
        "ref2va_dit": dit,
        "text_encoder": text_encoder,
        "video_vae": video_vae,
        "audio_vae": audio_vae,
    }
    pipe.resident = official.ResidentWeights()
    pipe.sample_rate = int(audio_vae.config.sampling_rate)
    return pipe


def model() -> Any:
    """The runtime's own H3Model over the tiny pipeline, with its reference-latent memo."""
    built: Any = H3Model.for_test(pipe=pipeline())
    object.__setattr__(
        built,
        "reference_latents",
        DerivedCache("reference_latents", max_entries=64, max_bytes=1 << 26, owner=built),
    )
    return built


def backend(built: Any) -> Any:
    """The prepared construction `Executor._install_group` reads, at `SIZES`."""
    return prepared(dict(built.pipe.components), SIZES)


def references(pipe: Any) -> list[Any]:
    """Two pictures, a short silent clip and a voice: every kind the encode step handles."""
    generator = np.random.default_rng(5)
    pictures = [
        Image.fromarray(generator.integers(0, 255, (200, 300, 3), dtype=np.uint8)),
        Image.fromarray(generator.integers(0, 255, (320, 200, 3), dtype=np.uint8)),
    ]
    rate = 16000
    wave = (0.25 * np.sin(2 * np.pi * 440.0 * np.arange(2 * rate) / rate)).astype("<f4")
    voice = DecodedAudio(
        channels=1,
        sample_count=2 * rate,
        sample_rate=rate,
        channel_layout="mono",
        channel_names=("FC",),
        pcm_f32le=(wave.tobytes(),),
        start_time=Fraction(0),
    )
    clip = generator.integers(0, 255, (22, 32, 48, 3), dtype=np.uint8)
    return [
        *(pipe.image_reference(picture) for picture in pictures),
        MiniMaxH3VideoReference(frames=clip),
        pipe.audio_reference(voice),
    ]


def start(built: Any) -> Any:
    pipe = built.pipe
    return pipe.start_ref2va(
        prompt="<Picture 1> and <Picture 2> walk through <Video 1> while <Audio 1> sings",
        references=references(pipe),
        generator=torch.Generator().manual_seed(1),
        steps=pipe._plans[TASK].steps[0],
        frames=official.frames_for(5),
        reference_image_short_edges=(64, 64),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path)
    parser.add_argument("--rank", type=int)
    parser.add_argument("--world", type=int)
    parser.add_argument("--rank-fd", type=int)
    parser.add_argument("--leader")
    args = parser.parse_args()
    torch.set_num_threads(1)
    _capture_seal()
    channel = Channel(socket.socket(fileno=args.rank_fd))
    executor: Any = Executor(channel, args.root, rank=args.rank, world=args.world)
    executor.device_kind = "cpu"
    built = model()
    executor._group_model_key = "fixture"
    executor.backend = backend(built)
    executor.torch = torch
    executor.ready = True
    while (command := channel.recv()) is not None:
        name = command["cmd"]
        if name == "shutdown":
            return
        if name == "join":
            reply = executor.join(decode(command))
            if reply.get("ok"):
                refused = executor._install_group(torch, built, HOSTED_CAPACITY)
                assert refused is None, refused
        elif name == "run":
            reply = executor.run(command)
        else:
            raise AssertionError(name)
        channel.send({**reply, "reply": name})


if __name__ == "__main__":
    main()
