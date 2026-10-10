"""Actual H3 packing/scheduler and bounded completed-AV continuation proofs."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")

from diffusers.models.autoencoders.autoencoder_kl_minimax_h3_audio import (  # noqa: E402
    AutoencoderKLMiniMaxH3Audio,
)
from diffusers.modular_pipelines.minimax_h3 import (  # noqa: E402
    MiniMaxH3Blocks,
    MiniMaxH3ModularPipeline,
    MiniMaxH3VideoReference,
)
from diffusers.modular_pipelines.modular_pipeline import PipelineState  # noqa: E402

from cozy_runtime.author import InvalidRequest  # noqa: E402
from cozy_runtime.models.minimax_h3.continuation import (  # noqa: E402
    AVContext,
    CompletedTail,
    ContextReferenceLayout,
    align_context_rows,
    audio_sample_range,
    decode_context,
    encode_context,
    plan_continuation,
)
from cozy_runtime.models.minimax_h3.official import OfficialH3Pipeline  # noqa: E402
from cozy_runtime.models.minimax_h3.vae_tiles import TileBatchedVideoVAE  # noqa: E402


def context(frames: int = 56, end: int = 262, windows: tuple[int, ...] = (22, 39, 56)) -> AVContext:
    encoded = {
        n: (
            torch.full((1, 24, 5 * ((n - 5) // 17) + 2, 2, 4), n / 100),
            torch.full((2 * ((n * 5 + 2) // 3), 32), n / 100),
        )
        for n in windows
    }
    return AVContext(encoded, 32, 64, end, "model-and-inference-v1", frames)


@pytest.mark.parametrize("prefix,sampled", [(0, 243), (22, 277), (39, 294), (56, 311)])
def test_delivery_clock_charges_context_without_delivering_it(prefix: int, sampled: int) -> None:
    plan = plan_continuation(240, context_frames=prefix)
    assert plan.sample_frames == sampled
    assert plan.delivery_end - plan.delivery_start == 240
    start, end = audio_sample_range(plan.delivery_start, plan.delivery_end, 32000)
    assert end - start == 320000
    assert sampled % 17 == 5
    with pytest.raises(InvalidRequest, match="362"):
        plan_continuation(360, context_frames=22)
    with pytest.raises(InvalidRequest, match="362"):
        plan_continuation(312, context_frames=56)
    with pytest.raises(InvalidRequest):
        plan_continuation(240, context_frames=32)


@pytest.mark.parametrize("frames", [22, 39, 56])
def test_context_codec_and_selection_preserve_independent_native_windows(frames: int) -> None:
    original = context()
    selected = decode_context(encode_context(original)).select(frames)
    assert torch.equal(selected.video_latents, original.windows[frames][0])
    assert torch.equal(selected.audio_latents, original.windows[frames][1])
    assert selected.source_start_frame == original.source_end_frame - frames
    assert selected.frame_count == frames
    if frames != 56:
        assert not torch.equal(
            selected.video_latents, original.video_latents[:, :, -selected.video_latents.shape[2] :]
        )
    assert selected.provenance == original.provenance
    with pytest.raises(InvalidRequest):
        decode_context(encode_context(original)[:-1])
    with pytest.raises(InvalidRequest):
        decode_context(b"\xff" * 8)
    broken = context()
    broken.audio_latents[0, 0] = float("nan")
    with pytest.raises(InvalidRequest, match="finite"):
        encode_context(broken)


def test_a_context_carries_only_the_windows_its_producer_was_asked_for() -> None:
    single = decode_context(encode_context(context(22, windows=(22,))))
    assert set(single.windows) == {22} and single.select(22).frame_count == 22
    with pytest.raises(InvalidRequest, match="absent"):
        single.select(39)
    with pytest.raises(InvalidRequest, match="selected"):
        context(56, windows=(22,))


@pytest.mark.parametrize("prefix", [0, 22, 39, 56])
def test_streaming_tail_comes_from_delivered_end_not_sample_padding(prefix: int) -> None:
    plan = plan_continuation(240, context_frames=prefix)
    collector = CompletedTail(plan)
    parts = []
    for first in range(0, plan.sample_frames, 17):
        indexes = torch.arange(first, min(first + 17, plan.sample_frames))
        chunk = (indexes.float() / 400)[:, None, None, None].expand(-1, 3, 32, 64)
        part = collector.push(chunk)
        parts.append(part)
    delivered = torch.cat(parts)
    landed = (delivered.clamp(0, 1) * 255).round().byte().permute(0, 2, 3, 1).contiguous()
    soundtrack = torch.arange(500000, dtype=torch.float32)[None].repeat(2, 1) / 500000
    with pytest.raises(InvalidRequest, match="completed"):
        collector.export(landed, soundtrack, 32000, "model")
    collector.finish()
    assert len(delivered) == 240
    assert delivered[0, 0, 0, 0] == prefix / 400
    assert delivered[-1, 0, 0, 0] == (prefix + 239) / 400
    with pytest.raises(InvalidRequest, match="completed"):
        collector.export(landed[1:], soundtrack, 32000, "model")
    tail = collector.export(landed, soundtrack, 32000, "model")
    assert tail.frame_count == 56 and tail.source_end_frame == prefix + 240
    assert np.array_equal(tail.frames, landed[-56:].numpy())
    start, end = audio_sample_range(prefix + 240 - 56, prefix + 240, 32000)
    assert torch.equal(tail.audio, soundtrack[:, start:end])
    assert tail.frames[-1, 0, 0, 0] != round((plan.sample_frames - 1) / 400 * 255)


@pytest.mark.parametrize("frames", [22, 39, 56])
def test_native_context_layout_preserves_refs_and_every_scheduler_step(frames: int) -> None:
    from diffusers import MiniMaxH3Scheduler
    from diffusers.modular_pipelines.minimax_h3 import MiniMaxH3ImageReference
    from diffusers.modular_pipelines.minimax_h3.before_denoise import (
        MiniMaxH3Ref2VAPrepareLayoutStep,
    )
    from diffusers.modular_pipelines.minimax_h3.denoise import MiniMaxH3LoopSchedulerStep

    ctx = context(frames)
    latent_frames = 5 * ((frames - 5) // 17) + 2
    ticks = (frames * 5 + 2) // 3
    refs = [
        MiniMaxH3ImageReference(image=np.zeros((32, 64, 3), dtype=np.uint8)),
        ContextReferenceLayout(),
    ]
    conditions = [torch.zeros(1, 24, 1, 2, 4), torch.zeros(1, 24, latent_frames, 2, 4)]
    audios = [torch.zeros(2 * ticks, 32)]
    packed = MiniMaxH3Ref2VAPrepareLayoutStep.build_ref2va_packed_sequence(
        torch.ones(3, dtype=torch.long), refs, conditions, audios, 20, 2, 4, 120, (1, 2, 2), 2, 2, 0
    )
    positions, _tags, video, audio, text, cv, ca = packed
    before = positions.clone()
    state = SimpleNamespace(
        position_ids=positions,
        video_indices=video,
        audio_indices=audio,
        text_indices=text,
        condition_latents=conditions,
        audio_condition_latents=audios,
        num_condition_video_rows=cv,
        num_condition_audio_rows=ca,
    )
    align_context_rows(state, ctx, (1, 2, 2))
    assert torch.equal(positions[video[:2]], before[video[:2]])
    assert torch.equal(positions[video[cv:]], before[video[cv:]])
    assert torch.equal(positions[text], before[text])
    origin = positions[video[cv], 0]
    offsets = []
    cursor = 0
    for i in range(latent_frames):
        offsets.extend([cursor * 5 / 3] * 2)
        cursor += (1, 4, 4, 4, 4)[i % 5]
    assert torch.allclose(
        positions[video[2:cv], 0],
        origin + torch.tensor(offsets, dtype=torch.float64),
        rtol=0,
        atol=1e-12,
    )
    assert torch.equal(
        positions[audio[:ca], 0], origin + torch.arange(ticks, dtype=torch.float64).repeat(2)
    )
    sched = MiniMaxH3Scheduler(shift=12.0)
    audio_sched = MiniMaxH3Scheduler(shift=3.0)
    sched.set_timesteps(9)
    audio_sched.set_timesteps(9)
    state.latents = torch.ones(len(video), 96)
    state.audio_latents = torch.ones(len(audio), 32)
    state.noise_pred = torch.full((1, len(video), 96), 0.25)
    state.audio_noise_pred = torch.full((1, len(audio), 32), 0.25)
    state.audio_timesteps = audio_sched.timesteps
    components = SimpleNamespace(scheduler=sched, audio_scheduler=audio_sched)
    for index, timestep in enumerate(sched.timesteps):
        MiniMaxH3LoopSchedulerStep()(components, state, i=index, t=timestep)
        assert torch.equal(state.latents[:cv], torch.ones(cv, 96))
        assert torch.equal(state.audio_latents[:ca], torch.ones(ca, 32))
    assert not torch.equal(state.latents[cv:], torch.ones_like(state.latents[cv:]))


def test_export_encodes_only_the_named_windows_of_the_landed_tail() -> None:
    """The real streamed decode and reference encoder at toy widths: a one-window export is
    that window of a full export, read from the frames the output landed."""
    torch.manual_seed(3)
    video_vae = TileBatchedVideoVAE(
        latent_channels=24,
        block_out_channels=(8,) * 6,
        layers_per_block=1,
        norm_num_groups=4,
        decoder_num_layers=1,
        decoder_num_attention_heads=2,
        decoder_attention_head_dim=8,
        decoder_num_register_tokens=1,
        decoder_ffn_mult=1,
        latents_mean=(0.0,) * 24,
        latents_std=(1.0,) * 24,
    ).eval()
    audio_vae = AutoencoderKLMiniMaxH3Audio(
        encoder_dim=4,
        latent_dim=32,
        latent_channels=32,
        num_attention_heads=1,
        decoder_dim=128,
        latents_mean=[0.0] * 32,
        latents_std=[1.0] * 32,
    ).eval()
    blocks = MiniMaxH3Blocks().get_workflow("ref2va")
    upstream = MiniMaxH3ModularPipeline(blocks=blocks)
    upstream.register_components(vae=video_vae, audio_vae=audio_vae)
    pipe = OfficialH3Pipeline.__new__(OfficialH3Pipeline)
    pipe.components = {"video_vae": video_vae, "audio_vae": audio_vae}
    pipe._blocks = {"ref2va": blocks}
    pipe._pipes = {"ref2va": upstream}

    plan = plan_continuation(120, context_frames=22)
    state = PipelineState()
    latent_frames = 5 * ((plan.sample_frames - 5) // 17) + 2
    state.set("latents", torch.randn(1, 24, latent_frames, 2, 4))
    state.set("continuation_delivery", plan)
    landed: list[Any] = []
    delivered = pipe.decode_video_chunks(
        "ref2va",
        state,
        lambda chunk: landed.append((chunk.clamp(0, 1) * 255).round().byte().permute(0, 2, 3, 1)),
    )
    frames = torch.cat(landed)
    assert delivered == len(frames) == 120
    state.set("audio", torch.rand(1, 2, plan.sample_frames * 32000 // 24) - 0.5)
    state.set("sampling_rate", 32000)

    one = pipe.export_completed_av_tail(state, frames=frames, windows=(22,), provenance="p")
    every = pipe.export_completed_av_tail(
        state, frames=frames, windows=(22, 39, 56), provenance="p"
    )
    assert set(one.windows) == {22} and one.frame_count == 22
    assert set(every.windows) == {22, 39, 56} and one.source_end_frame == plan.delivery_end
    for ours, theirs in zip(one.windows[22], every.windows[22], strict=True):
        assert torch.equal(ours, theirs)
    assert decode_context(encode_context(one)).select(22).frame_count == 22
    # The export's encoder, its clips spread as uint8, is Diffusers' reference encoder step
    # over the same window, value for value.
    window = state.completed_av_tail.export(frames, state.audio[0], 32000, "p").select(56)
    official = PipelineState()
    official.set(
        "normalized_references",
        [
            MiniMaxH3VideoReference(
                frames=window.frames, fps=24.0, audio=window.audio, sample_rate=window.sample_rate
            )
        ],
    )
    pipe._run("ref2va", "vae_encoder", official, component="video_vae")
    theirs = (official.condition_latents[0], official.audio_condition_latents[0])
    assert all(torch.equal(a, b.float()) for a, b in zip(every.windows[56], theirs, strict=True))
    with pytest.raises(InvalidRequest):
        pipe.export_completed_av_tail(state, frames=frames, windows=(), provenance="p")
    with pytest.raises(InvalidRequest, match="completed"):
        pipe.export_completed_av_tail(state, frames=frames[1:], windows=(22,), provenance="p")
