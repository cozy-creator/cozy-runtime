"""H3 continuation proofs over the real packing, scheduler, unpack and streamed decode."""

from __future__ import annotations

from fractions import Fraction
from itertools import pairwise
from types import SimpleNamespace
from typing import Any

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")

from diffusers import MiniMaxH3Scheduler  # noqa: E402
from diffusers.models.autoencoders.autoencoder_kl_minimax_h3_audio import (  # noqa: E402
    AutoencoderKLMiniMaxH3Audio,
)
from diffusers.modular_pipelines.minimax_h3 import (  # noqa: E402
    MiniMaxH3Blocks,
    MiniMaxH3ImageReference,
    MiniMaxH3ModularPipeline,
)
from diffusers.modular_pipelines.minimax_h3.before_denoise import (  # noqa: E402
    MiniMaxH3Ref2VAPrepareLayoutStep,
    MiniMaxH3SetTimestepsStep,
    patchify_video_latents,
)
from diffusers.modular_pipelines.minimax_h3.decoders import (  # noqa: E402
    MiniMaxH3AfterDenoiseStep,
)
from diffusers.modular_pipelines.minimax_h3.denoise import (  # noqa: E402
    MiniMaxH3LoopSchedulerStep,
)
from diffusers.modular_pipelines.modular_pipeline import PipelineState  # noqa: E402

from cozy_runtime.author import InvalidRequest  # noqa: E402
from cozy_runtime.models.minimax_h3.continuation import (  # noqa: E402
    AUDIO_TICKS,
    AVContext,
    Carry,
    ContextReferenceLayout,
    ContinuationPlan,
    PrefixNoise,
    align_context_rows,
    audio_ticks,
    decode_context,
    encode_context,
    export_context,
    video_tokens,
)
from cozy_runtime.models.minimax_h3.official import OfficialH3Pipeline  # noqa: E402
from cozy_runtime.models.minimax_h3.vae_tiles import TileBatchedVideoVAE  # noqa: E402

RATE = 32000
PICTURE, SOUND, BOTH = Carry(22), Carry(audio=True), Carry(22, True)


def chain(seconds: list[int], carries: list[Carry]) -> list[ContinuationPlan]:
    carries = [Carry(), *carries]
    return [
        ContinuationPlan(s, carried, successor)
        for s, (carried, successor) in zip(seconds, pairwise([*carries, Carry()]), strict=True)
    ]


@pytest.mark.parametrize(
    "seconds,carried,sampled,delivered",
    [
        (15, Carry(), 362, 362),
        (14, PICTURE, 362, 340),
        (10, PICTURE, 277, 255),
        (15, SOUND, 362, 362),
    ],
)
def test_a_segment_delivers_its_whole_sample_after_the_regenerated_head(
    seconds: int, carried: Carry, sampled: int, delivered: int
) -> None:
    plan = ContinuationPlan(seconds, carried)
    assert (plan.sample_frames, plan.delivered_frames) == (sampled, delivered)
    assert plan.delivered_frames >= 24 * seconds
    with pytest.raises(InvalidRequest, match="362"):
        ContinuationPlan(15, PICTURE)
    with pytest.raises(InvalidRequest):
        Carry(32)


def test_a_sound_only_successor_ends_the_picture_on_the_last_frame_its_sound_covers() -> None:
    # 362 frames is 603.33 ticks; the grid rounds to 603, so frame 362 has no sound.
    assert audio_ticks(362) == 603
    assert ContinuationPlan(15, successor=SOUND).delivery_end == 361
    assert ContinuationPlan(15, successor=SOUND).exit_offset == Fraction(1, 3)
    assert ContinuationPlan(15, successor=SOUND).sound_end == 602
    assert ContinuationPlan(15, successor=BOTH).delivery_end == 362
    assert ContinuationPlan(15, successor=BOTH).exit_offset == Fraction(-1, 3)
    assert ContinuationPlan(14, successor=SOUND).exit_offset == 0  # 345 frames: 575 ticks


@pytest.mark.parametrize(
    "carries",
    [
        [BOTH] * 4,
        [SOUND] * 4,
        [PICTURE] * 4,
        [BOTH, SOUND, Carry(), PICTURE],
        [SOUND, BOTH, SOUND, BOTH],
    ],
)
def test_joined_sound_is_continuous_and_never_drifts_from_the_picture(carries: list[Carry]) -> None:
    """Each segment's delivered sound starts where its predecessor's ended on that clip's own
    clock, so concatenation is continuous; the clocks telescope, so it never drifts."""
    plans = chain([15, 14, 14, 13, 14], carries)
    picture = sound = 0
    entry = Fraction(0)
    for plan, carried in zip(plans, [Carry(), *carries], strict=True):
        start, end = plan.audio_samples(entry, RATE)
        assert start == round(
            (Fraction(5, 3) * plan.prefix_frames + (entry if carried.audio else 0)) * 800
        )
        sound += end - start
        picture += plan.delivered_frames
        # Within a third of a tick at every join, exact (to rounding) where no sound continues.
        assert abs(sound - picture * Fraction(RATE, 24)) <= Fraction(RATE, 120) + 1
        entry = plan.exit_offset
    assert abs(sound - picture * Fraction(RATE, 24)) <= len(plans)


def context(carry: Carry, offset: Fraction = Fraction(0), *, value: float = 0.5) -> AVContext:
    video = torch.full((1, 24, video_tokens(carry.frames), 2, 4), value) if carry.frames else None
    audio = torch.full((2 * AUDIO_TICKS, 32), value) if carry.audio else None
    return AVContext(video, audio, offset if carry.audio else Fraction(0), "renderer")


@pytest.mark.parametrize("carry", [PICTURE, SOUND, BOTH, Carry(56, True)])
def test_context_codec_round_trips_exactly_what_a_successor_carries(carry: Carry) -> None:
    original = context(carry, Fraction(-1, 3))
    restored = decode_context(encode_context(original))
    assert restored.carry == carry and restored.audio_offset == original.audio_offset
    for ours, theirs in ((restored.video, original.video), (restored.audio, original.audio)):
        assert (ours is None and theirs is None) or torch.equal(ours, theirs)
    assert restored.provenance == "renderer"
    with pytest.raises(InvalidRequest):
        decode_context(encode_context(original)[:-1])
    with pytest.raises(InvalidRequest):
        decode_context(b"\xff" * 8)


def test_context_refuses_what_no_segment_produces() -> None:
    with pytest.raises(InvalidRequest, match="neither"):
        AVContext(None, None, Fraction(0), "renderer")
    with pytest.raises(InvalidRequest, match="offset"):
        context(SOUND, Fraction(1, 2))
    with pytest.raises(InvalidRequest, match="offset"):
        context(SOUND, Fraction(2, 3))
    with pytest.raises(InvalidRequest, match="finite"):
        context(SOUND, value=float("nan"))
    with pytest.raises(InvalidRequest, match="window"):
        AVContext(torch.zeros(1, 24, 8, 2, 4), None, Fraction(0), "renderer")


@pytest.mark.parametrize("successor", [PICTURE, SOUND, BOTH, Carry(56)])
def test_export_slices_the_successors_carry_from_the_unpacked_denoised_latents(
    successor: Carry,
) -> None:
    """The real after-denoise unpack, then the export: the picture is the last whole cycles of
    the sampled video latents and the sound the last 40 ticks, as packed rows."""
    plan = ContinuationPlan(15, successor=successor)  # 362 frames, 603.33 ticks rounded to 603
    frames, ticks = video_tokens(plan.sample_frames), audio_ticks(plan.sample_frames)
    video = torch.arange(24 * frames * 2 * 4, dtype=torch.float32).reshape(1, 24, frames, 2, 4)
    audio = torch.arange(2 * 32 * ticks, dtype=torch.float32).reshape(2, 32, ticks)
    rows = torch.cat([torch.full((5, 96), -1.0), patchify_video_latents(video, (1, 2, 2))])
    packed_audio = torch.cat([torch.full((3, 32), -1.0), audio.permute(0, 2, 1).reshape(-1, 32)])
    state = PipelineState()
    for name, value in dict(
        latents=rows,
        audio_latents=packed_audio,
        num_condition_video_rows=5,
        num_condition_audio_rows=3,
        num_latent_frames=frames,
        latent_height=2,
        latent_width=4,
        num_audio_latents=ticks,
    ).items():
        state.set(name, value)
    components = SimpleNamespace(patch_size=(1, 2, 2), vae_latent_channels=24, audio_channels=2)
    MiniMaxH3AfterDenoiseStep()(components, state)
    assert torch.equal(state.latents, video) and torch.equal(state.audio_latents, audio)
    carried = export_context(state.latents, state.audio_latents, plan, "renderer")
    assert carried is not None and carried.carry == successor
    if successor.frames:
        tokens = video_tokens(successor.frames)
        assert torch.equal(carried.video, video[:, :, -tokens:])
        assert (frames - tokens) % 5 == 0
    if successor.audio:
        end = int(plan.sound_end)
        assert torch.equal(
            carried.audio,
            audio[:, :, end - AUDIO_TICKS : end].permute(0, 2, 1).reshape(-1, 32),
        )
        assert carried.audio_offset == plan.exit_offset
    assert export_context(state.latents, state.audio_latents, ContinuationPlan(5), "r") is None


@pytest.mark.parametrize(
    "carry,offset", [(BOTH, Fraction(-1, 3)), (SOUND, Fraction(1, 3)), (Carry(39), Fraction(0))]
)
def test_carried_rows_land_on_this_clips_timeline_and_stay_fixed(
    carry: Carry, offset: Fraction
) -> None:
    carried = context(carry, offset)
    refs = [
        MiniMaxH3ImageReference(image=np.zeros((32, 64, 3), dtype=np.uint8)),
        ContextReferenceLayout.of(carried),
    ]
    conditions = [torch.zeros(1, 24, 1, 2, 4)] + (
        [carried.video] if carried.video is not None else []
    )
    audios = [carried.audio] if carried.audio is not None else []
    packed = MiniMaxH3Ref2VAPrepareLayoutStep.build_ref2va_packed_sequence(
        torch.ones(3, dtype=torch.long), refs, conditions, audios, 20, 2, 4, 120, (1, 2, 2), 2, 2, 0
    )
    positions, _tags, video, audio, text, cv, ca = packed
    before = positions.clone()
    state = SimpleNamespace(
        position_ids=positions,
        video_indices=video,
        audio_indices=audio,
        num_condition_video_rows=cv,
        num_condition_audio_rows=ca,
    )
    align_context_rows(state, carried, (1, 2, 2))
    assert torch.equal(positions[video[:2]], before[video[:2]])
    assert torch.equal(positions[video[cv:]], before[video[cv:]])
    assert torch.equal(positions[audio[ca:]], before[audio[ca:]])
    assert torch.equal(positions[text], before[text])
    origin = positions[video[cv], 0]
    if carried.video is not None:
        # In phase with the target head: token k at 5/3 of its first frame.
        tokens = video_tokens(carry.frames)
        starts = np.cumsum([0] + [(1, 4, 4, 4, 4)[i % 5] for i in range(tokens - 1)])
        expected = origin + torch.tensor(starts * 5 / 3, dtype=torch.float64).repeat_interleave(2)
        assert torch.allclose(positions[video[2:cv], 0], expected, rtol=0, atol=1e-12)
        assert torch.equal(positions[video[2:cv], 0], positions[video[cv : cv + 2 * tokens], 0])
    if carried.audio is not None:
        end = origin + float(Fraction(5, 3) * carry.frames + offset)
        ticks = torch.arange(AUDIO_TICKS, dtype=torch.float64).repeat(2)
        assert torch.allclose(
            positions[audio[:ca], 0], end - AUDIO_TICKS + ticks, rtol=0, atol=1e-9
        )
    sched, audio_sched = MiniMaxH3Scheduler(shift=12.0), MiniMaxH3Scheduler(shift=3.0)
    sched.set_timesteps(9)
    audio_sched.set_timesteps(9)
    step = SimpleNamespace(
        latents=torch.ones(len(video), 96),
        audio_latents=torch.ones(len(audio), 32),
        noise_pred=torch.full((1, len(video), 96), 0.25),
        audio_noise_pred=torch.full((1, len(audio), 32), 0.25),
        audio_timesteps=audio_sched.timesteps,
        num_condition_video_rows=cv,
        num_condition_audio_rows=ca,
    )
    for index, timestep in enumerate(sched.timesteps):
        MiniMaxH3LoopSchedulerStep()(
            SimpleNamespace(scheduler=sched, audio_scheduler=audio_sched), step, i=index, t=timestep
        )
        assert torch.equal(step.latents[:cv], torch.ones(cv, 96))
        assert torch.equal(step.audio_latents[:ca], torch.ones(ca, 32))


def test_streamed_decode_delivers_the_plans_picture_and_sound() -> None:
    """The real streamed decode and audio decode at toy widths: the regenerated head is cut,
    the sound is cut on its own clock from the predecessor's end, padded only past a grid
    that rounds the sound short."""
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

    def decode(plan: ContinuationPlan, carried: AVContext | None) -> tuple[Any, Any]:
        state = PipelineState()
        state.set("latents", torch.randn(1, 24, video_tokens(plan.sample_frames), 2, 4))
        state.set("audio_latents", torch.randn(2, 32, audio_ticks(plan.sample_frames)))
        state.set("continuation_delivery", plan)
        state.set("continuation_context", carried)
        landed: list[Any] = []
        delivered = pipe.decode_video_chunks("ref2va", state, landed.append)
        assert delivered == sum(len(chunk) for chunk in landed) == plan.delivered_frames
        with torch.no_grad():
            sound = pipe.decode_audio("ref2va", state)
        assert state.audio.shape[-1] == audio_ticks(plan.sample_frames) * 800
        return sound, state.audio

    plan = ContinuationPlan(5, BOTH, SOUND)
    sound, full = decode(plan, context(BOTH, Fraction(-1, 3)))
    start, end = plan.audio_samples(Fraction(-1, 3), RATE)
    # 158 frames: 263 ticks; the picture ends at 157 (261.67 ticks), the sound at tick 262.
    assert start == round((Fraction(110, 3) - Fraction(1, 3)) * 800)
    assert end == 262 * 800 == full.shape[-1] - 800
    assert torch.equal(sound, full[..., start:end])
    sound, full = decode(ContinuationPlan(15), None)  # 362 frames, 603 ticks
    assert sound.shape[-1] == round(Fraction(362 * RATE, 24)) == full.shape[-1] + 267
    assert torch.equal(sound[..., : full.shape[-1]], full)
    assert not sound[..., full.shape[-1] :].any()


def test_prefix_noise_labels_carried_tokens_with_schedule_levels_tapering_to_the_anchor() -> None:
    """Far tokens ride the target head's own noise at the next level, the last four step down
    the schedule to the anchor; only schedule or anchor timesteps, so pruned tables cover it."""
    carried = context(PICTURE)
    refs = [
        MiniMaxH3ImageReference(image=np.zeros((32, 64, 3), dtype=np.uint8)),
        ContextReferenceLayout.of(carried),
    ]
    packed = MiniMaxH3Ref2VAPrepareLayoutStep.build_ref2va_packed_sequence(
        torch.ones(3, dtype=torch.long),
        refs,
        [torch.zeros(1, 24, 1, 2, 4), carried.video],
        [],
        20,
        2,
        4,
        120,
        (1, 2, 2),
        2,
        2,
        0,
    )
    _positions, _tags, video, audio, text, cv, ca = packed
    sched = MiniMaxH3Scheduler(shift=12.0)
    sched.set_timesteps(9)
    plan = [
        tuple(
            MiniMaxH3SetTimestepsStep.build_row_timesteps(
                video, audio, cv, ca, len(text), float(t), float(t), max(float(t), 0.999), 1.0
            )
        )
        for t in sched.timesteps
    ]
    latents = torch.randn(len(video), 96)
    state = SimpleNamespace(
        latents=latents.clone(),
        num_condition_video_rows=cv,
        timesteps=sched.timesteps,
        row_timestep_plan=list(plan),
        denoiser_input_fields={"video_indices": video},
    )
    prefix = PrefixNoise(state, carried, (1, 2, 2), 0.999)
    allowed = {*sched.timesteps.tolist(), torch.tensor(0.999).item()}
    rows = slice(cv - 14, cv)
    for index in range(len(sched.timesteps)):
        prefix.apply(state, index)
        unique, inverse = state.row_timestep_plan[index]
        labels = unique[inverse][video[rows]]
        assert set(labels.tolist()) <= allowed
        anchor = torch.tensor(0.999).item()
        ahead = [
            sched.timesteps[index + k].item() if index + k < 8 else anchor for k in (1, 2, 3, 4)
        ]
        assert labels.reshape(7, 2)[:, 0].tolist() == [*ahead[:1] * 3, *ahead[1:], anchor]
        weight = labels[:, None]
        assert torch.allclose(
            state.latents[rows], weight * latents[rows] + (1 - weight) * latents[cv : cv + 14]
        )
        others = torch.ones(len(unique[inverse]), dtype=torch.bool)
        others[video[rows]] = False
        before_unique, before_inverse = plan[index]
        assert torch.equal(unique[inverse][others], before_unique[before_inverse][others])
    assert torch.equal(state.latents[cv:], latents[cv:])
