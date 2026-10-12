"""Retained H3 conditioning and the real upstream refinement loop, on CPU."""
from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")
from diffusers import MiniMaxH3Blocks, MiniMaxH3ModularPipeline, MiniMaxH3Transformer3DModel  # noqa: E402
from diffusers.modular_pipelines.minimax_h3.modular_pipeline import audio_latent_num_frames  # noqa: E402

from cozy_runtime.author import InvalidRequest  # noqa: E402
from cozy_runtime.models.minimax_h3.official import OfficialH3Pipeline, canonical_timestep_plan  # noqa: E402
from cozy_runtime.models.minimax_h3.refinement import (  # noqa: E402
    H3Latents, ReferenceLayout, capture_latents, decode_latents, encode_latents,
    refinement_state, resize_video_chunk,
)


def artifact(task: str, anchored: bool = False) -> H3Latents:
    generator = torch.Generator().manual_seed(42)
    random = lambda shape: torch.randn(shape, generator=generator)
    conditions = (random((1, 24, 1, 2, 4)),) if task == "ref2va" or anchored else ()
    return H3Latents(
        random((1, 24, 37, 2, 4)), random((2, 32, audio_latent_num_frames(124))),
        random((1, 2, 124 * 32000 // 24)), random((1, 3, 8)), torch.ones(3, dtype=torch.long),
        conditions, (), (ReferenceLayout("image"),) if task == "ref2va" else (),
        ("first",) if anchored else (), task, 124, 42, "model@1.0.0/original",  # type: ignore[arg-type]
    )


def pipeline(task: str) -> OfficialH3Pipeline:
    dit = MiniMaxH3Transformer3DModel(
        num_attention_heads=1, attention_head_dim=8, hidden_size=8, num_layers=2,
        num_refiner_layers=1, ffn_dim=16, in_channels=24, audio_in_channels=32,
        patch_size=(1, 2, 2), text_dim=8, freq_dim=8, time_embed_hidden_dim=8,
        time_embed_dim=8, rope_freq_dim=1,
    ).eval()
    workflows = [task, "t2va"] if task == "fl2va" else [task]
    pipe = object.__new__(OfficialH3Pipeline)
    pipe.components = {f"{task}_dit": dit}
    pipe._blocks = {name: MiniMaxH3Blocks().get_workflow(name) for name in workflows}
    pipe._pipes = {}
    for name, blocks in pipe._blocks.items():
        upstream = MiniMaxH3ModularPipeline(blocks=blocks)
        upstream.register_components(**{"transformer_ref" if task == "ref2va" else "transformer": dit})
        pipe._pipes[name] = upstream
    pipe._plans = {task: canonical_timestep_plan(task)}  # type: ignore[dict-item,arg-type]
    pipe._dit_specs = {task: (None, None, None)}  # type: ignore[dict-item]
    return pipe


@pytest.mark.parametrize("task,anchored", [("fl2va", False), ("fl2va", True), ("ref2va", False)])
def test_retained_generation_refines_with_real_upstream_blocks_and_keeps_audio(task: Any, anchored: bool) -> None:
    initial = artifact(task, anchored)
    restored = decode_latents(encode_latents(initial))
    original = encode_latents(restored)
    pipe = pipeline(task)
    state = refinement_state(restored, torch.Generator().manual_seed(77))
    calls: list[tuple[Any, Any]] = []

    def inspect_audio(_module: Any, _args: Any, kwargs: Any) -> None:
        tags, timesteps = kwargs["token_tags"], kwargs["timestep"]
        actual = timesteps[kwargs["timestep_indices"]]
        assert torch.all(actual[tags == 2] == 1), "refinement must treat retained audio as clean"
        calls.append((kwargs["hidden_states"].detach().clone(), kwargs["audio_hidden_states"].detach().clone()))

    hook = pipe.components[f"{task}_dit"].register_forward_pre_hook(inspect_audio, with_kwargs=True)
    prior_threads = torch.get_num_threads()
    steps: list[int] = []
    try:
        torch.set_num_threads(1)
        facts = pipe.denoise(task, state, on_step=steps.append, cancel=lambda: None, refine_steps=2)
    finally:
        hook.remove()
        torch.set_num_threads(prior_threads)
    assert steps == [0, 1] and facts.transformer_evaluations == 2
    assert facts.sigma_grid_points == 3 and len(calls) == 2
    assert torch.equal(calls[0][1], calls[1][1])
    assert torch.equal(state.audio_latents.cpu(), initial.audio)
    assert torch.equal(pipe.decode_audio(task, state), initial.waveform)
    assert state.latents.shape == initial.video.shape and torch.isfinite(state.latents).all()
    assert not torch.equal(state.latents, initial.video)
    assert encode_latents(restored) == original, "refining must not mutate a reusable baseline"
    saved = capture_latents(task, state, checkpoint=initial.checkpoint, seed=77)
    assert torch.equal(decode_latents(encode_latents(saved)).waveform, initial.waveform)


def test_latent_carrier_rejects_geometry_and_preserves_conditioning() -> None:
    value = artifact("ref2va")
    roundtrip = decode_latents(encode_latents(value))
    assert torch.equal(roundtrip.conditions[0], value.conditions[0])
    assert roundtrip.references == value.references
    with pytest.raises(InvalidRequest, match="time axis"):
        encode_latents(replace(value, video=value.video[:, :, :-1]))
    with pytest.raises(InvalidRequest, match="32-pixel grid"):
        encode_latents(replace(value, video=torch.zeros(1, 24, 37, 3, 4)))
    with pytest.raises(InvalidRequest):
        decode_latents(b"not a tensor carrier")


def test_standard_resize_preserves_time_and_aspect_with_exact_output_canvas() -> None:
    source = torch.ones(2, 3, 32, 56)
    result = resize_video_chunk(source, (192, 108))
    assert result.shape == (2, 3, 108, 192)
    assert torch.equal(source, torch.ones_like(source))
    assert torch.all(result[..., 1:190] > 0.99)
    assert torch.all(result[..., 0] == 0) and torch.all(result[..., -1] == 0)
