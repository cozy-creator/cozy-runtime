"""Real upstream H3 forwards verify schedule/layout delivery, not GPU Sol quality."""

from collections import OrderedDict
from copy import deepcopy
from inspect import signature
from types import SimpleNamespace
from typing import Any

import pytest


@pytest.mark.parametrize("task", ["fl2va_turbo", "ref2va_turbo"])
def test_turbo_loop_delivers_policy_to_every_real_attention_site(task: Any) -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("diffusers")
    from diffusers import MiniMaxH3Transformer3DModel
    from diffusers.modular_pipelines.minimax_h3 import MiniMaxH3Blocks, MiniMaxH3ModularPipeline
    from diffusers.modular_pipelines.minimax_h3.before_denoise import MiniMaxH3PrepareLayoutStep
    from diffusers.modular_pipelines.modular_pipeline import PipelineState

    from cozy_runtime.author._attention_scope import _ACTIVE_LAYOUT, AttentionLayout
    from cozy_runtime.models.minimax_h3.model import H3TurboBase, H3TurboLoRA
    from cozy_runtime.models.minimax_h3.official import OfficialH3Pipeline, canonical_timestep_plan

    # The workflow omits this keyword. Both public Turbo entry paths must agree;
    # the general denoiser retains the standard model's ten-step default.
    default = (
        signature(getattr(H3TurboBase, "sample_" + task)).parameters["sol_dense_steps"].default
    )
    adapter_default = (
        signature(getattr(H3TurboLoRA, "sample_" + task.removesuffix("_turbo")))
        .parameters["sol_dense_steps"]
        .default
    )
    assert default == adapter_default == 4
    assert signature(OfficialH3Pipeline.denoise).parameters["sol_dense_steps"].default == 10

    # Keep the real 50+2 layer topology; only widths and token counts are reduced.
    dit = MiniMaxH3Transformer3DModel(
        num_attention_heads=1,
        attention_head_dim=8,
        hidden_size=8,
        num_layers=50,
        num_refiner_layers=2,
        ffn_dim=16,
        in_channels=1,
        audio_in_channels=2,
        patch_size=(1, 2, 2),
        text_dim=8,
        freq_dim=8,
        time_embed_hidden_dim=8,
        time_embed_dim=8,
        rope_freq_dim=1,
    ).eval()
    workflow = "t2va_turbo" if task == "fl2va_turbo" else task
    blocks = MiniMaxH3Blocks().get_workflow(workflow.removesuffix("_turbo"))
    upstream = MiniMaxH3ModularPipeline(blocks=blocks)
    upstream.register_components(
        **{"transformer" if task == "fl2va_turbo" else "transformer_ref": dit}
    )
    pipe = OfficialH3Pipeline.__new__(OfficialH3Pipeline)
    pipe.components = {task.replace("_turbo", "_dit"): dit}
    # Start at prepared latents. Retain the actual upstream scheduler and denoiser
    # blocks, avoiding irrelevant text/VAE construction and model downloads.
    pipe._blocks = {
        workflow: SimpleNamespace(
            sub_blocks=OrderedDict(
                (name, blocks.sub_blocks[name])
                for name in ("denoise.set_timesteps", "denoise.denoise")
            )
        )
    }
    pipe._pipes = {workflow: upstream}
    pipe._plans = {task: canonical_timestep_plan(task)}
    position, tags, video, audio, text, cv, ca = MiniMaxH3PrepareLayoutStep.build_packed_sequence(
        text_token_tags=torch.ones(3, dtype=torch.long),
        num_latent_frames=2,
        latent_height=4,
        latent_width=6,
        num_audio_latents=4,
        patch_size=(1, 2, 2),
        audio_channels=2,
        audio_tag=2,
        video_tag=0,
        keyframe_anchors=(),
    )
    generator = torch.Generator().manual_seed(88)
    state = PipelineState()
    for name, value in {
        "num_inference_steps": 9,  # Eight transformer evaluations plus terminal sigma.
        "num_condition_video_rows": cv,
        "num_condition_audio_rows": ca,
        "latents": torch.randn(len(video), 4, generator=generator),
        "audio_latents": torch.randn(len(audio), 2, generator=generator),
        "prompt_embeds": torch.randn(1, len(text), 8, generator=generator),
    }.items():
        state.set(name, value)
    for name, value in {
        "position_ids": position,
        "token_tags": tags,
        "video_indices": video,
        "audio_indices": audio,
        "text_indices": text,
    }.items():
        state.set(name, value, kwargs_type="denoiser_input_fields")

    seen: list[tuple[str, AttentionLayout]] = []

    def observe(path: str) -> Any:
        def before(_module: Any, _args: Any) -> None:
            layout = _ACTIVE_LAYOUT.get()
            assert layout is not None
            seen.append((path, layout))

        return before

    hooks = [
        module.register_forward_pre_hook(observe(path))
        for path, module in dit.named_modules()
        if type(module).__name__ == "MiniMaxH3Attention"
    ]
    assert len(hooks) == 52

    class DenoisingComplete(Exception):
        pass

    steps: list[int] = []

    def on_step(step: int) -> None:
        steps.append(step)
        if step == 7:
            raise DenoisingComplete  # Stop before unrelated VAE reshaping/decoding.

    initial = state
    baseline: tuple[Any, Any] | None = None
    prior_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        for dense_steps in (default, 3, 8, 10):
            state = deepcopy(initial)
            seen.clear()
            steps.clear()
            with pytest.raises(DenoisingComplete):
                pipe.denoise(
                    task,
                    state,
                    on_step=on_step,
                    cancel=lambda: None,
                    sol_dense_steps=dense_steps,
                )
            assert _ACTIVE_LAYOUT.get() is None
            assert steps == list(range(8))
            assert len(seen) == 8 * 52
            for step in range(8):
                rows = seen[step * 52 : (step + 1) * 52]
                assert len({path for path, _layout in rows}) == 52
                for _path, layout in rows:
                    assert layout.step == step
                    assert layout.dense_until_step == dense_steps
                    assert layout.protected_prefix == int(video[0])
                    assert layout.live_tokens == len(tags)
                    assert layout.dense_paths == (
                        "token_refiner",
                        "transformer_blocks.0",
                        "transformer_blocks.1",
                    )
            outputs = (state.get("latents"), state.get("audio_latents"))
            assert all(torch.isfinite(value).all() for value in outputs)
            # No Sol was selected: changing only its policy leaves the real dense
            # CPU SDPA result equal at zero tolerance. FA3/Sol CUDA arithmetic is a GPU gate.
            if baseline is None:
                baseline = outputs
            else:
                for actual, expected in zip(outputs, baseline, strict=True):
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        seen.clear()
        trunk = task.removesuffix("_turbo")
        pipe._plans[trunk] = canonical_timestep_plan(trunk)
        pipe.warm_dit(trunk)
        assert len(seen) == 52
        assert all(layout.step == 0 and layout.dense_until_step == 10 for _, layout in seen)
        assert _ACTIVE_LAYOUT.get() is None
    finally:
        torch.set_num_threads(prior_threads)
        for hook in hooks:
            hook.remove()


@pytest.mark.parametrize("invalid", [-1, True, False, 3.0, "3", None])
def test_invalid_policy_refuses_before_pipeline_or_model_access(invalid: Any) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("diffusers")
    from cozy_runtime.models.minimax_h3.model import H3Model, H3TurboLoRA
    from cozy_runtime.models.minimax_h3.official import NumericalChecks, OfficialH3Pipeline

    adapter = H3TurboLoRA.__new__(H3TurboLoRA)
    with pytest.raises(ValueError, match="sol_dense_steps must be a nonnegative integer"):
        adapter._sample(
            H3Model.__new__(H3Model),
            "fl2va",
            None,
            on_step=lambda _: None,
            cancel=lambda: None,
            checks=NumericalChecks.__new__(NumericalChecks),
            sol_dense_steps=invalid,
        )
    pipe = OfficialH3Pipeline.__new__(OfficialH3Pipeline)
    with pytest.raises(ValueError, match="sol_dense_steps must be a nonnegative integer"):
        pipe.denoise(
            "fl2va_turbo",
            None,
            on_step=lambda _: None,
            cancel=lambda: None,
            sol_dense_steps=invalid,
        )
