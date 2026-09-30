"""Real H3 video/audio forwards retain generic adapters and the model-owned Turbo overlay."""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import pytest

if TYPE_CHECKING:
    from torch import Tensor, nn


@pytest.mark.parametrize("component", ["fl2va_dit", "ref2va_dit"])
@pytest.mark.parametrize("turbo", [False, True])
def test_real_h3_forward_matches_ordered_reference_adapters(component: str, turbo: bool) -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("peft")
    pytest.importorskip("diffusers")
    from cozy_runtime.internal.lora_composition import composer
    from cozy_runtime.internal.lora_contract import FORMAT, Graph, Linear
    from diffusers.modular_pipelines.minimax_h3.before_denoise import MiniMaxH3PrepareLayoutStep

    from cozy_runtime.author import AdapterRef
    from cozy_runtime.models.minimax_h3.adaln_pruned import AdaLNPrunedMiniMaxH3Transformer
    from cozy_runtime.models.minimax_h3.turbo import (
        ATTENTION_KWARG,
        OVERLAY_KWARG,
        TURBO_BANK,
        TurboOverlay,
        TurboSchedule,
    )

    torch.manual_seed(292)
    times = (0.0, 0.25, 0.75)
    keys = [(row, tag) for row in range(3) for tag in range(3)]
    original = AdaLNPrunedMiniMaxH3Transformer(
        table_timesteps=times,
        table_block_keys=keys,
        num_attention_heads=2,
        attention_head_dim=8,
        hidden_size=16,
        num_layers=1,
        num_refiner_layers=1,
        ffn_dim=32,
        in_channels=2,
        audio_in_channels=2,
        patch_size=(1, 1, 1),
        text_dim=4,
        freq_dim=8,
        time_embed_hidden_dim=16,
        time_embed_dim=8,
        rope_freq_dim=1,
    ).eval()
    original.install_lora_consumers()
    overlay = TurboOverlay(
        hidden_size=16,
        inner_dim=16,
        ffn_dim=32,
        num_layers=1,
        num_refiner_layers=1,
        video_out=2,
        audio_out=2,
        rank=1,
        alpha=1,
        schedule=TurboSchedule((0.75, 0.0), (0.25, 0.0)),
        table_timesteps=times,
        table_block_keys=keys,
        block_table_dtype=torch.float32,
        final_table_dtype=torch.float32,
    ).eval()
    with torch.no_grad():
        # Both table constructors intentionally allocate weight values without initialization.
        for model in (original, overlay):
            for parameter in model.parameters():
                parameter.normal_(0, 0.1)
    expected, actual = deepcopy(original), deepcopy(original)
    paths = (
        "proj_in",
        "audio_proj_in",
        "context_embedder",
        "token_refiner.refiner_blocks.0.attn.to_q",
        "transformer_blocks.0.attn.to_q",
        "transformer_blocks.0.attn.to_out.0",
        "transformer_blocks.0.ff.net.2",
        "proj_out",
        "audio_proj_out",
    )
    rows = tuple(
        Linear(component, path, f"adapter_{index}", rank, alpha, strength, dtype)
        for path in paths
        for index, (rank, alpha, strength, dtype) in enumerate(
            ((1, 2.0, 0.5, "f32"), (2, 1.0, -0.25, "f16"))
        )
    )
    graph = Graph(
        FORMAT,
        rows,
        tuple(
            AdapterRef("sha256:" + digest * 32, strength, "lora", component, "adapter")
            for digest, strength in (("12", 0.5), ("34", -0.25))
        ),
    )
    composer(graph)(SimpleNamespace(components={component: actual}))
    from cozy_runtime.internal import attention

    for path in ("token_refiner.refiner_blocks.0.attn", "transformer_blocks.0.attn"):
        assert (
            attention._dtype(actual.get_submodule(path))
            == attention._dtype(original.get_submodule(path))
            == "float32"
        )
    untouched = {name: value.clone() for name, value in original.state_dict().items()}

    def reference_hook(
        factors: tuple[tuple[Tensor, Tensor, float], ...],
    ) -> Callable[[nn.Module, tuple[Tensor, ...], Tensor], Tensor]:
        def apply(_module: nn.Module, operands: tuple[Tensor, ...], output: Tensor) -> Tensor:
            value = operands[0]
            result = output
            for a, b, scaling in factors:
                value = value.to(a.dtype)
                result = (
                    result
                    + torch.nn.functional.linear(torch.nn.functional.linear(value, a), b) * scaling
                )
            return result.to(output.dtype)

        return apply

    hooks = []
    with torch.no_grad():
        for path in paths:
            layer = actual.get_submodule(path)
            factors = []
            for row in (row for row in rows if row.target == path):
                a = cast("Tensor", layer.lora_A[row.adapter].weight)
                b = cast("Tensor", layer.lora_B[row.adapter].weight)
                a.normal_(0, 0.1)
                b.normal_(0, 0.1)
                factors.append((a.clone(), b.clone(), row.alpha / row.rank * row.strength))
            hooks.append(
                expected.get_submodule(path).register_forward_hook(reference_hook(tuple(factors)))
            )

    position, tags, video, audio, text, _cv, _ca = MiniMaxH3PrepareLayoutStep.build_packed_sequence(
        text_token_tags=torch.ones(3, dtype=torch.long),
        num_latent_frames=2,
        latent_height=2,
        latent_width=2,
        num_audio_latents=4,
        patch_size=(1, 1, 1),
        audio_channels=2,
        audio_tag=2,
        video_tag=0,
        keyframe_anchors=(),
    )
    prior_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        with torch.no_grad():
            selector = {ATTENTION_KWARG: TURBO_BANK, OVERLAY_KWARG: overlay} if turbo else {}
            inputs = dict(
                hidden_states=torch.randn(1, len(video), 2),
                audio_hidden_states=torch.randn(1, len(audio), 2),
                encoder_hidden_states=torch.randn(1, len(text), 4),
                timestep=torch.tensor([0.75, 0.25]),
                timestep_indices=torch.where(tags == 2, 1, 0),
                token_tags=tags,
                position_ids=position,
                video_indices=video,
                audio_indices=audio,
                text_indices=text,
                attention_kwargs=selector,
                return_dict=False,
            )
            wanted = expected(**inputs)
            observed = actual(**inputs)
            baseline = original(**inputs)
            for got, want in zip(observed, wanted, strict=True):
                assert torch.isfinite(got).all()
                torch.testing.assert_close(got, want, rtol=0, atol=0)
            assert all(
                not torch.equal(got, plain) for got, plain in zip(observed, baseline, strict=True)
            )
            assert observed[0].shape == (1, len(video), 2)
            assert observed[1].shape == (1, len(audio), 2)
            assert actual._arming is None
            assert actual.proj_out.base_layer._armed is None
            assert actual.audio_proj_out.base_layer._armed is None
            after = {
                name.replace(".base_layer.", "."): value
                for name, value in actual.state_dict().items()
                if ".lora_" not in name
            }
            assert set(after) == set(untouched)
            assert all(torch.equal(value, after[name]) for name, value in untouched.items())
            repeated = actual(**inputs)
            assert all(torch.equal(a, b) for a, b in zip(observed, repeated, strict=True))
            assert not torch.cuda.is_initialized()
    finally:
        torch.set_num_threads(prior_threads)
        for hook in hooks:
            hook.remove()
