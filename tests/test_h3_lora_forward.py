"""Real H3 video/audio forwards retain generic adapters and the model-owned Turbo overlay."""

from __future__ import annotations

import os
import socket
import sys
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("peft")
pytest.importorskip("diffusers")

from torch.utils._python_dispatch import TorchDispatchMode  # noqa: E402
from torch.utils._pytree import tree_leaves  # noqa: E402

from cozy_runtime.internal import spawn  # noqa: E402
from cozy_runtime.internal.executor import Executor  # noqa: E402
from cozy_runtime.internal.parallel.group import RankGroup  # noqa: E402
from cozy_runtime.internal.seam import Channel  # noqa: E402
from testdata import h3_lora_runtime  # noqa: E402

if TYPE_CHECKING:
    from torch import Tensor, nn


@pytest.mark.parametrize("component", ["fl2va_dit", "ref2va_dit"])
@pytest.mark.parametrize("turbo", [False, True])
def test_real_h3_forward_matches_ordered_reference_adapters(component: str, turbo: bool) -> None:
    from diffusers.modular_pipelines.minimax_h3.before_denoise import MiniMaxH3PrepareLayoutStep

    from cozy_runtime.author import AdapterRef
    from cozy_runtime.internal.lora_composition import composer
    from cozy_runtime.internal.lora_contract import FORMAT, Graph, Linear
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
            # The exact update, in float64; served updates round once into the output.
            value, result = operands[0].double(), output.double()
            for a, b, scaling in factors:
                result += value @ a.double().T @ b.double().T * scaling
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
                torch.testing.assert_close(got, want)
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


@pytest.mark.parametrize(
    ("degree", "dtype"), [(2, torch.float32), (4, torch.float32), (2, torch.float16)]
)
def test_a_prepared_lora_view_serves_its_exact_factors_on_every_gpu(
    tmp_path: Path, degree: int, dtype: Any
) -> None:
    """The production path end to end: the adapter-view child composes a real fp32 LoRA onto a
    bf16 H3 checkpoint, the executor's bind/substrate/plane constructs and fills it, and the
    group's real followers construct their own replica. Every GPU must hold the stored factors
    and scale, a group must compute what one GPU does, and both what merged weights compute."""
    root = tmp_path / "store"
    view, values, factors = h3_lora_runtime.checkpoints(root, dtype)
    model, backend = h3_lora_runtime.construct(root, view.digest)
    held = h3_lora_runtime.adapters(model)
    assert list(held) == h3_lora_runtime.targets(h3_lora_runtime.dit())
    for path, (a, b, scaling, active) in held.items():
        assert torch.equal(a, factors[path + ".lora_A.weight"]), path
        assert torch.equal(b, factors[path + ".lora_B.weight"]), path
        assert (scaling, active) == (h3_lora_runtime.SCALING, ("adapter_0",)), path
    inputs = h3_lora_runtime.inputs()
    one = model.denoise(inputs)
    merged, plain = h3_lora_runtime.dit().eval(), h3_lora_runtime.dit().eval()
    merged.load_state_dict(values)
    plain.load_state_dict(values)
    with torch.inference_mode():
        for path in held:
            layer = merged.get_submodule(path)
            delta = (
                factors[path + ".lora_B.weight"].float() @ factors[path + ".lora_A.weight"].float()
            )
            weight = values[path + ".weight"].float() + h3_lora_runtime.SCALING * delta
            layer.weight.copy_(weight.to(layer.weight.dtype))
        wanted, base = merged(**inputs), plain(**inputs)
    for got, want, without in zip(one, wanted, base, strict=True):
        effect = (want.float() - without.float()).norm()
        assert (got.float() - want.float()).norm() < 0.05 * effect

    fixture = Path(h3_lora_runtime.__file__)

    def launch(module_argv: list[str], fd: int) -> spawn.Child:
        return spawn.spawn_follower(
            python=sys.executable,
            module_argv=[str(fixture), *module_argv[2:], "--view", view.digest],
            inherit_fd=fd,
            env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
        )

    group = RankGroup(degree=degree, backend="cpu:gloo", root=str(tmp_path), launch=launch)
    ours, theirs = socket.socketpair()
    leader: Any = Executor(Channel(ours), tmp_path, world=degree)
    leader.device_kind = "cpu"
    leader.group = group
    leader._group_model_key = "fixture"
    leader.backend = backend
    try:
        group.spawn()
        group.form(torch, {})
        leader.pg = group.pg
        assert leader._install_group(torch, model) is None
        mine = h3_lora_runtime.digest(held)
        for rank in range(1, degree):
            assert (tmp_path / f"adapters-{rank}.txt").read_text() == mine, f"GPU {rank}"
        leader._attempt_spool = tmp_path / "attempt"
        many = model.denoise(inputs)
    finally:
        group.close()
        ours.close()
        theirs.close()
    for got, single in zip(many, one, strict=True):
        torch.testing.assert_close(got, single, rtol=0.02, atol=0.02)
    assert not torch.cuda.is_initialized()


class _Outputs(TorchDispatchMode):  # type: ignore[misc]
    """Every tensor each dispatched op produces: what a forward allocates, by dtype and size."""

    def __init__(self) -> None:
        super().__init__()
        self.seen: list[tuple[Any, int]] = []

    def __torch_dispatch__(self, func: Any, types: Any, args: Any = (), kwargs: Any = None) -> Any:
        kwargs = kwargs or {}
        held = {
            t.untyped_storage().data_ptr()
            for t in tree_leaves((args, kwargs))
            if isinstance(t, torch.Tensor)
        }
        out = func(*args, **kwargs)
        self.seen += [
            (t.dtype, t.numel())
            for t in tree_leaves(out)
            if isinstance(t, torch.Tensor) and t.untyped_storage().data_ptr() not in held
        ]
        return out


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_a_lora_update_accumulates_in_place_at_the_activation_precision(
    tmp_path: Path, dtype: Any
) -> None:
    """Factors keep their stored dtype, yet a bf16 activation is never copied to it and no
    full-width update is materialized (#1077): the only temporaries are rank-wide. The served
    projection stays within bf16 rounding of the exact merged weight."""
    root = tmp_path / "store"
    view, values, factors = h3_lora_runtime.checkpoints(root, dtype)
    model, _ = h3_lora_runtime.construct(root, view.digest)
    dit = model.pipe.components[h3_lora_runtime.COMPONENT]
    residency: Any = model._cozy_residency
    rows = 4096
    residency.admit("denoise", (h3_lora_runtime.COMPONENT,))
    try:
        for path in h3_lora_runtime.targets(dit):
            layer = dit.get_submodule(path)
            x = torch.randn(rows, layer.in_features, generator=torch.Generator().manual_seed(5))
            x = x.to(torch.bfloat16)
            with torch.inference_mode(), _Outputs() as outputs:
                got = layer(x)
            # Only the base output is full width; views and the in-place update allocate nothing.
            wide = [(kind, n) for kind, n in outputs.seen if n > rows * h3_lora_runtime.RANK]
            assert wide == [(torch.bfloat16, rows * layer.out_features)], (path, wide)
            a, b = (factors[f"{path}.lora_{role}.weight"].double() for role in "AB")
            weight = values[path + ".weight"].double() + h3_lora_runtime.SCALING * (b @ a)
            want = x.double() @ weight.T
            base = x.double() @ values[path + ".weight"].double().T
            error = (got.double() - want).norm()
            assert error < 2**-8 * want.norm(), path
            assert error < 0.01 * (want - base).norm(), path
    finally:
        residency.release("denoise", (h3_lora_runtime.COMPONENT,))
