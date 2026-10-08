"""H3's turbo overlay baked on the weight plane: a turbo forward bakes the overlay's factors into
the DiT's resident block bytes and its consumer hooks add nothing; a base forward on the same
DiT finds those bytes refilled from the unmodified host copy first. A CPU executor (no bake)
computes the same forwards per step."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from cozy_runtime.author import Artifact, Config, Loader
from cozy_runtime.internal import accel, plane
from cozy_runtime.internal.fill import Checkpoint, tensor_schema_of
from cozy_runtime.internal.weights import LoRASource, PlaneBackend, WeightResidency, Weights

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")

DEVICES = ["cpu"] + (["cuda"] if accel.present(torch, "cuda") and plane.available() else [])
TIMES = (0.0, 0.25, 0.75)
KEYS = [(row, tag) for row in range(3) for tag in range(3)]


def _dit() -> Any:
    from cozy_runtime.models.minimax_h3.adaln_pruned import AdaLNPrunedMiniMaxH3Transformer

    dit = AdaLNPrunedMiniMaxH3Transformer(
        table_timesteps=TIMES,
        table_block_keys=KEYS,
        num_attention_heads=2,
        attention_head_dim=8,
        hidden_size=16,
        num_layers=2,
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
    dit.install_lora_consumers()
    return dit


def _overlay(device: str) -> Any:
    from cozy_runtime.models.minimax_h3.turbo import TurboOverlay, TurboSchedule

    overlay = TurboOverlay(
        hidden_size=16,
        inner_dim=16,
        ffn_dim=32,
        num_layers=2,
        num_refiner_layers=1,
        video_out=2,
        audio_out=2,
        rank=2,
        alpha=2,
        schedule=TurboSchedule((0.75, 0.0), (0.25, 0.0)),
        table_timesteps=TIMES,
        table_block_keys=KEYS,
        block_table_dtype=torch.float32,
        final_table_dtype=torch.float32,
    ).eval()
    with torch.no_grad():
        for parameter in overlay.parameters():
            parameter.normal_(0, 0.1)
        for _, site in overlay.lora_sites():
            site.lora_down.data = site.lora_down.data.float()
            site.lora_up.data = site.lora_up.data.float()
    return overlay.to(device)


def _checkpoint(tmp_path: Path, dit: Any) -> Checkpoint:
    import tensorfs

    from test_lora_composition import Value, model

    store = tensorfs.Store.init(tmp_path / "store")
    tensors = {
        key: Value(tuple(value.shape), tuple(value.float().flatten().tolist()))
        for key, value in dit.state_dict().items()
    }
    written = model(store, tmp_path, "dit", "fl2va_dit", tensors, b"{}")
    return Checkpoint.from_store(store, written.manifest.digest)


def _inputs(device: str) -> dict[str, Any]:
    from diffusers.modular_pipelines.minimax_h3.before_denoise import MiniMaxH3PrepareLayoutStep

    position, tags, video, audio, text, _, _ = MiniMaxH3PrepareLayoutStep.build_packed_sequence(
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
    generator = torch.Generator().manual_seed(5)
    values = dict(
        hidden_states=torch.randn(1, len(video), 2, generator=generator),
        audio_hidden_states=torch.randn(1, len(audio), 2, generator=generator),
        encoder_hidden_states=torch.randn(1, len(text), 4, generator=generator),
        timestep=torch.tensor([0.75, 0.25]),
        timestep_indices=torch.where(tags == 2, 1, 0),
        token_tags=tags,
        position_ids=position,
        video_indices=video,
        audio_indices=audio,
        text_indices=text,
    )
    return {k: v.to(device) for k, v in values.items()} | {"return_dict": False}


@pytest.mark.parametrize("device", DEVICES)
def test_a_turbo_forward_bakes_and_a_base_forward_finds_the_bytes_refilled(
    tmp_path: Path, device: str
) -> None:
    from cozy_runtime.models.minimax_h3.turbo import ATTENTION_KWARG, OVERLAY_KWARG, TURBO_BANK

    torch.manual_seed(26)
    reference = _dit()
    with torch.no_grad():
        for parameter in reference.parameters():
            parameter.normal_(0, 0.1)
    checkpoint = _checkpoint(tmp_path, reference)
    reference = reference.to(device)
    overlay = _overlay(device)
    rows = checkpoint.rows("fl2va_dit")
    weights = Weights(
        torch, torch.device(device, 0) if device == "cuda" else torch.device(device), device
    )
    weights.set_budget(-1, pinned=64 << 20)
    backend = PlaneBackend.for_script(
        {"fl2va_dit": checkpoint}, rows, weights=weights, construction="turbo", release="turbo/1"
    )

    class Pipeline:
        def __init__(self, config: Config) -> None:
            # Built for real: its non-persistent buffers (rope, tables' indexes) keep the
            # values construction computed; the plane binds every weight.
            self.components = {"fl2va_dit": _dit()}

    try:
        backend.expect({"fl2va_dit": tuple(row.key for row in rows)})
        artifact = Artifact(checkpoint.manifest_id, tensor_schema_of(rows), Config({}))
        built = Loader(artifact, owner=SimpleNamespace(), backend=backend).construct(
            Pipeline, factory=Pipeline
        )
        dit = built.components["fl2va_dit"]
        regions = backend.components["fl2va_dit"].regions
        baking = [r for r in regions if r.bakes]
        hooks = [h for r in baking for b in r.bakes for h in b.hooks]
        assert all(isinstance(h, LoRASource) for h in hooks) and len(hooks) == 18
        assert all(r.bake is (device == "cuda") for r in baking)
        residency = WeightResidency(weights, backend.components, {"run": ("fl2va_dit",)})
        inputs = _inputs(device)

        def forward(model: Any, turbo: bool) -> tuple[list[Any], list[bool]]:
            selector = {ATTENTION_KWARG: TURBO_BANK, OVERLAY_KWARG: overlay} if turbo else {}
            served = model is dit
            if served:
                residency.admit("run", ("fl2va_dit",))
            try:
                with torch.no_grad():
                    out = model(**inputs, attention_kwargs=selector)
                return [t.float() for t in out], [h._cozy_baked for h in hooks]
            finally:
                if served:
                    residency.release("run", ("fl2va_dit",))

        turbo_ref, _ = forward(reference, True)
        base_ref, _ = forward(reference, False)
        cuda = device == "cuda"
        for turbo in (True, False, True, False):
            got, baked = forward(dit, turbo)
            want = turbo_ref if turbo else base_ref
            for g, w in zip(got, want, strict=True):
                torch.testing.assert_close(g, w, rtol=1e-4, atol=1e-5)
            assert baked == [cuda and turbo] * len(hooks)
        assert not torch.equal(turbo_ref[0], base_ref[0])
    finally:
        backend.close()
