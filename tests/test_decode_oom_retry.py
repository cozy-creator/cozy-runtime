"""STOPGAP until memory v3 deletes it: a VAE decode that runs out of device memory frees
the components its scope does not hold and runs once more tiled (runs 1582/1590/1597: SDXL
decode OOMed on an 8 GB card with the UNet and text encoders still resident)."""

from __future__ import annotations

from typing import Any

import pytest

from cozy_runtime.author import Model, uses_components
from test_staged_demand_pull import ENVELOPE, SCOPES, Device, device, plane

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")
from testdata.decode_oom import Card  # noqa: E402

__all__ = ["device"]


class Decoder(Model[object]):
    video_vae: Any

    @uses_components("video_vae")
    def decode_video(self, latents: Any) -> Any:
        with torch.inference_mode():
            return self.video_vae.decode(latents).sample


def test_a_decode_oom_sheds_retries_tiled_and_leaves_the_executor_serving(
    device: Device,
) -> None:
    residency, backend = plane(device, ("ref2va_dit", "text_encoder"))
    vae = Card(block_out_channels=(8,), norm_num_groups=8, layers_per_block=1, sample_size=16)
    vae.card = backend
    backend.parked["video_vae"] = vae
    model = Decoder.for_test(video_vae=vae)
    object.__setattr__(model, "_cozy_residency", residency)
    latents = torch.randn(1, 4, 4, 4, generator=torch.Generator().manual_seed(0))
    # Measured scopes keep what fits beside them, so the DiT is still resident at decode.
    residency.open_attempt("component_staged", ENVELOPE, dict(SCOPES), tuple(SCOPES))

    first = model.decode_video(latents)

    assert first.shape == (1, 3, 4, 4) and torch.isfinite(first).all()
    assert "ref2va_dit" not in backend.components and not vae.use_tiling
    assert {"method": "decode_video", "action": "decode retried tiled after OOM"} in backend.stage_log
    # The same executor serves the next request: nothing poisoned, the scope released.
    assert not residency.poisoned and model._cozy_active is None
    assert torch.equal(model.decode_video(latents), first)
