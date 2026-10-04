"""Real native VAE/subclass CPU decodes; only device allocator probes are replaced."""

from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest
import torch
from diffusers import AutoencoderKL
from diffusers.configuration_utils import FrozenDict

from cozy_runtime.internal import accel, image_vae, weights
from cozy_runtime.internal.weights import ResidencyRefusal


class StatefulVAE(AutoencoderKL):
    def __init__(self, failure):
        torch.nn.Module.__init__(self)
        self._internal_dict = FrozenDict({"block_out_channels": [1, 2, 3, 4]})
        self.use_tiling = False
        self.calls = 0
        self.failure = failure

    def decode(self, z, *args, **kwargs):
        self.calls += 1
        z.add_(1)
        torch.rand(1)
        if self.failure:
            raise self.failure
        return z


def residency(monkeypatch, vae, *, peak=0, free=1 << 30):
    owner = SimpleNamespace(
        torch=torch,
        kind="cuda",
        adaptive=set(),
        activation={"decode": peak},
        cap=free,
        modes={},
        poisoned="",
        residencies=set(),
        host_only=False,
        _recovery_local=threading.local(),
    )
    owner.recovery_refusal = lambda detail: weights.Weights.recovery_refusal(owner, detail)
    row = weights.WeightResidency.__new__(weights.WeightResidency)
    row.weights, row.active, row.poisoned = owner, ("decode", ("vae",)), ""
    row.components = {"vae": SimpleNamespace(root=vae)}
    row.room = lambda *_: (free, 0)
    row._room_for_decode = lambda _name, available, _evictable, _need: available
    owner.residencies.add(row)
    monkeypatch.setattr(accel, "allocated", lambda *_: 0)
    monkeypatch.setattr(accel, "peak_allocated", lambda *_: 0)
    monkeypatch.setattr(accel, "reset_peak", lambda *_: None)
    monkeypatch.setattr(accel, "release_cached", lambda *_: None)
    monkeypatch.setattr(weights, "inexact", lambda: None)
    return row, owner


@pytest.mark.parametrize(
    "failure",
    [
        torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 16.00 MiB"),
        ResidencyRefusal(weights.INDIVISIBLE, "aten.mm operation needs more room"),
    ],
)
def test_uncontracted_decode_failure_keeps_effects_once_and_retires(monkeypatch, failure):
    torch.set_num_threads(2)
    vae = StatefulVAE(failure)
    row, owner = residency(monkeypatch, vae)
    z = torch.zeros(1, 4, 128, 128)
    torch.manual_seed(324)
    torch.rand(1)
    expected_random = torch.get_rng_state().clone()
    torch.manual_seed(324)
    with pytest.raises(ResidencyRefusal, match="whole-decode replay") as caught:
        row._decode("vae", vae.decode)(z)
    assert caught.value.code == "recovery_contract_unavailable"
    assert vae.calls == 1 and torch.equal(z, torch.ones_like(z))
    assert torch.equal(torch.get_rng_state(), expected_random)
    assert owner.poisoned and row.poisoned
    with pytest.raises(ResidencyRefusal) as next_call, weights.Weights.recovering(owner):
        pytest.fail("a retained poisoned executor must not enter its next handler")
    assert next_call.value.code == "residency_poisoned"


def test_successful_subclass_still_decodes_once(monkeypatch):
    vae = StatefulVAE(None)
    row, owner = residency(monkeypatch, vae)
    z = torch.zeros(1, 4, 16, 16)
    out = row._decode("vae", vae.decode)(z)
    assert vae.calls == 1 and out is z and not owner.poisoned


def test_noncapacity_error_is_not_hidden_or_replayed(monkeypatch):
    vae = StatefulVAE(ValueError("authored refusal"))
    row, _owner = residency(monkeypatch, vae)
    with pytest.raises(ValueError, match="authored refusal"):
        row._decode("vae", vae.decode)(torch.zeros(1, 4, 16, 16))
    assert vae.calls == 1


def test_native_decode_matches_baseline_and_hook_runs_once(monkeypatch):
    torch.set_num_threads(2)
    vae = AutoencoderKL(
        down_block_types=("DownEncoderBlock2D",) * 2,
        up_block_types=("UpDecoderBlock2D",) * 2,
        block_out_channels=(32, 64),
        layers_per_block=1,
        sample_size=32,
        norm_num_groups=8,
        latent_channels=4,
    )
    z = torch.randn(1, 4, 16, 16)
    with torch.inference_mode():
        expected = vae.decode(z).sample
    calls = []
    vae.decoder.register_forward_pre_hook(lambda *_: calls.append(True))
    row, owner = residency(monkeypatch, vae)
    with torch.inference_mode():
        actual = row._decode("vae", vae.decode)(z).sample
    assert calls == [True] and not owner.poisoned
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_native_hook_oom_is_not_replayed(monkeypatch):
    torch.set_num_threads(2)
    vae = AutoencoderKL(
        down_block_types=("DownEncoderBlock2D",) * 2,
        up_block_types=("UpDecoderBlock2D",) * 2,
        block_out_channels=(32, 64),
        layers_per_block=1,
        sample_size=32,
        norm_num_groups=8,
        latent_channels=4,
    )
    calls = []

    def fail(*_):
        calls.append(True)
        raise torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 16.00 MiB")

    vae.decoder.register_forward_pre_hook(fail)
    row, owner = residency(monkeypatch, vae)
    with pytest.raises(ResidencyRefusal) as caught:
        row._decode("vae", vae.decode)(torch.zeros(1, 4, 16, 16))
    assert caught.value.code == "recovery_contract_unavailable"
    assert calls == [True] and owner.poisoned


def test_measured_tile_is_selected_before_first_decode(monkeypatch):
    vae = StatefulVAE(None)
    row, owner = residency(monkeypatch, vae, peak=2 << 30, free=512 << 20)
    shape = (1, 4, 128, 128)
    result = row._decode("vae", vae.decode)(torch.zeros(shape))
    assert result.shape == shape and vae.calls == 1 and not owner.poisoned
    assert vae.use_tiling and owner.modes["vae.decode"].startswith("tiled ")
    assert vae.tile_sample_min_size <= 512


def native_sdxl_geometry(dtype=torch.bfloat16):
    # Real native decoder structure and parameter metadata, without allocating weights.
    with torch.device("meta"):
        vae = AutoencoderKL(
            down_block_types=("DownEncoderBlock2D",) * 4,
            up_block_types=("UpDecoderBlock2D",) * 4,
            block_out_channels=(128, 256, 512, 512),
            layers_per_block=2,
            sample_size=1024,
        ).to(dtype=dtype)
    return vae


def test_cold_estimate_follows_real_upsample_channels_dtype_and_batch_slicing():
    vae = native_sdxl_geometry()
    geometry = image_vae.geometry(vae)
    assert geometry is not None and geometry.decode_estimate is not None
    single = geometry.decode_estimate((1, 4, 128, 128), 2)
    assert single is not None
    # The last upsampler still has256 channels at1024 square, larger than final128.
    assert single.peak == (3 * 256 * 1024 * 1024 + 2 * 3 * 1024 * 1024 + 4 * 128 * 128) * 2
    assert single.retained == (2 * 3 * 1024 * 1024 + 4 * 128 * 128) * 2
    doubled = geometry.decode_estimate((2, 4, 128, 128), 2)
    assert doubled is not None and doubled.peak == single.peak * 2
    vae.enable_slicing()
    sliced = geometry.decode_estimate((2, 4, 128, 128), 2)
    assert sliced is not None and single.peak < sliced.peak < doubled.peak
    vae.to(dtype=torch.float32)
    upcast = geometry.decode_estimate((1, 4, 128, 128), 2)
    assert upcast is not None and upcast.peak > single.peak * 1.99


@pytest.mark.parametrize("owner_seed", [0, 1008372224, 400 << 20])
def test_first_native_decode_selects_tiles_before_call_despite_partial_or_tiled_owner_seed(
    monkeypatch, owner_seed
):
    vae = native_sdxl_geometry()
    row, owner = residency(monkeypatch, vae, peak=owner_seed, free=1200 << 20)
    seen = []

    def decode(z):
        seen.append((vae.use_tiling, vae.tile_sample_min_size))
        return z

    z = torch.zeros(1, 4, 128, 128, dtype=torch.bfloat16)
    assert row._decode("vae", decode)(z) is z
    assert seen == [(True, 512)] and owner.modes["vae.decode"] == "tiled 512"
    assert not owner.poisoned


def test_cold_native_decode_keeps_large_room_untiled_fast_path(monkeypatch):
    vae = native_sdxl_geometry()
    row, owner = residency(monkeypatch, vae, free=3 << 30)
    seen = []

    def decode(z):
        seen.append(vae.use_tiling)
        return z

    z = torch.zeros(1, 4, 128, 128, dtype=torch.bfloat16)
    assert row._decode("vae", decode)(z) is z
    assert seen == [False] and owner.modes["vae.decode"] == "untiled"


def test_unsupported_custom_decoder_keeps_its_existing_one_call_path(monkeypatch):
    vae = StatefulVAE(None)
    vae.decoder = torch.nn.Identity()
    geometry = image_vae.geometry(vae)
    assert geometry is not None and geometry.decode_estimate is not None
    assert geometry.decode_estimate((1, 4, 16, 16), 4) is None
    row, owner = residency(monkeypatch, vae)
    z = torch.zeros(1, 4, 16, 16)
    assert row._decode("vae", vae.decode)(z) is z
    assert vae.calls == 1 and not owner.poisoned


def test_tile_budget_leaves_full_image_latent_and_output_retention(monkeypatch):
    vae = native_sdxl_geometry()
    row, _owner = residency(monkeypatch, vae, free=391 << 20)
    seen = []

    def decode(z):
        seen.append(vae.tile_sample_min_size)
        return z

    row._decode("vae", decode)(torch.zeros(1, 4, 128, 128, dtype=torch.bfloat16))
    assert seen == [448]
