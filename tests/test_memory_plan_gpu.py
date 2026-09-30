"""The executor's memory planner on a real card, with real diffusers models and real kernels.

Real: CUDA, the caching allocator and its cap, SDXL's UNet and VAE classes, their nested
`_no_split_modules`, host copies and the planner's ops. Random weights stand in for a
checkpoint: every comparison is bit-for-bit against the same weights resident.
"""

from __future__ import annotations

from types import MethodType
from typing import Any

import pytest

torch = pytest.importorskip("torch")
diffusers = pytest.importorskip("diffusers")

from cozy_runtime.author import pure  # noqa: E402
from cozy_runtime.internal import memory, tiers  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
# The executor's allocator configuration (worker session `alloc_conf`).
torch.cuda.memory._set_allocator_settings("expandable_segments:True")
MiB = 1 << 20
DEVICE = torch.device("cuda", 0)


def _unet() -> Any:
    torch.manual_seed(0)
    return pure(
        diffusers.UNet2DConditionModel(
            sample_size=64,
            down_block_types=("DownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D"),
            up_block_types=("CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "UpBlock2D"),
            block_out_channels=(320, 640, 1280),
            layers_per_block=2,
            transformer_layers_per_block=(1, 2, 2),
            cross_attention_dim=2048,
            attention_head_dim=(5, 10, 20),
            use_linear_projection=True,
            addition_embed_type="text_time",
            addition_time_embed_dim=256,
            projection_class_embeddings_input_dim=2816,
        )
        .to(torch.float16)
        .eval()
    )


def _vae() -> Any:
    torch.manual_seed(1)
    return pure(
        diffusers.AutoencoderKL(
            block_out_channels=(128, 256, 512, 512),
            down_block_types=("DownEncoderBlock2D",) * 4,
            up_block_types=("UpDecoderBlock2D",) * 4,
            latent_channels=4,
            sample_size=1024,
            layers_per_block=2,
        ).eval()
    )


def _inputs() -> dict[str, Any]:
    generator = torch.Generator().manual_seed(2)

    def randn(*shape: int) -> Any:
        return torch.randn(*shape, generator=generator).to(DEVICE, torch.float16)

    return {
        "sample": randn(2, 4, 128, 128),
        "encoder_hidden_states": randn(2, 77, 2048),
        "added_cond_kwargs": {"text_embeds": randn(2, 1280), "time_ids": randn(2, 6)},
    }


def _loop(unet: Any, inputs: dict[str, Any], steps: int = 3) -> Any:
    x = inputs["sample"].clone()
    with torch.inference_mode():
        for step in range(steps):
            t = torch.tensor([900.0 - 200 * step], device=DEVICE)
            eps = unet(
                x,
                t,
                encoder_hidden_states=inputs["encoder_hidden_states"],
                added_cond_kwargs=inputs["added_cond_kwargs"],
            ).sample
            x = x - 0.1 * eps
    return x


_planners: list[memory.Planner] = []


def _planner() -> memory.Planner:
    torch.cuda.set_per_process_memory_fraction(1.0, DEVICE)
    torch.cuda.empty_cache()
    free, total = torch.cuda.mem_get_info(DEVICE)
    planner = memory.Planner(torch, DEVICE)
    # Everything the card holds beyond this process's allocator counts against it here.
    planner.overhead = total - free - torch.cuda.memory_reserved(DEVICE)
    _planners.append(planner)
    return planner


@pytest.fixture(autouse=True)
def _one_process_one_planner() -> Any:
    """An executor has one planner. A test's leaves nothing on the card for the next test's."""
    yield
    while _planners:
        planner = _planners.pop()
        planner.vacate()
        for construction in {held.construction for held in planner.held.values()}:
            planner.forget(construction)
    torch.cuda.set_per_process_memory_fraction(1.0, DEVICE)
    torch.cuda.empty_cache()


@pytest.fixture
def reference() -> Any:
    # The cap a previous test's grant left on this process is not this test's.
    torch.cuda.set_per_process_memory_fraction(1.0, DEVICE)
    unet = _unet().to(DEVICE)
    inputs = _inputs()
    torch.cuda.reset_peak_memory_stats(DEVICE)
    base = torch.cuda.memory_allocated(DEVICE)
    resident = _loop(unet, inputs)
    working = torch.cuda.max_memory_allocated(DEVICE) - base
    unet.to("cpu")
    torch.cuda.empty_cache()
    yield unet, inputs, resident, working
    del unet
    torch.cuda.empty_cache()


def test_innermost_units_hold_sdxls_unet_to_one_block(reference: Any) -> None:
    unet, _, _, _ = reference
    part = tiers.partition("unet", unet, tiers.HOST)
    # Innermost repeated no-split instances: no unit contains another, and the largest one
    # is a transformer or resnet block, never a whole up block.
    names = [type(unit.module).__name__ for unit in part.blocks]
    assert set(names) == {"BasicTransformerBlock", "ResnetBlock2D"}
    assert max(unit.nbytes for unit in part.blocks) < part.nbytes // 10
    assert part.common.nbytes + sum(unit.nbytes for unit in part.blocks) == part.nbytes


@pytest.mark.parametrize("fraction", [2, 4])
def test_streaming_at_a_grant_is_bit_identical_to_resident(reference: Any, fraction: int) -> None:
    unet, inputs, resident, working = reference
    planner = _planner()
    held = planner.adopt("c", "unet", unet, None, tiers.HOST)
    grant = held.part.nbytes // fraction + working + planner.overhead
    planner.begin(grant)
    streamed = _loop(unet, inputs)
    facts = planner.end()
    assert torch.equal(streamed, resident)
    assert 0 < held.part.on(tiers.DEVICE) < held.part.nbytes
    assert facts["peak_bytes"] <= grant
    planner.vacate()
    assert held.part.on(tiers.DEVICE) == 0


def test_a_forward_its_author_replaced_is_still_a_planned_op(reference: Any) -> None:
    unet, inputs, resident, working = reference
    original = unet.forward
    unet.forward = MethodType(pure(lambda _self, *a, **k: original(*a, **k)), unet)
    planner = _planner()
    held = planner.adopt("c", "unet", unet, None, tiers.HOST)
    planner.begin(held.part.nbytes // 2 + working + planner.overhead)
    replaced = _loop(unet, inputs)
    planner.end()
    assert "unet.forward" in planner.plans
    assert torch.equal(replaced, resident)


def test_a_forced_out_of_memory_step_recovers_bit_identical(reference: Any) -> None:
    unet, inputs, resident, working = reference
    planner = _planner()
    held = planner.adopt("c", "unet", unet, None, tiers.HOST)
    grant = held.part.nbytes // 2 + working + planner.overhead
    planner.begin(grant)
    _loop(unet, inputs, steps=1)
    planner.end()
    for record in planner.records.values():
        record.ok //= 4  # understated: the next plan keeps too much resident and runs out
    planner.begin(grant)
    recovered = _loop(unet, inputs)
    planner.end()
    assert planner.log, "the understated plan ran out and was recovered"
    assert torch.equal(recovered, resident)


def test_a_decode_keeps_its_authored_tiling_mode() -> None:
    vae = _vae()
    vae.enable_tiling()
    vae.tile_sample_min_size = 256
    vae.tile_latent_min_size = 32
    z = torch.randn(1, 4, 128, 128, generator=torch.Generator().manual_seed(3)).to(DEVICE)
    planner = _planner()
    held = planner.adopt("c", "vae", vae, None, tiers.HOST)
    mode = memory._mode(vae)
    with torch.inference_mode():
        planner.begin(None)
        control = vae.decode(z).sample.clone()
        planner.end()
        measured = next(iter(planner.records.values())).ok
        planner.begin(held.part.nbytes + measured + planner.overhead)
        assert torch.equal(vae.decode(z).sample, control)
        planner.end()
    assert memory._mode(vae) == mode
    assert all(key.mode == mode for key in planner.records)


def test_an_op_with_a_foreign_hook_fails_typed_and_leaves_the_device_clean(
    reference: Any,
) -> None:
    unet, inputs, _, working = reference
    planner = _planner()
    held = planner.adopt("c", "unet", unet, None, tiers.HOST)
    grant = held.part.nbytes // 2 + working + planner.overhead
    handle = unet.down_blocks[0].register_forward_hook(lambda *_: None)
    try:
        planner.begin(grant)
        _loop(unet, inputs, steps=1)
        planner.end()
        for record in planner.records.values():
            record.ok //= 4
        planner.begin(grant)
        # Clean: nothing but the weights the plan placed outlives the refused op.
        before = torch.cuda.memory_allocated(DEVICE) - held.part.on(tiers.DEVICE)
        with pytest.raises(memory.MemoryRefusal) as refused:
            _loop(unet, inputs, steps=1)
        assert refused.value.code == "device_out_of_memory"
        assert "cannot be put back" in refused.value.message
        planner.end()
        assert torch.cuda.memory_allocated(DEVICE) - held.part.on(tiers.DEVICE) <= before
    finally:
        handle.remove()
