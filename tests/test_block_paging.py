"""A block floor must reach the real planner; regions are the innermost declared blocks, and
aliases may not cross them."""

from __future__ import annotations

import pytest

from cozy_runtime.internal import derive
from cozy_runtime.internal.paging import partition


def test_declared_blocks_with_shared_storage_are_not_independently_pageable() -> None:
    torch = pytest.importorskip("torch")

    class Residual(torch.nn.Module):  # type: ignore[name-defined,misc]  # optional real Torch
        def __init__(self) -> None:
            super().__init__()
            self.linear = torch.nn.Linear(4, 4, bias=False, device="meta")

    class Network(torch.nn.Module):  # type: ignore[name-defined,misc]  # optional real Torch
        _no_split_modules = ("Residual",)

        def __init__(self) -> None:
            super().__init__()
            self.blocks = torch.nn.ModuleList([Residual(), Residual()])

    model = Network()
    layout = partition(model)
    assert layout is not None and len(layout.blocks) == 2
    assert layout.working_bytes < layout.total_bytes
    model.blocks[1].linear.weight = model.blocks[0].linear.weight
    with pytest.raises(ValueError, match="shared across paging units"):
        partition(model)


def test_nested_declarations_page_the_innermost_blocks() -> None:
    torch = pytest.importorskip("torch")

    class Resnet(torch.nn.Module):  # type: ignore[name-defined,misc]  # optional real Torch
        def __init__(self) -> None:
            super().__init__()
            self.conv = torch.nn.Linear(4, 4, device="meta")

    class UpBlock(torch.nn.Module):  # type: ignore[name-defined,misc]  # optional real Torch
        def __init__(self, resnets: int) -> None:
            super().__init__()
            self.resnets = torch.nn.ModuleList([Resnet() for _ in range(resnets)])
            self.upsampler = torch.nn.Linear(4, 4, device="meta")

    class Unet(torch.nn.Module):  # type: ignore[name-defined,misc]  # optional real Torch
        # SDXL's shape: a declared up block around declared resnets.
        _no_split_modules = ("UpBlock", "Resnet")

        def __init__(self) -> None:
            super().__init__()
            self.conv_in = torch.nn.Linear(4, 4, device="meta")
            self.up_blocks = torch.nn.ModuleList([UpBlock(2), UpBlock(0)])

    layout = partition(Unet())
    assert layout is not None
    # An outer declaration keeps only its own weights (in `common`); one with no declared
    # block inside is itself innermost.
    assert [block.path for block in layout.blocks] == [
        "up_blocks.0.resnets.0",
        "up_blocks.0.resnets.1",
        "up_blocks.1",
    ]
    assert set(layout.common.keys) == {
        "conv_in.weight",
        "conv_in.bias",
        "up_blocks.0.upsampler.weight",
        "up_blocks.0.upsampler.bias",
    }
    assert layout.blocks[2].keys == ("up_blocks.1.upsampler.weight", "up_blocks.1.upsampler.bias")
    assert layout.total_bytes == 5 * 80 and layout.common.nbytes == 2 * 80


def test_a_subclass_of_a_declared_block_is_a_block() -> None:
    """Run 5439: SeedVR2's VAE inherits diffusers' `_no_split_modules = ["ResnetBlock2D"]` and
    builds `ResnetBlock3D(ResnetBlock2D)`; matched by name alone, the whole VAE was one region
    and nothing knew it was in use while its decode ran out of memory."""
    torch = pytest.importorskip("torch")

    class ResnetBlock2D(torch.nn.Module):  # type: ignore[name-defined,misc]  # optional real Torch
        def __init__(self) -> None:
            super().__init__()
            self.conv = torch.nn.Linear(4, 4, device="meta")

    class ResnetBlock3D(ResnetBlock2D):
        pass

    class Vae(torch.nn.Module):  # type: ignore[name-defined,misc]  # optional real Torch
        _no_split_modules = ("ResnetBlock2D",)

        def __init__(self) -> None:
            super().__init__()
            self.conv_in = torch.nn.Linear(4, 4, device="meta")
            self.resnets = torch.nn.ModuleList([ResnetBlock3D(), ResnetBlock2D()])

    layout = partition(Vae())
    assert layout is not None
    assert [block.path for block in layout.blocks] == ["resnets.0", "resnets.1"]
    assert set(layout.common.keys) == {"conv_in.weight", "conv_in.bias"}


def test_at_the_sub_block_grain_a_module_too_small_for_a_region_stays_in_common() -> None:
    """Run 2744: split into regions of at most 16 MiB, Anima's attention left each 256-byte
    norm scale a region of its own, bound only inside the norm's forward. The fused kernel
    reads the scale without calling the norm, got a pointer with nothing mapped behind it,
    and faulted the device. A unit that small stays in `common`, bound for the whole stage."""
    torch = pytest.importorskip("torch")

    class Scale(torch.nn.Module):  # type: ignore[name-defined,misc]  # optional real Torch
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.empty(8, device="meta"))

    class Attention(torch.nn.Module):  # type: ignore[name-defined,misc]  # optional real Torch
        def __init__(self) -> None:
            super().__init__()
            self.to_q = torch.nn.Linear(64, 64, bias=False, device="meta")
            self.to_k = torch.nn.Linear(64, 64, bias=False, device="meta")
            self.norm_q, self.norm_k = Scale(), Scale()

    class Network(torch.nn.Module):  # type: ignore[name-defined,misc]  # optional real Torch
        def __init__(self) -> None:
            super().__init__()
            self.blocks = torch.nn.ModuleList([Attention(), Attention()])

    layout = partition(Network(), 64 * 64 * 4)  # one projection: a scale is 1/512 of it
    assert layout is not None
    assert [block.path for block in layout.blocks] == [
        f"blocks.{index}.{name}" for index in range(2) for name in ("to_q", "to_k")
    ]
    assert set(layout.common.keys) == {
        f"blocks.{index}.{name}.weight" for index in range(2) for name in ("norm_q", "norm_k")
    }


def test_at_the_sub_block_grain_a_module_nothing_calls_is_never_a_unit() -> None:
    """H3's turbo LoRA holds factors the DiT's hooks read and never calls: as units they were
    bound only inside a forward that never runs, each costing a 2 MiB region. They stay in
    `common`; a projection that is called is a unit."""
    torch = pytest.importorskip("torch")

    class Factors(torch.nn.Module):  # type: ignore[name-defined,misc]  # optional real Torch
        def __init__(self) -> None:
            super().__init__()
            self.down = torch.nn.Parameter(torch.empty(64, 64, device="meta"))

    class Overlay(torch.nn.Module):  # type: ignore[name-defined,misc]  # optional real Torch
        def __init__(self) -> None:
            super().__init__()
            self.sites = torch.nn.ModuleList([Factors(), Factors()])
            self.head = torch.nn.Linear(64, 64, bias=False, device="meta")

    layout = partition(Overlay(), 64 * 64 * 4)
    assert layout is not None
    assert [block.path for block in layout.blocks] == ["head"]
    assert set(layout.common.keys) == {"sites.0.down", "sites.1.down"}


def test_module_lists_without_a_declared_forward_boundary_are_not_guessed() -> None:
    torch = pytest.importorskip("torch")
    model = torch.nn.Sequential(torch.nn.Linear(4, 4, device="meta"))
    assert partition(model) is None


@pytest.mark.parametrize("substrate_name", ["meta_substrate", "serving_substrate"])
def test_derived_rotary_initialization_keeps_checkpoint_mutation_guard(substrate_name: str) -> None:
    torch = pytest.importorskip("torch")
    substrate = getattr(derive, substrate_name)
    # Upstream Qwen registers inv_freq, then its post_init copies the config-derived
    # values. It is not a checkpoint destination and serving must retain its values.
    with substrate("cpu", derive.Observations()):
        model = torch.nn.Module()
        model.register_buffer("inv_freq", torch.zeros(4), persistent=False)
        model.inv_freq.copy_(torch.arange(4))
    if substrate_name == "serving_substrate":
        assert torch.equal(model.inv_freq, torch.arange(4))
    else:
        assert model.inv_freq.is_meta
    for parameter in (False, True):
        with (
            pytest.raises(derive.ConstructionFault, match="in-place weight transform"),
            substrate("cpu", derive.Observations()),
            torch.no_grad(),
        ):
            model = torch.nn.Module()
            if parameter:
                model.register_parameter("weight", torch.nn.Parameter(torch.zeros(4)))
            else:
                model.register_buffer("weight", torch.zeros(4), persistent=True)
            model.weight.copy_(torch.ones(4))
