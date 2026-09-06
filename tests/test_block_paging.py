"""A block floor must reach the real planner; aliases may not cross paging lifetimes."""

from __future__ import annotations

import pytest

from cozy_runtime.internal.paging import partition
from cozy_runtime.internal.worker.ledger import Ledger
from cozy_runtime.internal.worker.plan import DeclaredBinding, PlanChooser


def test_h3_encoder_block_floor_reaches_the_attempt_planner() -> None:
    # Exact full-H3 meta census, including its non-persistent buffers. These are
    # shape facts, not a claimed activation peak or a full-video fit certificate.
    nominal, working = 51_506_192_132, 2_898_621_988
    ledger = Ledger(worker_pid=1)
    ledger.begin_generation(1)
    ledger.observe_construction(
        {
            "filled_bytes": nominal,
            "allocator_bytes": 0,
            "reserved_bytes": 0,
            "resident": {},
            "parked": ["text_encoder"],
            "evicted": {"text_encoder": nominal},
            "paging": {"text_encoder": {"total_bytes": nominal, "working_bytes": working}},
            "declared_scopes": {"condition_text": ["text_encoder"]},
            "device_free_bytes": 8 << 30,
            "device_total_bytes": 24 << 30,
        }
    )
    binding = DeclaredBinding(
        entrypoint_binding_digest="sha256:fixture",
        entrypoint="generate",
        model_class="H3Model",
        model_binding_path="h3:H3Model",
        model_parameter_name="model",
        release="fixture@1",
        model_weight_bytes=nominal,
    )
    chooser = PlanChooser(ledger)
    _, rows, chosen, _ = chooser._walk(binding, "fixture")
    assert chosen == "component_staged", rows
    assert rows[0]["verdict"] == "construction_parked"
    staged = next(row for row in rows if row["rung"] == "component_staged")
    assert working <= staged["needs_bytes"] < nominal
    assert staged["absent_bytes"] == nominal
    ledger.device_free = working - 1
    _, rows, chosen, _ = chooser._walk(binding, "fixture")
    assert not chosen, rows
    assert next(row for row in rows if row["rung"] == "component_staged")["verdict"] == "short"


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


def test_module_lists_without_a_declared_forward_boundary_are_not_guessed() -> None:
    torch = pytest.importorskip("torch")
    model = torch.nn.Sequential(torch.nn.Linear(4, 4, device="meta"))
    assert partition(model) is None


@pytest.mark.parametrize("substrate_name", ["meta_substrate", "serving_substrate"])
def test_derived_rotary_initialization_keeps_checkpoint_mutation_guard(substrate_name: str) -> None:
    torch = pytest.importorskip("torch")
    from cozy_runtime.internal import derive

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
