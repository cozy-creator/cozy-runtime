"""A weight read outside its module's forward that binds a LoRA-baked block, whose bake reads the
overlay's factors through the stand-in dispatch: the second read nests inside the first.

Runs 10440, 10443 and 10454 (H3 long_form's second segment on an RTX 5090) were refused there,
`weight_unbound: aten.mul.Tensor ... inside another weight read's mapping`. Reads now nest, and
only a read that re-enters its own mapping is refused (run 10466 on the same pod and request)."""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from cozy_runtime.internal import plane  # noqa: E402
from cozy_runtime.internal.weights import Weights, _unbaked, let_go, unbind  # noqa: E402
from test_weight_plane import dit_inputs, lora_models, lora_reference, on_card  # noqa: E402


def _evict(weights: Weights, component: object, regions: list[object]) -> None:
    """The regions leave the card mid-stage, as paging under a long sequence leaves them: the
    stage's holds let go, the plane drops what it may (a stage's common lease stays mapped), and
    every one is unbound with its bake gone."""
    let_go(component)  # type: ignore[arg-type]
    for region in regions:
        try:
            weights.plane.drop(component.ws, weights.index, [region.index])  # type: ignore[attr-defined]
        except plane.error("LeaseViolation"):
            pass  # mapped under the stage's lease: its bytes stay, unbaked overlays have no bake
        unbind(region)  # type: ignore[arg-type]
        _unbaked(region)  # type: ignore[arg-type]


def _region_of(component: object, table: object, name: str) -> object:
    return next(
        region
        for region in component.regions  # type: ignore[attr-defined]
        if any(t is table and n == name for t, n, _ in region.unbound)
    )


@on_card
def test_a_read_outside_forward_binds_a_baked_block_whose_bake_reads_the_overlay(
    tmp_path: Path,
) -> None:
    """The read is served, and reads the bytes the block's forward bakes."""
    from testdata.consumer_lora import armed

    weights = Weights(torch, torch.device("cuda", 0), "cuda")
    weights.set_budget(-1, pinned=1 << 30)
    dit, adapter = lora_models(weights, tmp_path, "nested")
    weights.set_budget(1 << 30)
    lora_reference(weights, dit, adapter, dit_inputs("cuda"))  # every block baked, overlay armed
    site = dit.stack.get_submodule("transformer_blocks.0.attn.to_v")  # the overlay's own site
    baked = site.weight.detach().clone()
    factors = adapter.stack.factors("transformer_blocks.0.attn.to_v")
    blocks = dit.residency.components["fl2va_dit"]
    overlay = adapter.residency.components["fl2va_overlay"]
    dit.residency.admit("sample_adapted", ("fl2va_dit",))
    try:
        adapter.residency.admit("sample_fl2va", ("fl2va_overlay",))
        try:
            _evict(weights, blocks, [_region_of(blocks, site._parameters, "weight")])
            _evict(weights, overlay, list(overlay.regions))
            with torch.inference_mode(), armed(dit.stack, adapter.stack):
                # Both are stand-ins: the block's bake must map the overlay inside this read.
                assert type(site.weight) is not torch.nn.Parameter
                assert type(factors.lora_up) is not torch.nn.Parameter
                read = site.weight * 1
        finally:
            adapter.residency.release("sample_fl2va", ("fl2va_overlay",))
    finally:
        dit.residency.release("sample_adapted", ("fl2va_dit",))
    assert torch.equal(read, baked)
