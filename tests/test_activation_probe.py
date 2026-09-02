"""cr-075 proof arms: the activation-probe core over REAL torch modules, nothing mocked.

Every arm builds a real tiny diffusion-shaped stack — timestep embedding, projections, a
ModuleList of attention/MLP blocks — free-runs a real denoising loop to record the
trajectory, and teacher-forces real forward passes through the real engine. torch is the
`derive` extra: absent from the checks venv and from CI's lanes, so these arms follow the
`needs_torch` discipline (test_group_lanes.py) and run wherever torch is installed
(`uv sync --extra derive`).
"""

from __future__ import annotations

import math
from typing import Any

import pytest

from cozy_runtime import canonical_json
from cozy_runtime.probe import (
    REPORT_FORMAT,
    Lane,
    ProbeConfig,
    ProbeRefusal,
    TapPlan,
    compare,
    resolve_taps,
    run_probe,
)
from cozy_runtime.probe._sketch import TapStepAccumulator


def _torch_here() -> str:
    try:
        import torch  # noqa: F401
    except ImportError as exc:
        return f"torch must be importable in the test interpreter for the probe arms: {exc}"
    return ""


NO_TORCH = _torch_here()
pytestmark = pytest.mark.skipif(bool(NO_TORCH), reason=NO_TORCH or "")

FEATURES = 8
WIDTH = 16
TOKENS = 6
STEPS = 4

ORDERED_TAPS = (
    "denoiser.embed",
    "denoiser.proj_in",
    "denoiser.blocks.0",
    "denoiser.blocks.1",
    "denoiser.blocks.2",
    "denoiser.blocks.3",
    "denoiser.proj_out",
)


# ------------------------------------------------------------------ the real tiny stack


def _denoiser_class() -> Any:
    from torch import nn

    class Block(nn.Module):  # type: ignore[misc]
        def __init__(self) -> None:
            super().__init__()
            self.attn = nn.MultiheadAttention(WIDTH, 2, batch_first=True)
            self.norm = nn.LayerNorm(WIDTH)
            self.mlp = nn.Sequential(
                nn.Linear(WIDTH, WIDTH * 2), nn.GELU(), nn.Linear(WIDTH * 2, WIDTH)
            )

        def forward(self, hidden: Any, emb: Any) -> Any:
            attended, _ = self.attn(hidden, hidden, hidden, need_weights=False)
            hidden = hidden + attended
            return hidden + self.mlp(self.norm(hidden) + emb)

    class TinyDenoiser(nn.Module):  # type: ignore[misc]
        def __init__(self) -> None:
            super().__init__()
            self.embed = nn.Linear(1, WIDTH)
            self.proj_in = nn.Linear(FEATURES, WIDTH)
            self.blocks = nn.ModuleList(Block() for _ in range(4))
            self.proj_out = nn.Linear(WIDTH, FEATURES)

        def forward(self, latent: Any, timestep: Any, conditioning: Any) -> Any:
            wide = self.proj_in.weight.dtype
            emb = self.embed(timestep.to(wide)) + conditioning.to(wide)
            hidden = self.proj_in(latent.to(wide))
            for block in self.blocks:
                hidden = block(hidden, emb)
            return self.proj_out(hidden)

    return TinyDenoiser


def _reference_build() -> Any:
    import torch

    torch.manual_seed(0)
    return _denoiser_class()()


def _scaled_build(path: str, factor: float) -> Any:
    """The reference weights exactly, with ONE submodule's weights scaled: the corrupted
    lane whose degraded block the probe must name."""
    import torch

    model = _reference_build()
    with torch.no_grad():
        for parameter in model.get_submodule(path).parameters():
            parameter.mul_(factor)
    return model


def _drive(call: Any) -> None:
    """A real free-running denoising loop: euler-ish steps over the model's own output."""
    import torch

    generator = torch.Generator().manual_seed(7)
    latent = torch.randn(1, TOKENS, FEATURES, generator=generator)
    conditioning = torch.randn(1, 1, WIDTH, generator=generator)
    for index in range(STEPS):
        timestep = torch.full((1, 1, 1), 1.0 - index / STEPS)
        eps = call(latent, timestep, conditioning)
        latent = latent - 0.3 * eps.float()


def _aggregate(report: dict[str, Any], section: str) -> dict[str, dict[str, Any]]:
    return {entry["tap"]: entry for entry in report[section]}


def _over_floor(report: dict[str, Any], k: float, epsilon: float = 1e-9) -> list[str]:
    """Tap paths whose worst rel_l2 clears K x the measured floor, in FIRING order — the
    gate's short-circuit read, computed here because thresholds are the CALLER's."""
    floor = _aggregate(report, "floor_aggregates")
    over: list[str] = []
    for entry in report["aggregates"]:
        worst = entry["worst"]["rel_l2"]
        floor_worst = floor[entry["tap"]]["worst"]["rel_l2"]
        if worst is not None and worst > k * max(floor_worst or 0.0, epsilon):
            over.append(entry["tap"])
    return over


def _round_trips(report: dict[str, Any]) -> None:
    """The document is canonical-JSON clean: no NaN or infinity anywhere in it."""
    assert canonical_json.decode(canonical_json.encode(report)) == report


# ------------------------------------------------------------------------- proof arms


def test_corrupted_block_is_named_first_with_upstream_at_floor() -> None:
    """Scale ONE block's weights in the candidate lane: the FIRST tap over K x floor is
    that block's exact module path. Upstream taps sit AT the floor on EVERY step — teacher
    forcing localizes per forward, so step N's upstream is as clean as step 0's even
    though the corrupt block wrecked the free-running trajectory after step 0."""
    report = run_probe(
        Lane("fp32-ref", _reference_build),
        Lane("scaled", lambda: _scaled_build("blocks.2", 1.05)),
        _drive,
        component="denoiser",
    )
    assert report["format"] == REPORT_FORMAT
    assert [tap["path"] for tap in report["taps"]] == list(ORDERED_TAPS)
    assert all(tap["kind"] == "module" for tap in report["taps"])
    assert len(report["rows"]) == len(ORDERED_TAPS) * STEPS

    over = _over_floor(report, k=10.0)
    assert over[0] == "denoiser.blocks.2"
    assert "denoiser.blocks.3" in over and "denoiser.proj_out" in over  # in-forward wake

    upstream = set(ORDERED_TAPS[:4])
    for row in report["rows"]:
        assert row["nan_inf"] == {"ref": 0, "cand": 0}
        if row["tap"] in upstream:
            assert row["rel_l2"] == 0.0 and row["cosine"] == 1.0  # at the (CPU-zero) floor
        elif row["tap"] == "denoiser.blocks.2":
            assert row["rel_l2"] is not None and row["rel_l2"] > 0.0
    for row in report["floor"]:
        assert row["rel_l2"] == 0.0 and row["max_abs_delta"] == 0.0
    _round_trips(report)


def test_nan_arm_counts_nonfinite_at_the_poisoned_block() -> None:
    """A NaN weight in one block: the report carries exact NaN/Inf counts from that exact
    block onward, zero upstream and zero on the reference side — the signal the gate's
    NaN/Inf short-circuit rejects on. Geometry survives: no NaN reaches the document."""
    import torch

    def poisoned() -> Any:
        model = _reference_build()
        with torch.no_grad():
            model.blocks[2].mlp[0].weight.fill_(float("nan"))
        return model

    report = run_probe(
        Lane("fp32-ref", _reference_build),
        Lane("poisoned", poisoned),
        _drive,
        component="denoiser",
    )
    flagged = [entry["tap"] for entry in report["aggregates"] if entry["nan_inf"]["cand"] > 0]
    assert flagged and flagged[0] == "denoiser.blocks.2"
    assert all(entry["nan_inf"]["ref"] == 0 for entry in report["aggregates"])
    upstream = set(ORDERED_TAPS[:4])
    for row in report["rows"]:
        if row["tap"] in upstream:
            assert row["nan_inf"]["cand"] == 0
        if row["tap"] == "denoiser.blocks.2":
            assert row["nan_inf"]["cand"] > 0
        if row["rel_l2"] is not None:
            assert math.isfinite(row["rel_l2"])
    _round_trips(report)


def test_identical_lane_sits_at_the_floor() -> None:
    """A candidate with the exact reference weights reads as pure floor everywhere."""
    report = run_probe(
        Lane("fp32-ref", _reference_build),
        Lane("twin", _reference_build),
        _drive,
        component="denoiser",
    )
    for row in report["rows"] + report["floor"]:
        assert row["rel_l2"] == 0.0 and row["cosine"] == 1.0
        assert row["snr_db"] is None and row["max_abs_delta"] == 0.0
        assert row["nan_inf"] == {"ref": 0, "cand": 0}
    _round_trips(report)


def test_bf16_lane_moves_off_the_floor_everywhere_but_stays_coherent() -> None:
    """A real encoding lane: the same weights in bfloat16, run in bfloat16 (its natural
    dtype). Every tap shows rounding-scale degradation, none is wrecked — the fp32
    sketches are what make the two dtypes comparable at all."""
    report = run_probe(
        Lane("fp32-ref", _reference_build),
        Lane("bf16", lambda: _reference_build().bfloat16()),
        _drive,
        component="denoiser",
    )
    assert _over_floor(report, k=10.0) == list(ORDERED_TAPS)
    for entry in report["aggregates"]:
        worst = entry["worst"]
        assert worst["rel_l2"] is not None and 0.0 < worst["rel_l2"] < 0.2
        assert worst["cosine"] is not None and worst["cosine"] > 0.99
        assert entry["nan_inf"] == {"ref": 0, "cand": 0}
    _round_trips(report)


def test_zoom_reruns_one_block_subdivided_and_names_the_sublayer() -> None:
    """--zoom in core terms: the plan becomes the zoomed block plus its children, and the
    corrupted sublayer's exact module path is the first tap over K x floor."""
    report = run_probe(
        Lane("fp32-ref", _reference_build),
        Lane("scaled", lambda: _scaled_build("blocks.2.mlp.0", 1.05)),
        _drive,
        component="denoiser",
        zoom="denoiser.blocks.2",
    )
    assert {tap["path"] for tap in report["taps"]} == {
        "denoiser.blocks.2",
        "denoiser.blocks.2.attn",
        "denoiser.blocks.2.norm",
        "denoiser.blocks.2.mlp.0",
        "denoiser.blocks.2.mlp.2",
    }
    assert _over_floor(report, k=10.0)[0] == "denoiser.blocks.2.mlp.0"
    for row in report["rows"]:
        if row["tap"] in ("denoiser.blocks.2.attn", "denoiser.blocks.2.norm"):
            assert row["rel_l2"] == 0.0  # upstream of the corruption inside the block
    _round_trips(report)


def test_non_module_component_degrades_to_a_typed_output_tap() -> None:
    """§1.1 guarantees only state_dict(): a component that is not an nn.Module cannot be
    hooked, so it degrades to ONE component-output tap — typed, never silent."""
    import torch

    def closure(scale: float) -> Any:
        torch.manual_seed(3)
        weight = torch.randn(FEATURES, FEATURES) * scale

        def call(latent: Any, timestep: Any, conditioning: Any) -> Any:
            return latent @ weight

        return call

    report = run_probe(
        Lane("ref", lambda: closure(1.0)),
        Lane("scaled", lambda: closure(1.5)),
        _drive,
        component="fn",
    )
    assert report["taps"] == [{"path": "fn", "kind": "component_output"}]
    assert len(report["rows"]) == STEPS
    for row in report["rows"]:
        assert row["rel_l2"] is not None and row["rel_l2"] > 0.4
    for row in report["floor"]:
        assert row["rel_l2"] == 0.0
    _round_trips(report)


# ------------------------------------------------------------- resolution and refusals


def test_block_resolution_taps_blocks_never_every_module() -> None:
    plan = resolve_taps(_reference_build(), component="denoiser")
    assert isinstance(plan, TapPlan)
    assert tuple(tap.path for tap in plan.taps) == ORDERED_TAPS
    # Nothing inside a block resolved: no norm, no mlp linear, no attention projection.
    assert all(len(plan.module_path(tap).split(".")) <= 2 for tap in plan.taps)


def test_resolution_refuses_every_module() -> None:
    import torch
    from torch import nn

    torch.manual_seed(0)
    wall = nn.Sequential(*(nn.Linear(2, 2) for _ in range(600)))
    with pytest.raises(ProbeRefusal) as caught:
        resolve_taps(wall, component="wall")
    assert caught.value.code == "too_many_taps"


def test_refusals_are_typed() -> None:
    import torch

    def silent_drive(call: Any) -> None:
        return None

    with pytest.raises(ProbeRefusal) as empty:
        run_probe(
            Lane("ref", _reference_build),
            Lane("twin", _reference_build),
            silent_drive,
            component="denoiser",
        )
    assert empty.value.code == "empty_trajectory"

    def shallow() -> Any:
        model = _reference_build()
        del model.blocks[3]
        return model

    with pytest.raises(ProbeRefusal) as topology:
        run_probe(
            Lane("ref", _reference_build),
            Lane("shallow", shallow),
            _drive,
            component="denoiser",
        )
    assert topology.value.code == "topology_divergence"

    def unrecordable_drive(call: Any) -> None:
        call(torch.zeros(1, TOKENS, FEATURES), {"a-set"}, torch.zeros(1, 1, WIDTH))

    with pytest.raises(ProbeRefusal) as unrecordable:
        run_probe(
            Lane("ref", _reference_build),
            Lane("twin", _reference_build),
            unrecordable_drive,
            component="denoiser",
        )
    assert unrecordable.value.code == "unrecordable_input"

    with pytest.raises(ProbeRefusal) as unknown:
        resolve_taps(_reference_build(), component="denoiser", zoom="denoiser.nowhere")
    assert unknown.value.code == "zoom_unknown_path"

    with pytest.raises(ProbeRefusal) as outside:
        resolve_taps(_reference_build(), component="denoiser", zoom="unet.blocks.2")
    assert outside.value.code == "zoom_outside_component"

    with pytest.raises(ProbeRefusal) as unhookable:
        resolve_taps(lambda latent: latent, component="fn", zoom="fn.blocks.0")
    assert unhookable.value.code == "zoom_needs_module"


# ------------------------------------------------------------------- sketch fidelity


def test_sketch_estimates_track_exact_rel_l2_and_cosine() -> None:
    """CountSketch preserves rel-L2 and cosine within ~sqrt(2/dim). The probe judges
    MULTIPLES of a measured floor, so a few percent of sketch noise is resolution enough —
    and this arm pins that epsilon against exact math on a 50k-element activation."""
    import torch

    config = ProbeConfig()
    generator = torch.Generator().manual_seed(11)
    reference_values = torch.randn(50_000, generator=generator)
    noise = torch.randn(50_000, generator=generator)
    for target in (1e-4, 1e-2, 0.3):
        candidate_values = reference_values + noise * (
            target * float(reference_values.norm()) / float(noise.norm())
        )
        ref_side = TapStepAccumulator(config)
        ref_side.fold("tap", reference_values)
        cand_side = TapStepAccumulator(config)
        cand_side.fold("tap", candidate_values)
        stats = compare("tap", 0, ref_side.finalize(), cand_side.finalize())
        exact = float((reference_values - candidate_values).norm() / reference_values.norm())
        assert stats.rel_l2 is not None and abs(stats.rel_l2 - exact) < 0.15 * exact
        exact_cosine = float(
            torch.dot(reference_values, candidate_values)
            / (reference_values.norm() * candidate_values.norm())
        )
        assert stats.cosine is not None and abs(stats.cosine - exact_cosine) < 0.02
        assert stats.max_abs_delta == abs(
            float(reference_values.abs().max()) - float(candidate_values.abs().max())
        )


def test_diverged_shapes_are_refused_not_compared() -> None:
    import torch

    config = ProbeConfig()
    square = TapStepAccumulator(config)
    square.fold("tap", torch.zeros(4, 4))
    flat = TapStepAccumulator(config)
    flat.fold("tap", torch.zeros(16))
    with pytest.raises(ProbeRefusal) as caught:
        compare("tap", 0, square.finalize(), flat.finalize())
    assert caught.value.code == "tap_shape_divergence"
