"""Real author telemetry through the Runtime's lossy progress document projection."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from cozy_runtime.author import _activity
from cozy_runtime.author._context import Context
from cozy_runtime.author._observations import Observation
from cozy_runtime.author._services import Attempt, ProgressFrame, Telemetry


def _telemetry(tmp_path: Path) -> tuple[Telemetry, list[ProgressFrame | Observation]]:
    frames: list[ProgressFrame | Observation] = []
    attempt = Attempt("progress-test", tmp_path, sink=frames.append)
    context = Context("progress-test", time.monotonic() + 30)
    return Telemetry(attempt, context), frames


def test_invalid_or_regressing_progress_is_omitted_and_diagnosed(tmp_path: Path) -> None:
    telemetry, emitted = _telemetry(tmp_path)
    telemetry.progress(0.5, stage="denoising", overall_fraction=0.7)
    telemetry.progress(0.6, stage="denoising", overall_fraction=0.6)
    telemetry.progress(float("nan"), stage="denoising", overall_fraction=float("nan"))
    telemetry.progress(0.8, stage="")

    frames = [row for row in emitted if isinstance(row, ProgressFrame)]
    assert [(row.stage_fraction, row.overall_fraction) for row in frames] == [
        (0.5, 0.7),
        (0.6, None),
    ]
    diagnostics = [row for row in emitted if isinstance(row, Observation)]
    assert {str(row.fields.get("reason")) for row in diagnostics} == {
        "overall_fraction must not move backward",
        "stage_fraction must be a finite number in 0..1",
        "overall_fraction must be a finite number in 0..1",
        "stage must be 1..120 printable characters",
    }


def test_composition_scope_maps_nested_stage_once_and_never_completes_failure(
    tmp_path: Path,
) -> None:
    telemetry, emitted = _telemetry(tmp_path)
    with (
        telemetry.scope("Shot 2 of 4", overall_range=(0.25, 0.5)),
        telemetry.stage("denoise", overall_range=(0.2, 0.8)),
    ):
        telemetry.step_callback(4, stage="denoise", overall_range=(0.2, 0.8))(1)
    frames = [row for row in emitted if isinstance(row, ProgressFrame)]
    assert [(row.stage, row.overall_fraction) for row in frames] == [
        ("Shot 2 of 4", 0.25),
        ("Shot 2 of 4 / denoise", 0.3),
        ("Shot 2 of 4 / denoise", 0.375),
        ("Shot 2 of 4 / denoise", 0.45),
        ("Shot 2 of 4", 0.5),
    ]
    with pytest.raises(RuntimeError), telemetry.scope("Shot 3 of 4", overall_range=(0.5, 0.75)):
        telemetry.progress(0.2, stage="denoise", overall_fraction=0.2)
        raise RuntimeError("failed shot")
    frames = [row for row in emitted if isinstance(row, ProgressFrame)]
    assert frames[-1].overall_fraction == 0.55
    with telemetry.scope("Partial video", overall_range=(0.5, 0.51)):
        telemetry.progress(1.0, stage="assemble", overall_fraction=1.0)
    assert (
        max(row.overall_fraction or 0 for row in emitted if isinstance(row, ProgressFrame)) == 0.55
    )


def test_nested_scope_and_unweighted_children_keep_whole_work_unknown(tmp_path: Path) -> None:
    telemetry, emitted = _telemetry(tmp_path)
    with (
        telemetry.scope("Video", overall_range=(0.0, 0.8)),
        telemetry.scope("Shot 1", overall_range=(0.0, 0.5)),
    ):
        telemetry.progress(0.5, stage="denoise", overall_fraction=0.5)
    frames = [row for row in emitted if isinstance(row, ProgressFrame)]
    assert frames[2].stage == "Video / Shot 1 / denoise"
    assert frames[2].overall_fraction == 0.2
    telemetry, emitted = _telemetry(tmp_path / "unknown")
    with telemetry.scope("Unknown plan"):
        telemetry.progress(0.5, stage="denoise", overall_fraction=0.5)
    assert all(row.overall_fraction is None for row in emitted if isinstance(row, ProgressFrame))


def test_a_wait_for_weights_is_off_the_stage_clock_and_its_own_stage(tmp_path: Path) -> None:
    telemetry, frames = _telemetry(tmp_path)
    attempt = telemetry._attempt

    def record(name: str, ms: float) -> None:
        attempt.attribution.stage(name, ms)
        attempt.emit("stage", name, round(ms, 3))

    with _activity.asides(record), telemetry.stage("encode"):
        with _activity.aside("loading weights"):
            time.sleep(0.2)
        with _activity.aside("loading weights"):
            pass  # a wait that was none is not a row
    stages = attempt.attribution.stages
    assert stages["loading weights"].total_ms >= 200 and stages["loading weights"].count == 1
    assert stages["encode"].total_ms < 100, "the wait was counted in the stage open around it"
    rows = [row.name for row in frames if isinstance(row, Observation) and row.kind == "stage"]
    assert rows == ["loading weights", "encode"]


def test_a_stage_ends_when_its_device_work_does(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    telemetry, _ = _telemetry(tmp_path)
    # What a device does: the kernels a stage queued finish after the host left it.
    monkeypatch.setattr(_activity, "_SETTLE", lambda: time.sleep(0.2))
    with telemetry.stage("denoise"):
        pass
    assert telemetry._attempt.attribution.stages["denoise"].total_ms >= 200
