"""Real author telemetry through the Runtime's lossy progress document projection."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from cozy_runtime.author._context import Context
from cozy_runtime.author._observations import Observation
from cozy_runtime.author._services import Attempt, ProgressFrame, Telemetry
from cozy_runtime.internal.worker.session import _progress_document


def _telemetry(tmp_path: Path) -> tuple[Telemetry, list[ProgressFrame | Observation]]:
    frames: list[ProgressFrame | Observation] = []
    attempt = Attempt("progress-test", tmp_path, sink=frames.append)
    context = Context("progress-test", time.monotonic() + 30)
    return Telemetry(attempt, context), frames


def _document(frame: ProgressFrame) -> dict[str, object]:
    return _progress_document(
        {
            "kind": "progress",
            "stage": frame.stage,
            "stage_fraction": frame.stage_fraction,
            "overall_fraction": frame.overall_fraction,
            "position": frame.position,
            "total": frame.total,
            "advance": frame.advance,
            "step_ms": frame.step_ms,
        }
    )


def test_counted_stage_keeps_stage_and_overall_progress_distinct(tmp_path: Path) -> None:
    telemetry, emitted = _telemetry(tmp_path)
    on_step = telemetry.step_callback(4, stage="denoising", overall_range=(0.2, 0.6))

    on_step(0)
    on_step(1)
    decode_step = telemetry.step_callback(2, stage="decode", overall_range=(0.6, 1.0))
    decode_step(0)

    frames = [row for row in emitted if isinstance(row, ProgressFrame)]
    assert [(row.stage_fraction, row.overall_fraction) for row in frames] == [
        (0.25, 0.3),
        (0.5, 0.4),
        (0.5, 0.8),
    ]
    assert [(row.position, row.total, row.advance) for row in frames] == [
        (1, 4, 1),
        (2, 4, 2),
        (1, 2, 3),
    ]
    assert all(row.step_ms >= 0 for row in frames)
    assert _document(frames[1]) == {
        "type": "progress",
        "payload": {
            "stage": "denoising",
            "stage_fraction": 0.5,
            "position": 2,
            "total": 4,
            "overall_fraction": 0.4,
            "step_ms": frames[1].step_ms,
        },
    }


def test_uncounted_stage_emits_no_invented_fraction_and_finishes_only_on_success(
    tmp_path: Path,
) -> None:
    telemetry, emitted = _telemetry(tmp_path)
    with telemetry.stage("conditioning", overall_range=(0.0, 0.1)):
        pass

    frames = [row for row in emitted if isinstance(row, ProgressFrame)]
    assert [(row.stage, row.stage_fraction, row.overall_fraction) for row in frames] == [
        ("conditioning", None, 0.0),
        ("conditioning", None, 0.1),
    ]
    assert all(row.position is None and row.total is None for row in frames)

    failed, failed_emitted = _telemetry(tmp_path / "failed")
    with pytest.raises(RuntimeError), failed.stage("decode", overall_range=(0.9, 1.0)):
        raise RuntimeError("planted")
    failed_frames = [row for row in failed_emitted if isinstance(row, ProgressFrame)]
    assert [row.overall_fraction for row in failed_frames] == [0.9]

    uncounted, uncounted_emitted = _telemetry(tmp_path / "uncounted")
    with uncounted.stage("upload"):
        pass
    (uncounted_frame,) = [row for row in uncounted_emitted if isinstance(row, ProgressFrame)]
    assert _document(uncounted_frame) == {
        "type": "progress",
        "payload": {"stage": "upload", "step_ms": 0.0},
    }


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
