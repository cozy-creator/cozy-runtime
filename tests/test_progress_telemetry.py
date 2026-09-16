"""Real author telemetry through the Runtime's lossy progress document projection."""

from __future__ import annotations

import socket
import time
from pathlib import Path

import pytest

from cozy_runtime.author._context import Context
from cozy_runtime.author._observations import Observation
from cozy_runtime.author._services import Attempt, ProgressFrame, Telemetry
from cozy_runtime.internal.executor import Executor
from cozy_runtime.internal.seam import Channel
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


def test_author_observation_details_cross_the_real_executor_seam(tmp_path: Path) -> None:
    left, right = socket.socketpair()
    with left, right:
        executor = Executor(Channel(left), tmp_path)
        peer = Channel(right)
        attempt = Attempt("public-events", tmp_path, sink=executor._progress_sink("public-events"))
        telemetry = Telemetry(attempt, Context("public-events", time.monotonic() + 30))

        telemetry.log(
            "reference geometry",
            label="woman",
            normalized="1024x1360",
            source="Bearer " + "x" * 32,
            detail="x" * 500,
        )
        frame = peer.recv(timeout=1)
        assert frame is not None
        document = _progress_document(frame)
        assert document["type"] == "log"
        payload = document["payload"]
        assert isinstance(payload, dict)
        assert payload["name"] == "reference geometry"
        assert payload["value"] == "info"
        assert payload["fields"] == {
            "label": "woman",
            "normalized": "1024x1360",
            "source": "<redacted>",
            "detail": "x" * 399 + "…",
        }
        assert payload["at_unix_ms"] > 0
        assert "event" not in payload and "request_id" not in payload and "kind" not in payload
        assert "Bearer" not in str(document)
        assert attempt.ring.redactions == 1 and attempt.ring.truncated == 1

        telemetry.metric("resident_bytes", 1234, unit="bytes")
        frame = peer.recv(timeout=1)
        assert frame is not None
        metric = _progress_document(frame)
        assert metric["payload"]["name"] == "resident_bytes"
        assert metric["payload"]["value"] == 1234
        assert metric["payload"]["fields"] == {"unit": "bytes"}

        with telemetry.stage("conditioning"):
            pass
        entered = peer.recv(timeout=1)
        elapsed = peer.recv(timeout=1)
        assert entered is not None and elapsed is not None
        assert _progress_document(entered)["type"] == "progress"
        stage = _progress_document(elapsed)
        assert stage["type"] == "stage"
        assert stage["payload"]["name"] == "conditioning"
        assert stage["payload"]["value"] >= 0


def test_load_progress_preserves_component_and_measured_position() -> None:
    assert _progress_document(
        {
            "event": "progress",
            "request_id": "prepare:text_encoder",
            "kind": "load",
            "name": "text_encoder",
            "value": None,
            "position": 4096,
        }
    ) == {
        "type": "load",
        "payload": {"name": "text_encoder", "value": None, "position": 4096},
    }
