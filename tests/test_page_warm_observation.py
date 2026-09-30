"""Formatting keeps actual work separate from complete readiness and denied admission."""

from cozy_runtime.internal.worker.page_warm import describe


def test_complete_read_reports_its_measured_work_once() -> None:
    line = describe(
        {
            "component": "encoder",
            "bytes": 8192,
            "ms": 4.5,
            "read_bytes": 2048,
            "batches": 2,
            "complete": True,
        }
    )
    assert line.startswith("page-warmed 'encoder':")
    assert line.count("8192 B") == 1 and "4.5 ms" in line
    assert "2048 B read from the block layer" in line and "2 batches" in line
    assert "stopped:" not in line and "partial" not in line


def test_partial_read_keeps_metrics_and_stop_reason_before_component_clipping() -> None:
    line = describe(
        {
            "component": "long_component_" * 32,
            "bytes": 8192,
            "ms": 4.5,
            "read_bytes": 2048,
            "batches": 2,
            "complete": False,
            "skipped": "optional unreclaimed headroom exhausted",
        }
    )
    # The Worker prefixes these notes with the tenant and bounds them to 256 characters.
    visible = ("'machine-" + "a" * 32 + "' " + line)[:256]
    assert "partial page-warm" in visible and "page-warmed" not in visible
    assert visible.count("8192 B") == 1 and "4.5 ms" in visible
    assert "2048 B read from the block layer" in visible and "2 batches" in visible
    assert "stopped: optional unreclaimed headroom exhausted" in visible


def test_denied_read_preserves_reason_without_fabricating_measurements() -> None:
    reason = "host admission declined 16777216 B of prefetch"
    assert describe({"component": "encoder", "skipped": reason}) == reason


def test_unknown_completion_and_unreadable_io_are_not_inferred() -> None:
    line = describe({"component": "encoder", "bytes": 8192, "ms": 4.5, "read_bytes": -1})
    assert line.startswith("page read:") and "page-warmed" not in line
    assert "block-device reads unreadable" in line
    assert "0 B read" not in line
