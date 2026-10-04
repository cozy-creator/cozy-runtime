"""The worker COUNTS its cards.

A rental is billed by width and Creator reads the paid width back against
`WorkerResources.device_count`, which is this measurement. When `gpu_count` was
`1 if gpu_name else 0`, every wide pod reported one card and was refused
`rental.accelerator_count_mismatch` after it had already been paid for — the
whole sequence-parallel lane was unreachable for that one line.

These drive the real `subprocess` path with an `nvidia-smi` on PATH, not a patched
`measure`, because the parsing of the driver's rows is the thing under test.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from cozy_runtime.internal.hostfacts import measure

H100 = "NVIDIA H100 80GB HBM3, 81559, 580.95.05, 9.0"
H200 = "NVIDIA H200, 143771, 580.95.05, 9.0"


@pytest.fixture(autouse=True)
def _cards_visible(monkeypatch: pytest.MonkeyPatch) -> None:
    """These hosts' cards are the process's to see: a run that hides its own GPU
    (`CUDA_VISIBLE_DEVICES=""`) would otherwise measure none without asking the driver."""
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)


def _driver(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *rows: str) -> None:
    """Put an `nvidia-smi` on PATH that answers `--query-gpu` with these CSV rows."""
    binary = tmp_path / "nvidia-smi"
    body = "\n".join(rows)
    binary.write_text(f'#!/bin/sh\ncat <<"EOF"\n{body}\nEOF\n')
    binary.chmod(binary.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ["PATH"])


def test_one_card_counts_one(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _driver(tmp_path, monkeypatch, H100)
    facts = measure("cuda")
    assert facts.backend == "cuda"
    assert facts.gpu_name == "NVIDIA H100 80GB HBM3"
    assert facts.gpu_count == 1
    assert facts.gpu_sm == 90
    assert "gpu_name" not in facts.unreadable


@pytest.mark.parametrize("width", [2, 4, 8])
def test_a_wide_host_reports_its_width(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, width: int
) -> None:
    """The regression that cost a paid 4xH100 pod: four rows must read as four cards."""
    _driver(tmp_path, monkeypatch, *([H100] * width))
    facts = measure("cuda")
    assert facts.gpu_count == width
    # The per-card facts stay per-card: VRAM is one card's, never the sum.
    assert facts.gpu_name == "NVIDIA H100 80GB HBM3"
    assert facts.vram_total_bytes == 81559 * 1024 * 1024
    assert "gpu_name" not in facts.unreadable


def test_a_mixed_host_is_unreadable_not_guessed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """(device_name, device_count) cannot describe disagreeing cards, so it refuses."""
    _driver(tmp_path, monkeypatch, H100, H200)
    facts = measure("cuda")
    assert facts.gpu_count == 0
    assert facts.gpu_name == ""
    assert "gpu_name" in facts.unreadable


def test_no_driver_is_absent_not_one_card(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PATH", str(tmp_path))
    facts = measure("cuda")
    assert facts.gpu_count == 0
    assert "gpu_name" in facts.unreadable


def test_cpu_profile_holds_no_card(monkeypatch: pytest.MonkeyPatch) -> None:
    facts = measure("none")
    assert facts.backend == "none"
    assert facts.gpu_count == 0
    assert facts.state("gpu_count") == "absent"


def test_a_process_that_sees_no_gpu_measures_none_without_asking(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`CUDA_VISIBLE_DEVICES=""` hides the host's GPUs: the driver on PATH is never run."""
    asked = tmp_path / "asked"
    binary = tmp_path / "nvidia-smi"
    binary.write_text(f"#!/bin/sh\ntouch {asked}\necho '{H100}'\n")
    binary.chmod(binary.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ["PATH"])
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    facts = measure("cuda")
    assert facts.backend == "none" and facts.gpu_count == 0 and not asked.exists()
