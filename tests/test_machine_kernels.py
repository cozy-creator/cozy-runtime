"""The machine's kernel boot through its real entry: real builder processes, a real source
registry and the sealed store an executor reads. The card is the one thing stated: the box
running this has none to compile for.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

from cozy_runtime.internal import (
    accel,
    kernel_cache,
    kernel_compile,
    kernel_sources,
    machine_kernels,
)
from test_kernel_compile import _registry


def _boot(capsys: pytest.CaptureFixture[str]) -> tuple[int, list[dict[str, Any]]]:
    code = machine_kernels.main()
    return code, [json.loads(line) for line in capsys.readouterr().out.splitlines()]


def test_the_boot_compiles_what_the_cards_admit_into_the_store_its_executors_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _registry(tmp_path / "sources")
    scope = tmp_path / "run"
    scope.mkdir()
    monkeypatch.setattr(kernel_sources, "ROOT", root)
    monkeypatch.setattr(accel, "device_capability", lambda entry: {"GPU-a": 120}.get(entry, 0))
    monkeypatch.setattr(kernel_compile, "waiting", None)
    monkeypatch.setattr(kernel_cache, "_MACHINE_STORE", [None])
    monkeypatch.setenv("TMPDIR", str(scope))
    monkeypatch.setattr("tempfile.tempdir", None)
    monkeypatch.setenv(kernel_cache.OWN_ENV, str(tmp_path / "kernels" / "u0"))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-a,GPU-unknown")
    monkeypatch.setattr(sys, "path", list(sys.path))  # the boot puts Sol's sources on it

    code, lines = _boot(capsys)
    # Sol's sources are the registry's one artifact here; SageAttention2 has no source in it,
    # FA3 serves no sm120, and Sol's own object needs a DSL this tree does not carry.
    job = kernel_sources.job(kernel_sources.PYTHON, 120)
    built = [row for row in lines if row.get("state") == "ready"]
    assert [(row["kernel"], row["key"]) for row in built] == [
        (kernel_sources.PYTHON, job.key.digest[:16])
    ]
    assert built[0]["compile_ms"] > 0
    last = lines[-1]
    assert code == 0 and last["done"] is True and last["cards"] == [120], last
    assert (last["kernels"], last["failed"]) == (1, 0) and last["driver"] == accel.driver_version()

    # An executor of this generation reads the same store by the same key.
    executor = kernel_cache.from_sealed({kernel_cache.OWN_ENV: str(tmp_path / "kernels" / "u0")})
    assert executor is not None and last["store"] == str(executor.own)
    served = kernel_compile.submit(executor, job)
    assert served.state == "ready" and Path(served.path, kernel_cache.SITE, "tiny_kernel").is_dir()
    assert kernel_compile.machine_slot(executor.own).exists()

    # The next machine run finds every key built: it starts no builder.
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-a")
    started = dict(kernel_compile._STARTED)
    code, lines = _boot(capsys)
    assert code == 0 and started == kernel_compile._STARTED
    assert next(row["state"] for row in lines if "state" in row) == "ready"


def test_a_machine_without_a_card_or_a_store_compiles_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(accel, "device_capability", lambda entry: 0)
    monkeypatch.setenv(kernel_cache.OWN_ENV, str(tmp_path / "u0"))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    code, lines = _boot(capsys)
    assert code == 0 and lines == [
        {
            "done": True,
            "cards": [],
            "driver": accel.driver_version(),
            "store": str(tmp_path / "u0"),
            "kernels": 0,
            "failed": 0,
            "ms": lines[0]["ms"],
        }
    ]
    assert not (tmp_path / "u0").exists()
