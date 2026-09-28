"""A real kernel compiled at boot on this box's card, from pinned source, and then served.

Needs a CUDA device, torch, diffusers, a CUDA toolkit (`nvcc`) and the kernel-source registry
(`python -m cozy_runtime.internal.kernel_sources fetch <dir>`) at `COZY_KERNEL_SOURCES` or the
image's own location; it skips naming whichever is missing. SageAttention's own `setup.py`
detects the card: nothing is patched and no architecture is named.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from cozy_runtime.internal import kernel_sources


def _skip() -> str:
    for module in ("torch", "diffusers"):
        if importlib.util.find_spec(module) is None:
            return f"{module} is not installed"
    import torch

    if not torch.cuda.is_available():
        return "no CUDA device"
    try:
        kernel_sources.compiler()
    except kernel_sources.Absent as exc:
        return str(exc)
    sources, why = kernel_sources.registry(_sources())
    return "" if "sageattention" in sources else why or "the registry carries no sageattention"


def _sources() -> Path:
    return Path(os.environ.get("COZY_KERNEL_SOURCES") or kernel_sources.ROOT)


def test_sageattention_compiles_on_this_card_and_serves_once_ready(tmp_path: Path) -> None:
    why = _skip()
    if why:
        pytest.skip(why)
    child = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).parent / "testdata" / "kernel_boot_proof.py"),
            str(tmp_path / "kernels" / "u0"),
            str(_sources()),
        ],
        env={**os.environ, "TMPDIR": str(tmp_path)},
        capture_output=True,
        text=True,
        check=True,
    )
    out = json.loads(child.stdout.splitlines()[-1])
    print(json.dumps(out["compile"]))

    # While it compiles, the chain falls through and every reader says why.
    assert out["while_compiling"]["hosts"] == {"dit": {"sdpa": 1}}
    assert "compiling (42%)" in out["while_compiling"]["skipped"]["sageattention"]
    pinned = out["pin_while_compiling"]
    assert pinned["code"] == "attention_kernel_compiling" and "42%" in pinned["detail"]
    [row] = [r for r in out["evidence_while_compiling"] if r["kernel"] == "sageattention"]
    assert row["state"] == "compiling" and row["progress"] == 0.42 and not row["served"]
    assert out["pending_while_compiling"] == []

    # The real compile, started by the warm, never by a request.
    warm = {row["kernel"]: row for row in out["warm"]}
    assert warm["sageattention"]["state"] == "compiling" and warm["sdpa"]["status"] == "ready"
    compiled = out["compile"]
    assert compiled["state"] == "ready", compiled
    assert compiled["compile_ms"] > 0 and compiled["progress"], compiled

    # The next request boundary takes it, and it computes attention.
    assert out["pending_after"] == ["sageattention"]
    assert out["after"]["hosts"] == {"dit": {"sageattention": 1}}
    assert out["after"]["artifacts"]["sageattention"]["artifact"] == "sageattention"
    forward = out["forward"]
    assert forward["shape"] == [1, 4096, 1024] and forward["finite"]
    assert forward["rel_l2_to_sdpa"] < 0.2, forward
    [row] = [r for r in out["evidence_after"] if r["kernel"] == "sageattention"]
    assert row["state"] == "ready" and row["served"]
