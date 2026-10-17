"""`verify_op`'s verdicts on a real filesystem, across real processes: kept per machine by
(kernel, artifact, arch, driver), probed once, and refused for good when a probe raises or ends
its process (a kernel that leaves a sticky CUDA error). No GPU: the probes here are plain code."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from cozy_runtime.internal import accel, kernel_cache, verify
from cozy_runtime.internal.verify import Verdict


@pytest.fixture(autouse=True)
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(kernel_cache, "_MACHINE_STORE", [kernel_cache.Store(tmp_path / "u0")])
    monkeypatch.setattr(verify, "_KNOWN", {})
    return tmp_path / "u0"


def test_a_verdict_is_kept_per_machine_and_a_driver_update_verifies_again(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    probes: list[str] = []

    def probe() -> Verdict:
        probes.append(accel.driver_version())
        return verify.within({"bf16 linear M=3": 2e-3, "fp16 tile 0": 4e-4}, 1e-2)

    first = verify.verified("prebuilt-int8-mmasync", "sha256:abc", 86, probe)
    assert first.ok and first.detail.startswith("rel L2 2.00e-03 at bf16 linear M=3"), first
    monkeypatch.setattr(verify, "_KNOWN", {})  # another process of this machine
    assert verify.verified("prebuilt-int8-mmasync", "sha256:abc", 86, probe) == first
    assert len(probes) == 1
    monkeypatch.setattr(accel, "driver_version", lambda: "999.1")
    assert verify.verified("prebuilt-int8-mmasync", "sha256:abc", 86, probe) == first
    assert verify.verified("prebuilt-int8-mmasync", "sha256:new", 86, probe).ok
    assert probes[1:] == ["999.1", "999.1"]


def test_a_probe_over_budget_or_raising_refuses_the_kernel_for_good() -> None:
    over = verify.verified(
        "prebuilt-fp8-wgmma", "sha256:abc", 90, lambda: verify.within({"tile 3": 0.7}, 1e-2)
    )
    assert not over.ok and "7.00e-01 at tile 3" in over.detail

    def raises() -> Verdict:
        raise RuntimeError("8-bit linear: the GEMM: cudaErrorInvalidKernelImage")

    refused = verify.verified("prebuilt-int8-wgmma", "sha256:abc", 90, raises)
    assert refused == Verdict(
        False, "RuntimeError: 8-bit linear: the GEMM: cudaErrorInvalidKernelImage"
    )
    assert not verify.within({}, 1e-2).ok


def test_a_probe_that_ends_its_process_leaves_the_kernel_refused(store: Path) -> None:
    """A kernel that crashes its process mid-probe: the next process reads the refusal written
    before the probe started, and never probes it again."""
    code = textwrap.dedent(
        """
        import os, sys
        from pathlib import Path
        from cozy_runtime.internal import kernel_cache, verify
        kernel_cache.configure(kernel_cache.Store(Path(sys.argv[1])))
        verify.verified("prebuilt-int8-mmasync", "sha256:abc", 120, lambda: os._exit(3))
        """
    )
    ended = subprocess.run([sys.executable, "-c", code, str(store)], env=dict(os.environ))
    assert ended.returncode == 3

    def never() -> Verdict:
        raise AssertionError("a refused kernel is probed again")

    verdict = verify.verified("prebuilt-int8-mmasync", "sha256:abc", 120, never)
    assert verdict == Verdict(False, "its probe ended the process before a verdict")
