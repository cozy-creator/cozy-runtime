"""`verify_op` on a real filesystem, across real processes. A verdict is this process's own: a
failing probe never turns its kernel off for the next process. The one thing a machine keeps is a
probe that ended its process (a kernel that leaves a sticky CUDA error): the next process of that
Runtime build refuses it, and an upgrade checks it again. No GPU: the probes here are plain code.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from cozy_runtime.internal import derive, kernel_cache, verify
from cozy_runtime.internal.verify import Verdict

KERNEL, ARTIFACT = "prebuilt-int8-mmasync", "sha256:abc"


@pytest.fixture(autouse=True)
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(kernel_cache, "_MACHINE_STORE", [kernel_cache.Store(tmp_path / "u0")])
    monkeypatch.setattr(verify, "_KNOWN", {})
    return tmp_path / "u0"


def process(store: Path, probe: str) -> int:
    """Another process of this machine opening the kernel, its probe's body `probe`."""
    code = textwrap.dedent(
        f"""
        import os, sys
        from pathlib import Path
        from cozy_runtime.internal import kernel_cache, verify
        kernel_cache.configure(kernel_cache.Store(Path(sys.argv[1])))
        def probe():
            {probe}
        found = verify.verified({KERNEL!r}, {ARTIFACT!r}, 89, probe)
        sys.exit(0 if found.ok else 1)
        """
    )
    return subprocess.run([sys.executable, "-c", code, str(store)], env=dict(os.environ)).returncode


def test_a_verdict_is_this_processs_and_the_next_process_checks_again(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    probes: list[int] = []

    def probe() -> Verdict:
        probes.append(1)
        return verify.within({"bf16 linear M=3": 2e-3, "fp16 tile 0": 4e-4}, 1e-2)

    first = verify.verified(KERNEL, ARTIFACT, 89, probe)
    assert first.ok and first.detail.startswith("rel L2 2.00e-03 at bf16 linear M=3"), first
    assert verify.verified(KERNEL, ARTIFACT, 89, probe) == first and len(probes) == 1
    monkeypatch.setattr(verify, "_KNOWN", {})  # another process of this machine
    assert verify.verified(KERNEL, ARTIFACT, 89, probe) == first and len(probes) == 2


def test_a_failing_self_check_in_one_process_leaves_the_next_processs_kernel_alone(
    store: Path,
) -> None:
    """0.30.10 kept a failing verdict per machine (a probe the construction guard refused), and
    the prebuilt op stayed off there. A failure is now this process's alone."""
    assert process(store, "raise RuntimeError('cudaErrorInvalidKernelImage')") == 1
    assert process(store, "return verify.within({'tile 0': 0.7}, 1e-2)") == 1
    assert not list((store / verify.VERDICTS).glob("*.json"))
    found = verify.verified(KERNEL, ARTIFACT, 89, lambda: verify.within({"tile 0": 1e-3}, 1e-2))
    assert found.ok and not list((store / verify.VERDICTS).glob("*.json"))


def test_a_probe_over_budget_or_raising_fails_in_its_process() -> None:
    over = verify.verified(
        "prebuilt-fp8-wgmma", ARTIFACT, 90, lambda: verify.within({"tile 3": 0.7}, 1e-2)
    )
    assert not over.ok and "7.00e-01 at tile 3" in over.detail

    def raises() -> Verdict:
        raise RuntimeError("8-bit linear: the GEMM: cudaErrorInvalidKernelImage")

    refused = verify.verified("prebuilt-int8-wgmma", ARTIFACT, 90, raises)
    assert refused == Verdict(
        False, "RuntimeError: 8-bit linear: the GEMM: cudaErrorInvalidKernelImage"
    )
    assert not verify.within({}, 1e-2).ok


def test_a_probe_that_ends_its_process_is_refused_until_the_runtime_changes(
    store: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The crash-loop guard: the next process of this build refuses the kernel without probing
    it; an upgraded Runtime checks it again."""
    assert process(store, "os._exit(3)") == 3

    def never() -> Verdict:
        raise AssertionError("a kernel whose probe ended a process is probed again")

    assert verify.verified(KERNEL, ARTIFACT, 89, never) == Verdict(False, verify.CRASHED)
    monkeypatch.setattr(verify, "_KNOWN", {})
    monkeypatch.setattr(verify, "RUNTIME", verify.RUNTIME + ".next")
    assert verify.verified(KERNEL, ARTIFACT, 89, lambda: verify.within({"a": 0.0}, 1e-2)).ok


def test_a_device_that_ran_out_during_the_probe_keeps_nothing(store: Path) -> None:
    """Out of memory says nothing about the kernel: it raises for the memory manager, and the
    next open checks again."""
    torch = pytest.importorskip("torch")
    probes: list[int] = []

    def short() -> Verdict:
        probes.append(1)
        if len(probes) == 1:
            raise torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 20.00 MiB")
        return verify.within({"bf16 linear M=3": 2e-3}, 1e-2)

    with pytest.raises(torch.OutOfMemoryError):
        verify.verified(KERNEL, ARTIFACT, 89, short)
    assert not list((store / verify.VERDICTS).glob("*.json"))
    assert verify.verified(KERNEL, ARTIFACT, 89, short).ok
    assert len(probes) == 2


def test_a_kernel_opened_inside_a_models_construction_checks_with_the_guard_aside() -> None:
    """A leaf binds, and opens its kernel, inside its model's construction. The probe names
    its device; the guard that refuses an author's device pin steps aside for it."""
    torch = pytest.importorskip("torch")

    def probe() -> Verdict:
        x = torch.ones(4, 4, device="cpu")
        return verify.within({"cpu": verify.error(torch, x @ x, torch.full((4, 4), 4.0))}, 1e-6)

    with derive.serving_substrate("sm89", derive.Observations()):
        with pytest.raises(derive.ConstructionFault):
            torch.ones(1, device="cpu")  # an author's pin is still refused here
        assert verify.verified(KERNEL, ARTIFACT, 89, probe).ok
