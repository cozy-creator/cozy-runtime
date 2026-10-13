"""torch's allocator is configured in code at the accelerator's init (`accel.initialize`), not
by an environment variable: in a process whose environment names no allocator setting, a
freed 1 GiB block grows by the 512 MiB a 1.5 GiB request adds (expandable segments) instead
of torch reserving a second segment for the whole request."""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

torch = pytest.importorskip("torch")

from cozy_runtime.internal import accel, config  # noqa: E402

PROBE = """
import json, torch
from cozy_runtime.internal import accel
accel.initialize(torch, "cuda")
MIB = 1 << 20
a = torch.empty(1024 * MIB, dtype=torch.uint8, device="cuda")
del a
b = torch.empty(1536 * MIB, dtype=torch.uint8, device="cuda")
print(json.dumps({"reserved_mib": torch.cuda.memory_reserved() // MIB}))
"""


@pytest.mark.skipif(not accel.present(torch, "cuda"), reason="a CUDA card")
def test_the_allocator_grows_a_segment_with_no_environment_naming_it() -> None:
    env = {k: v for k, v in os.environ.items() if not k.startswith("PYTORCH_")}
    done = subprocess.run(
        [sys.executable, "-c", PROBE], env=env, capture_output=True, text=True, timeout=120
    )
    assert done.returncode == 0, done.stderr[-2000:]
    reserved = json.loads(done.stdout.strip().splitlines()[-1])["reserved_mib"]
    assert 1536 <= reserved < 2048, f"one segment grown to the request, not two: {reserved} MiB"


def test_the_seal_report_keeps_the_name_an_older_machine_imposes() -> None:
    """A machine that still imposes the variable compares it with the executor's report and
    refuses the executor when it is missing: the report names it, whatever its value."""
    reported = config.seal_snapshot({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    assert reported["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True"
    assert "PYTORCH_CUDA_ALLOC_CONF" not in config.seal_snapshot({})
