"""CPU content/forward qualification for the separate real-worker GPU diagnosis."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


def test_native_tiny_checkpoint_and_actual_official_warm(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("diffusers")
    fixture = Path(__file__).parent / "testdata"
    code = """import json,sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import h3_group_warm_fixture as fixture
sys.argv = ["fixture", "--root", sys.argv[2], "--prepare-only"]
fixture.main()
from h3_group_warm import Base,BasePipe,CONFIG
from cozy_runtime.author import Config,Device
from cozy_runtime.internal.warm import warm_generation
from cozy_runtime.internal.discovery import discover
from cozy_runtime.internal.package_interface import build
import torch
model=Base.for_test(pipe=BasePipe(Config({"dit":CONFIG})))
warm_generation(model,device=Device("cpu"),cancel=lambda:False)
build(discover(fixture.PACKAGE))
assert not torch.cuda.is_initialized()
print(json.dumps({"cpu_only":True,"interface_built":True}))
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(fixture), str(tmp_path)],
        env={
            **os.environ,
            "CUDA_VISIBLE_DEVICES": "",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    receipts = json.loads((tmp_path / "checkpoints.json").read_text())
    assert sum(row["weight_bytes"] for row in receipts) == 3_705_752
    events = [json.loads(line) for line in result.stderr.splitlines() if line.startswith("{")]
    entries = [row for row in events if row.get("event") == "h3a093.component_enter"]
    assert [(row["tokens"], row["text_tokens"]) for row in entries] == [
        (24, 8),
        (1029, 16),
        (1032, 16),
    ]
    assert json.loads(result.stdout.splitlines()[-1]) == {"cpu_only": True, "interface_built": True}
