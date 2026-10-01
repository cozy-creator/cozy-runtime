"""`CUDA_VISIBLE_DEVICES` is where a machine's GPUs are: the Runtime's inventory honours it.

Each case is one worker-shaped process: the host handoff is read once through the config
authority, and its devices drive the driver inventory, as `cozy-runtime-worker` boots."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROWS = "0, Card, GPU-a, 00000000:01:00.0, 8192, 1.0\n1, Card, GPU-b, 00000000:02:00.0, 8192, 1.0\n"

BOOT = """
import json, sys
from pathlib import Path
from cozy_runtime.internal import readiness
from cozy_runtime.internal.config import read_worker_host_config
body = json.load(sys.stdin)
config = read_worker_host_config(body["env"], publication_authority_path=Path(body["absent"]))
gpus = readiness.runtime_gpus(config.child_base_env, config.visible_devices, required=body["cuda"])
print(json.dumps([gpu["device_index"] for gpu in gpus]))
"""


def _inventory(tmp_path: Path, visible: str | None, *, cuda: bool) -> tuple[list[int], bool]:
    """Boot-read the handoff with a recording two-card nvidia-smi on the child PATH."""
    asked = tmp_path / "asked"
    tool = tmp_path / "bin" / "nvidia-smi"
    tool.parent.mkdir()
    tool.write_text(f"#!/bin/sh\necho asked >> {asked}\nprintf '{ROWS}'\n")
    tool.chmod(0o755)
    env = {
        "COZY_WORKER_ID": "worker-test",
        "COZY_WORKER_INTERNAL_PORT": "8443",
        "COZY_MEDIA_INTERNAL_PORT": "8444",
        "COZY_RECORD_OWNER_AUTH_JSON": json.dumps(
            {"control_public_key_ed25519_b64url": "A" * 43, "media_token_sha256": ["b" * 64]}
        ),
        "PATH": f"{tool.parent}:/usr/bin:/bin",
    }
    if visible is not None:
        env["CUDA_VISIBLE_DEVICES"] = visible
    body = {"env": env, "absent": str(tmp_path / "absent.json"), "cuda": cuda}
    run = subprocess.run(
        [sys.executable, "-c", BOOT], input=json.dumps(body), text=True, capture_output=True
    )
    assert run.returncode == 0, run.stderr
    return json.loads(run.stdout), asked.exists()


@pytest.mark.parametrize(
    "visible,indexes",
    [(None, [0, 1]), ("1", [1]), ("GPU-a", [0]), ("1,GPU-a", [0, 1]), ("7", [])],
)
def test_inventory_keeps_the_visible_devices(
    tmp_path: Path, visible: str | None, indexes: list[int]
) -> None:
    # A CUDA base whose operator names devices the driver lacks serves CPU; it is not refused.
    assert _inventory(tmp_path, visible, cuda=visible is None) == (indexes, True)


@pytest.mark.parametrize("visible", ["", " , "])
def test_hidden_devices_serve_cpu_without_asking_the_driver(tmp_path: Path, visible: str) -> None:
    assert _inventory(tmp_path, visible, cuda=True) == ([], False)
