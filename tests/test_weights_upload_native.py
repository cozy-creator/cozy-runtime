"""The actual concurrent gRPC byte path must stay live on a shared native lease."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_concurrent_native_lease_uploads_complete_with_exact_bytes() -> None:
    child = subprocess.run(
        [sys.executable, str(Path(__file__).parent / "testdata/concurrent_weights_upload.py")],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert child.returncode == 0, child.stdout + child.stderr
    rows = [json.loads(line) for line in child.stdout.splitlines() if line.startswith("{")]
    assert rows[-1] == {"phase": "complete", "uploads": 4, "bytes": 32 << 20}
