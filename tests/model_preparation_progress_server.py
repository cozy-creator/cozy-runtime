"""A real Runtime and native slow-object fixture for the agent's observer tests."""

from __future__ import annotations

import base64
import json
import sys
import time
from pathlib import Path

from test_model_preparation_progress import running_progress_machine

root = Path(sys.argv[1])
fixture = running_progress_machine(root)
machine = next(fixture)
try:
    (root / "ready.json").write_text(
        json.dumps(
            {
                "address": machine.address,
                "request": base64.b64encode(machine.request.SerializeToString()).decode(),
                "manifest": machine.manifest,
                "manifest_length": machine.manifest_length,
                "alternate_manifest": machine.alternate_manifest,
                "first_adapter_manifest": machine.first_adapter_manifest,
                "second_adapter_manifest": machine.second_adapter_manifest,
                "total": machine.total,
            }
        )
    )
    while not (root / "finish").exists():
        if (root / "release").exists():
            machine.release_source.set()
        if machine.source_blocked.is_set():
            (root / "blocked").touch()
        if (root / "hold-description").exists():
            # A real authenticated read stalls on Runtime's owner lock. Model-only
            # preparation does not acquire it, so its native transfer remains independent.
            with machine.worker.claim_lock:
                (root / "description-held").touch()
                while (
                    not (root / "release-description").exists() and not (root / "finish").exists()
                ):
                    if (root / "release").exists():
                        machine.release_source.set()
                    time.sleep(0.02)
        time.sleep(0.02)
finally:
    machine.release_source.set()
    fixture.close()
