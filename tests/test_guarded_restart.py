"""`cozy-runtime version` reports the guarded-restart capability the CLI reads."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_runtime_cli_reports_guarded_restart_capability() -> None:
    command = [str(Path(sys.executable).with_name("cozy-runtime")), "version", "--json"]
    value = json.loads(subprocess.run(command, capture_output=True, text=True, check=True).stdout)
    assert value["supports_guarded_restart"] is True
    selected = json.loads(
        subprocess.run(
            [*command, "--fields", "distribution"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    )
    assert set(selected) == {"distribution", "next"}
