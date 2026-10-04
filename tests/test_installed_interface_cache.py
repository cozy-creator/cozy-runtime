"""An explicit builder deadline still bounds a package import."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_explicit_builder_deadline_still_bounds_import(tmp_path: Path) -> None:
    (tmp_path / "slow.py").write_text(
        "import time\ntime.sleep(1)\nfrom cozy_runtime.author import App\napp=App()\n"
    )
    (tmp_path / "package.toml").write_text('[application]\nobject="slow:app"\n')
    done = subprocess.run(
        [
            sys.executable,
            "-c",
            "from pathlib import Path; "
            "from cozy_runtime.internal.discovery import discover; "
            f"discover(Path({str(tmp_path)!r}), seconds=0.01)",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode != 0
    assert "import_timeout" in done.stderr or "0.01" in done.stderr
