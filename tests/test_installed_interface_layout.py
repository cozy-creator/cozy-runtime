"""Child metadata belongs to the selected venv, independently of control Python."""

import json
import sys
from pathlib import Path

import pytest

from cozy_runtime.internal import installed_interfaces


@pytest.mark.skipif(sys.platform == "win32", reason="Unix venv layout")
@pytest.mark.parametrize("version_key", ["version", "version_info"])
def test_child_interfaces_use_selected_venv_minor(tmp_path: Path, version_key: str) -> None:
    minor = "3.13" if sys.version_info.minor == 12 else "3.12"
    (tmp_path / "pyvenv.cfg").write_text(f"{version_key} = {minor}.12\n")
    info = tmp_path / f"lib/python{minor}/site-packages/child-1.0.dist-info"
    info.mkdir(parents=True)
    (info / "package-interface.json").write_text(
        json.dumps(
            {
                "format": "cozy.package.interface/1",
                "application": "child:app",
                "entrypoints": [],
                "jobs": [
                    {
                        "name": "child",
                        "models": [],
                        "request": {"fields": []},
                        "result": {"fields": []},
                        "publishes": False,
                        "invocable": {
                            "module": "child",
                            "export": "child",
                            "memoize": False,
                            "capabilities": [],
                            "context": "ctx",
                            "parameters": [],
                            "defaults": {},
                            "type_names": {"request": "Request", "result": "Result"},
                            "enum_members": {},
                        },
                    }
                ],
            }
        )
    )
    # No executable exists: this is metadata discovery, never an interpreter probe.
    rows = installed_interfaces.for_job(str(tmp_path / "bin/python"), "selected")
    assert len(rows) == 1
    assert rows[0]["module"] == "child"
    assert rows[0]["interface_path"] == str(info / "package-interface.json")
