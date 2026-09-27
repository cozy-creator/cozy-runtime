"""Creator-to-Runtime CLI inputs tolerate fields and formatting from a newer Creator."""

from __future__ import annotations

import hashlib
import io
import json
import os
import random
import subprocess
import sys
import zipfile
from pathlib import Path

from PIL import Image

from cozy_runtime.internal import package_interface, static_interface
from test_image_prepare_cli import request, run
from test_serving_interfaces import project

ROOT = Path(__file__).resolve().parents[1]


def test_image_prepare_ignores_additive_request_fields(tmp_path: Path) -> None:
    source = tmp_path / "input.jpg"
    Image.frombytes("RGB", (256, 256), random.Random(3).randbytes(256 * 256 * 3)).save(source)
    body = request(source)
    body["priority"] = "interactive"
    body["prepare"]["quality_hint"] = 90
    assert run(body)["status"] == "prepared"


def test_interface_wheel_accepts_ordinary_json_with_additive_fields(tmp_path: Path) -> None:
    root = project(tmp_path)
    interface = tmp_path / "package-interface.json"
    interface.write_bytes(package_interface.canonical_bytes(static_interface.build(root)))
    implementation = io.BytesIO()
    with zipfile.ZipFile(implementation, "w") as archive:
        archive.writestr("serving_fixture.py", (root / "serving_fixture.py").read_text())
        archive.writestr(
            "serving_fixture-0.1.0.dist-info/METADATA",
            "Metadata-Version: 2.4\nName: serving-fixture\nVersion: 0.1.0\n",
        )
        archive.writestr(
            "serving_fixture-0.1.0.dist-info/WHEEL",
            "Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
        )
    wheel = tmp_path / "serving_fixture-0.1.0-py3-none-any.whl"
    wheel.write_bytes(implementation.getvalue())
    body = {
        "version": "0.1.0",
        "distribution": "serving-fixture",
        "interface_path": str(interface),
        "output_directory": str(tmp_path / "out"),
        "implementation_wheel": str(wheel),
        "implementation_wheel_digest": "sha256:" + hashlib.sha256(wheel.read_bytes()).hexdigest(),
        "requested_by": "a newer Creator",
    }
    result = subprocess.run(
        [str(Path(sys.executable).with_name("cozy-runtime")), "--json", "interface-wheel"],
        input=json.dumps(body, indent=2),
        text=True,
        capture_output=True,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    document = json.loads(result.stdout)
    built = Path(document["path"])
    assert document["digest"] == "sha256:" + hashlib.sha256(built.read_bytes()).hexdigest()
    assert built.name == wheel.name and built.parent == tmp_path / "out"
    with zipfile.ZipFile(built) as archive:
        assert "serving_fixture.py" in archive.namelist()
