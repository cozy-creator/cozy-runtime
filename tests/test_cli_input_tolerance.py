"""Creator-to-Runtime CLI inputs tolerate fields and formatting from a newer Creator."""

from __future__ import annotations

import random
from pathlib import Path

from PIL import Image

from test_image_prepare_cli import request, run


def test_image_prepare_ignores_additive_request_fields(tmp_path: Path) -> None:
    source = tmp_path / "input.jpg"
    Image.frombytes("RGB", (256, 256), random.Random(3).randbytes(256 * 256 * 3)).save(source)
    body = request(source)
    body["priority"] = "interactive"
    body["prepare"]["quality_hint"] = 90
    assert run(body)["status"] == "prepared"
