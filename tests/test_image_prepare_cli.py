"""Real stdin/process/media handoff for the package-free preparation capability."""

from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

from PIL import Image

from cozy_runtime.author import ImagePreparation
from cozy_runtime.author._decode import DEFAULT_DECODE_LIMITS, _Budget, decode_image_path
from cozy_runtime.internal.exits import Exit

ROOT = Path(__file__).resolve().parents[1]


def request(source: Path, *, edge: int = 64) -> dict[str, Any]:
    # Deliberately ordinary, unsorted JSON: this is a CLI request, not a signed document.
    return {
        "source_path": str(source),
        "prepare": {"profile": "image-fit/1", "max_edge": edge},
        "media_types": ["image/jpeg", "image/png", "image/webp"],
        "max_bytes": 1 << 20,
        "max_decoded_bytes": 0,
    }


def run(body: dict[str, Any], *, expect: int = 0) -> dict[str, Any]:
    directory = Path(body["source_path"]).parent
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "TMPDIR": str(directory)}
    result = subprocess.run(
        [str(Path(sys.executable).with_name("cozy-runtime")), "--json", "image-prepare"],
        input=json.dumps(body),
        text=True,
        capture_output=True,
        cwd=directory,
        env=env,
        timeout=60,
    )
    assert result.returncode == expect, (result.stdout, result.stderr)
    assert not result.stderr if expect == 0 else not result.stdout
    document = json.loads(result.stdout if expect == 0 else result.stderr)
    assert isinstance(document, dict)
    return document


def test_cli_image_handoff_and_raw_paths(tmp_path: Path) -> None:
    source = tmp_path / "large.jpg"
    Image.frombytes("RGB", (512, 512), random.Random(7).randbytes(512 * 512 * 3)).save(
        source, quality=95
    )
    before = source.read_bytes()
    body = request(source)
    answer = run(body)
    assert answer["status"] == "prepared"
    assert "digest" not in answer  # Creator fingerprints the selected carrier.
    derivative = Path(answer["path"])
    assert derivative.parent == tmp_path / "cozy"
    assert derivative.name == hashlib.sha256(derivative.read_bytes()).hexdigest() + ".webp"
    inode = derivative.stat().st_ino
    assert run(body)["path"] == str(derivative)
    assert derivative.stat().st_ino == inode
    assert list(derivative.parent.iterdir()) == [derivative]
    assert answer["length"] == derivative.stat().st_size < len(before)
    expected = decode_image_path(
        source, DEFAULT_DECODE_LIMITS, _Budget(1 << 20), ImagePreparation(max_edge=64)
    )
    actual = decode_image_path(derivative, DEFAULT_DECODE_LIMITS, _Budget(1 << 20))
    assert actual == expected
    assert source.read_bytes() == before

    # Raw paths retain the caller's path, including symlinks.
    linked = tmp_path / "linked.jpg"
    linked.symlink_to(source)
    answer = run(request(linked, edge=1024))
    assert answer["status"] == "raw" and answer["path"] == str(linked)

    # There is no admitted lossless carrier: Runtime performs the bounded resize.
    body["media_types"] = ["image/jpeg"]
    answer = run(body)
    assert answer["status"] == "raw" and answer["path"] == str(source)


def test_cli_keeps_a_smaller_original(tmp_path: Path) -> None:
    # A repeated tile compresses well before resizing; a near-size resample removes
    # that redundancy. Encoding a derivative must not force more bytes onto the wire.
    tile = Image.frombytes("RGB", (32, 32), random.Random(1).randbytes(32 * 32 * 3))
    image = Image.new("RGB", (512, 512))
    for y in range(0, 512, 32):
        for x in range(0, 512, 32):
            image.paste(tile, (x, y))
    source = tmp_path / "tiled.png"
    image.save(source)
    before = source.read_bytes()
    answer = run(request(source, edge=511))
    assert answer["status"] == "raw"
    assert source.read_bytes() == before
    assert sorted(path.name for path in tmp_path.iterdir()) == ["tiled.png"]


def test_cli_rejects_bad_policy_before_writing(tmp_path: Path) -> None:
    source = tmp_path / "image.png"
    Image.new("RGB", (32, 32)).save(source)
    body = request(source)
    body["prepare"]["max_edge"] = 0
    answer = run(body, expect=int(Exit.validation))
    assert answer["error"]["name"] == "image_preparation"
    assert sorted(path.name for path in tmp_path.iterdir()) == ["image.png"]


def test_cli_refuses_corrupt_cached_derivative_without_replacing_it(tmp_path: Path) -> None:
    source = tmp_path / "large.jpg"
    Image.frombytes("RGB", (512, 512), random.Random(7).randbytes(512 * 512 * 3)).save(
        source, quality=95
    )
    body = request(source)
    answer = run(body)
    derivative = Path(answer["path"])
    inode = derivative.stat().st_ino
    derivative.chmod(0o600)
    derivative.write_bytes(b"corrupt")
    answer = run(body, expect=int(Exit.validation))
    assert answer["error"]["name"] == "image_preparation_refused"
    assert derivative.stat().st_ino == inode
    assert derivative.read_bytes() == b"corrupt"
    assert list(derivative.parent.iterdir()) == [derivative]
