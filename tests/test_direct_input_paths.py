"""Input grants verify existing files without borrowing ownership of their bytes."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Annotated, Any

import msgspec
import pytest
from PIL import Image

from cozy_runtime.author import App, AssetBound, ImageAsset, Invocation, MediaDecoder, attempt
from cozy_runtime.author._assets import GrantedInput, file_state
from cozy_runtime.author._media import SNIFF_BYTES, reconcile, sniff
from cozy_runtime.internal.executor_commands import InputFile
from cozy_runtime.internal.sandbox import refuse_network

ROOT = Path(__file__).resolve().parents[1]


class Payload(msgspec.Struct):
    image: Annotated[ImageAsset, AssetBound(max_bytes=1 << 20, max_decoded_bytes=1 << 20)]


class Result(msgspec.Struct):
    path: str
    inode: int
    pixel: str


app = App()


@app.entrypoint
def inspect_input(payload: Payload, decoder: MediaDecoder) -> Result:
    frame = decoder.decode_image(payload.image)
    # This fixture inspects private path custody; real packages only see Assets values.
    source = payload.image._local
    assert source is not None
    return Result(str(source), source.stat().st_ino, frame.rgb[:3].hex())


def grant(source: Path, declared: str = "image/png") -> dict[str, GrantedInput]:
    source = source.resolve(strict=True)
    raw = source.read_bytes()
    actual = reconcile(declared, sniff(raw[:SNIFF_BYTES]))
    assert actual is not None
    return {
        "image": InputFile(
            local=str(source),
            media_type=actual,
            digest="sha256:" + hashlib.sha256(raw).hexdigest(),
            length=len(raw),
            file_state=file_state(source),
        ).granted("image")
    }


def invoke(root: Path, inputs: dict[str, GrantedInput]) -> tuple[Any, Any]:
    # Project across the same JSON seam as a real worker/executor, including file identity.
    command = json.loads(
        json.dumps(
            {
                "inputs": {
                    key: {
                        "local": str(row.local),
                        "file_state": row.file_state,
                        "media_type": row.media_type,
                        "digest": row.digest,
                        "length": row.length,
                        "order": row.order,
                    }
                    for key, row in inputs.items()
                }
            }
        )
    )
    assets = {
        key: msgspec.convert(row, InputFile).granted(key) for key, row in command["inputs"].items()
    }
    result, outcome, _ = attempt(
        app.get("inspect_input"),
        {"image": inputs["image"].digest},
        Invocation("direct", root, time.monotonic() + 10, assets=assets),
    )
    return result, outcome


def test_original_path_inode_and_permissions_survive_sandboxed_execution(tmp_path: Path) -> None:
    source = tmp_path / "original #? 100% café.png"
    Image.new("RGB", (8, 8), (12, 34, 56)).save(source)
    source.chmod(0o600)
    original = source.stat()
    child = subprocess.run(
        [sys.executable, str(Path(__file__)), str(source)],
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert child.returncode == 0, (child.stdout, child.stderr)
    result = json.loads(child.stdout)
    assert result == {"path": str(source), "inode": original.st_ino, "pixel": "0c2238"}
    after = source.stat()
    assert (after.st_ino, after.st_mode, after.st_uid, after.st_gid, after.st_mtime_ns) == (
        original.st_ino,
        original.st_mode,
        original.st_uid,
        original.st_gid,
        original.st_mtime_ns,
    )
    assert not (tmp_path / "spool" / "inputs").exists()


def test_symlink_grant_borrows_its_resolved_source(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    Image.new("RGB", (8, 8), (12, 34, 56)).save(source)
    link = tmp_path / "selected.png"
    link.symlink_to(source)
    verified = grant(link)
    assert verified["image"].local == source
    result, outcome = invoke(tmp_path / "result", verified)
    assert outcome.terminal == "succeeded", outcome
    assert result.result.inode == source.stat().st_ino
    assert link.is_symlink() and source.is_file()


@pytest.mark.parametrize("change", ["write", "replace", "delete"])
def test_changed_borrowed_source_refuses_before_decode(tmp_path: Path, change: str) -> None:
    source = tmp_path / "image.png"
    Image.new("RGB", (8, 8), (12, 34, 56)).save(source)
    verified = grant(source)
    if change == "write":
        # Same size, same inode: checking only size or inode would miss this edit.
        raw = source.read_bytes()
        source.write_bytes(raw[:-1] + bytes([raw[-1] ^ 1]))
    elif change == "replace":
        replacement = tmp_path / "replacement"
        replacement.write_bytes(source.read_bytes())
        replacement.replace(source)
    else:
        source.unlink()
    result, outcome = invoke(tmp_path / "result", verified)
    assert result is None
    assert outcome.code == ("input_unavailable" if change == "delete" else "input_changed")
    assert source.exists() == (change != "delete")


@pytest.mark.parametrize("declared", ["image/jpeg", "image/jpg", "application/octet-stream"])
def test_an_image_declared_under_another_image_type_decodes_as_its_bytes(
    tmp_path: Path, declared: str
) -> None:
    """A `.jpg` that is really a PNG, or an `image/jpg` label, is still the image the field
    asked for: it used to refuse `input_media_type` before the handler ran."""
    source = tmp_path / "photo.jpg"
    Image.new("RGB", (8, 8), (12, 34, 56)).save(source, format="PNG")
    verified = grant(source, declared)
    result, outcome = invoke(tmp_path / "result", verified)
    assert outcome.terminal == "succeeded", outcome
    assert result.result.pixel == "0c2238"


if __name__ == "__main__":
    path = Path(sys.argv[1])
    spool = path.parent / "spool"
    refuse_network("sandboxed direct-input custody proof")
    verified = grant(path)
    result, outcome = invoke(path.parent / "result", verified)
    assert outcome.terminal == "succeeded", outcome
    assert result is not None
    print(msgspec.json.encode(result.result).decode())
