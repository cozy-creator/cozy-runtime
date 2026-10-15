"""Large granted artifacts honor installed field bounds."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

from cozy_runtime.author._assets import InputMetadata, file_state
from cozy_runtime.internal.discovery import discover
from cozy_runtime.internal.executor_commands import InputFile
from cozy_runtime.internal.seam import RESULT_DOCUMENT
from session import session

MODULE = "declared_input_capacity_fixture"
SOURCE = """
from typing import Annotated
import msgspec
from cozy_runtime.author import App, AssetBound, FileAsset
app = App()
class Request(msgspec.Struct):
    declared: Annotated[FileAsset, AssetBound(max_bytes=2 << 30)]
    ordinary: FileAsset
class Result(msgspec.Struct):
    lengths: list[int]
@app.entrypoint(accelerator=False)
def read(payload: Request) -> Result:
    return Result([len(payload.declared.read_bytes()), len(payload.ordinary.read_bytes())])
"""


@pytest.mark.parametrize(
    ("declared", "ordinary", "cap", "accepted"),
    [
        (128 << 20, 1, None, True),
        (1, 128 << 20, None, False),
        (128 << 20, 128 << 20, None, False),
        (128 << 20, 1, 32 << 20, False),
        (128 << 20, 1, 0, False),
        (128 << 20, 1, 256 << 20, True),
    ],
)
def test_installed_field_capacity_and_explicit_ceiling(
    tmp_path: Path, declared: int, ordinary: int, cap: int | None, accepted: bool,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    (project / f"{MODULE}.py").write_text(SOURCE)
    (project / "package.toml").write_text(f'[application]\nobject="{MODULE}:app"\n')
    inputs: dict[str, InputFile] = {}
    for name, size in (("declared", declared), ("ordinary", ordinary)):
        source = tmp_path / name
        with source.open("wb") as target:
            target.truncate(size)
        with source.open("rb") as source_bytes:
            digest = "sha256:" + hashlib.file_digest(source_bytes, "sha256").hexdigest()
        inputs[name] = InputFile(
            local=str(source), media_type="application/octet-stream", digest=digest,
            length=size, order=0, file_state=file_state(source),
        )
    payload = {name: value.digest for name, value in inputs.items()}
    spool = tmp_path / "root" / "capacity"
    (tmp_path / "root").mkdir()
    # Omit the field exactly as existing workers do; a present ceiling remains policy.
    ceiling = {} if cap is None else {"max_input_bytes": cap}
    with session(tmp_path / "root", discover(project)) as host:
        reply = host.run(
            "read",
            payload,
            spool,
            input_metadata={name: InputMetadata(
                input_id=name, media_type=value.media_type, digest=value.digest,
                length=value.length, order=value.order,
            ) for name, value in inputs.items()},
            inputs=inputs,
            **ceiling,
        )
    sys.modules.pop(MODULE, None)
    if accepted:
        assert reply["outcome"]["terminal"] == "succeeded", reply["outcome"]
        assert json.loads((spool / RESULT_DOCUMENT).read_bytes()) == {
            "lengths": [declared, ordinary],
        }
    else:
        assert reply["outcome"]["code"] == "asset_too_large", reply["outcome"]
        assert not (spool / RESULT_DOCUMENT).exists()
