"""Large granted artifacts honor installed field bounds in both executor call paths."""

from __future__ import annotations

import hashlib
import json
import socket
import sys
import threading
from pathlib import Path
from typing import Any

import msgspec
import pytest

from cozy_runtime.author._assets import InputMetadata, file_state
from cozy_runtime.internal import package_interface
from cozy_runtime.internal.discovery import discover
from cozy_runtime.internal.executor import Executor
from cozy_runtime.internal.executor_commands import (
    Activate, InputFile, Invoke, JobBudget, Load, PrepareRequest, RunJob, Start,
)
from cozy_runtime.internal.seam import RESULT_DOCUMENT, Channel

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
@app.entrypoint
def read(payload: Request) -> Result:
    return Result([len(payload.declared.read_bytes()), len(payload.ordinary.read_bytes())])
@app.job(accelerator=False)
def job(payload: Request) -> Result:
    return read(payload)
"""


@pytest.mark.parametrize("kind", ["invoke", "job"])
@pytest.mark.parametrize(
    ("declared", "ordinary", "cap", "accepted"),
    [
        (128 << 20, 1, None, True),
        (1, 128 << 20, None, False),
        (128 << 20, 128 << 20, None, False),
        (128 << 20, 1, 32 << 20, False),
        (128 << 20, 1, 256 << 20, True),
    ],
)
def test_installed_field_capacity_and_explicit_ceiling(
    tmp_path: Path, kind: str, declared: int, ordinary: int, cap: int | None, accepted: bool,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    (project / f"{MODULE}.py").write_text(SOURCE)
    (project / "package.toml").write_text(f'[application]\nobject="{MODULE}:app"\n')
    found = discover(project)
    interface = tmp_path / "package-interface.json"
    interface.write_bytes(package_interface.canonical_bytes(package_interface.build(found)))
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
    root = tmp_path / "root"
    root.mkdir()
    spool = root / "spool"
    binding = {"application": f"{MODULE}:app", "package_interface": str(interface)}
    left, right = socket.socketpair()
    with left, right:
        peer = Channel(right)

        def drain() -> None:
            while peer.recv() is not None:
                pass

        reader = threading.Thread(target=drain, daemon=True)
        reader.start()
        host = Executor(Channel(left), root)
        common: dict[str, Any] = {
            "request_id": "capacity", "spool": str(spool), "deadline_s": 30,
            "inputs": inputs,
        }
        # Omit the field exactly as existing workers do; a present ceiling remains policy.
        if cap is not None:
            common["max_input_bytes"] = cap
        if kind == "job":
            host.discovered = found
            host.surface_names = {surface.name: surface for surface in found.surfaces}
            reply = host.run_job(msgspec.convert(
                {**common, **binding, "job": "job", "payload": payload,
                 "budget": JobBudget(gpu_count=0)}, RunJob,
            ))
        else:
            assert host.start(msgspec.convert({**binding, "devices": ""}, Start))["ok"]
            assert host.load(msgspec.convert(
                {"devices": "", "binding": binding, "budgets": {}, "construction": "a"},
                Load,
            ))["ok"]
            assert host.activate(Activate(construction="a"))["ok"]
            prepared = host.serve_request(PrepareRequest(
                request_id="capacity", entrypoint="read", payload=payload,
                input_metadata={name: InputMetadata(
                    input_id=name, media_type=value.media_type, digest=value.digest,
                    length=value.length, order=value.order,
                ) for name, value in inputs.items()},
            ))
            assert prepared["ok"], prepared
            spool.mkdir(exist_ok=True)
            reply = host.serve_invoke(msgspec.convert({**common, "entrypoint": "read"}, Invoke))
        left.shutdown(socket.SHUT_WR)
        reader.join(timeout=5)
        assert not reader.is_alive()
    sys.modules.pop(MODULE, None)
    if accepted:
        assert reply["outcome"]["terminal"] == "succeeded", reply["outcome"]
        assert json.loads((spool / RESULT_DOCUMENT).read_bytes()) == {
            "lengths": [declared, ordinary],
        }
    else:
        assert reply["outcome"]["code"] == "asset_too_large", reply["outcome"]
        assert not (spool / RESULT_DOCUMENT).exists()
