"""Installed full TensorFS gives a function no output authority: the attempt's grant does, and a
granted output is opened by TensorD's writer, never by the executor."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import tensorfs

from cozy_runtime.internal.discovery import discover
from cozy_runtime.internal.seam import RESULT_DOCUMENT
from session import session


@pytest.mark.parametrize("granted", [False, True])
def test_storage_comes_from_the_attempts_grant_and_refuses_only_when_used(
    tmp_path: Path, granted: bool
) -> None:
    # The regression was hidden by reduced builds: prove Store is actually installed.
    tensorfs.Store(tmp_path / "store")
    project = tmp_path / "project"
    project.mkdir()
    (project / "package.toml").write_text('[application]\nobject="storage_grant_fn:app"\n')
    (project / "storage_grant_fn.py").write_text("""
import msgspec
from cozy_runtime.author import App, CapabilityError, Context, WeightsOutput
from cozy_runtime.tensord_channels import Derivation
app = App()
class Request(msgspec.Struct):
    use_output: bool
class Result(msgspec.Struct):
    code: str
@app.entrypoint(accelerator=False, weights=(WeightsOutput("model", max_new_bytes=4096),))
def compute(payload: Request, ctx: Context) -> Result:
    if not payload.use_output:
        return Result("no-storage-needed")
    try:
        ctx.output("model", Derivation(None))
    except CapabilityError as exc:
        return Result(exc.code)
    return Result("granted")
""")
    with session(tmp_path, discover(project)) as host:
        for use_output in (False, True):
            spool = tmp_path / f"spool-{use_output}"
            reply = host.run(
                "compute",
                {"use_output": use_output},
                spool,
                weights={"model": 4096} if granted else {},
            )
            assert reply["outcome"]["terminal"] == "succeeded", reply
            expected = (
                "no-storage-needed"
                if not use_output
                else "no_durable_exchange"  # this harness answers no TensorD request
                if granted
                else "weights_output_ungranted"
            )
            assert json.loads((spool / RESULT_DOCUMENT).read_bytes()) == {"code": expected}
    writer = [f for f in host.frames if f.get("kind") == "weights_writer"]
    assert [f["operation"] for f in writer] == (["output"] if granted else [])
