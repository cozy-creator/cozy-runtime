"""Installed full TensorFS does not give a function source or output authority."""

from __future__ import annotations

import json
import socket
import threading
from pathlib import Path

import pytest
import tensorfs

from cozy_runtime.internal.discovery import discover
from cozy_runtime.internal.executor import Executor
from cozy_runtime.internal.executor_commands import JobBudget, RunJob
from cozy_runtime.internal.seam import RESULT_DOCUMENT, Channel


@pytest.mark.parametrize("granted", [False, True])
def test_job_storage_comes_from_its_owner_and_refuses_only_when_used(
    tmp_path: Path, granted: bool
) -> None:
    # The regression was hidden by reduced builds: prove Store is actually installed.
    tensorfs.Store.init(tmp_path / "store")
    project = tmp_path / "project"
    project.mkdir()
    (project / "package.toml").write_text('[application]\nobject="storage_grant_job:app"\n')
    (project / "storage_grant_job.py").write_text("""
import msgspec
from cozy_runtime.author import App, CapabilityError, Context, WeightsOutput
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
        ctx.output("model")
    except CapabilityError as exc:
        return Result(exc.code)
    return Result("granted")
""")
    found = discover(project)
    left, right = socket.socketpair()
    with left, right:
        peer = Channel(right)
        frames: list[dict[str, object]] = []

        def receive() -> None:
            while (frame := peer.recv()) is not None:
                frames.append(frame)

        reader = threading.Thread(target=receive, daemon=True)
        reader.start()
        executor = Executor(Channel(left), tmp_path)
        executor.discovered = found
        executor.surface_names = {surface.name: surface for surface in found.surfaces}
        for use_output in (False, True):
            spool = tmp_path / f"spool-{use_output}"
            result = executor.run_job(
                RunJob(
                    job="compute",
                    request_id=f"storage-{use_output}",
                    payload={"use_output": use_output},
                    deadline_s=10,
                    spool=str(spool),
                    application="",
                    package_interface="",
                    budget=JobBudget(gpu_count=0),
                    weights=granted,
                )
            )
            assert result["outcome"]["terminal"] == "succeeded", result
            expected = (
                "no-storage-needed"
                if not use_output
                else "granted"
                if granted
                else "weights_output_ungranted"
            )
            assert json.loads((spool / RESULT_DOCUMENT).read_bytes()) == {"code": expected}
        left.shutdown(socket.SHUT_WR)
        reader.join(timeout=5)
        assert not reader.is_alive()
    assert not any(frame.get("kind") == "weights_writer" for frame in frames)
