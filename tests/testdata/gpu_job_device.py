# Actual executor, channel, native Store and CUDA; no replacement implementations.
from __future__ import annotations

import json
import socket
import sys
import threading
from pathlib import Path
from typing import Any

import tensorfs

from cozy_runtime.internal.discovery import discover
from cozy_runtime.internal.executor import Executor
from cozy_runtime.internal.executor_commands import JobBudget, RunJob
from cozy_runtime.internal.seam import RESULT_DOCUMENT, Channel, SeamError


def main() -> None:
    mode, directory = sys.argv[1:]
    assert mode in {"cpu", "gpu", "early"}
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    project = root / "project"
    project.mkdir()
    (project / "device_job.py").write_text("""
import msgspec
from cozy_runtime.author import App, Context
app = App()
class Request(msgspec.Struct):
    pass
class Result(msgspec.Struct):
    device: str
    values: list[float]
@app.job
def compute(payload: Request, ctx: Context) -> Result:
    if ctx.device.type == "cpu":
        return Result("cpu", [])
    import torch
    assert torch.cuda.is_initialized(), "Runtime must initialize before the handler"
    values = torch.full((512,), 7.0, device=str(ctx.device)) * 2
    assert values.device.type == "cuda"
    return Result(str(values.device), values.cpu().tolist())
""")
    (project / "package.toml").write_text('[application]\nobject="device_job:app"\n')
    found = discover(project)
    if mode == "early":
        import torch

        torch.cuda.init()
    left, right = socket.socketpair()
    with left, right:
        peer = Channel(right)

        def receive() -> None:
            try:
                while peer.recv() is not None:
                    pass
            except SeamError:
                return

        reader = threading.Thread(target=receive, daemon=True)
        reader.start()
        process = Executor(Channel(left), root)
        process.discovered = found
        process.surface_names = {surface.name: surface for surface in found.surfaces}
        spool = root / "spool"
        reply = process.run_job(
            RunJob(
                job="compute",
                request_id="job-device",
                payload={},
                deadline_s=60,
                spool=str(spool),
                application="",
                package_interface="",
                budget=JobBudget(gpu_count=0 if mode == "cpu" else 1),
            )
        )
        left.shutdown(socket.SHUT_WR)
        reader.join(timeout=5)
        assert not reader.is_alive()
    if mode == "early":
        assert reply["ok"] is False
        print(
            json.dumps(
                {
                    "code": reply["code"],
                    "executions": process.attempts,
                    "result_written": (spool / RESULT_DOCUMENT).exists(),
                }
            )
        )
        return
    assert reply["outcome"]["terminal"] == "succeeded", reply
    result_bytes = (spool / RESULT_DOCUMENT).read_bytes()
    store = tensorfs.Store.init(root / "store")
    saved = store.put_file(spool / RESULT_DOCUMENT)
    assert store.document(saved["id"], len(result_bytes)) == result_bytes
    result: dict[str, Any] = {
        "result": json.loads(result_bytes),
        "torch_imported": "torch" in sys.modules,
        "device_initialized": process._device_ready,
        "quiescent": reply["quiescent"],
        "gpu_count": reply["metrics"]["gpu_count"],
        "peak_vram_bytes": reply["metrics"]["peak_vram_bytes"],
    }
    print(json.dumps(result))


if __name__ == "__main__":
    main()
