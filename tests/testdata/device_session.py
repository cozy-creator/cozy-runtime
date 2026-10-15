# Actual executor, channel, native Store and CUDA; no replacement implementations.
from __future__ import annotations

import json
import os
import socket
import sys
import threading
from pathlib import Path
from typing import Any

import msgspec
import tensorfs

from cozy_runtime.internal import package_interface
from cozy_runtime.internal.discovery import discover
from cozy_runtime.internal.executor import Executor, _capture_seal
from cozy_runtime.internal.executor_commands import Activate, Invoke, Load, PrepareRequest, Start
from cozy_runtime.internal.seam import RESULT_DOCUMENT, Channel, SeamError

# Run 6132: an undeclared function holds the machine's GPU, but its environment has no torch.
TORCHLESS = """
import os
import msgspec
from cozy_runtime.author import App, Context
app = App()
class Request(msgspec.Struct):
    pass
class Result(msgspec.Struct):
    device: str
    values: list[float]
@app.entrypoint
def compute(payload: Request, ctx: Context) -> Result:
    return Result(os.environ.get("CUDA_VISIBLE_DEVICES", ""), [42.0])
"""


def main() -> None:
    """A function with no constructed model, run on the session's GPU or (`cpu`) on none."""
    _capture_seal()
    mode, directory = sys.argv[1:]
    assert mode in {"cpu", "gpu", "early", "torchless"}
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    project = root / "project"
    project.mkdir()
    (project / "device_fn.py").write_text(
        TORCHLESS
        if mode == "torchless"
        else """
import msgspec
from cozy_runtime.author import App, Context
app = App()
class Request(msgspec.Struct):
    pass
class Result(msgspec.Struct):
    device: str
    values: list[float]
@app.entrypoint
def compute(payload: Request, ctx: Context) -> Result:
    if ctx.device.type == "cpu":
        return Result("cpu", [])
    import torch
    assert torch.cuda.is_initialized(), "Runtime must initialize before the handler"
    values = torch.full((512,), 7.0, device=str(ctx.device)) * 2
    assert values.device.type == "cuda"
    return Result(str(values.device), values.cpu().tolist())
"""
    )
    (project / "package.toml").write_text('[application]\nobject="device_fn:app"\n')
    found = discover(project)
    interface = root / "package-interface.json"
    interface.write_bytes(package_interface.canonical_bytes(package_interface.build(found)))
    release = {"application": found.application, "package_interface": str(interface)}
    devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
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
        started = process.start(msgspec.convert({**release, "devices": devices}, Start))
        if mode == "early":
            print(json.dumps({"code": started.get("code"), "executions": process.attempts}))
            return
        assert started["ok"], started
        load = {"devices": devices, "binding": release, "construction": "plan"}
        assert process.load(msgspec.convert(load, Load))["ok"]
        assert process.activate(Activate(construction="plan"))["ok"]
        prepared = PrepareRequest(request_id="device", entrypoint="compute", payload={})
        assert process.serve_request(prepared)["ok"]
        spool = root / "spool"
        reply = process.serve_invoke(
            Invoke(request_id="device", entrypoint="compute", spool=str(spool), deadline_s=60)
        )
        left.shutdown(socket.SHUT_WR)
        reader.join(timeout=5)
        assert not reader.is_alive()
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
