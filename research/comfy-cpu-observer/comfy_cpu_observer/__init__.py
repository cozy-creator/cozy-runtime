"""An actual CPU request owns submission, observation and artifact retention."""

from __future__ import annotations

import sys
from typing import Annotated

import msgspec

from cozy_runtime.author import App, AssetBound, Context, Outputs, Scratch, Tree

from .runner import observe

app = App()


class Request(msgspec.Struct, kw_only=True, forbid_unknown_fields=True):
    graph_json: str
    port: int = 8188
    timeout_s: int = 3600
    trace: bool = False
    output_root: str = "/root/h3-single-bench/ComfyUI/output"
    trace_root: str = "/root/h3-single-bench/full-trace/cuda-only-v2/traces"


class Result(msgspec.Struct):
    status: str
    prompt_id: str
    server_execution_seconds: float | None
    detail: str
    cpu_device: str
    torch_imported: bool
    artifacts: Annotated[Tree, AssetBound(max_bytes=64 << 30)]


@app.job(accelerator=False)
def run(payload: Request, ctx: Context, scratch: Scratch, out: Outputs) -> Result:
    """Submit once to a local Comfy server and retain the actual terminal evidence."""
    if ctx.device.type != "cpu":
        raise ValueError("Comfy observation requires a CPU allocation")
    if not 1 <= payload.port <= 65535 or not 1 <= payload.timeout_s <= 7200:
        raise ValueError("port must be1..65535 and timeout_s1..7200")
    directory = scratch.checkpoint_dir(key="comfy-observer")
    result = observe(
        payload.graph_json,
        port=payload.port,
        timeout_s=payload.timeout_s,
        trace=payload.trace,
        output_root=payload.output_root,
        trace_root=payload.trace_root,
        directory=directory,
        deadline=ctx.deadline,
        check_cancelled=ctx.raise_if_cancelled,
    )
    return Result(
        result["status"],
        result["prompt_id"],
        result["server_execution_seconds"],
        result["detail"],
        str(ctx.device),
        "torch" in sys.modules,
        out.save_tree(directory / "artifacts"),
    )
