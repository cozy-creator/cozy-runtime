"""Ordinary CLI CUDA-stream/lifetime diagnostic; no full-video timing claim."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory

import msgspec

from cozy_runtime.author import App, Context

app = App()


class Request(msgspec.Struct, forbid_unknown_fields=True):
    pass


class Result(msgspec.Struct):
    runtime: str
    tensorfs: str
    gpu: str
    passed: bool
    evidence_json: str


@app.entrypoint
def verify(payload: Request, ctx: Context) -> Result:
    import torch

    from cozy_runtime.internal import attention
    from cozy_runtime.internal import attention_upstream as upstream
    from cozy_runtime.internal.encoding import DeviceFacts

    if ctx.device.type != "cuda":
        raise ValueError("This diagnostic requires an allocated CUDA GPU")
    device = torch.device(str(ctx.device))
    index = device.index or 0
    capability = torch.cuda.get_device_capability(device)
    if capability not in {(12, 0), (12, 1)}:
        raise ValueError("Pinned Sage3 CUDA entrypoint requires SM120 or SM121")
    candidate = attention.BY_NAME["sageattention3-fp4-global-lowmem"]
    facts = DeviceFacts(
        "cuda", torch.cuda.get_device_name(index), capability[0] * 10 + capability[1], "", "", index
    )
    # The normal Runtime store supplies this exact candidate's native dependency.
    artifact = attention._compiled(candidate, facts, wait=True)
    attention._import(candidate)
    generator = torch.Generator(device=device).manual_seed(71831)
    rows = []
    with torch.inference_mode(), TemporaryDirectory(prefix="h3-sage-stream-") as directory:
        for label, ql, kl, dtype in [
            ("aligned_bf16", 128, 128, torch.bfloat16),
            ("tail_bf16", 4097, 4097, torch.bfloat16),
            ("prefix_fp16", 129, 257, torch.float16),
        ]:
            q = torch.randn((1, ql, 2, 128), device=device, dtype=dtype, generator=generator)
            kv = torch.randn((1, kl, 2, 2, 128), device=device, dtype=dtype, generator=generator)
            k, v = kv[:, :, :, 0], kv[:, :, :, 1]
            saved_q, saved_kv = q.clone(), kv.clone()
            expected = upstream._sage3_fp4(q, k, v, per_block_mean=False)
            warmed = upstream.sage3_fp4_global_lowmem(q, k, v)
            del warmed
            torch.cuda.synchronize(device)
            stream = torch.cuda.Stream(device=device)
            default = torch.cuda.default_stream(device)
            if stream.cuda_stream == default.cuda_stream:
                raise ValueError("The diagnostic did not obtain a non-default stream")
            stream.wait_stream(torch.cuda.current_stream(device))
            values = []
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CUDA],
                record_shapes=False,
                profile_memory=False,
                with_stack=False,
            ) as profile:
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        value = upstream.sage3_fp4_global_lowmem(q, k, v)
                        # Reuse the just-freed owned sizes on the SAME stream before waiting.
                        churn = [
                            torch.empty_like(q.transpose(1, 2)),
                            torch.empty_like(k.transpose(1, 2)),
                        ]
                        for scratch in churn:
                            scratch.fill_(17)
                        values.append(value.clone())
                        del value, churn, scratch
                    finished = torch.cuda.Event()
                    finished.record(stream)
                # No whole-device synchronize hides stream dependencies inside this region.
                finished.synchronize()
            path = Path(directory) / (label + ".json")
            profile.export_chrome_trace(str(path))
            raw = path.read_bytes()
            trace = json.loads(raw)
            kernels = [
                event for event in trace.get("traceEvents", []) if event.get("cat") == "kernel"
            ]
            stream_ids = sorted({int(event["args"]["stream"]) for event in kernels})
            exact = all(torch.equal(value, expected) for value in values)
            finite = all(bool(torch.isfinite(value).all()) for value in values)
            unchanged = torch.equal(q, saved_q) and torch.equal(kv, saved_kv)
            if not kernels or len(stream_ids) != 1 or not exact or not finite or not unchanged:
                raise ValueError(f"Sage3 current-stream/lifetime contract failed: {label}")
            rows.append(
                {
                    "case": label,
                    "shape_q": list(q.shape),
                    "shape_kv": list(k.shape),
                    "dtype": str(dtype),
                    "exact_global_reference_and_repeat": exact,
                    "finite": finite,
                    "inputs_unchanged": unchanged,
                    "nondefault_stream_handle": stream.cuda_stream,
                    "default_stream_handle": default.cuda_stream,
                    "cuda_trace_stream_ids": stream_ids,
                    "kernel_names_and_counts": dict(Counter(event["name"] for event in kernels)),
                    "trace_sha256": hashlib.sha256(raw).hexdigest(),
                    "trace_bytes": len(raw),
                    "torch_cpu_operators": False,
                }
            )
    return Result(
        version("cozy-runtime"),
        version("tensorfs"),
        torch.cuda.get_device_name(index),
        True,
        json.dumps(
            {
                "cases": rows,
                "artifact": artifact,
                "scope": "Non-default-stream CUDA arithmetic and allocator-reuse check; no full-video speed claim.",
            },
            allow_nan=False,
        ),
    )
