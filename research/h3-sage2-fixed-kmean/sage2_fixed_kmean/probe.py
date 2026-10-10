"""Fixed-input H100 K-mean, quantizer, output and timing comparisons."""

from __future__ import annotations

import hashlib
import importlib.metadata
import inspect
import json
import time
from pathlib import Path

from . import adapter
from .metrics import accumulate, compared, finish, new_metric, operand_layout

HEADS, DIM = 56, 128
DEGREES = (2, 4, 7, 8)
TOKENS = {"small": 129, "full": 109225}


def fresh(torch, values, start=0, stop=HEADS):
    return [x[:, :, start:stop].clone(memory_format=torch.contiguous_format) for x in values]


def capture(torch, core, values, fixed):
    return adapter.canonical_record(
        torch, adapter.stages(core, *values, fixed=fixed), values[0].shape[1]
    )


def case(torch, core, local, tokens, seed):
    free, total = torch.cuda.mem_get_info()
    reusable = torch.cuda.memory_reserved() - torch.cuda.memory_allocated()
    required = 28 << 30 if tokens > 129 else 2 << 30
    if free + reusable < required:
        raise RuntimeError(f"Requires {required} free/reusable bytes; no downsizing")
    torch.cuda.reset_peak_memory_stats()
    gen = torch.Generator(device="cuda").manual_seed(seed)
    values = [
        torch.randn(1, tokens, HEADS, DIM, dtype=torch.bfloat16, device="cuda", generator=gen)
        for _ in range(3)
    ]
    public = local.forward(*fresh(torch, values), scale=DIM**-0.5)[0].transpose(0, 1).contiguous()
    reference = capture(torch, core, fresh(torch, values), False)
    public_control = compared(torch, reference["output"], public)
    if not public_control["bytes_equal"] or public_control["nonfinite_elements"]:
        raise RuntimeError("Exact upstream replay differs from the actual Runtime adapter")
    del public
    rows = []
    for fixed in (False, True):
        full = reference if not fixed else capture(torch, core, fresh(torch, values), True)
        vs_upstream = {}
        if fixed:
            vs_upstream = {k: compared(torch, v, reference[k]) for k, v in full.items()}
            # Keep at most two full records while checking repeats, not the old
            # upstream record plus the candidate and its full repeat together.
            del reference
        repeat = capture(torch, core, fresh(torch, values), fixed)
        repeat_result = {k: compared(torch, v, full[k]) for k, v in repeat.items()}
        del repeat
        parts = []
        for degree in DEGREES:
            width = HEADS // degree
            metrics = {k: new_metric() for k in full}
            repeats = {k: new_metric() for k in full}
            layouts = []
            for start in range(0, HEADS, width):
                stop = start + width
                operands = fresh(torch, values, start, stop)
                layouts.append([operand_layout(x) for x in operands])
                actual = capture(torch, core, operands, fixed)
                del operands
                again = capture(torch, core, fresh(torch, values, start, stop), fixed)
                for k, v in actual.items():
                    accumulate(torch, metrics[k], v, full[k][start:stop])
                    accumulate(torch, repeats[k], again[k], v)
                del actual, again
            parts.append(
                {
                    "degree": degree,
                    "heads_per_group": width,
                    "vs_full": {k: finish(v) for k, v in metrics.items()},
                    "repeat": {k: finish(v) for k, v in repeats.items()},
                    "input_layouts": layouts,
                }
            )
        rows.append(
            {
                "variant": "fixed_kmean" if fixed else "upstream_replay",
                "full_repeat": repeat_result,
                "vs_upstream": vs_upstream,
                "partitions": parts,
                "record_layouts": {k: operand_layout(v) for k, v in full.items()},
            }
        )
        if fixed:
            del full
    # Check all original bytes without keeping an additional full input copy.
    gen = torch.Generator(device="cuda").manual_seed(seed)
    unchanged = []
    for value in values:
        again = torch.randn(value.shape, dtype=value.dtype, device=value.device, generator=gen)
        unchanged.append(compared(torch, value[0].transpose(0, 1), again[0].transpose(0, 1)))
        del again
    if any(not x["bytes_equal"] for x in unchanged):
        raise RuntimeError("Original inputs changed")
    return {
        "tokens": tokens,
        "seed": seed,
        "shape_bthd": [1, tokens, HEADS, DIM],
        "public_replay_control": public_control,
        "inputs_unchanged": unchanged,
        "variants": rows,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "required_headroom_bytes": required,
        "device_total_bytes": total,
    }


def timed(torch, call):
    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    begin.record()
    out = call()
    end.record()
    end.synchronize()
    ms = float(begin.elapsed_time(end))
    del out
    return ms


def timings(torch, core, local, tokens, seed):
    from .kernels import mean

    gen = torch.Generator(device="cuda").manual_seed(seed)
    values = [
        torch.randn(1, tokens, HEADS, DIM, dtype=torch.bfloat16, device="cuda", generator=gen)
        for _ in range(3)
    ]
    calls = {
        "public": lambda: local.forward(*values, scale=DIM**-0.5),
        "replay": lambda: adapter.forward(core, *values),
        "fixed_kmean": lambda: adapter.forward(core, *values, fixed=True),
    }
    for call in calls.values():
        out = call()
        del out
    torch.cuda.synchronize()
    rows = []
    for rep in range(5):
        for name in list(calls) if rep % 2 == 0 else list(reversed(calls)):
            torch.cuda.reset_peak_memory_stats()
            elapsed = timed(torch, calls[name])
            rows.append(
                {
                    "variant": name,
                    "repeat": rep,
                    "complete_attention_cuda_ms": elapsed,
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                }
            )
    return {
        "tokens": tokens,
        "samples": rows,
        "mean_only_cuda_ms": {
            "public": timed(torch, lambda: values[1].mean(dim=1, keepdim=True)),
            "fixed": timed(torch, lambda: mean(values[1])),
        },
        "scope": (
            "Warm CUDA events; no input creation, metrics or compilation. "
            "Primary speed baseline is actual Runtime public callable; not full-video timing."
        ),
    }


def run(size, seed):
    import torch

    from cozy_runtime.internal import accel, attention, attention_upstream
    from cozy_runtime.internal.encoding import DeviceFacts

    software = {
        n: importlib.metadata.version(n) for n in ["cozy-runtime", "tensorfs", "torch", "triton"]
    }
    cohort = json.loads(Path(__file__).with_name("cohort.json").read_text())
    if (
        software["cozy-runtime"] != cohort["runtime_version"]
        or software["tensorfs"] != cohort["tensorfs_version"]
    ):
        raise RuntimeError("Research cohort changed")
    root = Path(inspect.getfile(attention_upstream)).parents[1]
    hashes = {
        n: hashlib.sha256((root / n).read_bytes()).hexdigest() for n in cohort["source_sha256"]
    }
    if hashes != cohort["source_sha256"]:
        raise RuntimeError("Called source/native closure changed")
    if torch.cuda.get_device_capability() != (9, 0):
        raise RuntimeError("This probe admits H100 SM90 only")
    device = DeviceFacts(
        "cuda",
        torch.cuda.get_device_name(),
        90,
        accel.driver_version(),
        "fixed K-mean diagnostic",
        index=torch.cuda.current_device(),
    )
    candidate = attention.BY_NAME["sageattention-hopper-8bit-accurate"]
    artifact = attention._compiled(candidate, device, wait=True)
    attention._import(candidate)
    core = adapter.core()
    started = time.monotonic()
    cases = []
    times = []
    with torch.inference_mode():
        for label in TOKENS if size == "all" else [size]:
            cases.append({"size": label, **case(torch, core, candidate.local, TOKENS[label], seed)})
        for label in TOKENS if size == "all" else [size]:
            times.append(
                {"size": label, **timings(torch, core, candidate.local, TOKENS[label], seed)}
            )
    return {
        "software": software,
        "source_sha256": hashes,
        "upstream_core_sha256": adapter.CORE_SHA,
        "artifact": artifact,
        "device": torch.cuda.get_device_name(),
        "capability": [9, 0],
        "cases": cases,
        "timings": times,
        "elapsed_seconds": time.monotonic() - started,
        "limits": [
            "One GPU with fixed synthetic QKV, no model/LoRA/collectives/VAE/codec.",
            "Only K mean reduction changes; upstream per-thread QK quantization, "
            "V FP8 conversion and native SM90 buffered FP32 accumulation remain.",
            "Fixed mean may differ from upstream full-head reduction; "
            "no visible-quality/full-video identity claim.",
            "28GiB free/reusable admission is an estimate; actual peak retained; no downsizing.",
            "No serving/default change; any further GPU use requires explicit owner lease.",
        ],
    }
