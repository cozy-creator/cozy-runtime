"""Actual registry dispatch: full/short-Q fixed Sage2 and CuTe plus its fixed prefix."""

from __future__ import annotations

import hashlib
import importlib.metadata
import inspect
import os
import time
from pathlib import Path
from typing import Any

from .metrics import accumulate, compared, finish, new_metric

N, H, D, PREFIX = 109225, 56, 128, 257
DEGREES = (2, 4, 7, 8)
OPTIONS = {
    "attn_mask": None,
    "dropout_p": 0.0,
    "is_causal": False,
    "scale": D**-0.5,
    "enable_gqa": False,
}


def dispatch(ready: Any, values: Any) -> Any:
    from diffusers.models.attention_dispatch import dispatch_attention_fn

    return dispatch_attention_fn(*values, backend=ready.member, parallel_config=None, **OPTIONS)


def canonical(value: Any) -> Any:
    return value[0].transpose(0, 1).contiguous()


def fresh(torch: Any, values: Any, start: int = 0, stop: int = H) -> list[Any]:
    return [t[:, :, start:stop].clone(memory_format=torch.contiguous_format) for t in values]


def exact(metric: dict[str, Any]) -> bool:
    return bool(metric["bytes_equal"] and metric["nonfinite_elements"] == 0)


def head_groups(torch: Any, values: Any, capture: Any) -> dict[str, Any]:
    full = capture(values)
    again = capture(fresh(torch, values))
    repeat = {name: compared(torch, again[name], value) for name, value in full.items()}
    del again
    rows: list[dict[str, Any]] = []
    for degree in DEGREES:
        width = H // degree
        metrics = {k: new_metric() for k in full}
        repeats = {k: new_metric() for k in full}
        for start in range(0, H, width):
            actual = capture(fresh(torch, values, start, start + width))
            again = capture(fresh(torch, values, start, start + width))
            for name, value in actual.items():
                accumulate(torch, metrics[name], value, full[name][start : start + width])
                accumulate(torch, repeats[name], again[name], value)
            del actual, again
        rows.append(
            {
                "local_head_groups": degree,
                "heads_per_call": width,
                "vs_full": {k: finish(v) for k, v in metrics.items()},
                "repeat": {k: finish(v) for k, v in repeats.items()},
            }
        )
    return {
        "full_repeat": repeat,
        "partitions": rows,
        "passed": all(exact(m) for m in repeat.values())
        and all(
            exact(m) for row in rows for group in ["vs_full", "repeat"] for m in row[group].values()
        ),
    }


def run(seed: int) -> dict[str, Any]:
    import torch

    from cozy_runtime.author._attention_scope import AttentionLayout, attention_scope
    from cozy_runtime.internal import (
        accel,
        attention,
        attention_ulysses,
    )
    from cozy_runtime.internal import (
        attention_sage2_fixed as fixed,
    )
    from cozy_runtime.internal import (
        attention_sol as sol,
    )
    from cozy_runtime.internal.encoding import DeviceFacts

    if torch.cuda.get_device_capability() != (9, 0):
        raise RuntimeError("This registered check admits Hopper SM90 only")
    facts = DeviceFacts(
        "cuda",
        torch.cuda.get_device_name(),
        90,
        accel.driver_version(),
        "registered fixed-Kmean check",
        index=torch.cuda.current_device(),
    )
    # Normal Runtime pin resolution compiles/imports/launches the exact registered pair.
    hybrid = attention.pinned(fixed.SOL_NAME, facts)[0]
    dense = attention.pinned(fixed.DENSE_NAME, facts)[0]
    public = attention.pinned("sageattention", facts)[0]
    # The later H3 VAE uses FA3. Resolve its normal public artifact now so a cold
    # build cannot change that later request's automatic dense preference.
    fa3_started = time.monotonic()
    fa3 = attention.pinned("flash-attn3", facts)[0]
    fa3_ready_seconds = time.monotonic() - fa3_started
    compile_rows: list[dict[str, Any]] = []
    with sol.observing(kernels=compile_rows):
        sol.specialize(N, torch.device("cuda"), 3)
        loaded = sol._ready(N, torch.device("cuda"), 4)
    if loaded is None or sol.backend(torch.device("cuda")) != "cute_sm90":
        raise RuntimeError("Actual CuTe SM90 sparse kernel is required; no fallback")
    free, total = torch.cuda.mem_get_info()
    if free + torch.cuda.memory_reserved() - torch.cuda.memory_allocated() < 28 << 30:
        raise RuntimeError("Requires28GiB free/reusable; no shape reduction")
    torch.cuda.reset_peak_memory_stats()
    began = time.monotonic()
    calls = []
    prefix_checks = []
    with torch.inference_mode(), attention_ulysses.observing() as seen:
        generator = torch.Generator(device="cuda").manual_seed(seed)
        values = [
            torch.randn(1, N, H, D, dtype=torch.bfloat16, device="cuda", generator=generator)
            for _ in range(3)
        ]
        cases = []
        for query_rows in (N, PREFIX):
            operands = [values[0][:, :query_rows], values[1], values[2]]
            old = canonical(dispatch(public, operands))
            new = canonical(dispatch(dense, operands))
            original_delta = compared(torch, new, old)
            del old, new

            def dense_capture(v: Any) -> dict[str, Any]:
                return {"output": canonical(dispatch(dense, v))}

            cases.append(
                {
                    "query_rows": query_rows,
                    "kv_rows": N,
                    "candidate_vs_original": original_delta,
                    **head_groups(torch, operands, dense_capture),
                }
            )
            del operands

        def hybrid_capture(v: Any) -> dict[str, Any]:
            raw = sol._native(*v, D**-0.5, PREFIX, loaded)
            token = sol._SITE.set(sol.Site("diagnostic", "transformer_blocks.2.attn"))
            try:
                with attention_scope(AttentionLayout(N, PREFIX, 4, 4)), sol.observing() as counts:
                    result = dispatch(hybrid, v)
            finally:
                sol._SITE.reset(token)
            if counts != {"sparse": 1, "dense_step": 0, "dense_path": 0, "dense_prefix": 1}:
                raise RuntimeError(f"Hybrid did not serve actual sparse andprefix: {counts}")
            calls.append(
                {
                    "heads": v[0].shape[2],
                    "step": 4,
                    "counts": dict(counts),
                    "sparse_impl": sol.backend(v[0].device),
                }
            )
            prefix = dispatch(dense, [v[0][:, :PREFIX], v[1], v[2]])
            prefix_checks.append(compared(torch, canonical(result[:, :PREFIX]), canonical(prefix)))
            return {"raw_sparse": canonical(raw), "hybrid": canonical(result)}

        sparse = head_groups(torch, values, hybrid_capture)
        generator = torch.Generator(device="cuda").manual_seed(seed)
        custody = []
        for value in values:
            original = torch.randn(
                value.shape, dtype=value.dtype, device=value.device, generator=generator
            )
            custody.append(compared(torch, canonical(value), canonical(original)))
            del original
    root = Path(inspect.getfile(attention)).parent
    source = {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in [
            "attention.py",
            "attention_sol.py",
            "attention_sage2_fixed.py",
            "attention_sage2_fixed_mean.py",
        ]
    }
    return {
        "passed": all(
            c["passed"] and c["candidate_vs_original"]["nonfinite_elements"] == 0 for c in cases
        )
        and sparse["passed"]
        and all(exact(m) for m in custody + prefix_checks),
        "seed": seed,
        "software": {
            n: importlib.metadata.version(n)
            for n in ["cozy-runtime", "tensorfs", "torch", "triton"]
        },
        "device": torch.cuda.get_device_name(),
        "device_uuid": str(getattr(torch.cuda.get_device_properties(), "uuid", "")),
        "current_device_index": torch.cuda.current_device(),
        "visible_gpu_count": torch.cuda.device_count(),
        "pid": os.getpid(),
        "driver": accel.driver_version(),
        "capability": [9, 0],
        "shape": [1, N, H, D],
        "registered_dense": dense.name,
        "registered_hybrid": hybrid.name,
        "kernel_artifacts": {
            "dense": dense.artifact,
            "hybrid": hybrid.artifact,
            "public": public.artifact,
        },
        "actual_implementations": {k: sorted(v) for k, v in seen.items()},
        "compile_rows": compile_rows,
        "dependency_warmup": {
            "flash-attn3": fa3.artifact,
            "actual_readiness_seconds": fa3_ready_seconds,
            "scope": "Public FA3 dependency resolve/launch for later VAE; no added comparison",
            "compile_ms_note": (
                "Artifact compile_ms is stored build cost, not necessarily this request's wait"
            ),
        },
        "dense_cases": cases,
        "hybrid_step4": sparse,
        "prefix_matches_registered_dense": prefix_checks,
        "actual_sparse_calls": calls,
        "input_custody": custody,
        "source_sha256": source,
        "elapsed_seconds": time.monotonic() - began,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "device_total_bytes": total,
        "limits": [
            "OneH100 localheadgroups2/4/7/8; noactualcollective/fullpipeline/videoquality.",
            "Same8bitSage2mode; fixedreductioncanchange originalfullheadoutputbits.",
            "No new timingmatrix or whole-generation speed claim.",
        ],
    }
