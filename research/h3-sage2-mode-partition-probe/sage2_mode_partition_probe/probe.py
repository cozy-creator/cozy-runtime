"""Actual Runtime dense adapters on fixed operands; no models, hooks or collectives."""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import inspect
import time
from pathlib import Path
from typing import Any

from . import preprocess
from .metrics import accumulate, compared, finish, measured, new_metric, operand_layout

HEADS, DIM = 56, 128
DEGREES = (2, 4, 7, 8)
TOKENS = {"small": 129, "full": 109225}
BACKENDS = {
    "accurate": "sageattention-hopper-8bit-accurate",
    "per-warp": "sageattention-hopper-8bit-per-warp",
    "fp16pv": "sageattention-hopper-mixed8-16",
    "kitchen": "kitchen-int8",
}


def verify_cohort(versions: dict[str, str], source_root: Path) -> dict[str, str]:
    import json

    expected = json.loads(Path(__file__).with_name("cohort.json").read_text())
    if not expected["runtime_version"]:
        raise RuntimeError("The diagnostic cohort has not yet been frozen")
    if (
        versions["cozy-runtime"] != expected["runtime_version"]
        or versions["tensorfs"] != expected["tensorfs_version"]
    ):
        raise RuntimeError(f"Unsupported diagnostic cohort: {versions}")
    actual = {
        name: hashlib.sha256((source_root / name).read_bytes()).hexdigest()
        for name in expected["source_sha256"]
    }
    if actual != expected["source_sha256"]:
        raise RuntimeError("Called kernel source/native closure differs from reviewed cohort")
    return actual


def ranges(degree: int) -> list[tuple[int, int]]:
    if degree not in DEGREES or HEADS % degree:
        raise ValueError("Only the reviewed head partitions are admitted")
    width = HEADS // degree
    return [(start, start + width) for start in range(0, HEADS, width)]


def canonical(value: Any) -> Any:
    """BTHD -> HTD, so accumulating head groups preserves one global index order."""
    return value[0].transpose(0, 1).contiguous()


def metric_rows(value: Any) -> Any:
    # Keep a head as one row. Flattening tokens into 128-column rows would make
    # the shared metric's row cap launch thousands of tiny reductions. At full
    # H3 length one row is about14M elements; its temporary memory remains bounded.
    return value


def fresh(tensors: list[Any], start: int = 0, stop: int = HEADS) -> list[Any]:
    import torch

    return [
        value[:, :, start:stop].clone(memory_format=torch.contiguous_format) for value in tensors
    ]


def case(
    torch: Any, local: Any, name: str, tokens: int, seed: int, preprocessing_core: Any = None
) -> dict[str, Any]:
    # Original operands, full repeat and upstream centering/quantization scratch
    # remain under the already qualified conservative headroom bound.
    required = 24 << 30 if tokens > 129 else 2 << 30
    free, total = torch.cuda.mem_get_info()
    reusable = torch.cuda.memory_reserved() - torch.cuda.memory_allocated()
    if free + reusable < required:
        raise RuntimeError(
            f"{name}/N{tokens} requires {required} free/reusable bytes; no shape reduced"
        )
    torch.cuda.reset_peak_memory_stats()
    generator = torch.Generator(device="cuda").manual_seed(seed)
    tensors = [
        torch.randn(1, tokens, HEADS, DIM, dtype=torch.bfloat16, device="cuda", generator=generator)
        for _ in range(3)
    ]

    def forward(values: Any) -> Any:
        return local.forward(*values, scale=DIM**-0.5)

    operand_layouts = [operand_layout(value) for value in tensors]
    full_operands = fresh(tensors)
    actual_full_layouts = [operand_layout(value) for value in full_operands]
    reference, full_ms = measured(torch, forward, full_operands)
    full_output_layout = operand_layout(reference)
    del full_operands
    implementation = local.impl(tensors[0], tensors[1], tensors[2], {})
    reference = canonical(reference)
    repeated, repeat_ms = measured(torch, forward, fresh(tensors))
    repeated = canonical(repeated)
    repeat = compared(torch, metric_rows(repeated), metric_rows(reference))
    del repeated

    def inspect_qk(start: int = 0, stop: int = HEADS) -> tuple[dict[str, Any], dict[str, Any]]:
        q, k = fresh(tensors[:2], start, stop)
        return preprocess.run(torch, preprocessing_core, name, q, k)

    pre_reference, pre_contract = inspect_qk() if preprocessing_core is not None else ({}, {})
    pre_repeat = {}
    if pre_reference:
        values, _ = inspect_qk()
        pre_repeat = {
            key: compared(torch, values[key], value) for key, value in pre_reference.items()
        }
        del values
    groups = []
    for degree in DEGREES:
        output_metric, repeat_metric = new_metric(), new_metric()
        pre_metrics = {key: new_metric() for key in pre_reference}
        layouts, output_layouts, elapsed = [], [], 0.0
        for start, stop in ranges(degree):
            operands = fresh(tensors, start, stop)
            layouts.append([operand_layout(x) for x in operands])
            actual, ms = measured(torch, forward, operands)
            output_layouts.append(operand_layout(actual))
            elapsed += ms
            del operands
            actual = canonical(actual)
            accumulate(
                torch, output_metric, metric_rows(actual), metric_rows(reference[start:stop])
            )
            repeated, _ = measured(torch, forward, fresh(tensors, start, stop))
            repeated = canonical(repeated)
            accumulate(torch, repeat_metric, metric_rows(repeated), metric_rows(actual))
            del actual, repeated
            if pre_reference:
                parts, _ = inspect_qk(start, stop)
                for key, metric in pre_metrics.items():
                    accumulate(torch, metric, parts[key], pre_reference[key][start:stop])
                del parts
        groups.append(
            {
                "degree_simulated_on_one_gpu": degree,
                "heads_per_group": HEADS // degree,
                "output_vs_full": finish(output_metric),
                "same_group_repeat": finish(repeat_metric),
                "preprocessing_vs_full": {key: finish(value) for key, value in pre_metrics.items()},
                "fresh_operand_layouts": layouts,
                "raw_output_layouts": output_layouts,
                "isolated_group_cuda_event_ms": elapsed,
            }
        )
    return {
        "backend": name,
        "implementation": implementation,
        "shape_bthd": [1, tokens, HEADS, DIM],
        "seed": seed,
        "full_operand_layouts": operand_layouts,
        "full_repeat": repeat,
        "actual_full_operand_layouts": actual_full_layouts,
        "raw_full_output_layout": full_output_layout,
        "preprocessing_contract": pre_contract,
        "preprocessing_full_repeat": pre_repeat,
        "preprocessing_layouts": {
            key: operand_layout(value) for key, value in pre_reference.items()
        },
        "full_cuda_event_ms": full_ms,
        "repeat_cuda_event_ms": repeat_ms,
        "groups": groups,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "estimated_required_headroom_bytes": required,
        "device_total_bytes": total,
    }


def run(selected: str, size: str, seed: int) -> dict[str, Any]:
    import torch

    from cozy_runtime.internal import accel, attention, attention_upstream
    from cozy_runtime.internal.encoding import DeviceFacts

    versions = {
        name: importlib.metadata.version(name)
        for name in ("cozy-runtime", "tensorfs", "torch", "triton")
    }
    source_root = Path(inspect.getfile(attention_upstream)).parents[1]
    actual = verify_cohort(versions, source_root)
    capability = torch.cuda.get_device_capability()
    if capability != (9, 0):
        raise RuntimeError("This precision-mode diagnostic admits SM90 H100 only")
    if selected != "all" and selected not in BACKENDS:
        raise ValueError(f"Unknown diagnostic mode: {selected}")
    names = list(BACKENDS) if selected == "all" else [selected]
    device = DeviceFacts(
        "cuda",
        torch.cuda.get_device_name(),
        capability[0] * 10 + capability[1],
        accel.driver_version(),
        "head-partition diagnostic",
        index=torch.cuda.current_device(),
    )
    prepared, results = {}, []
    started = time.monotonic()
    with torch.inference_mode():
        for name in names:
            candidate = attention.BY_NAME[BACKENDS[name]]
            # Use Runtime's ordinary pinned cache/compile/import path. The diagnostic
            # calls its raw adapter directly, so it needs no Diffusers/model registry.
            artifact = attention._compiled(candidate, device, wait=True)
            module = attention._import(candidate)
            module_path = Path(inspect.getfile(module))
            prepared[name] = {
                "artifact": artifact,
                "module_path": str(module_path),
                "module_sha256": hashlib.sha256(module_path.read_bytes()).hexdigest(),
            }
            core = None
            if name != "kitchen":
                core, source_hashes = preprocess.load()
                prepared[name]["preprocessing_source_sha256"] = source_hashes
            for label in TOKENS if size == "all" else [size]:
                results.append(
                    {"size": label, **case(torch, candidate.local, name, TOKENS[label], seed, core)}
                )
    return {
        "device": torch.cuda.get_device_name(),
        "capability": list(capability),
        "software": versions,
        "cuda_build": torch.version.cuda,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "source_sha256": actual,
        "prepared": prepared,
        "cases": results,
        "elapsed_seconds": time.monotonic() - started,
        "limits": [
            "Fixed synthetic Q/K/V, one GPU; no model, LoRA, normalization/rope, collective, "
            "VAE or codec.",
            "Full and partitioned calls use fresh contiguous BTHD inputs; comparison indices "
            "are head/token/channel order.",
            "Same per-head mathematical inputs; batch/head geometry can select different "
            "reduction or GPU kernel schedules.",
            "K mean and exposed Q/K quantizers are separately re-executed with exact upstream "
            "arguments; this is not an in-call capture. V/P conversion and accumulation "
            "are not separately inspected. Unused padded scale slots are excluded.",
            "Event timings are diagnostic, include cold effects and exclude comparison/copy "
            "work; not a speed benchmark.",
            "No precision flags, deterministic algorithms or serving defaults changed. No "
            "unsupported backend fallback.",
        ],
    }
