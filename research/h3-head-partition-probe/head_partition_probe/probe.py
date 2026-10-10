"""Actual Runtime dense adapters on fixed operands; no models, hooks or collectives."""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import inspect
import time
from pathlib import Path
from typing import Any

from .metrics import accumulate, compared, finish, measured, new_metric, operand_layout

HEADS, DIM = 56, 128
DEGREES = (2, 4, 7, 8)
TOKENS = {"small": 129, "full": 109225}
BACKENDS = {
    "sage3": "sageattention3-fp4-global-lowmem",
    "sage2": "sageattention",
    "kitchen": "kitchen-int8",
}
VERSIONS = {"0.23.2", "0.23.2+dev.sage2producer.795c7d74"}
# Filled from the exact public .23.2 wheel; private795c retains these bytes unchanged.
ATTENTION_HASHES = {
    "0.23.2": "49b92880f01de5c6093fab3344eeab9b2cb72ab5a43b49f31dc7bf93e0b8bab0",
    "0.23.2+dev.sage2producer.795c7d74": (
        "5c857d28bda6ddc76b9f2a25847523974cea59d3604a3945cada15245d7786e5"
    ),
}
SOURCE_HASHES = {
    "internal/attention_upstream.py": (
        "a53013c5e268093f6ac6ef38619673cc571666121d5863894c5b0d0178ebd6a5"
    ),
    "internal/attention_ulysses.py": (
        "0423ec5a9fc46c5d8a82b0a4b9ca6ddf1bf87a96ce0f13ec09ed89bc08fc47dd"
    ),
    "internal/kernel_sources.py": (
        "4473f6d6484249db4830059d2fee04af2102bdf4b9460ff3554db69a2ce5fcdc"
    ),
    "_kernels/_C.abi3.so": "06dd8828d620334912b97bc5bf1ef6ba363d4586ccab882b79b4e4a0d7d7e812",
}


def ranges(degree: int) -> list[tuple[int, int]]:
    if degree not in DEGREES or HEADS % degree:
        raise ValueError("Only the reviewed head partitions are admitted")
    width = HEADS // degree
    return [(start, start + width) for start in range(0, HEADS, width)]


def canonical(value: Any) -> Any:
    """BTHD -> HTD, so accumulating head groups preserves one global index order."""
    return value[0].transpose(0, 1).contiguous()


def metric_rows(value: Any) -> Any:
    return value.reshape(-1, value.shape[-1])


def fresh(tensors: list[Any], start: int = 0, stop: int = HEADS) -> list[Any]:
    import torch

    return [
        value[:, :, start:stop].clone(memory_format=torch.contiguous_format) for value in tensors
    ]


def preprocess(torch: Any, tensors: list[Any], start: int = 0, stop: int = HEADS) -> list[Any]:
    api = importlib.import_module("sageattn3.api")
    q, k, v = (value.transpose(1, 2) for value in fresh(tensors, start, stop))
    # Match Runtime's K ownership/layout before upstream's in-place centering.
    output = api.preprocess_qkv(q, k.contiguous(), v, per_block_mean=False)
    return [value[0].contiguous() for value in output]


def case(torch: Any, local: Any, name: str, tokens: int, seed: int) -> dict[str, Any]:
    # Original operands, full repeat, centering/quantization scratch and optional
    # retained Sage3 preprocessing references remain under this conservative bound.
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
    preprocessing = preprocess(torch, tensors) if name == "sage3" else []
    preprocessing_repeat = []
    if preprocessing:
        repeated_pre = preprocess(torch, tensors)
        preprocessing_repeat = [
            compared(torch, metric_rows(a), metric_rows(b))
            for a, b in zip(repeated_pre, preprocessing, strict=True)
        ]
        del repeated_pre
    groups = []
    for degree in DEGREES:
        output_metric, repeat_metric = new_metric(), new_metric()
        pre_metrics = [new_metric() for _ in preprocessing]
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
            if preprocessing:
                parts = preprocess(torch, tensors, start, stop)
                for metric, actual_pre, original in zip(
                    pre_metrics, parts, preprocessing, strict=True
                ):
                    accumulate(
                        torch, metric, metric_rows(actual_pre), metric_rows(original[start:stop])
                    )
                del parts
        groups.append(
            {
                "degree_simulated_on_one_gpu": degree,
                "heads_per_group": HEADS // degree,
                "output_vs_full": finish(output_metric),
                "same_group_repeat": finish(repeat_metric),
                "preprocessing_vs_full": {
                    key: finish(value)
                    for key, value in zip(
                        ("q_centered", "k_centered", "v_padded", "delta_s"),
                        pre_metrics,
                        strict=bool(pre_metrics),
                    )
                },
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
        "preprocessing_layouts": [operand_layout(value) for value in preprocessing],
        "preprocessing_full_repeat": dict(
            zip(
                ("q_centered", "k_centered", "v_padded", "delta_s"),
                preprocessing_repeat,
                strict=bool(preprocessing_repeat),
            )
        ),
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
    if versions["cozy-runtime"] not in VERSIONS or versions["tensorfs"] != "0.6.1":
        raise RuntimeError(f"Unsupported diagnostic cohort: {versions}")
    source_root = Path(inspect.getfile(attention_upstream)).parents[1]
    actual = {
        name: hashlib.sha256((source_root / name).read_bytes()).hexdigest()
        for name in SOURCE_HASHES
    }
    if (
        actual != SOURCE_HASHES
        or hashlib.sha256((source_root / "internal/attention.py").read_bytes()).hexdigest()
        != ATTENTION_HASHES[versions["cozy-runtime"]]
    ):
        raise RuntimeError(
            "Called kernel adapters/source/native closure differs from the reviewed cohort"
        )
    capability = torch.cuda.get_device_capability()
    if capability not in ((9, 0), (12, 0)):
        raise RuntimeError("Only the owned H100/RTX5090 classes are admitted")
    primary = "sage2" if capability == (9, 0) else "sage3"
    names = [primary, "kitchen"] if selected == "auto" else [selected]
    if "sage3" in names and capability != (12, 0):
        raise RuntimeError("Sage3 is not supported on H100; no fallback")
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
            for label in TOKENS if size == "all" else [size]:
                results.append(
                    {"size": label, **case(torch, candidate.local, name, TOKENS[label], seed)}
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
            "Sage3 preprocessing is separately re-executed through its exact API; not "
            "captured inside the attention call.",
            "Event timings are diagnostic, include cold effects and exclude comparison/copy "
            "work; not a speed benchmark.",
            "No precision flags, deterministic algorithms or serving defaults changed. No "
            "unsupported backend fallback.",
        ],
    }
