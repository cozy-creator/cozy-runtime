"""Isolated native SM90 FP8/PEFT probe; does not install or restart Runtime.

Candidate modules are loaded only into this process. Full-M versus row partitions
isolates GEMM geometry on one H100, not distributed communication or video quality.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any


def load_candidates(root: Path) -> dict[str, str]:
    loaded = {}
    for suffix, filename in (
        ("encoding.torchao_rowwise", "torchao_rowwise.py"),
        ("base_weights", "base_weights.py"),
        ("forensics", "forensics.py"),
        ("lora", "lora.py"),
    ):
        name = f"cozy_runtime.internal.{suffix}"
        path = root / filename
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot load candidate {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        loaded[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return loaded


def compare(actual: Any, reference: Any) -> dict[str, Any]:
    """Bounded CPU comparison; excluded from the CUDA timings."""
    import torch

    if torch.equal(actual, reference):
        # Exact CPU comparisons need no floating norm. Keep finite checking
        # chunked so a multi-billion-element output needs no full boolean mask.
        flat = actual.reshape(-1, actual.shape[-1])
        nonfinite = 2 * sum(
            int(torch.count_nonzero(~torch.isfinite(flat[start : start + 256])))
            for start in range(0, flat.shape[0], 256)
        )
        return {
            "exact": True,
            "nonfinite_elements": nonfinite,
            "different_elements": 0,
            "elements": actual.numel(),
            "max_absolute_error": 0.0 if not nonfinite else None,
            "relative_rmse": 0.0 if not nonfinite else None,
            "sha256": hashlib.sha256(
                memoryview(actual.view(torch.uint8).numpy()).cast("B")
            ).hexdigest(),
        }
    different = 0
    nonfinite = 0
    max_abs = error2 = reference2 = 0.0
    a, b = actual.reshape(-1, actual.shape[-1]), reference.reshape(-1, reference.shape[-1])
    for start in range(0, a.shape[0], 256):
        aa, bb = a[start : start + 256], b[start : start + 256]
        nonfinite += int(torch.count_nonzero(~torch.isfinite(aa)))
        nonfinite += int(torch.count_nonzero(~torch.isfinite(bb)))
        different += int(torch.count_nonzero(aa != bb))
        delta = aa.float() - bb.float()
        max_abs = max(max_abs, float(delta.abs().max()))
        error2 += float(delta.double().square().sum())
        reference2 += float(bb.double().square().sum())
    return {
        "exact": different == 0,
        "nonfinite_elements": nonfinite,
        "different_elements": different,
        "elements": actual.numel(),
        "max_absolute_error": max_abs if not nonfinite else None,
        "relative_rmse": math.sqrt(error2 / max(reference2, 1e-100)) if not nonfinite else None,
        "sha256": hashlib.sha256(
            memoryview(actual.view(torch.uint8).numpy()).cast("B")
        ).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--factor-dir", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=8192)
    parser.add_argument("--in-features", type=int, default=5376)
    parser.add_argument("--out-features", type=int, default=7168)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--legacy-lora", action="store_true")
    parser.add_argument("--no-bf16-reduced-precision-reduction", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.rows, args.in_features, args.out_features, args.repeats) < 1:
        raise SystemExit("geometry and repeats must be positive")

    import torch

    from cozy_runtime.author._loader import census
    from cozy_runtime.internal.encoding.leaves import (
        PreQuantized,
        RowwiseNativeLeaf,
        quantize_activation_rowwise,
    )

    legacy_lora = None
    legacy_hash = None
    if args.legacy_lora:
        from cozy_runtime.internal import lora as legacy_lora

        if getattr(legacy_lora, "TILE_ROWS", None) != 4096:
            raise SystemExit("legacy reference must be the installed4096-row implementation")
        legacy_hash = hashlib.sha256(Path(legacy_lora.__file__).read_bytes()).hexdigest()

    candidate_hashes = load_candidates(args.candidate_dir)
    from tensorfs.derived import Part, Tensor

    from cozy_runtime.internal.base_weights import BaseWeights
    from cozy_runtime.internal.encoding.formats import SPEC_ROWWISE_KEEPDIM
    from cozy_runtime.internal.encoding.torchao_rowwise import bind_rowwise_weight
    from cozy_runtime.internal.forensics import distinct_storage_bytes
    from cozy_runtime.internal.lora import AdapterInput, LinearAdapters

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        raise SystemExit("this native qualification is scoped to one SM90 GPU")
    torch.set_num_threads(2)
    if args.no_bf16_reduced_precision_reduction:
        # An isolated upstream-kernel experiment. The worker's process-wide flag
        # is unchanged, and arbitrary-M bit identity remains an empirical check.
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    device = torch.device("cuda")
    in_features, out_features = args.in_features, args.out_features
    weight_bytes = in_features * out_features + out_features * 4
    input_bytes = args.rows * in_features * 2
    # Matches the candidate's conservative admission, plus a context/library margin.
    scratch_bound = max(
        args.rows * (in_features * 10 + out_features * 2 + 12),
        args.rows * (3 * out_features + 32) * 2,
    )
    required_free = 2 * input_bytes + weight_bytes + scratch_bound + (1 << 30)
    free, total = torch.cuda.mem_get_info()
    if free < required_free:
        raise SystemExit(
            f"probe needs {required_free} free bytes including margin; available {free}"
        )
    report: dict[str, Any] = {
        "scope": __doc__,
        "rows": args.rows,
        "in_features": in_features,
        "out_features": out_features,
        "repeats": args.repeats,
        "device": torch.cuda.get_device_name(),
        "capacity": {"free": free, "total": total},
        "required_free_bytes": required_free,
        "temporary_scratch_bound": scratch_bound,
        "candidate_sha256": candidate_hashes,
        "legacy_lora_source_sha256": legacy_hash,
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("torch", "torchao", "peft", "cozy-runtime")
        },
        "flags": {
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "bf16_reduced_precision_reduction": (
                torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
            ),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "scaled_mm_use_fast_accum": True,
        },
        "arms": {},
    }
    torch.manual_seed(7381)
    base = torch.nn.Linear(
        in_features, out_features, bias=False, dtype=torch.bfloat16, device="meta"
    )
    base.bias = torch.nn.Parameter(
        torch.randn(out_features, device=device, dtype=torch.bfloat16), requires_grad=False
    )
    bias = base.bias.detach().clone()
    model = torch.nn.Sequential(base)
    owned = BaseWeights("dit", model, census(SimpleNamespace(components={"dit": model})))
    payload = torch.randint(-100, 101, (out_features, in_features), device=device).to(
        torch.float8_e4m3fn
    )
    scales = torch.linspace(1e-4, 1e-2, out_features, device=device).reshape(-1, 1)
    legacy = RowwiseNativeLeaf("rowwise").leaf(
        torch, {"data": payload, "scale": scales}, base, torch.bfloat16
    )
    spec = Tensor(
        "bf16",
        (out_features, in_features),
        SPEC_ROWWISE_KEEPDIM,
        {
            "data": Part("f8_e4m3fn", (out_features, in_features)),
            "scale": Part("f32", (out_features, 1)),
        },
    )
    bind_rowwise_weight(base, payload, scales, spec=spec)
    x = torch.randn(1, args.rows, in_features, device=device, dtype=torch.bfloat16)
    report["physical_base_parameter_bytes"] = distinct_storage_bytes(base.parameters())

    def parts(partitions: int, pad_to_degree: bool = False) -> tuple[Any, ...]:
        padding = (-args.rows) % partitions if pad_to_degree else 0
        source = torch.nn.functional.pad(x, (0, 0, 0, padding)) if padding else x
        return source.tensor_split(partitions, dim=1)

    def cpu_result(fn: Any, partitions: int = 1, *, pad_to_degree: bool = False) -> Any:
        output = torch.cat(
            [fn(part).detach().cpu() for part in parts(partitions, pad_to_degree)], dim=1
        )
        return output[:, : args.rows].contiguous()

    def measure(fn: Any, partitions: int = 1) -> dict[str, Any]:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        for part in parts(partitions):
            result = fn(part)
            del result
        torch.cuda.synchronize()
        first_ms = (time.perf_counter() - start) * 1000
        times = []
        for _ in range(args.repeats):
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            begin.record()
            for part in parts(partitions):
                result = fn(part)
                del result
            end.record()
            end.synchronize()
            times.append(begin.elapsed_time(end))
        return {
            "first_wall_ms": first_ms,
            "cuda_ms": times,
            "median_cuda_ms": statistics.median(times),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        }

    def capacity(nbytes: int) -> None:
        available, _ = torch.cuda.mem_get_info()
        available += torch.cuda.memory_reserved() - torch.cuda.memory_allocated()
        if nbytes > available:
            raise RuntimeError(f"temporary admission needs {nbytes} bytes; available {available}")

    with torch.no_grad():
        # Identical prequantized operands isolate GEMM/scale/bias from quantization.
        q, scale = quantize_activation_rowwise(torch, x)
        pq = PreQuantized(q, scale, tuple(x.shape))
        old_pq, new_pq = legacy(pq).cpu(), base(pq).cpu()
        report["same_encoded_operands"] = compare(new_pq, old_pq)
        del q, scale, pq, old_pq, new_pq
        reference = cpu_result(legacy)
        report["arms"]["legacy_base"] = measure(legacy)
        report["arms"]["legacy_base"]["row_partitions"] = {
            str(parts): compare(cpu_result(legacy, parts), reference) for parts in (2, 4)
        }
        report["arms"]["legacy_base"]["degree4_pad_crop"] = compare(
            cpu_result(legacy, 4, pad_to_degree=True), reference
        )
        report["arms"]["standard_base"] = {
            **measure(base),
            "versus_legacy": compare(cpu_result(base), reference),
        }
        standard_reference = cpu_result(base)
        report["arms"]["standard_base"]["row_partitions"] = {
            str(parts): compare(cpu_result(base, parts), standard_reference) for parts in (2, 4)
        }
        report["arms"]["standard_base"]["degree4_pad_crop"] = compare(
            cpu_result(base, 4, pad_to_degree=True), standard_reference
        )
        del standard_reference
        binding = LinearAdapters({"dit": model}, {"dit": "base"}, {"dit": ["0"]})
        legacy_model = torch.nn.Sequential(legacy)
        legacy_binding = (
            legacy_lora.LinearAdapters({"dit": legacy_model}, {"dit": "base"}, {"dit": ["0"]})
            if legacy_lora is not None
            else None
        )
        report["arms"]["inactive_peft"] = {
            **measure(model),
            "versus_legacy": compare(cpu_result(model), reference),
        }
        factors = json.loads((args.factor_dir / "cpu-real-factors.json").read_text())["factors"]
        first_spatial = None
        try:
            for label in ("spatial", "wushu", "spatial"):
                row = next(row for row in factors if row["label"] == label)
                tensors = {}
                for letter, info in zip(("A", "B"), row["parts"], strict=True):
                    raw = bytearray((args.factor_dir / f"{label}-{letter}.bin").read_bytes())
                    assert hashlib.sha256(raw).hexdigest() == info["sha256"].removeprefix("sha256:")
                    value = (
                        torch.frombuffer(
                            raw, dtype={"f16": torch.float16, "f32": torch.float32}[info["dtype"]]
                        )
                        .clone()
                        .reshape(info["shape"])
                    )
                    tensors[f"0.lora_{letter}.weight"] = value
                adapter = AdapterInput(
                    label, "dit", "base", tensors, scale={"spatial": 0.3, "wushu": 0.8}[label]
                )
                legacy_arm = None
                legacy_output = None
                if legacy_binding is not None:
                    with legacy_binding.request(
                        [
                            legacy_lora.AdapterInput(
                                label, "dit", "base", tensors, scale=adapter.scale
                            )
                        ],
                        device=device,
                        max_device_bytes=8 << 20,
                        check_cancelled=lambda: None,
                        check_capacity=capacity,
                    ):
                        legacy_output = cpu_result(legacy_model)
                        legacy_arm = measure(legacy_model)
                torch.cuda.empty_cache()
                started = time.perf_counter()
                with binding.request(
                    [adapter],
                    device=device,
                    max_device_bytes=8 << 20,
                    check_cancelled=lambda: None,
                    check_capacity=capacity,
                ):
                    preparation_ms = (time.perf_counter() - started) * 1000
                    full = cpu_result(model)
                    layer = model[0]

                    def adapter_only(value: Any, layer: Any = layer) -> Any:
                        # Diagnostic isolation of the same native PEFT A/B
                        # modules, not a serving implementation or new kernel.
                        return (
                            layer.lora_B["request_0"](layer.lora_A["request_0"](value))
                            * layer.scaling["request_0"]
                        )

                    adapter_full = cpu_result(adapter_only)
                    arm = {
                        "preparation_wall_ms": preparation_ms,
                        "preparation": binding.preparation_facts(),
                        "full": measure(model),
                        "repeat": compare(cpu_result(model), full),
                        "row_partitions": {},
                        "adapter_only_row_partitions": {},
                        "degree4_padded_rows": args.rows + (-args.rows) % 4,
                        "source_parts": row["parts"],
                    }
                    if legacy_output is not None:
                        arm["legacy_fp32_tiled"] = {
                            "timing": legacy_arm,
                            "comparison": compare(full, legacy_output),
                        }
                        del legacy_output
                    for partitions in (2, 4):
                        arm["row_partitions"][str(partitions)] = {
                            "comparison": compare(cpu_result(model, partitions), full),
                            "timing": measure(model, partitions),
                        }
                        arm["adapter_only_row_partitions"][str(partitions)] = compare(
                            cpu_result(adapter_only, partitions), adapter_full
                        )
                    arm["degree4_pad_crop"] = compare(
                        cpu_result(model, 4, pad_to_degree=True), full
                    )
                    arm["adapter_only_degree4_pad_crop"] = compare(
                        cpu_result(adapter_only, 4, pad_to_degree=True), adapter_full
                    )
                    del adapter_full
                    before = [
                        (p, p.data_ptr())
                        for name, p in model.named_parameters()
                        if ".lora_" in name and p.device.type == "cuda"
                    ]
                    owned.evict()
                    owned.reserve(device)
                    bind_rowwise_weight(base, payload, scales, spec=spec)
                    base.bias.copy_(bias)
                    arm["base_evict_reload"] = compare(cpu_result(model), full)
                    assert all(p.data_ptr() == pointer for p, pointer in before)
                    del before
                    arm["factor_objects_survived_base_eviction"] = True
                    if first_spatial is None:
                        first_spatial = full
                    elif label == "spatial":
                        arm["switch_back"] = compare(full, first_spatial)
                    report["arms"][label if label not in report["arms"] else "spatial_repeat"] = arm
                    args.output.write_text(json.dumps(report, indent=2) + "\n")
            report["base_after_requests"] = compare(cpu_result(model), reference)
        finally:
            binding.close()
            if legacy_binding is not None:
                legacy_binding.close()
    report["complete"] = True
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "complete": True, "rows": args.rows}))


if __name__ == "__main__":
    main()
