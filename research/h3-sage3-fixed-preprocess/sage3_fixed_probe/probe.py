"""Bounded fixed-operand localization, identity and timing; ordinary CLI only."""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import inspect
import json
import time
from pathlib import Path

from . import stages
from .metrics import accumulate, compared, finish, new_metric, operand_layout

HEADS, DIM = 56, 128
DEGREES = (2, 4, 7, 8)
TOKENS = {"small": 129, "full": 109225}


def fresh(values, start=0, stop=HEADS):
    import torch

    return [x[:, :, start:stop].clone(memory_format=torch.contiguous_format) for x in values]


def capture(api, values, variant):
    tokens = values[0].shape[1]
    prepared = stages.preprocess(values, variant)
    del values
    quantized = stages.quantize(api, prepared)
    output = stages.finish(api, prepared, quantized, tokens)
    # All record tensors use head as their first axis; offsets are comparable
    # between full56 and groups28/14/8/7 without a concatenated group allocation.
    record = {name: value[0].contiguous() for name, value in {**prepared, **quantized}.items()}
    record["output"] = output[0].transpose(0, 1).contiguous()
    return record


def compare_records(torch, actual, reference):
    if actual.keys() != reference.keys():
        raise RuntimeError("Diagnostic stages changed between compared records")
    return {name: compared(torch, value, reference[name]) for name, value in actual.items()}


def api_control(torch, api, values):
    q, k, v = fresh(values)
    actual = api.preprocess_qkv(
        q.transpose(1, 2), k.transpose(1, 2).contiguous(), v.transpose(1, 2), per_block_mean=False
    )
    expected = stages.upstream_preprocess(*fresh(values))
    result = {
        name: compared(torch, a[0].contiguous(), expected[name][0].contiguous())
        for name, a in zip(("q", "k", "v", "delta"), actual, strict=True)
    }
    if any(not r["bytes_equal"] or r["nonfinite_elements"] for r in result.values()):
        raise RuntimeError("Owned upstream replay differs from the actual pinned API")
    return result


def case(torch, api, public_local, tokens, seed):
    free, total = torch.cuda.mem_get_info()
    reusable = torch.cuda.memory_reserved() - torch.cuda.memory_allocated()
    required = 28 << 30 if tokens > 129 else 2 << 30
    if free + reusable < required:
        raise RuntimeError(
            f"Full diagnostic needs{required}free/reusable bytes; no shape reduction"
        )
    torch.cuda.reset_peak_memory_stats()
    gen = torch.Generator(device="cuda").manual_seed(seed)
    original = [
        torch.randn(1, tokens, HEADS, DIM, dtype=torch.bfloat16, device="cuda", generator=gen)
        for _ in range(3)
    ]
    layouts = [operand_layout(x) for x in original]
    replay = api_control(torch, api, original)
    variants = []
    # Only one full reference and one comparison record coexist. No retained
    # copies per degree, no CPU tensor dumps, no atomics or changing precisionflags.
    for variant in stages.VARIANTS:
        reference = capture(api, fresh(original), variant)
        actual_callable = (
            public_local.forward(*fresh(original), scale=DIM**-0.5)
            if variant == "upstream"
            else stages.fixed_complete(api, fresh(original))
            if variant == "fixed_all"
            else None
        )
        callable_control = {}
        if actual_callable is not None:
            canonical = actual_callable[0].transpose(0, 1).contiguous()
            callable_control = compared(torch, canonical, reference["output"])
            if not callable_control["bytes_equal"] or callable_control["nonfinite_elements"]:
                raise RuntimeError("Instrumented replay differs from its actual timed callable")
            del actual_callable, canonical
        repeated = capture(api, fresh(original), variant)
        repeat = compare_records(torch, repeated, reference)
        del repeated
        versus_upstream = {}
        if variant != "upstream":
            upstream = capture(api, fresh(original), "upstream")
            versus_upstream = compare_records(torch, reference, upstream)
            del upstream
        partitions = []
        for degree in DEGREES:
            width = HEADS // degree
            metrics = {name: new_metric() for name in reference}
            repeats = {name: new_metric() for name in reference}
            group_layouts = []
            for start in range(0, HEADS, width):
                stop = start + width
                operands = fresh(original, start, stop)
                group_layouts.append([operand_layout(x) for x in operands])
                actual = capture(api, operands, variant)
                del operands
                for name, value in actual.items():
                    accumulate(torch, metrics[name], value, reference[name][start:stop])
                repeated = capture(api, fresh(original, start, stop), variant)
                for name, value in repeated.items():
                    accumulate(torch, repeats[name], value, actual[name])
                del actual, repeated
            partitions.append(
                {
                    "degree_simulated_on_one_gpu": degree,
                    "heads_per_group": width,
                    "versus_full": {name: finish(m) for name, m in metrics.items()},
                    "same_group_repeat": {name: finish(m) for name, m in repeats.items()},
                    "operand_layouts": group_layouts,
                }
            )
        variants.append(
            {
                "variant": variant,
                "full_repeat": repeat,
                "versus_upstream": versus_upstream,
                "partitions": partitions,
                "actual_callable_replay_control": callable_control,
                "full_record_layouts": {name: operand_layout(x) for name, x in reference.items()},
            }
        )
        del reference
    output = {
        "shape_bthd": [1, tokens, HEADS, DIM],
        "seed": seed,
        "input_layouts": layouts,
        "upstream_replay_control": replay,
        "variants": variants,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "headroom_required_bytes": required,
        "device_total_bytes": total,
    }
    check_generator = torch.Generator(device="cuda").manual_seed(seed)
    unchanged = []
    for original_value in original:
        regenerated = torch.randn(
            1, tokens, HEADS, DIM, dtype=torch.bfloat16, device="cuda", generator=check_generator
        )
        unchanged.append(compared(torch, original_value[0], regenerated[0]))
        del regenerated
    output["original_inputs_unchanged"] = unchanged
    if any(not item["bytes_equal"] for item in unchanged):
        raise RuntimeError("A diagnostic path mutated fixed original inputs")
    del original
    return output


def measure(torch, function):
    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    begin.record()
    value = function()
    end.record()
    end.synchronize()
    elapsed = float(begin.elapsed_time(end))
    del value
    return elapsed


def timings(torch, api, public_local, tokens, seed):
    from .kernels import fixed_preprocess

    gen = torch.Generator(device="cuda").manual_seed(seed)
    values = [
        torch.randn(1, tokens, HEADS, DIM, dtype=torch.bfloat16, device="cuda", generator=gen)
        for _ in range(3)
    ]

    def public_preprocess():
        q, k, v = (x.transpose(1, 2) for x in values)
        return api.preprocess_qkv(
            q, k.clone(memory_format=torch.contiguous_format), v, per_block_mean=False
        )

    calls = {
        "public_upstream": (
            public_preprocess,
            lambda: public_local.forward(*values, scale=DIM**-0.5),
        ),
        "fixed_all_candidate": (
            lambda: fixed_preprocess(*values),
            lambda: stages.fixed_complete(api, values),
        ),
        "upstream_instrumented_replay": (
            lambda: stages.preprocess(values, "upstream"),
            lambda: stages.complete(api, values, "upstream"),
        ),
        "token_last_control": (
            lambda: stages.preprocess(values, "token_last"),
            lambda: stages.complete(api, values, "token_last"),
        ),
        "fixed_means_control": (
            lambda: stages.preprocess(values, "fixed_means"),
            lambda: stages.complete(api, values, "fixed_means"),
        ),
    }
    for pre, complete in calls.values():
        value = pre()
        del value
        value = complete()
        del value
    torch.cuda.synchronize()
    rows = []
    for repetition in range(3):
        order = list(calls) if repetition % 2 == 0 else list(reversed(calls))
        for variant in order:
            torch.cuda.reset_peak_memory_stats()
            pre, complete = calls[variant]
            pre_ms = measure(torch, pre)
            complete_ms = measure(torch, complete)
            rows.append(
                {
                    "variant": variant,
                    "repetition": repetition,
                    "preprocessing_cuda_ms": pre_ms,
                    "complete_attention_cuda_ms": complete_ms,
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                }
            )
    return {
        "shape_bthd": list(values[0].shape),
        "samples": rows,
        "primary_speed_comparison": ["public_upstream", "fixed_all_candidate"],
        "scope": (
            "Warm CUDA event intervals only; excludes input generation/copies, "
            "compilation and metrics. Not full-video timing."
        ),
        "control_costs": (
            "Instrumented controls retain extra means/correction/dictionaries. "
            "fixed_means_control computes and discards fixed correction before "
            "upstream matmul; it is not an optimized means-only timing."
        ),
    }


def run(size, seed):
    import torch

    from cozy_runtime.internal import accel, attention, attention_upstream
    from cozy_runtime.internal.encoding import DeviceFacts

    versions = {
        name: importlib.metadata.version(name)
        for name in ("cozy-runtime", "tensorfs", "torch", "triton")
    }
    cohort = json.loads(Path(__file__).with_name("cohort.json").read_text())
    if not cohort["runtime_version"] or versions["cozy-runtime"] != cohort["runtime_version"]:
        raise RuntimeError("Diagnostic cohort is unfrozen or Runtime changed")
    if any(versions[name] != expected for name, expected in cohort["dependencies"].items()):
        raise RuntimeError("Diagnostic dependency cohort changed")
    root = Path(inspect.getfile(attention_upstream)).parents[1]
    hashes = {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in cohort["source_sha256"]
    }
    if hashes != cohort["source_sha256"]:
        raise RuntimeError("Called Runtime source/native artifacts changed")
    if torch.cuda.get_device_capability() != (12, 0):
        raise RuntimeError("This first experiment admits the assigned RTX5090/SM120 only")
    facts = DeviceFacts(
        "cuda",
        torch.cuda.get_device_name(),
        120,
        accel.driver_version(),
        "Sage3 fixed-preprocessing diagnostic",
        index=torch.cuda.current_device(),
    )
    candidate = attention.BY_NAME["sageattention3-fp4-global-lowmem"]
    artifact = attention._compiled(candidate, facts, wait=True)
    attention._import(candidate)
    api = importlib.import_module("sageattn3.api")
    proof = stages.verify_api(api)
    started = time.monotonic()
    results, times = [], []
    with torch.inference_mode():
        for label in TOKENS if size == "all" else (size,):
            results.append(
                {"size": label, **case(torch, api, candidate.local, TOKENS[label], seed)}
            )
        for label in TOKENS if size == "all" else (size,):
            times.append(
                {"size": label, **timings(torch, api, candidate.local, TOKENS[label], seed)}
            )
    return {
        "software": versions,
        "source_sha256": hashes,
        "upstream_api": proof,
        "artifact": artifact,
        "device": torch.cuda.get_device_name(),
        "capability": [12, 0],
        "driver": accel.driver_version(),
        "cuda_build": torch.version.cuda,
        "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cases": results,
        "timings": times,
        "elapsed_seconds": time.monotonic() - started,
        "limits": [
            "One GPU with fixed operands; not communication or whole-video identity.",
            "Fixed trees may differ from upstream; BF16 rounding remains. Quality review required.",
            "Upstream quantizers/attention stay unchanged; no global patch or precision changes.",
            "28GiB headroom is estimated; actual peak retained. No downsizing.",
            "Input, finite, repeat and partition checks must pass before full-video experiments.",
        ],
    }
