"""Compare actual unequal-length QKV/O exchanges to each rank's full-head reference."""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import inspect
import os
import time
from pathlib import Path

from . import fixed_means
from .metrics import compared, operand_layout

TOKENS, HEADS, DIM, DEGREE = 109225, 56, 128, 2
API_SHA = "9aec83f8d7165e3b563ba4fc91f6ca5e09b07ef3445b6db5d1105912e35dfb90"


def token_bounds(rank, tokens=TOKENS, degree=DEGREE):
    width, remainder = divmod(tokens, degree)
    start = rank * width + min(rank, remainder)
    return start, start + width + (rank < remainder)


def nhd_metric(torch, actual, reference):
    return compared(torch, actual[0].transpose(0, 1), reference[0].transpose(0, 1))


def exact(metric):
    return metric["bytes_equal"] and metric["nonfinite_elements"] == 0


def validate_rank_identities(rows):
    if (
        len(rows) != DEGREE
        or sorted(row["rank"] for row in rows) != list(range(DEGREE))
        or len({row["uuid"] for row in rows}) != DEGREE
        or len({row["pid"] for row in rows}) != DEGREE
        or any(
            row["world_size"] != DEGREE
            or row["group_global_ranks"] != list(range(DEGREE))
            or row["sm"] != [12, 0]
            or row["runtime"] != "0.23.5"
            or row["tensorfs"] != "0.6.1"
            for row in rows
        )
    ):
        raise RuntimeError("Probe requires two distinct granted SM120 ranks on its public SDK")


def capture(torch, api, values, candidate):
    tokens = values[0].shape[1]
    if candidate:
        prepared = fixed_means.fixed_preprocess(*values)
    else:
        q, k, v = (x.transpose(1, 2) for x in values)
        prepared = dict(
            zip(
                ("q", "k", "v", "delta"),
                api.preprocess_qkv(
                    q, k.clone(memory_format=torch.contiguous_format), v, per_block_mean=False
                ),
                strict=True,
            )
        )
    batch, heads, padded, dim = prepared["q"].shape
    quant = {}
    for name, fn in (
        ("q", api.scale_and_quant_fp4),
        ("k", api.scale_and_quant_fp4_permute),
        ("v", api.scale_and_quant_fp4_transpose),
    ):
        payload, scale = fn(prepared[name])
        ps = (batch, heads, dim, padded // 2) if name == "v" else (batch, heads, padded, dim // 2)
        ss = (batch, heads, dim, padded // 16) if name == "v" else (batch, heads, padded, dim // 16)
        if tuple(payload.shape) != ps or tuple(scale.shape) != ss:
            raise RuntimeError("Quantizer head/row ownership changed")
        if payload.dtype != torch.uint8 or scale.dtype != torch.float8_e4m3fn:
            raise RuntimeError("Quantizer dtype changed")
        quant[name + "_payload"], quant[name + "_scale"] = payload, scale
    pairs = [(quant[name + "_payload"], quant[name + "_scale"]) for name in ("q", "k", "v")]
    output = api.blockscaled_fp4_attn(*pairs, prepared["delta"], tokens, False, False, True)[0]
    return {**prepared, **quant, "output": output[:, :, :tokens, :]}


def time_call(torch, dist, handle, call):
    dist.barrier(group=handle)
    torch.cuda.synchronize()
    start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    wall = time.perf_counter()
    start.record()
    value = call()
    stop.record()
    stop.synchronize()
    elapsed = time.perf_counter() - wall
    del value
    return {"cuda_ms": float(start.elapsed_time(stop)), "wall_ms": elapsed * 1000}


def sensitivity_control(torch, api, handle, rank, seed):
    """Two real two-rank exchanges with fourteen local heads; not four-rank transport.

    Reproduce run5228's input generation (three contiguous draws from one generator),
    while preserving all56heads/full109225tokens. No timing is reported for this control.
    """
    from cozy_runtime.internal import attention_ulysses

    torch.cuda.reset_peak_memory_stats()
    generator = torch.Generator(device="cuda").manual_seed(seed)
    full = tuple(
        torch.randn(1, TOKENS, HEADS, DIM, dtype=torch.bfloat16, device="cuda", generator=generator)
        for _ in range(3)
    )
    start, stop = token_bounds(rank)
    variants = []
    for name, candidate in (("public", False), ("fixed_means", True)):
        reference = capture(torch, api, full, candidate)
        rounds = []
        outputs = []
        for offset in (0, 28):
            hs, he = offset + rank * 14, offset + (rank + 1) * 14
            local = tuple(x[:, start:stop, offset : offset + 28] for x in full)
            observed = {}

            def compute(
                q,
                k,
                v,
                *,
                hs=hs,
                he=he,
                candidate=candidate,
                reference=reference,
                observed=observed,
            ):
                observed["received_layouts"] = [operand_layout(x) for x in (q, k, v)]
                observed["qkv"] = {
                    key: nhd_metric(torch, value, original[:, :, hs:he])
                    for key, value, original in zip(("q", "k", "v"), (q, k, v), full, strict=True)
                }
                actual = capture(torch, api, (q, k, v), candidate)
                observed["head_stages_vs_full"] = {
                    key: compared(torch, value[0], reference[key][0, hs:he])
                    for key, value in actual.items()
                }
                return actual["output"].transpose(1, 2).contiguous()

            output = attention_ulysses.exchange(*local, handle, compute)
            outputs.append(output)
            rounds.append(
                {"input_head_range": [offset, offset + 28], "rank_head_range": [hs, he], **observed}
            )
            del compute, output, local
        reconstructed = torch.cat(outputs, dim=2)
        output_metric = nhd_metric(
            torch, reconstructed, reference["output"].transpose(1, 2)[:, start:stop]
        )
        variants.append(
            {
                "variant": name,
                "rounds": rounds,
                "output_vs_full_token_shard": output_metric,
                "partition_exact": exact(output_metric)
                and all(
                    exact(metric)
                    for row in rounds
                    for metric in [*row["qkv"].values(), *row["head_stages_vs_full"].values()]
                ),
            }
        )
        del reconstructed, outputs, reference
    generator = torch.Generator(device="cuda").manual_seed(seed)
    unchanged = {}
    for key, value in zip(("q", "k", "v"), full, strict=True):
        regenerated = torch.randn(
            value.shape, dtype=value.dtype, device=value.device, generator=generator
        )
        unchanged[key] = nhd_metric(torch, value, regenerated)
        del regenerated
    transport_and_inputs_exact = all(exact(x) for x in unchanged.values()) and all(
        exact(metric)
        for variant in variants
        for row in variant["rounds"]
        for metric in row["qkv"].values()
    )
    public_arithmetic_differs = not exact(variants[0]["output_vs_full_token_shard"]) or any(
        not exact(metric)
        for row in variants[0]["rounds"]
        for metric in row["head_stages_vs_full"].values()
    )
    return {
        "scope": "two real ranks, two28head batches, fourteen local heads; not four-rank coverage",
        "fixture": "run5228 contiguous QKV draws; public mismatch is observed, not assumed",
        "seed": seed,
        "variants": variants,
        "input_unchanged": unchanged,
        "candidate_partition_exact": variants[1]["partition_exact"]
        and all(exact(x) for x in unchanged.values()),
        "transport_and_inputs_exact": transport_and_inputs_exact,
        "public_difference_observed": transport_and_inputs_exact and public_arithmetic_differs,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
    }


def run(seed, handle):
    import torch
    import torch.distributed as dist

    from cozy_runtime.internal import accel, attention, attention_ulysses
    from cozy_runtime.internal.encoding import DeviceFacts

    rank, degree = dist.get_rank(handle), dist.get_world_size(handle)
    identity_fields = {
        "rank": rank,
        "world_size": degree,
        "pid": os.getpid(),
        "uuid": str(torch.cuda.get_device_properties(torch.cuda.current_device()).uuid),
        "sm": list(torch.cuda.get_device_capability()),
        "runtime": importlib.metadata.version("cozy-runtime"),
        "tensorfs": importlib.metadata.version("tensorfs"),
        "group_global_ranks": dist.get_process_group_ranks(handle),
    }
    identities = [None] * degree
    dist.all_gather_object(identities, identity_fields, group=handle)
    validate_rank_identities(identities)
    device = DeviceFacts(
        "cuda",
        torch.cuda.get_device_name(),
        120,
        accel.driver_version(),
        "actual_ulysses_probe",
        index=torch.cuda.current_device(),
    )
    selection = attention.pinned("sageattention3-fp4-global-lowmem", device)[0]
    baseline = selection.candidate.local.forward
    api = importlib.import_module("sageattn3.api")
    if hashlib.sha256(Path(api.__file__).read_bytes()).hexdigest() != API_SHA:
        raise RuntimeError("Reviewed Sage3 preprocessing source changed")
    free, total = torch.cuda.mem_get_info()
    if free + torch.cuda.memory_reserved() - torch.cuda.memory_allocated() < 28 << 30:
        raise RuntimeError("Full-shape comparison needs28GiB headroom; no shape reduction")
    torch.cuda.reset_peak_memory_stats()
    begin = time.perf_counter()
    with torch.inference_mode():
        packed = torch.randn(
            1,
            TOKENS,
            HEADS * 3,
            DIM,
            dtype=torch.bfloat16,
            device="cuda",
            generator=torch.Generator(device="cuda").manual_seed(seed),
        )
        full = packed.chunk(3, dim=2)
        start, stop = token_bounds(rank)
        local = tuple(x[:, start:stop] for x in full)
        hs, he = rank * (HEADS // degree), (rank + 1) * (HEADS // degree)
        transport = {}

        def identity(q, k, v, *, full=full):
            transport["received_layouts"] = [operand_layout(x) for x in (q, k, v)]
            transport["qkv"] = {
                name: nhd_metric(torch, x, ref[:, :, hs:he])
                for name, x, ref in zip(("q", "k", "v"), (q, k, v), full, strict=True)
            }
            return q.clone()

        identity_result = attention_ulysses.exchange(*local, handle, identity)
        transport["inverse_identity"] = nhd_metric(torch, identity_result, local[0])
        del identity_result
        transport["passed"] = all(exact(x) for x in transport["qkv"].values()) and exact(
            transport["inverse_identity"]
        )
        agreed = torch.tensor([int(transport["passed"])], device="cuda")
        dist.all_reduce(agreed, op=dist.ReduceOp.MIN, group=handle)
        if not agreed.item():
            refused = [None] * degree
            dist.all_gather_object(
                refused, {**identity_fields, "transport": transport}, group=handle
            )
            return {
                "seed": seed,
                "shape": [1, TOKENS, HEADS, DIM],
                "candidate_partition_exact": False,
                "attention_skipped": "QKV redistribution/inverse is not byte-exact",
                "ranks": refused,
            }
        variants = []
        previous_full_output = None
        for name, candidate in (("public", False), ("fixed_means", True)):
            reference = capture(torch, api, full, candidate)
            full_output = reference["output"].transpose(1, 2)
            direct = fixed_means.forward(*full) if candidate else baseline(*full, scale=DIM**-0.5)
            api_replay = nhd_metric(torch, full_output, direct)
            del direct
            stages = {}

            def compute(q, k, v, *, candidate=candidate, stages=stages, reference=reference):
                actual = capture(torch, api, (q, k, v), candidate)
                for key in actual:
                    stages[key] = compared(torch, actual[key][0], reference[key][0, hs:he])
                return actual["output"].transpose(1, 2).contiguous()

            output = attention_ulysses.exchange(*local, handle, compute)
            reconstructed = nhd_metric(torch, output, full_output[:, start:stop])
            repeat = attention_ulysses.exchange(
                *local,
                handle,
                (lambda q, k, v: fixed_means.forward(q, k, v))
                if candidate
                else (lambda q, k, v: baseline(q, k, v, scale=DIM**-0.5)),
            )
            repeat_metric = nhd_metric(torch, repeat, output)
            del output, repeat
            row = {
                "variant": name,
                "replay_equals_actual_callable": api_replay,
                "head_stages_vs_full": stages,
                "output_vs_full_token_shard": reconstructed,
                "repeated_exchange_output": repeat_metric,
                "partition_exact": exact(api_replay)
                and all(exact(x) for x in stages.values())
                and exact(reconstructed)
                and exact(repeat_metric),
            }
            if candidate:
                row["full_candidate_vs_public"] = nhd_metric(
                    torch, full_output, previous_full_output
                )
            else:
                previous_full_output = full_output
            variants.append(row)
            del compute, reference, full_output
        del previous_full_output
        callbacks = {
            "public": lambda q, k, v: baseline(q, k, v, scale=DIM**-0.5),
            "fixed_means": fixed_means.forward,
        }
        times = []
        for callback in callbacks.values():
            warm = attention_ulysses.exchange(*local, handle, callback)
            del warm
        for repetition in range(3):
            for name in list(callbacks) if repetition % 2 == 0 else list(reversed(callbacks)):
                callback = callbacks[name]
                times.append(
                    {
                        "variant": name,
                        "repetition": repetition,
                        "scope": "actual_ulysses_exchange_and_attention",
                        **time_call(
                            torch,
                            dist,
                            handle,
                            lambda callback=callback, local=local: attention_ulysses.exchange(
                                *local, handle, callback
                            ),
                        ),
                    }
                )
                times.append(
                    {
                        "variant": name,
                        "repetition": repetition,
                        "scope": "full_head_reference_no_collective",
                        **time_call(
                            torch,
                            dist,
                            handle,
                            lambda callback=callback, full=full: callback(*full),
                        ),
                    }
                )
        regenerated = torch.randn(
            packed.shape,
            dtype=torch.bfloat16,
            device="cuda",
            generator=torch.Generator(device="cuda").manual_seed(seed),
        )
        input_unchanged = compared(torch, packed[0], regenerated[0])
        del regenerated
        input_layouts = [operand_layout(x) for x in local]
        primary_peak = torch.cuda.max_memory_allocated()
        del packed, full, local, identity
        sensitivity = sensitivity_control(torch, api, handle, rank, seed)
        report = {
            **identity_fields,
            "rank": rank,
            "world_size": degree,
            "pid": os.getpid(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "device": torch.cuda.get_device_name(),
            "uuid": str(torch.cuda.get_device_properties(torch.cuda.current_device()).uuid),
            "backend": str(dist.get_backend(handle)),
            "group_global_ranks": dist.get_process_group_ranks(handle),
            "token_range": [start, stop],
            "head_range": [hs, he],
            "input_layouts": input_layouts,
            "packed_max_element_index": TOKENS * HEADS * 3 * DIM - 1,
            "transport": transport,
            "input_unchanged": input_unchanged,
            "variants": variants,
            "timings": times,
            "sensitivity_control": sensitivity,
            "primary_peak_allocated_bytes": primary_peak,
            "peak_allocated_bytes": max(primary_peak, sensitivity["peak_allocated_bytes"]),
            "total_device_bytes": total,
            "elapsed_seconds": time.perf_counter() - begin,
            "source_sha256": {
                m.__name__: hashlib.sha256(Path(inspect.getfile(m)).read_bytes()).hexdigest()
                for m in (attention_ulysses, api, fixed_means)
            },
            "runtime": importlib.metadata.version("cozy-runtime"),
            "tensorfs": importlib.metadata.version("tensorfs"),
            "driver": accel.driver_version(),
        }
        reports = [None] * degree
        dist.all_gather_object(reports, report, group=handle)
        return {
            "seed": seed,
            "shape": [1, TOKENS, HEADS, DIM],
            "ranks": reports,
            "rank_identity_validated": True,
            "candidate_partition_exact": all(
                r["variants"][1]["partition_exact"] and exact(r["input_unchanged"]) for r in reports
            ),
            "candidate_sensitivity_control_exact": all(
                r["sensitivity_control"]["candidate_partition_exact"] for r in reports
            ),
            "limits": [
                "Fixed activations and real two-GPU Ulysses only; no H3 sampling or video claim.",
                "Three warm timing samples check regression; no default or generation-speed claim.",
            ],
        }
