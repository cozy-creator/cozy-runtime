"""Compare the exact selected-tile kernels with identical prepacked synthetic inputs."""

from __future__ import annotations

import argparse
import ctypes as C
import hashlib
import json
import os
import random
import statistics
import subprocess
import time
from pathlib import Path


def check_device_free() -> None:
    result = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader",
        ],
        text=True,
    )
    others = [row for row in result.splitlines() if not row.startswith(f"{os.getpid()},")]
    if others:
        raise RuntimeError(f"Other GPU compute processes: {others}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=13)
    parser.add_argument("--variants", nargs="+", default=["baseline", "stage3"])
    args = parser.parse_args()
    check_device_free()
    import torch

    torch.cuda.init()
    root = Path(__file__).resolve().parent
    kernels = {}
    resources = {}
    for name in args.variants:
        path = root / "build" / f"{name}.so"
        lib = C.CDLL(str(path))
        launch = lib.launch_sol_exact
        launch.argtypes = (
            [C.c_void_p] * 14
            + [C.c_int, C.c_void_p]
            + [C.c_int] * 6
            + [C.c_float, C.c_int, C.c_void_p]
        )
        launch.restype = None
        r, s, b = C.c_int(), C.c_int(), C.c_int()
        code = lib.exact_resources(C.byref(r), C.byref(s), C.byref(b))
        if code:
            raise RuntimeError(f"{name}: cuda resource query returned {code}")
        resources[name] = dict(
            registers=r.value,
            shared_bytes=s.value,
            ctas_per_sm=b.value,
            binary_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        )
        kernels[name] = launch

    def inputs(t: int, h: int, selected: int, seed: int, ntok: int = 0, mixed: bool = False):
        torch.manual_seed(seed)
        tp = (t + 63) // 64 * 64
        nq = ntb = tp // 64

        def normal(*shape):
            return torch.randn(shape, device="cuda")

        def ints(*shape):
            return torch.randint(-127, 128, shape, device="cuda", dtype=torch.int8)

        qi, ki, vt = ints(1, t, h, 128), ints(h, tp, 128), ints(h, 128, tp)
        qs = torch.full((1, t, h), 0.01, device="cuda")
        ksb = torch.stack((torch.full((h, tp), 0.01, device="cuda"), normal(h, tp) * 0.1), dim=-1)
        vsc = torch.full((h, 128), 0.01, device="cuda")
        if mixed:
            qs *= 2 ** (normal(1, t, h) * 2)
            ksb[..., 0] *= 2 ** (normal(h, tp) * 2)
            vsc *= 2 ** (normal(h, 128) * 2)
        # Exactly the same fixed, sorted, unique route on both arms. No routing work is timed.
        route = (
            torch.rand((h, nq, ntb), device="cuda")
            .argsort(dim=-1)[..., :selected]
            .sort(dim=-1)
            .values
        )
        indices = torch.zeros((h, nq, ntb), device="cuda", dtype=torch.uint16)
        indices[..., :selected] = route.to(torch.uint16)
        counts = torch.full((h, nq), selected, device="cuda", dtype=torch.int32)
        if mixed and selected:
            counts = torch.randint(0, selected + 1, (h, nq), device="cuda", dtype=torch.int32)
        partial = normal(h, nq, 128).to(torch.bfloat16)
        maxes = normal(h, nq)
        masses = torch.rand((h, nq), device="cuda") + 0.1
        vrow = ints(h, tp, 128) if ntok else torch.empty(0, device="cuda", dtype=torch.int8)
        tokens = torch.randint(0, t, (h, (nq + 1) // 2, ntok), device="cuda", dtype=torch.int64).to(
            torch.uint32
        )
        token_counts = torch.full((h, (nq + 1) // 2), ntok, device="cuda", dtype=torch.int32)
        tensors = [
            qi,
            qs,
            ki,
            ksb,
            vt,
            vsc,
            indices,
            counts,
            partial,
            maxes,
            masses,
            vrow,
            tokens,
            token_counts,
        ]
        return tensors, (ntok, 1, t, tp, h, nq, ntb)

    def call(name, data, out):
        tensors, (ntok, b, t, tp, h, nq, ntb) = data
        kernels[name](
            *[x.data_ptr() for x in tensors],
            ntok,
            out.data_ptr(),
            b,
            t,
            tp,
            h,
            nq,
            ntb,
            1.4426950408889634 / 128**0.5,
            0 if out.dtype == torch.bfloat16 else 1,
            torch.cuda.current_stream().cuda_stream,
        )

    result = dict(
        timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        source_revision="f8b7e97a1418d2e529bd5405fbf86312be198ae6",
        torch=torch.__version__,
        gpu=str(torch.cuda.get_device_properties(0)),
        driver=subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True
        ).strip(),
        resources=resources,
        correctness=[],
        timings=[],
        scope=(
            "Synthetic prepacked selected-tile kernel only; "
            "no H3, producer, route, or end-to-end qualification."
        ),
        method=(
            "CUDA graph replay of identical fixed routes, shuffled arm order, "
            "two input sets, 3 replays/sample; warm cache, no clock lock."
        ),
    )
    # Tails, zero/few/odd/all blocks, token augmentation and both output dtypes.
    for t, h, selected, ntok in [
        (65, 2, 0, 0),
        (65, 2, 1, 0),
        (129, 3, 3, 0),
        (512, 2, 8, 129),
        (4096, 4, 64, 0),
    ]:
        for dtype in (torch.bfloat16, torch.float16):
            for seed in (298, 299):
                check_device_free()
                data = inputs(t, h, selected, seed, ntok, mixed=True)
                before = [x.clone() for x in data[0]]
                outputs = {}
                for name in [*args.variants, "baseline"]:
                    out = torch.full((1, t, h, 128), float("nan"), device="cuda", dtype=dtype)
                    call(name, data, out)
                    torch.cuda.synchronize()
                    if not torch.isfinite(out).all():
                        raise AssertionError(f"nonfinite output: {name}, {t}, {dtype}")
                    if outputs and not torch.equal(
                        outputs["baseline"].view(torch.uint8), out.view(torch.uint8)
                    ):
                        raise AssertionError(f"output bytes differ: {name}, {t}, {dtype}")
                    outputs[name] = out
                if not all(
                    torch.equal(a.view(torch.uint8), b.view(torch.uint8))
                    for a, b in zip(before, data[0], strict=True)
                ):
                    raise AssertionError("an input changed")
                result["correctness"].append(
                    dict(
                        tokens=t,
                        heads=h,
                        selected=selected,
                        token_aug=ntok,
                        dtype=str(dtype),
                        seed=seed,
                        equal_bytes=True,
                        inputs_unchanged=True,
                    )
                )
    print("correctness:", len(result["correctness"]), "cases passed", flush=True)
    del data, before, outputs, out
    torch.cuda.empty_cache()
    rng = random.Random(298)
    for t, h, selected in [
        (4096, 8, 8),
        (4096, 8, 32),
        (4096, 8, 64),
        (16384, 8, 32),
        (16384, 8, 128),
        (65536, 2, 64),
        (65536, 2, 256),
    ]:
        check_device_free()
        datasets = [inputs(t, h, selected, seed) for seed in (2980, 2981)]
        graphs, output_sets = {}, {}
        for name in args.variants:
            graphs[name], output_sets[name] = [], []
            for data in datasets:
                out = torch.empty((1, t, h, 128), device="cuda", dtype=torch.bfloat16)
                for _ in range(3):
                    call(name, data, out)
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    call(name, data, out)
                graphs[name].append(graph)
                output_sets[name].append(out)
        for name in args.variants:
            assert all(
                torch.equal(a.view(torch.uint8), b.view(torch.uint8))
                for a, b in zip(output_sets["baseline"], output_sets[name], strict=True)
            )
        samples = {name: [] for name in args.variants}
        orders = []
        for i in range(args.rounds):
            check_device_free()
            order = args.variants.copy()
            rng.shuffle(order)
            orders.append(order)
            for name in order:
                start, end = (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                start.record()
                for j in range(3):
                    graphs[name][(i + j) % 2].replay()
                end.record()
                end.synchronize()
                samples[name].append(start.elapsed_time(end) / 3)
        baseline = samples["baseline"]
        row = dict(
            tokens=t,
            heads=h,
            head_dim=128,
            selected_blocks=selected,
            input_set_count=2,
            input_bytes=sum(x.numel() * x.element_size() for d in datasets for x in d[0]),
            samples_ms=samples,
            arm_orders=orders,
            median_ms={name: statistics.median(v) for name, v in samples.items()},
            paired_ratio={
                name: statistics.median(a / b for a, b in zip(samples[name], baseline, strict=True))
                for name in args.variants
            },
        )
        result["timings"].append(row)
        print(
            json.dumps(
                {
                    k: row[k]
                    for k in ("tokens", "heads", "selected_blocks", "median_ms", "paired_ratio")
                }
            ),
            flush=True,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        del graphs, output_sets, datasets, data, out, graph
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
