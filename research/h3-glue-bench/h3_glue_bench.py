"""h3a-015 yardstick: eager vs fused (AOT Triton) vs regional `torch.compile`, same pod.

A model-less GPU job (random weights at MiniMax-H3's real widths, never a checkpoint), so
`cozy run local/h3-glue-bench/bench --rental-only` measures the three execution plans of
one H3 transformer block at the served sequence length on whatever card the rental holds.
Not shipped: the fused plan is the product; compile is measured here only as the yardstick
decision #702 allows (regional, post-fill, never at construction).

Numbers per variant: forward ms over `layers` blocks (median of `iterations` after two warm
calls), per-block ms and the 50-block step equivalent; fused-vs-eager and compiled-vs-eager
output distance; cold costs (artifact load, a from-scratch JIT build of the same kernels,
torch.compile's first call); optionally the CUDA kernel self-time by category.
"""

from __future__ import annotations

import argparse
import copy
import json
import statistics
import tempfile
import time
from pathlib import Path
from typing import Any

import msgspec
import torch
from diffusers import MiniMaxH3Transformer3DModel
from torch.profiler import ProfilerActivity, profile

from cozy_runtime.author import App, Context
from cozy_runtime.internal import fusion, fusion_build, fusion_install
from cozy_runtime.internal.encoding.leaves import RowwiseNativeLeaf

HIDDEN, FFN, HEADS, HEAD_DIM, ROTARY = 5376, 14336, 56, 128, 96
TEXT_ROWS, AUDIO_ROWS = 950, 1150
BLOCKS_PER_STEP = 50


class BenchRequest(msgspec.Struct, forbid_unknown_fields=True):
    rows: int = 104_916
    layers: int = 2
    iterations: int = 5
    fp8: bool = False
    compile: bool = True
    profile: bool = False


class Variant(msgspec.Struct):
    name: str
    forward_ms: float
    block_ms: float
    step_s: float


class Distance(msgspec.Struct):
    pair: str
    max_abs: float
    max_abs_over_range: float


class Cold(msgspec.Struct):
    artifact_load_ms: float
    jit_build_ms: float
    torch_compile_ms: float
    artifact_source: str


class KernelShare(msgspec.Struct):
    variant: str
    category: str
    ms: float


class BenchResult(msgspec.Struct):
    device: str
    arch: str
    rows: int
    layers: int
    fp8: bool
    variants: list[Variant]
    distance: list[Distance]
    cold: Cold
    kernels: list[KernelShare]


class _Holder:
    def __init__(self, transformer: Any) -> None:
        self.pipe = type("Pipe", (), {"components": {"dit": transformer}})()


def _transformer(layers: int, device: Any) -> Any:
    torch.manual_seed(0)
    transformer = MiniMaxH3Transformer3DModel(num_layers=layers, num_refiner_layers=1)
    with torch.no_grad():
        for parameter in transformer.parameters():
            parameter.normal_(std=0.02)
    for name, child in transformer.named_children():
        child.to(torch.float32 if name in transformer._keep_in_fp32_modules else torch.bfloat16)
    return transformer.to(device).eval()


def _inputs(device: Any, rows: int) -> dict[str, Any]:
    torch.manual_seed(1)
    video = rows - TEXT_ROWS - AUDIO_ROWS
    return {
        "hidden_states": torch.randn(1, video, 96, device=device),
        "audio_hidden_states": torch.randn(1, AUDIO_ROWS, 32, device=device),
        "encoder_hidden_states": torch.randn(1, TEXT_ROWS, 5120, device=device),
        "timestep": torch.tensor([0.7, 0.0], device=device),
        "timestep_indices": torch.arange(rows, device=device) % 2,
        "token_tags": torch.cat(
            [
                torch.ones(TEXT_ROWS, dtype=torch.int64),
                torch.zeros(video, dtype=torch.int64),
                torch.full((AUDIO_ROWS,), 2, dtype=torch.int64),
            ]
        ).to(device),
        "position_ids": torch.randint(0, 128, (rows, 3), device=device),
        "video_indices": torch.arange(TEXT_ROWS, TEXT_ROWS + video, device=device),
        "audio_indices": torch.arange(TEXT_ROWS + video, rows, device=device),
        "text_indices": torch.arange(TEXT_ROWS, device=device),
        "return_dict": False,
    }


def _encode_leaves(transformer: Any) -> None:
    provider = RowwiseNativeLeaf(encoding="fp8-rowwise/1")
    for block in transformer.transformer_blocks:
        sites = [
            (block.attn, "to_q"),
            (block.attn, "to_k"),
            (block.attn, "to_v"),
            (block.ff.net[0], "proj"),
            (block.ff.net, 2),
        ]
        for owner, attribute in sites:
            linear = owner[attribute] if isinstance(attribute, int) else getattr(owner, attribute)
            weight = linear.weight.detach().float()
            scale = (weight.abs().amax(dim=1, keepdim=True) / 448.0).clamp(min=1e-12)
            payload = (weight / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
            leaf = provider.leaf(
                torch, {"data": payload, "scale": scale.reshape(-1)}, linear, torch.bfloat16
            )
            if isinstance(attribute, int):
                owner[attribute] = leaf
            else:
                setattr(owner, attribute, leaf)


def _time(transformer: Any, inputs: dict[str, Any], iterations: int) -> float:
    for _ in range(2):
        transformer(**inputs)
    torch.cuda.synchronize()
    samples = []
    for _ in range(iterations):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        transformer(**inputs)
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end))
    return float(statistics.median(samples))


def _distance(pair: str, reference: Any, actual: Any) -> Distance:
    gap = (reference[0].float() - actual[0].float()).abs().max().item()
    return Distance(pair, gap, gap / max(reference[0].float().abs().max().item(), 1e-30))


def _category(name: str) -> str:
    lowered = name.lower()
    if any(tag in lowered for tag in ("norm_mod", "gate_add", "swiglu", "rope_norm")):
        return "fused_glue"
    if any(tag in lowered for tag in ("flash", "fmha", "cudnn", "attention", "sdpa")):
        return "attention"
    if any(tag in lowered for tag in ("gemm", "cutlass", "nvjet", "scaled_mm", "matmul")):
        return "gemm"
    if lowered.startswith("triton_"):
        return "inductor"
    if "rowwise" in lowered or "quantize" in lowered:
        return "leaf_quant"
    return "eager_glue"


def _profile(variant: str, transformer: Any, inputs: dict[str, Any]) -> list[KernelShare]:
    with profile(activities=[ProfilerActivity.CUDA]) as trace:
        transformer(**inputs)
        torch.cuda.synchronize()
    shares: dict[str, float] = {}
    for row in trace.key_averages():
        if row.self_device_time_total:
            category = _category(row.key)
            shares[category] = shares.get(category, 0.0) + row.self_device_time_total / 1000
    return [KernelShare(variant, category, round(ms, 2)) for category, ms in sorted(shares.items())]


def _record(variants: list[Variant], name: str, forward_ms: float, layers: int) -> None:
    block_ms = forward_ms / layers
    variants.append(
        Variant(
            name,
            round(forward_ms, 2),
            round(block_ms, 2),
            round(block_ms * BLOCKS_PER_STEP / 1000, 3),
        )
    )


def run(request: BenchRequest, device: Any) -> BenchResult:
    device = torch.device(str(device))
    major, minor = torch.cuda.get_device_capability(device)
    arch = major * 10 + minor
    # The runtime's own artifact when the image carries one; otherwise the same build the
    # image stage runs, here and now, so the yardstick measures on a pod whose image predates
    # the cubin bake. Either way the JIT cost is measured from scratch.
    scratch = tempfile.TemporaryDirectory(prefix="h3-glue-bench-jit-")
    started = time.perf_counter()
    fusion_build.build(Path(scratch.name), (arch,))
    jit_build_ms = (time.perf_counter() - started) * 1000
    started = time.perf_counter()
    try:
        fusion.load(device)
        artifact_source = str(fusion.artifact_dir())
    except fusion.FusionUnavailable as absent:
        if absent.code != "fusion_kernels_absent":
            raise
        fusion.load(device, Path(scratch.name))
        artifact_source = f"self-built ({absent})"
    artifact_load_ms = (time.perf_counter() - started) * 1000

    variants: list[Variant] = []
    distance: list[Distance] = []
    kernels: list[KernelShare] = []
    torch_compile_ms = 0.0
    with torch.no_grad():
        eager = _transformer(request.layers, device)
        if request.fp8:
            _encode_leaves(eager)
        inputs = _inputs(device, request.rows)
        reference = eager(**inputs)

        arms: list[tuple[str, Any]] = [("eager", eager)]
        fused = copy.deepcopy(eager)
        applied = fusion_install.apply_fusion_plan(_Holder(fused), torch=torch, device=device)
        assert applied.applied, applied.reason
        distance.append(_distance("fused_vs_eager", reference, fused(**inputs)))
        arms.append(("fused", fused))
        for name, module in arms:
            _record(variants, name, _time(module, inputs, request.iterations), request.layers)
            if request.profile:
                kernels += _profile(name, module, inputs)
        del fused, arms
        torch.cuda.empty_cache()
        if request.compile and not request.fp8:
            # The eager copy is compiled IN PLACE once its own numbers are taken: two block
            # copies plus inductor's workspace is what an 8 GB developer card cannot hold.
            eager.compile_repeated_blocks(fullgraph=True, dynamic=False)
            started = time.perf_counter()
            eager(**inputs)
            torch.cuda.synchronize()
            torch_compile_ms = (time.perf_counter() - started) * 1000
            distance.append(_distance("compiled_vs_eager", reference, eager(**inputs)))
            _record(
                variants,
                "compiled",
                _time(eager, inputs, request.iterations),
                request.layers,
            )
            if request.profile:
                kernels += _profile("compiled", eager, inputs)
    return BenchResult(
        device=torch.cuda.get_device_name(device),
        arch=f"sm{arch}",
        rows=request.rows,
        layers=request.layers,
        fp8=request.fp8,
        variants=variants,
        distance=distance,
        cold=Cold(
            round(artifact_load_ms, 1),
            round(jit_build_ms, 1),
            round(torch_compile_ms, 1),
            artifact_source,
        ),
        kernels=kernels,
    )


app = App()


@app.job
def bench(payload: BenchRequest, ctx: Context) -> BenchResult:
    """A GPU job with no model: the card is the subject, random weights are the fixture."""
    return run(payload, ctx.device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--rows", type=int, default=4096)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--fp8", action="store_true")
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    request = BenchRequest(
        rows=args.rows,
        layers=args.layers,
        iterations=args.iterations,
        fp8=args.fp8,
        compile=not args.no_compile,
        profile=args.profile,
    )
    print(json.dumps(msgspec.to_builtins(run(request, "cuda")), indent=1))
