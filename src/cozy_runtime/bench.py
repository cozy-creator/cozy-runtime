"""Unstable: for the release gate only, outside the stable API; it may change or go in any release.

The Runtime's exercisers of its own code: built-in calls of the Runtime's own App (`derive.operations`), which every
environment holds, so the release gate measures and checks the Runtime from inside it (a package can't import its
internals). A gate script calls them with `call(bench.attention_speed, ...)`; each returns one JSON document.
`internal/bench/` says what each does: the attention bench (xs-171), the GEMM bench (xs-177), the kernels suite and the
varena and execution suites' Runtime cells (xs-174)."""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Callable

import msgspec

from cozy_runtime.author import Context, Outputs, Telemetry
from cozy_runtime.internal.bench.qwen_overlay import Lightning, QwenDiT
from cozy_runtime.media import File


class Measured(msgspec.Struct, frozen=True):
    """One call's findings, as JSON (the release gate's suites read them)."""

    detail: str


def _run(work: Callable[[], object]) -> Measured:
    from cozy_runtime.internal.bench import attention

    found = attention.off_request(work)
    return Measured(json.dumps(found, sort_keys=True))


def attention_inventory(ctx: Context) -> Measured:
    """This card's hardware class, its attention kernels, and the sets of them that can share one process."""
    from cozy_runtime.internal.bench import attention

    return _run(attention.inventory)


def attention_speed(
    ctx: Context,
    tel: Telemetry,
    *,
    kernels: str = "",
    points: str = "",
    seed: int = 171,
    corpus: str = "",
) -> Measured:
    """`kernels` (comma-separated; SDPA joins as the anchor) timed over the grid's points (`points`, comma-separated
    ids; empty is every one), interleaved, three rounds; with `corpus` (a capture directory on this machine) the sparse
    points take its H3 sparse steps' Q/K/V and layouts. Its progress counts points under the stage "measure", not a
    sampler's steps."""
    from cozy_runtime.internal.bench import attention

    names = [n for n in kernels.split(",") if n]
    wanted = {p for p in points.split(",") if p}
    chosen = [p for p in attention.GRID if not wanted or p.id in wanted]
    with tel.stage("measure"):
        return _run(lambda: attention.speed(names, chosen, seed, ctx, corpus))


def attention_quality(
    ctx: Context, tel: Telemetry, *, kernel: str, seed: int = 171, corpus: str = ""
) -> Measured:
    """`kernel`'s worst-head relative L2 against FP32 attention on every site, with the best answer a kernel could give
    without computing; with `corpus` (a capture directory on this machine) every capture at the kernel's layout and at
    or below its level is a site too (`kind` "captured"). Its progress counts sites under the stage "measure"."""
    from cozy_runtime.internal.bench import attention

    with tel.stage("measure"):
        return _run(lambda: attention.quality(kernel, seed, ctx, corpus))


def gemm_speed(
    ctx: Context, tel: Telemetry, *, lanes: str = "", shapes: str = "", seed: int = 177
) -> Measured:
    """Every GEMM candidate of each lane (`lanes`, comma-separated; empty is every one) at our packages' shapes
    (`shapes`: shape ids or model names, comma-separated; empty is every one), interleaved a sample at a time, beside the
    pick the Runtime's rule makes (xs-177). Its progress counts shapes under the stage "measure"."""
    from cozy_runtime.internal.bench import gemm

    chosen = [n for n in lanes.split(",") if n]
    wanted = [n for n in shapes.split(",") if n]
    with tel.stage("measure"):
        return _run(lambda: gemm.speed(chosen, wanted, seed, ctx))


def host_calibrate(ctx: Context) -> Measured:
    """This pod's sustained bf16 GEMM TFLOPS and memory GB/s, which the gate holds against its card's reference: a host
    far below it (an H100 that ran every kernel 3x slow, release gate 43061813774) is refused (xs-177)."""
    from cozy_runtime.internal.bench import gemm

    return _run(gemm.calibrate)


def gemm_tiles(ctx: Context, tel: Telemetry, *, seed: int = 177) -> Measured:
    """Two of the Runtime's Triton kernels (nvfp4's weight decode and GEMV) over the tile settings that keep every output
    bit, at our shapes: whether one card of a class wants another tile than its sibling (xs-177). Its progress counts
    tile sets under the stage "measure"."""
    from cozy_runtime.internal.bench import gemm

    with tel.stage("measure"):
        return _run(lambda: gemm.tiles(seed, ctx))


def kernels_inventory(ctx: Context) -> Measured:
    """Every kernel this card claims, read from the Runtime's code: the names `kernel_check` takes, each Triton set's
    entries and each module's jit functions."""
    from cozy_runtime.internal.bench import kernels

    return Measured(json.dumps(kernels.inventory(), sort_keys=True))


def kernel_check(ctx: Context, *, name: str, index: int = 0, seed: int = 400) -> Measured:
    """The kernel `name` (an inventory name) on GPU `index`, through its real launch path, against a slow reference:
    its verdict and the entries and jit functions it launched. Raises when it fails."""
    from cozy_runtime.internal.bench import kernels

    return Measured(json.dumps(kernels.exercise(name, index, seed), sort_keys=True))


def kernel_ready(ctx: Context, *, mode: str, seed: int = 202610270201) -> Measured:
    """Kernel readiness publication under two real openers of one kernel, on the CPU: `late-success`, `late-error` or
    `dropped-slot` (`internal/bench/kernel_ready.py`). Raises when the registry publishes wrong."""
    from cozy_runtime.internal.bench import kernel_ready as check

    return Measured(json.dumps(check.check(mode, seed), sort_keys=True))


def varena_core(ctx: Context, *, case: str) -> Measured:
    """`case`, one of varena's rules that only the Runtime's own adapter reaches (`internal/bench/varena_cases.py`), in a
    fresh process: varena's allocator installs there before CUDA starts. Raises when it fails."""
    from tensorfs.varena import _runtime as varena

    command = [sys.executable, "-I", "-B", "-m", "cozy_runtime.internal.bench.varena_cases", case]
    lines: list[str] = []
    device = ctx.device.index or 0
    # fill-paths-agree pins its tier at idle, which an idle tier does only while no peer is in a request (TensorFS
    # a563efcd6): this executor, waiting on the case's process, steps out of its request for that one. The other cases
    # run beside the executor's request (split-waits-and-caps evicts the plan's weights without it, gate 43054657524).
    idle = case == "fill-paths-agree"
    if idle:
        varena.request_boundary(device, begin=False)
    try:
        code = _child(command, lines, ctx)
    finally:
        if idle:
            varena.request_boundary(device, begin=True)
    rows = [json.loads(line) for line in lines if line.startswith("{")]
    found = {"case": case, "exit": code, "rows": rows, "tail": lines[-40:] if code else []}
    if code:
        raise RuntimeError(f"varena case {case} failed: {json.dumps(found)[-4000:]}")
    return Measured(json.dumps(found, sort_keys=True))


def _child(command: list[str], lines: list[str], ctx: Context) -> int:
    """`command`'s exit code, its output lines into `lines`; killed when the call is canceled."""
    with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as child:
        assert child.stdout is not None
        try:
            for line in child.stdout:
                lines.append(line.rstrip())
                ctx.raise_if_cancelled()
        except BaseException:
            child.kill()
            raise
    return child.returncode


def offload_in_executor(ctx: Context) -> Measured:
    """`vt.offload` inside this executor, on the allocator it installed and the stream it chose: a small model offloaded
    serves what it served on the CPU, and leaves the executor's stream current. Raises when not."""
    import tensorfs.varena.torch as vt
    import torch
    from torch import nn

    device = torch.device("cuda", 0)
    stream = torch.cuda.current_stream(device)
    x = torch.randn(1, 8)
    model = nn.Sequential(nn.Linear(8, 8), nn.Linear(8, 8)).eval()
    with torch.inference_mode():
        want = model(x).to(device)
    vt.offload(model, device=0)
    with torch.inference_mode():
        got = [model(x.to(device)) for _ in range(6)][-1]
    found = {"close": bool(torch.allclose(got, want, atol=1e-5)), "same_stream": torch.cuda.current_stream(device) == stream,
             "explicit_stream": stream.cuda_stream > 2}
    if not all(found.values()):
        raise RuntimeError(f"vt.offload inside an executor: {json.dumps(found, sort_keys=True)}")
    return Measured(json.dumps(found, sort_keys=True))


def quantize_reuse(ctx: Context, tel: Telemetry, *, case: str, seed: int = 156) -> Measured:
    """The Runtime's per-component quantization across an Output's lifetime, on the CPU: `components` or `resume`
    (`internal/bench/quantize_reuse.py`). Raises when a conversion is reused, refused or written wrong."""
    from cozy_runtime.internal.bench import quantize_reuse as check

    return Measured(json.dumps(check.exercise(ctx, tel, case, seed), sort_keys=True, default=str))


class Wired(msgspec.Struct, frozen=True):
    """`child_wire`'s refusal codes, as JSON, and the file it saved for them (returned, as every saved file is)."""

    detail: str
    file: File


def child_wire(ctx: Context, out: Outputs) -> Wired:
    """Child calls whose wire lies about what this attempt holds, as a package could send them: a saved file's ref with
    another digest, a tree manifest, a spool symlink (`internal/bench/child_wire.py`). Raises unless TensorD refuses
    each."""
    from cozy_runtime.internal.bench import child_wire as check

    found, file = check.check(out)
    return Wired(json.dumps(found, sort_keys=True), file)


class OverlayProbe(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    """`overlay_replay`'s request: replay (unarmed twice, then armed before the loop, against graphs bypassed) or inside
    (armed in the step, against armed before the loop)."""

    seed: int = 331
    variant: str = "replay"


def overlay_replay(payload: OverlayProbe, model: QwenDiT, turbo_lora: Lightning, ctx: Context, tel: Telemetry) -> Measured:
    """Qwen-Image-Edit-2511's DiT with the Lightning LoRA's hooks, armed as packages arm it: the armed calls through the
    Runtime's graphs equal the same calls with graphs bypassed (`internal/bench/qwen_overlay.py`). Raises when not."""
    from cozy_runtime.internal.bench import qwen_overlay

    found = qwen_overlay.replay(model, turbo_lora, payload.seed, payload.variant, ctx, tel)
    return Measured(json.dumps({"variant": payload.variant, "checks": found}, sort_keys=True))



class RefillProbe(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    """`lora_overlay_refill`'s request."""

    seed: int = 4242
    #: square image side in pixels (16 px a token)
    size: int = 512
    #: DiT forwards per call
    steps: int = 2
    #: GiB the ballast leaves the DiT's pages: less than the DiT, so it pages within each call
    room_gib: float = 6.0
    #: warm calls after the references, alternating LoRA and base
    rounds: int = 3


def lora_overlay_refill(payload: RefillProbe, model: QwenDiT, turbo_lora: Lightning, ctx: Context, tel: Telemetry) -> Measured:
    """The Lightning LoRA armed in its DiT's own forward, as H3 arms Turbo, through the int8 lane's baked pages while a
    ballast pages the DiT within and between calls: every warm call equals its reference (`internal/bench/lora_refill.py`).
    Raises when one doesn't."""
    from cozy_runtime.internal.bench import lora_refill

    result = lora_refill.exercise(payload, model, turbo_lora, ctx, tel)
    detail = json.dumps(result, sort_keys=True, default=str)
    tel.log("lora overlay refill result", result=detail)
    if not result["passed"]:
        brief = {k: result.get(k) for k in ("failures", "lora_effect", "bake_error")}
        raise RuntimeError(f"LoRA overlay refill failed: {json.dumps(brief, sort_keys=True, default=str)}")
    return Measured(detail)
