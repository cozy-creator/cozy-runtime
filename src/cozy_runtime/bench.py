"""Unstable: for the release gate only, outside the stable API; it may change or go in any release.

The Runtime's exercisers of its own code: built-in calls of the Runtime's own App (`derive.operations`), which every
environment holds, so the release gate measures and checks the Runtime from inside it (a package can't import its
internals). A gate script calls them with `call(bench.attention_speed, ...)`; each returns one JSON document.
`internal/bench/` says what each does: the attention bench (xs-171) and the kernels suite (xs-174)."""

from __future__ import annotations

import json
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


def attention_speed(ctx: Context, *, kernels: str = "", points: str = "", seed: int = 171) -> Measured:
    """`kernels` (comma-separated; SDPA joins as the anchor) timed over the grid's points (`points`, comma-separated
    ids; empty is every one), interleaved, three rounds."""
    from cozy_runtime.internal.bench import attention

    names = [n for n in kernels.split(",") if n]
    wanted = {p for p in points.split(",") if p}
    chosen = [p for p in attention.GRID if not wanted or p.id in wanted]
    return _run(lambda: attention.speed(names, chosen, seed, ctx))


def attention_quality(ctx: Context, *, kernel: str, seed: int = 171) -> Measured:
    """`kernel`'s worst-head relative L2 against FP32 attention on every site, with the best answer a kernel could give
    without computing."""
    from cozy_runtime.internal.bench import attention

    return _run(lambda: attention.quality(kernel, seed, ctx))


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
