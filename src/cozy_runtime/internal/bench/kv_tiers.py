"""How K/V the card can't hold should be served, measured (cr-336's open question; the varena
suite's `kv-tiers` case, `cozy_runtime.bench.varena_core`).

One row's decode at Qwen3.5-9B's K/V layout, every attention layer a step, with none, a quarter,
half and all of its blocks off the card, served three ways:
- `in-place`: the kernel reads the blocks from host memory the device addresses;
- `window`: each layer's blocks are copied to a window on the card, then the kernel reads it;
- `ahead`: as `window`, but each layer's copy runs on a copy stream into one of two windows while
  the layer before it computes, as a step's graph would order it.

Each way is timed on attention alone (the K/V's share of a token), and with a stand-in for the
step's weight reads before each layer's attention (`STANDINS`): the work a copy ahead can hide
behind and an in-place read can't. The stand-in moves the bytes a plain step of the model it
names reads, through bf16 GEMVs from the card. Every arm's output is held against a float32
reference before it is timed. It prints the device's answer about host addressing first; where
that says no, only the card arms run.
"""

from __future__ import annotations

import statistics
import subprocess
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import cast

import torch
from tensorfs.varena import _runtime as varena

from cozy_runtime.engine.attention import HostTier, paged_attention
from cozy_runtime.engine.kv import KvLayout

type JSON = None | bool | int | float | str | Sequence[JSON] | Mapping[str, JSON]

#: Qwen3.5-9B's K/V (design inference-engine.md §4.2): 8 attention layers, 4 K/V heads of 256.
LAYOUT = KvLayout(attn_layers=8, kv_heads=4, head_dim=256, dtype=torch.bfloat16)
HEADS = 16
TOKENS = 65536
#: Parts a row is read in at once.
PARTS = 256
#: What an arm's output may differ from the float32 reference by, relative to its largest value.
BUDGET = 2e-2
#: An arm is timed for at least this long, so the card's utilisation and power are sampled, and
#: for at most `ARM_MOST_S` once it has three steps: a slow way of serving still reports.
ARM_S = 3.0
ARM_MOST_S = 120.0

#: The weight bytes a plain decode step of Qwen3.5-9B reads (one row: every body linear and the
#: head; the token table gives one row), by lane, from its config's shapes (hidden 4096, MLP 12288,
#: 24 gated-delta and 8 attention layers, a 248,320-token head). q8_0: 34 bytes a 32 weights; the
#: head's 1,080,688,640 B is the lane's as measured (44290207312). q4_k_m: llama-quantize's Q4_K_M
#: rule as `derive.ggml` ports it (q6_k head, ffn_down and attn_v/attn_qkv in its 16 "more bits"
#: layers; q4_k elsewhere): that places 33 of the lane's 35 q6_k tensors (ingest 44282983121), so
#: within about 1%. An arm stands for the largest lane the card holds beside this K/V.
STANDINS = {"qwen3.5-9b q8_0": 8_431_599_616, "qwen3.5-9b q4_k_m": 5_041_766_400}
#: The stand-in's matrix: bf16 rows of the hidden width, re-read so it moves its bytes from the
#: card's memory (far larger than any L2) without holding them.
STANDIN_ROWS, STANDIN_COLS = 32768, 4096


def _values(layer: int, blocks: int) -> torch.Tensor:
    """`layer`'s K and V for every block, `[blocks, 2, block_tokens, kv_heads, head_dim]`."""
    made = torch.Generator(device="cuda").manual_seed(213 + layer)
    shape = (blocks, 2, LAYOUT.block_tokens, LAYOUT.kv_heads, LAYOUT.head_dim)
    values = torch.randn(shape, generator=made, device="cuda", dtype=torch.float32)
    return values.mul_(0.25).to(LAYOUT.dtype)


def _reference(q: torch.Tensor, layer: int, blocks: int, scale: float) -> torch.Tensor:
    """`q` attended over every token of `layer`, in float32, by torch."""
    both = _values(layer, blocks)
    keys = both[:, 0].flatten(0, 1).float()  # [tokens, kv_heads, head_dim]
    values = both[:, 1].flatten(0, 1).float()
    group = HEADS // LAYOUT.kv_heads
    outs = []
    for kvh in range(LAYOUT.kv_heads):
        mine = q[0, kvh * group : (kvh + 1) * group].float() * scale
        weights = torch.softmax(mine @ keys[:, kvh].T, dim=-1)
        outs.append(weights @ values[:, kvh])
    return torch.cat(outs)


class _Watch:
    """The card's utilisation and power while an arm runs, where the pod has nvidia-smi."""

    def __init__(self) -> None:
        self.process: subprocess.Popen[str] | None = None
        try:
            self.process = subprocess.Popen(
                ["nvidia-smi", "-i", "0", "--query-gpu=utilization.gpu,power.draw",
                 "--format=csv,noheader,nounits", "-lms", "100"],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
            )  # fmt: skip
        except OSError:
            pass

    def read(self) -> dict[str, JSON]:
        if self.process is None:
            return {"watched": False}
        self.process.terminate()
        out, _ = self.process.communicate()
        rows = [line.split(",") for line in out.splitlines() if line.count(",") == 1]
        try:
            pairs = [(float(u), float(w)) for u, w in rows]
        except ValueError:
            return {"watched": False}
        if not pairs:
            return {"watched": False}
        return {
            "watched": True,
            "samples": len(pairs),
            "utilization_pct": statistics.fmean(u for u, _ in pairs),
            "power_w": statistics.fmean(w for _, w in pairs),
        }


def _windows(first_byte: int, block_bytes: int, blocks: int, unit: int) -> Iterator[tuple[int, int]]:
    """Blocks `[0, blocks)` of a range starting at `first_byte`, as runs none of which crosses a
    multiple of `unit`: a copy from registered host memory may not span two registrations."""
    at = 0
    while at < blocks:
        room = unit - (first_byte + at * block_bytes) % unit
        take = min(max(room // block_bytes, 1), blocks - at)
        yield at, at + take
        at += take


def _standin() -> tuple[str, int]:
    """The largest lane in `STANDINS` this card holds beside the K/V and a GiB, and its bytes."""
    _, card = torch.cuda.mem_get_info(0)
    kv = TOKENS * LAYOUT.block_bytes // LAYOUT.block_tokens
    fits = [(b, name) for name, b in STANDINS.items() if b + kv + (1 << 30) <= card]
    bytes_, name = max(fits) if fits else min((b, n) for n, b in STANDINS.items())
    return name, bytes_


def measure() -> dict[str, JSON]:
    layout = LAYOUT
    blocks = TOKENS // layout.block_tokens
    layers = layout.attn_layers
    scale = layout.head_dim**-0.5
    layer_bytes = layout.block_bytes // layers
    tile = (2, layout.block_tokens, layout.kv_heads, layout.head_dim)
    gpu = varena.Device(0)
    stream, copier = torch.cuda.Stream(), torch.cuda.Stream()
    probe = varena.host_addressing(0)
    standin, standin_bytes = _standin()
    row: dict[str, JSON] = {
        "probe": cast("JSON", dict(probe)),
        "layout": {"attn_layers": layers, "kv_heads": layout.kv_heads,
                   "head_dim": layout.head_dim, "heads": HEADS, "tokens": TOKENS, "parts": PARTS,
                   "kv_gib": blocks * layout.block_bytes / 2**30},
        "standin": {"model": standin, "bytes_step": standin_bytes,
                    "how": "bf16 GEMVs from the card's memory, the step's weight bytes before each layer's attention"},
    }  # fmt: skip

    card_cache = gpu.cache(blocks * layout.block_bytes)
    card_cache.commit(0, card_cache.reserved_bytes)
    card = card_cache.tensor(layout.dtype, (blocks, layers, *tile))
    window = torch.empty((2, blocks, *tile), dtype=layout.dtype, device="cuda")
    matrix = torch.randn((STANDIN_ROWS, STANDIN_COLS), device="cuda").mul_(0.02).to(torch.bfloat16)
    vector = torch.randn(STANDIN_COLS, device="cuda").to(torch.bfloat16)
    host_cache: varena.HostCache | None = None
    host_dev = host_cpu = None
    if probe["supported"]:
        tier = varena.Pages(0, cap=1 << 34)
        host_cache = tier.cache(layers * blocks * layer_bytes)
        if host_cache.commit(0, host_cache.reserved_bytes):
            shape = (layers, blocks, *tile)
            host_dev, host_cpu = host_cache.tensor(layout.dtype, shape), host_cache.host(layout.dtype, shape)
        else:
            row["host"] = "the host tier has no room for the K/V now"
    else:
        row["host"] = probe["why"]

    unit = host_cache.granularity if host_cache is not None else 1
    d2h_s = 0.0
    with torch.cuda.stream(stream):
        for layer in range(layers):
            values = _values(layer, blocks)
            card[:, layer].copy_(values)
            if host_cpu is not None:  # window by window: a copy may not span two registrations
                stream.synchronize()
                began = time.perf_counter()
                for a, b in _windows(layer * blocks * layer_bytes, layer_bytes, blocks, unit):
                    host_cpu[layer, a:b].copy_(values[a:b])
                d2h_s += time.perf_counter() - began
            del values
        # Scores a few units wide, so a wrong key shows in the output as surely as a wrong value.
        q = torch.randn((1, HEADS, layout.head_dim), device="cuda").mul_(4).to(layout.dtype)
        lengths = torch.full((1,), TOKENS, dtype=torch.int32, device="cuda")
        stream.synchronize()
    if host_cpu is not None:
        row["d2h_gb_s"] = layers * blocks * layer_bytes / d2h_s / 1e9

    matrix_bytes = STANDIN_ROWS * STANDIN_COLS * 2
    per_layer = standin_bytes // layers

    def weights() -> None:
        """One layer's share of the stand-in's bytes, read on the current stream."""
        left = per_layer
        while left >= matrix_bytes:
            torch.mv(matrix, vector)
            left -= matrix_bytes
        if rows := left // (STANDIN_COLS * 2):
            torch.mv(matrix[:rows], vector)

    def table(off_card: int) -> torch.Tensor:
        """The row's blocks, its oldest `off_card` on the host tier."""
        entries = torch.arange(blocks, dtype=torch.int32, device="cuda")
        entries[:off_card] += blocks
        return entries[None].contiguous()

    def copy(layer: int, slot: int, off_card: int) -> None:
        assert host_cpu is not None
        for a, b in _windows(layer * blocks * layer_bytes, layer_bytes, off_card, unit):
            window[slot, a:b].copy_(host_cpu[layer, a:b], non_blocking=True)

    def attend(host: HostTier, layer: int, entries: torch.Tensor) -> torch.Tensor:
        return paged_attention(q, card, host, entries, lengths, layout, layer, PARTS, scale)

    def serial(mode: str, off_card: int, entries: torch.Tensor, work: bool) -> Callable[[], list[torch.Tensor]]:
        """A step served one layer after another on `stream`: card, in place, or a window."""

        def step() -> list[torch.Tensor]:
            outs = []
            for layer in range(layers):
                if work:
                    weights()
                if mode == "window":
                    copy(layer, 0, off_card)
                    host = HostTier(window[:1], 0)
                elif mode == "in-place" and host_dev is not None:
                    host = HostTier(host_dev, layer)
                else:
                    host = HostTier(window[:1], 0)  # no block is off the card: never read
                outs.append(attend(host, layer, entries))
            return outs

        return step

    def ahead(off_card: int, entries: torch.Tensor, work: bool) -> Callable[[], list[torch.Tensor]]:
        """A step whose copies run on `copier`, layer l + 1's into the other window while layer l
        computes; a window is refilled only after the attention that read it."""
        copied = [torch.cuda.Event(), torch.cuda.Event()]
        read: list[torch.cuda.Event | None] = [None, None]
        issued = {"next": 0}

        def issue(index: int) -> None:
            slot = index % 2
            with torch.cuda.stream(copier):
                if (last := read[slot]) is not None:
                    copier.wait_event(last)
                copy(index % layers, slot, off_card)
                copied[slot].record(copier)

        def step() -> list[torch.Tensor]:
            outs = []
            if issued["next"] == 0:
                issue(0)
                issued["next"] = 1
            for layer in range(layers):
                index = issued["next"] - 1
                slot = index % 2
                if work:
                    weights()
                stream.wait_event(copied[slot])
                outs.append(attend(HostTier(window[slot : slot + 1], 0), layer, entries))
                done = torch.cuda.Event()
                done.record(stream)
                read[slot] = done
                issue(issued["next"])
                issued["next"] += 1
            return outs

        return step

    arms: list[JSON] = []
    shares = [0.0] if host_cpu is None else [0.0, 0.25, 0.5, 1.0]
    for share in shares:
        off_card = round(share * blocks)
        entries = table(off_card)
        modes = ["card"] if not off_card else ["in-place", "window", "ahead"]
        for work in (False, True):
            for mode in modes:
                if mode == "ahead" and not work:
                    continue  # with nothing to hide behind it is `window`
                step = ahead(off_card, entries, work) if mode == "ahead" else serial(mode, off_card, entries, work)
                with torch.cuda.stream(stream):
                    outs = step()  # also compiles the kernel, untimed
                    worst = 0.0
                    for layer in (0, layers - 1):
                        want = _reference(q, layer, blocks, scale)
                        got = outs[layer][0].float()
                        worst = max(worst, float((got - want).abs().max() / want.abs().max()))
                    assert worst <= BUDGET, f"{mode} at {share}: off its reference by {worst}"
                    del outs
                    stream.synchronize()
                    watch = _Watch()
                    steps: list[float] = []
                    began = time.perf_counter()
                    while True:
                        spent = time.perf_counter() - began
                        if len(steps) >= 3 and (spent >= ARM_MOST_S or spent >= ARM_S and len(steps) >= 5):
                            break
                        at = time.perf_counter()
                        step()
                        stream.synchronize()
                        steps.append(time.perf_counter() - at)
                    seen = watch.read()
                    copier.synchronize()
                step_s = statistics.median(steps)
                arms.append({
                    "off_card": share, "mode": mode, "standin": work, "steps": len(steps),
                    "ms_step": 1e3 * step_s, "ms_step_min": 1e3 * min(steps), "steps_s": 1 / step_s,
                    "host_gib_step": off_card * layout.block_bytes / 2**30,
                    "worst_error": worst, **seen,
                })  # fmt: skip
    row["arms"] = arms

    if host_cpu is not None:  # the link's rate: one layer's blocks to the card, window by window
        with torch.cuda.stream(stream):
            stream.synchronize()
            began = time.perf_counter()
            for layer in range(layers):
                copy(layer, 0, blocks)
            stream.synchronize()
            spent = time.perf_counter() - began
        row["h2d_gb_s"] = layers * blocks * layer_bytes / spent / 1e9

    del card, host_dev, host_cpu, window, matrix
    try:  # tidy; the measurement above stands whether or not it succeeds (the V2 cells judge this)
        card_cache.release(0, card_cache.reserved_bytes, stream.cuda_stream)
        stream.synchronize()
        card_cache.close()
        if host_cache is not None and "host" not in row:
            host_cache.release(0, host_cache.reserved_bytes)
            host_cache.close()
    except RuntimeError as exc:
        row["cleanup"] = str(exc)
    torch.cuda.synchronize()
    return row
