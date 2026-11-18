"""How K/V the card can't hold should be served, measured (cr-336's open question; the varena
suite's `kv-tiers` case, `cozy_runtime.bench.varena_core`).

One row's decode attention at Qwen3.5-9B's K/V layout, every attention layer a step, with none,
a quarter, half and all of its blocks off the card, served two ways:
- `in-place`: the kernel reads the blocks from host memory the device addresses;
- `window`: each layer's blocks are copied to a window on the card first, and the kernel reads that.

It times attention alone: no weights are read, so a step here is the K/V's share of a token, not a
token. Every arm's output is held against a float32 reference before it is timed. It prints the
device's answer about host addressing first; where that says no, only the card arm runs.
"""

from __future__ import annotations

import statistics
import subprocess
import time
from collections.abc import Iterator, Mapping, Sequence
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


def measure() -> dict[str, JSON]:
    layout = LAYOUT
    blocks = TOKENS // layout.block_tokens
    scale = layout.head_dim**-0.5
    layer_bytes = layout.block_bytes // layout.attn_layers
    tile = (2, layout.block_tokens, layout.kv_heads, layout.head_dim)
    gpu = varena.Device(0)
    stream = torch.cuda.Stream()
    probe = varena.host_addressing(0)
    row: dict[str, JSON] = {
        "probe": cast("JSON", dict(probe)),
        "layout": {"attn_layers": layout.attn_layers, "kv_heads": layout.kv_heads,
                   "head_dim": layout.head_dim, "heads": HEADS, "tokens": TOKENS, "parts": PARTS,
                   "kv_gib": blocks * layout.block_bytes / 2**30},
    }  # fmt: skip

    card_cache = gpu.cache(blocks * layout.block_bytes)
    card_cache.commit(0, card_cache.reserved_bytes)
    card = card_cache.tensor(layout.dtype, (blocks, layout.attn_layers, *tile))
    window = torch.empty((1, blocks, *tile), dtype=layout.dtype, device="cuda")
    host_cache: varena.HostCache | None = None
    host_dev = host_cpu = None
    if probe["supported"]:
        tier = varena.Pages(0, cap=1 << 34)
        host_cache = tier.cache(layout.attn_layers * blocks * layer_bytes)
        if host_cache.commit(0, host_cache.reserved_bytes):
            shape = (layout.attn_layers, blocks, *tile)
            host_dev, host_cpu = host_cache.tensor(layout.dtype, shape), host_cache.host(layout.dtype, shape)
        else:
            row["host"] = "the host tier has no room for the K/V now"
    else:
        row["host"] = probe["why"]

    d2h_s = 0.0
    with torch.cuda.stream(stream):
        for layer in range(layout.attn_layers):
            values = _values(layer, blocks)
            card[:, layer].copy_(values)
            if host_cpu is not None:
                stream.synchronize()
                began = time.perf_counter()
                host_cpu[layer].copy_(values)
                d2h_s += time.perf_counter() - began
            del values
        # Scores a few units wide, so a wrong key shows in the output as surely as a wrong value.
        q = torch.randn((1, HEADS, layout.head_dim), device="cuda").mul_(4).to(layout.dtype)
        lengths = torch.full((1,), TOKENS, dtype=torch.int32, device="cuda")
        stream.synchronize()
    if host_cpu is not None:
        row["d2h_gb_s"] = layout.attn_layers * blocks * layer_bytes / d2h_s / 1e9

    unit = host_cache.granularity if host_cache is not None else 1

    def table(off_card: int) -> torch.Tensor:
        """The row's blocks, its oldest `off_card` on the host tier."""
        entries = torch.arange(blocks, dtype=torch.int32, device="cuda")
        entries[:off_card] += blocks
        return entries[None].contiguous()

    def tier_for(mode: str, layer: int, off_card: int) -> HostTier:
        if mode == "window" and host_cpu is not None:
            first = layer * blocks * layer_bytes
            for a, b in _windows(first, layer_bytes, off_card, unit):
                window[0, a:b].copy_(host_cpu[layer, a:b], non_blocking=True)
            return HostTier(window, 0)
        if mode == "in-place" and host_dev is not None:
            return HostTier(host_dev, layer)
        return HostTier(window, 0)  # no block is off the card: never read

    def attend(mode: str, layer: int, off_card: int, entries: torch.Tensor) -> torch.Tensor:
        host = tier_for(mode, layer, off_card)
        return paged_attention(q, card, host, entries, lengths, layout, layer, PARTS, scale)

    arms: list[JSON] = []
    shares = [0.0] if host_cpu is None else [0.0, 0.25, 0.5, 1.0]
    for share in shares:
        off_card = round(share * blocks)
        for mode in ["card"] if not off_card else ["in-place", "window"]:
            entries = table(off_card)
            with torch.cuda.stream(stream):
                worst = 0.0
                for layer in (0, layout.attn_layers - 1):  # also compiles the kernel, untimed
                    got = attend(mode, layer, off_card, entries)[0].float()
                    want = _reference(q, layer, blocks, scale)
                    worst = max(worst, float((got - want).abs().max() / want.abs().max()))
                assert worst <= BUDGET, f"{mode} at {share}: off its reference by {worst}"
                stream.synchronize()
                watch = _Watch()
                steps: list[float] = []
                began = time.perf_counter()
                while True:
                    spent = time.perf_counter() - began
                    if len(steps) >= 3 and (spent >= ARM_MOST_S or spent >= ARM_S and len(steps) >= 5):
                        break
                    at = time.perf_counter()
                    for layer in range(layout.attn_layers):
                        attend(mode, layer, off_card, entries)
                    stream.synchronize()
                    steps.append(time.perf_counter() - at)
                seen = watch.read()
            step_s = statistics.median(steps)
            host_bytes = off_card * layout.block_bytes
            arms.append({
                "off_card": share, "mode": mode, "steps": len(steps), "ms_step": 1e3 * step_s,
                "ms_step_min": 1e3 * min(steps), "steps_s": 1 / step_s,
                "host_gib_step": host_bytes / 2**30,
                "worst_error": worst, **seen,
            })  # fmt: skip
    row["arms"] = arms

    if host_cpu is not None:  # the link's rate: one layer's blocks to the card, window by window
        with torch.cuda.stream(stream):
            stream.synchronize()
            began = time.perf_counter()
            for layer in range(layout.attn_layers):
                for a, b in _windows(layer * blocks * layer_bytes, layer_bytes, blocks, unit):
                    window[0, a:b].copy_(host_cpu[layer, a:b], non_blocking=True)
            stream.synchronize()
            spent = time.perf_counter() - began
        row["h2d_gb_s"] = layout.attn_layers * blocks * layer_bytes / spent / 1e9

    del card, host_dev, host_cpu, window
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
