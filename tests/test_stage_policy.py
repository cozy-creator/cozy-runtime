"""Streaming arithmetic, driven with recorded real-run facts.

Every size, time and rate comes from `testdata/stage-costs/recorded.json`: Runtime receipts of
the six-job SDXL/Anima benchmark on the RTX 4070 Laptop, H3 fp8 runs on 1/2/4 H100 SXM, Qwen-
Image-2 calls on H100, and `paging.partition` of each architecture. One modelling assumption
is named where it is used: per-block compute is not recorded, so a step's measured compute is
spread over its blocks in proportion to their bytes.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cozy_runtime.internal import weight_policy

FX = json.loads((Path(__file__).parent / "testdata/stage-costs/recorded.json").read_text())
LAPTOP, H100 = FX["machines"]["rtx4070_laptop"], FX["machines"]["h100_sxm"]
SDXL, ANIMA, H3, QWEN = FX["sdxl"], FX["anima"], FX["h3"], FX["qwen_image_2"]


# --------------------------------------------------------------------------- the cost model


def steady_pass(compute: list[float], sizes: list[int], resident: list[bool], link: float) -> float:
    """C2's per-region event model (`weight_policy.simulate`), two regions in flight."""
    streamed = [not hot for hot in resident]
    return weight_policy.simulate(sizes, compute, streamed, 2, link * 1e9)


def proportional(step_s: float, sizes: list[int]) -> list[float]:
    """Per-block compute is unrecorded: spread the measured step by bytes (named assumption)."""
    total = sum(sizes)
    return [step_s * size / total for size in sizes]


def test_overlapped_streaming_hides_the_tail_sdxl_streamed_synchronously_at_3_gib() -> None:
    blocks = SDXL["unet"]["blocks"]
    keep = SDXL["streamed_3gib"]["keep"]
    # The fixture's block order reproduces the tail the R19 run streamed every step.
    assert sum(blocks[keep:]) == SDXL["streamed_3gib"]["tail_bytes_per_step"]
    sizes = [SDXL["unet"]["common"], *blocks]
    resident = [True] * (1 + keep) + [False] * (len(blocks) - keep)
    compute = proportional(SDXL["step_s"], sizes)

    pinned = steady_pass(compute, sizes, resident, LAPTOP["pinned_gbps"])
    # 2.64 GB at 11.9 GB/s hides; what is left is the model's per-copy and per-block costs
    assert pinned == pytest.approx(SDXL["step_s"], rel=0.02)
    assert SDXL["streamed_3gib"]["step_s"] > 2.5 * pinned  # the recorded synchronous step

    disk = steady_pass(compute, sizes, resident, LAPTOP["disk_gbps"])
    copy = SDXL["streamed_3gib"]["tail_bytes_per_step"] / (LAPTOP["disk_gbps"] * 1e9)
    assert copy <= disk <= copy + SDXL["step_s"]  # from disk the link is the step


def test_overlapped_streaming_hides_anima_serial_cfg_at_3_gib() -> None:
    blocks = ANIMA["transformer"]["blocks"]
    keep = ANIMA["streamed_3gib"]["keep"]
    assert (
        ANIMA["calls_per_step"] * sum(blocks[keep:])
        == ANIMA["streamed_3gib"]["tail_bytes_per_step"]
    )
    sizes = [ANIMA["transformer"]["common"], *blocks]
    resident = [True] * (1 + keep) + [False] * (len(blocks) - keep)
    call = steady_pass(
        proportional(ANIMA["step_s"] / 2, sizes), sizes, resident, LAPTOP["pinned_gbps"]
    )
    assert 2 * call == pytest.approx(ANIMA["step_s"], rel=0.01)
    assert ANIMA["streamed_3gib"]["step_s"] > 1.8 * 2 * call


def test_h3_fp8_streams_its_whole_dit_on_h100_within_the_phase_2_gate() -> None:
    """Gate: H3 fp8 streaming <= 1.10x all-resident. Block bytes are the recorded DiT size over
    its 50 blocks (derived)."""
    sizes = [H3["dit_block_bytes_derived"]] * H3["dit_blocks"]
    step = H3["widths"]["1"]["step_s"]
    streamed = steady_pass(
        proportional(step, sizes), sizes, [False] * len(sizes), H100["link_gbps"]
    )
    assert streamed <= 1.10 * step
    assert streamed == pytest.approx(step, rel=0.01)
