"""A pure cut site's level from its live shapes against the agent's room, never an OOM probe
(status/s4-leases-runtime-plan.md §5): the ladder, the descent, group agreement and each site's
bytes. The card proof that each formula covers the measured peak is tests/test_band_live_bytes."""

from __future__ import annotations

import pytest

from cozy_runtime.internal.execution import sizing
from cozy_runtime.internal.execution.sizing import Attention, Ffn, Room

MIB = 1 << 20
GRAIN = 2 * MIB


def _need(units: int, per_unit: int, out: int = 0) -> sizing.Need:
    return sizing.levelled(units, lambda n, granularity: out + n * per_unit)


def test_the_coarsest_level_that_fits_in_cold_room_evicts_nothing_of_the_step() -> None:
    need = _need(64, MIB, out=8 * MIB)  # 64 units: whole 72 MiB, halves 40, quarters 24
    assert sizing.choose(6, need, Room(free=80 * MIB, cold=0, warm=0, granularity=1, floor=0)) == (
        0,
        "cold",
    )
    assert sizing.choose(
        64, need, Room(free=20 * MIB, cold=20 * MIB, warm=0, granularity=1, floor=0)
    ) == (
        1,
        "cold",
    )


def test_the_warm_tier_leaves_the_streaming_floor_bindable() -> None:
    """P1: evicting the step's own blocks needs SLOTS of the largest left free, or they cannot
    stream back within the step."""
    need = _need(64, MIB, out=8 * MIB)
    room = Room(free=4 * MIB, cold=4 * MIB, warm=40 * MIB, granularity=1, floor=24 * MIB)
    assert sizing.choose(6, need, room) == (2, "warm")  # 24 MiB fits 48 - 24, 40 does not
    assert sizing.choose(6, need, Room(free=1, cold=0, warm=0, granularity=1, floor=0)) == (
        6,
        "floor",
    )


def test_a_refusal_descends_one_level_at_least_and_rereads_the_room() -> None:
    """P2: the room is a snapshot; a refusal above the finest re-reads it and runs finer, never
    the same size again; only the finest's refusal is the floor."""

    class Refused(Exception):
        pass

    rooms = iter(
        [Room(100 * MIB, 0, 0, 1, 0), Room(100 * MIB, 0, 0, 1, 0), Room(30 * MIB, 0, 0, 1, 0)]
    )
    ran: list[int] = []

    def run(level: int) -> str:
        ran.append(level)
        if len(ran) < 3:
            raise Refused
        return "ok"

    result = sizing.descend(
        6, _need(64, MIB, 8 * MIB), lambda: next(rooms), run, lambda e: isinstance(e, Refused)
    )
    assert result == ("ok", 2, 2, "cold") and ran == [0, 1, 2]

    def always(level: int) -> str:
        ran.append(level)
        raise Refused

    ran.clear()
    with pytest.raises(Refused):
        sizing.descend(2, _need(4, MIB), lambda: Room(1 << 40, 0, 0, 1, 0), always, lambda e: True)
    assert ran == [0, 1, 2]  # at most `levels` steps, then the finest raises

    with pytest.raises(ValueError):  # anything but a refusal runs once
        sizing.descend(2, _need(4, MIB), lambda: Room(1 << 40, 0, 0, 1, 0), _raise, lambda e: False)


def _raise(level: int) -> str:
    raise ValueError(level)


def test_a_group_takes_the_finest_level_any_gpu_needs() -> None:
    """P3/P4: every GPU crosses its collectives at one level, agreed before capture too."""
    assert sizing.agreed([1, 3, 2]) == 3 and sizing.agreed([]) == 0


def test_each_term_takes_whole_granules() -> None:
    assert sizing.rounded(1, GRAIN) == GRAIN and sizing.rounded(0, GRAIN) == 0
    assert sizing.peak(GRAIN, 1, (1, 1), (GRAIN + 1,)) == GRAIN + 2 * GRAIN


def test_an_ffn_counts_its_widest_phase_and_both_quantized_inputs() -> None:
    """The review's MEDs: the down projection's quantized input is n x inner wide, and a
    composite activation holds several n x inner temporaries."""
    gelu = Ffn(
        din=3072, dout=3072, inner=12288, projected=12288, temps=1, itemsize=2, quantized=False
    )
    need = gelu.need(tokens=4096)
    n = 1024
    assert need(n, 1) == 4096 * 3072 * 2 + 2 * n * 12288 * 2  # output + projection and activation
    composite = Ffn(3072, 3072, 12288, 12288, temps=3, itemsize=2, quantized=False).need(4096)
    assert composite(n, 1) - need(n, 1) == 2 * n * 12288 * 2
    quantized = Ffn(3072, 3072, 12288, 12288, temps=1, itemsize=2, quantized=True).need(4096)
    down = n * 12288 * 2 + (n * 12288 + 4 * n) + n * 3072 * 2  # act out, its int8 rows, result
    assert quantized(n, 1) == 4096 * 3072 * 2 + max(down, 2 * n * 12288 * 2)
    gated = Ffn(3072, 3072, 12288, 2 * 12288, temps=2, itemsize=2, quantized=False).need(4096)
    assert gated(n, 1) == 4096 * 3072 * 2 + n * 2 * 12288 * 2 + 2 * n * 12288 * 2
    assert [need(sizing.pieces(4096, level), 1) for level in range(4)] == sorted(
        (need(sizing.pieces(4096, level), 1) for level in range(4)), reverse=True
    )


def test_attention_counts_scores_for_math_and_whole_kv_copies_in_the_cut() -> None:
    flash = Attention(1, 4096, 24, 24, 4096, 128, 2, "flash", kv_copies=0)
    math = Attention(1, 4096, 24, 24, 4096, 128, 2, "math", kv_copies=0)
    out = 4096 * 24 * 128 * 2
    piece = 1024 * 24 * 128 * 2 + 24 * 1024 * 4
    assert flash.need(24, 1024, 1) == out + piece
    assert math.need(24, 1024, 1) == out + piece + 24 * 1024 * 4096 * 6
    copied = Attention(1, 4096, 24, 24, 4096, 128, 2, "flash", kv_copies=1)
    assert copied.need(24, 1024, 1) - flash.need(24, 1024, 1) == 2 * 4096 * 24 * 128 * 2


def test_a_conv_band_counts_its_padded_input_copy_and_workspace() -> None:
    need = sizing.conv(
        out=64 * MIB,
        per_unit_in=MIB,
        per_unit_out=2 * MIB,
        halo=MIB // 2,
        workspace=4 * MIB,
    )
    assert need(4, 1) == 64 * MIB + 4 * MIB + MIB // 2 + 8 * MIB + 4 * MIB
    view = sizing.conv(64 * MIB, 0, 2 * MIB, MIB // 2, 0)
    assert view(4, 1) == 64 * MIB + 8 * MIB


def test_a_norm_holds_its_output_its_piece_and_its_intermediate() -> None:
    need = sizing.norm(rows=8192, row=3072, itemsize=2, intermediate=4)
    assert need(1024, 1) == 8192 * 3072 * 2 + 1024 * 3072 * (2 + 4)


def test_a_split_measured_at_two_windows_scales_a_quadratic_op() -> None:
    """Measured at the finest window and the next, never extrapolated from one point: an op
    quadratic in its window (attention inside it) is planned as one."""

    def quadratic(w: int) -> int:
        return 3 * w + w * w // 10

    fit = sizing.fitted((100, quadratic(100)), (200, quadratic(200)))
    assert fit(1000) == quadratic(1000)
    linear = sizing.fitted((100, 500), (200, 1000))
    assert linear(1000) == 5000
    noisy = sizing.fitted((100, 520), (200, 1000))  # a concave reading: the line through 200
    assert noisy(1000) == 5000
