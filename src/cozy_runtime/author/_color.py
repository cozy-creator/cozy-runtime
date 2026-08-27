"""The one normalized RGB/YUV color contract shared by decode and encode."""

from __future__ import annotations

from typing import Any


def normalize(
    *,
    width: int,
    height: int,
    primaries: int,
    transfer: int,
    matrix: int,
    color_range: int,
) -> tuple[int, int, int, int]:
    """Resolve unspecified tags to explicit BT.709-HD or ITU601-SD video facts."""
    hd = width > 1024 or height > 576
    if matrix == 6:  # SMPTE170M and ITU601 share the matrix in PyAV's swscale enum.
        matrix = 5
    if matrix not in (1, 4, 5, 7, 9):
        matrix = 1 if hd else 5
    if primaries in (0, 2):
        primaries = 1 if hd else 6
    if transfer in (0, 2):
        transfer = 1 if hd else 6
    if color_range not in (1, 2):
        color_range = 1
    return primaries, transfer, matrix, color_range


def colorspace(av: Any, value: int) -> Any:
    names = {1: "ITU709", 4: "FCC", 5: "ITU601", 7: "SMPTE240M", 9: "BT2020"}
    try:
        return av.video.reformatter.Colorspace[names[value]]
    except KeyError as exc:
        raise ValueError(f"unsupported video color matrix {value}") from exc


def range_value(av: Any, value: int) -> Any:
    names = {1: "MPEG", 2: "JPEG"}
    try:
        return av.video.reformatter.ColorRange[names[value]]
    except KeyError as exc:
        raise ValueError(f"unsupported video color range {value}") from exc
