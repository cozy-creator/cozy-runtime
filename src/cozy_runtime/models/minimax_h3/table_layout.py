"""The checkpoint's ordered AdaLN row labels, independent of a sampling preset.

Canonical source: cozy_runtime.models.minimax_h3.table_layout. The producer vendors
these exact bytes as h3_tables._table_layout; scripts/sync-h3-table-layout.py checks the copy.
"""

from __future__ import annotations

import math
import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import NoReturn

from cozy_runtime.author import ConformanceError

_MODALITIES = ("video", "text", "audio")


def _refuse(message: str) -> NoReturn:
    raise ConformanceError(
        f"model modulation tables {message}", code="artifact_config", fields=["table_keys"]
    )


def _timestep(value: object) -> float:
    if not isinstance(value, str):
        _refuse("need exact float32 hexadecimal timestep labels")
    try:
        number = float.fromhex(value)
        rounded = struct.unpack("<f", struct.pack("<f", number))[0]
    except (ValueError, OverflowError, TypeError):
        _refuse("contain an invalid timestep label")
    if not math.isfinite(number) or number != rounded or number.hex() != value:
        _refuse("need finite, canonical float32 timestep labels")
    return number


@dataclass(frozen=True, slots=True)
class TableLayout:
    timesteps: tuple[float, ...]
    block_keys: tuple[tuple[int, int], ...]

    @classmethod
    def parse(cls, value: object) -> TableLayout:
        if not isinstance(value, Mapping) or set(value) != {
            "final_normalization",
            "block_modulation",
        }:
            _refuse("must include their final-normalization and block-modulation row labels")
        final = value["final_normalization"]
        blocks = value["block_modulation"]
        if not isinstance(final, list) or not final or not isinstance(blocks, list) or not blocks:
            _refuse("must contain nonempty row-label lists")
        timesteps: list[float] = []
        rows: dict[float, int] = {}
        for index, row in enumerate(final):
            if (
                not isinstance(row, Mapping)
                or set(row) != {"index", "timestep"}
                or type(row["index"]) is not int
                or row["index"] != index
            ):
                _refuse("need contiguous final-normalization row indexes starting at zero")
            timestep = _timestep(row["timestep"])
            if timestep in rows:
                _refuse("contain duplicate final-normalization timestep labels")
            rows[timestep] = index
            timesteps.append(timestep)
        keys: list[tuple[int, int]] = []
        seen: set[tuple[int, int]] = set()
        for index, row in enumerate(blocks):
            if (
                not isinstance(row, Mapping)
                or set(row) != {"index", "timestep", "modality", "modality_tag"}
                or type(row["index"]) is not int
                or row["index"] != index
            ):
                _refuse("need contiguous block-modulation row indexes starting at zero")
            tag = row["modality_tag"]
            if type(tag) is not int or tag not in range(3) or row["modality"] != _MODALITIES[tag]:
                _refuse("must label modalities as video=0, text=1, or audio=2")
            timestep = _timestep(row["timestep"])
            if timestep not in rows:
                _refuse("reference a timestep absent from their final-normalization rows")
            key = (rows[timestep], tag)
            if key in seen:
                _refuse("contain a duplicate timestep/modality pair")
            seen.add(key)
            keys.append(key)
        return cls(tuple(timesteps), tuple(keys))

    def require(self, video: Sequence[float], audio: Sequence[float]) -> None:
        """Check the requested schedule before its first transformer evaluation.

        Conditional rows are checked against the actual packed tokens by the transformer's
        existing pre-hook; a text-only call need not carry unused reference rows.
        """
        available = {(self.timesteps[row], tag) for row, tag in self.block_keys}
        for values, tags in ((video, (0, 1)), (audio, (2,))):
            for timestep in values:
                for tag in tags:
                    if (timestep, tag) not in available:
                        _refuse(
                            f"do not cover the requested schedule: missing {_MODALITIES[tag]} "
                            f"timestep {timestep:g}. Select a covered step count or regenerate "
                            "the checkpoint's modulation tables for this schedule."
                        )
