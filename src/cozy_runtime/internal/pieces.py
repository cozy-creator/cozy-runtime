"""The Runtime's side of `author.split`: how finely an op runs, from the peak bytes per input
element it measured in this process and what the device has free for it beside the weights its
model keeps whole (`Weights.activation_room`)."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING

from cozy_runtime.internal import accel, weight_policy

if TYPE_CHECKING:
    from cozy_runtime.internal.weights import Weights


class Pieces:
    def __init__(self, weights: Weights) -> None:
        self.weights = weights
        #: op key -> the most bytes one input element took at a piece's peak
        self.ratio: dict[str, float] = {}

    def room(self, key: str) -> tuple[float | None, int]:
        weights = self.weights
        if weights.open:  # it fits itself to the room it finds from now on
            weights.adaptive.add(weights.open[-1][1])
        free, movable = weights.activation_room(whole=True)
        return self.ratio.get(key), max(free + movable - weight_policy.MARGIN, 0)

    @contextmanager
    def piece(self, key: str, elements: int, need: int) -> Iterator[None]:
        weights = self.weights
        torch, kind = weights.torch, weights.kind
        accel.hold_peak(torch, kind)
        weights.free_for(need, whole=True)
        base = accel.allocated(torch, kind)
        try:
            yield
        except BaseException:
            accel.drop_peak(torch, kind)  # a piece that failed never showed what it needs
            raise
        peak = accel.section_peak(torch, kind) - base
        self.ratio[key] = max(self.ratio.get(key, 0.0), max(peak, 0) / max(elements, 1))

    def ran(self, name: str, pieces: int) -> None:
        if pieces > 1:
            self.weights.modes[f"{name}.pieces"] = str(pieces)
