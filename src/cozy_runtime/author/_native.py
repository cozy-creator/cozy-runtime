"""What brings a native launch's operands back to their device first (`kernels.readable`): a
storage the memory ladder took out has no bytes behind its pointer until then. Runtime's memory
registers here; the author surface only calls."""

from __future__ import annotations

from collections.abc import Callable

NATIVE_READS: list[Callable[[tuple[object, ...]], None]] = []
