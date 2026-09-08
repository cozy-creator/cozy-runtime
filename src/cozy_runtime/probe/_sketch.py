"""Fixed-size activation evidence per (tap, step): a JL sketch plus exact scalars.

An ordinary render retains no full activation tensors. Each (tap, step) keeps KILOBYTES:
a CountSketch random projection of the flattened
activation — linear, so the sketch of a DIFF is the diff of the sketches, and rel-L2 and
cosine come out within ~sqrt(2/dim) of exact — beside exact per-run scalars (mean/std,
max-abs, outlier fraction, NaN/Inf counts). Sketches accumulate in fp32 whatever dtype the
lane runs in; scalar sums accumulate in fp64. Non-finite elements are counted exactly and
enter the sketch as ZERO: a NaN reaches the report as a count, never as poisoned geometry.

Projection identity is seeded from (tap, firing, numel, dim), so both lanes of one probe
project through the SAME buckets and signs — which is what makes their sketches
comparable. A module fired K times in one step folds K seed-distinct projections into one
sum: exactly the sketch of the concatenated firings.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any

import msgspec

_SEED_DOMAIN = b"cozy.probe.sketch/1"


class ProbeConfig(msgspec.Struct, frozen=True):
    """Mechanism knobs — resolution only, never thresholds (policy is the gate's)."""

    sketch_dim: int = 4096
    outlier_sigma: float = 6.0


class Observation(msgspec.Struct):
    """One lane's finalized evidence for one (tap, step)."""

    sketch: Any  # fp32 CPU tensor of sketch_dim
    shapes: tuple[tuple[int, ...], ...]  # one per firing, in firing order
    numel: int
    finite: int
    nan: int
    inf: int
    mean: float
    std: float
    max_abs: float
    outlier_frac: float


class TapStepAccumulator:
    """Folds every firing of one tap within one step; finalize() emits the Observation."""

    def __init__(self, config: ProbeConfig) -> None:
        self._config = config
        self.sketch: Any = None
        self.shapes: list[tuple[int, ...]] = []
        self.numel = 0
        self.count = 0
        self.total = 0.0
        self.sumsq = 0.0
        self.max_abs = 0.0
        self.outliers = 0
        self.nan = 0
        self.inf = 0

    def fold(self, tap: str, tensor: Any) -> None:
        import torch

        firing = len(self.shapes)
        self.shapes.append(tuple(int(extent) for extent in tensor.shape))
        flat = tensor.detach().reshape(-1)
        if self.sketch is None:
            self.sketch = torch.zeros(
                self._config.sketch_dim, dtype=torch.float32, device=flat.device
            )
        n = int(flat.numel())
        self.numel += n
        if not n:
            return
        values = flat.to(torch.float32)
        finite = torch.isfinite(values)
        nan = int(torch.isnan(values).sum())
        inf = int(torch.isinf(values).sum())
        self.nan += nan
        self.inf += inf
        if nan or inf:
            values = torch.where(finite, values, values.new_zeros(()))
        self._fold_scalars(values[finite] if (nan or inf) else values)
        buckets, signs = _projection(tap, firing, n, self._config.sketch_dim, values.device)
        self.sketch.index_add_(0, buckets, values * signs)

    def _fold_scalars(self, finite_values: Any) -> None:
        import torch

        count = int(finite_values.numel())
        if not count:
            return
        wide = finite_values.double()
        total = float(wide.sum())
        sumsq = float(torch.dot(wide, wide))
        self.count += count
        self.total += total
        self.sumsq += sumsq
        self.max_abs = max(self.max_abs, float(wide.abs().max()))
        mean = total / count
        deviation = math.sqrt(max(sumsq / count - mean * mean, 0.0))
        if deviation > 0.0:
            limit = self._config.outlier_sigma * deviation
            self.outliers += int((torch.abs(wide - mean) > limit).sum())

    def finalize(self) -> Observation:
        import torch

        sketch = (
            self.sketch.detach().to("cpu", copy=True)
            if self.sketch is not None
            else torch.zeros(self._config.sketch_dim, dtype=torch.float32)
        )
        mean = self.total / self.count if self.count else 0.0
        deviation = (
            math.sqrt(max(self.sumsq / self.count - mean * mean, 0.0)) if self.count else 0.0
        )
        return Observation(
            sketch=sketch,
            shapes=tuple(self.shapes),
            numel=self.numel,
            finite=self.count,
            nan=self.nan,
            inf=self.inf,
            mean=mean,
            std=deviation,
            max_abs=self.max_abs,
            outlier_frac=self.outliers / self.count if self.count else 0.0,
        )


def _projection(tap: str, firing: int, numel: int, dim: int, device: Any) -> tuple[Any, Any]:
    """The (buckets, signs) of one firing's CountSketch, deterministic on any host: drawn
    from a CPU generator seeded by (tap, firing, numel, dim), then moved to the lane."""
    import torch

    preimage = b"\0".join((_SEED_DOMAIN, tap.encode(), b"%d" % firing, b"%d" % numel, b"%d" % dim))
    seed = int.from_bytes(hashlib.blake2b(preimage, digest_size=8).digest(), "big")
    generator = torch.Generator()
    generator.manual_seed(seed & 0x7FFF_FFFF_FFFF_FFFF)
    buckets = torch.randint(0, dim, (numel,), generator=generator)
    signs = torch.randint(0, 2, (numel,), generator=generator, dtype=torch.float32) * 2.0 - 1.0
    if str(device) != "cpu":
        return buckets.to(device), signs.to(device)
    return buckets, signs
