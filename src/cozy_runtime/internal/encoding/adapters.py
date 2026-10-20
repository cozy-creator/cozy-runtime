"""`CoreDequant`: every row of `decode_schemes.ADAPTERS` (one per scheme the decode cores serve) as
a decode provider, on CUDA by the Triton kernels and elsewhere by their eager reference. The fill
plane checks a tensor's parts by the same row (`formats.part_rule`), and the probe qualifies each
row with its own oracle (`probe.py`, suite `cores-decode`): random parts at the row's
`probe_params`, kernel against eager, bit for bit, plus the scheme's spec vectors.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import ModuleType

from cozy_runtime.internal.decode_cores import HAS_TRITON, Cores
from cozy_runtime.internal.decode_schemes import ADAPTERS, Adapter, geometry_rule, params_rule, run
from cozy_runtime.internal.encoding.formats import Params, Tensor


@dataclass(frozen=True, slots=True)
class CoreDequant:
    """A scheme -> float through its row's core calls."""

    encoding: str = ""
    params: Params = ()
    adapters: tuple[Adapter, ...] = ADAPTERS

    @property
    def adapter(self) -> Adapter:
        return next(a for a in self.adapters if a.scheme == self.encoding)

    @property
    def name(self) -> str:
        return f"cozy.{self.encoding}.cores/1"

    @property
    def route(self) -> str:
        return "decoded_float"

    @property
    def device_predicate(self) -> str:
        return "any"

    @property
    def schemes(self) -> frozenset[str]:
        return frozenset(a.scheme for a in self.adapters)

    @property
    def code_revision(self) -> str:
        return "1"

    @property
    def suite(self) -> str:
        return "cores-decode"

    @property
    def required(self) -> frozenset[str]:
        return self.adapter.required

    @property
    def optional(self) -> frozenset[str]:
        return self.adapter.optional

    def supports(self, shape: Sequence[int], params: Mapping[str, object]) -> str:
        return params_rule(self.adapter, params) or geometry_rule(self.adapter, shape, params)

    def decode(self, torch: ModuleType, parts: Mapping[str, Tensor], destination: Tensor) -> None:
        cores = Cores(torch, kernel=destination.is_cuda and HAS_TRITON)
        run(self.adapter, cores, parts, dict(self.params), destination)
