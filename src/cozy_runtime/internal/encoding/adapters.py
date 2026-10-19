"""Schemes decoded through the generic cores (`internal/decode_cores.py`): one `Adapter` row per
scheme maps its parts and params onto one core call, and `CoreDequant` serves every row in
`ADAPTERS` as a decode provider (on CUDA by the Triton kernels, elsewhere by their eager
reference).

Adding a scheme: one `Adapter` in `ADAPTERS`. Its params and parts are the spec's (cozytensors/1
§6); its `sample` builds random stored parts (at `probe_params`) the probe decodes on the device
against the eager reference, bit for bit, before the provider is admitted (`probe.py`, suite
`cores-decode`).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import ModuleType

from cozy_runtime.internal.decode_cores import HAS_TRITON, Cores
from cozy_runtime.internal.encoding.formats import Params, Tensor, _params


def matrix_rule(shape: Sequence[int], params: Mapping[str, object]) -> str:
    """The rank-2-or-more view every non-tensorwise scheme takes (spec §6 shapes)."""
    return "" if len(shape) >= 2 else f"the scheme views rank 2 or more, not {tuple(shape)}"


@dataclass(frozen=True, slots=True)
class Adapter:
    """One scheme onto the cores."""

    scheme: str
    required: frozenset[str]
    optional: frozenset[str]
    #: every param the decode reads, with its allowed values (all required)
    params: Mapping[str, tuple[object, ...]]
    #: (cores, parts, params, logical shape, destination): the core call(s)
    decode: Callable[
        [Cores, Mapping[str, Tensor], Mapping[str, object], tuple[int, ...], Tensor], None
    ]
    #: the params the probe qualifies the provider at
    probe_params: Mapping[str, object]
    #: (torch, rows, cols, generator, params) -> random stored parts of an [R, C] tensor
    sample: Callable[[ModuleType, int, int, Tensor, Mapping[str, object]], dict[str, Tensor]]
    #: the shape rule beyond the params: "" or why not
    geometry: Callable[[Sequence[int], Mapping[str, object]], str] = matrix_rule


#: Every scheme the cores decode, one row each.
ADAPTERS: tuple[Adapter, ...] = (
    # fp8.blockwise, mxfp4
    # gguf.*
    # gptq.*, awq.*, ct.*
    # bnb.*
    # comfy.*
)


@dataclass(frozen=True, slots=True)
class CoreDequant:
    """A scheme -> float through its adapter's core call. On CUDA the Triton kernels run; on any
    other device their eager torch reference, the same arithmetic."""

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
        return _params(params, self.adapter.params) or self.adapter.geometry(shape, params)

    def decode(self, torch: ModuleType, parts: Mapping[str, Tensor], destination: Tensor) -> None:
        cores = Cores(torch, kernel=destination.is_cuda and HAS_TRITON)
        self.adapter.decode(cores, parts, dict(self.params), tuple(destination.shape), destination)
