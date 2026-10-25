"""Decode providers for the shared scheme rows. NVFP4 uses TensorFS on supported targets;
other rows and unsupported targets retain the existing cores. The fill plane checks each
tensor's parts through the same row (`formats.part_rule`).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import cache
from types import ModuleType

from tensorfs import UnsupportedError, _ext, raw

from cozy_runtime.internal.decode_cores import HAS_TRITON, Cores
from cozy_runtime.internal.decode_schemes import ADAPTERS, Adapter, geometry_rule, params_rule, run
from cozy_runtime.internal.encoding.comfy import _decoder_identity, _descriptor
from cozy_runtime.internal.encoding.formats import Params, Tensor


@cache
def _nvfp4_kernel(family: int, dtype: str) -> bool:
    """Use the installed decoder's catalog, including its metadata conversion stage."""
    try:
        for stage in ("codes", "cast_metadata"):
            _ext._kernel_binary(family, stage, dtype)
    except UnsupportedError:
        return False
    return True


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
        if self.encoding == "nvfp4":
            return f"2/{_decoder_identity()}"
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
        if self.encoding == "nvfp4":
            dtype, shape, _ = _descriptor(destination)
            device = destination.device.type
            family = 0
            if device == "cuda":
                major, minor = torch.cuda.get_device_capability(destination.device)
                family = major * 10 + minor
            if device == "cpu" or (device == "cuda" and _nvfp4_kernel(family, dtype)):
                raw.decode_into(
                    self.encoding,
                    {
                        role: ("F4", shape, part) if role == "weight" else _descriptor(part)
                        for role, part in parts.items()
                    },
                    destination,
                    dtype=dtype,
                    shape=shape,
                    params=dict(self.params),
                )
                return
        cores = Cores(torch, kernel=destination.is_cuda and HAS_TRITON)
        run(self.adapter, cores, parts, dict(self.params), destination)
