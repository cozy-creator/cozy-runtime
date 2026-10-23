"""Comfy serving through TensorFS's shared decoder, using already resident parts.

This provider owns routing and pure metadata preflight. TensorFS owns format
validation, arithmetic, device programs, completion and the borrowed buffer ABI.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from types import ModuleType

from tensorfs import _ext, raw

from cozy_runtime.author._loader import TENSORFS_DTYPES
from cozy_runtime.internal import canonical
from cozy_runtime.internal.encoding.formats import TORCH_DTYPES, Params, Tensor

SCHEMES = frozenset({"comfy.asym_w4a8_int8", "w4a8.convrot", "comfy.w6a8_int8", "int4.convrot"})
_ROTATIONS = frozenset(4**n for n in range(1, 9))
_FORMAT_DTYPES = {
    TORCH_DTYPES[value]: key for key, value in TENSORFS_DTYPES.items() if value in TORCH_DTYPES
}


@cache
def _decoder_identity() -> str:
    """The actual native decoder/catalog/marshal and its installed public adapter.

    A package version alone does not identify arithmetic in a development wheel.
    This joins the existing private capability key, without another inventory.
    """
    digests = {}
    for name, module in (("native", _ext), ("adapter", raw)):
        location = module.__file__
        if not isinstance(location, str):
            raise RuntimeError(f"TensorFS {name} has no installed implementation file")
        with Path(location).open("rb") as source:
            digests[name] = hashlib.file_digest(source, "sha256").hexdigest()
    return canonical.digest(digests)


def _descriptor(tensor: Tensor) -> tuple[str, list[int], Tensor]:
    name = str(tensor.dtype).removeprefix("torch.")
    dtype = _FORMAT_DTYPES.get(name)
    if dtype is None:
        raise ValueError(f"no TensorFS format dtype for resident {name}")
    return dtype, [int(n) for n in tensor.shape], tensor


@dataclass(frozen=True, slots=True)
class ComfyDequant:
    """Three reviewed Comfy schemes decoded directly from their resident role buffers."""

    encoding: str = ""
    params: Params = ()

    @property
    def name(self) -> str:
        return f"tensorfs.{self.encoding}.decode/1"

    @property
    def route(self) -> str:
        return "decoded_float"

    @property
    def device_predicate(self) -> str:
        return "any"

    @property
    def schemes(self) -> frozenset[str]:
        return SCHEMES

    @property
    def code_revision(self) -> str:
        return f"1/{_decoder_identity()}"

    @property
    def suite(self) -> str:
        return "comfy-decode"

    @property
    def required(self) -> frozenset[str]:
        if self.encoding == "int4.convrot":
            return frozenset({"weight", "scale"})
        return frozenset({"weight", "group_scale", "scale"})

    @property
    def optional(self) -> frozenset[str]:
        if self.encoding == "comfy.asym_w4a8_int8":
            return frozenset({"codebook", "correction"})
        if self.encoding == "w4a8.convrot":
            return frozenset({"codebook"})
        return frozenset()

    def supports(self, shape: Sequence[int], params: Mapping[str, object]) -> str:
        if self.encoding not in SCHEMES:
            return "unknown Comfy scheme"
        grouped = self.encoding != "int4.convrot"
        wanted = (
            {"rotation", "rotation_size", "group_size"}
            if grouped
            else {"rotation", "rotation_size"}
        )
        if set(params) != wanted:
            return f"{self.encoding}: params must name exactly {sorted(wanted)}"
        if params.get("rotation") != "kron_regular_h4":
            return "rotation must be kron_regular_h4"
        rotation = params.get("rotation_size")
        if not isinstance(rotation, int) or type(rotation) is not int or rotation not in _ROTATIONS:
            return "rotation_size must be a supported power of four"
        if len(shape) < 2 or any(type(n) is not int or n < 0 for n in shape):
            return "Comfy decoding needs a nonnegative matrix or stack of matrices"
        columns = shape[-1]
        if columns % rotation:
            return "columns do not contain whole rotation blocks"
        if grouped:
            group = params.get("group_size")
            if not isinstance(group, int) or type(group) is not int:
                return "group_size must be an integer"
            if self.encoding == "comfy.asym_w4a8_int8":
                valid = group >= 4 and (16 % group == 0 or group % 16 == 0)
            elif self.encoding == "w4a8.convrot":
                valid = group == 16
            else:
                valid = group >= 16 and group % 16 == 0
            if not valid or columns % group:
                return "columns do not contain supported whole quantization groups"
        return ""

    def decode(self, torch: ModuleType, parts: Mapping[str, Tensor], destination: Tensor) -> None:
        if destination.device.type not in ("cpu", "cuda"):
            # TensorFS decodes on the host or on CUDA; any other device (MPS) decodes on the host.
            host = torch.empty_like(destination, device="cpu")
            self.decode(torch, {role: part.cpu() for role, part in parts.items()}, host)
            destination.copy_(host)
            return
        dtype, shape, _ = _descriptor(destination)
        raw.decode_into(
            self.encoding,
            {role: _descriptor(part) for role, part in parts.items()},
            destination,
            dtype=dtype,
            shape=shape,
            params=dict(self.params),
        )
