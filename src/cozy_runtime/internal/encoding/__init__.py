"""Which implementation executes which stored bytes, in three seams, each importing only the
ones before it: `formats.py` (the stored contracts and the decoding providers), `leaves.py`
(the providers that replace a module) and `selection.py` (device identity, admission by rule
and the launch table).
"""

from __future__ import annotations

from cozy_runtime.internal.encoding.formats import (
    E4M3_MAX,
    PLAIN,
    REFUSALS,
    ROW_SCALE_FLOOR,
    TORCH_DTYPES,
    Encoded,
    EncodingRefusal,
    Int8ConvRotDequant,
    LeafProvider,
    Nvfp4Dequant,
    Params,
    Provider,
    RolePart,
    RowwiseDequant,
    TensorwiseDequant,
    Verbatim,
    params_of,
    refuse,
)
from cozy_runtime.internal.encoding.leaves import (
    Int8ConvRotNativeLeaf,
    Nvfp4NativeLeaf,
    RowwiseNativeLeaf,
)
from cozy_runtime.internal.encoding.selection import (
    FLOAT_DTYPES,
    NAMED_SEAMS,
    DeviceFacts,
    Refusal,
    RuntimeIdentity,
    Selection,
    Selector,
    TensorCandidates,
    admit,
    device_line,
    implementation_digest,
    launch_providers,
    measure_device,
    measure_runtime,
)

__all__ = [
    "E4M3_MAX",
    "FLOAT_DTYPES",
    "NAMED_SEAMS",
    "PLAIN",
    "REFUSALS",
    "ROW_SCALE_FLOOR",
    "TORCH_DTYPES",
    "DeviceFacts",
    "Encoded",
    "EncodingRefusal",
    "Int8ConvRotDequant",
    "Int8ConvRotNativeLeaf",
    "LeafProvider",
    "Nvfp4Dequant",
    "Nvfp4NativeLeaf",
    "Params",
    "Provider",
    "Refusal",
    "RolePart",
    "RowwiseDequant",
    "RowwiseNativeLeaf",
    "RuntimeIdentity",
    "Selection",
    "Selector",
    "TensorCandidates",
    "TensorwiseDequant",
    "Verbatim",
    "admit",
    "device_line",
    "implementation_digest",
    "launch_providers",
    "measure_device",
    "measure_runtime",
    "params_of",
    "refuse",
]
