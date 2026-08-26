"""cr-006 — the selected-encoding compute plane: which module executes which stored bytes.

cr-005 moves bytes that already ARE the destination's dtype. This package is the other
case: a checkpoint whose per-tensor encoding says the stored bytes are a CODE for the
logical tensor, and something has to turn the one into the other.

It was ONE 1,500-line module until #549.10 split it at its three real seams. The seams are
where the file's own sections already were, and each one now imports only the ones before it:

* `formats.py` — the refusal vocabulary, the registry lookups, the reviewed spec digests,
  the role parts a header declares, and every provider whose answer is a FLOAT TENSOR.
* `leaves.py` — the providers whose answer is a REPLACED MODULE (the `encoded_gemm` route),
  the activation quantizers they dispatch through, and the swizzle their scale grids need.
* `selection.py` — device and runtime identity, capability records, admission, and the
  launch table that maps reviewed digests to implementations.

The table stays STATIC and there is no plugin framework: a provider is a class in one of
these files, offered by `launch_providers`, and the only thing that admits it is a
capability record a probe minted. Nothing registers at import, nothing is discovered from
the environment, and there is no entry-point group. `probe.py` is the fourth part of the
same story and stays where it is, because it is the thing that MEASURES rather than the
thing that decides.

This module re-exports the whole surface, so every existing
`from cozy_runtime.internal.encoding import X` keeps working and the split cost its
consumers nothing.
"""

from __future__ import annotations

from cozy_runtime.internal.encoding.formats import (
    E4M3_MAX,
    E8M0_BIAS,
    E8M0_NAN,
    MX_BLOCK,
    REFUSALS,
    ROW_SCALE_FLOOR,
    SATURATION_MARGIN,
    SPEC_MXFP8,
    SPEC_PLAIN,
    SPEC_ROWWISE,
    SPEC_ROWWISE_KEEPDIM,
    SPEC_SCALED_SCALAR,
    SPEC_SCALED_SCALAR_WEIGHT_ONLY,
    SPEC_VECTORS,
    TORCH_DTYPES,
    Encoded,
    EncodingRefusal,
    LeafProvider,
    MicroScaledDequant,
    Provider,
    RolePart,
    RowwiseDequant,
    ScaledScalarDequant,
    Verbatim,
    aliases,
    digests_of,
    refuse,
    registry_digests,
)
from cozy_runtime.internal.encoding.leaves import (
    ACT_SCALE_FLOOR,
    MicroScaledNativeLeaf,
    RowwiseNativeLeaf,
    quantize_activation_mx,
    quantize_activation_rowwise,
    to_blocked,
)
from cozy_runtime.internal.encoding.selection import (
    IDENTITY_SOURCES,
    NAMED_SEAMS,
    OBJECTIVE_AXIS,
    Capabilities,
    CapabilityRecord,
    DeviceFacts,
    RuntimeIdentity,
    Selection,
    Selector,
    SpecUnreviewed,
    TensorCandidates,
    device_line,
    geometry_class,
    implementation_digest,
    launch_providers,
    measure_device,
    measure_runtime,
    runtime_source_digest,
)

__all__ = [
    "ACT_SCALE_FLOOR",
    "E4M3_MAX",
    "E8M0_BIAS",
    "E8M0_NAN",
    "IDENTITY_SOURCES",
    "MX_BLOCK",
    "NAMED_SEAMS",
    "OBJECTIVE_AXIS",
    "REFUSALS",
    "ROW_SCALE_FLOOR",
    "SATURATION_MARGIN",
    "SPEC_MXFP8",
    "SPEC_PLAIN",
    "SPEC_ROWWISE",
    "SPEC_ROWWISE_KEEPDIM",
    "SPEC_SCALED_SCALAR",
    "SPEC_SCALED_SCALAR_WEIGHT_ONLY",
    "SPEC_VECTORS",
    "TORCH_DTYPES",
    "Capabilities",
    "CapabilityRecord",
    "DeviceFacts",
    "Encoded",
    "EncodingRefusal",
    "LeafProvider",
    "MicroScaledDequant",
    "MicroScaledNativeLeaf",
    "Provider",
    "RolePart",
    "RowwiseDequant",
    "RowwiseNativeLeaf",
    "RuntimeIdentity",
    "ScaledScalarDequant",
    "Selection",
    "Selector",
    "SpecUnreviewed",
    "TensorCandidates",
    "Verbatim",
    "aliases",
    "device_line",
    "digests_of",
    "geometry_class",
    "implementation_digest",
    "launch_providers",
    "measure_device",
    "measure_runtime",
    "quantize_activation_mx",
    "quantize_activation_rowwise",
    "refuse",
    "registry_digests",
    "runtime_source_digest",
    "to_blocked",
]
