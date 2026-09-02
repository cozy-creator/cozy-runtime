"""The code-side TensorRequirements passed to exact TensorFS fit.

One projection, two producers: `cozy-runtime derive` projects the meta-device derivation
(the PlacementSet's `Slot.components`), and the fill plane projects the serving
census the moment before a byte is reserved. Both hand `tensorfs.code_topology` the same
rows, so `tensorfs.fit` answers the same question at every clock. Nothing here matches.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Sequence
from typing import Any

from cozy_runtime.internal import canonical

#: The dtype-diet default (`internal/derive.py`): a row that names no dtype is bf16.
DEFAULT_DTYPE = "bfloat16"

FORMAT = "cozy.runtime.tensor-requirements/1"

#: One code destination: (component, key within the component, torch dtype name or None
#: for the default, logical shape).
Row = tuple[str, str, str | None, Sequence[int]]


def accepts(encoded_leaves: str) -> list[str] | str:
    """What stored encodings the code admits, from the class's one keyword.

    `accept`: any encoding the pinned registry knows — the leaf may be replaced by a native
    encoded op. `refuse`: the module must be handed a float weight, so a stored encoding is
    admitted exactly when this runtime decodes it to float (`decoded_float`) or stores it
    plain (`verbatim`). The set is a fact about the code — package plus runtime — and it is
    hardware-independent (D3): whether a device can run the decoder is rule (6)'s question.
    """
    if encoded_leaves == "accept":
        return "any"
    from cozy_runtime.internal.encoding import launch_providers

    admitted = sorted(
        digest
        for digest, providers in launch_providers().items()
        if any(getattr(p, "route", "") in ("verbatim", "decoded_float") for p in providers)
    )
    if not admitted:
        raise RuntimeError("this runtime registers no float route; no encoding is admissible")
    return admitted


def tensorfs_dtype(name: str | None) -> str:
    """torch's spelling (`bfloat16`) to TensorFS's closed enum (`bf16`); a TensorFS
    spelling passes through, and an unknown one refuses at the TensorFS boundary."""
    from cozy_runtime.internal.encoding import TORCH_DTYPES

    spelled = (name or DEFAULT_DTYPE).removeprefix("torch.")
    for stored, torch_name in TORCH_DTYPES.items():
        if torch_name == spelled:
            return stored
    return spelled


def document(rows: Iterable[Row], constructed: Sequence[str]) -> dict[str, Any]:
    """Canonical TensorRequirements meaning, sorted by `(component, key)`."""
    out: dict[str, dict[str, Any]] = {}
    for component, key, dtype, shape in sorted(rows, key=lambda r: (r[0], r[1])):
        row: dict[str, Any] = {"shape": list(shape)}
        if dtype is not None and dtype != DEFAULT_DTYPE:
            row["dtype"] = dtype
        out.setdefault(component, {})[key] = row
    return {"format": FORMAT, "components": out, "constructed": list(constructed)}


def canonical_bytes(
    rows: Iterable[Row], constructed: Sequence[str], *, encoded_leaves: str
) -> bytes:
    """Canonical TensorRequirements bytes for one construction. A row whose dtype is
    not the default is PINNED: the destination fixes it (an integer buffer, an fp32
    island, a whole-fp16 construction) and the checkpoint supplies it or a lossless
    widening (D4); the fill plane admits no other dtype either."""
    import tensorfs

    held = sorted(rows, key=lambda r: (r[0], r[1]))
    admitted = accepts(encoded_leaves)
    contract = "sha256:" + hashlib.sha256(canonical.write(document(held, constructed))).hexdigest()
    tensors = [
        tensorfs.CodeTensor(
            component=component,
            key=key,
            dtype=tensorfs_dtype(dtype),
            shape=list(shape),
            accepts=admitted,
            pinned=dtype is not None and dtype != DEFAULT_DTYPE,
        )
        for component, key, dtype, shape in held
    ]
    return bytes(tensorfs.code_topology(contract, tensors, list(constructed)))
