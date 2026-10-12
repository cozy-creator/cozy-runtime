"""BiRefNet configuration for the standard native safetensors ingest path."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from tensorfs.derived import SourceInspection

REPOSITORY = "ZhengPeng7/BiRefNet"
REVISION = "e2bf8e4460fc8fa32bba5ea4d94b3233d367b0e4"
FILES = {
    "config.json": (405, "c97ea21569daf66b205491a4635147dd3bc42c7c168b89d7d75b53f67ef548ae"),
}


def preparation(
    inspection: SourceInspection, root: Path
) -> tuple[dict[str, Any], tuple[tuple[str, str], ...], dict[str, bytes]]:
    """Keep native weights unchanged and attach their model-owned constructor config."""
    raw = (root / "config.json").read_bytes()
    length, digest = FILES["config.json"]
    if len(raw) != length or hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError("BiRefNet metadata does not match its selected source revision")
    config = json.loads(raw)
    if config.get("architectures") != ["BiRefNet"] or config.get("bb_pretrained") is not False:
        raise ValueError("BiRefNet requires its complete checkpoint, without backbone downloads")
    census = json.loads(Path(__file__).with_name("source_tensors.json").read_text())
    tensors = inspection.components.get("model", {})
    if set(inspection.components) != {"model"} or set(tensors) != {row[0] for row in census}:
        raise ValueError("BiRefNet source tensor keys do not match its architecture")
    for key, shape, dtype in census:
        tensor = tensors[key]
        if list(tensor.shape) != shape or tensor.logical_dtype != dtype:
            raise ValueError(f"BiRefNet source tensor geometry differs: {key}")
    provenance = {
        "source_repository": REPOSITORY,
        "source_revision": REVISION,
        "license": "MIT",
        "conversion": (
            "Native CozyTensors storage; original tensor names, dtypes and values retained."
        ),
    }
    return (
        {"architecture": "BiRefNet", "bb_pretrained": False, "provenance": provenance},
        tuple(("model", row[0]) for row in census),
        {
            "SOURCE.json": (json.dumps(provenance, sort_keys=True) + "\n").encode(),
            "LICENSE": Path(__file__).with_name("LICENSE.model.txt").read_bytes(),
        },
    )
