"""A native receipt names its declaration inline (older Runtimes) or by digest; both read alike."""

from __future__ import annotations

import hashlib
import io
import struct
from pathlib import Path

import tensorfs

from cozy_runtime import canonical_json
from cozy_runtime.internal.encoding import SPEC_PLAIN
from cozy_runtime.internal.weights_sink import declaration_digest, same_receipt


def test_both_receipt_forms_name_the_same_declaration(tmp_path: Path) -> None:
    root = tmp_path / "store"
    tensorfs.Store.init(root)
    inline, digest = (tensorfs.Store.open(root, receipts=form) for form in ("inline", "digest"))
    keys = [f"block.{index:04d}.weight" for index in range(64)]
    tensor = {
        "logical_dtype": "f16",
        "shape": [4],
        "encoding": SPEC_PLAIN,
        "parts": {"value": {"dtype": "f16", "shape": [4]}},
    }
    targets = {"unet": {"drop": [], "add": dict.fromkeys(keys, tensor)}}
    order = [("unet", key) for key in keys]
    fingerprint = "sha256:" + "81" * 32
    declared = inline.derived_declaration(
        {}, targets, {}, order, 8 * len(keys), work_fingerprint=fingerprint
    )
    transaction = "sha256:" + "82" * 32
    writer = digest.begin_derived(
        transaction, 1, {}, targets, {}, order, 8 * len(keys), work_fingerprint=fingerprint
    )
    for index, key in enumerate(keys):
        writer.add_part("unet", key, "value", io.BytesIO(struct.pack("<4e", index, 1, 2, 3)))
    committed = writer.commit()
    old, new = (store.derived_lookup(transaction)["receipt"] for store in (inline, digest))
    assert committed == new and "declaration" in old and "declaration" not in new
    assert new["declaration_length"] == len(declared)
    assert declaration_digest(old) == declaration_digest(new) == hashlib.sha256(declared).digest()
    assert same_receipt(old, new) and same_receipt(new, old)
    assert len(canonical_json.encode(new)) < len(canonical_json.encode(old)) - len(declared) // 2
    changed = dict(old, manifest={**old["manifest"], "length": old["manifest"]["length"] + 1})
    assert not same_receipt(changed, new)
