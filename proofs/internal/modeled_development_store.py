"""Build the tiny exact TensorFS Repository used by modeled source-checkout proofs."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any


def create(
    root: Path, *, config: bytes = b'{"width":4}', tensor: bytes = b"\x80\x3f" * 4
) -> dict[str, Any]:
    import tensorfs

    store = tensorfs.Store.init(str(root.resolve()))
    plain = dict(tensorfs.seed_digests())["plain/1"]
    transaction = "sha256:" + hashlib.sha256(str(root.resolve()).encode()).hexdigest()
    writer = store.begin_derived(
        transaction,
        1,
        {},
        {
            "core": {
                "drop": [],
                "add": {
                    "weight": {
                        "logical_dtype": "bf16",
                        "shape": [4],
                        "encoding": plain,
                        "parts": {"value": {"dtype": "bf16", "shape": [4]}},
                    }
                },
            }
        },
        {"model": {"media_type": "application/json", "length": len(config)}},
        [("core", "weight")],
        1024,
    )
    writer.add_part("core", "weight", "value", io.BytesIO(tensor))
    writer.add_config("model", io.BytesIO(config))
    receipt = writer.commit()
    manifest = "sha256:" + str(receipt["manifest"]["sha256"])
    manifest_length = int(receipt["manifest"]["length"])
    selected = store.manifest(manifest)
    closure = store.walk(manifest)
    operation = store.begin_operation("modeled-development-proof", "proof", "tiny-core", 1)
    operation.hold_manifest(manifest, manifest_length)
    for reached in closure:
        operation.hold_blob(str(reached["id"]), int(reached["length"]))
    operation.progress(
        2,
        len(closure) + 1,
        manifest_length + sum(int(reached["length"]) for reached in closure),
    )
    evidence = json.dumps(
        {"classification_digest": tensorfs.topology_digest(bytes(selected["header"]))},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    operation.commit_release(None, "1.0.0", "cpu", manifest, manifest_length, evidence)
    return {
        "config_digest": "sha256:" + hashlib.sha256(config).hexdigest(),
        "config_length": len(config),
        "manifest_digest": manifest,
        "manifest_length": manifest_length,
        "store": str(root.resolve()),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(create(args.root), sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
