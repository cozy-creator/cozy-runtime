"""Seed a tiny real TensorFS repository; no placement or preparation reply is invented."""

from __future__ import annotations

import hashlib
import io
import json
import struct
import sys
from pathlib import Path

import tensorfs


def main() -> None:
    root = Path(sys.argv[1]).resolve()
    store = tensorfs.Store.ensure(str(root))
    encoding = dict(tensorfs.seed_digests())["plain/1"]
    targets = {
        component: {
            "drop": [],
            "add": {
                "weight": {
                    "logical_dtype": "f32",
                    "shape": [2, 2],
                    "encoding": encoding,
                    "parts": {"value": {"dtype": "f32", "shape": [2, 2]}},
                }
            },
        }
        for component in ("alpha", "spare", "zeta")
    }
    writer = store.begin_derived(
        "sha256:" + hashlib.sha256(b"local-serving-seed/1").hexdigest(),
        1,
        {},
        targets,
        {"pipeline": {"kind": "add"}},
        [(component, "weight") for component in targets],
        1 << 20,
        work_fingerprint="sha256:" + hashlib.sha256(b"local-serving-fixture/1").hexdigest(),
    )
    for component, scale in (("alpha", 2.0), ("spare", 7.0), ("zeta", 1.0)):
        writer.add_part(
            component, "weight", "value", io.BytesIO(struct.pack("<4f", scale, 0, 0, scale))
        )
    writer.add_config("pipeline", io.BytesIO(b"{}"))
    receipt = writer.commit()
    manifest = "sha256:" + receipt["manifest"]["sha256"]
    length = receipt["manifest"]["length"]
    operation = store.begin_operation("local-serving-release", "proof", "ordered")
    operation.hold_manifest(manifest, length)
    operation.commit_release(None, "1.0.0", "bf16", manifest, length)
    held = store.manifest(manifest)
    assert held["header"] is not None
    header = tensorfs.parse_header(bytes(held["header"]))
    print(
        json.dumps(
            {
                "manifest": manifest,
                "length": length,
                "manifest_json": bytes(held["manifest"]).decode(),
                "header_components": list(header["components"]),
                "component_bytes": {component: 16 for component in header["components"]},
                "weight_bytes": 48,
            }
        )
    )


if __name__ == "__main__":
    main()
