"""Freeze this research fixture to one reviewed candidate wheel, without installing it."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from email.parser import BytesParser
from pathlib import Path

FILES = (
    "internal/attention.py",
    "internal/attention_producer.py",
    "internal/attention_sol.py",
    "internal/attention_upstream.py",
    "internal/attention_ulysses.py",
    "internal/kernel_sources.py",
    "_kernels/_C.abi3.so",
)


def freeze(wheel: Path, expected_sha256: str, expected_version: str) -> dict[str, object]:
    digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
    if digest != expected_sha256:
        raise ValueError("Reviewed candidate wheel digest does not match")
    with zipfile.ZipFile(wheel) as archive:
        metadata_names = [
            name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
        ]
        if len(metadata_names) != 1:
            raise ValueError("Expected exactly one wheel distribution")
        metadata = BytesParser().parsebytes(archive.read(metadata_names[0]))
        if metadata["Name"].replace("_", "-") != "cozy-runtime":
            raise ValueError("Expected a Runtime wheel")
        if metadata["Version"] != expected_version:
            raise ValueError("Reviewed candidate Runtime version does not match")
        sources = {
            name: hashlib.sha256(archive.read("cozy_runtime/" + name)).hexdigest() for name in FILES
        }
    return {
        "runtime_version": expected_version,
        "tensorfs_version": "0.6.1",
        "wheel_sha256": digest,
        "source_sha256": sources,
        "status": "Research fixture frozen to reviewed candidate; no GPU qualification implied.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--expected-version", required=True)
    args = parser.parse_args()
    result = freeze(args.wheel, args.expected_sha256, args.expected_version)
    destination = Path(__file__).parent / "sage2_mode_partition_probe/cohort.json"
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"path": str(destination), **result}, indent=2))
