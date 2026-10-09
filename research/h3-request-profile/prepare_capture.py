"""Make an explicit local profiling package from the baseline's exact wheel artifacts.

No network or inference. Normal Cozy CLI owns the subsequent capture and execution.
"""

from __future__ import annotations

import argparse
import email.parser
import hashlib
import json
import re
import shutil
import zipfile
from pathlib import Path


def prepare(destination: Path, wheels: dict[str, Path]) -> None:
    expected = {"minimax-h3": "1.26.3", "qwen-image-2": "0.3.0", "diffusers": "0.41.0.dev0"}
    receipts = {}
    for name, wheel in wheels.items():
        wheel = wheel.resolve(strict=True)
        with zipfile.ZipFile(wheel) as archive:
            entries = [n for n in archive.namelist() if n.endswith(".dist-info/METADATA")]
            if len(entries) != 1 or archive.getinfo(entries[0]).file_size > 1 << 20:
                raise ValueError(f"{name}: expected one bounded wheel metadata document")
            metadata = email.parser.BytesParser().parsebytes(archive.read(entries[0]))
        if (
            metadata["Name"].lower().replace("_", "-") != name
            or metadata["Version"] != expected[name]
        ):
            raise ValueError(f"{name}: wheel does not match the benchmark package cohort")
        receipts[name] = {
            "path": str(wheel),
            "version": expected[name],
            "sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
        }
    if set(receipts) != set(expected):
        raise ValueError("all three non-PyPI source wheels are required")
    source = Path(__file__).resolve().parent
    destination.mkdir(parents=True, exist_ok=False)
    for name in ("package.toml", "pyproject.toml"):
        shutil.copyfile(source / name, destination / name)
    shutil.copytree(
        source / "h3_request_profile",
        destination / "h3_request_profile",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    project = destination / "pyproject.toml"
    document = project.read_text()
    for name, receipt in receipts.items():
        replacement = name + " = {path=" + json.dumps(receipt["path"]) + "}"
        document, count = re.subn(
            r"^" + re.escape(name) + r" = .*?$",
            lambda _, replacement=replacement: replacement,
            document,
            count=1,
            flags=re.MULTILINE,
        )
        if count != 1:
            raise ValueError(f"{name}: profiling template lost its explicit source")
    project.write_text(document)
    (destination / "captured-wheel-provenance.json").write_text(
        json.dumps(receipts, indent=2) + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--h3-wheel", required=True, type=Path)
    parser.add_argument("--qwen-wheel", required=True, type=Path)
    parser.add_argument("--diffusers-wheel", required=True, type=Path)
    args = parser.parse_args()
    prepare(
        args.out,
        {
            "minimax-h3": args.h3_wheel,
            "qwen-image-2": args.qwen_wheel,
            "diffusers": args.diffusers_wheel,
        },
    )
