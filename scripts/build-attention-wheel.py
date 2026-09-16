#!/usr/bin/env python3
"""Package an already-built kernel-builder snapshot as an ordinary dependency wheel.

No download or compiler runs here. Wheel requirements must describe the complete
upstream runtime dependencies, including the exact Torch ABI used to build it.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json
import re
import zipfile
from pathlib import Path

from packaging.requirements import Requirement
from packaging.tags import parse_tag
from packaging.version import Version


def build(args: argparse.Namespace) -> Path:
    name = "cozy-kernel-" + args.kernel
    package = name.replace("-", "_")
    version = str(Version(args.version))
    if not re.fullmatch(r"[a-fA-F0-9]{40}|[a-fA-F0-9]{64}", args.revision):
        raise ValueError("revision must be the full immutable upstream commit")
    if any("\n" in value or "\r" in value for value in (args.source_url, args.license)):
        raise ValueError("metadata values cannot contain newlines")
    tags = parse_tag(args.tag)
    if len(tags) != 1 or any(t.interpreter != "cp312" or t.abi != "cp312" for t in tags):
        raise ValueError("use an exact cp312-cp312 platform tag for these Torch kernels")
    requirements = [Requirement(value) for value in args.requirement]
    torch = [value for value in requirements if value.name == "torch"]
    if len(torch) != 1 or not re.fullmatch(r"==[^*,]+", str(torch[0].specifier)):
        raise ValueError("declare the exact build Torch version with --requirement torch==VERSION")
    if not any(value.name == "kernels" for value in requirements):
        raise ValueError("declare the kernels loader dependency")
    snapshot = args.snapshot.resolve()
    if not list(snapshot.glob("build/*/metadata.json")):
        raise ValueError("snapshot must contain build/<variant>/metadata.json")
    members: dict[str, bytes] = {}
    for path in sorted(snapshot.rglob("*")):
        relative = path.relative_to(snapshot)
        if any(part.startswith(".") or part == "__pycache__" for part in relative.parts):
            continue
        if path.is_symlink() and path.is_dir():
            raise ValueError(f"directory symlink is unsupported: {path}")
        if path.is_file() and path.suffix != ".pyc":
            members[f"{package}/{relative.as_posix()}"] = path.read_bytes()
    for path, data in members.items():
        if path.endswith("/metadata.json") and "/build/" in path:
            json.loads(data)  # Refuse an invalid snapshot before emitting a wheel.
    info = f"{package}-{version}.dist-info"
    metadata = [
        "Metadata-Version: 2.4",
        f"Name: {name}",
        f"Version: {version}",
        "Requires-Python: >=3.12,<3.13",
        f"License-Expression: {args.license}",
        f"Project-URL: Source, {args.source_url.rstrip('/')}/commit/{args.revision}",
        *(f"Requires-Dist: {requirement}" for requirement in requirements),
        "",
        "",
    ]
    members[f"{info}/METADATA"] = "\n".join(metadata).encode()
    members[f"{info}/WHEEL"] = (
        f"Wheel-Version: 1.0\nGenerator: cozy-runtime\nRoot-Is-Purelib: false\nTag: {args.tag}\n"
    ).encode()
    record = io.StringIO(newline="")
    writer = csv.writer(record, lineterminator="\n")
    for path, data in sorted(members.items()):
        digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
        writer.writerow((path, "sha256=" + digest, len(data)))
    writer.writerow((f"{info}/RECORD", "", ""))
    members[f"{info}/RECORD"] = record.getvalue().encode()
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    wheel = output / f"{package}-{version}-{args.tag}.whl"
    with zipfile.ZipFile(wheel, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path, data in sorted(members.items()):
            entry = zipfile.ZipInfo(path, date_time=(1980, 1, 1, 0, 0, 0))
            entry.external_attr = 0o100644 << 16
            entry.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(entry, data)
    return wheel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--kernel", choices=("flash-attn3", "flash-attn4"), required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--source-url", required=True)
    parser.add_argument("--license", required=True)
    parser.add_argument("--requirement", action="append", default=[])
    parser.add_argument("--tag", default="cp312-cp312-linux_x86_64")
    parser.add_argument("--out", type=Path, default=Path("dist"))
    args = parser.parse_args()
    print(build(args))


if __name__ == "__main__":
    main()
