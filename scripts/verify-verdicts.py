#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Release gate: every offline kernel fact a wheel ships binds to a kernel that wheel carries.

Verdicts and tables are measured on the manylinux x86_64 native wheel (`cli.qualify`). This
reads that wheel's RECORD and its own `qualification_tables/`, without importing CUDA, and fails
when a shipped `prebuilt-*` verdict names a binary or interface the wheel does not carry, or a
table measured an encoded GEMM against another `_gemm` build: a stale fact would otherwise be
silently unmatched and its card would lose its measured rules. Other platforms' wheels carry
other builds and serve their family defaults by design.
"""

from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import zipfile
from pathlib import Path

KERNELS = "cozy_runtime/_kernels/"
TABLES = "cozy_runtime/internal/qualification_tables/"


def module(kernel: str) -> str:
    return "_fa3" if kernel == "prebuilt-fa3" else "_gemm"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist", required=True, type=Path)
    wheel = next(parser.parse_args().dist.resolve().glob("*-manylinux_2_28_x86_64.whl"))
    with zipfile.ZipFile(wheel) as archive:
        record = next(n for n in archive.namelist() if n.endswith(".dist-info/RECORD"))
        digests = {
            row[0]: row[1].replace("=", ":", 1)
            for row in csv.reader(io.StringIO(archive.read(record).decode()))
            if row and row[1]
        }
        tables = {
            name: json.loads(archive.read(name))
            for name in archive.namelist()
            if name.startswith(TABLES) and name.endswith(".json")
        }
    binary = {
        name.removeprefix(KERNELS).split(".")[0]: digest
        for name, digest in digests.items()
        if name.startswith(KERNELS) and name.endswith(".so")
    }
    stub = {
        name.removeprefix(KERNELS).removesuffix(".pyi"): digest
        for name, digest in digests.items()
        if name.startswith(KERNELS) and name.endswith(".pyi")
    }
    stale = []
    for name, table in sorted(tables.items()):
        if name.removeprefix(TABLES).startswith("kernels-sm"):
            for row in table["verdicts"]:
                if not row["name"].startswith("prebuilt-"):
                    continue  # an environment's attention package, not this wheel's binary
                carried = (binary.get(module(row["name"])), stub.get(module(row["name"])))
                if (row["build"], row["abi"]) != carried:
                    stale.append(f"{name}: {row['name']} measured {row['build']} {row['abi']}")
        else:
            measured = dict(table["runtime"]["native_backends"]).get("_gemm", "")
            raw = bytes.fromhex(measured.rpartition("sha256:")[2])
            spelled = "sha256:" + base64.urlsafe_b64encode(raw).rstrip(b"=").decode()
            if spelled != binary.get("_gemm"):
                stale.append(f"{name}: measured against _gemm {measured}")
    assert not stale, "shipped offline facts name kernels this wheel lacks:\n" + "\n".join(stale)
    print(f"{wheel.name}: {len(tables)} offline fact files bind to its kernels")


if __name__ == "__main__":
    main()
