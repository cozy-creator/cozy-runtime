"""Release gate: every kernel verdict a wheel ships binds to code that wheel carries, and passed.

    uv run python scripts/verify-verdicts.py --dist dist

Verdicts are measured on the manylinux x86_64 native wheel (`cli.qualify`). This reads that
wheel's own `qualification_tables/kernels-sm<N>.json` and its native modules, without CUDA, and
fails when a shipped `prebuilt-*` verdict names cubins (`fatbin.cubins`, per architecture) or an
interface (`.pyi`) the wheel does not carry, or ships `ok: false`: a stale fact would be
silently unmatched, and a refused one is a bug to fix, never a fact to ship. Other platforms'
wheels carry other builds and serve their family defaults by design.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import sys
import zipfile
from pathlib import Path

from cozy_runtime.internal.fatbin import cubins

KERNELS = "cozy_runtime/_kernels/"
TABLES = "cozy_runtime/internal/qualification_tables/kernels-sm"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist", required=True, type=Path)
    wheel = next(parser.parse_args().dist.resolve().glob("*-manylinux_2_28_x86_64.whl"))
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        tables = {n: json.loads(archive.read(n)) for n in names if n.startswith(TABLES)}
        stubs = {
            n.removeprefix(KERNELS).removesuffix(".pyi"): archive.read(n)
            for n in names
            if n.startswith(KERNELS) and n.endswith(".pyi")
        }
        binaries = {
            n.removeprefix(KERNELS).split(".")[0]: archive.read(n)
            for n in names
            if n.startswith(KERNELS) and n.endswith(".so")
        }
    wrong = []
    for name, table in sorted(tables.items()):
        for row in table["verdicts"]:
            if not row["name"].startswith("prebuilt-"):
                continue  # an environment's attention package, not this wheel's binary
            module = "_fa3" if row["name"] == "prebuilt-fa3" else "_gemm"
            stub = stubs.get(module, b"")
            carried = (
                cubins(binaries.get(module, b""), table["sm"]),
                "sha256:"
                + base64.urlsafe_b64encode(hashlib.sha256(stub).digest()).rstrip(b"=").decode(),
            )
            if (row["build"], row["abi"]) != carried:
                wrong.append(f"{name}: {row['name']} measured {row['build']} {row['abi']}")
            if row["ok"] is not True:
                wrong.append(f"{name}: {row['name']} ships a refusal: {row['detail']}")
    if wrong:
        print("shipped kernel verdicts that cannot ship:\n" + "\n".join(wrong), file=sys.stderr)
        return 1
    print(f"{wheel.name}: {len(tables)} kernel verdict files bind to its kernels")
    return 0


if __name__ == "__main__":
    sys.exit(main())
