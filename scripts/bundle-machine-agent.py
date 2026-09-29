#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["wheel==0.47.0"]
# ///
"""Install the standalone Go agent through Linux Runtime wheels.

The portable wheel and source distribution stay toolchain-free. Linux wheels
carry the native executable in the standard wheel scripts installation scheme.
This step runs after both Runtime wheels and the static agents have been built.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from email.parser import BytesParser
from pathlib import Path

NATIVE_TAG = "cp312-abi3-manylinux_2_28_x86_64"
ARCHITECTURES = ("amd64", "arm64")


def bundle(source: Path, binary: Path, tag: str, destination: Path) -> Path:
    with tempfile.TemporaryDirectory(prefix=".machine-wheel-", dir=destination) as directory:
        work = Path(directory)
        subprocess.run(
            [
                sys.executable,
                "-m",
                "wheel",
                "unpack",
                str(source),
                "--dest",
                str(work / "unpacked"),
            ],
            check=True,
        )
        (tree,) = (work / "unpacked").iterdir()
        (info,) = tree.glob("*.dist-info")
        metadata = BytesParser().parsebytes((info / "METADATA").read_bytes())
        if metadata["Name"] != "cozy-runtime":
            raise ValueError("input wheel is not cozy-runtime")
        version = metadata["Version"]
        wheel_path = info / "WHEEL"
        wheel = BytesParser().parsebytes(wheel_path.read_bytes())
        if wheel.get_all("Tag") not in (["py3-none-any"], [NATIVE_TAG]):
            raise ValueError("input wheel has an unsupported platform")
        wheel.replace_header("Root-Is-Purelib", "false")
        del wheel["Tag"]
        wheel["Tag"] = tag
        wheel_path.write_bytes(wheel.as_bytes())
        command = tree / f"cozy_runtime-{version}.data/scripts/cozy-machine"
        command.parent.mkdir(parents=True)
        command.write_bytes(binary.read_bytes())
        command.chmod(0o755)
        output = work / "packed"
        output.mkdir()
        subprocess.run(
            [sys.executable, "-m", "wheel", "pack", str(tree), "--dest-dir", str(output)],
            check=True,
            env={
                **os.environ,
                "SOURCE_DATE_EPOCH": os.environ.get("SOURCE_DATE_EPOCH", "946684800"),
            },
        )
        (built,) = output.glob("*.whl")
        target = destination / built.name
        os.replace(built, target)
        return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist", required=True, type=Path)
    parser.add_argument("--agents", required=True, type=Path)
    args = parser.parse_args()
    manifest = json.loads((args.agents / "machine-agent.json").read_text())
    version = manifest["version"]
    pure = args.dist / f"cozy_runtime-{version}-py3-none-any.whl"
    native = args.dist / f"cozy_runtime-{version}-{NATIVE_TAG}.whl"
    if not pure.is_file() or not native.is_file():
        parser.error(
            "build the pure and native Runtime wheels at the agent's release version first"
        )
    binaries = {}
    for arch in ARCHITECTURES:
        binary = args.agents / f"cozy-machine-linux-{arch}"
        expected = manifest["platforms"][f"linux-{arch}"]["sha256"]
        if hashlib.sha256(binary.read_bytes()).hexdigest() != expected:
            raise ValueError(f"{binary.name} differs from the built agent manifest")
        binaries[arch] = binary
    # Keep one Linux wheel per architecture: older updaters select the first
    # matching wheel, so a second x86 wheel must not silently drop CUDA kernels.
    # No published artifact is modified by this packaging command.
    bundle(pure, binaries["arm64"], "py3-none-manylinux_2_17_aarch64", args.dist)
    bundle(native, binaries["amd64"], NATIVE_TAG, args.dist)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        sys.exit(f"machine-agent wheel packaging failed: {error}")
