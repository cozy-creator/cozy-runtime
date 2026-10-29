#!/usr/bin/env python3
"""Verify the exact approved release files without importing or executing their contents."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path


def selected(sums: str) -> dict[str, str]:
    words = sums.split()
    if not words or len(words) % 2:
        raise ValueError("provide nonempty '<sha256> <filename>' pairs")
    files: dict[str, str] = {}
    for digest, name in zip(words[::2], words[1::2], strict=True):
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError(f"invalid SHA-256 for {name!r}")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", name) or name in files:
            raise ValueError(f"invalid or duplicate release filename {name!r}")
        files[name] = digest
    return files


def part(version: str, linux: list[str]) -> set[str]:
    """The files a release part carries: x86_64 brings the sdist, the pure and Windows wheels."""
    if not linux or set(linux) - {"x86_64", "aarch64"} or len(set(linux)) != len(linux):
        raise ValueError(f"linux must name x86_64 and/or aarch64 once each, not {linux!r}")
    files = {f"cozy_runtime-{version}-cp312-abi3-manylinux_2_28_{cpu}.whl" for cpu in linux}
    if "x86_64" in linux:
        files |= {
            f"cozy_runtime-{version}.tar.gz",
            f"cozy_runtime-{version}-py3-none-any.whl",
            f"cozy_runtime-{version}-cp312-abi3-win_amd64.whl",
        }
    return files


def verify(dist: Path, sums: str, version: str, linux: list[str]) -> str:
    files = selected(sums)
    if not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", version):
        raise ValueError("release version must be the verified X.Y.Z version")
    required = part(version, linux)
    if files.keys() != required:
        raise ValueError(
            f"approved files are not the complete {version} release part {linux}: "
            f"missing={sorted(required - files.keys())}, extra={sorted(files.keys() - required)}"
        )
    actual = {path.name for path in dist.iterdir() if not path.name.startswith(".")}
    if actual != files.keys():
        raise ValueError(
            f"release files differ: missing={sorted(files.keys() - actual)}, "
            f"extra={sorted(actual - files.keys())}"
        )
    for name, expected in files.items():
        path = dist / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"release asset is not an ordinary file: {name}")
        with path.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != expected:
                raise ValueError(f"release asset digest differs: {name}")
    return "\n".join(f"{files[name]}  {name}" for name in sorted(files))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist", type=Path, required=True)
    parser.add_argument("--version", required=True, help="Version verified from the immutable tag")
    parser.add_argument("--linux", default='["x86_64", "aarch64"]', help="The part's Linux CPUs, a JSON list")
    args = parser.parse_args()
    sums = verify(args.dist, os.environ.get("SUMS", ""), args.version, json.loads(args.linux))
    if output := os.environ.get("GITHUB_OUTPUT"):
        with open(output, "a") as stream:
            stream.write(f"sha256sums<<SUMS\n{sums}\nSUMS\n")
    print(sums)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError) as error:
        raise SystemExit(f"release assets refused: {error}") from error
