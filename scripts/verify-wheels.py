#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["wheel==0.47.0", "packaging>=24"]
# ///
"""Check the published-wheel contract without importing CUDA: one version, its RECORDs and tags,
each native wheel's kernel modules, and which wheel each platform selects. No wheel carries a
machine (`cozy machine serve` is it)."""

from __future__ import annotations

import argparse
import platform
from email.parser import BytesParser
from pathlib import Path

from packaging.tags import compatible_tags, cpython_tags
from packaging.utils import parse_wheel_filename
from wheel.wheelfile import WheelFile

WINDOWS = "cp312-abi3-win_amd64"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist", required=True, type=Path)
    args = parser.parse_args()
    wheels = sorted(args.dist.resolve().glob("*.whl"))
    versions = {str(parse_wheel_filename(w.name)[1]) for w in wheels}
    assert len(versions) == 1, "the wheel set must contain one version"
    (version,) = versions
    # Each Linux arch's release job builds its own native wheel beside the pure one; x86_64's also
    # builds the Windows one (runtime-publish.yaml holds the exact release set).
    arch = platform.machine()
    linux = f"cp312-abi3-manylinux_2_28_{arch}"
    assert platform.system() == "Linux" and arch in ("x86_64", "aarch64"), (
        f"no Linux wheel for {arch}"
    )
    kernels: dict[str, tuple[str, ...]] = {
        "py3-none-any": (),
        linux: ("_gemm.abi3.so", "_norm.abi3.so", *(("_fa3.abi3.so",) * (arch == "x86_64"))),
    }
    if arch == "x86_64" and any(w.name.endswith(f"-{WINDOWS}.whl") for w in wheels):
        kernels[WINDOWS] = ("_gemm.pyd", "_norm.pyd")
    assert {w.name for w in wheels} == {f"cozy_runtime-{version}-{tag}.whl" for tag in kernels}, (
        "missing or unexpected platform wheels"
    )
    tags_by_name = {}
    for wheel in wheels:
        name, _, _, tags = parse_wheel_filename(wheel.name)
        assert name == "cozy-runtime" and len(tags) == 1
        (tag,) = tags
        tags_by_name[wheel.name] = tags
        with WheelFile(wheel) as archive:
            # Reading each member verifies the standard RECORD hashes.
            for member in archive.infolist():
                archive.read(member.filename)
            metadata = BytesParser().parsebytes(
                archive.read(f"cozy_runtime-{version}.dist-info/WHEEL")
            )
            assert metadata.get_all("Tag") == [str(tag)]
            pure = str(tag) == "py3-none-any"
            assert metadata["Root-Is-Purelib"] == ("true" if pure else "false")
            assert not any(".data/scripts/" in n for n in archive.namelist()), (
                f"{wheel.name} carries a script"
            )
            modules = {
                n.removeprefix("cozy_runtime/_kernels/")
                for n in archive.namelist()
                if n.startswith("cozy_runtime/_kernels/") and n.endswith((".so", ".pyd"))
            }
            assert modules == set(kernels[str(tag)]), f"{wheel.name} kernels: {modules}"

    for minor in (12, 13, 14):
        for platforms in (
            ["manylinux_2_28_x86_64", "manylinux_2_17_x86_64"],
            ["manylinux_2_17_x86_64"],
            ["manylinux_2_28_aarch64", "manylinux_2_17_aarch64"],
            ["win_amd64"],
            ["macosx_14_0_arm64"],
            ["musllinux_1_2_x86_64"],
        ):
            want = "py3-none-any"
            if f"manylinux_2_28_{arch}" in platforms:
                want = linux
            elif platforms == ["win_amd64"] and WINDOWS in kernels:
                want = WINDOWS
            supported = list(cpython_tags((3, minor), platforms=platforms))
            supported += list(compatible_tags((3, minor), platforms=platforms))
            ranks = {tag: index for index, tag in enumerate(supported)}
            selected = min(
                tags_by_name,
                key=lambda name: min(
                    (ranks[tag] for tag in tags_by_name[name] if tag in ranks),
                    default=len(ranks),
                ),
            )
            assert selected == f"cozy_runtime-{version}-{want}.whl", selected

    print(f"{version}: RECORDs, tags, kernels and selection verified")


if __name__ == "__main__":
    main()
