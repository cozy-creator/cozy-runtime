#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["wheel==0.47.0", "packaging>=24"]
# ///
"""Check the published-wheel contract without starting a machine or importing CUDA."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import tempfile
from email.parser import BytesParser
from pathlib import Path

from packaging.tags import compatible_tags, cpython_tags
from packaging.utils import parse_wheel_filename
from wheel.wheelfile import WheelFile


def run(*args: str) -> str:
    return subprocess.check_output(args, text=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist", required=True, type=Path)
    args = parser.parse_args()
    dist = args.dist.resolve()
    wheels = sorted(dist.glob("*.whl"))
    versions = {str(parse_wheel_filename(w.name)[1]) for w in wheels}
    assert len(versions) == 1, "the wheel set must contain one version"
    (version,) = versions
    expected = {
        "py3-none-any": None,
        "cp312-abi3-manylinux_2_28_x86_64": 62,
        "py3-none-manylinux_2_17_aarch64": 183,
    }
    assert {w.name for w in wheels} == {f"cozy_runtime-{version}-{tag}.whl" for tag in expected}, (
        "missing or unexpected platform wheels"
    )
    installed_command = f"cozy_runtime-{version}.data/scripts/cozy-machine"
    tags_by_name = {}
    binaries = {}
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
            architecture = expected[str(tag)]
            if architecture is None:
                assert installed_command not in archive.namelist()
                assert metadata["Root-Is-Purelib"] == "true"
                continue
            assert metadata["Root-Is-Purelib"] == "false"
            entry = archive.getinfo(installed_command)
            assert (entry.external_attr >> 16) & 0o777 == 0o755
            binary = archive.read(installed_command)
            assert binary[:6] == b"\x7fELF\x02\x01", "expected a 64-bit little-endian ELF"
            assert int.from_bytes(binary[18:20], "little") == architecture
            binaries[str(tag)] = binary

    for minor in (12, 13, 14):
        for platforms, want in (
            (
                ["manylinux_2_28_x86_64", "manylinux_2_17_x86_64"],
                "cp312-abi3-manylinux_2_28_x86_64",
            ),
            (["manylinux_2_17_x86_64"], "py3-none-any"),
            (["manylinux_2_17_aarch64"], "py3-none-manylinux_2_17_aarch64"),
            (["win_amd64"], "py3-none-any"),
            (["macosx_14_0_arm64"], "py3-none-any"),
            (["musllinux_1_2_x86_64"], "py3-none-any"),
        ):
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

    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise SystemExit("actual install verification requires Linux x86-64")
    with tempfile.TemporaryDirectory(prefix="cozy-machine-install-") as directory:
        venv = Path(directory) / "venv"
        run("uv", "venv", str(venv), "--python", "3.12")
        python = str(venv / "bin/python")
        run(
            "uv",
            "pip",
            "install",
            "--python",
            python,
            "--no-deps",
            "--no-index",
            "--find-links",
            str(dist),
            f"cozy-runtime=={version}",
        )
        command = venv / "bin/cozy-machine"
        assert command.read_bytes() == binaries["cp312-abi3-manylinux_2_28_x86_64"]
        identity = json.loads(run(str(command), "version", "--json"))
        assert identity["name"] == "cozy-machine" and identity["version"] == version
        run("uv", "pip", "uninstall", "--python", python, "cozy-runtime")
        assert not command.exists(), "uninstall must remove the owned command"
    print(f"{version}: RECORDs, ELF platforms, modes, selection, installs and uninstall verified")


if __name__ == "__main__":
    main()
