#!/usr/bin/env -S uv run --script
# /// script
# requires-python = "==3.12.*"
# dependencies = ["wheel==0.47.0"]
# ///
"""Build current Python edits with an unchanged, verified native wheel's CUDA extension."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import tomllib
from email.parser import BytesParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KERNEL = Path("cozy_runtime/_kernels/_C.abi3.so")
NATIVE_INPUTS = (
    "native",
    "CMakeLists.txt",
    "scripts/build-kernel-wheel.sh",
    "scripts/dual_build_backend.py",
    "src/cozy_runtime/_kernels",
)
ENV = {"PATH": os.defpath, "PYTHONNOUSERSITE": "1", "SOURCE_DATE_EPOCH": "946684800"}


def run(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, cwd=ROOT, env=ENV, capture_output=True, text=True, check=check)


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def native_inputs(commit: str) -> None:
    """The single comparison used before and after the Python build."""
    changed = run("git", "diff", "--quiet", commit, "--", *NATIVE_INPUTS, check=False)
    added = run("git", "ls-files", "--others", "--exclude-standard", "--", *NATIVE_INPUTS)
    old = tomllib.loads(run("git", "show", f"{commit}:pyproject.toml").stdout)
    new = tomllib.loads((ROOT / "pyproject.toml").read_text())

    def config(value: dict[str, object]) -> object:
        project, tool = value["project"], value["tool"]
        assert isinstance(project, dict) and isinstance(tool, dict)
        return (
            value["build-system"],
            project["requires-python"],
            tool["scikit-build"],
            tool["hatch"],
        )

    if changed.returncode or added.stdout or config(old) != config(new):
        raise ValueError("native build inputs changed; build a new native wheel before reusing it")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-wheel", type=Path, required=True)
    parser.add_argument("--native-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    donor = args.native_wheel.resolve()
    donor_digest = digest(donor)
    if donor_digest != args.native_sha256.removeprefix("sha256:"):
        raise ValueError("native wheel differs from its supplied SHA-256")
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="cozy-dev-wheel-") as directory:
        work = Path(directory)
        # These standard commands verify input RECORD hashes and regenerate output RECORD.
        run(sys.executable, "-m", "wheel", "unpack", str(donor), "--dest", str(work / "donor"))
        native = next((work / "donor").iterdir())
        stamp = (native / "cozy_runtime/_build_provenance.py").read_text()
        match = re.search(r'^COMMIT = "([0-9a-f]{40})"$', stamp, re.MULTILINE)
        if match is None:
            raise ValueError("native donor must identify its exact release source commit")
        native_commit = match.group(1)
        tags = (
            BytesParser()
            .parsebytes(next(native.glob("*.dist-info/WHEEL")).read_bytes())
            .get_all("Tag")
        )
        if not tags or len(tags) != 1 or not tags[0].startswith("cp312-abi3-"):
            raise ValueError("native donor must be one CPython 3.12 abi3 platform wheel")
        if set(native.rglob("*.so*")) != {native / KERNEL}:
            raise ValueError("native donor has an unsupported extension or bundled-library layout")
        native_inputs(native_commit)
        # Build edits as they exist now. This immutable wheel, not Git HEAD, is their identity.
        run(
            shutil.which("uv") or "uv",
            "--no-config",
            "build",
            "--wheel",
            "--no-sources",
            "--python",
            sys.executable,
            "--out-dir",
            str(work / "pure"),
            str(ROOT),
        )
        native_inputs(native_commit)
        pure = next((work / "pure").glob("*.whl"))
        source_digest = digest(pure)
        run(sys.executable, "-m", "wheel", "unpack", str(pure), "--dest", str(work / "python"))
        tree = next((work / "python").iterdir())
        if (tree / KERNEL).exists() or list(tree.glob("*.data")):
            raise ValueError("Python build must be a pure Runtime wheel with no spread-data tree")
        info = next(tree.glob("*.dist-info"))
        metadata = BytesParser().parsebytes((info / "METADATA").read_bytes())
        if metadata["Name"] != "cozy-runtime":
            raise ValueError("Python build is not cozy-runtime")
        recipe_digest = digest(Path(__file__))
        build_digest = hashlib.sha256(
            f"{source_digest}:{donor_digest}:{recipe_digest}".encode()
        ).hexdigest()
        version = f"{metadata['Version'].split('+')[0]}+dev.h{build_digest}"
        metadata.replace_header("Version", version)
        (info / "METADATA").write_bytes(metadata.as_bytes())
        wheel = BytesParser().parsebytes((info / "WHEEL").read_bytes())
        wheel.replace_header("Root-Is-Purelib", "false")
        wheel.replace_header("Generator", "cozy-runtime dev (wheel pack)")
        del wheel["Tag"]
        wheel["Tag"] = tags[0]
        (info / "WHEEL").write_bytes(wheel.as_bytes())
        info.rename(tree / f"cozy_runtime-{version}.dist-info")
        (tree / KERNEL).write_bytes((native / KERNEL).read_bytes())
        (tree / "cozy_runtime/_build_provenance.py").write_text(
            f'"""Development wheel snapshot; no Git commit is claimed."""\n\n'
            f'COMMIT = "unknown"\nSOURCE_WHEEL_SHA256 = "sha256:{source_digest}"\n'
        )
        provenance = {
            "recipe_sha256": recipe_digest,
            "native_source_commit": native_commit,
            "native_wheel_sha256": donor_digest,
            "native_extension_sha256": digest(native / KERNEL),
        }
        (tree / KERNEL.parent / "native-provenance.json").write_text(
            json.dumps(provenance, sort_keys=True, separators=(",", ":")) + "\n"
        )
        run(sys.executable, "-m", "wheel", "pack", str(tree), "--dest-dir", str(out))
        snapshot = out / "source-wheels" / source_digest
        snapshot.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(pure, snapshot / pure.name)
        built = out / f"cozy_runtime-{version}-{tags[0]}.whl"
        print(
            json.dumps(
                {
                    "wheel": str(built),
                    "sha256": digest(built),
                    "source_wheel_sha256": source_digest,
                    **provenance,
                },
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        raise SystemExit(f"development wheel refused: {exc}") from exc
