"""Freeze the real FFN fixture and derive required cohort hashes from supplied wheels."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import shutil
import subprocess
import zipfile
from pathlib import Path


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-wheel", type=Path, required=True)
    parser.add_argument("--tensorfs-wheel", type=Path, required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = Path(__file__).resolve().parent
    assert not args.output.exists(), "preserve prior artifacts"
    assert not subprocess.check_output(["git", "-C", str(source), "status", "--porcelain"])
    commit = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    tree = ast.parse((source / "fixture/ci_varena_refiner_ffn/__init__.py").read_text())
    modules = ast.literal_eval(next(
        node.value for node in tree.body if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "RUNTIME" for target in node.targets)
    ))
    with (zipfile.ZipFile(args.runtime_wheel) as runtime,
          zipfile.ZipFile(args.tensorfs_wheel) as tfs):
        extension = next(name for name in tfs.namelist()
                         if name.startswith("tensorfs/_ext.") and name.endswith(".so"))
        payload = {
            "runtime_modules": {name: digest(runtime.read("cozy_runtime/" + name))
                                for name in modules},
            "extension_sha256": digest(tfs.read(extension)),
            "shim_sha256": digest(tfs.read("tensorfs/_varena_torch.so")),
            "gpu_uuid": args.gpu_uuid,
        }
    project = args.output / "project"
    shutil.copytree(source / "fixture", project,
                    ignore=shutil.ignore_patterns("__pycache__", ".venv", ".task", "dist"))
    (args.output / "input.json").write_text(json.dumps(payload, indent=2) + "\n")
    (args.output / "provenance.json").write_text(json.dumps({
        "fixture_commit": commit,
        "runtime_wheel_sha256": digest(args.runtime_wheel.read_bytes()),
        "tensorfs_wheel_sha256": digest(args.tensorfs_wheel.read_bytes()),
        "files": {str(path.relative_to(project)): digest(path.read_bytes())
                  for path in sorted(project.rglob("*")) if path.is_file()},
        "scope": "fixture preparation only; no GPU execution",
    }, indent=2) + "\n")
    print(json.dumps({"project": str(project), "input": str(args.output / "input.json")}))


if __name__ == "__main__":
    main()
