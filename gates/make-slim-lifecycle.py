#!/usr/bin/env python3
"""Create a proof-only SDXL source copy from one exact packages commit.

--repo PACKAGES --ref FULL_SHA --output NEW_DIRECTORY [--path sdxl]
The copied computation and dependency lock stay intact. Only generate's memo policy,
diagnostic inputs and the after-first-step self-kill hook are changed.
"""

import argparse
import ast
import io
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import tarfile


def patch(source):
    tree = ast.parse(source)
    (generate,) = [
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "generate"
    ]
    (payload,) = [
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Txt2ImgInput"
    ]
    (memo,) = [
        keyword.value
        for decorator in generate.decorator_list
        if isinstance(decorator, ast.Call)
        for keyword in decorator.keywords
        if keyword.arg == "memoize"
    ]
    (step,) = [
        node
        for node in ast.walk(generate)
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "on_step"
    ]
    (future,) = [
        node
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.module == "__future__"
    ]
    last_field = max(node.end_lineno for node in payload.body if isinstance(node, ast.AnnAssign))
    lines = source.splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))
    changes = [
        (
            offsets[memo.lineno - 1] + memo.col_offset,
            offsets[memo.end_lineno - 1] + memo.end_col_offset,
            "False",
        ),
        (
            offsets[future.end_lineno],
            offsets[future.end_lineno],
            "\nfrom ._slim_lifecycle import after_step as _slim_lifecycle_step\n",
        ),
        (
            offsets[last_field],
            offsets[last_field],
            '    lifecycle_kill: bool = False\n    lifecycle_case: str = ""\n',
        ),
        (
            offsets[step.end_lineno],
            offsets[step.end_lineno],
            " " * step.col_offset
            + "_slim_lifecycle_step(index, payload.lifecycle_kill, payload.lifecycle_case, device, tel)\n",
        ),
    ]
    for start, end, replacement in sorted(changes, reverse=True):
        source = source[:start] + replacement + source[end:]
    ast.parse(source)
    return source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--ref", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--path", default="sdxl")
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", args.ref):
        parser.error("--ref must be the full packages commit")
    relative = PurePosixPath(args.path)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        parser.error("--path must stay inside the packages repository")
    if args.output.exists():
        parser.error("--output must be a new directory")
    archive = subprocess.check_output(
        ["git", "-C", str(args.repo), "archive", "--format=tar", args.ref, args.path]
    )
    args.output.mkdir(parents=True)
    prefix = args.path.rstrip("/") + "/"
    with tarfile.open(fileobj=io.BytesIO(archive)) as source:
        for item in source.getmembers():
            if not item.name.startswith(prefix):
                continue
            item.name = item.name.removeprefix(prefix)
            if item.name:
                source.extract(item, args.output, filter="data")
    module = args.output / "sdxl" / "__init__.py"
    module.write_text(patch(module.read_text()))
    shutil.copyfile(
        Path(__file__).with_name("slim-lifecycle-fixture.py"),
        module.with_name("_slim_lifecycle.py"),
    )
    # An old generated interface must not hide the new input or restore memoization.
    (args.output / "metadata" / "package-interface.json").unlink(missing_ok=True)
    (args.output / "lifecycle-source.json").write_text(
        json.dumps(
            {
                "packages_commit": args.ref,
                "package_path": args.path,
                "proof_only": True,
                "hook": "synchronize first denoise CUDA step, then self-SIGKILL",
            },
            indent=2,
        )
        + "\n"
    )
    print(args.output.resolve())


if __name__ == "__main__":
    main()
