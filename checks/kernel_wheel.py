#!/usr/bin/env python3
"""Kernel wheel formation: a built cozy-kernels wheel is the file its target promises
(`.github/workflows/kernels.yaml`). Its name carries exactly `cp312-abi3-<platform>`. Each
module's cubins and PTX are exactly `cozy_kernels.ARCHS` (cuobjdump). Nothing is bundled beside
it, and on Linux it needs only glibc (readelf): no torch, no CUDA library, no libpython. It
imports on every Python given, and a call on a machine without a GPU returns CUDA's error
rather than crashing. Its driver floor equals cozy-platform's (`--platform-floor`, the Rust
source TensorD checks at startup). Any failure is named, and the run exits 1.
"""

from __future__ import annotations

import argparse
import ast
import re
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any

#: What a Linux module may need: glibc's own libraries.
GLIBC = re.compile(r"(libc|libm|libdl|libpthread|librt|ld-linux[\w.-]*)\.so(\.\d+)*")
#: cozy-platform's `gpu::DRIVER_FLOOR`: Windows', then everything else's.
RUST_FLOOR = re.compile(
    r'DRIVER_FLOOR: &str = if cfg!\(windows\) \{ "([\d.]+)" \} else \{ "([\d.]+)" \}'
)
#: A module on a machine without a GPU: one gemv returns CUDA's error, a bad family is refused,
#: FA3 refuses a head dim it has no kernel for.
CALL = """
import sys
sys.path.insert(0, sys.argv[1])
from cozy_kernels import _gemm
code = _gemm.gemv(0, 0, False, 0, 0, 0, 0, False, 1, 8, 256, 0)
assert code != 0, "a gemv ran with no GPU"
assert _gemm.gemm(0, 3, False, 0, 0, 0, 0, 0, 0, False, 1, 8, 256, -1, None, 0) == 1
if "_fa3" not in sys.argv[2:]:
    from cozy_kernels import _fa3
    strides = ((0, 0, 0),) * 4
    head_96 = (0, 0, 0, 0, 0, 0, True, 1, 8, 8, 2, 2, 96, *strides)
    assert _fa3.fwd(*head_96, 1.0, False, 0, 0) == _fa3.NOT_SUPPORTED
print(f"{sys.version.split()[0]}: gemv without a GPU -> {code} {_gemm.error_name(code)}")
"""


def declared(package: Path, name: str) -> Any:
    """`cozy_kernels.<name>`, read from the source without running it."""
    target: ast.expr
    value: ast.expr | None
    for node in ast.parse((package / "__init__.py").read_text()).body:
        if isinstance(node, ast.AnnAssign):
            target, value = node.target, node.value
        elif isinstance(node, ast.Assign) and len(node.targets) == 1:
            target, value = node.targets[0], node.value
        else:
            continue
        if isinstance(target, ast.Name) and target.id == name and value is not None:
            return ast.literal_eval(value)
    raise SystemExit(f"cozy_kernels declares no {name}")


def embedded(module: Path) -> tuple[set[str], set[str]]:
    """The cubin targets and PTX the module carries, as cuobjdump lists them."""

    def listed(kind: str) -> set[str]:
        out = subprocess.run(
            ["cuobjdump", f"--list-{kind}", str(module)], capture_output=True, text=True
        ).stdout
        return set(re.findall(r"\.(sm_\d+[af]?)\.(?:cubin|ptx)\s*$", out, re.MULTILINE))

    return listed("elf"), {f"compute_{t.removeprefix('sm_')}" for t in listed("ptx")}


def needed(module: Path) -> list[str]:
    out = subprocess.run(["readelf", "-d", str(module)], capture_output=True, text=True).stdout
    return re.findall(r"\(NEEDED\)\s+Shared library: \[([^\]]+)\]", out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("wheel", type=Path)
    parser.add_argument("platform", help="the wheel's platform tag, e.g. manylinux_2_28_x86_64")
    parser.add_argument("--python", action="append", default=[], help="an interpreter to import on")
    parser.add_argument("--platform-floor", type=Path, help="cozy-platform's src/gpu/mod.rs")
    parser.add_argument("--without", action="append", default=[], help="a module this wheel omits")
    options = parser.parse_args()
    wheel: Path = options.wheel
    failed: list[str] = []
    if not wheel.name.endswith(f"-cp312-abi3-{options.platform}.whl"):
        failed.append(f"{wheel.name} is not tagged cp312-abi3-{options.platform}")
    with tempfile.TemporaryDirectory() as scratch:
        site = Path(scratch)
        with zipfile.ZipFile(wheel) as archive:
            archive.extractall(site)
        package = site / "cozy_kernels"
        bundled = [p.name for p in site.iterdir() if p.name.endswith(".libs")]
        if bundled:
            failed.append(f"the wheel bundles libraries: {bundled}")
        if options.platform_floor is not None:
            found = RUST_FLOOR.search(options.platform_floor.read_text())
            rust = {"win32": found[1], "linux": found[2]} if found else {}
            if declared(package, "DRIVER_FLOOR") != rust:
                failed.append(f"DRIVER_FLOOR differs from cozy-platform's {rust or '(none)'}")
        for name, want in declared(package, "ARCHS").items():
            modules = [p for p in package.glob(f"{name}.*") if p.suffix in (".so", ".pyd")]
            if name in options.without:
                if modules:
                    failed.append(f"{name}: carried, but this target omits it")
                continue
            if len(modules) != 1:
                failed.append(f"{name}: {len(modules)} module files")
                continue
            module = modules[0]
            cubin, ptx = embedded(module)
            for kind, have, wanted in (("cubin", cubin, want["cubin"]), ("ptx", ptx, want["ptx"])):
                if have != set(wanted):
                    failed.append(
                        f"{name} {kind}: missing {sorted(set(wanted) - have)}, "
                        f"undeclared {sorted(have - set(wanted))}"
                    )
            print(f"{name}: cubin {' '.join(sorted(cubin))}; ptx {' '.join(sorted(ptx))}")
            if module.suffix == ".so":
                foreign = [lib for lib in needed(module) if not GLIBC.fullmatch(lib)]
                if foreign:
                    failed.append(f"{name} needs {foreign}")
        for python in options.python:
            target = site / f"py-{Path(python).parent.parent.name}"
            install = [python, "-m", "pip", "install", "-q", "--no-deps", "--target", str(target)]
            run = subprocess.run([*install, str(wheel)], capture_output=True, text=True)
            if run.returncode == 0:
                run = subprocess.run(
                    [python, "-I", "-c", CALL, str(target), *options.without],
                    capture_output=True,
                    text=True,
                )
            print(run.stdout.strip())
            if run.returncode != 0:
                failed.append(f"{python}: {run.stderr.strip()[-600:]}")
    for line in failed:
        print(f"FAILED {line}", flush=True)
        print(f"::error title=kernel wheel::{line}", flush=True)
    print(f"{wheel.name}: {'ok' if not failed else f'{len(failed)} failed'}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
