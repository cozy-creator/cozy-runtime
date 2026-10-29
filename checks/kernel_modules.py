#!/usr/bin/env python3
"""Kernel module formation: the modules kernels.yaml built for one OS and CPU are what the native
cozy-runtime wheels promise. `<modules>/cozy_runtime/_kernels` holds exactly the modules
`cozy_runtime._kernels.ARCHS` declares (less `--without`), each carrying exactly its cubins and
PTX (cuobjdump). Each needs only the OS's C runtime (readelf, dumpbin): no torch, no CUDA library,
no version-specific libpython (Windows' stable-ABI `python3.dll` is the one Python it may name).
Each loads on every Python given, and a call on a machine without a GPU returns CUDA's error
rather than crashing. The driver floor equals the `platform` crate's (`--platform-floor`, the
Rust source TensorD checks at startup). Rust execution may use the OS libgcc unwinder.
Each exports `PyInit_*` alone (nm, dumpbin): an exported libstdc++
symbol binds process-wide and crashed torch imported after it. With `--torch-python`, each module
loads before torch in a fresh interpreter. Any failure is named, and the run exits 1.
"""

from __future__ import annotations

import argparse
import ast
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

#: What a module may need: glibc's own libraries; Windows' kernel, C runtime and stable ABI.
SYSTEM = re.compile(
    r"(libc|libm|libdl|libpthread|librt|libgcc_s|ld-linux[\w.-]*)\.so(\.\d+)*"
    r"|(?i:(kernel32|python3|ucrtbase|vcruntime140(_1)?|api-ms-win-[\w-]+)\.dll)"
)
#: The `platform` crate's `gpu::DRIVER_FLOOR`: Windows', then everything else's.
RUST_FLOOR = re.compile(
    r'DRIVER_FLOOR: &str = if cfg!\(windows\) \{ "([\d.]+)" \} else \{ "([\d.]+)" \}'
)
#: A module on a machine without a GPU: one gemv returns CUDA's error, a bad family is refused,
#: FA3 and SageAttention3 refuse a head dim they have no kernel for.
CALL = """
import importlib.util, pathlib, sys
def load(name):
    [path] = [p for p in pathlib.Path(sys.argv[1]).iterdir() if p.name.split(".")[0] == name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
_gemm, _norm = load("_gemm"), load("_norm")
code = _gemm.gemv(0, 0, False, 0, 0, 0, 0, False, 1, 8, 256, 0)
assert code != 0, "a gemv ran with no GPU"
assert _gemm.gemm(0, 3, False, 0, 0, 0, 0, 0, 0, False, 1, 8, 256, -1, None, 0) == 1
assert _gemm.TMA == ("tma" not in sys.argv[2:]), f"_gemm.TMA is {_gemm.TMA}"
assert _gemm.svdq_gemm(0, *(0,) * 11, False, 1, 100, 64, 32, 0) == 1  # N % 128: refused
assert callable(_norm.rms_rope_split_half)
if "_fa3" not in sys.argv[2:]:
    _fa3 = load("_fa3")
    strides = ((0, 0, 0),) * 4
    head_96 = (0, 0, 0, 0, 0, 0, True, 1, 8, 8, 2, 2, 96, *strides)
    assert _fa3.fwd(*head_96, 1.0, False, 0, 0) == _fa3.NOT_SUPPORTED
if "_sage3" not in sys.argv[2:]:
    _sage3 = load("_sage3")
    strides = ((0, 0, 0),) * 8
    head_96 = (0, *(0,) * 10, 1, 2, 2, 128, 128, 128, 96, *strides, 1.0, False, False, True, 0)
    assert _sage3.fwd(*head_96) == _sage3.NOT_SUPPORTED
    assert _sage3.quant(0, 0, 0, 0, 0, True, 1, 2, 128, 96, (0, 0, 0), (0, 0, 0), (0, 0, 0), 0) == (
        _sage3.NOT_SUPPORTED
    )
print(f"{sys.version.split()[0]}: gemv without a GPU -> {code} {_gemm.error_name(code)}")
"""


#: The catalog the modules are held to: `cozy_runtime._kernels`.
CATALOG = Path(__file__).resolve().parents[1] / "src/cozy_runtime/_kernels/__init__.py"


def declared(name: str) -> Any:
    """`cozy_runtime._kernels.<name>`, read from the source without running it."""
    target: ast.expr
    value: ast.expr | None
    for node in ast.parse(CATALOG.read_text()).body:
        if isinstance(node, ast.AnnAssign):
            target, value = node.target, node.value
        elif isinstance(node, ast.Assign) and len(node.targets) == 1:
            target, value = node.targets[0], node.value
        else:
            continue
        if isinstance(target, ast.Name) and target.id == name and value is not None:
            return ast.literal_eval(value)
    raise SystemExit(f"cozy_runtime._kernels declares no {name}")


def embedded(module: Path) -> tuple[set[str], set[str]]:
    """The cubin targets and PTX the module carries, as cuobjdump lists them."""

    def listed(kind: str) -> set[str]:
        out = subprocess.run(
            ["cuobjdump", f"--list-{kind}", str(module)], capture_output=True, text=True
        ).stdout
        return set(re.findall(r"\.(sm_\d+[af]?)\.(?:cubin|ptx)\s*$", out, re.MULTILINE))

    return listed("elf"), {f"compute_{t.removeprefix('sm_')}" for t in listed("ptx")}


def needed(module: Path) -> list[str]:
    windows = module.suffix == ".pyd"
    tool = ["dumpbin", "/dependents"] if windows else ["readelf", "-d"]
    out = subprocess.run([*tool, str(module)], capture_output=True, text=True).stdout
    if windows:
        return re.findall(r"^\s+(\S+\.dll)\s*$", out, re.MULTILINE | re.IGNORECASE)
    return re.findall(r"\(NEEDED\)\s+Shared library: \[([^\]]+)\]", out)


#: A module loaded by path, then torch, in one fresh interpreter.
BEFORE_TORCH = """
import importlib.util, sys
spec = importlib.util.spec_from_file_location(sys.argv[1], sys.argv[2])
spec.loader.exec_module(importlib.util.module_from_spec(spec))
import torch
print(f"{sys.argv[1]} before torch {torch.__version__}: ok")
"""


def exported(module: Path) -> list[str]:
    """The module's exported (dynamic, defined) symbols."""
    if module.suffix == ".pyd":
        out = subprocess.run(["dumpbin", "/exports", str(module)], capture_output=True, text=True)
        return re.findall(r"^\s+\d+\s+[0-9A-F]+\s+[0-9A-F]{8}\s+(\S+)", out.stdout, re.MULTILINE)
    out = subprocess.run(
        ["nm", "-D", "--defined-only", str(module)], capture_output=True, text=True
    )
    return [line.split()[-1] for line in out.stdout.splitlines() if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("modules", type=Path, help="the install prefix: <it>/cozy_runtime/_kernels")
    parser.add_argument("--python", action="append", default=[], help="an interpreter to import on")
    parser.add_argument("--platform-floor", type=Path, help="cozy's crates/platform/src/gpu/mod.rs")
    parser.add_argument("--torch-python", help="an interpreter with torch: each module before it")
    parser.add_argument(
        "--without",
        action="append",
        default=[],
        help="a module this target omits, or `tma`: CUTLASS's TMA GEMMs (_gemm's `sm_*a` cubins)",
    )
    options = parser.parse_args()
    package: Path = options.modules / "cozy_runtime" / "_kernels"
    failed: list[str] = []
    archs = declared("ARCHS")
    stray = [p.name for p in package.iterdir() if p.name.split(".")[0] not in archs]
    if stray:
        failed.append(f"files beside the modules: {stray}")
    if options.platform_floor is not None:
        found = RUST_FLOOR.search(options.platform_floor.read_text())
        rust = {"win32": found[1], "linux": found[2]} if found else {}
        if declared("DRIVER_FLOOR") != rust:
            failed.append(f"DRIVER_FLOOR differs from the platform crate's {rust or '(none)'}")
    for name, want in archs.items():
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
        if name == "_gemm" and "tma" in options.without:
            # The svdq GEMM is no TMA kernel: every target carries its sm_120a and sm_121a.
            keep = ("sm_120a", "sm_121a")
            cubins = tuple(t for t in want["cubin"] if not t.endswith("a") or t in keep)
            want = {**want, "cubin": cubins}
        for kind, have, wanted in (("cubin", cubin, want["cubin"]), ("ptx", ptx, want["ptx"])):
            if have != set(wanted):
                failed.append(
                    f"{name} {kind}: missing {sorted(set(wanted) - have)}, "
                    f"undeclared {sorted(have - set(wanted))}"
                )
        print(f"{name}: cubin {' '.join(sorted(cubin))}; ptx {' '.join(sorted(ptx))}")
        libraries = needed(module)
        print(f"{name} needs {' '.join(libraries)}")
        foreign = [lib for lib in libraries if not SYSTEM.fullmatch(lib)]
        if foreign or not libraries:
            failed.append(f"{name} needs {foreign or 'nothing readable'}")
        # A DLL's exports bind no other DLL's names: Windows may also export nanobind's own
        # exception types (dllexport for cross-module translation) and cudart_static's Optimus hint.
        windows = module.suffix == ".pyd"
        ours = ("@nanobind@@", "NvOptimusEnablementCuda")
        exports = [e for e in exported(module) if not (windows and any(o in e for o in ours))]
        if exports != [f"PyInit_{name}"]:
            failed.append(
                f"{name} exports {len(exports)} symbols, not PyInit_{name} alone: {exports[:5]}"
            )
        if options.torch_python:
            call = [options.torch_python, "-I", "-c", BEFORE_TORCH, name, str(module)]
            run = subprocess.run(call, capture_output=True, text=True)
            print(run.stdout.strip())
            if run.returncode != 0:
                failed.append(
                    f"{name} before torch exits {run.returncode}: {run.stderr.strip()[-400:]}"
                )
    for python in options.python:
        call = [python, "-I", "-c", CALL, str(package), *options.without]
        run = subprocess.run(call, capture_output=True, text=True)
        print(run.stdout.strip())
        if run.returncode != 0:
            failed.append(f"{python}: {run.stderr.strip()[-600:]}")
    for line in failed:
        print(f"FAILED {line}", flush=True)
        print(f"::error title=kernel modules::{line}", flush=True)
    print(f"{package}: {'ok' if not failed else f'{len(failed)} failed'}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
