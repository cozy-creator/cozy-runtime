"""The Runtime's own Triton sets prebuilt for the native wheel (`triton_kernels.PREBUILT`), on a
CPU: `python -m cozy_runtime.internal.triton_prebuild OUT [--launchers DIR]`, OUT being
`cozy_runtime/_triton`.

Every set a card row runs (`machine_kernels.triton_sets`, the LoRA bake set, tcgen05's Sol merge)
compiles for every row of `arch.ROWS` as the machine's builder compiles it (`job`,
`compile_entries`), into a scratch Triton cache; a set that does not compile for a row is left out
and named, and machines compile it as before. Each kernel's metadata and cubin go to
`kernels/<key>/` (its other stages are Triton's debugging output). Triton's launcher modules come
from `--launchers` (built where the wheel's C library floor is, kernels.yaml), else are built
here against `TRITON_LIBCUDA_PATH`'s libcuda (a driver stub on a CPU host). `manifest.json` names
them all with the Triton release they hold for: another Triton matches nothing, and a machine
compiles as before.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
import time
from pathlib import Path

import triton
from triton.runtime.build import compile_so_from_file
from triton.runtime.cache import get_cache_manager

from cozy_runtime.author.kernels import TritonKernels
from cozy_runtime.internal import (
    arch,
    attention_sol_fa4,
    lora_bake,
    machine_kernels,
    triton_kernels,
)


def sets() -> list[tuple[TritonKernels, int]]:
    """Every (set, architecture) a card row of `arch.ROWS` builds at machine start."""
    found: dict[tuple[str, int], TritonKernels] = {}
    for sm in sorted(arch.ROWS):
        row = arch.row(sm)
        extra = [attention_sol_fa4.KERNELS] if row.family == arch.TCGEN05 else []
        for kernels in (*machine_kernels.triton_sets(row), lora_bake.KERNELS, *extra):
            if kernels.entries:
                found[(kernels.name, sm)] = kernels
    return [(kernels, sm) for (_, sm), kernels in sorted(found.items())]


def _kernels(out: Path) -> tuple[dict[str, dict[str, object]], dict[str, object]]:
    shipped: dict[str, dict[str, object]] = {}
    report: dict[str, object] = {}
    for kernels, sm in sets():
        spec = triton_kernels.job(kernels, sm).spec
        began = time.perf_counter()
        try:
            triton_kernels.compile_entries(spec, lambda _: None)
        except Exception as exc:  # a row this set does not compile for: machines build it
            report[f"{kernels.name} sm{sm}"] = f"left out: {type(exc).__name__}: {exc}"[:300]
            continue
        report[f"{kernels.name} sm{sm}"] = round(time.perf_counter() - began, 2)
        for entry in kernels.entries:
            key, name = spec["entries"][entry.key], f"{entry.kernel.__name__}.json"
            group = get_cache_manager(key).get_group(name)
            if group is None:
                raise RuntimeError(f"{kernels.name} sm{sm} {entry.key}: Triton cached no group")
            kept = sorted(n for n in group if n.endswith((".json", ".cubin")))
            (out / "kernels" / key).mkdir(parents=True, exist_ok=True)
            for file in kept:
                shutil.copyfile(group[file], out / "kernels" / key / file)
            shipped[key] = {"group": name, "files": kept, "set": kernels.name, "sm": sm}
    return shipped, report


def _launchers(out: Path, given: Path | None) -> dict[str, dict[str, str]]:
    """Triton's launcher modules to ship, by file name: `given`'s
    (`scripts/build-triton-launchers.sh`, each with its driver.c digest beside it; only those of
    this Triton's driver.c), else one built here for this Python."""
    (out / "launcher").mkdir()
    digest = hashlib.sha256(triton_kernels.launcher_source()).hexdigest()
    if given is None:
        config = triton_kernels.launcher_config()
        built = compile_so_from_file(
            str(triton_kernels.launcher_driver()),
            "cuda_utils",
            library_dirs=config["library_dirs"],
            include_dirs=config["include_dirs"],
            libraries=config["libraries"],
        )
        shutil.copyfile(built, out / "launcher" / triton_kernels.LAUNCHER)
    else:
        for module in sorted(given.glob("cuda_utils.*.so")):
            sidecar = Path(f"{module}.driver-sha256")
            if sidecar.is_file() and sidecar.read_text().strip() == digest:
                shutil.copyfile(module, out / "launcher" / module.name)
    return {
        module.name: {"file": module.name, "driver_sha256": digest}
        for module in sorted((out / "launcher").iterdir())
    }


def build(out: Path, launchers: Path | None) -> dict[str, object]:
    out.mkdir(parents=True)
    with tempfile.TemporaryDirectory(prefix="cozy-triton-") as scratch:
        triton.knobs.cache.dir = scratch
        shipped, report = _kernels(out)
    modules = _launchers(out, launchers)
    manifest = {"triton": triton.__version__, "kernels": shipped, "launcher": modules}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True))
    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    return {
        "kernels": len(shipped),
        "bytes": size,
        "launchers": sorted(modules),
        "sets": report,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("out", type=Path, help="the cozy_runtime/_triton directory to write")
    parser.add_argument("--launchers", type=Path, help="prebuilt cuda_utils.*.so to ship")
    arguments = parser.parse_args()
    print(json.dumps(build(arguments.out, arguments.launchers), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
