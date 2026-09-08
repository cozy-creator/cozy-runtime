"""Build the fused-kernel artifact ahead of time: `python -m cozy_runtime.internal.fusion_build`.

Compiles every `fusion.specs` entry of every width profile for each requested architecture
with Triton's compiler, offline (no GPU, no driver: `ptxas` ships in the Triton wheel), plus
Triton's own launcher module for this platform, into one directory:

    <out>/manifest.json               triton, kernel-source digest, utils key, targets
    <out>/sm<arch>/<hash>/<kernel>.*  Triton's cache entry for (kernel, signature, arch)
    <out>/utils/<key>/cuda_utils.*.so Triton's launcher module

The worker image's cu130 build stage runs this after installing the runtime wheel, into
the package's own `_kernels/fusion` (the default `--out`), so a pod loads and never
compiles. A developer runs the same command once per box for their card. `--check`
rebuilds into a scratch directory and proves the build is deterministic and that the
loader's key recomputation matches — what CI runs, with no GPU.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

from cozy_runtime.internal import fusion


def build(out: Path, archs: tuple[int, ...], *, utils: bool = True) -> dict[str, Any]:
    """Compile the artifact into `out` (replacing it) and return its manifest.

    `utils=False` skips Triton's launcher module, which links against the host's
    `libcuda.so.1`: a driverless CI runner can still prove the cubins.
    """
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.runtime.cache import get_cache_manager

    if triton.__version__ != fusion.TRITON_VERSION:
        raise SystemExit(
            f"fusion_build: triton {triton.__version__} != pinned {fusion.TRITON_VERSION}"
        )
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    targets: dict[str, dict[str, str]] = {}
    with tempfile.TemporaryDirectory(prefix="cozy-fusion-build-") as scratch:
        triton.knobs.cache.dir = scratch  # Triton's own cache goes to scratch, not the box's
        for arch in archs:
            entries: dict[str, str] = {}
            for profile in fusion.PROFILES.values():
                for spec in fusion.specs(**profile):
                    compiled = triton.compile(
                        fusion.ast_source(spec),
                        target=GPUTarget("cuda", arch, fusion.WARP_SIZE),
                        options={"num_warps": spec.num_warps},
                    )
                    expected = fusion.cache_key(spec, arch)
                    if compiled.hash != expected:
                        raise SystemExit(
                            f"fusion_build: {spec.key} sm{arch} compiled under key "
                            f"{compiled.hash[:12]} but the loader computes {expected[:12]}"
                        )
                    entry = out / f"sm{arch}" / compiled.hash
                    entry.mkdir(parents=True, exist_ok=True)
                    for name, path in compiled.metadata_group.items():
                        shutil.copy2(path, entry / name)
                    entries[spec.key] = compiled.hash
            targets[f"sm{arch}"] = entries
        utils_key = fusion.utils_key()
        if utils:
            from triton.backends.nvidia.driver import CudaUtils

            CudaUtils()  # compiles Triton's launcher module into the scratch cache
            built = Path(get_cache_manager(utils_key).cache_dir)
            destination = out / "utils" / utils_key
            destination.mkdir(parents=True)
            for path in built.iterdir():
                if path.is_file() and path.name != "lock":
                    shutil.copy2(path, destination / path.name)
    manifest: dict[str, Any] = {
        "triton": fusion.TRITON_VERSION,
        "kernels": fusion.source_digest(),
        "profiles": fusion.PROFILES,
        "utils": utils_key if utils else "",
        "targets": targets,
    }
    (out / fusion.MANIFEST).write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    return manifest


def check(archs: tuple[int, ...]) -> None:
    """Two builds agree byte for byte, and every key is what the loader recomputes."""
    with tempfile.TemporaryDirectory(prefix="cozy-fusion-check-") as scratch:
        first, second = Path(scratch, "a"), Path(scratch, "b")
        manifest = build(first, archs, utils=False)
        if build(second, archs, utils=False) != manifest:
            raise SystemExit("fusion_build: two builds produced different manifests")
        for arch in archs:
            for entry, digest in manifest["targets"][f"sm{arch}"].items():
                a = (first / f"sm{arch}" / digest).glob("*.cubin")
                b = (second / f"sm{arch}" / digest).glob("*.cubin")
                if [p.read_bytes() for p in sorted(a)] != [p.read_bytes() for p in sorted(b)]:
                    raise SystemExit(f"fusion_build: {entry} sm{arch} cubin is not deterministic")
    print(f"fusion_build: deterministic for {', '.join(f'sm{a}' for a in archs)}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--arch", type=int, action="append", required=True, help="e.g. 90")
    parser.add_argument("--out", type=Path, default=fusion.KERNELS_DIR)
    parser.add_argument("--check", action="store_true", help="prove determinism, write nothing")
    args = parser.parse_args(argv)
    archs = tuple(args.arch)
    if args.check:
        check(archs)
        return
    manifest = build(args.out, archs)
    entries = sum(len(v) for v in manifest["targets"].values())
    print(f"fusion_build: {entries} entries for {sorted(manifest['targets'])} at {args.out}")


if __name__ == "__main__":
    main(sys.argv[1:])
