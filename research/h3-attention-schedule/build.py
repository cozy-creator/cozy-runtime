"""Build only the pinned Sol exact kernel; do not install or patch Kitchen."""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import tarfile
import urllib.request
from pathlib import Path

REVISION = "f8b7e97a1418d2e529bd5405fbf86312be198ae6"
SHA256 = "2053f19423a1cd4092d9278a23e049e17de494f25ddb69c94aa06e728af0ad82"
ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nvcc", required=True)
    parser.add_argument("--arch", default="sm_89")
    parser.add_argument("--variants", nargs="+", default=["baseline", "stage3", "l1kv"])
    args = parser.parse_args()
    vendor = ROOT / "vendor"
    vendor.mkdir(exist_ok=True)
    archive = vendor / f"{REVISION}.tar.gz"
    if not archive.exists():
        urllib.request.urlretrieve(
            f"https://codeload.github.com/Comfy-Org/comfy-kitchen/tar.gz/{REVISION}", archive
        )
    if hashlib.sha256(archive.read_bytes()).hexdigest() != SHA256:
        raise ValueError("Kitchen archive differs from Runtime's source pin")
    with tarfile.open(archive) as packed:
        packed.extractall(vendor, filter="data")
    source = vendor / f"comfy-kitchen-{REVISION}" / "comfy_kitchen/backends/cuda/sage_attention"
    original = (source / "sol_attn_exact.cu").read_text()
    build = ROOT / "build"
    build.mkdir(exist_ok=True)
    for name in args.variants:
        depth = {"baseline": 2, "stage3": 3, "l1kv": 2}[name]
        body = original.replace("constexpr int NSTAGE = 2;", f"constexpr int NSTAGE = {depth};")
        assert body != original or depth == 2
        if name == "l1kv":
            start = body.index("#define SOLX_STAGE")
            end = body.index("    #pragma unroll", start)
            staged = body[start:end]
            assert staged.count("cp_async16(") == 2
            body = body[:start] + staged.replace("cp_async16(", "cp_async16_ca(") + body[end:]
        body = body.replace("void launch_sol_exact(", 'extern "C" void launch_sol_exact(')
        # Introspection is outside the kernel and does not change its arithmetic.
        body += """
extern "C" int exact_resources(int* registers, int* shared, int* blocks) {
    cudaFuncAttributes a;
    cudaError_t err = cudaFuncGetAttributes(&a, sol_exact_kernel<__nv_bfloat16>);
    if (err != cudaSuccess) return err;
    *registers = a.numRegs; *shared = a.sharedSizeBytes;
    return cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        blocks, sol_exact_kernel<__nv_bfloat16>, NTHREADS, 0);
}
"""
        unit = build / f"{name}.cu"
        unit.write_text(body)
        cuda_lib = Path(args.nvcc).resolve().parent.parent / "lib"
        command = [
            args.nvcc,
            str(unit),
            "-I",
            str(source),
            "--shared",
            "-Xcompiler=-fPIC",
            "--cudart=shared",
            "--cudadevrt=none",
            "-L",
            str(cuda_lib),
            f"-Xlinker=-rpath,{cuda_lib}",
            "-std=c++17",
            "-O3",
            "--use_fast_math",
            "--expt-relaxed-constexpr",
            "--expt-extended-lambda",
            "-lineinfo",
            "-Xptxas=-v",
            f"-arch={args.arch}",
            "-o",
            str(build / f"{name}.so"),
        ]
        completed = subprocess.run(command, text=True, capture_output=True)
        (build / f"{name}.log").write_text(
            " ".join(command) + "\n" + completed.stdout + completed.stderr
        )
        print(name, completed.returncode, completed.stderr, flush=True)
        completed.check_returncode()


if __name__ == "__main__":
    main()
