#!/usr/bin/env bash
# Build cozy-runtime's OTHER wheel file: same source, same version, tagged cp312-abi3 and
# carrying `cozy_runtime/_kernels/_C.abi3.so` (cr-094). ONE recipe, used by the release
# workflow and by a developer who wants fused kernels on a box with no CUDA toolkit --
# nvcc lives in the container, never on the host.
#
#   scripts/build-kernel-wheel.sh [--out DIR] [--commit SHA]
#
# WHY manylinux rather than the GPU worker image's own base: a wheel's glibc floor is the
# floor of the machine that COMPILED it. Building in the tensorhub profile's Ubuntu 24.04
# base (glibc 2.39, gcc 13) yields a .so that auditwheel will only certify manylinux_2_39 --
# unpublishable in practice, since it excludes every glibc 2.28-2.38 host. AlmaLinux 8's
# glibc 2.28 plus gcc-toolset is what buys the floor. The part that actually has to match
# the worker image matches exactly: nvcc 13.0.88, cp312, static cudart.
set -euo pipefail

# Pinned by digest: the toolchain that compiles a PUBLISHED binary is part of the artifact.
# Tag at the time of pinning: quay.io/pypa/manylinux_2_28_x86_64:latest (AlmaLinux 8,
# glibc 2.28, gcc-toolset-14, CPython 3.12 at /opt/python/cp312-cp312).
IMAGE=quay.io/pypa/manylinux_2_28_x86_64@sha256:0536c364004fa2a3c5041120b6fe35d84fc5bfe31f04c6a6304f13eac4a67b63
# The floor is a CHOSEN CONTRACT, not whatever the compiler happened to emit. The build
# currently clears it with room to spare (eligible for 2_26); `--only-plat` keeps the
# published tag stable at that contract instead of drifting with the low-water mark, and
# auditwheel REFUSES the repair outright if a future change ever pushes the floor above it.
PLAT=manylinux_2_28_x86_64
CUDA=13-0

if [ "${1:-}" = "--in-container" ]; then
    dnf install -y -q dnf-plugins-core
    dnf config-manager --add-repo \
        https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/cuda-rhel8.repo
    # nvcc and the static CUDA runtime only. No driver, no cuDNN, no torch: CMakeLists asks
    # for CUDAToolkit, Python and nanobind, and nothing else.
    dnf install -y -q "cuda-nvcc-$CUDA" "cuda-cudart-devel-$CUDA"
    export PATH="/usr/local/cuda-${CUDA/-/.}/bin:$PATH"
    # Explicit: the image also has a system gcc 8.5 and a clang, and which one CMake picks
    # by PATH order is not something a release artifact should depend on.
    export CC=/opt/rh/gcc-toolset-14/root/usr/bin/gcc
    export CXX=/opt/rh/gcc-toolset-14/root/usr/bin/g++
    python=/opt/python/cp312-cp312/bin/python
    # The checkout is mounted read-only; stamping and the CMake build both write.
    cp -a /src /build
    cd /build
    "$python" scripts/stamp-build-provenance.py "$COZY_RUNTIME_BUILD_COMMIT"
    "$python" -m pip install --no-cache-dir -q uv==0.12.7
    COZY_RUNTIME_BUILD_KERNELS=1 "$python" -m uv build \
        --wheel --python "$python" --out-dir /tmp/raw /build
    auditwheel show /tmp/raw/*.whl
    # REPAIR, not `wheel tags`: the .so needs nothing outside the manylinux allow-list, so
    # repair bundles no library and only rewrites the tag -- but it REFUSES a tag the binary
    # cannot honour, which a blind retag would happily write.
    auditwheel repair --plat "$PLAT" --only-plat -w /out /tmp/raw/*.whl
    "$python" - "$PLAT" <<'INNER'
import pathlib, sys, zipfile
plat, = sys.argv[1:]
whl, = pathlib.Path("/out").glob("*-cp312-abi3-*.whl")
if not whl.name.endswith(f"-cp312-abi3-{plat}.whl"):
    sys.exit(f"wheel is not tagged exactly cp312-abi3-{plat}: {whl.name}")
if "cozy_runtime/_kernels/_C.abi3.so" not in zipfile.ZipFile(whl).namelist():
    sys.exit(f"wheel carries no kernels: {whl.name}")
print(f"built {whl.name}")
INNER
    # The build runs as root; hand the artifacts back to the caller.
    chown -R "$HOST_OWNER" /out
    exit 0
fi

repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
out=$repo/dist
commit=""
while [ $# -gt 0 ]; do
    case $1 in
        --out) out=$2; shift 2 ;;
        --commit) commit=$2; shift 2 ;;
        *) echo "usage: $0 [--out DIR] [--commit SHA]" >&2; exit 2 ;;
    esac
done
[ -n "$commit" ] || commit=$(git -C "$repo" rev-parse HEAD)
mkdir -p "$out"
out=$(cd "$out" && pwd)

exec docker run --rm \
    -v "$repo:/src:ro" -v "$out:/out" \
    -e COZY_RUNTIME_BUILD_COMMIT="$commit" \
    -e HOST_OWNER="$(id -u):$(id -g)" \
    "$IMAGE" bash /src/scripts/build-kernel-wheel.sh --in-container
