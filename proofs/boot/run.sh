#!/bin/sh
# cr-048 boot proof: build this tree's wheel, bake it into a substrate stand-in image, and
# BOOT it with no provision document. See proofs/boot/pod.py for what is measured.
set -eu

repo=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
here="$repo/proofs/boot"
uv_bin=$(command -v uv)
proof_root=$(mktemp -d /tmp/cozy-boot-proof.XXXXXX)
image="cozy-runtime-boot-proof:$(basename "$proof_root")"

cleanup() {
    docker rm -f "$container" >/dev/null 2>&1 || true
    docker image rm -f "$image" >/dev/null 2>&1 || true
    chmod -R u+w "$proof_root" 2>/dev/null || true
    rm -rf -- "$proof_root"
}
container="cozy-boot-proof-$$"
trap cleanup EXIT HUP INT TERM

mkdir -p "$proof_root/context/wheels"
UV_OFFLINE=1 "$uv_bin" build --offline --wheel --out-dir "$proof_root/context/wheels" "$repo"
cp "$here/Dockerfile" "$here/pod.py" "$here/nvidia-smi" "$proof_root/context/"

nice -n 19 docker build --quiet -t "$image" "$proof_root/context" >/dev/null
docker run --rm --name "$container" --network none "$image"
