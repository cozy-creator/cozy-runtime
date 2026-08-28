#!/bin/sh
# cr-048 + cr-049 boot proof: build this tree's wheel and the endpoint fixture's, bake them
# into a substrate stand-in image beside the pinned `uv`, and BOOT it with no provision
# document. The pod then receives its FIRST placement set over the wire and must fetch,
# materialize and serve it. See proofs/boot/pod.py for what is measured.
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
UV_OFFLINE=1 "$uv_bin" build --offline --wheel --out-dir "$proof_root/context/wheels" \
    "$repo/proofs/fixtures/cli-project"
# The materializer's `uv` is BAKED, exactly as the substrate bakes it: internal/environment.py
# pins its digest, length and version, so any other binary refuses before a venv is created.
cp "$(readlink -f "$uv_bin")" "$proof_root/context/uv"
cp "$here/Dockerfile" "$here/pod.py" "$here/hub.py" "$here/nvidia-smi" \
    "$repo/proofs/support/proof_data.py" "$proof_root/context/"

nice -n 19 docker build --quiet -t "$image" "$proof_root/context" >/dev/null
nice -n 19 docker run --rm --name "$container" --network none "$image"
