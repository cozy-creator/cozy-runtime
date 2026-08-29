#!/bin/sh
# cr-048 + cr-049 boot proof: build this tree's wheel and the package fixture, install only
# the Runtime/base family into the base-worker-image stand-in, and BOOT with no provision document.
# The pod then receives its FIRST placement and must fetch, overlay, and serve it.
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
cp "$here/Dockerfile" "$here/pod.py" "$here/hub.py" "$here/nvidia-smi" \
    "$repo/proofs/support/proof_data.py" "$proof_root/context/"

nice -n 19 docker build --quiet -t "$image" "$proof_root/context" >/dev/null
# THE WORKER'S CGROUP GRANT, in its local form. The launcher hands the pod a
# DELEGATED cgroup-v2 subtree, and internal/proctree.py refuses to run an executor without
# one — it never silently falls back to a weaker scope, which is the property, not an
# inconvenience. `--cgroupns=private` gives the container its own namespaced cgroup root
# and docker mounts THAT subtree rw; the container reaches no host cgroup and still has no
# network at all. Bind-mounting the host's /sys/fs/cgroup would have handed the proof the
# host root instead, which is why it is not what this does.
nice -n 19 docker run --rm --name "$container" --network none \
    --cgroupns=private --privileged "$image"
