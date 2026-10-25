#!/usr/bin/env bash
set -euo pipefail
: "${COZY_CUDA_TOOLKIT:?Set COZY_CUDA_TOOLKIT to the build-time CUDA 13 toolkit}"
root=$(cd "$(dirname "$0")/.." && pwd)
destination=${1:?Usage: build-execution-native.sh INSTALL_PREFIX}
target=${CARGO_TARGET_DIR:-"$root/native/execution/target"}
export NVCC="$COZY_CUDA_TOOLKIT/bin/nvcc"
cargo build --locked --release --manifest-path "$root/native/execution/Cargo.toml"
mkdir -p "$destination/cozy_runtime/_kernels"
case "$(uname -s)" in
  Linux) cp "$target/release/lib_execution.so" "$destination/cozy_runtime/_kernels/_execution.abi3.so" ;;
  *) printf '%s\n' 'This build entry point currently targets Linux; no other platform is qualified.' >&2; exit 1 ;;
esac
