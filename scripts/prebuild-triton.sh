#!/usr/bin/env bash
# The Runtime's prebuilt Triton sets (`triton_prebuild`) for each Triton worker images serve
# (torch-triton.txt, at most two): each in its own CPU environment, that pair's CPU torch and Triton
# plus this source and its locked dependencies, into OUT/<Triton release>/ with
# LAUNCHERS/<Triton release>/'s launcher modules. Triton's keys need a libcuda: an empty stub.
set -euo pipefail
out=${1:?Usage: prebuild-triton.sh OUT LAUNCHERS}; launchers=${2:?Usage: prebuild-triton.sh OUT LAUNCHERS}
root=$(cd "$(dirname "$0")/.." && pwd)
pairs=$(grep -o '^torch==[^ ]* triton==[^ ]*' "$root/torch-triton.txt")
[ "$(wc -l <<< "$pairs")" -le 2 ] || { echo "torch-triton.txt serves at most two Tritons" >&2; exit 1; }
scratch=$(mktemp -d); trap 'rm -rf "$scratch"' EXIT
mkdir "$scratch/stub"
echo 'void cozy_libcuda_stub(void) {}' | "${CC:-cc}" -shared -x c -o "$scratch/stub/libcuda.so.1" -
uv export -q --project "$root" --frozen --no-dev --no-emit-project --no-header -o "$scratch/locked.txt"
while read -r torch triton; do
  venv=$scratch/${triton#triton==}
  uv venv -q --python 3.12 "$venv"
  uv pip install -q --python "$venv" "$torch" --index-url https://download.pytorch.org/whl/cpu
  uv pip install -q --python "$venv" "$triton" -r "$scratch/locked.txt"
  uv pip install -q --python "$venv" --no-deps "$root"
  TRITON_LIBCUDA_PATH=$scratch/stub "$venv/bin/python" -m cozy_runtime.internal.triton_prebuild "$out" --launchers "$launchers"
done <<< "$pairs"
