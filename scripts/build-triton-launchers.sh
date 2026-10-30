#!/usr/bin/env bash
# Triton's CUDA launcher module (`cuda_utils`) for the Runtime's prebuilt Triton sets
# (`python -m cozy_runtime.internal.triton_prebuild OUT --launchers DIR`): the locked Triton's
# driver.c built by Triton's own builder at this host's C library floor, once per Python given,
# against an empty libcuda stub (a machine's driver supplies the real one). Each module's
# driver.c digest goes beside it, so a launcher of another Triton is never shipped.
set -euo pipefail
destination=${1:?Usage: build-triton-launchers.sh DIR PYTHON...}; shift
root=$(cd "$(dirname "$0")/.." && pwd)
triton=$(awk '/^name = "triton"$/ { getline; gsub(/^version = "|"$/, ""); print; exit }' "$root/uv.lock")
stub=$(mktemp -d)
echo 'void cozy_libcuda_stub(void) {}' | "${CC:-cc}" -shared -x c -o "$stub/libcuda.so.1" -
mkdir -p "$destination"
for python in "$@"; do
  "$python" -m pip install -q --root-user-action ignore "triton==$triton"
  TRITON_LIBCUDA_PATH=$stub "$python" -I - "$destination" <<'EOF'
import hashlib, pathlib, shutil, sys
from triton.backends.nvidia import driver
from triton.runtime.build import compile_so_from_file
source = pathlib.Path(driver.__file__).parent / "driver.c"
built = pathlib.Path(compile_so_from_file(str(source), "cuda_utils", library_dirs=driver.library_dirs(),
                                          include_dirs=driver.include_dirs, libraries=driver.libraries))
target = pathlib.Path(sys.argv[1]) / built.name
shutil.copyfile(built, target)
pathlib.Path(f"{target}.driver-sha256").write_text(hashlib.sha256(source.read_bytes()).hexdigest())
print(target.name, target.stat().st_size)
EOF
done
