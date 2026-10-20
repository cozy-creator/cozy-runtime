# /// script
# requires-python = ">=3.12"
# dependencies = ["cozy-runtime==0.27.4"]
# ///
"""Measure the real store and assigned GPUs before the model matrix.

Run through cozy run --on RENTAL --gpus N before the model matrix. The child
uses the supplied TensorFS wheel in isolation;
the outer Runtime and worker installation are not changed by this script.
TensorFS owns both probes and keeps their results in the sealed shared store.
"""

import json
import os
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Annotated

from cozy_runtime.author import AssetBound, FileAsset, Outputs, Tree

CHILD = r"""
import ctypes
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, sys.argv[1])
import tensorfs
from tensorfs.varena import ceiling

wheel = Path(sys.argv[1]).resolve()
assert Path(tensorfs.__file__).resolve().is_relative_to(wheel), tensorfs.__file__
home = Path(os.environ['TENSORFS_HOME']).resolve(strict=True)
cuda = ctypes.CDLL('libcuda.so.1')
def checked(name, *args):
    result = getattr(cuda, name)(*args)
    if result:
        raise RuntimeError(f'{name} failed with CUDA {result}')
checked('cuInit', 0)
count = ctypes.c_int()
checked('cuDeviceGetCount', ctypes.byref(count))
assert 1 <= count.value <= 2, f'expected one or two assigned GPUs, got {count.value}'
devices = []
# A private, non-sparse fixture measures the same volume even before the first
# model arrives. TensorFS performs the actual O_DIRECT reads and H2D probes.
scratch = home / 'tmp'
scratch.mkdir(exist_ok=True)
with tempfile.TemporaryDirectory(prefix='slim-ceiling-', dir=scratch) as folder:
    sample = Path(folder) / 'probe.bin'
    with sample.open('wb') as stream:
        for _ in range(32):
            stream.write(os.urandom(8 << 20))
        stream.flush()
        os.fdatasync(stream.fileno())
    for device in range(count.value):
        uuid = (ctypes.c_ubyte * 16)()
        checked('cuDeviceGetUuid', ctypes.byref(uuid), device)
        measured = ceiling(device=device, path=sample)
        assert measured.disk_direct, 'the store filesystem refused uncached disk measurement'
        assert measured.disk_gbps > 0 and measured.h2d_gbps > 0
        assert measured.streams in (1, 2, 4), measured.streams
        devices.append({
            'device': device, 'uuid': bytes(uuid).hex(),
            'disk_gbps': measured.disk_gbps, 'disk_direct': measured.disk_direct,
            'h2d_gbps': measured.h2d_gbps, 'streams': measured.streams,
        })
print(json.dumps({
    'tensorfs': importlib.metadata.version('tensorfs'), 'store': str(home),
    'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
    'devices': devices,
}))
"""


def main(
    *,
    tensorfs_wheel: Annotated[FileAsset, AssetBound(max_bytes=268435456)],
    out: Outputs,
) -> Tree:
    if not os.environ.get("TENSORFS_HOME"):
        raise RuntimeError("The worker must provide its sealed shared TENSORFS_HOME")
    root = Path(tempfile.mkdtemp(prefix="slim-ceilings-"))
    wheel_file = root / "tensorfs.whl"
    wheel_file.write_bytes(tensorfs_wheel.read_bytes())
    extracted = root / "wheel"
    with zipfile.ZipFile(wheel_file) as archive:
        archive.extractall(extracted)
    process = subprocess.run(
        [sys.executable, "-I", "-c", CHILD, str(extracted)],
        text=True,
        capture_output=True,
        timeout=300,
    )
    print(process.stdout, end="", flush=True)
    print(process.stderr, end="", file=sys.stderr, flush=True)
    process.check_returncode()
    measured = json.loads(process.stdout)
    if not str(measured["tensorfs"]).startswith("0.13."):
        raise RuntimeError(f"Expected the candidate TensorFS 0.13 wheel: {measured}")
    result = root / "results"
    result.mkdir()
    (result / "ceilings.json").write_text(json.dumps(measured, indent=2) + "\n")
    return out.save_tree(result)
