# /// script
# requires-python = ">=3.12"
# dependencies = ["cozy-runtime>=0.30", "torch"]
# ///
"""The offline kernel oracle on one rented card (`scripts/kernel-verdicts.sh` runs it).

    cozy run kernel-verdicts.py --on <rental> --gpus 1 --asset runtime=<native wheel>
        --asset tensorfs=<wheel> --out <dir>

Installs the exact candidate wheels into a child's path and runs `python -m
cozy_runtime.cli.qualify`: the decode suite's table and the card's kernel verdicts, rules and
raw calibration (`internal/calibrate.py`). Returns them with the manifest that pins what ran.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Annotated

from cozy_runtime.author import AssetBound, FileAsset, Outputs, ScriptContext, Tree

FACTS = """import json, torch, tensorfs, cozy_runtime
from cozy_runtime import _kernels
from cozy_runtime._build_provenance import COMMIT
from cozy_runtime.internal.encoding import measure_runtime
assert not _kernels.ABSENT, _kernels.ABSENT
print(json.dumps({"torch": torch.__version__, "tensorfs": tensorfs.__version__,
    "runtime": cozy_runtime.__version__, "commit": COMMIT, "gpu": torch.cuda.get_device_name(),
    "identity": measure_runtime(torch, cozy_runtime.__version__).document()}))
"""


def wheel(asset: FileAsset, into: Path) -> tuple[Path, str]:
    """The asset under its own wheel filename (pip takes no other), and its sha256."""
    data = asset.read_bytes()
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        meta = next(n for n in archive.namelist() if n.endswith(".dist-info/WHEEL"))
        tags = archive.read(meta).decode().splitlines()
        tag = next(line[5:] for line in tags if line.startswith("Tag: "))
    path = into / f"{meta.split('.dist-info/', 1)[0]}-{tag}.whl"
    path.write_bytes(data)
    return path, hashlib.sha256(data).hexdigest()


def main(
    ctx: ScriptContext,
    *,
    runtime: Annotated[FileAsset, AssetBound(max_bytes=1 << 28)],
    tensorfs: Annotated[FileAsset, AssetBound(max_bytes=1 << 28)],
    out: Outputs,
) -> Tree:
    root = Path(tempfile.mkdtemp(prefix="kernel-verdicts-"))
    results, site = root / "results", root / "site"
    results.mkdir()
    (runtime_wheel, runtime_sha), (tensorfs_wheel, tensorfs_sha) = (
        wheel(runtime, root),
        wheel(tensorfs, root),
    )
    install = [shutil.which("uv") or "uv", "pip", "install", "--python", sys.executable]
    subprocess.run(
        [*install, "--target", str(site), "--no-deps", str(runtime_wheel), str(tensorfs_wheel)],
        check=True,
    )
    env = {**os.environ, "PYTHONPATH": str(site)}
    facts = subprocess.run(
        [sys.executable, "-c", FACTS], cwd=root, env=env, check=True, capture_output=True, text=True
    )
    manifest = {
        "runtime_wheel_sha256": runtime_sha,
        "tensorfs_wheel_sha256": tensorfs_sha,
        "installed": json.loads(facts.stdout.strip().splitlines()[-1]),
        "nvidia_smi": subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout,
    }
    command = [sys.executable, "-u", "-m", "cozy_runtime.cli.qualify", str(results / "tables")]
    ctx.log(f"DEVELOPMENT ORACLE {' '.join(command)}", level="warn")
    with (
        (results / "oracle.log").open("w") as log,
        subprocess.Popen(
            command, cwd=root, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        ) as child,
    ):
        for line in child.stdout or ():
            log.write(line)
            log.flush()
            ctx.log(line.rstrip()[:400], level="warn")
    manifest["exit_code"] = child.returncode
    (results / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    ctx.log(f"ORACLE exit={child.returncode} runtime={runtime_sha[:16]}", level="warn")
    return out.save_tree(results)
