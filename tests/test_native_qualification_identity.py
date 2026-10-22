"""Actual wheel binaries and their presence: the provenance a table records, and the per-arch
kernel identity a verdict binds to; no GPU."""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]


def inspect(root: Path) -> dict[str, Any]:
    program = """import json
from cozy_runtime.internal import prebuilt
from cozy_runtime.internal.encoding import selection
kernels = {sm: prebuilt.gemm_kernel("int8", "mma.sync", sm).build for sm in (86, 89, 120)}
print(json.dumps({"source":selection.__file__,"kernels":kernels,"backends":selection.native_backend_identity()}))
"""
    done = subprocess.run(
        [sys.executable, "-c", program],
        cwd=root,
        env={**os.environ, "PYTHONPATH": str(root), "CUDA_VISIBLE_DEVICES": ""},
        capture_output=True,
        text=True,
        check=True,
    )
    result = json.loads(done.stdout)
    assert Path(result["source"]).is_relative_to(root)
    return result


def test_built_native_pure_unloadable_and_foreign_extension_have_distinct_identity(
    tmp_path: Path,
) -> None:
    pure = next((ROOT / "dist").glob("*-py3-none-any.whl"), None)
    native = next((ROOT / "dist").glob("*-manylinux*.whl"), None)
    if pure is None or native is None:
        pytest.skip("run after building both actual wheel variants (dev-build wheels)")
    for label, wheel in (("pure", pure), ("native", native)):
        with zipfile.ZipFile(wheel) as archive:
            archive.extractall(tmp_path / label)
    without, actual = inspect(tmp_path / "pure"), inspect(tmp_path / "native")
    assert without["backends"]["_gemm"] == without["backends"]["_norm"] == "ABSENT"
    assert actual["backends"]["platform"] == f"{platform.system()}/{platform.machine()}"
    assert "sha256:" in actual["backends"]["_gemm"]
    # Each family's verdict binds to the cubins its card runs: distinct per arch, absent without
    # a native module.
    assert len(set(actual["kernels"].values())) == 3
    assert all(build.startswith("sha256:") for build in actual["kernels"].values())
    assert set(without["kernels"].values()) == {""}

    # These are real temporary files and real loader failures, not mocked modules. Renaming
    # a host binary to a foreign extension only proves non-loadability here, not Windows execution.
    for label in ("broken", "foreign"):
        shutil.copytree(tmp_path / "native", tmp_path / label)
    broken = next((tmp_path / "broken" / "cozy_runtime" / "_kernels").glob("_gemm*.so"))
    broken.write_bytes(b"invalid native module for negative loader proof")
    foreign = next((tmp_path / "foreign" / "cozy_runtime" / "_kernels").glob("_gemm*.so"))
    foreign.rename(foreign.with_name("_gemm.foreign.pyd"))
    failed, other = inspect(tmp_path / "broken"), inspect(tmp_path / "foreign")
    assert failed["backends"]["_gemm"].startswith("unavailable:sha256:")
    assert other["backends"]["_gemm"].startswith("unavailable:sha256:")
    assert failed["kernels"] != actual["kernels"]
