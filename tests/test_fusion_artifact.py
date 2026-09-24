"""Native launcher packaging and installation without loading model weights."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from cozy_runtime.internal import fusion, fusion_build

triton = pytest.importorskip("triton")
if fusion.TRITON_ABSENT or triton.__version__ != fusion.TRITON_VERSION:
    pytest.skip("the pinned Triton compiler is required", allow_module_level=True)


@pytest.fixture(scope="module")
def built_artifact(tmp_path_factory: pytest.TempPathFactory) -> Path:
    # The image builder provides CUDA's link stub; a development host provides
    # libcuda itself. Neither path needs a CUDA device or a model.
    from triton.backends.nvidia import driver

    try:
        driver.library_dirs()
    except (AssertionError, RuntimeError) as exc:
        pytest.skip(f"a CUDA driver or link stub is required: {exc}")
    out = tmp_path_factory.mktemp("fusion-launcher") / "artifact"
    fusion_build.build(out, (90,))
    return out


def test_archive_roundtrip_installs_native_launcher_without_a_compiler(
    built_artifact: Path, tmp_path: Path
) -> None:
    archive = shutil.make_archive(str(tmp_path / "artifact"), "zip", built_artifact)
    restored = tmp_path / "restored"
    shutil.unpack_archive(archive, restored)
    manifest = fusion.check_artifact(restored, 90)
    module = fusion._utils_artifact(restored, manifest["utils"])
    assert module.read_bytes()[:4] == b"\x7fELF"
    proof = subprocess.run(
        [
            sys.executable,
            "-c",
            "from pathlib import Path; import json, sys, triton; "
            "from cozy_runtime.internal import fusion; "
            "from triton.backends.nvidia import driver; "
            "root=Path(sys.argv[1]); triton.knobs.cache.dir=sys.argv[2]; "
            "triton.knobs.build.cc='/compiler-must-not-run'; "
            "manifest=json.loads((root/'manifest.json').read_text()); "
            "fusion._place_utils(root,manifest); driver.CudaUtils(); "
            "print(fusion.utils_cache_dir())",
            str(restored),
            str(tmp_path / "cold-cache"),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    installed = Path(proof.stdout.strip()) / module.name
    assert installed.read_bytes() == module.read_bytes()


def test_verifier_rejects_missing_or_empty_launcher_before_gpu_use(
    built_artifact: Path, tmp_path: Path
) -> None:
    out = tmp_path / "incomplete"
    shutil.copytree(built_artifact, out)
    manifest = json.loads((out / fusion.MANIFEST).read_text())
    launcher = fusion._utils_artifact(out, manifest["utils"])
    launcher.unlink()
    for missing in (True, False):
        if not missing:
            launcher.touch()
        with pytest.raises(fusion.FusionUnavailable) as refused:
            fusion.check_artifact(out, 90)
        assert refused.value.code == "fusion_kernels_corrupt"
        assert "launcher" in str(refused.value)
