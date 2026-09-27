"""The image kernel site is placed after the package's own packages, and only for its build.

Each arm runs in a fresh interpreter, because placing a site is a process-wide import-path
change: exactly what the executor does once at entry.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

PROBE = """
import json, sys
from pathlib import Path
from cozy_runtime.internal import attention, kernel_site
from cozy_runtime.internal.encoding import DeviceFacts
placed = kernel_site.activate(Path(sys.argv[1]))
out = {"site": placed.document(), "path": sys.path}
try:
    import cozy_site_kernel
    out["kernel"] = cozy_site_kernel.__file__
except ImportError:
    out["kernel"] = ""
try:
    import cozy_pth_kernel
    out["pth"] = cozy_pth_kernel.__file__
except ImportError:
    out["pth"] = ""
import packaging
out["packaging"] = packaging.__file__
try:
    attention.pinned("kitchen-int8", DeviceFacts("cuda", "H100", 90, "", "", 0))
except attention.AttentionRefusal as exc:
    out["refusal"] = [exc.code, str(exc)]
print(json.dumps(out))
"""


def _image(root: Path, **manifest: str) -> Path:
    site = root / "site"
    (site / "extra").mkdir(parents=True)
    (site / "cozy_site_kernel.py").write_text("")
    # A package the environment already has: the package's own copy must win.
    (site / "packaging").mkdir()
    (site / "packaging" / "__init__.py").write_text("")
    # The CuTe DSL wheels expose `cutlass` through an import line in a .pth file.
    (site / "extra" / "cozy_pth_kernel.py").write_text("")
    (site / "cozy.pth").write_text(f"import sys; sys.path.insert(0, {str(site / 'extra')!r})\n")
    if manifest:
        (root / "manifest.json").write_text(json.dumps(manifest))
    return root


#: The CUDA torch build the probe's environment carries. `torch_build` reads files and never
#: imports torch, so a dist-info and `torch/version.py` are the whole build on any host.
TORCH, CUDA = "2.14.0", "13.0"


def _torch(root: Path) -> Path:
    environment = root / "cuda-torch"
    (environment / "torch").mkdir(parents=True)
    (environment / "torch" / "version.py").write_text(
        f"__version__ = '{TORCH}+cu130'\ncuda = '{CUDA}'\n"
    )
    info = environment / f"torch-{TORCH}+cu130.dist-info"
    info.mkdir()
    (info / "METADATA").write_text(f"Metadata-Version: 2.1\nName: torch\nVersion: {TORCH}+cu130\n")
    return environment


def _run(root: Path) -> dict[str, object]:
    environment = os.pathsep.join(filter(None, (str(_torch(root)), os.environ.get("PYTHONPATH"))))
    result = subprocess.run(
        [sys.executable, "-c", PROBE, str(root)],
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "PYTHONPATH": environment},
    )
    placed: dict[str, object] = json.loads(result.stdout)
    return placed


def _built() -> dict[str, str]:
    python = f"cp{sys.version_info.major}{sys.version_info.minor}"
    return {"torch": TORCH + "+cu", "cuda": CUDA, "python_abi": python}


def test_a_matching_site_is_appended_after_the_package_environment(tmp_path: Path) -> None:
    placed = _run(_image(tmp_path, site=str(tmp_path / "site"), **_built()))
    site = str(tmp_path / "site")
    assert placed["site"] == {**placed["site"], "path": site, "reason": ""}  # type: ignore[dict-item]
    path = placed["path"]
    assert isinstance(path, list)
    own = next(i for i, entry in enumerate(path) if entry.endswith("site-packages"))
    assert path.index(site) > own
    assert placed["kernel"] == str(tmp_path / "site" / "cozy_site_kernel.py")
    assert placed["pth"] == str(tmp_path / "site" / "extra" / "cozy_pth_kernel.py")
    assert not str(placed["packaging"]).startswith(site)
    # A kernel it does not ship is absent by name, and the refusal says the site was placed.
    pytest.importorskip("diffusers")
    refusal = placed["refusal"]
    assert isinstance(refusal, list)
    code, detail = refusal
    assert code == "attention_kernel_absent" and f"kernel site {site}" in detail


def test_another_torch_build_places_nothing_and_says_why(tmp_path: Path) -> None:
    placed = _run(_image(tmp_path, site="site", **{**_built(), "torch": "9.9.9"}))
    site = placed["site"]
    assert isinstance(site, dict) and site["path"] == ""
    assert "built for torch 9.9.9" in site["reason"]
    assert str(tmp_path / "site") not in placed["path"]  # type: ignore[operator]
    assert placed["kernel"] == "" and placed["pth"] == ""
    pytest.importorskip("diffusers")
    assert "built for torch 9.9.9" in placed["refusal"][1]  # type: ignore[index]


def test_an_image_without_a_manifest_or_with_a_bad_one_places_nothing(tmp_path: Path) -> None:
    absent = _run(_image(tmp_path / "absent"))
    assert "bakes no" in absent["site"]["reason"]  # type: ignore[index]
    unreadable = _image(tmp_path / "bad")
    (unreadable / "manifest.json").write_text("{")
    assert "unreadable" in _run(unreadable)["site"]["reason"]  # type: ignore[index]
    partial = _run(_image(tmp_path / "partial", site="site"))
    assert "names no torch/cuda build" in partial["site"]["reason"]  # type: ignore[index]
    other = _run(_image(tmp_path / "abi", site="site", **{**_built(), "python_abi": "cp399"}))
    assert "cp399" in other["site"]["reason"]  # type: ignore[index]


def test_torch_is_read_from_its_files_and_never_imported() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from cozy_runtime.internal import kernel_site; "
            "print(kernel_site.torch_build(), 'torch' in sys.modules)",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip().endswith("False")
