"""Exercise real Python package shadowing, without replacing attention arithmetic."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from cozy_runtime.internal import attention


def test_matching_loaded_diffusers_registry_is_accepted() -> None:
    diffusers = pytest.importorskip("diffusers")
    if diffusers.__version__ != attention.DIFFUSERS_VERSION:
        pytest.skip("requires the reviewed Diffusers package")
    registry = pytest.importorskip("diffusers.models.attention_dispatch")
    assert attention._registry() is registry._AttentionBackendRegistry


def test_imported_version_cannot_shadow_distribution_metadata(
    tmp_path: Path,
) -> None:
    diffusers = pytest.importorskip("diffusers")
    if diffusers.__version__ != attention.DIFFUSERS_VERSION:
        pytest.skip("requires the reviewed Diffusers package")
    original = Path(diffusers.__file__).parent
    shadow = tmp_path / "diffusers"
    shadow.mkdir()
    # Reuse every real module, changing only the shadow package's version declaration.
    # Distribution metadata remains the installed 0.40 package; no kernel is mocked.
    for child in original.iterdir():
        if child.name not in {"__init__.py", "__pycache__"}:
            (shadow / child.name).symlink_to(child, target_is_directory=child.is_dir())
    text = (original / "__init__.py").read_text()
    declaration = f'__version__ = "{attention.DIFFUSERS_VERSION}"'
    assert declaration in text
    (shadow / "__init__.py").write_text(text.replace(declaration, '__version__ = "0.41.0.dev0"'))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")
    env["PYTHONPATH"] = os.pathsep.join(
        (str(tmp_path), str(Path(__file__).resolve().parents[1] / "src"))
    )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import json, importlib.metadata, diffusers
from cozy_runtime.internal import attention
facts = {'installed': importlib.metadata.version('diffusers'),
         'loaded': diffusers.__version__, 'path': diffusers.__file__}
try:
    attention._registry()
    facts['code'] = 'accepted'
except attention.AttentionRefusal as error:
    facts.update(code=error.code, detail=str(error))
print(json.dumps(facts))
""",
        ],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    facts = json.loads(result.stdout)
    assert facts["installed"] == attention.DIFFUSERS_VERSION
    assert facts["loaded"] == "0.41.0.dev0" and facts["path"].startswith(str(shadow))
    assert facts["code"] == "attention_kernel_absent", facts
    assert "0.41.0.dev0" in facts["detail"] and str(shadow) in facts["detail"]
