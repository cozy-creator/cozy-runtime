"""Plain-main interfaces are read without importing scripts or captured libraries."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from cozy_runtime.author import ConformanceError
from cozy_runtime.cli.main import parse
from cozy_runtime.internal import package_interface, static_interface
from cozy_runtime.internal.discovery import discover
from cozy_runtime.internal.static_interface import StaticRefusal


def project(root: Path, code: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "package.toml").write_text('[application]\nobject="cozy_script_entry:app"\n')
    (root / "cozy_script_entry.py").write_text(
        'from cozy_runtime.author import script_app\napp = script_app("cozy_script")\n'
    )
    (root / "cozy_script.py").write_text(code)
    return root


@pytest.mark.parametrize(
    "body",
    [
        "def main(): pass\n",
        "from typing import Annotated\n"
        "from cozy_runtime.author import AssetBound, FileAsset, Tree\n"
        "def main(ctx, *, report: Annotated[FileAsset, AssetBound(max_bytes=200000, "
        "media_types=('application/json',))], files: Tree) -> Tree:\n"
        "    raise RuntimeError('body must not execute')\n",
        "async def main(ctx): ctx.log('started')\n",
        "from cozy_runtime.author import ImageAsset, Outputs, Telemetry\n"
        "def main(*, out: Outputs, tel: Telemetry) -> ImageAsset: raise RuntimeError('body')\n",
        "from typing import Annotated\n"
        "from cozy_runtime.author import AssetBound, ImageAsset, Outputs\n"
        "def main(*, out: Outputs) -> Annotated[ImageAsset, "
        "AssetBound(media_types=('image/png',))]:\n"
        "    raise RuntimeError('body')\n",
        "# /// script\n# [tool.cozy.weights]\n# checkpoint=4096\n# [tool.cozy.models]\n"
        "# source='owner/model'\n# ///\n"
        "from cozy_runtime.author import Context, Model, ModelArtifact, Telemetry\n"
        "def main(ctx: Context, *, source: Model[object],\n"
        "         tel: Telemetry) -> ModelArtifact:\n"
        "    raise RuntimeError('body')\n",
        "# /// script\n# [tool.cozy.weights]\n# checkpoint=4096\n# ///\n"
        "from cozy_runtime.author import Context, ModelArtifact\n"
        "def main(ctx: Context) -> ModelArtifact:\n"
        "    raise RuntimeError('body')\n",
    ],
)
def test_static_main_equals_execution_adapter(tmp_path: Path, body: str) -> None:
    root = project(tmp_path, body)
    modules = ("cozy_script", "cozy_script_entry")
    previous = {name: sys.modules.pop(name) for name in modules if name in sys.modules}
    try:
        expected = package_interface.build(discover(root))
    finally:
        for name in modules:
            sys.modules.pop(name, None)
        sys.modules.update(previous)
    actual = static_interface.build(root)
    assert package_interface.canonical_bytes(actual) == package_interface.canonical_bytes(expected)


def test_captured_model_source_and_pth_are_never_executed(tmp_path: Path) -> None:
    marker = tmp_path / "executed"
    root = project(
        tmp_path / "project",
        "from library import Probe\n"
        "from cozy_runtime.author import Outputs, VideoAsset\n"
        "raise RuntimeError('script must not execute')\n"
        "def main(ctx, *, model: Probe, out: Outputs) -> list[VideoAsset]:\n"
        "    raise RuntimeError('body must not execute')\n",
    )
    venv = tmp_path / "venv"
    binary = venv / "bin/python"
    binary.parent.mkdir(parents=True)
    binary.write_text(f'#!/bin/sh\ntouch "{marker}"\nexit 99\n')
    binary.chmod(0o700)
    (venv / "pyvenv.cfg").write_text("home = /unused\n")
    site = venv / "lib/python3.12/site-packages"
    site.mkdir(parents=True)
    library = tmp_path / "captured-library"
    library.mkdir()
    (library / "library.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).touch()\n"
        "import unavailable_accelerator_dependency\n"
        "from cozy_runtime.author import Model, uses_components\n"
        "class Probe(Model[object], encoded_leaves='accept'):\n"
        "    @uses_components('video_vae')\n"
        "    def decode(self): pass\n"
    )
    (site / "captured.pth").write_text(
        f"import pathlib; pathlib.Path({str(marker)!r}).touch()\n{library}\n"
    )
    document = static_interface.build(root, environment_python=binary)
    jobs = document["jobs"]
    assert isinstance(jobs, list) and isinstance(jobs[0], dict)
    models = jobs[0]["models"]
    assert isinstance(models, list) and isinstance(models[0], dict)
    model = models[0]
    assert model["class"] == "Probe"
    assert model["component_use"] == {"decode": ["video_vae"]}
    assert model["encoded_leaves"] == "accept"
    assert not marker.exists()
    assert "library" not in sys.modules
    assert "unavailable_accelerator_dependency" not in sys.modules
    command = subprocess.run(
        [
            str(Path(sys.executable).with_name("cozy-runtime")),
            "--json",
            "--dir",
            str(root),
            "describe",
            "--environment-python",
            str(binary),
        ],
        cwd="/",
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert command.returncode == 0, command.stderr
    assert json.loads(command.stdout) == document
    assert not marker.exists()
    with pytest.raises(StaticRefusal):
        static_interface.build(root)  # No ambient sys.path/host environment discovery.


@pytest.mark.parametrize(
    "body",
    [
        "from cozy_runtime.author import Model\ndef main(*, model=Model): pass\n",
        "def main(*args): pass\n",
        "# /// script\n# tool=1\n# ///\ndef main(): pass\n",
        "def main():\n    yield 1\n",
        "from cozy_runtime.author import Model\nclass Local(Model[object]): pass\n"
        "def main(*, model: Local): pass\n",
        "from cozy_runtime.author import Model\ndef main(*, model: Model[factory()]): pass\n",
        "from typing import Annotated\n"
        "from cozy_runtime.author import AssetBound, ImageAsset\n"
        "from external import formats\n"
        "def main() -> Annotated[ImageAsset, AssetBound(media_types=formats())]: pass\n",
        "# /// script\n# [tool.cozy.weights]\n# checkpoint=1\n# ///\ndef main(): pass\n",
        "from cozy_runtime.author import WeightsSink\ndef main(*, weights: WeightsSink): pass\n",
        "# /// script\n# [tool.cozy.weights]\n# checkpoint=4096\n# ///\n"
        "from cozy_runtime.author import Context, WeightsReader\n"
        "def main(ctx: Context, *, source: WeightsReader): pass\n",
        "# /// script\n# [tool.cozy.weights]\n# checkpoint=4096\n# ///\n"
        "from cozy_runtime.author import Context, Outputs\n"
        "def main(ctx: Context, *, out: Outputs): pass\n",
    ],
)
def test_static_script_refuses_unresolvable_or_mismatched_contract(
    tmp_path: Path, body: str
) -> None:
    with pytest.raises(ConformanceError) as caught:
        static_interface.build(project(tmp_path, body))
    assert getattr(caught.value, "code", "").startswith(("static_", "script_"))


def test_cli_environment_python_is_a_describe_source_option() -> None:
    verb, args, opts, _ = parse(["describe", "--environment-python", "/owned/venv/bin/python"])
    assert verb == "describe" and not args
    assert opts.environment_python == "/owned/venv/bin/python"


def test_venv_pth_cannot_extend_reads_outside_its_retained_root(tmp_path: Path) -> None:
    owned = tmp_path / "retained"
    root = project(owned / "project", "def main(): pass\n")
    prefix = owned / "venv"
    (prefix / "bin").mkdir(parents=True)
    binary = prefix / "bin/python"
    binary.touch()
    (prefix / "pyvenv.cfg").touch()
    site = prefix / "lib/python3.12/site-packages"
    site.mkdir(parents=True)
    outside = tmp_path / "another-install"
    outside.mkdir()
    (site / "outside.pth").write_text(str(outside) + "\n")
    with pytest.raises(StaticRefusal, match="escapes"):
        static_interface.build(root, environment_python=binary)
