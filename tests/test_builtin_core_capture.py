"""Core-only SDK capture resolves numerical dependencies only on the owned call path."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any

from test_builtin_operation_carrier import carrier
from test_installed_dependencies import _distribution


def test_core_sdk_capture_is_offline_and_later_preparation_uses_captured_code(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[1]
    wheels = tmp_path / "wheels"
    subprocess.run(
        ["uv", "build", "--wheel", "--out-dir", str(wheels)],
        cwd=root,
        check=True,
        capture_output=True,
    )
    wheel = next(wheels.glob("*.whl"))
    core = tmp_path / "core"
    subprocess.run(
        ["uv", "venv", str(core), "--python", sys.executable], check=True, capture_output=True
    )
    subprocess.run(
        ["uv", "pip", "install", "--python", str(core / "bin/python"), str(wheel), "uv==0.12.11"],
        check=True,
        capture_output=True,
    )
    env = dict(os.environ)
    env["PATH"] = str(core / "bin") + os.pathsep + env.get("PATH", "")
    env.pop("UV_OFFLINE", None)

    def python(executable: str, source: str) -> str:
        return subprocess.check_output([executable, "-I", "-c", source], text=True, env=env).strip()

    probe = """
import importlib.util
assert importlib.util.find_spec('numpy') is None
from cozy_runtime.derive.operations import app
from cozy_runtime.internal import builtin_operations
from cozy_runtime.internal.executor import Executor
broker = Executor._call_broker(object.__new__(Executor), {
    'request_id': 'core-only-proof', 'call_interfaces': builtin_operations.rows()})
import sys
assert broker is not None and 'numpy' not in sys.modules
"""
    python(str(core / "bin/python"), probe)
    version = python(
        str(core / "bin/python"), "import importlib.metadata as m; print(m.version('cozy-runtime'))"
    )
    capsule = carrier(tmp_path / "carrier", version=version)
    site = next(core.glob("lib/python*/site-packages"))
    # Creator can select a root package extra during uv sync, but the local
    # preparation message currently carries wheels, not that extra selection.
    # A real authored capture must retain the installed optional dependency.
    _distribution(site, "snapshotextra", {"snapshotextra.py": b"value = 47\n"})
    extra_root = tmp_path / "extra-wheel"
    _distribution(
        extra_root,
        "snapshotproject",
        {
            "snapshotproject.py": b"from snapshotextra import value\n",
            "snapshotproject-1.0.dist-info/WHEEL": (
                b"Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n"
            ),
        },
        requirements=('snapshotextra; extra == "cuda"',),
    )
    extra_wheel = tmp_path / "snapshotproject-1.0-py3-none-any.whl"
    with zipfile.ZipFile(extra_wheel, "w") as archive:
        for member in sorted(extra_root.rglob("*")):
            if member.is_file():
                archive.write(member, str(member.relative_to(extra_root)))
    authored_python = python(
        str(core / "bin/python"),
        "from pathlib import Path; import sys; "
        "from cozy_runtime.internal import package_environment as p; "
        f"path = Path({str(extra_wheel)!r}); base = p.observe_base(); "
        "wheel, verified = p.measure_wheel(path, 'project', base); "
        "source = p.Input('sha256:' + 'a'*64, wheel.digest, (wheel,), "
        "(p.BlobGrant(wheel.digest, wheel.length, path, verified),)); "
        f"result = p.materialize(source, Path({str(tmp_path / 'authored')!r}), "
        "base=base, dependency_python=Path(sys.executable), exact_dependencies=True); "
        "print(result.python)",
    )
    (site / "snapshotextra.py").write_text("raise AssertionError('original extra used')\n")
    python(authored_python, "from snapshotproject import value; assert value == 47")
    # Restore its recorded bytes; the following builtin should omit it entirely.
    (site / "snapshotextra.py").write_text("value = 47\n")
    unrelated = _distribution(site, "unrelated-sdk-tool", {"unused.dat": b"not needed"})
    # Deliberately absent bytes prove capture never tries copying an unrelated
    # installed dependency; the real Runtime closure still gets captured below.
    (site / "unused.dat").unlink()

    def operation(
        executable: str, verb: str, target: Path, *, offline: bool = False
    ) -> dict[str, Any]:
        result = subprocess.run(
            [
                str(Path(executable).with_name("cozy-runtime")),
                "--json",
                verb,
                "operations",
                "--wheel",
                str(capsule),
                "--out",
                str(target),
            ],
            capture_output=True,
            text=True,
            env={**env, **({"UV_OFFLINE": "1"} if offline else {})},
        )
        assert result.returncode == 0, result.stderr + result.stdout
        value = json.loads(result.stdout)
        assert isinstance(value, dict)
        return value

    captured = operation(
        str(core / "bin/python"), "builtin-capture", tmp_path / "captured", offline=True
    )
    assert "unrelated-sdk-tool" not in captured["closure"]
    assert "snapshotextra" not in captured["closure"]
    assert not (
        Path(captured["environment_python"]).parent.parent / unrelated.relative_to(core)
    ).exists()
    python(captured["environment_python"], probe)
    # Replace code in the original SDK after capture. The later operation must use
    # the retained snapshot; source freezing is byte custody, not a version check.
    original = next(core.glob("lib/python*/site-packages/cozy_runtime/derive/operations.py"))
    original.write_text("raise AssertionError('mutable original SDK was used')\n")
    prepared = operation(captured["environment_python"], "builtin-prepare", tmp_path / "prepared")
    assert prepared["implementation_digest"] == captured["implementation_digest"]
    assert prepared["content_digest"] != captured["content_digest"]
    python(captured["environment_python"], probe)
    python(
        prepared["environment_python"],
        "import numpy; from cozy_runtime.derive.operations import app; "
        "from cozy_runtime.author import describe; assert describe(app)",
    )
    assert "numpy==" in prepared["closure"] and "numpy==" not in captured["closure"]
    manifest = next((tmp_path / "prepared/contents").glob("*/installed-dependencies.json"))
    snapshot = json.loads(manifest.read_bytes())
    assert any(row["distribution"] == "numpy" for row in snapshot["distributions"])
    assert any("/numpy/" in row["path"] for row in snapshot["files"])
