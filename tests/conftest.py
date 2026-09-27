"""Shared test plumbing."""

from __future__ import annotations

import atexit
import functools
import shutil
import tempfile
from pathlib import Path


@functools.cache
def image_python() -> Path:
    """A REAL base interpreter that owns the runtime closure in its own site-packages —
    the shape a worker image's python actually has.

    A venv python cannot stand in: `uv venv --python <venv-python>` resolves the venv's
    BASE as the new environment's base, so nothing installed in a venv is ever visible to
    a `--system-site-packages` child. The one faithful stand-in is a private copy of the
    real interpreter prefix with the runtime wheel installed into its own site-packages.
    Built once per test session, in temp, deleted with it."""

    import atexit
    import shutil
    import subprocess
    import sys
    import tempfile

    base = Path(getattr(sys, "_base_executable", sys.executable)).resolve()
    prefix = base.parent.parent
    root = Path(tempfile.mkdtemp(prefix="cozy-image-prefix."))
    atexit.register(shutil.rmtree, root, ignore_errors=True)
    copy = root / "python"
    shutil.copytree(prefix, copy, symlinks=True)
    python = copy / "bin" / "python3"
    assert python.is_file(), python
    # The copy is ours to modify; the PEP 668 marker guards the shared uv-managed original.
    for marker in copy.glob("lib/python*/EXTERNALLY-MANAGED"):
        marker.unlink()
    uv = shutil.which("uv")
    assert uv is not None
    wheels = root / "wheels"
    subprocess.run(
        [uv, "build", "--wheel", "--out-dir", str(wheels), str(Path(__file__).parent.parent)],
        check=True,
        capture_output=True,
    )
    runtime_wheel = next(wheels.glob("cozy_runtime-*.whl"))
    subprocess.run(
        [uv, "pip", "install", "--python", str(python), str(runtime_wheel)],
        check=True,
        capture_output=True,
    )
    return python


@functools.cache
def authored_python() -> Path:
    """An explicit author SDK venv, suitable for complete immutable source capture."""
    import shutil
    import subprocess

    image = image_python()
    root = image.parent.parent.parent
    sdk = root / "author-sdk"
    uv = shutil.which("uv")
    assert uv is not None
    subprocess.run([uv, "venv", "--python", str(image), str(sdk)], check=True, capture_output=True)
    wheel = next((root / "wheels").glob("cozy_runtime-*.whl"))
    python = sdk / "bin/python"
    subprocess.run(
        [uv, "pip", "install", "--python", str(python), str(wheel)], check=True, capture_output=True
    )
    return python


@functools.cache
def sample_python() -> Path:
    """Install the sample project so executor tests never borrow their working directory."""
    import shutil
    import subprocess

    image = image_python()
    root = image.parent.parent.parent
    sdk = root / "sample-sdk"
    uv = shutil.which("uv")
    assert uv is not None
    subprocess.run([uv, "venv", "--python", str(image), str(sdk)], check=True, capture_output=True)
    wheel = next((root / "wheels").glob("cozy_runtime-*.whl"))
    python = sdk / "bin/python"
    project = Path(__file__).parent.parent / "examples/marco-polo"
    subprocess.run(
        [uv, "pip", "install", "--python", str(python), str(wheel), str(project)],
        check=True,
        capture_output=True,
    )
    return python


@functools.cache
def mypy_cache() -> Path:
    """One mypy cache per session: every generated caller type-checks against the same
    `cozy_runtime`, so only the first check pays for analysing it."""
    root = Path(tempfile.mkdtemp(prefix="cozy-mypy-cache."))
    atexit.register(shutil.rmtree, root, ignore_errors=True)
    return root
