"""The native-wheel proof reads the environment `cozy package install` left: a generation root
as TensorD publishes it (`generation.json`, `env/`, `.hold`), held while the proof runs."""

from __future__ import annotations

import json
import sys
import venv
from pathlib import Path

import pytest

from cozy_runtime.cli import native_wheel_proof
from cozy_runtime.internal import filelock
from cozy_runtime.internal.package_environment import EnvironmentRefusal

IDENTITY = "0123456789abcdef0123456789abcdef"


def _generation(generations: Path, identity: str = IDENTITY, named: str = IDENTITY) -> Path:
    """A generation directory laid out as TensorD's installer leaves it."""
    root = generations / identity
    venv.EnvBuilder(symlinks=sys.platform != "win32").create(root / "env")
    (root / ".hold").touch()
    record = {
        "identity": named,
        "package": "native-proof",
        "version": "1.0.0",
        "application": "native_proof:app",
        "python": str(root / "env" / "bin" / "python"),
        "dependencies": [],
        "interface": {},
        "sdk_fallback": "",
        "runtime_build": "",
    }
    (root / "generation.json").write_text(json.dumps(record))
    return root


def test_the_proof_holds_the_generation_it_reads(tmp_path: Path) -> None:
    root = _generation(tmp_path)
    with native_wheel_proof.held(root) as (generation, site_packages):
        assert (generation.identity, generation.package) == (IDENTITY, "native-proof")
        assert site_packages.parent.parent.parent == root / "env"
        # TensorD's collector takes the hold exclusively, without waiting: it cannot now.
        with (root / ".hold").open("rb") as collector, pytest.raises(BlockingIOError):
            filelock.lock(collector, wait=False)
    with (root / ".hold").open("rb") as collector:
        filelock.lock(collector, wait=False)


def test_a_root_its_record_does_not_name_is_refused(tmp_path: Path) -> None:
    root = _generation(tmp_path, named="f" * 32)
    with pytest.raises(EnvironmentRefusal) as refused, native_wheel_proof.held(root):
        pass
    assert refused.value.code == "package_generation_invalid"


def test_a_root_without_a_record_is_refused(tmp_path: Path) -> None:
    root = _generation(tmp_path)
    (root / "generation.json").unlink()
    with pytest.raises(EnvironmentRefusal) as refused, native_wheel_proof.held(root):
        pass
    assert refused.value.code == "package_generation_invalid"
