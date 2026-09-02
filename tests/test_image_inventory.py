"""cr-070: the pod package venv is built by NAME against the placed image's inventory.

Real wheels, real uv venvs, the real preparation path — nothing mocked. The three arms:

* an inventory-owned name is SKIPPED: the image's site-packages copy serves it, and the
  lock wheel the set delivered never enters the venv;
* a declared range the image version cannot satisfy BOOT-INSTALLS an in-range version from
  the index into the venv — logged, never refused — and the venv copy shadows the image's;
* a LYING inventory — one that promises a distribution the image does not actually provide,
  or provides at another version — is a typed refusal naming the distribution and both
  versions. The base observation is the cross-check that catches it, not the authority.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import shutil
import subprocess
import sys
import sysconfig
import tempfile
from pathlib import Path

import pytest
from packaging.specifiers import SpecifierSet
from packaging.version import Version

from cozy_runtime.internal import base_observation, image_inventory, package_environment
from cozy_runtime.internal.worker.package_prepare import PreparationRefusal, prepare_package
from cozy_runtime.protocol import worker_pb2 as pb

ROOT = Path(__file__).resolve().parent.parent
PACKAGE = ROOT / "examples" / "marco-polo"
TIMEOUT = 420

#: The one image-owned name under test. It is REAL on PyPI (0.0.1..0.0.5), tiny and pure,
#: so the boot-install arm resolves against the actual index the slow path uses.
DEPENDENCY = "annotated-doc"
DECLARED = ">=0.0.2,<0.0.6"


def _run(*command: str, cwd: str | None = None) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        command,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=TIMEOUT,
        cwd=cwd,
        check=False,
    )
    assert result.returncode == 0, f"{command}: {result.stderr[-4000:]}"
    return result


def _write_ranged_package(root: Path) -> None:
    """The real marco-polo package, renamed, with one image-owned declared range."""

    shutil.copytree(PACKAGE, root)
    (root / "uv.lock").unlink()
    text = (root / "pyproject.toml").read_text()
    replacements = (
        ('name = "marco-polo"', 'name = "ranged-marco"'),
        ('"msgspec>=0.19",', f'"msgspec>=0.19",\n    "{DEPENDENCY}{DECLARED}",'),
    )
    for old, new in replacements:
        assert old in text, old
        text = text.replace(old, new)
    # The example locks cozy-runtime from PyPI since #215; this copy resolves it
    # from the checkout so the arm never depends on a published wheel.
    text += '\n[tool.uv.sources]\ncozy-runtime = { path = "../..", editable = true }\n'
    (root / "pyproject.toml").write_text(text)


def _write_lock_dependency(root: Path, version: str) -> None:
    package = root / "src" / "annotated_doc"
    package.mkdir(parents=True)
    (root / "pyproject.toml").write_text(
        "[build-system]\n"
        'requires = ["hatchling"]\n'
        'build-backend = "hatchling.build"\n\n'
        "[project]\n"
        f'name = "{DEPENDENCY}"\n'
        f'version = "{version}"\n'
        'requires-python = ">=3.12,<3.13"\n\n'
        "[tool.hatch.build.targets.wheel]\n"
        'packages = ["src/annotated_doc"]\n'
    )
    (package / "__init__.py").write_text(f"VALUE = {version!r}\n")


def _download_row(kind: pb.LocalDownloadKind, path: Path) -> pb.LocalDownloadFile:
    raw = path.read_bytes()
    return pb.LocalDownloadFile(
        digest=hashlib.sha256(raw).digest(),
        filename=path.name,
        kind=kind,
        length=len(raw),
        path=str(path),
    )


def _image(
    observed: base_observation.Observation, version: str | None
) -> base_observation.Observation:
    """The real observation with the image's annotated-doc pin fabricated exactly."""

    rows = tuple(row for row in observed.distributions if row[0] != DEPENDENCY)
    if version is not None:
        rows = (*rows, (DEPENDENCY, version))
    return dataclasses.replace(observed, distributions=rows)


def _inventory(
    observed: base_observation.Observation, pins: dict[str, str]
) -> image_inventory.Inventory:
    """A `tensorhub.image_inventory/1` document through the real reader."""

    versions = {**dict(observed.distributions), **pins}
    raw = json.dumps(
        {
            "format": image_inventory.INVENTORY_FORMAT,
            "profile": "test-profile",
            "python": "3.12.0",
            "distributions": [
                {"name": name, "version": versions[name]}
                for name in sorted({"cozy-runtime", "msgspec", *pins})
            ],
        }
    ).encode()
    return image_inventory.read_bytes(raw)


def _describe(selected: package_environment.InstalledEnvironment, distribution: str) -> bytes:
    """Derive the package interface inside the prepared venv, exactly as the worker would.

    On a pod the image interpreter carries cozy-runtime and its dependencies in its real
    site-packages, so the `--system-site-packages` venv sees them. On this test host that
    copy lives in the checkout's venv (an editable install a child cannot reach through
    `PYTHONPATH`), so the child is pointed at the source tree and the checkout's
    third-party site directly.
    """

    command = (
        "from cozy_runtime.internal import package_interface; "
        "from cozy_runtime.internal.discovery import discover_distribution; "
        f"found=discover_distribution({distribution!r}); "
        "print(package_interface.canonical_bytes(package_interface.build(found)).decode())"
    )
    result = subprocess.run(
        (str(selected.python), "-c", command),
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=TIMEOUT,
        cwd="/",
        check=False,
        env={
            **os.environ,
            "PYTHONPATH": f"{ROOT / 'src'}:{sysconfig.get_path('purelib')}",
        },
    )
    assert result.returncode == 0, result.stderr[-4000:]
    return result.stdout.encode()


def _prepare(
    root: Path,
    *,
    files: list[pb.LocalDownloadFile],
    base: base_observation.Observation,
    inventory: image_inventory.Inventory,
    seed: bytes,
) -> tuple[pb.PreparePackageSetResult, Path]:
    environment = root / f"environment-{seed.hex()}"
    rows = sorted(files, key=lambda row: row.digest)
    result = prepare_package(
        package_name="paul/ranged-marco",
        release="1.0.4",
        files=rows,
        placement_seed=seed,
        artifact_cache=root / "artifacts",
        artifact_store=root / "artifact-store",
        environment_root=environment,
        python=Path(sys.executable),
        describe=_describe,
        verified=lambda _digest, _length, _path: None,
        base=base,
        inventory=inventory,
    )
    generations = list((environment / "contents").iterdir())
    assert len(generations) == 1, generations
    return result, generations[0] / "lib" / "python3.12" / "site-packages"


def test_pod_venv_skips_image_owned_names_and_boot_installs_out_of_range(
    capsys: pytest.CaptureFixture[str],
) -> None:
    uv = shutil.which("uv")
    assert uv is not None
    observed = package_environment.observe_base(Path(sys.executable))
    assert "cozy-runtime" in observed.distribution_versions

    with tempfile.TemporaryDirectory(prefix="cozy-image-inventory.") as raw:
        root = Path(raw)
        _write_ranged_package(root / "ranged-marco")
        _write_lock_dependency(root / "lock-dependency", "0.0.5")
        wheel_set = root / "artifacts" / "sets" / "package"
        wheel_set.mkdir(parents=True)
        (root / "artifact-store").mkdir()
        _run(uv, "build", "--wheel", "--out-dir", str(wheel_set), str(ROOT))
        _run(uv, "build", "--wheel", "--out-dir", str(wheel_set), str(root / "ranged-marco"))
        _run(uv, "build", "--wheel", "--out-dir", str(wheel_set), str(root / "lock-dependency"))
        runtime_wheel = next(wheel_set.glob("cozy_runtime-*.whl"))
        project_wheel = next(wheel_set.glob("ranged_marco-*.whl"))
        lock_wheel = next(wheel_set.glob("annotated_doc-0.0.5-*.whl"))

        # GREEN ARM ONE — export minus provided: the image pins annotated-doc 0.0.5 inside
        # the declared range, so BOTH delivered dependency wheels stay out of the venv: the
        # runtime wheel through the protected-family path, the annotated-doc lock wheel by
        # inventory NAME. The image's copies serve them; the lock pin constrained nothing.
        in_range = _image(observed, "0.0.5")
        prepared, site_packages = _prepare(
            root,
            files=[
                _download_row(pb.LOCAL_DOWNLOAD_KIND_PROJECT_WHEEL, project_wheel),
                _download_row(pb.LOCAL_DOWNLOAD_KIND_DEPENDENCY_WHEEL, runtime_wheel),
                _download_row(pb.LOCAL_DOWNLOAD_KIND_DEPENDENCY_WHEEL, lock_wheel),
            ],
            base=in_range,
            inventory=_inventory(in_range, {DEPENDENCY: "0.0.5"}),
            seed=b"\x51",
        )
        placement = json.loads(prepared.placement_set.placement_set_canonical_bytes)
        assert placement["placements"][0]["environment"]["wheels"] == []
        assert not list(site_packages.glob("annotated_doc*"))
        out = capsys.readouterr().out
        assert "[package-prepare] image inventory serves" in out
        assert f"{DEPENDENCY} (image 0.0.5 over lock 0.0.5)" in out
        assert "boot-install" not in out

        # GREEN ARM TWO — the boot-install slow path: the image's 0.0.1 falls outside the
        # declared range and no lock wheel was delivered for the name, so preparation
        # installs an in-range version from the real index into the venv — logged with the
        # name, the image version, the required range and the chosen version — and the venv
        # copy shadows the image's through ordinary venv precedence.
        out_of_range = _image(observed, "0.0.1")
        prepared, site_packages = _prepare(
            root,
            files=[
                _download_row(pb.LOCAL_DOWNLOAD_KIND_PROJECT_WHEEL, project_wheel),
                _download_row(pb.LOCAL_DOWNLOAD_KIND_DEPENDENCY_WHEEL, runtime_wheel),
            ],
            base=out_of_range,
            inventory=_inventory(out_of_range, {DEPENDENCY: "0.0.1"}),
            seed=b"\x52",
        )
        placement = json.loads(prepared.placement_set.placement_set_canonical_bytes)
        assert placement["placements"][0]["environment"]["wheels"] == []
        installed = list(site_packages.glob("annotated_doc-*.dist-info"))
        assert len(installed) == 1, installed
        chosen = installed[0].name.removesuffix(".dist-info").rpartition("-")[2]
        assert SpecifierSet(DECLARED).contains(Version(chosen))
        out = capsys.readouterr().out
        line = next(row for row in out.splitlines() if "boot-install" in row)
        assert f"boot-install distribution={DEPENDENCY}" in line
        assert "image_version=0.0.1" in line
        assert f"chosen={chosen}" in line
        assert "reused=false" in line
        for clause in DECLARED.split(","):
            assert clause in line

        # RED ARM — the lying inventory. Same wheels, same declared range; the document
        # promises annotated-doc 0.0.4 while the image actually provides 0.0.5. The
        # demoted observation is the cross-check that names the distribution and both
        # versions, before any wheel is touched.
        with pytest.raises(PreparationRefusal) as lying:
            _prepare(
                root,
                files=[
                    _download_row(pb.LOCAL_DOWNLOAD_KIND_PROJECT_WHEEL, project_wheel),
                    _download_row(pb.LOCAL_DOWNLOAD_KIND_DEPENDENCY_WHEEL, runtime_wheel),
                ],
                base=in_range,
                inventory=_inventory(in_range, {DEPENDENCY: "0.0.4"}),
                seed=b"\x53",
            )
        assert lying.value.code == "package_prepare_image_inventory_disagrees"
        assert lying.value.detail == f"distribution={DEPENDENCY} inventory=0.0.4 image=0.0.5"

        # The absence flavor of the same lie: a promised distribution the image never had.
        with pytest.raises(PreparationRefusal) as phantom:
            _prepare(
                root,
                files=[
                    _download_row(pb.LOCAL_DOWNLOAD_KIND_PROJECT_WHEEL, project_wheel),
                    _download_row(pb.LOCAL_DOWNLOAD_KIND_DEPENDENCY_WHEEL, runtime_wheel),
                ],
                base=_image(observed, None),
                inventory=_inventory(observed, {DEPENDENCY: "0.0.5"}),
                seed=b"\x54",
            )
        assert phantom.value.code == "package_prepare_image_inventory_disagrees"
        assert phantom.value.detail == f"distribution={DEPENDENCY} inventory=0.0.5 image=absent"


def test_image_inventory_document_reader_is_the_exact_writer_rule() -> None:
    body = {
        "format": image_inventory.INVENTORY_FORMAT,
        "profile": "cuda13-torch213",
        "python": "3.12.8",
        "distributions": [
            {"name": "cozy-runtime", "version": "0.0.36"},
            {"name": "torch", "version": "2.13.1"},
        ],
    }
    inventory = image_inventory.read_bytes(json.dumps(body).encode())
    assert inventory.profile == "cuda13-torch213"
    assert inventory.versions == {"cozy-runtime": "0.0.36", "torch": "2.13.1"}

    for mutation, _reason in (
        ({"format": "tensorhub.image_inventory/2"}, "format"),
        ({"distributions": []}, "distributions absent"),
        ({"python": "not-a-version"}, "python"),
        (
            {
                "distributions": [
                    {"name": "torch", "version": "2.13.1"},
                    {"name": "cozy-runtime", "version": "0.0.36"},
                ]
            },
            "unsorted names",
        ),
        ({"distributions": [{"name": "Torch", "version": "2.13.1"}]}, "unnormalized name"),
        ({"distributions": [{"name": "torch", "version": "nope"}]}, "version"),
    ):
        with pytest.raises(image_inventory.ImageInventoryRefusal):
            image_inventory.read_bytes(json.dumps({**body, **mutation}).encode())

    # Version-spelling equivalence is a Version comparison, not a string comparison.
    assert (
        image_inventory.disagreement(inventory, {"cozy-runtime": "0.0.36.0", "torch": "2.13.1"})
        is None
    )
