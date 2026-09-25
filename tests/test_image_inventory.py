"""Image inventory is checked as a fact but cannot prune the private dependency closure."""

from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

import pytest

from cozy_runtime.internal import base_observation, image_inventory, package_environment
from cozy_runtime.internal.worker.package_prepare import (
    PreparationRefusal,
    _filtered_rows,
    prepare_package_set,
)
from cozy_runtime.protocol import worker_pb2 as pb

DEPENDENCY = "annotated-doc"


def _image(
    observed: base_observation.Observation, version: str | None
) -> base_observation.Observation:
    """The real observation with the image's annotated-doc pin fabricated exactly."""

    rows = tuple(row for row in observed.distributions if row[0] != DEPENDENCY)
    if version is not None:
        rows = (*rows, (DEPENDENCY, version))
    return dataclasses.replace(observed, distributions=rows)


def _locked(rows: list[str]) -> package_environment.LockedRequirements:
    body = "\n".join(["--index-url https://pypi.org/simple", *rows, ""]).encode()
    return package_environment.read_locked_requirements(body)


def _pin(name: str, version: str) -> str:
    return f"{name}=={version} --hash=sha256:" + "a" * 64


def _inventory(pins: dict[str, str]) -> image_inventory.Inventory:
    return image_inventory.from_wire(
        "test-profile",
        "3.12.0",
        [
            pb.ImageDistribution(distribution=name, version=version)
            for name, version in sorted(pins.items())
        ],
    )


def test_locked_export_keeps_inventory_names() -> None:
    """Every selected row enters the private venv, even when the image also has it."""

    locked = _locked(
        [
            _pin(DEPENDENCY, "0.0.5"),
            _pin("ranged-marco", "1.0.4"),
            _pin("some-helper", "2.0.0"),
        ]
    )
    rows = _filtered_rows(
        "paul/ranged-marco", "1.0.4", locked, _inventory({DEPENDENCY: "0.0.5"}), {}
    )
    assert rows == list(locked.rows)

    # The package's own pin is what the org-index install actually installs; an export
    # without it — or with the wrong release — is not this release's export.
    with pytest.raises(PreparationRefusal) as absent:
        _filtered_rows(
            "paul/ranged-marco",
            "1.0.4",
            _locked([_pin("some-helper", "2.0.0")]),
            _inventory({}),
            {},
        )
    assert absent.value.code == "package_prepare_project_pin_missing"
    with pytest.raises(PreparationRefusal) as wrong:
        _filtered_rows(
            "paul/ranged-marco",
            "1.0.4",
            _locked([_pin("ranged-marco", "1.0.5")]),
            _inventory({}),
            {},
        )
    assert wrong.value.code == "package_prepare_project_pin_missing"

    # Even an inventory-owned NAME equal to the project's is not served by the image: the
    # project row always installs.
    kept = _filtered_rows(
        "paul/ranged-marco",
        "1.0.4",
        _locked([_pin("ranged-marco", "1.0.4")]),
        _inventory({"ranged-marco": "1.0.4"}),
        {},
    )
    assert [row.name for row in kept] == ["ranged-marco"]


def test_lying_inventory_refuses_before_any_install() -> None:
    """The demoted observation is the cross-check that catches a lying inventory, and it
    fires before a single byte is installed."""

    from cozy_runtime.internal import canonical

    observed = package_environment.observe_base(Path(sys.executable))
    download_set = canonical.write(
        {
            "format": "cozy.worker.v1.DownloadDelegation/1",
            "packages": [{"package": "paul/ranged-marco", "release": "1.0.4"}],
            "models": [],
        }
    )

    def prepare(
        base: base_observation.Observation, inventory_pins: dict[str, str]
    ) -> pb.PreparePackageSetResult:
        pins = {**dict(base.distributions), **inventory_pins}
        selected = sorted({"cozy-runtime", "msgspec", *inventory_pins})
        return prepare_package_set(
            pb.PreparePackageSetRequest(
                download_delegation=download_set,
                install_root="/tmp/unused-environment",
                application="ranged_marco_package:app",
                model_slot_paths=[],
                image_inventory=pb.ImageInventory(
                    profile="test-profile",
                    python="3.12.0",
                    distributions=[
                        pb.ImageDistribution(distribution=name, version=pins[name])
                        for name in selected
                    ],
                ),
                locked_requirements=b"--index-url https://pypi.org/simple\n"
                b"ranged-marco==1.0.4 --hash=sha256:" + b"a" * 64 + b"\n",
                package_interface=b"{}",
            ),
            artifact_cache=Path("/tmp/unused-artifacts"),
            tensorfs_root=Path("/tmp/unused-store"),
            install_root=Path("/tmp/unused-environment"),
            python=Path(sys.executable),
            verified=lambda _digest, _length, _path: None,
            base=base,
        )

    in_range = _image(observed, "0.0.5")
    with pytest.raises(PreparationRefusal) as lying:
        prepare(in_range, {DEPENDENCY: "0.0.4"})
    assert lying.value.code == "package_prepare_image_inventory_disagrees"
    assert lying.value.detail == f"distribution={DEPENDENCY} inventory=0.0.4 image=0.0.5"

    with pytest.raises(PreparationRefusal) as phantom:
        prepare(_image(observed, None), {DEPENDENCY: "0.0.5"})
    assert phantom.value.code == "package_prepare_image_inventory_disagrees"
    assert phantom.value.detail == f"distribution={DEPENDENCY} inventory=0.0.5 image=absent"


def test_frameworks_remain_in_the_complete_locked_dependencies() -> None:
    locked = _locked(
        [
            _pin("cozy-runtime", "0.2.24"),
            _pin("ranged-marco", "1.0.4"),
            _pin("some-helper", "2.0.0"),
            _pin("tensorfs", "0.3.10"),
        ]
    )
    rows = _filtered_rows(
        "paul/ranged-marco",
        "1.0.4",
        locked,
        _inventory({}),
        {"cozy-runtime": "0.2.25", "tensorfs": "0.3.11", "some-helper": "99.0.0"},
    )
    assert rows == list(locked.rows)
    # Importability in an unrelated parent venv is not inherited platform custody.
    assert _filtered_rows("paul/ranged-marco", "1.0.4", locked, _inventory({}), {}) == list(
        locked.rows
    )


def test_actual_interpreter_layout_distinguishes_managed_venv_from_base() -> None:
    assert not package_environment.is_base_interpreter(Path(sys.executable))
    assert package_environment.is_base_interpreter(Path(sys.base_prefix) / "bin/python3.12")


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
