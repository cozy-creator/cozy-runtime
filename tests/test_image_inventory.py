"""Image inventory is checked as a fact but cannot prune the private dependency closure."""

from __future__ import annotations

import json

import pytest

from cozy_runtime.internal import image_inventory, package_environment
from cozy_runtime.internal.worker.package_prepare import (
    PreparationRefusal,
    _filtered_rows,
)

DEPENDENCY = "annotated-doc"


def _locked(rows: list[str]) -> package_environment.LockedRequirements:
    body = "\n".join(["--index-url https://pypi.org/simple", *rows, ""]).encode()
    return package_environment.read_locked_requirements(body)


def _pin(name: str, version: str) -> str:
    return f"{name}=={version} --hash=sha256:" + "a" * 64


def test_locked_export_keeps_inventory_names() -> None:
    """Every selected row enters the private venv, even when the image also has it."""

    locked = _locked(
        [
            _pin(DEPENDENCY, "0.0.5"),
            _pin("ranged-marco", "1.0.4"),
            _pin("some-helper", "2.0.0"),
        ]
    )
    rows = _filtered_rows("paul/ranged-marco", "1.0.4", locked)
    assert rows == list(locked.rows)

    # The package's own pin is what the org-index install actually installs; an export
    # without it — or with the wrong release — is not this release's export.
    with pytest.raises(PreparationRefusal) as absent:
        _filtered_rows(
            "paul/ranged-marco",
            "1.0.4",
            _locked([_pin("some-helper", "2.0.0")]),
        )
    assert absent.value.code == "package_prepare_project_pin_missing"
    with pytest.raises(PreparationRefusal) as wrong:
        _filtered_rows(
            "paul/ranged-marco",
            "1.0.4",
            _locked([_pin("ranged-marco", "1.0.5")]),
        )
    assert wrong.value.code == "package_prepare_project_pin_missing"

    # Even an inventory-owned NAME equal to the project's is not served by the image: the
    # project row always installs.
    kept = _filtered_rows(
        "paul/ranged-marco",
        "1.0.4",
        _locked([_pin("ranged-marco", "1.0.4")]),
    )
    assert [row.name for row in kept] == ["ranged-marco"]
    # A release spelled `1.0.4-rc.1` is the wheel's normalised `1.0.4rc1`.
    assert _filtered_rows("paul/ranged-marco", "1.0.4-rc.1", _locked([_pin("ranged-marco", "1.0.4rc1")]))


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
    )
    assert rows == list(locked.rows)
    # Importability in an unrelated parent venv is not inherited platform custody.
    assert _filtered_rows("paul/ranged-marco", "1.0.4", locked) == list(locked.rows)


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
