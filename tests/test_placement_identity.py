"""Placement and model identity are stable as a package's model set grows (wire 61).

One placement per package release: `placement_id = "package-" + sha256(package \\0 release)`,
never the bound model rows, so adding a slot never retires the process that serves the
others. A Model's id is a function of its slot path alone, so every other slot's binding
digest survives the addition.
"""

from __future__ import annotations

import hashlib

from cozy_runtime.internal.worker.package_prepare import model_id, placement_id


def test_placement_id_is_the_package_release() -> None:
    expected = "package-" + hashlib.sha256(b"paul/minimax-h3\x001.2.0").hexdigest()[:24]
    assert placement_id("paul/minimax-h3", "1.2.0") == expected
    assert placement_id("paul/minimax-h3", "1.2.1") != expected
    assert placement_id("paul/minimax-h3-tools", "1.2.0") != expected


def test_model_id_is_the_slot_path() -> None:
    slot = "fl2va.models.model"
    assert model_id(slot) == "model-" + hashlib.sha256(slot.encode()).hexdigest()[:16]
    assert model_id(slot) != model_id("ref2va_turbo.models.base_model")
