"""Installed Runtime observation bridges tolerate additional producer diagnostics."""

from __future__ import annotations

import copy
import json
from typing import Any

import pytest

from cozy_runtime.internal import base_observation, image_inventory


def test_observation_accepts_additions_but_retains_required_facts_and_types() -> None:
    current = base_observation.Observation(
        python_abi="cp312",
        os_arch="linux/amd64",
        libc="glibc2.39",
        accelerator_backend="none",
        accelerator_build="cpu",
        torch_release="none",
        cuda_version="none",
        uv_version="0.10.0",
        distributions=(("cozy-runtime", "0.18.34"),),
        import_roots=(("cozy", ("cozy-runtime",)),),
        python_full_version="3.12.12",
    )
    body = base_observation._body(current)
    extended = json.loads(json.dumps({**body, "future_diagnostic": {"available": True}}))
    assert base_observation._parse(extended) == current
    for name in body:
        missing = dict(extended)
        missing.pop(name)
        with pytest.raises(base_observation.ObservationRefusal):
            base_observation._parse(missing)
    for name, invalid in (
        ("python_abi", 312),
        ("distributions", [["cozy-runtime", 34]]),
        ("import_roots", [["cozy", [34]]]),
    ):
        with pytest.raises(base_observation.ObservationRefusal):
            base_observation._parse({**extended, name: invalid})


def test_image_inventory_accepts_additive_metadata_but_not_missing_or_invalid_facts() -> None:
    body: dict[str, Any] = {
        "format": image_inventory.INVENTORY_FORMAT,
        "profile": "cpu",
        "python": "3.12.12",
        "distributions": [{"name": "cozy-runtime", "version": "0.18.34"}],
        "interpreters": [{"version": "3.12.12", "abi": "cp312"}],
    }
    expected = image_inventory.read_bytes(json.dumps(body).encode())
    extra = copy.deepcopy(body)
    extra["future_diagnostic"] = {"available": True}
    extra["distributions"][0]["origin"] = "index"
    extra["interpreters"][0]["implementation"] = "cpython"
    assert image_inventory.read_bytes(json.dumps(extra).encode()) == expected
    for name in ("format", "profile", "python", "distributions"):
        missing = dict(extra)
        missing.pop(name)
        with pytest.raises(image_inventory.ImageInventoryRefusal):
            image_inventory.read_bytes(json.dumps(missing).encode())
    for path, name, invalid in (
        ("distributions", "version", "bad"),
        ("interpreters", "abi", "cp313"),
    ):
        broken = copy.deepcopy(extra)
        broken[path][0][name] = invalid
        with pytest.raises(image_inventory.ImageInventoryRefusal):
            image_inventory.read_bytes(json.dumps(broken).encode())
        del broken[path][0][name]
        with pytest.raises(image_inventory.ImageInventoryRefusal):
            image_inventory.read_bytes(json.dumps(broken).encode())
