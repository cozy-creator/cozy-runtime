"""Installed Runtime observation bridges tolerate additional producer diagnostics."""

from __future__ import annotations

import json

import pytest

from cozy_runtime.internal import base_observation


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
