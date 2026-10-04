"""Documents from newer or older writers keep working when they carry additive fields."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cozy_runtime.internal import canonical, interface_wheel, package_interface, static_interface
from test_serving_interfaces import project


def _interface() -> bytes:
    return canonical.write(
        {
            "application": "additive_fixture:app",
            "format": package_interface.SCHEMA,
            "entrypoints": [],
            "jobs": [
                {
                    "name": "main",
                    "request": {
                        "fields": [{"name": "steps", "type": "int", "since": "a later writer"}],
                        "doc": "an additive schema attribute",
                    },
                    "result": {"fields": []},
                    "publishes": False,
                    "timeout_hint": 30,
                }
            ],
            "generator": "a newer Runtime",
        }
    )


def test_missing_required_interface_fields_still_refuse() -> None:
    body = json.loads(_interface())
    del body["jobs"][0]["result"]
    with pytest.raises(
        package_interface.StalePackageInterface, match="missing required field `result`"
    ):
        package_interface.read_bytes(canonical.write(body))
    del body["jobs"]
    with pytest.raises(
        package_interface.StalePackageInterface, match="missing required field `jobs`"
    ):
        package_interface.read_bytes(canonical.write(body))


def test_additive_asset_bound_keys_are_read_and_not_generated(tmp_path: Path) -> None:
    root = project(tmp_path)
    source = (root / "serving_fixture.py").read_text()
    source = source.replace(
        "from cozy_runtime.author import App, Context, Model",
        "from typing import Annotated\n"
        "from cozy_runtime.author import App, AssetBound, Context, ImageAsset, Model",
    ).replace(
        "    prompt: str\n",
        "    prompt: str\n    image: Annotated[ImageAsset, AssetBound(max_bytes=4096)]\n",
    )
    (root / "serving_fixture.py").write_text(source)
    body = json.loads(package_interface.canonical_bytes(static_interface.build(root)))
    bounds = [
        field["asset_bound"]
        for entry in body["entrypoints"]
        for field in entry["request"]["fields"]
        if "asset_bound" in field
    ]
    assert bounds
    for bound in bounds:
        bound["max_frames"] = 3
    raw = canonical.write(body)
    package_interface.read_bytes(raw)
    for name, generated in interface_wheel.generate(raw).items():
        if name.endswith(".py"):
            assert b"max_frames" not in generated
            compile(generated, name, "exec")
