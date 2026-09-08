"""Assigned input aliases retain the same complete interface as imported annotations."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from cozy_runtime.internal import package_interface, static_interface
from cozy_runtime.internal.discovery import discover
from cozy_runtime.internal.static_interface import StaticRefusal


def fixture(root: Path, *, poison: bool = False) -> Path:
    (root / "package.toml").write_text('[application]\nobject="alias_app:app"\n')
    (root / "alias_types.py").write_text(
        ('raise RuntimeError("alias module must not execute")\n' if poison else "")
        + """from typing import Annotated
import msgspec
from cozy_runtime.author import Assets, AssetLimits, AssetBound, Image, Context, Outputs
class Request(msgspec.Struct):
    prompt: str
class Result(msgspec.Struct):
    done: bool
Input = Request
Ctx = Context
Save = Outputs
Images = Annotated[
    Assets[Annotated[Image,
        AssetBound(media_types=("image/png",), max_bytes=4096, max_decoded_bytes=8192)]],
    AssetLimits(images=2, total=2), msgspec.Meta(min_length=1),
]
Inputs = Images
"""
    )
    (root / "alias_app.py").write_text(
        """from cozy_runtime.author import App
from alias_types import Input, Result, Ctx, Save, Inputs
app=App()
@app.entrypoint
def render(ctx: Ctx, payload: Input, assets: Inputs, out: Save) -> Result:
    raise RuntimeError("handler must not execute")
"""
    )
    return root


def test_assigned_assets_context_and_service_aliases_equal_imported_surface(tmp_path: Path) -> None:
    root = fixture(tmp_path)
    for name in ("alias_app", "alias_types"):
        sys.modules.pop(name, None)
    expected = package_interface.build(discover(root))
    actual = static_interface.build(root)
    assert package_interface.canonical_bytes(actual) == package_interface.canonical_bytes(expected)
    entries = actual["entrypoints"]
    assert isinstance(entries, list) and isinstance(entries[0], dict)
    assets = entries[0]["assets"]
    assert isinstance(assets, dict)
    assert assets["parameter"] == "assets" and assets["view"] == "decoded"
    assert assets["kinds"] == [
        {
            "kind": "image",
            "media_types": ["image/png"],
            "max_count": 2,
            "max_bytes": 4096,
            "max_decoded_bytes": 8192,
        }
    ]
    fixture(root, poison=True)
    for name in ("alias_app", "alias_types"):
        sys.modules.pop(name, None)
    assert static_interface.build(root) == actual  # File reading does not use sys.modules/imports.


@pytest.mark.parametrize(
    "value", ["Inputs", "Images", "Annotated[Inputs, msgspec.Meta(min_length=1)]"]
)
def test_alias_cycles_refuse_without_recursing(tmp_path: Path, value: str) -> None:
    root = fixture(tmp_path)
    path = root / "alias_types.py"
    path.write_text(
        path.read_text().replace("Inputs = Images", f"Inputs = {value}")
        + ("\nImages = Inputs\n" if value == "Images" else "")
    )
    with pytest.raises(StaticRefusal, match="cyclic"):
        static_interface.build(root)
