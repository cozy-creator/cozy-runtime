"""Native output slots consume typed declarations and tolerate producer additions."""

from __future__ import annotations

from copy import deepcopy
from typing import Annotated

import msgspec
import pytest

from cozy_runtime.author import AssetBound, FileAsset, ImageAsset, Tree
from cozy_runtime.internal import schema
from cozy_runtime.internal.worker.machine_byte_results import paths
from cozy_runtime.internal.worker.workspace import WorkspaceRefusal


class _Nested(msgspec.Struct):
    pair: tuple[FileAsset, Tree]


class _Result(msgspec.Struct):
    nested: Annotated[_Nested, AssetBound(max_bytes=80)]
    references: Annotated[list[ImageAsset], AssetBound(max_bytes=40)]
    info: dict[str, int]


def test_declared_nested_slots_keep_inherited_and_list_item_bounds() -> None:
    document = schema.render(_Result)
    expected = [("nested.pair.0", 80), ("nested.pair.1", 80), ("references.*", 40)]
    assert list(paths(document)) == expected
    assert list(paths(document, bound=50)) == [
        ("nested.pair.0", 50),
        ("nested.pair.1", 50),
        ("references.*", 40),
    ]
    evolved = deepcopy(document)
    evolved["future"] = {"asset": "new-schema-advisory"}
    for field in evolved["fields"]:
        field["future"] = {"kind": "unknown"}
        field["type"]["future"] = ["producer addition"]
    assert list(paths(evolved)) == expected


@pytest.mark.parametrize("annotation", [list[Tree], dict[str, FileAsset], FileAsset | None])
def test_dynamic_native_slots_still_refuse(annotation: object) -> None:
    with pytest.raises(WorkspaceRefusal, match="dynamic native result fields"):
        list(paths(schema.render(annotation), "result"))


def test_malformed_consumed_slot_fields_refuse_before_custody() -> None:
    document = schema.render(_Result)
    document["fields"][0]["asset_bound"]["max_bytes"] = True
    with pytest.raises(WorkspaceRefusal, match="invalid output schema"):
        list(paths(document))
