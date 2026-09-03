"""The package interface's request/result projection over the author's own type system.

`internal/schema.render` is the real production renderer — the one `describe` runs and the
one whose output is a PUBLICATION fact. The arms below are the types msgspec codes and
JSON carries: a projection narrower than the runtime it describes refuses packages that
would have served, so every one of these must produce a NODE rather than a refusal.
"""

from __future__ import annotations

import datetime
import enum
from typing import Annotated, Literal

import msgspec
import pytest

from cozy_runtime.author._errors import ConformanceError
from cozy_runtime.internal.schema import render


class Palette(enum.Enum):
    """A tuple-valued enum. msgspec encodes a member as the JSON array `[1, 2]`."""

    LOW = (1, 2)
    HIGH = (3, 4)


class Color(enum.Enum):
    RED = "red"
    BLUE = "blue"


def test_msgspec_native_containers_project() -> None:
    """The walker descends these; the renderer must agree with it about the same types."""
    assert render(dict[str, str]) == {"map": {"key": "str", "value": "str"}}
    assert render(dict[str, list[int]]) == {"map": {"key": "str", "value": {"list": "int"}}}
    assert render(tuple[int, int]) == {"tuple": ["int", "int"]}
    assert render(tuple[str, ...]) == {"list": "str"}
    assert render(set[str]) == {"list": "str"}
    assert render(frozenset[int]) == {"list": "int"}


def test_types_with_no_interoperable_spelling_are_opaque_not_refusals() -> None:
    """`bytes` and `datetime` are msgspec-native. The package interface names them and moves on."""
    assert render(bytes) == {"opaque": "bytes"}
    assert render(datetime.datetime) == {"opaque": "datetime"}


def test_an_enumeration_the_wire_cannot_name_degrades_instead_of_refusing() -> None:
    """A tuple-valued enum serves under msgspec, so it must BUILD under the projection."""
    assert render(Palette) == {"opaque": "Palette"}
    assert render(Literal[b"x"]) == {"opaque": "typing.Literal[b'x']"}

    # And an enum the wire CAN name still enumerates, member by member, by its VALUE —
    # including a Literal written over enum members rather than over their values.
    assert render(Color) == {"literal": ["blue", "red"]}
    assert render(Literal[Color.RED, Color.BLUE]) == {"literal": ["blue", "red"]}


def test_unknown_meta_constraints_pass_through_verbatim() -> None:
    """A constraint msgspec APPLIES must reach the package interface, learned by Creator or not."""

    class Request(msgspec.Struct):
        count: Annotated[int, msgspec.Meta(lt=64, ge=1)]
        name: Annotated[str, msgspec.Meta(pattern="^[a-z]+$", max_length=32)]

    rendered = render(Request)
    assert isinstance(rendered, dict)
    projected = rendered["fields"]
    assert isinstance(projected, list)
    fields = {field["name"]: field for field in projected}
    assert fields["count"]["constraints"] == {"lt": 64, "ge": 1}
    assert fields["name"]["constraints"] == {"pattern": "^[a-z]+$", "max_length": 32}


def test_an_incoherent_tagged_union_still_refuses() -> None:
    """What the projection KEEPS refusing: a grammar a consumer could not discriminate."""

    class Tagged(msgspec.Struct, tag="a"):
        x: int

    class Untagged(msgspec.Struct):
        y: int

    with pytest.raises(ConformanceError) as refusal:
        render(Tagged | Untagged)
    assert refusal.value.code == "tagged_union_incoherent"
