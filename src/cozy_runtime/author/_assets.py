"""xs-135 — named assets and `<name>` mentions: an opt-in convention for prompts that talk about
their production elements (characters, props, environments).

A package declares `assets: list[A]` for an `Asset` subclass carrying its media and fields
(`PictureAsset` for images), and renders its prompt with `render` and a label: how its model refers
to the Nth asset. Nothing below this module knows the convention; an asset is an ordinary struct
whose files bind like any other, and a package wanting another shape declares it.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from typing import NamedTuple

import msgspec

from cozy_runtime.author._errors import InvalidRequest
from cozy_runtime.author._services import AdjustmentRows, Adjustments

#: A mention: `<`, any text without brackets or line breaks, `>`.
MENTION = re.compile(r"<([^<>\n]+)>")


class Asset(msgspec.Struct, kw_only=True):
    """A production element the prompt names as `<name>`. Subclasses add its media (`image`,
    `video`, `audio`, or any other file field) and their own fields; `kind` (`character`, `prop`,
    `environment`, ...) is free text for the package to validate, narrow (a `Literal`) or ignore."""

    name: str
    kind: str | None = None

    def __post_init__(self) -> None:
        # Prompts quote a name as <name>: any text is one once its brackets and extra spaces go.
        self.name = " ".join(self.name.replace("<", " ").replace(">", " ").split())
        if not self.name:
            raise ValueError("an asset name needs at least one visible character")


def name_key(name: str) -> str:
    """How a mention matches a name: any case, with `_`, `-` and space alike."""
    return " ".join(re.sub(r"[_-]", " ", name).split()).casefold()


class Mention(NamedTuple):
    start: int
    end: int
    #: As written, brackets included.
    text: str
    #: The position of the asset it names; None when no asset has that name.
    position: int | None


def mentions(text: str, assets: Sequence[Asset]) -> list[Mention]:
    """Every `<...>` in `text`, in order, with the asset it names. Two assets with one name
    refuse."""
    names: dict[str, int] = {}
    for index, asset in enumerate(assets):
        key = name_key(asset.name)
        if key in names:
            raise InvalidRequest(
                f"two assets are named <{asset.name}>", code="asset_name", fields=("assets",)
            )
        names[key] = index
    return [
        Mention(m.start(), m.end(), m.group(0), names.get(name_key(m.group(1))))
        for m in MENTION.finditer(text)
    ]


def name_label(n: int, asset: Asset) -> str:
    """The label for a model with no words of its own for its assets: the name."""
    return asset.name


def render[A: Asset](
    text: str,
    assets: Sequence[A],
    label: Callable[[int, A], str] = name_label,
    *,
    adj: Adjustments | AdjustmentRows | None = None,
    field: str = "prompt",
) -> str:
    """`text` with each mention of an asset replaced by `label(n, asset)`, `n` its 1-based
    position. A mention of no asset stays as written, with one warning."""
    parts: list[str] = []
    unknown: dict[str, None] = {}
    at = 0
    for mention in mentions(text, assets):
        parts.append(text[at : mention.start])
        if mention.position is None:
            parts.append(mention.text)
            unknown[mention.text] = None
        else:
            parts.append(label(mention.position + 1, assets[mention.position]))
        at = mention.end
    parts.append(text[at:])
    if unknown and adj is not None:
        verb = "names" if len(unknown) == 1 else "name"
        adj.warn(field, f"{', '.join(unknown)} {verb} no asset; left as written")
    return "".join(parts)
