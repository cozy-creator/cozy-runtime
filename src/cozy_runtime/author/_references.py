"""Named references and `<name>` mentions (xs-135, renamed in xs-139): an opt-in convention for
prompts that name the files guiding their output.

A package declares a list of a `Reference` subclass carrying its media and fields (`Picture` for
images, the shot kinds in `_video`), and renders its prompt with `render` and a label: how its model
refers to the Nth reference. Nothing below this module knows the convention; a reference is an
ordinary struct whose files bind like any other, and a package wanting another shape declares it.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from typing import NamedTuple

import msgspec

from cozy_runtime.author._services import AdjustmentRows, Adjustments

#: A mention: `<`, any text without brackets or line breaks, `>`.
MENTION = re.compile(r"<([^<>\n]+)>")


class Reference(msgspec.Struct, kw_only=True):
    """A file guiding the output, which the prompt names as `<name>`. Subclasses add its media
    (`image`, `video`, `audio`, or any other file field) and their own fields."""

    name: str

    def __post_init__(self) -> None:
        # Prompts quote a name as <name>: any text is one once its brackets and extra spaces go.
        self.name = " ".join(self.name.replace("<", " ").replace(">", " ").split())
        if not self.name:
            raise ValueError("a reference name needs at least one visible character")


def name_key(name: str) -> str:
    """How a mention matches a name: any case, with `_`, `-` and space alike."""
    return " ".join(re.sub(r"[_-]", " ", name).split()).casefold()


class Mention(NamedTuple):
    start: int
    end: int
    #: As written, brackets included.
    text: str
    #: The position of the reference it names; None when no reference has that name.
    position: int | None


def _shared_names(references: Sequence[Reference]) -> list[str]:
    """The names a second reference repeats, which a mention reads as the first."""
    seen: set[str] = set()
    shared = []
    for reference in references:
        key = name_key(reference.name)
        if key in seen:
            shared.append(reference.name)
        seen.add(key)
    return shared


def mentions(text: str, references: Sequence[Reference]) -> list[Mention]:
    """Every `<...>` in `text`, in order, with the reference it names, the first of any that share
    a name."""
    names: dict[str, int] = {}
    for index, reference in enumerate(references):
        names.setdefault(name_key(reference.name), index)
    return [
        Mention(m.start(), m.end(), m.group(0), names.get(name_key(m.group(1))))
        for m in MENTION.finditer(text)
    ]


def name_label(n: int, reference: Reference) -> str:
    """The label for a model with no words of its own for its references: the name."""
    return reference.name


def render[R: Reference](
    text: str,
    references: Sequence[R],
    label: Callable[[int, R], str] = name_label,
    *,
    adj: Adjustments | AdjustmentRows | None = None,
    field: str = "prompt",
    listed: str = "references",
) -> str:
    """`text` with each mention of a reference replaced by `label(n, reference)`, `n` its 1-based
    position in the list `listed` names. A mention of no reference stays as written, with one
    warning; so does a second reference with a taken name, which mentions never reach."""
    parts: list[str] = []
    unknown: dict[str, None] = {}
    at = 0
    for mention in mentions(text, references):
        parts.append(text[at : mention.start])
        if mention.position is None:
            parts.append(mention.text)
            unknown[mention.text] = None
        else:
            parts.append(label(mention.position + 1, references[mention.position]))
        at = mention.end
    parts.append(text[at:])
    for name in _shared_names(references) if adj is not None else ():
        adj.warn(listed, f"two references are named <{name}>; a mention means the first")
    if unknown and adj is not None:
        verb = "names" if len(unknown) == 1 else "name"
        adj.warn(field, f"{', '.join(unknown)} {verb} no reference; left as written")
    return "".join(parts)
