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

from cozy_runtime.author._errors import InvalidRequest
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


def mentions(
    text: str, references: Sequence[Reference], field: str = "references"
) -> list[Mention]:
    """Every `<...>` in `text`, in order, with the reference it names. Two references with one
    name refuse, naming `field`."""
    names: dict[str, int] = {}
    for index, reference in enumerate(references):
        key = name_key(reference.name)
        if key in names:
            raise InvalidRequest(
                f"two references are named <{reference.name}>",
                code="reference_name",
                fields=(field,),
            )
        names[key] = index
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
    warning."""
    parts: list[str] = []
    unknown: dict[str, None] = {}
    at = 0
    for mention in mentions(text, references, listed):
        parts.append(text[at : mention.start])
        if mention.position is None:
            parts.append(mention.text)
            unknown[mention.text] = None
        else:
            parts.append(label(mention.position + 1, references[mention.position]))
        at = mention.end
    parts.append(text[at:])
    if unknown and adj is not None:
        verb = "names" if len(unknown) == 1 else "name"
        adj.warn(field, f"{', '.join(unknown)} {verb} no reference; left as written")
    return "".join(parts)
