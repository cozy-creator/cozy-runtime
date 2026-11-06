"""A tool's box: the values Calcifer fills in, and the `tool=` record of `app.tool`.

A tool is a struct plus a function (design/tools.md). The struct is the box, made of the SDK's
types below and Python's own (`str`, `bool`, bounded numbers, `Literal`, a package's own structs);
the function turns it into calls. A prompt arrives as its text, as typed, and the references that
text mentions, each under its mention's spelling: `text.replace(f"<{key}>", ...)` is all a package
does with them.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Annotated, Generic, NewType, TypeVar

import msgspec

from cozy_runtime.author._errors import ConformanceError
from cozy_runtime.author._files import AudioFile, ImageFile, References, VideoFile

#: W:H as written: `16:9`, `2.39:1`, `16/9`. The box's own shape; `Ratios` names what it takes.
AspectRatio = NewType("AspectRatio", str)
#: A reproducible draw: unset, a new one for each render.
Seed = NewType("Seed", Annotated[int, msgspec.Meta(ge=0, le=(1 << 53) - 1)])

#: A mention: `<`, any text without brackets or line breaks, `>`.
MENTION = re.compile(r"<([^<>\n]+)>")


def name_key(name: str) -> str:
    """How a mention matches a name: any case, with `_`, `-` and space alike."""
    return " ".join(re.sub(r"[_-]", " ", name).split()).casefold()


def spoken(text: str) -> str:
    """A dotted mention in a text as a model reads it: `<kyogre.2>` as `<kyogre 2>`."""
    return MENTION.sub(lambda found: f"<{found[1].replace('.', ' ')}>", text)


class Image(msgspec.Struct, frozen=True, kw_only=True, tag_field="type", tag="image"):
    """A picture and what it shows: what a mention of a picture box gives, or a slot holds."""

    file: ImageFile
    description: str = ""


class Video(msgspec.Struct, frozen=True, kw_only=True, tag_field="type", tag="video"):
    """A clip, cut to its mention's `@in-out` range, and what it shows. A package's subclass names
    extras of the version it holds (H3's `latents`), filled when that package made it."""

    file: VideoFile
    description: str = ""


class Audio(msgspec.Struct, frozen=True, kw_only=True, tag_field="type", tag="audio"):
    """A sound and what it is."""

    file: AudioFile
    description: str = ""


#: What a prompt's references may be: `Prompt[Image]` narrows it.
T = TypeVar("T", bound=Image | Video | Audio)


# msgspec resolves this module's string annotations without a class's type parameters.
class Prompt(msgspec.Struct, Generic[T], frozen=True, kw_only=True):  # noqa: UP046
    """Text as typed, and the references it mentions: `Prompt[Image]` takes pictures only."""

    text: str = ""
    #: Each `<name>` of `text`, in order of first mention, spelled as the text spells it. On the
    #: wire, a list of the values, each with its `key`.
    references: References[T] = msgspec.field(default_factory=References)

    def __post_init__(self) -> None:
        # Built in Python, a prompt may be given a plain dict; the wire's list comes decoded.
        given: object = self.references
        if not isinstance(given, References):
            msgspec.structs.force_setattr(self, "references", References(self.references))


class Pickup(msgspec.Struct, frozen=True, kw_only=True):
    """Pick up from here: a take and the frame its kept start ends on, filled only by the gesture.
    Never drawn, written or keyed. A package subclasses it (H3 adds `beats_from`) and marks the
    field with `Snap` when the frame lands on a grid."""

    take: Video
    out_frame: Annotated[int, msgspec.Meta(ge=0)]


def _words(name: str, value: object) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or not all(isinstance(w, str) and w for w in value):
        raise ConformanceError(f"tool {name} is a list of words", code="tool")
    return tuple(value)


@dataclass(frozen=True, slots=True, kw_only=True)
class Tool:
    """`app.tool`'s own options. `label`: its name in the tool list. `named`: a new box's name,
    numbered (`shot` makes `shot_1`). `reserved`: the model's own tags in its text, never
    mentions. `ui`: a component in this package that draws the box."""

    label: str = ""
    named: str = ""
    reserved: Sequence[str] = ()
    ui: str = ""

    def __post_init__(self) -> None:
        if not all(type(word) is str for word in (self.label, self.named, self.ui)):
            raise ConformanceError("tool label, named and ui are text", code="tool")
        if "." in self.named:
            raise ConformanceError(
                "tool named= holds no `.`: a dot selects inside a box", code="tool"
            )
        object.__setattr__(self, "reserved", _words("reserved", self.reserved))

    def document(self) -> dict[str, object]:
        """The interface's `tool` record: only what was declared."""
        written = {
            "label": self.label,
            "named": self.named,
            "reserved": list(self.reserved),
            "ui": self.ui,
        }
        return {key: value for key, value in written.items() if value}
