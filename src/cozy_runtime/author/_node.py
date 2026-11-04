"""What a function says of its box on a canvas that its types can't. Literal, read statically. A function is a node with or without one."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from cozy_runtime.author._errors import ConformanceError


def _words(key: str, value: object) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or not all(isinstance(word, str) and word for word in value):
        raise ConformanceError(f"Node {key} is a list of words", code="node")
    return tuple(value)


@dataclass(frozen=True, slots=True, kw_only=True)
class Node:
    """`label`: its name in the tool list, by default from its package and function. `show`: inputs on its default face beside its first
    text and what's required. `hide`: inputs a box never draws, which a file or `cozy run` can still set. `fill`: input to `take`, or to
    `<input>.<member>`, which the platform fills from what that input mentions, undrawn. `reserved`: the model's own tags in a text, never
    mentions. `ui`: a component in this package that draws the box."""

    label: str = ""
    show: Sequence[str] = ()
    hide: Sequence[str] = ()
    fill: Mapping[str, str] = field(default_factory=dict)
    reserved: Sequence[str] = ()
    ui: str = ""

    def __post_init__(self) -> None:
        if type(self.label) is not str or type(self.ui) is not str:
            raise ConformanceError("Node label and ui are text", code="node")
        if not isinstance(self.fill, Mapping) or not all(
            isinstance(key, str) and key and isinstance(value, str) and value for key, value in self.fill.items()
        ):
            raise ConformanceError("Node fill maps an input to `take` or `<input>.<member>`", code="node")
        for key in ("show", "hide", "reserved"):
            object.__setattr__(self, key, _words(key, getattr(self, key)))

    def document(self) -> dict[str, object]:
        """What the package interface says: only what was declared."""
        written = {"label": self.label, "show": list(self.show), "hide": list(self.hide), "fill": dict(self.fill),
                   "reserved": list(self.reserved), "ui": self.ui}  # fmt: skip
        return {key: value for key, value in written.items() if value}
