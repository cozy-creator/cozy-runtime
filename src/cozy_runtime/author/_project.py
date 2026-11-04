"""A project's operations as a package's project adapter reads them, and what it returns (`App.project_adapter`).

A project (`project.yaml`) is general and a model isn't: an adapter turns one asset's inputs into this package's own call, so the
package owns its templating and vocabulary. It is pure, on the render machine in the package's environment: no GPU, no
network, no file contents. A file is its digest, which goes into the call's input as is. All of this is opt-in: a function no
adapter speaks for is called directly with its asset's inputs, each `<mention>` in a file input the file it names.
"""

from __future__ import annotations

import re
from typing import Any, Literal

import msgspec

from cozy_runtime.author._errors import InvalidRequest


class OperationFile(msgspec.Struct, kw_only=True):
    """A file by its digest, and what the machine measured of it. `alpha`: the share of a picture that is transparent."""

    sha256: str
    width: int = 0
    height: int = 0
    fps: float = 0
    frames: int = 0
    alpha: float = 0


class Read(msgspec.Struct, kw_only=True):
    """What an operation mentions. `name` is as the mention spells it. `kind`: its file's type. `asset_type`: `character`, `prop`,
    `environment`, or none. `description`: the words for whatever mentions it, the only ones that cross a mention: what made it is for its
    maker alone. `files`: its state by role (`image`, `video`, `audio`, `latents`, `voice`). `inputs`: how to take the file, `use_audio`
    and `follow`. `made_by`: `org/package/function@release`."""

    name: str
    kind: str = ""
    asset_type: str = ""
    description: str = ""
    files: dict[str, OperationFile] = {}
    inputs: dict[str, Any] = {}
    made_by: str = ""

    def files_of(self, field: str) -> list[OperationFile]:
        """A file field's files in order: `image`, then a list's `image-2`, `image-3`, ... ."""
        listed = {int(key[len(field) + 1 :]): file for key, file in self.files.items()
                  if key.startswith(f"{field}-") and key[len(field) + 1 :].isdigit()}  # fmt: skip
        return [*([self.files[field]] if field in self.files else []), *(listed[index] for index in sorted(listed))]


class Keep(msgspec.Struct, kw_only=True):
    """`--take`, `--in`: the version a new one keeps, up to `at_s` or to its end at `ends_s`."""

    take: int
    video: OperationFile
    latents: OperationFile | None = None
    at_s: float | None = None
    ends_s: float
    made_by: str


class Operation(msgspec.Struct, kw_only=True):
    """The work that makes one asset. `kind` is the file it produces, `image`, `video` or `audio`. `function` is the one its tool names.
    `inputs` is its box, after `defaults`; an adapter types them by subclassing this, which puts a signature in the package's interface
    and is that tool's controls."""

    version: Literal[3]
    asset: str
    kind: str = ""
    function: str = ""
    inputs: dict[str, Any]
    reads: list[Read]
    keep: Keep | None = None


class Invocation(msgspec.Struct, kw_only=True):
    """A function of this package and its exact input. Later render options, models and LoRAs go on top. A string in `input` that
    is exactly `$0.video` is that field of the result of the invocation at that position, one run earlier in the list."""

    function: str
    input: dict[str, Any]
    warnings: list[str] = []


class Invocations(msgspec.Struct, kw_only=True):
    """What an adapter's `list[Invocation]` is on the wire: run in order, usually one. None: this asset's files are what it reads."""

    invocations: list[Invocation]


class Refuse(InvalidRequest):
    """An adapter's no: nothing is made, and these words say why."""

    default_code = "project_refused"


# The inputs first-party packages share, so a project can swap one model for another without rewriting its file. A package opts in by
# annotating its adapter with them. Calcifer passes inputs through and knows none.


class ShotBeat(msgspec.Struct, kw_only=True):
    description: str = ""
    duration: float | None = None


class ShotPlate(msgspec.Struct, kw_only=True):
    video: str = ""
    use_audio: bool = False


class ShotInputs(msgspec.Struct, kw_only=True):
    """A shot. `first_frame`, `last_frame` and `plate`'s `video` mention an asset."""

    summary: str = ""
    beats: list[ShotBeat] = []
    style: str = ""
    soundscape: str = ""
    music: str = ""
    aspect_ratio: str | None = None
    first_frame: str = ""
    last_frame: str = ""
    plate: ShotPlate | None = None


class ShotOperation(Operation, kw_only=True):
    inputs: ShotInputs


class PictureInputs(msgspec.Struct, kw_only=True):
    """A picture: how to draw it, which nothing else reads."""

    instructions: str = ""
    style: str = ""
    aspect_ratio: str | None = None


class PictureOperation(Operation, kw_only=True):
    inputs: PictureInputs


_MENTION = re.compile(r"<([^<>\n]+)>")


def spoken(text: str) -> str:
    """A dotted name's mention in a text as a model reads it: `<kyogre.primordial>` as `<kyogre primordial>`."""
    return _MENTION.sub(lambda found: f"<{found[1].replace('.', ' ')}>", text)


def adapter_result(annotation: object) -> object:
    """A result annotated `list[Invocation]` has the schema of `Invocations`."""
    return Invocations if annotation == list[Invocation] else annotation
