"""A project's operations as a package's project adapter reads them, and what it returns (`App.project_adapter`).

A project (`project.yaml`) is general and a model isn't: an adapter turns one operation into this package's own call, so the
package owns its templating and vocabulary. It is pure, on the render machine in the package's environment: no GPU, no
network, no file contents. A file is its digest, which goes into the call's input as is. All of this is opt-in: a function no
adapter speaks for is called directly with its node's fields, each `<mention>` the file it names.
"""

from __future__ import annotations

import re
from collections.abc import Collection
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
    """What an operation mentions: what the file says of it, and its state now by role (`image`, `video`, `audio`, `voice`, ...). `name`
    is as the mention spells it, so `coast.2` for that revision. `kind`: an entity's (`character`, `prop`, `environment`), or an asset's
    file type. `made_by`: `org/package/function@release`."""

    name: str
    kind: str = ""
    fields: dict[str, Any] = {}
    files: dict[str, OperationFile] = {}
    made_by: str = ""

    def files_of(self, field: str) -> list[OperationFile]:
        """A file field's files in order: `image`, then a list's `image-2`, `image-3`, ... ."""
        listed = {int(key[len(field) + 1 :]): file for key, file in self.files.items()
                  if key.startswith(f"{field}-") and key[len(field) + 1 :].isdigit()}  # fmt: skip
        return [*([self.files[field]] if field in self.files else []), *(listed[index] for index in sorted(listed))]


class Write(msgspec.Struct, kw_only=True):
    """A revision of a series this operation's text names (`coast.1`), to screenshot `at_s` into it, or at its end. `fields`: what
    the file says of that revision as an asset of its own, if it does."""

    name: str
    at_s: float | None = None
    fields: dict[str, Any] = {}


class Keep(msgspec.Struct, kw_only=True):
    """`--take`, `--in`: the version a new one keeps, up to `at_s` or to its end at `ends_s`."""

    take: int
    video: OperationFile
    latents: OperationFile | None = None
    at_s: float | None = None
    ends_s: float
    made_by: str


class Operation(msgspec.Struct, kw_only=True):
    """The work that makes one asset. `kind` is the file it produces, `image`, `video` or `audio`, which names the adapter. `function` is
    the one its tool names, for this adapter to make the input of; empty when it names only this package, for the adapter to pick. `fields`
    is everything else the file says of it, after defaults; an adapter types them by subclassing this, which puts a signature in the
    package's interface and, where it speaks for a function, is that tool's box."""

    version: Literal[2]
    asset: str
    kind: str = ""
    function: str = ""
    fields: dict[str, Any]
    #: What the file says for this package alone, any field to any value: `turbo`, a model's own settings. Not part of the standard.
    options: dict[str, Any] = {}
    reads: list[Read]
    writes: list[Write] = []
    keep: Keep | None = None

    def known_options(self, known: Collection[str]) -> tuple[dict[str, Any], list[str]]:
        """The options this adapter reads, and one warning for each key it doesn't know."""
        return ({key: value for key, value in self.options.items() if key in known},
                [f"{self.asset}: this package has no option `{key}`: it's ignored" for key in self.options if key not in known])  # fmt: skip


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


# The fields first-party packages share, so a project can swap one model for another without rewriting its file. A package opts in by
# annotating its adapter with them. Calcifer passes fields through and knows none.


class BeatWrite(msgspec.Struct, kw_only=True):
    """A screenshot: `image` is a revision (`coast.1`), `at` seconds into its beat, else the beat's end."""

    image: str
    at: float | None = None


class ShotBeat(msgspec.Struct, kw_only=True):
    description: str = ""
    duration: float | None = None
    writes: list[BeatWrite] = []


class ShotPlate(msgspec.Struct, kw_only=True):
    video: str = ""
    use_audio: bool = False


class ShotFields(msgspec.Struct, kw_only=True):
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
    fields: ShotFields


class PictureFields(msgspec.Struct, kw_only=True):
    """A picture. `description` is what it is, which what reads it reads too; `instructions` how to draw this picture, which nothing
    else reads."""

    description: str = ""
    instructions: str = ""
    style: str = ""
    aspect_ratio: str | None = None


class PictureOperation(Operation, kw_only=True):
    fields: PictureFields


_MENTION = re.compile(r"<([^<>\n]+)>")


def spoken(text: str) -> str:
    """A dotted name's mention in a text as a model reads it: `<kyogre.primordial>` as `<kyogre primordial>`, `<coast.2>` as `<coast 2>`."""
    return _MENTION.sub(lambda found: f"<{found[1].replace('.', ' ')}>", text)


def adapter_result(annotation: object) -> object:
    """A result annotated `list[Invocation]` has the schema of `Invocations`."""
    return Invocations if annotation == list[Invocation] else annotation


class ProjectPicture(msgspec.Struct, kw_only=True):
    """A picture a node is made from: what it shows, and its file's digest."""

    name: str
    image: str


class ProjectBeat(msgspec.Struct, kw_only=True):
    """From `at` seconds into the shot, what happens."""

    at: float
    description: str = ""


class ProjectReference(msgspec.Struct, kw_only=True):
    """Something a shot reads. `kind`: `character`, `prop`, `environment`, `world_state`, `video` or `audio`. `fields`: what
    `project.yaml` says of it beyond these (`use_audio`, a window), for the adapter to use or leave."""

    name: str
    kind: str
    description: str = ""
    image: str | None = None
    voice: str | None = None
    video: str | None = None
    audio: str | None = None
    fields: dict[str, Any] = msgspec.field(default_factory=dict)


class ProjectContinue(msgspec.Struct, kw_only=True):
    """The call before this one of a shot made in consecutive calls: its video and latents, by digest."""

    video: str
    latents: str | None = None


class ProjectShot(msgspec.Struct, kw_only=True):
    """One call of a shot: its whole text, and from `start_s` to `end_s` on its timeline. It is the whole shot unless a
    mid-shot write splits it. A cut is a new shot; `beats` are in order."""

    version: Literal[1] = 1
    name: str
    function: str
    summary: str = ""
    beats: list[ProjectBeat] = msgspec.field(default_factory=list)
    soundscape: str = ""
    music: str = ""
    style: str = ""
    aspect_ratio: str | None = None
    duration_s: float
    start_s: float = 0.0
    end_s: float
    references: list[ProjectReference] = msgspec.field(default_factory=list)
    plate: dict[str, Any] | None = None
    first_frame: str | None = None
    last_frame: str | None = None
    continues: ProjectContinue | None = None


class ProjectUpscale(msgspec.Struct, kw_only=True):
    """A shot's video taken to the project's `resolution`, with the text it was made from. `latents` only when this package
    made them."""

    version: Literal[1] = 1
    name: str
    function: str
    video: str
    latents: str | None = None
    resolution: str | None = None
    summary: str = ""
    beats: list[ProjectBeat] = msgspec.field(default_factory=list)
    style: str = ""
    references: list[ProjectReference] = msgspec.field(default_factory=list)


class ProjectEntity(msgspec.Struct, kw_only=True):
    """A character, prop or environment's picture. `kind`: `character`, `prop` or `environment`. `description` is what it is,
    which shots read too; `instructions` is how to draw this picture, which nothing else reads."""

    version: Literal[1] = 1
    name: str
    function: str
    kind: str
    description: str = ""
    instructions: str = ""
    style: str = ""
    aspect_ratio: str | None = None
    pictures: list[ProjectPicture] = msgspec.field(default_factory=list)


class ProjectSetup(msgspec.Struct, kw_only=True):
    """An environment from a camera position. `pictures` open with the environment, or with `from_frame` a shot's frame."""

    version: Literal[1] = 1
    name: str
    function: str
    environment: str
    description: str = ""
    camera: str = ""
    from_frame: bool = False
    style: str = ""
    aspect_ratio: str | None = None
    pictures: list[ProjectPicture] = msgspec.field(default_factory=list)


class ProjectWorldState(msgspec.Struct, kw_only=True):
    """A world state's first picture: how it starts, and who and what is where."""

    version: Literal[1] = 1
    name: str
    function: str
    init: str = ""
    blocking: str = ""
    style: str = ""
    aspect_ratio: str | None = None
    pictures: list[ProjectPicture] = msgspec.field(default_factory=list)
