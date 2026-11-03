"""A project's nodes as a package's project adapter reads them, and what it returns (`App.project_adapter`).

A project (`project.yaml`) is general and a model isn't: an adapter turns one node into this package's own call, so the
package owns its templating and vocabulary. It is pure, on the render machine in the package's environment: no GPU, no
network, no file contents. A file is its digest, which goes into the call's input as is. `version` is the project format
it reads: an adapter that reads more than one takes a union. All of this is opt-in: a package with none gets one flat
`prompt`.
"""

from __future__ import annotations

from typing import Any, Literal

import msgspec


class Invocation(msgspec.Struct, kw_only=True):
    """A function of this package and its exact input. Later render options, models and LoRAs go on top."""

    function: str
    input: dict[str, Any]


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
