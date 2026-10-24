"""The commands an executor answers, decoded once where it reads them.

TensorD's commands are `tensorfs.executor`'s classes: the Rust structs TensorD writes them from
(`protocol::executor`), decoded by serde, so this Runtime keeps no copy of them. Rank 0's commands
to its followers (`join`, `attention`, `kernels`, `request_memory`) are this Runtime's own, and
ride beside a forwarded TensorD command what only a follower reads (`Forwarded`).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, get_args

import msgspec
from msgspec import UNSET, UnsetType
from tensorfs import executor as wire
from tensorfs.executor import (
    Activate,
    Adapter,
    Bearer,
    Binding,
    CallInterface,
    Construction,
    Fork,
    Hello,
    InputFile,
    Invoke,
    Load,
    Malformed,
    Map,
    MemoSettings,
    ModelLoad,
    PrepareRequest,
    Probe,
    Shutdown,
    Start,
    TensorFSOptions,
    Unload,
)

from cozy_runtime.author._assets import GrantedInput
from cozy_runtime.internal.attention import Choice

__all__ = [
    "Activate",
    "Adapter",
    "Attention",
    "Bearer",
    "Binding",
    "CallInterface",
    "Construction",
    "Fork",
    "Forwarded",
    "GroupHardware",
    "Hello",
    "InputFile",
    "Invoke",
    "Join",
    "Kernels",
    "KnownCommand",
    "Load",
    "Malformed",
    "Map",
    "MemoSettings",
    "ModelLoad",
    "PrepareRequest",
    "Probe",
    "Release",
    "RequestMemory",
    "Shutdown",
    "Start",
    "TensorFSOptions",
    "Unload",
]


class Command(msgspec.Struct, frozen=True, kw_only=True, omit_defaults=True, tag_field="cmd"):
    """One of rank 0's commands to a follower; `cmd` names it."""


class Join(Command, frozen=True, kw_only=True, tag="join"):
    port: int
    backend: str = "nccl"
    #: A socket follows: a byte on it aborts this follower's collectives (`RankGroup.interrupt`).
    abort: bool = False
    #: The group forms again after a call ended short of memory (`mirror.mirror_component`).
    again: bool = False
    #: The NCCL route rank 0 chose from the group's measured links; set before joining.
    nccl: dict[str, str] = {}
    #: Every GPU then times NCCL against its copy engines (`copy_engine.calibrate`).
    calibrate: bool = False


class RequestMemory(Command, frozen=True, kw_only=True, tag="request_memory"):
    construction: str
    request_id: str
    phase: Literal["begin", "plan", "end"]
    steps: tuple[tuple[str, ...], ...] = ()


class Attention(Command, frozen=True, kw_only=True, tag="attention"):
    """A follower's attention: install rank 0's whole selection (`adopt`), hold a `pin`, or
    restore (neither)."""

    construction: str = ""
    pin: str | UnsetType = UNSET
    model_key: str | None = None
    adopt: Choice | UnsetType = UNSET
    #: rank 0 asks while it serves a request: this GPU opens no kernel on its own thread
    serving: bool = False


class Kernels(Command, frozen=True, kw_only=True, tag="kernels"):
    """A follower's deferred kernel installs (`Plan.when_ready`) at rank 0's boundary: which
    of `sets` this GPU has built, or, with `install`, install those rank 0 installed."""

    construction: str = ""
    model_key: str | None = None
    sets: tuple[str, ...] = ()
    install: bool = False


class GroupHardware(msgspec.Struct, frozen=True, kw_only=True):
    """Rank 0's hardware plan, broadcast with a group load."""

    degree: int
    devices: tuple[str, ...]


class Forwarded(msgspec.Struct, frozen=True, kw_only=True):
    """What rank 0 adds to a TensorD load it forwards, read by a follower alone: its hardware
    plan and, in a many-model load, which model."""

    group: GroupHardware | None = None
    model_key: str | None = None


class Release(msgspec.Struct, frozen=True, kw_only=True):
    """Where a package's code and interface live on this worker."""

    application: str = ""
    package_interface: str = ""


type Own = Join | RequestMemory | Attention | Kernels
type KnownCommand = (
    Hello
    | Shutdown
    | Probe
    | Fork
    | Map
    | Unload
    | Activate
    | Start
    | Load
    | PrepareRequest
    | Invoke
    | Join
    | RequestMemory
    | Attention
    | Kernels
)

_OWN = frozenset(str(kind.__struct_config__.tag) for kind in get_args(Own.__value__))

#: Every command name this Runtime's executor answers (a `bearer` is the seam's: `_supplied`).
NAMES = (wire.COMMANDS - {"bearer"}) | _OWN


def decode(frame: dict[str, object]) -> KnownCommand | Bearer:
    """The command a frame states. TensorD's decode through `tensorfs.executor` (`Malformed`
    names a needed member's path); rank 0's through msgspec (`msgspec.ValidationError`)."""
    if frame.get("cmd") in _OWN:
        own: Own = msgspec.convert(frame, Own)
        return own
    command: KnownCommand | Bearer = wire.decode_command(msgspec.json.encode(frame))
    return command


def forwarded(frame: dict[str, object]) -> Forwarded:
    """What rank 0 added to a forwarded command for this follower."""
    return msgspec.convert(frame, Forwarded)


def encode(command: Command | KnownCommand, **added: object) -> dict[str, object]:
    """The frame for one command, `cmd` included, with what rank 0 adds for its followers."""
    if isinstance(command, Command):
        frame = msgspec.to_builtins(command)
    else:
        assert not isinstance(command, Hello | Shutdown)
        frame = json.loads(command.json())
    assert isinstance(frame, dict)
    return {**frame, **msgspec.to_builtins(added)}


def release(start: Start) -> Release:
    return Release(application=start.application, package_interface=str(start.package_interface))


def parameter_names(binding: Binding) -> tuple[str, ...]:
    """Every declared name for this shared construction, its primary first."""
    names = (binding.model_parameter_name, *binding.model_parameter_names)
    return tuple(dict.fromkeys(name for name in names if name))


def bindings(load: Load) -> tuple[Binding, ...]:
    """A load's models: its many-model form's, else its one model."""
    return tuple(row.binding for row in load.models) or (load.binding,)


def granted(file: InputFile, input_id: str) -> GrantedInput:
    """One verified input file in the attempt's own spool, as the author's input."""
    return GrantedInput(
        input_id=input_id,
        local=Path(file.local),
        file_state=None,
        media_type=file.media_type,
        digest=file.digest,
        length=file.length,
        order=file.order,
    )
