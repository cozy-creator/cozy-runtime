"""GPU 0's commands to the other GPUs of its group, and how an executor reads any command.

TensorD's commands are `tensorfs.executor`'s classes, the Rust structs TensorD writes them from
(`executor-api`): this Runtime keeps no copy. GPU 0 hands its followers TensorD's `start`, `load`
and `map` with what only a follower reads beside them (`Forwarded`), and these commands of its own.
"""

from __future__ import annotations

import json
from typing import get_args

import msgspec
from msgspec import UNSET, UnsetType
from tensorfs import executor as wire


class Choice(msgspec.Struct, frozen=True, kw_only=True):
    """One GPU's whole selection, as a group's followers install it (`attention.install`): what
    each site serves, by (component, module path), per token band (`attention.Spec`: each band's
    upper bound on `min(Lq, Lk)`, 0 for none, and the kernels a call tries in turn), and the
    dense reference Sol recomputes rows with where Sol serves, with the exact kernel beside a
    quantizing one ("" for none)."""

    sites: tuple[tuple[str, str, tuple[tuple[int, tuple[str, ...]], ...]], ...] = ()
    dense: str = ""
    exact: str = ""


class _Own(msgspec.Struct, frozen=True, kw_only=True, omit_defaults=True, tag_field="cmd"):
    pass


class Shutdown(_Own, frozen=True, kw_only=True, tag="shutdown"):
    """Ends a follower's command loop."""


class Join(_Own, frozen=True, kw_only=True, tag="join"):
    """A follower's first half: rank 0's store reached, then it says it is ready (`Form`)."""

    port: int
    backend: str = "nccl"
    #: A socket follows: a byte on it aborts this follower's collectives (`RankGroup.interrupt`).
    abort: bool = False
    #: The group forms again after an attempt ended on one of its GPUs (`spmd`).
    again: bool = False
    #: The NCCL route rank 0 chose from the group's measured links; set before joining.
    nccl: dict[str, str] = {}


class Form(_Own, frozen=True, kw_only=True, tag="form"):
    """Every follower said it is ready: all GPUs enter group init together."""

    #: Every GPU then maps this exchange and checks its bytes (`copy_engine.form`).
    exchange: str = ""


class Attention(_Own, frozen=True, kw_only=True, tag="attention"):
    """A follower's attention: install rank 0's whole selection (`adopt`), hold a `pin`, or
    restore (neither)."""

    construction: str = ""
    pin: str | UnsetType = UNSET
    model_key: str | None = None
    adopt: Choice | UnsetType = UNSET
    #: rank 0 asks while it serves a request: this GPU opens no kernel on its own thread
    serving: bool = False
    #: the kernels the group serves from here (`kernel_ready.agreed`)
    kernels: tuple[str, ...] = ()


class Kernels(_Own, frozen=True, kw_only=True, tag="kernels"):
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


class Forwarded(msgspec.Struct, frozen=True, kw_only=True, omit_defaults=True):
    """What rank 0 adds to a TensorD load it forwards, read by a follower alone: its hardware
    plan and, in a many-model load, which model."""

    group: GroupHardware | None = None
    model_key: str | None = None


type Own = Shutdown | Join | Form | Attention | Kernels
type Command = wire.Command | Own

_OWN = frozenset(str(kind.__struct_config__.tag) for kind in get_args(Own.__value__))

#: Every command name an executor answers (a follower's `run` is `parallel.wire`'s).
NAMES = wire.COMMANDS | _OWN


def decode(frame: dict[str, object]) -> tuple[Command, Forwarded]:
    """The command a frame states and what rank 0 added to it. TensorD's decode through
    `tensorfs.executor` (`Malformed` names a needed member's path); rank 0's through msgspec
    (`msgspec.ValidationError`)."""
    if frame.get("cmd") in _OWN:
        own: Own = msgspec.convert(frame, Own)
        return own, Forwarded()
    return wire.decode_command(msgspec.json.encode(frame)), msgspec.convert(frame, Forwarded)


def encode(
    command: _Own | wire.Start | wire.Load | wire.Map | wire.PrepareRequest | wire.Invoke,
    added: Forwarded | None = None,
) -> dict[str, object]:
    """The frame of one command to a follower, `cmd` included, with what rank 0 adds."""
    if isinstance(command, _Own):
        frame = msgspec.to_builtins(command)
    else:
        frame = json.loads(command.json())
    assert isinstance(frame, dict)
    if added is not None:
        frame.update(msgspec.to_builtins(added))
    return frame
