"""Successful eager working-demand samples; no Runtime memory policy lives here."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import fields, is_dataclass
from enum import Enum
from typing import Any, Literal

import tensorfs.varena.torch as vt
import torch

Mode = Literal["eager", "capture", "replay", "preparation"]


def _semantic(value: object) -> object:
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    if isinstance(value, Enum):
        return (type(value).__module__, type(value).__qualname__, _semantic(value.value))
    if isinstance(value, torch.dtype | torch.device):
        return (type(value).__name__, str(value))
    if isinstance(value, tuple):
        return tuple(_semantic(item) for item in value)
    if isinstance(value, frozenset):
        return sorted((_semantic(item) for item in value), key=_encoded)
    if is_dataclass(value) and not isinstance(value, type):
        parameters = vars(type(value)).get("__dataclass_params__")
        if parameters is None or not parameters.frozen:
            raise TypeError("mutable phase configuration")
        return (
            type(value).__module__, type(value).__qualname__,
            tuple((field.name, _semantic(getattr(value, field.name))) for field in fields(value)),
        )
    raise TypeError(f"unrepresented phase configuration: {type(value).__qualname__}")


def _encoded(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), allow_nan=False).encode()


def configuration(value: object) -> str | None:
    """Only declared immutable metadata can identify a reusable execution."""
    try:
        return _encoded(_semantic(value)).decode()
    except (TypeError, ValueError):
        return None


class Observation:
    def __init__(self, device: int, token: int, completion_stream: Any = None) -> None:
        self.device, self.token = device, token
        self.completion_stream = completion_stream
        self.key: str | None = None
        self.result: dict[str, Any] | None = None
        self.completed = False
        self.choices = 0
        self._cuts = hashlib.sha256()

    def invalidate(self, reason: str) -> None:
        vt.phase_invalidate(self.device, self.token, reason=reason)

    def choice(self, value: object) -> None:
        self.choices += 1
        try:
            encoded = _encoded(_semantic(value))
        except (TypeError, ValueError):
            self.invalidate("unrepresented_dispatch")
            return
        self._cuts.update(encoded + b"\n")

    def key_for(self, configuration: object | None, namespace: str) -> str | None:
        """A sealed dispatch fingerprint; making a key alone never publishes a sample."""
        if configuration is None:
            return None
        try:
            encoded = _encoded(_semantic(configuration))
        except (TypeError, ValueError):
            return None
        return namespace + ":" + hashlib.sha256(
            encoded + b"\n" + self._cuts.digest()
        ).hexdigest()

    def complete(self, configuration: object | None) -> None:
        self.completed = True
        self.key = self.key_for(configuration, "runtime.eager/1")
        # Unknown phase identity prevents its watermark from publishing, but does not erase
        # a valid completion proof for fully represented leaf sites on the same stream.


_ACTIVE: ContextVar[Observation | None] = ContextVar("cozy_working_phase", default=None)


def invalidate(reason: str) -> None:
    if (active := _ACTIVE.get()) is not None:
        active.invalidate(reason)


def choice(value: object) -> None:
    if (active := _ACTIVE.get()) is not None:
        active.choice(value)


def completion_owner(device: int, stream: int) -> int | None:
    """Only the eager owner's actual synchronized stream can qualify a submitted leaf."""
    active = _ACTIVE.get()
    if (
        active is not None and active.device == device and active.completion_stream is not None
        and int(active.completion_stream.cuda_stream) == stream
    ):
        return active.token
    return None


@contextmanager
def observing(
    device: int | None, mode: Mode, records: list[dict[str, Any]] | None
) -> Iterator[Observation | None]:
    if device is None or records is None:
        yield None
        return
    # Actual adaptive cuts are unknown at entry; no preceding pass is a promise about this one.
    observation = Observation(
        device, vt.phase_enter(device, None, mode=mode),
        torch.cuda.current_stream(device) if mode == "eager" else None,
    )
    token = _ACTIVE.set(observation)
    succeeded = False
    try:
        yield observation
        succeeded = True
    finally:
        try:
            record = vt.phase_leave(
                device, observation.token,
                completed=succeeded and (mode != "eager" or observation.completed),
                key=observation.key,
            )
            observation.result = {**record, "cut_choices": observation.choices}
            records.append(observation.result)
        finally:
            _ACTIVE.reset(token)
