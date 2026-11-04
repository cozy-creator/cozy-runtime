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
            type(value).__module__,
            type(value).__qualname__,
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


def execution_key(configuration: str | None, namespace: str) -> str | None:
    """The key of a declared configuration's passes; a pass with an unknown one publishes nothing.
    A site's level can change between passes, so a sample is a measured hint for the next."""
    if configuration is None:
        return None
    return f"{namespace}:{hashlib.sha256(configuration.encode()).hexdigest()}"


class Observation:
    def __init__(self, device: int, token: int, completion_stream: Any = None) -> None:
        self.device, self.token = device, token
        self.completion_stream = completion_stream
        self.result: dict[str, Any] | None = None
        self.completed = False
        self.choices = 0

    def invalidate(self, reason: str) -> None:
        vt.phase_invalidate(self.device, self.token, reason=reason)

    def complete(self) -> None:
        self.completed = True


_ACTIVE: ContextVar[Observation | None] = ContextVar("cozy_working_phase", default=None)


def invalidate(reason: str) -> None:
    if (active := _ACTIVE.get()) is not None:
        active.invalidate(reason)


def choice() -> None:
    if (active := _ACTIVE.get()) is not None:
        active.choices += 1


def completion_owner(device: int, stream: int) -> int | None:
    """Only the eager owner's actual synchronized stream can qualify a submitted leaf."""
    active = _ACTIVE.get()
    if (
        active is not None
        and active.device == device
        and active.completion_stream is not None
        and int(active.completion_stream.cuda_stream) == stream
    ):
        return active.token
    return None


def begin_split(
    device: int,
    key: str,
    choice: str,
    selection: dict[str, Any],
    *,
    stream: int,
) -> int:
    """Forward the current native cohort's exact selection generation."""
    return vt.split_begin(
        device,
        key,
        choice,
        stream=stream,
        completion_owner=completion_owner(device, stream),
        plan_generation=selection["plan_generation"],
    )


@contextmanager
def observing(
    device: int | None, mode: Mode, records: list[dict[str, Any]] | None, key: str | None = None
) -> Iterator[Observation | None]:
    """One execution's observation; none where another owns the device (an enclosing loop)."""
    if device is None or records is None:
        yield None
        return
    try:
        token = vt.phase_enter(device, key, mode=mode)
    except vt.PhaseOverlapError:
        yield None
        return
    observation = Observation(
        device, token, torch.cuda.current_stream(device) if mode == "eager" else None
    )
    active = _ACTIVE.set(observation)
    succeeded = False
    try:
        yield observation
        succeeded = True
    finally:
        try:
            record = vt.phase_leave(
                device,
                observation.token,
                completed=succeeded and (mode != "eager" or observation.completed),
            )
            sites = record.pop("split_observations", ())
            observation.result = {
                **record, "cut_choices": observation.choices, "sites": len(sites), "passes": 1,
            }
            last = records[-1] if records else None
            if last is not None and key is not None and last.get("key") == key:
                # Repeated passes of one key are one row: an LM's tokens, a DiT's steps.
                observation.result["passes"] += last["passes"]
                observation.result["peak_bytes"] = max(record["peak_bytes"], last["peak_bytes"])
                records[-1] = observation.result
            else:
                records.append(observation.result)
        finally:
            _ACTIVE.reset(active)
