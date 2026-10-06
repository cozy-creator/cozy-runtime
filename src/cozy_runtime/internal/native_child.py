"""Bounded JSON framing for a standalone native CPU helper process."""

import faulthandler
import os
import sys
import threading
import traceback
from collections.abc import Callable
from typing import Annotated, Literal

import msgspec

MAX_BYTES = 2 << 20


class Launch[S](msgspec.Struct, frozen=True, kw_only=True):
    store: str
    parent_pid: Annotated[int, msgspec.Meta(gt=0)]
    step: S


class Answer(msgspec.Struct, frozen=True, kw_only=True, tag_field="answer"):
    pass


class Sample(Answer, frozen=True, tag="progress"):
    stage: Literal["download", "convert"]
    position: int
    total: int


class Refused(Answer, frozen=True, tag="refused"):
    code: str


_STDOUT = threading.Lock()


def _line(body: bytes) -> None:
    with _STDOUT:
        sys.stdout.buffer.write(body + b"\n")
        sys.stdout.buffer.flush()


def report(stage: Literal["download", "convert"], position: int, total: int) -> None:
    _line(msgspec.json.encode(Sample(stage=stage, position=position, total=total)))


def serve[S](kind: type[Launch[S]], execute: Callable[[Launch[S]], Answer]) -> None:
    """Execute the bounded launch on stdin; the helper ends with its parent."""
    faulthandler.enable()
    try:
        try:
            command = sys.stdin.buffer.read(MAX_BYTES + 1)
            if len(command) > MAX_BYTES:
                _line(msgspec.json.encode(Refused(code="native_step_bound")))
                return
            launch = msgspec.json.decode(command, type=kind)
        except msgspec.DecodeError:
            _line(msgspec.json.encode(Refused(code="native_step_malformed")))
            return
        parent = launch.parent_pid

        def parent_lifetime() -> None:
            tick = threading.Event()
            while os.getppid() == parent:
                tick.wait(0.25)
            os._exit(70)

        threading.Thread(target=parent_lifetime, daemon=True, name="source-parent").start()
        reply = execute(launch)
    except Exception as exc:
        traceback.print_exc()
        code = getattr(exc, "code", "native_source_failed")
        reply = Refused(code=code if isinstance(code, str) else "native_source_failed")
    body = msgspec.json.encode(reply)
    _line(
        body if len(body) <= MAX_BYTES else msgspec.json.encode(Refused(code="source_result_bound"))
    )
