"""The executor's work toward its calls' results, told to the machine (cl-648): its CPU no longer
says a call moves, so the points the Runtime already passes count. A construction leg ends, a
graph replays, an LM step decodes, a kernel build its call waits on moves (`done`); varena binds a
block (`tensorfs.varena.torch.progress`). Once a second, while that count moved, each call in
flight is told it. Calls side by side in one executor share it, as they share its GPU."""

from __future__ import annotations

import sys
import threading
import time
from collections.abc import Callable
from typing import Any

#: Work units done in this process outside varena.
done = 0

SAMPLE_SECONDS = 1.0

_calls: set[str] = set()
#: Held while the calls are told: a call that ended is told nothing after its reply.
_telling = threading.Lock()


def count() -> int:
    """This process's work so far: the Runtime's units and the blocks varena bound."""
    varena = sys.modules.get("tensorfs.varena.torch")
    return done + int(getattr(varena, "progress", 0))


def begin(call: str) -> None:
    with _telling:
        _calls.add(call)


def end(call: str) -> None:
    with _telling:
        _calls.discard(call)


def follow(send: Callable[[dict[str, Any]], None]) -> None:
    """Tells each call in flight the work count once a second while it moves, on a daemon
    thread of its own."""

    def run() -> None:
        told = count()
        while True:
            time.sleep(SAMPLE_SECONDS)
            now = count()
            if now == told:
                continue
            told = now
            with _telling:
                for call in _calls:
                    send({"event": "progress", "request_id": call, "kind": "work", "work": now})

    threading.Thread(target=run, name="work", daemon=True).start()
