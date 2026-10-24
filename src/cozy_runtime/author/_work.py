"""The executor's work toward its calls' results, told to the machine (cl-648): its CPU no longer
says a call moves, so the points the Runtime already passes count. A construction leg ends, a
graph replays, an LM step decodes, a kernel build its call waits on moves (`done`); varena binds a
block (`tensorfs.varena.torch.progress`) and lands bytes (`landed()`: fills and page-ins, as each
copy lands); the GPU completes work queued on the executor's stream (`gpu`: a mark recorded only
while the stream is busy, counted once the device passes it, so a long kernel-only phase moves
and an idle GPU says nothing). Once a second, while that count moved, each call in flight is told
it. Calls side by side in one executor share it, as they share its GPU."""

from __future__ import annotations

import sys
import threading
import time
from collections import deque
from collections.abc import Callable
from typing import Any

from cozy_runtime.author._observations import DeviceClock

#: Work units done in this process outside varena.
done = 0
#: GPU marks the device passed (`follow`)
gpu = 0

SAMPLE_SECONDS = 1.0

_calls: set[str] = set()
#: Held while the calls are told: a call that ended is told nothing after its reply.
_telling = threading.Lock()


def count() -> int:
    """This process's work so far: the Runtime's units, the GPU marks passed, and varena's bound
    blocks and landed bytes (a TensorFS without one counts none of it)."""
    varena = sys.modules.get("tensorfs.varena.torch")
    landed = getattr(varena, "landed", None)
    return done + gpu + int(getattr(varena, "progress", 0)) + (int(landed()) if landed else 0)


def begin(call: str) -> None:
    with _telling:
        _calls.add(call)


def end(call: str) -> None:
    with _telling:
        _calls.discard(call)


def follow(send: Callable[[dict[str, Any]], None], clock: DeviceClock | None = None) -> None:
    """Tells each call in flight the work count once a second while it moves, on a daemon
    thread of its own. With the device's `clock`, a mark goes on its stream each sample while
    work is queued there, and each the device passed counts (`gpu`)."""

    def run() -> None:
        global gpu
        told, marks = count(), deque[object]()
        while True:
            time.sleep(SAMPLE_SECONDS)
            if clock is not None:
                while marks and clock.passed(marks[0]):
                    marks.popleft()
                    gpu += 1
                if clock.busy():
                    marks.append(clock.mark())
            now = count()
            if now == told:
                continue
            told = now
            with _telling:
                for call in _calls:
                    send({"event": "progress", "request_id": call, "kind": "work", "work": now})

    threading.Thread(target=run, name="work", daemon=True).start()
