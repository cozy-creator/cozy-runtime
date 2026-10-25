"""Host stalls inside a step: the request thread sampled while its GPU's queue is empty, named by
the code it was in, so a step that waits on the host says where.

cr-211: H3's first step spent minutes planning a `split` in Python with the GPU idle, and the run
record said only that step 1 was long. Every `PERIOD` this samples the stepping thread's stack;
when the GPU it steps on has no work queued, the sample counts for the innermost frame outside
the standard library and torch (`cozy_runtime.author._split._finest`, `triton.compiler...`).
Each step's counts become stages `host · <code>` (`cozy run show` lists them under the stage they
fall in); a code under `FLOOR_MS` in a step is launch gaps, not a stall, and is dropped.
"""

from __future__ import annotations

import sys
import threading
from collections import Counter
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from types import FrameType

PERIOD = 0.1
FLOOR_MS = 1000.0
_LIBRARY = frozenset(sys.stdlib_module_names) | {"torch"}


def where(frame: FrameType | None) -> str:
    """The innermost frame outside the standard library and torch, as `module.qualname`."""
    innermost = frame
    while frame is not None:
        module = str(frame.f_globals.get("__name__", ""))
        if module.partition(".")[0] not in _LIBRARY:
            return f"{module}.{frame.f_code.co_qualname}"
        frame = frame.f_back
    if innermost is None:
        return "?"
    return f"{innermost.f_globals.get('__name__', '')}.{innermost.f_code.co_qualname}"


class Watch:
    """Samples one thread while `idle()` says its GPU has nothing queued, until `stop` or until
    `alive()` says its attempt ended."""

    def __init__(self, thread: int, idle: Callable[[], bool], alive: Callable[[], bool]) -> None:
        self.thread, self.idle, self.alive = thread, idle, alive
        self.counts: Counter[str] = Counter()
        self.lock = threading.RLock()
        self.done = threading.Event()
        threading.Thread(target=self._run, name="cozy-host-stalls", daemon=True).start()

    def _run(self) -> None:
        while not self.done.wait(PERIOD) and self.alive():
            with self.lock:
                if self.done.is_set() or not self.alive():
                    return
                try:
                    if not self.idle():
                        continue
                except Exception:  # the GPU's queue cannot be read: nothing is said
                    return
                label = where(sys._current_frames().get(self.thread))
                self.counts[label] += 1

    @contextmanager
    def pause(self) -> Iterator[None]:
        """Drain an in-flight observation before capture and exclude new CUDA queries."""
        with self.lock:
            yield

    def take(self) -> dict[str, float]:
        """Each code's stalled milliseconds since the last take, those over `FLOOR_MS`."""
        with self.lock:
            counts, self.counts = self.counts, Counter()
        return {k: n * PERIOD * 1000 for k, n in counts.items() if n * PERIOD * 1000 >= FLOOR_MS}

    def stop(self) -> None:
        with self.lock:
            self.done.set()


def watch(alive: Callable[[], bool]) -> Watch | None:
    """A watch on the calling thread's GPU, or None where no CUDA runs in this process (torch
    not imported or CUDA not started: there is no queue to read, and imports stay the caller's)."""
    torch = sys.modules.get("torch")
    if torch is None or not torch.cuda.is_initialized():
        return None
    stream = torch.cuda.default_stream(torch.cuda.current_device())
    return Watch(threading.get_ident(), stream.query, alive)
