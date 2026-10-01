"""Bounded, shared CPU preparation; a speculative task never consumes every worker."""

from __future__ import annotations

import threading
from collections.abc import Callable, Hashable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field


@dataclass
class _Entry[T]:
    load: Callable[[Callable[[], bool]], T]
    #: cancels the running load; given the demanding interest when demand pauses it
    stop: Callable[[Hashable | None], None]
    future: Future[T] = field(default_factory=Future)
    interests: dict[Hashable, bool] = field(default_factory=dict)
    cancel: threading.Event = field(default_factory=threading.Event)
    started: bool = False
    paused: bool = False

    @property
    def demanded(self) -> bool:
        return any(self.interests.values())


class Preparations[T]:
    """One future per immutable selection; demand promotes an already queued hint."""

    def __init__(self, maximum: int = 32, changed: Callable[[], None] = lambda: None):
        self.maximum = maximum
        #: told whenever an entry finishes or goes, so a request refused for room retries
        self.changed = changed
        self.pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="machine-prepare")
        self.lock = threading.RLock()
        self.entries: dict[Hashable, _Entry[T]] = {}
        self.closed = False

    def request(
        self,
        key: Hashable,
        interest: Hashable,
        load: Callable[[Callable[[], bool]], T],
        *,
        speculative: bool,
        cancel: Callable[[Hashable | None], None] = lambda _: None,
        stale: Callable[[T], bool] = lambda _: False,
    ) -> Future[T] | None:
        with self.lock:
            if self.closed:
                raise RuntimeError("preparation queue is closed")
            entry = self.entries.get(key)
            if entry is not None and entry.cancel.is_set() and not entry.paused:
                # Its last consumer left. Await its native cancellation before replacing
                # the transfer, so the same bytes never acquire two writers.
                if not entry.future.done():
                    return None
                del self.entries[key]
                entry = None
            if (
                entry is not None
                and entry.future.done()
                and not speculative
                and (entry.future.exception() is not None or stale(entry.future.result()))
            ):
                # A failed speculative attempt is not a permanent refusal. Demand reuses
                # an exact finished preparation unless `stale` says its custody may be gone.
                entry = self.entries[key] = _Entry(load, cancel, interests=entry.interests.copy())
            if entry is None:
                if speculative and len(self.entries) >= self.maximum - 1:
                    return None
                if len(self.entries) >= self.maximum:
                    completed = next((k for k, e in self.entries.items() if e.future.done()), None)
                    if completed is None:
                        return None
                    del self.entries[completed]
                entry = self.entries[key] = _Entry(load, cancel)
            entry.interests[interest] = entry.interests.get(interest, False) or not speculative
            self._schedule()
            return entry.future

    def retain(self, active: Callable[[Hashable], bool]) -> None:
        with self.lock:
            for key, entry in list(self.entries.items()):
                entry.interests = {k: v for k, v in entry.interests.items() if active(k)}
                if not entry.interests:
                    entry.paused = False  # abandoned work must not restart
                    entry.cancel.set()
                    entry.stop(None)
                    if not entry.started:
                        entry.future.cancel()
                    if entry.future.done():
                        del self.entries[key]
            self._schedule()

    def started(self) -> list[str]:
        """The preparations under way. A queued one, a hint still waiting for a worker
        among them, is no work yet."""
        with self.lock:
            return [
                str(key)
                for key, entry in self.entries.items()
                if entry.started and not entry.future.done()
            ]

    def _schedule(self) -> None:
        if self.closed:
            return
        running = [e for e in self.entries.values() if e.started and not e.future.done()]
        queued = sorted(
            (e for e in self.entries.values() if not e.started and not e.future.done()),
            key=lambda e: not e.demanded,
        )
        for entry in queued:
            if len(running) >= 2:
                break
            if not entry.demanded and any(not e.demanded for e in running):
                continue  # one hint at a time
            entry.started = True
            running.append(entry)
            self.pool.submit(self._run, entry)
        # A demand left waiting for a worker takes one hint's; a stopping hint is already
        # giving its worker up.
        waiting = [e for e in queued if e.demanded and not e.started]
        hints = [e for e in running if not e.demanded]
        yielding = sum(e.cancel.is_set() for e in hints)
        for demand, hint in zip(
            waiting[yielding:], [h for h in hints if not h.cancel.is_set()], strict=False
        ):
            hint.paused = True
            hint.cancel.set()
            hint.stop(next(interest for interest, wanted in demand.interests.items() if wanted))

    def _run(self, entry: _Entry[T]) -> None:
        try:
            result = entry.load(entry.cancel.is_set)
        except BaseException as exc:
            with self.lock:
                if entry.paused and entry.interests and not self.closed:
                    # Keep one shared result, and do not overlap the canceled transfer
                    # with its replacement. The loader has finished releasing its holds.
                    entry.started = False
                    entry.paused = False
                    entry.cancel = threading.Event()
                else:
                    entry.future.set_exception(exc)
        else:
            entry.future.set_result(result)
        finally:
            with self.lock:
                self._schedule()
            self.changed()

    def close(self) -> None:
        with self.lock:
            self.closed = True
            for entry in self.entries.values():
                entry.cancel.set()
                entry.stop(None)
                if not entry.started:
                    entry.future.cancel()
        self.pool.shutdown(wait=True)
