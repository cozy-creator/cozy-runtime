"""The machine's pinned host tier: each weight set's memfd, kept across executors.

An executor's weight plane fills one memfd per weight set (its layout plus a state page that
marks which regions are Ready) and offers it here. The Worker keeps it, so a replacement
executor, or another package's executor registering the same layout, adopts the filled
regions instead of reading the disk (weight-plane.md §3). Pinned memory is unreclaimable, so
the tier keeps at most half of what the host has, closing the least recently used first:
page cache and disk stay beneath it.
"""

from __future__ import annotations

import contextlib
import os
import threading
from collections.abc import Callable
from dataclasses import dataclass

from cozy_runtime.author._executor_requests import (
    Answer,
    DescriptorReply,
    HostTier,
    Reply,
    Tier,
    refuse,
)


@dataclass(slots=True)
class Held:
    fd: int
    nbytes: int
    used: int


class HostTiers:
    def __init__(self, available: Callable[[], int]) -> None:
        #: the host's available bytes now (cgroup-aware); the tier keeps half of it with
        #: what it already holds
        self.available = available
        self.held: dict[tuple[str, str], Held] = {}
        self.clock = 0
        #: loads on separate lanes answer at once
        self.lock = threading.Lock()

    def answer(self, request: HostTier) -> Reply:
        """An executor's `HostTier` exchange: keep an offered memfd, or hand one back."""
        with self.lock:
            return self._answer(request)

    def _answer(self, request: HostTier) -> Reply:
        key = (request.name, request.layout)
        self.clock += 1
        if request.offer:
            if request.memfd < 0:
                return refuse("host_tier_absent", "an offer arrived without its memfd")
            kept = Held(request.memfd, _resident(request.memfd), self.clock)  # may raise: not kept
            prior = self.held.pop(key, None)
            if prior is not None:
                os.close(prior.fd)
            self.held[key] = kept
            self._trim()
            return Answer(ok=True)
        if request.memfd >= 0:  # an ask carries none; one that does is not kept
            os.close(request.memfd)
        row = self.held.get(key)
        if row is None:
            return Tier(ok=True, held=False)
        row.used = self.clock
        return DescriptorReply(Tier(ok=True, held=True), os.dup(row.fd))

    def nbytes(self) -> int:
        with self.lock:
            return self._nbytes()

    def _nbytes(self) -> int:
        return sum(row.nbytes for row in self.held.values())

    def _trim(self) -> None:
        for row in self.held.values():
            with contextlib.suppress(OSError):
                row.nbytes = _resident(row.fd)
        while self.held and self._nbytes() > (max(self.available(), 0) + self._nbytes()) // 2:
            key = min(self.held, key=lambda k: self.held[k].used)
            os.close(self.held.pop(key).fd)

    def close(self) -> None:
        with self.lock:
            for row in self.held.values():
                os.close(row.fd)
            self.held.clear()


def _resident(fd: int) -> int:
    """The memfd's allocated bytes: evicted regions are holes and cost nothing."""
    return os.fstat(fd).st_blocks * 512
