"""The machine's pinned host tier: each weight set's memfd, kept across executors.

An executor's weight plane fills one memfd per weight set (its layout plus a state page that
marks which regions are Ready) and offers it here. The Worker keeps it, holding a claim on its
regions (`plane.hold_tier`), so a replacement executor, or another package's executor
registering the same layout, adopts the filled regions instead of reading the disk
(weight-plane.md §3). Pinned memory is unreclaimable, so the tiers held here stay inside the
machine's pinned total (`weight_policy.pinned_total`: half of what the host has under its
tightest limit), the least recently used released first (which frees what no executor
claims). Page cache and disk stay beneath it.
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
from cozy_runtime.internal import plane, weight_policy


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
            # either raising leaves the offered memfd to the caller, which closes it
            nbytes = _resident(request.memfd)
            kept = Held(plane.hold_tier(request.memfd), nbytes, self.clock)
            os.close(request.memfd)  # the hold is its own descriptor
            prior = self.held.pop(key, None)
            if prior is not None:
                plane.release_tier(prior.fd)
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

    def contents(self) -> dict[str, int]:
        """Bytes held per weight set (its construction and component), for placement."""
        with self.lock:
            held: dict[str, int] = {}
            for (name, _layout), row in self.held.items():
                held[name] = held.get(name, 0) + row.nbytes
            return held

    def _nbytes(self) -> int:
        return sum(row.nbytes for row in self.held.values())

    def _trim(self) -> None:
        for row in self.held.values():
            with contextlib.suppress(OSError):
                row.nbytes = _resident(row.fd)
        while self.held and self._nbytes() > weight_policy.pinned_total(
            self.available(), self._nbytes()
        ):
            key = min(self.held, key=lambda k: self.held[k].used)
            plane.release_tier(self.held.pop(key).fd)

    def trim(self) -> int:
        """Release the least recently used tiers the host no longer has room for (its limit
        moved, or other memory grew); returns the bytes still held."""
        with self.lock:
            self._trim()
            return self._nbytes()

    def close(self) -> None:
        with self.lock:
            for row in self.held.values():
                plane.release_tier(row.fd)
            self.held.clear()


def _resident(fd: int) -> int:
    """The memfd's allocated bytes: evicted regions are holes and cost nothing."""
    return os.fstat(fd).st_blocks * 512
