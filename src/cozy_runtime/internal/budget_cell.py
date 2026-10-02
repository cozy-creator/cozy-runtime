"""A running executor's plane budget, which the Worker lowers mid-call (weight-plane.md).

The executor creates a 32-byte memfd and hands it to the Worker once, at its first load
(`BudgetCell`). The Worker writes a wanted budget and then bumps `asked`; the executor reads
the cell at every block boundary, applies a new ask there (unmapping, then placing its open
stages again) and writes what it applied, then `done = asked`. Only the Worker writes the
first two words and only the executor the last two, each in that order, so a reader that
sees the counter has the value it guards.
"""

from __future__ import annotations

import ctypes
import mmap
import os
import struct

#: asked, wanted, done, applied
_LAYOUT = struct.Struct("<qqqq")
_MFD_CLOEXEC = 1


class Cell:
    def __init__(self, fd: int) -> None:
        self.map = mmap.mmap(fd, _LAYOUT.size)
        #: the executor's last applied ask
        self.seen = 0

    @classmethod
    def create(cls) -> tuple[Cell, int]:
        """A fresh cell and its memfd (the caller hands the fd on, then closes it)."""
        # libc's own: python-build-standalone's old glibc build lacks `os.memfd_create`
        fd = ctypes.CDLL(None, use_errno=True).memfd_create(b"cozy-budget-cell", _MFD_CLOEXEC)
        if fd < 0:
            raise OSError(ctypes.get_errno(), "memfd_create")
        os.ftruncate(fd, _LAYOUT.size)
        return cls(fd), fd

    def _read(self) -> tuple[int, int, int, int]:
        asked, wanted, done, applied = _LAYOUT.unpack_from(self.map)
        return asked, wanted, done, applied

    # -- the Worker

    def ask(self, vram_bytes: int) -> int:
        """Ask for `vram_bytes`; returns the ticket `answered` takes."""
        asked = self._read()[0] + 1
        struct.pack_into("<q", self.map, 8, vram_bytes)
        struct.pack_into("<q", self.map, 0, asked)
        return asked

    def answered(self, ticket: int) -> int | None:
        """The bytes the executor applied for `ticket` or a later ask, once it has."""
        _, _, done, applied = self._read()
        return applied if done >= ticket else None

    # -- the executor

    def poll(self) -> int | None:
        """A budget asked since the last `acknowledge`, else None."""
        asked, wanted, _, _ = self._read()
        if asked == self.seen:
            return None
        self.seen = asked
        return wanted

    def acknowledge(self, applied: int) -> None:
        struct.pack_into("<q", self.map, 24, applied)
        struct.pack_into("<q", self.map, 16, self.seen)

    def close(self) -> None:
        self.map.close()
