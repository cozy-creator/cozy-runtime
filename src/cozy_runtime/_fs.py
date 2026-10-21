"""Open flags and directory durability by what the OS offers, so one code path serves every OS.
`os.open` is already non-inheritable (PEP 446): no flag here asks for that."""

from __future__ import annotations

import os
from pathlib import Path

#: Bytes as written: Windows opens a descriptor in text mode without it.
BINARY = getattr(os, "O_BINARY", 0)
#: A final symlink refused, where the OS can refuse one.
NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
#: Never waits on a FIFO's writer, where the OS has FIFOs.
NONBLOCK = getattr(os, "O_NONBLOCK", 0)
_DIRECTORY = getattr(os, "O_DIRECTORY", None)


def sync_directory(path: Path) -> None:
    """A rename or link into `path` made durable. Windows cannot open a directory to sync it."""
    if _DIRECTORY is None:
        return
    fd = os.open(path, os.O_RDONLY | _DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
