"""Advisory whole-file locks that end with their holder: `flock(2)` on POSIX, `LockFileEx` on
Windows, where one byte far past any data stands for the file so the lock never blocks a
reader or writer of its contents. A lock that is not free raises `BlockingIOError` when the
caller will not wait, as `fcntl.flock` does."""

from __future__ import annotations

import sys
from typing import Protocol


class Fileno(Protocol):
    def fileno(self) -> int: ...


def _fd(file: int | Fileno) -> int:
    return file if isinstance(file, int) else file.fileno()


if sys.platform == "win32":
    import ctypes
    import msvcrt
    from ctypes import wintypes

    class _Overlapped(ctypes.Structure):
        _fields_ = [
            ("Internal", ctypes.c_size_t),
            ("InternalHigh", ctypes.c_size_t),
            ("Offset", wintypes.DWORD),
            ("OffsetHigh", wintypes.DWORD),
            ("hEvent", wintypes.HANDLE),
        ]

    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _kernel32.LockFileEx.argtypes = [
        wintypes.HANDLE,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.POINTER(_Overlapped),
    ]
    _kernel32.LockFileEx.restype = wintypes.BOOL
    _kernel32.UnlockFileEx.argtypes = [
        wintypes.HANDLE,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.POINTER(_Overlapped),
    ]
    _kernel32.UnlockFileEx.restype = wintypes.BOOL
    _EXCLUSIVE, _NO_WAIT, _LOCK_VIOLATION = 0x2, 0x1, 33

    def _far() -> _Overlapped:
        # The byte at 2**63 - 2, which no file reaches.
        return _Overlapped(0, 0, 0xFFFFFFFE, 0x7FFFFFFF, None)

    def lock(file: int | Fileno, *, shared: bool = False, wait: bool = True) -> None:
        """Take the lock, waiting for its holder unless `wait` is False."""
        flags = (0 if shared else _EXCLUSIVE) | (0 if wait else _NO_WAIT)
        handle = msvcrt.get_osfhandle(_fd(file))
        if not _kernel32.LockFileEx(handle, flags, 0, 1, 0, ctypes.byref(_far())):
            error = ctypes.get_last_error()
            if error == _LOCK_VIOLATION:
                raise BlockingIOError("the lock is held")
            raise ctypes.WinError(error)

    def unlock(file: int | Fileno) -> None:
        handle = msvcrt.get_osfhandle(_fd(file))
        if not _kernel32.UnlockFileEx(handle, 0, 1, 0, ctypes.byref(_far())):
            raise ctypes.WinError(ctypes.get_last_error())

else:
    import fcntl

    def lock(file: int | Fileno, *, shared: bool = False, wait: bool = True) -> None:
        """Take the lock, waiting for its holder unless `wait` is False."""
        mode = fcntl.LOCK_SH if shared else fcntl.LOCK_EX
        fcntl.flock(_fd(file), mode | (0 if wait else fcntl.LOCK_NB))

    def unlock(file: int | Fileno) -> None:
        fcntl.flock(_fd(file), fcntl.LOCK_UN)
