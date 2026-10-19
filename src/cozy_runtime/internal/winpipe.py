"""The seam on Windows: a byte-mode named pipe, overlapped so one thread reads while others
write, with the socket calls the seam makes. The machine's listener (TensorD's
`platform::ipc`) names its pipe from the path it hands the executor: `\\\\.\\pipe\\cozy-`
and the first 16 bytes of the path text's SHA-256, in hex. A handle crosses by being
duplicated into the peer's process: a zero marker byte, then its value there, 8 bytes
little-endian."""

from __future__ import annotations

import _winapi
import contextlib
import ctypes
import errno
import hashlib
import sys
from ctypes import wintypes

# Windows's alone: the type checker skips the rest elsewhere.
assert sys.platform == "win32"

_kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
_kernel32.GetNamedPipeInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)] + [
    ctypes.c_void_p
] * 3
_kernel32.GetNamedPipeServerProcessId.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.ULONG)]
_kernel32.GetNamedPipeClientProcessId.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.ULONG)]
_kernel32.GetFileType.argtypes = [wintypes.HANDLE]
_kernel32.GetFileType.restype = wintypes.DWORD
_kernel32.CancelIoEx.argtypes = [wintypes.HANDLE, ctypes.c_void_p]
_ntdll = ctypes.WinDLL("ntdll")
_ntdll.NtQueryObject.argtypes = [
    wintypes.HANDLE,
    ctypes.c_int,
    ctypes.c_void_p,
    wintypes.ULONG,
    ctypes.c_void_p,
]
_ntdll.NtQueryObject.restype = ctypes.c_long

FILE_TYPE_DISK, FILE_TYPE_PIPE = 0x1, 0x3
_PIPE_SERVER_END = 0x1
_FILE_WRITE_DATA, _FILE_APPEND_DATA = 0x2, 0x4
_ERROR_BROKEN_PIPE, _ERROR_PIPE_BUSY, _ERROR_OPERATION_ABORTED = 109, 231, 995
_WAIT_TIMEOUT = 0x102


class _BasicInformation(ctypes.Structure):
    _fields_ = [
        ("Attributes", wintypes.ULONG),
        ("GrantedAccess", wintypes.ULONG),
        ("HandleCount", wintypes.ULONG),
        ("PointerCount", wintypes.ULONG),
        ("Reserved", wintypes.ULONG * 10),
    ]


def pipe_name(path: str) -> str:
    return "\\\\.\\pipe\\cozy-" + hashlib.sha256(path.encode()).digest()[:16].hex()


def file_type(handle: int) -> int:
    return int(_kernel32.GetFileType(handle))


def writable(handle: int) -> bool:
    """Whether `handle` was granted writing or appending."""
    info = _BasicInformation()
    status = _ntdll.NtQueryObject(handle, 0, ctypes.byref(info), ctypes.sizeof(info), None)
    if status < 0:
        raise OSError(errno.EINVAL, f"NtQueryObject: NTSTATUS {status & 0xFFFFFFFF:#x}")
    return bool(info.GrantedAccess & (_FILE_WRITE_DATA | _FILE_APPEND_DATA))


class Pipe:
    """One connected end, its handle the caller's to close."""

    def __init__(self, handle: int) -> None:
        self.handle = handle
        self.timeout: float | None = None
        self.closed = False

    @classmethod
    def connect(cls, path: str) -> Pipe:
        """Dial the listener at `path`, waiting while every instance is busy."""
        name = pipe_name(path)
        while True:
            try:
                return cls(
                    _winapi.CreateFile(
                        name,
                        _winapi.GENERIC_READ | _winapi.GENERIC_WRITE,
                        0,
                        _winapi.NULL,
                        _winapi.OPEN_EXISTING,
                        _winapi.FILE_FLAG_OVERLAPPED,
                        _winapi.NULL,
                    )
                )
            except OSError as exc:
                if exc.winerror != _ERROR_PIPE_BUSY:
                    raise
            with contextlib.suppress(OSError):  # gone or free again: the next dial says which
                _winapi.WaitNamedPipe(name, _winapi.INFINITE)

    def settimeout(self, timeout: float | None) -> None:
        self.timeout = timeout

    def fileno(self) -> int:
        return self.handle

    def recv(self, size: int) -> bytes:
        """Up to `size` bytes; b"" once the peer is gone. Raises TimeoutError after
        `timeout` seconds without any."""
        try:
            overlapped, error = _winapi.ReadFile(self.handle, size, overlapped=True)
        except BrokenPipeError:
            return b""
        timed_out = False
        try:
            if error == _winapi.ERROR_IO_PENDING:
                wait = (
                    _winapi.INFINITE if self.timeout is None else max(0, round(self.timeout * 1000))
                )
                if _winapi.WaitForMultipleObjects([overlapped.event], False, wait) == _WAIT_TIMEOUT:
                    timed_out = True
                    overlapped.cancel()
        except BaseException:
            overlapped.cancel()
            overlapped.GetOverlappedResult(True)
            raise
        _, error = overlapped.GetOverlappedResult(True)
        if error == _ERROR_BROKEN_PIPE:
            return b""
        if error == _ERROR_OPERATION_ABORTED:
            if self.closed:
                raise OSError(errno.EBADF, "the pipe is closed")
            if timed_out:
                raise TimeoutError("timed out")
        return bytes(overlapped.getbuffer())

    def sendall(self, data: bytes | bytearray | memoryview) -> None:
        view = memoryview(data)
        while view:
            overlapped, error = _winapi.WriteFile(self.handle, view, overlapped=True)
            try:
                if error == _winapi.ERROR_IO_PENDING:
                    _winapi.WaitForMultipleObjects([overlapped.event], False, _winapi.INFINITE)
            except BaseException:
                overlapped.cancel()
                overlapped.GetOverlappedResult(True)
                raise
            written, error = overlapped.GetOverlappedResult(True)
            if error == _ERROR_OPERATION_ABORTED:
                raise BrokenPipeError(errno.EPIPE, "the pipe closed during a write")
            view = view[written:]

    def peer_pid(self) -> int:
        flags = wintypes.DWORD()
        if not _kernel32.GetNamedPipeInfo(self.handle, ctypes.byref(flags), None, None, None):
            raise ctypes.WinError(ctypes.get_last_error())
        server = flags.value & _PIPE_SERVER_END
        ask = (
            _kernel32.GetNamedPipeClientProcessId
            if server
            else _kernel32.GetNamedPipeServerProcessId
        )
        pid = wintypes.ULONG()
        if not ask(self.handle, ctypes.byref(pid)):
            raise ctypes.WinError(ctypes.get_last_error())
        return int(pid.value)

    def send_handle(self, handle: int) -> None:
        """Duplicate `handle` into the peer and name it there; the caller keeps its own."""
        peer = _winapi.OpenProcess(_winapi.PROCESS_DUP_HANDLE, False, self.peer_pid())
        try:
            value = _winapi.DuplicateHandle(
                _winapi.GetCurrentProcess(), handle, peer, 0, False, _winapi.DUPLICATE_SAME_ACCESS
            )
        finally:
            _winapi.CloseHandle(peer)
        self.sendall(b"\0" + value.to_bytes(8, "little"))

    def recv_handle(self) -> int:
        """One handle the peer duplicated into this process, the caller's to close."""
        message = bytearray()
        while len(message) < 9:
            chunk = self.recv(9 - len(message))
            if not chunk:
                raise ConnectionResetError(errno.ECONNRESET, "the peer closed before its handle")
            message += chunk
        if message[0] != 0:
            raise ValueError("expected a handle marker")
        return int.from_bytes(message[1:], "little")

    def close(self) -> None:
        if not self.closed:
            self.closed = True
            _kernel32.CancelIoEx(self.handle, None)
            _winapi.CloseHandle(self.handle)
