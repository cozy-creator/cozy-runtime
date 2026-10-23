"""The worker <-> device-executor seam: length-framed JSON, byte-accounted.

One byte stream carries length-framed canonical-JSON commands and replies: a fixed 4-byte
big-endian length prefix, then exactly that many body bytes (#448). A byte stream plus
explicit framing is the ONE protocol on every OS: a `SOCK_STREAM` unix socket on Linux and
macOS, a byte-mode named pipe on Windows (`winpipe`). Neither of the last two has
SOCK_SEQPACKET (which silently truncates an over-buffer message, cr-007's wedge).

Three properties are structural rather than reviewed:

* **No size rule.** A frame is as large as its structure; `MAX_FRAME` only refuses a corrupt
  or hostile length prefix, before any body byte is read. Every wire byte in both directions
  is counted. Output bytes never cross this seam — they land in the attempt spool the worker
  brokered, which keeps the executor's writable storage to exactly that spool (§3.1).
* **Machine credentials stay outside it.** `send` refuses keys from `SECRET_KEYS`; Load,
  Invoke and `bearer` may carry a signer's supplied bearer at `tensorfs.token`.
* **A torn frame is typed.** EOF or a peer reset exactly at a frame boundary is the
  disposable executor's normal end (`None`); EOF inside a header or body is
  `seam_eof_mid_frame`, never a silent short read.

The frames are the ONE vocabulary; there is no second RPC between these two processes.
"""

from __future__ import annotations

import array
import contextlib
import ctypes
import json
import os
import socket
import stat
import struct
import sys
import tempfile
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, NoReturn

from cozy_runtime.internal.canonical import DOC_MAX_BYTES

if sys.platform == "win32":
    import _winapi
    import msvcrt

    from cozy_runtime.internal import winpipe
else:
    import fcntl

#: Worker/executor command-and-reply compatibility, independent of SDK package
#: version or source build. Compatible additive fields keep this revision; only
#: an incompatible change to required commands/semantics raises it.
EXECUTOR_PROTOCOL_REVISION = 1

#: Not a size rule: the most one length prefix may make a reader allocate (the documents'
#: guard, shared with TensorD's `protocol::MAX_FRAME`). It refuses only a corrupt or hostile
#: prefix.
MAX_FRAME = DOC_MAX_BYTES

#: 4-byte big-endian unsigned frame-length prefix. Fixed width so the reader always knows
#: exactly how many header bytes to collect before it commits to anything.
HEADER = struct.Struct("!I")

#: Secret field names outside the explicitly supplied TensorFS command options.
SECRET_KEYS = frozenset({"token", "credential", "authorization", "secret", "jwt"})

#: The attempt-spool file the executor writes its serialized typed result into: the bytes
#: take the road the output blobs take, and only their digest and length cross here. The
#: WORKER names the file, so no path ever comes off the child (cr-007).
RESULT_DOCUMENT = "result.canonical"

#: One decoded frame: a JSON object, routed by its `event`, `cmd`, `seq` and `request_id`.
type Frame = dict[str, object]

#: Peer-gone errnos: ECONNRESET, EPIPE, ENOTCONN.
_GONE_ERRNOS = (104, 32, 107)


class SeamError(Exception):
    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


#: The commands that may carry a signer's supplied bearer, at `tensorfs.token` only.
SUPPLYING = frozenset({"load", "invoke", "bearer"})


def _scan(value: Any, path: str = "", *, supplied: bool = False) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            at = f"{path}{key}"
            allowed = supplied and at == "tensorfs.token" and isinstance(item, str)
            if key.lower() in SECRET_KEYS and not allowed:
                raise SeamError(
                    "credential_at_seam",
                    f"{path}{key!r}: credentials never cross the worker/executor seam",
                )
            _scan(item, f"{at}.", supplied=supplied)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _scan(item, path, supplied=supplied)


@dataclass(slots=True)
class Channel:
    """One end of the seam, with its own byte accounting.

    `recv(timeout)` is a per-read stall clock: the clock resets every time bytes arrive, so
    a slow frame that keeps moving never times out and a stalled peer does (#434). Lifecycle
    handshakes can additionally provide `total_timeout`; that one bounds the entire frame so
    a peer cannot retain executor ownership forever by dripping bytes.
    """

    sock: socket.socket
    sent_bytes: int = 0
    recv_bytes: int = 0
    frames_out: int = 0
    frames_in: int = 0
    refusals: list[str] = field(default_factory=list)
    #: One frame at a time, and a frame with the descriptors that follow it (held across both
    #: by `Mux.ask`): attempts, follower readers and overlapped calls all send.
    sending: threading.RLock = field(default_factory=threading.RLock, repr=False, compare=False)

    def send(self, message: dict[str, Any]) -> None:
        _scan(message, supplied=message.get("cmd") in SUPPLYING)
        data = json.dumps(message, separators=(",", ":"), sort_keys=True).encode()
        if len(data) > MAX_FRAME:
            self.refusals.append("frame_too_large")
            raise SeamError("frame_too_large", f"{len(data)} B over the {MAX_FRAME} B guard")
        with self.sending:
            self.sock.sendall(HEADER.pack(len(data)) + data)
            self.sent_bytes += HEADER.size + len(data)
            self.frames_out += 1

    def send_descriptor(self, descriptor: socket.socket) -> None:
        """Transfer one already-connected local capability after its control answer."""
        if descriptor.family != socket.AF_UNIX or descriptor.type != socket.SOCK_STREAM:
            raise SeamError("seam_descriptor", "capability must be a local stream")
        descriptor.getpeername()
        try:
            sent = self.sock.sendmsg(
                [b"\0"],
                [(socket.SOL_SOCKET, socket.SCM_RIGHTS, array.array("i", [descriptor.fileno()]))],
            )
        except OSError as exc:
            self._raise(exc)
        if sent != 1:
            raise SeamError("seam_descriptor", "capability handoff was truncated")
        self.sent_bytes += 1

    def recv_descriptor(self) -> int:
        """Receive exactly one connected socket with close-on-exec descriptor custody."""
        # Reject a regular file, listener, TCP socket or datagram before a storage handle
        # sees it. No address is opened or connected here.
        received = self._recv_fd()
        try:
            descriptor = socket.socket(fileno=received)
        except OSError as exc:
            os.close(received)
            self._raise(exc)
        try:
            if descriptor.family != socket.AF_UNIX or descriptor.type != socket.SOCK_STREAM:
                raise SeamError("seam_descriptor", "capability must be a local stream")
            descriptor.getpeername()
            descriptor.set_inheritable(False)
            return descriptor.detach()
        except OSError as exc:
            self._raise(exc)
        finally:
            descriptor.close()

    def recv_file(self) -> int:
        """Receive exactly one read-only regular file (a verified object file the machine
        hands over); anything else is closed and refused."""
        received = self._recv_fd()
        try:
            info = os.fstat(received)
            if (
                not stat.S_ISREG(info.st_mode)
                or fcntl.fcntl(received, fcntl.F_GETFL) & os.O_ACCMODE
            ):
                raise SeamError(
                    "seam_descriptor", "an object file must be a read-only regular file"
                )
        except BaseException:
            os.close(received)
            raise
        return received

    def recv_fds(self, count: int) -> list[int]:
        """`count` descriptors, unopened."""
        received: list[int] = []
        try:
            for _ in range(count):
                received.append(self._recv_fd())
        except BaseException:
            for fd in received:
                os.close(fd)
            raise
        return received

    def _send_fd(self, fd: int, what: str) -> None:
        try:
            sent = self.sock.sendmsg(
                [b"\0"], [(socket.SOL_SOCKET, socket.SCM_RIGHTS, array.array("i", [fd]))]
            )
        except OSError as exc:
            self._raise(exc)
        if sent != 1:
            raise SeamError("seam_descriptor", f"{what} handoff was truncated")
        self.sent_bytes += 1

    def _recv_fd(self) -> int:
        """One descriptor after a one-byte marker, close-on-exec, and nothing else."""
        received: list[int] = []
        width = array.array("i").itemsize
        try:
            marker, control, flags, _ = self.sock.recvmsg(
                1, socket.CMSG_SPACE(width), getattr(socket, "MSG_CMSG_CLOEXEC", 0)
            )
            valid = marker == b"\0" and not flags & (socket.MSG_CTRUNC | socket.MSG_TRUNC)
            for level, kind, data in control:
                if (level, kind) != (socket.SOL_SOCKET, socket.SCM_RIGHTS):
                    valid = False
                    continue
                descriptors = array.array("i")
                descriptors.frombytes(data[: len(data) - len(data) % width])
                received.extend(descriptors)
                valid = valid and len(data) % width == 0
            if not valid or len(received) != 1:
                raise SeamError("seam_descriptor", "expected exactly one scoped capability")
            if not hasattr(socket, "MSG_CMSG_CLOEXEC"):  # macOS: close-on-exec after the fact
                os.set_inheritable(received[0], False)
            self.recv_bytes += 1
            return received.pop()
        except OSError as exc:
            self._raise(exc)
        finally:
            for descriptor_fd in received:
                os.close(descriptor_fd)

    def recv(
        self, timeout: float | None = None, *, total_timeout: float | None = None
    ) -> dict[str, Any] | None:
        """One frame, or None when the peer is gone (a disposable executor's normal end)."""
        deadline = time.monotonic() + total_timeout if total_timeout is not None else None
        head = self._read_exact(
            HEADER.size, boundary=True, stall_timeout=timeout, deadline=deadline
        )
        if head is None:
            return None
        (length,) = HEADER.unpack(head)
        if not 0 < length <= MAX_FRAME:
            # Refused on the DECLARED length, before any body byte is read or allocated.
            # The stream is desynchronized past this point; the channel is done.
            self.refusals.append("frame_over_cap")
            raise SeamError(
                "frame_over_cap",
                f"declared {length} B against the {MAX_FRAME} B guard — refused before "
                "allocation; the channel is not recoverable past a lying header",
            )
        body = self._read_exact(length, boundary=False, stall_timeout=timeout, deadline=deadline)
        assert body is not None  # boundary=False never yields None
        self.recv_bytes += HEADER.size + length
        self.frames_in += 1
        value = json.loads(body)
        assert isinstance(value, dict)
        return value

    def _read_exact(
        self,
        want: int,
        *,
        boundary: bool,
        stall_timeout: float | None,
        deadline: float | None,
    ) -> bytes | None:
        """Exactly `want` bytes, looping over short reads.

        `boundary=True` marks the frame boundary (the first header byte): EOF or a peer
        reset THERE returns None. Anywhere else the frame is torn and the failure is typed.
        """
        buf = bytearray()
        while len(buf) < want:
            at_boundary = boundary and not buf
            read_timeout = stall_timeout
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise SeamError(
                        "seam_total_timeout",
                        f"frame deadline expired with {len(buf)}/{want} B read",
                    )
                read_timeout = remaining if read_timeout is None else min(read_timeout, remaining)
            self.sock.settimeout(read_timeout)
            try:
                chunk = self.sock.recv(min(want - len(buf), 1 << 20))
            except TimeoutError:
                if deadline is not None and time.monotonic() >= deadline:
                    raise SeamError(
                        "seam_total_timeout",
                        f"frame deadline expired with {len(buf)}/{want} B read",
                    ) from None
                raise SeamError(
                    "seam_timeout",
                    f"no bytes for {read_timeout} s with {len(buf)}/{want} B read",
                ) from None
            except OSError as exc:
                if exc.errno in _GONE_ERRNOS:
                    if at_boundary:
                        return None
                    raise SeamError(
                        "seam_eof_mid_frame",
                        f"peer reset with {len(buf)}/{want} B of a "
                        f"{'header' if boundary else 'body'} read",
                    ) from None
                self._raise(exc)
            if not chunk:
                if at_boundary:
                    return None
                raise SeamError(
                    "seam_eof_mid_frame",
                    f"EOF with {len(buf)}/{want} B of a {'header' if boundary else 'body'} read",
                )
            buf += chunk
        return bytes(buf)

    @staticmethod
    def _raise(exc: OSError) -> NoReturn:
        raise SeamError("seam_io", f"{exc.__class__.__name__}: {exc}") from exc

    def close(self) -> None:
        with contextlib.suppress(OSError):
            self.sock.close()

    def accounting(self) -> dict[str, int]:
        return {
            "frames_out": self.frames_out,
            "frames_in": self.frames_in,
            "sent_bytes": self.sent_bytes,
            "recv_bytes": self.recv_bytes,
        }


if sys.platform == "win32":

    class PipeChannel(Channel):
        """Windows's seam end, on a `winpipe.Pipe`: a handle crosses duplicated into the
        peer. A file arrives as a C runtime descriptor; a stream capability as its raw
        handle, for TensorFS to adopt."""

        def send_descriptor(self, descriptor: socket.socket) -> None:
            raise SeamError("seam_descriptor", "no socket crosses a Windows seam")

        def recv_descriptor(self) -> int:
            handle = self._recv_handle()
            if winpipe.file_type(handle) != winpipe.FILE_TYPE_PIPE:
                _winapi.CloseHandle(handle)
                raise SeamError("seam_descriptor", "capability must be a local stream")
            return handle

        def recv_file(self) -> int:
            received = self._recv_fd()
            try:
                if not stat.S_ISREG(os.fstat(received).st_mode) or winpipe.writable(
                    msvcrt.get_osfhandle(received)
                ):
                    raise SeamError(
                        "seam_descriptor", "an object file must be a read-only regular file"
                    )
            except BaseException:
                os.close(received)
                raise
            return received

        def _send_fd(self, fd: int, what: str) -> None:
            try:
                self.sock.send_handle(msvcrt.get_osfhandle(fd))
            except OSError as exc:
                self._raise(exc)
            self.sent_bytes += 9

        def _recv_fd(self) -> int:
            handle = self._recv_handle()
            try:
                return msvcrt.open_osfhandle(handle, os.O_NOINHERIT)
            except OSError:
                _winapi.CloseHandle(handle)
                raise

        def _recv_handle(self) -> int:
            try:
                handle = self.sock.recv_handle()
            except ValueError as exc:
                raise SeamError(
                    "seam_descriptor", f"expected one scoped capability: {exc}"
                ) from exc
            except OSError as exc:
                self._raise(exc)
            self.recv_bytes += 9
            return handle


#: Cancel marks kept for calls this executor has not served (yet, or ever).
CANCELS_KEPT = 256


def _nothing(*_: object) -> None:
    return None


@dataclass(slots=True)
class _Asked:
    receive: Callable[[Frame], None]
    answer: Frame | None = None
    error: SeamError | None = None


class Mux:
    """One seam shared by every thread of the process. Whoever needs a frame reads the next
    one and routes it: an answer to the request waiting on its `seq`, a `cancel` to its call's
    mark and to `cancelled`, a frame that may carry a bearer (`SUPPLYING`) to `bearer` as it is
    read (a `bearer` frame goes nowhere else), anything else to the commands, in order. No
    reader thread exists, so a process that has served no call stays single-threaded and
    forkable (`zygote`)."""

    def __init__(
        self,
        channel: Channel,
        bearer: Callable[[Frame], None] = _nothing,
        cancelled: Callable[[str], None] = _nothing,
    ) -> None:
        self.channel = channel
        self._bearer, self._cancelled = bearer, cancelled
        self._seq = 0
        self._asked: dict[int, _Asked] = {}
        self._commands: deque[Frame] = deque()
        self._cancels: dict[str, None] = {}
        self._reading = False
        self._ended: SeamError | None = None
        self._changed = threading.Condition()

    def ask(
        self,
        request: Frame,
        send: Callable[[], None] = _nothing,
        receive: Callable[[Frame], None] = _nothing,
    ) -> Frame:
        """Send `request` and the descriptors `send` writes after it, and return its answer.
        `receive` takes the descriptors following the answer, on the thread that read it."""
        asked = _Asked(receive)
        with self.channel.sending:
            with self._changed:
                self._seq += 1
                seq = self._seq
                self._asked[seq] = asked
            try:
                self.channel.send({**request, "seq": seq})
                send()
            except BaseException:
                with self._changed:
                    del self._asked[seq]
                raise
        self._pump(lambda: asked.answer is not None)
        if asked.error is not None:
            raise asked.error
        if asked.answer is None:
            assert self._ended is not None
            raise SeamError(self._ended.code, f"{request.get('kind')}#{seq}: {self._ended.detail}")
        return asked.answer

    def command(self) -> Frame | None:
        """The next command; None once the peer closed the seam."""
        self._pump(lambda: bool(self._commands))
        with self._changed:
            if self._commands:
                return self._commands.popleft()
        assert self._ended is not None
        if self._ended.code == "seam_closed":
            return None
        raise self._ended

    def cancel(self, request_id: str) -> None:
        with self._changed:
            self._cancels[request_id] = None
            while len(self._cancels) > CANCELS_KEPT:
                del self._cancels[next(iter(self._cancels))]
            self._changed.notify_all()

    def cancelled(self, request_id: str, wait: float = 0.0) -> bool:
        """Whether `request_id` is cancelled, waiting up to `wait` seconds for it."""
        with self._changed:
            return self._changed.wait_for(lambda: request_id in self._cancels, wait)

    def forget(self, request_id: str) -> None:
        with self._changed:
            self._cancels.pop(request_id, None)

    def _pump(self, done: Callable[[], bool]) -> None:
        while True:
            with self._changed:
                self._changed.wait_for(
                    lambda: done() or self._ended is not None or not self._reading
                )
                if done() or self._ended is not None:
                    return
                self._reading = True
            ended: SeamError | None = None
            try:
                frame = self.channel.recv()
                if frame is None:
                    ended = SeamError("seam_closed", "the worker closed the seam")
                else:
                    self._route(frame)
            except Exception as exc:  # the stream is past a frame no one can place: it ends
                ended = exc if isinstance(exc, SeamError) else SeamError("seam_io", repr(exc))
            finally:
                with self._changed:
                    self._reading = False
                    self._ended = self._ended or ended
                    self._changed.notify_all()

    def _route(self, frame: Frame) -> None:
        if frame.get("event") == "answer":
            seq = frame.get("seq")
            with self._changed:
                asked = self._asked.pop(seq, None) if isinstance(seq, int) else None
            if asked is None:
                raise SeamError("seam_out_of_order", f"an answer to no request: {seq!r}")
            try:
                asked.receive(frame)  # its descriptors follow it on the stream
            except (
                SeamError
            ) as exc:  # that exchange fails; a receive that lost the stream closed it
                asked.error = exc
            with self._changed:
                asked.answer = frame
        elif frame.get("cmd") == "cancel":
            self.cancel(str(frame.get("request_id", "")))
            self._cancelled(str(frame.get("request_id", "")))
        elif frame.get("cmd") == "bearer":
            self._bearer(frame)
        else:
            if frame.get("cmd") in SUPPLYING:
                self._bearer(frame)
            with self._changed:
                self._commands.append(frame)


def listener(path: str) -> socket.socket:
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.bind(path)
    sock.listen(1)
    return sock


def connect(path: str) -> Channel:
    if sys.platform == "win32":
        return PipeChannel(winpipe.Pipe.connect(path))
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.connect(path)
    return Channel(sock)


def anonymous(name: str, sealing: bool = False) -> int:
    """A new close-on-exec in-memory file no path names: a memfd (`sealing`: one that may be
    sealed); on macOS, which has none, an unlinked file; on Windows, a temporary file gone
    with its last handle."""
    if sys.platform == "win32":
        path = os.path.join(tempfile.gettempdir(), f"{name}.{os.urandom(8).hex()}")
        flags = os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_BINARY | os.O_NOINHERIT
        return os.open(path, flags | os.O_TEMPORARY | os.O_SHORT_LIVED, 0o600)
    if sys.platform != "linux":
        fd, path = tempfile.mkstemp(prefix=f"{name}.")
        os.unlink(path)
        return fd
    # libc's own: python-build-standalone's old glibc build lacks `os.memfd_create`.
    # MFD_CLOEXEC, and MFD_ALLOW_SEALING when asked
    flags = 0x1 | (0x2 if sealing else 0)
    fd: int = ctypes.CDLL(None, use_errno=True).memfd_create(name.encode(), flags)
    if fd < 0:
        raise OSError(ctypes.get_errno(), "memfd_create")
    return fd


#: Linux `fcntl` seals by number (python-build-standalone may lack the names): seal, shrink,
#: grow and write.
_ADD_SEALS, _SEALS = 1033, 0x8 | 0x4 | 0x2 | 0x1


def sealed(name: str, body: bytes) -> int:
    """`body` in a new memfd, sealed: a document that crosses beside its request. macOS has
    no memfd and no seals: there it is an unlinked file, which no path names."""
    fd = anonymous(name, sealing=True)
    try:
        view = memoryview(body)
        while view:
            view = view[os.write(fd, view) :]
        if sys.platform == "linux":
            fcntl.fcntl(fd, _ADD_SEALS, _SEALS)
    except BaseException:
        os.close(fd)
        raise
    return fd
