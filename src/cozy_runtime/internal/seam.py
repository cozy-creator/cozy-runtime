"""The worker <-> device-executor seam: CONTROL, not data, byte-accounted.

One `SOCK_STREAM` unix socket carries length-framed canonical-JSON commands and replies:
a fixed 4-byte big-endian length prefix, then exactly that many body bytes (#448). A byte
stream plus explicit framing is the ONE protocol on every OS — Linux today, Windows
socketpair and macOS AF_UNIX next, neither of which has SOCK_SEQPACKET (and SEQPACKET
silently truncates an over-buffer message, the wedge defect class of cr-007).

Three properties are structural rather than reviewed:

* **Control, not data.** Every frame is capped (`MAX_FRAME`) and the declared length is
  refused BEFORE the body is read or allocated. Every wire byte in both directions is
  counted. Output bytes never cross this seam — they land in the attempt spool the
  worker brokered, and the worker reads them from there, which is also what keeps
  the executor's writable storage to exactly that spool (§3.1).
* **No credentials cross it.** `send` refuses a frame carrying a key from `SECRET_KEYS`, so a
  grant token or worker JWT reaching the executor is a typed refusal at the boundary and not
  a code-review rule. The executor holds no RecordOwner, storage or billing credential.
* **A torn frame is typed.** EOF or a peer reset exactly at a frame boundary is the
  disposable executor's normal end (`None`); EOF inside a header or body is
  `seam_eof_mid_frame`, never a silent short read.

The frames are the ONE vocabulary; there is no second RPC between these two processes.
"""

from __future__ import annotations

import array
import contextlib
import json
import os
import socket
import stat
import struct
import threading
import time
from dataclasses import dataclass, field
from typing import Any, NoReturn

#: Worker/executor command-and-reply compatibility, independent of SDK package
#: version or source build. Compatible additive fields keep this revision; only
#: an incompatible change to required commands/semantics raises it.
EXECUTOR_PROTOCOL_REVISION = 1

#: Frames are control. 64 KiB is generous for a payload document and small enough that a
#: caller trying to move tensors or media through here fails immediately.
MAX_FRAME = 64 * 1024

#: 4-byte big-endian unsigned frame-length prefix. Fixed width so the reader always knows
#: exactly how many header bytes to collect before it commits to anything.
HEADER = struct.Struct("!I")

#: Key names that must never appear in a frame, at any depth. Credentials stop at the
#: worker (§3.1); the executor cannot leak what it was never handed.
SECRET_KEYS = frozenset({"token", "credential", "authorization", "secret", "jwt"})

#: The attempt-spool file the executor writes its serialized typed result into. The public
#: contract admits an inline result up to 4 MiB and this frame admits 64 KiB, so the bytes
#: take the same road the output blobs already take — the brokered spool — and only their
#: digest and length cross here. The WORKER names the file, so no path ever comes off
#: the child (cr-007).
RESULT_DOCUMENT = "result.canonical"

#: Peer-gone errnos: ECONNRESET, EPIPE, ENOTCONN.
_GONE_ERRNOS = (104, 32, 107)


class SeamError(Exception):
    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def _scan(value: Any, path: str = "") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if key.lower() in SECRET_KEYS:
                raise SeamError(
                    "credential_at_seam",
                    f"{path}{key!r}: credentials never cross the worker/executor seam",
                )
            _scan(item, f"{path}{key}.")
    elif isinstance(value, (list, tuple)):
        for item in value:
            _scan(item, path)


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
    #: One frame at a time: follower progress is forwarded from reader threads and overlapped
    #: calls log from theirs, beside the thread that owns the attempt.
    sending: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def send(self, message: dict[str, Any]) -> None:
        _scan(message)
        data = json.dumps(message, separators=(",", ":"), sort_keys=True).encode()
        if len(data) > MAX_FRAME:
            self.refusals.append("frame_too_large")
            raise SeamError(
                "frame_too_large",
                f"{len(data)} B over the {MAX_FRAME} B control cap — the seam carries "
                "control, not data (bytes go through the brokered attempt spool)",
            )
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

    def send_memfd(self, fd: int) -> None:
        """Transfer one weight set's pinned host tier (`tensorfs.plane` memfd)."""
        _require_memfd(fd)
        try:
            sent = self.sock.sendmsg(
                [b"\0"], [(socket.SOL_SOCKET, socket.SCM_RIGHTS, array.array("i", [fd]))]
            )
        except OSError as exc:
            self._raise(exc)
        if sent != 1:
            raise SeamError("seam_descriptor", "memfd handoff was truncated")
        self.sent_bytes += 1

    def recv_memfd(self) -> int:
        """Receive exactly one memfd; anything else is closed and refused."""
        received = self._recv_fd()
        try:
            _require_memfd(received)
        except SeamError:
            os.close(received)
            raise
        return received

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
                f"declared {length} B against the {MAX_FRAME} B control cap — refused "
                "before allocation; the channel is not recoverable past a lying header",
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
                chunk = self.sock.recv(want - len(buf))
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


def listener(path: str) -> socket.socket:
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.bind(path)
    sock.listen(1)
    return sock


def connect(path: str) -> Channel:
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.connect(path)
    return Channel(sock)


def _require_memfd(fd: int) -> None:
    """A regular file the kernel names `/memfd:…`: an anonymous in-memory tier, never a path."""
    try:
        regular = stat.S_ISREG(os.fstat(fd).st_mode)
        named = os.readlink(f"/proc/self/fd/{fd}")
    except OSError as exc:
        raise SeamError("seam_descriptor", f"memfd unreadable: {exc}") from exc
    if not regular or not named.startswith("/memfd:"):
        raise SeamError("seam_descriptor", "capability must be a memfd")
