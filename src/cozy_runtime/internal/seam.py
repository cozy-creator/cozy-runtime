"""The supervisor <-> device-executor seam: CONTROL, not data, byte-accounted.

One `SOCK_SEQPACKET` unix socket carries length-framed canonical-JSON commands and replies.
Two properties are structural rather than reviewed:

* **Control, not data.** Every frame is capped (`MAX_FRAME`) and every byte in both
  directions is counted. Output bytes never cross this seam — they land in the attempt spool
  the supervisor brokered, and the supervisor reads them from there, which is also what keeps
  the executor's writable storage to exactly that spool (§3.1).
* **No credentials cross it.** `send` refuses a frame carrying a key from `SECRET_KEYS`, so a
  grant token or worker JWT reaching the executor is a typed refusal at the boundary and not
  a code-review rule. The executor holds no coordinator, storage or billing credential.

The frames are the ONE vocabulary; there is no second RPC between these two processes.
"""

from __future__ import annotations

import contextlib
import json
import socket
from dataclasses import dataclass, field
from typing import Any

#: Frames are control. 64 KiB is generous for a payload document and small enough that a
#: caller trying to move tensors or media through here fails immediately.
MAX_FRAME = 64 * 1024

#: Key names that must never appear in a frame, at any depth. Credentials stop at the
#: supervisor (§3.1); the executor cannot leak what it was never handed.
SECRET_KEYS = frozenset({"token", "credential", "authorization", "secret", "jwt"})


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
                    f"{path}{key!r}: credentials never cross the supervisor/executor seam",
                )
            _scan(item, f"{path}{key}.")
    elif isinstance(value, (list, tuple)):
        for item in value:
            _scan(item, path)


@dataclass(slots=True)
class Channel:
    """One end of the seam, with its own byte accounting."""

    sock: socket.socket
    sent_bytes: int = 0
    recv_bytes: int = 0
    frames_out: int = 0
    frames_in: int = 0
    refusals: list[str] = field(default_factory=list)

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
        self.sock.sendall(data)
        self.sent_bytes += len(data)
        self.frames_out += 1

    def recv(self, timeout: float | None = None) -> dict[str, Any] | None:
        """One frame, or None when the peer is gone (a disposable executor's normal end)."""
        self.sock.settimeout(timeout)
        try:
            data = self.sock.recv(MAX_FRAME)
        except TimeoutError:
            raise SeamError("seam_timeout", f"no frame within {timeout} s") from None
        except OSError as exc:
            return None if exc.errno in (104, 32, 107) else self._raise(exc)
        if not data:
            return None
        self.recv_bytes += len(data)
        self.frames_in += 1
        value = json.loads(data)
        assert isinstance(value, dict)
        return value

    @staticmethod
    def _raise(exc: OSError) -> None:
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
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    sock.bind(path)
    sock.listen(1)
    return sock


def connect(path: str) -> Channel:
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    sock.connect(path)
    return Channel(sock)
