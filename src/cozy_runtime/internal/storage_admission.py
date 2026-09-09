"""Borrow the supervisor's disk admission for one bounded Runtime write.

Only the pod parent receives the inherited channel. Local execution has no host
controller. Each operation passes a private socket; closing it releases the
supervisor's existing exclusion, including after process death. No disk policy,
quota, credentials, public listener or persistent ownership lives here.
"""

from __future__ import annotations

import array
import json
import os
import socket
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from cozy_runtime.internal import fill


class StorageRefusal(Exception):
    code = "insufficient_storage"

    def __init__(self, detail: str, facts: dict[str, int] | None = None) -> None:
        super().__init__(detail)
        self.detail, self.facts = detail, facts or {}


@dataclass(frozen=True)
class Write:
    path: Path
    bytes: int
    inodes: int


_channel: socket.socket | None = None


def native_write(path: Path | None, payload: int = 0, objects: int = 1) -> Write:
    """Conservative CAS payload, atomic metadata and directory allocation bound.

    Metadata uses the native document ceiling; object fanout charges an inode
    per MiB (well above native's 64 MiB pack grid), plus per-file fanout. These
    are transient write estimates, never TensorFS reachability or quota policy.
    """
    if _channel is None:
        return Write(path or Path("/"), 0, 0)
    if path is None:
        raise StorageRefusal("native writer has no configured Store root")
    metadata = int(fill.tensorfs_module().manifest_max_bytes())
    blocks = (payload + (1 << 20) - 1) >> 20
    return Write(path, payload + 4 * metadata, 3 * (objects + blocks) + 16)


def inherit() -> None:
    """Consume the pod's fixed descriptor before any executor is launched."""
    global _channel
    value = os.environ.pop("COZY_STORAGE_ADMISSION_FD", None)
    if value is None:
        return
    if value != "4" or _channel is not None:
        raise StorageRefusal("invalid supervisor storage channel")
    try:
        _channel = socket.socket(fileno=4)
        _channel.set_inheritable(False)
        if _channel.family != socket.AF_UNIX or _channel.type != socket.SOCK_SEQPACKET:
            raise ValueError("wrong socket kind")
    except (OSError, ValueError) as exc:
        raise StorageRefusal("supervisor storage channel is unavailable") from exc


@contextmanager
def admit(*writes: Write) -> Iterator[None]:
    """Hold admission through the actual write/child exit, never just its launch."""
    if _channel is None:
        yield
        return
    if not 1 <= len(writes) <= 8 or any(
        not row.path.is_absolute()
        or type(row.bytes) is not int
        or type(row.inodes) is not int
        or not 0 <= row.bytes < 1 << 63
        or not 0 <= row.inodes < 1 << 63
        for row in writes
    ):
        raise StorageRefusal("Runtime write requires bounded absolute destinations")
    body = json.dumps(
        [{"path": str(row.path), "bytes": row.bytes, "inodes": row.inodes} for row in writes],
        separators=(",", ":"),
    ).encode()
    if len(body) > 4096:
        raise StorageRefusal("Runtime write admission exceeds its control bound")
    client, host = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    with client, host:
        try:
            sent = _channel.sendmsg(
                [body],
                [(socket.SOL_SOCKET, socket.SCM_RIGHTS, array.array("i", [host.fileno()]))],
            )
            if sent != len(body):
                raise OSError("short admission command")
            host.close()
            with client.makefile("rb") as response:
                raw = response.readline(4097)
            if len(raw) > 4096 or not raw.endswith(b"\n"):
                raise ValueError("invalid admission response")
            answer = json.loads(raw)
            if not isinstance(answer, dict) or answer.get("ok") is not True:
                facts = answer.get("facts", {}) if isinstance(answer, dict) else {}
                raise StorageRefusal("supervisor refused Runtime disk write", facts)
        except (OSError, ValueError) as exc:
            raise StorageRefusal("supervisor storage admission is unavailable") from exc
        yield
