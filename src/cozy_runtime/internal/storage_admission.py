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
import threading
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path

from cozy_runtime.internal import fill


class StorageRefusal(Exception):
    def __init__(
        self,
        detail: str,
        facts: dict[str, int] | None = None,
        *,
        code: str = "insufficient_storage",
    ) -> None:
        super().__init__(detail)
        self.code = code
        self.facts = facts or {}
        counts = ", ".join(
            f"{key}={value}" for key, value in sorted(self.facts.items()) if type(value) is int
        )
        self.detail = detail + (": " + counts if counts else "")
        self.args = (self.detail,)


@dataclass(frozen=True)
class Write:
    path: Path
    bytes: int
    inodes: int


_channel: socket.socket | None = None
_held: ContextVar[bool] = ContextVar("storage_admission_held", default=False)


def enabled() -> bool:
    return _channel is not None


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


def inherit(descriptor: int | None) -> None:
    """Consume the pod's fixed descriptor before any executor is launched."""
    global _channel
    if descriptor is None:
        return
    if type(descriptor) is not int or descriptor != 4 or _channel is not None:
        raise StorageRefusal("invalid supervisor storage channel")
    try:
        _channel = socket.socket(fileno=4)
        _channel.set_inheritable(False)
        if _channel.family != socket.AF_UNIX or _channel.type != socket.SOCK_SEQPACKET:
            raise ValueError("wrong socket kind")
    except (OSError, ValueError) as exc:
        raise StorageRefusal("supervisor storage channel is unavailable") from exc


class Lease:
    """One explicit operation hold; close may run in a different RPC thread."""

    def __init__(self, context: AbstractContextManager[dict[str, object]] | None = None) -> None:
        self._context = context
        self._lock = threading.RLock()
        self._closed = False

    @contextmanager
    def scope(self) -> Iterator[None]:
        with self._lock:
            if self._closed:
                raise StorageRefusal("storage admission was already released")
            token = _held.set(True)
            try:
                yield
            finally:
                _held.reset(token)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            if self._context is not None:
                self._context.__exit__(None, None, None)
                self._context = None


def acquire(*writes: Write) -> Lease:
    if _channel is None:
        return Lease()
    if _held.get():
        raise StorageRefusal("nested storage admission would deadlock; combine write destinations")
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
    context = _exchange(body, tuple(row.path for row in writes))
    context.__enter__()
    return Lease(context)


@contextmanager
def admit(*writes: Write) -> Iterator[None]:
    """Hold admission through the actual write/child exit, never just its launch."""
    if _channel is None:
        yield
        return
    lease = acquire(*writes)
    try:
        with lease.scope():
            yield
    finally:
        lease.close()


@contextmanager
def _exchange(
    body: bytes,
    directories: tuple[Path, ...] = (),
    *,
    failure: tuple[str, str] = (
        "storage_admission_unavailable",
        "supervisor could not assess Runtime disk write",
    ),
) -> Iterator[dict[str, object]]:
    if _channel is None or _held.get():
        raise StorageRefusal("supervisor storage channel is unavailable or already held")
    if len(body) > 4096:
        raise StorageRefusal("Runtime write admission exceeds its control bound")
    client, host = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    with client, host:
        try:
            descriptors: list[int] = []
            try:
                for directory in directories:
                    while True:
                        try:
                            descriptors.append(
                                os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
                            )
                            break
                        except FileNotFoundError:
                            if directory.parent == directory:
                                raise
                            directory = directory.parent
                sent = _channel.sendmsg(
                    [body],
                    [
                        (
                            socket.SOL_SOCKET,
                            socket.SCM_RIGHTS,
                            array.array("i", [host.fileno(), *descriptors]),
                        )
                    ],
                )
            finally:
                for descriptor in descriptors:
                    os.close(descriptor)
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
                if isinstance(facts, dict) and facts:
                    raise StorageRefusal(
                        "supervisor refused Runtime disk write",
                        facts,
                        code="insufficient_storage",
                    )
                raise StorageRefusal(failure[1], code=failure[0])
        except (OSError, ValueError) as exc:
            raise StorageRefusal(
                "supervisor storage admission is unavailable", code="storage_admission_unavailable"
            ) from exc
        yield answer


def fetch_wheel(operation: str, filename: str, digest: str, url: str) -> tuple[Path, int]:
    """The existing supervisor downloader owns redirects, admission and hashing."""
    body = json.dumps(
        {"operation": operation, "filename": filename, "digest": digest, "url": url},
        separators=(",", ":"),
    ).encode()
    with _exchange(
        body,
        failure=("wheel_download_failed", "supervisor could not download the locked wheel"),
    ) as answer:
        path, length = answer.get("path"), answer.get("length")
        if not isinstance(path, str) or type(length) is not int or not 0 < length <= 2 << 30:
            raise StorageRefusal("supervisor returned an invalid wheel file")
        return Path(path), length


def release_wheels(operation: str) -> None:
    """A completed immutable install releases its existing wheel staging operation."""
    with _exchange(
        json.dumps({"operation": operation, "release": True}).encode(),
        failure=("wheel_release_failed", "supervisor could not release locked wheel staging"),
    ):
        pass
