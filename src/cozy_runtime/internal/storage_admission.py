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
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from weakref import WeakValueDictionary


class StorageRefusal(Exception):
    """A Runtime write that storage admission did not grant.

    `code` is the owner's stable reason, `facts` its measured budget, if any. A refusal
    that carries a budget and names no other code is a capacity refusal.
    """

    def __init__(
        self,
        detail: str,
        facts: dict[str, int] | None = None,
        *,
        code: str | None = None,
    ) -> None:
        super().__init__(detail)
        self.facts = facts or {}
        self.code = code or ("insufficient_storage" if self.facts else "storage_admission_refused")
        counts = ", ".join(
            f"{key}={value}" for key, value in sorted(self.facts.items()) if type(value) is int
        )
        self.detail = detail + (" [" + counts + "]" if counts else "")
        self.args = (self.detail,)


@dataclass(frozen=True)
class Write:
    path: Path
    bytes: int
    inodes: int


_channel: socket.socket | None = None
_held: ContextVar[bool] = ContextVar("storage_admission_held", default=False)
_reclaimers: WeakValueDictionary[int, Registration] = WeakValueDictionary()
_reclaiming: ContextVar[bool] = ContextVar("storage_reclaiming", default=False)


def enabled() -> bool:
    return _channel is not None


def pressure_enabled() -> bool:
    return enabled() or bool(_reclaimers)


@dataclass(eq=False)
class Registration:
    """A worker lifetime reference, not another durable storage owner."""

    root: Path
    reclaim: Callable[[int], object]

    def close(self) -> None:
        _reclaimers.pop(id(self), None)


def register_workspace(root: Path, reclaim: Callable[[int], object]) -> Registration:
    """Only a worker's already-open native workspace supplies deletion authority."""
    if not root.is_absolute() or not (root / ".cozy-workspace").is_dir():
        raise StorageRefusal("pressure recovery requires an existing Runtime workspace")
    registration = Registration(root, reclaim)
    _reclaimers[id(registration)] = registration
    return registration


def native_write(path: Path | None, payload: int = 0, objects: int = 1) -> Write:
    """Conservative CAS payload, atomic metadata and directory allocation bound.

    Metadata uses the native document ceiling; object fanout charges an inode
    per MiB (well above native's 64 MiB pack grid), plus per-file fanout. These
    are transient write estimates, never TensorFS reachability or quota policy.
    """
    if not pressure_enabled():
        return Write(path or Path("/"), 0, 0)
    if path is None:
        raise StorageRefusal("native writer has no configured Store root")
    from cozy_runtime.internal import fill

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


def acquire(*writes: Write, recover: bool = True) -> Lease:
    """Admit the writes, reclaiming workspace cache once on a refusal when ``recover``.

    Without ``recover`` a refusal is a pure observation: its facts measure the disk and
    nothing is evicted for a request the caller may shrink instead.
    """
    if not pressure_enabled():
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

    def admission() -> AbstractContextManager[dict[str, object]]:
        if _channel is not None:
            return _exchange(body, tuple(row.path for row in writes))
        from . import local_storage_admission

        registrations = tuple(_reclaimers.values())
        if not registrations:
            raise StorageRefusal("local write has no live workspace admission owner")
        return local_storage_admission.acquire(registrations[0].root, writes)

    context = admission()
    try:
        context.__enter__()
    except StorageRefusal as exc:
        # The failed exchange/exclusion is closed before native recovery. A single
        # retry re-observes real capacity; a busy collector cannot fabricate space.
        if not recover or not _recover(writes, exc.facts):
            raise
        context = admission()
        context.__enter__()
    return Lease(context)


def _recover(writes: tuple[Write, ...], facts: dict[str, int]) -> bool:
    if _reclaiming.get() or not facts or not _reclaimers:
        return False
    required = facts.get("required")
    available = facts.get("available")
    reserve = facts.get("reserve", 0) + facts.get("reserved", 0)
    if type(required) is not int or type(available) is not int or min(required, available) < 0:
        return False
    target = max(1, required + reserve - available)
    if facts.get("required_inodes", 0) > facts.get("available_inodes", 0) - facts.get(
        "reserve_inodes", 0
    ) - facts.get("reserved_inodes", 0):
        target = (1 << 63) - 1
    from . import local_storage_admission

    token = _reclaiming.set(True)
    try:
        devices = {local_storage_admission.device(row.path) for row in writes}
        matched = False
        seen: set[Path] = set()
        for registration in tuple(_reclaimers.values()):
            root = registration.root
            if root not in seen and local_storage_admission.device(root) in devices:
                registration.reclaim(target)
                seen.add(root)
                matched = True
        return matched
    except OSError as exc:
        raise StorageRefusal(
            f"reclamation filesystem identity is unavailable: {exc}",
            code="storage_capacity_unavailable",
        ) from exc
    finally:
        _reclaiming.reset(token)


@contextmanager
def admit(*writes: Write) -> Iterator[None]:
    """Hold admission through the actual write/child exit, never just its launch."""
    if not pressure_enabled():
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
    body: bytes, directories: tuple[Path, ...] = (), *, subject: str = "Runtime disk write"
) -> Iterator[dict[str, object]]:
    if _channel is None or _held.get():
        raise StorageRefusal(
            "supervisor storage channel is unavailable or already held",
            code="storage_admission_unavailable",
        )
    if len(body) > 4096:
        raise StorageRefusal(
            "Runtime write admission exceeds its control bound", code="storage_request_invalid"
        )
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
                raise _refusal(subject, answer)
        except (OSError, ValueError) as exc:
            raise StorageRefusal(
                f"supervisor storage admission is unavailable for {subject}: "
                f"{type(exc).__name__}: {exc}",
                code="storage_admission_unavailable",
            ) from exc
        yield answer


def _refusal(subject: str, answer: object) -> StorageRefusal:
    """The supervisor's own code and reason; a reasonless answer says that it was one."""
    fields = answer if isinstance(answer, dict) else {}
    code, reason, facts = fields.get("code"), fields.get("reason"), fields.get("facts")
    if not isinstance(code, str) or not code:
        code = None
    if not isinstance(reason, str) or not reason:
        reason = "the supervisor gave no reason"
    return StorageRefusal(
        f"supervisor refused {subject}: {reason}",
        {key: value for key, value in facts.items() if type(value) is int}
        if isinstance(facts, dict)
        else None,
        code=code,
    )
