"""Negotiated machine-owned model byte sources over the existing durable executor seam.

Only preparation requests descriptors. Header/fit/plan/model behavior remain the existing
TensorFS and Runtime implementations; no step or block crosses this broker.
"""
from __future__ import annotations

import hashlib
import os
import sys
import time
from collections.abc import Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING, Protocol

import msgspec

try:
    from tensorfs import plane as native_plane
except ImportError:
    native_plane = None

from cozy_runtime.author._errors import CapabilityError
from cozy_runtime.author._executor_requests import Exchange, ModelSourceRead, SourceBlob
from cozy_runtime.internal import proctree

if TYPE_CHECKING:
    from tensorfs import ReadPlan
    from tensorfs.plane import Plane, Source, SourceDescriptor

CAPABILITY = "model_sources.descriptors/1"


class ModelSourceProvider(Protocol):
    def header(self, manifest: str) -> bytes: ...
    def assets(self, manifest: str, names: Sequence[str]) -> dict[str, bytes]: ...
    def source(self, plane: Plane, manifest: str, plan: ReadPlan) -> Source: ...
    def close(self) -> None: ...


class _ObjectItem(msgspec.Struct, tag="object", tag_field="source", frozen=True):
    object: str
    object_length: int
    what: str = ""


class _InlineItem(msgspec.Struct, tag="inline", tag_field="source", frozen=True):
    what: str = ""


class SourceProfile(msgspec.Struct, frozen=True):
    event: str
    components: list[str]
    objects: int
    new_objects: int
    new_bytes: int
    rpc_ms: float
    constructor_ms: float
    source_build_ms: float
    total_ms: float
    verification_workers: int = 1
    pending_fd_limit: int = 1
    pending_byte_limit: int = 0


@dataclass(frozen=True, slots=True)
class VerificationLimits:
    workers: int
    pending_fds: int
    pending_bytes: int

    @classmethod
    def for_process(cls) -> VerificationLimits:
        cpus = float(len(os.sched_getaffinity(0)))
        root = Path("/sys/fs/cgroup")
        try:
            relative = next(
                line.split("::", 1)[1]
                for line in Path("/proc/self/cgroup").read_text().splitlines()
                if line.startswith("0::")
            )
            current = root / relative.lstrip("/")
            while current.is_relative_to(root):
                try:
                    quota, period = (current / "cpu.max").read_text().split()
                    if quota.isdigit() and period.isdigit() and int(period):
                        cpus = min(cpus, int(quota) / int(period))
                except (OSError, ValueError):
                    pass
                if current == root:
                    break
                current = current.parent
        except (OSError, StopIteration):
            cpus = 1
        available_bytes = proctree.available_host_bytes()
        workers = max(1, min(4, int(cpus // 2)))
        if available_bytes < workers * (16 << 20):
            workers = 1
        window = min(128 << 20, max(64 << 10, available_bytes // 64))
        return cls(workers, workers * 2, window)


@dataclass(frozen=True, slots=True)
class _Verified:
    descriptor: SourceDescriptor
    seconds: float


@dataclass(slots=True)
class _OwnedFD:
    descriptor: int
    lock: Lock = field(default_factory=Lock)

    def take(self) -> int:
        with self.lock:
            descriptor, self.descriptor = self.descriptor, -1
        if descriptor < 0:
            raise CapabilityError("source import was canceled", code="model_source_unavailable")
        return descriptor

    def close_unclaimed(self) -> None:
        with self.lock:
            descriptor, self.descriptor = self.descriptor, -1
        if descriptor >= 0:
            os.close(descriptor)


@dataclass(frozen=True, slots=True)
class _Offer:
    key: tuple[str, str]
    reply: SourceBlob
    owned: _OwnedFD


def _verify(reply: SourceBlob, owned: _OwnedFD) -> _Verified:
    """The submitted task owns this received fd until native import completes or fails."""
    fd = owned.take()
    try:
        started = time.perf_counter()
        descriptor = native_plane.SourceDescriptor(reply.sha256, reply.length, fd)
        return _Verified(descriptor, time.perf_counter() - started)
    finally:
        os.close(fd)


def _drain(pending: dict[Future[_Verified], _Offer]) -> None:
    for future, offer in pending.items():
        if future.cancel():
            offer.owned.close_unclaimed()
    for future in pending:
        if not future.cancelled():
            with suppress(BaseException):
                future.result()
    pending.clear()


def available() -> bool:
    return native_plane is not None and hasattr(native_plane, "SourceDescriptor") and hasattr(
        native_plane.Plane, "source_from_descriptors"
    )


class BrokeredModelSources:
    def __init__(self, exchange: Exchange, *, limits: VerificationLimits | None = None):
        if not available():
            raise CapabilityError("this weight plane has no descriptor source operation",
                                  code="model_source_unavailable")
        self.exchange = exchange
        self.headers: dict[str, bytes] = {}
        self.descriptors: dict[tuple[str, str], SourceDescriptor] = {}
        self.rpc_seconds = 0.0
        self.limits = limits or VerificationLimits.for_process()
        if min(self.limits.workers, self.limits.pending_fds, self.limits.pending_bytes) < 1:
            raise ValueError("source verification limits must be positive")

    def _open(self, request: ModelSourceRead) -> SourceBlob:
        started = time.perf_counter()
        try:
            reply = self.exchange(request, SourceBlob)
        finally:
            self.rpc_seconds += time.perf_counter() - started
        if not reply.ok or reply.descriptor < 0:
            if reply.descriptor >= 0:
                os.close(reply.descriptor)
            raise CapabilityError(reply.detail or "machine declined this source operation",
                                  code=reply.code or "model_source_unavailable")
        return reply

    def _bytes(self, request: ModelSourceRead) -> bytes:
        reply = self._open(request)
        try:
            if os.fstat(reply.descriptor).st_size != reply.length:
                raise CapabilityError("source blob length changed", code="model_source_changed")
            data = bytearray()
            digest = hashlib.sha256()
            while len(data) < reply.length:
                chunk = os.pread(
                    reply.descriptor, min(1 << 20, reply.length - len(data)), len(data)
                )
                if not chunk:
                    raise CapabilityError("source blob ended early", code="model_source_changed")
                data.extend(chunk)
                digest.update(chunk)
            if digest.hexdigest() != reply.sha256:
                raise CapabilityError("source blob digest differs", code="model_source_changed")
            return bytes(data)
        finally:
            os.close(reply.descriptor)

    def header(self, manifest: str) -> bytes:
        if manifest not in self.headers:
            self.headers[manifest] = self._bytes(ModelSourceRead(manifest=manifest, role="header"))
        return self.headers[manifest]

    def assets(self, manifest: str, names: Sequence[str]) -> dict[str, bytes]:
        return {name: self._bytes(ModelSourceRead(manifest=manifest, role="asset", name=name))
                for name in names}

    def source(self, plane: Plane, manifest: str, plan: ReadPlan) -> Source:
        started = time.perf_counter()
        rpc_before = self.rpc_seconds
        constructor_seconds = 0.0
        new_objects, new_bytes = 0, 0
        rows = msgspec.convert(plan.items(), list[_ObjectItem | _InlineItem], strict=True)
        wanted: dict[str, int] = {}
        for row in rows:
            if isinstance(row, _ObjectItem):
                if row.object in wanted and wanted[row.object] != row.object_length:
                    raise CapabilityError("read plan conflicts on object length",
                                          code="model_source_changed")
                wanted[row.object] = row.object_length
        pending: dict[Future[_Verified], _Offer] = {}
        staged: dict[tuple[str, str], SourceDescriptor] = {}
        pending_bytes = 0

        def finish() -> None:
            nonlocal pending_bytes, constructor_seconds
            done, not_done = wait(pending, return_when=FIRST_COMPLETED)
            future = next(iter(done))
            done.clear()
            not_done.clear()
            offer = pending.pop(future)
            pending_bytes -= offer.reply.length
            result = future.result()
            staged[offer.key] = result.descriptor
            constructor_seconds += result.seconds

        with ThreadPoolExecutor(
            max_workers=self.limits.workers, thread_name_prefix="model-source"
        ) as pool:
            complete = False
            try:
                for object_id, length in wanted.items():
                    key = (manifest, object_id)
                    if key in self.descriptors:
                        continue
                    # One object larger than the byte window proceeds alone. Size never
                    # refuses a model; bounded native verification uses only64KiB per task.
                    while pending and (len(pending) >= self.limits.pending_fds or
                                       pending_bytes + length > self.limits.pending_bytes):
                        finish()
                    reply = self._open(ModelSourceRead(manifest=manifest, role="object",
                                                       name=object_id, length=length))
                    owned = _OwnedFD(reply.descriptor)
                    try:
                        if (reply.sha256 != object_id.removeprefix("sha256:") or
                                reply.length != length):
                            raise CapabilityError("object grant differs from its read plan",
                                                  code="model_source_changed")
                        future = pool.submit(_verify, reply, owned)
                    except BaseException:
                        # submit may fail after queuing work (e.g. thread creation).
                        # Atomic custody prevents closing an fd a started worker took.
                        owned.close_unclaimed()
                        raise
                    pending[future] = _Offer(key, reply, owned)
                    del future
                    pending_bytes += length
                    new_objects += 1
                    new_bytes += length
                while pending:
                    finish()
                complete = True
            finally:
                # Only a successfully canceled UNSTARTED task leaves fd custody here.
                # Started/done tasks close their own fd; drain before releasing mappings.
                _drain(pending)
                if not complete:
                    staged.clear()
        self.descriptors.update(staged)
        descriptors = [self.descriptors[(manifest, object_id)] for object_id in wanted]
        before = time.perf_counter()
        source = plane.source_from_descriptors(descriptors)
        build_seconds = time.perf_counter() - before
        record = SourceProfile(
            "diagnostic.model_source", sorted({row.what.partition("/")[0] for row in rows}),
            len(wanted), new_objects, new_bytes, (self.rpc_seconds - rpc_before) * 1000,
            constructor_seconds * 1000, build_seconds * 1000,
            (time.perf_counter() - started) * 1000,
            self.limits.workers, self.limits.pending_fds, self.limits.pending_bytes,
        )
        print(msgspec.json.encode(record).decode(), file=sys.stderr)
        return source

    def close(self) -> None:
        self.descriptors.clear()
        self.headers.clear()
