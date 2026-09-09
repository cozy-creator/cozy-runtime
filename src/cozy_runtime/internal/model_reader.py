"""Parent-owned read leases over exact admitted model inputs.

The existing executor control lane and bounded attempt spool carry reads. There is
no output declaration, writer epoch, publication, or new durable custody authority.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import math
import re
import secrets
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import msgspec

from cozy_runtime import canonical_json
from cozy_runtime.author._errors import CapabilityError
from cozy_runtime.author._weights import (
    MAX_WEIGHTS_COMPONENTS,
    MAX_WEIGHTS_SOURCE_CONFIGS,
    MAX_WEIGHTS_SOURCE_READ,
    MAX_WEIGHTS_SOURCE_ROLES,
    MAX_WEIGHTS_SOURCE_TENSORS,
    MAX_WEIGHTS_SOURCES,
    WeightsSource,
    WeightsSourcePart,
)
from cozy_runtime.internal.weights_sink import source_structure
from cozy_runtime.internal.weights_writer import _read, _write

_VIEW = re.compile(r"[0-9a-f]{32}\Z")
_METADATA_BYTES = MAX_WEIGHTS_SOURCE_READ


def _path(spool: Path, view: str) -> Path:
    if _VIEW.fullmatch(view) is None:
        raise CapabilityError("invalid model view handle", code="weights_reader_identity")
    return spool / f"model-reader-{view}-output"


def _bound(value: Any, *, positive: bool = False) -> int:
    if type(value) is not int or value < int(positive):
        raise CapabilityError("invalid model read range", code="weights_reader_bounds")
    return value


class ModelReaderProxy:
    def __init__(
        self,
        spool: Path,
        exchange: Callable[..., dict[str, Any]],
        view: str,
        facts: Mapping[str, Any],
    ) -> None:
        self.spool, self.exchange, self.view = spool, exchange, view
        self.structure = msgspec.convert(facts["structure"], type=WeightsSource)
        self.identities: Mapping[tuple[str, str], str] = {
            (row[0], row[1]): row[2] for row in facts["identities"]
        }
        self.tensors = {(row.component, row.key): row for row in self.structure.tensors}

    @classmethod
    def open(
        cls, spool: Path, exchange: Callable[..., dict[str, Any]], manifest: str
    ) -> ModelReaderProxy:
        result = exchange("model_reader", {"operation": "open", "manifest": manifest})
        cls._answer(result)
        view = str(result["view_id"])
        path = _path(spool, view)
        try:
            with _read(path, int(result["length"]), _METADATA_BYTES) as source:
                return cls(spool, exchange, view, json.load(source))
        except BaseException:
            exchange("model_reader", {"operation": "close", "view_id": view})
            raise
        finally:
            path.unlink(missing_ok=True)

    @staticmethod
    def _answer(result: Mapping[str, Any]) -> None:
        if not result.get("ok"):
            raise CapabilityError(
                str(result.get("detail", "model read refused")),
                code=str(result.get("code", "weights_reader_refused")),
            )

    def call(self, operation: str, **fields: Any) -> dict[str, Any]:
        result = self.exchange(
            "model_reader", {"operation": operation, "view_id": self.view, **fields}
        )
        self._answer(result)
        return result

    def check(self) -> None:
        self.call("check")

    def close(self) -> None:
        self.call("close")

    def read_part_into(
        self, component: str, key: str, role: str, offset: int, into: memoryview
    ) -> None:
        length = into.nbytes
        self.call("read", component=component, key=key, role=role, offset=offset, length=length)
        path = _path(self.spool, self.view)
        try:
            with _read(path, length, MAX_WEIGHTS_SOURCE_READ) as source:
                if cast(Any, source).readinto(into) != length:
                    raise CapabilityError("short model read handoff", code="weights_reader_bounds")
        finally:
            path.unlink(missing_ok=True)

    def read_values_into(self, component: str, key: str, offset: int, into: memoryview) -> None:
        from cozy_runtime.internal.model_values import read_values_into

        read_values_into(self, component, key, offset, into)


@dataclass
class _Held:
    lease: Any
    header: bytes
    structure: WeightsSource
    identities: list[tuple[str, str, str]]
    digest: bytes
    parts: dict[tuple[str, str, str], WeightsSourcePart]

    def close(self) -> None:
        try:
            self.lease.recheck()
        finally:
            self.lease.release()


class ModelReaderBroker:
    """Ephemeral attempt handles; TensorFS's existing native lease owns the bytes."""

    def __init__(self, store: Callable[[], Any]) -> None:
        self.store = store
        self.lock = threading.RLock()
        self.views: dict[tuple[str, int, bytes, str], _Held] = {}

    @staticmethod
    def _running(attempt: Any) -> None:
        if attempt.canceling or attempt.state != "running" or attempt.spool is None:
            raise CapabilityError("model read attempt is not running", code="weights_reader_closed")

    def _open(self, attempt: Any, manifest: str) -> tuple[str, _Held]:
        import tensorfs

        from cozy_runtime.internal.worker.grants import MODEL_PREFIX

        allowed = {
            row["digest"]: row["length"]
            for row in attempt.spec.get("inputs", [])
            if str(row.get("input_id", "")).startswith(MODEL_PREFIX)
        }
        length = allowed.get(manifest)
        if length is None:
            raise CapabilityError(
                "model is outside the input grant", code="weights_source_ungranted"
            )
        if (
            sum(key[:2] == (attempt.request_id, attempt.attempt) for key in self.views)
            >= MAX_WEIGHTS_SOURCES
        ):
            raise CapabilityError("too many open model views", code="weights_reader_bounds")
        store = self.store()
        document = store.manifest(manifest)
        if len(document["manifest"]) != length:
            raise CapabilityError("model manifest length changed", code="weights_reader_identity")
        raw = bytes(document["header"])
        header = tensorfs.parse_header(raw)
        structure = source_structure(header)
        if (
            not 0 < len(structure.tensors) <= MAX_WEIGHTS_SOURCE_TENSORS
            or len(structure.configs) > MAX_WEIGHTS_SOURCE_CONFIGS
            or len(header["components"]) > MAX_WEIGHTS_COMPONENTS
            or any(len(row.parts) > MAX_WEIGHTS_SOURCE_ROLES for row in structure.tensors)
        ):
            raise CapabilityError("model geometry exceeds bounds", code="weights_reader_bounds")
        identities = [
            (
                component,
                key,
                "sha256:"
                + hashlib.sha256(
                    b"cozy.model.encoded-tensor/1\0"
                    + canonical_json.encode(msgspec.to_builtins(entry))
                ).hexdigest(),
            )
            for component, tensors in header["components"].items()
            for key, entry in tensors.items()
        ]
        lease = store.acquire_cozytensors(manifest)
        try:
            lease.recheck()
            # The header projected above must be in the acquired native closure.
            if (tensorfs.object_id(raw), len(raw)) not in lease.objects:
                raise CapabilityError("model header is not held", code="weights_reader_identity")
            self._running(attempt)
            return secrets.token_hex(16), _Held(
                lease,
                raw,
                structure,
                identities,
                attempt.digest,
                {
                    (row.component, row.key, part.name): part
                    for row in structure.tensors
                    for part in row.parts
                },
            )
        except BaseException:
            lease.release()
            raise

    def handle(self, attempt: Any, frame: Mapping[str, Any]) -> dict[str, Any]:
        with self.lock:
            view = str(frame.get("view_id", ""))
            key = (attempt.request_id, attempt.attempt, attempt.digest, view)
            try:
                operation = frame.get("operation")
                if operation == "close":
                    held = self.views.pop(key, None)
                    if held is None or held.digest != attempt.digest:
                        raise CapabilityError(
                            "model view is not current", code="weights_reader_closed"
                        )
                    held.close()
                    return {"ok": True}
                self._running(attempt)
                if operation == "open":
                    view, held = self._open(attempt, str(frame.get("manifest", "")))
                    key = (attempt.request_id, attempt.attempt, attempt.digest, view)
                    self.views[key] = held
                    raw = msgspec.json.encode(
                        {"structure": held.structure, "identities": held.identities}
                    )
                    length = _write(_path(attempt.spool, view), io.BytesIO(raw), _METADATA_BYTES)
                    self._running(attempt)
                    return {"ok": True, "view_id": view, "length": length}
                held = self.views.get(key)
                if held is None or held.digest != attempt.digest:
                    raise CapabilityError("model view is not current", code="weights_reader_closed")
                held.lease.recheck()
                if operation == "check":
                    return {"ok": True}
                if operation != "read":
                    raise CapabilityError(
                        "unknown model read operation", code="weights_reader_refused"
                    )
                component, tensor_key, role = (
                    str(frame.get(name, "")) for name in ("component", "key", "role")
                )
                part = held.parts.get((component, tensor_key, role))
                if part is None:
                    raise CapabilityError("model part is absent", code="weights_reader_tensor")
                offset, length = (
                    _bound(frame.get("offset")),
                    _bound(frame.get("length"), positive=True),
                )
                if length > MAX_WEIGHTS_SOURCE_READ:
                    raise CapabilityError("model read exceeds bound", code="weights_reader_bounds")
                # Native geometry remains authoritative; preflight avoids oversized allocations.
                import tensorfs

                if offset + length > math.prod(part.shape) * tensorfs.dtypes()[part.dtype]:
                    raise CapabilityError(
                        "model read leaves the part", code="weights_reader_bounds"
                    )
                value = bytearray(length)
                held.lease.read_part_into(held.header, component, tensor_key, role, offset, value)
                held.lease.recheck()
                self._running(attempt)
                _write(_path(attempt.spool, view), io.BytesIO(value), length)
                return {"ok": True}
            except Exception as exc:
                held = self.views.pop(key, None)
                if held is not None:
                    with contextlib.suppress(Exception):
                        held.close()
                return {
                    "ok": False,
                    "code": str(getattr(exc, "code", "weights_reader_refused")),
                    "detail": "native model read refused: " + type(exc).__name__,
                }

    def close_attempt(self, attempt: Any) -> None:
        with self.lock:
            for key in list(self.views):
                if key[:3] == (attempt.request_id, attempt.attempt, attempt.digest):
                    held = self.views.pop(key)
                    with contextlib.suppress(Exception):
                        held.close()
