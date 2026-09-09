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
from cozy_runtime.internal import storage_admission
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

    def __init__(self, store: Callable[[], Any], store_root: Path | None = None) -> None:
        self.store, self.store_root = store, store_root
        self.lock = threading.RLock()
        self.views: dict[tuple[str, int, bytes, str], _Held] = {}

    @staticmethod
    def _running(attempt: Any) -> None:
        if attempt.canceling or attempt.state != "running" or attempt.spool is None:
            raise CapabilityError("model read attempt is not running", code="weights_reader_closed")

    @staticmethod
    def _source_length(attempt: Any, manifest: str) -> int:
        from cozy_runtime.internal.worker.grants import MODEL_PREFIX

        allowed = {
            row["digest"]: row["length"]
            for row in attempt.spec.get("inputs", [])
            if str(row.get("input_id", "")).startswith(MODEL_PREFIX)
        }
        length = allowed.get(manifest)
        if type(length) is not int or length <= 0:
            raise CapabilityError(
                "model is outside the input grant", code="weights_source_ungranted"
            )
        return int(length)

    def _open(self, attempt: Any, manifest: str) -> tuple[str, _Held]:
        import tensorfs

        length = self._source_length(attempt, manifest)
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
            if (tensorfs.object_id(raw), len(raw)) not in lease.objects:
                raise CapabilityError("model header is not held", code="weights_reader_identity")
            self._running(attempt)
            parts = {
                (row.component, row.key, part.name): part
                for row in structure.tensors
                for part in row.parts
            }
            return secrets.token_hex(16), _Held(
                lease, raw, structure, identities, attempt.digest, parts
            )
        except BaseException:
            lease.release()
            raise

    @staticmethod
    def _read_span(held: _Held, frame: Mapping[str, Any]) -> tuple[str, str, str, int, int]:
        import tensorfs

        component, key, role = (str(frame.get(name, "")) for name in ("component", "key", "role"))
        part = held.parts.get((component, key, role))
        if part is None:
            raise CapabilityError("model part is absent", code="weights_reader_tensor")
        offset, length = _bound(frame.get("offset")), _bound(frame.get("length"), positive=True)
        if (
            length > MAX_WEIGHTS_SOURCE_READ
            or offset + length > math.prod(part.shape) * tensorfs.dtypes()[part.dtype]
        ):
            raise CapabilityError(
                "model read leaves its bounded part", code="weights_reader_bounds"
            )
        return component, key, role, offset, length

    def _current(self, attempt: Any, view: str) -> _Held:
        held = self.views.get((attempt.request_id, attempt.attempt, attempt.digest, view))
        if held is None:
            raise CapabilityError("model view is not current", code="weights_reader_closed")
        held.lease.recheck()
        return held

    def _writes(
        self, attempt: Any, frame: Mapping[str, Any]
    ) -> tuple[storage_admission.Write, ...]:
        operation = frame.get("operation")
        if operation not in {"open", "read"}:
            return ()
        self._running(attempt)
        if operation == "open":
            self._source_length(attempt, str(frame.get("manifest", "")))
            return (
                storage_admission.native_write(self.store_root),
                storage_admission.Write(attempt.spool, _METADATA_BYTES, 1),
            )
        held = self._current(attempt, str(frame.get("view_id", "")))
        *_, length = self._read_span(held, frame)
        return (storage_admission.Write(attempt.spool, length, 1),)

    def handle(self, attempt: Any, frame: Mapping[str, Any]) -> dict[str, Any]:
        view = str(frame.get("view_id", ""))
        try:
            with self.lock:
                writes = self._writes(attempt, frame)
            # No broker mutex is held while admission waits. Cancellation can close
            # the view; grant, scope and ranges are rechecked after reservation.
            with (
                storage_admission.admit(*writes) if writes else contextlib.nullcontext(),
                self.lock,
            ):
                return self._handle(attempt, frame)
        except Exception as exc:
            with self.lock:
                return self._failure(attempt, view, exc)

    def _failure(self, attempt: Any, view: str, exc: Exception) -> dict[str, Any]:
        held = self.views.pop((attempt.request_id, attempt.attempt, attempt.digest, view), None)
        if held is not None:
            with contextlib.suppress(Exception):
                held.close()
        return {
            "ok": False,
            "code": str(getattr(exc, "code", "weights_reader_refused")),
            "detail": "native model read refused: " + type(exc).__name__,
        }

    def _handle(self, attempt: Any, frame: Mapping[str, Any]) -> dict[str, Any]:
        view = str(frame.get("view_id", ""))
        try:
            operation = frame.get("operation")
            if operation == "close":
                held = self.views.pop(
                    (attempt.request_id, attempt.attempt, attempt.digest, view), None
                )
                if held is not None:
                    held.close()
                return {"ok": True}
            self._running(attempt)
            if operation == "open":
                view, held = self._open(attempt, str(frame.get("manifest", "")))
                self.views[attempt.request_id, attempt.attempt, attempt.digest, view] = held
                raw = msgspec.json.encode(
                    {"structure": held.structure, "identities": held.identities}
                )
                length = _write(_path(attempt.spool, view), io.BytesIO(raw), _METADATA_BYTES)
                self._running(attempt)
                return {"ok": True, "view_id": view, "length": length}
            held = self._current(attempt, view)
            if operation == "check":
                return {"ok": True}
            if operation != "read":
                raise CapabilityError("unknown model read operation", code="weights_reader_refused")
            component, key, role, offset, length = self._read_span(held, frame)
            value = bytearray(length)
            held.lease.read_part_into(held.header, component, key, role, offset, value)
            held.lease.recheck()
            self._running(attempt)
            _write(_path(attempt.spool, view), io.BytesIO(value), length)
            return {"ok": True}
        except Exception as exc:
            return self._failure(attempt, view, exc)

    def close_attempt(self, attempt: Any) -> None:
        with self.lock:
            for key in list(self.views):
                if key[:3] == (attempt.request_id, attempt.attempt, attempt.digest):
                    held = self.views.pop(key)
                    with contextlib.suppress(Exception):
                        held.close()

    def close(self) -> None:
        with self.lock:
            for key in list(self.views):
                held = self.views.pop(key)
                with contextlib.suppress(Exception):
                    held.close()
