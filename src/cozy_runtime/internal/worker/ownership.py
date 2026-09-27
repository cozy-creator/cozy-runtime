"""One fixed pod's accepted control authority, under the existing worker-root lease."""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path
from typing import Any

from cozy_runtime.internal.refusal import LaunchRefusal

_MAX_BYTES = 4096
_MAX_COUNTER = (1 << 64) - 1
_FIELDS = frozenset(
    {"worker_boot_id", "record_owner_epoch", "record_owner_id", "control_stream_epoch"}
)


def read(path: Path, boot_id: str, *, required: bool) -> tuple[int, str, int]:
    """Load the existing counters after ExecutorSupervision owns the worker root."""
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except FileNotFoundError as exc:
        if required:
            raise LaunchRefusal(
                "worker_ownership_absent", "a resumed pod has no accepted control history"
            ) from exc
        write(path, boot_id, 0, "", 0)
        return 0, "", 0
    except OSError as exc:
        raise LaunchRefusal("worker_ownership_unreadable", "cannot open control history") from exc
    try:
        with os.fdopen(descriptor, "rb") as stream:
            info = os.fstat(stream.fileno())
            if (
                not stat.S_ISREG(info.st_mode)
                or info.st_uid != os.geteuid()
                or info.st_mode & 0o077
            ):
                raise ValueError("control history must be a private owned regular file")
            raw = stream.read(_MAX_BYTES + 1)
        value: Any = json.loads(raw, object_pairs_hook=_unambiguous)
        if len(raw) > _MAX_BYTES or not isinstance(value, dict) or not _FIELDS.issubset(value):
            raise ValueError("control history fields or size are invalid")
        if value["worker_boot_id"] != boot_id:
            raise ValueError("control history belongs to another pod boot")
        owner, identity, stream_epoch = (
            value["record_owner_epoch"],
            value["record_owner_id"],
            value["control_stream_epoch"],
        )
        _validate(owner, identity, stream_epoch)
        return owner, identity, stream_epoch
    except (OSError, ValueError, TypeError) as exc:
        raise LaunchRefusal("worker_ownership_invalid", "control history is invalid") from exc


def write(path: Path, boot_id: str, owner: int, identity: str, stream_epoch: int) -> None:
    """Publish and sync the next stream before any ClaimAck can name it."""
    _validate(owner, identity, stream_epoch)
    raw = _encode(
        {
            "worker_boot_id": boot_id,
            "record_owner_epoch": owner,
            "record_owner_id": identity,
            "control_stream_epoch": stream_epoch,
        }
    )
    if len(raw) > _MAX_BYTES:
        raise ValueError("control history exceeds its fixed bound")
    temporary = path.with_suffix(".pending")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            os.fchmod(stream.fileno(), 0o600)
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def _validate(owner: Any, identity: Any, stream_epoch: Any) -> None:
    if (
        type(owner) is not int
        or type(stream_epoch) is not int
        or not 0 <= owner <= _MAX_COUNTER
        or not 0 <= stream_epoch <= _MAX_COUNTER
        or not isinstance(identity, str)
    ):
        raise ValueError("control authority or counter is invalid")
    if (stream_epoch == 0 and (owner != 0 or identity != "")) or (
        stream_epoch > 0 and not identity
    ):
        raise ValueError("control authority is incomplete")


def _encode(value: dict[str, Any]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def _unambiguous(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value = dict(pairs)
    if len(value) != len(pairs):
        raise ValueError("control history repeats a field")
    return value
