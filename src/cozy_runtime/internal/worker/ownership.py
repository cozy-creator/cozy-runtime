"""One fixed pod's accepted control authority, under the existing worker-root lease."""

from __future__ import annotations

import os
import stat
from pathlib import Path
from typing import Annotated

import msgspec

from cozy_runtime import canonical_json
from cozy_runtime.internal.refusal import LaunchRefusal

_MAX_BYTES = 4096
_Counter = Annotated[int, msgspec.Meta(ge=0)]


class _History(msgspec.Struct, frozen=True):
    worker_boot_id: str
    record_owner_epoch: _Counter
    record_owner_id: str
    control_stream_epoch: _Counter

    def __post_init__(self) -> None:
        if (
            self.control_stream_epoch == 0 and (self.record_owner_epoch or self.record_owner_id)
        ) or (self.control_stream_epoch > 0 and not self.record_owner_id):
            raise ValueError("control authority is incomplete")


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
        if len(raw) > _MAX_BYTES:
            raise ValueError("control history exceeds its fixed bound")
        history = canonical_json.decode_as(raw, _History)
        if history.worker_boot_id != boot_id:
            raise ValueError("control history belongs to another pod boot")
        return history.record_owner_epoch, history.record_owner_id, history.control_stream_epoch
    except (OSError, ValueError) as exc:
        raise LaunchRefusal("worker_ownership_invalid", "control history is invalid") from exc


def write(path: Path, boot_id: str, owner: int, identity: str, stream_epoch: int) -> None:
    """Publish and sync the next stream before any ClaimAck can name it."""
    raw = canonical_json.encode(
        msgspec.to_builtins(_History(boot_id, owner, identity, stream_epoch))
    )
    if min(owner, stream_epoch) < 0 or len(raw) > _MAX_BYTES:
        raise ValueError("control history counter or size is invalid")
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
