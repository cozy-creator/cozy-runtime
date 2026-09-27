"""Explicit completion of one file registered by the running parent's Outputs."""

from __future__ import annotations

import hashlib
import os
import stat
from typing import Annotated, cast

import msgspec

from cozy_runtime.author._assets import FileAsset
from cozy_runtime.author._errors import OutputError
from cozy_runtime.author._markers import AssetBound
from cozy_runtime.author._services import MAX_OUTPUT_BYTES, Attempt
from cozy_runtime.author.sources import _call


class CommittedFile(msgspec.Struct, frozen=True):
    file: Annotated[FileAsset, AssetBound(max_bytes=MAX_OUTPUT_BYTES)]


async def commit(attempt: Attempt, pending: FileAsset) -> FileAsset:
    attempt.check_open("Outputs.commit")
    if any(pending is ready for ready in attempt.committed_files.values()):
        return pending
    if type(pending) is not FileAsset or attempt.pending.get(pending.ref) is not pending:
        raise OutputError(
            "commit requires this Outputs instance's pending FileAsset", code="output_owner"
        )
    prefix = f"attempt:{attempt.request_id}/"
    if pending._attempt != attempt.request_id or not pending.ref.startswith(prefix + "file/"):
        raise OutputError("pending file belongs to another attempt", code="output_owner")
    if ready := attempt.committed_files.get(pending.ref):
        return ready
    path = attempt.spool / f"file-{pending.ref.rsplit('/', 1)[-1]}"
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    digest, length = hashlib.sha256(), 0
    with os.fdopen(fd, "rb") as reader:
        if not stat.S_ISREG(os.fstat(reader.fileno()).st_mode):
            raise OutputError("pending file is not regular", code="output_file")
        while chunk := reader.read(min(1 << 20, pending.size_bytes - length + 1)):
            length += len(chunk)
            if length > pending.size_bytes:
                raise OutputError("pending file changed before commit", code="output_changed")
            digest.update(chunk)
    if length != pending.size_bytes:
        raise OutputError("pending file changed before commit", code="output_changed")
    result = cast(
        CommittedFile,
        await _call(
            "commit_file",
            {
                "slot": pending.ref.removeprefix(prefix),
                "digest": "sha256:" + digest.hexdigest(),
                "size_bytes": length,
                "media_type": pending.media_type,
            },
        ),
    )
    attempt.check_open("Outputs.commit")
    if (
        type(result.file) is not FileAsset
        or result.file.digest != "sha256:" + digest.hexdigest()
        or result.file.size_bytes != length
        or result.file.media_type != pending.media_type
        or not result.file.hydrated
    ):
        raise OutputError("committed file differs from its owned snapshot", code="output_changed")
    attempt.committed_files[pending.ref] = result.file
    return result.file
