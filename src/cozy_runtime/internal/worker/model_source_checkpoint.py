"""Bounded checkpoint inspection and granted transport through the TensorFS owner."""

from __future__ import annotations

import base64
import re
from pathlib import Path
from typing import Any

from cozy_runtime.internal import fill
from cozy_runtime.protocol import documents, weights_limits
from cozy_runtime.protocol import worker_pb2 as pb

from .model_source_prepare import _IDENTIFIER, _OBJECT_ID, ModelSourceRefusal, _printable

Request = pb.SourceCheckpointPageRequest | pb.SourceCheckpointTransferRequest


def _subject(request: Request) -> dict[str, Any]:
    return {
        "record_owner_epoch": request.record_owner_epoch,
        "control_stream_epoch": request.control_stream_epoch,
        "worker_boot_id": request.worker_boot_id,
        "operation_id": request.operation_id,
        "source_selection_digest": request.source_selection_digest,
        "slot": request.slot,
        "plan_digest": request.plan_digest,
        "head": request.head,
    }


def _identity(request: Request, root: Path) -> None:
    if (
        not root.is_absolute()
        or _IDENTIFIER.fullmatch(request.operation_id) is None
        or _IDENTIFIER.fullmatch(request.slot) is None
        or len(request.source_selection_digest) != 32
        or len(request.plan_digest) != 32
        or len(request.head.digest) != 32
        or request.head.length <= 0
    ):
        raise ModelSourceRefusal(
            "source_checkpoint_identity_invalid", "invalid checkpoint identity"
        )


def _uint(value: object, *, positive: bool = False) -> int:
    if type(value) is not int or not int(positive) <= value < 1 << 64:
        raise ModelSourceRefusal("source_checkpoint_result_invalid", "invalid checkpoint counter")
    return value


def _ref(object_id: object, length: object) -> pb.Ref:
    if not isinstance(object_id, str) or _OBJECT_ID.fullmatch(object_id) is None:
        raise ModelSourceRefusal("source_checkpoint_result_invalid", "invalid checkpoint object")
    return pb.Ref(digest=documents.raw(object_id), length=_uint(length, positive=True))


def _optional_ref(value: object) -> pb.Ref | None:
    if value is None:
        return None
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ModelSourceRefusal("source_checkpoint_result_invalid", "invalid checkpoint reference")
    return _ref(*value)


def _page(store: Any, request: Request, offset: int, limit: int) -> pb.SourceCheckpointPageResult:
    if not 1 <= limit <= weights_limits.MAX_SOURCE_CHECKPOINT_OBJECTS:
        raise ModelSourceRefusal(
            "source_checkpoint_page_bound", "checkpoint page exceeds its bound"
        )
    value = store.source_checkpoint_page(
        documents.spell(bytes(request.head.digest)),
        request.head.length,
        operation_id=request.operation_id,
        slot=request.slot,
        plan_digest=documents.spell(bytes(request.plan_digest)),
        offset=offset,
        limit=limit,
    )
    if not isinstance(value, dict):
        raise ModelSourceRefusal("source_checkpoint_result_invalid", "invalid checkpoint page")
    objects = value.get("objects")
    if not isinstance(objects, list) or len(objects) > limit:
        raise ModelSourceRefusal("source_checkpoint_result_invalid", "invalid checkpoint inventory")
    result = pb.SourceCheckpointPageResult(
        **_subject(request),
        index=_uint(value.get("index")),
        bytes=_uint(value.get("bytes")),
        previous=_optional_ref(value.get("previous")),
        progress=_optional_ref(value.get("progress")),
    )
    for row in objects:
        if not isinstance(row, dict) or row.get("kind") not in {"blob", "manifest"}:
            raise ModelSourceRefusal("source_checkpoint_result_invalid", "invalid object kind")
        result.objects.append(
            pb.SourceCheckpointObject(
                ref=_ref(row.get("object_id"), row.get("length")),
                manifest=row["kind"] == "manifest",
            )
        )
    next_offset = value.get("next_offset")
    if next_offset is not None:
        next_offset = _uint(next_offset)
        if not offset < next_offset < 1 << 32:
            raise ModelSourceRefusal(
                "source_checkpoint_result_invalid", "invalid page continuation"
            )
        result.next_offset = next_offset
        result.has_more = True
    return result


def _failure(exc: Exception) -> tuple[str, str]:
    # Transfer exceptions can carry capabilities. Keep the typed reason, never their text.
    code = re.sub(r"[^A-Za-z0-9._-]", "_", str(getattr(exc, "code", "failed")))[:128]
    if not code.startswith("source_checkpoint_"):
        code = "source_checkpoint_" + code
    return code.lower(), f"checkpoint operation refused ({type(exc).__name__})"


def page(
    request: pb.SourceCheckpointPageRequest, *, tensorfs_root: Path
) -> pb.SourceCheckpointPageResult:
    try:
        _identity(request, tensorfs_root)
        store = fill.tensorfs_module().Store.ensure(str(tensorfs_root))
        return _page(store, request, request.offset, request.limit)
    except Exception as exc:
        code, detail = _failure(exc)
        return pb.SourceCheckpointPageResult(
            **_subject(request), safe_code=code, safe_detail=detail
        )


def _member(store: Any, request: pb.SourceCheckpointTransferRequest) -> None:
    """The admitted Link, not the request, determines which object may move."""
    offset = 0
    while True:
        result = _page(store, request, offset, weights_limits.MAX_SOURCE_CHECKPOINT_OBJECTS)
        if not request.object.manifest and request.object.ref in (
            request.head,
            result.previous,
            result.progress,
        ):
            return
        if request.object in result.objects:
            return
        if not result.has_more:
            raise ModelSourceRefusal("object_ungranted", "object is outside checkpoint Link")
        offset = result.next_offset


def _grant(request: pb.SourceCheckpointTransferRequest) -> str:
    grant = request.upload_grant
    expected_id = documents.spell(bytes(request.object.ref.digest))
    headers = {header.name: header.value for header in grant.required_headers}
    expected_headers = {
        "if-none-match": "*",
        "x-amz-checksum-sha256": base64.b64encode(request.object.ref.digest).decode("ascii"),
    }
    if (
        grant.object_id != expected_id
        or grant.length != request.object.ref.length
        or len(grant.required_headers) != len(expected_headers)
        or headers != expected_headers
        or not _printable(grant.url, weights_limits.MAX_WEIGHTS_GRANT_URL_BYTES, empty=False)
    ):
        raise ModelSourceRefusal("grant_invalid", "upload grant differs from checkpoint object")
    return "\n".join((grant.url, *(f"{name}: {value}" for name, value in headers.items())))


def transfer(
    request: pb.SourceCheckpointTransferRequest,
    *,
    tensorfs_root: Path,
    allow_local: bool = False,
) -> pb.SourceCheckpointTransferStatus:
    result = pb.SourceCheckpointTransferStatus(
        **_subject(request),
        object=request.object,
        transfer_id=request.transfer_id,
        grant_revision=request.grant_revision,
    )
    try:
        _identity(request, tensorfs_root)
        if (
            _IDENTIFIER.fullmatch(request.transfer_id) is None
            or request.grant_revision == 0
            or len(request.object.ref.digest) != 32
            or request.object.ref.length == 0
        ):
            raise ModelSourceRefusal("transfer_invalid", "invalid checkpoint transfer")
        store = fill.tensorfs_module().Store.ensure(str(tensorfs_root))
        decision = request.WhichOneof("decision")
        restores_head = (
            decision == "download_url"
            and not request.object.manifest
            and request.object.ref == request.head
        )
        if not restores_head:
            _member(store, request)
        object_id = documents.spell(bytes(request.object.ref.digest))
        if decision == "upload_grant":
            receipt = store.source_checkpoint_push(
                object_id,
                request.object.ref.length,
                request.object.manifest,
                _grant(request),
                allow_local=allow_local,
            )
            result.http_status = _uint(receipt["http_status"])
            result.state = (
                pb.WEIGHTS_TRANSFER_STATE_ALREADY_PRESENT
                if receipt["held"]
                else pb.WEIGHTS_TRANSFER_STATE_UPLOADED
            )
        elif decision == "download_url":
            if not _printable(
                request.download_url, weights_limits.MAX_MODEL_SOURCE_URL_BYTES, empty=False
            ):
                raise ModelSourceRefusal("grant_invalid", "invalid checkpoint download capability")
            receipt = store.source_checkpoint_fetch(
                object_id,
                request.object.ref.length,
                request.object.manifest,
                request.download_url,
                allow_local=allow_local,
            )
            # A replacement Store can first fetch its authorized head. Its contents must
            # bind the claimed operation/slot/plan before it can authorize any other object.
            if restores_head:
                _page(store, request, 0, 1)
            result.state = pb.WEIGHTS_TRANSFER_STATE_HELD
        else:
            raise ModelSourceRefusal("grant_absent", "checkpoint transfer has no decision")
        result.transferred_bytes = _uint(receipt["transferred"])
        result.checksum_sha256 = object_id
    except Exception as exc:
        result.state = pb.WEIGHTS_TRANSFER_STATE_FAILED
        result.safe_code, result.safe_detail = _failure(exc)
    return result
