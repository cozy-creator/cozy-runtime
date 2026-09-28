"""Bounded checkpoint inspection and granted transport through the TensorFS owner."""

from __future__ import annotations

import base64
import re
from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from pathlib import Path
from typing import TypedDict

from cozy_runtime.internal import fill
from cozy_runtime.protocol import documents, weights_limits
from cozy_runtime.protocol import worker_pb2 as pb

from .model_source_prepare import _IDENTIFIER, _OBJECT_ID, ModelSourceRefusal, _printable

Request = pb.CheckpointPageRequest | pb.CheckpointTransferRequest


class _Subject(TypedDict):
    record_owner_epoch: int
    control_stream_epoch: int
    worker_boot_id: str
    subject: pb.CheckpointSubject
    plan_digest: bytes
    head: pb.Ref


def _subject(request: Request) -> _Subject:
    return {
        "record_owner_epoch": request.record_owner_epoch,
        "control_stream_epoch": request.control_stream_epoch,
        "worker_boot_id": request.worker_boot_id,
        "subject": request.subject,
        "plan_digest": request.plan_digest,
        "head": request.head,
    }


def _operation_slot(subject: pb.CheckpointSubject) -> tuple[str, str]:
    kind = subject.WhichOneof("kind")
    if kind == "source":
        source = subject.source
        if len(source.source_selection_digest) != 32:
            raise ModelSourceRefusal("checkpoint_identity_invalid", "invalid source subject")
        return source.operation_id, source.slot
    if kind == "weights":
        weights = subject.weights
        if (
            len(weights.invocation_spec_digest) != 32
            or len(weights.tensorfs_declaration_digest) != 32
            or _OBJECT_ID.fullmatch(weights.weights_transaction_id) is None
            or weights.writer_epoch == 0
        ):
            raise ModelSourceRefusal("checkpoint_identity_invalid", "invalid weights subject")
        return weights.request_id, weights.output_slot
    raise ModelSourceRefusal("checkpoint_identity_invalid", "checkpoint needs one exact subject")


def _identity(request: Request, root: Path) -> None:
    operation, slot = _operation_slot(request.subject)
    if (
        not root.is_absolute()
        or _IDENTIFIER.fullmatch(operation) is None
        or _IDENTIFIER.fullmatch(slot) is None
        or len(request.plan_digest) != 32
        or len(request.head.digest) != 32
        or request.head.length <= 0
    ):
        raise ModelSourceRefusal("checkpoint_identity_invalid", "invalid checkpoint identity")


def _uint(value: int, *, positive: bool = False) -> int:
    if not int(positive) <= value < 1 << 64:
        raise ModelSourceRefusal("checkpoint_result_invalid", "invalid checkpoint counter")
    return value


def _ref(object_id: str, length: int) -> pb.Ref:
    if _OBJECT_ID.fullmatch(object_id) is None:
        raise ModelSourceRefusal("checkpoint_result_invalid", "invalid checkpoint object")
    return pb.Ref(digest=documents.raw(object_id), length=_uint(length, positive=True))


def _optional_ref(value: tuple[str, int] | None) -> pb.Ref | None:
    return None if value is None else _ref(*value)


def _page(store: fill.Store, request: Request, offset: int, limit: int) -> pb.CheckpointPageResult:
    if not 1 <= limit <= weights_limits.MAX_CHECKPOINT_OBJECTS:
        raise ModelSourceRefusal("checkpoint_page_bound", "checkpoint page exceeds its bound")
    operation, slot = _operation_slot(request.subject)
    value = store.checkpoint_page(
        documents.spell(bytes(request.head.digest)),
        request.head.length,
        operation_id=operation,
        slot=slot,
        plan_digest=documents.spell(bytes(request.plan_digest)),
        offset=offset,
        limit=limit,
    )
    if len(value["objects"]) > limit:
        raise ModelSourceRefusal("checkpoint_result_invalid", "invalid checkpoint inventory")
    result = pb.CheckpointPageResult(
        **_subject(request),
        index=_uint(value["index"]),
        bytes=_uint(value["bytes"]),
        previous=_optional_ref(value["previous"]),
        progress=_optional_ref(value["progress"]),
    )
    for row in value["objects"]:
        if row["kind"] not in {"blob", "manifest"}:
            raise ModelSourceRefusal("checkpoint_result_invalid", "invalid object kind")
        result.objects.append(
            pb.CheckpointObject(
                ref=_ref(row["object_id"], row["length"]),
                manifest=row["kind"] == "manifest",
            )
        )
    next_offset = value["next_offset"]
    if next_offset is not None:
        next_offset = _uint(next_offset)
        if not offset < next_offset < 1 << 32:
            raise ModelSourceRefusal("checkpoint_result_invalid", "invalid page continuation")
        result.next_offset = next_offset
        result.has_more = True
    return result


def _failure(exc: Exception) -> tuple[str, str]:
    # Transfer exceptions can carry capabilities. Keep the typed reason, never their text.
    code = re.sub(r"[^A-Za-z0-9._-]", "_", str(getattr(exc, "code", "failed")))[:128]
    if not code.startswith("checkpoint_"):
        code = "checkpoint_" + code
    return code.lower(), f"checkpoint operation refused ({type(exc).__name__})"


def page(request: pb.CheckpointPageRequest, *, tensorfs_root: Path) -> pb.CheckpointPageResult:
    try:
        _identity(request, tensorfs_root)
        store = fill.ensure_store(tensorfs_root)
        return _page(store, request, request.offset, request.limit)
    except Exception as exc:
        code, detail = _failure(exc)
        return pb.CheckpointPageResult(**_subject(request), safe_code=code, safe_detail=detail)


def _member(store: fill.Store, request: pb.CheckpointTransferRequest) -> None:
    """The admitted Link, not the request, determines which object may move."""
    offset = 0
    while True:
        result = _page(store, request, offset, weights_limits.MAX_CHECKPOINT_OBJECTS)
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


def _grant(request: pb.CheckpointTransferRequest) -> str:
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
    request: pb.CheckpointTransferRequest,
    *,
    tensorfs_root: Path,
    allow_local: bool = False,
    restore_scope: Callable[[pb.CheckpointTransferRequest], AbstractContextManager[tuple[str, int]]]
    | None = None,
) -> pb.CheckpointTransferStatus:
    result = pb.CheckpointTransferStatus(
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
        store = fill.ensure_store(tensorfs_root)
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
            receipt = store.checkpoint_push(
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
            scope: AbstractContextManager[tuple[str, int] | None] = nullcontext(None)
            if request.subject.WhichOneof("kind") == "weights":
                if restore_scope is None:
                    raise ModelSourceRefusal(
                        "restore_ungranted", "weights restore needs a current intent"
                    )
                scope = restore_scope(request)
            with scope as derived_restore:
                receipt = store.checkpoint_fetch(
                    object_id,
                    request.object.ref.length,
                    request.object.manifest,
                    request.download_url,
                    allow_local=allow_local,
                    **({"derived_restore": derived_restore} if derived_restore is not None else {}),
                )
                # A replacement Store can first fetch its authorized head. Its contents
                # must bind the claimed operation/slot/plan before authorizing members.
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
