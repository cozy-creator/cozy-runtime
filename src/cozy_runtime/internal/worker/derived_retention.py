"""Native independent result custody on the privileged Host loopback."""

from __future__ import annotations

import re
from pathlib import Path

from cozy_runtime.internal import fill
from cozy_runtime.protocol import documents
from cozy_runtime.protocol import worker_pb2 as pb


class RetentionRefusal(ValueError):
    """A request whose identities are malformed or name another result; retrying cannot help."""


def change(
    request: pb.DerivedRetentionRequest, *, tensorfs_root: Path, release: bool
) -> pb.DerivedRetentionResult:
    if (
        not tensorfs_root.is_absolute()
        or re.fullmatch(r"sha256:[0-9a-f]{64}", request.weights_transaction_id) is None
        or len(request.tensorfs_receipt_digest) != 32
        or re.fullmatch(r"sha256:[0-9a-f]{64}", request.retention_id) is None
    ):
        raise RetentionRefusal("derived retention requires exact bounded identities")
    store = fill.store(tensorfs_root)
    action = store.release_derived_retention if release else store.retain_derived_result
    value = action(
        request.weights_transaction_id,
        documents.spell(request.tensorfs_receipt_digest),
        request.retention_id,
    )
    if (
        value.get("transaction_id") != request.weights_transaction_id
        or value.get("receipt_digest") != documents.spell(request.tensorfs_receipt_digest)
        or value.get("retention_id") != request.retention_id
        or value.get("released") is not release
    ):
        raise RetentionRefusal("native derived retention result changed identity")
    result = pb.DerivedRetentionResult(
        weights_transaction_id=request.weights_transaction_id,
        tensorfs_receipt_digest=request.tensorfs_receipt_digest,
        retention_id=request.retention_id,
        released=release,
    )
    if manifest := value.get("manifest"):
        result.manifest.CopyFrom(
            pb.Ref(digest=documents.raw("sha256:" + manifest["sha256"]), length=manifest["length"])
        )
    elif not release:
        raise RetentionRefusal("native retained result has no manifest")
    return result


def release_result(
    request: pb.DerivedResultReleaseRequest, *, tensorfs_root: Path
) -> pb.DerivedResultReleaseResult:
    if (
        not tensorfs_root.is_absolute()
        or re.fullmatch(r"sha256:[0-9a-f]{64}", request.weights_transaction_id) is None
        or len(request.tensorfs_receipt_digest) != 32
    ):
        raise RetentionRefusal("derived result release requires exact identities")
    store = fill.store(tensorfs_root)
    # The transaction id is the committed result's identity (one commit per transaction),
    # so it is what release compares; re-encoding TensorFS's receipt would tie release to
    # one receipt format forever.
    if store.derived_lookup(request.weights_transaction_id).get("state") != "committed":
        raise RetentionRefusal("derived result is not committed")
    store.derived_dispose(request.weights_transaction_id)
    return pb.DerivedResultReleaseResult(
        weights_transaction_id=request.weights_transaction_id,
        tensorfs_receipt_digest=request.tensorfs_receipt_digest,
        released=True,
    )
