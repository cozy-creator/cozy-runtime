"""Release, collection and retention refuse only what retrying cannot change."""

from __future__ import annotations

from pathlib import Path

import pytest
import tensorfs

from cozy_runtime.internal import canonical
from cozy_runtime.internal.worker import derived_retention
from cozy_runtime.internal.worker.derived_retention import RetentionRefusal
from cozy_runtime.protocol import worker_pb2 as pb
from test_inline_model_configs import FA3_CONFIG, _lane_checkpoint


def test_release_names_the_transaction_not_the_receipt_encoding(tmp_path: Path) -> None:
    store, _manifest = _lane_checkpoint(tmp_path, None, canonical.write(FA3_CONFIG))
    transaction = tensorfs.object_id(b"inline-config-proof")
    request = pb.DerivedResultReleaseRequest(
        weights_transaction_id=transaction, tensorfs_receipt_digest=b"\x01" * 32
    )
    for _ in range(2):
        result = derived_retention.release_result(request, tensorfs_root=Path(store.root))
        assert result.released and result.weights_transaction_id == transaction
    assert store.derived_lookup(transaction)["disposition"]["kind"] == "released"
    absent = pb.DerivedResultReleaseRequest(
        weights_transaction_id=tensorfs.object_id(b"never"), tensorfs_receipt_digest=b"\x01" * 32
    )
    with pytest.raises(RetentionRefusal):
        derived_retention.release_result(absent, tensorfs_root=Path(store.root))
