"""Release, collection and retention refuse only what retrying cannot change."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import grpc
import pytest
import tensorfs
from tensorfs import errors

from cozy_runtime.internal import canonical
from cozy_runtime.internal.worker import derived_retention
from cozy_runtime.internal.worker.control import _PreparationServicer
from cozy_runtime.internal.worker.derived_retention import RetentionRefusal
from cozy_runtime.protocol import worker_pb2 as pb
from cozy_runtime.protocol import worker_pb2_grpc as rpc
from test_inline_model_configs import FA3_CONFIG, _lane_checkpoint


@contextmanager
def _client(**handlers: Any) -> Iterator[rpc.RuntimePreparationStub]:
    server = grpc.server(ThreadPoolExecutor(max_workers=2))
    rpc.add_RuntimePreparationServicer_to_server(
        _PreparationServicer(None, None, None, None, **handlers), server
    )
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()
    try:
        with grpc.insecure_channel(f"127.0.0.1:{port}") as channel:
            yield rpc.RuntimePreparationStub(channel)
    finally:
        server.stop(None).wait()


def _raising(exc: BaseException) -> Callable[..., Any]:
    def handler(*_args: Any) -> Any:
        raise exc

    return handler


@pytest.mark.parametrize(
    ("exc", "code"),
    [
        (errors.StoreBusy("busy"), grpc.StatusCode.UNAVAILABLE),
        (OSError("disk hiccup"), grpc.StatusCode.UNAVAILABLE),
        (RetentionRefusal("names another result"), grpc.StatusCode.FAILED_PRECONDITION),
        (errors.CODES["ROOT_ABSENT"]("no such transaction"), grpc.StatusCode.FAILED_PRECONDITION),
    ],
)
def test_release_and_collection_are_retryable_unless_retry_cannot_help(
    exc: BaseException, code: grpc.StatusCode
) -> None:
    handler = _raising(exc)
    with _client(
        derived_result_releaser=handler, store_collector=handler, derived_retainer=handler
    ) as client:
        calls: list[Callable[[], Any]] = [
            lambda: client.ReleaseDerivedResult(pb.DerivedResultReleaseRequest(), timeout=5),
            lambda: client.RetainDerivedResult(pb.DerivedRetentionRequest(), timeout=5),
        ]
        for call in calls:
            with pytest.raises(grpc.RpcError) as failure:
                call()
            assert failure.value.code() == code
        with pytest.raises(grpc.RpcError) as failure:
            client.CollectStoreGarbage(pb.CollectStoreGarbageRequest(), timeout=5)
        want = code if isinstance(exc, errors.Refusal) else grpc.StatusCode.UNAVAILABLE
        assert failure.value.code() == want


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
