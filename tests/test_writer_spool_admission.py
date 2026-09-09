from __future__ import annotations

import hashlib
import socket
import threading
from collections.abc import Callable, Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import NoReturn

import pytest
import tensorfs

from cozy_runtime.internal import storage_admission as storage
from cozy_runtime.internal.weights_writer import StoreProxy, WriterBroker, _path
from test_storage_admission import operation


@pytest.mark.parametrize("finish", ["part", "cancel", "spool_error"])
def test_spool_admission_survives_rpc_thread_changes_and_releases_on_finish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, finish: str
) -> None:
    store = tensorfs.Store.init(tmp_path / "store")
    spool = tmp_path / "spool"
    spool.mkdir()
    encoding = next(d for a, d in tensorfs.seed_digests() if a == "plain/1")
    targets = {
        "model": {
            "drop": [],
            "add": {
                "w": {
                    "logical_dtype": "bf16",
                    "shape": [2],
                    "encoding": encoding,
                    "parts": {"value": {"dtype": "bf16", "shape": [2]}},
                }
            },
        }
    }
    order, fingerprint, transaction = [("model", "w")], "sha256:" + "71" * 32, "sha256:" + "72" * 32
    declaration = store.derived_declaration({}, targets, {}, order, 4, work_fingerprint=fingerprint)
    attempt = SimpleNamespace(
        request_id="r",
        attempt=1,
        state="running",
        canceling="",
        spool=spool,
        weights_work_fingerprint=fingerprint,
        spec={"inputs": [], "outputs": [{"output_id": "model", "max_bytes": 4}]},
    )
    broker = WriterBroker(lambda: store, store_root=tmp_path / "store")
    broker.authorize(attempt, transaction, 1, hashlib.sha256(declaration).digest(), "model")
    proxy = StoreProxy(store, spool, lambda _kind, frame: broker.handle(attempt, frame))
    writer = proxy.begin_derived(
        transaction, 1, {}, targets, {}, order, 4, work_fingerprint=fingerprint
    )
    host, client = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    released = threading.Event()
    seen, failures = [], []

    def controller() -> None:
        try:
            rows, conn = operation(host)
            seen.extend(rows)
            with conn:
                conn.sendall(b'{"ok":true}\n')
                assert conn.recv(1) == b""
            released.set()
        except BaseException as exc:
            failures.append(exc)

    def on_thread(call: Callable[[], None]) -> None:
        def execute() -> None:
            try:
                call()
            except BaseException as exc:
                failures.append(exc)

        thread = threading.Thread(target=execute, daemon=True)
        thread.start()
        thread.join(3)
        assert not thread.is_alive(), (
            "broker held its lock while awaiting another writer's admission"
        )
        assert not failures

    with host, client:
        monkeypatch.setattr(storage, "_channel", client)
        server = threading.Thread(target=controller, daemon=True)
        server.start()
        fields = {
            "weights_transaction_id": transaction,
            "writer_epoch": 1,
            "component": "model",
            "key": "w",
            "role": "value",
        }
        if finish == "spool_error":

            class BrokenSource:
                def read(self, _size: int) -> NoReturn:
                    raise RuntimeError("input source stopped")

            with pytest.raises(RuntimeError, match="input source stopped"):
                writer.add_part("model", "w", "value", BrokenSource())
        else:

            def bound() -> None:
                assert broker.handle(
                    attempt, {**fields, "operation": "bound", "target_operation": "part"}
                ) == {"ok": True, "length": 4}

            on_thread(bound)
            assert not released.is_set()
            assert (
                broker.handle(
                    attempt, {**fields, "operation": "bound", "target_operation": "part"}
                )["code"]
                == "weights_writer_payload_pending"
            )
            assert not released.is_set(), "overlapping bound released the first writer"
            if finish == "part":
                _path(spool, transaction, "input").write_bytes(b"\0" * 4)
                on_thread(
                    lambda: assert_part(
                        broker.handle(attempt, {**fields, "operation": "part", "length": 4})
                    )
                )
            else:
                on_thread(lambda: broker.close_attempt(attempt))
        assert released.wait(3), "writer completion/cancel left disk admission held"
        server.join(3)
        assert not failures and not broker.pending
        assert len(seen) == 2  # Native body plus actual spool/scratch copy.
        assert seen[1]["bytes"] >= 4
        monkeypatch.setattr(storage, "_channel", None)
        if finish == "part":
            writer.commit()
        broker.close_attempt(attempt)


def assert_part(result: Mapping[str, object]) -> None:
    assert result == {"ok": True}
