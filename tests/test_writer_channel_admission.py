from __future__ import annotations

import hashlib
import socket
import threading
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import tensorfs
from tensorfs.derived import Derivation, Part, Target, Tensor

from cozy_runtime.internal import storage_admission as storage
from cozy_runtime.internal.weights_writer import WriterBroker
from test_storage_admission import operation
from weights_channel import open_output


@pytest.mark.parametrize("finish", ["part", "cancel", "producer_error"])
def test_native_channel_admission_releases_on_completion_cancel_or_producer_loss(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, finish: str
) -> None:
    store = tensorfs.Store.init(tmp_path / "store")
    spool = tmp_path / "spool"
    spool.mkdir()
    targets = {
        "model": {
            "drop": [],
            "add": {
                "w": {
                    "logical_dtype": "bf16",
                    "shape": [2],
                    "encoding": dict(tensorfs.seed_digests())["plain/1"],
                    "parts": {"value": {"dtype": "bf16", "shape": [2]}},
                }
            },
        }
    }
    order = [("model", "w")]
    fingerprint, transaction = "sha256:" + "71" * 32, "sha256:" + "72" * 32
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
    recorded: list[Mapping[str, object]] = []
    broker = WriterBroker(
        lambda: store,
        store_root=Path(store.root),
        checkpoint=lambda _a, _t, _b, facts: recorded.append(facts),
        receipt=lambda _a, _t, _b, facts: recorded.append(facts),
    )
    broker.authorize(attempt, transaction, 1, hashlib.sha256(declaration).digest(), "model")
    definition = Derivation(
        {},
        {
            "model": Target(
                add={
                    "w": Tensor(
                        "bf16",
                        (2,),
                        dict(tensorfs.seed_digests())["plain/1"],
                        {"value": Part("bf16", (2,))},
                    )
                }
            )
        },
        {},
        order,
    )
    writer = open_output(broker, attempt, transaction, definition)
    host, client = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    released, pulling, supply = threading.Event(), threading.Event(), threading.Event()
    seen: list[dict[str, Any]] = []
    owner_errors: list[BaseException] = []
    producer_errors: list[BaseException] = []

    def controller() -> None:
        try:
            rows, connection = operation(host)
            seen.extend(rows)
            with connection:
                connection.sendall(b'{"ok":true}\n')
                assert connection.recv(1) == b""
            released.set()
        except BaseException as exc:
            owner_errors.append(exc)

    class Source:
        sent = 0

        def read(self, size: int) -> bytes:
            pulling.set()
            assert supply.wait(5), "test failed to release its controlled producer"
            if finish == "producer_error":
                raise RuntimeError("producer stopped")
            count = min(size, 4 - self.sent)
            self.sent += count
            return bytes(count)

    def produce() -> None:
        try:
            writer.add_part("model", "w", "value", Source())
        except BaseException as exc:
            producer_errors.append(exc)

    with host, client:
        monkeypatch.setattr(storage, "_channel", client)
        owner = threading.Thread(target=controller, daemon=True)
        producer = threading.Thread(target=produce, daemon=True)
        owner.start()
        producer.start()
        try:
            assert pulling.wait(5), "native owner did not start its bounded pull"
            assert broker.payload_pending(attempt)
            assert not released.is_set()
            assert len(seen) == 1 and seen[0]["bytes"] == 4 + 4 * tensorfs.manifest_max_bytes()
            assert not list(spool.glob("*-input")) and not list(spool.glob("*-output"))
            if finish == "cancel":
                attempt.canceling = "operator"
                broker.close_attempt(attempt)
                assert released.wait(5), "cancel waited for the stalled producer"
            supply.set()
            producer.join(5)
            owner.join(5)
            assert not producer.is_alive() and not owner.is_alive()
            assert released.is_set() and not owner_errors
            assert bool(producer_errors) == (finish != "part")
            assert not broker.payload_pending(attempt)
        finally:
            supply.set()
            monkeypatch.setattr(storage, "_channel", None)
            broker.close_attempt(attempt)
        if finish == "part":
            assert store.derived_lookup(transaction)["state"] == "open"
