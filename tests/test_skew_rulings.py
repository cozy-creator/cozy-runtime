"""Cache keys, receipts, ledgers and refusals that survive Runtime updates on one pod."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from cozy_runtime.internal import canonical, readiness
from cozy_runtime.internal.refusal import LaunchRefusal
from cozy_runtime.internal.worker.child import ExecutorProtocolMismatch, require_executor_protocol
from cozy_runtime.internal.worker.ledger import Ledger


def _receipt(boot: str = "boot-1", port: int = 7000, **extra: object) -> bytes:
    payload = readiness.receipt_payload(
        control_public_key_ed25519_b64url="key",
        media_token_sha256=["sha256:" + "b" * 64],
        pod_boot_id_value=boot,
        certificate_der=b"certificate",
        worker_internal_port=port,
        media_internal_port=port + 1,
        gpus=[
            {
                "device_index": 0,
                "device_name": "H100",
                "device_uuid": "GPU-1",
                "driver_version": "590",
                "memory_bytes": 1 << 30,
                "pci_bus_id": "0000:01:00.0",
            }
        ],
    )
    return canonical.write({**json.loads(payload), **extra}) if extra else payload


def test_restarted_worker_republishes_readiness_when_attested_facts_agree(
    tmp_path: Path,
) -> None:
    path = tmp_path / "readiness-payload"
    readiness.publish(path, _receipt())
    newer = _receipt(port=7100, runtime_extra="a newer Runtime field")
    readiness.publish(path, newer)
    assert path.read_bytes() == newer
    with pytest.raises(LaunchRefusal, match="pod_boot_id"):
        readiness.publish(path, _receipt(boot="boot-2"))
    assert path.read_bytes() == newer


def test_ledger_releasing_more_than_it_took_moves_the_baseline_not_a_leak() -> None:
    ledger = Ledger(worker_pid=os.getpid())
    ledger.device_total = 80 << 30
    ledger.device_baseline = ledger.device_allocated = 10 << 30
    before = ledger.snapshot()
    ledger.device_allocated = 9 << 30
    record = ledger.close_attempt("released", before)
    assert not record["leaks"] and not ledger.unreconciled
    assert ledger.device_baseline == 9 << 30
    before = ledger.snapshot()
    ledger.device_allocated = (9 << 30) + 1
    assert ledger.close_attempt("grew", before)["leaks"] and ledger.unreconciled


@pytest.mark.parametrize(
    ("hello", "remedy"),
    [
        ({"runtime_version": "0.18.33"}, "publish the package against cozy-runtime>=0.18.51"),
        (
            {"runtime_version": "0.18.49", "executor_protocol_revision": 1},
            "publish the package against cozy-runtime>=0.18.51",
        ),
        ({"executor_protocol_revision": 2}, "cozy rental update <rental>"),
    ],
)
def test_executor_protocol_refusal_names_the_side_to_update(
    hello: dict[str, object], remedy: str
) -> None:
    with pytest.raises(ExecutorProtocolMismatch, match=remedy):
        require_executor_protocol(hello)
