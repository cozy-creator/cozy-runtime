"""Cache keys, receipts, ledgers and refusals that survive Runtime updates on one pod."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from cozy_runtime.internal import (
    canonical,
    readiness,
    source_interfaces,
)
from cozy_runtime.internal.config import Credentials, RuntimeConfig
from cozy_runtime.internal.refusal import LaunchRefusal
from cozy_runtime.internal.worker.child import ExecutorProtocolMismatch, require_executor_protocol
from cozy_runtime.internal.worker.control import InMemoryControlHost
from cozy_runtime.internal.worker.ledger import Ledger
from cozy_runtime.internal.worker.session import (
    REPREPARE,
    Worker,
    WorkerOptions,
    _same_parent,
    read_placement_set,
)
from cozy_runtime.internal.worker.source_calls import computation_key
from cozy_runtime.protocol import documents
from cozy_runtime.protocol import worker_pb2 as pb

SOURCE = {"source": {"digest": "sha256:" + "a" * 64, "length": 1}}


def test_native_memo_key_is_operation_version_output_format_and_inputs_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Pinned: a Runtime or TensorFS release that keeps output meaning and format keeps this key.
    key = computation_key("download_huggingface", SOURCE)
    assert key.hex() == "2739c9b1e6771dee3872314aee8548064ad7736c6f0a9aeacfb3ca4fb6da363b"
    assert computation_key("download_huggingface", SOURCE, "cozytensors/1") == key
    assert computation_key("download_huggingface", SOURCE, "cozytensors/2") != key
    monkeypatch.setitem(source_interfaces.OPERATION_VERSIONS, "download_huggingface", 2)
    assert computation_key("download_huggingface", SOURCE) != key


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


def test_orchestration_parent_is_its_prepared_job_not_its_policy() -> None:
    running = pb.JobDirective(
        installation_id="install-a", job_descriptor_id="sha256:" + "c" * 64, orchestration=True
    )
    changed = pb.JobDirective()
    changed.CopyFrom(running)
    changed.publication_contract.grant_id = "owner/_job-a"
    changed.resource_caps.max_rss_bytes = 1 << 30
    assert _same_parent(running, changed)
    changed.job_descriptor_id = "sha256:" + "d" * 64
    assert not _same_parent(running, changed)


def test_stale_placement_shape_asks_its_owner_to_prepare_again(tmp_path: Path) -> None:
    stale = canonical.write(
        {
            "format": "cozy.worker.v1.PlacementSet/1",
            "placements": [
                {
                    "placement_id": "package-old",
                    "package": {"package": "test/old", "release": "1.0"},
                    "package_interface": {"digest": "sha256:" + "e" * 64, "length": 10},
                    "bindings_digest": "sha256:" + "f" * 64,
                    "entrypoints": [],
                }
            ],
        }
    )
    desired = pb.DesiredPlacementSet(
        placement_set_digest=documents.digest_of(stale), placement_set_canonical_bytes=stale
    )
    with pytest.raises(documents.DocumentError, match=REPREPARE):
        read_placement_set(desired)
    worker = Worker(
        RuntimeConfig(cozy_home=tmp_path / "home", credentials=Credentials()),
        WorkerOptions(root=tmp_path / "worker", tensorfs_root=tmp_path / "store", worker_id="w"),
        InMemoryControlHost(),
    )
    try:
        worker._apply_desired_state(pb.DesiredWorkerState(revision=3, placement_set=desired))
        (fault,) = [row for row in worker.engine.faults if row.reason == REPREPARE]
        assert fault.kind == pb.FAULT_KIND_ARTIFACT_FETCH_FAILED
        assert fault.desired_state_revision == 3
        assert not worker.latched
    finally:
        worker.shutdown()
