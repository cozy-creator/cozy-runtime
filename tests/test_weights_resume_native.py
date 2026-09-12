"""Real native custody, rather than surviving receipt metadata, governs replay."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest
import tensorfs

from cozy_runtime.author import WeightsPart, WeightsReceipt, WeightsTarget, WeightsTensor
from cozy_runtime.author._errors import CapabilityError
from cozy_runtime.author._weights import WeightsCommit
from cozy_runtime.internal.base_observation import Observation
from cozy_runtime.internal.hostfacts import HostFacts
from cozy_runtime.internal.weights_sink import WeightsTransactionHost, work_fingerprint


def _committed(
    tmp_path: Path,
) -> tuple[tensorfs.Store, WeightsTransactionHost, WeightsCommit, WeightsReceipt]:
    store = tensorfs.Store.init(tmp_path / "store")
    plain = next(digest for alias, digest in tensorfs.seed_digests() if alias == "plain/1")
    request = WeightsCommit(
        output_slot="result",
        sources={},
        targets={
            "model": WeightsTarget(
                add={
                    "weight": WeightsTensor(
                        logical_dtype="f32",
                        shape=(512,),
                        encoding=plain,
                        parts={"value": WeightsPart(dtype="f32", shape=(512,))},
                    )
                }
            )
        },
        configs={},
        order=(("model", "weight"),),
        max_new_bytes=2048,
    )
    host = WeightsTransactionHost(
        store=store,
        owner_scope="resume-proof",
        request_id="request",
        invocation_spec_digest="sha256:" + "41" * 32,
        work_fingerprint="sha256:" + "42" * 32,
        writer_session_id=1,
        allowed_sources={},
        output_bounds={"result": 2048},
    )
    writer = host.open(request)
    writer.add_part("model", "weight", "value", b"\x31" * 2048)
    receipt = writer.commit()
    return store, host, request, receipt


def test_pending_and_adopted_results_replay_but_disposed_metadata_refuses(tmp_path: Path) -> None:
    store, host, request, receipt = _committed(tmp_path)
    for disposition in ("pending", "adopted"):
        if disposition == "adopted":
            store.derived_adopt(receipt.weights_transaction_id, "owned-output")
        replay = host.open(request)
        assert replay.replayed
        assert replay.commit().tensorfs_receipt == receipt.tensorfs_receipt
        observed = store.derived_lookup(receipt.weights_transaction_id)
        assert observed["disposition"]["kind"] == disposition

    store.derived_dispose(receipt.weights_transaction_id)
    retained_metadata = store.derived_lookup(receipt.weights_transaction_id)
    assert retained_metadata["state"] == "committed"
    assert retained_metadata["receipt"] == json.loads(receipt.tensorfs_receipt)
    with pytest.raises(CapabilityError) as error:
        host.open(request)
    assert error.value.code == "weights_transaction_closed"


def test_pending_receipt_without_native_root_is_not_reusable(tmp_path: Path) -> None:
    store, host, request, receipt = _committed(tmp_path)
    # Corrupt only this fixture's private root. The metadata still says Pending and
    # all payload bytes remain resident; neither fact may replace native retention.
    root = (
        Path(store.root)
        / "roots/derived"
        / (receipt.weights_transaction_id.removeprefix("sha256:") + ".json")
    )
    root.unlink()
    assert store.derived_lookup(receipt.weights_transaction_id)["state"] == "committed"
    with pytest.raises(CapabilityError) as error:
        host.open(request)
    assert error.value.code == "weights_receipt_unavailable"


def test_work_identity_ignores_attempt_authority_but_binds_numerical_inputs() -> None:
    # Use the actual closed InvocationSpec fixture, which includes both a deadline
    # and publication grant. Hashing that whole document would bind those clocks.
    spec = json.loads(
        (
            Path(__file__).parent / "testdata/worker-protocol/canonical/invocation_spec_job.json"
        ).read_text()
    )
    spec["inputs"] = [
        {
            "input_id": "model:base",
            "digest": "sha256:" + "31" * 32,
            "length": 2048,
            "kind_mime": "application/x-cozytensors",
            "order": 0,
        }
    ]
    base = Observation(
        python_abi="cp312",
        python_full_version="3.12.12",
        os_arch="linux/amd64",
        libc="glibc2.28",
        accelerator_backend="cuda",
        accelerator_build="cu128",
        torch_release="2.8.0",
        cuda_version="12.8",
        uv_version="0.8.0",
        distributions=(
            ("cozy-runtime", "0.2.26"),
            ("tensorfs", "0.3.12"),
            ("torch", "2.8.0+cu128"),
            ("numpy", "2.3.0"),
        ),
        import_roots=(),
    )
    device = HostFacts(gpu_name="H200", gpu_sm=90, driver_version="580.1", backend="cuda")
    environment = "sha256:" + "32" * 32
    expected = work_fingerprint(spec, environment, base, device)
    replacement = deepcopy(spec)
    replacement["deadline_unix_ms"] += 1000
    replacement["job"]["publication_contract"]["grant_id"] = "new-owner-grant"
    # These are outside InvocationSpec in production, and must never become work inputs.
    replacement.update(
        attempt_ordinal=9,
        worker_boot_id="new-pod",
        owner_epoch=22,
        lease_id="new-lease",
        scratch="/new/path",
    )
    assert (
        work_fingerprint(
            replacement,
            environment,
            replace(
                base, uv_version="another-installer", import_roots=(("torch", ("/new/path",)),)
            ),
            replace(
                device,
                gpu_name="other H200",
                vram_total_bytes=1,
                host_ram_total_bytes=2,
                vcpu_count=128,
                platform="another host",
            ),
        )
        == expected
    )
    for field in ("build_id", "job_descriptor_id"):
        changed = deepcopy(spec)
        changed["job"][field] = "changed-producer"
        assert work_fingerprint(changed, environment, base, device) != expected
    changed = deepcopy(spec)
    changed["payload_digest"] = "sha256:" + "33" * 32
    assert work_fingerprint(changed, environment, base, device) != expected
    changed = deepcopy(spec)
    changed["inputs"][0]["digest"] = "sha256:" + "34" * 32
    assert work_fingerprint(changed, environment, base, device) != expected
    assert work_fingerprint(spec, "sha256:" + "35" * 32, base, device) != expected
    for name, version in base.distributions:
        distributions = tuple(
            (key, value if key != name else version + "+changed")
            for key, value in base.distributions
        )
        assert (
            work_fingerprint(spec, environment, replace(base, distributions=distributions), device)
            != expected
        )
    assert work_fingerprint(spec, environment, base, replace(device, gpu_sm=89)) != expected
    assert (
        work_fingerprint(spec, environment, base, replace(device, driver_version="other"))
        != expected
    )
