"""Real native custody, rather than surviving receipt metadata, governs replay."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import tensorfs
from tensorfs.derived import Derivation, Part, Target, Tensor

from cozy_runtime.author import WeightsReceipt
from cozy_runtime.author._errors import CapabilityError
from cozy_runtime.internal.hostfacts import HostFacts
from cozy_runtime.internal.numerical_environment import fingerprint
from cozy_runtime.internal.weights_sink import receipt_from_native, same_receipt, work_fingerprint
from native_weights import NativeExecution


def _committed(
    tmp_path: Path,
) -> tuple[tensorfs.Store, NativeExecution, Derivation, WeightsReceipt]:
    store = tensorfs.Store.init(tmp_path / "store")
    plain = next(digest for alias, digest in tensorfs.seed_digests() if alias == "plain/1")
    request = Derivation(
        {},
        {
            "model": Target(
                add={"weight": Tensor("f32", (512,), plain, {"value": Part("f32", (512,))})}
            )
        },
        {},
        (("model", "weight"),),
    )
    host = NativeExecution(store, tmp_path, "request", {}, {"result": 2048})
    with host.client.open_output("result", request) as writer:
        writer.add_part("model", "weight", "value", b"\x31" * 2048)
        facts = writer.commit()
    receipt = receipt_from_native(
        "result", facts["transaction_id"], facts, replayed=False, request_id="request"
    )
    return store, host, request, receipt


def test_pending_and_adopted_results_replay_but_disposed_metadata_refuses(tmp_path: Path) -> None:
    store, host, request, receipt = _committed(tmp_path)
    for disposition in ("pending", "adopted"):
        if disposition == "adopted":
            store.derived_adopt(receipt.weights_transaction_id, "owned-output")
        replay = host.client.open_output("result", request)
        assert replay.receipt is not None
        assert replay.commit() == json.loads(receipt.tensorfs_receipt)
        observed = store.derived_lookup(receipt.weights_transaction_id)
        assert observed["disposition"]["kind"] == disposition

    store.derived_dispose(receipt.weights_transaction_id)
    retained_metadata = store.derived_lookup(receipt.weights_transaction_id)
    assert retained_metadata["state"] == "committed"
    assert same_receipt(retained_metadata["receipt"], json.loads(receipt.tensorfs_receipt))
    with pytest.raises(CapabilityError) as error:
        host.client.open_output("result", request)
    assert error.value.code == "weights_transaction_abandoned"


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
        host.client.open_output("result", request)
    assert error.value.code == "weights_receipt_unavailable"


def test_work_identity_ignores_attempt_authority_but_binds_numerical_inputs() -> None:
    spec: dict[str, Any] = {
        "job": {
            "installation_id": "first-installation",
            "job_descriptor_id": "first-job",
            "publication_contract": {"grant_id": "first-grant"},
        },
        "deadline_unix_ms": 1000,
        "payload_digest": "sha256:" + "30" * 32,
    }
    spec["inputs"] = [
        {
            "input_id": "model:base",
            "digest": "sha256:" + "31" * 32,
            "length": 2048,
            "kind_mime": "application/x-cozytensors",
            "order": 0,
        }
    ]
    device = HostFacts(gpu_name="H200", gpu_sm=90, driver_version="580.1", backend="cuda")
    numerical = fingerprint(device, threads=4, inherited={})
    implementation = "sha256:" + "42" * 32

    def work(
        value: dict[str, Any], policy: bytes = numerical, identity: str = implementation
    ) -> str:
        return work_fingerprint(value, policy, operation_identity=identity)

    expected = work(spec)
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
    assert work(replacement) == expected
    # A known callee, not its installing SDK or caller package, owns native work; without
    # one the job descriptor does. An installation id never does (a reinstall resumes).
    for field in ("installation_id", "job_descriptor_id"):
        changed = deepcopy(spec)
        changed["job"][field] = "changed-sdk-or-caller"
        assert work(changed) == expected
        assert (work(changed, identity="") != work(spec, identity="")) == (
            field == "job_descriptor_id"
        )
    changed = deepcopy(spec)
    changed["payload_digest"] = "sha256:" + "33" * 32
    assert work(changed) != expected
    changed = deepcopy(spec)
    changed["inputs"][0]["digest"] = "sha256:" + "34" * 32
    assert work(changed) != expected
    assert work(spec, identity="sha256:" + "43" * 32) != expected
    assert work(spec, fingerprint(replace(device, gpu_sm=89), threads=4, inherited={})) != expected
    updated = replace(device, driver_version="590.4")
    assert work(spec, fingerprint(updated, threads=4, inherited={})) == expected
    assert work(spec, fingerprint(device, threads=1, inherited={})) != expected


def _partial_then_resume(tmp_path: Path, first: HostFacts, second: HostFacts) -> dict[str, Any]:
    """Interrupt real native work under one device's policy, resume it under another."""
    store = tensorfs.Store.init(tmp_path / "store")
    plain = next(digest for alias, digest in tensorfs.seed_digests() if alias == "plain/1")
    request = Derivation(
        {},
        {
            "model": Target(
                add={"weight": Tensor("f32", (512,), plain, {"value": Part("f32", (512,))})}
            )
        },
        {},
        (("model", "weight"),),
    )
    for epoch, device in ((1, first), (2, second)):
        with NativeExecution(store, tmp_path, "resume", {}, {"result": 2048}, epoch=epoch) as host:
            host.attempt.weights_work_fingerprint = work_fingerprint(
                host.attempt.spec,
                fingerprint(device, threads=4, inherited={}),
                operation_identity="sha256:" + "42" * 32,
            )
            writer = host.client.open_output("result", request)
            writer.add_part("model", "weight", "value", b"\x31" * 2048)
            if epoch == 2:
                return dict(writer.commit())
    raise AssertionError("unreachable")


def test_interrupted_work_resumes_after_a_host_driver_update(tmp_path: Path) -> None:
    before = HostFacts(gpu_name="H200", gpu_sm=90, driver_version="580.1", backend="cuda")
    facts = _partial_then_resume(tmp_path, before, replace(before, driver_version="590.4"))
    assert facts["transaction_id"]


def test_interrupted_work_does_not_resume_on_another_architecture(tmp_path: Path) -> None:
    before = HostFacts(gpu_name="H200", gpu_sm=90, driver_version="580.1", backend="cuda")
    with pytest.raises(CapabilityError):
        _partial_then_resume(tmp_path, before, replace(before, gpu_sm=100))
