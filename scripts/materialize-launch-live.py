#!/usr/bin/env python3
"""Narrow local contract proof for the fixed materialize/launch adapter."""

from __future__ import annotations

import base64
import hashlib
import tempfile
import time
from pathlib import Path

from cozy_runtime.internal import canonical, pull, readiness
from cozy_runtime.internal.config import ConfigError, read_adapter_config
from cozy_runtime.internal.materialize_launch import (
    REQUIRED_DOCUMENT_ROLES,
    LaunchRefusal,
    _provision_pair,
)
from cozy_runtime.protocol import SCHEMA_DIGEST


def ref(raw: bytes) -> dict[str, object]:
    return {"digest": "sha256:" + hashlib.sha256(raw).hexdigest(), "length": len(raw)}


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="cozy-materialize-launch-") as raw_root:
        root = Path(raw_root)
        token = "1" * 64
        document = b"x"
        document_ref = ref(document)
        control_ref = {"digest": "sha256:" + "2" * 64, "length": 4096}
        spec = {
            "acquisition_attempt_id": "attempt-live",
            "acquisition_attempt_ordinal": 1,
            "artifact_object_set": document_ref,
            "capacity": {
                "accelerator_count": 1,
                "accelerator_model": "NVIDIA H200",
                "cache_mount_path": "/cache",
                "cache_volume_id": "cache-live",
                "datacenter": "US",
                "minimum_local_disk_bytes": 1 << 30,
                "provider": "local",
                "provider_offer_key": "offer-live",
                "provider_sku": "h200-sxm",
                "provider_volume_id": "volume-live",
            },
            "execution_substrate": {
                "control_runtime": control_ref,
                "oci_image": "cozy/substrate@sha256:" + "3" * 64,
                "wheelhouse_manifest_digest": "sha256:" + "4" * 64,
            },
            "format": "tensorhub.rental_provision_spec/1",
            "initial_placement_set": document_ref,
            "network": {
                "media_port": 43101,
                "media_protocol": "https",
                "worker_port": 43100,
                "worker_protocol": "cozy.worker.v1",
                "worker_wire_schema_digest": SCHEMA_DIGEST,
            },
            "renter_token_sha256": [token],
            "rental_id": "rental-live",
        }
        spec_bytes = canonical.write(spec)
        spec_ref = ref(spec_bytes)
        documents = [
            {**document_ref, "role": role, "url": f"https://delivery.invalid/{role}"}
            for role in REQUIRED_DOCUMENT_ROLES
        ]
        bundle = {
            "artifact_grants": [],
            "control_runtime": control_ref,
            "documents": documents,
            "format": "tensorhub.rental_provision_bundle/1",
            "provision_spec": spec_ref,
        }
        bundle_bytes = canonical.write(bundle)
        bundle_ref = ref(bundle_bytes)
        env = {
            "AWS_SECRET_ACCESS_KEY": "must-not-cross",
            "HF_TOKEN": "must-not-cross",
            "LANG": "C.UTF-8",
            "PATH": "/usr/bin",
            "RUNPOD_API_KEY": "must-not-cross",
            "COZY_ADAPTER_PROVISION_SPEC_PATH": str(root / "spec.json"),
            "COZY_ADAPTER_PROVISION_SPEC_DIGEST": str(spec_ref["digest"]),
            "COZY_ADAPTER_PROVISION_BUNDLE_PATH": str(root / "bundle.json"),
            "COZY_ADAPTER_PROVISION_BUNDLE_DIGEST": str(bundle_ref["digest"]),
            "COZY_ADAPTER_POD_BOOT_ID_PATH": str(root / "boot-id"),
            "COZY_ADAPTER_TLS_CERT_PATH": str(root / "tls.crt"),
            "COZY_ADAPTER_TLS_KEY_PATH": str(root / "tls.key"),
            "COZY_ADAPTER_RECEIPT_PAYLOAD_PATH": str(root / "receipt.json"),
            "COZY_ADAPTER_WORKER_INTERNAL_PORT": "43100",
            "COZY_ADAPTER_MEDIA_INTERNAL_PORT": "43101",
            "COZY_ADAPTER_PROVIDER_POD_ID": "pod-live",
            "COZY_ADAPTER_RENTER_TOKEN_SHA256_JSON": f'["{token}"]',
            "COZY_ADAPTER_READINESS_DEADLINE_UNIX": str(int(time.time()) + 60),
        }
        try:
            read_adapter_config({**env, "PYTHONPATH": "/injected/source"})
        except ConfigError as exc:
            assert exc.name == "adapter_config"
        else:
            raise AssertionError("PYTHONPATH source injection was accepted")
        config = read_adapter_config(env)
        assert dict(config.child_base_env) == {"LANG": "C.UTF-8", "PATH": "/usr/bin"}
        parsed_spec, parsed_bundle = _provision_pair(spec_bytes, bundle_bytes, config)
        assert parsed_spec == spec and parsed_bundle == bundle

        snapshot = "a" * 64

        class FakeStore:
            def __init__(self, path: Path) -> None:
                self.root = path

        class FakeTensorFS:
            def __init__(self) -> None:
                self.completed = False

            def fetch_plan(
                self,
                _store: FakeStore,
                _session: str,
                root_digest: str,
                root_length: int,
                declared: list[tuple[str, int]],
            ) -> dict[str, object]:
                assert root_digest == snapshot and root_length == 17
                assert declared == [("sha256:" + "b" * 64, 19)]
                return {
                    "document": b"tensorfs-plan",
                    "held": [("sha256:" + snapshot, 17), *declared],
                    "held_bytes": 36,
                    "presence": {},
                    "wanted": [],
                    "wanted_bytes": 0,
                }

            def fetch_complete(self, _store: FakeStore, document: bytes) -> None:
                assert document == b"tensorfs-plan"
                self.completed = True

        tensorfs = FakeTensorFS()
        outcome = pull.pull_granted_snapshot(
            tensorfs,
            FakeStore(root / "store"),
            "sha256:" + snapshot,
            17,
            [
                pull.Read("sha256:" + snapshot, 17, "https://delivery.invalid/root"),
                pull.Read("sha256:" + "b" * 64, 19, "https://delivery.invalid/object"),
            ],
        )
        assert tensorfs.completed and outcome.held == 2 and outcome.fetched == 0
        try:
            pull.pull_granted_snapshot(
                tensorfs,
                FakeStore(root / "store"),
                snapshot,
                17,
                [pull.Read("sha256:" + snapshot, 17, "http://delivery.invalid/root")],
            )
        except pull.PullRefusal as exc:
            assert exc.code == "artifact_incomplete"
        else:
            raise AssertionError("plaintext direct object grant was accepted")

        boot_raw = bytes(range(32))
        boot_id = base64.urlsafe_b64encode(boot_raw).rstrip(b"=").decode()
        config.pod_boot_id_path.write_text(boot_id + "\n")
        assert readiness.pod_boot_id(config.pod_boot_id_path) == boot_id
        payload = readiness.receipt_payload(
            spec=spec,
            provision_spec_digest=str(spec_ref["digest"]),
            provision_bundle_digest=str(bundle_ref["digest"]),
            provider_pod_id="pod-live",
            pod_boot_id_value=boot_id,
            certificate_der=b"certificate-der",
            installed_receipt_digest="sha256:" + "5" * 64,
            placement_set_digest=str(document_ref["digest"]),
            gpus=[
                {
                    "device_index": 0,
                    "device_name": "NVIDIA H200",
                    "device_uuid": "GPU-live",
                    "driver_version": "999.0",
                    "memory_bytes": 141 << 30,
                    "pci_bus_id": "00000000:01:00.0",
                }
            ],
        )
        receipt = canonical.parse_canonical(payload)
        assert isinstance(receipt, dict)
        assert receipt["worker_listener_bound"] is True
        assert receipt["media_foreign_credential_refused"] is True
        readiness.publish(config.receipt_payload_path, payload)
        assert config.receipt_payload_path.read_bytes() == payload
        try:
            readiness.publish(config.receipt_payload_path, payload)
        except LaunchRefusal as exc:
            assert exc.code == "readiness_receipt_conflict"
        else:
            raise AssertionError("readiness receipt was overwritten")
    print("materialize-launch contract: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
