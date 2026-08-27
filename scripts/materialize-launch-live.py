#!/usr/bin/env python3
"""Narrow local contract proof for the fixed materialize/launch adapter."""

from __future__ import annotations

import base64
import hashlib
import json
import subprocess
import tempfile
import time
from pathlib import Path
from typing import TypedDict, cast

from cozy_runtime.internal import canonical, contract, fill, local, provisioned, pull, readiness
from cozy_runtime.internal.config import ConfigError, read_adapter_config
from cozy_runtime.internal.materialize_launch import (
    REQUIRED_DOCUMENT_ROLES,
    LaunchRefusal,
    _materialize_objects,
    _provision_pair,
)
from cozy_runtime.internal.worker.plan import PlanRefusal
from cozy_runtime.protocol import SCHEMA_DIGEST

ROOT = Path(__file__).resolve().parent.parent


class Ref(TypedDict):
    digest: str
    length: int


class ObjectSetVector(TypedDict):
    name: str
    document: dict[str, object]
    canonical: str
    digest: str


def ref(raw: bytes) -> Ref:
    return {"digest": "sha256:" + hashlib.sha256(raw).hexdigest(), "length": len(raw)}


def mint_certificate(root: Path, name: str, san: str) -> Path:
    certificate, key = root / f"{name}.crt", root / f"{name}.key"
    subprocess.run(
        [
            "openssl",
            "req",
            "-x509",
            "-newkey",
            "rsa:2048",
            "-nodes",
            "-keyout",
            str(key),
            "-out",
            str(certificate),
            "-days",
            "1",
            "-subj",
            "/CN=cozy-worker",
            "-addext",
            f"subjectAltName={san}",
        ],
        check=True,
        capture_output=True,
    )
    return certificate


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="cozy-materialize-launch-") as raw_root:
        root = Path(raw_root)
        vectors: dict[str, tuple[bytes, provisioned.ArtifactObjectSet]] = {}
        for line in (ROOT / "vectors" / "artifact-object-sets.jsonl").read_text().splitlines():
            vector = cast(ObjectSetVector, json.loads(line))
            raw = canonical.write(vector["document"])
            assert raw == vector["canonical"].encode()
            assert "sha256:" + hashlib.sha256(raw).hexdigest() == vector["digest"]
            vectors[vector["name"]] = (raw, provisioned.read_artifact_object_set(raw))
        empty_bytes, empty_object_set = vectors["empty"]
        object_set_ref = ref(empty_bytes)

        token = "1" * 64
        document = b"x"
        document_ref = ref(document)
        control_ref = {"digest": "sha256:" + "2" * 64, "length": 4096}
        spec = {
            "acquisition_attempt_id": "attempt-live",
            "acquisition_attempt_ordinal": 1,
            "artifact_object_set": object_set_ref,
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
            {
                **(object_set_ref if role == "artifact_object_set" else document_ref),
                "role": role,
                "url": f"https://delivery.invalid/{role}",
            }
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
            "COZY_ADAPTER_RENTER_TOKEN_SHA256_JSON": f'["{token}"]',
            "COZY_ADAPTER_READINESS_DEADLINE_UNIX": str(int(time.time()) + 60),
        }
        try:
            read_adapter_config({**env, "PYTHONPATH": "/injected/source"})
        except ConfigError as exc:
            assert exc.name == "adapter_config"
        else:
            raise AssertionError("PYTHONPATH source injection was accepted")
        try:
            read_adapter_config({**env, "COZY_ADAPTER_PROVIDER_POD_ID": "retired-provider-id"})
        except ConfigError as exc:
            assert exc.name == "adapter_config" and "unknown" in exc.message
        else:
            raise AssertionError("retired provider identity entered the adapter handoff")
        config = read_adapter_config(env)
        assert dict(config.child_base_env) == {"LANG": "C.UTF-8", "PATH": "/usr/bin"}
        parsed_spec, parsed_bundle = _provision_pair(spec_bytes, bundle_bytes, config)
        assert parsed_spec == spec and parsed_bundle == bundle

        assert empty_bytes == b'{"kind":"tensorhub.resolved_object_set/1","roots":[]}'
        assert empty_object_set.roots == () and empty_object_set.refs == {}
        local_weightless = local.weightless_release_documents(
            {
                "application": "weightless:app",
                "entrypoints": [
                    {"hidden": False, "kind": "entrypoint", "models": [], "name": "health"}
                ],
            }
        )
        assert local_weightless.artifact_object_set == empty_bytes
        assert ref(local_weightless.artifact_object_set) == {
            "digest": "sha256:d60755ff474871fce1e9705850245780895c1d2b798750f7692962c2d64d8856",
            "length": 53,
        }
        empty_store = _materialize_objects(
            {"capacity": {"cache_mount_path": str(root / "empty-cache")}},
            {"artifact_grants": []},
            empty_object_set,
            deadline=int(time.time()) + 60,
            store_root=root / "empty-local-store",
        )
        assert empty_store.store == root / "empty-local-store" and empty_store.store.is_dir()
        assert not (root / "empty-cache" / "tensorfs").exists()
        assert empty_store.sources == {
            "cloud": {"bytes": 0, "objects": 0},
            "local": {"bytes": 0, "objects": 0},
            "network_cache": {"bytes": 0, "objects": 0},
        }

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
        try:
            pull.open_guarded(
                "https://127.0.0.1/grant",
                pull.Deadline.after(1),
                no_proxy=True,
            )
        except pull.PullRefusal as exc:
            assert exc.code == "source_not_allowed"
        else:
            raise AssertionError("private small-artifact grant bypassed the shared guard")

        construction = contract.ModelConstructionContract(
            candidate=contract.Candidate(
                release="endpoint@release",
                application="endpoint:app",
                model_class="endpoint.models:Model",
                factory="sha256:" + "c" * 64,
                snapshot="sha256:" + snapshot,
                config_digest="sha256:" + "d" * 64,
                hardware_variant="cuda-h200",
            ),
            logical_tensors=(
                contract.LogicalTensor(
                    key="transformer.weight",
                    component="transformer",
                    shape=(16, 16),
                    kind="parameter",
                    module_class="Linear",
                    alias_group=0,
                    dtype=None,
                ),
            ),
            destination_sets=(
                contract.DestinationSet(
                    component="transformer",
                    module_class="Transformer",
                    training=False,
                    compiled=False,
                    keys=("transformer.weight",),
                ),
            ),
            derived=(
                contract.DerivedTensor(
                    key="transformer.rotary",
                    component="transformer",
                    shape=(8,),
                    dtype="float32",
                ),
            ),
            caches=(),
        )
        contract_bytes = canonical.write(construction.render())
        parsed_contract = contract.read_contract(contract_bytes)
        assert parsed_contract == construction
        assert contract.logical_model_digest(parsed_contract).startswith("sha256:")
        malformed_contract = dict(construction.render())
        malformed_contract["asserted_digest"] = "sha256:" + "e" * 64
        try:
            contract.read_contract(canonical.write(malformed_contract))
        except contract.ContractError as exc:
            assert exc.code == "contract_document_invalid"
        else:
            raise AssertionError("open ModelConstructionContract field set was accepted")

        binding_documents = root / "binding-documents"
        binding_documents.mkdir()
        config_bytes = b"{}"
        config_ref = ref(config_bytes)
        config_asset_bytes = b"tokenizer-asset"
        config_asset_ref = ref(config_asset_bytes)
        bound_contract = contract.ModelConstructionContract(
            candidate=contract.Candidate(
                release="endpoint@release",
                application="endpoint:app",
                model_class="endpoint.models:Model",
                factory="factory-identity",
                snapshot="sha256:" + snapshot,
                config_digest=str(config_ref["digest"]),
                hardware_variant="cuda-h200",
            ),
            logical_tensors=construction.logical_tensors,
            destination_sets=construction.destination_sets,
            derived=construction.derived,
            caches=(),
        )
        contract_bytes = canonical.write(bound_contract.render())
        contract_ref = ref(contract_bytes)
        descriptor_bytes = canonical.write(
            {
                "application": "endpoint:app",
                "entrypoints": [
                    {
                        "hidden": False,
                        "kind": "entrypoint",
                        "models": [
                            {
                                "class": "Model",
                                "component_use": {"run": ["transformer"]},
                                "param": "model",
                                "path": "generate.models.model",
                                "stamps": {},
                            }
                        ],
                        "name": "generate",
                    },
                    {
                        "hidden": False,
                        "kind": "entrypoint",
                        "models": [],
                        "name": "health",
                    },
                ],
            }
        )
        descriptor_ref = ref(descriptor_bytes)
        layout_ref: Ref = {"digest": "sha256:" + "e" * 64, "length": 23}
        fp8_weights_ref: Ref = {"digest": "sha256:" + "f" * 64, "length": 256}
        model_root_ref: Ref = {"digest": "sha256:" + snapshot, "length": 17}
        binding_bytes = canonical.write(
            {
                "checkpoint": {
                    "snapshot_root": model_root_ref,
                    "tensor_layout": layout_ref,
                },
                "config": {
                    "assets": [{"name": "tokenizer", "ref": config_asset_ref}],
                    "document": config_ref,
                },
                "execution_layout": [
                    {
                        "component": "transformer",
                        "snapshot_root": model_root_ref,
                        "tensor_layout": layout_ref,
                    }
                ],
                "format": "cozy.endpoint.ResolvedModelBinding/1",
                "model_construction_contract": contract_ref,
                "stamps": [],
            }
        )
        binding_ref = ref(binding_bytes)
        plan_bytes = canonical.write(
            {
                "bindings": [{"binding": binding_ref, "slot": "model"}],
                "descriptor": descriptor_ref,
                "entrypoint": "generate",
                "format": "cozy.endpoint.EntrypointBindingPlan/1",
            }
        )
        weightless_plan_bytes = canonical.write(
            {
                "bindings": [],
                "descriptor": descriptor_ref,
                "entrypoint": "health",
                "format": "cozy.endpoint.EntrypointBindingPlan/1",
            }
        )
        for artifact_ref, raw in (
            (config_ref, config_bytes),
            (contract_ref, contract_bytes),
            (descriptor_ref, descriptor_bytes),
            (binding_ref, binding_bytes),
        ):
            (binding_documents / f"{str(artifact_ref['digest'])[7:]}.json").write_bytes(raw)
        object_set_bytes = canonical.write(
            {
                "kind": "tensorhub.resolved_object_set/1",
                "roots": [
                    {
                        "objects": sorted(
                            [config_asset_ref, fp8_weights_ref, layout_ref],
                            key=lambda row: str(row["digest"]),
                        ),
                        "root": model_root_ref,
                    },
                ],
            }
        )
        object_set = provisioned.read_artifact_object_set(object_set_bytes)
        assert [root.root.digest for root in object_set.roots] == [model_root_ref["digest"]]
        legacy_role_bytes = canonical.write(
            {
                "kind": "tensorhub.resolved_object_set/1",
                "roots": [
                    {
                        "objects": [layout_ref],
                        "role": "endpoint",
                        "root": model_root_ref,
                    }
                ],
            }
        )
        try:
            provisioned.read_artifact_object_set(legacy_role_bytes)
        except PlanRefusal as exc:
            assert exc.code == "artifact_object_set.roots[0]_invalid"
        else:
            raise AssertionError("deleted ArtifactObjectSet role field was accepted")
        plan_ref = ref(plan_bytes)
        weightless_plan_ref = ref(weightless_plan_bytes)
        plan_path = root / "plan.json"
        weightless_plan_path = root / "weightless-plan.json"
        plan_path.write_bytes(plan_bytes)
        weightless_plan_path.write_bytes(weightless_plan_bytes)
        plan_documents = {
            str(plan_ref["digest"]): provisioned.ExactDocument(
                provisioned.ArtifactRef(str(plan_ref["digest"]), int(plan_ref["length"])),
                plan_path,
                plan_bytes,
            ),
            str(weightless_plan_ref["digest"]): provisioned.ExactDocument(
                provisioned.ArtifactRef(
                    str(weightless_plan_ref["digest"]), int(weightless_plan_ref["length"])
                ),
                weightless_plan_path,
                weightless_plan_bytes,
            ),
        }
        selected = provisioned.read_provisioned_release(
            plan_documents,
            read_document=provisioned.staged_document_reader(binding_documents),
            endpoint_release="endpoint@release",
            store_root=root / "store",
            object_set=object_set,
            dtype_widths={"bfloat16": 2, "float32": 4},
        )
        resolved_binding = selected.bindings[str(plan_ref["digest"])]
        assert resolved_binding.application == "endpoint:app"
        assert resolved_binding.components == ("transformer",)
        # The TensorFS payload is FP8-sized, while the allocator destinations are BF16
        # plus one derived f32 buffer. Stored/custody bytes never become the fill budget.
        assert int(fp8_weights_ref["length"]) < 16 * 16 * 2
        assert resolved_binding.model_weight_bytes == 16 * 16 * 2 + 8 * 4
        header_envelope = fill.price_envelope(
            [("transformer", 16 * 16 * 2, int(fp8_weights_ref["length"]), "decoded_float")],
            decode_scratch_bytes=64,
        )
        complete_envelope = fill.complete_device_envelope(
            header_envelope, resolved_binding.model_weight_bytes
        )
        assert complete_envelope["derived_destination_bytes"] == 8 * 4
        assert complete_envelope["required_bytes"] == 16 * 16 * 2 + 8 * 4 + 256
        assert resolved_binding.expected_logical_model_digest == contract.logical_model_digest(
            bound_contract
        )
        weightless_binding = selected.bindings[str(weightless_plan_ref["digest"])]
        assert weightless_binding.weightless and weightless_binding.model_weight_bytes == 0
        assert selected.config_assets == {
            str(config_asset_ref["digest"]): int(config_asset_ref["length"])
        }
        assert str(config_asset_ref["digest"]) not in selected.documents
        extra_root_ref: Ref = {"digest": "sha256:" + "c" * 64, "length": 31}
        extra_layout_ref: Ref = {"digest": "sha256:" + "d" * 64, "length": 37}
        extra_object_set = provisioned.read_artifact_object_set(
            canonical.write(
                {
                    "kind": "tensorhub.resolved_object_set/1",
                    "roots": [
                        {
                            "objects": sorted(
                                [config_asset_ref, fp8_weights_ref, layout_ref],
                                key=lambda row: str(row["digest"]),
                            ),
                            "root": model_root_ref,
                        },
                        {"objects": [extra_layout_ref], "root": extra_root_ref},
                    ],
                }
            )
        )
        try:
            provisioned.read_provisioned_release(
                plan_documents,
                read_document=provisioned.staged_document_reader(binding_documents),
                endpoint_release="endpoint@release",
                store_root=root / "store",
                object_set=extra_object_set,
                dtype_widths={"bfloat16": 2, "float32": 4},
            )
        except PlanRefusal as exc:
            assert exc.code == "artifact_object_set_mismatch"
        else:
            raise AssertionError("an unconsumed model root was accepted")
        try:
            provisioned.read_provisioned_release(
                {str(plan_ref["digest"]): plan_documents[str(plan_ref["digest"])]},
                read_document=provisioned.staged_document_reader(binding_documents),
                endpoint_release="endpoint@release",
                store_root=root / "store",
                object_set=object_set,
                dtype_widths={"bfloat16": 2, "float32": 4},
            )
        except PlanRefusal as exc:
            assert exc.code == "binding_plan_set_invalid"
        else:
            raise AssertionError("incomplete visible-entrypoint plan set was accepted")

        boot_raw = bytes(range(32))
        boot_id = base64.urlsafe_b64encode(boot_raw).rstrip(b"=").decode()
        config.pod_boot_id_path.write_text(boot_id + "\n")
        assert readiness.pod_boot_id(config.pod_boot_id_path) == boot_id
        fixed_certificate = mint_certificate(root, "fixed-worker", "DNS:cozy-worker")
        assert readiness.certificate(fixed_certificate)
        ip_certificate = mint_certificate(root, "provider-ip", "IP:127.0.0.1")
        try:
            readiness.certificate(ip_certificate)
        except LaunchRefusal as exc:
            assert exc.code == "tls_certificate_invalid"
        else:
            raise AssertionError("provider IP SAN was accepted as worker identity")
        payload = readiness.receipt_payload(
            spec=spec,
            provision_spec_digest=str(spec_ref["digest"]),
            provision_bundle_digest=str(bundle_ref["digest"]),
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
            artifact_sources={
                "cloud": {"bytes": 0, "objects": 0},
                "local": {"bytes": 36, "objects": 2},
                "network_cache": {"bytes": 0, "objects": 0},
            },
            network_cache_degradations=(),
        )
        receipt = canonical.parse_canonical(payload)
        assert isinstance(receipt, dict)
        assert receipt["worker_listener_bound"] is True
        assert receipt["media_foreign_credential_refused"] is True
        assert receipt["artifact_sources"]["local"] == {"bytes": 36, "objects": 2}
        assert receipt["acquisition_attempt_id"] == "attempt-live"
        assert "provider_pod_id" not in receipt
        readiness.publish(config.receipt_payload_path, payload)
        assert config.receipt_payload_path.read_bytes() == payload
        readiness.publish(config.receipt_payload_path, payload)
        assert config.receipt_payload_path.read_bytes() == payload
        try:
            readiness.publish(config.receipt_payload_path, payload + b"\n")
        except LaunchRefusal as exc:
            assert exc.code == "readiness_receipt_conflict"
        else:
            raise AssertionError("different readiness receipt overwrote the published bytes")
    print("materialize-launch contract: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
