#!/usr/bin/env python3
"""Real RuntimePreparation -> TensorFS 0.0.5 source/replay/refusal proof.

The source directory is read-only. The proof hashes the exact supervisor-side files once,
then Runtime sends those immutable facts through its actual loopback gRPC host.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import threading
import time
from pathlib import Path

import grpc
import tensorfs

from cozy_runtime.internal.worker import model_source_prepare
from cozy_runtime.internal.worker.control import GrpcControlHost
from cozy_runtime.protocol import documents
from cozy_runtime.protocol import worker_pb2 as pb
from cozy_runtime.protocol import worker_pb2_grpc as pb_grpc

PROFILE = "hf/stabilityai/stable-diffusion-xl-base-1.0/diffusers-fp16"
SOURCE_URI = (
    "hf://stabilityai/stable-diffusion-xl-base-1.0@462165984030d82259a11f4367a4eed129e94a7b"
)
MEMBERS = (
    "text_encoder/model.fp16.safetensors",
    "text_encoder_2/model.fp16.safetensors",
    "unet/diffusion_pytorch_model.fp16.safetensors",
    "vae/diffusion_pytorch_model.fp16.safetensors",
)


def digest(path: Path) -> str:
    hashed = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 << 20):
            hashed.update(chunk)
    return "sha256:" + hashed.hexdigest()


def request(source: Path) -> pb.PrepareModelSourceRequest:
    files: list[pb.LocalModelSourceFile] = []
    identities: list[str] = []
    for member in MEMBERS:
        path = source / member
        object_id = digest(path)
        identities.append(f"{member}\0{object_id}\0{path.stat().st_size}")
        files.append(
            pb.LocalModelSourceFile(
                member=member,
                object_id=object_id,
                length=path.stat().st_size,
                path=str(path),
            )
        )
    selection = hashlib.sha256("\n".join(identities).encode()).digest()
    return pb.PrepareModelSourceRequest(
        operation_id="runtime-model-source-live",
        source_selection_digest=selection,
        profiles=[pb.ModelSourceProfile(slot="model", profile=PROFILE)],
        files=files,
        source_uri=SOURCE_URI,
        declared_license="openrail++",
    )


def await_address(path: Path) -> str:
    for _ in range(200):
        if path.is_file():
            return path.read_text().strip()
        time.sleep(0.05)
    raise RuntimeError("loopback RuntimePreparation address was not published")


def refused(result: pb.PrepareModelSourceResult, code: str) -> None:
    if (
        result.outcome != pb.MODEL_SOURCE_PREPARE_OUTCOME_REFUSED
        or result.safe_code != code
        or not result.safe_detail
        or result.sources
    ):
        raise RuntimeError(f"wrong refusal {result}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--store", required=True, type=Path)
    args = parser.parse_args()
    source = args.source.resolve(strict=True)
    store_root = args.store.resolve()
    store_root.parent.mkdir(parents=True, exist_ok=True)
    store = (
        tensorfs.Store.open(str(store_root))
        if store_root.exists()
        else tensorfs.Store.init(str(store_root))
    )

    address_file = store_root.parent / "runtime-preparation.addr"
    address_file.unlink(missing_ok=True)
    host = GrpcControlHost(
        "127.0.0.1:0",
        address_file,
        model_source_preparer=lambda value: model_source_prepare.prepare_model_source(
            value, artifact_store=store_root
        ),
    )
    thread = threading.Thread(target=host.serve, args=(object(),), daemon=True)
    thread.start()
    channel = grpc.insecure_channel(await_address(address_file))
    stub = pb_grpc.RuntimePreparationStub(channel)
    prepared_request = request(source)
    first = stub.PrepareModelSource(prepared_request, timeout=600)
    if first.outcome != pb.MODEL_SOURCE_PREPARE_OUTCOME_PREPARED or len(first.sources) != 1:
        raise RuntimeError(f"first preparation failed: {first}")
    replay = stub.PrepareModelSource(prepared_request, timeout=600)
    if (
        replay.outcome != pb.MODEL_SOURCE_PREPARE_OUTCOME_REPLAYED
        or replay.sources != first.sources
    ):
        raise RuntimeError("exact source replay changed")

    row = first.sources[0]
    evidence = json.loads(row.checkpoint_evidence_canonical_bytes)
    manifest_id = documents.spell(row.manifest.digest)
    manifest = store.manifest(manifest_id)
    if (
        row.slot != "model"
        or row.profile != PROFILE
        or row.manifest.length != len(manifest["manifest"])
        or evidence["classification_digest"] != tensorfs.topology_digest(manifest["header"])
        or evidence["ingest"]["source_uri"] != SOURCE_URI
        or evidence["ingest"]["declared_license"] != "openrail++"
        or evidence["verification"]["kind"] != "approved_producer"
    ):
        raise RuntimeError("prepared source Manifest/CheckpointEvidence binding is incomplete")

    changed = pb.PrepareModelSourceRequest()
    changed.CopyFrom(prepared_request)
    changed.declared_license = "other"
    refused(
        stub.PrepareModelSource(changed, timeout=30),
        "model_source_tensorfs_transaction_conflict",
    )
    unordered = pb.PrepareModelSourceRequest()
    unordered.CopyFrom(prepared_request)
    unordered.files.reverse()
    refused(stub.PrepareModelSource(unordered, timeout=30), "model_source_file_order")
    short = pb.PrepareModelSourceRequest()
    short.CopyFrom(prepared_request)
    short.files[0].length -= 1
    refused(stub.PrepareModelSource(short, timeout=30), "model_source_file_identity_invalid")

    link = store_root.parent / "source-link.safetensors"
    link.unlink(missing_ok=True)
    os.symlink(prepared_request.files[0].path, link)
    linked = pb.PrepareModelSourceRequest()
    linked.CopyFrom(prepared_request)
    linked.files[0].path = str(link)
    refused(stub.PrepareModelSource(linked, timeout=30), "model_source_file_identity_invalid")

    model_source_prepare.release_model_source(
        artifact_store=store_root, operation_id=prepared_request.operation_id
    )
    channel.close()
    host.stop()
    thread.join(timeout=5)
    if thread.is_alive():
        raise RuntimeError("loopback RuntimePreparation host did not stop")
    print(
        "MODEL SOURCE PASS "
        f"profile={PROFILE} manifest={manifest_id} files={len(MEMBERS)} "
        "prepared=1 replay=1 conflict=1 order=1 length=1 symlink=1 evidence=exact release=1"
    )


if __name__ == "__main__":
    main()
