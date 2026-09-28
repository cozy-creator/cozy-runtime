"""Real worker capture qualification over the ordinary local serving path."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import grpc
from cozy_eval import capture as reader  # type: ignore[import-untyped]

from cozy_runtime.author import ActivationCapture
from cozy_runtime.internal import package_interface
from cozy_runtime.internal.config import read_config
from cozy_runtime.internal.discovery import discover
from cozy_runtime.internal.local import (
    LocalRecordOwner,
    LocalRequest,
    run_slice,
)
from cozy_runtime.internal.worker.session import Worker, WorkerOptions
from cozy_runtime.internal.worker.workspace_rpc import WorkspaceRPC
from cozy_runtime.protocol import WIRE_MINOR, documents
from cozy_runtime.protocol import worker_pb2 as pb
from cozy_runtime.protocol import worker_pb2_grpc as rpc


class Handler(WorkspaceRPC, rpc.RuntimePreparationServicer):  # type: ignore[misc] # generated gRPC base
    pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--capture", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    project = Path(__file__).parent
    root.mkdir(parents=True, exist_ok=True)
    os.chdir(project)
    found = discover(project)
    interface = package_interface.build(found)
    evidence: dict[str, Any] = {}
    errors: list[str] = []
    monitor: list[threading.Thread] = []
    servers: list[tuple[Any, ThreadPoolExecutor]] = []
    key = "captured" if args.capture else "plain"
    fixture_token = "owned-capture-qualification"
    config = replace(read_config(), cozy_home=root / "home", bootstrap_credential=fixture_token)

    def hook(owner: LocalRecordOwner, worker: Worker) -> None:
        # The ordinary in-memory owner has no external socket or credential. This proof
        # additionally exercises existing authenticated workspace RPCs on loopback.
        initial = owner.host.inbound.get_nowait()
        assert initial is not None
        assert initial.WhichOneof("msg") == "claim"
        initial.claim.proof = fixture_token.encode()
        owner.host.deliver(initial)
        handler = Handler()
        handler.workspace_service = worker.workspace_service
        pool = ThreadPoolExecutor(max_workers=2)
        server = grpc.server(pool)
        rpc.add_RuntimePreparationServicer_to_server(handler, server)
        port = server.add_insecure_port("127.0.0.1:0")
        server.start()
        servers.append((server, pool))

        def collect() -> None:
            try:
                while owner.terminal is None and not owner.done.is_set():
                    time.sleep(0.02)
                terminal = owner.terminal
                assert terminal is not None, owner.error
                body = documents.read(terminal.outcome_canonical_bytes, pb.AttemptOutcomeBody)
                assert body["status"] == pb.OUTCOME_STATUS_SUCCEEDED, body
                evidence["outcome"] = body
                observation = body["observation"]
                environment = observation["environment"]
                assert environment["runtime_version"] == "0.8.0", environment
                assert environment["accelerator"] and environment["worker_boot_id"] == owner.boot_id
                assert not environment.get("kernel_symbol")
                if args.capture:
                    output_doc = next(
                        row
                        for row in body["output_manifest"]["outputs"]
                        if row["output_id"] == "runtime.capture"
                    )
                    native = output_doc["native_tree"]
                    source = pb.NativeByteTreeRef(
                        producer_root_id=native["producer_root_id"],
                        receipt_digest=documents.raw(native["receipt_digest"]),
                        manifest=pb.Ref(
                            digest=documents.raw(native["manifest"]["digest"]),
                            length=native["manifest"]["length"],
                        ),
                        content_bytes=native["content_bytes"],
                    )
                    assert observation["capture"]["output_id"] == "runtime.capture"
                    hold = pb.NativeByteRetentionRequest(
                        source=source,
                        retention_id="sha256:"
                        + hashlib.sha256(b"capture-proof-recipient").hexdigest(),
                    )
                    claim = pb.Claim(
                        record_owner_id="local-run",
                        record_owner_epoch=owner.record_owner_epoch,
                        wire_minor=WIRE_MINOR,
                        proof=fixture_token.encode(),
                    )
                    with grpc.insecure_channel(f"127.0.0.1:{port}") as channel:
                        client = rpc.RuntimePreparationStub(channel)
                        received = client.WorkspaceRetainByteTree(
                            pb.NativeByteRetentionCall(claim=claim, request=hold)
                        )
                        assert received.source == source and not received.released
                        manifest_request = pb.NativeByteReadCall(
                            claim=claim, source=hold, object=source.manifest
                        )
                        manifest = b"".join(
                            chunk.data
                            for chunk in client.WorkspaceReadByteTreeObject(manifest_request)
                        )
                        assert hashlib.sha256(manifest).digest() == source.manifest.digest
                        entries = json.loads(manifest)["entries"]
                        destination = root / "capture-readback"
                        destination.mkdir()
                        for entry in entries:
                            blob = entry["blob"]
                            request = pb.NativeByteReadCall(
                                claim=claim,
                                source=hold,
                                object=pb.Ref(
                                    digest=bytes.fromhex(blob["sha256"]), length=blob["length"]
                                ),
                            )
                            data = b"".join(
                                chunk.data for chunk in client.WorkspaceReadByteTreeObject(request)
                            )
                            assert (
                                len(data) == blob["length"]
                                and hashlib.sha256(data).hexdigest() == blob["sha256"]
                            )
                            (destination / entry["path"]).write_bytes(data)
                        captured = reader.read(destination)
                        assert captured.identity == observation["capture"]["content_digest"]
                        assert captured.identity != documents.spell(source.manifest.digest)
                        assert captured.manifest.steps == (0, 2)
                        assert [tap.path for tap in captured.manifest.taps] == ["zeta", "alpha"]
                        assert len(captured.manifest.rows) == 4
                        evidence["capture"] = {
                            "identity": captured.identity,
                            "native_manifest": documents.spell(source.manifest.digest),
                            "content_bytes": source.content_bytes,
                            "rows": len(captured.manifest.rows),
                        }
                        # The independent recipient is verified before the producer ACK.
                        client.WorkspaceReleaseByteTree(
                            pb.NativeByteRetentionCall(claim=claim, request=hold)
                        )
                else:
                    assert "capture" not in observation
            except BaseException as exc:
                errors.append(repr(exc))
            finally:
                if owner.terminal is not None:
                    owner.auto_ack = True
                    owner._on_outcome(owner.terminal)
                else:
                    owner.done.set()
                    owner.host.close()

        thread = threading.Thread(target=collect, daemon=True)
        monitor.append(thread)
        thread.start()

    outcome = run_slice(
        config,
        LocalRequest(
            entrypoint="generate",
            payload={"seed": 1234, "steps": 3},
            outputs=(),
            request_id="capture-worker-" + key,
            capture=ActivationCapture(("alpha", "zeta"), (0, 2)) if args.capture else None,
        ),
        workspace=root / key,
        package_installation_id="local-" + uuid.uuid4().hex,
        package_interface_document=interface,
        project=project,
        development_tensorfs_root=root / "tensorfs",
        options=WorkerOptions(root=root / key / "worker"),
        auto_ack=False,
        record_owner_hook=hook,
        delete_workspace=False,
    )
    for thread in monitor:
        thread.join(10)
        assert not thread.is_alive()
    for server, pool in servers:
        server.stop(grace=0).wait()
        pool.shutdown(wait=True)
    evidence["result"] = outcome.result
    evidence["exit_code"] = outcome.exit_code
    evidence["ended"] = outcome.ended
    evidence["errors"] = errors
    evidence["worker_records"] = [asdict(row) for row in outcome.attempt_records]
    (root / (key + ".json")).write_text(json.dumps(evidence, indent=2))
    assert not errors, errors
    assert outcome.exit_code == 0 and outcome.ended == "terminal", (
        outcome.exit_code,
        outcome.error,
    )
    assert outcome.result and outcome.result["device"] == "cuda"
    assert any(row.kind == "closed" for row in outcome.attempt_records)
    print(
        json.dumps(
            {
                "proof": key,
                "result": outcome.result,
                "capture": evidence.get("capture"),
                "seconds": outcome.timings["wall_ms"] / 1000,
            }
        )
    )


if __name__ == "__main__":
    main()
