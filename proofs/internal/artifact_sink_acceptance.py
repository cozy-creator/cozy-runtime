#!/usr/bin/env python3
"""Real-kernel + real-TensorFS acceptance for Runtime ArtifactSink.

The proof creates a TensorFS checkpoint with the shipping ``tfs`` binary, roots it through
the Python facade, invokes an ordinary typed job through the author kernel, and drives the
actual tfs-019 transaction. No store or path enters the job surface.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import msgspec
import tensorfs

from cozy.worker.v1 import worker_pb2 as pb
from cozy_runtime.author import (
    App,
    ArtifactOutput,
    ArtifactPart,
    ArtifactSink,
    ArtifactTarget,
    ArtifactTensor,
    Model,
    describe,
)
from cozy_runtime.author.fakes import invoke_with_fakes
from cozy_runtime.internal import descriptor
from cozy_runtime.internal.artifact_sink import ArtifactTransactionHost, protocol_receipt
from cozy_runtime.internal.discovery import Discovered
from cozy_runtime.protocol import documents


class Request(msgspec.Struct, forbid_unknown_fields=True):
    marker: str


class Result(msgspec.Struct):
    transaction: str
    receipt: str


class SourceModel(Model[object]):
    def load(self, loader: Any) -> None:
        del loader


app = App()


@app.job(artifacts=(ArtifactOutput("model", max_new_bytes=4096),))
def derive(payload: Request, model: SourceModel, artifacts: ArtifactSink) -> Result:
    receipt = artifacts.derive(
        "model",
        sources={"full": model},
        targets={
            "derived": ArtifactTarget(
                source="full",
                source_component="unet",
                add={
                    payload.marker: ArtifactTensor(
                        logical_dtype="bf16",
                        shape=(32, 32),
                        encoding=PLAIN,
                        parts={"value": ArtifactPart("bf16", (32, 32), b"x" * 2048)},
                    )
                },
            )
        },
        order=ORDER,
    )
    return Result(receipt.artifact_transaction_id, receipt.tensorfs_receipt_digest)


PLAIN = ""
ORDER: tuple[tuple[str, str], ...] = ()


def command(*argv: str) -> str:
    result = subprocess.run(argv, text=True, capture_output=True, check=False)
    if result.returncode:
        raise RuntimeError(f"{' '.join(argv)} failed:\n{result.stdout}\n{result.stderr}")
    return result.stdout


def source(tfs: Path, root: Path) -> tuple[tensorfs.Store, str, int, dict[str, Any]]:
    output = command(str(tfs), "checkpoint", "write", str(root), "--blocks", "22", "--scale", "32")
    found = re.search(r"manifest_id\s+(sha256:[0-9a-f]{64})", output)
    if found is None:
        raise RuntimeError("checkpoint write named no manifest")
    manifest_id = found.group(1)
    store = tensorfs.Store.open(str(root))
    held = store.manifest(manifest_id)
    manifest_bytes = held["manifest"]
    operation = store.begin_operation("artifact-source", "proof", "source", 1)
    operation.hold_manifest(manifest_id, len(manifest_bytes))
    for reached in store.walk(manifest_id):
        operation.hold_blob(reached["id"], reached["length"])
    evidence = json.dumps(
        {"classification_digest": tensorfs.topology_digest(held["header"])},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    operation.commit_release(None, "1.0.0", "local", manifest_id, len(manifest_bytes), evidence)
    return store, manifest_id, len(manifest_bytes), tensorfs.parse_header(held["header"])


def model(manifest_id: str) -> SourceModel:
    value = object.__new__(SourceModel)
    object.__setattr__(value, "checkpoint_ref", manifest_id)
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tfs", type=Path, required=True)
    args = parser.parse_args()
    found = Discovered(
        app=app,
        application="artifact_sink_acceptance:app",
        project=Path(__file__).parent,
        module=sys.modules[__name__],
        surfaces=describe(app),
        bindings={},
    )
    described = descriptor.build(found)
    artifact_outputs = described["jobs"][0].get("artifact_outputs")
    if artifact_outputs != [
        {
            "output_id": "model",
            "mime_type": "application/vnd.cozy.model-manifest",
            "max_bytes": 4096,
        }
    ]:
        raise RuntimeError(f"descriptor artifact outputs changed: {artifact_outputs!r}")
    with tempfile.TemporaryDirectory(prefix="cozy-artifact-sink-") as temporary:
        root = Path(temporary)
        store, manifest_id, manifest_length, header = source(args.tfs, root / "store")
        component = header["components"]["unet"]
        global PLAIN, ORDER
        PLAIN = next(
            tensor["encoding"]
            for tensor in component.values()
            if list(tensor["parts"]) == ["value"]
        )
        ORDER = (*(("derived", key) for key in component), ("derived", "table"))
        chosen = model(manifest_id)

        def run(writer_session: int) -> tuple[Result, ArtifactTransactionHost]:
            spool = root / f"spool-{writer_session}"
            spool.mkdir()
            host = ArtifactTransactionHost(
                store=store,
                spool=spool,
                owner_scope="proof-owner",
                request_id="artifact-proof",
                invocation_spec_digest="sha256:" + "1" * 64,
                writer_session_id=writer_session,
                allowed_sources={manifest_id: manifest_length},
                output_bounds={"model": 4096},
            )
            result = invoke_with_fakes(
                app.get("derive"),
                {"marker": "table"},
                request_id="artifact-proof",
                spool=spool,
                models={"model": chosen},
                artifacts=host.commit,
            ).result
            assert isinstance(result, Result)
            return result, host

        first, host = run(1)
        second, _ = run(2)
        if first != second:
            raise RuntimeError("identical retry returned a different artifact receipt")
        lookup = store.derived_lookup(first.transaction)
        receipt = lookup.get("receipt")
        if lookup.get("state") != "committed" or not isinstance(receipt, dict):
            raise RuntimeError("TensorFS did not retain the committed artifact transaction")
        manifest = receipt["manifest"]
        manifest_id_out = "sha256:" + manifest["sha256"]
        verified = store.manifest(manifest_id_out)
        if "table" not in tensorfs.parse_header(verified["header"])["components"]["derived"]:
            raise RuntimeError("derived Manifest omitted the produced table")

        author_receipt = host._wrap("model", first.transaction, receipt, replayed=True)
        reference, canonical, digest = protocol_receipt(
            author_receipt,
            owner_scope="proof-owner",
            request_id="artifact-proof",
            invocation_spec_digest="sha256:" + "1" * 64,
        )
        parsed = documents.read(canonical, pb.ArtifactReceipt)
        if reference.artifact_receipt_digest != digest or parsed["output_slot"] != "model":
            raise RuntimeError("Worker Protocol ArtifactReceipt did not preserve identity")

        released = host.release("model")
        if released.tensorfs_receipt_digest != first.receipt:
            raise RuntimeError("release changed the TensorFS receipt")
        print(
            "ARTIFACT SINK PASS",
            manifest_id_out,
            first.transaction,
            first.receipt,
            "replay=true protocol=true release=true",
        )


if __name__ == "__main__":
    main()
