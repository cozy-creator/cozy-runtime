#!/usr/bin/env python3
"""Real-kernel + real-TensorFS acceptance for Runtime ArtifactSink.

The proof creates a TensorFS checkpoint with the shipping ``tfs`` binary, roots it through
the Python facade, invokes an ordinary typed job through the author kernel, and drives the
actual tfs-019 transaction. No store or path enters the job surface.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from types import SimpleNamespace
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
from cozy_runtime.internal.artifact_sink import (
    ArtifactHostBinding,
    ArtifactTransactionHost,
    protocol_receipt,
)
from cozy_runtime.internal.discovery import Discovered
from cozy_runtime.internal.worker.artifacts import ArtifactExchange
from cozy_runtime.protocol import documents


class Request(msgspec.Struct, forbid_unknown_fields=True):
    marker: str
    overlong: bool = False


class Result(msgspec.Struct):
    transaction: str
    receipt: str


class ReadRequest(msgspec.Struct, forbid_unknown_fields=True):
    alias: str = "full"
    offset: int = 0
    size: int = 2048
    readonly: bool = False


class ReadResult(msgspec.Struct):
    transaction: str
    source_digest: str


class SourceModel(Model[object]):
    def load(self, loader: Any) -> None:
        del loader


class Chunks:
    """A bounded reader proving TensorFS pulls production directly under backpressure."""

    def __init__(self, data: bytes) -> None:
        self.data = data
        self.offset = 0
        self.reads = 0
        STREAMS.append(self)

    def read(self, size: int = -1) -> bytes:
        self.reads += 1
        if self.offset == len(self.data):
            return b""
        take = 257 if size < 0 else min(size, 257)
        chunk = self.data[self.offset : self.offset + take]
        self.offset += len(chunk)
        return chunk


app = App()
STREAMS: list[Chunks] = []


def produce(payload: Request, model: SourceModel, artifacts: ArtifactSink) -> Result:
    stream = Chunks(b"x" * (2049 if payload.overlong else 2048))
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
                        parts={"value": ArtifactPart("bf16", (32, 32), stream)},
                    )
                },
            )
        },
        order=ORDER,
    )
    if not receipt.replayed and stream.reads < 2:
        raise RuntimeError("TensorFS did not stream the produced value")
    return Result(receipt.artifact_transaction_id, receipt.tensorfs_receipt_digest)


@app.job(artifacts=(ArtifactOutput("model", max_new_bytes=4096),))
def derive(payload: Request, model: SourceModel, artifacts: ArtifactSink) -> Result:
    return produce(payload, model, artifacts)


@app.job(name="derive-tight", artifacts=(ArtifactOutput("model", max_new_bytes=1024),))
def derive_tight(payload: Request, model: SourceModel, artifacts: ArtifactSink) -> Result:
    return produce(payload, model, artifacts)


@app.job(name="derive-read", artifacts=(ArtifactOutput("model", max_new_bytes=4096),))
def derive_read(payload: ReadRequest, model: SourceModel, artifacts: ArtifactSink) -> ReadResult:
    targets = {
        "derived": ArtifactTarget(
            source="full",
            source_component="unet",
            drop=(SOURCE_KEY,),
            add={
                SOURCE_KEY: ArtifactTensor(
                    logical_dtype="bf16",
                    shape=(32, 32),
                    encoding=PLAIN,
                    parts={"value": ArtifactPart("bf16", (32, 32))},
                )
            },
        )
    }
    order = (
        *(("derived", key) for key in SOURCE_KEYS if key != SOURCE_KEY),
        ("derived", SOURCE_KEY),
    )
    with artifacts.open(
        "model", sources={"full": model}, targets=targets, order=order
    ) as transaction:
        if transaction.replayed:
            receipt = transaction.receipt
            assert receipt is not None
            return ReadResult(receipt.artifact_transaction_id, EXPECTED_SOURCE_DIGEST)
        writable = bytes(payload.size) if payload.readonly else bytearray(payload.size)
        transaction.source_read_into(
            payload.alias,
            "unet",
            SOURCE_KEY,
            "value",
            payload.offset,
            writable,
        )
        raw = bytes(writable)
        digest = hashlib.sha256(raw).hexdigest()
        transaction.add_part("derived", SOURCE_KEY, "value", raw)
        receipt = transaction.commit()
        return ReadResult(receipt.artifact_transaction_id, digest)


PLAIN = ""
ORDER: tuple[tuple[str, str], ...] = ()
SOURCE_KEY = ""
SOURCE_KEYS: tuple[str, ...] = ()
EXPECTED_SOURCE_DIGEST = ""


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
        global EXPECTED_SOURCE_DIGEST, PLAIN, SOURCE_KEY, SOURCE_KEYS, ORDER
        SOURCE_KEYS = tuple(component)
        SOURCE_KEY, source_tensor = next(
            (key, tensor) for key, tensor in component.items() if list(tensor["parts"]) == ["value"]
        )
        PLAIN = source_tensor["encoding"]
        source_part = source_tensor["parts"]["value"]
        source_segment = source_part["segments"][0]
        source_bytes = store.document(source_segment["sha256"], 1 << 20)
        EXPECTED_SOURCE_DIGEST = hashlib.sha256(source_bytes[128 : 128 + 2048]).hexdigest()
        ORDER = (*(("derived", key) for key in component), ("derived", "table"))
        chosen = model(manifest_id)

        def run(writer_session: int) -> tuple[Result, ArtifactTransactionHost]:
            spool = root / f"spool-{writer_session}"
            spool.mkdir()
            host = ArtifactTransactionHost(
                store=store,
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
                artifacts=host.open,
            ).result
            assert isinstance(result, Result)
            return result, host

        first, host = run(1)
        second, _ = run(2)
        if first != second:
            raise RuntimeError("identical retry returned a different artifact receipt")
        if any(any((root / f"spool-{writer}").iterdir()) for writer in (1, 2)):
            raise RuntimeError("ArtifactSink made a model-sized attempt-spool copy")
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

        order: list[str] = []
        durable_digest = b""
        ordered_host: ArtifactTransactionHost

        def bind_ordered(
            output_slot: str, declaration_digest: bytes, declaration: bytes
        ) -> ArtifactHostBinding:
            del declaration_digest, declaration
            state = store.derived_lookup(ordered_host.transaction_id(output_slot))["state"]
            order.append(f"intent:{state}")
            return ArtifactHostBinding(ordered_host.transaction_id(output_slot), 11)

        def record_ordered(value: Any) -> None:
            nonlocal durable_digest
            state = store.derived_lookup(value.artifact_transaction_id)["state"]
            order.append(f"receipt:{state}")
            durable_digest = protocol_receipt(
                value,
                owner_scope="proof-owner",
                request_id="artifact-host-order",
                invocation_spec_digest="sha256:" + "3" * 64,
            )[2]

        ordered_host = ArtifactTransactionHost(
            store=store,
            owner_scope="proof-owner",
            request_id="artifact-host-order",
            invocation_spec_digest="sha256:" + "3" * 64,
            writer_session_id=1,
            allowed_sources={manifest_id: manifest_length},
            output_bounds={"model": 4096},
            bind_intent=bind_ordered,
            record_receipt=record_ordered,
        )
        ordered_result = invoke_with_fakes(
            app.get("derive"),
            {"marker": "table"},
            request_id="artifact-host-order",
            spool=root / "spool-host-order",
            models={"model": chosen},
            artifacts=ordered_host.open,
        ).result
        if order != ["intent:absent", "receipt:committed"] or not durable_digest:
            raise RuntimeError(f"host durability order was {order}")
        replay_records = 0

        def bind_committed(
            output_slot: str, declaration_digest: bytes, declaration: bytes
        ) -> ArtifactHostBinding:
            del declaration_digest, declaration
            return ArtifactHostBinding(ordered_host.transaction_id(output_slot), 12, durable_digest)

        def record_replay(value: Any) -> None:
            nonlocal replay_records
            del value
            replay_records += 1

        replay_host = ArtifactTransactionHost(
            store=store,
            owner_scope="proof-owner",
            request_id="artifact-host-order",
            invocation_spec_digest="sha256:" + "3" * 64,
            writer_session_id=1,
            allowed_sources={manifest_id: manifest_length},
            output_bounds={"model": 4096},
            bind_intent=bind_committed,
            record_receipt=record_replay,
        )
        replayed_result = invoke_with_fakes(
            app.get("derive"),
            {"marker": "table"},
            request_id="artifact-host-order",
            spool=root / "spool-host-replay",
            models={"model": chosen},
            artifacts=replay_host.open,
        ).result
        if replayed_result != ordered_result or replay_records:
            raise RuntimeError("committed host replay rewrote or re-recorded the artifact")

        sent: list[pb.WorkerFrame] = []
        exchange: ArtifactExchange

        def host_send(frame: pb.WorkerFrame) -> None:
            sent.append(frame)
            kind = frame.WhichOneof("msg")
            if kind == "artifact_intent":
                intent = frame.artifact_intent
                exchange.absorb_ack(
                    pb.ArtifactHostAck(
                        request_id=intent.request_id,
                        attempt_ordinal=intent.attempt_ordinal,
                        invocation_spec_digest=intent.invocation_spec_digest,
                        output_slot=intent.output_slot,
                        artifact_transaction_id=first.transaction,
                        writer_generation=9,
                        tensorfs_declaration_digest=intent.tensorfs_declaration_digest,
                        stage=pb.ARTIFACT_HOST_STAGE_INTENT,
                        outcome=pb.ARTIFACT_HOST_OUTCOME_RECORDED,
                    )
                )
            elif kind == "artifact_receipt":
                carried = frame.artifact_receipt
                exchange.absorb_ack(
                    pb.ArtifactHostAck(
                        request_id=carried.request_id,
                        attempt_ordinal=carried.attempt_ordinal,
                        invocation_spec_digest=carried.invocation_spec_digest,
                        output_slot=carried.output_slot,
                        artifact_transaction_id=carried.artifact_transaction_id,
                        writer_generation=carried.writer_generation,
                        tensorfs_declaration_digest=carried.tensorfs_declaration_digest,
                        stage=pb.ARTIFACT_HOST_STAGE_RECEIPT,
                        outcome=pb.ARTIFACT_HOST_OUTCOME_RECORDED,
                        artifact_receipt=carried.artifact_receipt,
                        manifest=carried.manifest,
                    )
                )
            else:
                raise RuntimeError(f"unexpected host frame {kind!r}")

        exchange = ArtifactExchange(
            store_root=root / "store",
            send=host_send,
            stop=threading.Event(),
            owner_scope=lambda: "proof-owner",
        )
        attempt_record = SimpleNamespace(
            request_id="artifact-proof",
            attempt=1,
            digest=documents.raw("sha256:" + "1" * 64),
            canceling="",
            artifact_receipts={},
        )
        declaration = json.dumps(
            receipt["declaration"], separators=(",", ":"), sort_keys=True
        ).encode()
        bound = exchange.intent(
            attempt_record,
            output_slot="model",
            declaration_digest=hashlib.sha256(declaration).digest(),
            declaration=declaration,
            requested_writer_generation=0,
        )
        if bound["artifact_transaction_id"] != first.transaction or bound["writer_generation"] != 9:
            raise RuntimeError(f"host intent did not bind exact transaction: {bound}")
        recorded = exchange.receipt(
            attempt_record,
            output_slot="model",
            transaction_id=first.transaction,
            receipt_digest=digest,
            canonical_receipt=canonical,
        )
        receipt_frame = sent[-1].artifact_receipt
        if (
            recorded["artifact_receipt_digest"] != documents.spell(digest)
            or not receipt_frame.objects
            or not receipt_frame.HasField("manifest")
            or list(receipt_frame.objects)
            != sorted(receipt_frame.objects, key=lambda row: row.object_id)
        ):
            raise RuntimeError("host receipt did not carry sorted exact object rows")
        manifest_id_from_frame = documents.spell(receipt_frame.manifest.digest)
        if manifest_id_from_frame != manifest_id_out:
            raise RuntimeError("typed produced Manifest does not match TensorFS receipt")
        first_object = receipt_frame.objects[0]
        read = pb.ArtifactReadRequest(
            artifact_transaction_id=first.transaction,
            writer_generation=9,
            object_id=first_object.object_id,
            source_ref=first_object.source_ref,
            read_id="proof-read",
            offset=0,
            max_bytes=min(64, first_object.length),
        )
        artifact_read_result = exchange.read(read).artifact_read_result
        if (
            artifact_read_result.outcome != pb.ARTIFACT_READ_OUTCOME_DATA
            or hashlib.sha256(artifact_read_result.data).digest()
            != artifact_read_result.data_digest
        ):
            raise RuntimeError("host object read did not return digest-bound bytes")
        read.source_ref = "tensorfs-object:sha256:" + "0" * 64
        if (
            exchange.read(read).artifact_read_result.refusal
            != pb.ARTIFACT_READ_REFUSAL_SOURCE_REF_MISMATCH
        ):
            raise RuntimeError("foreign source ref did not refuse")
        exchange.close()

        released = host.release("model")
        if released.tensorfs_receipt_digest != first.receipt:
            raise RuntimeError("release changed the TensorFS receipt")

        refused_spool = root / "spool-overlong"
        refused_spool.mkdir()
        refused_host = ArtifactTransactionHost(
            store=store,
            owner_scope="proof-owner",
            request_id="artifact-overlong",
            invocation_spec_digest="sha256:" + "2" * 64,
            writer_session_id=1,
            allowed_sources={manifest_id: manifest_length},
            output_bounds={"model": 4096},
        )
        try:
            invoke_with_fakes(
                app.get("derive"),
                {"marker": "table", "overlong": True},
                request_id="artifact-overlong",
                spool=refused_spool,
                models={"model": chosen},
                artifacts=refused_host.open,
            )
        except Exception as exc:
            if getattr(exc, "code", "") != "artifact_tensorfs_refused":
                raise RuntimeError(f"overlong stream refused as {exc!r}") from exc
        else:
            raise RuntimeError("overlong produced stream committed")
        refused_state = store.derived_lookup(refused_host.transaction_id("model"))
        if refused_state.get("state") != "open" or "writer_session_id" in refused_state:
            raise RuntimeError(f"overlong stream did not fence its writer: {refused_state}")
        refused_host.abandon("model")

        bounded_host = ArtifactTransactionHost(
            store=store,
            owner_scope="proof-owner",
            request_id="artifact-bounded",
            invocation_spec_digest="sha256:" + "3" * 64,
            writer_session_id=1,
            allowed_sources={manifest_id: manifest_length},
            output_bounds={"model": 1024},
        )
        try:
            invoke_with_fakes(
                app.get("derive-tight"),
                {"marker": "table"},
                request_id="artifact-bounded",
                models={"model": chosen},
                artifacts=bounded_host.open,
            )
        except Exception as exc:
            if getattr(exc, "code", "") != "artifact_tensorfs_refused":
                raise RuntimeError(f"over-bound declaration refused as {exc!r}") from exc
        else:
            raise RuntimeError("over-bound artifact declaration reached its stream")
        if STREAMS[-1].reads != 0:
            raise RuntimeError("TensorFS read bytes before enforcing the aggregate output cap")
        if store.derived_lookup(bounded_host.transaction_id("model")).get("state") != "absent":
            raise RuntimeError("over-bound declaration created a transaction")

        def read_run(
            request_id: str,
            digest_seed: str,
            payload: dict[str, object],
            *,
            writer_session: int = 1,
            cancelled: bool = False,
        ) -> tuple[ReadResult, ArtifactTransactionHost]:
            host = ArtifactTransactionHost(
                store=store,
                owner_scope="proof-owner",
                request_id=request_id,
                invocation_spec_digest="sha256:" + digest_seed * 64,
                writer_session_id=writer_session,
                allowed_sources={manifest_id: manifest_length},
                output_bounds={"model": 4096},
            )
            result = invoke_with_fakes(
                app.get("derive-read"),
                payload,
                request_id=request_id,
                models={"model": chosen},
                artifacts=host.open,
                cancelled=cancelled,
            ).result
            assert isinstance(result, ReadResult)
            return result, host

        read_result, read_host = read_run("artifact-read", "4", {"offset": 128})
        if read_result.source_digest != EXPECTED_SOURCE_DIGEST:
            raise RuntimeError("transaction source read returned different bytes")
        replay, _ = read_run("artifact-read", "4", {"offset": 128}, writer_session=2)
        if replay != read_result:
            raise RuntimeError("source-read transaction replay changed its result")
        read_host.release("model")

        refusal_cases: tuple[tuple[str, str, dict[str, object], str, bool], ...] = (
            (
                "foreign-source",
                "5",
                {"alias": "foreign", "offset": 128},
                "artifact_source_ungranted",
                False,
            ),
            (
                "offset-bound",
                "6",
                {"offset": 1 << 30},
                "artifact_tensorfs_refused",
                False,
            ),
            (
                "readonly-buffer",
                "7",
                {"offset": 128, "readonly": True},
                "artifact_source_buffer",
                False,
            ),
            (
                "empty-buffer",
                "8",
                {"offset": 128, "size": 0},
                "artifact_source_buffer",
                False,
            ),
            (
                "oversize-buffer",
                "9",
                {"offset": 128, "size": (64 << 20) + 1},
                "artifact_source_buffer",
                False,
            ),
            (
                "canceled-read",
                "a",
                {"offset": 128},
                "cancelled",
                True,
            ),
        )
        for name, seed, payload, expected, canceled in refusal_cases:
            try:
                _, failed_host = read_run(f"artifact-{name}", seed, payload, cancelled=canceled)
            except Exception as exc:
                if getattr(exc, "code", "") != expected:
                    raise RuntimeError(f"{name} refused as {exc!r}, want {expected}") from exc
                failed_host = ArtifactTransactionHost(
                    store=store,
                    owner_scope="proof-owner",
                    request_id=f"artifact-{name}",
                    invocation_spec_digest="sha256:" + seed * 64,
                    writer_session_id=1,
                    allowed_sources={manifest_id: manifest_length},
                    output_bounds={"model": 4096},
                )
            else:
                raise RuntimeError(f"{name} source read was admitted")
            state = store.derived_lookup(failed_host.transaction_id("model"))
            if state.get("state") != "open" or "writer_session_id" in state:
                raise RuntimeError(f"{name} did not fence its writer: {state}")
            failed_host.abandon("model")
        print(
            "ARTIFACT SINK PASS",
            manifest_id_out,
            first.transaction,
            first.receipt,
            "replay=true protocol=true release=true direct_stream=true "
            "cap_before_read=true overlong_fenced=true source_read=true "
            "foreign_offset_buffer_cancel_red=true host_intent_before_write=true "
            "host_committed_replay=true host_intent_receipt_read=true typed_manifest=true",
        )


if __name__ == "__main__":
    main()
