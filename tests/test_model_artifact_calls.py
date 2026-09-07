from __future__ import annotations

import hashlib
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import grpc
import msgspec
import pytest
import tensorfs

from cozy_runtime.author import (
    App,
    Context,
    Invocation,
    Loader,
    Model,
    ModelArtifact,
    ObjectRef,
    attempt,
    describe,
    invocable,
)
from cozy_runtime.author._model import _derive_model
from cozy_runtime.internal import interface_wheel, package_interface
from cozy_runtime.internal.discovery import Discovered
from cozy_runtime.internal.worker import derived_retention
from cozy_runtime.internal.worker.control import _PreparationServicer
from cozy_runtime.protocol import documents
from cozy_runtime.protocol import worker_pb2 as pb
from cozy_runtime.protocol import worker_pb2_grpc as rpc
from test_weights_resume_native import _committed


class Source(Model[object]):
    def load(self, loader: Loader) -> None:
        raise AssertionError("a derive-only Model must not be loaded")


class Result(msgspec.Struct):
    manifest: str


@invocable(reusable=True)
async def consume(ctx: Context, *, source: Source) -> Result:
    ctx.raise_if_cancelled()
    return Result(source.checkpoint_ref)


@invocable(reusable=True)
async def echo(ctx: Context, *, value: ModelArtifact) -> ModelArtifact:
    return value


def test_model_proxy_is_an_exact_artifact_and_dispatch_checks_bound_model(tmp_path: Path) -> None:
    app = App()
    app.job(consume)
    app.job(echo)
    found = Discovered(app, __name__ + ":app", tmp_path, sys.modules[__name__], describe(app), {})
    raw = package_interface.canonical_bytes(package_interface.build(found))
    metadata = package_interface.read_bytes(raw)
    consuming = next(row for row in metadata["jobs"] if row["name"] == "consume")
    assert consuming["invocable"]["parameters"] == ["source"]
    assert consuming["request"]["fields"][0]["type"] == {"union": ["null", {"input": "model"}]}
    assert next(row for row in metadata["jobs"] if row["name"] == "echo")["result"] == {
        "input": "model"
    }
    for name, body in interface_wheel.generate(raw).items():
        if name.endswith(".py"):
            compile(body, name, "exec")
    artifact = ModelArtifact(
        "producer", "result", ObjectRef("sha256:" + "17" * 32, 512), "sha256:" + "18" * 32
    )
    invocation = Invocation(
        "consumer",
        tmp_path / "valid",
        time.monotonic() + 5,
        models={"source": _derive_model(Source, artifact.manifest.digest)},
    )
    result, outcome, _ = attempt(
        app.get("consume"), {"source": msgspec.to_builtins(artifact)}, invocation
    )
    assert (
        outcome.terminal == "succeeded"
        and result is not None
        and result.result.manifest == artifact.manifest.digest
    )
    changed = msgspec.to_builtins(artifact)
    changed["manifest"]["digest"] = "sha256:" + "19" * 32
    _, refused, _ = attempt(app.get("consume"), {"source": changed}, invocation)
    assert refused.code == "model_artifact_binding"
    # Existing top-level fixed Model bindings remain valid without a derived handle.
    result, outcome, _ = attempt(app.get("consume"), {}, invocation)
    assert outcome.terminal == "succeeded" and result is not None


def test_native_artifact_receipt_and_independent_loopback_retention_survive_original_dispose(
    tmp_path: Path,
) -> None:
    store, _, _, receipt = _committed(tmp_path)
    artifact = receipt.artifact
    native = json.loads(receipt.tensorfs_receipt)
    assert artifact.producer_request_id == "request"
    assert artifact.manifest == ObjectRef(
        "sha256:" + native["manifest"]["sha256"], native["manifest"]["length"]
    )
    assert (
        artifact.tensorfs_receipt_digest
        == "sha256:" + hashlib.sha256(receipt.tensorfs_receipt).hexdigest()
    )
    root = tmp_path / "store"
    server = grpc.server(ThreadPoolExecutor(max_workers=2))
    handler = _PreparationServicer(
        None,
        None,
        None,
        None,
        None,
        derived_retainer=lambda req, release: derived_retention.change(
            req, tensorfs_root=root, release=release
        ),
        derived_result_releaser=lambda req: derived_retention.release_result(
            req, tensorfs_root=root
        ),
    )
    rpc.add_RuntimePreparationServicer_to_server(handler, server)
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()
    request = pb.DerivedRetentionRequest(
        weights_transaction_id=receipt.weights_transaction_id,
        tensorfs_receipt_digest=documents.raw(artifact.tensorfs_receipt_digest),
        retention_id="sha256:" + "19" * 32,
    )
    try:
        with grpc.insecure_channel(f"127.0.0.1:{port}") as channel:
            client = rpc.RuntimePreparationStub(channel)
            held = client.RetainDerivedResult(request, timeout=5)
            assert not held.released and held.manifest.digest == documents.raw(
                artifact.manifest.digest
            )
            assert client.RetainDerivedResult(request, timeout=5) == held
            store.derived_adopt(receipt.weights_transaction_id, "original-result")
            original_release = pb.DerivedResultReleaseRequest(
                weights_transaction_id=request.weights_transaction_id,
                tensorfs_receipt_digest=request.tensorfs_receipt_digest,
            )
            for _ in range(2):
                released_original = client.ReleaseDerivedResult(original_release, timeout=5)
                assert released_original.released
            assert store.derived_lookup(receipt.weights_transaction_id)["receipt"] == native
            tensorfs.gc(str(root))
            assert client.RetainDerivedResult(request, timeout=5) == held
            assert store.manifest(artifact.manifest.digest)
            changed = pb.DerivedRetentionRequest()
            changed.CopyFrom(request)
            changed.tensorfs_receipt_digest = b"x" * 32
            with pytest.raises(grpc.RpcError) as refused:
                client.RetainDerivedResult(changed, timeout=5)
            assert refused.value.code() == grpc.StatusCode.FAILED_PRECONDITION
            released = client.ReleaseDerivedRetention(request, timeout=5)
            assert (
                released.released and client.ReleaseDerivedRetention(request, timeout=5) == released
            )
            with pytest.raises(grpc.RpcError):
                client.RetainDerivedResult(request, timeout=5)
    finally:
        server.stop(0).wait()
