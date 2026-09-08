"""A native child result remains readable after producer cleanup and forwards as input."""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any

import msgspec
import pytest
import tensorfs

from cozy_runtime import canonical_json
from cozy_runtime.author import FileAsset, Tree
from cozy_runtime.author._call_results import decode
from cozy_runtime.author._errors import CapabilityError
from cozy_runtime.internal.worker import byte_inputs, child_byte_results, grants
from cozy_runtime.internal.worker import workspace_byte_outputs as outputs
from cozy_runtime.internal.worker.workspace import Workspace, WorkspaceRefusal
from cozy_runtime.protocol import documents
from cozy_runtime.protocol import worker_pb2 as pb
from test_workspace_byte_outputs import producing


class Result(msgspec.Struct):
    report: FileAsset
    bundle: Tree


def finish_producer(
    workspace: Workspace, spec: bytes, source: pb.NativeByteTreeRef, data: bytes
) -> None:
    raw, digest = documents.identity(
        pb.AttemptOutcomeBody(
            request_id="producer",
            attempt_ordinal=1,
            invocation_spec_digest=documents.spell(spec),
            status=pb.OUTCOME_STATUS_SUCCEEDED,
            execution_started=True,
            output_manifest=pb.OutputManifest(
                outputs=[
                    pb.OutputEntry(
                        output_id="report",
                        digest=hashlib.sha256(data).digest(),
                        length=len(data),
                        mime_type="application/json",
                        native_tree=source,
                    )
                ]
            ),
        )
    )
    workspace.outcome(
        "owner",
        pb.AttemptOutcome(
            request_id="producer",
            attempt_ordinal=1,
            invocation_spec_digest=spec,
            outcome_id="complete",
            outcome_digest=digest,
            outcome_canonical_bytes=raw,
        ),
    )
    workspace.acknowledge(
        "owner",
        pb.AttemptOutcomeAck(
            request_id="producer",
            attempt_ordinal=1,
            invocation_spec_digest=spec,
            outcome_id="complete",
            outcome_digest=digest,
        ),
    )


def retained(tmp_path: Path) -> tuple[Workspace, pb.ChildCallResult, bytes]:
    workspace, spec, manifest, path = producing(tmp_path)
    data = path.read_bytes()
    source = outputs.commit(
        workspace,
        "owner",
        "producer",
        1,
        spec,
        "report",
        manifest,
        [("payload", path)],
        200000,
    )
    received = []
    for index, output_id in enumerate(("report", "bundle")):
        retention_id = "sha256:" + f"{index + 1:02x}" * 32
        hold = pb.NativeByteRetentionRequest(source=source, retention_id=retention_id)
        outputs.change_hold(workspace, "owner", hold, release=False)
        received.append(
            pb.ChildByteResultGrant(output_id=output_id, source=source, retention_id=retention_id)
        )
    value = {
        "report": {
            "asset_ref": "sha256:" + hashlib.sha256(data).hexdigest(),
            "kind": "file",
            "digest": "sha256:" + hashlib.sha256(data).hexdigest(),
            "size_bytes": len(data),
            "media_type": "application/json",
        },
        "bundle": {
            "asset_ref": documents.spell(source.manifest.digest),
            "kind": "tree",
            "digest": documents.spell(source.manifest.digest),
            "size_bytes": len(data),
        },
    }
    result = pb.ChildCallResult(
        parent_request_id="parent",
        parent_attempt_ordinal=1,
        call_index=0,
        parent_invocation_spec_digest=b"p" * 32,
        intent_digest=b"i" * 32,
        child_request_id="producer",
        state=pb.CHILD_CALL_STATE_SUCCEEDED,
        result_canonical_bytes=canonical_json.encode(value),
        byte_result_grants=received,
    )
    path.unlink()
    finish_producer(workspace, spec, source, data)
    tensorfs.gc(str(workspace.store_root))
    return Workspace(workspace.store_root), result, data


def test_result_materializes_after_ack_and_gc_then_forwards_as_native_inputs(
    tmp_path: Path,
) -> None:
    workspace, reply, data = retained(tmp_path)
    rows = child_byte_results.materialize(workspace, "owner", reply, tmp_path / "parent")
    closed = False

    def guard() -> None:
        if closed:
            raise CapabilityError("closed", code="escaped_handle")

    result, observation = decode(
        canonical_json.decode(reply.result_canonical_bytes),
        Result,
        rows,
        request_id="parent",
        guard=guard,
        observation=None,
    )
    assert observation is None
    assert isinstance(result.report, FileAsset) and isinstance(result.bundle, Tree)
    assert result.report.read_bytes() == data
    assert (result.bundle.path / "payload").read_bytes() == data
    assert result.bundle.size_bytes == len(data)
    assert not any(Path(row["local"]).stat().st_mode & 0o222 for row in rows)

    source = reply.byte_result_grants[0].source
    input_hold = pb.NativeByteRetentionRequest(source=source, retention_id="sha256:" + "03" * 32)
    outputs.change_hold(workspace, "owner", input_hold, release=False)
    tree_id = "tree:" + result.bundle.digest
    spec = pb.InvocationSpec(
        job=pb.JobInvocationSpec(),
        inputs=[
            pb.InputBinding(
                input_id="report",
                digest="sha256:" + hashlib.sha256(data).hexdigest(),
                length=len(data),
                kind_mime="application/json",
            ),
            pb.InputBinding(
                input_id=tree_id,
                digest=documents.spell(source.manifest.digest),
                length=source.manifest.length,
                kind_mime=byte_inputs.TREE_MIME,
            ),
        ],
    )
    _, digest = documents.identity(spec)
    access = pb.DeliveryGrant(
        invocation_spec_digest=digest,
        expires_at_unix=int(time.time()) + 60,
        inputs=[
            pb.InputAccess(input_id="report", native_tree=input_hold),
            pb.InputAccess(input_id=tree_id, native_tree=input_hold),
        ],
    )
    bound = grants.bind(documents.body(spec), access, digest)
    downstream = tmp_path / "downstream"
    inputs = grants.hydrate_inputs(
        bound, grants.Authorizer(), spool=downstream, workspace=workspace, owner="owner"
    )
    trees = grants.read_trees(
        bound, grants.Authorizer(), spool=downstream, workspace=workspace, owner="owner"
    )
    assert inputs["report"].local.read_bytes() == data
    assert (trees[result.bundle.digest][0] / "payload").read_bytes() == data
    assert inputs["report"].local != result.report._local
    assert trees[result.bundle.digest][0] != result.bundle.path
    closed = True
    with pytest.raises(CapabilityError, match="escaped_handle"):
        result.report.read_bytes()
    with pytest.raises(CapabilityError, match="escaped_handle"):
        result.bundle.files()
    grants.release_inputs(downstream)
    assert not (downstream / "native-input-trees").exists()


@pytest.mark.parametrize("mutation", ["owner", "digest", "member", "inventory", "unretained"])
def test_incorrect_native_result_never_becomes_a_hydrated_handle(
    tmp_path: Path, mutation: str
) -> None:
    workspace, reply, _ = retained(tmp_path)
    owner = "owner"
    if mutation == "owner":
        owner = "another-owner"
    elif mutation == "digest":
        value = canonical_json.decode(reply.result_canonical_bytes)
        value["report"]["digest"] = value["report"]["asset_ref"] = "sha256:" + "ff" * 32
        reply.result_canonical_bytes = canonical_json.encode(value)
    elif mutation == "member":
        reply.byte_result_grants[0].source.content_bytes += 1
    elif mutation == "inventory":
        reply.byte_result_grants.append(reply.byte_result_grants[0])
    else:
        reply.byte_result_grants[0].retention_id = "sha256:" + "ab" * 32
    with pytest.raises((ValueError, WorkspaceRefusal)):
        child_byte_results.materialize(workspace, owner, reply, tmp_path / "parent")


def test_author_request_decoder_still_refuses_hydration_records() -> None:
    from cozy_runtime.author._assets import asset_dec_hook

    for target in (FileAsset, Tree):
        with pytest.raises(TypeError, match="reference string"):
            asset_dec_hook(target, {"asset_ref": "sha256:" + "00" * 32, "local": "/tmp"})


def test_capture_native_manifest_and_semantic_identity_remain_distinct(tmp_path: Path) -> None:
    import numpy as np

    capture = pytest.importorskip("cozy_eval.capture")

    from cozy_runtime.author import ExecutionObservation
    from cozy_runtime.internal.capture_observation import document

    workspace, spec, _, _ = producing(tmp_path)
    root = tmp_path / "captured"
    semantic = capture.write(
        root,
        capture.Manifest(
            format="cozy.capture/1",
            components=("model",),
            taps=(capture.Tap("model", "module"),),
            steps=(0,),
            sketch=capture.Sketch("countsketch", 4, capture.SEED_DOMAIN),
            rows=(capture.Row("model", 0, ((1, 4),), 4, 0, 0, 1.0, 1.0, 0.0),),
        ),
        np.ones((1, 4), dtype="<f4"),
    )
    members: list[dict[str, Any]] = [
        {
            "path": path.name,
            "kind": "file",
            "blob": {
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "length": path.stat().st_size,
            },
        }
        for path in sorted(root.iterdir())
    ]
    manifest = canonical_json.encode({"entries": members})
    source = outputs.commit(
        workspace,
        "owner",
        "producer",
        1,
        spec,
        "runtime.capture",
        manifest,
        [(member["path"], root / member["path"]) for member in members],
        1 << 20,
    )
    hold = pb.NativeByteRetentionRequest(source=source, retention_id="sha256:" + "12" * 32)
    outputs.change_hold(workspace, "owner", hold, release=False)
    reply = pb.ChildCallResult(
        state=pb.CHILD_CALL_STATE_SUCCEEDED,
        result_canonical_bytes=b'{"value":1}',
        byte_result_grants=[
            pb.ChildByteResultGrant(
                output_id="runtime.capture", source=source, retention_id=hold.retention_id
            )
        ],
        observation=pb.ExecutionObservation(
            environment=pb.ExecutionEnvironment(runtime_version="test", accelerator="CPU"),
            capture=pb.ActivationCaptureResult(
                output_id="runtime.capture", content_digest=bytes.fromhex(semantic[7:])
            ),
        ),
    )
    rows = child_byte_results.materialize(workspace, "owner", reply, tmp_path / "parent")
    observed: ExecutionObservation | None = msgspec.convert(
        document(reply.observation), type=ExecutionObservation
    )
    _, observed = decode(
        {"value": 1},
        dict[str, int],
        rows,
        request_id="parent",
        guard=lambda: None,
        observation=observed,
    )
    assert (
        observed is not None and observed.capture is not None and observed.capture.tree is not None
    )
    assert (
        capture.read(observed.capture.tree.path).identity
        == observed.capture.content_digest
        == semantic
    )
    assert observed.capture.tree.digest != observed.capture.content_digest
    reply.observation.capture.content_digest = b"a" * 32
    with pytest.raises(WorkspaceRefusal, match="semantic"):
        child_byte_results.materialize(workspace, "owner", reply, tmp_path / "wrong-semantic")


def test_worker_poll_verifies_caches_and_releases_its_owned_checkout(tmp_path: Path) -> None:
    from cozy_runtime.internal.worker.attempts import AttemptRecord
    from cozy_runtime.internal.worker.calls import Calls, _Pending

    workspace, reply, data = retained(tmp_path)
    sent: list[pb.WorkerFrame] = []
    worker = Calls(sent.append, lambda: 1, lambda: 43, workspace=workspace, owner=lambda: "owner")
    request = pb.ChildCallRequest(
        parent_request_id="parent",
        parent_attempt_ordinal=1,
        parent_invocation_spec_digest=reply.parent_invocation_spec_digest,
        intent_digest=reply.intent_digest,
        call_index=0,
    )
    worker.pending[("parent", 1, 0)] = _Pending(request, 1)
    parent = AttemptRecord(
        "parent",
        1,
        reply.parent_invocation_spec_digest,
        {},
        state="running",
        spool=tmp_path / "parent",
    )
    assert worker.absorb(reply)
    first = worker.handle(parent, {"kind": "child_poll", "call_index": 0})
    assert first["ok"], first
    assert Path(first["byte_grants"][0]["local"]).read_bytes() == data
    assert worker.handle(parent, {"kind": "child_poll", "call_index": 0}) == first
    assert not sent
    worker.close("parent", 1)
    assert not Path(first["byte_grants"][0]["local"]).exists()
    assert not worker.pending
