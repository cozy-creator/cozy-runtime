"""A native child result remains readable after producer cleanup and forwards as input."""

from __future__ import annotations

import hashlib
import time
from fractions import Fraction
from pathlib import Path
from typing import Annotated, Any

import msgspec
import pytest
import tensorfs

from cozy_runtime import canonical_json
from cozy_runtime.author import (
    App,
    AssetBound,
    Context,
    FileAsset,
    ImageAsset,
    Invocation,
    Tree,
    attempt,
)
from cozy_runtime.author._call_results import decode
from cozy_runtime.author._errors import CapabilityError
from cozy_runtime.author._executor_requests import Answer, CallState, ChildForget, ChildPoll
from cozy_runtime.internal.executor_replies import OutputRow
from cozy_runtime.internal.worker import byte_inputs, child_byte_results, grants
from cozy_runtime.internal.worker import workspace_byte_outputs as outputs
from cozy_runtime.internal.worker.workspace import Workspace, WorkspaceRefusal
from cozy_runtime.protocol import documents
from cozy_runtime.protocol import worker_pb2 as pb
from test_workspace_byte_outputs import producing


class NativeInputs(msgspec.Struct):
    report: Annotated[FileAsset, AssetBound(max_bytes=200000, media_types=("application/json",))]
    bundle: Tree


class VerifiedLength(msgspec.Struct):
    length: int


class Result(msgspec.Struct):
    report: FileAsset
    bundle: Tree


def finish_producer(
    workspace: Workspace,
    spec: bytes,
    source: pb.NativeByteTreeRef,
    data: bytes,
    *,
    media_type: str = "application/json",
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
                        mime_type=media_type,
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


def test_streaming_mp4_survives_native_child_handoff(tmp_path: Path) -> None:
    pytest.importorskip("av")
    from cozy_runtime.author import _codec

    workspace, spec, _, _ = producing(tmp_path)
    path = tmp_path / "stream.mp4"
    encoder = _codec.StreamingMP4Encoder(
        path,
        width=16,
        height=16,
        frame_rate=Fraction(24),
        pixel_aspect_ratio=Fraction(1),
        color_primaries=1,
        color_transfer=1,
        color_matrix=1,
        color_range=1,
        max_bytes=200000,
    )
    for pixel in (32, 96, 160):
        encoder.write_video(bytes([pixel]) * (16 * 16 * 3))
    assert encoder.finish().frame_count == 3
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    manifest = canonical_json.encode(
        {
            "entries": [
                {
                    "path": "payload",
                    "kind": "file",
                    "blob": {"sha256": digest, "length": len(data)},
                }
            ]
        }
    )
    source = outputs.commit(
        workspace, "owner", "producer", 1, spec, "report", manifest, [("payload", path)], 200000
    )
    retention = "sha256:" + "a1" * 32
    outputs.change_hold(
        workspace,
        "owner",
        pb.NativeByteRetentionRequest(source=source, retention_id=retention),
        release=False,
    )
    finish_producer(workspace, spec, source, data, media_type="video/mp4")
    path.unlink()
    tensorfs.gc(str(workspace.store_root))
    value = {
        "clip": {
            "asset_ref": "sha256:" + digest,
            "digest": "sha256:" + digest,
            "kind": "video",
            "media_type": "video/mp4",
            "size_bytes": len(data),
        }
    }
    result = pb.ChildCallResult(
        state=pb.CHILD_CALL_STATE_SUCCEEDED,
        call_index=0,
        result_canonical_bytes=canonical_json.encode(value),
        byte_result_grants=[
            pb.ChildByteResultGrant(output_id="clip", source=source, retention_id=retention)
        ],
    )
    rows = child_byte_results.materialize(workspace, "owner", result, tmp_path / "parent")
    delivered = Path(rows[0]["local"])
    assert delivered.read_bytes() == data
    assert _codec.probe_mp4(delivered).frame_count == 3


@pytest.mark.parametrize("legacy_prefix", [False, True])
def test_result_materializes_after_ack_and_gc_then_forwards_as_native_inputs(
    tmp_path: Path,
    legacy_prefix: bool,
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
    tree_id = "tree:" + result.bundle.digest if legacy_prefix else "bundle"
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
    assert inputs["report"].media_type == "application/json"
    assert (trees[result.bundle.digest][0] / "payload").read_bytes() == data
    consumer = App()

    @consumer.job
    async def verify(ctx: Context, payload: NativeInputs) -> VerifiedLength:
        assert payload.report.read_bytes() == (payload.bundle.path / "payload").read_bytes()
        return VerifiedLength(len(payload.report.read_bytes()))

    verified, outcome, _ = attempt(
        consumer.get("verify"),
        {"report": result.report.digest, "bundle": result.bundle.digest},
        Invocation("downstream", downstream, time.monotonic() + 30, assets=inputs, trees=trees),
    )
    assert outcome.terminal == "succeeded", outcome
    assert verified is not None and verified.result.length == len(data)
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
    # The parent's poll reads the settled child call (`Worker.child_call`).
    worker = Calls(lambda *_: reply, workspace=workspace, owner=lambda: "owner")
    request = pb.ChildCallRequest(
        parent_request_id="parent",
        parent_attempt_ordinal=1,
        parent_invocation_spec_digest=reply.parent_invocation_spec_digest,
        intent_digest=reply.intent_digest,
        call_index=0,
    )
    worker.pending[("parent", 1, 0)] = _Pending(request, reply)
    parent = AttemptRecord(
        "parent",
        1,
        reply.parent_invocation_spec_digest,
        {},
        state="running",
        spool=tmp_path / "parent",
    )
    first = worker.handle(parent, ChildPoll(call_index=0))
    assert isinstance(first, CallState) and first.ok, first
    local = first.byte_grants[0]["local"]
    assert isinstance(local, str) and Path(local).read_bytes() == data
    assert worker.handle(parent, ChildPoll(call_index=0)) == first
    forgotten = worker.handle(parent, ChildForget(call_index=0))
    assert isinstance(forgotten, Answer) and forgotten.ok
    assert not worker.pending
    assert worker.materialized[("parent", 1)][1] > 0
    assert Path(local).read_bytes() == data
    returned = canonical_json.decode(reply.result_canonical_bytes)["report"]
    forwarded = worker.received_output(
        parent, msgspec.convert({**returned, "output_id": "report"}, OutputRow)
    )
    assert forwarded is not None and forwarded[0].read_bytes() == data
    changed = dict(returned, size_bytes=len(data) + 1)
    with pytest.raises(ValueError, match="exact received"):
        worker.received_output(
            parent, msgspec.convert({**changed, "output_id": "report"}, OutputRow)
        )
    worker.close("parent", 1)
    assert not Path(local).exists()
    assert not worker.pending
    assert not worker.materialized
    with pytest.raises(ValueError, match="exact received"):
        worker.received_output(
            parent, msgspec.convert({**returned, "output_id": "report"}, OutputRow)
        )


class ImageResult(msgspec.Struct):
    image: ImageAsset


def png_result(tmp_path: Path) -> tuple[Workspace, pb.ChildCallResult, bytes]:
    from PIL import Image

    from cozy_runtime.internal.worker.byte_outputs import roster

    workspace, spec, _, path = producing(tmp_path)
    Image.new("RGB", (8, 8), (12, 34, 56)).save(path, format="PNG")
    data = path.read_bytes()
    manifest, files = roster(path, tree=False, max_bytes=1 << 20)
    source = outputs.commit(
        workspace, "owner", "producer", 1, spec, "image", manifest, files, 1 << 20
    )
    hold = pb.NativeByteRetentionRequest(source=source, retention_id="sha256:" + "13" * 32)
    outputs.change_hold(workspace, "owner", hold, release=False)
    digest = "sha256:" + hashlib.sha256(data).hexdigest()
    reply = pb.ChildCallResult(
        state=pb.CHILD_CALL_STATE_SUCCEEDED,
        result_canonical_bytes=canonical_json.encode(
            {
                "image": {
                    "asset_ref": digest,
                    "kind": "image",
                    "digest": digest,
                    "size_bytes": len(data),
                    "media_type": "image/png",
                }
            }
        ),
        byte_result_grants=[
            pb.ChildByteResultGrant(
                output_id="image", source=source, retention_id=hold.retention_id
            )
        ],
    )
    path.unlink()
    return workspace, reply, data


def test_encoded_png_result_hydrates_as_existing_image_asset(tmp_path: Path) -> None:
    workspace, reply, data = png_result(tmp_path)
    rows = child_byte_results.materialize(workspace, "owner", reply, tmp_path / "parent")
    assert rows[0]["media_type"] == "image/png"
    result, _ = decode(
        canonical_json.decode(reply.result_canonical_bytes),
        ImageResult,
        rows,
        request_id="parent",
        guard=lambda: None,
        observation=None,
    )
    assert isinstance(result.image, ImageAsset)
    assert result.image.media_type == "image/png"
    assert result.image.read_bytes() == data


def test_native_grants_and_result_leaves_tolerate_producer_additions(tmp_path: Path) -> None:
    workspace, reply, data = png_result(tmp_path)
    value = canonical_json.decode(reply.result_canonical_bytes)
    value["image"]["future"] = {"label": "newer producer"}
    reply.result_canonical_bytes = canonical_json.encode(value)
    grants = child_byte_results.materialize(workspace, "owner", reply, tmp_path / "parent")
    answer = msgspec.to_builtins(CallState(ok=True, state="succeeded", byte_grants=tuple(grants)))
    answer["byte_grants"][0]["future"] = ["newer worker"]
    received = msgspec.convert(answer, CallState, strict=True)
    assert received.byte_grants == ({**grants[0], "future": ["newer worker"]},)
    result, _ = decode(
        value,
        ImageResult,
        received.byte_grants,
        request_id="parent",
        guard=lambda: None,
        observation=None,
    )
    assert result.image.read_bytes() == data


@pytest.mark.parametrize(("index", "field"), [(0, "length"), (1, "content_bytes")])
def test_native_grant_reader_rejects_boolean_byte_counts(
    tmp_path: Path, index: int, field: str
) -> None:
    workspace, reply, _ = retained(tmp_path)
    grants = child_byte_results.materialize(workspace, "owner", reply, tmp_path / "parent")
    answer = msgspec.to_builtins(CallState(ok=True, state="succeeded", byte_grants=tuple(grants)))
    answer["byte_grants"][index][field] = True
    received = msgspec.convert(answer, CallState, strict=True)
    with pytest.raises(msgspec.ValidationError, match="got `bool`"):
        decode(
            canonical_json.decode(reply.result_canonical_bytes),
            Result,
            received.byte_grants,
            request_id="parent",
            guard=lambda: None,
            observation=None,
        )


def test_file_grant_without_tree_byte_count_still_hydrates(tmp_path: Path) -> None:
    workspace, reply, data = png_result(tmp_path)
    grants = child_byte_results.materialize(workspace, "owner", reply, tmp_path / "parent")
    answer = msgspec.to_builtins(CallState(ok=True, state="succeeded", byte_grants=tuple(grants)))
    del answer["byte_grants"][0]["content_bytes"]
    received = msgspec.convert(answer, CallState, strict=True)
    result, _ = decode(
        canonical_json.decode(reply.result_canonical_bytes),
        ImageResult,
        received.byte_grants,
        request_id="parent",
        guard=lambda: None,
        observation=None,
    )
    assert result.image.read_bytes() == data


def test_encoded_png_mislabeled_as_jpeg_refuses_at_result_decode(tmp_path: Path) -> None:
    workspace, reply, data = png_result(tmp_path)
    value = canonical_json.decode(reply.result_canonical_bytes)
    value["image"]["media_type"] = "image/jpeg"
    reply.result_canonical_bytes = canonical_json.encode(value)
    rows = child_byte_results.materialize(workspace, "owner", reply, tmp_path / "parent")
    # Materialization corrects same-family MIME claims from the bytes. A native
    # result must still agree with its verified grant when the SDK hydrates it.
    assert rows[0]["media_type"] == "image/png"
    assert Path(rows[0]["local"]).read_bytes() == data
    with pytest.raises(ValueError, match="differs from its verified host grant"):
        decode(value, ImageResult, rows, request_id="parent", guard=lambda: None, observation=None)


def test_native_grant_reader_ignores_fields_unused_by_its_kind(tmp_path: Path) -> None:
    workspace, reply, data = retained(tmp_path)
    grants = child_byte_results.materialize(workspace, "owner", reply, tmp_path / "parent")
    answer = msgspec.to_builtins(CallState(ok=True, state="succeeded", byte_grants=tuple(grants)))
    answer["byte_grants"][0]["content_bytes"] = {"newer": "unused on file"}
    answer["byte_grants"][1]["media_type"] = ["unused on tree"]
    answer["byte_grants"][1]["length"] = {"newer": "unused on tree"}
    value = canonical_json.decode(reply.result_canonical_bytes)
    value["bundle"]["media_type"] = {"newer": "unused on tree result"}
    received = msgspec.convert(answer, CallState, strict=True)
    result, _ = decode(
        value,
        Result,
        received.byte_grants,
        request_id="parent",
        guard=lambda: None,
        observation=None,
    )
    assert result.report.read_bytes() == (result.bundle.path / "payload").read_bytes() == data


@pytest.mark.parametrize("kind,mime", [("audio", "audio/wav"), ("video", "image/png")])
def test_encoded_png_refuses_a_wrong_kind_or_media_family(
    tmp_path: Path, kind: str, mime: str
) -> None:
    workspace, reply, _ = png_result(tmp_path)
    value = canonical_json.decode(reply.result_canonical_bytes)
    value["image"].update(kind=kind, media_type=mime)
    reply.result_canonical_bytes = canonical_json.encode(value)
    with pytest.raises(WorkspaceRefusal, match="media type"):
        child_byte_results.materialize(workspace, "owner", reply, tmp_path / "parent")


def test_abort_reclaims_readonly_projections_without_following_storage_links(
    tmp_path: Path,
) -> None:
    original = tmp_path / "native-storage"
    original.mkdir()
    source = original / "payload"
    source.write_bytes(b"native bytes must survive")
    source.chmod(0o444)
    original.chmod(0o555)
    spool = tmp_path / "attempt"
    projected = spool / "child-results" / "0" / "0"
    nested = projected / "nested"
    nested.mkdir(parents=True)
    copy = nested / "payload"
    copy.write_bytes(source.read_bytes())
    copy.chmod(0o444)
    nested.chmod(0o555)
    projected.chmod(0o555)
    (spool / "storage-link").symlink_to(original, target_is_directory=True)
    (spool / "file-link").symlink_to(source)
    assert grants.abort_outputs(spool) == 3
    assert spool.is_dir() and list(spool.iterdir()) == []
    assert source.read_bytes() == b"native bytes must survive"
    assert source.stat().st_mode & 0o777 == 0o444
    assert original.stat().st_mode & 0o777 == 0o555
    original.chmod(0o755)  # the test's own fixture cleanup
