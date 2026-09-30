"""Typed native member projection, independent forwarding and refusal boundaries."""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any

import msgspec
import pytest
import tensorfs
from PIL import Image

from cozy_runtime.author import AssetBound, Context, ImageAsset, Tree
from cozy_runtime.author._calls import _Broker, _current, _input_ref
from cozy_runtime.author._decode import MediaDecoder
from cozy_runtime.author._errors import CapabilityError
from cozy_runtime.author._executor_requests import Reply, Request, TreeMember
from cozy_runtime.author._services import Attempt
from cozy_runtime.internal.worker import (
    byte_inputs,
    byte_outputs,
    grants,
    machine_byte_inputs,
    machine_models,
    tree_members,
    workspace_byte_outputs,
)
from cozy_runtime.internal.worker.attempts import AttemptRecord
from cozy_runtime.internal.worker.calls import Calls, _Received
from cozy_runtime.internal.worker.workspace import Workspace, WorkspaceRefusal
from cozy_runtime.internal.worker.workspace_executions import Executions
from cozy_runtime.protocol import documents
from cozy_runtime.protocol import worker_pb2 as pb
from durable_seam import seam
from test_workspace_byte_outputs import producing


def prepared(tmp_path: Path, *, root_input: bool = False) -> tuple[Any, ...]:
    workspace, spec, _, _ = producing(tmp_path)
    Executions(workspace)
    source = tmp_path / "source"
    source.mkdir()
    Image.new("RGB", (4, 3), "red").save(source / "frame.png")
    (source / "unrelated.txt").write_text("not part of the forwarded file")
    manifest, files = byte_outputs.roster(source, tree=True, max_bytes=1000)
    native = workspace_byte_outputs.commit(
        workspace, "owner", "producer", 1, spec, "bundle", manifest, files, 1000
    )
    held = machine_models.retain_bytes(workspace, "owner", "producer", "input/bundle", native)
    hold = pb.NativeByteRetentionRequest(source=held.source, retention_id=held.retention_id)
    local = tmp_path / "spool" / "bundle"
    byte_inputs.copy_retained(workspace, "owner", hold, local)
    parent = AttemptRecord("producer", 1, spec, {}, state="running", spool=tmp_path / "spool")
    calls = Calls(lambda *_: pb.ChildCallResult(), workspace=workspace, owner=lambda: "owner")
    digest = documents.spell(native.manifest.digest)
    if root_input:
        parent.grant.inputs["bundle"] = grants.BoundInput(
            "bundle",
            "",
            native.manifest.digest,
            native.manifest.length,
            byte_inputs.TREE_MIME,
            0,
            hold,
        )
        parent.trees[digest] = (local, digest)
    else:
        calls.received[("producer", 1)] = {
            "tree": _Received(
                {
                    "kind": "tree",
                    "digest": digest,
                    "length": native.manifest.length,
                    "local": str(local),
                    "media_type": byte_inputs.TREE_MIME,
                    "content_bytes": native.content_bytes,
                },
                hold,
            )
        }
    frame = {
        "tree": digest,
        "path": "frame.png",
        "asset_kind": "image",
        "max_bytes": 1000,
        "media_types": ["image/png"],
    }
    return workspace, parent, calls, hold, local, frame


@pytest.mark.parametrize("root_input", [False, True])
def test_member_projects_without_spool_copy_and_child_survives_parent_gc(
    tmp_path: Path,
    root_input: bool,
) -> None:
    workspace, parent, calls, hold, local, frame = prepared(tmp_path, root_input=root_input)
    inode = (local / "frame.png").stat().st_ino

    def project(request: Request) -> Reply:
        assert isinstance(request, TreeMember)
        return tree_members.project(calls, parent, request)

    broker = _Broker("producer", {}, seam(project))
    broker.bind(Context("producer", time.monotonic() + 30))
    tree = Tree(frame["tree"], digest=frame["tree"], root=local, attempt="producer")
    tree._member_token = broker.grant_token
    token = _current.set(broker)
    try:
        asset = tree.member(
            "frame.png",
            ImageAsset,
            bound=AssetBound(
                max_bytes=1000,
                max_decoded_bytes=1000,
                media_types=("image/png",),
            ),
        )
        assert asset.read_bytes() == (local / "frame.png").read_bytes()
        assert asset._local is not None
        assert asset._local == local / "frame.png" and asset._local.stat().st_ino == inode
        decoder = MediaDecoder(Attempt("producer", tmp_path / "decode"), active=lambda: False)
        decoded = decoder.decode_image(asset)
        assert (decoded.width, decoded.height) == (4, 3)
        assert calls.materialized == {}
        again = tree.member("frame.png", ImageAsset, bound=AssetBound(max_bytes=1000))
        assert again.digest == asset.digest and _input_ref(asset, "producer") == asset.digest
        with workspace.locked() as db:
            assert db.execute("SELECT count(*) FROM byte_outputs").fetchone()[0] == 2
        schema = {
            "fields": [
                {"name": "image", "type": {"asset": "image"}, "asset_bound": {"max_bytes": 1000}}
            ]
        }
        bindings, access = machine_byte_inputs.inputs(
            workspace,
            "owner",
            parent,
            "child",
            schema,
            {"image": asset.digest},
            calls,
        )
        assert access[0].native_tree.source.content_bytes == len(asset.read_bytes())
        assert access[0].native_tree.source.manifest.digest != hold.source.manifest.digest
        broker.closed = True
        with pytest.raises(CapabilityError, match="escaped_handle"):
            asset.read_bytes()
        with pytest.raises(CapabilityError, match="escaped_handle"):
            decoder.decode_image(asset)
    finally:
        _current.reset(token)
    machine_models.release(workspace, "owner", "producer")
    with workspace.locked() as db:
        roots = [row[0] for row in db.execute("SELECT id FROM byte_outputs").fetchall()]
        db.execute("UPDATE byte_outputs SET state='releasing'")
    for root in roots:
        workspace_byte_outputs.release(workspace, "owner", root)
    byte_inputs.remove_checkout(local)
    tensorfs.gc(str(workspace.store_root))
    spec = pb.InvocationSpec(inputs=bindings)
    _, digest = documents.identity(spec)
    grant = grants.bind(
        documents.body(spec),
        pb.DeliveryGrant(
            invocation_spec_digest=digest,
            inputs=access,
        ),
        digest,
    )
    hydrated = grants.hydrate_inputs(
        grant,
        grants.Authorizer(),
        spool=tmp_path / "child",
        workspace=workspace,
        owner="owner",
    )
    assert (
        "sha256:" + hashlib.sha256(hydrated["image"].local.read_bytes()).hexdigest()
        == bindings[0].digest
    )
    with Image.open(hydrated["image"].local) as image:
        assert image.size == (4, 3)
    machine_models.release(workspace, "owner", "child")


@pytest.mark.parametrize(
    "mutation",
    [
        "../escape",
        "/absolute",
        "frame.png/",
        "absent",
        "size",
        "kind",
        "media",
        "changed",
        "symlink",
        "released",
        "foreign",
        "cancelled",
        "missing_file",
    ],
)
def test_member_refuses_before_minting_custody(tmp_path: Path, mutation: str) -> None:
    workspace, parent, calls, hold, local, frame = prepared(tmp_path)
    if mutation in ("../escape", "/absolute", "frame.png/", "absent"):
        frame["path"] = mutation
    elif mutation == "size":
        frame["max_bytes"] = 1
    elif mutation == "kind":
        frame["asset_kind"] = "video"
    elif mutation == "media":
        frame["media_types"] = ["image/jpeg"]
    elif mutation == "changed":
        (local / "frame.png").chmod(0o644)
        (local / "frame.png").write_bytes(b"changed")
    elif mutation == "missing_file":
        local.chmod(0o755)
        (local / "frame.png").unlink()
    elif mutation == "symlink":
        local.chmod(0o755)
        (local / "frame.png").unlink()
        (local / "frame.png").symlink_to(tmp_path / "source" / "frame.png")
    elif mutation == "released":
        workspace_byte_outputs.change_hold(workspace, "owner", hold, release=True)
    elif mutation == "foreign":
        parent.request_id = "foreign"
    else:
        parent.canceling = "requested"
    reply = tree_members.project(calls, parent, msgspec.convert(frame, TreeMember))
    assert not reply.ok, reply
    with workspace.locked() as db:
        assert db.execute("SELECT count(*) FROM byte_outputs").fetchone()[0] == 1


def terminal_bundle(
    workspace: Any, parent: Any, hold: Any, extra: pb.OutputEntry | None = None
) -> None:
    raw, digest = documents.identity(
        pb.AttemptOutcomeBody(
            request_id=parent.request_id,
            attempt_ordinal=parent.attempt,
            invocation_spec_digest=documents.spell(parent.digest),
            status=pb.OUTCOME_STATUS_SUCCEEDED,
            execution_started=True,
            output_manifest=pb.OutputManifest(
                outputs=[pb.OutputEntry(output_id="bundle", native_tree=hold.source)]
                + ([extra] if extra is not None else [])
            ),
        )
    )
    workspace.outcome(
        "owner",
        pb.AttemptOutcome(
            request_id=parent.request_id,
            attempt_ordinal=parent.attempt,
            invocation_spec_digest=parent.digest,
            outcome_id="finished",
            outcome_digest=digest,
            outcome_canonical_bytes=raw,
        ),
    )


@pytest.mark.parametrize("root_input", [False, True])
def test_projected_members_are_not_required_terminal_outputs(
    tmp_path: Path,
    root_input: bool,
) -> None:
    workspace, parent, calls, hold, _, frame = prepared(tmp_path, root_input=root_input)
    assert tree_members.project(calls, parent, msgspec.convert(frame, TreeMember)).ok
    terminal_bundle(Workspace(workspace.store_root), parent, hold)
    with workspace.locked() as db:
        assert db.execute("SELECT state FROM attempts").fetchone()[0] == "outcome"
        assert db.execute("SELECT count(*) FROM byte_outputs").fetchone()[0] == 2


@pytest.mark.parametrize("mutation", ["missing_binding", "unfinished_hold", "wrong_slot"])
def test_projection_exclusion_requires_the_exact_completed_hold(
    tmp_path: Path,
    mutation: str,
) -> None:
    workspace, parent, calls, hold, _, frame = prepared(tmp_path)
    assert tree_members.project(calls, parent, msgspec.convert(frame, TreeMember)).ok
    with workspace.locked() as db:
        if mutation == "missing_binding":
            db.execute("DELETE FROM execution_model_holds WHERE path LIKE 'runtime.tree_member.%'")
        elif mutation == "unfinished_hold":
            db.execute("UPDATE holds SET state='retaining'")
        else:
            db.execute(
                "UPDATE byte_outputs SET slot='runtime.tree_member.changed' "
                "WHERE slot LIKE 'runtime.tree_member.%'"
            )
    with pytest.raises(WorkspaceRefusal):
        terminal_bundle(workspace, parent, hold)


@pytest.mark.parametrize("hash_shaped", [False, True])
def test_ordinary_output_can_use_a_projection_like_name(tmp_path: Path, hash_shaped: bool) -> None:
    workspace, parent, _, hold, local, _ = prepared(tmp_path)
    manifest, files = byte_outputs.roster(local / "frame.png", tree=False, max_bytes=1000)
    slot = "runtime.tree_member." + (
        hashlib.sha256(manifest).hexdigest() if hash_shaped else "frame"
    )
    native = workspace_byte_outputs.commit(
        workspace, "owner", "producer", 1, parent.digest, slot, manifest, files, 1000
    )
    terminal_bundle(workspace, parent, hold, pb.OutputEntry(output_id=slot, native_tree=native))


def test_projection_terminal_replays_after_retained_collection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def producing_execution(path: Path) -> tuple[Any, ...]:
        workspace = Workspace(path / "store")
        executions = Executions(workspace)
        raw, spec = documents.identity(pb.InvocationSpec(job=pb.JobInvocationSpec()))
        executions.submit(
            "owner",
            "submission",
            b"c" * 32,
            pb.AttemptOffer(
                request_id="producer",
                attempt_ordinal=1,
                invocation_spec_digest=spec,
                invocation_spec_canonical_bytes=raw,
            ),
            expected_execution_workspace_id=executions.workspace_id,
        )
        workspace.mark_running(
            "owner",
            pb.AttemptAccepted(
                request_id="producer",
                attempt_ordinal=1,
                invocation_spec_digest=spec,
            ),
        )
        return workspace, spec, b"", path / "unused"

    monkeypatch.setattr("test_tree_member_assets.producing", producing_execution)
    workspace, parent, calls, hold, _, frame = prepared(tmp_path)
    assert tree_members.project(calls, parent, msgspec.convert(frame, TreeMember)).ok
    terminal_bundle(workspace, parent, hold)
    executions = Executions(workspace)
    # The execution unit projects a recorded outcome onto its execution (`reconcile`).
    executions.reconcile("owner", "producer")
    terminal = executions.collect("owner", "producer")
    workspace.outcome("owner", terminal)
    ack = pb.AttemptOutcomeAck(
        request_id=terminal.request_id,
        attempt_ordinal=terminal.attempt_ordinal,
        invocation_spec_digest=terminal.invocation_spec_digest,
        outcome_id=terminal.outcome_id,
        outcome_digest=terminal.outcome_digest,
        retain_work=True,
    )
    assert executions.acknowledge_collection("owner", ack).collected
    workspace.acknowledge("owner", ack)
    with workspace.locked() as db:
        assert db.execute("SELECT state FROM attempts").fetchone()[0] == "outcome"
        assert {row[0] for row in db.execute("SELECT state FROM holds")} == {"released"}
    reopened = Workspace(workspace.store_root)
    assert Executions(reopened).collect("owner", "producer") == terminal
    reopened.outcome("owner", terminal)
    changed = pb.AttemptOutcome()
    changed.CopyFrom(terminal)
    changed.outcome_id = "different-terminal"
    with pytest.raises(WorkspaceRefusal):
        reopened.outcome("owner", changed)
