"""Ordinary output custody is native and survives independent recipient ownership."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import tensorfs

from cozy_runtime import canonical_json
from cozy_runtime.internal.worker import workspace_byte_outputs as outputs
from cozy_runtime.internal.worker.workspace import Workspace, WorkspaceRefusal
from cozy_runtime.protocol import documents
from cozy_runtime.protocol import worker_pb2 as pb


def producing(tmp_path: Path) -> tuple[Workspace, bytes, bytes, Path]:
    workspace = Workspace(tmp_path / "store")
    raw, digest = documents.identity(pb.InvocationSpec(job=pb.JobInvocationSpec()))
    workspace.accept(
        "owner",
        pb.AttemptOffer(
            request_id="producer",
            attempt_ordinal=1,
            invocation_spec_digest=digest,
            invocation_spec_canonical_bytes=raw,
        ),
    )
    workspace.mark_running(
        "owner",
        pb.AttemptAccepted(
            request_id="producer",
            attempt_ordinal=1,
            invocation_spec_digest=digest,
        ),
    )
    path = tmp_path / "report.json"
    data = canonical_json.encode({"report": "observed" * 12000})
    path.write_bytes(data)
    manifest = canonical_json.encode(
        {
            "entries": [
                {
                    "path": "payload",
                    "kind": "file",
                    "blob": {
                        "sha256": hashlib.sha256(data).hexdigest(),
                        "length": len(data),
                    },
                }
            ]
        }
    )
    return workspace, digest, manifest, path


def test_real_tree_commit_replay_retention_gc_and_release(tmp_path: Path) -> None:
    workspace, spec, manifest, path = producing(tmp_path)
    data = path.read_bytes()
    source = outputs.commit(
        workspace, "owner", "producer", 1, spec, "report", manifest, [("payload", path)], 200000
    )
    path.unlink()
    reopened = Workspace(workspace.store_root)
    assert (
        outputs.commit(
            reopened, "owner", "producer", 1, spec, "report", manifest, [("payload", path)], 200000
        )
        == source
    )
    hold = pb.NativeByteRetentionRequest(source=source, retention_id="sha256:" + "aa" * 32)
    assert not outputs.change_hold(reopened, "owner", hold, release=False).released
    outcome_bytes, outcome_digest = documents.identity(
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
    reopened.outcome(
        "owner",
        pb.AttemptOutcome(
            request_id="producer",
            attempt_ordinal=1,
            invocation_spec_digest=spec,
            outcome_id="outcome",
            outcome_digest=outcome_digest,
            outcome_canonical_bytes=outcome_bytes,
        ),
    )
    ack = pb.AttemptOutcomeAck(
        request_id="producer",
        attempt_ordinal=1,
        invocation_spec_digest=spec,
        outcome_id="outcome",
        outcome_digest=outcome_digest,
    )
    reopened.acknowledge("owner", ack)
    tensorfs.gc(str(workspace.store_root))
    with outputs.leased(reopened, "owner", hold) as (_, lease, members):
        assert len(members) == 1
        actual = bytearray(len(data))
        lease.read_into("sha256:" + members[0]["blob"]["sha256"], len(data), 0, len(data), actual)
        assert actual == data
    assert outputs.change_hold(reopened, "owner", hold, release=True).released
    with pytest.raises(WorkspaceRefusal):
        outputs.change_hold(reopened, "owner", hold, release=False)
    with pytest.raises(WorkspaceRefusal), outputs.leased(reopened, "owner", hold):
        pytest.fail("released bytes became readable")


def test_changed_output_and_unowned_retention_refuse(tmp_path: Path) -> None:
    workspace, spec, manifest, path = producing(tmp_path)
    source = outputs.commit(
        workspace, "owner", "producer", 1, spec, "report", manifest, [("payload", path)], 200000
    )
    with pytest.raises(WorkspaceRefusal, match="changed"):
        outputs.commit(
            workspace,
            "owner",
            "producer",
            1,
            spec,
            "report",
            manifest.replace(b"payload", b"another"),
            [("another", path)],
            200000,
        )
    hold = pb.NativeByteRetentionRequest(source=source, retention_id="sha256:" + "bb" * 32)
    with pytest.raises(WorkspaceRefusal, match="not retained"):
        outputs.change_hold(workspace, "another-owner", hold, release=False)
    hold.source.content_bytes += 1
    with pytest.raises(WorkspaceRefusal):
        outputs.change_hold(workspace, "owner", hold, release=False)


def test_release_before_acquire_cannot_resurrect_recipient(tmp_path: Path) -> None:
    workspace, spec, manifest, path = producing(tmp_path)
    source = outputs.commit(
        workspace, "owner", "producer", 1, spec, "report", manifest, [("payload", path)], 200000
    )
    hold = pb.NativeByteRetentionRequest(source=source, retention_id="sha256:" + "cc" * 32)
    assert outputs.change_hold(workspace, "owner", hold, release=True).released
    with pytest.raises(WorkspaceRefusal, match="released"):
        outputs.change_hold(workspace, "owner", hold, release=False)
