"""Real native input commits, replay, cancellation and exact independent custody."""

from __future__ import annotations

import hashlib
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest
import tensorfs

from cozy_runtime import canonical_json
from cozy_runtime.internal.worker import machine_models, workspace_byte_outputs
from cozy_runtime.internal.worker.workspace import Workspace, WorkspaceBusy, WorkspaceRefusal
from cozy_runtime.internal.worker.workspace_input_trees import receive
from cozy_runtime.protocol import documents
from cozy_runtime.protocol import worker_pb2 as pb


def intake(
    request: str = "input-root", paths: tuple[str, ...] = ("nested/report", "copy", "empty")
) -> tuple[pb.InputTreeImportHeader, list[pb.InputTreeImportFrame], bytes]:
    data = b"immutable root input\n" * 8192
    payloads = {hashlib.sha256(data).hexdigest(): data, hashlib.sha256(b"").hexdigest(): b""}
    members: list[dict[str, Any]] = [
        {
            "path": path,
            "kind": "file",
            "blob": {
                "sha256": hashlib.sha256(b"" if path == "empty" else data).hexdigest(),
                "length": 0 if path == "empty" else len(data),
            },
        }
        for path in sorted(paths)
    ]
    body = canonical_json.encode({"entries": members})
    header = pb.InputTreeImportHeader(
        request_id=request,
        input_id="files",
        manifest=pb.Ref(digest=hashlib.sha256(body).digest(), length=len(body)),
        manifest_canonical_bytes=body,
        content_bytes=sum(row["blob"]["length"] for row in members),
    )
    used = {row["blob"]["sha256"] for row in members}
    frames = [
        pb.InputTreeImportFrame(
            blob=pb.InputTreeImportBlob(
                object=pb.Ref(digest=bytes.fromhex(digest), length=len(raw)), data=raw
            )
        )
        for digest, raw in payloads.items()
        if digest in used
    ]
    frames.append(pb.InputTreeImportFrame(commit=pb.InputTreeImportCommit()))
    return header, frames, data


def read(workspace: Workspace, result: Any, data: bytes) -> None:
    request = pb.NativeByteRetentionRequest(source=result.source, retention_id=result.retention_id)
    with workspace_byte_outputs.leased(workspace, "owner", request) as (_, lease, members):
        assert len(members) == 3
        target = bytearray(len(data))
        lease.read_into(
            "sha256:" + hashlib.sha256(data).hexdigest(), len(data), 0, len(data), target
        )
        assert target == data


def test_input_native_commit_replay_abort_and_independent_recipient(tmp_path: Path) -> None:
    workspace = Workspace(tmp_path)
    header, frames, data = intake()
    result = receive(workspace, "owner", header, frames)
    assert result.source.content_bytes == len(data) * 2 and not result.released
    read(workspace, result, data)
    replay = receive(workspace, "owner", header, frames[-1:])
    assert replay == result
    with workspace.locked() as db:
        assert all(
            db.execute(f"SELECT count(*) FROM {table}").fetchone()[0] == 0
            for table in ("attempts", "executions", "byte_outputs")
        )
        assert db.execute("SELECT count(*) FROM input_tree_intakes").fetchone()[0] == 1
    recipient = machine_models.retain_bytes(
        workspace, "owner", "accepted-root", "input/files", result.source
    )
    assert recipient.retention_id != result.retention_id
    aborted = receive(
        workspace,
        "owner",
        header,
        [pb.InputTreeImportFrame(commit=pb.InputTreeImportCommit(abort=True))],
    )
    assert aborted.released and aborted.source == result.source
    assert (
        receive(
            workspace,
            "owner",
            header,
            [pb.InputTreeImportFrame(commit=pb.InputTreeImportCommit(abort=True))],
        )
        == aborted
    )
    tensorfs.gc(str(tmp_path))
    read(workspace, recipient, data)
    with pytest.raises(WorkspaceRefusal, match="cannot be reopened"):
        receive(workspace, "owner", header, frames)
    with pytest.raises(WorkspaceRefusal):
        workspace_byte_outputs.change_hold(
            workspace,
            "other",
            pb.NativeByteRetentionRequest(source=result.source, retention_id=result.retention_id),
            release=False,
        )


def test_input_eof_keeps_replayable_identity_without_native_custody(tmp_path: Path) -> None:
    workspace = Workspace(tmp_path)
    header, frames, _ = intake()
    with pytest.raises(WorkspaceRefusal, match="final commit"):
        receive(workspace, "owner", header, frames[:-1])
    with workspace.locked() as db:
        row = db.execute("SELECT id,state,receipt FROM input_tree_intakes").fetchone()
        assert row["state"] == "receiving" and not row["receipt"]
    assert tensorfs.Store.open(str(tmp_path)).tree_root(row["id"]) is None
    changed = pb.InputTreeImportHeader()
    changed.CopyFrom(header)
    changed.content_bytes += 1
    with pytest.raises(WorkspaceRefusal):
        receive(workspace, "owner", changed, frames)
    assert not receive(workspace, "owner", header, frames).released


def test_abort_tombstone_fences_concurrent_stream_before_commit(tmp_path: Path) -> None:
    workspace = Workspace(tmp_path)
    header, frames, _ = intake()
    paused, resume = threading.Event(), threading.Event()

    def stream() -> Any:
        yield frames[0]
        paused.set()
        assert resume.wait(10), "test did not release its own input stream"
        yield from frames[1:]

    with ThreadPoolExecutor(max_workers=1) as pool:
        running = pool.submit(receive, workspace, "owner", header, stream())
        assert paused.wait(10), "native intake did not reach its controlled pause"
        with pytest.raises(WorkspaceBusy):
            receive(
                workspace,
                "owner",
                header,
                [pb.InputTreeImportFrame(commit=pb.InputTreeImportCommit(abort=True))],
            )
        resume.set()
        with pytest.raises(WorkspaceRefusal, match="aborted"):
            running.result()
    result = receive(
        workspace,
        "owner",
        header,
        [pb.InputTreeImportFrame(commit=pb.InputTreeImportCommit(abort=True))],
    )
    assert result.released and not result.HasField("source")
    with pytest.raises(WorkspaceRefusal, match="cannot be reopened"):
        receive(workspace, "owner", header, frames)
    with workspace.locked() as db:
        assert db.execute("SELECT count(*) FROM holds").fetchone()[0] == 0


@pytest.mark.parametrize(
    "paths",
    [("../outside",), ("/absolute",), ("a/./b",), ("a\\b",), ("a", "a"), ("a", "a/b"), (".",)],
)
def test_input_paths_refuse_before_intake_reservation(
    tmp_path: Path, paths: tuple[str, ...]
) -> None:
    workspace = Workspace(tmp_path)
    header, frames, _ = intake(paths=paths)
    with pytest.raises(WorkspaceRefusal):
        receive(workspace, "owner", header, frames)
    with workspace.locked() as db:
        assert db.execute("SELECT count(*) FROM input_tree_intakes").fetchone()[0] == 0


@pytest.mark.parametrize("failure", ["unknown", "digest", "offset", "missing", "extra", "frame"])
def test_input_stream_refuses_unclosed_or_changed_objects(tmp_path: Path, failure: str) -> None:
    workspace = Workspace(tmp_path)
    header, frames, _ = intake()
    if failure == "unknown":
        frames[0].blob.object.digest = b"x" * 32
    elif failure == "digest":
        frames[0].blob.data = b"x" * len(frames[0].blob.data)
    elif failure == "offset":
        frames[0].blob.offset = 1
    elif failure == "missing":
        frames = frames[1:]
    elif failure == "extra":
        frames.append(pb.InputTreeImportFrame(commit=pb.InputTreeImportCommit()))
    else:
        frames[0].blob.data = b"x" * ((1 << 20) + 1)
    with pytest.raises(WorkspaceRefusal):
        receive(workspace, "owner", header, frames)
    with workspace.locked() as db:
        assert db.execute("SELECT count(*) FROM holds").fetchone()[0] == 0
        assert (
            db.execute("SELECT count(*) FROM input_tree_intakes WHERE receipt!=x''").fetchone()[0]
            == 0
        )


@pytest.mark.parametrize("accept_first", [False, True])
def test_root_acceptance_and_intake_abort_have_one_durable_order(
    tmp_path: Path, accept_first: bool
) -> None:
    from cozy_runtime.internal.worker import machine_byte_inputs
    from cozy_runtime.internal.worker.workspace_executions import Executions

    workspace = Workspace(tmp_path)
    executions = Executions(workspace)
    header, frames, data = intake()
    imported = receive(workspace, "owner", header, frames)
    source = imported.source
    spec = pb.InvocationSpec(
        job=pb.JobInvocationSpec(
            installation_id="local-" + "11" * 16, job_descriptor_id="sha256:" + "12" * 32
        ),
        inputs=[
            pb.InputBinding(
                input_id="files",
                digest=documents.spell(source.manifest.digest),
                length=source.manifest.length,
                kind_mime=workspace_byte_outputs.TREE_MIME,
            )
        ],
    )
    raw, digest = documents.identity(spec)
    submitted = pb.AttemptOffer(
        request_id=header.request_id,
        attempt_ordinal=1,
        invocation_spec_digest=digest,
        invocation_spec_canonical_bytes=raw,
        grant=pb.DeliveryGrant(
            invocation_spec_digest=digest,
            inputs=[
                pb.InputAccess(
                    input_id="files",
                    native_tree=pb.NativeByteRetentionRequest(
                        source=source, retention_id=imported.retention_id
                    ),
                )
            ],
        ),
    )
    machine_byte_inputs.retain_root(
        workspace,
        "owner",
        submitted,
        {"fields": [{"name": "files", "type": {"input": "tree"}}]},
        {"files": documents.spell(source.manifest.digest)},
    )
    own = pb.NativeByteRetentionResult(
        source=source,
        retention_id=machine_models.byte_retention_id(
            "owner", header.request_id, "input/files", source
        ),
    )
    if accept_first:
        receipt = executions.submit(
            "owner",
            "submission",
            b"c" * 32,
            submitted,
            expected_execution_workspace_id=executions.workspace_id,
        )
    receive(
        workspace,
        "owner",
        header,
        [pb.InputTreeImportFrame(commit=pb.InputTreeImportCommit(abort=True))],
    )
    tensorfs.gc(str(tmp_path))
    if accept_first:
        assert (
            executions.submit(
                "owner",
                "submission",
                b"c" * 32,
                submitted,
                expected_execution_workspace_id=executions.workspace_id,
            )
            == receipt
        )
        read(workspace, own, data)
        dispatch = executions.offer("owner", header.request_id)
        machine_byte_inputs.bind_root_grant("owner", dispatch)
        assert dispatch.grant.inputs[0].native_tree.retention_id == own.retention_id
        assert submitted.grant.inputs[0].native_tree.retention_id == imported.retention_id
    else:
        with pytest.raises(WorkspaceRefusal, match="aborted"):
            executions.submit(
                "owner",
                "submission",
                b"c" * 32,
                submitted,
                expected_execution_workspace_id=executions.workspace_id,
            )
        with workspace.locked() as db:
            held = db.execute("SELECT state FROM holds WHERE id=?", (own.retention_id,)).fetchone()
            assert held["state"] == "released"
            assert db.execute("SELECT count(*) FROM executions").fetchone()[0] == 0


def test_serving_root_dispatches_from_the_submitters_native_hold(tmp_path: Path) -> None:
    from cozy_runtime.internal.worker import grants, machine_byte_inputs
    from cozy_runtime.internal.worker.workspace_executions import Executions

    workspace = Workspace(tmp_path)
    executions = Executions(workspace)
    header, frames, data = intake(paths=("payload",))
    imported = receive(workspace, "owner", header, frames)
    digest = "sha256:" + hashlib.sha256(data).hexdigest()
    spec = pb.InvocationSpec(
        serving=pb.ServingInvocationSpec(
            entrypoint_binding_digest="sha256:" + "13" * 32,
            attempt_binding_id="sha256:" + "13" * 32,
            bindings_digest="sha256:" + "14" * 32,
        ),
        inputs=[
            pb.InputBinding(
                input_id="files", digest=digest, length=len(data), kind_mime="text/plain"
            )
        ],
    )
    raw, identity = documents.identity(spec)
    submitted = pb.AttemptOffer(
        request_id=header.request_id,
        attempt_ordinal=1,
        invocation_spec_digest=identity,
        invocation_spec_canonical_bytes=raw,
        grant=pb.DeliveryGrant(
            invocation_spec_digest=identity,
            inputs=[
                pb.InputAccess(
                    input_id="files",
                    native_tree=pb.NativeByteRetentionRequest(
                        source=imported.source, retention_id=imported.retention_id
                    ),
                )
            ],
        ),
    )
    executions.submit(
        "owner",
        "submission",
        b"c" * 32,
        submitted,
        expected_execution_workspace_id=executions.workspace_id,
    )

    def hydrate(spool: str) -> dict[str, Any]:
        dispatch = executions.offer("owner", header.request_id)
        machine_byte_inputs.bind_root_grant("owner", dispatch)
        bound = grants.bind(spec_of(dispatch), dispatch.grant, dispatch.invocation_spec_digest)
        return grants.hydrate_inputs(
            bound, grants.Authorizer(), spool=tmp_path / spool, workspace=workspace, owner="owner"
        )

    def spec_of(offer: pb.AttemptOffer) -> Any:
        return documents.read(offer.invocation_spec_canonical_bytes, pb.InvocationSpec)

    assert hydrate("first")["files"].local.read_bytes() == data
    receive(
        workspace,
        "owner",
        header,
        [pb.InputTreeImportFrame(commit=pb.InputTreeImportCommit(abort=True))],
    )
    with pytest.raises(grants.GrantRefusal, match="no exact received retention"):
        hydrate("released")


def test_a_later_input_reuses_only_its_owners_held_objects(tmp_path: Path) -> None:
    workspace = Workspace(tmp_path)
    header, frames, data = intake("first-root")
    receive(workspace, "owner", header, frames)
    # The same bytes for the owner's next root: the commit alone, no object sent.
    again, _, _ = intake("second-root")
    read(workspace, receive(workspace, "owner", again, frames[-1:]), data)
    # Another owner never learns what this owner holds: it sends its bytes.
    other, _, _ = intake("third-root")
    with pytest.raises(WorkspaceRefusal, match="input_objects_absent"):
        receive(workspace, "stranger", other, frames[-1:])
    assert receive(workspace, "stranger", other, frames).source.content_bytes == len(data) * 2
