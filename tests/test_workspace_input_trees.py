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
