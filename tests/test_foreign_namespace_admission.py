"""A namespace cut cannot silently duplicate another owner's retained work."""

from pathlib import Path

import pytest

from cozy_runtime.internal.worker.workspace import Workspace
from cozy_runtime.internal.worker.workspace_executions import Executions, ExecutionWorkspaceRefusal
from test_machine_execution import offer


def test_foreign_paused_work_blocks_only_new_supervisor_admission(tmp_path: Path) -> None:
    executions = Executions(Workspace(tmp_path / "store"))
    identity = executions.workspace_id
    owner = "cozy-local-client"
    accepted = executions.submit(
        owner, "already", b"c" * 32, offer("already"), expected_execution_workspace_id=identity
    )
    executions.submit(
        "hub-public:old",
        "foreign",
        b"c" * 32,
        offer("foreign"),
        expected_execution_workspace_id=identity,
    )
    with executions.workspace.locked() as db:
        db.execute("UPDATE executions SET state='paused' WHERE owner='hub-public:old'")
        db.execute("UPDATE attempts SET state='outcome' WHERE owner='hub-public:old'")
    assert (
        executions.submit(
            owner, "already", b"c" * 32, offer("already"), expected_execution_workspace_id=identity
        )
        == accepted
    )
    with pytest.raises(ExecutionWorkspaceRefusal) as failure:
        executions.submit(
            owner, "new", b"c" * 32, offer("new"), expected_execution_workspace_id=identity
        )
    assert failure.value.code == "execution_foreign_namespace_retained"
    with executions.workspace.locked() as db:
        assert (
            db.execute("SELECT state FROM executions WHERE owner='hub-public:old'").fetchone()[0]
            == "paused"
        )
        assert db.execute("SELECT count(*) FROM executions").fetchone()[0] == 2
        # Model an explicit release by the original owner, never migration deletion.
        db.execute("UPDATE executions SET retention_waived=1 WHERE owner='hub-public:old'")
        db.execute("UPDATE attempts SET state='released' WHERE owner='hub-public:old'")
    assert (
        executions.submit(
            owner, "new", b"c" * 32, offer("new"), expected_execution_workspace_id=identity
        ).request_id
        == "new"
    )
