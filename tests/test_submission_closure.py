"""Real journal races: negative lookup is not authority to duplicate a submission."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import pytest

from cozy_runtime.internal.worker.workspace import Workspace, WorkspaceRefusal
from cozy_runtime.internal.worker.workspace_executions import Executions, ExecutionWorkspaceRefusal
from test_machine_execution import offer


def test_closed_absence_survives_restart_and_rejects_delayed_submit(tmp_path: Path) -> None:
    executions = Executions(Workspace(tmp_path / "store"))
    identity = executions.workspace_id
    assert executions.accepted("owner", "submission", "root", identity) is None
    assert executions.close_submission("owner", "submission", "root", identity) is None
    reopened = Executions(Workspace(tmp_path / "store"))
    assert reopened.close_submission("owner", "submission", "root", identity) is None
    with pytest.raises(ExecutionWorkspaceRefusal, match="closed"):
        reopened.submit(
            "owner", "submission", b"c" * 32, offer(), expected_execution_workspace_id=identity
        )
    with pytest.raises(ExecutionWorkspaceRefusal, match="closed"):
        reopened.accepted("owner", "submission", "root", identity)
    for submission, request in [("submission", "different"), ("different", "root")]:
        with pytest.raises(WorkspaceRefusal, match="identity changed"):
            reopened.close_submission("owner", submission, request, identity)
    with pytest.raises(ExecutionWorkspaceRefusal, match="workspace"):
        reopened.close_submission("owner", "submission", "root", "another-journal")
    assert reopened.close_submission("another-owner", "submission", "root", identity) is None


def test_close_returns_existing_receipt_without_fabricating_cancellation(tmp_path: Path) -> None:
    executions = Executions(Workspace(tmp_path / "store"))
    identity = executions.workspace_id
    accepted = executions.submit(
        "owner", "submission", b"c" * 32, offer(), expected_execution_workspace_id=identity
    )
    assert executions.close_submission("owner", "submission", "root", identity) == accepted
    assert executions.accepted("owner", "submission", "root", identity) == accepted
    assert (
        executions.submit(
            "owner", "submission", b"c" * 32, offer(), expected_execution_workspace_id=identity
        )
        == accepted
    )
    assert executions.offer("owner", "root").request_id == "root"


@pytest.mark.parametrize("index", range(12))
def test_close_and_accept_have_one_serialized_winner(tmp_path: Path, index: int) -> None:
    executions = Executions(Workspace(tmp_path / "store"))
    identity = executions.workspace_id
    start = Barrier(2)
    submission, request = f"submission-{index}", f"request-{index}"

    def submit() -> object:
        start.wait()
        try:
            return executions.submit(
                "owner",
                submission,
                b"c" * 32,
                offer(request),
                expected_execution_workspace_id=identity,
            )
        except ExecutionWorkspaceRefusal as exc:
            assert exc.code == "execution_submission_closed"
            return None

    def close() -> object:
        start.wait()
        return executions.close_submission("owner", submission, request, identity)

    with ThreadPoolExecutor(max_workers=2) as pool:
        submitted, closed = pool.submit(submit), pool.submit(close)
        assert submitted.result() == closed.result()
