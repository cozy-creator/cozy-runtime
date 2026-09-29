"""The machine agent can read durable public history while Runtime is absent."""

from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from cozy_runtime.internal.worker.workspace import Workspace
from cozy_runtime.internal.worker.workspace_executions import Executions
from test_machine_execution import complete, offer


def test_public_views_preserve_outcomes_without_exposing_execution_authority(
    tmp_path: Path,
) -> None:
    workspace = Workspace(tmp_path / "store")
    executions = Executions(workspace)
    receipt = executions.submit(
        "owner",
        "submission",
        b"c" * 32,
        offer("request"),
        expected_execution_workspace_id=executions.workspace_id,
    )
    complete(executions, "request")
    executions.reconcile("owner", "request")
    path = workspace.directory / "journal.sqlite3"
    with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True) as db:
        db.row_factory = sqlite3.Row
        run = db.execute("SELECT * FROM machine_runs_v1").fetchone()
        assert run["run_number"] == receipt.number
        assert run["workspace_id"] == executions.workspace_id
        assert run["state"] == "succeeded"
        assert run["finished_at_ms"] > 0
        assert not {"capture", "offer", "preparation", "capture_document", "owner"} & set(
            run.keys()
        )
        events = db.execute("SELECT * FROM machine_events_v1 ORDER BY sequence").fetchall()
        terminal = events[-1]
        assert terminal["kind"] == "outcome"
        assert json.loads(terminal["body_json"])["state"] == "succeeded"
        outcome = executions.outcome("owner", "request")
        assert terminal["outcome_json"] == outcome.outcome_canonical_bytes
        assert terminal["invocation_spec_digest"] == outcome.invocation_spec_digest
        assert terminal["outcome_id"] == outcome.outcome_id
        assert terminal["outcome_digest"] == outcome.outcome_digest
        with pytest.raises(sqlite3.OperationalError):
            db.execute("DELETE FROM executions")


def test_supervisor_capability_probe_needs_no_machine_settings_or_cuda() -> None:
    answer = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from cozy_runtime.cli.runtime_worker import main; "
            "assert 'torch' not in sys.modules; "
            "raise SystemExit(main(['capabilities','--json']))",
        ],
        text=True,
        capture_output=True,
        check=True,
    )
    assert "machine-supervisor/1" in json.loads(answer.stdout)["capabilities"]
