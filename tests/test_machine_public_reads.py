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


def test_public_reads_keep_legacy_request_collisions_in_their_record_namespace(
    tmp_path: Path,
) -> None:
    workspace = Workspace(tmp_path / "store")
    executions = Executions(workspace)
    namespaces = ("legacy-A", "legacy-B", "cozy-local-client")
    numbers = {}
    for namespace in namespaces:
        receipt = executions.submit(
            namespace,
            "submission",
            b"c" * 32,
            offer("same-request"),
            expected_execution_workspace_id=executions.workspace_id,
        )
        numbers[namespace] = receipt.number
        executions.record(namespace, "same-request", "log", {"value": namespace})
        # These are historical settled rows, not a new admission beside unfinished
        # foreign-namespace work (which the current G8 boundary correctly refuses).
        with workspace.locked() as db:
            db.execute(
                "UPDATE executions SET state='canceled',retention_waived=1 WHERE owner=?",
                (namespace,),
            )
            db.execute("UPDATE attempts SET state='released' WHERE owner=?", (namespace,))
    assert len(set(numbers.values())) == len(namespaces)
    with workspace.locked() as db:
        # Emulate an existing pre-G8 journal, not current admission authority.
        db.execute("UPDATE executions SET state='queued',retention_waived=0")
        db.execute("UPDATE attempts SET state='accepted'")

    path = workspace.directory / "journal.sqlite3"
    with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True) as db:
        # Exactly the domain of the agent's authenticated Runtime calls. Historical
        # identities remain present and are never flattened into that namespace.
        for namespace in namespaces:
            assert db.execute(
                "SELECT run_number FROM machine_runs_v1 WHERE record_namespace=? AND request_id=?",
                (namespace, "same-request"),
            ).fetchall() == [(numbers[namespace],)]
            events = db.execute(
                "SELECT body_json FROM machine_events_v1 "
                "WHERE record_namespace=? AND request_id=? AND kind='log' ORDER BY sequence",
                (namespace, "same-request"),
            ).fetchall()
            assert [json.loads(row[0]) for row in events] == [{"value": namespace}]
        assert db.execute("SELECT count(*) FROM executions").fetchone()[0] == 3


def test_software_update_view_keeps_rental_first_use_after_collection(tmp_path: Path) -> None:
    workspace = Workspace(tmp_path / "store")
    executions = Executions(workspace)
    with workspace.locked() as db:
        row = db.execute("SELECT * FROM machine_software_update_v1").fetchone()
        assert not row["accepted_work"] and row["quiescent"]
    executions.submit(
        "owner",
        "submission",
        b"c" * 32,
        offer("request"),
        expected_execution_workspace_id=executions.workspace_id,
    )
    with workspace.locked() as db:
        db.execute("UPDATE executions SET state='canceled',collected=1,retention_waived=1")
        db.execute("UPDATE attempts SET state='released'")
        row = db.execute("SELECT * FROM machine_software_update_v1").fetchone()
        assert row["quiescent"]
        assert row["accepted_work"]  # collection never restores a fresh rental
        assert row["workspace_id"]


def test_software_update_view_checks_native_work_and_waived_retention(tmp_path: Path) -> None:
    workspace = Workspace(tmp_path / "store")
    executions = Executions(workspace)
    executions.submit(
        "owner",
        "submission",
        b"c" * 32,
        offer("request"),
        expected_execution_workspace_id=executions.workspace_id,
    )
    with workspace.locked() as db:
        db.execute("UPDATE executions SET state='failed',retention_waived=1")
        db.execute("UPDATE attempts SET state='released'")
        assert db.execute("SELECT quiescent FROM machine_software_update_v1").fetchone()[0]
        db.execute(
            "INSERT INTO native_calls(owner,parent_request,call_index,parent_ordinal,"
            "parent_spec,intent_digest,service_id,operation,accepted,native_owner,state) "
            "VALUES('owner','request',0,0,x'',x'','native','operation',x'','owner','executing')"
        )
        assert not db.execute("SELECT quiescent FROM machine_software_update_v1").fetchone()[0]
        db.execute("UPDATE native_calls SET state='released'")
        assert db.execute("SELECT quiescent FROM machine_software_update_v1").fetchone()[0]
        db.execute("UPDATE attempts SET state='running'")
        assert not db.execute("SELECT quiescent FROM machine_software_update_v1").fetchone()[0]
