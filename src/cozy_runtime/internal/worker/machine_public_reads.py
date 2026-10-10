"""Versioned read-only machine API projections of Runtime's sole journal.

The agent opens SQLite with mode=ro and only selects these views. Views contain no
captured arguments, invocation/preparation documents, credentials, or control state.
Additive internal journal changes cannot widen the public data selected here.
"""

from __future__ import annotations

from .workspace import Journal

# Shared with Executions.retention_required; the machine projection spans every
# namespace and includes child attempts and native work that executions omit.
RETENTION_REQUIRED = "retention_waived=0 AND (state NOT IN ('succeeded','canceled') OR collected=0)"


def install(db: Journal) -> None:
    """An additive public read contract; old views remain readable across Runtime upgrades."""
    db.executescript(f"""
    CREATE VIEW IF NOT EXISTS machine_software_update_v1 AS
      SELECT execution_workspace_id AS workspace_id,
        (EXISTS(SELECT 1 FROM executions) OR EXISTS(SELECT 1 FROM attempts)
         OR EXISTS(SELECT 1 FROM native_calls)) AS accepted_work,
        NOT (EXISTS(SELECT 1 FROM executions WHERE {RETENTION_REQUIRED})
          OR EXISTS(SELECT 1 FROM attempts WHERE state IN ('accepted','running'))
          OR EXISTS(SELECT 1 FROM native_calls WHERE state<>'released')
          OR EXISTS(SELECT 1 FROM holds WHERE state<>'released')
          OR EXISTS(SELECT 1 FROM weights WHERE state='intent')
          OR EXISTS(SELECT 1 FROM input_tree_intakes WHERE state<>'released')) AS quiescent
      FROM workspace_state;
    CREATE VIEW IF NOT EXISTS machine_workspace_v1 AS
      SELECT execution_workspace_id AS workspace_id FROM workspace_state;
    CREATE VIEW IF NOT EXISTS machine_runs_v1 AS
      SELECT s.execution_workspace_id AS workspace_id, n.number AS run_number,
        e.request AS request_id, e.submission AS submission_id,
        e.ordinal AS attempt_ordinal, e.generation, e.state, e.collected,
        e.sequence, e.compacted_through, e.accepted_ms AS accepted_at_ms,
        CASE WHEN e.state IN ('succeeded','failed','paused','canceled') THEN
          (SELECT coalesce(max(v.at_ms),0) FROM execution_events v
           WHERE v.owner=e.owner AND v.request=e.request AND v.kind='outcome')
          ELSE 0 END AS finished_at_ms,
        e.worker_id, e.accepted_boot AS worker_boot_id, e.owner AS record_namespace
      FROM executions e JOIN execution_numbers n
        ON n.owner=e.owner AND n.request=e.request
      CROSS JOIN workspace_state s;
    CREATE VIEW IF NOT EXISTS machine_events_v1 AS
      SELECT s.execution_workspace_id AS workspace_id, n.number AS run_number,
        e.request AS request_id, e.sequence, e.ordinal AS attempt_ordinal,
        e.at_ms, e.kind, e.body AS body_json,
        coalesce(a.spec,x'') AS invocation_spec_digest,
        coalesce(a.outcome_id,'') AS outcome_id,
        coalesce(a.outcome_digest,x'') AS outcome_digest,
        coalesce(a.outcome,x'') AS outcome_json, e.owner AS record_namespace
      FROM execution_events e JOIN execution_numbers n
        ON n.owner=e.owner AND n.request=e.request
      LEFT JOIN attempts a ON e.kind='outcome'
        AND a.owner=e.owner AND a.request=e.request AND a.ordinal=e.ordinal
      CROSS JOIN workspace_state s;
    CREATE VIEW IF NOT EXISTS machine_outputs_v1 AS
      SELECT s.execution_workspace_id AS workspace_id, n.number AS run_number,
        b.request AS request_id, b.ordinal AS attempt_ordinal, b.slot,
        b.manifest_body AS manifest_json, b.content_bytes, b.state,
        b.owner AS record_namespace
      FROM byte_outputs b JOIN execution_numbers n
        ON n.owner=b.owner AND n.request=b.request
      CROSS JOIN workspace_state s;
    """)
