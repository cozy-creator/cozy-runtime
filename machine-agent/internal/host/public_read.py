"""The guardian's read-only adapter. Only versioned public views are queried."""
import base64
import json
import sqlite3
import sys
from pathlib import Path

from cozy_runtime.protocol import documents
from cozy_runtime.protocol import worker_pb2 as pb


def state(row):
    return pb.MachineExecutionState(
        request_id=row["request_id"], attempt_ordinal=row["attempt_ordinal"],
        generation=row["generation"], state=row["state"], collected=bool(row["collected"]),
        sequence=row["sequence"], worker_id=row["worker_id"],
        worker_boot_id=row["worker_boot_id"], execution_workspace_id=row["workspace_id"],
        number=row["run_number"], accepted_at_ms=row["accepted_at_ms"], finished_at_ms=row["finished_at_ms"],
    )


RUN_COLUMNS = "workspace_id,run_number,request_id,attempt_ordinal,generation,state,collected,sequence,compacted_through,accepted_at_ms,finished_at_ms,worker_id,worker_boot_id"
db = sqlite3.connect(Path(sys.argv[1]).as_uri() + "?mode=ro", uri=True, timeout=2)
db.row_factory = sqlite3.Row
db.execute("PRAGMA query_only=ON")
db.execute("BEGIN")
request = json.loads(sys.argv[2])
workspace = db.execute("SELECT workspace_id FROM machine_workspace_v1").fetchone()[0]
expected = request.get("workspace", "")
if expected and expected != workspace:
    raise ValueError("execution_workspace_changed")
op = request["op"]
limit = min(max(int(request.get("limit") or 256), 1), 256)
if op == "workspace":
    result = pb.MachineExecutionWorkspace(execution_workspace_id=workspace, run_output_log=True)
elif op == "list":
    filters, params = ["run_number > ?"], [request.get("after", 0)]
    if request.get("before", 0):
        filters.append("run_number < ?")
        params.append(request["before"])
    states = request.get("states", [])
    if states:
        if len(states) > 32:
            raise ValueError("too many state filters")
        filters.append("state IN (" + ",".join("?" for _ in states) + ")")
        params.extend(states)
    order = "DESC" if request.get("newest") else "ASC"
    rows = db.execute("SELECT " + RUN_COLUMNS + " FROM machine_runs_v1 WHERE " + " AND ".join(filters) + " ORDER BY run_number " + order + " LIMIT ?", params + [limit])
    head = db.execute("SELECT coalesce(max(run_number),0) FROM machine_runs_v1").fetchone()[0]
    result = pb.MachineExecutionList(execution_workspace_id=workspace, head_number=head, executions=[state(row) for row in rows])
elif op in ("get", "events"):
    row = db.execute("SELECT " + RUN_COLUMNS + " FROM machine_runs_v1 WHERE request_id=?", (request["request"],)).fetchone()
    if row is None:
        raise ValueError("execution_not_found")
    if op == "get":
        result = state(row)
    else:
        result = pb.MachineExecutionEventPage(head_sequence=row["sequence"], compacted_through=row["compacted_through"], next_after=request.get("after", 0))
        rows = db.execute("SELECT request_id,sequence,attempt_ordinal,at_ms,kind,body_json,invocation_spec_digest,outcome_id,outcome_digest,outcome_json FROM machine_events_v1 WHERE request_id=? AND sequence>? ORDER BY sequence LIMIT ?", (request["request"], request.get("after", 0), limit))
        size = 0
        for item in rows:
            body = bytes(item["body_json"])
            if size and size + len(body) > 256 << 10:
                break
            size += len(body)
            event = result.events.add(sequence=item["sequence"], attempt_ordinal=item["attempt_ordinal"], at_ms=item["at_ms"], kind=item["kind"], body_canonical_bytes=body)
            if item["kind"] == "product":
                event.product.CopyFrom(documents.parse(body, pb.RunProduct))
            elif item["kind"] == "outcome":
                event.outcome.CopyFrom(pb.AttemptOutcome(request_id=item["request_id"], attempt_ordinal=item["attempt_ordinal"], invocation_spec_digest=bytes(item["invocation_spec_digest"]), outcome_id=item["outcome_id"], outcome_digest=bytes(item["outcome_digest"]), outcome_canonical_bytes=bytes(item["outcome_json"])))
            result.next_after = item["sequence"]
            if item["kind"] == "outcome":
                break
else:
    raise ValueError("unsupported public read")
print(base64.b64encode(result.SerializeToString()).decode("ascii"))
