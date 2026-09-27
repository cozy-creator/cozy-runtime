"""Acknowledged results belong to Creator; workspace keeps only custody facts."""

from pathlib import Path

import pytest

from cozy_runtime.internal.worker import workspace_finalize, workspace_memo
from cozy_runtime.internal.worker.workspace import Workspace, WorkspaceRefusal
from cozy_runtime.protocol import documents
from cozy_runtime.protocol import worker_pb2 as pb
from test_workspace_custody import produced, retention
from test_workspace_memo import completed


def test_scalar_ack_retires_execution_bodies_without_erasing_memo_result(tmp_path: Path) -> None:
    workspace = Workspace(tmp_path / "store")
    invocation, spec = documents.identity(
        pb.InvocationSpec(
            job=pb.JobInvocationSpec(
                installation_id="local-" + "11" * 16, job_descriptor_id="12" * 32
            )
        )
    )
    workspace.accept(
        "owner",
        pb.AttemptOffer(
            request_id="scalar",
            attempt_ordinal=1,
            invocation_spec_digest=spec,
            invocation_spec_canonical_bytes=invocation,
        ),
    )
    body, digest = documents.identity(
        pb.AttemptOutcomeBody(
            request_id="scalar",
            attempt_ordinal=1,
            invocation_spec_digest=documents.spell(spec),
            status=pb.OUTCOME_STATUS_SUCCEEDED,
            execution_started=True,
            result=pb.ResultEnvelope(inline_result=b'{"value":"' + b"x" * 20000 + b'"}'),
        )
    )
    workspace.outcome(
        "owner",
        pb.AttemptOutcome(
            request_id="scalar",
            attempt_ordinal=1,
            invocation_spec_digest=spec,
            outcome_id="outcome",
            outcome_digest=digest,
            outcome_canonical_bytes=body,
        ),
    )
    call = pb.RecordOperationResultCall(
        computation_digest=b"k" * 32,
        request_id="scalar",
        attempt_ordinal=1,
        invocation_spec_digest=spec,
        outcome_id="outcome",
        outcome_digest=digest,
    )
    assert workspace_memo.record(workspace, "owner", call).recorded
    ack = pb.AttemptOutcomeAck(
        request_id="scalar",
        attempt_ordinal=1,
        invocation_spec_digest=spec,
        outcome_id="outcome",
        outcome_digest=digest,
    )
    workspace.acknowledge("owner", ack)
    workspace.acknowledge("owner", ack)
    with workspace.locked() as db:
        assert db.execute("SELECT count(*) FROM attempts").fetchone()[0] == 0
    hit = workspace_memo.lookup(
        workspace,
        "owner",
        pb.LookupOperationCall(
            computation_digest=call.computation_digest, consumer_request_id="fresh-scalar"
        ),
    )
    assert hit.found and hit.source.outcome_canonical_bytes == body
    ack.retain_work = True
    with pytest.raises(WorkspaceRefusal):
        workspace.acknowledge("owner", ack)


def test_live_native_custody_keeps_identity_without_acknowledged_execution_bodies(
    tmp_path: Path,
) -> None:
    _, workspace, receipt = produced(tmp_path)
    call = completed(workspace, receipt)
    with workspace.locked() as db:
        invocation = db.execute("SELECT invocation FROM attempts").fetchone()[0]
        receipt_digest = db.execute("SELECT receipt_digest FROM weights").fetchone()[0]
    workspace.acknowledge(
        "owner",
        pb.AttemptOutcomeAck(
            request_id=call.request_id,
            attempt_ordinal=1,
            invocation_spec_digest=call.invocation_spec_digest,
            outcome_id=call.outcome_id,
            outcome_digest=call.outcome_digest,
        ),
    )
    with workspace.locked() as db:
        row = db.execute("SELECT invocation,outcome,result_schema FROM attempts").fetchone()
        assert tuple(row) == (b"", b"", b"")
    assert workspace.retain("owner", retention(receipt, 121)).manifest.digest
    adopted = workspace_finalize.finalize(
        workspace,
        "owner",
        pb.WeightsFinalizeRequest(
            request_id=call.request_id,
            invocation_spec_digest=call.invocation_spec_digest,
            invocation_spec_canonical_bytes=invocation,
            owner_authority_scope="owner",
            output_slot="model",
            weights_receipt_digest=receipt_digest,
            disposition=pb.WEIGHTS_FINALIZE_DISPOSITION_ADOPT,
            scratch_root_id="retirement/owned",
        ),
    )
    assert adopted.outcome == pb.WEIGHTS_FINALIZE_OUTCOME_ADOPTED
