"""Cache interruption and cancellation cross actual processes and native roots."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest
import tensorfs

from cozy_runtime import canonical_json
from cozy_runtime.internal.weights_sink import weights_transaction_id
from cozy_runtime.internal.worker import workspace_memo
from cozy_runtime.internal.worker.workspace import Workspace
from cozy_runtime.protocol import documents
from cozy_runtime.protocol import worker_pb2 as pb

RECORD_PROCESS = """
import base64,os,signal,sys
from pathlib import Path
from cozy_runtime.internal.worker import workspace_memo
from cozy_runtime.internal.worker.workspace import Workspace
from cozy_runtime.protocol import worker_pb2 as pb
mode=sys.argv[3];changes=0
def schedule(frame,event,arg):
    global changes
    pause=False
    if (mode=='after-first' and event=='return' and frame.f_code.co_name=='native'
            and frame.f_code.co_filename.endswith('/workspace.py')):
        pause=True
    if (mode=='before-second' and event=='call' and frame.f_code.co_name=='change'
            and frame.f_code.co_filename.endswith('/derived_retention.py')):
        changes+=1;pause=changes==2
    if pause:
        sys.settrace(None)
        print('PAUSED',flush=True)
        os.kill(os.getpid(),signal.SIGSTOP)
    return schedule
sys.settrace(schedule)
try:
    workspace=Workspace(Path(sys.argv[1]))
    call=pb.RecordOperationResultCall.FromString(base64.b64decode(sys.argv[2]))
    if sys.argv[4]=='record':workspace_memo.record(workspace,'owner',call)
    else:workspace_memo.lookup(workspace,'owner',pb.LookupOperationCall(
        computation_digest=call.computation_digest,consumer_request_id='recipient'))
except Exception as error:
    print('REFUSED '+str(error),flush=True)
else:
    print('RECORDED',flush=True)
"""


def native_outputs(
    root: Path,
) -> tuple[
    tensorfs.Store, Workspace, list[pb.DerivedResultReleaseRequest], pb.RecordOperationResultCall
]:
    store = tensorfs.Store.init(root / "store")
    workspace = Workspace(Path(store.root))
    invocation = documents.canonical_bytes(
        pb.InvocationSpec(
            job=pb.JobInvocationSpec(
                installation_id="local-" + "11" * 16, job_descriptor_id="sha256:" + "12" * 32
            ),
            outputs=[
                {
                    "output_id": slot,
                    "mime_type": "application/vnd.cozy.model-manifest",
                    "max_bytes": 2048,
                }
                for slot in ("a", "b")
            ],
        )
    )
    spec = hashlib.sha256(invocation).digest()
    workspace.accept(
        "owner",
        pb.AttemptOffer(
            request_id="producer",
            attempt_ordinal=1,
            invocation_spec_digest=spec,
            invocation_spec_canonical_bytes=invocation,
        ),
    )
    plain = next(digest for alias, digest in tensorfs.seed_digests() if alias == "plain/1")
    targets = {
        "model": {
            "drop": [],
            "add": {
                "weight": {
                    "logical_dtype": "f32",
                    "shape": [512],
                    "encoding": plain,
                    "parts": {"value": {"dtype": "f32", "shape": [512]}},
                }
            },
        }
    }
    fingerprint = "sha256:" + "42" * 32
    order = [("model", "weight")]
    declaration = store.derived_declaration(
        {}, targets, {}, order, 2048, work_fingerprint=fingerprint
    )
    declared = hashlib.sha256(declaration).digest()
    references, releases = [], []
    for slot in ("a", "b"):
        transaction = weights_transaction_id("owner", "producer", documents.spell(spec), slot)
        workspace.begin_weights(
            "owner",
            pb.WeightsIntentFrame(
                request_id="producer",
                attempt_ordinal=1,
                invocation_spec_digest=spec,
                output_slot=slot,
                weights_transaction_id=transaction,
                tensorfs_declaration_digest=declared,
                tensorfs_declaration_canonical_bytes=declaration,
            ),
        )
        writer = store.begin_derived(
            transaction, 1, {}, targets, {}, order, 2048, work_fingerprint=fingerprint
        )
        writer.add_part("model", "weight", "value", io.BytesIO(b"\x07" * 2048))
        native = writer.commit()
        native_bytes = canonical_json.encode(native)
        native_digest = hashlib.sha256(native_bytes).digest()
        raw, digest = documents.identity(
            pb.WeightsReceipt(
                owner_authority_scope="owner",
                request_id="producer",
                invocation_spec_digest=documents.spell(spec),
                output_slot=slot,
                weights_transaction_id=transaction,
                tensorfs_receipt_digest=documents.spell(native_digest),
                tensorfs_receipt_canonical_bytes=native_bytes,
            )
        )
        reference = pb.WeightsReceiptRef(
            weights_receipt_digest=digest, weights_receipt_canonical_bytes=raw
        )
        references.append(reference)
        workspace.record_receipt(
            "owner",
            pb.WeightsReceiptFrame(
                request_id="producer",
                attempt_ordinal=1,
                invocation_spec_digest=spec,
                output_slot=slot,
                weights_transaction_id=transaction,
                writer_epoch=1,
                tensorfs_declaration_digest=declared,
                weights_receipt=reference,
                manifest=pb.Ref(
                    digest=bytes.fromhex(native["manifest"]["sha256"]),
                    length=native["manifest"]["length"],
                ),
            ),
        )
        releases.append(
            pb.DerivedResultReleaseRequest(
                weights_transaction_id=transaction, tensorfs_receipt_digest=native_digest
            )
        )
    raw, digest = documents.identity(
        pb.AttemptOutcomeBody(
            request_id="producer",
            attempt_ordinal=1,
            invocation_spec_digest=documents.spell(spec),
            status=pb.OUTCOME_STATUS_SUCCEEDED,
            result=pb.ResultEnvelope(inline_result=b'{"value":7}'),
            weights_receipts=references,
        )
    )
    workspace.outcome(
        "owner",
        pb.AttemptOutcome(
            request_id="producer",
            attempt_ordinal=1,
            invocation_spec_digest=spec,
            outcome_id="out-producer",
            outcome_digest=digest,
            outcome_canonical_bytes=raw,
        ),
    )
    call = pb.RecordOperationResultCall(
        request_id="producer",
        attempt_ordinal=1,
        invocation_spec_digest=spec,
        outcome_id="out-producer",
        outcome_digest=digest,
        computation_digest=b"\x71" * 32,
    )
    return store, workspace, releases, call


@pytest.mark.parametrize("mode", ["after-first", "before-second"])
@pytest.mark.parametrize("operation", ["record", "lookup"])
def test_all_output_holds_survive_interruption_or_late_acquisition(
    tmp_path: Path, mode: str, operation: str
) -> None:
    store, workspace, releases, call = native_outputs(tmp_path)
    if operation == "lookup":
        assert workspace_memo.record(workspace, "owner", call).recorded
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            RECORD_PROCESS,
            str(store.root),
            base64.b64encode(call.SerializeToString()).decode(),
            mode,
            operation,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert process.stdout is not None
        assert process.stdout.readline().strip() == "PAUSED"
        _, stopped = os.waitpid(process.pid, os.WUNTRACED)
        assert os.WIFSTOPPED(stopped)
        if mode == "after-first":
            process.kill()
            process.wait(timeout=10)
        with workspace.locked() as db:
            assert db.execute("SELECT count(*) FROM holds").fetchone()[0] == (
                4 if operation == "lookup" else 2
            )
            table = "operation_lookups" if operation == "lookup" else "operation_cache"
            assert db.execute("SELECT state FROM " + table).fetchone()[0] == "retaining"
            held = json.loads(db.execute("SELECT body FROM " + table).fetchone()[0])["holds"]
        for release in releases:
            workspace.release_result("owner", release)
        workspace.acknowledge(
            "owner",
            pb.AttemptOutcomeAck(
                request_id="producer",
                attempt_ordinal=1,
                invocation_spec_digest=call.invocation_spec_digest,
                outcome_id=call.outcome_id,
                outcome_digest=call.outcome_digest,
                retain_work=False,
            ),
        )
        if operation == "lookup":
            for hold in held:
                workspace.retain(
                    "owner",
                    pb.DerivedRetentionRequest(
                        weights_transaction_id=hold["transaction_id"],
                        tensorfs_receipt_digest=documents.raw(hold["native_receipt_digest"]),
                        retention_id=hold["retention_id"],
                    ),
                    release=True,
                )
        pruned = workspace_memo.prune(workspace, "owner")
        assert pruned.removed_entries == 1
        if mode == "before-second":
            os.kill(process.pid, signal.SIGCONT)
            stdout, stderr = process.communicate(timeout=20)
            assert process.returncode == 0, stderr
            assert "REFUSED " in stdout and "RECORDED" not in stdout
        with workspace.locked() as db:
            assert {row[0] for row in db.execute("SELECT state FROM holds")} == {"released"}
        assert not workspace_memo.lookup(
            workspace,
            "owner",
            pb.LookupOperationCall(
                computation_digest=b"\x79" * 32, consumer_request_id="unrelated"
            ),
        ).found
        tensorfs.gc(str(store.root))
        assert not store.contains(tensorfs.object_id(b"\x07" * 2048))
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=10)
        for pipe in (process.stdout, process.stderr):
            if pipe is not None:
                pipe.close()


def scalar(
    workspace: Workspace,
    request: str,
    value: int,
    key: bytes,
    *,
    evidence: str = "",
    owner: str = "owner",
) -> bool:
    raw = documents.canonical_bytes(
        pb.InvocationSpec(
            job=pb.JobInvocationSpec(
                installation_id="local-" + "11" * 16, job_descriptor_id="sha256:" + "12" * 32
            )
        )
    )
    spec = hashlib.sha256(raw).digest()
    workspace.accept(
        owner,
        pb.AttemptOffer(
            request_id=request,
            attempt_ordinal=1,
            invocation_spec_digest=spec,
            invocation_spec_canonical_bytes=raw,
        ),
    )
    terminal, digest = documents.identity(
        pb.AttemptOutcomeBody(
            request_id=request,
            attempt_ordinal=1,
            invocation_spec_digest=documents.spell(spec),
            status=pb.OUTCOME_STATUS_SUCCEEDED,
            result=pb.ResultEnvelope(
                inline_result=canonical_json.encode({"value": value, "evidence": evidence})
            ),
        )
    )
    workspace.outcome(
        owner,
        pb.AttemptOutcome(
            request_id=request,
            attempt_ordinal=1,
            invocation_spec_digest=spec,
            outcome_id="out-" + request,
            outcome_digest=digest,
            outcome_canonical_bytes=terminal,
        ),
    )
    return workspace_memo.record(
        workspace,
        owner,
        pb.RecordOperationResultCall(
            request_id=request,
            attempt_ordinal=1,
            invocation_spec_digest=spec,
            outcome_id="out-" + request,
            outcome_digest=digest,
            computation_digest=key,
        ),
    ).recorded


def test_capacity_pressure_preserves_contradiction_tombstones(tmp_path: Path) -> None:
    workspace = Workspace(tmp_path / "store")
    key = b"\x78" * 32
    assert scalar(workspace, "first", 7, key)
    assert not scalar(workspace, "conflicting", 8, key)
    for index in range(workspace_memo.MAX_ENTRIES + 1):
        assert scalar(
            workspace,
            "other-" + str(index),
            index,
            hashlib.sha256(("other" + str(index)).encode()).digest(),
        )
    with workspace.locked() as db:
        row = db.execute("SELECT state,body FROM operation_cache WHERE key=?", (key,)).fetchone()
        assert row["state"] == "disabled"
        assert json.loads(row["body"])["conflicting_source"]["request_id"] == "conflicting"
        assert (
            db.execute("SELECT count(*) FROM operation_cache WHERE state='ready'").fetchone()[0]
            == workspace_memo.MAX_ENTRIES
        )
    assert not scalar(workspace, "third", 9, key)
    assert not workspace_memo.lookup(
        workspace,
        "owner",
        pb.LookupOperationCall(computation_digest=key, consumer_request_id="after-pressure"),
    ).found


def test_missing_old_recipient_does_not_clear_disabled_mapping(tmp_path: Path) -> None:
    store, workspace, _, call = native_outputs(tmp_path)
    assert workspace_memo.record(workspace, "owner", call).recorded
    lookup = pb.LookupOperationCall(
        computation_digest=call.computation_digest, consumer_request_id="already-held"
    )
    assert workspace_memo.lookup(workspace, "owner", lookup).found
    assert not scalar(workspace, "contradictory", 8, call.computation_digest)
    # Actual disk-loss fixture: the prior recipient becomes unavailable, but
    # cleaning that old result must not erase the computation's contradiction.
    payload = hashlib.sha256(b"\x07" * 2048).hexdigest()
    paths = list(Path(store.root).glob("**/" + payload))
    assert len(paths) == 1
    paths[0].unlink()
    assert not workspace_memo.lookup(workspace, "owner", lookup).found
    with workspace.locked() as db:
        assert (
            db.execute(
                "SELECT state FROM operation_cache WHERE key=?", (call.computation_digest,)
            ).fetchone()[0]
            == "disabled"
        )


def test_disabled_compaction_preserves_history_without_scalar_lookup_copies(tmp_path: Path) -> None:
    workspace = Workspace(tmp_path / "store")
    key = b"\x79" * 32
    assert scalar(workspace, "original", 7, key, evidence="x" * 16384)
    lookup = pb.LookupOperationCall(computation_digest=key, consumer_request_id="owned-result")
    before = workspace_memo.lookup(workspace, "owner", lookup)
    assert before.found and len(before.source.outcome_canonical_bytes) > 16384
    assert not scalar(workspace, "contradiction", 8, key, evidence="y" * 16384)
    with workspace.locked() as db:
        row = db.execute("SELECT * FROM operation_cache WHERE key=?", (key,)).fetchone()
        assert row["state"] == "disabled" and len(row["body"]) < 1024
        payload = json.loads(row["body"])
        assert payload["holds"] == []
        assert payload["source"]["outcome_digest"] == documents.spell(before.source.outcome_digest)
        assert payload["conflicting_source"]["request_id"] == "contradiction"
        assert "outcome_canonical_bytes" not in payload["source"]
        assert "outcome_canonical_bytes" not in payload["conflicting_source"]
    # Scalar delivery has no native effect to replay. Creator owns any accepted
    # result; a fresh query must obey the now-disabled mapping.
    assert not workspace_memo.lookup(workspace, "owner", lookup).found
    assert any(
        item.outcome_canonical_bytes == before.source.outcome_canonical_bytes
        for item in workspace.retained_outcomes("owner")
    )
    assert not workspace_memo.lookup(
        workspace,
        "owner",
        pb.LookupOperationCall(computation_digest=key, consumer_request_id="fresh-result"),
    ).found
