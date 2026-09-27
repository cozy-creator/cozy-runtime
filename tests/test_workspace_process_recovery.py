"""A dead Runtime can be retried; another live Runtime is never reclaimed."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

import pytest
import tensorfs

from cozy_runtime.internal.worker import workspace_recovery
from cozy_runtime.internal.worker.workspace import Workspace, WorkspaceRefusal
from cozy_runtime.protocol import documents
from cozy_runtime.protocol import worker_pb2 as pb

CHILD = r"""
import base64, hashlib, io, json, sys
from pathlib import Path
import tensorfs
from cozy_runtime.internal.weights_sink import weights_transaction_id
from cozy_runtime.internal.worker.workspace import Workspace
from cozy_runtime.protocol import documents,worker_pb2 as pb
root=Path(sys.argv[1]);store=tensorfs.Store.ensure(root);workspace=Workspace(root)
invocation,spec=documents.identity(pb.InvocationSpec(job=pb.JobInvocationSpec(
    installation_id='local-'+'11'*16,job_descriptor_id='sha256:'+'12'*32),
    outputs=[pb.OutputBinding(output_id='model',mime_type='application/vnd.cozy.model-manifest',max_bytes=4096)]))
workspace.accept('owner',pb.AttemptOffer(request_id='interrupted',attempt_ordinal=1,
    invocation_spec_digest=spec,invocation_spec_canonical_bytes=invocation))
transaction=weights_transaction_id('owner','interrupted',documents.spell(spec),'model')
plain=next(value for alias,value in tensorfs.seed_digests() if alias=='plain/1')
tensor={'logical_dtype':'f32','shape':[1],'encoding':plain,'parts':{'value':{'dtype':'f32','shape':[1]}}}
targets={'model':{'drop':[],'add':{'a':tensor,'b':tensor}}};order=[('model','a'),('model','b')]
fingerprint='sha256:'+'42'*32
declaration=store.derived_declaration({},targets,{},order,8,work_fingerprint=fingerprint)
row=workspace.begin_weights('owner',pb.WeightsIntentFrame(request_id='interrupted',attempt_ordinal=1,
    invocation_spec_digest=spec,output_slot='model',weights_transaction_id=transaction,
    tensorfs_declaration_digest=hashlib.sha256(declaration).digest(),tensorfs_declaration_canonical_bytes=declaration))
writer=store.begin_derived(transaction,row['epoch'],{},targets,{},order,8,work_fingerprint=fingerprint)
writer.add_part('model','a','value',io.BytesIO(b'1234'))
head=writer.checkpoint('interrupted','model')
workspace.checkpoint('owner',pb.WeightsCheckpointFrame(request_id='interrupted',attempt_ordinal=1,
    invocation_spec_digest=spec,output_slot='model',weights_transaction_id=transaction,
    writer_epoch=row['epoch'],tensorfs_declaration_digest=hashlib.sha256(declaration).digest(),
    checkpoint=pb.CheckpointRef(head=pb.Ref(digest=documents.raw(head['head']),length=head['head_length']),
        plan_digest=documents.raw(head['plan_digest']),index=head['index'],bytes=head['bytes'])))
if sys.argv[2]=='committed':
    writer.add_part('model','b','value',io.BytesIO(b'5678'))
    writer.commit() # simulate death before the normal receipt callback
print(json.dumps({'invocation':base64.b64encode(invocation).decode(),
    'declaration':base64.b64encode(declaration).decode(),'transaction':transaction,
    'targets':targets,'order':order,'fingerprint':fingerprint,'head':head}),flush=True)
sys.stdin.readline()
"""


@pytest.mark.parametrize("state", ["partial", "committed"])
def test_dead_process_recovers_partial_or_unreported_complete_native_work(
    tmp_path: Path, state: str
) -> None:
    root = tmp_path / "store"
    child = subprocess.Popen(
        [sys.executable, "-c", CHILD, str(root), state],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert child.stdout is not None
        line = child.stdout.readline()
        assert line, child.stderr.read() if child.stderr is not None else "no child output"
        facts = json.loads(line)
        workspace = Workspace(root)
        assert workspace.retained_outcomes("owner") == []
        before = workspace.weights_row("owner", facts["transaction"])
        assert before["epoch"] == 1
        child.kill()
        child.wait(timeout=10)
        outcomes = workspace.retained_outcomes("owner")
        assert len(outcomes) == 1
        body = documents.read(outcomes[0].outcome_canonical_bytes, pb.AttemptOutcomeBody)
        assert body["status"] == pb.OUTCOME_STATUS_ABANDONED
        assert body["cause"]["code"] == pb.CAUSE_CODE_EXECUTOR_INVALIDATED
        assert body["execution_started"]
        assert Workspace(root).retained_outcomes("owner") == outcomes
        invocation = base64.b64decode(facts["invocation"])
        spec = hashlib.sha256(invocation).digest()
        declaration = base64.b64decode(facts["declaration"])
        workspace.accept(
            "owner",
            pb.AttemptOffer(
                request_id="interrupted",
                attempt_ordinal=2,
                invocation_spec_digest=spec,
                invocation_spec_canonical_bytes=invocation,
            ),
        )
        successor = workspace.begin_weights(
            "owner",
            pb.WeightsIntentFrame(
                request_id="interrupted",
                attempt_ordinal=2,
                invocation_spec_digest=spec,
                output_slot="model",
                weights_transaction_id=facts["transaction"],
                tensorfs_declaration_digest=hashlib.sha256(declaration).digest(),
                tensorfs_declaration_canonical_bytes=declaration,
            ),
        )
        store = tensorfs.Store.open(str(root))
        if state == "partial":
            assert successor["epoch"] == 2
            assert successor["checkpoint"] == before["checkpoint"]
            resumed = store.begin_derived(
                facts["transaction"],
                2,
                {},
                facts["targets"],
                {},
                [tuple(value) for value in facts["order"]],
                8,
                work_fingerprint=facts["fingerprint"],
            )
            assert resumed.completed_parts() == [("model", "a", "value")]
            resumed.add_part("model", "b", "value", io.BytesIO(b"5678"))
            assert resumed.commit()["transaction_id"] == facts["transaction"]
        else:
            assert len(body["weights_receipts"]) == 1
            assert successor["receipt"]
            assert store.derived_lookup(facts["transaction"])["state"] == "committed"
    finally:
        if child.poll() is None:
            child.kill()
        child.communicate(timeout=10)


def test_live_originating_process_does_not_authorize_replacement_writer(tmp_path: Path) -> None:
    root = tmp_path / "store"
    child = subprocess.Popen(
        [sys.executable, "-c", CHILD, str(root), "partial"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert child.stdout is not None
        facts = json.loads(child.stdout.readline())
        workspace = Workspace(root)
        invocation = base64.b64decode(facts["invocation"])
        spec = hashlib.sha256(invocation).digest()
        declaration = base64.b64decode(facts["declaration"])
        workspace.accept(
            "owner",
            pb.AttemptOffer(
                request_id="interrupted",
                attempt_ordinal=2,
                invocation_spec_digest=spec,
                invocation_spec_canonical_bytes=invocation,
            ),
        )
        with pytest.raises(WorkspaceRefusal, match="previous writer"):
            workspace.begin_weights(
                "owner",
                pb.WeightsIntentFrame(
                    request_id="interrupted",
                    attempt_ordinal=2,
                    invocation_spec_digest=spec,
                    output_slot="model",
                    weights_transaction_id=facts["transaction"],
                    tensorfs_declaration_digest=hashlib.sha256(declaration).digest(),
                    tensorfs_declaration_canonical_bytes=declaration,
                ),
            )
        assert workspace.retained_outcomes("owner") == []
        assert workspace.weights_row("owner", facts["transaction"])["epoch"] == 1
    finally:
        child.kill()
        child.communicate(timeout=10)


def test_slow_recovery_cannot_disrupt_an_already_admitted_successor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "store"
    child = subprocess.Popen(
        [sys.executable, "-c", CHILD, str(root), "partial"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert child.stdout is not None
        facts = json.loads(child.stdout.readline())
        child.kill()
        child.communicate(timeout=10)
        entered, finish = threading.Event(), threading.Event()
        errors: list[Exception] = []
        native = workspace_recovery.native_result

        def delayed(workspace: Workspace, row: dict[str, Any]) -> pb.WeightsReceiptFrame | None:
            if threading.current_thread().name == "slow-recovery":
                entered.set()
                assert finish.wait(5)
            return native(workspace, row)

        monkeypatch.setattr(workspace_recovery, "native_result", delayed)

        def recover() -> None:
            try:
                Workspace(root).retained_outcomes("owner")
            except Exception as exc:
                errors.append(exc)

        thread = threading.Thread(target=recover, name="slow-recovery")
        thread.start()
        resumed = None
        try:
            assert entered.wait(5)
            workspace = Workspace(root)
            assert len(workspace.retained_outcomes("owner")) == 1
            invocation = base64.b64decode(facts["invocation"])
            spec = hashlib.sha256(invocation).digest()
            declaration = base64.b64decode(facts["declaration"])
            workspace.accept(
                "owner",
                pb.AttemptOffer(
                    request_id="interrupted",
                    attempt_ordinal=2,
                    invocation_spec_digest=spec,
                    invocation_spec_canonical_bytes=invocation,
                ),
            )
            workspace.begin_weights(
                "owner",
                pb.WeightsIntentFrame(
                    request_id="interrupted",
                    attempt_ordinal=2,
                    invocation_spec_digest=spec,
                    output_slot="model",
                    weights_transaction_id=facts["transaction"],
                    tensorfs_declaration_digest=hashlib.sha256(declaration).digest(),
                    tensorfs_declaration_canonical_bytes=declaration,
                ),
            )
            store = tensorfs.Store.open(str(root))
            resumed = store.begin_derived(
                facts["transaction"],
                2,
                {},
                facts["targets"],
                {},
                [tuple(value) for value in facts["order"]],
                8,
                work_fingerprint=facts["fingerprint"],
            )
        finally:
            finish.set()
            thread.join(5)
        assert not thread.is_alive() and errors == []
        assert store.derived_lookup(facts["transaction"])["writer_session_id"] == 2
        assert workspace.weights_row("owner", facts["transaction"])["ordinal"] == 2
        assert resumed is not None
        resumed.fence()
    finally:
        if child.poll() is None:
            child.kill()
        child.communicate(timeout=10)
