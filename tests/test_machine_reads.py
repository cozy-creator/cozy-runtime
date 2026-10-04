"""A journal numbers the roots an older Runtime accepted when it opens, and an acknowledged
product stays until the store's GC evicts it.
"""

from __future__ import annotations

import gc
import hashlib
import sqlite3
from pathlib import Path

import pytest
import tensorfs

from cozy_runtime.internal import fill
from cozy_runtime.internal.worker import machine_models, products
from cozy_runtime.internal.worker import workspace_byte_outputs as outputs
from cozy_runtime.internal.worker.workspace import NativeHold, Workspace
from cozy_runtime.internal.worker.workspace_executions import Executions
from cozy_runtime.protocol import worker_pb2 as pb
from test_machine_execution import offer
from test_workspace_byte_outputs import producing


def test_roots_an_older_runtime_accepted_are_numbered_when_the_journal_opens(
    tmp_path: Path,
) -> None:
    executions = Executions(Workspace(tmp_path / "store"))
    for name in ("a", "b", "c"):
        executions.submit(
            "owner",
            name,
            b"c" * 32,
            offer(name),
            expected_execution_workspace_id=executions.workspace_id,
        )
    del executions
    gc.collect()
    # Runtime 0.18.75 and older never number: its journal has no numbers table at all.
    with sqlite3.connect(tmp_path / "store" / ".cozy-workspace" / "journal.sqlite3") as db:
        db.execute("DROP TABLE execution_numbers")
    reopened = Executions(Workspace(tmp_path / "store"))
    runs, head = reopened.runs("owner")
    assert [(run.number, run.request) for run in runs] == [(1, "a"), (2, "b"), (3, "c")]
    assert head == 3
    receipt = reopened.submit(
        "owner", "d", b"c" * 32, offer("d"), expected_execution_workspace_id=reopened.workspace_id
    )
    assert receipt.number == 4


@pytest.mark.parametrize("delivery", [True, False])
def test_an_acknowledged_product_stays_until_gc_where_the_store_tracks_delivery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, delivery: bool
) -> None:
    """The run log's hold on a product, as `Products` takes it; then the owner's ack."""
    if not delivery:
        monkeypatch.setattr(fill, "capabilities", frozenset)
    elif "deliver/1" not in fill.capabilities():
        pytest.skip("the installed TensorFS does not track delivery")
    workspace, spec, manifest, path = producing(tmp_path)
    source = outputs.commit(
        workspace, "owner", "producer", 1, spec, "video", manifest, [("payload", path)], 200000
    )
    data = path.read_bytes()
    content = pb.Ref(digest=hashlib.sha256(data).digest(), length=len(data))
    grant = machine_models.retain_bytes(
        workspace,
        "owner",
        products.recipient("producer"),
        products.hold_path(content, source),
        source,
    )
    store = tensorfs.Store.open(str(workspace.store_root))
    assert store.tree_root(grant.retention_id) is not None

    products.Products(workspace, Executions(workspace), None).deliver("owner", "producer")
    tensorfs.gc(str(workspace.store_root))  # no pressure: only unreferenced bytes go
    with workspace.locked() as db:
        held = db.one(NativeHold, "SELECT * FROM holds WHERE id=?", (grant.retention_id,))
    assert held is not None and held.state == ("held" if delivery else "released")
    root = store.tree_root(grant.retention_id)
    if delivery:
        assert root is not None and not root["released"]
        # A replayed acknowledgement delivers again, the root already delivered.
        products.Products(workspace, Executions(workspace), None).deliver("owner", "producer")
    else:
        assert root is None or root["released"]
