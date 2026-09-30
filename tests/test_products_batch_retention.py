"""Returned products retain only journaled native custody across publication failure."""

from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from cozy_runtime import canonical_json
from cozy_runtime.internal.worker import products
from cozy_runtime.internal.worker import workspace_byte_outputs as outputs
from cozy_runtime.internal.worker.attempts import AttemptRecord
from cozy_runtime.internal.worker.workspace import NativeHold, Workspace
from cozy_runtime.internal.worker.workspace_executions import Executions
from cozy_runtime.protocol import documents
from cozy_runtime.protocol import worker_pb2 as pb


@pytest.mark.parametrize("fail_after", [None, 1, 2])
def test_finish_keeps_journaled_files_without_retaining_unjournaled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fail_after: int | None
) -> None:
    workspace = Workspace(tmp_path / "store")
    executions = Executions(workspace)
    body, digest = documents.identity(pb.InvocationSpec(job=pb.JobInvocationSpec()))
    spec = digest
    executions.submit(
        "owner",
        "producer",
        b"c" * 32,
        pb.AttemptOffer(
            request_id="producer",
            attempt_ordinal=1,
            invocation_spec_digest=spec,
            invocation_spec_canonical_bytes=body,
        ),
        expected_execution_workspace_id=executions.workspace_id,
    )
    workspace.mark_running(
        "owner",
        pb.AttemptAccepted(request_id="producer", attempt_ordinal=1, invocation_spec_digest=spec),
    )
    entries = []
    payloads = {}
    for name in ("image", "raw", "rgb", "witness"):
        data = (name + ":retained bytes").encode()
        path = tmp_path / (name + ".bin")
        path.write_bytes(data)
        manifest = canonical_json.encode(
            {
                "entries": [
                    {
                        "path": "payload",
                        "kind": "file",
                        "blob": {"sha256": hashlib.sha256(data).hexdigest(), "length": len(data)},
                    }
                ]
            }
        )
        source = outputs.commit(
            workspace, "owner", "producer", 1, spec, name, manifest, [("payload", path)], 200000
        )
        entries.append(
            pb.OutputEntry(
                output_id=name,
                native_tree=source,
                digest=hashlib.sha256(data).digest(),
                length=len(data),
                mime_type="application/octet-stream",
            )
        )
        payloads[name] = data
    log = products.Products(workspace, executions, None)
    attempt = SimpleNamespace(request_id="producer", grant=SimpleNamespace(outputs=payloads))
    notice = executions.notice
    count = 0

    def failing_notice(*args: Any, **kwargs: Any) -> Any:
        nonlocal count
        result = notice(*args, **kwargs)
        count += 1
        if count == fail_after:
            # Fail after the durable notice, before _journal updates its local fold.
            raise RuntimeError("lost reply after journal commit")
        return result

    if fail_after is not None:
        monkeypatch.setattr(executions, "notice", failing_notice)
        with pytest.raises(RuntimeError, match="lost reply"):
            log.finish("owner", cast(AttemptRecord, attempt), entries)
    else:
        assert log.finish("owner", cast(AttemptRecord, attempt), entries) is None
    fold = log.fold("owner", "producer")
    assert len(fold.single) == (fail_after or 4)
    for name, product in fold.single.items():
        with outputs.leased(workspace, "owner", product.source) as (_, lease, members):
            blob = members[0].blob
            readback = bytearray(blob.length)
            lease.read_into("sha256:" + blob.sha256, blob.length, 0, blob.length, readback)
            assert readback == payloads[name]
    with workspace.locked() as db:
        held = db.all(NativeHold, "SELECT * FROM holds WHERE owner=?", ("owner",))
    # Master acquires each hold immediately before its journal entry. A lost reply
    # must not cause either dangling future holds or tombstones for unacquired ones.
    assert len(held) == (fail_after or 4)
    assert all(h.state == "held" for h in held)
    if fail_after is not None:
        monkeypatch.setattr(executions, "notice", notice)
        assert log.finish("owner", cast(AttemptRecord, attempt), entries) is None
        assert len(log.fold("owner", "producer").single) == 4
        with workspace.locked() as db:
            resumed = db.all(NativeHold, "SELECT * FROM holds WHERE owner=?", ("owner",))
        assert len(resumed) == 4 and all(h.state == "held" for h in resumed)
