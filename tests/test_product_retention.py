"""Every final product remains readable through its own exact native hold."""

from pathlib import Path

import pytest
import tensorfs

from cozy_runtime.internal.worker import byte_outputs, machine_byte_results, products
from cozy_runtime.internal.worker import workspace_byte_outputs as outputs
from cozy_runtime.internal.worker.attempts import AttemptRecord
from cozy_runtime.internal.worker.grants import BoundGrant, BoundOutput
from cozy_runtime.internal.worker.workspace import Workspace
from cozy_runtime.internal.worker.workspace_executions import Executions
from cozy_runtime.protocol import documents
from cozy_runtime.protocol import worker_pb2 as pb
from test_machine_execution import ack, offer


@pytest.mark.parametrize("list_output", [False, True], ids=["three-fields", "three-list-items"])
def test_returning_several_products_keeps_every_exact_hold(
    tmp_path: Path, list_output: bool
) -> None:
    workspace = Workspace(tmp_path / "store")
    executions = Executions(workspace)
    submitted = offer("producer")
    executions.submit(
        "owner",
        "submission",
        b"c" * 32,
        submitted,
        expected_execution_workspace_id=executions.workspace_id,
    )
    workspace.mark_running(
        "owner",
        pb.AttemptAccepted(
            request_id="producer",
            attempt_ordinal=1,
            invocation_spec_digest=submitted.invocation_spec_digest,
        ),
    )
    names = [f"clips.{i}" for i in range(3)] if list_output else ["metadata", "native", "preview"]
    slots = ["clips" + machine_byte_results.LIST] if list_output else names
    attempt = AttemptRecord(
        request_id="producer",
        attempt=1,
        digest=submitted.invocation_spec_digest,
        spec={},
        grant=BoundGrant(
            outputs={
                name: BoundOutput(name, "", 4096, "application/octet-stream") for name in slots
            }
        ),
    )
    entries: list[pb.OutputEntry] = []
    expected: dict[bytes, bytes] = {}
    for index, name in enumerate(names):
        path = tmp_path / f"output-{index}"
        data = f"distinct output {index}".encode()
        path.write_bytes(data)
        entry = byte_outputs.commit(
            workspace,
            "owner",
            "producer",
            1,
            attempt.digest,
            name,
            path,
            tree=False,
            max_bytes=4096,
            media_type="application/octet-stream",
        )
        entries.append(entry)
        expected[entry.digest] = data

    log = products.Products(workspace, executions, None)
    assert log.finish("owner", attempt, entries) is None
    recorded = executions.product_bodies("owner", "producer")
    assert len(recorded) == 3
    # Repeating finalization does not create another product or change its custody.
    assert log.finish("owner", attempt, entries) is None
    assert executions.product_bodies("owner", "producer") == recorded

    def read_all(current: Workspace) -> None:
        for raw in recorded:
            product = documents.parse(raw, pb.RunProduct)
            with outputs.leased(current, "owner", product.source) as (_, lease, members):
                assert len(members) == 1
                data = bytearray(product.content.length)
                lease.read_into(
                    documents.spell(product.content.digest), len(data), 0, len(data), data
                )
                assert bytes(data) == expected[product.content.digest]

    read_all(Workspace(workspace.store_root))
    # Collection releases producer custody, but delivered product holds remain readable.
    body, digest = documents.identity(
        pb.AttemptOutcomeBody(
            request_id="producer",
            attempt_ordinal=1,
            invocation_spec_digest=documents.spell(attempt.digest),
            status=pb.OUTCOME_STATUS_SUCCEEDED,
            output_manifest=pb.OutputManifest(outputs=entries),
        )
    )
    terminal = pb.AttemptOutcome(
        request_id="producer",
        attempt_ordinal=1,
        invocation_spec_digest=attempt.digest,
        outcome_id="outcome",
        outcome_digest=digest,
        outcome_canonical_bytes=body,
    )
    workspace.outcome("owner", terminal)
    executions.reconcile("owner", "producer")
    executions.acknowledge_collection("owner", ack(terminal))
    log.deliver("owner", "producer")
    workspace.acknowledge("owner", ack(terminal))
    tensorfs.gc(str(workspace.store_root))
    read_all(Workspace(workspace.store_root))
