"""Fixed byte-output budgets remain authoritative across executor and native custody."""

from __future__ import annotations

import hashlib
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Annotated, Any, cast

import msgspec
import pytest

from cozy_runtime import canonical_json
from cozy_runtime.author import App, AssetBound, Invocation, Outputs, Tree, attempt
from cozy_runtime.internal import (
    output_budget,
    package_interface,
    source_interfaces,
    static_interface,
)
from cozy_runtime.internal.discovery import discover
from cozy_runtime.internal.executor import _serialize_result
from cozy_runtime.internal.executor_replies import AttemptReply
from cozy_runtime.internal.worker import (
    commit_files,
    grants,
    workspace_byte_outputs,
    workspace_sources,
)
from cozy_runtime.internal.worker.attempts import AttemptEngine, AttemptRecord
from cozy_runtime.internal.worker.calls import Calls, _Pending
from cozy_runtime.internal.worker.plan import AttemptPlan, DeclaredBinding
from cozy_runtime.internal.worker.workspace import Workspace
from cozy_runtime.protocol import documents
from cozy_runtime.protocol import worker_pb2 as pb


def spec_for(*sizes: int) -> dict[str, Any]:
    return {
        "outputs": [{"output_id": f"out{i}", "max_bytes": size} for i, size in enumerate(sizes)]
    }


def test_admitted_slots_preserve_defaults_and_ignore_weights_capacity() -> None:
    assert output_budget.admitted({}) == 256 << 20
    assert output_budget.admitted(spec_for(256 << 20, 256 << 20)) == 512 << 20
    assert output_budget.admitted(spec_for(7, 11)) == 18
    assert output_budget.intermediate(spec_for(7, 11)) == 256 << 20
    assert output_budget.intermediate(spec_for(256 << 20, 256 << 20)) == 512 << 20
    assert output_budget.admitted(spec_for(1 << 40)) == 256 << 20
    assert (
        output_budget.admitted(
            {
                "outputs": [
                    {
                        "output_id": "weights",
                        "mime_type": "application/vnd.cozy.model-manifest",
                        "max_bytes": 1 << 40,
                    },
                    {"output_id": "report", "max_bytes": 17},
                ]
            }
        )
        == 17
    )
    with pytest.raises(ValueError):
        output_budget.admitted(spec_for(*([1] * 33)))
    for bad in (-1, True, "100"):
        with pytest.raises(ValueError):
            output_budget.admitted({"outputs": [{"max_bytes": bad}]})
    assert output_budget.effective(spec_for(7, 11), {"max_output_bytes": 13}) == 13
    assert output_budget.effective(spec_for(7, 11), {"max_output_bytes": 256 << 20}) == 18
    for bad in ((256 << 20) + 1, -1, True, "18"):
        with pytest.raises(ValueError):
            output_budget.effective(spec_for(7, 11), {"max_output_bytes": bad})


def test_optional_tree_bound_static_imported_parity(tmp_path: Path) -> None:
    sys.modules.pop("bounded_trees", None)
    (tmp_path / "pyproject.toml").write_text('[project]\nname="bounded-trees"\nversion="0.1.0"\n')
    (tmp_path / "package.toml").write_text('[application]\nobject="bounded_trees:app"\n')
    (tmp_path / "bounded_trees.py").write_text("""from typing import Annotated
import msgspec
from cozy_runtime.author import App, AssetBound, MediaDecoder, Tree
class Request(msgspec.Struct):
    prefix: Annotated[Tree | None, AssetBound(max_bytes=268435456)] = None
class Result(msgspec.Struct):
    video: Annotated[Tree, AssetBound(max_bytes=268435456)]
    prefix: Annotated[Tree, AssetBound(max_bytes=268435456)]
app = App()
@app.job
def assemble(payload: Request, media: MediaDecoder) -> Result:
    raise RuntimeError("describe only")
""")
    source = static_interface.build(tmp_path)
    imported = package_interface.build(discover(tmp_path))
    assert package_interface.canonical_bytes(source) == package_interface.canonical_bytes(imported)


@pytest.mark.parametrize("final_bytes", [140 << 20, 8])
def test_native_file_commit_uses_the_recorded_parent_budget(
    tmp_path: Path,
    final_bytes: int,
) -> None:
    workspace = Workspace(tmp_path / "store")
    spec = pb.InvocationSpec(
        job=pb.JobInvocationSpec(),
        outputs=[
            pb.OutputBinding(output_id=name, max_bytes=final_bytes) for name in ("first", "second")
        ],
    )
    raw, digest = documents.identity(spec)
    workspace.accept(
        "owner",
        pb.AttemptOffer(
            request_id="producer",
            attempt_ordinal=1,
            invocation_spec_digest=digest,
            invocation_spec_canonical_bytes=raw,
        ),
    )
    workspace.mark_running(
        "owner",
        pb.AttemptAccepted(
            request_id="producer",
            attempt_ordinal=1,
            invocation_spec_digest=digest,
        ),
    )
    for index, size in enumerate((129 << 20, 129 << 20, 23 << 20, 257 << 20), start=1):
        arguments = {
            "slot": f"file/{index:04d}",
            "digest": "sha256:" + "aa" * 32,
            "size_bytes": size,
            "media_type": "application/octet-stream",
        }
        intent = canonical_json.encode(
            {
                "module": source_interfaces.MODULE,
                "export": "commit_file",
                "request": arguments,
            }
        )
        command = pb.NativeSourceCommand(
            service_id=workspace_sources.identity("owner", "producer", index),
            operation=pb.NATIVE_SOURCE_OPERATION_COMMIT_FILE,
            parent_call=pb.ChildCallRequest(
                parent_request_id="producer",
                parent_attempt_ordinal=1,
                parent_invocation_spec_digest=digest,
                call_index=index,
                module=source_interfaces.MODULE,
                export="commit_file",
                request_canonical_bytes=canonical_json.encode(arguments),
                intent_digest=hashlib.sha256(intent).digest(),
            ),
        )
        workspace_sources.accepted(workspace, "owner", command)
        if index <= (2 if final_bytes > 8 else 1):
            prepared = commit_files.prepare(workspace, "owner", command, b"c" * 32, tmp_path)
            assert prepared.basename == f"file-{index:04d}"
        else:
            with pytest.raises(ValueError, match=r"budget|bounded"):
                commit_files.prepare(workspace, "owner", command, b"c" * 32, tmp_path)
    # Preparation records bounded intent but has written no native payload bytes.
    with workspace.locked() as db:
        assert (
            db.execute("SELECT count(*) FROM byte_outputs WHERE state='complete'").fetchone()[0]
            == 0
        )


def test_worker_command_keeps_intermediate_room_for_a_small_final_report(tmp_path: Path) -> None:
    # The executor receives enough room for intermediate files even when final
    # output fields are small. The host independently retains final authority.
    parent = AttemptRecord(
        "report",
        1,
        b"",
        spec_for(7, 11),
        declared=DeclaredBinding("", "report", "", "", "", ""),
        plan=AttemptPlan("", "", "", "cpu", 0, 0, "", ""),
    )
    command = AttemptEngine._command(
        cast(AttemptEngine, SimpleNamespace(attention_pin="")), parent, "report#1", tmp_path, 30
    )
    assert command.max_output_bytes == 256 << 20
    assert (
        output_budget.effective(parent.spec, {"max_output_bytes": command.max_output_bytes}) == 18
    )


class Request(msgspec.Struct):
    pass


class Pair(msgspec.Struct):
    first: Annotated[Tree, AssetBound(max_bytes=140 << 20)]
    second: Annotated[Tree, AssetBound(max_bytes=140 << 20)]


def test_two_large_outputs_commit_within_their_admitted_aggregate(tmp_path: Path) -> None:
    size = 129 << 20
    source = tmp_path / "source"
    source.mkdir()
    with (source / "payload").open("wb") as stream:
        stream.write(b"bounded output")
        stream.truncate(size)
    app = App()

    @app.job
    def write(payload: Request, out: Outputs) -> Pair:
        return Pair(out.save_tree(source), out.save_tree(source))

    spec = pb.InvocationSpec(
        job=pb.JobInvocationSpec(),
        outputs=[
            pb.OutputBinding(output_id=name, max_bytes=140 << 20) for name in ("first", "second")
        ],
    )
    raw, digest = documents.identity(spec)
    body = documents.body(spec)
    spool = tmp_path / "spool"
    result, outcome, _ = attempt(
        app.get("write"),
        {},
        Invocation(
            "producer",
            spool,
            time.monotonic() + 120,
            max_output_bytes=output_budget.admitted(body),
        ),
    )
    assert outcome.terminal == "succeeded", outcome
    _, rows, _ = _serialize_result(result, spool)
    assert sum(row["size_bytes"] for row in rows) == size * 2 > 256 << 20
    workspace = Workspace(tmp_path / "store")
    workspace.accept(
        "owner",
        pb.AttemptOffer(
            request_id="producer",
            attempt_ordinal=1,
            invocation_spec_digest=digest,
            invocation_spec_canonical_bytes=raw,
        ),
    )
    workspace.mark_running(
        "owner",
        pb.AttemptAccepted(
            request_id="producer",
            attempt_ordinal=1,
            invocation_spec_digest=digest,
        ),
    )
    record = AttemptRecord("producer", 1, digest, body, spool=spool)
    record.grant.outputs = {
        name: grants.BoundOutput(name, "", 140 << 20, "") for name in ("first", "second")
    }
    engine = SimpleNamespace(
        tensorfs_root=workspace.store_root,
        calls=SimpleNamespace(),
        owner_scope=lambda: "owner",
        _encode_frames=lambda *args: None,
        _blob_path=AttemptEngine._blob_path,
        _publication=lambda *args: SimpleNamespace(document=lambda: {}, digest=lambda: b"r" * 32),
        records=SimpleNamespace(append=lambda *args: None),
        products=None,
    )
    manifest, error = AttemptEngine._tail(
        cast(AttemptEngine, engine),
        record,
        msgspec.convert(
            {"ok": True, "outputs": rows, "max_output_bytes": output_budget.admitted(body)},
            AttemptReply,
        ),
    )
    assert error is None and manifest is not None, error
    assert sum(row.native_tree.content_bytes for row in manifest.outputs) == size * 2
    # Received intermediate outputs use that same parent declaration, including
    # the aggregate, rather than a second unrelated 256 MiB hard-coded ceiling.
    result_value, byte_grants = {}, []
    for index, output in enumerate(manifest.outputs):
        retention = pb.NativeByteRetentionRequest(
            source=output.native_tree,
            retention_id="sha256:" + f"{index + 1:02x}" * 32,
        )
        workspace_byte_outputs.change_hold(workspace, "owner", retention, release=False)
        byte_grants.append(
            pb.ChildByteResultGrant(
                output_id=output.output_id,
                source=retention.source,
                retention_id=retention.retention_id,
            )
        )
        identity = documents.spell(output.native_tree.manifest.digest)
        result_value[output.output_id] = {
            "asset_ref": identity,
            "kind": "tree",
            "digest": identity,
            "size_bytes": size,
        }
    child_result = pb.ChildCallResult(
        state=pb.CHILD_CALL_STATE_SUCCEEDED,
        call_index=1,
        result_canonical_bytes=canonical_json.encode(result_value),
        byte_result_grants=byte_grants,
    )
    consumer = AttemptRecord("consumer", 1, digest, body, spool=tmp_path / "consumer")
    calls = Calls(lambda *_: pb.ChildCallResult(), workspace=workspace, owner=lambda: "owner")
    pending = _Pending(pb.ChildCallRequest(), child_result)
    materialized = calls._materialize(consumer, pending)
    assert sum(row["content_bytes"] for row in materialized) == size * 2
    assert calls.materialized[("consumer", 1)][1] == size * 2
    child_result.call_index = 2
    with pytest.raises(ValueError, match="byte bound"):
        calls._materialize(consumer, _Pending(pb.ChildCallRequest(), child_result))
    # Authority is rechecked after encoding/commit even when a reply claims more.
    manifest, error = AttemptEngine._tail(
        cast(AttemptEngine, engine),
        record,
        msgspec.convert(
            {"ok": True, "outputs": rows, "max_output_bytes": output_budget.admitted(body) + 1},
            AttemptReply,
        ),
    )
    assert manifest is None and error and "budget" in error


def test_lower_aggregate_stops_the_author_before_a_second_large_copy(tmp_path: Path) -> None:
    # Small real bytes exercise the same aggregate check without a duplicate large fixture.
    source = tmp_path / "source"
    source.mkdir()
    (source / "payload").write_bytes(b"01234567")
    app = App()

    @app.job
    def write(payload: Request, out: Outputs) -> Pair:
        return Pair(out.save_tree(source), out.save_tree(source))

    _, outcome, _ = attempt(
        app.get("write"),
        {},
        Invocation(
            "producer",
            tmp_path / "spool",
            time.monotonic() + 30,
            max_output_bytes=15,
        ),
    )
    assert outcome.code == "output_too_large"
