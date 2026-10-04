"""Supplemental custody/refusal checks; actual function proof uses Creator's CLI."""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any

import msgspec
import pytest

from cozy_runtime import canonical_json
from cozy_runtime.author import App, FileAsset, Invocation, OutputError, Outputs, attempt
from cozy_runtime.author._calls import _Broker
from cozy_runtime.internal import source_interfaces
from cozy_runtime.internal.pathkey import opaque_key
from cozy_runtime.internal.worker import workspace_sources
from cozy_runtime.internal.worker.workspace import Workspace
from cozy_runtime.protocol import documents
from cozy_runtime.protocol import worker_pb2 as pb
from durable_seam import wire
from test_workspace_native_memo import accepted


class Empty(msgspec.Struct):
    pass


@pytest.mark.parametrize(
    "fault", ["", "foreign", "wrong_attempt", "unreturned", "return_committed"]
)
def test_owned_commit_facade_keeps_budget_and_explicit_result_boundary(
    tmp_path: Path, fault: str
) -> None:
    app = App()
    data = b"parent generated file" * 4000
    ready_path = tmp_path / "received"
    ready_path.write_bytes(data)
    digest = documents.spell(hashlib.sha256(data).digest())
    submitted = 0

    @app.job
    async def write(payload: Empty, out: Outputs) -> Empty:
        saved = out.save_bytes(data, media_type="application/json")
        if fault == "foreign":
            await out.commit(FileAsset(saved.ref))
        if fault == "wrong_attempt":
            saved._attempt = "another#1"
        ready = await out.commit(saved)
        assert ready.read_bytes() == data
        assert await out.commit(saved) is ready
        assert await out.commit(ready) is ready
        with pytest.raises(OutputError, match="aggregate limit"):
            out.save_bytes(b"x", media_type="application/json")
        if fault == "unreturned":
            out.save_bytes(b"", media_type="application/json")
        if fault == "return_committed":
            return ready  # type: ignore[return-value] # intentionally wrong R
        return Empty()

    def exchange(kind: str, body: dict[str, Any]) -> dict[str, Any]:
        nonlocal submitted
        if kind == "child_call":
            submitted += 1
            assert body["export"] == "commit_file"
            args = canonical_json.decode(body["payload"].encode())
            assert args["digest"] == digest and args["size_bytes"] == len(data)
            assert args["slot"] == "file/0001"
            return {"ok": True}
        if kind == "child_forget":
            return {"ok": True}
        assert kind == "child_poll"
        value = {
            "file": {
                "asset_ref": digest,
                "kind": "file",
                "digest": digest,
                "size_bytes": len(data),
                "media_type": "application/json",
            }
        }
        return {
            "ok": True,
            "state": "succeeded",
            "result": canonical_json.encode(value).decode(),
            "byte_grants": [
                {
                    "output_id": "file",
                    "kind": "file",
                    "digest": digest,
                    "length": len(data),
                    "local": str(ready_path),
                    "media_type": "application/json",
                }
            ],
        }

    broker = _Broker(
        "script#1",
        {(b.module, b.export): b for b in source_interfaces.bindings().values()},
        wire(exchange),
    )
    _, outcome, _ = attempt(
        app.get("write"),
        {},
        Invocation(
            "script#1", tmp_path, time.monotonic() + 30, calls=broker, max_output_bytes=len(data)
        ),
    )
    if fault:
        assert (
            outcome.code
            == {
                "foreign": "output_owner",
                "wrong_attempt": "output_owner",
                "unreturned": "unreturned_handle",
                "return_committed": "result_schema",
            }[fault]
        ), outcome
    else:
        assert outcome.terminal == "succeeded", outcome
    assert submitted == (0 if fault in ("foreign", "wrong_attempt") else 1)


def pending(
    tmp_path: Path, data: bytes | None = None
) -> tuple[Workspace, pb.NativeSourceCommand, Path, bytes]:
    workspace = Workspace(tmp_path / "store")
    source = accepted(workspace, "script")
    if data is None:
        data = canonical_json.encode({"report": "observed" * 12000})
    spool = tmp_path / "spool" / opaque_key("attempt-input-spool", "script", 1)
    spool.mkdir(parents=True)
    path = spool / "file-0001"
    path.write_bytes(data)
    arguments = {
        "slot": "file/0001",
        "digest": documents.spell(hashlib.sha256(data).digest()),
        "size_bytes": len(data),
        "media_type": "application/json",
    }
    intent = canonical_json.encode(
        {
            "module": source_interfaces.MODULE,
            "export": "commit_file",
            "request": arguments,
        }
    )
    command = pb.NativeSourceCommand(
        service_id=workspace_sources.identity("owner", "script", 1),
        operation=pb.NATIVE_SOURCE_OPERATION_COMMIT_FILE,
        phase=pb.NATIVE_SOURCE_PHASE_EXECUTE,
        parent_call=pb.ChildCallRequest(
            parent_request_id="script",
            parent_attempt_ordinal=1,
            parent_invocation_spec_digest=source.parent_call.parent_invocation_spec_digest,
            call_index=1,
            module=source_interfaces.MODULE,
            export="commit_file",
            request_canonical_bytes=canonical_json.encode(arguments),
            intent_digest=hashlib.sha256(intent).digest(),
        ),
    )
    return workspace, command, path, data
