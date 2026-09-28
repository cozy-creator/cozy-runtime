"""A newer SDK's unknown native option is omitted at its default, else refused for that call."""

from __future__ import annotations

import hashlib
import queue
import socket
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import msgspec
import pytest

from cozy_runtime import canonical_json
from cozy_runtime.author import CapabilityError
from cozy_runtime.author.fakes import fake_context
from cozy_runtime.internal import native_interfaces, source_interfaces
from cozy_runtime.internal.executor import Executor
from cozy_runtime.internal.executor_commands import CallInterface
from cozy_runtime.internal.seam import Channel
from cozy_runtime.internal.worker import workspace_sources
from cozy_runtime.internal.worker.machine_sources import Sources
from cozy_runtime.internal.worker.source_calls import SourceCalls
from cozy_runtime.protocol import worker_pb2 as pb
from test_owned_file_commit import pending, replace_arguments

UPDATE = "update the rental's Runtime (`cozy rental update <name>`) or pin an older SDK"


def _broker(tmp_path: Path, rows: list[dict[str, Any]]) -> Any:
    left, right = socket.socketpair()
    with left, right:
        broker = Executor(Channel(left), tmp_path)._call_broker(
            "parent", msgspec.convert(rows, tuple[CallInterface, ...])
        )
    broker.bind(fake_context(request_id="parent"))
    return broker


def _rows() -> list[dict[str, Any]]:
    stated = {"native_interfaces": native_interfaces.stated()}
    return [r for r in native_interfaces.rows(stated) if r["module"] == source_interfaces.MODULE]


def _served(without: dict[str, str]) -> list[dict[str, Any]]:
    """The rows a Runtime that predates one option of an operation advertises."""
    return [
        {
            **row,
            "request_fields": [f for f in row["request_fields"] if f != without.get(row["export"])],
        }
        for row in _rows()
    ]


def _payload(broker: Any, export: str, arguments: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = canonical_json.decode(
        broker.reserve(source_interfaces.MODULE, export, arguments).payload
    )
    return payload


def test_sdk_omits_a_defaulted_option_an_older_runtime_does_not_serve(tmp_path: Path) -> None:
    arguments = {"repository": "org/model", "revision": "a" * 40}
    current = _payload(_broker(tmp_path, _rows()), "download_huggingface", arguments)
    assert current["files"] == []
    older = _broker(tmp_path, _served({"download_huggingface": "files"}))
    assert "files" not in _payload(older, "download_huggingface", arguments)
    with pytest.raises(CapabilityError, match="does not support files") as refused:
        older.reserve(
            source_interfaces.MODULE,
            "download_huggingface",
            {**arguments, "files": ("config.json",)},
        )
    assert UPDATE in str(refused.value) and refused.value.code == "child_capability_unavailable"
    # A Runtime that does not advertise its fields receives the full request.
    unstated = [{k: v for k, v in row.items() if k != "request_fields"} for row in _served({})]
    assert _payload(_broker(tmp_path, unstated), "download_huggingface", arguments)["files"] == []


def test_runtime_refuses_only_the_commit_carrying_an_unknown_option(tmp_path: Path) -> None:
    workspace, command, _, data = pending(tmp_path)
    arguments = canonical_json.decode(command.parent_call.request_canonical_bytes)
    replace_arguments(command, {**arguments, "overwrite": True})
    frames: queue.Queue[pb.NativeSourceStatus] = queue.Queue()
    calls = SourceCalls(
        workspace, lambda: "owner", frames.put, lambda _: True, spool_root=tmp_path / "spool"
    )
    calls.handle(command)
    refused = frames.get(timeout=30)
    assert refused.state == pb.NATIVE_SOURCE_STATE_FAILED
    assert refused.safe_code == "commit_request_unsupported"
    assert "does not support overwrite" in refused.safe_detail and UPDATE in refused.safe_detail
    with workspace.locked() as db:
        assert db.execute("SELECT count(*) FROM byte_outputs").fetchone()[0] == 0

    # The machine path hands the package the same refusal, also after a worker restart.
    for service in (calls, None):
        host = SimpleNamespace(workspace=workspace, source_calls=service, stamp=lambda _: None)
        answer = Sources(cast(Any, host)).result(
            "owner", cast(Any, None), command.parent_call, pb.ChildCallResult()
        )
        assert answer.state == pb.CHILD_CALL_STATE_FAILED
        assert answer.safe_code == "commit_request_unsupported" and UPDATE in answer.safe_detail

    # The same attempt's next commit, without the option, still completes.
    (spool,) = (tmp_path / "spool").iterdir()
    (spool / "file-0002").write_bytes(data)
    second = pb.NativeSourceCommand()
    second.CopyFrom(command)
    second.parent_call.call_index = 2
    second.service_id = workspace_sources.identity("owner", "script", 2)
    replace_arguments(second, {**arguments, "slot": "file/0002"})
    calls.handle(second)
    committed = frames.get(timeout=30)
    assert committed.state == pb.NATIVE_SOURCE_STATE_SUCCEEDED, committed.safe_code
    assert committed.byte_output.content_bytes == len(data)
    assert hashlib.sha256(data).hexdigest() in committed.result_canonical_bytes.decode()
