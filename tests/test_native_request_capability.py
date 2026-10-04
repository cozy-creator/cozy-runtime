"""A newer SDK's unknown native option is omitted at its default, else refused for that call."""

from __future__ import annotations

import socket
from pathlib import Path
from typing import Any

import msgspec
import pytest

from cozy_runtime import canonical_json
from cozy_runtime.author import CapabilityError
from cozy_runtime.author.fakes import fake_context
from cozy_runtime.internal import native_interfaces, source_interfaces
from cozy_runtime.internal.executor import Executor
from cozy_runtime.internal.executor_commands import CallInterface
from cozy_runtime.internal.seam import Channel

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
