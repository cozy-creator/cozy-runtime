"""An undeclared request field is dropped with one warning, never a refusal (owner, 2026-09-28).

Real Worker, real executor, real journal: `test_execution_containment`'s machine, with
request structs that forbid unknown fields, receiving the payload an older Creator or a newer
caller sends. Types and required fields stay strict.
"""

from __future__ import annotations

from typing import Any

import pytest

import test_execution_containment as containment
from cozy_runtime import canonical_json
from cozy_runtime.protocol import worker_pb2 as pb

pytestmark = containment.pytestmark

SOURCE = containment.SOURCE.replace(
    "class Request(msgspec.Struct):\n    segments: int\n    gate: str = ''\n",
    "class Label(msgspec.Struct, forbid_unknown_fields=True):\n    text: str\n"
    "class Request(msgspec.Struct, forbid_unknown_fields=True):\n"
    "    segments: int\n    gate: str = ''\n    labels: list[Label] = []\n",
)


def warnings(m: containment.Machine, request_id: str) -> list[Any]:
    assert m.worker.executions is not None
    events = m.worker.executions.events(m.worker.fence.record_owner_id, request_id).events
    return [canonical_json.decode(e.body) for e in events if e.kind == "warning"]


def test_undeclared_fields_run_with_one_warning(monkeypatch: pytest.MonkeyPatch) -> None:
    assert SOURCE != containment.SOURCE
    monkeypatch.setattr(containment, "SOURCE", SOURCE)
    with containment.machine(monkeypatch) as m:
        labels = [{"text": "a"}, {"text": "b", "foo": 1}]
        m.submit("tolerant", segments=2, style="noir", labels=labels)
        state, _, body = m.settle("tolerant")
        assert state == "succeeded", body
        assert canonical_json.decode(body.result.inline_result) == {"sizes": [9, 0]}
        assert warnings(m, "tolerant") == [
            {
                "code": "request_fields_ignored",
                "message": "ignored unknown fields labels[1].foo, style"
                " — not in nested's interface",
                "fields": ["labels[1].foo", "style"],
            }
        ]

        refused: list[tuple[str, dict[str, Any], str]] = [
            ("missing", {"style": "noir"}, "segments"),
            ("mistyped", {"segments": "two"}, "segments"),
            ("nested-mistyped", {"segments": 1, "labels": [{"text": 7}]}, "labels"),
        ]
        for request_id, payload, named in refused:
            m.submit(request_id, **payload)
            state, _, body = m.settle(request_id)
            assert (state, body.status) == ("failed", pb.OUTCOME_STATUS_REFUSED), body
            assert body.cause.code == pb.CAUSE_CODE_INVALID_REQUEST, body
            assert named in body.safe_message, body.safe_message
            assert not warnings(m, request_id)
        m.idle()
