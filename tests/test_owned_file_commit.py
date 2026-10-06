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
from durable_seam import wire


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
    digest = "sha256:" + hashlib.sha256(data).hexdigest()
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
