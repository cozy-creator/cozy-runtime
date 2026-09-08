"""A regular-file output tree is snapshotted and bounded before leaving an attempt."""

from __future__ import annotations

import json
import time
from pathlib import Path

import msgspec
import pytest

from cozy_runtime.author import App, Invocation, Outputs, Tree, attempt
from cozy_runtime.internal.executor import _serialize_result


class Request(msgspec.Struct):
    pass


class Result(msgspec.Struct):
    files: Tree


@pytest.mark.parametrize("bad", ["", "symlink", "oversize", "unreturned"])
def test_tree_snapshot_and_result_serialization(tmp_path: Path, bad: str) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "capture.json").write_text('{"rows":2}')
    (source / "nested").mkdir()
    (source / "nested/sketches.f32").write_bytes(b"12345678")
    if bad == "symlink":
        (source / "escape").symlink_to(tmp_path)
    app = App()

    @app.job
    def main(payload: Request, out: Outputs) -> Result:
        tree = out.save_tree(source)
        (source / "capture.json").write_text("changed after snapshot")
        assert (tree.path / "capture.json").read_text() == '{"rows":2}'
        if bad == "unreturned":
            out.save_tree(source)
        return Result(tree)

    spool = tmp_path / "spool"
    result, outcome, _ = attempt(
        app.get("main"),
        {},
        Invocation(
            "producer",
            spool,
            time.monotonic() + 30,
            max_output_bytes=4 if bad == "oversize" else 4096,
        ),
    )
    if bad:
        assert result is None
        assert (
            outcome.code
            == {
                "symlink": "output_tree",
                "oversize": "output_too_large",
                "unreturned": "unreturned_handle",
            }[bad]
        )
    else:
        assert outcome.terminal == "succeeded", outcome
        assert result is not None
        reference, rows, size = _serialize_result(result, spool)
        assert reference is not None and size == 0 and len(rows) == 1
        assert rows[0]["output_id"] == "files" and rows[0]["kind"] == "tree"
        assert str(tmp_path) not in json.dumps(rows)
