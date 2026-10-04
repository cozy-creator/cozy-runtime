"""Supplemental native custody checks; the client CLI owns function qualification."""

from __future__ import annotations

from pathlib import Path

import pytest

from cozy_runtime.internal.worker import workspace_byte_outputs as byte_outputs
from cozy_runtime.internal.worker.workspace import Workspace, WorkspaceRefusal
from test_workspace_native_memo import accepted


def test_author_cannot_forge_source_view_namespace(tmp_path: Path) -> None:
    workspace = Workspace(tmp_path / "store")
    source = accepted(workspace, "script")
    with pytest.raises(WorkspaceRefusal, match="namespace"):
        byte_outputs.commit(
            workspace,
            "owner",
            "script",
            1,
            source.parent_call.parent_invocation_spec_digest,
            byte_outputs.native_slot("source_files", 1),
            b"{}",
            [],
            64 << 20,
        )
