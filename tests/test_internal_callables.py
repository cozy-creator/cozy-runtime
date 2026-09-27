"""Internal visibility preserves same-package execution without opening a root API."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from cozy_runtime import canonical_json
from cozy_runtime.author import ConformanceError
from cozy_runtime.internal import interface_wheel, package_interface, static_interface
from cozy_runtime.internal.discovery import discover
from cozy_runtime.internal.worker.internal_calls import require_child
from cozy_runtime.internal.worker.workspace import Workspace, WorkspaceRefusal
from cozy_runtime.internal.worker.workspace_calls import Calls
from cozy_runtime.internal.worker.workspace_executions import Executions
from cozy_runtime.protocol import worker_pb2 as pb
from test_machine_calls import request, running
from test_machine_execution import offer

SOURCE = """import msgspec
from cozy_runtime.author import App, Context, invocable
app = App()
class Request(msgspec.Struct):
    n: int
class Result(msgspec.Struct):
    value: int
@app.entrypoint
def public(payload: Request) -> Result:
    return Result(payload.n)
@app.entrypoint(internal=True)
def segment(payload: Request) -> Result:
    return Result(payload.n)
@invocable
async def child(ctx: Context, *, n: int) -> Result:
    return Result(n)
app.job(child, internal=True)
@app.entrypoint(hidden=True)
def unfinished(payload: Request) -> Result:
    return Result(payload.n)
"""


def project(tmp_path: Path, source: str = SOURCE) -> Path:
    sys.modules.pop("internal_fixture", None)
    root = tmp_path / "source"
    root.mkdir()
    (root / "pyproject.toml").write_text('[project]\nname="internal-fixture"\nversion="0.1.0"\n')
    (root / "package.toml").write_text('[application]\nobject="internal_fixture:app"\n')
    (root / "internal_fixture.py").write_text(source)
    return root


def test_internal_interface_is_identical_for_static_and_imported_discovery(tmp_path: Path) -> None:
    root = project(tmp_path)
    static = static_interface.build(root)
    imported = package_interface.build(discover(root))
    raw = package_interface.canonical_bytes(static)
    assert raw == package_interface.canonical_bytes(imported)
    document = package_interface.read_bytes(raw)
    assert [entry["name"] for entry in document["entrypoints"]] == ["public", "segment"]
    assert document["entrypoints"][1]["internal"] is True
    assert "internal" not in document["entrypoints"][0]
    assert document["jobs"][0]["internal"] is True
    generated = interface_wheel.generate(raw)["internal_fixture/__init__.py"].decode()
    assert "def public(" in generated
    assert "def segment(" not in generated and "def child(" not in generated


@pytest.mark.parametrize("value", ['"yes"', "1"])
def test_internal_requires_a_boolean_in_both_readers(tmp_path: Path, value: str) -> None:
    root = project(tmp_path, SOURCE.replace("internal=True", "internal=" + value))
    with pytest.raises(ConformanceError, match="internal must be a boolean"):
        static_interface.build(root)
    with pytest.raises(ConformanceError, match="internal must be a boolean"):
        discover(root)


def test_hidden_and_internal_are_distinct_and_mutually_exclusive(tmp_path: Path) -> None:
    root = project(tmp_path, SOURCE.replace("hidden=True", "internal=True, hidden=True"))
    for reader in (static_interface.build, discover):
        with pytest.raises(ConformanceError, match="cannot also be hidden"):
            reader(root)


@pytest.mark.parametrize("value", ["true", 1, None])
def test_interface_reader_rejects_nonboolean_internal(tmp_path: Path, value: object) -> None:
    document = static_interface.build(project(tmp_path))
    document["entrypoints"][1]["internal"] = value
    with pytest.raises(ConformanceError, match="internal"):
        package_interface.read_bytes(canonical_json.encode(document))


def test_child_authority_survives_reopen_but_cannot_be_reused_by_roots_or_other_installations(
    tmp_path: Path,
) -> None:
    workspace = Workspace(tmp_path / "store")
    executions = Executions(workspace)
    parent = offer("parent")
    executions.submit(
        "owner",
        "parent",
        b"c" * 32,
        parent,
        expected_execution_workspace_id=executions.workspace_id,
    )
    calls = Calls(workspace)
    call = calls.accept("owner", request(running(executions)))
    installation = "local-" + "11" * 16
    calls.freeze(
        "owner",
        call,
        canonical_json.encode({"installation_id": installation, "entrypoint": "segment"}),
    )
    child = offer(call.child_request)
    executions.submit(
        "owner",
        call.child_request,
        b"c" * 32,
        child,
        expected_execution_workspace_id=executions.workspace_id,
    )
    reopened = Workspace(tmp_path / "store")
    require_child(reopened, "owner", child, installation, "segment", machine_owned=True)
    for selected, owner, requested_installation, name, machine_owned in (
        (child, "owner", installation, "segment", False),
        (parent, "owner", installation, "segment", True),
        (child, "another-owner", installation, "segment", True),
        (child, "owner", "local-" + "22" * 16, "segment", True),
        (child, "owner", installation, "another-helper", True),
    ):
        with pytest.raises(WorkspaceRefusal, match="internal_callable"):
            require_child(
                reopened,
                owner,
                selected,
                requested_installation,
                name,
                machine_owned=machine_owned,
            )
    foreign_request = request(running(executions))
    foreign_request.call_index = 1
    foreign_call = calls.accept("owner", foreign_request)
    foreign_installation = "local-" + "22" * 16
    calls.freeze(
        "owner",
        foreign_call,
        canonical_json.encode({"installation_id": foreign_installation, "entrypoint": "segment"}),
    )
    foreign_child = offer(foreign_call.child_request)
    executions.submit(
        "owner",
        foreign_call.child_request,
        b"c" * 32,
        foreign_child,
        expected_execution_workspace_id=executions.workspace_id,
    )
    with pytest.raises(WorkspaceRefusal, match="internal_callable"):
        require_child(
            reopened,
            "owner",
            foreign_child,
            foreign_installation,
            "segment",
            machine_owned=True,
        )
    altered = pb.AttemptOffer()
    altered.CopyFrom(child)
    altered.invocation_spec_digest = b"x" * 32
    with pytest.raises(WorkspaceRefusal, match="internal_callable"):
        require_child(reopened, "owner", altered, installation, "segment", machine_owned=True)
