"""Embedded full-wheel interfaces do not turn a root's App exports into dependencies."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from cozy_runtime.internal import package_interface, static_interface
from cozy_runtime.protocol import worker_pb2 as pb
from test_end_to_end import NO_EXECUTOR, _wheel_rows
from test_machine_partial_work import machine
from test_unpublished_exact_dependencies import _wheel

SOURCE = """import os
import msgspec
from cozy_runtime.author import App, Context, invocable
app = App()
class Request(msgspec.Struct):
    value: int = 12
class Result(msgspec.Struct):
    value: int
    pid: int

@app.entrypoint
def ordinary(ctx: Context, payload: Request) -> Result:
    return Result(payload.value, os.getpid())

@invocable()
async def child(ctx: Context, *, value: int) -> Result:
    return Result(value + 1, os.getpid())
app.job(child)

@app.job
async def nested(ctx: Context, payload: Request) -> Result:
    result = await child(value=payload.value)
    assert result.pid != os.getpid(), "child was not independently executed"
    return result
"""


@pytest.mark.skipif(bool(NO_EXECUTOR), reason=NO_EXECUTOR or "")
@pytest.mark.parametrize("embedded", [False, True])
def test_real_installed_root_keeps_self_exports_and_calls_child(
    monkeypatch: pytest.MonkeyPatch, embedded: bool
) -> None:
    import test_job_preparation_isolation as fixture

    original = fixture.package

    def package(root: Path, label: str) -> tuple[Path, pb.PrepareLocalPackageRequest]:
        environment, preparation = original(root, label)
        if embedded:
            project = root / ("project_" + label)
            module = "prepare_" + label
            (project / "package.toml").write_text(f'[application]\nobject="{module}:app"\n')
            wheels = environment / ".stage" / label / "wheels"
            info = f"{module}-1.0.0.dist-info"
            document = package_interface.canonical_bytes(static_interface.build(project))
            wheel = _wheel(
                wheels,
                module,
                ("cozy-runtime>=0.16.8,<1",),
                (project / f"{module}.py").read_bytes(),
                resources={
                    f"{info}/entry_points.txt": f"[cozy.application]\ndefault={module}:app\n".encode(),
                    f"{info}/package-interface.json": document,
                },
                version="1.0.0",
            )
            with zipfile.ZipFile(wheel) as archive:
                assert archive.read(f"{info}/package-interface.json") == document
            del preparation.files[:]
            preparation.files.extend(_wheel_rows(sorted(wheels.glob("*.whl"))))
        return environment, preparation

    monkeypatch.setattr(fixture, "package", package)
    with machine(monkeypatch, SOURCE) as pod:
        pod.start(("ordinary", "child"))
        executor_pids: set[int] = set()
        for request in ("first", "reused"):
            pod.submit(request)
            body = pod.outcome(request)
            assert body["status"] == pb.OUTCOME_STATUS_SUCCEEDED, body
            rows = pod.rows(
                "SELECT body FROM execution_events WHERE request='" + request + "' AND kind='executor'"
            )
            for row in rows:
                raw = row["body"]
                assert isinstance(raw, bytes)
                executor_pids.add(json.loads(raw)["pid"])
            calls = pod.rows(
                "SELECT child_request FROM execution_calls WHERE parent_request='" + request + "'"
            )
            assert len(calls) == 1
        assert executor_pids
