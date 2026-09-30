"""Embedded full-wheel interfaces do not turn a root's App exports into dependencies."""

from __future__ import annotations

import base64
import json
import shutil
import zipfile
from pathlib import Path

import msgspec
import pytest

from cozy_runtime.internal import package_interface, static_interface
from cozy_runtime.internal.executor_commands import CallInterface
from cozy_runtime.internal.worker.attempts import AttemptRecord
from cozy_runtime.protocol import documents
from cozy_runtime.protocol import worker_pb2 as pb
from test_end_to_end import NO_EXECUTOR, _wheel_rows
from test_machine_partial_work import machine
from test_unpublished_exact_dependencies import _wheel

SOURCE = """import os
import msgspec
from cozy_runtime.author import App, Context, invocable
from prepare_dependency import double
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
    other = await double(value=result.value)
    assert other.pid != os.getpid(), "dependency was not independently executed"
    return Result(other.value, other.pid)
"""

DEPENDENCY_SOURCE = """import os
import msgspec
from cozy_runtime.author import App, Context, invocable
app = App()
class Result(msgspec.Struct):
    value: int
    pid: int
@invocable()
async def double(ctx: Context, *, value: int) -> Result:
    return Result(value * 2, os.getpid())
app.job(double)
"""


class _Result(msgspec.Struct):
    value: int
    pid: int


@pytest.mark.skipif(bool(NO_EXECUTOR), reason=NO_EXECUTOR or "")
@pytest.mark.parametrize("embedded", [False, True])
def test_real_installed_root_keeps_self_exports_and_calls_child(
    monkeypatch: pytest.MonkeyPatch, embedded: bool
) -> None:
    import test_job_preparation_isolation as fixture

    original = fixture.package
    dependencies: list[pb.PrepareLocalPackageRequest] = []

    def package(root: Path, label: str) -> tuple[Path, pb.PrepareLocalPackageRequest]:
        with monkeypatch.context() as context:
            context.setattr(fixture, "SOURCE", DEPENDENCY_SOURCE)
            environment, dependency = original(root, "dependency")
        environment, preparation = original(root, label)
        for name, selected in [("dependency", dependency), (label, preparation)]:
            project = root / ("project_" + name)
            module = "prepare_" + name
            (project / "package.toml").write_text(f'[application]\nobject="{module}:app"\n')
            wheels = environment / ".stage" / name / "wheels"
            info = f"{module}-1.0.0.dist-info"
            document = package_interface.canonical_bytes(static_interface.build(project))
            resources = {
                f"{info}/entry_points.txt": f"[cozy.application]\ndefault={module}:app\n".encode()
            }
            if embedded:
                resources[f"{info}/package-interface.json"] = document
            wheel = _wheel(
                wheels,
                module,
                ("cozy-runtime>=0.16.8,<1",)
                + (("prepare-dependency==1.0.0",) if name == label else ()),
                (project / f"{module}.py").read_bytes(),
                resources=resources,
                version="1.0.0",
            )
            if embedded:
                with zipfile.ZipFile(wheel) as archive:
                    assert archive.read(f"{info}/package-interface.json") == document
            if name == label:
                dependency_wheel = next(
                    (environment / ".stage/dependency/wheels").glob("prepare_dependency-*.whl")
                )
                shutil.copyfile(dependency_wheel, wheels / dependency_wheel.name)
            del selected.files[:]
            selected.files.extend(_wheel_rows(sorted(wheels.glob("*.whl"))))
        dependencies.append(dependency)
        return environment, preparation

    monkeypatch.setattr(fixture, "package", package)
    with machine(monkeypatch, SOURCE) as pod:
        dependency = pod.worker.prepare_local_package(dependencies[0]).installed_package
        pod.start(("ordinary", "child"))
        capture = documents.parse(pod.captured, pb.MachineExecutionCapture)
        capture.installed_packages.add().CopyFrom(dependency)
        capture.bindings.add(
            caller_installation_id=capture.root_installation_id,
            callee_installation_id=dependency.installation_id,
            module="prepare_dependency",
            export="double",
            entrypoint="double",
        )
        pod.captured, pod.capture_digest = documents.identity(capture)
        original_bindings = pod.worker.calls.bindings
        observed_roots: set[str] = set()

        def bindings(attempt: AttemptRecord) -> tuple[CallInterface, ...]:
            rows = original_bindings(attempt)
            indexed = {(row.module, row.export): row for row in rows}
            assert len(indexed) == len(rows), "an export has multiple executor bindings"
            if attempt.request_id in {"first", "reused"}:
                ordinary = indexed["prepare_nested", "ordinary"]
                child = indexed["prepare_nested", "child"]
                dependency = indexed["prepare_dependency", "double"]
                assert ordinary.self_call and ordinary.kind == "entrypoint"
                assert child.self_call and child.kind == "job"
                assert not dependency.self_call and dependency.kind == "job"
                assert dependency.interface_document is not None
                observed_roots.add(attempt.request_id)
            return rows

        monkeypatch.setattr(pod.worker.calls, "bindings", bindings)
        executor_pids: set[int] = set()
        for request in ("first", "reused"):
            pod.submit(request)
            body = pod.outcome(request)
            assert body["status"] == pb.OUTCOME_STATUS_SUCCEEDED, body
            result = body["result"]
            assert isinstance(result, dict)
            inline = result["inline_result"]
            assert isinstance(inline, str)
            value = msgspec.json.decode(base64.b64decode(inline), type=_Result)
            assert value.value == 26 and value.pid > 0
            rows = pod.rows(
                "SELECT body FROM execution_events WHERE request='"
                + request
                + "' AND kind='executor'"
            )
            for row in rows:
                raw = row["body"]
                assert isinstance(raw, bytes)
                executor_pids.add(json.loads(raw)["pid"])
            calls = pod.rows(
                "SELECT child_request FROM execution_calls WHERE parent_request='" + request + "'"
            )
            assert len(calls) == 2
        assert observed_roots == {"first", "reused"}
        assert executor_pids
        pod.wait(
            "child attempts to close",
            lambda: all(
                attempt.state == "closed" for attempt in pod.worker.engine.history.values()
            ),
            120,
        )
        executors = [
            supervision.current
            for supervision in (
                pod.worker.supervision,
                *(slot.supervision for slot in pod.worker.job_slots.values()),
                *(hosted.supervision for hosted in pod.worker.hosted.values()),
            )
            if supervision.current is not None
        ]
    assert not pod.thread.is_alive()
    assert all(not executor.alive() for executor in executors)
