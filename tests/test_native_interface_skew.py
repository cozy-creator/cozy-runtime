"""A package locked to an older released Runtime calls native operations under this worker.

The executor runs a released cozy-runtime from PyPI. 0.18.32 compares its whole native
interface digest; 0.18.48 refuses any row naming an operation it lacks. This worker has
since added `Convert.profiles` and the upload operations.
"""

from __future__ import annotations

import base64
import json
import os
import subprocess
import tempfile
import threading
import time
from pathlib import Path

import grpc
import msgspec
import pytest

from cozy_runtime import canonical_json
from cozy_runtime.author import Tree
from cozy_runtime.internal import child_env, native_interfaces, schema, source_interfaces
from cozy_runtime.internal.config import Credentials, RuntimeConfig
from cozy_runtime.internal.local import LocalRecordOwner, LocalRequest
from cozy_runtime.internal.worker.control import GrpcControlHost
from cozy_runtime.internal.worker.plan import JobBinding
from cozy_runtime.internal.worker.session import Worker, WorkerOptions
from cozy_runtime.protocol import WIRE_MINOR, documents
from cozy_runtime.protocol import worker_pb2 as pb
from cozy_runtime.protocol import worker_pb2_grpc as rpc
from test_end_to_end import NO_EXECUTOR
from test_machine_children import native_origin

PACKAGE = """import msgspec
from cozy_runtime.author import App, ChildCallError, Context
from cozy_runtime.author.sources import (
    convert_cozytensors, download_civitai, download_huggingface, source_files)
app = App()
class Request(msgspec.Struct):
    pass
class Result(msgspec.Struct):
    manifest: str
    files: int
    refused: str
@app.job
async def JOB(ctx: Context, payload: Request) -> Result:
    source = await download_huggingface(
        'example/model', revision='a' * 40, carriers=('provider/first.safetensors',))
    tree = await source_files(source)
    model = await convert_cozytensors(source, profile='fixture/first/1')
    try:
        await download_civitai(123)
        refused = ''
    except ChildCallError as exc:
        refused = str(exc)
    return Result(model.manifest.digest, sum(1 for _ in tree.path.rglob('*.*')), refused)
"""


@pytest.mark.skipif(bool(NO_EXECUTOR), reason=NO_EXECUTOR or "")
@pytest.mark.parametrize("older", ["0.18.32", "0.18.48"])
def test_older_executor_calls_native_operations_under_this_worker(
    monkeypatch: pytest.MonkeyPatch, older: str
) -> None:
    import test_job_preparation_isolation as fixture
    from conftest import image_python

    # This worker is modelled as one that predates download_civitai; every other native
    # operation stays exactly as this checkout serves it.
    offered = canonical_json.decode(source_interfaces.DOCUMENT)
    del offered["operations"]["download_civitai"]
    monkeypatch.setattr(source_interfaces, "DOCUMENT", canonical_json.encode(offered))
    monkeypatch.setattr(native_interfaces, "RUNTIME_VERSION", "0.18.31")
    monkeypatch.setattr(fixture, "SOURCE", PACKAGE)
    with tempfile.TemporaryDirectory(prefix="cz-skew.") as directory:
        root = Path(directory)
        environment, preparation = fixture.package(root, "older", runtime=older)
        (root / "artifacts").mkdir()
        worker = Worker(
            RuntimeConfig(
                cozy_home=root / "home",
                credentials=Credentials(),
                bootstrap_credential="offline",
                child_base_env=tuple(
                    sorted(
                        (key, value)
                        for key, value in os.environ.items()
                        if not child_env.erased(key)
                        and key not in ("PYTHONPATH", "COZY_BOOTSTRAP_CREDENTIAL")
                    )
                ),
            ),
            WorkerOptions(
                root=root / "worker",
                devices="",
                accelerator_backend="none",
                python=str(image_python()),
                install_root=environment,
                artifact_cache=root / "artifacts",
                tensorfs_root=root / "store",
                grant_roots=(str(root),),
            ),
            GrpcControlHost("127.0.0.1:0", root / "address"),
        )
        thread = threading.Thread(target=worker.run, daemon=True)
        origin = native_origin(worker)
        try:
            prepared = worker.prepare_local_package(preparation)
            if worker.supervision.current is not None:
                worker.supervision.retire_current(worker.supervision.current, "begin lifecycle")
            plans = [
                json.loads(path.read_bytes()) for path in (root / "home/job-plans").glob("*/*.json")
            ]
            binding = JobBinding.read(next(row for row in plans if row.get("job") == "older"))
            # The released executor's own documents are the ones this worker assumes for it.
            probe = subprocess.run(
                [
                    binding.python,
                    "-c",
                    "from cozy_runtime.internal import source_interfaces as s, "
                    "effect_interfaces as e; print(s.DIGEST, e.DIGEST)",
                ],
                capture_output=True,
                text=True,
                env={k: v for k, v in os.environ.items() if k != "PYTHONPATH"},
            )
            assert probe.returncode == 0, probe.stderr
            released = probe.stdout.split()
            assert released == [
                canonical_json.digest(native_interfaces.released(older)[module])
                for module in ("cozy_runtime.author.sources", "cozy_runtime.author.publication")
            ]
            captured, capture_digest = documents.identity(
                pb.MachineExecutionCapture(
                    root_installation_id=prepared.installed_package.installation_id,
                    installed_packages=[prepared.installed_package],
                )
            )
            thread.start()
            deadline = time.monotonic() + 120
            while not (root / "address").exists():
                assert thread.is_alive() and time.monotonic() < deadline
                time.sleep(0.02)
            claim = pb.Claim(
                record_owner_id="owner",
                record_owner_epoch=1,
                wire_minor=WIRE_MINOR,
                proof=b"offline",
            )
            worker.serve_stream(iter([pb.RecordOwnerFrame(claim=claim)]), lambda _: None)
            generator = LocalRecordOwner(
                LocalRequest(
                    entrypoint="older",
                    payload={},
                    outputs=(),
                    kind="job",
                    request_id="root",
                    job_descriptor_id=binding.job_descriptor_id,
                    timeout_ms=90_000,
                ),
                {},
                root / "grants",
                package_installation_id=binding.installation_id,
            )
            generator.host.inbound.get_nowait()
            generator._dispatch()
            offer = generator.host.inbound.get_nowait()
            assert offer is not None
            state = generator._desired_state()
            state.job.orchestration = True
            client = rpc.WorkerControlStub(
                grpc.insecure_channel((root / "address").read_text().strip())
            )
            client.SubmitMachineExecution(
                pb.MachineExecutionSubmit(
                    claim=claim,
                    submission_id="root",
                    capture_digest=capture_digest,
                    capture_canonical_bytes=captured,
                    offer=offer.attempt_offer,
                    prepared_state=state,
                    payload_canonical_bytes=canonical_json.encode({}),
                    expected_execution_workspace_id=client.GetMachineExecutionWorkspace(
                        pb.MachineExecutionWorkspaceQuery(claim=claim)
                    ).execution_workspace_id,
                )
            )
            while worker.execution_status(claim, "root").state not in ("succeeded", "failed"):
                assert thread.is_alive() and time.monotonic() < deadline
                time.sleep(0.05)
            outcome = worker.collect_execution(claim, "root")
            body = documents.read(outcome.outcome_canonical_bytes, pb.AttemptOutcomeBody)
            assert worker.workspace is not None
            with worker.workspace.locked() as db:
                calls = [
                    tuple(row)
                    for row in db.execute(
                        "SELECT call_index,safe_code,safe_detail FROM execution_calls "
                        "ORDER BY call_index"
                    )
                ]
                natives = [
                    tuple(row)
                    for row in db.execute(
                        "SELECT operation,state,native_error,service_id FROM native_calls"
                    )
                ]
            assert worker.source_calls is not None
            details = [worker.source_calls.failure_detail(row[3]) for row in natives]
            assert body["status"] == pb.OUTCOME_STATUS_SUCCEEDED, (calls, natives, details)
            result = canonical_json.decode(base64.b64decode(body["result"]["inline_result"]))
            assert result["manifest"].startswith("sha256:") and result["files"] == 1
            # Only the operation this worker lacks is refused, naming it and the remedy.
            assert result["refused"] == (
                'child_capability_unavailable: native operation "download_civitai" is not '
                "offered by this machine's Runtime 0.18.31 (executor needs it); "
                "run `cozy rental update <rental>`"
            )
        finally:
            worker.request_stop()
            worker.host.stop()
            if thread.ident is not None:
                thread.join(30)
            assert not thread.is_alive()
            origin[0].shutdown()
            origin[0].server_close()
            origin[1].join()


class _Older(msgspec.Struct, frozen=True):
    repository: str
    revision: str


class _NewerOptional(msgspec.Struct, frozen=True):
    repository: str
    revision: str
    mirror: str = ""


class _NewerRequired(msgspec.Struct, frozen=True):
    repository: str
    revision: str
    mirror: str


class _WithoutRevision(msgspec.Struct, frozen=True):
    repository: str


class _Files(msgspec.Struct, frozen=True):
    files: Tree
    listing: str


def _stating(
    version: str,
    request: type[msgspec.Struct] | None = None,
    result: type[msgspec.Struct] | None = None,
) -> dict[str, object]:
    """A hello stating `download_huggingface` / `source_files` rendered from these types."""
    stated = native_interfaces.stated()
    operations = stated[source_interfaces.MODULE]["operations"]
    if request is not None:
        operations["download_huggingface"]["request"] = schema.render(request)
    if result is not None:
        operations["source_files"]["result"] = schema.render(result)
    return {"native_interfaces": stated, "runtime_version": version}


def _unavailable(hello: dict[str, object], export: str) -> str:
    row = next(row for row in native_interfaces.rows(hello) if row["export"] == export)
    return str(row.get("unavailable", ""))


def test_additive_optional_fields_are_admitted_in_both_directions() -> None:
    assert not any("unavailable" in row for row in native_interfaces.rows({}))
    assert not _unavailable(_stating("0.18.1", _Older), "download_huggingface")
    assert not _unavailable(_stating("99.0.0", _NewerOptional), "download_huggingface")


def test_required_fields_are_enforced_naming_the_side_to_update() -> None:
    newer = _unavailable(_stating("99.0.0", _NewerRequired), "download_huggingface")
    assert "request mirror is not read" in newer and "cozy rental update" in newer
    older = _unavailable(_stating("0.18.1", _WithoutRevision), "download_huggingface")
    assert "request revision is not sent" in older
    assert "publish the package against cozy-runtime>=" in older
    # The refusal is confined to that operation.
    rows = native_interfaces.rows(_stating("0.18.1", _WithoutRevision))
    assert [row["export"] for row in rows if "unavailable" in row] == ["download_huggingface"]


def test_results_need_what_the_executor_requires() -> None:
    refused = _unavailable(_stating("0.18.1", result=_Files), "source_files")
    assert "result listing is not produced" in refused and "result files" not in refused


def test_an_operation_this_worker_lacks_is_refused_alone() -> None:
    hello = _stating("99.0.0")
    operations = hello["native_interfaces"][source_interfaces.MODULE]["operations"]  # type: ignore[index]
    operations["download_dataset"] = operations["download_huggingface"]
    rows = native_interfaces.rows(hello)
    (refused,) = [row for row in rows if "unavailable" in row]
    assert (
        refused["export"] == "download_dataset" and "cozy rental update" in refused["unavailable"]
    )


def test_released_executors_get_the_documents_they_carry() -> None:
    digests = {
        version: canonical_json.digest(
            native_interfaces.released(version)[source_interfaces.MODULE]
        )
        for version in ("0.18.32", "0.18.39", "0.18.40", "0.18.48", "0.18.49", "0.18.50", "dev")
    }
    assert {v: d[7:19] for v, d in digests.items()} == {
        "0.18.32": "a0e88796caf5",
        "0.18.39": "a0e88796caf5",
        "0.18.40": "efc5f80ab313",
        "0.18.48": "efc5f80ab313",
        "0.18.49": "e8eca59e0baa",
        "0.18.50": "e8eca59e0baa",
        "dev": "a0e88796caf5",
    }
    # 0.18.48 refuses a row naming an operation it lacks, so the uploads are not listed.
    listed = {row["export"] for row in native_interfaces.rows({"runtime_version": "0.18.48"})}
    assert "upload_huggingface" not in listed and "download_huggingface" in listed
