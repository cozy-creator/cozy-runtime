"""A published callee is installed when the running root first selects it, never before."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import threading
import time
from pathlib import Path

import grpc
import pytest

import signed_claims
from conftest import image_python
from cozy_runtime import canonical_json
from cozy_runtime.internal import child_env, package_interface, static_interface
from cozy_runtime.internal.config import Credentials, RuntimeConfig
from cozy_runtime.internal.worker.control import GrpcControlHost
from cozy_runtime.internal.worker.plan import JobBinding
from cozy_runtime.internal.worker.session import Worker, WorkerOptions
from cozy_runtime.protocol import documents
from cozy_runtime.protocol import worker_pb2 as pb
from cozy_runtime.protocol import worker_pb2_grpc as rpc
from local_owner import LocalRecordOwner, LocalRequest
from test_end_to_end import NO_EXECUTOR, _run
from test_job_preparation_isolation import package
from test_publish_derivation import _download_set, _SimpleIndex

CALLEE = """
from cozy_runtime.author import invocable
@invocable
async def triple(ctx: Context, *, n: int) -> Result:
    return Result(n * 3)
app.job(triple)
"""


@pytest.mark.skipif(bool(NO_EXECUTOR), reason=NO_EXECUTOR or "")
@pytest.mark.parametrize("case", ["selected", "broken"])
def test_deferred_callee_installs_on_first_selection(case: str) -> None:
    uv = shutil.which("uv")
    assert uv is not None
    with tempfile.TemporaryDirectory(prefix="cz-defer.") as raw:
        root = Path(raw)
        environment, root_request = package(root, "offline")
        package(root, "callee")
        package(root, "unused")
        for label in ("callee", "unused"):
            source = root / f"project_{label}" / f"prepare_{label}.py"
            source.write_text(source.read_text() + CALLEE)
        project = root / "project_offline"
        source = project / "prepare_offline.py"
        source.write_text(
            source.read_text()
            .replace("import msgspec\n", "import msgspec\nimport prepare_callee\n")
            .replace("def offline(", "async def offline(")
            .replace("return Result(7)", "return Result((await prepare_callee.triple(n=7)).value)")
        )
        pyproject = project / "pyproject.toml"
        pyproject.write_text(
            pyproject.read_text().replace(
                'dependencies=["cozy-runtime>=0.16.8,<1"]',
                'dependencies=["cozy-runtime>=0.16.8,<1","prepare-callee==1.0.0","prepare-unused==1.0.0"]',
            )
        )
        wheels: dict[str, Path] = {}
        for label in ("offline", "callee", "unused"):
            (root / f"project_{label}" / "package.toml").write_text(
                f'[application]\nobject="prepare_{label}:app"\n'
            )
            out = root / "wheels" / label
            _run(uv, "build", "--wheel", "--out-dir", str(out), str(root / f"project_{label}"))
            (wheels[label],) = out.glob("*.whl")
        runtime = next(
            Path(row.path) for row in root_request.files if row.filename.startswith("cozy_runtime-")
        )
        index = _SimpleIndex(root / "index", [*wheels.values(), runtime])

        def pin(path: Path) -> str:
            name, version = path.name.split("-")[:2]
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            return f"{name.replace('_', '-')}=={version} --hash=sha256:{digest}\n"

        def locked(*labels: str) -> bytes:
            return (
                f"--index-url https://pypi.org/simple\n--extra-index-url {index.url}\n"
                + "".join(pin(wheels[label]) for label in labels)
                + pin(runtime)
            ).encode() + bytes(root_request.dependency_requirements)

        worker = Worker(
            RuntimeConfig(
                cozy_home=root / "home",
                credentials=Credentials(),
                record_owner_public_key=signed_claims.PUBLIC_KEY,
                child_base_env=tuple(
                    sorted(
                        (k, v)
                        for k, v in os.environ.items()
                        if not child_env.erased(k) and k != "PYTHONPATH"
                    )
                ),
            ),
            WorkerOptions(
                **signed_claims.IDENTITY,
                root=root / "worker",
                devices="",
                accelerator_backend="none",
                python=str(image_python()),
                install_root=environment,
                artifact_cache=root / "artifacts",
                tensorfs_root=root / "store",
                grant_roots=(str(root),),
            ),
            GrpcControlHost("127.0.0.1:0", root / "control.addr"),
        )
        (root / "artifacts").mkdir()
        thread = threading.Thread(target=worker.run, daemon=True)

        def preparation(label: str, *labels: str) -> pb.PreparePackageSetRequest:
            return pb.PreparePackageSetRequest(
                download_delegation=_download_set(f"published/prepare-{label}", "1.0.0"),
                application=f"prepare_{label}:app",
                image_inventory=pb.ImageInventory(
                    profile="test",
                    python="3.12",
                    distributions=[
                        pb.ImageDistribution(
                            distribution="cozy-runtime",
                            version=worker.base.distribution_versions["cozy-runtime"],
                        )
                    ],
                ),
                locked_requirements=locked(label, *labels),
                package_interface=package_interface.canonical_bytes(
                    static_interface.build(root / f"project_{label}")
                ),
            )

        try:
            request = preparation("offline", "callee", "unused")
            request.install_root = str(environment)
            prepared = worker.prepare_package_set(request)
            installation = prepared.installed_package.installation_id

            def deferred(label: str) -> pb.DeferredInstallation:
                callee = preparation(label)
                if case == "broken" and label == "callee":
                    callee.locked_requirements = b"prepare-callee==9.9.9\n"
                return pb.DeferredInstallation(
                    key=f"published/prepare-{label}@1.0.0",
                    package=f"published/prepare-{label}",
                    release="1.0.0",
                    preparation=callee,
                )

            capture, capture_digest = documents.identity(
                pb.MachineExecutionCapture(
                    root_installation_id=installation,
                    installed_packages=[prepared.installed_package],
                    deferred_installations=[deferred("callee"), deferred("unused")],
                    bindings=[
                        pb.MachineCallableBinding(
                            caller_installation_id=installation,
                            module=f"prepare_{label}",
                            export="triple",
                            callee_deferred_key=f"published/prepare-{label}@1.0.0",
                            entrypoint="triple",
                        )
                        for label in ("callee", "unused")
                    ],
                )
            )
            if worker.supervision.current is not None:
                worker.supervision.retire_current(worker.supervision.current, "begin lifecycle")
            plans = [
                json.loads(path.read_bytes()) for path in (root / "home/job-plans").glob("*/*.json")
            ]
            binding = JobBinding.read(next(row for row in plans if row.get("job") == "offline"))
            thread.start()
            deadline = time.monotonic() + 600
            while not (root / "control.addr").exists():
                assert thread.is_alive() and time.monotonic() < deadline
                time.sleep(0.02)
            claim = signed_claims.claim()
            worker.serve_stream(iter([pb.RecordOwnerFrame(claim=claim)]), lambda _: None)
            generator = LocalRecordOwner(
                LocalRequest(
                    entrypoint="offline",
                    payload={},
                    outputs=(),
                    kind="job",
                    request_id="root",
                    job_descriptor_id=binding.job_descriptor_id,
                    timeout_ms=480_000,
                ),
                {},
                root / "grants",
                package_installation_id=binding.installation_id,
            )
            offered = generator.offer()
            state = generator.desired_state()
            state.job.orchestration = True
            client = rpc.WorkerControlStub(
                grpc.insecure_channel((root / "control.addr").read_text().strip())
            )
            workspace = client.GetMachineExecutionWorkspace(
                pb.MachineExecutionWorkspaceQuery(claim=claim)
            )

            def installed(label: str) -> bool:
                return any(
                    placement.document.package.package == f"published/prepare-{label}"
                    for placement in worker.prepared_installations.values()
                )

            client.SubmitMachineExecution(
                pb.MachineExecutionSubmit(
                    claim=claim,
                    submission_id="root",
                    capture_digest=capture_digest,
                    capture_canonical_bytes=capture,
                    offer=offered,
                    prepared_state=state,
                    payload_canonical_bytes=canonical_json.encode({}),
                    expected_execution_workspace_id=workspace.execution_workspace_id,
                )
            )
            # Acceptance waited for no callee.
            assert not installed("callee") and not installed("unused")
            while worker.execution_status(claim, "root").state not in ("succeeded", "failed"):
                assert thread.is_alive() and time.monotonic() < deadline
                time.sleep(0.05)
            body = documents.read(
                worker.collect_execution(claim, "root").outcome_canonical_bytes,
                pb.AttemptOutcomeBody,
            )
            assert not installed("unused")
            if case == "broken":
                assert body["status"] == pb.OUTCOME_STATUS_FAILED, body
                assert worker.workspace is not None
                with worker.workspace.locked() as db:
                    refused = db.execute(
                        "SELECT safe_detail FROM execution_calls WHERE parent_request='root'"
                    ).fetchone()
                assert "deferred_installation_failed: published/prepare-callee@1.0.0" in refused[0]
                assert not worker.stop.is_set()
                return
            assert body["status"] == pb.OUTCOME_STATUS_SUCCEEDED, body
            assert installed("callee")
        finally:
            worker.request_stop()
            worker.host.stop()
            if thread.ident is not None:
                thread.join(30)
            index.close()
