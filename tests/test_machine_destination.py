"""A rented conversion publishes its weights outputs from the machine it ran on.

A memoized job root is submitted with a ``model://`` destination on its weights output and
the root's publication authorization. The real worker runs it in a real executor, which
uploads the returned Model through the ordinary upload_checkpoint effect to a stand-in Hub
before the attempt closes; the execution journals the published checkpoint for its owner.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import tempfile
import threading
import time
from pathlib import Path

import grpc
import pytest
import tensorfs

import signed_claims
from cozy_runtime import canonical_json
from cozy_runtime.internal import child_env
from cozy_runtime.internal.config import Credentials, RuntimeConfig
from cozy_runtime.internal.worker.control import GrpcControlHost
from cozy_runtime.internal.worker.machine_publication import PublicationAuthority
from cozy_runtime.internal.worker.plan import JobBinding
from cozy_runtime.internal.worker.session import Worker, WorkerOptions
from cozy_runtime.protocol import WIRE_MINOR, documents
from cozy_runtime.protocol import worker_pb2 as pb
from cozy_runtime.protocol import worker_pb2_grpc as rpc
from test_checkpoint_publication import (
    AUTHORIZATION,
    WORKER_TOKEN,
    Bucket,
    Hub,
    bucket_handler,
    certificate,
    hub_handler,
    serving,
)
from test_end_to_end import NO_EXECUTOR

SOURCE = """import struct
import tensorfs
from tensorfs.derived import Config, Derivation, Part, Target, Tensor
from cozy_runtime.author import App, Context, ModelArtifact, WeightsOutput, invocable

app = App()
PLAIN = dict(tensorfs.seed_digests())["plain/1"]

@invocable(memoize=True)
async def convert(ctx: Context, *, n: int) -> ModelArtifact:
    definition = Derivation(
        sources={},
        targets={"body": Target(add={
            "layer.weight": Tensor("f16", (16,32), PLAIN, {"value": Part("f16", (16,32))}),
        })},
        configs={"pipeline": Config("add")},
        order=(("body", "layer.weight"),),
    )
    with tensorfs.derive(ctx.output("model"), definition) as output:
        if output.receipt is None:
            output.add_part(
                "body", "layer.weight", "value", struct.pack("<512e", *(i/n for i in range(512)))
            )
            output.add_config("pipeline", b"{}")
        receipt = output.receipt or output.commit()
    return ctx.adopt_model(receipt)

app.job(convert, weights=(WeightsOutput("model", max_new_bytes=65536),))
"""


def root_offer(
    request: str, binding: JobBinding, weights: list[dict[str, object]]
) -> pb.AttemptOffer:
    payload = canonical_json.encode({"n": 3})
    digest = documents.spell(hashlib.sha256(payload).digest())
    spec = pb.InvocationSpec(
        payload_digest=digest,
        inputs=[
            pb.InputBinding(
                input_id="payload", digest=digest, length=len(payload), kind_mime="application/json"
            )
        ],
        outputs=[pb.OutputBinding(**row) for row in weights],  # type: ignore[arg-type]
        deadline_unix_ms=int((time.time() + 300) * 1000),
        job=pb.JobInvocationSpec(
            installation_id=binding.installation_id, job_descriptor_id=binding.job_descriptor_id
        ),
    )
    raw, spec_digest = documents.identity(spec)
    return pb.AttemptOffer(
        request_id=request,
        attempt_ordinal=1,
        invocation_spec_digest=spec_digest,
        invocation_spec_canonical_bytes=raw,
        grant=pb.DeliveryGrant(
            invocation_spec_digest=spec_digest,
            inputs=[
                pb.InputAccess(
                    input_id="payload",
                    url="data:application/json;base64," + base64.b64encode(payload).decode(),
                )
            ],
            outputs=[
                # Typed with capitals: the Hub's names fold, so this is alice/model.
                pb.OutputAccess(output_id=str(row["output_id"]), url="model://Alice/Model")
                for row in weights
            ],
        ),
    )


@pytest.mark.skipif(bool(NO_EXECUTOR), reason=NO_EXECUTOR or "")
def test_job_root_publishes_its_weights_output_to_the_granted_destination(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import test_job_preparation_isolation as fixture
    from conftest import image_python

    monkeypatch.setattr(fixture, "SOURCE", SOURCE)
    with tempfile.TemporaryDirectory(prefix="cz-dest.") as directory:
        root = Path(directory)
        environment, preparation = fixture.package(root, "convert")
        (root / "artifacts").mkdir()
        trusted, context = certificate(root)
        monkeypatch.setenv("SSL_CERT_FILE", str(trusted))
        bucket = Bucket()
        hub = Hub(bucket)
        with (
            serving(bucket_handler(bucket)) as bucket_port,
            serving(hub_handler(hub), context) as hub_port,
        ):
            hub.bucket_url = f"http://127.0.0.1:{bucket_port}"
            worker = Worker(
                RuntimeConfig(
                    cozy_home=root / "home",
                    credentials=Credentials(),
                    record_owner_public_key=signed_claims.PUBLIC_KEY,
                    child_base_env=tuple(
                        sorted(
                            (key, value)
                            for key, value in os.environ.items()
                            if not child_env.erased(key) and key != "PYTHONPATH"
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
                    publication_authority=PublicationAuthority(
                        f"https://localhost:{hub_port}", "worker", WORKER_TOKEN
                    ),
                ),
                GrpcControlHost("127.0.0.1:0", root / "address"),
            )
            assert worker.machine_effects is not None
            worker.machine_effects.allow_local = True  # the stand-in bucket is plain HTTP
            thread = threading.Thread(target=worker.run, daemon=True)
            try:
                installed = worker.prepare_local_package(preparation).installed_package
                captured, capture_digest = documents.identity(
                    pb.MachineExecutionCapture(
                        root_installation_id=installed.installation_id,
                        installed_packages=[installed],
                        bindings=[
                            pb.MachineCallableBinding(
                                caller_installation_id=installed.installation_id,
                                callee_installation_id=installed.installation_id,
                                module="prepare_convert",
                                export="convert",
                                entrypoint="convert",
                            )
                        ],
                    )
                )
                if worker.supervision.current is not None:
                    worker.supervision.retire_current(worker.supervision.current, "begin lifecycle")
                plans = [
                    json.loads(path.read_bytes())
                    for path in (root / "home/job-plans").glob("*/*.json")
                ]
                binding = JobBinding.read(next(row for row in plans if row.get("job") == "convert"))
                interface = json.loads(installed.package_interface)
                weights = next(job for job in interface["jobs"] if job["name"] == "convert")[
                    "weights_outputs"
                ]
                thread.start()
                deadline = time.monotonic() + 120
                while not (root / "address").exists():
                    assert thread.is_alive() and time.monotonic() < deadline
                    time.sleep(0.02)
                claim = signed_claims.claim()
                worker.serve_stream(iter([pb.RecordOwnerFrame(claim=claim)]), lambda _: None)
                client = rpc.WorkerControlStub(
                    grpc.insecure_channel((root / "address").read_text().strip())
                )
                workspace = client.GetMachineExecutionWorkspace(
                    pb.MachineExecutionWorkspaceQuery(claim=claim)
                ).execution_workspace_id

                def submit(request: str, authorization: str) -> None:
                    client.SubmitMachineExecution(
                        pb.MachineExecutionSubmit(
                            claim=claim,
                            submission_id=request,
                            capture_digest=capture_digest,
                            capture_canonical_bytes=captured,
                            offer=root_offer(request, binding, weights),
                            prepared_state=pb.DesiredWorkerState(
                                revision=1,
                                wire_minor=WIRE_MINOR,
                                posture=pb.POSTURE_ACCEPTING,
                                job=pb.JobDirective(
                                    installation_id=binding.installation_id,
                                    job_descriptor_id=binding.job_descriptor_id,
                                ),
                            ),
                            payload_canonical_bytes=canonical_json.encode({"n": 3}),
                            expected_execution_workspace_id=workspace,
                            publication_authorization_id=authorization,
                        )
                    )

                # A destination without its owner's publication grant is refused at intake.
                with pytest.raises(grpc.RpcError, match="publication authority") as refused:
                    submit("ungranted", "")
                assert refused.value.code() == grpc.StatusCode.FAILED_PRECONDITION
                assert worker.executions is not None
                assert not worker.executions.owns("owner", "ungranted")

                submit("root", AUTHORIZATION)
                while (state := worker.execution_status(claim, "root").state) not in (
                    "succeeded",
                    "failed",
                ):
                    assert thread.is_alive() and time.monotonic() < deadline
                    time.sleep(0.05)
                body = documents.read(
                    worker.collect_execution(claim, "root").outcome_canonical_bytes,
                    pb.AttemptOutcomeBody,
                )
                assert state == "succeeded", body
                (retained,) = body["result"]["retained_models"]
                artifact = json.loads(base64.b64decode(retained["model_artifact_canonical_bytes"]))
                manifest = artifact["manifest"]["digest"]

                # The memoized root's own result reached the destination, every closure object.
                assert set(hub.checkpoints) == {manifest}
                closure = {manifest} | {
                    str(row["id"])
                    for row in tensorfs.Store.ensure(root / "store").walk_cozytensors(manifest)
                }
                assert set(bucket.objects) == closure
                events = worker.executions.events("owner", "root").events
                published = [json.loads(e.body) for e in events if e.kind == "checkpoint"]
                assert published and all(
                    event
                    == {
                        "destination": "alice/model",
                        "checkpoint": manifest,
                        "output_slot": "model",
                        "observation": "acknowledged",
                    }
                    for event in published
                )
            finally:
                worker.request_stop()
                worker.host.stop()
                if thread.ident is not None:
                    thread.join(30)
                assert not thread.is_alive()
